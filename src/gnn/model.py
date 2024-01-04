from typing import Any
import numba as nb
import numpy as np
import scipy.sparse
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
import torch_geometric.nn as gnn
from torch_scatter import scatter_mean

from .bistride import bstride_selection
    

def init_model(num_features, num_classes, args, **kwargs):
    """
        Parameters:
        -----------
            num_features: int
            num_classes: int
            args: argparse.Namespace
        Returns:
        --------
            model: torch.nn.Module
            is_trained: bool
    """
    if args.model == "MLP":
        return MLP(num_features, num_classes, args.hidden_dim, args.num_layers)
    elif args.model == "SIGN":
        return SIGN(num_features, num_classes, args.hidden_dim, args.num_layers, args.num_hops)
    elif args.model == "GCN":
        return GCN(num_features, num_classes, args.hidden_dim, args.num_layers)
    elif args.model == "GAT":
        return GAT(num_features, num_classes, args.hidden_dim, args.num_layers, heads=args.num_heads)
    elif args.model == "GraphUNet":
        pool = {
            "topk": TopKPooling,
            "sag": SAGPooling,
            "bistride": TwoStridePooling,
        }[args.pool]
        conv = {
            "gcn": gnn.GCNConv,
            "gat": gnn.GATConv,
        }[args.conv]
        return GraphUNet(num_features, num_classes, args.hidden_dim, depth=args.depth, pool_ratios=args.pool_ratios, conv=conv, pool=pool)
    elif args.model == "NodeEdgeGNN":
        return NodeEdgeGNN(num_features, num_classes, kwargs["num_edge_features"], args.hidden_dim, args.num_layers)
    else:
        raise NotImplementedError(f"no such model {args.model}")
    
class Activation(nn.Module):
    def __init__(self, act):
        super().__init__()
        if act == 'prelu':
            self.act = nn.PReLU()
        elif act in ['sigmoid', 'tanh']:
            self.act = getattr(torch, act) 
        else:
            self.act = getattr(F, act)
    def forward(self, x):
        return self.act(x)

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_layers=2, act='relu', res=True, norm=False):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers   = num_layers
        self.act  = act
        self.act_fn = Activation(act)
        if norm:
            bns = [nn.BatchNorm1d(in_channels)]
            for _ in range(num_layers - 1):
                bns.append(nn.BatchNorm1d(hidden_channels))
            self.bns = nn.ModuleList(bns)
        else:
            self.bns = None
        layers = [nn.Linear(in_channels, hidden_channels)]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_channels, hidden_channels)]
        layers += [nn.Linear(hidden_channels, out_channels)]
        self.layers = nn.ModuleList(layers)
        if res:
            self.res = nn.Linear(in_channels, out_channels)
        self.reset_parameters()
    def __str__(self):
        return f"MLP_{self.in_channels}_{self.out_channels}_{self.hidden_channels}_{self.num_layers}"
    def __repr__(self):
        return (f"MLP(in_channels={self.in_channels}, "
                f"out_channels={self.out_channels}, "
                f"hidden_channels={self.hidden_channels}, "
                f"num_layers={self.num_layers})")
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        if hasattr(self, "res"):
            self.res.reset_parameters()
        if self.bns is not None:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x):
        if self.bns is not None:
            x = self.bns[0](x)
        if hasattr(self, "res"):
            res = self.res(x)
        for i in range(self.num_layers - 1):
            x = self.layers[i](x)
            x = self.act_fn(x)
            if self.bns is not None:
                x = self.bns[i+1](x)
        x = self.layers[-1](x)
        if hasattr(self, "res"):
            return x + res
        return x     
    

class GCN(gnn.GCN):
    def __init__(self, in_channels, out_channels, hidden_channels, num_layers=2):
        super().__init__(in_channels, hidden_channels, num_layers, out_channels, bias=True)
    def __str__(self):
        return f"GCN_{self.in_channels}_{self.out_channels}_{self.hidden_channels}_{self.num_layers}"
    def __repr__(self):
        return (f"GCN(in_channels={self.in_channels}, "
                f"out_channels={self.out_channels}, "
                f"hidden_channels={self.hidden_channels}, "
                f"num_layers={self.num_layers})")
    
class GAT(gnn.GAT):
    def __init__(self, in_channels, out_channels, hidden_channels, num_layers=2, heads=4):
        super().__init__(in_channels, hidden_channels, num_layers, out_channels, heads=heads, concat=True, bias=True)
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers   = num_layers
        self.heads        = heads
    def __str__(self):
        return f"GAT_{self.in_channels}_{self.out_channels}_{self.hidden_channels}_{self.num_layers}_{self.heads}"
    def __repr__(self):
        return (f"GAT(in_channels={self.in_channels}, "
                f"out_channels={self.out_channels}, "
                f"hidden_channels={self.hidden_channels}, "
                f"num_layers={self.num_layers}, "
                f"heads={self.heads})")

class GCNConvNoWeight(gnn.MessagePassing):
    def __init__(self):
        # Initialize with 'add' aggragation
        super(GCNConvNoWeight, self).__init__(aggr='add') 
    def forward(self, x, edge_index):
        # Normalize node features
        row, col = edge_index
        deg = pyg.utils.degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, norm=norm)

    def message(self, x_j, norm):
        # Normalize message
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # Pass aggregated node features through linear layer
        return aggr_out

class SIGN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_layers=2, num_hops=3, act='relu'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels= out_channels
        self.hidden_channels = hidden_channels
        self.num_layers  = num_layers
        self.num_hops    = num_hops
        self.branches = nn.ModuleList()
        for _ in range(num_hops):
            self.branches.append(MLP(in_channels, hidden_channels, hidden_channels, num_layers, act))
        self.merger   = MLP(hidden_channels * num_hops, out_channels, hidden_channels, num_layers, act)
        self.reset_parameters()
    def __str__(self):
        return f"SIGN_{self.in_channels}_{self.out_channels}_{self.hidden_channels}_{self.num_layers}_{self.num_hops}"
    def __repr__(self):
        return (f"SIGN(in_channels={self.in_channels}, "
                f"out_channels={self.out_channels}, "
                f"hidden_channels={self.hidden_channels}, "
                f"num_layers={self.num_layers}, "
                f"num_hops={self.num_hops})")
    def reset_parameters(self):
        for branch in self.branches:
            branch.reset_parameters()
        self.merger.reset_parameters()
    def forward(self, x, edge_index):
        xs = [self.branches[0](x)]
        for branch in self.branches[1:]:
            x = GCNConvNoWeight()(x, edge_index)
            xs.append(branch(x))
        x = torch.cat(xs, dim=-1)
        x = self.merger(x)
        return x

class TwoStridePooling(nn.Module):
    """The ratio can only be 0.5
    """
    def __init__(self):
        super().__init__()

    def __str__(self):
        if hasattr(self, "input_num_nodes"):
            return f"TwoStridePooling({self.input_num_nodes},{self.input_num_edges}->{self.output_num_nodes},{self.output_num_edges})"
        else:
            return f"TwoStridePooling()"

    def __repr__(self):
        return self.__str__()
    
    def pool(self, x, edge_index,  pos):
        assert edge_index.device.type == "cpu", "edge_index must be on cpu"
        self.input_num_nodes = x.size(0)
        self.input_num_edges = edge_index.size(1)
        self.input_graph = edge_index
        row, col = edge_index
        row = row.cpu().numpy()
        col = col.cpu().numpy()
        edge_index = edge_index
        index_to_keep, edge_index_ = bstride_selection(edge_index.numpy(), pos_mesh=pos.numpy(), n=x.shape[0])
        x_     = x[index_to_keep]
        self.S = index_to_keep
        edge_index_ = torch.from_numpy(edge_index_)

        self.output_num_nodes = x_.size(0)
        self.output_num_edges = edge_index_.size(1)
        return x_, edge_index_
    def unpool(self, x_, edge_index_):
        x = torch.zeros(self.input_num_nodes, *x_.shape[1:], device=x_.device, dtype=x_.dtype)
        x[self.S] = x_

        edge_index = self.input_graph

        return x, edge_index

class SAGPooling(nn.Module):
    def __init__(self, in_channels, ratio=0.5) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.ratio       = ratio
        self.gnn         = gnn.GCNConv(in_channels, 1)
        self.w           = nn.Parameter(torch.Tensor(1,))
    def pool(self, x, edge_index):
        """
            Parameters:
            -----------
                x: torch.Tensor, shape [N, in_channels]
                edge_index: torch.Tensor, shape [2, E]

            Returns:
            --------
                x: torch.Tensor, shape [N', in_channels]
                edge_index: torch.Tensor, shape [2, E']
        """
        self.input_num_nodes = x.size(0)
        self.input_graph = edge_index

        score = self.gnn(x, edge_index).squeeze()
        score = score * self.w 
        score = torch.tanh(score)
        topk  = torch.topk(score, k=int(self.ratio * x.size(0))).indices

        self.S = topk 
        x_     = x[topk]
        edge_index_, _ = pyg.utils.subgraph(topk, edge_index, relabel_nodes=True)
        
        return x_, edge_index_
    
    def unpool(self, x_, edge_index_):
        """
            Parameters:
            -----------
                x: torch.Tensor, shape [N', in_channels]
                edge_index: torch.Tensor, shape [2, E']

            Returns:
            --------
                x: torch.Tensor, shape [N, in_channels]
                edge_index: torch.Tensor, shape [2, E]
        """
        x = torch.zeros(self.input_num_nodes, self.in_channels, device=x.device, dtype=x.dtype).to(x.device)
        x[self.S] = x_

        edge_index = self.input_graph

        return x, edge_index

class TopKPooling(nn.Module):
    """Only select the top k (ratio * N) nodes subgraph from the input graph.
    """
    def __init__(self, in_channels, ratio=0.5):
        super().__init__()
        self.in_channels = in_channels
        self.ratio       = ratio
        self.w           = nn.Parameter(torch.Tensor(1, in_channels))

    def __str__(self):
        if hasattr(self, "input_num_nodes"):
            return f"TopKPooling({self.input_num_nodes},{self.input_num_edges}->{self.output_num_nodes},{self.output_num_edges})"
        else:
            return f"TopKPooling(in_chanels={self.in_channels}, ratio={self.ratio})"

    def __repr__(self):
        return self.__str__()

    def pool(self, x, edge_index):
        """
            Parameters:
            -----------
                x: torch.Tensor, shape [N, in_channels]
                edge_index: torch.Tensor, shape [2, E]

            Returns:
            --------
                x: torch.Tensor, shape [N', in_channels]
                edge_index: torch.Tensor, shape [2, E']
        """
 
        self.input_num_nodes = x.size(0)
        self.input_num_edges = edge_index.size(1)
        self.input_graph = edge_index

        score = (x * self.w).sum(dim=-1) / self.w.norm(p=2, dim=-1)
        score = torch.tanh(score)
        topk  = torch.topk(score, k=int(self.ratio * x.size(0))).indices

        self.S = topk 
        x_     = x[topk]
        edge_index_, _ = pyg.utils.subgraph(topk, edge_index, relabel_nodes=True)
        
        self.output_num_nodes = x_.size(0)
        self.output_num_edges = edge_index_.size(1)
        return x_, edge_index_
     
    def unpool(self, x_, edge_index_):
        """
            Parameters:
            -----------
                x: torch.Tensor, shape [N', in_channels]
                edge_index: torch.Tensor, shape [2, E']

            Returns:
            --------
                x: torch.Tensor, shape [N, in_channels]
                edge_index: torch.Tensor, shape [2, E]
        """
        assert self.output_num_nodes == x_.size(0)
        assert self.output_num_edges == edge_index_.size(1)
        x = torch.zeros(self.input_num_nodes, self.in_channels, device=x_.device, dtype=x_.dtype)
        x[self.S] = x_

        edge_index = self.input_graph

        return x, edge_index

class GraphUNet(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 hidden_channels = 64,
                 depth = 3,
                 pool_ratios = 0.5,
                 bn    = True,
                 act   = nn.ReLU(),
                 conv  = gnn.GCNConv,
                 pool  = TopKPooling,):
        super(GraphUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels= out_channels
        self.hidden_channels = hidden_channels
        self.depth      = depth
        self.use_bn     = bn
        self.depth      = depth
        self.pool_ratios= pool_ratios
        self.act        = act
        self.conv       = conv
        self.pool       = pool
        self.up_convs = nn.ModuleList()
        self.up_bns   = nn.ModuleList() if bn else None
        self.dn_convs = nn.ModuleList()
        self.dn_bns   = nn.ModuleList() if bn else None
        self.pools    = nn.ModuleList()
        for i in range(depth):
            self.dn_convs.append(conv(hidden_channels, hidden_channels))
            if bn:
                self.dn_bns.append(nn.BatchNorm1d(hidden_channels))
            if i < self.depth:
                if pool is TwoStridePooling:
                    self.pools.append(pool())
                else:
                    self.pools.append(pool(hidden_channels, ratio=pool_ratios))
        
        for i in range(depth):
            self.up_convs.append(conv(2 * hidden_channels, hidden_channels))
            if bn:
                self.up_bns.append(nn.BatchNorm1d(hidden_channels))
        self.in_transform     = conv(in_channels, hidden_channels)
        self.bottom_transform = conv(hidden_channels, hidden_channels)
        self.out_transform    = conv(hidden_channels, out_channels)

        self.reset_parameters()

    def __str__(self):
        conv = {
            gnn.GCNConv: "gcn",
            gnn.GATConv: "gat",
        }[self.conv]
        pool = {
            TopKPooling: "topk",
            SAGPooling: "sag",
            TwoStridePooling: "bistride",
        }[self.pool]
        return f"GraphUNet_{self.in_channels}_{self.out_channels}_{self.hidden_channels}_{self.depth}_{self.pool_ratios}_{conv}_{pool}"

    def __repr__(self):
        return (f"GraphUNet(in_channels={self.in_channels}, "
                f"out_channels={self.out_channels}, "
                f"hidden_channels={self.hidden_channels}, "
                f"depth={self.depth}, "
                f"pool_ratios={self.pool_ratios}, "
                f"act={self.act}, "
                f"conv={self.conv}, "
                f"pool={self.pool})")

    def reset_parameters(self):
        for conv in self.dn_convs:
            conv.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, **kwargs):
        xs = []
        x = self.in_transform(x, edge_index)
        if isinstance(self.pools[0], TwoStridePooling):
            assert "pos" in kwargs, "pos must be in kwargs"
            pos = kwargs["pos"]

        for i in range(self.depth):
            x = self.dn_convs[i](x, edge_index)
            if self.use_bn:
                x = self.dn_bns[i](x)
            x = self.act(x)
            xs.append(x)
            if i < self.depth:
                if isinstance(self.pools[i], TwoStridePooling):
                    x, edge_index = self.pools[i].pool(x, edge_index, pos)
                    pos = pos[self.pools[i].S]
                else:
                    x, edge_index = self.pools[i].pool(x, edge_index)
        x = self.bottom_transform(x, edge_index)
        x = self.act(x)
        for i in range(self.depth):
            x, edge_index = self.pools[self.depth-i-1].unpool(x, edge_index)
            x = torch.cat([x, xs[self.depth-i-1]], dim=1)
            x = self.up_convs[i](x, edge_index)
            if self.use_bn:
                x = self.up_bns[i](x)
            x = self.act(x)
        x = self.out_transform(x, edge_index)
        return x
    

class NodeEdgeConv(nn.Module):
    """
        edata = linear(edata, ndata_u, ndata_v)
        ndata = linear(ndata, mean(edata))
    """
    def __init__(self, n_in_channels, n_out_channels, e_in_channels, e_out_channels):
        super().__init__()
        self.n_lin = nn.Linear(n_in_channels + e_out_channels, n_out_channels)
        self.e_lin = nn.Linear(e_in_channels + n_in_channels * 2, e_out_channels)
        self.prop  = GCNConvNoWeight()
        self.reset_parameters()

    def reset_parameters(self):
        self.n_lin.reset_parameters()
        self.e_lin.reset_parameters()

    def forward(self, x, edata, edge_index):
        row, col = edge_index
        x_src, x_dst = x[row], x[col]
        edata = self.e_lin(torch.cat([edata, x_src, x_dst], -1)) # [n_edge, e_out_channels]
        ndata = scatter_mean(edata, row, dim=0, dim_size=x.size(0)) # [n_node, e_out_channels]
        x = self.n_lin(torch.cat([x,ndata], -1))  # [n_node, n_out_channels]

        return x, edata

class NodeEdgeGNN(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, e_in_channels, num_hidden=64, num_layers=3, activation="relu"):
        super().__init__()
        self.n_in_channels  = n_in_channels
        self.n_out_channels = n_out_channels
        self.e_in_channels  = e_in_channels
        self.num_hidden     = num_hidden
        self.num_layers     = num_layers
        self.layers         = nn.ModuleList([NodeEdgeConv(n_in_channels, num_hidden, e_in_channels, num_hidden)])
        for _ in range(num_layers-2):
            self.layers.append(NodeEdgeConv(num_hidden, num_hidden, num_hidden, num_hidden))
        # self.layers.append(NodeEdgeConv(num_hidden, n_out_channels, num_hidden, e_out_channels))
        self.layers.append(nn.Linear(num_hidden, n_out_channels))
        self.act = Activation(activation)
        self.reset_parameters()
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
    def forward(self, ndata, edata, edge_index):
        for layer in self.layers[:-1]:
            ndata, edata = layer(ndata, edata, edge_index)
        ndata = self.layers[-1](ndata)
        return ndata
    def __str__(self):
        return f"NodeEdgeGNN_{self.n_in_channels}_{self.n_out_channels}_{self.e_in_channels}_{self.num_hidden}_{self.num_layers}"
    def __repr__(self):
        return (f"NodeEdgeGNN(n_in_channels={self.n_in_channels}, "
                f"n_out_channels={self.n_out_channels}, "
                f"e_in_channels={self.e_in_channels}, "
                f"num_hidden={self.num_hidden}, "
                f"num_layers={self.num_layers})")


class GNNPipeline(nn.Module):
    def __init__(self, num_features, num_classes, 
                        args, encoder=None, 
                       decoder=None, 
                       **kwargs):
        super().__init__()
        if encoder is None:
            encoder = None  
            processor_num_features = num_features
        elif encoder == 'mlp':
            encoder = MLP(num_features, args.hidden_dim, args.hidden_dim, args.num_layers, args.act, norm=True)
            processor_num_features = args.hidden_dim
        else:
            raise NotImplementedError(f"encoder {encoder} is not implemented")
        if decoder is None:
            decoder = None 
            processor_num_classes = num_classes
        elif decoder == 'mlp':
            decoder = MLP(args.hidden_dim, args.num_classes, args.hidden_dim, args.num_layers, args.act)
            processor_num_classes = args.hidden_dim
        else:
            raise NotImplementedError(f"decoder {decoder} is not implemented")
        self.encoder = encoder
        self.decoder = decoder
        self.processor = init_model(processor_num_features, processor_num_classes, args, **kwargs)
      
    def forward(self, x, edge_index, **kwargs):
        x = self.encoder(x) if self.encoder is not None else x
        x = self.processor(x, edge_index, **kwargs)
        x = self.decoder(x) if self.decoder is not None else x
        return x
        
    def __str__(self):
        return f"GNNPipeline_{self.encoder}_{self.processor}_{self.decoder}"
    def __repr__(self):
        return (f"GNNPipeline(encoder={self.encoder}, "
                f"processor={self.processor}, "
                f"decoder={self.decoder})")


class BipartiteModel(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64, num_layers=2, act='prelu', gcn_norm=False):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers   = num_layers
        self.act          = act
        self.mlp1         = MLP(in_channels, hidden_channels, hidden_channels, num_layers, act)
        self.mlp2         = MLP(hidden_channels, out_channels, hidden_channels, num_layers, act) 
        self.gcn_norm     = gcn_norm
        self.reset_parameters()


    def reset_parameters(self):
        self.mlp1.reset_parameters()
        self.mlp2.reset_parameters()
    def forward(self, x_src, edge_index, num_dst_nodes):
        num_src_nodes = x_src.size(0)
        x_src = self.mlp1(x_src)

        
        adj = torch.sparse_coo_tensor(
            edge_index,
            torch.ones(edge_index.size(1)),
            (num_dst_nodes, num_src_nodes),
            dtype=x_src.dtype,
            device=x_src.device
        )

        if self.gcn_norm:
            raise NotImplementedError("gcn_norm is not implemented")

        x_dst = adj @ x_src

        x_dst = self.mlp2(x_dst)

        return x_dst
    
class BipartiteDenseModel(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64, num_layers=2, act='prelu', gcn_norm=True):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers   = num_layers
        self.act          = act
        self.mlp1         = MLP(in_channels, hidden_channels, hidden_channels, num_layers, act)
        self.mlp2         = MLP(hidden_channels, out_channels, hidden_channels, num_layers, act) 
        self.gcn_norm     = gcn_norm
        self.reset_parameters()


    def reset_parameters(self):
        self.mlp1.reset_parameters()
        self.mlp2.reset_parameters()
    def forward(self, x_src, num_dst_nodes):
        num_src_nodes = x_src.size(0)
        x_src = self.mlp1(x_src)

        adj = torch.ones([num_dst_nodes, num_src_nodes]).type(x_src.dtype).to(x_src.device)
      
        if self.gcn_norm  : 
            adj  *= 1/torch.sqrt(num_src_nodes * num_dst_nodes)

        x_dst = adj @ x_src

        x_dst = self.mlp2(x_dst)

        return x_dst
    
class BipartiteEdgeModel(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, e_in_channels, hidden_channels=64, num_layers=2, act='prelu'):
        super().__init__()
        self.n_in_channels  = n_in_channels
        self.n_out_channels = n_out_channels
        self.e_in_channel   = e_in_channels
        self.hidden_channels = hidden_channels
        self.num_layers   = num_layers
        self.act          = act
        self.n_mlp        = MLP(hidden_channels, n_out_channels, hidden_channels, num_layers, act)
        self.e_mlp        = MLP(e_in_channels + n_in_channels, hidden_channels, hidden_channels, num_layers, act)
        self.reset_parameters()
    def reset_parameters(self):
        self.n_mlp.reset_parameters()
        self.e_mlp.reset_parameters()
    def forward(self, x_src, edata, edge_index, num_dst_nodes):
        u, v = edge_index
        edata = self.e_mlp(torch.cat([x_src[v], edata], -1))
        ndata = self.n_mlp(scatter_mean(edata, v, dim=0, dim_size=num_dst_nodes))
        return ndata
    
class StaticCondenseRHS(nn.Module):
    def __init__(self):
        super().__init__() 
        # self.register_buffer("K_ou2in",K_ou2in.to_sparse_coo())

    
    def forward(self,  x, K_ou2in):
        K_ou2in = K_ou2in.to(x.device).type(x.dtype)
        dim = x.size(-1)
        x = K_ou2in @ x.reshape(-1, 1)
        x = x.reshape(-1, dim)
        return x
    
class EqualLoss(nn.Module):
    def __init__(self, num_losses):
        super().__init__()
        self.num_losses = num_losses
    def forward(self, *args):
        assert len(args) == self.num_losses, f"expect {self.num_losses} losses, but got {len(args)}"
        return sum(args)
    
class WeightLoss(nn.Module):
    def __init__(self, *weights):
        super().__init__()
        self.weights = weights
    def forward(self, *args):
        assert len(args) == len(self.weights), f"expect {len(self.weights)} losses, but got {len(args)}"
        return sum([w * loss for w, loss in zip(self.weights, args)])

class AutoWeightLoss(nn.Module):
    """
    https://github.com/Mikoto10032/AutomaticWeightedLoss/blob/master/AutomaticWeightedLoss.py
    """
    def __init__(self, num_losses):
        super().__init__()
        self.num_losses = num_losses
        self.weights = nn.Parameter(torch.ones(num_losses))

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.weights[i] ** 2) * loss + torch.log(1 + self.weights[i] ** 2)
        return loss_sum