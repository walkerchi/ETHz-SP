import torch
import torch_geometric as pyg



class Normalizer:
    def __call__(self, x):
        """ scale to [0,1]
        Parameters:
        -----------
            x:  torch.Tensor [n_batch, n_dim]
        
        Returns:
        --------
            x:  torch.Tensor [n_batch, n_dim]
                normalized x
        """
        _min, _max = x.min(dim=0, keepdim=True).values, x.max(dim=0, keepdim=True).values
        self._min, self._max = _min, _max
        return (x - _min) / (_max - _min)
    def undo(self, x):
        """
        Parameters:
        -----------
            x:  torch.Tensor [n_batch, n_dim]
        
        Returns:
        --------
            x:  torch.Tensor [n_batch, n_dim]
                unnormalized x
        """
        return x * (self._max - self._min) + self._min

    
    
def partite_graph(graph, mask):
    """
        Partite a graph A into A_11, A_12, A_21, A_22
        where A_11 is the subgraph of A induced by the nodes
        in the mask and A_22 is the subgraph of A induced by
        mask is the node mask for the subgraph 1.

        Parameters:
        -----------
            graph: pyg.data.Data
                The graph to be partited
            mask: torch.Tensor
                The node mask for the subgraph 1
        Returns:
        --------
        [[graph_11, graph_12], [graph_21, graph_22]]: list of pyg.data.Data
            graph_11: pyg.data.Data
                The subgraph 1
            graph_22: pyg.data.Data
                The subgraph 2
            graph_12: pyg.data.Data
                The edge graph between subgraph 1 and subgraph 2
            graph_21: pyg.data.Data
                The edge graph between subgraph 2 and subgraph 1
    """
    assert graph.num_nodes == len(mask)
    n_mask_1 = mask
    n_mask_2 = ~mask
    num_1  = mask.sum()
    num_2  = (~mask).sum()
    u_mask_1 = n_mask_1[graph.edge_index[0]]
    v_mask_1 = n_mask_1[graph.edge_index[1]]
    u_mask_2 = n_mask_2[graph.edge_index[0]]
    v_mask_2 = n_mask_2[graph.edge_index[1]]
    lnids     = torch.zeros(graph.num_nodes, dtype=torch.long)
    lnids[n_mask_1] = torch.arange(num_1)
    lnids[n_mask_2] = torch.arange(num_2)
    e_mask_11 = u_mask_1 & v_mask_1
    e_mask_22 = u_mask_2 & v_mask_2
    e_mask_12 = u_mask_2 & v_mask_1
    e_mask_21 = u_mask_1 & v_mask_2
    

    edge_index_11 = lnids[graph.edge_index[:, e_mask_11].flatten()].view(2,-1)
    graph_11 = pyg.data.Data(
        num_nodes  = num_1,
        edge_index = edge_index_11,
    )
    edge_index_22 = lnids[graph.edge_index[:, e_mask_22].flatten()].view(2,-1)
    graph_22 = pyg.data.Data(
        num_nodes  = num_2,
        edge_index = edge_index_22,
    )
    edge_index_12 = lnids[graph.edge_index[:, e_mask_12].flatten()].view(2,-1)
    graph_12 = pyg.data.Data(
        num_src_nodes = num_1,
        num_dst_nodes = num_2,
        edge_index = edge_index_12
    )
    edge_index_21 = lnids[graph.edge_index[:, e_mask_21].flatten()].view(2,-1)
    graph_21 = pyg.data.Data(
        num_src_nodes = num_2,
        num_dst_nodes = num_1,
        edge_index = edge_index_21
    )
    for key, value in graph:
        if key.startswith("n_"):
            graph_11[key] = value[n_mask_1]
            graph_22[key] = value[n_mask_2]
            graph_12[f"{key}_u"] = value[n_mask_1]
            graph_12[f"{key}_v"] = value[n_mask_2]
            graph_21[f"{key}_u"] = value[n_mask_2]
            graph_21[f"{key}_v"] = value[n_mask_1]
        elif key.startswith("e_"):
            graph_11[key] = value[e_mask_11]
            graph_22[key] = value[e_mask_22]
            graph_12[key] = value[e_mask_12]
            graph_21[key] = value[e_mask_21]
        elif key.startswith("g_"):
            graph_11[key] = value
            graph_22[key] = value
            graph_12[key] = value
            graph_21[key] = value
        else:
            pass
 
    return [
        [graph_11, graph_12],
        [graph_21, graph_22]
    ]


 