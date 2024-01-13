import os
import torch
import torch.nn as nn
import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
import torch_geometric.nn as gnn
from torch_geometric.transforms import NormalizeFeatures
from itertools import chain
from copy import deepcopy


from .model import init_model, \
                    GNNPipeline,\
                    BipartiteModel, BipartiteDenseModel, BipartiteEdgeModel, StaticCondenseRHS, EqualLoss, WeightLoss, AutoWeightLoss
from dataset import Truss, Triangle, Tetra

from .utils import Normalizer, partite_graph
from linear_elasticity import partite


def init_dataset(args):
    if args.dataset in Truss.NAMES:
        return Truss(
            name=args.dataset,
            E  = args.E,
            A  = args.A,
            n_grid = args.n_grid,
            support = args.m_support,
            d = args.d,
            a = args.a,
            p = args.p,
        )
    elif args.dataset in Triangle.NAMES:
        return Triangle(
            name=args.dataset,
            E  = args.E,
            nu = args.nu,
            d = args.d,
            a = args.a,
            b = args.b,
            p = args.p,
        )
    elif args.dataset in Tetra.NAMES:
        return Tetra(
            name=args.dataset,
            E  = args.E,
            nu = args.nu,
            d = args.d,
            a = args.a,
            b = args.b,
            p = args.p,
        )
    else:
        raise NotImplementedError(f"{args.dataset} is not implemented")


class OneTrainer:
    def __init__(self, args):
        self.dataset = init_dataset(args)
        self.graph   = self.dataset.as_graph()
        self.args    = args
        self.device  = torch.device(args.device)

        self.init_loss(args)
        self.init_model(self.graph, args)
        self.init_scheduler(args)
        self.init_path(args)
        graph = self.process_graph(self.dataset, self.graph, args)
        if isinstance(graph, (list,tuple)): 
            self.graph_i, self.graph_b2i = graph
        else:
            self.graph = graph
        self.to(self.device)

        # nids         = torch.arange(self.graph.num_nodes)
        # inner_nids   = nids[~self.graph.n_dirichlet_mask]
        # inner_nids   = inner_nids[torch.randperm(len(inner_nids))]
        # outer_nids   = nids[self.graph.n_dirichlet_mask]
        # n_train      = int(len(inner_nids) * args.train_ratio)
        # n_test       = int(len(inner_nids) * args.test_ratio)
        # n_valid      = len(inner_nids) - n_train - n_test
        # # n_train     += self.graph.n_dirichlet_mask.sum()
        
        # if args.loss == "equal":
        #     self.loss = EqualLoss(2)
        # elif args.loss == "weight":
        #     self.loss = WeightLoss(1, self.args.physical_weight)
        # elif args.loss == "auto_weight":
        #     self.loss = AutoWeightLoss(2)
        # else:
        #     raise NotImplementedError(f"loss type {args.loss} is not implemented")
        
        # if args.condense is not None:
        #     [graph_i,_], [graph_b2i,_] = partite_graph(self.graph, ~self.graph.n_dirichlet_mask)
            
        #     train_mask = torch.zeros(graph_i.num_nodes, dtype=torch.bool)
        #     valid_mask = torch.zeros(graph_i.num_nodes, dtype=torch.bool)
        #     test_mask  = torch.zeros(graph_i.num_nodes, dtype=torch.bool)
     
        #     train_mask[:n_train] = True
        #     valid_mask[n_train:n_train+n_valid] = True
        #     test_mask[n_train+n_valid:] = True

        #     self.graph_i   = graph_i
        #     self.graph_b2i = graph_b2i
        #     self.norm_x_i  = Normalizer()
        #     self.norm_x_b2i= Normalizer()
        #     if args.model == "NodeEdgeGNN":
        #         num_features     = graph_i.n_source_value.shape[1]
        #         num_classes      = graph_i.n_displacement.shape[1]
        #         num_edge_features= graph_i.n_pos.shape[1]
        #         self.model_i     = GNNPipeline(num_features, num_classes,  args, encoder=args.encoder, decoder=args.decoder, num_edge_features=num_edge_features)
        #         u, v             = graph_i.edge_index
        #         edata_i          = graph_i.n_pos[v] - graph_i.n_pos[u]
        #         y_i              = graph_i.n_displacement
        #         num_node_features= graph_i.n_displacement.shape[1]
        #         num_edge_features= graph_b2i.n_pos_u.shape[1]
        #         num_node_classes = graph_b2i.n_source_value_u.shape[1]
                
        #         if args.condense == "static":
        #             _, K_ou2in,_ = partite(self.dataset.solver.K_torch, self.graph.n_dirichlet_mask[:,None].repeat(1, 2).ravel())
        #             self.model_b2i   = StaticCondenseRHS(K_ou2in) 
        #         else:
        #             self.model_b2i   = BipartiteEdgeModel(num_node_features, num_node_classes, num_edge_features)
            
        #         u,v              = graph_b2i.edge_index
        #         edata_b2i        = graph_b2i.n_pos_u[v] - graph_b2i.n_pos_v[u]
        #         x_b2i            = graph_b2i.n_displacement_u
        #     else:
        #         num_features     = graph_i.n_pos.shape[1]+graph_i.n_source_value.shape[1] if args.use_pos else graph_i.n_source_value.shape[1]
        #         num_classes      = graph_i.n_displacement.shape[1]
        #         self.model_i     = GNNPipeline(num_features, num_classes,  args, encoder=args.encoder, decoder=args.decoder)
        #         y_i              = graph_i.n_displacement
        #         num_node_features= graph_i.n_displacement.shape[1] + graph_i.n_pos.shape[1] if args.use_pos else graph_i.n_displacement.shape[1]
        #         num_node_classes = graph_b2i.n_source_value_u.shape[1]

        #         if args.condense == "static":
        #             _, K_ou2in, _ = partite(self.dataset.solver.K_torch, self.graph.n_dirichlet_mask[:,None].repeat(1, 2).ravel())
        #             self.model_b2i   = StaticCondenseRHS(K_ou2in)
        #         elif args.condense == "nn_dense":
        #             self.model_b2i   = BipartiteDenseModel(num_node_features, num_node_classes, hidden_channels=64, num_layers=3)
        #         else:
        #             self.model_b2i   = BipartiteModel(num_node_features, num_node_classes, hidden_channels=64, num_layers=3)
        #         if args.use_pos:
        #             x_b2i            = torch.cat([
        #                 graph_b2i.n_pos_u,
        #                 graph_b2i.n_displacement_u,
        #                 ],-1)
        #         else:
        #             x_b2i            = graph_b2i.n_displacement_u
        #         edata_i, edata_b2i = None, None

        #     self.optimizer   = torch.optim.Adam(chain(
        #                                     self.model_i.parameters(),
        #                                     self.model_b2i.parameters(),
        #                                     self.loss.parameters())
        #                                     , lr=args.lr)
        #     self.graph_i.y = y_i.type(self.dtype)
        #     self.graph_i.edata = edata_i.type(self.dtype) if edata_i is not None else None
        #     self.graph_b2i.x = x_b2i.type(self.dtype)
        #     self.graph_b2i.edata = edata_b2i.type(self.dtype) if edata_b2i is not None else None
        #     self.graph_i.train_mask = train_mask
        #     self.graph_i.valid_mask = valid_mask
        #     self.graph_i.test_mask  = test_mask

        # else:

        #     train_mask   = torch.zeros(self.graph.num_nodes, dtype=torch.bool)
        #     valid_mask   = torch.zeros(self.graph.num_nodes, dtype=torch.bool)
        #     test_mask    = torch.zeros(self.graph.num_nodes, dtype=torch.bool)

        #     train_mask[inner_nids[:n_train]] = True
        #     train_mask[outer_nids] = True
        #     valid_mask[inner_nids[n_train:n_train+n_valid]] = True
        #     test_mask[inner_nids[n_train+n_valid:]] = True
        
        #     self.norm_x  = Normalizer()
        #     y            = self.graph.n_displacement
        #     if args.model == "NodeEdgeGNN":
        #         x        = self.graph.n_source_value
        #         u,v      = self.graph.edge_index
        #         edata    = self.graph.n_pos[u] - self.graph.n_pos[v]
        #         num_features    = x.shape[1]
        #         num_labels      = y.shape[1]
        #         num_edge_features = edata.shape[1]
        #         self.model      = GNNPipeline(num_features, num_labels, args, encoder=args.encoder, decoder=args.decoder, num_edge_features=num_edge_features)
        #     else:
        #         if args.use_pos:
        #             x        = torch.cat([
        #                 self.graph.n_source_value,
        #                 self.graph.n_pos,
        #             ],-1)
        #         else:
        #             x        = self.graph.n_source_value
        #         edata    = None
        #         num_features    = x.shape[1]
        #         num_labels      = y.shape[1]
        #         self.model      = GNNPipeline(num_features, num_labels, args, encoder=args.encoder, decoder=args.decoder)
        
        #     self.optimizer  = torch.optim.Adam(chain(
        #         self.model.parameters(),
        #         self.loss.parameters()), 
        #         lr=args.lr)
        
        #     self.graph.x          = x.type(self.dtype)
        #     self.graph.y          = y.type(self.dtype)
        #     self.graph.edata      = edata.type(self.dtype) if edata is not None else None
        #     self.graph.train_mask = train_mask 
        #     self.graph.valid_mask = valid_mask    
        #     self.graph.test_mask  = test_mask

        # self.to(self.device)
        # model_str = str(self.model_i) if args.condense is not None else str(self.model)
        # self.model_weight_path = (f"./.result/{self.args.trainer}/{self.args.dataset.split('_')[0]}/"
        #     f"{self.args.train_ratio}_{model_str}_model"
        #     f"{f'_{args.condense}_condense' if args.condense is not None else ''}"
        #     f"{f'_{args.loss}'}"
        #     f"{f'_{args.physical_loss}_phy' if args.physical_loss is not None else ''}.pth")
        # self.loss_image_path   = (f"./.result/{self.args.trainer}/{self.args.dataset.split('_')[0]}/"
        #     f"{self.args.train_ratio}_{model_str}_loss"
        #     f"{f'_{args.condense}_condense' if args.condense is not None else ''}"
        #     f"{f'_{args.loss}'}"
        #     f"{f'_{args.physical_loss}_phy' if args.physical_loss is not None else ''}.png")
       
        # if args.scheduler == "none":
        #     self.scheduler = None
        # elif args.scheduler == "cos":
        #     self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.epochs, eta_min=1e-6)
        # elif args.scheduler == "step":
        #     self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)
        # elif args.scheduler == "exp":
        #     self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.999)
        # else:
        #     raise NotImplementedError(f"Scheduler {args.scheduler} is not implemented")
        # os.makedirs(os.path.dirname(self.model_weight_path), exist_ok=True)
        # os.makedirs(os.path.dirname(self.loss_image_path), exist_ok=True)


    def process_graph(self, dataset, graph, args):
        nids         = torch.arange(graph.num_nodes)
        inner_nids   = nids[~graph.n_dirichlet_mask]
        inner_nids   = inner_nids[torch.randperm(len(inner_nids))]
        outer_nids   = nids[graph.n_dirichlet_mask]
        n_train      = int(len(inner_nids) * args.train_ratio)
        n_test       = int(len(inner_nids) * args.test_ratio)
        n_valid      = len(inner_nids) - n_train - n_test
        
        if args.condense is not None:
            [graph_i,_], [graph_b2i,_] = partite_graph(graph, ~graph.n_dirichlet_mask)
            
            train_mask = torch.zeros(graph_i.num_nodes, dtype=torch.bool)
            valid_mask = torch.zeros(graph_i.num_nodes, dtype=torch.bool)
            test_mask  = torch.zeros(graph_i.num_nodes, dtype=torch.bool)
     
            train_mask[:n_train] = True
            valid_mask[n_train:n_train+n_valid] = True
            test_mask[n_train+n_valid:] = True

            if args.model == "NodeEdgeGNN":
                u, v             = graph_i.edge_index
                edata_i          = graph_i.n_pos[v] - graph_i.n_pos[u]
                y_i              = graph_i.n_displacement
                
                if args.condense == "static":
                    _, K_ou2in,_ = partite(dataset.solver.K_torch, graph.n_dirichlet_mask[:,None].repeat(1, 2).ravel())
                    graph_b2i.K  = K_ou2in
            
                u,v              = graph_b2i.edge_index
                edata_b2i        = graph_b2i.n_pos_u[v] - graph_b2i.n_pos_v[u]
                x_b2i            = graph_b2i.n_displacement_u
            else:
                y_i              = graph_i.n_displacement
   
                if args.condense == "static":
                    _, K_ou2in, _ = partite(dataset.solver.K_torch, graph.n_dirichlet_mask[:,None].repeat(1, 2).ravel())
                    graph_b2i.K  = K_ou2in
                if args.use_pos:
                    x_b2i            = torch.cat([
                        graph_b2i.n_pos_u,
                        graph_b2i.n_displacement_u,
                        ],-1)
                else:
                    x_b2i            = graph_b2i.n_displacement_u
                edata_i, edata_b2i = None, None

            graph_i.y = y_i.type(self.dtype)
            graph_i.edata = edata_i.type(self.dtype) if edata_i is not None else None
            graph_b2i.x = x_b2i.type(self.dtype)
            graph_b2i.edata = edata_b2i.type(self.dtype) if edata_b2i is not None else None
            graph_i.train_mask = train_mask
            graph_i.valid_mask = valid_mask
            graph_i.test_mask  = test_mask
            return [graph_i, graph_b2i]
        else:

            train_mask   = torch.zeros(self.graph.num_nodes, dtype=torch.bool)
            valid_mask   = torch.zeros(self.graph.num_nodes, dtype=torch.bool)
            test_mask    = torch.zeros(self.graph.num_nodes, dtype=torch.bool)

            train_mask[inner_nids[:n_train]] = True
            train_mask[outer_nids] = True
            valid_mask[inner_nids[n_train:n_train+n_valid]] = True
            test_mask[inner_nids[n_train+n_valid:]] = True
        
            y            = self.graph.n_displacement
            if args.model == "NodeEdgeGNN":
                x        = self.graph.n_source_value
                u,v      = self.graph.edge_index
                edata    = self.graph.n_pos[u] - self.graph.n_pos[v]
            else:
                if args.use_pos:
                    x        = torch.cat([
                        self.graph.n_source_value,
                        self.graph.n_pos,
                    ],-1)
                else:
                    x        = self.graph.n_source_value
                edata    = None
        
            self.optimizer  = torch.optim.Adam(chain(
                self.model.parameters(),
                self.loss.parameters()), 
                lr=args.lr)
        
            graph.x          = x.type(self.dtype)
            graph.y          = y.type(self.dtype)
            graph.edata      = edata.type(self.dtype) if edata is not None else None
            graph.train_mask = train_mask 
            graph.valid_mask = valid_mask    
            graph.test_mask  = test_mask
            return graph

    def init_loss(self, args):
        # init loss 
        if args.loss == "equal":
            self.loss = EqualLoss(2)
        elif args.loss == "weight":
            self.loss = WeightLoss(1, args.physical_weight)
        elif args.loss == "auto_weight":
            self.loss = AutoWeightLoss(2)
        else:
            raise NotImplementedError(f"loss type {args.loss} is not implemented")

    def init_model(self, graph, args):
    
        # init model
        if args.condense is not None:

            if args.model == "NodeEdgeGNN":
                num_features     = graph.n_source_value.shape[1]
                num_classes      = graph.n_displacement.shape[1]
                num_edge_features= graph.n_pos.shape[1]
                num_node_features= graph.n_displacement.shape[1]
                num_edge_features= graph.n_pos_u.shape[1]
                num_node_classes = graph.n_source_value_u.shape[1]
                self.model_i     = GNNPipeline(num_features, num_classes,  args, encoder=args.encoder, decoder=args.decoder, num_edge_features=num_edge_features)
                
                if args.condense == "static":
                    self.model_b2i   = StaticCondenseRHS() 
                else:
                    self.model_b2i   = BipartiteEdgeModel(num_node_features, num_node_classes, num_edge_features)
            
            else:
                num_features     = graph.n_pos.shape[1]+graph.n_source_value.shape[1] if args.use_pos else graph.n_source_value.shape[1]
                num_classes      = graph.n_displacement.shape[1]
                self.model_i     = GNNPipeline(num_features, num_classes,  args, encoder=args.encoder, decoder=args.decoder)
                num_node_features= graph.n_displacement.shape[1] + graph.n_pos.shape[1] if args.use_pos else graph.n_displacement.shape[1]
                num_node_classes = graph.n_source_value.shape[1]

                if args.condense == "static":
                    self.model_b2i   = StaticCondenseRHS()
                elif args.condense == "nn_dense":
                    self.model_b2i   = BipartiteDenseModel(num_node_features, num_node_classes, hidden_channels=64, num_layers=3)
                else:
                    self.model_b2i   = BipartiteModel(num_node_features, num_node_classes, hidden_channels=64, num_layers=3)

            self.optimizer   = torch.optim.Adam(chain(
                                            self.model_i.parameters(),
                                            self.model_b2i.parameters(),
                                            self.loss.parameters())
                                            , lr=args.lr)

        else:

          
            if args.model == "NodeEdgeGNN":
                num_features    = graph.n_source_value.shape[1]
                num_labels      = graph.n_displacement.shape[1]
                num_edge_features = graph.n_pos_v.shape[1]
                self.model      = GNNPipeline(num_features, num_labels, args, encoder=args.encoder, decoder=args.decoder, num_edge_features=num_edge_features)
            else:
                if args.use_pos:
                    num_features    = graph.n_source_value.shape[1] + graph.n_pos.shape[1]
                else:
                    num_features    = graph.n_source_value.shape[1]
                num_labels      = graph.n_displacement.shape[1]
                self.model      = GNNPipeline(num_features, num_labels, args, encoder=args.encoder, decoder=args.decoder)
        
            self.optimizer  = torch.optim.Adam(chain(
                self.model.parameters(),
                self.loss.parameters()), 
                lr=args.lr)
        
    def init_scheduler(self, args):
        if args.scheduler == "none":
            self.scheduler = None
        elif args.scheduler == "cos":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.epochs, eta_min=1e-6)
        elif args.scheduler == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)
        elif args.scheduler == "exp":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.999)
        else:
            raise NotImplementedError(f"Scheduler {args.scheduler} is not implemented")
        
    def init_path(self, args):
        model_str = str(self.model_i) if args.condense is not None else str(self.model)
        self.model_weight_path = (f"./.result/{self.args.trainer}/{self.args.dataset.split('_')[0]}/"
            f"{self.args.train_ratio}_{model_str}_model"
            f"{f'_{args.condense}_condense' if args.condense is not None else ''}"
            f"{f'_{args.loss}'}"
            f"{f'_{args.physical_loss}_phy' if args.physical_loss is not None else ''}.pth")
        self.loss_image_path   = (f"./.result/{self.args.trainer}/{self.args.dataset.split('_')[0]}/"
            f"{self.args.train_ratio}_{model_str}_loss"
            f"{f'_{args.condense}_condense' if args.condense is not None else ''}"
            f"{f'_{args.loss}'}"
            f"{f'_{args.physical_loss}_phy' if args.physical_loss is not None else ''}.png")
        os.makedirs(os.path.dirname(self.model_weight_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.loss_image_path), exist_ok=True)

    def predict(self, graph=None, graph_i=None, graph_b2i=None):
       
        if self.args.condense is not None:
            if self.args.condense == "static":
                f = - self.model_b2i(graph_b2i.n_displacement_u, graph_b2i.K)
            elif self.args.model == "NodeEdgeGNN":
                f = self.model_b2i(graph_b2i.x, graph_b2i.edata, graph_b2i.edge_index, graph_b2i.num_dst_nodes) # outer -> inner
            elif self.args.condense == "nn_dense":
                f = self.model_b2i(graph_b2i.x,  graph_b2i.num_dst_nodes) # outer -> inner
            else:
                f = self.model_b2i(graph_b2i.x, graph_b2i.edge_index, graph_b2i.num_dst_nodes) # outer -> inner
            
            n_source_value = f + graph_i.n_source_value
            
           
            if self.args.model == "MLP":
                x = torch.cat([
                    graph_i.n_pos,
                    n_source_value,
                ],-1).type(next(iter(self.model_i.parameters())).dtype)
                u = self.model_i(x)
            elif self.args.model == "NodeEdgeGNN":
                x = n_source_value.type(graph_i.edata.dtype)
                u = self.model_i(x, graph_i.edata, graph_i.edge_index)
            elif self.args.model == "GraphUNet":
                pos = graph_i.n_pos
                if self.args.use_pos:
                    x = torch.cat([
                        pos,
                        n_source_value,
                    ],-1).type(next(iter(self.model_i.parameters())).dtype)
                else:
                    x = n_source_value.type(next(iter(self.model_i.parameters())).dtype)
                u = self.model_i(x, graph_i.edge_index, pos=pos)
            else:
                if self.args.use_pos:
                    x = torch.cat([
                        graph_i.n_pos,
                        n_source_value,
                    ],-1).type(next(iter(self.model_i.parameters())).dtype)
                else:
                    x = n_source_value.type(next(iter(self.model_i.parameters())).dtype)
            
                u = self.model_i(x, graph_i.edge_index)

        else:
            if self.args.model == "MLP":
                u = self.model(graph.x)
            elif self.args.model == "NodeEdgeGNN":
                u = self.model(graph.x, graph.edata, graph.edge_index)
            elif self.args.model == "GraphUNet":
                u = self.model(graph.x, graph.edge_index, pos= graph.n_pos)
            else:
                u = self.model(graph.x, graph.edge_index)

        return u
    
    def compute_loss(self, mode="train", graph=None, graph_i=None, graph_b2i=None):
        graph = self.graph if graph is None else graph
        graph_i = self.graph_i if graph_i is None else graph_i
        graph_b2i = self.graph_b2i if graph_b2i is None else graph_b2i
        assert mode in ["train", "valid", "test", "all"]
        mask_key = f"{mode}_mask"
        if self.args.condense is not None:
            u = self.predict(graph, graph_i, graph_b2i)
            y = graph_i.y
            if mode != "all":
                u_labeled = u[graph_i[mask_key]]
                y_labeled = y[graph_i[mask_key]]
            else:
                u_labeled = u 
                y_labeled = y
        else:
            u = self.predict(graph, graph_i, graph_b2i)
            y = graph.y
            if mode != "all":
                u_labeled = u[graph[mask_key]]
                y_labeled = y[graph[mask_key]]
            else:
                u_labeled = u 
                y_labeled = y
            
        if u_labeled.numel() == 0:
            assert self.args.physical_loss is not None, f"no data is calculated as loss, it's only allowed when physical_loss is not None"
            loss = torch.tensor(0.0)
        else:
            loss = torch.nn.MSELoss()(u_labeled, y_labeled)

        if self.args.physical_loss is not None and mode == "train":
            if self.args.condense is not None:
                u_global = torch.zeros_like(graph.n_pos, dtype=self.dtype, device=u.device)
                u_global[~graph.n_dirichlet_mask] += u 
                u_global[graph.n_dirichlet_mask] += graph_b2i.n_displacement_u.to(u.device)
            else:
                u_global = u
            f = graph.n_source_value.type(self.dtype).to(u.device)
            physical_loss = self.dataset.compute_residual(u_global, f,  mse=True, form=self.args.physical_loss) # (n_nodes, 1)
            loss = {
                "data loss": loss,
                "physical loss": physical_loss 
            }
           
        return loss
    
    def save(self):
        if self.args.condense is not None:
            weight = (self.model_i.state_dict(), self.model_b2i.state_dict())
            torch.save(weight, self.model_weight_path)
        else:
            torch.save(self.model.state_dict(), self.model_weight_path)
        return self
    
    def load(self):
        if self.args.condense is not None:
            weight = torch.load(self.model_weight_path)
            self.model_i.load_state_dict(weight[0])
            if not self.args.condense == "static":
                self.model_b2i.load_state_dict(weight[1])
        else:
            self.model.load_state_dict(torch.load(self.model_weight_path))
        return self
    
    def state_dict(self):
        if self.args.condense is not None:
            return (self.model_i.state_dict(), self.model_b2i.state_dict())
        else:
            return self.model.state_dict()
        
    def load_state_dict(self, state_dict):
        if self.args.condense is not None:
            self.model_i.load_state_dict(state_dict[0])
            if not self.args.condense == "static":
                self.model_b2i.load_state_dict(state_dict[1])
        else:
            self.model.load_state_dict(state_dict)
        return self
    
    def train_mode(self):
        if self.args.condense is not None:
            self.model_i.train()
            self.model_b2i.train()  
        else:
            self.model.train()

    def eval_mode(self):
        if self.args.condense is not None:
            self.model_i.eval()
            self.model_b2i.eval()
        else:
            self.model.eval()

    def to(self, device):
        if self.args.condense is not None:
            self.model_i.to(device)
            self.model_b2i.to(device)
            self.graph_i.to(device)
            self.graph_b2i.to(device)
        else:
            self.model.to(device)
            self.graph.to(device)
        self.loss.to(device)
        return self
    
    @property
    def dtype(self):
        if self.args.condense is not None:
            return next(iter(self.model_i.parameters())).dtype
        else:
            return next(iter(self.model.parameters())).dtype

    def fit(self):

        self.train_mode()
        train_losses    = {}
        valid_losses    = {}

        best_model, best_epoch, best_loss = None, None, float('inf')

        pbar = tqdm.tqdm(range(self.args.epochs))
        for ep in pbar:
            self.optimizer.zero_grad()

            loss = self.compute_loss(mode="train")

            if isinstance(loss, dict):
                for key, value in loss.items():
                    train_losses.setdefault(key, []).append(value.item())
                loss = self.loss(*tuple(loss.values()))
                train_losses.setdefault("total loss", []).append(loss.item())
                loss.backward()
            elif isinstance(loss, torch.Tensor):
                loss.backward()
                train_losses.setdefault("loss", []).append(loss.item())
            else:
                raise NotImplementedError(f"loss type {type(loss)} is not implemented")
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            
            if (ep + 1) % self.args.eval_every_eps == 0:
                self.eval_mode()
                with torch.no_grad():
                    loss = self.compute_loss(mode="valid")
                self.train_mode()

                if isinstance(loss, dict):
                    for key, value in loss.items():
                        valid_losses.setdefault(key, []).append(value.item())
                    loss = self.loss(*tuple(loss.values())).item()
                    valid_losses.setdefault("total loss", []).append(loss)
                elif isinstance(loss, torch.Tensor):
                    valid_losses.setdefault("loss", []).append(loss.item())
                    loss = loss.item()
                if loss < best_loss:
                    best_loss = loss
                    best_epoch = ep
                    best_model = deepcopy(self.state_dict())
                pbar.set_postfix(
                    loss=loss,
                )

        self.load_state_dict(best_model)
        self.save()

        loss = self.test(on_all_nodes=False)

        self.plot_loss(train_losses, valid_losses, best_epoch, loss)

    def plot_loss(self, train_losses,  valid_losses, best_epoch, test_loss):
        # plot the train/valid/test losses best validation point in one figure
        fig, ax = plt.subplots(figsize=(10, 6))
        for key,value in train_losses.items():
            ax.plot(np.arange(len(value)), value, label=f'train_{key}', linestyle=":")
        for key,value in valid_losses.items():
            ax.plot(np.arange(len(value)) * self.args.eval_every_eps, value, label=f'valid_{key}', linestyle="-")
        ax.scatter([self.args.epochs], [test_loss], label=f'test', marker="x", color="red")
        ax.axvline(best_epoch, linestyle="--", color="green", label="best validate")
        # ax.scatter([best_epoch], [best_loss], label='best validate', marker="d", color="green")

        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.set_title('Losses')
        ax.set_yscale('log')
        ax.legend()
        fig.savefig(self.loss_image_path)
    
    def test(self, on_all_nodes=True, on_cpu=True):
        if on_cpu:
            self.to("cpu")

        if not os.path.exists(self.model_weight_path):
            raise FileNotFoundError(f"{self.model_weight_path}  , Try to train first")
        self.load()
        self.eval_mode()
        with torch.no_grad():
            loss        = self.compute_loss(mode="all" if on_all_nodes else "test")

            if isinstance(loss, dict):
                loss = loss['data loss'].item()
            elif isinstance(loss, torch.Tensor):
                loss = loss.item()
            else:
                raise NotImplementedError(f"loss type {type(loss)} is not implemented")
        return loss



class MultiTrainer(OneTrainer):
    def __init__(self, args):
        self.train_datasets = []
        source = "A"
        for _ in range(args.n_samples):
            for load in ["sin1","cos1", "sin2", "cos2", "sin4", "cos4"]:
                for source, boundary in [("A","B"),("A","C"),("A","D"),("A","B+C"),("A","B+D"),("A","C+D"),
                                         ("B","A"),("B","C"),("B","D"),("B","A+C"),("B","A+D"),("B","C+D"),
                                         ("C","A"),("C","B"),("C","D"),("C","A+B"),("C","A+D"),("C","B+D"),
                                         ("D","A"),("D","B"),("D","C"),("D","A+B"),("D","A+C"),("D","B+C")]:
                    self.train_datasets.append(Triangle(
                        name=f"quadrilateral_{load}_{boundary}_{source}",
                        E  = args.E,
                        nu = args.nu,
                        d = args.d,
                        a = args.a,
                        b = args.b,
                        p = args.p,
                    ))
                for source, boundary  in [("A","B"),("A","C"),
                                          ("B","A"),("B","C"),
                                         ("C","A"),("C","B")]:
                    self.train_datasets.append(Triangle(
                        name=f"triangle_{load}_{boundary}_{source}",
                        E  = args.E,
                        nu = args.nu,
                        d = args.d,
                        a = args.a,
                        b = args.b,
                        p = args.p,
                    ))

        # self.train_datasets.append(init_dataset(args))

        print(f"""
              
              {len(self.train_datasets)} training samples
              
                """)
        self.test_dataset = init_dataset(args)
        
        self.train_graphs  = [dataset.as_graph() for dataset in self.train_datasets]
        self.test_graph    = self.test_dataset.as_graph()
        self.args    = args
        self.device  = torch.device(args.device)

        self.init_loss(args)
        self.init_model(self.train_graphs[0], args)
        self.init_scheduler(args)
        self.init_path(args)
        if args.condense is not None:
            train_graphs = [self.process_graph(dataset, graph, args) for dataset,graph in zip(self.train_datasets, self.train_graphs)]
            self.train_graphs_i = [graph[0] for graph in train_graphs]
            self.train_graphs_b2i = [graph[1] for graph in train_graphs]
            self.test_graph_i, self.test_graph_b2i = self.process_graph(self.test_dataset, self.test_graph, args)
        else:
            self.train_graphs = [self.process_graph(dataset, graph, args) for dataset,graph in zip(self.train_datasets, self.train_graphs)]
            self.test_graph   = self.process_graph(self.test_dataset, self.test_graph, args)
        self.to(self.device)

    def compute_loss(self, mode="train", dataset=None, graph=None, graph_i=None, graph_b2i=None):
        assert mode in ["train", "valid", "test", "all"]
        mask_key = f"{mode}_mask"

        losses = []
        datasets = self.train_datasets if dataset is None else [dataset]
        graphs = self.train_graphs if graph is None else [graph]
        graphs_i = self.train_graphs_i if graph_i is None else [graph_i]
        graphs_b2i = self.train_graphs_b2i if graph_b2i is None else [graph_b2i]


        VALID_PHY = False
        # breakpoint()
        for i in range(len(graphs)):
            if self.args.condense is not None:
                u = self.predict(graphs[i], graphs_i[i], graphs_b2i[i])
                y = graphs_i[i].y
                if mode != "all":
                    u_labeled = u[graphs_i[i][mask_key]]
                    y_labeled = y[graphs_i[i][mask_key]]
                else:
                    u_labeled = u 
                    y_labeled = y
            else:
                u = self.predict(graphs[i], graphs_i[i], graphs_b2i[i])
                y = graphs[i].y
                if mode != "all":
                    u_labeled = u[graphs[i][mask_key]]
                    y_labeled = y[graphs[i][mask_key]]
                else:
                    u_labeled = u 
                    y_labeled = y
            
            if u_labeled.numel() == 0 or (VALID_PHY and mode=="valid"):
                assert self.args.physical_loss is not None, f"no data is calculated as loss, it's only allowed when physical_loss is not None"
                loss = torch.tensor(0.0)
            else:
                loss = torch.nn.MSELoss()(u_labeled, y_labeled)

            if self.args.physical_loss is not None and (mode == "train" or (VALID_PHY and mode=="valid")):
                if self.args.condense is not None:
                    u_global = torch.zeros_like(graphs[i].n_pos, dtype=self.dtype, device=u.device)
                    u_global[~graphs[i].n_dirichlet_mask] += u 
                    u_global[graphs[i].n_dirichlet_mask] += graphs_b2i[i].n_displacement_u.to(u.device)
                else:
                    u_global = u
                f = graphs[i].n_source_value.type(self.dtype).to(u.device)
                physical_loss = datasets[i].compute_residual(u_global, f,  mse=True, form=self.args.physical_loss) # (n_nodes, 1)
                loss = {
                    "data loss": loss,
                    "physical loss": physical_loss 
                }
            losses.append(loss)

        if isinstance(losses[0], dict):
            mean_loss = {}
            for key in losses[0].keys():
                mean_loss[key] = torch.stack([loss[key] for loss in losses]).mean()
        else:
            mean_loss = torch.stack(losses).mean()

        return mean_loss
             
    def to(self, device):
        if self.args.condense is not None:
            self.model_i.to(device)
            self.model_b2i.to(device)
            for i in range(len(self.train_graphs_i)):
                self.train_graphs_i[i].to(device)
                self.train_graphs_b2i[i].to(device)
            self.test_graph_i.to(device)
            self.test_graph_b2i.to(device)
        else:
            self.model.to(device)
            for  i in range(len(self.train_graphs_i)):
                self.train_graphs[i].to(device)
            self.test_graph.to(device)
        self.loss.to(device)
        return self
    
    def test(self, on_all_nodes=True, on_cpu=True):
        if on_cpu:
            self.to("cpu")

        if not os.path.exists(self.model_weight_path):
            raise FileNotFoundError(f"{self.model_weight_path}  , Try to train first")
        self.load()
        self.eval_mode()
        with torch.no_grad():
            loss        = self.compute_loss(mode="all" if on_all_nodes else "test",
                                            graph = self.test_graph,
                                            graph_i = self.test_graph_i,
                                            graph_b2i = self.test_graph_b2i)

            if isinstance(loss, dict):
                loss = loss['data loss'].item()
            elif isinstance(loss, torch.Tensor):
                loss = loss.item()
            else:
                raise NotImplementedError(f"loss type {type(loss)} is not implemented")
        return loss
    