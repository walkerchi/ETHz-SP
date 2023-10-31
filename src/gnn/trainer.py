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
from typing import Any

from .model import init_model, BipartiteModel, BipartiteEdgeModel
from dataset import SphericalShell, Truss

from .utils import Normalizer, partite_graph

def init_dataset(args):
 
    if args.dataset == "spherical_shell":
        return SphericalShell(
            d = args.d,
            E = args.E,
            nu = args.nu,
            a = args.a,
            b = args.b,
            p = args.p,
        )
    elif args.dataset == "truss":
        return Truss(
            name="bridge.pratt",	
            E  = args.E,
            nu = args.nu,
            n_grid = args.n_grid,
            support = args.m_support,
        )
    else:
        raise NotImplementedError()



class OneTrainer:
    def __init__(self, args):
        self.dataset = init_dataset(args)
        self.graph   = self.dataset.as_graph()
        self.args    = args
        self.device  = torch.device(args.device)

        nids         = torch.arange(self.graph.num_nodes)
        inner_nids   = nids[~self.graph.n_dirichlet_mask]
        inner_nids   = inner_nids[torch.randperm(len(inner_nids))]
        n_train      = int(len(inner_nids) * args.train_ratio)
        n_test       = int(len(inner_nids) * args.test_ratio)
        n_valid      = len(inner_nids) - n_train - n_test
        n_train     += self.graph.n_dirichlet_mask.sum()
        
        if args.use_condense:
            [graph_i,_], [graph_b2i,_] = partite_graph(self.graph, ~self.graph.n_dirichlet_mask)
            
            train_mask = torch.zeros(graph_i.num_nodes, dtype=torch.bool)
            valid_mask = torch.zeros(graph_i.num_nodes, dtype=torch.bool)
            test_mask  = torch.zeros(graph_i.num_nodes, dtype=torch.bool)

            train_mask[:n_train] = True
            valid_mask[n_train:n_train+n_valid] = True
            test_mask[n_train+n_valid:] = True

            self.graph_i   = graph_i
            self.graph_b2i = graph_b2i
            self.norm_x_i  = Normalizer()
            self.norm_x_b2i= Normalizer()
            if args.model == "NodeEdgeGNN":
                num_features     = graph_i.n_source_value.shape[1]
                num_classes      = graph_i.n_displacement.shape[1]
                num_edge_features= graph_i.n_pos.shape[1]
                self.model_i     = init_model(num_features, num_classes,  args, num_edge_features=num_edge_features)
                u, v             = graph_i.edge_index
                edata_i          = graph_i.n_pos[v] - graph_i.n_pos[u]
                y_i              = graph_i.n_displacement
                num_node_features= graph_i.n_displacement.shape[1]
                num_edge_features= graph_b2i.n_pos_u.shape[1]
                num_node_classes = graph_b2i.n_source_value_u.shape[1]
                self.model_b2i   = BipartiteEdgeModel(num_node_features, num_node_classes, num_edge_features)
                u,v              = graph_b2i.edge_index
                edata_b2i        = graph_b2i.n_pos_u[v] - graph_b2i.n_pos_v[u]
                x_b2i            = graph_b2i.n_displacement_u
            else:
                num_features     = graph_i.n_pos.shape[1]+graph_i.n_source_value.shape[1]
                num_classes      = graph_i.n_displacement.shape[1]
                self.model_i     = init_model(num_features, num_classes,  args)
                y_i              = graph_i.n_displacement
                num_node_features= graph_i.n_displacement.shape[1] + graph_i.n_pos.shape[1]
                num_node_classes = graph_b2i.n_source_value_u.shape[1]
                self.model_b2i   = BipartiteModel(num_node_features, num_node_classes, hidden_channels=64, num_layers=3)
                x_b2i            = torch.cat([
                    graph_b2i.n_pos_u,
                    graph_b2i.n_displacement_u,
                    ],-1)
                edata_i, edata_b2i = None, None

            self.optimizer   = torch.optim.Adam(chain(
                                            self.model_i.parameters(),
                                            self.model_b2i.parameters())
                                            , lr=args.lr)
            self.graph_i.y = y_i.type(self.dtype)
            self.graph_i.edata = edata_i.type(self.dtype) if edata_i is not None else None
            self.graph_b2i.x = x_b2i.type(self.dtype)
            self.graph_b2i.edata = edata_b2i.type(self.dtype) if edata_b2i is not None else None
            self.graph_i.train_mask = train_mask
            self.graph_i.valid_mask = valid_mask
            self.graph_i.test_mask  = test_mask

        else:

            train_mask   = torch.zeros(self.graph.num_nodes, dtype=torch.bool)
            valid_mask   = torch.zeros(self.graph.num_nodes, dtype=torch.bool)
            test_mask    = torch.zeros(self.graph.num_nodes, dtype=torch.bool)

            train_mask[inner_nids[:n_train]] = True
            valid_mask[inner_nids[n_train:n_train+n_valid]] = True
            test_mask[inner_nids[n_train+n_valid:]] = True
        
            self.norm_x  = Normalizer()
            y            = self.graph.n_displacement
            if args.model == "NodeEdgeGNN":
                x        = self.graph.n_source_value
                u,v      = self.graph.edge_index
                edata    = self.graph.n_pos[u] - self.graph.n_pos[v]
                num_features    = x.shape[1]
                num_labels      = y.shape[1]
                num_edge_features = edata.shape[1]
                self.model      = init_model(num_features, num_labels, args, num_edge_features=num_edge_features)
            else:
                x        = torch.cat([
                    self.graph.n_source_value,
                    self.graph.n_pos,
                ],-1)
                edata    = None
                num_features    = x.shape[1]
                num_labels      = y.shape[1]
                self.model      = init_model(num_features, num_labels,  args)
        
            self.optimizer  = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        
            self.graph.x          = x.type(self.dtype)
            self.graph.y          = y.type(self.dtype)
            self.graph.edata      = edata.type(self.dtype) if edata is not None else None
            self.graph.train_mask = train_mask 
            self.graph.valid_mask = valid_mask    
            self.graph.test_mask  = test_mask

        
        self.to(self.device)
        model_str = str(self.model_i) if args.use_condense else str(self.model)
        self.model_weight_path = f"./.result/OneTrainer_{self.args.dataset}_{self.args.train_ratio}_{model_str}_model{'_condense' if args.use_condense else ''}.pth"
        self.loss_image_path   = f"./.result/OneTrainer_{self.args.dataset}_{self.args.train_ratio}_{model_str}_loss{'_condense' if args.use_condense else ''}.png"

    def predict(self):

        if self.args.use_condense:
            if self.args.model == "NodeEdgeGNN":
                f = self.model_b2i(self.graph_b2i.x, self.graph_b2i.edata, self.graph_b2i.edge_index, self.graph_b2i.num_dst_nodes) # outer -> inner
            else:
                f = self.model_b2i(self.graph_b2i.x, self.graph_b2i.edge_index, self.graph_b2i.num_dst_nodes) # outer -> inner
            n_source_value = f + self.graph_i.n_source_value
            
           
            if self.args.model == "MLP":
                x = torch.cat([
                    self.graph_i.n_pos,
                    n_source_value,
                ],-1).type(next(iter(self.model_i.parameters())).dtype)
                u = self.model_i(x)
            elif self.args.model == "NodeEdgeGNN":
                x = n_source_value.type(self.graph_i.edata.dtype)
                u = self.model_i(x, self.graph_i.edata, self.graph_i.edge_index)
            else:
                x = torch.cat([
                    self.graph_i.n_pos,
                    n_source_value,
                ],-1).type(next(iter(self.model_i.parameters())).dtype)
                u = self.model_i(x, self.graph_i.edge_index)

        else:
            if self.args.model == "MLP":
                u = self.model(self.graph.x)
            elif self.args.model == "NodeEdgeGNN":
                u = self.model(self.graph.x, self.graph.edata, self.graph.edge_index)
            else:
                u = self.model(self.graph.x, self.graph.edge_index)

        return u
    
    def compute_loss(self, mode="train"):
        assert mode in ["train", "valid", "test", "all"]
        mask_key = f"{mode}_mask"
        if self.args.use_condense:
            u = self.predict()
            y = self.graph_i.y
            if mode != "all":
                u = u[self.graph_i[mask_key]]
                y = y[self.graph_i[mask_key]]
        else:
            u = self.predict()
            y = self.graph.y
            if mode != "all":
                u = u[self.graph[mask_key]]
                y = y[self.graph[mask_key]]
        loss = torch.nn.MSELoss()(u, y)
        return loss
    
    def save(self):
        if self.args.use_condense:
            weight = (self.model_i.state_dict(), self.model_b2i.state_dict())
            torch.save(weight, self.model_weight_path)
        else:
            torch.save(self.model.state_dict(), self.model_weight_path)
        return self
    
    def load(self):
        if self.args.use_condense:
            weight = torch.load(self.model_weight_path)
            self.model_i.load_state_dict(weight[0])
            self.model_b2i.load_state_dict(weight[1])
        else:
            self.model.load_state_dict(torch.load(self.model_weight_path))
        return self
    
    def state_dict(self):
        if self.args.use_condense:
            return (self.model_i.state_dict(), self.model_b2i.state_dict())
        else:
            return self.model.state_dict()
        
    def load_state_dict(self, state_dict):
        if self.args.use_condense:
            self.model_i.load_state_dict(state_dict[0])
            self.model_b2i.load_state_dict(state_dict[1])
        else:
            self.model.load_state_dict(state_dict)
        return self
    
    def train_mode(self):
        if self.args.use_condense:
            self.model_i.train()
            self.model_b2i.train()  
        else:
            self.model.train()

    def eval_mode(self):
        if self.args.use_condense:
            self.model_i.eval()
            self.model_b2i.eval()
        else:
            self.model.eval()

    def to(self, device):
        if self.args.use_condense:
            self.model_i.to(device)
            self.model_b2i.to(device)
            self.graph_i.to(device)
            self.graph_b2i.to(device)
        else:
            self.model.to(device)
            self.graph.to(device)
        return self
    
    @property
    def dtype(self):
        if self.args.use_condense:
            return next(iter(self.model_i.parameters())).dtype
        else:
            return next(iter(self.model.parameters())).dtype

    def fit(self):

        self.train_mode()
        train_losses    = []
        valid_losses    = []

        best_model, best_epoch, best_loss = None, None, float('inf')

        pbar = tqdm.tqdm(range(self.args.epochs))
        for ep in pbar:
            self.optimizer.zero_grad()

            loss = self.compute_loss(mode="train")
        
            loss.backward()
            train_losses.append(loss.item())

            self.optimizer.step()
            
            if (ep + 1) % self.args.eval_every_eps == 0:
                self.eval_mode()
                with torch.no_grad():
                    loss = self.compute_loss(mode="valid")
                self.train_mode()
                loss = loss.item()
                valid_losses.append(loss)
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

        # plot the train/valid/test losses best validation point in one figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(np.arange(len(train_losses)), train_losses, label='train', linestyle=":")
        ax.plot(np.arange(len(valid_losses)) * self.args.eval_every_eps, valid_losses, label='valid', linestyle="-")
        ax.scatter([self.args.epochs], [loss], label='test', marker="x", color="red")
        ax.scatter([best_epoch], [best_loss], label='best validate', marker="d", color="green")

        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.set_title('Losses')
        ax.set_yscale('log')
        ax.legend()
        fig.savefig(self.loss_image_path)
        if not os.path.exists("./.result"):
            os.makedirs("./.result")
    
    def test(self, on_all_nodes=True, on_cpu=True):
        if on_cpu:
            self.to("cpu")

        if not os.path.exists(self.model_weight_path):
            raise FileNotFoundError(f"{self.model_weight_path}  , Try to train first")
        self.load()
        self.eval_mode()
        with torch.no_grad():
            loss        = self.compute_loss(mode="all" if on_all_nodes else "test")
            loss        = loss.item()
        return loss



class MultiTrainer(OneTrainer):
    def __init__(self, args):
        self.args = args
        self.datasets = []
        for name in Truss.NAMES:
            for n_grid in [8, 12, 16, 20, 24]:
                for n_support in [1, 2, 3]:
                    self.datasets.append(Truss(name=name, n_grid=n_grid, support=n_support))
        
        self.graphs  = [dataset.as_graph() for dataset in self.datasets]
        self.device  = torch.device(args.device)

        self.optimizer = None 
        self.model     = None

        random.shuffle(self.graphs)

        for i,graph in enumerate([*self.graphs]):
        
            if args.use_condense:
                
                graph.n_dirichlet_mask

                [graph_i,_], [graph_b2i,_] = partite_graph(graph, ~graph.n_dirichlet_mask)
                

                if args.model == "NodeEdgeGNN":
                    num_features     = graph_i.n_source_value.shape[1]
                    num_classes      = graph_i.n_displacement.shape[1]
                    num_edge_features= graph_i.n_pos.shape[1]
                    if self.model is None:
                        model_i     = init_model(num_features, num_classes,  args, num_edge_features=num_edge_features)
                    u, v             = graph_i.edge_index
                    edata_i          = graph_i.n_pos[v] - graph_i.n_pos[u]
                    y_i              = graph_i.n_displacement
                    num_node_features= graph_i.n_displacement.shape[1]
                    num_edge_features= graph_b2i.n_pos_u.shape[1]
                    num_node_classes = graph_b2i.n_source_value_u.shape[1]
                    if self.model is None:
                        model_b2i   = BipartiteEdgeModel(num_node_features, num_node_classes, num_edge_features)
                    u,v              = graph_b2i.edge_index
                    edata_b2i        = graph_b2i.n_pos_u[v] - graph_b2i.n_pos_v[u]
                    x_b2i            = graph_b2i.n_displacement_u
                else:
                    num_features     = graph_i.n_pos.shape[1]+graph_i.n_source_value.shape[1]
                    num_classes      = graph_i.n_displacement.shape[1]
                    if self.model is None:
                        model_i     = init_model(num_features, num_classes,  args)
                    y_i              = graph_i.n_displacement
                    num_node_features= graph_i.n_displacement.shape[1] + graph_i.n_pos.shape[1]
                    num_node_classes = graph_b2i.n_source_value_u.shape[1]
                    if self.model is None:
                        model_b2i   = BipartiteModel(num_node_features, num_node_classes, hidden_channels=64, num_layers=3)
                    x_b2i            = torch.cat([
                        graph_b2i.n_pos_u,
                        graph_b2i.n_displacement_u,
                        ],-1)
                    edata_i, edata_b2i = None, None

                if self.model is None:
                    self.model = nn.ModuleDict(
                        {
                            "i": model_i,
                            "b2i": model_b2i,
                        }
                    )
                    self.optimizer   = torch.optim.Adam(self.model.parameters(), lr=args.lr)
                graph_i.y = y_i.type(self.dtype)
                graph_i.edata = edata_i.type(self.dtype) if edata_i is not None else None
                graph_b2i.x = x_b2i.type(self.dtype)
                graph_b2i.edata = edata_b2i.type(self.dtype) if edata_b2i is not None else None

                normalizer = NormalizeFeatures()
                self.graphs[i] = {
                    "i": normalizer(graph_i),
                    "b2i": normalizer(graph_b2i),
                }
            else:
                y            = graph.n_displacement
                if args.model == "NodeEdgeGNN":
                    x        = graph.n_source_value
                    u,v      = graph.edge_index
                    edata    = graph.n_pos[u] - graph.n_pos[v]
                    num_features    = x.shape[1]
                    num_labels      = y.shape[1]
                    num_edge_features = edata.shape[1]
                    if self.model is None:
                        self.model      = init_model(num_features, num_labels, args, num_edge_features=num_edge_features)
                else:
                    x        = torch.cat([
                        graph.n_source_value,
                        graph.n_pos,
                    ],-1)
                    edata    = None
                    num_features    = x.shape[1]
                    num_labels      = y.shape[1]
                    if self.model is None:
                        self.model      = init_model(num_features, num_labels,  args)

                if self.optimizer is None:
                    self.optimizer  = torch.optim.Adam(self.model.parameters(), lr=args.lr)
            
                graph.x          = x.type(self.dtype)
                graph.y          = y.type(self.dtype)
                graph.edata      = edata.type(self.dtype) if edata is not None else None

                self.graphs[i] = NormalizeFeatures()(graph)

        n_train = int(len(self.graphs) * args.train_ratio)
        n_test  = int(len(self.graphs) * args.test_ratio)
        n_valid = len(self.graphs) - n_train - n_test
        self.train_graphs  = self.graphs[:n_train]
        self.valid_graphs  = self.graphs[n_train:n_train+n_valid]
        self.test_graphs   = self.graphs[n_train+n_valid:]

        self.to(self.device)
        if isinstance(self.model, nn.ModuleDict):
            model_str = str(self.model['i'])
        else:
            model_str = str(self.model)
        self.model_weight_path = f"./.result/MultiTrainer_{self.args.dataset}_{self.args.train_ratio}_{model_str}_model{'_condense' if args.use_condense else ''}.pth"
        self.loss_image_path   = f"./.result/MultiTrainer_{self.args.dataset}_{self.args.train_ratio}_{model_str}_loss{'_condense' if args.use_condense else ''}.png"

    def predict(self, graph):

        if self.args.use_condense:
            graph_b2i = graph["b2i"]
            graph_i   = graph["i"]

            # outer -> inner
            if self.args.model == "NodeEdgeGNN":
                f = self.model['b2i'](graph_b2i.x, graph_b2i.edata, graph_b2i.edge_index, graph_b2i.num_dst_nodes) # outer -> inner
            else:
                f = self.model['b2i'](graph_b2i.x, graph_b2i.edge_index, graph_b2i.num_dst_nodes) # outer -> inner
            n_source_value = f + graph_i.n_source_value
        
            # inner -> inner
            if self.args.model == "MLP":
                x = torch.cat([
                    graph_i.n_pos,
                    n_source_value,
                ],-1).type(next(iter(self.model['i'].parameters())).dtype)
                u = self.model['i'](x)
            elif self.args.model == "NodeEdgeGNN":
                x = n_source_value.type(graph_i.edata.dtype)
                u = self.model['i'](x, graph_i.edata, graph_i.edge_index)
            else:
                x = torch.cat([
                    graph_i.n_pos,
                    n_source_value,
                ],-1).type(next(iter(self.model.parameters())).dtype)
                u = self.model['i'](x, graph_i.edge_index)

        else:

            if self.args.model == "MLP":
                u = self.model(graph.x)
            elif self.args.model == "NodeEdgeGNN":
                u = self.model(graph.x, graph.edata, graph.edge_index)
            else:
                u = self.model(graph.x, graph.edge_index)

        return u
    
    def compute_loss(self, mode="train"):
        assert mode in ["train", "valid", "test", "all"]
        graphs = {
            "train": self.train_graphs,
            "valid": self.valid_graphs,
            "test": self.test_graphs,
            "all": self.graphs,
        }[mode]
        total_loss = 0.0
        for graph in graphs:
            if self.args.use_condense:
                graph_i = graph["i"]
                u = self.predict(graph)
                y = graph_i.y
            else:
                u = self.predict(graph)
                y = graph.y
            loss = torch.nn.MSELoss()(u, y)

            if mode == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_loss += loss
        return total_loss.item() / len(graphs)
    
    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()

    def save(self):
        torch.save(self.model.state_dict(), self.model_weight_path)
     
    def load(self):
        self.model.load_state_dict(torch.load(self.model_weight_path))

    def state_dict(self):
        return self.model.state_dict()
        
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
        return self


    @property
    def dtype(self):
        return next(iter(self.model.parameters())).dtype  
             
    def to(self,device):
        self.model.to(device)
        for graph in chain(self.graphs, self.train_graphs, self.valid_graphs, self.test_graphs):
            if isinstance(graph, dict):
                graph["i"].to(device)
                graph["b2i"].to(device)
            else:
                graph.to(device)
        return self

    def fit(self):

        self.train_mode()
        train_losses    = []
        valid_losses    = []

        best_model, best_epoch, best_loss = None, None, float('inf')

        pbar = tqdm.tqdm(range(self.args.epochs))
        for ep in pbar:
            
            loss = self.compute_loss("train")
            train_losses.append(loss)

                
            if (ep + 1) % self.args.eval_every_eps == 0:
                self.eval_mode()
                with torch.no_grad():
                    loss = self.compute_loss("valid")
                self.train_mode()
                valid_losses.append(loss)
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

        # plot the train/valid/test losses best validation point in one figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(np.arange(len(train_losses)), train_losses, label='train', linestyle=":")
        ax.plot(np.arange(len(valid_losses)) * self.args.eval_every_eps, valid_losses, label='valid', linestyle="-")
        ax.scatter([self.args.epochs], [loss], label='test', marker="x", color="red")
        ax.scatter([best_epoch], [best_loss], label='best validate', marker="d", color="green")

        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.set_title('Losses')
        ax.set_yscale('log')
        ax.legend()
        fig.savefig(self.loss_image_path)
        if not os.path.exists("./.result"):
            os.makedirs("./.result")

    def test(self, on_all_nodes=True, on_cpu=True):
        if on_cpu:
            self.to("cpu")

        if not os.path.exists(self.model_weight_path):
            raise FileNotFoundError(f"{self.model_weight_path}  , Try to train first")
        self.load()
        self.eval_mode()
        with torch.no_grad():
            loss        = self.compute_loss(mode="all" if on_all_nodes else "test")
        return loss
