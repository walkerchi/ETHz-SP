import importlib
import sys 
import meshio 
import os
import numpy as np
import torch 
import torch_geometric as pyg
bridge = importlib.import_module(".bridge", package=__package__)
roof   = importlib.import_module(".roof", package=__package__)
from .hollow_rectangle import hollow_rectangle

from linear_elasticity import TrussSolver

class Truss:
    NAMES = ["bridge.pratt", "bridge.howe", "bridge.baltimore", "roof.pratt", "roof.howe", "hollow_rectangle"]
    PREFIX = ["bridge", "roof", "hollow_rectangle"]
    def __init__(self, name="bridge.pratt", E=1, A=1, n_grid=12, support=3,
                 d:float=0.2, # mesh size for hollow rectangle
                a:float=2.0, # outer length for hollow rectangle
                p:float=1.0, # pressure for hollow rectangle
    ):
        naive_truss = {
            "bridge.pratt": bridge.pratt,
            "bridge.howe": bridge.howe,
            "bridge.baltimore": bridge.baltimore,
            "roof.pratt": roof.pratt,
            "roof.howe": roof.howe,
        }
        if name in naive_truss:
            self.mesh = naive_truss[name](n_grid=n_grid, 
                        support=support, 
                        E = E, A=A)
        elif name == "hollow_rectangle":
            self.mesh = hollow_rectangle(d=d, E=E, A=A, a=a, p=p)
        else:
            raise NotImplementedError
        
        self.solver = TrussSolver(self.mesh)
        
    def solve(self):
        u = self.solver.scipy_solve()
        return u 
    
    def compute_residual(self, u,f, mse=True):
        return self.solver.compute_residual(u,f, mse=mse)
    
    def plot(self, **kwargs):
        return self.solver.plot(**kwargs)

    def as_graph(self):

        dtype               = torch.float64
        mesh                = self.mesh

        # inner dirichlet condition
        dirichlet_mask      = torch.from_numpy(self.mesh.point_data["dirichlet_mask"]).type(torch.bool)
        dirichlet_value     = torch.from_numpy(self.mesh.point_data["dirichlet_value"]).type(dtype)
        source_mask         = torch.from_numpy(self.mesh.point_data["source_mask"]).type(torch.bool)
        source_value        = torch.from_numpy(self.mesh.point_data["source_value"]).type(dtype)
        dirichlet_mask      = dirichlet_mask.any(-1, keepdim=True) # consider the dirichlet is for all dimensions

        # connectivity
        elements = mesh.get_cells_type("line")
        elements = torch.tensor(elements, dtype=torch.long)
        edges  = torch.vmap(lambda x: torch.stack(torch.meshgrid(x, x), -1))(elements) # (n_tetras, 4, 4, 2)
        edges  = edges.view(-1, 2) # (n_tetras * 4 * 4, 2)
        adj    = torch.sparse_coo_tensor(
            edges.T, 
            torch.ones(edges.shape[0], dtype=torch.float), 
            size=(mesh.points.shape[0], mesh.points.shape[0])
        ).coalesce()
        edges  = adj.indices().T

        # label data 
        u = self.solve()
        displacement = torch.from_numpy(u).type(dtype)
        graph = pyg.data.Data(
            num_nodes           =   mesh.points.shape[0],   
            n_pos               =   torch.tensor(mesh.points, dtype=torch.float),
            n_dirichlet_mask    =   dirichlet_mask.any(-1),
            n_dirichlet_value   =   dirichlet_value,
            n_source_mask       =   source_mask,
            n_source_value      =   source_value,
            n_displacement      =   displacement,
            # g_E         =   torch.tensor(self.pde_parameters["E"], dtype=dtype),
            # g_nu        =   torch.tensor(self.pde_parameters["nu"], dtype=dtype),
            edge_index  =   edges.T,
        )
        return graph