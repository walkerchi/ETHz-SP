import importlib
import sys 
import meshio 
import os
import numpy as np
import torch 
import torch_geometric as pyg
from .cube import cube  
from .spherical_shell  import spherical_shell

from linear_elasticity import TetraSolver

class Tetra:
    NAMES = ["cube", "spherical_shell"]
    PREFIX  = ["cube", "spherical_shell"]
    def __init__(self, 
                 name="cube", 
                 E=1, nu=0.3, 
                 d=0.1, # charateristic length 
                 a:float=1.0, # length or inner radius
                 b:float=2.0, # outer radius
                p:float=1.0, # pressure
    ):
        if name == "cube":
            self.mesh = cube(d=d, E=E, nu=nu, a=a, p=p)
        elif name == "spherical_shell":
            self.mesh = spherical_shell(d=d, E=E, nu=nu, a=a, b=b, p=p)
        else:
            raise NotImplementedError
        
        self.solver = TetraSolver(self.mesh)
        
    def solve(self):
        u = self.solver.scipy_solve()
        return u 
    
    def compute_residual(self, u, mse=True):
        return self.solver.compute_residual(u, mse=mse)
    
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
        elements = mesh.get_cells_type("tetra")
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