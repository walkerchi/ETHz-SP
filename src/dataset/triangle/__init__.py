import importlib
import sys 
import meshio 
import os
import numpy as np
import torch 
import torch_geometric as pyg

from .rectangle_shell import rectangle_shell
from .rectangle import rectangle
from .quadrilateral import quadrilateral
from .triangle import triangle

from linear_elasticity import TriangleSolver

class Triangle:
    NAMES = (["rectangle_shell", "rectangle"] +
            [f"rectangle_{i}{j}" for i in ("sin", "cos") for j in (1, 2, 4, 8)] + 
            [f"rectangle_{i}{j}_{k}" for i in ("sin", "cos") for j in (1, 2, 4, 8) for k in ("left","right","bottom","left+right","left+bottom","right+bottom","left+right+bottom")])
    PREFIX = ["rectangle", "quadrilateral", "rectangle_shell", "triangle"]
    def __init__(self, 
                 name="rectangle", 
                 E=1, nu=0.3, 
                 d=0.1, # charateristic length 
                 a:float=1.0, # length or inner length
                 b:float=2.0, # outer length
                 p:float=1.0, # pressure
    ):
        
        def get_force_fn(word):
            if "const" in word:
                return lambda p,x,y:p
            force = word[:3]
            freq  = int(word[3:])
            return {
                "sin": lambda p,x,y:p*np.sin( freq * np.pi*x/a),
                "cos": lambda p,x,y:p*np.cos( freq * np.pi*x/a),
            }[force]
        if name.startswith("rectangle"):
            words = name.split("_")
            if len(words) == 1:
                self.mesh = rectangle(d=d, E=E, nu=nu, a=a, p=p)
            elif len(words) == 2:
                self.mesh = rectangle(d=d, E=E, nu=nu, a=a, p=p, fn = get_force_fn(words[1]))
            elif len(words) == 3:
                self.mesh = rectangle(d=d, E=E, nu=nu, a=a, p=p, fn = get_force_fn(words[1]), boundary=words[2])
            elif len(words) == 4:
                self.mesh = rectangle(d=d, E=E, nu=nu, a=a, p=p, fn = get_force_fn(words[1]), boundary=words[2], source=words[3])
            else:
                raise NotImplementedError(f"Invalid name: {name}")
        elif name.startswith("quadrilateral"):
            words = name.split("_")
            if len(words) == 1:
                self.mesh = quadrilateral(d=d, E=E, nu=nu, a=a, p=p)
            elif len(words) == 2:
                self.mesh = quadrilateral(d=d, E=E, nu=nu, a=a, p=p, fn = get_force_fn(words[1]))
            else:
                self.mesh = quadrilateral(d=d, E=E, nu=nu, a=a, p=p, fn = get_force_fn(words[1]), boundary=words[2])
        elif name.startswith("triangle"):
            words = name.split("_")
            if len(words) == 1:
                self.mesh = triangle(d=d, E=E, nu=nu, a=a, p=p)
            elif len(words) == 2:
                self.mesh = triangle(d=d, E=E, nu=nu, a=a, p=p, fn = get_force_fn(words[1]))
            else:
                self.mesh = triangle(d=d, E=E, nu=nu, a=a, p=p, fn = get_force_fn(words[1]), boundary=words[2])
        elif name == "rectangle_shell":
            self.mesh = rectangle_shell(d=d, E=E, nu=nu, a=a, b=b, p=p)
        else:
            raise NotImplementedError
        
        self.solver = TriangleSolver(self.mesh)
        
    def solve(self):
        u, f = self.solver.scipy_solve()
        return u, f
    
    def compute_residual(self, u, f, mse=True, form="strong"):
        return self.solver.compute_residual(u, f, mse=mse, form=form)
    
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
        elements = mesh.get_cells_type("triangle")
        elements = torch.tensor(elements, dtype=torch.long)
        edges  = torch.vmap(lambda x: torch.stack(torch.meshgrid(x, x), -1))(elements) # (n_triangle, 4, 4, 2)
        edges  = edges.view(-1, 2) # (n_triangle * 4 * 4, 2)
        adj    = torch.sparse_coo_tensor(
            edges.T, 
            torch.ones(edges.shape[0], dtype=torch.float), 
            size=(mesh.points.shape[0], mesh.points.shape[0])
        ).coalesce()
        edges  = adj.indices().T

        # label data 
        u, f = self.solve()
        displacement = torch.from_numpy(u).type(dtype)
        load         = torch.from_numpy(f).type(dtype)  
        graph = pyg.data.Data(
            num_nodes           =   mesh.points.shape[0],   
            n_pos               =   torch.tensor(mesh.points, dtype=torch.float),
            n_dirichlet_mask    =   dirichlet_mask.any(-1),
            n_dirichlet_value   =   dirichlet_value,
            n_source_mask       =   source_mask,
            n_source_value      =   load,
            n_displacement      =   displacement,
            # g_E         =   torch.tensor(self.pde_parameters["E"], dtype=dtype),
            # g_nu        =   torch.tensor(self.pde_parameters["nu"], dtype=dtype),
            edge_index  =   edges.T,
        )
        return graph