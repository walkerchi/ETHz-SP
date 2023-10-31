import importlib
import sys 
import meshio 
import os
import numpy as np
import torch 
import torch_geometric as pyg
bridge = importlib.import_module(".bridge", package=__package__)
roof   = importlib.import_module(".roof", package=__package__)

class Truss:
    NAMES = ["bridge.pratt", "bridge.howe", "bridge.baltimore", "roof.pratt", "roof.howe"]
    def __init__(self, name="bridge.pratt", E=1, A=1, n_grid=12, support=3):
        self.mesh = {
            "bridge.pratt": bridge.pratt,
            "bridge.howe": bridge.howe,
            "bridge.baltimore": bridge.baltimore,
            "roof.pratt": roof.pratt,
            "roof.howe": roof.howe,
        }[name](n_grid=n_grid, 
                    support=support, 
                    E = E, A=A)
        
    def solve(self, ):
        sys.path.append("../..")
        from linear_elasticity import TrussSolver
        truss_sol = TrussSolver(self.mesh)
        u = truss_sol.scipy_solve()
        strain, stress = truss_sol.compute_stress(u, return_strain=True)
        return u, strain, stress

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
        u, strain, stress = self.solve()
        displacement = torch.from_numpy(u).type(dtype)
        strain       = torch.from_numpy(strain).type(dtype)
        stress       = torch.from_numpy(stress).type(dtype)
        graph = pyg.data.Data(
            num_nodes           =   mesh.points.shape[0],   
            n_pos               =   torch.tensor(mesh.points, dtype=torch.float),
            n_dirichlet_mask    =   dirichlet_mask.any(-1),
            n_dirichlet_value   =   dirichlet_value,
            n_source_mask       =   source_mask,
            n_source_value      =   source_value,
            n_displacement      =   displacement,
            g_strain            =   strain,
            g_stress            =   stress,
            # g_E         =   torch.tensor(self.pde_parameters["E"], dtype=dtype),
            # g_nu        =   torch.tensor(self.pde_parameters["nu"], dtype=dtype),
            edge_index  =   edges.T,
        )
        return graph