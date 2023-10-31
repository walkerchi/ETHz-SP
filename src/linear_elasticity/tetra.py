import numpy as np
import scipy.sparse
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
from .utils import element2edge, \
                  get_ele2msh_node, \
                  get_ele2msh_edge, \
                  scipy_bsr_matrix_from_coo, \
                 torch_bsr_matrix_from_coo, \
                 partite

class TetraSolver:
    def __init__(self, mesh, E=None, nu=None):
        self.update_mesh(mesh, E, nu)
    
    def update_mesh(self, mesh, E=None, nu=None):
        """
            Parameters:
            -----------
                mesh: meshio.Mesh
                    mesh of the truss
                E: float, or np.ndarray[n_element] could be found in mesh.cell_data_dict["E"]
                    Young's modulus
                    default None
                nu: float, or np.ndarray[n_element] could be found in mesh.cell_data_dict["nu"]
                    cross section area
                    default None
        """
        if E is None:
            assert "E" in mesh.field_data.keys(), "E should be provided"
            E = mesh.field_data["E"]        
        if nu is None:
            assert "nu" in mesh.field_data.keys(), "nu should be provided"
            nu = mesh.field_data["nu"]
    
        points = mesh.points
        elements  = mesh.cells_dict["tetra"]
        n_points  = points.shape[0]
        n_dim     = points.shape[1]
        n_basis   = elements.shape[1]
        n_element = elements.shape[0]

        # compute Galerkin matrix
        elem_coords = points[elements] # [n_element, n_basis, n_dim]
        shape_grad  = np.array([
            [-1, 1, 0, 0],
            [-1, 0, 1, 0],
            [-1, 0, 0, 1]
        ])  # [n_dim, n_basis]

        J           = np.einsum("ib,nbj->nij", shape_grad, elem_coords)  # [n_element, n_dim, n_dim]
        invJ        = np.linalg.inv(J) # [n_element, n_dim, n_dim]
        detJ        = np.linalg.det(J) # [n_element]
        shape_grad  = np.einsum("nij, jb->nib", invJ, shape_grad) # [n_element, n_dim, n_basis]
        
        B           = np.zeros((n_element, np.math.factorial(n_dim), n_basis*n_dim)) # [n_element, n_dim!,  n_basis*n_dim]
        B[:, 0, ::n_dim]  = shape_grad[:, 0]
        B[:, 1, 1::n_dim] = shape_grad[:, 1]
        B[:, 2, 2::n_dim] = shape_grad[:, 2]
        B[:, 3, ::n_dim]  = shape_grad[:, 1]
        B[:, 3, 1::n_dim] = shape_grad[:, 0]
        B[:, 4, 1::n_dim] = shape_grad[:, 2]
        B[:, 4, 2::n_dim] = shape_grad[:, 1]
        B[:, 5, ::n_dim]  = shape_grad[:, 2]
        B[:, 5, 2::n_dim] = shape_grad[:, 0]

        # lambda+2mu  lambda     lambda     0     0     0
        # lambda      lambda+2mu lambda     0     0     0
        # lambda      lambda     lambda+2mu 0     0     0
        # 0           0          0          mu    0     0
        # 0           0          0          0     mu    0
        # 0           0          0          0     0     mu
        var_dim       = n_dim * (n_dim + 1) // 2
        lambda_       = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu            = E / (2 * (1 + nu))
        D             = np.zeros((n_element, var_dim, var_dim)) # [n_element, n_dim!, n_dim!]
        D[:,:var_dim//2, :var_dim//2]              += lambda_ 
        D[:,np.arange(var_dim),np.arange(var_dim)] += mu
        D[:,np.arange(var_dim//2),np.arange(var_dim//2)] += mu
        weight        = 1/6

        K_local = weight * np.einsum("nia, nij, njb->nab", B, D, B) * detJ[:, None, None] # [n_element, n_basis*n_dim, n_basis*n_dim]
        K_local = K_local.reshape(n_element, n_basis, n_dim, n_basis, n_dim).transpose(0,1,3,2,4) # [n_element, n_basis, n_basis, n_dim, n_dim]

        # assemble Galerkin matrix
        edge_u, edge_v = element2edge(elements, n_points)
        n_edges        = edge_u.shape[0]
        ele2msh_edge   = get_ele2msh_edge(edge_u, edge_v, elements, n_points) # [n_edge, n_element * n_basis * n_basis]

        K_global = ele2msh_edge @ K_local.reshape(-1, n_dim * n_dim) # [n_edge, n_dim * n_dim] 

        K_coo = scipy_bsr_matrix_from_coo(
            K_global.reshape(ele2msh_edge.shape[0], n_dim, n_dim), edge_u, edge_v, shape=(n_points * n_dim,  n_points  * n_dim)
        ).tocoo() # [n_dim * n_point, n_dim * n_point]

        
        dirichlet_mask = mesh.point_data["dirichlet_mask"]
        source_value   = mesh.point_data["source_value"]
        source_mask    = mesh.point_data["source_mask"]
        self.K_inner, self.K_ou2in = partite(K_coo, dirichlet_mask.ravel())
        self.is_inner_dof = ~dirichlet_mask.ravel()
        self.is_outer_dof = dirichlet_mask.ravel()
        self.source_value  = source_value
        self.source_mask   = source_mask
        self.dirichlet_mask= dirichlet_mask

        self.D        = D      # [n_element, n_basis, n_basis]
        self.B        = B      # [n_element, n_basis, n_basis * n_dim]
        self.mesh     = mesh
        self.points   = points # [n_point, n_dim]
        self.elements = elements # [n_element, n_basis]
        self.K_coo    = K_coo  # [n_dim * n_point, n_dim * n_point]
        self.n_points = n_points
        self.n_dim    = n_dim
        self.n_basis  = n_basis
        self.n_element= n_element
        self.E        = E if isinstance(E, np.ndarray) else np.full(n_element, fill_value=E)
        self.nu       = nu if isinstance(nu, np.ndarray) else np.full(n_element, fill_value=nu)
        self.ele2msh_edge_torch = torch.sparse_csr_tensor(
            torch.from_numpy(ele2msh_edge.indptr),
            torch.from_numpy(ele2msh_edge.indices),
            torch.from_numpy(ele2msh_edge.data),
            size=ele2msh_edge.shape
        )
        self.K_torch  = torch_bsr_matrix_from_coo(
            K_global.reshape(n_edges, n_dim, n_dim), edge_u, edge_v, shape=(n_points * n_dim,  n_points*n_dim)
        ) # [n_dim * n_point, n_dim * n_point]
    
    def scipy_solve(self, 
              dirichlet_mask = None, 
              dirichlet_value = None, 
              source_mask = None, 
              source_value = None):
        """
            Parameters:
            -----------
                dirichlet_mask: np.ndarray, shape [n_point, n_dim]
                    The dirichlet mask of the points
                dirichlet_value: np.ndarray, shape [n_point, n_dim]
                    The dirichlet value of the points
                source_mask: np.ndarray, shape [n_point, n_dim]
                    The source mask of the points
                source_value: np.ndarray, shape [n_point, n_dim]
                    The source value of the points
            Returns:
            --------
                u: np.ndarray, shape [n_point, n_dim]
                    displacement of the points
        """
        if dirichlet_mask is None:
            assert "dirichlet_mask" in self.mesh.point_data.keys(), "dirichlet_mask should be provided"
            dirichlet_mask = self.mesh.point_data["dirichlet_mask"]
        if dirichlet_value is None:
            assert "dirichlet_value" in self.mesh.point_data.keys(), "dirichlet_value should be provided"
            dirichlet_value = self.mesh.point_data["dirichlet_value"]
        if source_mask is None:
            assert "source_mask" in self.mesh.point_data.keys(), "source_mask should be provided"
            source_mask = self.mesh.point_data["source_mask"]
        if source_value is None:
            assert "source_value" in self.mesh.point_data.keys(), "source_value should be provided"
            source_value = self.mesh.point_data["source_value"]
        assert dirichlet_mask.shape == (self.n_points, self.n_dim), f"dirichlet_mask.shape: {dirichlet_mask.shape}, should be {(self.n_points, self.n_dim)}"
        assert dirichlet_value.shape == (self.n_points, self.n_dim), f"dirichlet_value.shape: {dirichlet_value.shape}, should be {(self.n_points, self.n_dim)}"
        assert source_mask.shape    == (self.n_points, self.n_dim), f"source_mask.shape: {source_mask.shape}, should be {(self.n_points, self.n_dim)}"
        assert source_value.shape   == (self.n_points, self.n_dim), f"source_value.shape: {source_value.shape}, should be {(self.n_points, self.n_dim)}"
                                             
        # compute load vector
        F = np.zeros(self.n_points * self.n_dim) # [n_point * n_dim]
        F[source_mask.ravel()] = source_value[source_mask].ravel() # [n_point * n_dim]

        # init displacement vector
        u = np.zeros(self.n_points * self.n_dim) # [n_point * n_dim]
        u[dirichlet_mask.ravel()] = dirichlet_value[dirichlet_mask].ravel() # [n_point * n_dim]

        # condensing
        K_inner, K_ou2in = partite(self.K_coo, dirichlet_mask.ravel())
        F_inner = F[~dirichlet_mask.ravel()] - K_ou2in @ u[dirichlet_mask.ravel()]

        # solve
        u_inner = scipy.sparse.linalg.spsolve(K_inner, F_inner)
        u[~dirichlet_mask.ravel()] = u_inner

        return u.reshape(self.n_points, self.n_dim)
    
    def compute_vm_stress(self, u):
        """
            Parameters:
            -----------
                u: np.ndarray, shape [n_point, n_dim]
                    displacement of the points
            Returns:
            --------
                vm_stress: np.ndarray, shape [n_points]
                    stress of the elements
        """
        n_element, n_dim = self.elements.shape
        elem_u = u[self.elements].reshape(n_element, -1) # [n_element, n_basis * n_dim]
        strain = np.einsum('nbi,ni->nb', self.B, elem_u) # [n_element, n_basis]
        stress = np.einsum('nij,nj->ni', self.D, strain) # [n_element, n_basis]

        ele2msh_node = get_ele2msh_node(self.elements, self.n_points) # [n_points, n_element]
        stress = ele2msh_node @ stress # [n_element, n_basis] - > [n_points, n_basis]

        vm_stress = np.sqrt(
            0.5 * ()stress[:, 0] ** 2 + stress[:, 1] ** 2 - stress[:, 0] * stress[:, 1] + 3 * stress[:, 2] ** 2
        ) 

        return vm_stress

    def compute_residual(self, u, mse=True):
        """
            Parameters:
            -----------
                u: np.ndarray, shape [n_point, n_dim]
                    displacement of the points
            Returns:
            --------
                residual: np.ndarray, shape [n_point, n_dim] or float  if mse is True
                    residual of the points
        """
        # solve
        r = self.K_coo @ u.ravel() - self.source_value.ravel()
        if mse:
            return (r * r).mean()
        else:
            return  r.reshape(self.n_points, self.n_dim)

    def plot(self, **kwargs):
        """
            Parameters:
            -----------
                kwargs: dict[str, np.ndarray of shape[n_point]]
                    The displacement of the points
        """
        
        if len(kwargs) == 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.triplot(self.points[:, 0], self.points[:, 1], self.elements, color="k", zorder=1)
            ax.scatter(self.points[:, 0], self.points[:, 1], s=50, c="r", zorder=10, marker="o")
            for  i, (x, y) in enumerate(self.points):
                ax.text(x, y, f"n{i}", color="violet", fontsize=16)
            for  i, (x, y) in enumerate(self.points[self.elements].mean(1)):
                ax.text(x, y, f"e{i}", color="green", fontsize=16)
            ax.autoscale()
            ax.margins(0.1)
            plt.show()

        else:
            ncols = len(kwargs)
            fig, ax = plt.subplots(ncols=ncols, figsize=(ncols*5, 4),squeeze=False)
            for i,(k, value) in enumerate(kwargs.items()):
                tpc = ax[0,i].tripcolor(self.points[:, 0], self.points[:, 1], self.elements, value, shading="gouraud", cmap="jet", zorder=0)
                # ax[0,i].triplot(self.points[:, 0], self.points[:, 1], self.elements, alpha=0.2, color="k", zorder=1)
                # ax[0,i].scatter(self.points[:, 0], self.points[:, 1], s=20, alpha=0.2, c="r", zorder=10, marker="o")
                # ax[0,i].scatter(self.points[self.source_mask.any(1), 0], self.points[self.source_mask.any(1), 1], s=50, c="b", zorder=10, marker="v", label="force")
                # ax[0,i].scatter(self.points[self.dirichlet_mask.any(1), 0], self.points[self.dirichlet_mask.any(1), 1], s=50, c="g", zorder=10, marker="^", label="dirichlet")
                ax[0,i].autoscale()
                ax[0,i].margins(0.1)
                ax[0,i].legend()
                cb = plt.colorbar(tpc, ax=ax[i])
                cb.set_label(k)
            plt.show()
                



