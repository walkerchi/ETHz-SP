import numpy as np
import scipy.sparse
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import pyvista as pv
from .utils import element2edge, \
                  get_ele2msh_node, \
                  get_ele2msh_edge, \
                  scipy_bsr_matrix_from_coo, \
                 torch_bsr_matrix_from_coo, \
                 partite




def get_gauss_points(n:int):
    """
        Parameters:
        -----------
            n : int
                the order of the quadrature
        Returns:
        --------
            weights: torch.Tensor of shape [n]
                the quadrature weights
            points: torch.Tensor of shape [n, 2]
                the quadrature points
    """
    if n == 1:
        weights = np.array([ 0.16666666666666666,])
        points = np.array([ [ 0.25, 0.25, 0.25],]) 
    elif n == 2:
        weights = np.array([ 0.041666666666666664, 0.041666666666666664, 0.041666666666666664, 0.041666666666666664,])
        points = np.array([ [ 0.5854101966249685, 0.1381966011250105, 0.1381966011250105,], [ 0.1381966011250105, 0.1381966011250105, 0.1381966011250105,], [ 0.1381966011250105, 0.1381966011250105, 0.5854101966249685,], [ 0.1381966011250105, 0.5854101966249685, 0.1381966011250105,],]) 
    elif n == 3:
        weights = np.array([ -0.13333333333333333, 0.075, 0.075, 0.075, 0.075,])
        points = np.array([ [ 0.25, 0.25, 0.25,], [ 0.5, 0.1666666666666667, 0.1666666666666667,], [ 0.1666666666666667, 0.1666666666666667, 0.1666666666666667,], [ 0.1666666666666667, 0.1666666666666667, 0.5,], [ 0.1666666666666667, 0.5, 0.1666666666666667,],])
    else:
        raise Exception(f"n must be 1, 2 or 3, but got {n}")
    return weights, points

def shape_val_p1(quadrature):
    """
        Parameters:
        -----------
            quadrature: torch.Tensor [n_quadrature, n_dim]
                        n_dim = 3 for tetra
        Returns:
        --------
            phi      : torch.Tensor [n_quadrature, n_basis]
                        n_basis = 4
    """
    n_quadrature, n_dim = quadrature.shape[-2:]
    assert n_dim == 3, f"n_dim must be 3 for tetra , but got {n_dim}"

    phi = np.zeros((*quadrature.shape[:-1], 4), dtype=quadrature.dtype)
    x, y, z = quadrature[..., 0], quadrature[..., 1], quadrature[..., 2]
    phi[..., 0] = 1 - x - y - z
    phi[..., 1] = x
    phi[..., 2] = y 
    phi[..., 3] = z
    return phi

def shape_grad_p1(quadrature, element_coords, return_jac=False):
    """
        Parameters:
        -----------
            quadrature: torch.Tensor [n_quadrature, n_dim]
            element_coords: torch.Tensor [n_element, n_corner, n_dim]
                        n_dim = 3 for tetra
            return_jac: bool
                        whether to return the jacobian
                        default is False
        Returns:
        --------
            grad_phi: torch.Tensor of shape [n_element, n_quadrature, n_basis, n_dim]
                the gradient of the base functions
            jac     : torch.Tensor of shape [n_element, n_quadrature, n_dim, n_dim]
                the jacobian of the base functions
                if return_jac is False, then jac is None
    """
    assert element_coords.dtype == quadrature.dtype, f"element_coords.dtype must be {quadrature.dtype}, but got {element_coords.dtype}"
    assert len(element_coords.shape) == 3, f"element_coords must be 3D of shape [n_element, 4, 3], but got {element_coords.shape}"
    n_quadrature, n_dim = quadrature.shape 
    n_element, n_basis, _ = element_coords.shape
    assert n_dim == 3, f"n_dim must be 3 for tetra , but got {n_dim}"
    assert n_basis == 4, f"n_basis must be 4 for tetra , but got {n_basis}"
    
    grad_phi = np.zeros([n_quadrature, n_basis, n_dim], dtype=quadrature.dtype)
    grad_phi[..., 0, :] = -1
    grad_phi[..., 1, 0] = 1
    grad_phi[..., 2, 1] = 1
    grad_phi[..., 3, 2] = 1
    
    jac  = np.einsum("ebj,qbi->eqij", element_coords, grad_phi)
    ijac =  np.linalg.inv(jac)
    grad_phi = np.einsum("qbi,eqji->eqbj", grad_phi, ijac)

    if return_jac:
        return grad_phi, jac
    else:
        return grad_phi


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
        quadrature_weights, quadrature_points = get_gauss_points(2) # [n_quadrature], [n_quadrature, n_dim]
        shape_grad, jac = shape_grad_p1(quadrature_points, elem_coords, return_jac=True) # [n_element, n_quadrature, n_basis, n_dim]
        jac_det     = np.linalg.det(jac) # [n_element, n_quadrature]
        jxw         = np.abs(jac_det) * quadrature_weights # [n_element, n_quadrature]
        n_quadrature = quadrature_weights.shape[0]
        B           = np.zeros((n_element, n_quadrature, np.math.factorial(n_dim), n_basis*n_dim)) # [n_element, n_quadrature n_dim!,  n_basis*n_dim]
        B[:, :, 0, ::n_dim]  = shape_grad[..., 0]
        B[:, :, 1, 1::n_dim] = shape_grad[..., 1]
        B[:, :, 2, 2::n_dim] = shape_grad[..., 2]
        B[:, :, 3, ::n_dim]  = shape_grad[..., 1]
        B[:, :, 3, 1::n_dim] = shape_grad[..., 0]
        B[:, :, 4, 1::n_dim] = shape_grad[..., 2]
        B[:, :, 4, 2::n_dim] = shape_grad[..., 1]
        B[:, :, 5, ::n_dim]  = shape_grad[..., 2]
        B[:, :, 5, 2::n_dim] = shape_grad[..., 0]

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
        D[:,:var_dim//2, :var_dim//2]              += lambda_ [:,None, None]
        D[:,np.arange(var_dim),np.arange(var_dim)] += mu [:, None]
        D[:,np.arange(var_dim//2),np.arange(var_dim//2)] += mu [:, None]

        K_local = np.einsum("nqia, nij, nqjb, nq->nab", B, D, B, jxw) # [n_element, n_basis*n_dim, n_basis*n_dim]
        K_local = K_local.reshape(n_element, n_basis, n_dim, n_basis, n_dim).transpose(0,1,3,2,4) # [n_element, n_basis, n_basis, n_dim, n_dim]

        # assemble Galerkin matrix
        edge_u, edge_v = element2edge(elements, n_points)
        n_edges        = edge_u.shape[0]
        ele2msh_edge   = get_ele2msh_edge(edge_u, edge_v, elements, n_points) # [n_edge, n_element * n_basis * n_basis]

        K_global = ele2msh_edge @ K_local.reshape(-1, n_dim * n_dim) # [n_edge, n_dim * n_dim] 

        K_coo = scipy_bsr_matrix_from_coo(
            K_global.reshape(ele2msh_edge.shape[0], n_dim, n_dim), edge_u, edge_v, shape=(n_points,  n_points)
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
            K_global.reshape(n_edges, n_dim, n_dim), edge_u, edge_v, shape=(n_points,  n_points)
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
    
    # def compute_vm_stress(self, u):
    #     """
    #         Parameters:
    #         -----------
    #             u: np.ndarray, shape [n_point, n_dim]
    #                 displacement of the points
    #         Returns:
    #         --------
    #             vm_stress: np.ndarray, shape [n_points]
    #                 stress of the elements
    #     """
    #     n_element, n_dim = self.elements.shape
    #     elem_u = u[self.elements].reshape(n_element, -1) # [n_element, n_basis * n_dim]
    #     strain = np.einsum('nbi,ni->nb', self.B, elem_u) # [n_element, n_basis]
    #     stress = np.einsum('nij,nj->ni', self.D, strain) # [n_element, n_basis]

    #     ele2msh_node = get_ele2msh_node(self.elements, self.n_points) # [n_points, n_element]
    #     stress = ele2msh_node @ stress # [n_element, n_basis] - > [n_points, n_basis]

    #     vm_stress = np.sqrt(
    #         0.5 * ()stress[:, 0] ** 2 + stress[:, 1] ** 2 - stress[:, 0] * stress[:, 1] + 3 * stress[:, 2] ** 2
    #     ) 

    #     return vm_stress

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
        if isinstance(u, torch.Tensor):
            self.K_torch = self.K_torch.to(u.device).type(u.dtype)
            f = torch.from_numpy(self.source_value).type(u.dtype).to(u.device)
            r = self.K_torch @ u.reshape(-1, 1) - f.reshape(-1, 1)
        elif isinstance(u, np.ndarray):
            f = self.source_value
            r = self.K_coo @ u.ravel() - f.ravel()
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

            mesh = pv.from_meshio(mesh)
            p = pv.Plotter()
            p.add_mesh_clip_plane(mesh, assign_to_axis='z',  show_edges=True, cmap="jet")
            p.add_axes()
            p.show()

        else:
            ncols = len(kwargs)
            mesh = pv.from_meshio(self.mesh)

            plotter = pv.Plotter(shape=(1, ncols))
            for i,(k, value) in enumerate(kwargs.items()):
                mesh.point_data[k] = value
                plotter.subplot(0, i)
                plotter.add_mesh_clip_plane(mesh, assign_to_axis='z',scalars=k, cmap="jet", show_edges=True, show_scalar_bar=True)
                plotter.add_text(k, font_size=10, position='upper_edge')
                plotter.add_axes()
                
            plotter.show()
                



