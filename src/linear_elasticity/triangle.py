import numpy as np
import scipy.sparse
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import skfem
import skfem.models.elasticity
from .utils import element2edge, \
                  get_ele2msh_node, \
                  get_ele2msh_edge, \
                  scipy_bsr_matrix_from_coo, \
                 torch_bsr_matrix_from_coo, \
                 partite




def get_quadrature_points(n:int):
    """
        Parameters:
        -----------
            n: int
                the number of quadrature points
        Returns:
        --------
            weights: torch.Tensor of shape [n]
                the quadrature weights
            points: torch.Tensor of shape [n, 2]
                the quadrature points
    """
    if n == 1:
        weights = np.array([
            1/2
        ])
        points = np.array([
            [1/3, 1/3]
        ])
    elif n == 2:
        weights = np.array([
                1/6, 1/6, 1/6
            ])
        points = np.array([
            [1/6, 1/6],
            [2/3, 1/6],
            [1/6, 2/3]
        ])
    elif n == 3:
        weights = np.array([
            -27/96, 25/96, 25/96, 25/96
        ])
        points = np.array([
            [1/3, 1/3],
            [1/5, 1/5],
            [3/5, 1/5],
            [1/5, 3/5]
        ])
    else:
        raise NotImplementedError(f"n: {n} is not implemented")

    return weights, points
    
def get_shape_val(quadrature, p=1):
    """
        Parameters:
        -----------
            quadrature: torch.Tensor [n_quadrature, n_dim]
                        n_dim = 2 for triangle
        Returns:
        --------
            phi      : torch.Tensor [n_quadrature, n_basis]
                        n_basis = 3
    """
    n_quadrature, n_dim = quadrature.shape[-2:]
    assert n_dim == 2, f"n_dim must be 2 for triangle , but got {n_dim}"
    xi, eta = quadrature[..., 0], quadrature[..., 1]

    if p == 1:
        phi = np.zeros((*quadrature.shape[:-1], 3),  dtype=quadrature.dtype)
        phi[..., 0] = 1 - xi - eta
        phi[..., 1] = xi
        phi[..., 2] = eta
    elif p == 2:
        phi = np.zeros(*quadrature.shape[:-1], 6, dtype=quadrature.dtype)
        phi[..., 0] = (1 - xi - eta) * ( 1 - 2*xi - 2*eta)
        phi[..., 1] = xi * (2*xi - 1)
        phi[..., 2] = eta * (2*eta - 1)
        phi[..., 3] = 4*xi * (1 - xi - eta)
        phi[..., 4] = 4*xi * eta
        phi[..., 5] = 4*eta * (1 - xi - eta)
    else:
        raise NotImplementedError(f"p: {p} is not implemented")
    return phi

def get_shape_grad(quadrature, element_coords, p=1, return_jac=False):
    """
        Parameters:
        -----------
            quadrature: torch.Tensor [n_quadrature, n_dim]
            element_coords: torch.Tensor [n_element, n_corner, n_dim]
                        n_dim = 2 for triangle
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
    assert len(element_coords.shape) == 3, f"element_coords must be 3D of shape [n_element, 3, 2], but got {element_coords.dim()}"
    n_quadrature, n_dim = quadrature.shape 
    n_element, n_basis, _ = element_coords.shape
    assert n_dim == 2, f"n_dim must be 2 for triangle , but got {n_dim}"
    
    xi, eta = quadrature[..., 0], quadrature[..., 1]
    if p == 1:
        n_basis = 3
        grad_phi = np.zeros((n_quadrature, n_basis, n_dim), dtype=quadrature.dtype)
        grad_phi[..., 0, (0,1)] = -1
        grad_phi[..., 1, 0] = 1
        grad_phi[..., 2, 1] = 1
    elif p == 2:
        n_basis = 6
        grad_phi = np.zeros((n_quadrature, n_basis, n_dim),dtype=quadrature.dtype)
        grad_phi[..., 0, 0] = -3 + 4*xi + 4*eta
        grad_phi[..., 0, 1] = -3 + 4*xi + 4*eta
        grad_phi[..., 1, 0] = 2 - 4*xi - 2*eta
        grad_phi[..., 1, 1] = -2*xi
        grad_phi[..., 2, 0] = 2 - 4*xi - 2*eta
        grad_phi[..., 2, 1] = 0
        grad_phi[..., 3, 0] = 2*eta
        grad_phi[..., 3, 1] = 2*xi
        grad_phi[..., 4, 0] = -2*eta
        grad_phi[..., 4, 1] = 1 - 2*xi - 2*eta
        grad_phi[..., 5, 0] = 0
        grad_phi[..., 5, 1] = 4*eta - 2
    else:
        raise  NotImplementedError(f"p: {p} is not implemented")
    
    
    jac  = np.einsum("bhj,ghi->bgij", element_coords, grad_phi)
    ijac = np.linalg.inv(jac)
    grad_phi = np.einsum("gbi,ngji->ngbj", grad_phi, ijac)

    if return_jac:
        return grad_phi, jac
    else:
        return grad_phi



class TriangleSolver:
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
        elements  = mesh.cells_dict["triangle"]
        n_points  = points.shape[0]
        n_dim     = points.shape[1]
        n_basis   = elements.shape[1]
        n_element = elements.shape[0]
        assert n_dim == 2, f"n_dim must be 2 for triangle , but got {n_dim}"

        # compute Galerkin matrix
        elem_coords = points[elements] # [n_element, n_basis, n_dim]
        quadrature_weights, quadrature_points = get_quadrature_points(1) # [n_quadrature], [n_quadrature, n_dim]
        shape_grad, jac = get_shape_grad(quadrature_points, elem_coords, p=1, return_jac=True) # [n_element, n_quadrature, n_basis, n_dim]
        jac_det     = np.linalg.det(jac) # [n_element, n_quadrature]
        jxw         = np.abs(jac_det) * quadrature_weights # [n_element, n_quadrature]
        n_quadrature = jxw.shape[1]


        B           = np.zeros((n_element, n_quadrature, n_basis, n_dim*(n_dim+1)//2, n_dim))
        B[:, :, :, 0, 0] = shape_grad[:,:,:,0]
        B[:, :, :, 1, 1] = shape_grad[:,:,:,1]
        B[:, :, :, 2, 0] = shape_grad[:,:,:,1]
        B[:, :, :, 2, 1] = shape_grad[:,:,:,0]
        # B           = np.zeros((n_element, n_quadrature, n_dim*(n_dim+1)//2, n_basis*n_dim)) # [n_element, n_quadrature, n_dim*(n_dim+1)/2,  n_basis*n_dim]
        # B[:, :, 0, 0::n_dim] = shape_grad[:,:,:,0]
        # B[:, :, 1, 1::n_dim] = shape_grad[:,:,:,1]
        # B[:, :, 2, 0::n_dim] = shape_grad[:,:,:,1]
        # B[:, :, 2, 1::n_dim] = shape_grad[:,:,:,0]

        #  1, nu, 0
        #  nu, 1, 0
        #  0,  0, (1 - nu) / 2     
        D             = np.zeros((n_element, 3, 3)) # [n_element, n_dim*(n_dim+1)/2, n_dim*(n_dim+1)/2]
        D[:,(0,1),(0,1)] = 1 
        D[:,(1,0),(0,1)] = nu[:, None]
        D[:, 2, 2]       = (1 - nu) / 2
        D *= E[:, None, None] / (1 - nu[:, None, None] ** 2)
       

        K_local = np.einsum("eqaim, eij, eqbjn, eq->eqabmn", B, D, B, jxw) # [n_element, n_basis, n_basis, n_dim, n_dim]
        # K_local = np.einsum("nqia, nij, nqjb, nq->nab", B, D, B, jxw) # [n_element, n_basis*n_dim, n_basis*n_dim]
        # K_local = K_local.reshape(n_element, n_basis, n_dim, n_basis, n_dim).transpose(0,1,3,2,4) # [n_element, n_basis, n_basis, n_dim, n_dim]
     
        # assemble Galerkin matrix
        edge_u, edge_v = element2edge(elements, n_points)
        n_edges        = edge_u.shape[0]
        ele2msh_edge   = get_ele2msh_edge(edge_u, edge_v, elements, n_points) # [n_edge, n_element * n_basis * n_basis]

        K_global = ele2msh_edge @ K_local.reshape(-1, n_dim * n_dim) # [n_edge, n_dim * n_dim] 

        K_coo = scipy_bsr_matrix_from_coo(
            K_global.reshape(ele2msh_edge.shape[0], n_dim, n_dim), edge_u, edge_v, shape=(n_points,  n_points)
        ).tocoo() # [n_dim * n_point, n_dim * n_point]

        dirichlet_mask = mesh.point_data["dirichlet_mask"]
        dirichlet_value= mesh.point_data["dirichlet_value"]
        source_value   = mesh.point_data["source_value"]
        source_mask    = mesh.point_data["source_mask"]
        self.K_inner, self.K_ou2in = partite(K_coo, dirichlet_mask.ravel())
        self.is_inner_dof = ~dirichlet_mask.ravel()
        self.is_outer_dof = dirichlet_mask.ravel()
        self.source_value  = source_value
        self.source_mask   = source_mask
        self.dirichlet_mask= dirichlet_mask
        self.dirichlet_value= dirichlet_value

        self.ele2msh_node = get_ele2msh_node(elements, n_points) # [n_points, n_element]
        self.shape_val= get_shape_val(quadrature_points, p=1) # [n_quadrature, n_basis]
        self.D        = D      # [n_element, n_basis, n_basis]
        self.B        = B      # [n_element, n_quadrature, n_basis, n_basis * n_dim]
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
    
    def skfem_solve(self):
        """
            Returns:
            --------
                u: np.ndarray, shape [n_point, n_dim]
                    displacement of the points
        """
        mesh = skfem.MeshTri(self.points.T, self.elements.T)
        basis = skfem.InteriorBasis(mesh, skfem.ElementVectorH1(skfem.ElementTriP1())) 

        # assemble Galerkin matrix
        E, nu   = self.E[0], self.nu[0]
        K = skfem.asm(skfem.models.elasticity.linear_elasticity(*skfem.models.elasticity.lame_parameters(E,nu)), basis)
        u = np.zeros(self.n_points * self.n_dim)
        u[self.dirichlet_mask.ravel()] = self.dirichlet_value[self.dirichlet_mask].ravel()
        u = skfem.solve(*skfem.condense(K, b=self.source_value.ravel(), x=u, D=self.dirichlet_mask.ravel()))
        u = u.reshape(self.n_points, self.n_dim)

        return u

    def scipy_solve(self, 
              dirichlet_mask = None, 
              dirichlet_value = None, 
              source_mask = None, 
              source_value = None):
        """
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
    
    def compute_stress(self, u, return_strain=False, return_vm_stress=False):
        """
            Parameters:
            -----------
                u:  np.ndarray, shape [n_point, n_dim]
                    displacement of the points
                return_strain: bool
                    whether to return strain
                    default False
                return_vm_stress: bool
                    whether to return von mises stress
                    default False
            Returns:
            --------
                returns: list[np.ndarray]
                    the stress of the points
                    if return_strain is True, then returns[0] is strain
                    if return_vm_stress is True, then returns[-1] is von mises stress
        """
        n_element, n_basis = self.elements.shape
        n_point, n_dim     = u.shape 
        elem_u = u[self.elements] # [n_element, n_basis, n_dim]
        B      = self.B # [n_element, n_quadrature, n_dim*(n_dim+1)/2, n_basis * n_dim]
        D      = self.D # [n_element, n_dim*(n_dim+1)/2, n_dim*(n_dim+1)/2]
        shape_val = self.shape_val # [n_quadrature, n_basis]
        n_quadrature  = B.shape[1]

        B      = B.reshape(n_element, n_quadrature, -1, n_basis, n_dim) # [n_element, n_quadrature, n_dim*(n_dim+1)/2, n_basis, n_dim]
     
        strain = np.einsum('nqibd,qb,nbd->nbi', B, shape_val, elem_u) # [n_element, n_basis,n_dim*(n_dim+1)]
        stress = np.einsum('nij,nbi->nbi', D, strain) # [n_element, n_basis, n_dim*(n_dim+1)]
        
        strain = self.ele2msh_node @ strain.reshape(n_element *n_basis, -1 ) # [n_point, n_dim*(n_dim+1)]
        stress = self.ele2msh_node @ stress.reshape(n_element *n_basis, -1 ) # [n_point, n_dim*(n_dim+1)]

        vm_stress =  np.sqrt(stress[:, 0] ** 2 + stress[:, 1] ** 2 - stress[:, 0] * stress[:, 1] + 3 * stress[:, 2] ** 2)

        returns = []
        if return_strain:
            returns.append(strain)
        returns.append(stress)
        if return_vm_stress:
            returns.append(vm_stress)
        if len(returns) == 1:
            return returns[0]
        else:
            return returns

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
            self.K_torch = self.K_torch.type(u.dtype).to(u.device)
            f = torch.from_numpy(self.source_value).type(u.dtype).to(u.device)
            r = self.K_torch @ u.reshape(-1, 1) - f.reshape(-1, 1)
        elif isinstance(u, np.ndarray):
            r = self.K_coo @ u.ravel() - self.source_value.ravel()
        else:
            raise NotImplementedError(f"u should be torch.Tensor or np.ndarray, but got {type(u)}")
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
                ax[0,i].triplot(self.points[:, 0], self.points[:, 1], self.elements, alpha=0.2, color="k", zorder=1)
                # ax[0,i].scatter(self.points[:, 0], self.points[:, 1], s=20, alpha=0.2, c="r", zorder=10, marker="o")
                ax[0,i].scatter(self.points[self.source_mask.any(1), 0], self.points[self.source_mask.any(1), 1], s=50, c="b", zorder=10, marker="v", label="force")
                ax[0,i].scatter(self.points[self.dirichlet_mask.any(1), 0], self.points[self.dirichlet_mask.any(1), 1], s=50, c="g", zorder=10, marker="^", label="dirichlet")
                ax[0,i].autoscale()
                ax[0,i].margins(0.1)
                ax[0,i].legend()
                ax[0,i].set_title(k)
                cb = plt.colorbar(tpc, ax=ax[0,i])
                cb.set_label(k)
            plt.show()
                



