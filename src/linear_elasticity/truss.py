import numpy as np
import scipy.sparse
import skfem
import skfem.helpers
import trusspy
import torch 
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from .utils import scipy_bsr_matrix_from_coo, torch_bsr_matrix_from_coo, partite

class MappingAffine(skfem.MappingAffine):
    def __init__(self, mesh):
        dim = mesh.p.shape[0]
        
        if mesh.t.shape[0] > 0:
            nt = mesh.t.shape[1]
            # initialize the affine mapping
            self.A = np.empty((dim, dim, nt))
            self.b = np.empty((dim, nt))

            dist = mesh.p[:, mesh.t[0]] - mesh.p[:, mesh.t[1]] # [n_dim, n_element]
            L    = np.linalg.norm(dist, axis=0) # [n_element]
            cos  = dist[0] / L # [n_element]
            sin  = dist[1] / L # [n_element]
            self.A[0,0] = cos
            self.A[0,1] = -sin
            self.A[1,0] = sin
            self.A[1,1] = cos
            self.b      = mesh.p[:, mesh.t[0]]


            # determinants
            if dim == 1:
                self.detA = self.A[0, 0]
            elif dim == 2:
                self.detA = (self.A[0, 0] * self.A[1, 1] -
                             self.A[0, 1] * self.A[1, 0])
            elif dim == 3:
                self.detA = self.A[0, 0] * (self.A[1, 1] * self.A[2, 2] -
                                            self.A[1, 2] * self.A[2, 1]) -\
                            self.A[0, 1] * (self.A[1, 0] * self.A[2, 2] -
                                            self.A[1, 2] * self.A[2, 0]) +\
                            self.A[0, 2] * (self.A[1, 0] * self.A[2, 1] -
                                            self.A[1, 1] * self.A[2, 0])
            else:
                raise Exception("Not implemented for the given dimension.")
            
            # affine mapping inverses
            self.invA = np.empty((dim, dim, nt))
            if dim == 1:
                self.invA[0, 0] = 1. / self.A[0, 0]
            elif dim == 2:
                self.invA[0, 0] =  self.A[1, 1] / self.detA  # noqa
                self.invA[0, 1] = -self.A[0, 1] / self.detA
                self.invA[1, 0] = -self.A[1, 0] / self.detA
                self.invA[1, 1] =  self.A[0, 0] / self.detA  # noqa
            elif dim == 3:
                self.invA[0, 0] = (-self.A[1, 2] * self.A[2, 1] +
                                   self.A[1, 1] * self.A[2, 2]) / self.detA
                self.invA[1, 0] = (self.A[1, 2] * self.A[2, 0] -
                                   self.A[1, 0] * self.A[2, 2]) / self.detA
                self.invA[2, 0] = (-self.A[1, 1] * self.A[2, 0] +
                                   self.A[1, 0] * self.A[2, 1]) / self.detA
                self.invA[0, 1] = (self.A[0, 2] * self.A[2, 1] -
                                   self.A[0, 1] * self.A[2, 2]) / self.detA
                self.invA[1, 1] = (-self.A[0, 2] * self.A[2, 0] +
                                   self.A[0, 0] * self.A[2, 2]) / self.detA
                self.invA[2, 1] = (self.A[0, 1] * self.A[2, 0] -
                                   self.A[0, 0] * self.A[2, 1]) / self.detA
                self.invA[0, 2] = (-self.A[0, 2] * self.A[1, 1] +
                                   self.A[0, 1] * self.A[1, 2]) / self.detA
                self.invA[1, 2] = (self.A[0, 2] * self.A[1, 0] -
                                   self.A[0, 0] * self.A[1, 2]) / self.detA
                self.invA[2, 2] = (-self.A[0, 1] * self.A[1, 0] +
                                   self.A[0, 0] * self.A[1, 1]) / self.detA
            else:
                raise Exception("Not implemented for the given dimension.")

        self.dim = dim
        self.mesh = mesh  # this is required in ElementH2


class MeshLine(skfem.MeshLine1):
    def _mapping(self):
        from skfem import MappingIsoparametric
        if not hasattr(self, '_cached_mapping'):
            if self.affine:
                self._cached_mapping = MappingAffine(self)
            else:
                self._cached_mapping = MappingIsoparametric(
                    self,
                    self.elem(),
                    self.bndelem,
                )
        return self._cached_mapping


class TrussSolver:
    def __init__(self, mesh, E=None, A=None):
        self.update_mesh(mesh, E, A)

    def update_mesh(self, mesh, E=None, A=None):
        """
            Parameters:
            -----------
                mesh: meshio.Mesh
                    mesh of the truss
                E: float, or np.ndarray[n_element] could be found in mesh.cell_data_dict["E"]
                    Young's modulus
                    default None
                A: float, or np.ndarray[n_element] could be found in mesh.cell_data_dict["A"]
                    cross section area
                    default None
        """
        if E is None:
            assert "E" in mesh.cell_data_dict.keys(), "E should be provided"
            E = mesh.cell_data_dict["E"]["line"]
        if A is None:
            assert "A" in mesh.cell_data_dict.keys(), "A should be provided"
            A = mesh.cell_data_dict["A"]["line"]
    
        points = mesh.points
        truss  = mesh.cells_dict["line"]
        n_points  = points.shape[0]
        n_dim     = points.shape[1]
        n_basis   = truss.shape[1]
        n_element = truss.shape[0]

        src  = points[truss[:,0]]
        dst  = points[truss[:,1]]
        L    = np.linalg.norm(src - dst, axis=-1) # [n_element]
        dist = (dst - src) / L[:,None] # [n_element, 2]
        cos, sin = dist[:,0], dist[:,1] # [n_element] 

        # compute Galerkin matrix
        K_dim = np.stack([
            np.stack([cos * cos, cos * sin], -1),
            np.stack([cos * sin, sin * sin], -1)
        ], -2) # [n_element, n_dim, n_dim]
        K_node = np.array([
            [1, -1],
            [-1, 1]
        ]) # [n_basis, n_basis]
        K_local = np.einsum("nab,ij->nijab", K_dim, K_node)  # [n_element, n_src_basis, n_dst_basis, n_src_dim, n_dst_dim]
        K_local = (E * A / L)[:,None,None,None,None] * K_local # [n_element, n_src_basis, n_dst_basis, n_src_dim, n_dst_dim]
        
        # assemble Galerkin matrix
        edge_u = np.concatenate([
            truss[:,0], truss[:,1], np.arange(len(points))
        ]) # [2 * n_element + n_point] = [n_edge]
        edge_v = np.concatenate([
            truss[:,1], truss[:,0], np.arange(len(points))
        ]) # [2 * n_element + n_point] = [n_edge]
        n_edges = len(edge_u)
        eids_graph = scipy.sparse.coo_matrix((
            np.arange(n_edges),# [2 * n_element + n_point] since there are self-looped undirected graph
            (edge_u, edge_v)
        ), shape=(n_points, n_points)).tocsr()
        assert eids_graph.nnz == n_edges, f"eids_graph.nnz: {eids_graph.nnz}, should be {n_edges}, the truss elements are overlapped"
        K_local_nids_u = truss[:, :, None].repeat(n_basis, 2) # [n_element, n_src_basis, n_dst_basis] do as meshgrid
        K_local_nids_v = truss[:, None, :].repeat(n_basis, 1) # [n_element, n_src_basis, n_dst_basis]
        K_local_eids = np.array(eids_graph[K_local_nids_u.ravel(), K_local_nids_v.ravel()]).ravel() # [n_element * n_src_basis * n_dst_basis]
       
        ele2msh_edge = scipy.sparse.coo_matrix((
            np.ones(n_element * n_basis * n_basis),
            (K_local_eids, np.arange(n_element * n_basis * n_basis))
        ), shape=(n_edges, n_element * n_basis * n_basis)).tocsr()

        K_global = ele2msh_edge @ K_local.reshape(-1, n_dim * n_dim) # [n_edge, n_dim * n_dim] 

        K_coo = scipy_bsr_matrix_from_coo(
            K_global.reshape(n_edges, n_dim, n_dim), edge_u, edge_v, shape=(n_points * n_dim,  n_points*n_dim)
        ).tocoo() # [n_dim * n_point, n_dim * n_point]

        self.dirichlet_mask = mesh.point_data["dirichlet_mask"]
        self.dirichlet_value= mesh.point_data["dirichlet_value"]
        self.source_mask    = mesh.point_data["source_mask"]
        self.source_value   = mesh.point_data["source_value"]
        self.mesh     = mesh
        self.points   = points # [n_point, n_dim]
        self.truss    = truss # [n_element, n_basis]
        self.K_coo    = K_coo  # [n_dim * n_point, n_dim * n_point]
        self.n_points = n_points
        self.n_dim    = n_dim
        self.n_basis  = n_basis
        self.n_element= n_element
        self.E        = E if isinstance(E, np.ndarray) else np.full(n_element, fill_value=E)
        self.A        = A if isinstance(A, np.ndarray) else np.full(n_element, fill_value=A)
        self.ele2msh_edge_torch = torch.sparse_csr_tensor(
            torch.from_numpy(ele2msh_edge.indptr),
            torch.from_numpy(ele2msh_edge.indices),
            torch.from_numpy(ele2msh_edge.data),
            size=ele2msh_edge.shape
        )
        self.K_torch  = torch_bsr_matrix_from_coo(
            K_global.reshape(n_edges, n_dim, n_dim), edge_u, edge_v, shape=(n_points * n_dim,  n_points*n_dim)
        ) # [n_dim * n_point, n_dim * n_point]
        

    def trusspy_solve(self,
                    dirichlet_mask = None,
                    dirichlet_value = None,
                    source_mask = None,
                    source_value = None
                    ):
        """
            Parmaters:
            ----------
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
        M = trusspy.Model()
        with M.Nodes as MN:
            for i in range(self.n_points):
                MN.add_node(i, coord=(self.points[i,0],self.points[i,1],0))
        with M.Elements as ME:
            for i in range(self.n_element):
                ME.add_element(i, conn=self.truss[i], material_properties=[self.E[i]], geometric_properties=[self.A[i]])
        with M.Boundaries  as MB:
            for i in range(self.n_points):
                MB.add_bound_U(i, (~dirichlet_mask[i]).astype(int))
        with M.ExtForces as MF:
            for i in range(self.n_points):
                MF.add_force(i, (source_value[i,0],source_value[i,1],0))

        # Solve the problem
        M.Settings.nsteps = 1
        M.build()
        M.run()
        breakpoint()
        # Extract displacements
        u = np.array([M.NDof[node_id].U for node_id in range(self.n_points)])

        return u
            
    def skfem_solve(self,
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
        p = self.points.T # [n_dim, n_point]
        t = self.truss.T  # [n_basis, n_element]
        mesh = MeshLine(p, t)
        basis= skfem.Basis(mesh, skfem.ElementVectorH1(skfem.ElementLineP1(), 2))
       
        @skfem.BilinearForm
        def stiffness(u, v, w):
            x = w.x
            lengths = np.linalg.norm(x[:,:,1] - x[:,:,0], axis=0)
            # return (self.E * self.A / lengths)[:, None] * skfem.helpers.dot(u.grad, v.grad)
            # result =  (self.E * self.A / lengths)[:, None] * skfem.helpers.ddot(u.grad, v.grad)
            # result = (self.E * self.A / lengths)[:, None] * skfem.helpers.ddot(skfem.helpers.sym_grad(u), skfem.helpers.sym_grad(v))
            result = (self.E * self.A)[:, None] * skfem.helpers.ddot(skfem.helpers.sym_grad(u), skfem.helpers.sym_grad(v))
            return result
        @skfem.LinearForm
        def loading(v, w):
            # return  skfem.helpers.dot(w.f.transpose(1, 0, 2), v)
            return skfem.helpers.dot(w.f.transpose(2, 0, 1), v)
        A = skfem.asm(stiffness, basis)
       
        b = skfem.asm(loading, basis, f=source_value[self.truss])
        # b = source_value.ravel()
        # breakpoint()
        u = skfem.solve(*skfem.condense(A, b, D=dirichlet_mask.ravel(), x=dirichlet_value.ravel()))

        return u.reshape(self.n_points, self.n_dim)

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
        # src, dst, edata = self.K_coo.row, self.K_coo.col, self.K_coo.data
        # n_node      = self.n_points * self.n_dim
        # is_node_inner = ~dirichlet_mask.ravel() # [n_point * n_dim]
        # is_node_outer = dirichlet_mask.ravel() # [n_point * n_dim]
        # n_node_inner  = is_node_inner.sum()
        # n_node_outer  = is_node_outer.sum() 
        # nids          = np.full(n_node, fill_value=-1) # [n_point * n_dim]
        # nids[is_node_inner] = np.arange(n_node_inner) # [n_node_inner]
        # nids[is_node_outer] = np.arange(n_node_outer) # [n_node_outer]
        # is_src_inner, is_dst_inner = is_node_inner[src], is_node_inner[dst] # [n_edge]
        # is_src_outer, is_dst_outer = is_node_outer[src], is_node_outer[dst] # [n_edge]
        # is_e_inner    = is_src_inner & is_dst_inner # [n_edge]
        # is_e_ou2in    = is_src_inner & is_dst_outer # [n_edge]
        # local_nids    = np.full(n_node, fill_value=-1)
        # local_nids[is_node_inner] = np.arange(n_node_inner)
        # local_nids[is_node_outer] = np.arange(n_node_outer)
        # src_inner, dst_inner = local_nids[src[is_e_inner]], local_nids[dst[is_e_inner]] # [n_edge_inner]
        # src_ou2in, dst_ou2in = local_nids[src[is_e_ou2in]], local_nids[dst[is_e_ou2in]] # [n_edge_ou2in]
        # edata_inner, edata_ou2in = edata[is_e_inner], edata[is_e_ou2in] # [n_edge_inner]
        # K_inner = scipy.sparse.coo_matrix((
        #     edata_inner, (src_inner, dst_inner)
        # ), shape=(n_node_inner, n_node_inner)).tocsr() # [n_node_inner, n_node_inner]

        # K_ou2in = scipy.sparse.coo_matrix((
        #     edata_ou2in, (src_ou2in, dst_ou2in)
        # ), shape=(n_node_inner, n_node_outer)).tocsr() # [n_node_inner, n_node_outer]
        K_inner, K_ou2in = partite(self.K_coo, dirichlet_mask.ravel())
        F_inner = F[~dirichlet_mask.ravel()] - K_ou2in @ u[dirichlet_mask.ravel()]

        # solve
        u_inner = scipy.sparse.linalg.spsolve(K_inner, F_inner)
        u[~dirichlet_mask.ravel()] = u_inner

        return u.reshape(self.n_points, self.n_dim)
    
    def compute_torch_residual(self, u, f):
        """
            Parameters:
            -----------
                u: np.ndarray, shape [n_point, n_dim]
                    displacement of the points  
            Returns:
            --------
                residual: np.ndarray, shape [n_point, n_dim]
                    residual of the points
        """
        shape = u.shape
        r = (self.K_torch @ u.reshape(-1, 1)).reshape(shape) - f
        return r

    def compute_stress(self, u, return_strain=False):
        """
            Parameters:
            -----------
                u: np.ndarray, shape [n_point, n_dim]
                    displacement of the points
            Returns:
            --------
                stress: np.ndarray, shape [n_element]
                    stress of the elements
        """
        src  = self.points[self.truss[:,0]]
        dst  = self.points[self.truss[:,1]]
        L    = np.linalg.norm(src - dst, axis=-1)
        dist = (dst - src) / L[:,None]
        cos, sin = dist[:,0], dist[:,1]
        u_src = u[self.truss[:,0]]
        u_dst = u[self.truss[:,1]]
        # turn u into axial
        u_src = u_src[:,0] * cos + u_src[:,1] * sin
        u_dst = u_dst[:,0] * cos + u_dst[:,1] * sin
        strain = (u_dst - u_src) / L
        stress = self.E * self.A * strain
        if return_strain:
            return stress, strain
        else:
            return stress

    def plot(self, **kwargs):
        """
            Parameters:
            -----------
                kwargs: dict[str, np.ndarray of shape[n_point]]
                    The displacement of the points
        """
        
        if len(kwargs) == 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            lines = self.points[self.truss]
            lc    = LineCollection(lines, linewidths=2, cmap="jet")
            ax.add_collection(lc)
            ax.scatter(self.points[:, 0], self.points[:, 1], s=50, c="r", zorder=10, marker="o")
            for  i, (x, y) in enumerate(self.points):
                ax.text(x, y, f"n{i}", color="violet", fontsize=16)
            for  i, (x, y) in enumerate(lines.mean(1)):
                ax.text(x, y, f"e{i}", color="green", fontsize=16)
            ax.autoscale()
            ax.margins(0.1)
            plt.show()

        else:
            ncols = len(kwargs)
            fig, ax = plt.subplots(ncols=ncols, figsize=(ncols*5, 4),squeeze=False)
            for i,(k, value) in enumerate(kwargs.items()):
                lines = self.points[self.truss]
                lc    = LineCollection(lines, linewidths=2, cmap="jet")
                lc.set_array(value)
                ax[0,i].add_collection(lc)
                ax[0,i].scatter(self.points[:, 0], self.points[:, 1], alpha=0.3, s=50, c="g", zorder=10, marker="o")
                ax[0,i].scatter(self.points[self.dirichlet_mask.any(-1)][:, 0], self.points[self.dirichlet_mask.any(-1)][:, 1]-0.02, alpha=0.5, s=50, c="b", zorder=10, marker="^", label="dirichlet")
                ax[0,i].scatter(self.points[self.source_mask.any(-1)][:, 0], self.points[self.source_mask.any(-1)][:, 1]+0.02, alpha=0.5, s=50, c="r", zorder=10, marker="v", label="source")
                ax[0,i].autoscale()
                ax[0,i].margins(0.1)    
                ax[0,i].legend()
                ax[0,i].set_title(k)
                cb = plt.colorbar(lc, ax=ax[0,i])
                cb.set_label(k)
            plt.show()
                




        