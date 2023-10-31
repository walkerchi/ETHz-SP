import os
import numpy as np
import meshio
import scipy.sparse
import torch
import torch_geometric as pyg
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def bsr_matrix_from_coo(edata, u, v, shape):
    assert len(edata) == len(u) == len(v), f"edata: {len(edata)}, u: {len(u)}, v: {len(v)} should be same"
    assert len(edata.shape) == 3, f"edata should be 3D array, but got {len(edata.shape)}"
    nrows   = shape[0] // edata.shape[-2]
    args    = np.argsort(u)
    indptr  = np.zeros(nrows + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(np.bincount(u[args], minlength=nrows))
    indices = v[args]
    edata   = edata[args]
    return scipy.sparse.bsr_matrix((edata, indices, indptr), shape=shape, blocksize=edata.shape[1:])

def truss_linear_elasicity(points, truss, dirichlet_mask, dirichlet_value, source_mask, source_value, E, A):
    """
        Parametres:
        ----------
            points: np.ndarray, shape [n_point, n_dim]
                The coordinates of the points
            truss: np.ndarray, shape [n_element, n_dim]
                The indices of the points of each element
            dirichlet_mask: np.ndarray, shape [n_point, n_dim]
                The dirichlet mask of the points
            dirichlet_value: np.ndarray, shape [n_point, n_dim]
                The dirichlet value of the points
            source_mask: np.ndarray, shape [n_point, n_dim]
                The source mask of the points
            source_value: np.ndarray, shape [n_point, n_dim]
                The source value of the points
            E: float
                Young's modulus of the bridge
            A: float
                Cross-sectional area of the bridge
        Returns:
        --------
            u: np.ndarray, shape [n_point, n_dim]
                displacement of the points
    """ 
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
    K_local = E * A / L[:,None,None,None,None] * K_local # [n_element, n_src_basis, n_dst_basis, n_src_dim, n_dst_dim]
    
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
    K_local_nids_u = truss[:, :, None].repeat(n_basis, 2) # [n_element, n_src_basis, n_dst_basis] do as meshgrid
    K_local_nids_v = truss[:, None, :].repeat(n_basis, 1) # [n_element, n_src_basis, n_dst_basis]
    K_local_eids = np.array(eids_graph[K_local_nids_u.ravel(), K_local_nids_v.ravel()]).ravel() # [n_element * n_src_basis * n_dst_basis]
    ele2msh_edge = scipy.sparse.coo_matrix((
        np.ones(n_element * n_basis * n_basis),
        (K_local_eids, np.arange(n_element * n_basis * n_basis))
    ), shape=(n_edges, n_element * n_basis * n_basis)).tocsr()

    K_global = ele2msh_edge @ K_local.reshape(-1, n_dim * n_dim) # [n_edge, n_dim * n_dim] 


    K_coo = bsr_matrix_from_coo(
        K_global.reshape(n_edges, n_dim, n_dim), edge_u, edge_v, shape=(n_points * n_dim,  n_points*n_dim)
    ).tocoo() # [n_dim * n_point, n_dim * n_point]

    # compute load vector
    F = np.zeros(n_points * n_dim) # [n_point * n_dim]
    F[source_mask.ravel()] = source_value[source_mask].ravel() # [n_point * n_dim]

    # init displacement vector
    u = np.zeros(n_points * n_dim) # [n_point * n_dim]
    u[dirichlet_mask.ravel()] = dirichlet_value[dirichlet_mask].ravel() # [n_point * n_dim]

    # condensing
    src, dst, edata = K_coo.row, K_coo.col, K_coo.data
    n_node      = n_points * n_dim
    is_node_inner = ~dirichlet_mask.ravel() # [n_point * n_dim]
    is_node_outer = dirichlet_mask.ravel() # [n_point * n_dim]
    n_node_inner  = is_node_inner.sum()
    n_node_outer  = is_node_outer.sum() 
    nids          = np.full(n_node, fill_value=-1) # [n_point * n_dim]
    nids[is_node_inner] = np.arange(n_node_inner) # [n_node_inner]
    nids[is_node_outer] = np.arange(n_node_outer) # [n_node_outer]
    is_src_inner, is_dst_inner = is_node_inner[src], is_node_inner[dst] # [n_edge]
    is_src_outer, is_dst_outer = is_node_outer[src], is_node_outer[dst] # [n_edge]
    is_e_inner    = is_src_inner & is_dst_inner # [n_edge]
    is_e_ou2in    = is_src_inner & is_dst_outer # [n_edge]
    local_nids    = np.full(n_node, fill_value=-1)
    local_nids[is_node_inner] = np.arange(n_node_inner)
    local_nids[is_node_outer] = np.arange(n_node_outer)
    src_inner, dst_inner = local_nids[src[is_e_inner]], local_nids[dst[is_e_inner]] # [n_edge_inner]
    src_ou2in, dst_ou2in = local_nids[src[is_e_ou2in]], local_nids[dst[is_e_ou2in]] # [n_edge_ou2in]
    edata_inner, edata_ou2in = edata[is_e_inner], edata[is_e_ou2in] # [n_edge_inner]
    K_inner = scipy.sparse.coo_matrix((
        edata_inner, (src_inner, dst_inner)
    ), shape=(n_node_inner, n_node_inner)).tocsr() # [n_node_inner, n_node_inner]
    K_ou2in = scipy.sparse.coo_matrix((
        edata_ou2in, (src_ou2in, dst_ou2in)
    ), shape=(n_node_inner, n_node_outer)).tocsr() # [n_node_inner, n_node_outer]
    F_inner = F[is_node_inner] - K_ou2in @ u[is_node_outer]
   
    # solve
    u_inner = scipy.sparse.linalg.spsolve(K_inner, F_inner)
    u[is_node_inner] = u_inner

    return u

class Bridge:
    def __init__(self, 
                 n_grid:int=5, 
                 l:float=1, 
                 h:float=0.5,
                 E:float=1e6, 
                 A:float=1e-4,
                 nu:float=0.3, 
                 p:float=-1.,
                 cache_dir:str=".cache",
                 overwrite:bool=True
                 ):
        """
            Parameters:
            -----------
            n_grid: int
                Number of grid points along the length of the bridge
            l: float
                Length of the bridge
            h: float
                Height of the bridge
            E: float
                Young's modulus of the bridge
            A: float
                Cross-sectional area of the bridge
            nu: float
                Poisson's ratio of the bridge
            g: float
                Gravitational acceleration
        """
        self.cache_dir = os.path.join(cache_dir,self.__class__.__name__.lower())
        os.makedirs(self.cache_dir, exist_ok=True)
        self.mesh_parameters = {"l" : l, "h" : h, "n": n_grid}
        self.pde_parameters = {"E" : E, "nu" : nu, "A":A, "p" : p}
        mesh_parameters_str = "_".join([f"{k}={v}" for k,v in self.mesh_parameters.items()])
        pde_parameters_str  = "_".join([f"{k}={v}" for k,v in self.pde_parameters.items()])
        self.mesh_path    = os.path.join(self.cache_dir, f"{mesh_parameters_str}.msh")
        self.data_path    = os.path.join(self.cache_dir, f"{mesh_parameters_str}_{pde_parameters_str}.npz")
        self.file_format  = "gmsh"	
        self.fem_path     = os.path.join(self.cache_dir, f"{mesh_parameters_str}_{pde_parameters_str}.fem.npz")
        if not os.path.exists(self.mesh_path) or overwrite:
            self.mesh_gen(n_grid, l, h)
        if not os.path.exists(self.data_path) or overwrite:
            self.data_gen(h, p)

    def mesh_gen(self, n_grid=5, l=1, h=0.5):   

        dx = l / n_grid

        points_bottom = np.stack([
            np.linspace(0, l, n_grid+1),
            np.zeros(n_grid + 1)
        ], -1)
        nids_bottom = np.arange(n_grid+1)

        points_top    = np.stack([
            np.linspace(dx/2, l-dx/2, n_grid),
            np.ones(n_grid) * h
        ], -1)
        nids_top = np.arange(n_grid) + n_grid + 1

        truss = np.concatenate([
            np.stack([nids_top[:-1], nids_top[1:]], -1),
            np.stack([nids_bottom[:-1], nids_bottom[1:]], -1),
            np.stack([nids_bottom[:-1], nids_top], -1),
            np.stack([nids_bottom[1:], nids_top], -1),
        ])

        points = np.concatenate([points_bottom, points_top], axis=0)


        mesh = meshio.Mesh(
            points.astype(np.float64),
            cells={"line":truss.astype(np.int64)},
        )

        mesh.write(self.mesh_path, file_format=self.file_format)

    def data_gen(self, h=0.5, p=-1.):
        mesh = meshio.read(self.mesh_path, file_format=self.file_format)
        points = mesh.points[:, :2]
        x_axis, y_axis = 0, 1
        is_bottom = np.isclose(points[:, 1], 0)
        is_top    = np.isclose(points[:, 1], h)
        where_top_right = np.where(is_top)[0][-1]
        dirichlet_mask = np.zeros(points.shape, dtype=bool)
        dirichlet_mask[is_bottom, y_axis] = True
        dirichlet_value = np.zeros(points.shape, dtype=np.float64)
        dirichlet_value[is_bottom, y_axis] = 0.
        source_mask    = np.zeros(points.shape, dtype=bool)
        source_mask[where_top_right, y_axis] = True
        source_value   = np.zeros(points.shape, dtype=np.float64)
        source_value[where_top_right, y_axis] = p
        point_data={
                "dirichlet_mask" : dirichlet_mask,
                "dirichlet_value": dirichlet_value,
                "source_mask"    : source_mask,
                "source_value"   : source_value	
            }
        np.savez(self.data_path, **point_data)

    def as_graph(self):
        mesh       = meshio.read(self.mesh_path, file_format=self.file_format)
        data       = np.load(self.data_path)
        fem_result = np.load(self.fem_path)
        dtype      = torch.float64
        u,v        = torch.from_numpy(mesh.cells_dict["line"]).T
        nids       = torch.arange(mesh.points.shape[0])
        src        = torch.cat([u, v, nids])
        dst        = torch.cat([v, u, nids])
        edges      = torch.stack([src, dst], 0)
        graph = pyg.data.Data(
            num_nodes           =   mesh.points.shape[0],   
            n_pos               =   torch.from_numpy(mesh.points).type(dtype),
            n_dirichlet_mask    =   data['dirichlet_mask'],
            n_dirichlet_value   =   data['dirichlet_value'],
            n_source_mask       =   data['source_mask'],
            n_source_value      =   data['source_value'],
            n_displacement      =   torch.from_numpy(fem_result['displacement']).type(dtype),
            n_strain            =   torch.from_numpy(fem_result['strain']).type(dtype),
            n_stress            =   torch.from_numpy(fem_result['stress']).type(dtype),
            g_E         =   torch.tensor(self.pde_parameters["E"], dtype=dtype),
            g_nu        =   torch.tensor(self.pde_parameters["nu"], dtype=dtype),
            edge_index  =   edges,
        )

    def fem_sol(self):
        mesh = meshio.read(self.mesh_path, file_format=self.file_format)

        points = mesh.points[:, :2]
        truss  = mesh.cells_dict["line"]

        data = np.load(self.data_path)  
        dirichlet_mask = data["dirichlet_mask"].astype(bool)
        dirichlet_value= data["dirichlet_value"].astype(np.float64)
        source_mask    = data["source_mask"].astype(bool)
        source_value   = data["source_value"].astype(np.float64)
        
        u = truss_linear_elasicity(points, truss, dirichlet_mask, dirichlet_value, source_mask, source_value, E=self.pde_parameters["E"], A=self.pde_parameters["A"])
        
        displacement = u
        displacement_src = displacement[truss[:,0]]
        displacement_dst = displacement[truss[:,1]]
        src  = points[truss[:,0]]
        dst  = points[truss[:,1]]
        L    = np.linalg.norm(src - dst, axis=-1) # [n_element]
        strain = (displacement_src - displacement_dst) / L # [n_element]
        stress = self.pde_parameters['E'] * strain
        
        # save
        np.savez(self.fem_path, 
            u=u, 
            strain=strain, 
            stress=stress)

    def vis(self):
        mesh = meshio.read(self.mesh_path)
        stress = np.load(self.fem_path)['stress'] # [n_element]
        lines = mesh.points[:,:2][mesh.cells_dict["line"]] # [n_element, 2(nodes), 2(dim)]
        fig, ax = plt.subplots(figsize=(10, 6))

        lc = LineCollection(lines, linewidths=2, cmap="jet")
        lc.set_array(stress)
        ax.add_collection(lc)
        ax.autoscale()
        ax.margins(0.1)
        cb = plt.colorbar(lc, ax=ax)
        cb.set_label('stress')
        plt.show()  

if __name__ == "__main__":
    bridge = Bridge()
    bridge.fem_sol()
    bridge.vis()
