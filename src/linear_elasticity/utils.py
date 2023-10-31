import numpy as np
import scipy.sparse
import torch

def scipy_bsr_matrix_from_coo(edata, u, v, shape):
    """
        Parameters:
        -----------
            edata: np.ndarray, shape [n_edge, dim_1, dim_2]
            u: np.ndarray, shape [n_edge]
            v: np.ndarray, shape [n_edge]
            shape: tuple, shape [n_point, n_point]
    """
    assert len(edata) == len(u) == len(v), f"edata: {len(edata)}, u: {len(u)}, v: {len(v)} should be same"
    assert len(edata.shape) == 3, f"edata should be 3D array, but got {len(edata.shape)}"
    nrows   = shape[0] // edata.shape[-2]
    args    = np.argsort(u)
    indptr  = np.zeros(nrows + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(np.bincount(u[args], minlength=nrows))
    indices = v[args]
    edata   = edata[args]
    return scipy.sparse.bsr_matrix((edata, indices, indptr), shape=shape, blocksize=edata.shape[1:])

def torch_bsr_matrix_from_coo(edata, u, v, shape):
    if isinstance(edata, np.ndarray):
        edata = torch.from_numpy(edata)
    if isinstance(u, np.ndarray):
        u = torch.from_numpy(u)
    if isinstance(v, np.ndarray):
        v = torch.from_numpy(v)

    # dim_1, dim_2 = edata.shape[-2:]
    # off_1, off_2 = torch.meshgrid(torch.arange(dim_1), torch.arange(dim_2)) # [dim_1, dim_2]
    # u = u[:, None, None] + off_1[None, :, :] # [n_edge, dim_1, dim_2] 
    # v = v[:, None, None] + off_2[None, :, :] # [n_edge, dim_1, dim_2]
    # u = u.reshape(-1) # [n_edge * dim_1 * dim_2]
    # v = v.reshape(-1) # [n_edge * dim_1 * dim_2]
    # edata = edata.reshape( -1) # [n_edge, dim_1 * dim_2]
   
    # return torch.sparse_coo_tensor(torch.stack([u,v], 0), edata, size=shape)
    nrows   = shape[0] // edata.shape[-2]
    args    = torch.argsort(u)
    indptr  = torch.zeros(nrows + 1, dtype=torch.int64)
    indptr[1:] = torch.cumsum(torch.bincount(u[args], minlength=nrows), dim=0)
    indices = v[args]
    edata   = edata[args]
    return torch.sparse_bsr_tensor(indptr, indices, edata, size=shape)
    

def partite(K_coo:scipy.sparse.coo_matrix, dirichlet_mask:np.ndarray):
    """
        Parameters:
        -----------
            K_coo: scipy.sparse.coo_matrix, shape [n_point * n_dim, n_point * n_dim]
                The stiffness matrix of the mesh
            dirichlet_mask: np.ndarray, shape [n_point, n_dim]
                The dirichlet mask of the points
        Returns:
        --------
            K_inner: scipy.sparse.csr_matrix, shape [n_node_inner, n_node_inner]
                The stiffness matrix of the inner nodes
            K_ou2in: scipy.sparse.csr_matrix, shape [n_node_inner, n_node_outer]
                The stiffness matrix of the outer nodes to the inner nodes
    """
    assert K_coo.shape[0] == K_coo.shape[1], f"K_coo should be square matrix, but got {K_coo.shape}"
    assert len(dirichlet_mask) == K_coo.shape[0], f"dirichlet_mask.shape: {dirichlet_mask.shape}, should be {K_coo.shape[0]}"

    if isinstance(K_coo, scipy.sparse.coo_matrix):
        ravel = lambda x: x.ravel()
        full  = lambda x, fill_value: np.full(x, fill_value=fill_value)
        arange= lambda x: np.arange(x)
        coo_matrix = lambda edata, row, col, shape: scipy.sparse.coo_matrix((edata, (row, col)), shape=shape)
        tocsr = lambda x: x.tocsr()
    elif isinstance(K_coo, torch.Tensor):
        ravel = lambda x: x.view(-1)
        full  = lambda x, fill_value: torch.full(x, fill_value=fill_value)
        arange= lambda x: torch.arange(x)
        coo_matrix = lambda edata, row, col, shape: torch.sparse_coo_tensor((row, col), edata, shape=shape)
        tocsr = lambda x: x.coalesce().tocsr()


    src, dst, edata = K_coo.row, K_coo.col, K_coo.data
    n_node          = K_coo.shape[0]
    is_node_inner = ~ravel(dirichlet_mask) # [n_point * n_dim]
    is_node_outer = ravel(dirichlet_mask) # [n_point * n_dim]
    n_node_inner  = is_node_inner.sum()
    n_node_outer  = is_node_outer.sum() 
    nids          = full(n_node, fill_value=-1) # [n_point * n_dim]
    nids[is_node_inner] = arange(n_node_inner) # [n_node_inner]
    nids[is_node_outer] = arange(n_node_outer) # [n_node_outer]
    is_src_inner, is_dst_inner = is_node_inner[src], is_node_inner[dst] # [n_edge]
    is_src_outer, is_dst_outer = is_node_outer[src], is_node_outer[dst] # [n_edge]
    is_e_inner    = is_src_inner & is_dst_inner # [n_edge]
    is_e_ou2in    = is_src_inner & is_dst_outer # [n_edge]
    local_nids    = full(n_node, fill_value=-1)
    local_nids[is_node_inner] = arange(n_node_inner)
    local_nids[is_node_outer] = arange(n_node_outer)
    src_inner, dst_inner = local_nids[src[is_e_inner]], local_nids[dst[is_e_inner]] # [n_edge_inner]
    src_ou2in, dst_ou2in = local_nids[src[is_e_ou2in]], local_nids[dst[is_e_ou2in]] # [n_edge_ou2in]
    edata_inner, edata_ou2in = edata[is_e_inner], edata[is_e_ou2in] # [n_edge_inner]
    K_inner = coo_matrix(
        edata_inner, src_inner, dst_inner, shape=(n_node_inner, n_node_inner)
    )  # [n_node_inner, n_node_inner]
    K_ou2in = coo_matrix(
        edata_ou2in, src_ou2in, dst_ou2in, shape=(n_node_inner, n_node_outer)
    ) # [n_node_inner, n_node_outer]
    K_inner, K_ou2in = tocsr(K_inner), tocsr(K_ou2in)
    return K_inner, K_ou2in



def element2edge(elements, n_points):
    """
        Parameters:
        -----------
            elements: np.ndarray, shape [n_element, n_basis]
                The elements of the mesh
            n_points: int
        Returns:
        --------
            edges: np.ndarray, shape [2,  n_edge]
                The edges of the mesh
    """
    n_elements, n_basis = elements.shape
    if n_basis == 2:
        edge_u = np.concatenate([
            elements[:, 0], elements[:, 1], np.arange(n_points)
        ]) # [2 * n_element + n_point]
        edge_v = np.concatenate([
            elements[:, 1], elements[:, 0], np.arange(n_points)
        ]) # [2 * n_element + n_point]
    else:
        assert n_basis >= 3
        edge_u, edge_v = [], []
        for i in range(n_basis):
            for j in range(n_basis):
                edge_u.append(elements[:, i])
                edge_v.append(elements[:, j])
        edge_u = np.concatenate(edge_u) # [n_elements * n_basis * n_basis]
        edge_v = np.concatenate(edge_v) # [n_elements * n_basis * n_basis]
        # remove duplicated edges
        tmp = scipy.sparse.coo_matrix((
            np.ones(n_elements * n_basis * n_basis),
            (edge_u, edge_v)
        ), shape = (n_points, n_points)).tocsr().tocoo()
        edge_u, edge_v = tmp.row, tmp.col
    return np.stack([edge_u, edge_v], axis=0) # [2, n_edge]

def get_ele2msh_node(elements, n_points):
    """
        Parameteres:
        ------------
        Returns:
        --------
            ele2msh_node: scipy.sparse.csr_matrix, shape [n_point, n_element * n_basis]
                The mapping from element to mesh node
    """
    n_element, n_basis = elements.shape
    ele2msh_node = scipy.sparse.coo_matrix((
        np.ones(elements.ravel().shape[0]),
        (elements.ravel(), np.arange(n_element * n_basis))
    ), shape=(n_points, n_element * n_basis)).tocsr()

    return ele2msh_node


def get_ele2msh_edge(edge_u, edge_v, elements, n_points):
    """
        Parameters:
        -----------
            edge_u: np.ndarray, shape [n_edge]
                The u of the edges
            edge_v: np.ndarray, shape [n_edge]
                The v of the edges
            elements: np.ndarray, shape [n_element, n_basis]
                The elements of the mesh
            n_points: int
        Returns:
        --------
            ele2msh_edge: scipy.sparse.csr_matrix, shape [n_edge, n_element * n_basis * n_basis]
                The mapping from element to mesh edge
    """
    n_elements, n_basis = elements.shape
    n_edges = len(edge_u)
    eids_graph = scipy.sparse.coo_matrix((
        np.arange(n_edges),# [2 * n_element + n_point] since there are self-looped undirected graph
        (edge_u, edge_v)
    ), shape=(n_points, n_points)).tocsr()
    assert eids_graph.nnz == n_edges, f"eids_graph.nnz: {eids_graph.nnz}, should be {n_edges}, the truss elements are overlapped"
    K_local_nids_u = elements[:, :, None].repeat(n_basis, 2) # [n_element, n_src_basis, n_dst_basis] do as meshgrid
    K_local_nids_v = elements[:, None, :].repeat(n_basis, 1) # [n_element, n_src_basis, n_dst_basis]
    K_local_eids = np.array(eids_graph[K_local_nids_u.ravel(), K_local_nids_v.ravel()]).ravel() # [n_element * n_src_basis * n_dst_basis]
    
    ele2msh_edge = scipy.sparse.coo_matrix((
        np.ones(n_elements * n_basis * n_basis),
        (K_local_eids, np.arange(n_elements * n_basis * n_basis))
    ), shape=(n_edges, n_elements * n_basis * n_basis)).tocsr()
    
    return  ele2msh_edge


