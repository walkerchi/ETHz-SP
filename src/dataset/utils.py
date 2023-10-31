import meshio 
import numpy as np
import torch 
import torch_geometric as pyg 

def get_element_type(mesh):
    if "tetra" in mesh.cells_dict().keys():
        return "tetra"
    elif "triangle" in mesh.cells_dict().keys():
        return "triangle"
    elif "quad" in mesh.cells_dict().keys():
        return "quad"
    elif "line" in mesh.cells_dict().keys():
        return "line"
    else:
        raise NotImplementedError()


def mesh_to_pyg_graph(mesh, dtype=torch.float64):
    assert "dirichlet_mask" in mesh.point_data.keys()
    assert "dirichlet_value" in mesh.point_data.keys()
    assert "source_mask" in mesh.point_data.keys()
    assert "source_value" in mesh.point_data.keys()
    assert "displacement" in mesh.point_data.keys()
    assert "strain" in mesh.point_data.keys()
    assert "stress" in mesh.point_data.keys()

    element_key     = get_element_type(mesh)
    dirichlet_mask  = torch.from_numpy(mesh.point_data['dirichlet_mask']).bool() # (n_points, n_dim)
    dirichlet_value = torch.from_numpy(mesh.point_data['dirichlet_value']).type(dtype) # (n_points, n_dim)
    source_mask     = torch.from_numpy(mesh.point_data['source_mask']).bool() # (n_points, n_dim)
    source_value    = torch.from_numpy(mesh.point_data['source_value']).type(dtype) # (n_points, n_dim)
    displacement    = torch.from_numpy(mesh.point_data['displacement']).type(dtype) # (n_points, n_dim)
    strain          = torch.from_numpy(mesh.point_data['strain']).type(dtype) # (n_points, n_dim, n_dim) for truss (n_points,)
    stress          = torch.from_numpy(mesh.point_data['stress']).type(dtype) # (n_points, n_dim, n_dim) for truss (n_points,)
    gdata           = {}
    if "E" in mesh.cell_data_dict.keys():
        gdata["g_E"] = torch.tensor(mesh.cell_data_dict["E"][element_key], dtype=dtype)
    if "A" in mesh.cell_data_dict.keys():
        gdata["g_A"] = torch.tensor(mesh.cell_data_dict["A"][element_key], dtype=dtype)
    if "nu" in mesh.cell_data_dict.keys():
        gdata["g_nu"] = torch.tensor(mesh.cell_data_dict["nu"][element_key], dtype=dtype)

    # connectivity
    elements = torch.from_numpy(mesh.cells_dict()[element_key]).long()
    edges  = torch.vmap(lambda x: torch.stack(torch.meshgrid(x, x), -1))(elements) # (n_elements, n_basis, n_basis, 2)
    edges  = edges.view(-1, 2) # (n_elements*n_basis*n_basis, 2)
    adj    = torch.sparse_coo_tensor(
        edges.T, 
        torch.ones(edges.shape[0], dtype=torch.float), 
        size=(mesh.points.shape[0], mesh.points.shape[0])
    ).coalesce()
    edges  = adj.indices().T

    # label data 
    graph = pyg.data.Data(
        num_nodes           =   mesh.points.shape[0],   
        n_pos               =   torch.tensor(mesh.points, dtype=torch.float),
        n_dirichlet_mask    =   dirichlet_mask,
        n_dirichlet_value   =   dirichlet_value,
        n_source_mask       =   source_mask,
        n_source_value      =   source_value,
        n_displacement      =   torch.from_numpy(displacement).type(dtype),
        n_strain            =   torch.from_numpy(strain).type(dtype),
        n_stress            =   torch.from_numpy(stress).type(dtype),
        edge_index  =   edges.T,
        **gdata
    )
    return graph