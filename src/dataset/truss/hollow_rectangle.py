import os
import gmsh
import meshio 
import numpy as np
import scipy.sparse


def hollow_rectangle(
    d:float=0.2, # mesh size
    E:float=1.0, # Young's modulus
    A:float=0.4, # Poisson's ratio
    a:float=2.0, # outer length
    p:float=1.0, # pressure
):
    """
        hollow rectangle with bottom fixed boundary and top pressure
    """
    gmsh.initialize()
    gmsh.model.add("Rectangle")
    rectangle = gmsh.model.occ.addRectangle(0, 0, 0, a, a)
    gmsh.model.occ.synchronize()
    # gmsh.model.addPhysicalGroup(2, [outer_rectangle], 1)
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), d)
    gmsh.model.mesh.generate(3)
    gmsh.write("tmp.msh")
    gmsh.finalize()

    meshio.Mesh
    mesh = meshio.read("tmp.msh")

    # breakpoint()
    truss = np.concatenate([mesh.cells_dict['triangle'][:, (0,1)],
                                 mesh.cells_dict['triangle'][:, (1,2)],
                                 mesh.cells_dict['triangle'][:, (2,0)]]) # [n_triangle*3, 2]
    truss.sort(-1)
    mat = scipy.sparse.coo_matrix((np.ones(len(truss)), (truss[:,0], truss[:,1])), shape=(len(mesh.points),len(mesh.points))).tocsr().tocoo()
    truss = np.stack([mat.row, mat.col], -1)
    
    mesh = meshio.Mesh(
        mesh.points,
        {"line": truss},
        point_data = mesh.point_data,
        # cell_data  = {k:v[1:2] for k,v in mesh.cell_data.items()},
        field_data = mesh.field_data,
        cell_sets  = mesh.cell_sets,
    )

    os.remove("tmp.msh")
    mesh.points       = mesh.points[:, :2]
    y_axis            = 1
    is_bottom         = np.isclose(mesh.points[:, y_axis], 0)
    is_top            = np.isclose(mesh.points[:, y_axis], a)
    dirichlet_mask    = np.zeros_like(mesh.points).astype(bool)
    dirichlet_mask[is_bottom, y_axis] = True
    dirichlet_value   = np.zeros_like(mesh.points)
    source_mask       = np.zeros_like(mesh.points).astype(bool)
    source_mask[is_top, y_axis] = True
    source_value      = np.zeros_like(mesh.points)
    source_value[is_top, y_axis]= -p

    mesh.point_data = {
        "dirichlet_mask": dirichlet_mask,
        "dirichlet_value": dirichlet_value,
        "source_mask": source_mask,
        "source_value": source_value,
    }
    
    mesh.field_data["E"] = np.ones(len(mesh.cells_dict['line']),dtype=np.float64) * E 
    mesh.field_data["A"] = np.ones(len(mesh.cells_dict['line']),dtype=np.float64) * A
    
    return mesh