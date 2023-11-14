import os
import gmsh
import meshio 
import numpy as np


def rectangle(
    d:float=0.2, # mesh size
    E:float=1.0, # Young's modulus
    nu:float=0.4, # Poisson's ratio
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
    os.remove("tmp.msh")

    mesh.points       = mesh.points[:, :2]
    y_axis            = 1
    is_bottom         = np.isclose(mesh.points[:, y_axis], 0)
    is_top            = np.isclose(mesh.points[:, y_axis], a)
    dirichlet_mask    = np.zeros_like(mesh.points).astype(bool)
    dirichlet_mask[is_bottom, :] = True
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
    
    mesh.field_data["E"] = np.ones(len(mesh.cells_dict['triangle']),dtype=np.float64) * E 
    mesh.field_data["nu"] = np.ones(len(mesh.cells_dict['triangle']),dtype=np.float64) * nu

    return mesh