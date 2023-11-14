import os
import meshio
import gmsh 
import numpy as np


def cube(
                d:float=0.2, # mesh size
                E:float=1.0, # Young's modulus
                nu:float=0.4, # Poisson's ratio
                a:float=1.0, # length
                p:float=1.0, # pressure
                ):
        gmsh.initialize()
        gmsh.model.add("Cube")
        cube = gmsh.model.occ.addBox(0, 0, 0, a, a, a)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), d)
        gmsh.model.mesh.generate(3)
        gmsh.write("tmp.msh")
        gmsh.finalize()

        mesh = meshio.read("tmp.msh")
        os.remove("tmp.msh")
        mesh.points       = mesh.points
        y_axis            = 1
        z_axis            = 2
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
        mesh.field_data["E"] = np.ones(len(mesh.cells_dict['tetra']),dtype=np.float64) * E 
        mesh.field_data["nu"] = np.ones(len(mesh.cells_dict['tetra']),dtype=np.float64) * nu
        return mesh
