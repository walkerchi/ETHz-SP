import os
import meshio
import gmsh 
import numpy as np


def spherical_shell(
                d:float=0.2, # mesh size
                E:float=1.0, # Young's modulus
                nu:float=0.4, # Poisson's ratio
                a:float=1.0, # inner radius
                b:float=2.0, # outer radius
                p:float=1.0, # pressure
                ):
        gmsh.initialize()
        gmsh.model.add("SphericalShell")
        gmsh.option.setNumber("General.Terminal", 1)
        outer_sphere = gmsh.model.occ.addSphere(0, 0, 0, b)
        inner_sphere = gmsh.model.occ.addSphere(0, 0, 0, a)
        resulting_entities, subtracted_entities = gmsh.model.occ.cut([(3, outer_sphere)], [(3, inner_sphere)], removeObject=True, removeTool=True)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [outer_sphere], 1)
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), d)
        gmsh.model.mesh.generate(3)
        gmsh.write("tmp.msh")
        gmsh.finalize()

        mesh = meshio.read("tmp.msh")
        os.remove("tmp.msh")
        r    = np.sqrt((mesh.points ** 2).sum(-1))
        n    = mesh.points / r.reshape(-1, 1) 
        n_dim= mesh.points.shape[1]
        is_outer_boundary = np.isclose(r, b)
        is_inner_boundary = np.isclose(r, a)
        dirichlet_mask    = is_inner_boundary[:, None].repeat(n_dim, axis=1)
        dirichlet_value   = np.zeros_like(mesh.points)
        source_mask       = is_outer_boundary[:, None].repeat(n_dim, axis=1)
        source_value      = np.zeros_like(mesh.points)
        source_value      = -p * n

        mesh.point_data = {
            "dirichlet_mask": dirichlet_mask,
            "dirichlet_value": dirichlet_value,
            "source_mask": source_mask,
            "source_value": source_value,
        }
        mesh.field_data["E"] = np.ones(len(mesh.cells_dict['tetra']),dtype=np.float64) * E 
        mesh.field_data["nu"] = np.ones(len(mesh.cells_dict['tetra']),dtype=np.float64) * nu
        return mesh
