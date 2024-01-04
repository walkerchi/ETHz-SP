import os
import gmsh
import meshio 
import numpy as np
from scipy.spatial import ConvexHull
import re


def quadrilateral(
    d:float=0.2, # mesh size
    E:float=1.0, # Young's modulus
    nu:float=0.4, # Poisson's ratio
    a:float=2.0, # outer length
    p:float=1.0, # pressure
    seed:int=None,
    fn=lambda p,x,y:p, # pressure function
    boundary="A",
    source  ="B"
):
    boundary_sets = set(re.findall("A-D", boundary))
    source_sets   = set(re.findall("A-D", source))
    assert boundary_sets.intersection(source_sets) == set(), "Boundary and source sets must be disjoint"
    if fn is None:
        fn = lambda x: p

    # Step 1: Generate 4 random points
    if seed is not None:
        np.random.seed(seed)
    points = np.array([[0,0],
                       [a,0],
                       [a,a],
                       [0,a]])
    points += 0.4 * np.random.rand(4, 2)
    # points = np.random.rand(4, 2)

    # Step 2: Create a convex hull
    hull = ConvexHull(points)

    # Initialize Gmsh
    gmsh.initialize()
    gmsh.model.add("convex_hull")

    # Step 3: Define geometry in Gmsh
    for i, point in enumerate(hull.points[hull.vertices]):
        gmsh.model.geo.addPoint(point[0], point[1], 0, tag=i+1)

    lines = []
    for i in range(len(hull.vertices)):
        start = i + 1
        end = i + 2 if i + 2 <= len(hull.vertices) else 1
        lines.append(gmsh.model.geo.addLine(start, end))

    # Creating a Loop and Surface
    ll = gmsh.model.geo.addCurveLoop(lines)
    ps = gmsh.model.geo.addPlaneSurface([ll])


    # Step 4: Define Physical Groups for each boundary and name them
    boundary_names = ['A', 'B', 'C', 'D']
    for i, line in enumerate(lines):
        gmsh.model.addPhysicalGroup(1, [line], tag=i+1)
        gmsh.model.setPhysicalName(1, i+1, boundary_names[i])

    # Mesh Generation
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), d)
    gmsh.model.mesh.generate(3)

    node_ids_per_boundary = {}
    for i, name in enumerate(boundary_names, 1):
        boundary_nodes = set()
        entities = gmsh.model.getEntitiesForPhysicalGroup(1, i)
        for entity in entities:
            # Ensure that 'entity' is a tuple (dim, tag)
            if isinstance(entity, tuple):
                dim, tag = entity
            else:
                # If 'entity' is not a tuple, handle it accordingly
                dim = 1  # replace with the actual dimension if known
                tag = entity
            # Adjusting the unpacking to handle all returned values from getNodes
            node_tags, node_coords, _ = gmsh.model.mesh.getNodes(dim, tag)
            boundary_nodes.update(node_tags)
        node_ids_per_boundary[name] = np.array(list(boundary_nodes)) - 1

    # add all triangles generated to physical group
    gmsh.model.addPhysicalGroup(2, [ps], tag=1)
    gmsh.model.setPhysicalName(2, 1, "all_triangles")

    # Save and finalize
    gmsh.write("tmp.msh")
    gmsh.finalize()

    mesh = meshio.read("tmp.msh")
    # os.remove("tmp.msh")

    mesh.points       = mesh.points[:, :2]
    x_axis            = 0
    y_axis            = 1
    dirichlet_mask    = np.zeros_like(mesh.points).astype(bool)
    dirichlet_value   = np.zeros_like(mesh.points)
    source_mask       = np.zeros_like(mesh.points).astype(bool)
    source_value      = np.zeros_like(mesh.points)
    for name in boundary_names:
        if name in boundary:
            dirichlet_mask[node_ids_per_boundary[name], :] = True
            dirichlet_value[node_ids_per_boundary[name], y_axis] = 0.0

    for name in boundary_names:
        if name in source:
            p0, p1 = mesh.points[node_ids_per_boundary[name][0:2]]
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            n = np.array([-dy, dx])
            n /= np.linalg.norm(n)
            source_mask[node_ids_per_boundary[name], :] = True
            pressure = fn(p, mesh.points[node_ids_per_boundary[name], 0], mesh.points[node_ids_per_boundary[name], 1])
            source_value[node_ids_per_boundary[name], x_axis] = pressure * n[0]
            source_value[node_ids_per_boundary[name], y_axis] = pressure * n[1]
            # source_value[node_ids_per_boundary[name], y_axis] = fn(p, mesh.points[node_ids_per_boundary[name], 0], mesh.points[node_ids_per_boundary[name], 1])

    mesh.point_data = {
        "dirichlet_mask": dirichlet_mask,
        "dirichlet_value": dirichlet_value,
        "source_mask": source_mask,
        "source_value": source_value,
    }
    mesh.field_data["E"] = np.ones(len(mesh.cells_dict['triangle']),dtype=np.float64) * E 
    mesh.field_data["nu"] = np.ones(len(mesh.cells_dict['triangle']),dtype=np.float64) * nu

    return mesh


if __name__ == '__main__':
    quadrilateral()