
import numpy as np
from linear_elasticity import TriangleSolver, partite
import meshio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

if __name__ == '__main__':
    nodes = np.array([
    [0, 0],  
    [1, 0],  
    [1, 1],  
    [2, 0]  
    ]).astype(np.float64)

    # Define the connectivity for the two triangular elements
    # Meshio expects zero-based indexing
    elements = np.array([
        [0, 1, 2],  # Element 1
        [1, 3, 2]   # Element 2
    ])

    # Create the cells list, containing the cell type and the elements array
    cells = [("triangle", elements)]

    # Create the meshio mesh
    mesh = meshio.Mesh(points=nodes, cells=cells)
    dirichlet_mask = np.zeros_like(mesh.points).astype(bool)
    dirichlet_mask[0, :] = True
    dirichlet_value = np.zeros_like(mesh.points)
    source_mask = np.zeros_like(mesh.points).astype(bool)
    source_mask[2, 1] = True
    source_value = np.zeros_like(mesh.points)
    source_value[2, 1] = -1
    mesh.point_data = {
        "dirichlet_mask": dirichlet_mask,
        "dirichlet_value": dirichlet_value,
        "source_mask": source_mask,
        "source_value": source_value,
    }
    mesh.field_data["E"] = np.ones(len(mesh.cells_dict['triangle']),dtype=np.float64) * 1.0
    mesh.field_data["nu"] = np.ones(len(mesh.cells_dict['triangle']),dtype=np.float64) * 0.3


    solver  = TriangleSolver(mesh)

    solver.K_coo
    solver.ele2msh_edge_torch.to_dense()
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.spy(solver.ele2msh_edge_torch.to_dense())
    plt.show()
    breakpoint()

    fig, ax = plt.subplots()

    # Add the triangular elements as patches
    for element in elements:
        triangle_coords = nodes[element]
        triangle = patches.Polygon(triangle_coords, closed=True, edgecolor='black', fill=False)
        ax.add_patch(triangle)

    # Plot the nodes
    ax.plot(nodes[:, 0], nodes[:, 1], 'o', color='red')

    # Set equal scaling and show the plot
    ax.set_aspect('equal')
    plt.show()