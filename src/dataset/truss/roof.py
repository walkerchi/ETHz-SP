"""
    generate multilayer br
"""
import meshio
import numpy as np

def howe(
    n_grid:int = 10,
    L:float    = 10.0,
    H:float    = 2.0,
    g:float    = 1.0,
    E:float    = 1.0,
    A:float    = 1.0,
    support:int = 1,
):
    """https://pressbooks.library.upei.ca/statics/chapter/trusses-introduction/
        Parameters:
        -----------
            n_grid: int
                number of grid
            L: float
                length of bridge
            H: float
                height of bridge
            g: float
                gravity
            E: float
                Young's modulus
            A: float
                cross section area
            support: int
                0: free
                1: fixed both side one node
                2: fixed both side two node
                ...
        Returns:
        --------
            mesh: meshio.Mesh
                mesh of bridge
    """
    assert n_grid % 2 == 0, "n_grid must be even number"
    dx = L / n_grid
    bottom_points = np.stack([
        np.arange(n_grid + 1) * dx,
        np.zeros(n_grid + 1),
    ], -1)
    slope = H / (dx * n_grid / 2)    
    top_left_points = np.stack([
        bottom_points[1:n_grid//2,0],
        slope * bottom_points[1:n_grid//2,0]], -1)
    top_right_points = np.stack([
        bottom_points[n_grid//2:-1,0],
        2*H - slope * bottom_points[n_grid//2:-1,0]], -1)
    
    points = np.concatenate([
        bottom_points,
        top_left_points,
        top_right_points,
    ], 0)
   
    bottom_nids = np.arange(n_grid + 1) # [n_grid+1]
    top_left_nids = np.arange(n_grid + 1, n_grid + n_grid//2 ) # [n_grid//2-1]
    top_right_nids = np.arange(n_grid + n_grid//2, 2*n_grid) # [n_grid//2]
    
    truss = np.concatenate(
        [np.stack([bottom_nids[:-1], bottom_nids[1:]], -1), # [n_grid, 2]
        np.stack([bottom_nids[0:1], top_left_nids[0:1]], -1), # [1, 2]
        np.stack([bottom_nids[-1:], top_right_nids[-1:]], -1), # [1, 2]
        np.stack([top_left_nids[-1:], top_right_nids[0:1]], -1), # [1, 2]
        np.stack([top_left_nids[:-1], top_left_nids[1:]], -1), # [n_grid//2-2, 2]
        np.stack([top_right_nids[:-1], top_right_nids[1:]], -1),
        np.stack([bottom_nids[1:-1], np.concatenate([top_left_nids,top_right_nids])], -1),
        np.stack([bottom_nids[2:n_grid//2+1], top_left_nids], -1), # slope left [n_grid//2-1, 2]
        np.stack([bottom_nids[n_grid//2:-2], top_right_nids[1:]], -1) # slope  right [n_grid//2-1, 2]
    ], 0)

    y_axis             = 1
    boundary_mask = np.zeros(points.shape, dtype=bool)
    boundary_mask[:support,  y_axis] = True
    boundary_mask[n_grid+1-support:n_grid+1, y_axis] = True
    source_mask   = np.ones(points.shape, dtype=bool)
    source_value  = np.zeros(points.shape)
    source_value[:, y_axis] = -g
    mesh = meshio.Mesh(
        points=points,
        cells={"line": truss},
        point_data={
            "dirichlet_mask": boundary_mask,
            "dirichlet_value": np.zeros(points.shape),
            "source_mask": source_mask,
            "source_value": source_value,
            },
        cell_data={
         
            "E": [np.ones(len(truss), dtype=np.float64) * E],
            "A": [np.ones(len(truss), dtype=np.float64) * A],
        
        },
    )

    return mesh


def pratt(
    n_grid:int = 10,
    L:float    = 10.0,
    H:float    = 2.0,
    g:float    = 1.0,
    E:float    = 1.0,
    A:float    = 1.0,
    support:int = 1,
):
    """https://pressbooks.library.upei.ca/statics/chapter/trusses-introduction/
        Parameters:
        -----------
            n_grid: int
                number of grid
            L: float
                length of bridge
            H: float
                height of bridge
            g: float
                gravity
            E: float
                Young's modulus
            A: float
                cross section area
            support:
                0: free
                1: fixed both side one node
                2: fixed both side two node
                ...
        Returns:
        --------
            mesh: meshio.Mesh
                mesh of bridge
    """
    assert n_grid % 2 == 0, "n_grid must be even number"
    dx = L / n_grid
    bottom_points = np.stack([
        np.arange(n_grid + 1) * dx,
        np.zeros(n_grid + 1),
    ], -1)
    slope = H / (dx * n_grid / 2)    
    top_left_points = np.stack([
        bottom_points[1:n_grid//2,0],
        slope * bottom_points[1:n_grid//2,0]], -1)
    top_right_points = np.stack([
        bottom_points[n_grid//2:-1,0],
        2*H - slope * bottom_points[n_grid//2:-1,0]], -1)
    
    points = np.concatenate([
        bottom_points,
        top_left_points,
        top_right_points,
    ], 0)
   
    bottom_nids = np.arange(n_grid + 1) # [n_grid+1]
    top_left_nids = np.arange(n_grid + 1, n_grid + n_grid//2 ) # [n_grid//2-1]
    top_right_nids = np.arange(n_grid + n_grid//2, 2*n_grid) # [n_grid//2]

    truss = np.concatenate(
        [np.stack([bottom_nids[:-1], bottom_nids[1:]], -1), # [n_grid, 2]
        np.stack([bottom_nids[0:1], top_left_nids[0:1]], -1), # [1, 2]
        np.stack([bottom_nids[-1:], top_right_nids[-1:]], -1), # [1, 2]
        np.stack([top_left_nids[-1:], top_right_nids[0:1]], -1), # [1, 2]
        np.stack([top_left_nids[:-1], top_left_nids[1:]], -1), # [n_grid//2-2, 2]
        np.stack([top_right_nids[:-1], top_right_nids[1:]], -1),
        np.stack([bottom_nids[1:-1], np.concatenate([top_left_nids,top_right_nids])], -1),
        np.stack([bottom_nids[1:n_grid//2], np.concatenate([top_left_nids[1:], top_right_nids[0:1]])], -1), # slope left [n_grid//2-1, 2]
        np.stack([bottom_nids[n_grid//2+1:-1], top_right_nids[:-1]], -1) # slope  right [n_grid//2-1, 2]
    ], 0)

    y_axis             = 1
    boundary_mask = np.zeros(points.shape, dtype=bool)
    boundary_mask[:support,  y_axis] = True
    boundary_mask[n_grid+1-support:n_grid+1, y_axis] = True
    source_mask   = np.ones(points.shape, dtype=bool)
    source_value  = np.zeros(points.shape)
    source_value[:, y_axis] = -g
    mesh = meshio.Mesh(
        points=points,
        cells={"line": truss},
        point_data={
            "dirichlet_mask": boundary_mask,
            "dirichlet_value": np.zeros(points.shape),
            "source_mask": source_mask,
            "source_value": source_value,
            },
        cell_data={
         
            "E": [np.ones(len(truss), dtype=np.float64) * E],
            "A": [np.ones(len(truss), dtype=np.float64) * A],
        
        },
    )

    return mesh
