
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
    ], -1) # [n_grid+1, 2]
    top_points    = np.stack([
        np.arange(1, n_grid) * dx,
        np.ones(n_grid - 1) * H,
    ], -1) # [n_grid-1, 2]

    points = np.concatenate([
        bottom_points,
        top_points,
    ], 0)
   
    bottom_nids = np.arange(n_grid + 1) # [n_grid+1]
    top_nids    = np.arange(n_grid + 1, 2*n_grid) # [n_grid-1]
    truss       = np.concatenate([
        np.stack([bottom_nids[:-1], bottom_nids[1:]], -1), # [n_grid, 2]
        np.stack([top_nids[:-1], top_nids[1:]], -1), # [n_grid-2, 2]
        np.stack([bottom_nids[1:-1], top_nids], -1), # [n_grid-1, 2]
        np.stack([bottom_nids[:n_grid//2], top_nids[:n_grid//2]], -1), # [n_grid//2, 2]
        np.stack([bottom_nids[-n_grid//2:], top_nids[-n_grid//2:]], -1), # [n_grid//2, 2]
    ])


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
            "dirichlet_value": np.zeros_like(points),
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
    ], -1) # [n_grid+1, 2]
    top_points    = np.stack([
        np.arange(1, n_grid) * dx,
        np.ones(n_grid - 1) * H,
    ], -1) # [n_grid-1, 2]

    points = np.concatenate([
        bottom_points,
        top_points,
    ], 0)
   
    
    bottom_nids = np.arange(n_grid + 1) # [n_grid+1]
    top_nids    = np.arange(n_grid + 1, 2*n_grid) # [n_grid-1]
    truss       = np.concatenate([
        np.stack([bottom_nids[:-1], bottom_nids[1:]], -1), # [n_grid, 2]
        np.stack([top_nids[:-1], top_nids[1:]], -1), # [n_grid-2, 2]
        np.stack([bottom_nids[1:-1], top_nids], -1), # [n_grid-1, 2]
        np.stack([bottom_nids[0:1], top_nids[0:1]], -1), # [1, 2]
        np.stack([bottom_nids[-1:], top_nids[-1:]], -1), # [1, 2]
        np.stack([bottom_nids[2:n_grid//2+1], top_nids[:n_grid//2-1]], -1), # [n_grid//2 - 1, 2]
        np.stack([bottom_nids[n_grid//2:-2], top_nids[-n_grid//2+1:]], -1), # [n_grid//2 - 1, 2]
    ])


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
            "dirichlet_value": np.zeros_like(points),
            "source_mask": source_mask,
            "source_value": source_value,
            },
        cell_data={
            "E": [np.ones(len(truss), dtype=np.float64) * E],
            "A": [np.ones(len(truss), dtype=np.float64) * A],
        
        },
    )

    return mesh

def k(
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
    ], -1) # [n_grid+1, 2]
    mid_points    = np.stack([
        np.concatenate([np.arange(1, n_grid//2),np.arange(n_grid//2+1,n_grid)]) * dx,
        np.ones(n_grid - 2) * H/2,
    ], -1) # [n_grid-2, 2]
    top_points    = np.stack([
        np.arange(1, n_grid) * dx,
        np.ones(n_grid - 1) * H,
    ], -1) # [n_grid-1, 2]

    points = np.concatenate([
        bottom_points,
        mid_points,
        top_points,
    ], 0)
   
    bottom_nids = np.arange(n_grid + 1) # [n_grid+1]
    mid_nids    = np.arange(n_grid + 1, 2*n_grid-1) # [n_grid-2]
    top_nids    = np.arange(2*n_grid-1, 3*n_grid-2) # [n_grid-1]
    truss       = np.concatenate([
        np.stack([bottom_nids[:-1], bottom_nids[1:]], -1), # [n_grid, 2] bottom
        np.stack([top_nids[:-1], top_nids[1:]], -1), # [n_grid-2, 2] top
        np.stack([bottom_nids[1:n_grid//2], mid_nids[:n_grid//2-1]], -1), # [n_grid-2, 2] bottom-mid
        np.stack([bottom_nids[n_grid//2+1:-1], mid_nids[n_grid//2-1:]], -1), # [n_grid-2, 2] bottom-mid
        np.stack([mid_nids[:n_grid//2-1], top_nids[:n_grid//2-1]], -1), # [n_grid//2-1, 2] mid-top
        np.stack([mid_nids[n_grid//2-1:], top_nids[n_grid//2:]], -1), # [n_grid//2-1, 2] mid-top
        np.stack([bottom_nids[0:1], top_nids[0:1]], -1), # [1, 2] left
        np.stack([bottom_nids[-1:], top_nids[-1:]], -1), # [1, 2] right
        np.stack([bottom_nids[n_grid//2:n_grid//2+1], top_nids[n_grid//2-1:n_grid//2]], -1), # [1, 2] middle
        np.stack([mid_nids[:n_grid//2-1], top_nids[1:n_grid//2]], -1), # [n_grid//2, 2] left-mid
        np.stack([mid_nids[:n_grid//2-1], bottom_nids[2:n_grid//2+1]], -1), # [n_grid//2, 2] left-mid
        np.stack([mid_nids[-n_grid//2+1:], top_nids[-n_grid//2:-1]], -1), # [n_grid//2, 2] mid-right
        np.stack([mid_nids[-n_grid//2+1:], bottom_nids[-n_grid//2-1:-2]], -1), # [n_grid//2, 2] mid-right
    ])


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
            "dirichlet_value": np.zeros_like(points),
            "source_mask": source_mask,
            "source_value": source_value,
            },
        cell_data={
            "E": [np.ones(len(truss), dtype=np.float64) * E],
            "A": [np.ones(len(truss), dtype=np.float64) * A],
        
        },
    )

    return mesh

def baltimore(
    n_grid:int = 12,
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
    assert n_grid % 4 == 0, "n_grid must be multiplier of 4"
    assert n_grid >=8 , "n_grid must be greater than 8"
    dx = L / n_grid
    bottom_points = np.stack([
        np.arange(n_grid + 1) * dx,
        np.zeros(n_grid + 1),
    ], -1) # [n_grid+1, 2]
    mid_points    = np.stack([
        np.arange(1, n_grid+1, 2) * dx,
        np.ones(n_grid//2) * H/2,
    ], -1) # [n_grid//2, 2]
    top_points    = np.stack([
        np.arange(1, n_grid//2) * 2 * dx,
        np.ones(n_grid//2 - 1) * H,
    ], -1) # [n_grid//2 - 1, 2]

    points = np.concatenate([
        bottom_points,
        mid_points,
        top_points,
    ], 0)
   
    
    bottom_nids = np.arange(n_grid + 1) # [n_grid+1]
    mid_nids    = np.arange(n_grid + 1, n_grid + n_grid//2 + 1 ) # [n_grid//2]
    top_nids    = np.arange(n_grid + n_grid//2 + 1, 2*n_grid) # [n_grid//2 - 1]

    truss       = np.concatenate([
        np.stack([bottom_nids[:-1], bottom_nids[1:]], -1), # [n_grid, 2] bottom
        np.stack([top_nids[:-1], top_nids[1:]], -1), # [n_grid-2, 2] top
        np.stack([bottom_nids[2:-2:2], top_nids], -1), # [n_grid//2-1, 2] bottom-top
        np.stack([bottom_nids[1:-1:2], mid_nids], -1), # [n_grid//2, 2] bottom-mid
        np.stack([mid_nids[0:1], top_nids[0:1]], -1), # [1, 2]
        np.stack([mid_nids[-1:], top_nids[-1:]], -1), # [1, 2]
        np.stack([bottom_nids[2::2], mid_nids], -1), # [n_grid//2, 2]
        np.stack([bottom_nids[:-2:2], mid_nids], -1), # [n_grid//2, 2]
        np.stack([mid_nids[1:n_grid//4], top_nids[:n_grid//4-1]], -1), # [n_grid//4-1, 2
        np.stack([mid_nids[n_grid//4:-1], top_nids[-n_grid//4+1:]], -1), # [n_grid//4-1, 2
        # np.stack([bottom_nids[2:n_grid//2+1], top_nids[-n_grid//4+1:]], -1), # [n_grid//2 - 1, 2]
    ])


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
            "dirichlet_value": np.zeros_like(points),
            "source_mask": source_mask,
            "source_value": source_value,
            },
        cell_data={
            "E": [np.ones(len(truss), dtype=np.float64) * E],
            "A": [np.ones(len(truss), dtype=np.float64) * A],
        
        },
    )

    return mesh