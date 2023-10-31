import numpy as np
import numba as nb
import torch_geometric as pyg
import torch
import os
import gmsh
import meshio
import scipy
import matplotlib.pyplot as plt
from skfem import *
from skfem.helpers import ddot, trace, sym_grad, eye
from skfem.models.elasticity import linear_elasticity, linear_stress, lame_parameters
from skfem.helpers import dot, ddot, dd


def vm_stress_2d(stress):
    """
        Parameters:
        -----------
            stress: [3] [σxx, σyy, σxy]
    """
    stress_vm = np.sqrt(stress[0]**2 + stress[1]**2 - stress[0]*stress[1] + 3*stress[2]**2)
    return stress_vm

class UniAxial:
   
    def __init__(self, n_grid=10, cache_dir=".cache"):
        self.cache_dir = cache_dir
        self.n_grid    = n_grid
        self.mesh_path = os.path.join(self.cache_dir, f"{self.__class__.__name__}_{n_grid}.msh")
        self.fem_path  = os.path.join(self.cache_dir, f"{self.__class__.__name__}_{n_grid}.fem.vtk")
        self.ana_path  = os.path.join(self.cache_dir, f"{self.__class__.__name__}_{n_grid}.ana.vtk")
        self.parameters = {
            "E": 1.0,
            "nu": 0.4,
            "f" : 1.0,
            "L" : 1.0,
            "A" : 1.0,
        }
        
        if not os.path.exists(self.mesh_path):
            self.mesh_gen()

    def mesh_gen(self):

        gmsh.initialize()

        # Create a new Gmsh model
        gmsh.model.add("rectangle")

        # Define corner points of the rectangle
        A, L = self.parameters["A"], self.parameters["L"]
        lc = 1/self.n_grid  # characteristic length
        p1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
        p2 = gmsh.model.geo.addPoint(L, 0, 0, lc)
        p3 = gmsh.model.geo.addPoint(L, A, 0, lc)
        p4 = gmsh.model.geo.addPoint(0, A, 0, lc)

        # Create lines using the defined points
        l1 = gmsh.model.geo.addLine(p1, p2)
        l2 = gmsh.model.geo.addLine(p2, p3)
        l3 = gmsh.model.geo.addLine(p3, p4)
        l4 = gmsh.model.geo.addLine(p4, p1)

        # Loop the lines to create a surface
        ll = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
        surf = gmsh.model.geo.addPlaneSurface([ll])

        # Tag boundaries for boundary conditions
        gmsh.model.addPhysicalGroup(1, [l1], 1)
        gmsh.model.addPhysicalGroup(1, [l2], 2)
        gmsh.model.addPhysicalGroup(1, [l3], 3)
        gmsh.model.addPhysicalGroup(1, [l4], 4)
        gmsh.model.addPhysicalGroup(2, [surf], 5)

        gmsh.model.setPhysicalName(1, 1, "bottom_boundary")
        gmsh.model.setPhysicalName(1, 2, "right_boundary")
        gmsh.model.setPhysicalName(1, 3, "top_boundary")
        gmsh.model.setPhysicalName(1, 4, "left_boundary")
        gmsh.model.setPhysicalName(2, 5, "surface")

        # Generate 2D mesh
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)

        # Save the mesh
        os.makedirs(self.cache_dir, exist_ok=True)
        gmsh.write(self.mesh_path)

        # Cleanup
        gmsh.finalize()

    def vis(self, target="mesh"):
        assert target in ["mesh", "ana", "fem", "gnn"]
        assert os.path.exists(self.mesh_path)

        if target == "mesh":
            mesh = meshio.read(self.mesh_path)

            # plot the 2d mesh using matplotlib
            plt.figure(figsize=(10, 10))
            plt.triplot(mesh.points[:, 0], mesh.points[:, 1], mesh.cells_dict['triangle'], linewidth=0.5)
            plt.show()
        
        else:
            vtk_path = {
                "ana":self.ana_path,
                "fem":self.fem_path,
            }[target]
            mesh = meshio.read(vtk_path)
           
            # plot the 2d mesh using matplotlib
            fig, axes = plt.subplots(ncols=3, figsize=(15, 5))
            for ax in axes:
                ax.triplot(mesh.points[:,0], mesh.points[:,1], mesh.cells_dict['triangle'], linewidth=0.5, color="#00023D")
                ax.axis("off")
            # add stress tensor field on the mesh
            stress = mesh.point_data["stress"]
            strain = mesh.point_data["strain"]
            displacement = mesh.point_data["displacement"]
            axes[0].quiver(mesh.points[:,0], mesh.points[:,1], displacement[:,0], displacement[:,1], color="#F08650")
            axes[1].quiver(mesh.points[:,0], mesh.points[:,1], strain[:,0], strain[:,2], color="#75F94D")
            axes[2].quiver(mesh.points[:,0], mesh.points[:,1], stress[:,0], stress[:,2], color="#7E84F7")
            axes[0].set_title("displacement")
            axes[1].set_title("strain")
            axes[2].set_title("stress")
            plt.show()

    def analytical_solve(self):
        f,L,A,E,nu      = self.parameters["f"], self.parameters["L"], self.parameters["A"], self.parameters["E"], self.parameters["nu"]
        mesh            = meshio.read(self.mesh_path)
        x, y            = mesh.points[:, 0], mesh.points[:, 1]
        
        stress_x        = f / A / E
        stress_y        = - nu * stress_x
        stress          = np.zeros((len(x), 3))
        stress[:,0]     = stress_x
        stress[:,1]     = stress_y
        
        displacement_x  = f * L / E 
        displacement_y  = - nu * displacement_x
        displacement    = np.zeros((len(x), 2))
        displacement[:,0] = displacement_x
        displacement[:,1] = displacement_y

        strain_x        = f / A / E 
        strain_y        = - nu * f / A / E
        strain          = np.zeros((len(x), 3))
        strain[:,0]     = strain_x
        strain[:,1]     = strain_y

        mesh.point_data["displacement"] = displacement
        mesh.point_data["strain"] = strain
        mesh.point_data["stress"] = stress
        mesh.write(self.ana_path, file_format="vtk")

        return self


    def skfem_solve(self):
        E, nu, f = self.parameters["E"], self.parameters["nu"], self.parameters["f"]

        # Read the mesh using meshio
        mesh            = Mesh.load(self.mesh_path)
        element         = ElementTriP1()
        vector          = ElementVector(element)
        mapping         = MappingIsoparametric(mesh, element)
        vector_basis    = Basis(mesh, vector, MappingIsoparametric(mesh, element))

        # Applying boundary conditions
        # Here we assume a unit displacement in x-direction on the Dirichlet boundary
        u               = vector_basis.zeros()
        left_dofs       = vector_basis.get_dofs(mesh.boundaries["left_boundary"])
        right_dofs      = vector_basis.get_dofs(mesh.boundaries["right_boundary"])
        u[right_dofs.nodal["u^1"]]    = self.parameters["f"] / len(right_dofs.nodal["u^1"])

        K = asm(linear_elasticity(*lame_parameters(E, nu)), vector_basis)
        u = solve(*condense(K, x=u, D=right_dofs))
        
        matrix_basis    = vector_basis.with_element(ElementVector(vector))
        C               = linear_stress(*lame_parameters(E, nu))
        u_ele           = vector_basis.interpolate(u)
        strain_ele      = sym_grad(u_ele)
        stress_ele      = C(strain_ele)
        strain          = matrix_basis.project(strain_ele)
        stress          = matrix_basis.project(stress_ele)

        # visualize mesh_result using vtk
        points_2d       = np.stack([mesh.p[0],mesh.p[1]],-1)
        displacement_2d = u[vector_basis.nodal_dofs].T
        strain_2d       = strain[matrix_basis.nodal_dofs].T.reshape(-1,2,2)
        stress_2d       = stress[matrix_basis.nodal_dofs].T.reshape(-1,2,2)
        points_3d       = np.zeros((mesh.p.shape[1], 3))
        displacement_3d = np.zeros((mesh.p.shape[1], 3))
        strain_3d       = np.zeros((mesh.p.shape[1], 3))
        stress_3d       = np.zeros((mesh.p.shape[1], 3))
        points_3d[:,:2] = points_2d
        displacement_3d[:,:2] = displacement_2d
        strain_3d[:,:2] = strain_2d.diagonal(axis1=-2,axis2=-1)
        strain_3d[:, 2] = strain_2d[:, 1, 0]
        stress_3d[:,:2] = stress_2d.diagonal(axis1=-2,axis2=-1)
        stress_3d[:, 2] = stress_2d[:, 1, 0]
        mesh = meshio.read(self.mesh_path)
        mesh.point_data["displacement"] = displacement_2d
        mesh.point_data["strain"] = strain_3d
        mesh.point_data["stress"] = stress_3d #[stress_xx, stress_yy, stress_xy]
        mesh.write(self.fem_path, file_format="vtk", binary=False)
        return self
        # meshio.read(self.fem_path)

        # import pyvista as pv
        # mesh = pv.read(self.fem_path)
        # p = pv.Plotter()
        # p.add_mesh(mesh, scalars="displacement", cmap="viridis", show_edges=True)
        # p.add_scalar_bar(title="Displacement")
        # p.show()
        # return self

 

    def as_element_graph(self):
        """ Represent each (triangle, etc..)element as the node in the graph.
        The graph connectivity is the edge between two elements.

        The Node Features is 
        [element_center_coord, E, nu, # body properties
        S, n, d, # edge properties
        x, dx] # applied force
        
        """
        mesh       = meshio.read(self.mesh_path)
        ana_result = meshio.read(self.ana_path)
        fem_result = meshio.read(self.fem_path)
        node_coord = torch.from_numpy(mesh.points)[:, :2]
        elements   = torch.from_numpy(mesh.cells_dict["triangle"])
        n_basis    = elements.shape[1]
        n_ele      = elements.shape[0]
        n_point    = node_coord.shape[0]

        # prepare edge connectivity
        edges      = torch.vmap(lambda x:torch.stack(torch.meshgrid(x, x),-1))(elements).view(-1, 2).T 
        edges      = torch.sparse_coo_tensor(
                        edges, 
                        torch.ones(edges.shape[1]), 
                        size=(node_coord.shape[0], node_coord.shape[0])
                        ).coalesce()
        edges      = edges.indices().numpy()
        eids       = scipy.sparse.coo_matrix(
                        (np.arange(edges.shape[1]), (edges[0], edges[1])),
                        shape=(node_coord.shape[0], node_coord.shape[0])
                    ).tocsr()
        
        elements   = np.sort(elements, -1)
        ele_eids   = np.full([n_ele, 3], -1  )  # [eid1(0,1), eid2(0,2), eid3(1,2)]
        ele_esin   = np.full([n_ele, 3], -1.0)  # [(0,1)x(0,2), (0,2)x(0,1), (1,2)x(1,0)]
        ele_elid   = np.full([n_ele, 3], -1  )  # [eleid1(0,1), eleid2(0,2), eleid3(1,2)]
        ele_eids[:, 0] = eids[elements[:,0], elements[:,1]]
        ele_eids[:, 1] = eids[elements[:,0], elements[:,2]]
        ele_eids[:, 2] = eids[elements[:,1], elements[:,2]]
        ele_esin[:, 0] = np.cross(node_coord[elements[:,1]] - node_coord[elements[:,0]], node_coord[elements[:,2]] - node_coord[elements[:,0]])
        ele_esin[:, 1] = np.cross(node_coord[elements[:,2]] - node_coord[elements[:,0]], node_coord[elements[:,1]] - node_coord[elements[:,0]])
        ele_esin[:, 2] = np.cross(node_coord[elements[:,2]] - node_coord[elements[:,1]], node_coord[elements[:,0]] - node_coord[elements[:,1]])
        ele_elid[:, :] = np.arange(n_ele)[:,None].repeat(3, -1)
        assert (ele_esin != 0.0).all(), f"got zero area element {ele_esin[ele_esin==0.0]}"
        ele_edges      = np.full([edges.shape[1],2], -1, dtype=np.int64)
        pos_mask       = ele_esin.flatten() > 0 # element on left side of the edge
        neg_mask       = ele_esin.flatten() < 0 # element on right side of the edge
        ele_edges[ele_eids.flatten()[pos_mask],0] = ele_elid.flatten()[pos_mask]
        ele_edges[ele_eids.flatten()[neg_mask],1] = ele_elid.flatten()[neg_mask]
        
        ele_edges      = ele_edges[(ele_edges !=-1.0).all(-1)] # remove edge with only one element
        ele_edges      = torch.from_numpy(ele_edges).T

        # prepare mask data
        left_boundary_tag = mesh.field_data["left_boundary"][0]
        right_boundary_tag= mesh.field_data["right_boundary"][0]
        lines = mesh.get_cells_type("line")
        line_tag = mesh.get_cell_data("gmsh:physical", "line")
        left_boundary_edges = lines[line_tag == left_boundary_tag]
        right_boundary_edges= lines[line_tag == right_boundary_tag]
        left_boundary_nodes = np.unique(left_boundary_edges.flatten())
        right_boundary_nodes= np.unique(right_boundary_edges.flatten())
        boundary_nodes      = np.unique(line_tag.flatten())
        point2ele = scipy.sparse.coo_matrix(
            (
                np.ones(n_ele * n_basis),(
                    np.arange(n_ele*3),
                    elements.flatten())
                ),
            shape=(n_ele * 3, n_point)
        ).tocsr()
        left_boundary_nodes_mask = np.zeros(n_point, dtype=bool)
        right_boundary_nodes_mask= np.zeros(n_point, dtype=bool)
        boundary_nodes_mask      = np.zeros(n_point, dtype=bool)
        left_boundary_nodes_mask[left_boundary_nodes]   = True
        right_boundary_nodes_mask[right_boundary_nodes] = True
        boundary_nodes_mask[boundary_nodes]             = True
        left_boundary_element_mask = (point2ele @ left_boundary_nodes_mask).reshape(n_ele, 3).sum(-1) > 1
        right_boundary_element_mask= (point2ele @ right_boundary_nodes_mask).reshape(n_ele, 3).sum(-1) > 1
        boundary_element_mask      = (point2ele @ boundary_nodes_mask).reshape(n_ele, 3).sum(-1) > 1

        # prepare body properties: [element_center_coord, E, nu] 
        element_coord        = node_coord[elements] # [n_ele, n_basis, n_dim]
        element_center_coord = element_coord.mean(1) # [n_ele, n_dim]
        element_parameter    = torch.zeros([n_ele, 2]) # [n_ele, 2]
        element_parameter[:, 0] = self.parameters["E"]
        element_parameter[:, 1] = self.parameters["nu"]

        # prepare edge properties: [S, n, d] 
        element_edge_coord   = element_coord - torch.roll(element_coord, 1, -1) # [n_ele, n_basis, n_dim]
        element_surface_size = torch.norm(element_edge_coord, dim=-1) # [n_ele, n_basis]
        element_surface_norm = torch.cross(
                torch.cat([element_edge_coord, torch.zeros(n_ele, n_basis, 1)], -1),
                torch.cross(
                    torch.cat([element_edge_coord, torch.zeros(n_ele, n_basis, 1)], -1),
                    torch.cat([element_coord - torch.roll(element_coord, 2, -1),  torch.zeros(n_ele, n_basis, 1)], -1)
                ,-1),-1)[:, :, :2]
        element_center_dist  = np.abs(np.cross(
            element_edge_coord - element_center_coord[:, None, :],
            element_edge_coord, -1
        )) / element_surface_size

        # prepare external force properties: [x, dx]
        element_force_coord       = torch.zeros([n_ele, 2]).double()
        element_force_vec         = torch.zeros([n_ele, 2]).double()
        right_boundary_nodes_mask = (point2ele @ right_boundary_nodes_mask).reshape(n_ele, 3).astype(bool)
        right_boundary_nodes_mask[right_boundary_nodes_mask.sum(1) == 1] = False # only one node is at boundary which should be neglected 
        n_right_boundary_element  = right_boundary_element_mask.sum()            # because the edge is shared by two nodes
        assert (right_boundary_nodes_mask.sum(1) == 2).sum() == n_right_boundary_element
        element_force_coord[right_boundary_element_mask] = element_coord[right_boundary_nodes_mask, :].reshape(n_right_boundary_element, 2, 2).mean(1)
        element_force_vec[right_boundary_element_mask, 0]= self.parameters["f"] / self.n_grid   

        graph      = pyg.data.Data(
            x          = torch.cat([
                element_center_coord,          # [n_ele, 2]
                element_parameter,             # [n_ele, 2]
                element_surface_size,          # [n_ele, 3]
                element_surface_norm.reshape(n_ele,6),# [n_ele, 6]
                element_center_dist,           # [n_ele, 3]
                element_force_coord,           # [n_ele, 2]
                element_force_vec              # [n_ele, 2]
            ], -1),
            points     = torch.from_numpy(elements), # [n_ele, n_basis]
            fix_mask   = torch.from_numpy(left_boundary_element_mask), # [n_ele]
            force_mask = torch.from_numpy(right_boundary_element_mask), # [n_ele]
            boundary_mask = torch.from_numpy(boundary_element_mask),    # [n_ele]
            edge_index = ele_edges
        )
        graph      = pyg.transforms.AddSelfLoops()(graph)
        graph      = pyg.transforms.ToUndirected()(graph)
        graph      = pyg.transforms.RemoveDuplicatedEdges()(graph)

        return graph
            

    def as_graph(self):
        """ Represent each point as a graph 
        The Node Features is
        [node_coord, force] # node properties
        """
        mesh       = meshio.read(self.mesh_path)
        ana_result = meshio.read(self.ana_path)
        fem_result = meshio.read(self.fem_path)
        node_coord = torch.from_numpy(mesh.points)[:, :2]
        elements   = torch.from_numpy(mesh.cells_dict["triangle"])
        n_basis    = elements.shape[1]
        n_ele      = elements.shape[0]
        n_point    = node_coord.shape[0]
        
        edges      = torch.vmap(lambda x:torch.stack(torch.meshgrid(x, x),-1))(elements).view(-1, 2)
        
        
        left_boundary_tag = mesh.field_data["left_boundary"][0]
        right_boundary_tag= mesh.field_data["right_boundary"][0]
        lines = mesh.get_cells_type("line")
        line_tag = mesh.get_cell_data("gmsh:physical", "line")
        left_boundary_edges = torch.from_numpy(lines[line_tag == left_boundary_tag])
        right_boundary_edges= torch.from_numpy(lines[line_tag == right_boundary_tag])
        boundary_nodes      = torch.from_numpy(lines)
        left_boundary_nodes = torch.unique(left_boundary_edges.flatten())
        right_boundary_nodes= torch.unique(right_boundary_edges.flatten())
        boundary_nodes      = torch.unique(boundary_nodes.flatten())
        
        fix_mask   = torch.zeros([node_coord.shape[0],2], dtype=torch.bool)
        fix_mask[left_boundary_nodes, 0] = True # only x axis is fixed
        force_mask = torch.zeros([node_coord.shape[0],2], dtype=torch.bool)
        force_mask[right_boundary_nodes, 0] = True # only x axis is applied force
        force      = torch.zeros([node_coord.shape[0],2])
        force[right_boundary_nodes, 0] = self.parameters["f"] / len(right_boundary_nodes)
        boundary_mask = torch.zeros([node_coord.shape[0],2], dtype=torch.bool)
        boundary_mask[boundary_nodes, 0] = True # only x axis is applied force

        graph = pyg.data.Data(
            # node features
            x           = torch.cat([
                node_coord,    # [n_point, n_dim]
                force,         # [n_point, n_dim]
            ], -1),
            coord       = node_coord,    # [n_point, n_dim]
            fix_mask    = fix_mask,      # [n_point, n_dim]
            force_mask  = force_mask,    # [n_point, n_dim]
            force       = force,         # [n_point, n_dim]
            boundary_mask = boundary_mask,# [n_point, n_dim]
            # ground truth
            ana_displacement = torch.from_numpy(ana_result.point_data["displacement"]), # [n_point, n_dim]
            ana_strain       = torch.from_numpy(ana_result.point_data["strain"]),       # [n_point, n_dim]
            ana_stress       = torch.from_numpy(ana_result.point_data["stress"]),       # [n_point, n_dim]
            ana_vm_stress    = torch.vmap(vm_stress_2d, ana_reuslt.point_data["stress"]),# [n_point]
            fem_displacement = torch.from_numpy(fem_result.point_data["displacement"]), # [n_point, n_dim]
            fem_strain       = torch.from_numpy(fem_result.point_data["strain"]),       # [n_point, n_dim]
            fem_stress       = torch.from_numpy(fem_result.point_data["stress"]),       # [n_point, n_dim]
            fem_vm_stress    = torch.vmap(vm_stress_2d, fem_reuslt.point_data["stress"]),# [n_point]
            # connectivity
            edge_index  = edges.T,       # [2, n_edge]
        )
        graph = pyg.transforms.AddSelfLoops()(graph)
        graph = pyg.transforms.ToUndirected()(graph)
        graph = pyg.transforms.RemoveDuplicatedEdges()(graph)
        breakpoint()
        return graph
        
      
        

    def manual_solve(self):
        mesh    = meshio.read(self.file_path)
        points  = mesh.points  # [n_points, n_dim]
        n_point = points.shape[0]
        elements= mesh.cells_dict['triangle']
        edges   = mesh.cells_dict['line']
        n_basis = elements.shape[1]
        n_ele   = elements.shape[0]

        src,dst = edges.T
        src,dst = (np.concatenate([np.arange(n_point), src, dst]),
                   np.concatenate([np.arange(n_point), dst, src]))
        n_edge  = edges.shape[0] # nnz in stiffness matrix
        edge_id = scipy.sparse.coo_matrix((
            np.arange(n_edge),
            (src, dst)
        ), shape = (n_point, n_point)).to_csr()

        n_row   = n_point
        n_col   = n_ele * n_basis
        ele2msh_node = scipy.sparse.coo_matrix((
            np.ones(n_col),
            (
                elements.flatten(),
                np.arange(n_col)
            )
        ), shape=(n_row, n_col))

        msh_edges = np.stack([elements[:,:,None].repeat(n_basis,2), elements[:,None,:].repeat(n_basis,1)], -1)
        msh_edges = msh_edges.transpose(3, 0, 1, 2).reshape(2, -1) # potential error for axis 1(n_basis),2(n_basis) transpose
        msh_src, msh_dst = msh_edges
        msh_edge_id = edge_id[msh_src, msh_dst].toarray().flatten()

        ele2msh_edge = scipy.sparse.coo_matrix((
            np.arange(n_ele*n_basis*n_basis),
            (
                msh_edge_id,
                np.arange(n_ele*n_basis*n_basis)
            )            
        ), shape=(n_edge, n_ele*n_basis*n_basis)).to_csr()

        shape_fn_W = np.array([ # shape_fn_W @ [r,s,1] = [x, y, useless]
            [-1, -1, 1],
            [1, 0, 0],
            [0, 1, 0]
        ])
        
        shape_fn_grad = shape_fn_W[:,:2] # [n_basis, n_dim]
        element_coord = points[elements] # [n_ele, n_basis, n_dim] 
        # jacobian_{e,i,j} = shape_fn_grad_{b,i} * element_coord_{e,b,j} global -> local
        J     = np.einsum("bi, ebj->eij", element_coord, shape_fn_grad)
        J_det = np.linalg.det(J) # [n_ele] # 
        J_inv = np.linalg.inv(J) # [n_ele, n_dim, n_dim]  local -> global
        stress= self.parameters["lambda"] 
        sigma = self.parameters["lambda"] * shape_fn_grad.sum(-1)

        # K # [n_ele, n_basis, n_basis, n_dim, n_dim]


        # N1 = 1 - r - s
        # N2 = r
        # N3 = s
        # dN1/dx = -1
        # dN1/dy = -1
        # dN2/dx = 1
        # dN2/dy = 0
        # dN3/dx = 0
        # dN3/dy = 1
        B = np.array([
            [-1, 0, ]
        ])



        mesh = meshio.Mesh(
        points,
        cells,
        # Optionally provide extra data on points, cells, etc.
        point_data={"T": [0.3, -1.2, 0.5, 0.7, 0.0, -3.0]},
        # Each item in cell data must match the cells array
        cell_data={"a": [[0.1, 0.2], [0.4]]},
    )

if __name__ == '__main__':
    UniAxial(n_grid=10).skfem_solve().vis("fem")
    # UniAxial(n_grid=10).analytical_solve().vis("ana")
    # UniAxial(n_grid=10).as_graph()