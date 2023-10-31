import os
import toml
import numpy as np
import gmsh
import meshio
import scipy.spatial
import pyvista as pv
import torch
import torch_geometric as pyg
from skfem import *
from skfem.models.elasticity import linear_elasticity, linear_stress, lame_parameters
from skfem.helpers import grad, ddot, dot,sym_grad

def vector_spherical_to_cartesian(v, theta, phi):
    """Turn vectors in spherical cooridnates to cartesian cooridnates
        Parameters:
        -----------
            v : (n, 3)
            theta: (n,)
            phi: (n,)
        Returns:
        --------
            u_cartesian: (n, 3)
    """
    sin_t, sin_p = np.sin(theta), np.sin(phi)
    cos_t, cos_p = np.cos(theta), np.cos(phi)
    T = np.stack([
        np.stack([sin_t * cos_p, cos_t * cos_p, -sin_p],-1),
        np.stack([sin_t * sin_p, cos_t * sin_p, cos_p],-1),
        np.stack([cos_t, -sin_t, np.zeros_like(theta)],-1)
    ], -2)
    u_cartesian = np.einsum("nij,nj->ni", T, v)
    return u_cartesian

def vector_carteisan_to_spherical(v, position):
    """
        Parameters:
        -----------
            v : (n, 3)
            position: (n, 3)
        Returns:
        --------
            u_spherical: (n, 3)
    """
    theta, phi = np.arctan2(position[:, 1], position[:, 0]), np.arccos(position[:, 2] / np.sqrt((position**2).sum(-1)))
    sin_t, sin_p = np.sin(theta), np.sin(phi)
    cos_t, cos_p = np.cos(theta), np.cos(phi)
    T = np.stack([
        np.stack([sin_t * cos_p, sin_t * sin_p, cos_t],-1),
        np.stack([cos_t * cos_p, cos_t * sin_p, -sin_t],-1),
        np.stack([-sin_p, cos_p, np.zeros_like(theta)],-1)
    ], -2)
    u_spherical = np.einsum("nij,nj->ni", T, v)
    return u_spherical

def matrix_spherical_to_cartesian(u_rr, u_tt, theta, phi):
    """Turn matrixes in spherical cooridnates to cartesian cooridnates
        Parameters:
        -----------
            u_rr: (n,)
            u_tt: (n,)
            theta: (n,)
            phi: (n,)
        Returns:
        --------
            u_cartesian: (n, 3, 3)
    """
    sin_p, cos_p = np.sin(phi), np.cos(phi)
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    T = np.stack([
        np.stack([sin_t * cos_p, cos_t * cos_p, -sin_p],-1),
        np.stack([sin_t * sin_p, cos_t * sin_p, cos_p],-1),
        np.stack([cos_t, -sin_t, np.zeros_like(u_rr)],-1)
    ], -2) # (n, 3, 3)

    u_sphere = np.zeros((len(u_rr), 3, 3))
    u_sphere[:, 0, 0] = u_rr
    u_sphere[:, 1, 1] = u_tt
    u_sphere[:, 2, 2] = u_tt
    u_cartesian = np.einsum("nij, njk, nmk->nim", T, u_sphere, T)

    return u_cartesian

def vm_stress_cartesian(stress):
    """Calculate the von Mises stress from the stress tensor in cartesian coordinates.
        Parameters:
        -----------
        stress: (n, 3, 3)
        Returns:
        --------
        vm_stress: (n,)
    """
    return np.sqrt(0.5 * ((stress[:, 0, 0] - stress[:, 1, 1])**2 +
                          (stress[:, 1, 1] - stress[:, 2, 2])**2 + 
                          (stress[:, 2, 2] - stress[:, 0, 0])**2 + 
                          6 * (stress[:, 0, 1]**2 + stress[:, 1, 2]**2 + stress[:, 2, 0]**2)))

def vm_stress_spherical(radial_stress, hoop_stress):
    """Calculate the von Mises stress from the stress tensor in spherical coordinates.
        Parameters:
        -----------
        radial_stress: (n,)
        hoop_stress: (n,)
        Returns:
        --------
        vm_stress: (n,)
    """
    return np.sqrt(2 * (radial_stress - hoop_stress)**2)





class SphericalShell:
    """
    Pressurized Shell
    """
    def __init__(self, 
                d:float=0.2, # mesh size
                E:float=1.0, # Young's modulus
                nu:float=0.4, # Poisson's ratio
                a:float=1.0, # inner radius
                b:float=2.0, # outer radius
                p:float=1.0, # pressure
                cache_dir:str=".cache",
                overwrite:bool=False):
        self.cache_dir = os.path.join(cache_dir,self.__class__.__name__.lower())
        os.makedirs(self.cache_dir, exist_ok=True)
        self.mesh_parameters = {"a" : a, "b" : b, "d" : d}
        self.pde_parameters = {"E" : E, "nu" : nu, "p" : p}
        
        mesh_parameters_str = "_".join([f"{k}={v}" for k,v in self.mesh_parameters.items()])
        pde_parameters_str  = "_".join([f"{k}={v}" for k,v in self.pde_parameters.items()])
        self.mesh_path    = os.path.join(self.cache_dir, f"{mesh_parameters_str}.msh")
        self.ana_path     = os.path.join(self.cache_dir, f"{mesh_parameters_str}_{pde_parameters_str}.ana.npz")
        self.fem_path     = os.path.join(self.cache_dir, f"{mesh_parameters_str}_{pde_parameters_str}.fem.npz")
        if not os.path.exists(self.mesh_path) or overwrite:
            self.mesh_gen(a, b, d)

        self.config_path = os.path.join(self.cache_dir, f"{mesh_parameters_str}_{pde_parameters_str}.toml")
        if not os.path.exists(self.config_path) or overwrite:
            self.config_gen()

    def __str__(self):
        mesh_para = "_".join([str(v) for v in self.mesh_parameters.values()])
        pde_para  = "_".join([str(v) for v in self.pde_parameters.values()])
        return f"SphericalShell_{mesh_para}_{pde_para}"
    
    def __repr__(self):
        return (f"SphericalShell({' '.join([f'{k}={v}' for k,v in self.mesh_parameters.items()])} "
                " ".join([f'{k}={v}' for k,v in self.pde_parameters.items()]),
                ")")

    @classmethod
    def load_config(cls, config_path:str):
        with open(config_path, "r") as f:
            config = toml.load(f)
        cache_dir = os.path.dirname(config_path)
        return cls(**config["mesh"], **config["pde"], cache_dir=cache_dir)
     
    def config_gen(self):
        with open(self.config_path, "w") as f:
            toml.dump({"mesh":self.mesh_parameters, "pde":self.pde_parameters}, f)
    
    def mesh_gen(self, a, b, d, vis=False):
        """Generate mesh for a spherical shell with inner radius a, outer radius b, and mesh size d.
        The center of the sphere is at the origin. (0.0, 0.0, 0.0)
            Parameters:
            -----------
            a: float
                inner radius
            b: float
                outer radius
            d: float
                mesh size
        """

        # Initialize Gmsh
        gmsh.initialize()

        # Create a new model
        gmsh.model.add("SphericalShell")

        # Using the OpenCASCADE kernel
        gmsh.option.setNumber("General.Terminal", 1)

        # Create the two spheres representing the inner and outer boundaries
        outer_sphere = gmsh.model.occ.addSphere(0, 0, 0, b)
        inner_sphere = gmsh.model.occ.addSphere(0, 0, 0, a)

        # Use boolean fragments to create the shell between the two spheres
        resulting_entities, subtracted_entities = gmsh.model.occ.cut([(3, outer_sphere)], [(3, inner_sphere)], removeObject=True, removeTool=True)
         
        # Synchronize the data from Python to Gmsh
        gmsh.model.occ.synchronize()

        # # add physical tags to the inner and outer boundaries
        # shell_surfaces = gmsh.model.getBoundary(resulting_entities, oriented=False, combined=False)
        # gmsh.model.occ.cut(shell_surfaces[0:1], shell_surfaces[1:2], removeObject=True, removeTool=True)
        
        # gmsh.model.occ.synchronize()
        # inner_boundary_tag = gmsh.model.addPhysicalGroup(2, shell_surfaces[1], 1)  # inner surface
        # outer_boundary_tag = gmsh.model.addPhysicalGroup(2, shell_surfaces[0], 2)  # outer surface

        # # Name the physical groups
        # gmsh.model.setPhysicalName(2, inner_boundary_tag, "inner boundary")
        # gmsh.model.setPhysicalName(2, outer_boundary_tag, "boundary")

        # Define a physical volume for the shell
        gmsh.model.addPhysicalGroup(3, [outer_sphere], 1)

        # Define mesh size
        # gmsh.model.mesh.setSize(shell_entities, d)
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), d)

        # Mesh the geometry
        gmsh.model.mesh.generate(3)

        # Save the mesh to a file
        gmsh.write(self.mesh_path)

        # Visualize the mesh
        if vis:
            gmsh.fltk.run()

        # Finalize Gmsh
        gmsh.finalize()

 

    def as_graph(self):
        if not os.path.exists(self.ana_path):
            self.ana_sol()

        dtype               = torch.float64
        mesh                = meshio.read(self.mesh_path)

        # inner dirichlet condition
        r                   = np.sqrt((mesh.points**2).sum(-1))
        atol                = 1e-10
        is_inner_boundary   = np.isclose(r, self.mesh_parameters["a"],atol=atol)
        is_outer_boundary   = np.isclose(r, self.mesh_parameters["b"],atol=atol)
        dirichlet_mask      = torch.from_numpy(is_inner_boundary).type(torch.bool)
        dirichlet_value     = torch.zeros(mesh.points.shape, dtype=dtype)
        dirichlet_value[dirichlet_mask] = 0.0
        
        # outer pressure
        n                   = mesh.points / r[:, None]
        f                   = self.pde_parameters["p"] * (- n) * (4 * np.pi * self.mesh_parameters["b"]) / is_outer_boundary.sum()
        source_mask         = torch.from_numpy(is_outer_boundary).type(torch.bool)
        source_value        = torch.zeros(mesh.points.shape, dtype=dtype)
        source_value[source_mask] = torch.from_numpy(f).type(dtype)[source_mask]

        # connectivity
        tetras = mesh.get_cells_type("tetra")
        tetras = torch.tensor(tetras, dtype=torch.long)
        edges  = torch.vmap(lambda x: torch.stack(torch.meshgrid(x, x), -1))(tetras) # (n_tetras, 4, 4, 2)
        edges  = edges.view(-1, 2) # (n_tetras * 4 * 4, 2)
        adj    = torch.sparse_coo_tensor(
            edges.T, 
            torch.ones(edges.shape[0], dtype=torch.float), 
            size=(mesh.points.shape[0], mesh.points.shape[0])
        ).coalesce()
        edges  = adj.indices().T

        # label data 
        data = np.load(self.ana_path)
        displacement = data["displacement"]
        strain       = data["strain"]
        stress       = data["stress"]
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
            g_E         =   torch.tensor(self.pde_parameters["E"], dtype=dtype),
            g_nu        =   torch.tensor(self.pde_parameters["nu"], dtype=dtype),
            edge_index  =   edges.T,
        )
        return graph

    def manual_solve(self):
        mesh           = meshio.read(self.mesh_path)


    def skfem_solve(self):
        # Read the mesh
        atol = self.mesh_parameters['d']**3
        mesh           = Mesh.load(self.mesh_path
                                   ).with_boundaries({
                                       "inner boundary": lambda x: np.isclose(np.sqrt((x**2).sum(0)), self.mesh_parameters["a"], atol=atol),
                                       "outer boundary": lambda x: np.isclose(np.sqrt((x**2).sum(0)), self.mesh_parameters["b"], atol=atol),
                                   })
        vector         = ElementVector(ElementTetP1())
        vector_basis   = Basis(mesh, vector)
        

        # Material parameters: Young's modulus and Poisson's ratio
        E, nu = self.pde_parameters["E"], self.pde_parameters["nu"]
    
        u               = vector_basis.zeros()
        f               = vector_basis.zeros()
        inner_dofs      = vector_basis.get_dofs(facets=mesh.boundaries["inner boundary"])
        outer_dofs      = vector_basis.get_dofs(facets=mesh.boundaries["outer boundary"])

        u[inner_dofs.nodal["u^1"]]    = 0.0
        u[inner_dofs.nodal["u^2"]]    = 0.0
        u[inner_dofs.nodal["u^3"]]    = 0.0
        # breakpoint()
        outer_pos       = mesh.p.T[outer_dofs.nodal_ix]
        #  breakpoint()
        outer_rad       = np.sqrt((outer_pos**2).sum(1))
        outer_nor       = outer_pos / outer_rad[:, None]
        outer_force     = self.pde_parameters["p"] * (- outer_nor) * (4 * np.pi * self.mesh_parameters["b"] ** 2) / len(outer_dofs.nodal_ix)
        f[outer_dofs.nodal["u^1"]]    = outer_force[:,0]
        f[outer_dofs.nodal["u^2"]]    = outer_force[:,1]
        f[outer_dofs.nodal["u^3"]]    = outer_force[:,2]
      
        K = asm(linear_elasticity(*lame_parameters(E, nu)), vector_basis)

        # Apply the boundary conditions. Here, we consider only internal pressure and free outer boundary.
        u = solve(*condense(K, f, D=inner_dofs, x=u))
        breakpoint()
        matrix_basis    = vector_basis.with_element(ElementVector(vector))
        C               = linear_stress(*lame_parameters(E, nu))
        u_ele           = vector_basis.interpolate(u)
        strain_ele      = sym_grad(u_ele)
        stress_ele      = C(strain_ele)

        strain          = matrix_basis.project(strain_ele)
        stress          = matrix_basis.project(stress_ele)
        u               = u[vector_basis.nodal_dofs].transpose(1, 0)
        strain          = strain[matrix_basis.nodal_dofs].reshape(3, 3, -1).transpose(2, 0, 1)
        stress          = stress[matrix_basis.nodal_dofs].reshape(3, 3, -1).transpose(2, 0, 1)
        # breakpoint()
        u               = vector_carteisan_to_spherical(u, mesh.p.T)
        # breakpoint()

        vm_stress       = vm_stress_cartesian(stress)

        np.savez(self.fem_path,
            u = u[:, 0],
            displacement = u,
            stress = stress, 
            strain = strain,
            vm_stress=vm_stress)
        
    def fem_sol(self):
        self.skfem_solve()
        return self
  
    def ana_sol(self):
        """Analytical solution for the radial stress in a spherical shell under internal pressure.
        suppose the pressure is on the outer boundary
        https://www.brown.edu/Departments/Engineering/Courses/En1750/Notes/Elastic_Solutions/Elastic_Solutions.htm
        $$
        u = A R + \frac{B}{R^2}\\
        \epsilon_{rr} = A - 2 \frac{B}{R^3}\\
        \epsilon_{\theta\theta} = A + 2 \frac{B}{R^3}\\
        \sigma_{rr} = \frac{E}{(1+\nu)(1-2\nu)}\left\{(1-\nu)\epsilon_{rr} + 2\nu \epsilon_{\theta\theta}\right\}\\
        \sigma_{\theta\theta} =  \frac{E}{(1+\nu)(1-2\nu)}\left\{\epsilon_{\theta\theta} +\nu \epsilon_{rr}\right\}
        $$
        given $u(a) = 0$,  $\sigma_{rr}(b) = -p$, we have
        $$
        A = \frac{-(1-2\nu)(1+\nu)p}{E((1+\nu)b^3 + (2-4\nu)a^3)}\\
        B = \frac{(1-2\nu)(1+\nu)pa^3}{(1+\nu)b^3 + (2-4\nu)a^3}
        $$
        """
        mesh = meshio.read(self.mesh_path)
        r = np.sqrt((mesh.points**2).sum(-1))
        p, a, b = self.pde_parameters["p"], self.mesh_parameters["a"], self.mesh_parameters["b"]
        E, nu = self.pde_parameters["E"], self.pde_parameters["nu"]

        A = -(1-2*nu)*(1+nu)*p / (E * ((1+nu)*b**3 + (2-4*nu)*a**3))
        B = (1-2*nu)*(1+nu)*p*a**3 / ((1+nu)*b**3 + (2-4*nu)*a**3)

        e = mesh.points / r[:, None]

        u = A * r + B / r**2 
        
        strain_rr = A - 2 * B / r**3
        strain_tt = A + 2 * B / r**3

        stress_rr = E / ((1+nu)*(1-2*nu)) * ((1-nu)*strain_rr + 2*nu*strain_tt)
        stress_tt = E / ((1+nu)*(1-2*nu)) * (strain_tt + nu*strain_rr)

        # spherical coordinates
        theta, phi = np.arctan2(mesh.points[:, 1], mesh.points[:, 0]), np.arccos(mesh.points[:, 2] / r)

        # transform displacement(u), strain, stress from spherical coordinates to cartesian coordinates
        displacement_sphere = np.stack([u, np.zeros_like(u), np.zeros_like(u)], -1)
        displacement_cartesian = vector_spherical_to_cartesian(displacement_sphere, theta, phi)
        strain_cartesian = matrix_spherical_to_cartesian(strain_rr, strain_tt, theta, phi)
        stress_cartesian = matrix_spherical_to_cartesian(stress_rr, stress_tt, theta, phi)

        # calculate von Mises stress
        vm_stress_1 = vm_stress_cartesian(stress_cartesian)
        vm_stress_2 = vm_stress_spherical(stress_rr, stress_tt)

        np.savez(self.ana_path, 
            u = np.sqrt((displacement_cartesian**2).sum(-1)),
            displacement=displacement_cartesian, 
            strain=strain_cartesian, 
            stress=stress_cartesian,
            vm_stress=vm_stress_1)

        return self
    
    def vis(self, obj="fem", tgt="vm-stress"):
        assert obj in ["fem", "ana", "mesh"]
        assert tgt in ["stress-x", "stress-y", "stress-z",
                       "strain-x", "strain-y", "strain-z",
                       "displacement-x", "displacement-y", "displacement-z",
                       "vm-stress", "u" ]
        if obj == "mesh":
            gmsh.initialize()
            gmsh.open(self.mesh_path)
            gmsh.fltk.run()
            gmsh.finalize()
        elif obj in ["fem", "ana"]:
            if obj == "fem":
                data = np.load(self.fem_path) 
            elif obj == "ana":
                data = np.load(self.ana_path)

            if tgt == "vm-stress":
                target = data['vm_stress']
            elif tgt == "u":
                target = data["u"]
            else:
                idx = {"x":0, "y":1, "z":2}[tgt.split("-")[1]]
                target = data[tgt.split("-")[0]]
                if tgt.startswith("displacement"):
                    target = target[:, idx]
                else:
                    target = target[:, idx, idx]
          
            mesh = meshio.read(self.mesh_path)
            mesh = pv.from_meshio(mesh)
            mesh[tgt] = target
            p = pv.Plotter()
            p.add_mesh_clip_plane(mesh, assign_to_axis='z', scalars=tgt, cmap="jet", show_edges=False)
            p.add_axes()
            p.show()
        return self

       

if __name__ == '__main__':
    # SphericalShell().ana_sol().vis("ana", "vm-stress")
    # SphericalShell().ana_sol().vis("ana", "u")
    # SphericalShell().fem_sol().vis("fem", "vm-stress")
    # SphericalShell().fem_sol().vis("fem", "u")
    SphericalShell().vis("mesh")
    # phericalShell().as_graph()