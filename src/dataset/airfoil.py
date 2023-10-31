
import os
import numpy as np
import gmsh
import meshio
import matplotlib.pyplot as plt

def naca2412(x):
    # Mean camber line
    m = 0.02
    p = 0.4
    c = 1.0  # chord length
    
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)
    yc[x < p*c] = m / (p**2) * (2*p*x[ x < p*c] - x[ x < p*c]**2)
    yc[x >= p*c] = m / ((1-p)**2) * ((1 - 2*p) + 2*p*x[x >= p*c] - x[x >= p*c]**2)
    dyc_dx[x < p*c] = 2*m / (p**2) * (p - x[x < p*c])
    dyc_dx[x >= p*c] = 2*m / ((1-p)**2) * (p - x[x >= p*c])
    
    # Thickness distribution
    t = 0.12
    yt = 5 * t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1036*x**4)
    
    # Upper and lower surface
    xu = x - yt * np.sin(np.arctan(dyc_dx))
    xl = x + yt * np.sin(np.arctan(dyc_dx))
    yu = yc + yt * np.cos(np.arctan(dyc_dx))
    yl = yc - yt * np.cos(np.arctan(dyc_dx))
    
    return xu, yu, xl, yl


class AirFoil:
    def __init__(self, n_grid:int=100, name:str="naca2412", cache_dir:str=".cache",):
        """
            Parameters:
            -----------
            type: str
            tmp_dir: str
        """

        x = np.linspace(0, 1, n_grid)

        xu, yu, xl, yl = {
            "naca2412": naca2412,
        }[name](x)

        self.xu, self.yu, self.xl, self.yl = xu, yu, xl, yl

        self.cache_dir = cache_dir  

    def generate_mesh(self):

        # Initialize Gmsh
        gmsh.initialize()

        # Create a new model
        gmsh.model.add('airfoil')

        # Define the airfoil in Gmsh
        upper_points = [gmsh.model.geo.addPoint(x, y, 0) for x, y in zip(xu, yu)]
        lower_points = [gmsh.model.geo.addPoint(x, y, 0) for x, y in zip(xl, yl)]

        upper_lines = [gmsh.model.geo.addLine(upper_points[i], upper_points[i+1]) for i in range(len(upper_points)-1)]
        lower_lines = [gmsh.model.geo.addLine(lower_points[i], lower_points[i+1]) for i in range(len(lower_points)-1)]

        # Close the trailing edge
        te_line_upper = gmsh.model.geo.addLine(upper_points[-1], upper_points[0])
        te_line_lower = gmsh.model.geo.addLine(lower_points[0], lower_points[-1])

        # Create a surface from the lines
        gmsh.model.geo.addCurveLoop(upper_lines + [te_line_upper] + lower_lines + [-te_line_lower], 1)
        gmsh.model.geo.addPlaneSurface([1], 1)

        # Synchronize and generate 2D mesh
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)

        # Save the mesh
        os.makedirs(cache_dir, exist_ok=True)
        gmsh.write(os.path.join(cache_dir,'airfoil.msh'))

        # Finalize Gmsh
        gmsh.finalize()

        # Read the mesh using meshio
        mesh = meshio.read('naca2412.msh')

        self.points      = mesh.points
        self.elements    = mesh.cells['triangle']

    def as_graph(self):
        pass

    def plot(self):
        # Plot the mesh
        plt.figure(figsize=(10, 5))
        plt.triplot(self.points[:, 0], self.points[:, 1], self.elements)
        plt.axis('equal')
        plt.title('NACA 2412 Mesh')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.show()



if __name__ == '__main__':
    AirFoil().plot()