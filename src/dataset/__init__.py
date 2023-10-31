from .spherical_shell import SphericalShell
from .truss import Truss
from .utils import mesh_to_pyg_graph
import importlib
truss = importlib.import_module(".truss", package=__package__)
triangle = importlib.import_module(".triangle", package=__package__)