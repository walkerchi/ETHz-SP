import dataset
from linear_elasticity import TriangleSolver,TrussSolver


def truss():
    mesh = dataset.truss.bridge.baltimore(n_grid=16, support=3)
    truss_sol = TrussSolver(mesh)
    u_scipy = truss_sol.scipy_solve()
    #  u_skfem = truss_sol.skfem_solve()
    stress_scipy = truss_sol.compute_stress(u_scipy)
    # stress_skfem = truss_sol.compute_stress(u_skfem)
    truss_sol.plot(
        stress_scipy = stress_scipy,
    #    stress_skfem = stress_skfem,
    )

def triangle():
    mesh = dataset.triangle.hollow_rectangle(d=0.05,E=10)
    triangle_sol = TriangleSolver(mesh)
    u_scipy = triangle_sol.scipy_solve()
    #u_skfem = triangle_sol.skfem_solve()
    r_scipy = triangle_sol.compute_residual(u_scipy)
    # r_skfem = triangle_sol.compute_residual(u_skfem)
    vm_stress_scipy = triangle_sol.compute_stress(u_scipy,return_vm_stress=True)[-1]
    #vm_stress_skfem = triangle_sol.compute_vm_stress(u_skfem)
    print(f"residual: {r_scipy}")
    triangle_sol.plot(
        vm_stress_scipy = vm_stress_scipy,
        u_x_scipy = u_scipy[:,0],
        u_y_scipy = u_scipy[:,1],
    #    vm_stress_skfem = vm_stress_skfem,
    )

if __name__ == '__main__':
    truss()