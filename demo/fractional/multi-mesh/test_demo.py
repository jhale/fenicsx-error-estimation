from dolfinx.mesh       import create_unit_square
from dolfinx.fem        import Function, form, assemble_scalar
from mpi4py             import MPI
from demo               import solve_parametric, compute_est
from ufl                import inner, grad, Measure

import numpy as np
import ufl

def test_solve_parametric():
    # Analytical solution source data
    def source_data(x):
        return 8.0 * np.pi**2 * np.sin(2.0 * np.pi * x[0]) * np.sin(2.0 * np.pi * x[1])

    def exact_solution(x):
        return np.sin(2.0 * np.pi * x[0]) * np.sin(2.0 * np.pi * x[1])
    
    mesh = create_unit_square(MPI.COMM_WORLD, 80, 80)
    u = solve_parametric(2, source_data, mesh, 1., 0.)

    V = u.function_space
    dx = ufl.Measure("dx", domain=mesh)

    exact_solution_V = Function(V)
    exact_solution_V.interpolate(exact_solution)

    difference = u - exact_solution_V

    L2_norm_sq = form(inner(difference, difference) * dx(degree=2))

    L2_error = np.sqrt(assemble_scalar(L2_norm_sq))

    assert np.isclose(L2_error, 0.0, atol=1.e-6)

def test_compute_est():
    rational_constant = 1.
    ls_eta_param_vects = [np.zeros(10) for i in range(3)]
    for i in range(10):
        ls_eta_param_vect = np.zeros(10)
        ls_eta_param_vect[i] = 1.
        ls_eta_param_vects[2] = ls_eta_param_vect

        assert np.isclose(compute_est(ls_eta_param_vects, rational_constant), 1.)


if __name__ == "__main__":
    test_solve_parametric()