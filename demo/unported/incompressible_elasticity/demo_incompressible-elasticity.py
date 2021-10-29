# Copyright 2019-2020, Jack S. Hale, Raphaël Bulle
# SPDX-License-Identifier: LGPL-3.0-or-later

# Mixed robust nearly-incompressible linear elasticity error estimator from Khan,
# Powell and Silvester (2019) https://doi.org/10.1002/nme.6040. We solve
# the problem from Carstensen and Gedicke https://doi.org/10.1016/j.cma.2015.10.001.
#
# We implement the Poisson problem local estimator detailed in Section 3.5.
# Somewhat remarkably, despite the complexity of the mixed formulation, an
# highly efficient implicit estimator is derived involving the solution of two
# Poisson problems on a special local finite element space. An additional
# explicit estimator computing related to the pressure can be computed. No
# local inf-sup condition must be satisfied by the estimation problem.

# Differences with the presentation in Khan et al.:
# We do not split Equation 50a into two Poisson sub-problems. Instead, we solve
# it as a single monolithic system. In practice we found the performance
# benefits of the splitting negligible, especially given the additional
# complexity.  Note also that the original paper uses quadrilateral finite
# elements. We use the same estimation strategy on triangular finite elements
# without any issues (see Page 28).

import pandas as pd
import numpy as np
import scipy as sp

from dolfin import *

from fenics_error_estimation.interpolate import create_interpolation
import fenics_error_estimation

parameters["ghost_mode"] = "shared_facet"
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True

mu = 100.  # First Lamé coefficien
nu = .499    # Poisson ratio
lmbda = 2. * mu * nu / (1. - 2. * nu)  # Second Lamé coefficient


def main():
    K = 10
    mesh = UnitSquareMesh(K, K)

    X_el = VectorElement('CG', triangle, 2)
    M_el = FiniteElement('CG', triangle, 1)

    V_el = MixedElement([X_el, M_el])

    results = []
    # Standard solve, estimate, mark, refine loop.
    for i in range(0, 10):
        V = FunctionSpace(mesh, V_el)

        result = {}
        w_h, exact_err = solve(V)
        print('V dim = {}'.format(V.dim()))
        w_h, err = solve(V)
        print('Exact error = {}'.format(err))
        result['exact_error'] = err

        print('Estimating...')
        eta_h = estimate(w_h)
        result['error_bw'] = np.sqrt(eta_h.vector().sum())
        print('BW = {}'.format(np.sqrt(eta_h.vector().sum())))
        result['hmin'] = mesh.hmin()
        result['hmax'] = mesh.hmax()
        result['num_dofs'] = V.dim()

        print('Estimating (res)...')
        eta_res = residual_estimate(w_h)
        result['error_res'] = np.sqrt(eta_res.vector().sum())
        print('Res = {}'.format(np.sqrt(eta_res.vector().sum())))

        print('Marking...')
        markers = fenics_error_estimation.dorfler(eta_h, 0.5)
        print('Refining...')
        mesh = refine(mesh, markers, redistribute=True)

        with XDMFFile('output/mesh_{}.xdmf'.format(str(i).zfill(4))) as f:
            f.write(mesh)

        with XDMFFile('output/disp_{}.xdmf'.format(str(i).zfill(4))) as f:
            f.write_checkpoint(w_h.sub(0), 'u_{}'.format(str(i).zfill(4)))

        with XDMFFile('output/pres_{}.xdmf'.format(str(i).zfill(4))) as f:
            f.write_checkpoint(w_h.sub(1), 'p_{}'.format(str(i).zfill(4)))

        with XDMFFile('output/eta_{}.xdmf'.format(str(i).zfill(4))) as f:
            f.write_checkpoint(eta_h, 'eta_{}'.format(str(i).zfill(4)))

        results.append(result)

    if (MPI.comm_world.rank == 0):
        df = pd.DataFrame(results)
        df.to_pickle('output/results.pkl')
        print(df)


def solve(V):
    """Solve nearly-incompressible elasticity using a P2-P1 mixed finite
    element method. This is completely standard."""
    mesh = V.mesh()

    f = Expression(('-2.*mu*pow(pi,3)*cos(pi*x[1])*sin(pi*x[1])*(2.*cos(2.*pi*x[0]) - 1.)', '2.*mu*pow(pi,3)*cos(pi*x[0])*sin(pi*x[0])*(2.*cos(2.*pi*x[1]) -1.)'), mu=mu, degree=4)  # noqa: E501

    w_exact = Expression(('pi*cos(pi*x[1])*pow(sin(pi*x[0]), 2)*sin(pi*x[1])',
                          '-pi*cos(pi*x[0])*pow(sin(pi*x[1]), 2)*sin(pi*x[0])', '0'), degree=4)

    (u, p) = TrialFunctions(V)
    (v, q) = TestFunctions(V)

    a = 2. * mu * inner(sym(grad(u)), sym(grad(v))) * dx
    b_1 = - inner(p, div(v)) * dx
    b_2 = - inner(q, div(u)) * dx
    c = (1. / lmbda) * inner(p, q) * dx

    B = a + b_1 + b_2 - c
    L = inner(f, v) * dx

    bcs = DirichletBC(V.sub(0), Constant((0., 0.)), 'on_boundary')

    A, b = assemble_system(B, L, bcs=bcs)

    w_h = Function(V)

    PETScOptions.set('pc_type', 'lu')
    PETScOptions.set('pc_factor_mat_solver_type', 'mumps')
    solver = PETScKrylovSolver()
    solver.set_from_options()
    solver.solve(A, w_h.vector(), b)

    u_h = w_h.sub(0)
    p_h = w_h.sub(1)

    with XDMFFile('output/displacement.xdmf') as f:
        f.write_checkpoint(u_h, 'u_h')
    with XDMFFile('output/pressure.xdmf') as f:
        f.write_checkpoint(p_h, 'p_h')

    X_el_f = VectorElement('CG', triangle, 3)
    M_el_f = FiniteElement('CG', triangle, 2)

    V_el_f = MixedElement([X_el_f, M_el_f])

    V_f = FunctionSpace(mesh, V_el_f)

    w_h_f = project(w_h, V_f)
    w_f = project(w_exact, V_f)

    w_diff = Function(V_f)
    w_diff.vector()[:] = w_h_f.vector()[:] - w_f.vector()[:]
    local_exact_err_2 = energy_norm(w_diff)
    exact_err = sqrt(sum(local_exact_err_2[:]))
    return w_h, exact_err


def estimate(w_h):
    """Estimator described in Section 3.5 of Khan et al."""
    mesh = w_h.function_space().mesh()

    u_h = w_h.sub(0)
    p_h = w_h.sub(1)

    # The first estimation problem consists of two Poisson sub-problems. We
    # solve both simultaneously in block diagonal system.
    X_element_f = VectorElement('DG', triangle, 3)

    # We construct the interpolation operator between the fine and coarse space
    # by placing two scalar interpolation operators into a block diagonal
    # system.
    S_element_f = FiniteElement('DG', triangle, 3)
    S_element_g = FiniteElement('DG', triangle, 1)

    N_S = create_interpolation(
        S_element_f, S_element_g)
    N_X = sp.linalg.block_diag(N_S, N_S)

    X_f = FunctionSpace(mesh, X_element_f)

    f = Expression(('-2.*mu*pow(pi,3)*cos(pi*x[1])*sin(pi*x[1])*(2.*cos(2.*pi*x[0]) - 1.)', '2.*mu*pow(pi,3)*cos(pi*x[0])*sin(pi*x[0])*(2.*cos(2.*pi*x[1]) -1.)'), mu=mu, degree=4)  # noqa: E501

    e_X = TrialFunction(X_f)
    v_X = TestFunction(X_f)

    bcs = DirichletBC(X_f, Constant((0., 0.)), 'on_boundary', 'geometric')

    # Cell and edge residual equations from Khan et al.
    n = FacetNormal(mesh)
    R_K = f + div(2. * mu * sym(grad(u_h))) - grad(p_h)
    Id = Identity(2)

    R_E = (1. / 2.) * jump(p_h * Id - 2. * mu * sym(grad(u_h)), -n)
    rho_d = 1. / (lmbda**(-1) + (2. * mu)**(-1))

    # Local estimation problem
    a_X_e = 2. * mu * inner(grad(e_X), grad(v_X)) * dx
    L_X_e = inner(R_K, v_X) * dx - inner(R_E, avg(v_X)) * dS

    # Solve the two Poisson estimation problems on the special space.
    e_h = fenics_error_estimation.estimate(a_X_e, L_X_e, N_X, bcs)

    # The second estimation problem. Local projection of cell residual to
    # special space.
    M_element_f = FiniteElement('DG', triangle, 2)
    M_element_g = FiniteElement('DG', triangle, 1)

    N_M = create_interpolation(
        M_element_f, M_element_g)

    M_f = FunctionSpace(mesh, M_element_f)

    p_M_f = TrialFunction(M_f)
    q_M_f = TestFunction(M_f)

    # From Khan et al.
    a_M_e = rho_d**(-1) * inner(p_M_f, q_M_f) * dx

    r_K = div(u_h) + (1. / lmbda) * p_h
    L_M_e = inner(r_K, q_M_f) * dx

    eps_h = fenics_error_estimation.estimate(a_M_e, L_M_e, N_M)

    # Compute the Khan et al. local estimator.
    V_e = FunctionSpace(mesh, 'DG', 0)
    v = TestFunction(V_e)

    eta_h = Function(V_e)
    eta = assemble(2. * mu * inner(inner(grad(e_h), grad(e_h)), v) * dx
                   + rho_d**(- 1) * inner(inner(eps_h, eps_h), v) * dx)
    eta_h.vector()[:] = eta

    return eta_h


def residual_estimate(w_h):
    """Residual estimator described in Section 3.1 of Khan et al."""
    mesh = w_h.function_space().mesh()

    f = Expression(('-2.*mu*pow(pi,3)*cos(pi*x[1])*sin(pi*x[1])*(2.*cos(2.*pi*x[0]) - 1.)', '2.*mu*pow(pi,3)*cos(pi*x[0])*sin(pi*x[0])*(2.*cos(2.*pi*x[1]) -1.)'), mu=mu, degree=4)  # noqa: E501

    u_h = w_h.sub(0)
    p_h = w_h.sub(1)

    n = FacetNormal(mesh)
    R_K = f + div(2. * mu * sym(grad(u_h))) - grad(p_h)
    r_K = div(u_h) + (1. / lmbda) * p_h
    Id = Identity(2)
    R_E = (1. / 2.) * jump(p_h * Id - 2. * mu * sym(grad(u_h)), -n)

    V = FunctionSpace(mesh, "DG", 0)
    v = TestFunction(V)
    h = CellDiameter(mesh)

    eta_h = Function(V)

    rho_K = (h * (2. * mu)**(-0.5)) / 2.
    rho_E = (avg(h) * (2. * mu)**(-1)) / 2.
    rho_d = 1. / (lmbda**(-1) + (2. * mu)**(-1))

    eta = assemble(rho_K**2 * R_K**2 * v * dx + rho_d * r_K**2 * v * dx + rho_E * R_E**2 * avg(v) * dS)
    eta_h.vector()[:] = eta

    return eta_h


def energy_norm(x):
    mesh = x.function_space().mesh()
    u = x.sub(0)
    p = x.sub(1)

    W = FunctionSpace(mesh, 'DG', 0)
    v = TestFunction(W)

    form = Constant(2. * mu) * inner(inner(grad(u), grad(u)), v) * dx + Constant(1. / (2. * mu)) * \
        inner(inner(p, p), v) * dx + Constant(1. / lmbda) * inner(inner(p, p), v) * dx
    norm_2 = assemble(form)
    return norm_2


if __name__ == "__main__":
    main()
