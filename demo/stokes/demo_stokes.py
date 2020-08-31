# Copyright 2019-2020, Jack S. Hale, Raphaël Bulle
# SPDX-License-Identifier: LGPL-3.0-or-later

# Mixed robust Stokes error estimator from Liao and Silvester (2012)
# https://doi.org/10.1016/j.apnum.2010.05.003.

import pandas as pd
import numpy as np
import scipy as sp

from dolfin import *

import fenics_error_estimation
from fenics_error_estimation.interpolate import create_interpolation

parameters["ghost_mode"] = "shared_facet"
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True


def main():
    K = 10
    mesh = UnitSquareMesh(K, K, diagonal='crossed')
    mesh.coordinates()[:] -= 0.5
    mesh.coordinates()[:] *= 2.

    X_el = VectorElement('CG', triangle, 2)
    M_el = FiniteElement('CG', triangle, 1)
    L_el = FiniteElement('R', triangle, 0)

    V_el = MixedElement([X_el, M_el, L_el])

    results = []
    for i in range(0, 10):
        V = FunctionSpace(mesh, V_el)

        result = {}
        result['num_cells'] = V.mesh().num_cells()

        w_h, err = solve(V)

        print('Exact error = {}'.format(err))
        result['exact_error'] = err

        print('Estimating (Liao and Silvester)...')
        eta = estimate(w_h)
        result['error_bw'] = np.sqrt(eta.vector().sum())
        print('BW = {}'.format(np.sqrt(eta.vector().sum())))
        result['hmin'] = mesh.hmin()
        result['hmax'] = mesh.hmax()
        result['num_dofs'] = V.dim()

        print('Estimating (residual)...')
        eta_res = residual_estimate(w_h)
        result['error_res'] = np.sqrt(eta_res.vector().sum())
        print('Res = {}'.format(np.sqrt(eta_res.vector().sum())))

        print('Marking...')
        markers = fenics_error_estimation.dorfler(eta, 0.5)
        print('Refining...')
        mesh = refine(mesh, markers, redistribute=True)

        with XDMFFile('output/mesh_{}.xdmf'.format(str(i).zfill(4))) as f:
            f.write(mesh)

        with XDMFFile('output/velo_{}.xdmf'.format(str(i).zfill(4))) as f:
            f.write_checkpoint(w_h.sub(0), 'u_{}'.format(str(i).zfill(4)))

        with XDMFFile('output/pres_{}.xdmf'.format(str(i).zfill(4))) as f:
            f.write_checkpoint(w_h.sub(1), 'p_{}'.format(str(i).zfill(4)))

        with XDMFFile('output/eta_{}.xdmf'.format(str(i).zfill(4))) as f:
            f.write_checkpoint(eta, 'eta_{}'.format(str(i).zfill(4)))

        results.append(result)

    if (MPI.comm_world.rank == 0):
        df = pd.DataFrame(results)
        df.to_pickle('output/results.pkl')
        print(df)


def solve(V):
    """Solve Stokes problem for viscous incompressible flow using a P2-P1 mixed finite
    element method. This is completely standard."""
    mesh = V.mesh()

    f = Constant((0., 0.))

    u_exact = Expression(
        ('20.*x[0]*pow(x[1], 3)', '5.*pow(x[0], 4)-5.*pow(x[1], 4)'), degree=4)
    p_exact = Expression('60.*pow(x[0], 2)*x[1]- 20.*pow(x[1], 3)', degree=4)
    (u, p, u_l) = TrialFunctions(V)
    (v, q, v_l) = TestFunctions(V)

    a = inner(grad(u), grad(v)) * dx
    b = - inner(p, div(v)) * dx
    c = - inner(div(u), q) * dx

    d = inner(u_l, q) * dx + inner(v_l, p) * dx
    B = a + b + c + d

    L = inner(f, v) * dx

    bcs = DirichletBC(V.sub(0), u_exact, 'on_boundary')

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

    X_f = FunctionSpace(mesh, X_el_f)
    M_f = FunctionSpace(mesh, M_el_f)

    u_h_f = project(u_h, X_f)
    p_h_f = project(p_h, M_f)

    u_f = project(u_exact, X_f)
    p_f = project(p_exact, M_f)

    u_diff = Function(X_f)
    u_diff.vector()[:] = u_f.vector()[:] - u_h_f.vector()[:]

    p_diff = Function(M_f)
    p_diff.vector()[:] = p_f.vector()[:] - p_h_f.vector()[:]

    local_exact_err_2 = energy_norm(u_diff, p_diff)
    with XDMFFile('output/exact_disp.xdmf') as f:
        f.write_checkpoint(u_h_f, 'exact_disp')

    with XDMFFile('output/exact_p.xdmf') as f:
        f.write_checkpoint(p_h_f, 'exact_p')

    local_exact_err_2 = energy_norm(u_diff, p_diff)
    exact_err = sqrt(sum(local_exact_err_2[:]))

    return w_h, exact_err


def estimate(w_h):
    """Estimator described in Section 3.3 of Liao and Silvester"""
    mesh = w_h.function_space().mesh()

    u_h = w_h.sub(0)
    p_h = w_h.sub(1)

    X_element_f = VectorElement('DG', triangle, 3)
    S_element_f = FiniteElement('DG', triangle, 3)
    S_element_g = FiniteElement('DG', triangle, 1)

    N_S = create_interpolation(S_element_f, S_element_g)
    N_X = sp.linalg.block_diag(N_S, N_S)

    X_f = FunctionSpace(mesh, X_element_f)

    f = Constant((0., 0.))

    e_X = TrialFunction(X_f)
    v_X = TestFunction(X_f)

    bcs = DirichletBC(X_f, Constant((0., 0.)), 'on_boundary', 'geometric')

    n = FacetNormal(mesh)
    R_T = f + div(grad(u_h)) - grad(p_h)
    Id = Identity(2)

    R_E = (1. / 2.) * jump(-p_h * Id + grad(u_h), -n)

    a_X_e = inner(grad(e_X), grad(v_X)) * dx
    L_X_e = inner(R_T, v_X) * dx - inner(R_E, 2. * avg(v_X)) * dS

    e_h = fenics_error_estimation.estimate(a_X_e, L_X_e, N_X, bcs)

    M_element_f = FiniteElement('DG', triangle, 2)
    M_element_g = FiniteElement('DG', triangle, 1)

    N_M = create_interpolation(
        M_element_f, M_element_g)

    M_f = FunctionSpace(mesh, M_element_f)

    p_M_f = TrialFunction(M_f)
    q_M_f = TestFunction(M_f)

    a_M_e = inner(p_M_f, q_M_f) * dx
    r_T = div(u_h)
    L_M_e = inner(r_T, q_M_f) * dx

    eps_h = fenics_error_estimation.estimate(a_M_e, L_M_e, N_M)

    V_e = FunctionSpace(mesh, 'DG', 0)
    v = TestFunction(V_e)

    eta_h = Function(V_e)
    eta = assemble(inner(inner(grad(e_h), grad(e_h)), v)
                   * dx + inner(inner(eps_h, eps_h), v) * dx)
    eta_h.vector()[:] = eta

    return eta_h


def residual_estimate(w_h):
    """Residual estimator described in Section 3.1 of Liao and Silvester"""
    mesh = w_h.function_space().mesh()

    f = Constant((0., 0.))

    u_h = w_h.sub(0)
    p_h = w_h.sub(1)

    n = FacetNormal(mesh)
    R_T = f + div(grad(u_h)) - grad(p_h)
    r_T = div(u_h)
    Id = Identity(2)
    R_E = (1. / 2.) * jump(-p_h * Id + grad(u_h), -n)

    V = FunctionSpace(mesh, "DG", 0)
    v = TestFunction(V)
    h = CellDiameter(mesh)

    eta_h = Function(V)

    eta = assemble(h**2 * R_T**2 * v * dx + r_T ** 2 * v * dx + avg(h) * R_E**2 * avg(v) * dS)
    eta_h.vector()[:] = eta

    return eta_h


def energy_norm(u, p):
    u_mesh = u.function_space().mesh()
    p_mesh = p.function_space().mesh()

    assert u_mesh is p_mesh

    mesh = u_mesh

    W = FunctionSpace(mesh, 'DG', 0)
    v = TestFunction(W)

    form = inner(inner(grad(u), grad(u)), v) * dx + inner(inner(p, p), v) * dx
    norm_2 = assemble(form)
    return norm_2


if __name__ == "__main__":
    main()
