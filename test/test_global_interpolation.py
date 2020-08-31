# SPDX-License-Identifier: LGPL-3.0+
# Jack S. Hale

import numpy as np
from dolfin import *


def _global_pure_dirichlet(request, mesh, k, u_and_f_dirichlet):
    k_g = k

    V = FunctionSpace(mesh, "CG", k)
    V_f = FunctionSpace(mesh, "DG", k + 1)
    V_g = FunctionSpace(mesh, "DG", k_g)

    u_exact, f = u_and_f_dirichlet

    def boundary(x, on_boundary):
        return on_boundary

    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx

    bcs = [DirichletBC(V, Constant(0.0), boundary)]

    u_h = Function(V)
    A, b = assemble_system(a, L, bcs=bcs)

    solver = PETScKrylovSolver("cg", "hypre_amg")
    solver.solve(A, u_h.vector(), b)

    e = TrialFunction(V_f)
    v = TestFunction(V_f)

    if mesh.geometry().dim() == 1:
        bcs = [DirichletBC(V_f, Constant(0.0), boundary, "pointwise")]
    else:
        bcs = [DirichletBC(V_f, Constant(0.0), boundary, "geometric")]

    n = FacetNormal(mesh)
    a_e = inner(grad(e), grad(v)) * dx
    L_e = inner(f + div(grad(u_h)), v) * dx + \
        inner(jump(grad(u_h), -n), avg(v)) * dS

    Id = np.zeros((V_f.dim(), V_f.dim()))
    w = Function(V_f)
    for i in range(V_f.dim()):
        w.vector()[i] = 1.0
        w_V_g = interpolate(w, V_g)
        w_V_f = interpolate(w_V_g, V_f)
        Id[:, i] = w_V_f.vector()
        w.vector()[i] = 0.0

    M = np.eye(V_f.dim()) - Id
    eigs, P = np.linalg.eig(M)
    mask = np.abs(eigs) > 0.5
    try:
        N = np.array(P[:, mask], dtype=np.float64)
    except ComplexWarning:
        pass

    A_e, b_e = assemble_system(a_e, L_e, bcs=bcs)
    A_e_0 = N.T @ A_e.array() @ N
    b_e_0 = N.T @ b_e

    e_0 = np.linalg.solve(A_e_0, b_e_0)
    e = N @ e_0
    e_V_f = Function(V_f)
    e_V_f.vector()[:] = e

    error_bw = np.sqrt(assemble(inner(grad(e_V_f), grad(e_V_f)) * dx))
    error_exact = errornorm(u_exact, u_h, "H10")

    result = {}
    result["name"] = request.node.name
    result["hmin"] = mesh.hmin()
    result["hmax"] = mesh.hmax()
    result["k"] = k
    result["gdim"] = mesh.geometry().dim()
    result["num_dofs"] = V.dim()
    result["error_bank_weiser"] = error_bw
    result["error_exact"] = error_exact

    return result


def _global_pure_neumann(request, mesh, k, u_and_f_neumann):
    k_g = k

    V = FunctionSpace(mesh, "CG", k)
    V_f = FunctionSpace(mesh, "DG", k + 1)
    V_g = FunctionSpace(mesh, "DG", k_g)

    u_exact, f = u_and_f_neumann

    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v)) * dx + inner(u, v) * dx
    L = inner(f, v) * dx

    u_h = Function(V)
    A, b = assemble_system(a, L)

    solver = PETScKrylovSolver("cg", "hypre_amg")
    solver.solve(A, u_h.vector(), b)

    e = TrialFunction(V_f)
    v = TestFunction(V_f)

    n = FacetNormal(mesh)
    a_e = inner(grad(e), grad(v)) * dx + inner(e, v) * dx
    L_e = inner(f + div(grad(u_h)), v) * dx + inner(jump(grad(u_h), -n),
                                                    avg(v)) * dS - inner(inner(grad(u_h), n), v) * ds

    Id = np.zeros((V_f.dim(), V_f.dim()))
    w = Function(V_f)
    for i in range(V_f.dim()):
        w.vector()[i] = 1.0
        w_V_g = interpolate(w, V_g)
        w_V_f = interpolate(w_V_g, V_f)
        Id[:, i] = w_V_f.vector()
        w.vector()[i] = 0.0

    M = np.eye(V_f.dim()) - Id
    eigs, P = np.linalg.eig(M)
    mask = np.abs(eigs) > 0.5
    try:
        N = np.array(P[:, mask], dtype=np.float64)
    except ComplexWarning:
        pass

    A_e, b_e = assemble_system(a_e, L_e)
    A_e_0 = N.T @ A_e.array() @ N
    b_e_0 = N.T @ b_e

    e_0 = np.linalg.solve(A_e_0, b_e_0)
    e = N @ e_0
    e_V_f = Function(V_f)
    e_V_f.vector()[:] = e

    error_bw = np.sqrt(assemble(inner(e_V_f, e_V_f) * dx
                                + inner(grad(e_V_f), grad(e_V_f)) * dx))
    error_exact = errornorm(u_exact, u_h, "H1")

    result = {}
    result["name"] = request.node.name
    result["hmin"] = mesh.hmin()
    result["hmax"] = mesh.hmax()
    result["k"] = k
    result["gdim"] = mesh.geometry().dim()
    result["num_dofs"] = V.dim()
    result["error_bank_weiser"] = error_bw
    result["error_exact"] = error_exact

    return result
