import numpy as np
from dolfin import *


def _lagrange_pure_dirichlet(request, mesh, k, u_and_f_dirichlet):
    if k != 1:
        raise ValueError(
            "Lagrange multiplier approach only works for order 1.")

    V = FunctionSpace(mesh, "CG", 1)
    V_f = FunctionSpace(mesh,
                        MixedElement([
                            FiniteElement("DG", mesh.ufl_cell(), 2),
                            FiniteElement("DG", mesh.ufl_cell(), 1)]))

    u_exact, f = u_and_f_dirichlet

    def boundary(x, on_boundary):
        return on_boundary

    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v))*dx
    L = inner(f, v)*dx

    bcs = [DirichletBC(V, Constant(0.0), boundary)]

    u_h = Function(V)
    A, b = assemble_system(a, L, bcs[0])

    solver = PETScKrylovSolver("cg", "hypre_amg")
    solver.solve(A, u_h.vector(), b)

    e, l = TrialFunctions(V_f)
    v, m = TestFunctions(V_f)

    n = FacetNormal(mesh)
    h_T = CellVolume(mesh)
    a_e = inner(grad(e), grad(v))*dx + Constant(1E5)*inner(e, v)*ds
    b_e = inner(e, m)*dx + inner(v, l)*dx

    L_e = inner(f + div(grad(u_h)), v)*dx + \
        inner(jump(grad(u_h), -n), avg(v))*dS

    A_e = assemble(a_e)
    B_e = assemble(b_e, form_compiler_parameters={
                   "representation": "quadrature", "quadrature_rule": "vertex"})
    b_e = assemble(L_e)

    C_e = A_e + B_e

    el_V_f = Function(V_f)

    solver = PETScLUSolver()
    solver.solve(C_e, el_V_f.vector(), b_e)

    # Discard Lagrange multiplier
    e_V_f = el_V_f[0]

    error_bw = np.sqrt(assemble(inner(grad(e_V_f), grad(e_V_f))*dx))
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


def _lagrange_pure_neumann(request, mesh, k, u_and_f_neumann):
    if k != 1:
        raise ValueError(
            "Lagrange multiplier approach only works for order 1.")

    V = FunctionSpace(mesh, "CG", 1)
    V_f = FunctionSpace(mesh,
                        MixedElement([
                            FiniteElement("DG", mesh.ufl_cell(), 2),
                            FiniteElement("DG", mesh.ufl_cell(), 1)]))

    u_exact, f = u_and_f_neumann

    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v))*dx + inner(u, v)*dx
    L = inner(f, v)*dx

    u_h = Function(V)
    A, b = assemble_system(a, L)

    solver = PETScKrylovSolver("cg", "hypre_amg")
    solver.solve(A, u_h.vector(), b)

    e, l = TrialFunctions(V_f)
    v, m = TestFunctions(V_f)

    n = FacetNormal(mesh)
    h_T = CellVolume(mesh)
    a_e = inner(grad(e), grad(v))*dx + inner(e, v)*ds
    b_e = inner(e, m)*dx + inner(v, l)*dx

    L_e = inner(f + div(grad(u_h)), v)*dx + inner(jump(grad(u_h), -n),
                                                  avg(v))*dS - inner(inner(grad(u_h), n), v)*ds

    A_e = assemble(a_e)
    B_e = assemble(b_e, form_compiler_parameters={
                   "representation": "quadrature", "quadrature_rule": "vertex"})
    b_e = assemble(L_e)

    C_e = A_e + B_e

    el_V_f = Function(V_f)

    solver = PETScLUSolver()
    solver.solve(C_e, el_V_f.vector(), b_e)

    # Discard Lagrange multiplier
    e_V_f = el_V_f[0]

    error_bw = np.sqrt(
        assemble(inner(grad(e_V_f), grad(e_V_f))*dx + inner(e_V_f, e_V_f)*dx))
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
