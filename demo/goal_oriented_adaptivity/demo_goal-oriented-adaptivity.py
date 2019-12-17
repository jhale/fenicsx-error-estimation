import numpy as np

from dolfin import *
import ufl

import bank_weiser

with open("exact_solution.h", "r") as f:
    u_exact_code = f.read()

k = 1
u_exact = CompiledExpression(compile_cpp_code(u_exact_code).Exact(), degree=5)

def main():
    mesh = Mesh()
    try:
        with XDMFFile(MPI.comm_world, 'mesh.xdmf') as f:
            f.read(mesh)
    except:
        print("Generate the mesh using `python3 generate_mesh.py` before running this script.")
        exit()

    for i in range(0, 7):
        V = FunctionSpace(mesh, "CG", k)
        u_h = primal_solve(V)
        with XDMFFile("output/u_h_{}.xdmf".format(str(i).zfill(4))) as f:
            f.write(u_h)

        z_h = dual_solve(u_h)
        with XDMFFile("output/z_h_{}.xdmf".format(str(i).zfill(4))) as f:
            f.write(z_h)

        eta_hu = estimate(u_h)
        with XDMFFile("output/eta_hu_{}.xdmf".format(str(i).zfill(4))) as f:
            f.write(eta_hu)

        eta_hz = estimate(z_h)
        with XDMFFile("output/eta_hz_{}.xdmf".format(str(i).zfill(4))) as f:
            f.write(eta_hz)

        eta_hw = weighted_estimator(eta_hu, eta_hz)
        with XDMFFile("output/eta_hw_{}.xdmf".format(str(i).zfill(4))) as f:
            f.write(eta_hw)

        markers = mark(eta_hw, 0.1)

        mesh = refine(mesh, markers)

        with XDMFFile("output/mesh_{}.xdmf".format(str(i).zfill(4))) as f:
            f.write(mesh)

def primal_solve(V):
    u = TrialFunction(V)
    v = TestFunction(V)

    f = Constant(0.0)

    a = inner(grad(u), grad(v))*dx
    L = inner(f, v)*dx

    def all_boundary(x, on_boundary):
        return on_boundary

    bcs = DirichletBC(V, u_exact, all_boundary)

    A, b = assemble_system(a, L, bcs=bcs)

    u_h = Function(V, name="u_h")
    solver = PETScLUSolver()
    solver.solve(A, u_h.vector(), b)

    return u_h

def dual_solve(u_h):
    V = u_h.function_space()

    z = TrialFunction(V)
    v = TestFunction(V)

    eps_f = 0.2
    centre = 0.4
    cpp_f = """
    ((x[0] - centre)/eps_f)*((x[0] - centre)/eps_f) +((x[1] - centre)/eps_f)*((x[1] - centre)/eps_f) < 1.0 ? 
    (1.0/0.4665123965446233)*pow(eps_f, -2.0)*
    exp(-1.0/(1.0 - (((x[0] - centre)/eps_f)*((x[0] - centre)/eps_f) + ((x[1] - centre)/eps_f)*((x[1] - centre)/eps_f)))) :
    0.0"""

    c = Expression(cpp_f, eps_f=eps_f, centre=centre, degree=3)
    J = inner(c, v)*dx

    a = inner(grad(z), grad(v))*dx

    def all_boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(V, Constant(0.0), all_boundary)

    A, b = assemble_system(a, J, bcs=bc)

    z_h = Function(V, name="z_h")
    solver = PETScLUSolver()
    solver.solve(A, z_h.vector(), b)

    return z_h

def estimate(u_h):
    mesh = u_h.function_space().mesh()

    V_f = FunctionSpace(mesh, "DG", k + 1)
    V_g = FunctionSpace(mesh, "DG", k)

    N = bank_weiser.local_interpolation_to_V0(V_f, V_g)

    e = TrialFunction(V_f)
    v = TestFunction(V_f)

    f = Constant(0.0)

    def all_boundary(x, on_boundary):
        return on_boundary

    bcs = DirichletBC(V_f, Constant(0.0), all_boundary, 'geometric')

    n = FacetNormal(mesh)
    a_e = inner(grad(e), grad(v))*dx
    L_e = inner(f + div(grad(u_h)), v)*dx + \
          inner(jump(grad(u_h), -n), avg(v))*dS

    e_h = bank_weiser.estimate(a_e, L_e, N, bcs)
    error = norm(e_h, "H10")

    # Computation of local error indicator
    V_e = FunctionSpace(mesh, "DG", 0)
    v = TestFunction(V_e)

    eta_h = Function(V_e, name="eta")
    eta = assemble(inner(inner(grad(e_h), grad(e_h)), v)*dx)
    eta_h.vector()[:] = eta

    return eta_h

def weighted_estimator(eta_uh, eta_zh):
    eta_uh_vec = eta_uh.vector()
    eta_zh_vec = eta_zh.vector()

    sum_eta_uh = eta_uh_vec.sum()
    sum_eta_zh = eta_zh_vec.sum()

    eta_wh = Function(eta_uh.function_space(), name="eta")
    eta_wh.vector()[:] = ((sum_eta_zh/(sum_eta_uh + sum_eta_zh))*eta_uh_vec) + \
                         ((sum_eta_uh/(sum_eta_uh + sum_eta_zh))*eta_zh_vec)

    return eta_wh

def mark(eta_h, alpha):
    etas = eta_h.vector().get_local()
    indices = etas.argsort()[::-1]
    sorted = etas[indices]

    total = sum(sorted)
    fraction = alpha*total

    mesh = eta_h.function_space().mesh()
    markers = MeshFunction("bool", mesh, mesh.geometry().dim(), False)

    v = 0.0
    for i in indices:
        if v >= fraction:
            break
        markers[int(i)] = True
        v += sorted[i]

    return markers

if __name__ == "__main__":
    main()
