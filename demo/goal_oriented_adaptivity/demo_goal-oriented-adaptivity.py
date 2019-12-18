import os

import numpy as np
import pandas as pd

from dolfin import *
import ufl

import bank_weiser

current_dir = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(current_dir, "exact_solution.h"), "r") as f:
    u_exact_code = f.read()

k = 1
u_exact = CompiledExpression(compile_cpp_code(u_exact_code).Exact(), degree=5)

# Calculated using P3 on a very fine adapted mesh, good to ~10 s.f.
J_fine = 0.2010229183

def main():
    mesh = Mesh()
    try:
        with XDMFFile(MPI.comm_world, os.path.join(current_dir, 'mesh.xdmf')) as f:
            f.read(mesh)
    except:
        print(
            "Generate the mesh using `python3 generate_mesh.py` before running this script.")
        exit()

    results = []
    for i in range(0, 15):
        result = {}
        V = FunctionSpace(mesh, "CG", k)
        u_h = primal_solve(V)
        with XDMFFile("output/u_h_{}.xdmf".format(str(i).zfill(4))) as f:
            f.write(u_h)

        J_h = assemble(J(u_h))

        z_h = dual_solve(u_h)
        with XDMFFile("output/z_h_{}.xdmf".format(str(i).zfill(4))) as f:
            f.write(z_h)

        eta_hu = estimate(u_h)
        with XDMFFile("output/eta_hu_{}.xdmf".format(str(i).zfill(4))) as f:
            f.write(eta_hu)

        eta_hz = estimate(z_h)
        with XDMFFile("output/eta_hz_{}.xdmf".format(str(i).zfill(4))) as f:
            f.write(eta_hz)

        eta_hw = bank_weiser.weighted_estimate(eta_hu, eta_hz)
        with XDMFFile("output/eta_hw_{}.xdmf".format(str(i).zfill(4))) as f:
            f.write(eta_hw)

        markers = bank_weiser.mark(eta_hw, 0.1)
        error_hu = np.sqrt(eta_hu.vector().sum())
        error_hz = np.sqrt(eta_hz.vector().sum())

        result["J_h"] = J_h
        result["error"] = np.abs(J_h - J_fine)
        result["estimated_error"] = error_hu*error_hz
        result["hmin"] = mesh.hmin()
        result["hmax"] = mesh.hmax()
        result["num_dofs"] = V.dim()

        mesh = refine(mesh, markers)

        with XDMFFile("output/mesh_{}.xdmf".format(str(i).zfill(4))) as f:
            f.write(mesh)

        results.append(result)

    df = pd.DataFrame(results)
    df.to_pickle("output/results.pkl")
    print(df)


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


def J(v):
    eps_f = 0.35
    centre = 0.2
    cpp_f = """
    ((x[0] - centre)/eps_f)*((x[0] - centre)/eps_f) + ((x[1] - centre)/eps_f)*((x[1] - centre)/eps_f) < 1.0 ? 
    (1.0)*pow(eps_f, -2.0)*
    exp(-1.0/(1.0 - (((x[0] - centre)/eps_f)*((x[0] - centre)/eps_f) + ((x[1] - centre)/eps_f)*((x[1] - centre)/eps_f)))) :
    0.0"""

    c = Expression(cpp_f, eps_f=eps_f, centre=centre, degree=3)
    J = inner(c, v)*dx

    return J


def dual_solve(u_h):
    V = u_h.function_space()

    z = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(z), grad(v))*dx
    J_v = J(v)

    def all_boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(V, Constant(0.0), all_boundary)

    A, b = assemble_system(a, J_v, bcs=bc)

    z_h = Function(V, name="z_h")
    solver = PETScLUSolver()
    solver.solve(A, z_h.vector(), b)

    return z_h


def estimate(u_h):
    mesh = u_h.function_space().mesh()

    V_f = FunctionSpace(mesh, "DG", k + 1)
    V_g = FunctionSpace(mesh, "DG", k)

    N = bank_weiser.create_interpolation(V_f, V_g)

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

if __name__ == "__main__":
    main()


def test():
    main()
