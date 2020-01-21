## Copyright 2019-2020, Jack S. Hale, Raphaël Bulle
## SPDX-License-Identifier: LGPL-3.0-or-later
import os

import numpy as np
import pandas as pd

from dolfin import *
import ufl

import fenics_error_estimation

current_dir = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(current_dir, "exact_solution.h"), "r") as f:
    u_exact_code = f.read()

k = 2
u_exact = CompiledExpression(compile_cpp_code(u_exact_code).Exact(), degree=4)

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
        u_h = solve(V)
        with XDMFFile("output/u_h_{}.xdmf".format(str(i).zfill(4))) as f:
            f.write(u_h)
        result["error"] = errornorm(u_exact, u_h, "H10")

        u_exact_V = interpolate(u_exact, u_h.function_space())
        u_exact_V.rename("u_exact_V", "u_exact_V")
        with XDMFFile("output/u_exact_{}.xdmf".format(str(i).zfill(4))) as f:
            f.write(u_exact_V)

        eta_h = estimate(u_h)
        with XDMFFile("output/eta_hu_{}.xdmf".format(str(i).zfill(4))) as f:
            f.write(eta_h)
        result["error_bw"] = np.sqrt(eta_h.vector().sum())

        result["hmin"] = mesh.hmin()
        result["hmax"] = mesh.hmax()
        result["num_dofs"] = V.dim()

        markers = fenics_error_estimation.maximum(eta_h, 0.1)
        mesh = refine(mesh, markers)

        with XDMFFile("output/mesh_{}.xdmf".format(str(i).zfill(4))) as f:
            f.write(mesh)

        results.append(result)

    if (MPI.comm_world.rank == 0):
        df = pd.DataFrame(results)
        df.to_pickle("output/results.pkl")
        print(df)

def solve(V):
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
    solver = PETScLUSolver("mumps")
    solver.solve(A, u_h.vector(), b)

    return u_h


def estimate(u_h):
    mesh = u_h.function_space().mesh()

    element_f = FiniteElement("DG", triangle, k + 1)
    element_g = FiniteElement("DG", triangle, k)

    N = fenics_error_estimation.create_interpolation(element_f, element_g)

    V_f = FunctionSpace(mesh, element_f)

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

    e_h = fenics_error_estimation.estimate(a_e, L_e, N, bcs)
    error = norm(e_h, "H10")

    # Computation of local error indicator
    V_e = FunctionSpace(mesh, "DG", 0)
    v = TestFunction(V_e)

    eta_h = Function(V_e, name="eta_h")
    eta = assemble(inner(inner(grad(e_h), grad(e_h)), v)*dx)
    eta_h.vector()[:] = eta

    return eta_h


if __name__ == "__main__":
    main()


def test():
    main()
