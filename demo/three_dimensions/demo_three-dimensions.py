# Copyright 2019-2020, Jack S. Hale, Raphaël Bulle
# SPDX-License-Identifier: LGPL-3.0-or-later

# Three-dimensional Poisson problem using Bank-Weiser estimator.  Proof of
# reliability of this estimator in Removing the saturation assumption in
# Bank-Weiser error estimator analysis in dimension three Bulle et al.
# https://hal.archives-ouvertes.fr/hal-02482235

import pandas as pd
import numpy as np

from dolfin import *

import fenics_error_estimation

k = 1
parameters["ghost_mode"] = "shared_facet"
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True


def main():
    mesh = Mesh()
    try:
        with XDMFFile(MPI.comm_world, 'mesh.xdmf') as f:
            f.read(mesh)
    except RuntimeError:
        print(
            "Generate the mesh using `python3 generate_mesh.py` before running this script.")
        exit()

    results = []
    for i in range(0, 10):
        result = {}
        V = FunctionSpace(mesh, "CG", k)
        u_h = solve(V)

        print("Estimating...")
        eta_h = estimate(u_h)
        result["error_bw"] = np.sqrt(eta_h.vector().sum())
        result["hmin"] = mesh.hmin()
        result["hmax"] = mesh.hmax()
        result["num_dofs"] = V.dim()

        print("Marking...")
        markers = fenics_error_estimation.maximum(eta_h, 0.2)
        print("Refining...")
        mesh = refine(mesh, markers, redistribute=True)

        with XDMFFile("output/mesh_{}.xdmf".format(str(i).zfill(4))) as f:
            f.write(mesh)

        with XDMFFile("output/u_{}.xdmf".format(str(i).zfill(4))) as f:
            f.write(u_h)

        with XDMFFile("output/eta_{}.xdmf".format(str(i).zfill(4))) as f:
            f.write(eta_h)

        results.append(result)

    if (MPI.comm_world.rank == 0):
        df = pd.DataFrame(results)
        df.to_pickle("output/results.pkl")
        print(df)


def solve(V):
    u = TrialFunction(V)
    v = TestFunction(V)

    f = Constant(1.0)

    a = inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx

    bcs = DirichletBC(V, Constant(0.0), "on_boundary")

    A, b = assemble_system(a, L, bcs=bcs)

    u_h = Function(V)

    PETScOptions.set("ksp_type", "cg")
    PETScOptions.set("ksp_rtol", 1E-10)
    PETScOptions.set("ksp_monitor_true_residual")
    PETScOptions.set("pc_type", "hypre")
    PETScOptions.set("pc_hypre_type", "boomeramg")
    PETScOptions.set("pc_hypre_boomeramg_strong_threshold", 0.5)
    PETScOptions.set("pc_hypre_boomeramg_coarsen_type", "HMIS")
    PETScOptions.set("pc_hypre_boomeramg_agg_nl", 4)
    PETScOptions.set("pc_hypre_boomeramg_agg_num_paths", 2)
    PETScOptions.set("pc_hypre_boomeramg_interp_type", "ext+i")
    PETScOptions.set("pc_hypre_boomeramg_truncfactor", 0.35)
    PETScOptions.set("ksp_view")
    solver = PETScKrylovSolver()
    solver.set_from_options()
    solver.solve(A, u_h.vector(), b)

    return u_h


def estimate(u_h):
    mesh = u_h.function_space().mesh()

    element_f = FiniteElement("DG", tetrahedron, k + 1)
    element_g = FiniteElement("DG", tetrahedron, k)
    V_f = FunctionSpace(mesh, element_f)

    N = fenics_error_estimation.create_interpolation(element_f, element_g)

    e = TrialFunction(V_f)
    v = TestFunction(V_f)

    f = Constant(1.0)

    bcs = DirichletBC(V_f, Constant(0.0), "on_boundary", "geometric")

    n = FacetNormal(mesh)
    a_e = inner(grad(e), grad(v)) * dx
    L_e = inner(f + div(grad(u_h)), v) * dx + \
        inner(jump(grad(u_h), -n), avg(v)) * dS

    e_h = fenics_error_estimation.estimate(a_e, L_e, N, bcs)
    # error = norm(e_h, "H10")

    # Computation of local error indicator
    V_e = FunctionSpace(mesh, "DG", 0)
    v = TestFunction(V_e)

    eta_h = Function(V_e)
    eta = assemble(inner(inner(grad(e_h), grad(e_h)), v) * dx)
    eta_h.vector()[:] = eta

    return eta_h


if __name__ == "__main__":
    main()


def test():
    pass
