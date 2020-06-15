## Copyright 2019-2020, Jack S. Hale, Raphaël Bulle
## SPDX-License-Identifier: LGPL-3.0-or-later

# Poisson problem on L-shaped domain with adaptive mesh refinement.  This is a
# classic problem shown in almost every paper on the topic.

import os

import numpy as np
import pandas as pd

from dolfin import *
import ufl

import mpi4py.MPI

import fenics_error_estimation

# We use DG functionality, requiring ghost regions on facets.
parameters["ghost_mode"] = "shared_facet"

current_dir = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(current_dir, "exact_solution.h"), "r") as f:
    u_exact_code = f.read()

k = 2
u_exact = CompiledExpression(compile_cpp_code(u_exact_code).Exact(), element=FiniteElement("CG", triangle, k + 3))

def main():
    comm = MPI.comm_world
    mesh = Mesh(comm)

    # The mesh is generated via gmsh.
    try:
        with XDMFFile(comm, os.path.join(current_dir, 'mesh.xdmf')) as f:
            f.read(mesh)
    except:
        print(
            "Generate the mesh using `python3 generate_mesh.py` before running this script.")
        exit()

    # Adaptive refinement loops
    results = []
    for i in range(0, 25):
        result = {}

        # Solve
        print("Solving...")
        V = FunctionSpace(mesh, "CG", k)
        u_h = solve(V)
        with XDMFFile("output/u_h_{}.xdmf".format(str(i).zfill(4))) as f:
            f.write(u_h)
        result["error"] = errornorm(u_exact, u_h, "H10")

        u_exact_V = interpolate(u_exact, u_h.function_space())
        u_exact_V.rename("u_exact_V", "u_exact_V")
        with XDMFFile("output/u_exact_{}.xdmf".format(str(i).zfill(4))) as f:
            f.write(u_exact_V)

        # Estimate
        print("Estimating...")
        eta_h, e_h = estimate(u_h)
        with XDMFFile("output/eta_hu_{}.xdmf".format(str(i).zfill(4))) as f:
            f.write_checkpoint(eta_h, "eta_h")
        with XDMFFile("output/e_h_{}.xdmf".format(str(i).zfill(4))) as f:
            f.write_checkpoint(e_h, "e_h")
        result["error_bw"] = np.sqrt(eta_h.vector().sum())

        # Exact local error
        V_e = eta_h.function_space()
        eta_exact = Function(V_e, name="eta_exact")
        v = TestFunction(V_e)
        eta_exact.vector()[:] = assemble(inner(inner(grad(u_h - u_exact_V), grad(u_h - u_exact_V)), v)*dx(mesh))
        result["error_exact"] = np.sqrt(eta_exact.vector().sum())
        with XDMFFile("output/eta_exact_{}.xdmf".format(str(i).zfill(4))) as f:
            f.write(eta_exact)

        # Necessary for parallel operation
        result["hmin"] = comm.reduce(mesh.hmin(), op=mpi4py.MPI.MIN, root=0)
        result["hmax"] = comm.reduce(mesh.hmax(), op=mpi4py.MPI.MAX, root=0)
        result["num_dofs"] = V.dim()

        # Mark
        print("Marking...")
        markers = fenics_error_estimation.dorfler(eta_h, 0.5)

        # and refine.
        print("Refining...")
        mesh = refine(mesh, markers, redistribute=True)

        with XDMFFile("output/mesh_{}.xdmf".format(str(i).zfill(4))) as f:
            f.write(mesh)

        results.append(result)

    if (MPI.comm_world.rank == 0):
        df = pd.DataFrame(results)
        df.to_pickle("output/results.pkl")
        print(df)

def solve(V):
    """Entirely standard Poisson solve with non-homogeneous Dirichlet
    conditions set according to exact solution"""
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
    """Bank-Weiser error estimation procedure"""
    mesh = u_h.function_space().mesh()

    # Higher order space
    element_f = FiniteElement("DG", triangle, k + 2)
    # Low order space
    element_g = FiniteElement("DG", triangle, k)

    # Construct the Bank-Weiser interpolation operator according to the
    # definition of the high and low order spaces.
    N = fenics_error_estimation.create_interpolation(element_f, element_g)

    V_f = FunctionSpace(mesh, element_f)

    e = TrialFunction(V_f)
    v = TestFunction(V_f)

    f = Constant(0.0)

    # Homogeneous zero Dirichlet boundary conditions
    bcs = DirichletBC(V_f, Constant(0.0), "on_boundary", "geometric")

    # Define the local Bank-Weiser problem on the full higher order space
    n = FacetNormal(mesh)
    a_e = inner(grad(e), grad(v))*dx
    # Residual
    L_e = inner(f + div(grad(u_h)), v)*dx + \
        inner(jump(grad(u_h), -n), avg(v))*dS

    # Local solves on the implied Bank-Weiser space. The solution is returned
    # on the full space.
    e_h = fenics_error_estimation.estimate(a_e, L_e, N, bcs)

    # Estimate of global error
    error = norm(e_h, "H10")

    # Computation of local error indicator using the now classic assemble
    # tested against DG_0 trick.
    V_e = FunctionSpace(mesh, "DG", 0)
    v = TestFunction(V_e)

    eta_h = Function(V_e, name="eta_h")
    eta = assemble(inner(inner(grad(e_h), grad(e_h)), v)*dx)
    eta_h.vector()[:] = eta

    return eta_h, e_h


if __name__ == "__main__":
    main()


def test():
    main()
