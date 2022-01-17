# Copyright 2020, Jack S. Hale
# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np

import cffi
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
from dolfinx import DirichletBC, Function, FunctionSpace, UnitCubeMesh
from dolfinx.common import Timer
from dolfinx.fem import (apply_lifting, assemble_matrix, assemble_vector, assemble_scalar,
                         locate_dofs_topological, set_bc)
from dolfinx.mesh import locate_entities_boundary

from fenicsx_error_estimation import estimate, create_interpolation

import ufl
from ufl import avg, div, grad, inner, jump, pi, sin

ffi = cffi.FFI()

k = 2


def primal():
    mesh = UnitCubeMesh(MPI.COMM_WORLD, 256, 256, 256)

    element = ufl.FiniteElement("CG", ufl.tetrahedron, k)
    V = FunctionSpace(mesh, element)
    dx = ufl.Measure("dx", domain=mesh)

    x = ufl.SpatialCoordinate(mesh)
    f = 12.0 * pi**2 * sin(2.0 * pi * x[0]) * sin(2.0 * pi * x[1]) * sin(2.0 * pi * x[2])

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx(degree=k + 3)

    u0 = Function(V)
    u0.vector.set(0.0)
    facets = locate_entities_boundary(
        mesh, mesh.topology.dim - 1, lambda x: np.full(x.shape[1], True, dtype=bool))
    dofs = locate_dofs_topological(V, mesh.topology.dim - 1, facets)
    bcs = [DirichletBC(u0, dofs)]

    a_dolfin = dolfinx.Form(a)
    with Timer("Z Assemble matrix") as t:
        A = assemble_matrix(a_dolfin, bcs=bcs)
        A.assemble()

    L_dolfin = dolfinx.Form(L)
    with Timer("Z Assemble vector") as t:
        b = assemble_vector(L_dolfin)

    with Timer("Z Apply Dirichlet BCs") as t:
        apply_lifting(b, [a_dolfin], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, bcs)

    u = Function(V)
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    PETSc.Options()["ksp_type"] = "cg"
    PETSc.Options()["ksp_rtol"] = 1E-10
    PETSc.Options()["pc_type"] = "hypre"
    PETSc.Options()["pc_hypre_type"] = "boomeramg"
    PETSc.Options()["pc_hypre_boomeramg_strong_threshold"] = 0.5
    PETSc.Options()["pc_hypre_boomeramg_coarsen_type"] = "HMIS"
    PETSc.Options()["pc_hypre_boomeramg_interp_type"] = "ext+i"
    PETSc.Options()["pc_hypre_boomeramg_agg_nl"] = 4
    PETSc.Options()["pc_hypre_boomeramg_agg_num_paths"] = 2
    PETSc.Options()["pc_hypre_boomeramg_truncfactor"] = 0.35
    PETSc.Options()["ksp_monitor_true_residual"] = ""
    PETSc.Options()["ksp_view"] = ""
    solver.setOperators(A)
    solver.setFromOptions()

    with Timer("Z Solve") as t:
        solver.solve(b, u.vector)

    with Timer("Z Ghost update") as t:  # noqa: F841
        u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    u_exact = sin(2.0 * pi * x[0]) * sin(2.0 * pi * x[1]) * sin(2.0 * pi * x[2])
    local_error = assemble_scalar(inner(grad(u - u_exact), grad(u - u_exact)) * dx(degree=k + 3))
    error = mesh.mpi_comm().allreduce(local_error, op=MPI.SUM)
    if (MPI.COMM_WORLD.rank == 0):
        print("True error: {}".format(np.sqrt(error)))

    return u


def estimate_primal(u_h):
    mesh = u_h.function_space.mesh
    dx = ufl.Measure("dx", domain=mesh.ufl_domain())
    dS = ufl.Measure("dS", domain=mesh.ufl_domain())

    element_f = ufl.FiniteElement("DG", ufl.tetrahedron, k + 1)
    element_g = ufl.FiniteElement("DG", ufl.tetrahedron, k)
    element_e = ufl.FiniteElement("DG", ufl.tetrahedron, 0)
    N = create_interpolation(element_f, element_g)

    V_f = ufl.FunctionSpace(mesh.ufl_domain(), element_f)
    e = ufl.TrialFunction(V_f)
    v = ufl.TestFunction(V_f)

    n = ufl.FacetNormal(mesh.ufl_domain())

    # Data
    x = ufl.SpatialCoordinate(mesh.ufl_domain())
    f = 12.0 * pi**2 * sin(2.0 * pi * x[0]) * sin(2.0 * pi * x[1]) * sin(2.0 * pi * x[2])

    # Bilinear form
    a_e = inner(grad(e), grad(v)) * dx

    # Linear form
    L_e = inner(jump(grad(u_h), -n), avg(v)) * dS + inner(f + div((grad(u_h))), v) * dx(degree=k + 2)

    # Error form
    V_e = dolfinx.FunctionSpace(mesh, element_e)
    e_h = ufl.Coefficient(V_f)
    v_e = ufl.TestFunction(V_e)
    L_eta = inner(inner(grad(e_h), grad(e_h)), v_e) * dx

    # Functions to store results
    eta_h = Function(V_e)

    # Boundary conditions
    boundary_entities = locate_entities_boundary(
        mesh, mesh.topology.dim - 1, lambda x: np.full(x.shape[1], True, dtype=bool))
    boundary_entities_sorted = np.sort(boundary_entities)

    estimate(eta_h, a_e, L_e, L_eta, N, boundary_entities_sorted)

    bank_weiser_error = np.sqrt(eta_h.vector.sum())
    if (MPI.COMM_WORLD.rank == 0):
        print(f"Bank-Weiser error from estimator: {bank_weiser_error}")


def main():
    u = primal()
    estimate_primal(u)
    dolfinx.common.list_timings(MPI.COMM_WORLD, [dolfinx.cpp.common.TimingType.wall])


if __name__ == "__main__":
    main()
