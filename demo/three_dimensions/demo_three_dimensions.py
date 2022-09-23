# Copyright 2020, Jack S. Hale
# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np
from fenicsx_error_estimation import create_interpolation, estimate

import ufl
from dolfinx.fem import (Function, FunctionSpace, apply_lifting,
                         assemble_scalar, dirichletbc, form,
                         locate_dofs_topological, set_bc)
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_unit_cube, locate_entities_boundary
from ufl import avg, div, grad, inner, jump, pi, sin

from mpi4py import MPI
from petsc4py import PETSc

k = 2


def primal():
    mesh = create_unit_cube(MPI.COMM_WORLD, 64, 64, 64)

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
    bcs = [dirichletbc(u0, dofs)]

    A = assemble_matrix(form(a), bcs=bcs)
    A.assemble()

    b = assemble_vector(form(L))
    apply_lifting(b, [form(a)], [bcs])
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
    solver.solve(b, u.vector)
    u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    with XDMFFile(mesh.comm, "output/u.xdmf", "w") as of:
        of.write_mesh(mesh)
        of.write_function(u)

    u_exact = sin(2.0 * pi * x[0]) * sin(2.0 * pi * x[1]) * sin(2.0 * pi * x[2])
    error = mesh.comm.allreduce(assemble_scalar(
        form(inner(grad(u - u_exact), grad(u - u_exact)) * dx(degree=k + 3))), op=MPI.SUM)
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
    V_e = FunctionSpace(mesh, element_e)
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

    print("Bank-Weiser error from estimator: {}".format(np.sqrt(eta_h.vector.sum())))

    with XDMFFile(mesh.comm, "output/eta.xdmf", "w") as of:
        of.write_mesh(mesh)
        of.write_function(eta_h)


def main():
    u = primal()
    estimate_primal(u)


if __name__ == "__main__":
    main()
