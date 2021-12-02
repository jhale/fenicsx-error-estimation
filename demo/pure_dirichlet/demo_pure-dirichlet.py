# Copyright 2020, Jack S. Hale
# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
from dolfinx import cpp
from dolfinx import DirichletBC, Function, FunctionSpace, RectangleMesh
from dolfinx.cpp.mesh import CellType
from dolfinx.fem import (apply_lifting, assemble_matrix, assemble_vector, assemble_scalar, Form,
                         locate_dofs_topological, set_bc)
from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities_boundary

import fenicsx_error_estimation

import ufl
from ufl import avg, cos, div, dS, dx, grad, inner, jump, pi, sin

# Structured mesh
mesh = RectangleMesh(
    MPI.COMM_WORLD,
    [np.array([0, 0, 0]), np.array([1, 1, 0])], [32, 32],
    CellType.triangle)

k = 1
element = ufl.FiniteElement("CG", ufl.triangle, k)
V = FunctionSpace(mesh, element)
dx = ufl.Measure("dx", domain=mesh)

x = ufl.SpatialCoordinate(mesh)
f = 8.0 * pi**2 * sin(2.0 * pi * x[0]) * sin(2.0 * pi * x[1])

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = inner(grad(u), grad(v)) * dx
L = inner(f, v) * dx

u0 = Function(V)
with u0.vector.localForm() as u0_local:
    u0_local.set(0.0)

facets = locate_entities_boundary(
    mesh, mesh.topology.dim - 1, lambda x: np.full(x.shape[1], True, dtype=bool))
dofs = locate_dofs_topological(V, mesh.topology.dim - 1, facets)
bcs = [DirichletBC(u0, dofs)]

problem = dolfinx.fem.LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
u_h = problem.solve()

with XDMFFile(MPI.COMM_WORLD, "output/u.xdmf", "w") as of:
    of.write_mesh(mesh)
    of.write_function(u_h)

u_exact = sin(2.0 * pi * x[0]) * sin(2.0 * pi * x[1])
error = MPI.COMM_WORLD.allreduce(assemble_scalar(
    inner(grad(u_h - u_exact), grad(u_h - u_exact)) * dx(degree=k + 3)), op=MPI.SUM)
print("True error: {}".format(np.sqrt(error)))

# Now we specify the Bank-Weiser error estimation problem.
element_f = ufl.FiniteElement("DG", ufl.triangle, k + 1)
element_g = ufl.FiniteElement("DG", ufl.triangle, k)
element_e = ufl.FiniteElement("DG", ufl.triangle, 0)
N = fenicsx_error_estimation.create_interpolation(element_f, element_g)

# The local error estimation problem is written in pure UFL. This allows us to
# avoid making a dolfinx.FunctionSpace on the high-order discontinuous Galerkin
# space, which can use a lot of memory in three-dimensions.
V_f = ufl.FunctionSpace(mesh.ufl_domain(), element_f)
e = ufl.TrialFunction(V_f)
v = ufl.TestFunction(V_f)

# Bilinear form
a_e = inner(grad(e), grad(v)) * dx

# Linear form
n = ufl.FacetNormal(mesh)
dS = ufl.Measure("dS", domain=mesh)
L_e = inner(jump(grad(u_h), -n), avg(v)) * dS + inner(f + div((grad(u_h))), v) * dx

# Error form
# Note that e_h is a ufl.Coefficient, not a dolfinx.Function. Inside the
# assembler e_h is computed locally 'on-the-fly' and then discarded.
V_e = dolfinx.FunctionSpace(mesh, element_e)
e_h = ufl.Coefficient(V_f)
v_e = ufl.TestFunction(V_e)
L_eta = inner(inner(grad(e_h), grad(e_h)), v_e) * dx

# Functions to store results
eta_h = Function(V_e)

# We must apply homogeneous zero Dirichlet condition on the local problems when
# a (possibly non-zero) Dirichlet condition was applied to the original
# problem. Due to the way boundary conditions are enforced locally it is only
# necessary to compute a sorted list of entities (facets) on which homogeneous
# Dirichlet conditions should be applied.
facets_sorted = np.sort(facets)

# Estimate the error using the Bank-Weiser approach.
fenicsx_error_estimation.estimate(eta_h, a_e, L_e, L_eta, N, facets_sorted)

print("Bank-Weiser error from estimator: {}".format(np.sqrt(eta_h.vector.sum())))

with XDMFFile(MPI.COMM_WORLD, "output/eta.xdmf", "w") as of:
    of.write_mesh(mesh)
    of.write_function(eta_h)
