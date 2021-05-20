# Copyright 2020, Jack S. Hale, Raphaël Bulle
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

import fenics_error_estimation

import ufl
from ufl import avg, cos, div, dot, dS, dx, grad, inner, jump, pi, sin
from ufl.algorithms.elementtransformations import change_regularity

assert dolfinx.has_petsc_complex == False

# The first part of this script is completely standard. We solve a screened
# Poisson problem on a square mesh with known data and homogeneous Neumann
# boundary conditions.

mesh = RectangleMesh(
    MPI.COMM_WORLD,
    [np.array([0, 0, 0]), np.array([1, 1, 0])], [128, 128],
    CellType.triangle)

element = ufl.FiniteElement("CG", ufl.triangle, 1)
V = FunctionSpace(mesh, element)
dx = ufl.Measure("dx", domain=mesh)

x = ufl.SpatialCoordinate(mesh)
f = (2.0 * (2.0 * pi)**2 + 1.0) * sin(2.0 * pi * x[0] - 0.5 * pi) * sin(2 * pi * x[1] - 0.5 * pi)
g = dolfinx.Constant(mesh, 0.0)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = inner(grad(u), grad(v)) * dx + inner(u, v) * dx
L = inner(f, v) * dx

A = assemble_matrix(a)
A.assemble()

b = assemble_vector(L)

u_h = Function(V)
solver = PETSc.KSP().create(MPI.COMM_WORLD)
solver.setOperators(A)
solver.solve(b, u_h.vector)
u_h.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

with XDMFFile(mesh.mpi_comm(), "output/u.xdmf", "w") as of:
    of.write_mesh(mesh)
    of.write_function(u_h)

u_exact = sin(2.0 * pi * x[0] - 0.5 * pi) * sin(2.0 * pi * x[1] - 0.5 * pi)
error = MPI.COMM_WORLD.allreduce(assemble_scalar(
    inner(grad(u_h - u_exact), grad(u_h - u_exact)) * dx(degree=3)), op=MPI.SUM)
print("True error: {}".format(np.sqrt(error)))

# Now we specify the Bank-Weiser error estimation problem.
element_f = ufl.FiniteElement("DG", ufl.triangle, 2)
element_g = ufl.FiniteElement("DG", ufl.triangle, 1)
element_e = ufl.FiniteElement("DG", ufl.triangle, 0)
N = fenics_error_estimation.create_interpolation(element_f, element_g)

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
ds = ufl.Measure("ds", domain=mesh)
L_e = inner(jump(grad(u_h), -n), avg(v)) * dS + inner(f + div((grad(u_h))), v) * \
    dx + inner(g - dot(grad(u_h), n), v) * ds

# Error form
# Note that e_h is a ufl.Coefficient, not a dolfinx.Function. Inside the
# assembler e_h is computed locally 'on-the-fly' and then discarded.
V_e = dolfinx.FunctionSpace(mesh, element_e)
e_h = ufl.Coefficient(V_f)
v_e = ufl.TestFunction(V_e)
L_eta = inner(inner(grad(e_h), grad(e_h)), v_e) * dx

# Functions to store results
eta_h = Function(V_e)

# Estimate the error using the Bank-Weiser approach.
# As we are solving a Neumann problem we pass an empty list of facet ids so
# that no Dirichlet conditions are applied to the local Bank-Weiser problems.
facets = np.array([], dtype=np.int32)
fenics_error_estimation.estimate(eta_h, u_h, a_e, L_e, L_eta, N, facets)

print("Bank-Weiser error from estimator: {}".format(np.sqrt(eta_h.vector.sum())))

with XDMFFile(mesh.mpi_comm(), "output/eta.xdmf", "w") as of:
    of.write_mesh(mesh)
    of.write_function(eta_h)
