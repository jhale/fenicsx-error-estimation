# Copyright 2020, Jack S. Hale, Raphaël Bulle
# SPDX-License-Identifier: LGPL-3.0-or-later
import fenicsx_error_estimation
import numpy as np

import dolfinx
import ufl
from dolfinx.fem import (Constant, Function, FunctionSpace, assemble_scalar,
                         form)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_rectangle
from ufl import avg, div, dot, grad, inner, jump, pi, sin

from mpi4py import MPI

# The first part of this script is completely standard. We solve a screened
# Poisson problem on a square mesh with known data and homogeneous Neumann
# boundary conditions.

mesh = create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0, 0]), np.array([1, 1])], [4, 4],
    CellType.triangle)

k = 1
element = ufl.FiniteElement("CG", ufl.triangle, k)
V = FunctionSpace(mesh, element)
dx = ufl.Measure("dx", domain=mesh)

x = ufl.SpatialCoordinate(mesh)
f = (2.0 * (2.0 * pi)**2 + 1.0) * sin(2.0 * pi * x[0] - 0.5 * pi) * sin(2 * pi * x[1] - 0.5 * pi)
g = Constant(mesh, 0.0)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = inner(grad(u), grad(v)) * dx + inner(u, v) * dx
L = inner(f, v) * dx

problem = LinearProblem(a, L, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
u_h = problem.solve()

with XDMFFile(mesh.comm, "output/u.xdmf", "w") as of:
    of.write_mesh(mesh)
    of.write_function(u_h)

u_exact = sin(2.0 * pi * x[0] - 0.5 * pi) * sin(2.0 * pi * x[1] - 0.5 * pi)
error = mesh.comm.allreduce(assemble_scalar(
    form(inner(grad(u_h - u_exact), grad(u_h - u_exact)) * dx(degree=3))), op=MPI.SUM)
print("True error: {}".format(np.sqrt(error)))

# Now we specify the Bank-Weiser error estimation problem.
element_f = ufl.FiniteElement("DG", ufl.triangle, k + 1)
element_g = ufl.FiniteElement("DG", ufl.triangle, k)
element_e = ufl.FiniteElement("DG", ufl.triangle, 0)
N = fenicsx_error_estimation.create_interpolation(element_f, element_g)
print(N)
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
V_e = FunctionSpace(mesh, element_e)
e_h = ufl.Coefficient(V_f)
v_e = ufl.TestFunction(V_e)
L_eta = inner(inner(grad(e_h), grad(e_h)), v_e) * dx

# Functions to store results
eta_h = Function(V_e)

# Estimate the error using the Bank-Weiser approach.
# As we are solving a Neumann problem we pass an empty list of facet ids so
# that no Dirichlet conditions are applied to the local Bank-Weiser problems.
facets = np.array([], dtype=np.int32)
fenicsx_error_estimation.estimate(eta_h, a_e, L_e, L_eta, N, facets)

print(eta_h.x.array)

print("Bank-Weiser error from estimator: {}".format(np.sqrt(eta_h.vector.sum())))

with XDMFFile(mesh.comm, "output/eta.xdmf", "w") as of:
    of.write_mesh(mesh)
    of.write_function(eta_h)
