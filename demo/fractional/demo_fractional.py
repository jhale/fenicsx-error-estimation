# Copyright 2020, Jack S. Hale
# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
from dolfinx import cpp, Constant, form
from dolfinx import DirichletBC, Function, FunctionSpace, RectangleMesh
from dolfinx.cpp.mesh import CellType
from dolfinx.fem import (apply_lifting, assemble_matrix, assemble_vector, assemble_scalar, Form,
                         locate_dofs_topological, set_bc)
from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities_boundary

import fenics_error_estimation
from rational_sum_parameters import rational_param

import ufl
from ufl import avg, div, grad, inner, jump, Measure, TrialFunction, TestFunction, FacetNormal

# Fractional power (in (0, 1))
s = 0.5
# Lower bound spectrum
lmbda_0 = 1.
# Finite element degree
k = 1
# Tolerance (tolerance for rational sum will be tol * 1e-3 * l2_norm_data)
tol = 1e-3
# DÃrfler marking parameter
theta = 0.3

# Structured mesh
mesh = RectangleMesh(
        MPI.COMM_WORLD,
        [np.array([0, 0, 0]), np.array([1, 1, 0])], [4, 4],
        CellType.triangle)


# Input data
def f_e(x):
    values = (2./np.pi) * np.sin(x[0]) * np.sin(x[1])
    return values
l2_norm_data = 1.

# Generates rational sum parameters s.t. rational approx. error < tol * 1e-3 * l2_norm_data
rational_param(tol, s, lmbda_0, l2_norm_data)

c_1s = np.load("./rational_sum_parameters/c_1s.npy")
c_2s = np.load("./rational_sum_parameters/c_2s.npy")
weights = np.load("./rational_sum_parameters/weights.npy")
constant = np.load("./rational_sum_parameters/constant.npy")

# Initialize bilinear and linear forms
c_1 = Constant(0.)
c_2 = Constant(0.)

# Initialize estimator value
eta = 1.

ref_step = 0
while np.greater(eta, tol):
    # FE spaces
    element = ufl.FiniteElement("CG", mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, element)
    u = TrialFunction(V)
    v = TestFunction(V)

    element_f = ufl.FiniteElement("DG", ufl.triangle, 2)
    element_g = ufl.FiniteElement("DG", ufl.triangle, 1)
    element_e = ufl.FiniteElement("DG", ufl.triangle, 0)
    V_f = FunctionSpace(mesh.ufl_domain(), element_f)
    e_f = TrialFunction(V_f)
    v_f = TestFunction(V_f)
    
    V_e = FunctionSpace(mesh.ufl_domain(), element_e)
    v_e = TestFuntion(V_e)

    # Initialize BW estimator interpolation matrix
    N = fenics_error_estimation.create_interpolation(element_f, element_g)

    # Measures and normal
    dx = Measure("dx", domain=mesh)
    dS = Measure("dS", domain=mesh)
    n = ufl.FacetNormal(mesh.ufl_domain())

    # Interpolate input data into a FE function
    f = Function(V)
    f.interpolate(f_e)

    # Initialize bilinear and linear forms
    cst_1 = Constant(0.)
    cst_2 = Constant(0.)
    u_est = Coefficient(V)  #FIXME: see with Jack how to update Coeff in the loop
    e_h = Coefficient(V_f)

    a_form = form(cst_1 * inner(grad(u), grad(v)) * dx + cst_2 * inner(u, v) * dx)
    L_form = form(inner(f, v) * dx)

    # a_e_form = form(cst_1 * inner(grad(e_f), grad(v_f)) * dx + cst_2 * inner(e_f, v_f) * dx)
    # L_e_form = form(inner(f + cst_1 * div(grad(u_est)) - cst_2 * u_est, v_f) * dx\
    #         + inner(cst_1 * jump(grad(u_est), -n), avg(v_f)) * dS)

    # Homogeneous zero Dirichlet boundary condition
    u0 = Function(V)
    u0.vector.set(0.0)
    facets = locate_entities_boundary(
            mesh, 1, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = locate_dofs_topological(V, 1, facets)
    bcs = [DirichletBC(u0, dofs)]

    # BW estimator boundary conditions
    boundary_entities = locate_entities_boundary(
            mesh, 1, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_entities_sorted = np.sort(boundary_entities)

    # Initialize FE solution
    u_h = Function(V)
    # Initialize Bank-Weiser solution
    bw_f = Function(V_f)
    # Initialize estimator DG0 function
    eta_e = Function(V_e)
    for i, (c_1, c_2, weight) in enumerate(zip(c_1s, c_2s, weights)):
        '''
        Parametric problems solves
        '''
        # Linear system assembly
        cst_1.value = c_1
        cst_2.value = c_2
        A = assemble_matrix(a_form, bcs=bcs)
        b = assemble_vector(L_form)

        apply_lifting(b, [a_form], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, bcs)

        # Linear system solve
        options = PETSc.Options()
        options["ksp_type"] = "cg"
        options["ksp_view"] = None
        options["pc_type"] = "hypre"
        options["ksp_rtol"] = 1e-10
        options["pc_hypre_type"] = "boomeramg"
        options["ksp_monitor_true_residual"] = None

        u_param = Function(V)
        solver = PETSc.KSP().create(MPI.COMM_WORLD)
        solver.setOperators(A)
        solver.setFromOptions()
        solver.solve(b, u_param.vector)
        u_param.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                               mode=PETSc.ScatterMode.FORWARD)

        # Update fractional solution
        u_h.vector.axpy(weight, u_param.vector)

        '''
        A posteriori error estimation
        '''
        a_e_form = form(cst_1 * inner(grad(e_f), grad(v_f)) * dx + cst_2 * inner(e_f, v_f) * dx)
        L_e_form = form(inner(f + cst_1 * div(grad(u_h)) - cst_2 * u_h, v_f) * dx\
                + inner(cst_1 * jump(grad(u_h), -n), avg(v_f)) * dS)

        L_eta = form(inner(inner(e_h, e_h), v_e) * dx)

        # Functions to store results
        eta_h = Function(V_e)   # L2 norm of parametric BW solution (not used here)
        e_h_f = Function(V_f)   # Parametric Bank-Weiser solution
        e_D = Function(V_f)     # Zero dirichlet boundary condition

        fenicsx_error_estimation.estimate(
                eta_h, a_e, L_e, L_eta, N, boundary_entities_sorted, e_h=e_h_f, e_D=e_D, diagonal=max(1., cst_1.value))

        bw_f.vector.axpy(weight, e_h_f.vector)

    # Scale fractional solution and save
    u_h.vector.scale(constant)
    with XDMFFile(MPI.COMM_WORLD, "./output/u_{str(ref_step)}.xdmf", "w") as fo:
        fo.write_mesh(mesh)
        fo.write_function(u_h)

    # Scale Bank-Weiser solution
    bw_f.vector.scale(constant)

    # Compute L2 error estimator DG0 function and save
    eta_f = assemble_vector(inner(inner(bw_f, bw_f), v_e) * dx)
    eta_e.vector.setArray(eta_f)
    with XDMFFile(MPI.COMM_WORLD, "./output/eta_{str(ref_step)}.xdmf", "w") as fo:
        fo.write_mesh(mesh)
        fo.write_function(eta_e)

    # Compute L2 error estimator
    eta = np.sqrt(sum(eta_e.vector.array()))

    # Marking
    markers_tag = dorfler(mesh, eta_e, theta)

    # Refine mesh
    mesh.topology.create_entities(mesh.topology.dim - 1)
    mesh = dolfinx.mesh.refine(mesh, cell_markers=markers_tag)

    ref_step += 1
