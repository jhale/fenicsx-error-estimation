# Copyright 2020, Jack S. Hale, Raphael Bulle.
# SPDX-License-Identifier: LGPL-3.0-or-later

import fenicsx_error_estimation
import numpy as np

import dolfinx
import ufl
from dolfinx.fem import (Constant, Function, FunctionSpace, apply_lifting,
                         dirichletbc, form, locate_dofs_topological, set_bc)
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from dolfinx.io import XDMFFile
from dolfinx.mesh import (CellType, compute_incident_entities,
                          create_unit_square, locate_entities_boundary)
from ufl import (Coefficient, Measure, TestFunction, TrialFunction, avg, div,
                 grad, inner, jump)

from mpi4py import MPI
from petsc4py import PETSc
import pandas as pd

import sys
sys.path.append("../")

from rational_schemes import BP_rational_approximation, BURA_rational_approximation
from FE_utils import mesh_refinement, parametric_problem

def solve_parametric(cst_1, cst_2, refinement_index):
    # Read the mesh corresponding to the refinement_index
    with XDMFFile(MPI.COMM_WORLD, os.path.join(os.sep, "meshes", f"mesh_{str(refinement_index).zfill(4)}.xdmf"), "r") as fi:
        mesh = fi.read_mesh()

    # Measure, FE space and trial/test functions definitions
    dx = Measure("dx", domain=mesh)
    element = ufl.FiniteElement("CG", mesh.ufl_cell(), k)
    V = FunctionSpace(mesh, element)
    u = TrialFunction(V)
    v = TestFunction(V)

    # Interpolation of data
    f_V = Function(V)
    f_V.interpolate(f)

    # Bilinear and linear forms
    c_1 = Constant(mesh, cst_1)
    c_2 = Constant(mesh, cst_2)
    a_form = form(c_1 * inner(grad(u_param), grad(v)) * dx
                + c_2 * inner(u_param, v) * dx)
    L_form = form(inner(f_e, v) * dx)

    # Homogeneous Dirichlet boundary conditions
    u0 = Function(V)
    u0.vector.set(0.)
    facets = locate_entities_boundary(mesh, 1, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = locate_dofs_topological(V, 1, facets)
    bcs = [dirichletbc(u0, dofs)]

    # Assemble system
    A = assemble_matrix(a_form, bcs=bcs)
    A.assemble()

    b = assemble_vector(L_form)
    apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)

    options = PETSc.Options()
    options["ksp_type"] = "cg"
    options["ksp_view"] = None
    options["pc_type"] = "hypre"
    options["ksp_rtol"] = 1E-10
    options["pc_hypre_type"] = "boomeramg"
    options["ksp_monitor_true_residual"] = None

    u_param = Function(V)
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setOperators(A)
    solver.setFromOptions()
    solver.solve(b, u_param.vector)
    u_param.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                               mode=PETSc.ScatterMode.FORWARD)
    
    return u_param

def estimate_parametric(cst_1, cst_2, u_param):
    V = u_param.functionspace()
    mesh = V.mesh()
    ufl_domain = mesh.ufl_domain()

    dx = ufl.Measure("dx", domain=ufl_domain)
    dS = ufl.Measure("dS", domain=ufl_domain)

    element_f = ufl.FiniteElement("DG", ufl.triangle, 2)
    element_g = ufl.FiniteElement("DG", ufl.triangle, 1)
    element_e = ufl.FiniteElement("DG", ufl.triangle, 0)
    N = create_interpolation(element_f, element_g)

    V_ef = ufl.FunctionSpace(ufl_domain, element_f)
    e_ef = ufl.TrialFunction(V_ef)
    v_ef = ufl.TestFunction(V_ef)

    n = ufl.FacetNormal(ufl_domain)

    # Bilinear form (two constants)
    c_1 = Constant(mesh, cst_1)
    c_2 = Constant(mesh, cst_2)
    a_ef = c_1*inner(grad(e_ef), grad(v_ef)) * dx + c_2*inner(e_ef, v_ef) * dx

    # Linear form (two coefficients, two constants)
    r = f + c_1 * div(grad(u_param)) - c_2 * u_param
    L_ef = inner(r, v_ef)*dx + inner(c_1 * jump(grad(u_param), -n), avg(v_ef)) * dS

    # Error form (one coefficient)
    V_e = dolfinx.FunctionSpace(mesh, element_e)
    e_h = ufl.Coefficient(V_ef)
    v_e = ufl.TestFunction(V_e)
    # L_eta = inner(inner(e_h, e_h), v_e) * dx   # L2 estimation
    # Fractional alpha/2 norm estimation
    L_eta = inner(c_1*inner(grad(e_h), grad(e_h)) +
                  c_2*inner(e_h, e_h), v_e) * dx

    V_f = dolfinx.FunctionSpace(mesh, element_f)
    # Function to store result
    eta_h = Function(V_e)       # Norm of Bank-Weiser solution
    e_h_f = Function(V_f)       # Bank-Weiser solution

    # Boundary conditions
    boundary_entities = locate_entities_boundary(
        mesh, 1, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_entities_sorted = np.sort(boundary_entities)

    fenics_error_estimation.estimate(
        eta_h, u_param, a_ef, L_ef, L_eta, N, boundary_entities_sorted, e_h=e_h_f)    

    return eta_h, e_h_f


if __name__=="__main__":
    # FE degree
    k = 1

    # Fractional power
    s = 0.5

    # Max number refinement steps
    ref_step_max = 3

    # Rational scheme parameters
    fineness_parameter = 0.3
    rational_parameters, _ = BP_rational_approximation(s, fineness_parameter)
    ls_cst_1       = rational_parameters["c_1s"]
    ls_cst_2       = rational_parameters["c_2s"]
    ls_weight      = rational_parameters["weights"]
    constant         = rational_parameters["constant"]
    initial_constant = rational_parameters["initial constant"]

    # List of indices encoding which mesh to use
    refinement_indices = np.zeros_like(ls_cst_1)

    # Data
    def f(x):
        values = np.ones(x.shape[1])
        values[np.where(np.logical_and(x[0] < 0.5, x[1] > 0.5))] = -1.0
        values[np.where(np.logical_and(x[0] > 0.5, x[1] < 0.5))] = -1.0
        return values

    # Initial mesh
    initial_mesh = create_unit_square(MPI.COMM_WORLD, 8, 8)

    with XDMFFile(MPI.COMM_WORLD, os.path.join(os.sep, "meshes", f"mesh_{str(0).zfill(4)}.xdmf"), "w") as of:
        of.write_mesh(initial_mesh)

    mesh = initial_mesh
    for ref_step in range(ref_step_max):

        ls_eta_param = []
        for cst_1, cst_2, weight, refinement_index, pbm_num in zip(ls_cst_1, ls_cst_2, ls_weight, range(len(ls_cst_1))):
            # Output dir
            output_dir = f"pbm_{str(pbm_num).zfill(4)}"

            # Solve the parametric problem
            u_param = solve_parametric(cst_1, cst_2, refinement_index)
            with XDMFFile(mesh.comm, os.path.join(output_dir, os.sep, f"u_{str(ref_step).zfill(4)}.xdmf"), "w") as of:
                of.write_mesh(mesh)
                of.write_function(u_param)

            # Estimate the parametric L2 error
            eta_param, e_h_f = estimate_parametric(cst_1, cst_2, refinement_index)
            with XDMFFile(mesh.comm, os.path.join(output_dir, os.sep, f"eta_{str(ref_step).zfill(4)}.xdmf"), "w") as of:
                of.write_mesh(mesh)
                of.write_function(eta_param)
            with XDMFFile(mesh.comm, os.path.join(output_dir, os.sep, f"e_h_f_{str(ref_step).zfill(4)}.xdmf"), "w") as of:
                of.write_mesh(mesh)
                of.write_function(e_h_f)

