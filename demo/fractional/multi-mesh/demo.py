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

import os
import sys
sys.path.append("../")

from rational_schemes import BP_rational_approximation, BURA_rational_approximation
from FE_utils import mesh_refinement, parametric_problem

"""
solve_parametric: solves the finite element parametric problem given by
cst_1 * u - cst_2 Delta u = f
on the mesh corresponding to the given refinement_index.


"""
def solve_parametric(mesh, cst_1, cst_2):
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
    a_form = form(cst_1 * inner(grad(u), grad(v)) * dx
                + cst_2 * inner(u, v) * dx)
    L_form = form(inner(f_V, v) * dx)

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
    #apply_lifting(b, [A], [bcs])
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
    V = u_param.function_space
    mesh = V.mesh
    ufl_domain = mesh.ufl_domain()

    dx = ufl.Measure("dx", domain=ufl_domain)
    dS = ufl.Measure("dS", domain=ufl_domain)

    element_f = ufl.FiniteElement("DG", ufl.triangle, 2)
    element_g = ufl.FiniteElement("DG", ufl.triangle, 1)
    element_e = ufl.FiniteElement("DG", ufl.triangle, 0)
    N = fenicsx_error_estimation.create_interpolation(element_f, element_g)

    V_ef = ufl.FunctionSpace(ufl_domain, element_f)

    V_f = FunctionSpace(mesh, element_f)
    e_f = ufl.TrialFunction(V_f)
    v_f = ufl.TestFunction(V_f)
    f_f = Function(V_f)
    f_f.interpolate(f)

    n = ufl.FacetNormal(ufl_domain)

    # Bilinear form (two constants)
    a_ef = cst_1*inner(grad(e_f), grad(v_f)) * dx + cst_2*inner(e_f, v_f) * dx

    # Linear form (two coefficients, two constants)
    L_ef = inner(f_f + cst_1 * div(grad(u_param)) - cst_2 * u_param, v_f)*dx\
         + inner(cst_1 * jump(grad(u_param), -n), avg(v_f)) * dS

    # Error form (one coefficient)
    V_e = FunctionSpace(mesh, element_e)
    e_h = ufl.Coefficient(V_f)
    v_e = ufl.TestFunction(V_e)
    L_eta = inner(inner(e_h, e_h), v_e) * dx   # L2 estimation
    # Fractional alpha/2 norm estimation
    # L_eta = inner(cst_1*inner(grad(e_h), grad(e_h)) +
    #               cst_2*inner(e_h, e_h), v_e) * dx

    # Function to store result
    eta_h = Function(V_e)       # Norm of Bank-Weiser solution
    e_h_f = Function(V_f)       # Bank-Weiser solution

    # Boundary conditions
    boundary_entities = locate_entities_boundary(
        mesh, 1, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_entities_sorted = np.sort(boundary_entities)

    e_D_f = Function(V_f)
    e_D_f.vector.set(0.)

    fenicsx_error_estimation.estimate(
        eta_h, a_ef, L_ef, L_eta, N, boundary_entities_sorted, e_h=e_h_f, e_D=e_D_f, diagonal=max(1., cst_1.value))

    return eta_h, e_h_f

def mesh_refinement(mesh, sq_local_estimator, global_estimator, theta=0.3):
    """
    Uses Dörfler marking to refine the mesh based on the values of the estimator.
    Parameters:
        mesh: the initial mesh on which the estimator has been computed
        sq_local_estimator: the square root of the local contributions of the estimator
        global_estimtor: the global estimator
        theta: the Dörfler marking parameter (the standard value is theta=0.3)
    Returns:
        mesh: the refined mesh.
    """
    # Dörfler marking
    eta_global = global_estimator**2
    cutoff = theta * eta_global

    assert MPI.COMM_WORLD.size == 1
    sorted_cells = np.argsort(sq_local_estimator.vector.array)[::-1]
    rolling_sum = 0.
    for i, e in enumerate(sq_local_estimator.vector.array[sorted_cells]):
        rolling_sum += e
        if rolling_sum > cutoff:
            breakpoint = i
            break

    refine_cells = sorted_cells[0:breakpoint + 1]
    indices = np.array(np.sort(refine_cells), dtype=np.int32)
    edges = compute_incident_entities(mesh, indices, mesh.topology.dim, mesh.topology.dim - 1)

    # Refine mesh
    mesh.topology.create_entities(mesh.topology.dim - 1)
    mesh = dolfinx.mesh.refine(mesh, edges)

    return mesh

def parametric_refinement(e_frac_f, parametric_bw_solutions, parametric_estimators, refinement_indices, refinement_bool, theta=1.1):
    V_f = e_frac_f.function_space
    min_estimator = min(parametric_estimators)

    i = 0
    for parametric_bw_solution, parametric_estimator in zip(parametric_bw_solutions, parametric_estimators):
        if parametric_estimator >= theta * min_estimator:
            refinement_indices[i] += 1
            refinement_bool = True
            e_h_fine_mesh = Function(V_f)
            e_h_fine_mesh.interpolate(parametric_bw_solution)   # Doesn't work because interpolate does not support inter-mesh interpolation yet.
            e_frac_f.vector.axpy()



if __name__=="__main__":
    workdir = os.getcwd()

    # FE degree
    k = 1

    # Fractional power
    s = 0.5

    # Max number refinement steps
    ref_step_max = 3

    # Rational scheme parameters
    fineness_parameter = 0.3
    rational_parameters, _ = BP_rational_approximation(s, fineness_parameter)

    ls_c_1           = rational_parameters["c_1s"]
    ls_c_2           = rational_parameters["c_2s"]
    ls_weight        = rational_parameters["weights"]
    constant         = rational_parameters["constant"]
    initial_constant = rational_parameters["initial constant"] # The initial constant is zero in the case of BP scheme.

    # List of indices encoding which mesh to use
    refinement_indices = np.zeros_like(ls_c_1).astype(int)

    # Data
    def f(x):
        values = np.ones(x.shape[1])
        values[np.where(np.logical_and(x[0] < 0.5, x[1] > 0.5))] = -1.0
        values[np.where(np.logical_and(x[0] > 0.5, x[1] < 0.5))] = -1.0
        return values

    # Initial mesh
    initial_mesh = create_unit_square(MPI.COMM_WORLD, 8, 8)

    with XDMFFile(MPI.COMM_WORLD, os.path.join(workdir, "meshes", f"mesh_{str(0).zfill(4)}.xdmf"), "w") as of:
        of.write_mesh(initial_mesh)

    for ref_step in range(ref_step_max):
        fractional_output_dir = os.path.join(workdir, "fractional_solutions", f"{str(ref_step).zfill(4)}")
        ls_eta_param = []

        index_max = max(refinement_indices) # NOTE: here index_max could be replaced with ref_step.
        with XDMFFile(MPI.COMM_WORLD, os.path.join(workdir, "meshes", f"mesh_{str(index_max).zfill(4)}.xdmf"), "r") as fi:
            finest_mesh = fi.read_mesh()

        element_f = ufl.FiniteElement("DG", ufl.triangle, 2)
        V_f       = FunctionSpace(finest_mesh, element_f)
        e_frac_f  = Function(V_f)
        refinement_bool = False
        parametric_estimators = []
        parametric_bw_solutions = []
        for c_1, c_2, weight, refinement_index, pbm_num in zip(ls_c_1, ls_c_2, ls_weight, refinement_indices, range(len(ls_c_1))):
            # Output dirs
            meshes_dir = os.path.join(workdir, "meshes")
            param_pbm_output_dir = os.path.join(workdir, "parametric_problems", f"{str(pbm_num).zfill(4)}", f"refinement_{str(ref_step).zfill(4)}")

            # Read the mesh corresponding to the refinement_index
            with XDMFFile(MPI.COMM_WORLD, os.path.join(meshes_dir, f"mesh_{str(refinement_index).zfill(4)}.xdmf"), "r") as fi:
                mesh = fi.read_mesh()
            
            # Initialize parameters of the parametric problem
            cst_1 = Constant(mesh, c_1)
            cst_2 = Constant(mesh, c_2)

            # Solve the parametric problem
            u_param = solve_parametric(mesh, cst_1, cst_2)
            with XDMFFile(mesh.comm, os.path.join(param_pbm_output_dir, f"u_{str(ref_step).zfill(4)}.xdmf"), "w") as of:
                of.write_mesh(mesh)
                of.write_function(u_param)

            # Estimate the parametric L2 error
            eta_param, e_h_f = estimate_parametric(cst_1, cst_2, u_param)
            with XDMFFile(mesh.comm, os.path.join(param_pbm_output_dir, f"eta_{str(ref_step).zfill(4)}.xdmf"), "w") as of:
                of.write_mesh(mesh)
                of.write_function(eta_param)
            with XDMFFile(mesh.comm, os.path.join(param_pbm_output_dir, f"e_h_f_{str(ref_step).zfill(4)}.xdmf"), "w") as of:
                of.write_mesh(mesh)
                of.write_function(e_h_f)
        
            # Computation fractional estimator
            if (refinement_index == ref_step):  # If the refinement_index is maximal (i.e. e_h_f has been computed on the finest mesh), then we keep e_h_f as it is
                e_frac_f.vector.axpy(weight, e_h_f.vector)
            # else:                               # If not, we need to interpolate e_h_f to the finest mesh
                # e_h_f_int = interpolate(e_h_f) # TODO: add inter-mesh interpolation.
                # e_frac_f.vector.axpy(weight, e_h_f_int.vector)

            parametric_estimators.append(np.sqrt(eta_param.vector.sum()))
            parametric_bw_solutions.append(e_h_f)

        parametric_refinement(e_frac_f, parametric_bw_solutions, parametric_estimators, refinement_indices, refinement_bool)

        # Fractional BW solution
        e_h_f.vector.scale(constant)

        # Fractional error estimator


        with XDMFFile(finest_mesh.comm, os.path.join(fractional_output_dir, "e_h_f.xdmf"), "w") as of:
            of.write_mesh(finest_mesh)
            of.write_function(e_h_f)