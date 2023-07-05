# Copyright 2020, Jack S. Hale, Raphael Bulle.
# SPDX-License-Identifier: LGPL-3.0-or-later

import fenicsx_error_estimation
import numpy as np

import dolfinx
import ufl
from dolfinx.fem import (Constant,
                         Function,
                         FunctionSpace,
                         apply_lifting,
                         dirichletbc,
                         form,
                         locate_dofs_topological,
                         set_bc)
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from dolfinx.io import XDMFFile
from dolfinx.mesh import (CellType,
                          compute_incident_entities,
                          create_unit_square,
                          locate_entities_boundary)
from ufl import (Coefficient,
                 Measure,
                 TestFunction,
                 TrialFunction,
                 avg,
                 div,
                 grad,
                 inner,
                 jump)

from mpi4py import MPI
from petsc4py import PETSc
import pandas as pd


import os
import shutil
import sys
sys.path.append("../")

from rational_schemes import BP_rational_approximation, BURA_rational_approximation
from FE_utils import mesh_refinement, parametric_problem


"""
========================================================================
solve_parametric:   Solves the finite element parametric problem.

\param[in] mesh     The FE mesh on which the problem is solved.
\param[in] c_diff   Float, the diffusion coefficient.
\param[in] c_react  Float, the reaction coefficient.

return     u_param  The FE solution of the parametric problem.
========================================================================
"""
def solve_parametric(fe_degree, source_data, mesh, c_diff, c_react):
    assert (c_diff > 0.) or (c_react > 0.)

    # Measure, FE space and trial/test functions definitions
    dx = Measure("dx", domain=mesh)
    element = ufl.FiniteElement("CG", mesh.ufl_cell(), fe_degree)
    V = FunctionSpace(mesh, element)

    dof_num = V.dofmap.index_map.size_global

    u = TrialFunction(V)
    v = TestFunction(V)

    # Interpolation of data
    f_V = Function(V)
    f_V.interpolate(source_data)

    # Reaction and diffusion coefficients
    # cst_diff  = Coefficient(V, c_diff)
    # cst_react = Coefficient(V, c_react)

    # Bilinear and linear forms
    a_form = form(c_diff * inner(grad(u), grad(v)) * dx
                + c_react * inner(u, v) * dx)
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
    
    return u_param, dof_num


"""
========================================================================
estimate_parametric:    Estimate the discretization error for a
                        reaction-diffusion problem.

\param[in]  cst_1       The diffusion coefficient.
\param[in]  cst_2       The reaction coefficient.
\param[in]  u_param     The FE solution of the problem.

return      eta_h       DG0 FE function stocking the values of the local
                        estimators.
            e_h_f       CG V_f FE function stocking the local BW solutions.
========================================================================
"""
def estimate_parametric(source_data, c_diff, c_react, u_param):
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
    f_f.interpolate(source_data)

    n = ufl.FacetNormal(ufl_domain)

    # Bilinear form (two constants)
    a_ef = c_diff*inner(grad(e_f), grad(v_f)) * dx + c_react*inner(e_f, v_f) * dx

    # Linear form (two coefficients, two constants)
    L_ef = inner(f_f + c_diff * div(grad(u_param)) - c_react * u_param, v_f)*dx\
         + inner(c_diff * jump(grad(u_param), -n), avg(v_f)) * dS

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
        eta_h, a_ef, L_ef, L_eta, N, boundary_entities_sorted, e_h=e_h_f, e_D=e_D_f, diagonal=max(1., c_diff))

    return eta_h, e_h_f


"""
========================================================================
dorfler_marking:    Mark the meshes with respect to the given parametric
estimators using a Dörfler-like algorithm.

\param[in]  ls_eta_param_vects       List of lists, the weighted
                                     squared values of the local estimators.
\param[in]  rational_constant        Float, multiplicative constant in front of
                                     the rational sum.
\param[in]  theta                    Float, Dörfler's algorithm parameter
                                     (in (0,1)).

returns     unsorted_marked_cells    List of lists, unsorted lists of marked
                                     cells for each parametric mesh.
========================================================================
"""
def dorfler_marking(ls_eta_param_vects, num_param_pbms, rational_constant, theta=0.35):
    # Compute the global estimator value
    global_est = compute_est(ls_eta_param_vects, rational_constant)
    cutoff = theta * global_est

    # Compute and concatenate the indices arrays and estimator values array
    param_indices = []
    cells_indices = []
    for i, est in enumerate(ls_eta_param_vects):
        param_indices.append([i for _ in range(len(est))])
        cells_indices.append(list(range(len(est))))
    
    param_ind_concat        = np.concatenate(param_indices)
    cells_ind_concat        = np.concatenate(cells_indices)
    eta_param_vects_concat  = np.concatenate(ls_eta_param_vects)

    # Store the concatenated arrays in a single Nx3 numpy array
    param_cell_est      = np.zeros((len(eta_param_vects_concat), 3))
    param_cell_est[:,0] = param_ind_concat
    param_cell_est[:,1] = cells_ind_concat
    param_cell_est[:,2] = eta_param_vects_concat

    # Doesn't work in parallel
    assert MPI.COMM_WORLD.size == 1

    # Sort the estimators values in decreasing order and sorting
    # the indices accordingly
    sorted_ind = np.argsort(param_cell_est[:,2])[::-1]
    param_cell_est = param_cell_est[sorted_ind]

    sorted_param_indices    = param_cell_est[:,0].astype(np.intc)
    sorted_cells_indices    = param_cell_est[:,1].astype(np.intc)
    sorted_weighted_sq_est  = param_cell_est[:,2]

    # List of lists of marked meshes cells (each list corresponds to a parametric problem)
    selected_est_values     = [[] for _ in range(num_param_pbms)]
    unsorted_marked_cells   = [[] for _ in range(num_param_pbms)]

    rolling_sum = 0.
    for i, cell_index, param_index, est in zip(range(len(sorted_param_indices)), #
                                               sorted_cells_indices, #
                                               sorted_param_indices, #
                                               sorted_weighted_sq_est):
        printlog(f"\t \t Dörfler loop step {i} / {len(sorted_param_indices)}")

        selected_est_values[param_index].append(est)
        partial_est = compute_est(selected_est_values, rational_constant)
        unsorted_marked_cells[param_index].append(cell_index)
        if partial_est > cutoff:
            breakpoint = i
            break
    
    return unsorted_marked_cells

"""
========================================================================
maximum_marking:    Mark the meshes with respect to the given parametric
estimators using a maximum strategy

\param[in]  ls_eta_param_vects       List of lists, the weighted
                                     squared values of the local estimators.
\param[in]  rational_constant        Float, multiplicative constant in front of
                                     the rational sum.
\param[in]  theta                    Float, threshold for marking
                                     (in (0,1)).

returns     unsorted_marked_cells    List of lists, unsorted lists of marked
                                     cells for each parametric mesh.
========================================================================
"""
def maximum_marking(ls_eta_param_vects, num_param_pbms, rational_constant, theta=0.8):

    # Compute and concatenate the indices arrays and estimator values array
    param_indices = []
    cells_indices = []
    for i, est in enumerate(ls_eta_param_vects):
        param_indices.append([i for _ in range(len(est))])
        cells_indices.append(list(range(len(est))))
    
    param_ind_concat        = np.concatenate(param_indices)
    cells_ind_concat        = np.concatenate(cells_indices)
    eta_param_vects_concat  = np.concatenate(ls_eta_param_vects)

    # Store the concatenated arrays in a single Nx3 numpy array
    param_cell_est      = np.zeros((len(eta_param_vects_concat), 3))
    param_cell_est[:,0] = param_ind_concat
    param_cell_est[:,1] = cells_ind_concat
    param_cell_est[:,2] = eta_param_vects_concat

    # Doesn't work in parallel
    assert MPI.COMM_WORLD.size == 1

    # Sort the estimators values in decreasing order and sorting
    # the indices accordingly
    sorted_ind = np.argsort(param_cell_est[:,2])[::-1]
    param_cell_est = param_cell_est[sorted_ind]

    # Compute the global estimator value
    max_est = np.max(param_cell_est[:,2])
    cutoff = theta * max_est

    sorted_param_indices    = param_cell_est[:,0].astype(np.intc)
    sorted_cells_indices    = param_cell_est[:,1].astype(np.intc)
    sorted_weighted_sq_est  = param_cell_est[:,2]

    # List of lists of marked meshes cells (each list corresponds to a parametric problem)
    selected_est_values     = [[] for _ in range(num_param_pbms)]
    unsorted_marked_cells   = [[] for _ in range(num_param_pbms)]

    rolling_sum = 0.
    for i, cell_index, param_index, est in zip(range(len(sorted_param_indices)), #
                                               sorted_cells_indices, #
                                               sorted_param_indices, #
                                               sorted_weighted_sq_est):
        printlog(f"\t \t Dörfler loop step {i} / {len(sorted_param_indices)}")

        selected_est_values[param_index].append(est)
        unsorted_marked_cells[param_index].append(cell_index)
        if est < cutoff:
            breakpoint = i
            break
    
    return unsorted_marked_cells


"""
========================================================================
compute_est: Compute the fractional estimator based on a selection of local
             estimator values for each parametric problem. Used in the marking
             strategy.

\param[in]  ls_eta_param_vects  List of lists, the weighted
                                squared values of the local estimators.
\param[in]  rational_constant   Float, multiplicative constant in front of
                                the rational sum.

returns     fractional_est      Value of the partial fractional estimator.
========================================================================
"""
def compute_est(ls_eta_param_vects, rational_constant):
    sums_param = [sum(eta_param_vect) for eta_param_vect in ls_eta_param_vects]
    fractional_est = rational_constant * np.sqrt(sum(sums_param))
    return fractional_est


"""
========================================================================
meshes_refinement:  Adaptively refine the meshes based on a list of marked
                    cells for each parametric mesh.

\param[in,out]  meshes                  List, input the list of meshes,
                                        output the list of adaptively
                                        refined meshes.
\param[in,out]  mesh_ref_bools          List of booleans, True if the mesh
                                        was refined, False else.
\param[in]      unsorted_marked_cells   List of lists, unsorted lists of marked
                                        cells for each mesh.
========================================================================
"""
def mesh_refinement(meshes, mesh_ref_bools, unsorted_marked_cells):
    for i, unsorted_cells in enumerate(unsorted_marked_cells):
        if not len(unsorted_cells) == 0:
            printlog(f"\t \t Refinement mesh {i}")
            mesh_ref_bools[i] = True
            indices = np.array(np.sort(unsorted_cells), dtype=np.int32)
            edges = compute_incident_entities(meshes[i], #
                                            indices, #
                                            meshes[i].topology.dim, #
                                            meshes[i].topology.dim - 1)

            meshes[i].topology.create_entities(meshes[i].topology.dim - 1)
            meshes[i] = dolfinx.mesh.refine(meshes[i], edges)
        else:
            mesh_ref_bools[i] = False
    return meshes, mesh_ref_bools


"""
========================================================================
refinement_loop:  Refinement loop.

\param[in]      source_data             Python function, the reaction-diffusion
                                        PDEs source data.
\param[in]      rational_parameters     Dictionnary, rational scheme 
                                        parameters.
\param[in]      fe_deg                  Integer, finite element method degree.
\param[in]      ref_step_max            Integer, number of mesh 
                                        refinement steps.
========================================================================
"""
def refinement_loop(source_data, #
                    rational_parameters, #
                    fe_degree, #
                    ref_step_max=3, #
                    theta=0.35, #
                    param_pbm_dir="./"):
    workdir = os.getcwd()

    # Unpack the rational parameters
    ls_c_diff           = rational_parameters["c_1s"]
    ls_c_react          = rational_parameters["c_2s"]
    ls_weight           = rational_parameters["weights"]
    constant            = rational_parameters["constant"]
    initial_constant    = rational_parameters["initial constant"] # The initial constant is zero in the case of BP scheme.

    # Number of parametric problems
    num_param_pbms = len(ls_c_diff)
    printlog(f"Num. parametric problems to solve: {num_param_pbms}")

    # Initialize list of booleans
    mesh_ref_bools = np.ones(num_param_pbms).astype(bool)
    
    # Fill the list of meshes with inital 8x8 meshes
    printlog("Meshes initialization")
    meshes = []
    for i in range(num_param_pbms):
        mesh = create_unit_square(MPI.COMM_WORLD, 4, 4)
        meshes.append(mesh)

    # Refinement loop
    meshes_bools_history                    = np.zeros((ref_step_max, num_param_pbms)).astype(int)
    global_weighted_parametric_est_history  = np.zeros((ref_step_max, num_param_pbms))
    global_parametric_est_history           = np.zeros((ref_step_max, num_param_pbms))
    ls_frac_global_est                      = np.zeros(ref_step_max)
    ls_total_dof_num                        = np.zeros(ref_step_max)

    printlog("Entering refinement loop")
    for ref_step in range(ref_step_max):
        # List of local estimators
        ls_eta_param_vects = []

        # List of global parametric weighted estimators
        ls_global_weighted_eta_param    = np.zeros(num_param_pbms)
        ls_global_eta_param             = np.zeros(num_param_pbms)
        total_dof_num = 0

        # Parametric problems loop
        for mesh, mesh_ref_bool, c_diff, c_react, weight, pbm_num in zip(meshes, #
                                                                         mesh_ref_bools, #
                                                                         ls_c_diff, #
                                                                         ls_c_react, #
                                                                         ls_weight, #
                                                                         range(num_param_pbms)):                  
            printlog(f"\t Refinement step: {ref_step}, problem: {pbm_num}/{num_param_pbms}")

            # Solve parametric problem
            printlog(f"\t Solve")
            u_param, dof_num = solve_parametric(fe_degree, source_data, mesh, c_diff, c_react)
            total_dof_num += dof_num

            # Estimate the parametric L2 error
            printlog(f"\t Estimate")
            eta_param, e_h_f = estimate_parametric(source_data, c_diff, c_react, u_param)
            
            sq_eta_param_vect           = (eta_param.vector.array)**2
            weighted_sq_eta_param_vect  = sq_eta_param_vect * weight**2

            ls_eta_param_vects.append(weighted_sq_eta_param_vect)
            ls_global_weighted_eta_param[pbm_num] = np.sqrt(sum(weighted_sq_eta_param_vect))
            ls_global_eta_param[pbm_num] = np.sqrt(sum(sq_eta_param_vect))

        for mesh, mesh_ref_bool, i in zip(meshes, mesh_ref_bools, range(len(meshes))):
            param_pbm_subdir = f"{str(i).zfill(4)}"
            # Save the mesh
            with XDMFFile(MPI.COMM_WORLD, os.path.join(param_pbm_dir, param_pbm_subdir, "meshes", f"mesh_{str(ref_step).zfill(4)}.xdmf"), "w") as of:
                of.write_mesh(mesh)
            # Save the parametric solutions
            with XDMFFile(mesh.comm, os.path.join(param_pbm_dir, param_pbm_subdir, "solutions", f"solution_{str(ref_step).zfill(4)}.xdmf"), "w") as of:
                of.write_mesh(mesh)
                of.write_function(u_param)
            # Save the local estimators and BW solutions
            with XDMFFile(mesh.comm, os.path.join(param_pbm_dir, param_pbm_subdir, "estimators", f"estimator_{str(ref_step).zfill(4)}.xdmf"), "w") as of:
                of.write_mesh(mesh)
                of.write_function(eta_param)
            with XDMFFile(mesh.comm, os.path.join(param_pbm_dir, param_pbm_subdir, "local_bw_solutions", f"local_bw_solution_{str(ref_step).zfill(4)}.xdmf"), "w") as of:
                of.write_mesh(mesh)
                of.write_function(e_h_f)

        # Marking
        printlog('\t Dörfler marking')
        unsorted_marked_cells = dorfler_marking(ls_eta_param_vects, num_param_pbms, constant, theta=theta)

        #printlog('\t Maximum marking')
        #unsorted_marked_cells = maximum_marking(ls_eta_param_vects, num_param_pbms, constant, theta=theta)

        # Refinement
        printlog("\t Refinement")
        meshes, mesh_ref_bools = mesh_refinement(meshes, mesh_ref_bools, unsorted_marked_cells)
    
        # Store infos
        global_parametric_est_history[ref_step,:]            = ls_global_eta_param
        global_weighted_parametric_est_history[ref_step,:]   = ls_global_weighted_eta_param
        meshes_bools_history[ref_step,:]                     = mesh_ref_bools
    
        frac_global_est = constant * sum(ls_global_weighted_eta_param)
        ls_frac_global_est[ref_step] = frac_global_est
        ls_total_dof_num[ref_step] = total_dof_num

    return meshes_bools_history, global_weighted_parametric_est_history, global_parametric_est_history, ls_frac_global_est, ls_total_dof_num

def printlog(log_line):
    with open("log.txt", "a+") as logf:
        logf.write(f"{log_line}\n")

if __name__ == "__main__":
    # Fractional power
    s = 0.7
    # Rational scheme fineness parameter
    fineness_ra_sch = 0.35
    # FEM degree
    fe_degree = 1
    # Maximum number of refinement steps
    ref_step_max = 40
    # Dörfler marking parameter
    theta = 0.5
    # Maximum marking parameter
    # theta = 0.8

    workdir = os.getcwd()
    param_pbm_dir = os.path.join(workdir, f"parametric_problems_{str(int(10*s))}")

    with open("log.txt", 'w') as logf:
        logf.write("Multi-mesh refinement algorithm log.\n")
    
    # Empty results directory
    if os.path.isdir(param_pbm_dir):
        shutil.rmtree(param_pbm_dir)
        os.makedirs(param_pbm_dir)
    else:
        os.makedirs(param_pbm_dir)

    # Data
    def source_data(x):
        values = np.ones(x.shape[1])
        values[np.where(np.logical_and(x[0] < 0.5, x[1] > 0.5))] = -1.0
        values[np.where(np.logical_and(x[0] > 0.5, x[1] < 0.5))] = -1.0
        return values
    
    # Compute the rational parameters
    printlog(f"Computation rational parameters, fineness param = {fineness_ra_sch}")
    rational_parameters, _ = BP_rational_approximation(s, fineness_ra_sch)
    meshes_bools_history, global_weighted_parametric_est_history, global_parametric_est_history, ls_frac_global_est, ls_total_dof_num = refinement_loop(source_data, rational_parameters, fe_degree, ref_step_max=ref_step_max, theta=theta, param_pbm_dir=param_pbm_dir)

    meshes_num_refinement = np.sum(meshes_bools_history.astype(int), axis=0)

    ls_c_diff       = rational_parameters["c_1s"][:]
    ls_c_react      = rational_parameters["c_2s"][:]
    ls_weights      = rational_parameters["weights"][:]
    param_pbm_nums  = np.arange(len(ls_c_diff))

    results_dir_str = f"results_frac_pw_{str(int(10*s))}"
    if os.path.isdir(results_dir_str):
        shutil.rmtree(results_dir_str)
        os.makedirs(results_dir_str)
    else:
        os.makedirs(results_dir_str)
    np.save(os.path.join(results_dir_str, "meshes_num_refinement.npy"),                    meshes_num_refinement)
    np.save(os.path.join(results_dir_str, "global_weighted_parametric_est_history.npy"),   global_weighted_parametric_est_history)
    np.save(os.path.join(results_dir_str, "global_parametric_est_history.npy"),            global_parametric_est_history)
    np.save(os.path.join(results_dir_str, "ls_c_diff.npy"),                                ls_c_diff)
    np.save(os.path.join(results_dir_str, "ls_c_react.npy"),                               ls_c_react)
    np.save(os.path.join(results_dir_str, "ls_weights.npy"),                               ls_weights)
    np.save(os.path.join(results_dir_str, "frac_global_est.npy"),                          ls_frac_global_est)
    np.save(os.path.join(results_dir_str, "total_dof_num.npy"),                            ls_total_dof_num)