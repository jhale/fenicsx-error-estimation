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
import shutil
import sys
sys.path.append("../")

from rational_schemes import BP_rational_approximation, BURA_rational_approximation
from FE_utils import mesh_refinement, parametric_problem


def main(k, tol, ra_tol, theta, mesh, f, parameter):
    L2_norm_f = 1.    # ||f||_{L2} = 1 in this case

    if FE_adaptive:
        if rational_adaptive:
            output_dir = "output/" + method + "_FE_adaptive" + "_rational_adaptive" + f"_{str(s)[-1]}/"
        else:
            output_dir = "output/" + method + "_FE_adaptive" + f"_{str(s)[-1]}/"
    else:
        output_dir = "output/" + method + f"_{str(s)[-1]}/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)

    if not rational_adaptive:
        rational_error = np.Inf
        while(rational_error > ra_tol):
            if method == "bp":
                parameter -= 0.01
            elif method == "bura":
                parameter += 1

            rational_parameters, rational_scalar_estimator = rational_approximation(parameter, s)
    else:
        rational_parameters, rational_scalar_estimator = rational_approximation(parameter, s)

    rational_estimator = rational_scalar_estimator * L2_norm_f

    # Results storage
    results = {"dof num": [], "rational parameter": [], "num solves": [],  "L2 bw": [], "rational estimator": [], "total estimator": []}
    total_num_solves = 0
    total_estimator = np.Inf
    ref_step = 0
    while total_estimator > tol:
        dx = Measure("dx", domain=mesh)

        # Finite element spaces and functions
        element = ufl.FiniteElement("CG", mesh.ufl_cell(), k)
        element_e = ufl.FiniteElement("DG", mesh.ufl_cell(), 0)
        V = FunctionSpace(mesh, element)
        V_e = FunctionSpace(mesh, element_e)
        u = TrialFunction(V)
        v = TestFunction(V)

        # Interpolate input data into a FE function
        f_e = Function(V_e)
        f_e.interpolate(f)

        with XDMFFile(mesh.comm, output_dir + f"f_{str(ref_step).zfill(3)}.xdmf",
                "w") as of:
            of.write_mesh(mesh)
            of.write_function(f_e)

        # Initialize bilinear and linear form of parametric problem
        cst_1 = Constant(mesh, 0.)
        cst_2 = Constant(mesh, 0.)

        a_form = form(cst_1 * inner(grad(u), grad(v)) * dx
                    + cst_2 * inner(u, v) * dx)
        L_form = form(inner(f_e, v) * dx)

        # Homogeneous zero Dirichlet boundary condition
        u0 = Function(V)
        u0.vector.set(0.)
        facets = locate_entities_boundary(
                mesh, 1, lambda x: np.ones(x.shape[1], dtype=bool))
        dofs = locate_dofs_topological(V, 1, facets)
        bcs = [dirichletbc(u0, dofs)]

        boundary_entities = locate_entities_boundary(
                mesh, 1, lambda x: np.ones(x.shape[1], dtype=bool))
        boundary_entities_sorted = np.sort(boundary_entities)

        # Fractional solution
        print(f"Method: {method}, FE adaptive: {FE_adaptive}, rational adaptive: {rational_adaptive}, s: {s}, Ref step: {ref_step}")
        u_h, estimators = parametric_problem(f_e, V, k, rational_parameters, bcs,
                                             boundary_entities_sorted, a_form,
                                             L_form, cst_1, cst_2,
                                             ref_step=ref_step, output_dir=output_dir)

        # FE estimator
        bw_sq_local_estimator = estimators["L2 squared local BW"]
        bw_global_estimator = estimators["L2 global BW"]

        df_rational_parameters = pd.DataFrame(rational_parameters)
        df_rational_parameters.to_csv(output_dir + f"rational_parameters_{ref_step}.csv")

        # Total estimator
        total_estimator = bw_global_estimator + rational_estimator

        with XDMFFile(mesh.comm, output_dir + f"u_{str(ref_step).zfill(3)}.xdmf",
                "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(u_h)

        with XDMFFile(mesh.comm, output_dir + f"l2_bw_{str(ref_step).zfill(3)}.xdmf",
                "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(bw_sq_local_estimator)

        if FE_adaptive:
            mesh = mesh_refinement(mesh, bw_sq_local_estimator,  bw_global_estimator, theta)
        else:
            mesh = dolfinx.mesh.refine(mesh)
    
        rational_error = np.Inf
        if rational_adaptive:
            # Rational scheme refinement
            if ref_step <= 1:
                coef = 1.
            else:
                coef = results["L2 bw"][-1]/results["L2 bw"][-2]

            while(rational_estimator > coef * bw_global_estimator):
                if method == "bura":
                    parameter += 1
                elif method == "bp":
                    parameter -= 0.01
                rational_parameters, rational_scalar_estimator = rational_approximation(parameter, s)
                rational_estimator = rational_scalar_estimator * L2_norm_f

        results["dof num"].append(V.dofmap.index_map.size_global)
        results["L2 bw"].append(bw_global_estimator)
        results["rational estimator"].append(rational_estimator)
        results["rational parameter"].append(parameter)
        results["total estimator"].append(total_estimator)
        results["num solves"].append(len(rational_parameters["c_1s"]))

        df = pd.DataFrame(results)
        if FE_adaptive:
            if rational_adaptive:
                df.to_csv(f"results/results_{method}_{str(s)[-1]}_FE_adaptive_rational_adaptive.csv")
            else:
                df.to_csv(f"results/results_{method}_{str(s)[-1]}_FE_adaptive.csv")
        else:
            df.to_csv(f"results/results_{method}_{str(s)[-1]}.csv")
        print(df)

        ref_step += 1

if __name__ == "__main__":   
    # Finite element degree
    k = 1

    # Dorfler marking parameter
    theta = 0.3

    # Structured mesh
    mesh = create_unit_square(MPI.COMM_WORLD, 8, 8)

    # Data
    def f(x):
        values = np.ones(x.shape[1])
        values[np.where(np.logical_and(x[0] < 0.5, x[1] > 0.5))] = -1.0
        values[np.where(np.logical_and(x[0] > 0.5, x[1] < 0.5))] = -1.0
        return values

    for FE_adaptive in [True]:
        if FE_adaptive:
            rational_adaptive_choices = [True]
        else:
            rational_adaptive_choices = [False, True]
        for rational_adaptive in rational_adaptive_choices: #rational_adaptive_choices:
            for method in ["bp", "bura"]:
                for s in [0.1, 0.3, 0.5, 0.7, 0.9]:
                    if s==0.1:
                        tol = 1.e-2

                        ra_tol = tol/10.
                    if s==0.3:
                        tol = 1.e-3

                        ra_tol = tol/10.
                    if s>0.3:
                        tol = 1.e-4

                        ra_tol = tol/10.
                    if method == "bp":
                        parameter = 3.    # Fineness parameter (in (0., 1.), more precise if close to 0.)
                        # For coarse scheme
                        # parameter = 2.5   # (3 solves, error ~ 0.06)
                        rational_approximation = BP_rational_approximation

                    elif method == "bura":
                        parameter = 1      # Degree of the rational approximation (integer, more precise if large)
                        # For coarse scheme
                        # parameter = 3     # (3 solves, error ~ 0.005)
                        rational_approximation = BURA_rational_approximation

                    main(k, tol, ra_tol, theta, mesh, f, parameter)