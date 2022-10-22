# Copyright 2020, Jack S. Hale, Raphael Bulle.
# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np

import dolfinx
import ufl
from dolfinx.fem import (Constant, Function, FunctionSpace, apply_lifting,
                         dirichletbc, form, locate_dofs_topological, set_bc, assemble_scalar)
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


def main():
    output_dir = "output/" + method + f"_{str(s)[-1]}/"
    
    # Structured mesh
    mesh = create_unit_square(MPI.COMM_WORLD, 4, 4)
    mesh.geometry.x[:] *= np.pi

    # Data
    def f(x):
        values = np.sin(x[0])*np.sin(x[1])
        return values

    # Analytical solution
    def u_exact(x):
        values = 2.**(-s) * np.sin(x[0]) * np.sin(x[1])
        return values

    if method == "bp":
        parameter = 1.
    elif method == "bura":
        parameter = 1

    rational_tol = np.Inf
    while(rational_tol > r_tol):
        if method == "bp":
            parameter -= 0.01
        elif method == "bura":
            parameter += 1

        rational_parameters, rational_tol = rational_approximation(parameter, s)

    df_rational_parameters = pd.DataFrame(rational_parameters)
    df_rational_parameters.to_csv(output_dir + "rational_parameters.csv")
    # Results storage
    results = {"dof num": [], "rational parameter": [], "num solves": [], "rational error": [], "FE error": [], "total error": [], "sqrt(FE^2 + rat errors^2)": [], "L2 bw": [], "rational error": []}

    ref_step = 0
    bw_global_estimator = np.Inf
    while bw_global_estimator > FE_tol:
        dx = Measure("dx", domain=mesh)

        # Finite element spaces and functions
        element = ufl.FiniteElement("CG", mesh.ufl_cell(), k)
        element_f = ufl.FiniteElement("CG", mesh.ufl_cell(), k+1)
        V = FunctionSpace(mesh, element)
        V_f = FunctionSpace(mesh, element_f)
        u = TrialFunction(V)
        v = TestFunction(V)

        # Interpolate input data into a FE function
        f_V_f = Function(V_f)
        f_V_f.interpolate(f)

        with XDMFFile(mesh.comm, output_dir + f"/f_{str(ref_step).zfill(3)}.xdmf",
                "w") as of:
            of.write_mesh(mesh)
            of.write_function(f_V_f)

        # Initialize bilinear and linear form of parametric problem
        cst_1 = Constant(mesh, 0.)
        cst_2 = Constant(mesh, 0.)

        a_form = form(cst_1 * inner(grad(u), grad(v)) * dx
                    + cst_2 * inner(u, v) * dx)
        L_form = form(inner(f_V_f, v) * dx)

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
        u_h, estimators = parametric_problem(f_V_f, V, k, rational_parameters, bcs,
                                             boundary_entities_sorted, a_form,
                                             L_form, cst_1, cst_2,
                                             ref_step=ref_step,
                                             output_dir=output_dir,
                                             sines_test_case=True)

        # Compute the L2 projection of f onto V (only necessary for BURA)
        if method == "bura":
            u0 = Function(V)
            u0.vector.set(0.)
            facets = locate_entities_boundary(
                    mesh, 1, lambda x: np.ones(x.shape[1], dtype=bool))
            dofs = locate_dofs_topological(V, 1, facets)
            bcs = [dirichletbc(u0, dofs)]

            u = TrialFunction(V)
            v = TestFunction(V)
            f_l2_V = Function(V)
            a_V = form(inner(u, v)*dx)
            A_V = assemble_matrix(a_V)
            A_V.assemble()
            L_V = form(inner(f_V_f, v) * dx)
            b_V = assemble_vector(L_V)

            set_bc(b_V, bcs)

            # Linear system solve
            options = PETSc.Options()
            options["ksp_type"] = "cg"
            options["pc_type"] = "hypre"
            options["ksp_rtol"] = 1e-7
            options["pc_hypre_type"] = "boomeramg"

            solver = PETSc.KSP().create(MPI.COMM_WORLD)
            solver.setOperators(A_V)
            solver.setFromOptions()
            solver.solve(b_V, f_l2_V.vector)

            u_h.vector.array += rational_parameters["initial constant"] * f_l2_V.vector.array

        # Semi-discrete approximation
        def u_Q(x):
            c_1s = rational_parameters["c_1s"]
            c_2s = rational_parameters["c_2s"]
            weights = rational_parameters["weights"]
            constant = rational_parameters["constant"]
            initial_constant = rational_parameters["initial constant"]

            parametric_coefs = [weight * (1./(c_2 + 2.*c_1)) for weight, c_1, c_2 in zip(weights, c_1s, c_2s)]
            sum_parametric = sum(parametric_coefs)
            coef = initial_constant + constant * sum_parametric

            return coef * np.sin(x[0]) * np.sin(x[1])

        # Exact errors
        element_f = ufl.FiniteElement("CG", mesh.ufl_cell(), k+1)
        V_f = FunctionSpace(mesh, element_f)
        u_V_f = Function(V_f)
        u_V_f.interpolate(u_exact)
        u_Q_V_f = Function(V_f)
        u_Q_V_f.interpolate(u_Q)

        with XDMFFile(mesh.comm, output_dir + f"/u_Q_{str(ref_step).zfill(3)}.xdmf", "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(u_Q_V_f)

        u_h_V_f = Function(V_f)
        u_h_V_f.interpolate(u_h)

        # Exact rational error
        e_rational = u_V_f - u_Q_V_f
        err_2 = assemble_scalar(form(inner(e_rational, e_rational) * dx))
        rational_err = np.sqrt(err_2)

        # Exact FE error
        e_FE = u_Q_V_f - u_h_V_f
        err_2 = assemble_scalar(form(inner(e_FE, e_FE) * dx))
        FE_err = np.sqrt(err_2)
        
        # Exact total error
        e_total = u_V_f - u_h_V_f
        err_2 = assemble_scalar(form(inner(e_total, e_total) * dx))
        total_err = np.sqrt(err_2)

        # Estimator steering the refinement
        bw_sq_local_estimator = estimators["L2 squared local BW"]
        bw_global_estimator = estimators["L2 global BW"]

        with XDMFFile(mesh.comm, output_dir + f"/u_{str(ref_step).zfill(3)}.xdmf",
                "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(u_h)

        with XDMFFile(mesh.comm, output_dir + f"/u_exact_{str(ref_step).zfill(3)}.xdmf",
        "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(u_V_f)

        with XDMFFile(mesh.comm, output_dir + f"/l2_bw_{str(ref_step).zfill(3)}.xdmf",
                "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(bw_sq_local_estimator)

        mesh = dolfinx.mesh.refine(mesh)    # This test case doesn't require adaptive refinement
        ref_step += 1

        results["dof num"].append(V.dofmap.index_map.size_global)
        results["total error"].append(total_err)
        results["L2 bw"].append(bw_global_estimator)
        results["rational error"].append(rational_err)
        results["FE error"].append(FE_err)
        results["rational parameter"].append(parameter)
        results["num solves"].append(len(rational_parameters["c_1s"]))
        results["sqrt(FE^2 + rat errors^2)"].append(np.sqrt(rational_err**2 + FE_err**2))

        df = pd.DataFrame(results)
        df.to_csv(f"results_{method}_{str(s)[-1]}.csv")
        print(df)


if __name__ == "__main__":
    # Finite element degree
    k = 1

    # Rational approximation tolerance
    r_tol = 1.e-8

    # Number adaptive refinements
    FE_tol = 1.e-3

    # Dorfler marking parameter
    theta = 0.3

    sines_test_case=True

    for method in ["bp", "bura"]:
        for s in [0.3, 0.5, 0.7]:
            if method == "bp":
                # parameter = 0.35    # Fineness parameter (in (0., 1.), more precise if close to 0.)
                # For coarse scheme
                # parameter = 2.5   # (3 solves, error ~ 0.06)
                rational_approximation = BP_rational_approximation

            elif method == "bura":
                # parameter = 10      # Degree of the rational approximation (integer, more precise if large)
                # For coarse scheme
                # parameter = 3     # (3 solves, error ~ 0.005)
                rational_approximation = BURA_rational_approximation

            main()