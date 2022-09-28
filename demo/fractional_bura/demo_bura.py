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

method = "bura"   # methods "bp" or "bura"

def mesh_refinement(mesh, sq_local_estimator, global_estimator, theta):
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


def rational_approximation(lmbda, kappa, s):
    """
    Generates the parameters for the rational sum according to exponentially
    convergent scheme in Bonito and Pasciak 2013.

    Parameters:
        lmbda: eigenvalue at which the sum is evaluated
        kappa: fineness parameter
        s: fractional power

    Returns:
        c_1s: diffusion coefficients of the rational sum
        c_2s: reaction coefficients of the rational sum
        weights: multiplicative coefficients of the rational sum
        constant: multiplicative constant in front of the sum
    """
    rational_parameters = {"c_1s": None, "c_2s": None, "weights": None,
                           "constant": 0}

    M = np.ceil((np.pi**2) / (4. * s * kappa**2))
    N = np.ceil((np.pi**2) / (4. * (1. - s) * kappa**2))

    ls = np.arange(-M, N + 1, 1, dtype=np.float64)
    rational_parameters["c_1s"] = np.exp(2. * kappa * ls)
    rational_parameters["c_2s"] = np.ones_like(ls)
    rational_parameters["weights"] = np.exp(2. * s * kappa * ls)
    rational_parameters["constant"] = (2. * np.sin(np.pi * s) * kappa) / np.pi

    return rational_parameters


def parametric_problem(f, V, k, rational_parameters, bcs,
                       boundary_entities_sorted, a_form, L_form, cst_1, cst_2,
                       ref_step=0):
    mesh = V.mesh
    ufl_domain = mesh.ufl_domain()

    # Measures and normal
    dx = Measure("dx", domain=mesh)
    dS = Measure("dS", domain=mesh)
    n = ufl.FacetNormal(ufl_domain)

    element_f = ufl.FiniteElement("DG", ufl.triangle, k + 1)
    element_g = ufl.FiniteElement("DG", ufl.triangle, k)
    element_e = ufl.FiniteElement("DG", ufl.triangle, 0)

    # Initialize BW estimator interpolation matrix
    N = fenicsx_error_estimation.create_interpolation(element_f, element_g)

    V_f = FunctionSpace(mesh, element_f)
    e_f = TrialFunction(V_f)
    v_f = TestFunction(V_f)

    V_e = FunctionSpace(mesh, element_e)
    v_e = TestFunction(V_e)

    # Fractional solution and fractional BW solution
    u_h = Function(V)
    e_h_f = Function(V_f)
    eta_h = Function(V_e)   # To store L2 norm of BW parametric solution
                            # (unused)

    # Parametric solution and parametric BW solution
    u_param = Function(V)
    e_h_param = Function(V_f)

    # DBC
    e_D = Function(V_f)
    e_D.vector.set(0.)

    c_1s = rational_parameters["c_1s"]
    c_2s = rational_parameters["c_2s"]
    constant = rational_parameters["constant"]

    e_h = Coefficient(V_f)
    weights = rational_parameters["weights"]

    estimators = {"BW fractional solution": None}

    for i, (c_1, c_2, weight) in enumerate(zip(c_1s, c_2s, weights)):
        print(f'Refinement step {ref_step}: Parametric problem {i}: System solve and error estimation...')
        # Parametric problems solves
        cst_1.value = c_1
        cst_2.value = c_2
        # TODO: Sparsity pattern could be moved outside loop
        A = assemble_matrix(a_form, bcs=bcs)
        A.assemble()
        b = assemble_vector(L_form)

        u_param.vector.set(0.0)
        apply_lifting(b, [a_form], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, bcs)

        # Linear system solve
        options = PETSc.Options()
        options["ksp_type"] = "cg"
        options["pc_type"] = "hypre"
        options["ksp_rtol"] = 1e-7
        options["pc_hypre_type"] = "boomeramg"

        solver = PETSc.KSP().create(MPI.COMM_WORLD)
        solver.setOperators(A)
        solver.setFromOptions()
        solver.solve(b, u_param.vector)
        u_param.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                   mode=PETSc.ScatterMode.FORWARD)

        # Update fractional solution
        u_h.vector.axpy(weight, u_param.vector)

        # A posteriori error estimation
        a_e_form = cst_1 * inner(grad(e_f), grad(v_f)) * dx + cst_2 * inner(e_f, v_f) * dx
        L_e_form = inner(f + cst_1 * div(grad(u_param)) - cst_2 * u_param, v_f) * dx\
            + inner(cst_1 * jump(grad(u_param), -n), avg(v_f)) * dS

        L_eta = inner(inner(e_h, e_h), v_e) * dx

        eta_h.vector.set(0.0)       # L2 norm of parametric Bank-Weiser solution (unused)
        e_h_param.vector.set(0.0)   # Parametric Bank-Weiser solution

        fenicsx_error_estimation.estimate(
            eta_h, a_e_form, L_e_form, L_eta,
            N, boundary_entities_sorted, e_h=e_h_param,
            e_D=e_D, diagonal=max(1., cst_1.value))

        # Update fractional error solution
        e_h_f.vector.axpy(weight, e_h_param.vector)

    u_h.vector.scale(constant)
    e_h_f.vector.scale(constant)

    # Computation of the L2 BW estimator
    bw_vector = assemble_vector(form(inner(inner(e_h_f, e_h_f), v_e) * dx))
    bw = Function(V_e)
    bw.vector.setArray(bw_vector)

    print("bw local", bw.vector.array)

    estimators["L2 squared local BW"] = bw
    estimators["L2 global BW"] = np.sqrt(bw.vector.sum())
    return u_h, estimators


def main():
    # Fractional power in (0, 1)
    s = 0.5
    # Lowest eigenvalue of the Laplacian
    lower_bound_eigenvalues = 1.
    # Finite element degree
    k = 1
    # Number adaptive refinements
    num_refinement_steps = 15
    # Dorfler marking parameter
    theta = 0.3

    # Structured mesh
    mesh = create_unit_square(MPI.COMM_WORLD, 32, 32)

    # Data
    def f_e(x):
        values = np.ones(x.shape[1])
        values[np.where(np.logical_and(x[0] < 0.5, x[1] > 0.5))] = -1.0
        values[np.where(np.logical_and(x[0] > 0.5, x[1] < 0.5))] = -1.0
        return values


    if method == "bp":
        # Rational approximation scheme (BP)
        fineness_param = 0.35
        rational_parameters = rational_approximation(lower_bound_eigenvalues,
                                                     fineness_param, s)
    elif method == "bura":
        # Rational approximation scheme (BURA)
        brasil_bura_parameters = pd.read_csv("./brasil_bura_coefs.csv")
        fineness_param = brasil_bura_parameters["degree"].values
        residuals = brasil_bura_parameters["residuals"].values
        c_1s = brasil_bura_parameters["poles"].values
        c_2s = np.ones_like(c_1s)

        rational_parameters = {"c_1s": np.abs(c_1s), "c_2s": c_2s,
                               "weights": residuals/c_1s,
                               "constant": 1.}

        C_N = np.sum(-residuals/c_1s)

    # Results storage
    results = {"dof num": [], "L2 bw": []}
    for ref_step in range(num_refinement_steps):
        dx = Measure("dx", domain=mesh)

        # Finite element spaces and functions
        element = ufl.FiniteElement("CG", mesh.ufl_cell(), k)
        element_e = ufl.FiniteElement("DG", mesh.ufl_cell(), 0)
        V = FunctionSpace(mesh, element)
        V_e = FunctionSpace(mesh, element_e)
        u = TrialFunction(V)
        v = TestFunction(V)

        # Interpolate input data into a FE function
        f = Function(V_e)
        f.interpolate(f_e)

        with XDMFFile(mesh.comm, "output/" + method + f"/f_{str(ref_step).zfill(3)}.xdmf",
                "w") as of:
            of.write_mesh(mesh)
            of.write_function(f)

        # Initialize bilinear and linear form of parametric problem
        cst_1 = Constant(mesh, 0.)
        cst_2 = Constant(mesh, 0.)

        a_form = form(cst_1 * inner(grad(u), grad(v)) * dx
                    + cst_2 * inner(u, v) * dx)
        L_form = form(inner(f, v) * dx)

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
        u_h, estimators = parametric_problem(f, V, k, rational_parameters, bcs,
                                             boundary_entities_sorted, a_form,
                                             L_form, cst_1, cst_2,
                                             ref_step=ref_step)
        if method == "bura":
            f_V = Function(V)
            f_V.interpolate(f_e)
            u_h.vector.array += C_N * f_V.vector.array

        # Estimator steering the refinement
        bw_sq_local_estimator = estimators["L2 squared local BW"]
        bw_global_estimator = estimators["L2 global BW"]

        with XDMFFile(mesh.comm, "output/" + method + f"/u_{str(ref_step).zfill(3)}.xdmf",
                "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(u_h)

        with XDMFFile(mesh.comm, "output/" + method + f"/l2_bw_{str(ref_step).zfill(3)}.xdmf",
                "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(bw_sq_local_estimator)

        mesh = mesh_refinement(mesh, bw_sq_local_estimator, bw_global_estimator,
                               theta)

        results["dof num"].append(V.dofmap.index_map.size_global)
        results["L2 bw"].append(bw_global_estimator)

        df = pd.DataFrame(results)
        df.to_csv(f"results_{method}.csv")
        print(df)


if __name__ == "__main__":
    main()
