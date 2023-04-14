# Copyright 2020, Jack S. Hale, Raphael Bulle.
# SPDX-License-Identifier: LGPL-3.0-or-later

import fenicsx_error_estimation
import numpy as np
import baryrat as br

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


def BURA_rational_approximation(degree, s):
    """
    Generates the parameters for the BURA using the BRASIL method from Hofreither 2020.

    Parameters:
        degree: degree of the rational approximation (= number of parametric solves - 1)
        s: fractional power
    Returns a dict containing:
        rational_parameters: dict containing
            c_1s: diffusion coefficients of the rational sum
            c_2s: reaction coefficients of the rational sum
            weights: multiplicative coefficients of the rational sum
            constant: multiplicative constant in front of the sum
            initial_constant: once the parametric solutions are added initial_constant * f must be added to this sum to obtain the fractional approximation
        err: the rational approximation error estimation
    """

    def r(x):           # BURA method approximate x^s instead of x^{-s}
        return x**s

    domain = [1e-8, 1.] # The upper bound is lambda_1^{-1} where lambda_1 is the lowest eigenvalue, in this case lambda_1 = 1
    xs = np.linspace(domain[0], 1., 10000)

    r_brasil = br.brasil(r, domain, (degree-1,degree)) # (degree-1, degree) gives the best result
    pol_brasil, res_brasil = r_brasil.polres()

    c_1s = -pol_brasil
    c_2s = np.ones_like(c_1s)
    weights = res_brasil/pol_brasil
    constant = 1.

    rational_parameters = {"c_1s": c_1s, "c_2s": c_2s,
                        "weights": weights,
                        "constant": constant,
                        "initial constant": -np.sum(weights)}

    # Rational error estimation
    err = np.max(np.abs(r(xs) - r_brasil(xs)))

    return rational_parameters, err


def BP_rational_approximation(kappa, s):
    """
    Generates the parameters for the rational sum according to exponentially
    convergent scheme in Bonito and Pasciak 2013.

    Parameters:
        kappa: fineness parameter
        s: fractional power

    Returns a dict containing:
        rational_parameters: a dict containing:
            c_1s: diffusion coefficients of the rational sum
            c_2s: reaction coefficients of the rational sum
            weights: multiplicative coefficients of the rational sum
            constant: multiplicative constant in front of the sum
        err: the rational approximation error estimation
    """
    rational_parameters = {"c_1s": None, "c_2s": None, "weights": None,
                           "constant": 0}

    M = np.ceil((np.pi**2) / (4. * s * kappa**2))
    N = np.ceil((np.pi**2) / (4. * (1. - s) * kappa**2))

    ls = np.arange(-M, N + 1, 1, dtype=np.float64)
    c_1s = np.exp(2. * kappa * ls)
    c_2s = np.ones_like(c_1s)
    weights = np.exp(2. * s * kappa * ls)
    constant = (2. * np.sin(np.pi * s) * kappa) / np.pi

    rational_parameters = {"c_1s": c_1s, "c_2s": c_2s, "weights": weights,
                           "constant": constant,
                           "initial constant": 0.}   # There is no initial term in this method so initial_constant must be 0.
    
    # Rational error estimation
    xs = np.linspace(1., 1e8, 10000)

    ys = []
    for x in xs:
        bp_terms = np.multiply(weights, np.reciprocal(c_2s + c_1s * x))
        bp_sum = np.sum(bp_terms)
        bp_sum *= constant
        ys.append(bp_sum)
    
    err = np.max(np.abs(np.power(xs, -s) - ys))

    return rational_parameters, err


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
    # Finite element degree
    k = 1
    # Number adaptive refinements
    num_refinement_steps = 20
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

    parameter = initial_parameter

    rational_parameters, rational_error = rational_approximation(initial_parameter, s)
    total_num_solves = 0
    # Results storage
    results = {"dof num": [], "rational parameter": [], "solves num": [], "L2 bw": [], "rational error": [], "total solves num": []}
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

        # If method == "bp" then this step is useless (intial_constant=0.)
        f_V = Function(V)
        f_V.interpolate(f_e)
        u_h.vector.array += rational_parameters["initial constant"] * f_V.vector.array

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

        results["dof num"].append(V.dofmap.index_map.size_global)
        results["L2 bw"].append(bw_global_estimator)
        results["rational error"].append(rational_error)

        num_solves = len(rational_parameters["c_1s"])
        total_num_solves += num_solves
        results["solves num"].append(num_solves)
        results["total solves num"].append(total_num_solves)

        # Mesh refinement
        mesh = mesh_refinement(mesh, bw_sq_local_estimator, bw_global_estimator,
                               theta)
        
        # Rational scheme refinement
        results["rational parameter"].append(parameter)
        if ref_step <= 1:
            coef = 0.5
        else:
            coef = results["L2 bw"][-1]/results["L2 bw"][-2]

        if adaptive:
            rational_error = np.Inf
            if method == "bura":
                parameter = 1       # Initial parameter for BURA
            elif method == "bp":
                parameter = 1.   # Initial parameter for BP
            while(rational_error > coef * bw_global_estimator):
                if method == "bura":
                    parameter += 1
                elif method == "bp":
                    parameter -= 0.01
                rational_parameters, rational_error = rational_approximation(parameter, s)

        df = pd.DataFrame(results)
        if adaptive:
            df.to_csv(f"results_{method}_adaptive.csv")
        else:
            df.to_csv(f"results_{method}_non_adaptive.csv")
        print(df)


if __name__ == "__main__":
    for method in ["bura"]:
        for adaptive in [False, True]:
            if method == "bp":
                initial_parameter = 0.52    # Fineness parameter (in (0., 1.), more precise if close to 0.)
                # initial_parameter = 0.8   adaptive
                # initial_parameter = 0.52  non-adaptive
                rational_approximation = BP_rational_approximation

            elif method == "bura":
                initial_parameter = 8      # Degree of the rational approximation (integer, more precise if large)
                # initial_parameter = 4     adaptive
                # initial_parameter = 8     non-adaptive
                rational_approximation = BURA_rational_approximation
            main()