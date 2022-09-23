# Copyright 2020, Jack S. Hale, Raphael Bulle.
# SPDX-License-Identifier: LGPL-3.0-or-later

# This demo solves the fractional Laplacian problem using an adaptive finite
# element scheme outlined in Bulle et al. 2022. The problem specification is
# taken from Bonito and Pasciak 2013.

import numpy as np

import dolfinx
from dolfinx import Constant, DirichletBC, Function, FunctionSpace, RectangleMesh
from dolfinx.cpp.mesh import CellType
from dolfinx.fem import (apply_lifting, assemble_matrix, assemble_vector, Form,
                         locate_dofs_topological, set_bc)
from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities_boundary

from mpi4py import MPI
from petsc4py import PETSc
import pandas as pd

import fenicsx_error_estimation

import ufl
from ufl import avg, div, grad, inner, jump, Measure, TrialFunction, TestFunction, Coefficient

# Fractional power (in (0, 1))
s = 0.5
# Lower bound on spectrum of the Laplacian operator on square
lmbda_0 = 1.
# Finite element degree
k = 1
# Tolerance (tolerance for rational sum will be tol * 1e-3 * l2_norm_data,
# tolerance for FE will be tol)
tol = 1e-5
# Dorfler marking parameter
theta = 0.3


# Structured mesh
mesh = RectangleMesh(
    MPI.COMM_WORLD,
    [np.array([0, 0, 0]), np.array([1, 1, 0])], [8, 8],
    CellType.triangle)


def f_e(x):
    """Checkerboard problem data"""
    values = np.ones(x.shape[1])
    values[np.where(np.logical_and(x[0] < 0.5, x[1] > 0.5))] = -1.0
    values[np.where(np.logical_and(x[0] > 0.5, x[1] < 0.5))] = -1.0
    return values


# Find kappa s.t. rational approx. error < tol * 1e-3 * || f_e || according to
# a priori result in Bonito and Pasciak 2013
def bp_sum(lmbda, kappa, s):
    """
    Generates the parameters for the rational sum according to exponentially
    convergent scheme in Bonito and Pasciak 2013.

    Parameters:
        lmbda: eigenvalue at which the sum is evaluated
        kappa: fineness parameter
        s: fractional power

    Returns:
        q: value of the rational sum at lmbda
        c_1s: diffusion coefficients of the rational sum
        c_2s: reaction coefficients of the rational sum
        weights: multiplicative coefficients of the rational sum
        constant: multiplicative constant in front of the sum
    """
    M = np.ceil((np.pi**2) / (4. * s * kappa**2))
    N = np.ceil((np.pi**2) / (4. * (1. - s) * kappa**2))

    constant = (2. * np.sin(np.pi * s) * kappa) / np.pi

    ls = np.arange(-M, N + 1, 1, dtype=np.float64)
    c_1s = np.exp(2. * kappa * ls)
    c_2s = np.ones_like(c_1s)
    weights = np.exp(2. * s * kappa * ls)

    q = constant * np.sum(weights / (c_2s + c_1s * lmbda))

    return q, c_1s, c_2s, weights, constant


f_e_L2_norm = 1.
tol_rs = tol * 1e-3 * f_e_L2_norm
trial_kappas = np.flip(np.arange(1e-2, 1., step=0.01))
for kappa in trial_kappas:
    q, c_1s, c_2s, weights, constant = bp_sum(lmbda_0, kappa, s)
    diff = np.abs(lmbda_0 ** (-s) - q)
    if np.less(diff, tol_rs):
        break

# Initialize estimator value
eta = 1.

ref_step = 0
results = {"dof num": [], "L2 bw": []}
while np.greater(eta, tol):
    ufl_domain = mesh.ufl_domain()

    # FE spaces
    element = ufl.FiniteElement("CG", mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, element)
    u = TrialFunction(V)
    v = TestFunction(V)

    results["dof num"].append(V.dofmap.index_map.size_global)

    element_f = ufl.FiniteElement("DG", ufl.triangle, 2)
    element_g = ufl.FiniteElement("DG", ufl.triangle, 1)
    element_e = ufl.FiniteElement("DG", ufl.triangle, 0)
    V_f = FunctionSpace(mesh, element_f)
    e_f = TrialFunction(V_f)
    v_f = TestFunction(V_f)

    V_e = FunctionSpace(mesh, element_e)
    v_e = TestFunction(V_e)

    # Initialize BW estimator interpolation matrix
    N = fenicsx_error_estimation.create_interpolation(element_f, element_g)

    # Measures and normal
    dx = Measure("dx", domain=mesh)
    dS = Measure("dS", domain=mesh)
    n = ufl.FacetNormal(ufl_domain)

    # Interpolate input data into a FE function
    f = Function(V)
    f.interpolate(f_e)

    # Initialize bilinear and linear forms
    cst_1 = Constant(mesh, 0.)
    cst_2 = Constant(mesh, 0.)
    e_h = Coefficient(V_f)

    a_form = Form(cst_1 * inner(grad(u), grad(v)) * dx + cst_2 * inner(u, v) * dx)
    L_form = Form(inner(f, v) * dx)

    # Homogeneous zero Dirichlet boundary condition
    u0 = Function(V)
    u0.vector.set(0.0)
    facets = locate_entities_boundary(
        mesh, 1, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = locate_dofs_topological(V, 1, facets)
    bcs = [DirichletBC(u0, dofs)]

    # Homogeneous zero BW estimator boundary condition
    boundary_entities = locate_entities_boundary(
        mesh, 1, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_entities_sorted = np.sort(boundary_entities)

    # Fractional solution
    u_h = Function(V)
    # Parametric solution
    u_param = Function(V)
    # Bank-Weiser error solution for fractional problem
    bw_f = Function(V_f)
    # L^2 error estimator for fractional problem proposed in Bulle et al. 2022
    eta_e = Function(V_e)

    # Functions to store results
    # L2 norm of parametric Bank-Weiser solution (unused in this methodology)
    eta_h = Function(V_e)
    e_h_f = Function(V_f)   # Parametric Bank-Weiser solution
    e_D = Function(V_f)     # Zero dirichlet boundary condition
    e_D.vector.set(0.0)    # Zero dirichlet boundary condition

    for i, (c_1, c_2, weight) in enumerate(zip(c_1s, c_2s, weights)):
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
        print(f'Refinement step {ref_step}: Parametric problem {i}: System solve...')
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

        eta_h.vector.set(0.0)   # L2 norm of parametric Bank-Weiser solution (unused)
        e_h_f.vector.set(0.0)   # Parametric Bank-Weiser solution

        print(f'Refinement step {ref_step}: Parametric problem {i}: Estimate...')
        fenicsx_error_estimation.estimate(
            eta_h, a_e_form, L_e_form, L_eta,
            N, boundary_entities_sorted, e_h=e_h_f,
            e_D=e_D, diagonal=max(1., cst_1.value))

        # Update fractional error solution
        bw_f.vector.axpy(weight, e_h_f.vector)

    # Scale fractional solution
    print(f'Refinement step {ref_step}: Solution computation and solve...')
    u_h.vector.scale(constant)
    with XDMFFile(MPI.COMM_WORLD, f"./output/u_{str(ref_step)}.xdmf", "w") as fo:
        fo.write_mesh(mesh)
        fo.write_function(u_h)

    # Scale Bank-Weiser solution
    print(f'Refinement step {ref_step}: Estimator computation and solve...')
    bw_f.vector.scale(constant)
    with XDMFFile(MPI.COMM_WORLD, f"./output/bw_{str(ref_step)}.xdmf", "w") as fo:
        fo.write_mesh(mesh)
        fo.write_function(bw_f)

    # Compute L2 error estimator
    eta_f = assemble_vector(inner(inner(bw_f, bw_f), v_e) * dx)
    eta_e.vector.setArray(eta_f)
    with XDMFFile(MPI.COMM_WORLD, f"./output/eta_{str(ref_step)}.xdmf", "w") as fo:
        fo.write_mesh(mesh)
        fo.write_function(eta_e)

    # Compute L2 error estimator
    # TODO: Not MPI safe
    eta = np.sqrt(sum(eta_e.vector.array))
    results["L2 bw"].append(eta)
    print(f'Refinement step: {ref_step}: Error:', eta)

    # Dörfler marking
    print(f'Refinement step: {ref_step} Marking...')
    eta_global = eta**2
    cutoff = theta * eta_global

    assert MPI.COMM_WORLD.size == 1
    sorted_cells = np.argsort(eta_e.vector.array)[::-1]
    rolling_sum = 0.0
    for i, e in enumerate(eta_e.vector.array[sorted_cells]):
        rolling_sum += e
        if rolling_sum > cutoff:
            breakpoint = i
            break

    refine_cells = sorted_cells[0:breakpoint + 1]
    indices = np.array(np.sort(refine_cells), dtype=np.int32)
    markers = np.zeros(indices.shape, dtype=np.int8)
    markers_tag = dolfinx.MeshTags(mesh, mesh.topology.dim, indices, markers)

    # Refine mesh
    print(f'Refinement step {ref_step}: Refinement...')
    mesh.topology.create_entities(mesh.topology.dim - 1)
    mesh = dolfinx.mesh.refine(mesh, cell_markers=markers_tag)

    ref_step += 1

df = pd.DataFrame.from_dict(results, orient="index").transpose()
print(df)
df.to_csv("./results.csv")
print(f"Proposed kappa: {kappa}")
