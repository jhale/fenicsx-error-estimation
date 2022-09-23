# Copyright 2020, Jack S. Hale, Raphael Bulle.
# SPDX-License-Identifier: LGPL-3.0-or-later

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

def parametric_problem(f, V, k, rational_parameters, bcs,
                       boundary_entities_sorted, a_form, L_form, cst_1, cst_2):
    element_f = ufl.FiniteElement("DG", ufl.triangle, 2)
    element_g = ufl.FiniteElement("DG", ufl.triangle, 1)
    element_e = ufl.FiniteElement("DG", ufl.triangle, 0)

    V_f = FunctionSpace(mesh, element_f)
    e_f = TrialFunction(V_f)
    v_f = TestFunction(V_f)

    V_e = FunctionSpace(mesh, element_e)
    v_e = TestFunction(V_e)

    # Fractional solution and fractional BW solution
    u_h = Function(V)
    bw = Function(V_f)

    # Parametric solution and parametric BW solution
    u_param = Function(V)
    bw_param = Function(V_f)

    # DBC
    e_D = Function(V_f)
    e_D.vector.set(0.)

    eta_e = Function(V_e)

    c_1s = rational_parameters["c_1s"]
    c_2s = rational_parameters["c_2s"]
    weights = rational_parameters["weights"]

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

    u_h.vector.scale(constant)


def main():
    # Fractional power in (0, 1)
    s = 0.5
    #Lowest eigenvalue of the Laplacian
    lower_bound_eigenvalues = 1.
    # Finite element degree
    k = 1
    # Number adaptive refinements
    num_refinement_steps = 15
    # Dorfler marking parameter
    theta = 0.3

    # Initial mesh
    mesh = RectangleMesh(
            MPI.COMM_WORLD,
            [np.array([0, 0, 0]), np.array([1, 1, 0])], [8, 8],
            CellType.triangle)

    # Data
    def f_e(x):
        values = np.ones(x.shape[1])
        values[np.where(np.logical_and(x[0] < 0.5, x[1] > 0.5))] = -1.0
        values[np.where(np.logical_and(x[0] > 0.5, x[1] < 0.5))] = -1.0
        return values

    fineness_param = 0.3

    # Rational approximation scheme
    rational_parameters = rational_approximation(lower_bound_eigenvalues,
                                                 fineness_param, s)
    # q, c_1s, c_2s, weights, constant = rational_approximation(
    #                                    lower_bound_eigenvalues, fineness_param, s)

    # Results storage
    results = {"rational approximation parameter": [fineness_param], "dof num": [], "L2 bw": []}
    for i in range(num_refinement_steps):
        # Finite element spaces and functions
        ufl_domain = mesh.ufl_domain()

        element = ufl.FiniteElement("CG", mesh.ufl_cell(), k)
        V = FunctionSpace(mesh, element)
        u = TrialFunction(V)
        v = TestFunction(V)

        # Interpolate input data into a FE function
        f = Function(V)
        f.interpolate(f_e)

        # Initialize bilinear and linear form of parametric problem
        cst_1 = Constant(mesh, 0.)
        cst_2 = Constant(mesh, 0.)
        e_h = Coefficient(V_f)

        a_form = Form(cst_1 * inner(grad(u), grad(v)) * dx +\
                      cst_2 * inner(u, v) * dx)
        L_form = Form(inner(f, v) * dx)

        # Homogeneous zero Dirichlet boundary condition
        u0 = Function(V)
        u0.vector.set(0.)
        facets = locate_entities_boundary(
                mesh, 1, lambda x: np.ones(x.shape[1], dtype=bool))
        dofs = locate_dofs_topological(V, 1, facets)
        bcs = [DirichletBC(u0, dofs)]

        boundary_entities = locate_entities_boundary(
                mesh, 1, lambda x: np.ones(x.shape[1], dtype=bool))
        boundary_entities_sorted = np.sort(boundary_entities)

        # Fractional solution
        u_h, estimators = parametric_problem(f, V, k, rational_parameters, bcs,
                                             boundary_entities_sorted, a_form,
                                             L_form, cst_1, cst_2)

        # Estimator steering the refinement
        eta_e = estimators["L2 bw"]
        eta = np.sqrt(eta_e.vector.sum())

        # Dörfler marking
        eta_global = eta**2
        cutoff = theta * eta_global

        assert MPI.COMM_WORLD.size == 1
        sorted_cells = np.argsort(eta_e.vector.array)[::-1]
        rolling_sum = 0.
        for i, e in enumerate(eta_e.vector.array[sorted_cells]):
            rolling_sum += e
            if rolling_sum > cutoff:
                breakpoint = i
                break

        refine_cells = sorted_cells[0:breakpoint + 1]
        indices = np.array(np.sort(refine_cells), dtype=np.int32)
        markers = np.zeros(indices.shape, dtype=np.int8)
        markers_tag = dolfinx.MeshTags(mesh, mesh.topology.dim, indices,
                                       markers)

        # Refine mesh
        mesh.topology.create_entities(mesh.topology.dim - 1)
        mesh = dolfinx.mesh.refine(mesh, cell_markers=markers_tag)

        results["Num dof"].append(V.dofmap.index_map.size_global)
        results["L2 bw"].append(eta)

        df = pd.DataFrame(results)
        df.to_csv("results.csv")



if __name__=="__main__":
    main()
