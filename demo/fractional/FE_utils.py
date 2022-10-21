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


def parametric_problem(f, V, k, rational_parameters, bcs,
                       boundary_entities_sorted, a_form, L_form, cst_1, cst_2,
                       ref_step=0, bw_global_estimator=np.Inf):
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
        print("\t Param solve:", i)

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

    global_bw = np.sqrt(bw.vector.sum())
    estimators["L2 squared local BW"] = bw
    estimators["L2 global BW"] = global_bw
    return u_h, estimators