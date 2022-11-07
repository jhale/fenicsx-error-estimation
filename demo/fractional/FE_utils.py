import fenicsx_error_estimation
import numpy as np
import dolfinx
import ufl
from dolfinx.fem import (Constant, Function, FunctionSpace, apply_lifting,
                         dirichletbc, form, locate_dofs_topological, locate_dofs_geometrical, set_bc)
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
                       ref_step=0, output_dir=None, sines_test_case=False):
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
    initial_constant = rational_parameters["initial constant"]

    e_h = Coefficient(V_f)
    weights = rational_parameters["weights"]

    estimators = {"BW fractional solution": None}

    parametric_results = {"parametric index": [], "parametric exact error": []}
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
        
        if sines_test_case:
            # Analytical parametric solution
            def u_param_exact(x):
                values = 1./(c_2 + c_1 * 2.) * np.sin(x[0]) * np.sin(x[1])
                return values

            # Exact error
            element_f = ufl.FiniteElement("DG", mesh.ufl_cell(), k+1)
            V_f_CG = FunctionSpace(mesh, element_f)
            u_param_exact_V_f = Function(V_f_CG)
            u_param_exact_V_f.interpolate(u_param_exact)
            u_param_V_f = Function(V_f_CG)
            u_param_V_f.interpolate(u_param)
            v_e = TestFunction(V_e)

            e_param_f = u_param_exact_V_f - u_param_V_f
            local_err_2 = assemble_vector(form(inner(inner(e_param_f, e_param_f), v_e) * dx))
            err_param = np.sqrt(local_err_2.sum())

            parametric_results["parametric index"].append(i)
            parametric_results["parametric exact error"].append(err_param)
                
            df = pd.DataFrame(parametric_results)
            df.to_csv(output_dir + f"parametric_results_{str(ref_step)}.csv")

            with XDMFFile(mesh.comm, output_dir + "/parametric_solutions/" + f"u_{str(i).zfill(4)}.xdmf", "w") as of:
                of.write_mesh(mesh)
                of.write_function(u_param)

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

    # Compute the L2 projection of f onto V (only necessary for BURA)
    if rational_parameters["initial constant"] > 0.:
        u0 = Function(V)
        u0.vector.set(0.)
        facets = locate_entities_boundary(
                mesh, 1, lambda x: np.ones(x.shape[1], dtype=bool))
        dofs = locate_dofs_topological(V, 1, facets)
        bcs = [dirichletbc(u0, dofs)]

        u = TrialFunction(V)
        v = TestFunction(V)
        f_l2_V = Function(V)
        a_V = form(inner(u, v) * dx)
        A_V = assemble_matrix(a_V)
        A_V.assemble()
        L_V = form(inner(f, v) * dx)
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

        u0_f = Function(V_f)
        u0_f.vector.set(0.)
        facets = locate_entities_boundary(
                mesh, 1, lambda x: np.ones(x.shape[1], dtype=bool))

        dof_locate = lambda x: np.logical_or.reduce((np.isclose(x[0], 0.0),
                                                     np.isclose(x[0], np.pi),
                                                     np.isclose(x[1], 0.0),
                                                     np.isclose(x[1], np.pi)))
        dofs = locate_dofs_geometrical(V_f, dof_locate)
        print(dofs)
        # dofs = locate_dofs_topological(V_f, 1, facets)
        bcs = [dirichletbc(u0_f, dofs)]

        u_f = TrialFunction(V_f)
        v_f = TestFunction(V_f)
        f_l2_V_f = Function(V_f)
        a_V_f = form(inner(u_f, v_f) * dx)
        A_V_f = assemble_matrix(a_V_f)
        A_V_f.assemble()
        L_V_f = form(inner(f, v_f) * dx)
        b_V_f = assemble_vector(L_V_f)

        set_bc(b_V_f, bcs)

        # Linear system solve
        options = PETSc.Options()
        options["ksp_type"] = "cg"
        options["pc_type"] = "hypre"
        options["ksp_rtol"] = 1e-7
        options["pc_hypre_type"] = "boomeramg"

        solver = PETSc.KSP().create(MPI.COMM_WORLD)
        solver.setOperators(A_V_f)
        solver.setFromOptions()
        solver.solve(b_V_f, f_l2_V_f.vector)

        f_V_V_f = Function(V_f)
        f_V_V_f.interpolate(f_l2_V)

        e_h_f.vector.array += rational_parameters["initial constant"] * (f_V_V_f.vector.array - f_l2_V_f.vector.array)

    # Computation of the L2 BW estimator
    bw_vector = assemble_vector(form(inner(inner(e_h_f, e_h_f), v_e) * dx))
    bw = Function(V_e)
    bw.vector.setArray(bw_vector)

    estimators["L2 squared local BW"] = bw
    estimators["L2 global BW"] = np.sqrt(bw.vector.sum())
    return u_h, estimators