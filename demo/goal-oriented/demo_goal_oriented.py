import os
import numpy as np

import mpi4py.MPI as MPI
import petsc4py.PETSc as PETSc

import pandas

import dolfinx
from dolfinx.io import XDMFFile

import ufl
from ufl import avg, div, grad, inner, jump

import fenicsx_error_estimation.estimate
from fenicsx_error_estimation import create_interpolation

k = 1


def main():
    estimator = "bw"
    parameters = (1, 2)

    adaptive_refinement(estimator, parameters)
    return


def adaptive_refinement(estimator, parameters=None):
    if estimator != "bw" and parameters is not None:
        print("Parameters are ignored when the estimator is not 'bw'.")

    if estimator == "bw":
        estimator_path = "bw_" + str(parameters[0]) + "_" + str(parameters[1])
    else:
        estimator_path = estimator

    with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", 'r') as fi:
        mesh = fi.read_mesh()

    # Adaptive refinement loop
    results = []
    for i in range(0, 20):
        result = {}

        def u_exact(x):
            r = np.sqrt((x[0] - 1.) * (x[0] - 1.) + (x[1] - 1.) * (x[1] - 1.))
            theta = np.arctan2((x[1] - 1.), (x[0] - 1.)) + np.pi / 2.
            values = r**(2. / 3.) * np.sin((2. / 3.) * theta)
            values[np.where(np.logical_or(np.logical_and(np.isclose((x[0] - 1.), 0., atol=1e-10), (x[1] - 1.) < 0.),
                                          np.logical_and(np.isclose((x[1] - 1.), 0., atol=1e-10), (x[0] - 1.) < 0.)))] = 0.
            return values

        V = dolfinx.FunctionSpace(mesh, ("CG", k))

        # Zero data
        f = dolfinx.Function(V)

        u_exact_V = dolfinx.Function(V)
        u_exact_V.interpolate(u_exact)
        with XDMFFile(MPI.COMM_WORLD,
                      os.path.join("output", estimator_path, f"u_exact_{str(i).zfill(4)}.xdmf"),
                      "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(u_exact_V)

        facets = dolfinx.mesh.locate_entities_boundary(
            mesh, 1, lambda x: np.ones(x.shape[1], dtype=bool))
        dofs = dolfinx.fem.locate_dofs_topological(V, 1, facets)
        bcs = [dolfinx.DirichletBC(u_exact_V, dofs)]

        # Solve primal problem
        print(estimator + " steering " + f"STEP {i}:\n solving primal problem...")
        u_h = solve_primal(V, u_exact_V, bcs)
        with XDMFFile(MPI.COMM_WORLD, os.path.join("output", estimator_path, f"u_h_{str(i).zfill(4)}.xdmf"), "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(u_h)

        # Solve dual problem
        print(estimator + " steering " + f'STEP {i}:\n solving dual problem...')
        z_h = solve_dual(V)
        with XDMFFile(MPI.COMM_WORLD, os.path.join("output", estimator_path, f"z_h_{str(i).zfill(4)}.xdmf"), "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(z_h)

        # BW estimation primal problem
        print(estimator + " steering " + f"STEP {i}:\n estimating primal problem (bw)...")
        eta_bw_u = estimate_bw(u_h, f)
        with XDMFFile(MPI.COMM_WORLD, os.path.join("output", estimator_path, f"eta_u_{str(i).zfill(4)}.xdmf"), "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(eta_bw_u)
        result['error_bw_u'] = np.sqrt(eta_bw_u.vector.sum())

        # Residual estimation primal problem
        print(estimator + " steering " + f"STEP {i}:\n estimating primal problem (res)...")
        eta_res_u = estimate_residual(u_h, f)
        with XDMFFile(MPI.COMM_WORLD, os.path.join("output", estimator_path, f"eta_res_u_{str(i).zfill(4)}.xdmf"), "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(eta_res_u)
        result['error_res_u'] = np.sqrt(eta_res_u.vector.sum())

        # ZZ estimation primal problem
        print(estimator + " steering " + f"STEP {i}:\n estimating primal problem (zz)...")
        eta_zz_u = estimate_zz(u_h, f)
        with XDMFFile(MPI.COMM_WORLD, os.path.join("output", estimator_path, f"eta_zz_u_{str(i).zfill(4)}.xdmf"), "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(eta_zz_u)
        result['error_zz_u'] = np.sqrt(eta_zz_u.vector.sum())

        # BW estimation dual problem
        print(estimator + " steering " + f"STEP {i}:\n estimating dual problem (bw)...")
        eta_bw_z = estimate_bw(z_h, weight)
        with XDMFFile(MPI.COMM_WORLD, os.path.join("output", estimator_path, f"eta_z_{str(i).zfill(4)}.xdmf"), "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(eta_bw_z)
        result['error_bw_z'] = np.sqrt(eta_bw_z.vector.sum())

        # Residual estimation dual problem
        print(estimator + " steering " + f"STEP {i}:\n estimating dual problem (res)...")
        eta_res_z = estimate_residual(z_h, weight)
        with XDMFFile(MPI.COMM_WORLD, os.path.join("output", estimator_path, f"eta_res_z_{str(i).zfill(4)}.xdmf"), "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(eta_res_z)
        result['error_res_z'] = np.sqrt(eta_res_z.vector.sum())

        # ZZ estimation dual problem
        print(estimator + " steering " + f"STEP {i}:\n estimating dual problem (zz)...")
        eta_zz_z = estimate_zz(z_h, weight)
        with XDMFFile(MPI.COMM_WORLD, os.path.join("output", estimator_path, f"eta_zz_z_{str(i).zfill(4)}.xdmf"), "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(eta_zz_z)
        result['error_zz_z'] = np.sqrt(eta_zz_z.vector.sum())

        # Calculated using P3 on a very fine adapted mesh, good to ~10 s.f.
        J_fine = 0.2341612424405788

        V_f = dolfinx.FunctionSpace(mesh, ("CG", k + 2))
        weight_V_f = dolfinx.Function(V_f)
        weight_V_f.interpolate(weight)
        with XDMFFile(MPI.COMM_WORLD, os.path.join("output", estimator_path, f"weight_{str(i).zfill(4)}.xdmf"), "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(weight_V_f)

        u_V_f = dolfinx.Function(V_f)
        u_V_f.interpolate(u_h)

        dx = ufl.Measure("dx", domain=mesh)

        J_u_h = dolfinx.fem.assemble_scalar(inner(weight_V_f, u_V_f) * dx)

        result['J_u_h'] = J_u_h
        result['exact_error'] = np.abs(J_u_h - J_fine)

        # Necessary for parallel operation
        h_local = dolfinx.cpp.mesh.h(mesh, mesh.topology.dim, np.arange(
            0, mesh.topology.index_map(mesh.topology.dim).size_local, dtype=np.int32))
        h_max = MPI.COMM_WORLD.allreduce(h_local, op=MPI.MAX)
        result['h_max'] = h_max
        h_min = MPI.COMM_WORLD.allreduce(h_local, op=MPI.MIN)
        result['h_min'] = h_min
        result['num_dofs'] = V.dofmap.index_map.size_global

        # BW WGO estimation
        eta_bw_w = estimate_wgo(eta_bw_u, eta_bw_z)
        with XDMFFile(MPI.COMM_WORLD, os.path.join("output", estimator_path, f"eta_bw_w_{str(i).zfill(4)}.xdmf"), "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(eta_bw_w)
        result['error_bw_w'] = np.sqrt(eta_bw_u.vector.sum() * eta_bw_z.vector.sum())

        # Res WGO estimation
        eta_res_w = estimate_wgo(eta_res_u, eta_res_z)
        with XDMFFile(MPI.COMM_WORLD, os.path.join("output", estimator_path, f"eta_res_w_{str(i).zfill(4)}.xdmf"), "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(eta_res_w)
        result['error_res_w'] = np.sqrt(eta_res_u.vector.sum() * eta_res_z.vector.sum())

        # ZZ WGO estimation
        eta_zz_w = estimate_wgo(eta_zz_u, eta_zz_z)
        with XDMFFile(MPI.COMM_WORLD, os.path.join("output", estimator_path, f"eta_zz_w_{str(i).zfill(4)}.xdmf"), "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(eta_zz_w)
        result['error_zz_w'] = np.sqrt(eta_zz_u.vector.sum() * eta_zz_z.vector.sum())

        # Choose the estimator to steer the adaptive strategy depending on the
        # parameter
        if estimator == "bw":
            eta_w = eta_bw_w
        elif estimator == "res":
            eta_w = eta_res_w
        elif estimator == "zz":
            eta_w = eta_zz_w

        markers_tag = marking(eta_w)

        # Refine
        print('Refining...')
        mesh = dolfinx.mesh.refine(mesh, cell_markers=markers_tag)

        with XDMFFile(MPI.COMM_WORLD, os.path.join("output", estimator_path, f"mesh{str(i).zfill(4)}.xdmf"), "w") as fo:
            fo.write_mesh(mesh)

        results.append(result)

    df = pandas.DataFrame.from_dict(results)
    df.to_pickle(os.path.join("output", estimator_path, "results.pkl"))
    print(df)


def weight(x):  # Gaussian function to focus the goal functional on a particular region of the domain
    eps_f = 0.1
    center_x = 0.75
    center_y = 0.75
    r2 = (((x[0] - 1.) - center_x)**2 + ((x[1] - 1.) - center_y)**2) / eps_f**2.

    values = np.zeros_like(x[0])

    values = np.exp(- r2 / 10.)
    return values


def solve_primal(V, u_exact_V, bcs):
    mesh = V.mesh
    dx = ufl.Measure("dx", domain=mesh)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = inner(grad(u), grad(v)) * dx

    A = dolfinx.fem.assemble_matrix(a, bcs=bcs)
    A.assemble()

    f = dolfinx.Function(V)  # Zero data
    L = inner(f, v) * dx

    b = dolfinx.fem.assemble_vector(L)
    dolfinx.fem.apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(b, bcs)

    options = PETSc.Options()
    options["ksp_type"] = "cg"
    options["ksp_view"] = None
    options["pc_type"] = "hypre"
    options["ksp_rtol"] = 1e-10
    options["pc_hypre_type"] = "boomeramg"
    options["ksp_monitor_true_residual"] = None

    u_h = dolfinx.Function(V)
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setOperators(A)
    solver.setFromOptions()
    solver.solve(b, u_h.vector)
    u_h.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                           mode=PETSc.ScatterMode.FORWARD)
    return u_h


def solve_dual(V):
    mesh = V.mesh
    dx = ufl.Measure("dx", domain=mesh)

    z = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = inner(grad(v), grad(z)) * dx

    dbc = dolfinx.Function(V)
    facets = dolfinx.mesh.locate_entities_boundary(
        mesh, 1, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = dolfinx.fem.locate_dofs_topological(V, 1, facets)
    bcs = [dolfinx.DirichletBC(dbc, dofs)]

    A = dolfinx.fem.assemble_matrix(a, bcs=bcs)
    A.assemble()

    V_f = dolfinx.FunctionSpace(mesh, ("CG", k + 2))
    weight_V_f = dolfinx.Function(V_f)
    weight_V_f.interpolate(weight)

    L = inner(weight_V_f, v) * dx

    b = dolfinx.fem.assemble_vector(L)
    dolfinx.fem.apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(b, bcs)

    options = PETSc.Options()
    options["ksp_type"] = "cg"
    options["ksp_view"] = None
    options["pc_type"] = "hypre"
    options["ksp_rtol"] = 1e-10
    options["pc_hypre_type"] = "boomeramg"
    options["ksp_monitor_true_residual"] = None

    z_h = dolfinx.Function(V)
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setOperators(A)
    solver.setFromOptions()
    solver.solve(b, z_h.vector)
    z_h.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                           mode=PETSc.ScatterMode.FORWARD)
    return z_h


def estimate_bw(u_h, f, parameters=(1, 2)):
    V = u_h.function_space
    mesh = V.mesh
    ufl_domain = mesh.ufl_domain()

    dx = ufl.Measure('dx', domain=mesh)
    dS = ufl.Measure('dS', domain=mesh)

    element_f = ufl.FiniteElement("DG", ufl.triangle, parameters[1])
    element_g = ufl.FiniteElement("DG", ufl.triangle, parameters[0])
    element_e = ufl.FiniteElement("DG", ufl.triangle, 0)
    N = create_interpolation(element_f, element_g)

    V_f = ufl.FunctionSpace(ufl_domain, element_f)
    e = ufl.TrialFunction(V_f)
    v = ufl.TestFunction(V_f)

    n = ufl.FacetNormal(mesh)

    # Bilinear form
    a_e = inner(grad(e), grad(v)) * dx

    V_w = dolfinx.FunctionSpace(mesh, ("CG", k + 2))
    weight_V_w = dolfinx.Function(V_w)
    weight_V_w.interpolate(f)

    # Linear form
    L_e = inner(weight_V_w + div(grad(u_h)), v) * dx + \
        inner(jump(grad(u_h), -n), avg(v)) * dS

    # Error form
    V_e = dolfinx.FunctionSpace(mesh, element_e)
    e_h = ufl.Coefficient(V_f)
    v_e = ufl.TestFunction(V_e)

    L_eta = inner(inner(grad(e_h), grad(e_h)), v_e) * dx

    # Boundary conditions
    boundary_entities = dolfinx.mesh.locate_entities_boundary(
        mesh, 1, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_entities_sorted = np.sort(boundary_entities)

    eta_h = dolfinx.Function(V_e)

    # V_f_dolfin = dolfinx.FunctionSpace(mesh, element_f)
    # e_D = dolfinx.Function(V_f_dolfin)
    # e_h = dolfinx.Function(V_f_dolfin)

    fenicsx_error_estimation.estimate(
        eta_h, a_e, L_e, L_eta, N, boundary_entities_sorted) #, e_h=e_h, e_D=e_D)

    return eta_h


def estimate_residual(u_h, f):
    V = u_h.function_space
    mesh = V.mesh

    dx = ufl.Measure('dx', domain=mesh)
    dS = ufl.Measure('dS', domain=mesh)

    n = ufl.FacetNormal(mesh)
    h_T = ufl.CellDiameter(mesh)
    h_E = ufl.FacetArea(mesh)

    V_f = dolfinx.FunctionSpace(mesh, ("CG", k + 2))
    f_V_f = dolfinx.Function(V_f)
    f_V_f.interpolate(f)

    r = f_V_f + div(grad(u_h))
    J_h = jump(grad(u_h), -n)

    V_e = dolfinx.FunctionSpace(mesh, ("DG", 0))
    v_e = ufl.TestFunction(V_e)

    R = h_T**2 * inner(inner(r, r), v_e) * dx + avg(h_E) * \
                 inner(inner(J_h, J_h), avg(v_e)) * dS

    # Computation of local error indicator
    eta_h = dolfinx.Function(V_e)
    eta = dolfinx.fem.assemble_vector(R)[:]
    eta_h.vector.array[:] = eta
    return eta_h


def estimate_zz(u_h, f):
    V = u_h.function_space
    mesh = V.mesh

    dx = ufl.Measure('dx', domain=mesh, metadata={"quadrature_rule": "vertex"})

    W = dolfinx.VectorFunctionSpace(mesh, ("CG", 1), dim=2)

    # Global grad recovery
    w_h = ufl.TrialFunction(W)
    v_h = ufl.TestFunction(W)

    A = dolfinx.fem.assemble_matrix(inner(w_h, v_h) * dx)
    b = dolfinx.fem.assemble_vector(inner(grad(u_h), v_h) * dx)
    A.assemble()

    G_h = dolfinx.Function(W)

    options = PETSc.Options()
    options["ksp_type"] = "cg"
    options["ksp_view"] = None
    options["pc_type"] = "hypre"
    options["ksp_rtol"] = 1e-10
    options["pc_hypre_type"] = "boomeramg"

    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setOperators(A)
    solver.setFromOptions()
    solver.solve(b, G_h.vector)
    G_h.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                           mode=PETSc.ScatterMode.FORWARD)

    disc_zz = grad(u_h) - G_h

    # Computation of local error indicator
    V_e = dolfinx.FunctionSpace(mesh, ("DG", 0))
    v = ufl.TestFunction(V_e)

    eta_h = dolfinx.Function(V_e, name="eta_h")
    eta = dolfinx.fem.assemble_vector(inner(inner(disc_zz, disc_zz), v) * dx)
    eta_h.vector.array[:] = eta
    return eta_h


def estimate_wgo(eta_u, eta_z):
    # BW WGO estimator
    eta_u_vec = eta_u.vector.array
    eta_z_vec = eta_z.vector.array

    sum_eta_u = eta_u.vector.sum()
    sum_eta_z = eta_z.vector.sum()

    eta_w = dolfinx.Function(eta_u.function_space)
    eta_w_vec = ((sum_eta_z / (sum_eta_u + sum_eta_z)) * eta_u_vec) + \
                ((sum_eta_u / (sum_eta_u + sum_eta_z)) * eta_z_vec)
    eta_w.vector.setArray(eta_w_vec)
    return eta_w


def marking(eta_w):
    mesh = eta_w.function_space.mesh
    # Mark
    print('Marking...')
    assert(mesh.mpi_comm().size == 1)
    theta = 0.3

    eta_global = eta_w.vector.sum()
    cutoff = theta * eta_global

    sorted_cells = np.argsort(eta_w.vector.array)[::-1]
    rolling_sum = 0.0
    for j, e in enumerate(eta_w.vector.array[sorted_cells]):
        rolling_sum += e
        if rolling_sum > cutoff:
            breakpoint = j
            break

    refine_cells = sorted_cells[0:breakpoint + 1]
    indices = np.array(np.sort(refine_cells), dtype=np.int32)
    markers = np.zeros(indices.shape, dtype=np.int8)
    markers_tag = dolfinx.MeshTags(mesh, mesh.topology.dim, indices, markers)
    return markers_tag


if __name__ == "__main__":
    main()
