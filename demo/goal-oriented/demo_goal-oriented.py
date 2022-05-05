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
    with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", 'r') as fi:
        mesh = fi.read_mesh()

    # Adaptive refinement loop
    results = []
    for i in range(0, 20):
        result = {}

        def u_exact(x):
            r = np.sqrt((x[0] - 0.5) * (x[0] - 0.5) + (x[1] - 0.5) * (x[1] - 0.5))
            theta = np.arctan2(x[1], x[0]) + np.pi / 2.
            values = r**(2. / 3.) * np.sin((2. / 3.) * theta)
            values[np.where(np.logical_or(np.logical_and(np.isclose(x[0], 0., atol=1e-10), x[1] < 0.),
                                          np.logical_and(np.isclose(x[1], 0., atol=1e-10), x[0] < 0.)))] = 0.
            return values

        V = dolfinx.FunctionSpace(mesh, ("CG", k))
        u_exact_V = dolfinx.Function(V)
        u_exact_V.interpolate(u_exact)
        with XDMFFile(MPI.COMM_WORLD, f"output/u_exact_{str(i).zfill(4)}.xdmf", "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(u_exact_V)

        # Solve primal problem
        print(f'STEP {i}: solving primal problem...')
        u_h = solve_primal(V, u_exact_V)
        with XDMFFile(MPI.COMM_WORLD, f"output/u_h_{str(i).zfill(4)}.xdmf", "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(u_h)

        # Solve dual problem
        print(f'STEP {i}: solving dual problem...')
        z_h = solve_dual(V)
        with XDMFFile(MPI.COMM_WORLD, f"output/z_h_{str(i).zfill(4)}.xdmf", "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(z_h)

        # Estimate primal problem
        print(f"STEP {i}: estimating primal problem...")
        eta_u = estimate_primal(u_h)
        with XDMFFile(MPI.COMM_WORLD, f"output/eta_u_{str(i).zfill(4)}.xdmf", "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(eta_u)
        result['error_bw_u'] = np.sqrt(eta_u.vector.sum())

        # Estimate dual problem
        print(f"STEP {i}: estimating dual problem...")
        eta_z = estimate_dual(z_h)
        with XDMFFile(MPI.COMM_WORLD, f"output/eta_z_{str(i).zfill(4)}.xdmf", "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(eta_z)
        result['error_bw_u'] = np.sqrt(eta_z.vector.sum())

        # Calculated using P3 on a very fine adapted mesh, good to ~10 s.f.
        J_fine = 0.0326590077

        V_f = dolfinx.FunctionSpace(mesh, ("CG", k + 2))
        c = dolfinx.Function(V_f)
        c.interpolate(weight)
        with XDMFFile(MPI.COMM_WORLD, f"output/c_{str(i).zfill(4)}.xdmf", "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(c)

        dx = ufl.Measure("dx", domain=mesh)

        J_u_h = dolfinx.fem.assemble_scalar(inner(c, u_h) * dx)
        result['exact_error'] = np.abs(J_u_h - J_fine)

        # Necessary for parallel operation
        h_local = dolfinx.cpp.mesh.h(mesh, mesh.topology.dim, np.arange(
            0, mesh.topology.index_map(mesh.topology.dim).size_local, dtype=np.int32))
        h_max = MPI.COMM_WORLD.allreduce(h_local, op=MPI.MAX)
        result['h_max'] = h_max
        h_min = MPI.COMM_WORLD.allreduce(h_local, op=MPI.MIN)
        result['h_min'] = h_min
        result['num_dofs'] = V.dofmap.index_map.size_global

        # WGO estimator
        eta_u_vec = eta_u.vector.array
        eta_z_vec = eta_z.vector.array

        sum_eta_u = eta_u.vector.sum()
        sum_eta_z = eta_z.vector.sum()

        eta_w = dolfinx.Function(eta_u.function_space)
        eta_w_vec = ((sum_eta_z / (sum_eta_u + sum_eta_z)) * eta_u_vec) + \
                    ((sum_eta_u / (sum_eta_u + sum_eta_z)) * eta_z_vec)
        eta_w.vector.setArray(eta_w_vec)
        with XDMFFile(MPI.COMM_WORLD, f"output/eta_w_{str(i).zfill(4)}.xdmf", "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(eta_w)
        result['error_bw_w'] = np.sqrt(eta_u.vector.sum() * eta_z.vector.sum())

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

        # Refine
        print('Refining...')
        mesh = dolfinx.mesh.refine(mesh, cell_markers=markers_tag)

        with XDMFFile(MPI.COMM_WORLD, f"output/mesh{str(i).zfill(4)}.xdmf", "w") as fo:
            fo.write_mesh(mesh)

        results.append(result)

    df = pandas.DataFrame.from_dict(results)
    df.to_pickle("output/results.pkl")
    print(df)


def weight(x):  # Gaussian function to focus the goal functional on a particular region of the domain
    eps_f = 0.1
    center_x = 0.75
    center_y = 0.75
    r2 = ((x[0] - center_x)**2 + (x[1] - center_y)**2) / eps_f**2.

    values = np.zeros_like(x[0])

    values = np.exp(- r2)
    return values


def solve_primal(V, u_exact_V):
    mesh = V.mesh
    dx = ufl.Measure("dx", domain=mesh)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = inner(grad(u), grad(v)) * dx

    facets = dolfinx.mesh.locate_entities_boundary(
        mesh, 1, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = dolfinx.fem.locate_dofs_topological(V, 1, facets)
    bcs = [dolfinx.DirichletBC(u_exact_V, dofs)]

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

    V_c = dolfinx.FunctionSpace(mesh, ("CG", k + 2))
    c = dolfinx.Function(V_c)
    c.interpolate(weight)

    L = inner(c, v) * dx

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


def estimate_primal(u_h):
    V = u_h.function_space
    mesh = V.mesh
    ufl_domain = mesh.ufl_domain()

    dx = ufl.Measure('dx', domain=ufl_domain)
    dS = ufl.Measure('dS', domain=ufl_domain)

    element_f = ufl.FiniteElement("DG", ufl.triangle, 2)
    element_g = ufl.FiniteElement("DG", ufl.triangle, 1)
    element_e = ufl.FiniteElement("DG", ufl.triangle, 0)
    N = create_interpolation(element_f, element_g)

    V_f = ufl.FunctionSpace(ufl_domain, element_f)
    e = ufl.TrialFunction(V_f)
    v = ufl.TestFunction(V_f)

    n = ufl.FacetNormal(ufl_domain)

    # Bilinear form
    a_e = inner(grad(e), grad(v)) * dx

    f = dolfinx.Function(V)  # Zero data
    L = inner(f, v) * dx

    # Linear form
    L_e = L + inner(div(grad(u_h)), v) * dx + \
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

    fenicsx_error_estimation.estimate(
        eta_h, a_e, L_e, L_eta, N, boundary_entities_sorted)

    return eta_h


def estimate_dual(z_h):
    V = z_h.function_space
    mesh = V.mesh
    ufl_domain = mesh.ufl_domain()

    dx = ufl.Measure('dx', domain=ufl_domain)
    dS = ufl.Measure('dS', domain=ufl_domain)

    element_f = ufl.FiniteElement("DG", ufl.triangle, 2)
    element_g = ufl.FiniteElement("DG", ufl.triangle, 1)
    element_e = ufl.FiniteElement("DG", ufl.triangle, 0)
    N = create_interpolation(element_f, element_g)

    V_f = ufl.FunctionSpace(ufl_domain, element_f)
    e = ufl.TrialFunction(V_f)
    v = ufl.TestFunction(V_f)

    n = ufl.FacetNormal(ufl_domain)

    # Bilinear form
    a_e = inner(grad(e), grad(v)) * dx

    V_c = dolfinx.FunctionSpace(mesh, ("CG", k + 2))
    c = dolfinx.Function(V_c)
    c.interpolate(weight)

    L = inner(c, v) * dx

    # Linear form
    L_e = L + inner(div(grad(z_h)), v) * dx + \
              inner(jump(grad(z_h), -n), avg(v)) * dS

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

    fenicsx_error_estimation.estimate(
        eta_h, a_e, L_e, L_eta, N, boundary_entities_sorted)

    return eta_h


if __name__ == "__main__":
    main()
