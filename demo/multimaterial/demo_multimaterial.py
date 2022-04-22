import numpy as np

import mpi4py.MPI as MPI
import petsc4py.PETSc as PETSc

import pandas

import dolfinx
from dolfinx import RectangleMesh
from dolfinx.cpp.mesh import CellType
from dolfinx.io import XDMFFile
from dolfinx.fem import locate_dofs_topological

import ufl
from ufl import avg, div, grad, inner, jump

import fenicsx_error_estimation.estimate
from fenicsx_error_estimation import create_interpolation

k = 1

# Parameters:
phi = np.pi / 2.
psi = 0.

p1 = 161.4476387975881
p3 = p1
p2 = 1.
p4 = p2

a1 = 0.1
alpha1 = np.pi / 4.
beta1 = -14.92256510455152


def main():
    mesh = RectangleMesh(
        MPI.COMM_WORLD,
        [np.array([-0.5, -0.5, 0]), np.array([0.5, 0.5, 0])], [32, 32],
        CellType.triangle)

    # Adaptive refinement loop
    results = []
    for i in range(0, 1):
        result = {}

        def u_exact(x):
            r = np.sqrt(x[0]**2 + x[1]**2)
            theta = np.arctan2(x[1], x[0]) + np.pi

            values = np.zeros_like(x[0])

            val4 = np.multiply(np.power(r, a1), np.cos((phi - alpha1) * a1) * np.cos((theta - phi - np.pi - beta1) * a1))
            values[np.where(np.less(theta, 2. * np.pi))] = val4[np.where(np.less(theta, 2. * np.pi))]

            val3 = np.multiply(np.power(r, a1), np.cos(beta1 * a1) * np.cos((theta - np.pi - alpha1) * a1))
            values[np.where(np.less(theta, np.pi + phi))] = val3[np.where(np.less(theta, np.pi + phi))]

            val2 = np.multiply(np.power(r, a1), np.cos(alpha1 * a1) * np.cos((theta - np.pi + beta1) * a1))
            values[np.where(np.less(theta, np.pi))] = val2[np.where(np.less(theta, np.pi))]

            val1 = np.multiply(np.power(r, a1), np.cos((phi - beta1) * alpha1) * np.cos((theta - phi + alpha1) * a1))
            values[np.where(np.less(theta, phi))] = val1[np.where(np.less(theta, phi))]
            return values

        V = dolfinx.FunctionSpace(mesh, ('CG', k))
        f = dolfinx.Function(V)

        V_e = dolfinx.FunctionSpace(mesh, ('DG', 0))

        def p_python(x):
            theta = np.arctan2(x[1], x[0]) + np.pi
            values = np.zeros_like(x[0])

            values[np.where(np.less(theta, 2. * np.pi))] = p4
            values[np.where(np.less(theta, np.pi + phi))] = p3
            values[np.where(np.less(theta, np.pi))] = p2
            values[np.where(np.less(theta, phi))] = p1
            return values

        p = dolfinx.Function(V_e)
        p.interpolate(p_python)

        with XDMFFile(MPI.COMM_WORLD, "output/p.xdmf", "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(p)

        print(f'STEP {i}')

        V = dolfinx.FunctionSpace(mesh, ("CG", k))
        V_f = dolfinx.FunctionSpace(mesh, ('CG', k + 1))

        u_exact_f = dolfinx.Function(V_f)
        u_exact_f.interpolate(u_exact)
        with XDMFFile(MPI.COMM_WORLD, f"output/u_exact_{str(i).zfill(4)}.xdmf", "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(u_exact_f)

        u_dbc = dolfinx.Function(V)
        u_dbc.interpolate(u_exact)

        # Solve
        print('Solving...')
        u_h = solve(V, p, f, u_dbc)
        with XDMFFile(MPI.COMM_WORLD, f"output/u_h_{str(i).zfill(4)}.xdmf", "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(u_h)

        # Estimate
        print("Estimating...")
        eta_h = estimate(u_h, p, f, u_exact_f)
        with XDMFFile(MPI.COMM_WORLD, f"output/eta_h_{str(i).zfill(4)}.xdmf", "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(eta_h)
        result['error_bw'] = np.sqrt(np.sum(eta_h.vector.array))

        # Exact local error
        dx = ufl.Measure("dx", domain=mesh.ufl_domain())
        V_e = eta_h.function_space
        eta_exact = dolfinx.Function(V_e, name="eta_exact")
        v = ufl.TestFunction(V_e)
        eta = dolfinx.fem.assemble_vector(inner(inner(p * grad(u_h - u_exact_f), grad(u_h - u_exact_f)), v) * dx)
        eta_exact.vector.setArray(eta)
        with XDMFFile(MPI.COMM_WORLD, f"output/eta_exact_{str(i).zfill(4)}.xdmf", "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(eta_exact)
        result['error'] = np.sqrt(np.sum(eta_exact.vector.array))

        # Necessary for parallel operation
        h_local = dolfinx.cpp.mesh.h(mesh, mesh.topology.dim, np.arange(
            0, mesh.topology.index_map(mesh.topology.dim).size_local, dtype=np.int32))
        h_max = MPI.COMM_WORLD.allreduce(h_local, op=MPI.MAX)
        result['h_max'] = h_max
        h_min = MPI.COMM_WORLD.allreduce(h_local, op=MPI.MIN)
        result['h_min'] = h_min
        result['num_dofs'] = V.dofmap.index_map.size_global

        # Mark
        print('Marking...')
        assert(mesh.mpi_comm().size == 1)
        theta = 0.3

        eta_global = sum(eta_h.vector.array)
        cutoff = theta * eta_global

        sorted_cells = np.argsort(eta_h.vector.array)[::-1]
        rolling_sum = 0.0
        for i, e in enumerate(eta_h.vector.array[sorted_cells]):
            rolling_sum += e
            if rolling_sum > cutoff:
                breakpoint = i
                break

        refine_cells = sorted_cells[0:breakpoint + 1]
        indices = np.array(np.sort(refine_cells), dtype=np.int32)
        markers = np.zeros(indices.shape, dtype=np.int8)
        markers_tag = dolfinx.MeshTags(mesh, mesh.topology.dim, indices, markers)

        # Refine
        print('Refining...')
        #mesh = dolfinx.mesh.refine(mesh, cell_markers=markers_tag)
        mesh = dolfinx.mesh.refine(mesh)

        with XDMFFile(MPI.COMM_WORLD, f"output/mesh{str(i).zfill(4)}.xdmf", "w") as fo:
            fo.write_mesh(mesh)

        results.append(result)

    df = pandas.DataFrame.from_dict(results)
    df.to_pickle("output/results.pkl")
    print(df)


def solve(V, p, f, u_dbc):
    mesh = V.mesh
    dx = ufl.Measure("dx", domain=mesh)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = inner(p * grad(u), grad(v)) * dx
    L = inner(f, v) * dx

    facets = dolfinx.mesh.locate_entities_boundary(
        mesh, 1, lambda x: np.ones(x.shape[1], dtype=bool))

    dofs = locate_dofs_topological(V, 1, facets)
    bcs = [dolfinx.DirichletBC(u_dbc, dofs)]

    A = dolfinx.fem.assemble_matrix(a, bcs=bcs)
    A.assemble()

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


def estimate(u_h, p, f, u_dbc):
    mesh = u_h.function_space.mesh
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
    a_e = inner(p * grad(e), grad(v)) * dx
    # a_e = inner(grad(e), grad(v)) * dx

    r = f + div(p * grad(u_h))
    J = jump(p * grad(u_h), -n)
    # r = f + div(grad(u_h))
    # J = jump(grad(u_h), -n)

    L_e = inner(r, v) * dx + inner(J, avg(v)) * dS

    # Error form
    V_e = dolfinx.FunctionSpace(mesh, element_e)
    e_h = ufl.Coefficient(V_f)
    v_e = ufl.TestFunction(V_e)

    # L_eta = inner(p * inner(grad(e_h), grad(e_h)), v_e) * dx
    L_eta = inner(inner(grad(e_h), grad(e_h)), v_e) * dx

    boundary_entities = dolfinx.mesh.locate_entities_boundary(
        mesh, 1, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_entities_sorted = np.sort(boundary_entities)

    eta_h = dolfinx.Function(V_e)
    V_f_global = dolfinx.FunctionSpace(mesh, ('CG', k + 1))
    e_h = dolfinx.Function(V_f_global)

    fenicsx_error_estimation.estimate(
        eta_h, a_e, L_e, L_eta, N, boundary_entities_sorted)
    print(eta_h.vector.array)

    return eta_h


if __name__ == "__main__":
    main()
