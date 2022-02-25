import numpy as np

import mpi4py.MPI as MPI
import petsc4py.PETSc as PETSc

import pandas

import dolfinx
from dolfinx import UnitSquareMesh
from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities

import ufl
from ufl import avg, div, grad, inner, jump

import fenicsx_error_estimation.estimate
from fenicsx_error_estimation import create_interpolation

k = 1
kappa_1 = 0.1
kappa_2 = 1.

def main():
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 4, 4)

    # Adaptive refinement loop
    results = []
    for i in range(0, 10):
        result = {}

        def u_exact(x):
            values = x[0]*x[1]*(x[0]-1.)*(x[1]-1.)
            values[np.where(np.greater(x[1], 0.5))] *= (1./kappa_1)
            values[np.where(np.less(x[1], 0.5))] *= (1./kappa_2)
            return values

        def f(x):
            return -2.*x[1]**2 + 2.*x[1] - 2.*x[0]**2 + 2.*x[0]

        def Omega_1(x):
            return np.greater(x[1], 0.5)
        
        def Omega_2(x):
            return np.less(x[1], 0.5)

        cells_1 = locate_entities(mesh, mesh.topology.dim, Omega_1)
        cells_2 = locate_entities(mesh, mesh.topology.dim, Omega_2)

        V_e = dolfinx.FunctionSpace(mesh, ('DG', 0))
        kappa = dolfinx.Function(V_e)
        kappa.x.array[cells_1] = np.full(len(cells_1), kappa_1)
        kappa.x.array[cells_2] = np.full(len(cells_2), kappa_2)

        print(f'STEP {i}')

        V = dolfinx.FunctionSpace(mesh, ("CG", k))
        u_exact_V = dolfinx.Function(V)
        u_exact_V.interpolate(u_exact)
        with XDMFFile(MPI.COMM_WORLD, f"output/u_exact_{str(i).zfill(4)}.xdmf", "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(u_exact_V)

        V_f = dolfinx.FunctionSpace(mesh, ('CG', 2))
        f_V_f = dolfinx.Function(V_f)
        f_V_f.interpolate(f)

        # Solve
        print('Solving...')
        u_h = solve(V, kappa, u_exact_V, f_V_f)
        with XDMFFile(MPI.COMM_WORLD, f"output/u_h_{str(i).zfill(4)}.xdmf", "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(u_h)

        # Estimate
        print("Estimating...")
        eta_h = estimate(u_h, kappa, f_V_f)
        with XDMFFile(MPI.COMM_WORLD, f"output/eta_hu_{str(i).zfill(4)}.xdmf", "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(eta_h)
        result['error_bw'] = np.sqrt(np.sum(eta_h.vector.array))

        # Exact local error
        dx = ufl.Measure("dx", domain=mesh.ufl_domain())
        V_e = eta_h.function_space
        eta_exact = dolfinx.Function(V_e, name="eta_exact")
        v = ufl.TestFunction(V_e)
        eta = dolfinx.fem.assemble_vector(inner(inner(kappa * grad(u_h - u_exact_V), grad(u_h - u_exact_V)), v) * dx)
        eta_exact.vector.setArray(eta)
        result['error'] = np.sqrt(np.sum(eta_exact.vector.array))
        with XDMFFile(MPI.COMM_WORLD, f"output/eta_exact_{str(i).zfill(4)}.xdmf", "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(eta_exact)

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
        mesh = dolfinx.mesh.refine(mesh, cell_markers=markers_tag)

        with XDMFFile(MPI.COMM_WORLD, f"output/mesh{str(i).zfill(4)}.xdmf", "w") as fo:
            fo.write_mesh(mesh)

        results.append(result)

    df = pandas.DataFrame.from_dict(results)
    df.to_pickle("output/results.pkl")
    print(df)


def solve(V, kappa, u_exact_V, f):
    mesh = V.mesh
    dx = ufl.Measure("dx", domain=mesh)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = inner(kappa * grad(u), grad(v)) * dx
    L = inner(f, v) * dx

    dbc = dolfinx.Function(V)
    facets = dolfinx.mesh.locate_entities_boundary(
        mesh, 1, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = dolfinx.fem.locate_dofs_topological(V, 1, facets)
    bcs = [dolfinx.DirichletBC(u_exact_V, dofs)]

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


def estimate(u_h, kappa, f):
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
    a_e = inner(kappa * grad(e), grad(v)) * dx

    # Linear form
    V = u_h.function_space
    r = f + div(kappa * grad(u_h))
    L_e = inner(r, v) * dx + inner(jump(kappa * grad(u_h), -n), avg(v)) * dS

    # Error form
    V_e = dolfinx.FunctionSpace(mesh, element_e)
    e_h = ufl.Coefficient(V_f)
    v_e = ufl.TestFunction(V_e)

    L_eta = inner(inner(kappa * grad(e_h), grad(e_h)), v_e) * dx

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
