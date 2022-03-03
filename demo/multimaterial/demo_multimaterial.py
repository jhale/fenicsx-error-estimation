import numpy as np

import mpi4py.MPI as MPI
import petsc4py.PETSc as PETSc

import pandas

import dolfinx
from dolfinx import UnitSquareMesh
from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities
from dolfinx.fem import locate_dofs_topological

import ufl
from ufl import avg, div, grad, inner, jump

import fenicsx_error_estimation.estimate
from fenicsx_error_estimation import create_interpolation

k = 1
kappa_1 = 1.
kappa_2 = 1.

def main():
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 8, 8)

    # Adaptive refinement loop
    results = []
    for i in range(0, 5):
        result = {}

        def u_exact(x):
            values = np.zeros_like(x[0])
            values[np.where(np.greater(x[1], 0.5))] = ((2. * x[1][np.where(np.greater(x[1], 0.5))] - 1.) * kappa_1 + kappa_2)/(kappa_1 + kappa_2)
            values[np.where(np.less(x[1], 0.5))] = (2. * x[1][np.where(np.less(x[1], 0.5))] * kappa_1)/(kappa_1 + kappa_2)
            return values

        V = dolfinx.FunctionSpace(mesh, ('CG', k))
        f = dolfinx.Function(V)

        V_e = dolfinx.FunctionSpace(mesh, ('DG', 0))
        
        def kappa_python(x):
            values = np.ones_like(x[1]) * kappa_1
            values[np.where(np.less(x[1], 0.5))] = kappa_2
            return values
        kappa = dolfinx.Function(V_e)
        kappa.interpolate(kappa_python)

        with XDMFFile(MPI.COMM_WORLD, f"output/kappa.xdmf", "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(kappa)

        print(f'STEP {i}')

        V = dolfinx.FunctionSpace(mesh, ("CG", k))
        V_f = dolfinx.FunctionSpace(mesh, ('CG', k+1))

        u_exact_f = dolfinx.Function(V_f)
        u_exact_f.interpolate(u_exact)
        with XDMFFile(MPI.COMM_WORLD, f"output/u_exact_{str(i).zfill(4)}.xdmf", "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(u_exact_f)

        u_dbc = dolfinx.Function(V)
        u_dbc.interpolate(u_exact)

        # Solve
        print('Solving...')
        u_h = solve(V, kappa, f, u_dbc)
        with XDMFFile(MPI.COMM_WORLD, f"output/u_h_{str(i).zfill(4)}.xdmf", "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(u_h)

        # Estimate
        print("Estimating...")
        eta_h = estimate(u_h, kappa, f, u_exact_f)
        with XDMFFile(MPI.COMM_WORLD, f"output/eta_h_{str(i).zfill(4)}.xdmf", "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(eta_h)
        result['error_bw'] = np.sqrt(np.sum(eta_h.vector.array))

        # Exact local error
        dx = ufl.Measure("dx", domain=mesh.ufl_domain())
        V_e = eta_h.function_space
        eta_exact = dolfinx.Function(V_e, name="eta_exact")
        v = ufl.TestFunction(V_e)
        eta = dolfinx.fem.assemble_vector(inner(inner(kappa * grad(u_h - u_exact_f), grad(u_h - u_exact_f)), v) * dx)
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


def solve(V, kappa, f, u_dbc):
    mesh = V.mesh
    dx = ufl.Measure("dx", domain=mesh)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = inner(kappa * grad(u), grad(v)) * dx
    #a = inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx

    def boundary_D(x):
        return np.logical_or(np.isclose(x[1], 0.), np.isclose(x[1], 1.))

    facets = dolfinx.mesh.locate_entities_boundary(
        mesh, 1, lambda x: boundary_D(x))

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


def estimate(u_h, kappa, f, u_dbc):
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
    #a_e = inner(kappa * grad(e), grad(v)) * dx
    a_e = inner(grad(e), grad(v)) * dx

    # Linear form
    V = u_h.function_space
    #r = f + div(kappa * grad(u_h))
    r = f + div(grad(u_h))
    #L_e = inner(r, v) * dx + inner(jump(kappa * grad(u_h), -n), avg(v)) * dS
    L_e = inner(r, v) * dx + inner(jump(grad(u_h), -n), avg(v)) * dS

    # Error form
    V_e = dolfinx.FunctionSpace(mesh, element_e)
    e_h = ufl.Coefficient(V_f)
    v_e = ufl.TestFunction(V_e)

    #L_eta = inner(inner(kappa * grad(e_h), grad(e_h)), v_e) * dx
    L_eta = inner(inner(grad(e_h), grad(e_h)), v_e) * dx

    # Boundary conditions
    def boundary_D(x):
        return np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], 1))
    #dofs = locate_dofs_topological(V, 1, boundary_D)

    boundary_entities = dolfinx.mesh.locate_entities_boundary(
        mesh, 1, lambda x: boundary_D(x))

    #boundary_entities = dolfinx.mesh.locate_entities_boundary(
    #    mesh, 1, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_entities_sorted = np.sort(boundary_entities)

    eta_h = dolfinx.Function(V_e)
    V_f_global = dolfinx.FunctionSpace(mesh, ('CG', k+1))
    e_h = dolfinx.Function(V_f_global)

    fenicsx_error_estimation.estimate(
        eta_h, a_e, L_e, L_eta, N, boundary_entities_sorted, e_h=e_h, e_D=u_dbc)
    print(eta_h.vector.array)

    return eta_h


if __name__ == "__main__":
    main()
