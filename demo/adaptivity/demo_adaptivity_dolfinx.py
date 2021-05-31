import numpy as np

import mpi4py.MPI as MPI
import petsc4py.PETSc as PETSc

import dolfinx
from dolfinx.io import XDMFFile

import ufl
from ufl import avg, div, grad, inner, jump

import fenics_error_estimation.estimate
from fenics_error_estimation import create_interpolation

k=1


def main():
    with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", 'r') as fi:
        mesh = fi.read_mesh(name="Grid")

    # Adaptive refinement loop
    results = [["num_dofs", "hmax", "hmin", "error_bw", "error_exact"]]
    for i in range(0, 25):
        result = np.zeros(5)

        def u_exact(x):
            r = np.sqrt(x[0]*x[0] + x[1]*x[1])
            theta = np.arctan2(x[1], x[0]) + np.pi/2.
            values = r**(2./3.)*np.sin((2./3.)*theta)
            return values

        V = dolfinx.FunctionSpace(mesh, ("CG", k))
        u_exact_V = dolfinx.Function(V)
        u_exact_V.interpolate(u_exact)
        with XDMFFile(MPI.COMM_WORLD, f"output/u_exact_{str(i).zfill(4)}.xdmf", "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(u_exact_V)
        print("u_exact =", u_exact_V.vector[:])

        # Solve
        print('Solving...')
        u_h = solve(V, u_exact_V)
        with XDMFFile(MPI.COMM_WORLD, f"output/u_h_{str(i).zfill(4)}.xdmf", "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(u_h)

        # Estimate
        print("Estimating...")
        eta_h, e_h = estimate(u_h)
        print('eta_h.vector.array = ', eta_h.vector.array)
        with XDMFFile(MPI.COMM_WORLD, f"output/eta_hu_{str(i).zfill(4)}.xdmf", "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(eta_h)
        with XDMFFile(MPI.COMM_WORLD, f"output/e_h_{str(i).zfill(4)}.xdmf", "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(e_h)
        result[3] = np.sqrt(sum(eta_h.vector.array))

        # Exact local error
        dx = ufl.Measure("dx", domain=mesh.ufl_domain())
        V_e = eta_h.function_space
        eta_exact = dolfinx.Function(V_e, name="eta_exact")
        v = ufl.TestFunction(V_e)
        eta = dolfinx.fem.assemble_vector(inner(inner(grad(u_h - u_exact_V), grad(u_h - u_exact_V)), v) * dx)
        eta_exact.vector.setArray(eta)
        result[4] = np.sqrt(sum(eta_exact.vector.array))
        with XDMFFile(MPI.COMM_WORLD, f"output/eta_exact_{str(i).zfill(4)}.xdmf", "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(eta_exact)

        # Necessary for parallel operation
        h_local = dolfinx.cpp.mesh.h(mesh, mesh.topology.dim, np.arange(0, mesh.topology.index_map(mesh.topology.dim).size_local, dtype=np.int32))
        h_global = MPI.COMM_WORLD.allreduce(h_local, op=MPI.MAX)
        result[1] = h_global[0]
        h_global = MPI.COMM_WORLD.allreduce(h_local, op=MPI.MIN)
        result[2] = h_global[0]
        result[0] = V.dofmap.index_map.size_global

        # Mark
        print('Marking...')
        assert(mesh.mpi_comm().size == 1)
        theta = 0.3

        eta_global = sum(eta_h.vector.array)
        cutoff = theta*eta_global

        sorted_cells = np.argsort(eta_h.vector.array)[::-1]
        rolling_sum = 0.0
        for i, e in enumerate(eta_h.vector.array[sorted_cells]):
            rolling_sum += e
            if rolling_sum > cutoff:
                breakpoint = i
                break

        refine_cells = sorted_cells[0:breakpoint]
        indices = np.array(np.sort(refine_cells), dtype=np.int32)
        markers = np.zeros(indices.shape, dtype=np.int8)
        markers_tag = dolfinx.MeshTags(mesh, mesh.topology.dim, indices, markers)

        # Refine
        print('Refining...')
        mesh = dolfinx.mesh.refine(mesh, cell_markers=markers_tag)

        with XDMFFile(MPI.COMM_WORLD, f"output/mesh{str(i).zfill(4)}.xdmf", "w") as fo:
            fo.write(mesh)

        results = np.concatenate((results, result))
        print(results)
    np.save('output/results.npy', results)

def solve(V, u_exact_V):
    mesh = V.mesh
    dx = ufl.Measure("dx", domain=mesh)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    f = dolfinx.Function(V) # Zero data

    a = inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx

    facets = dolfinx.mesh.locate_entities_boundary(
            mesh, 1, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = dolfinx.fem.locate_dofs_topological(V, 1, facets)
    bcs = [dolfinx.DirichletBC(u_exact_V, dofs)]
    
    A = dolfinx.fem.assemble_matrix(a, bcs=bcs)
    A.assemble()
    A_mat = np.zeros((len(u_exact_V.vector.array), len(u_exact_V.vector.array)))
    for i in range(len(u_exact_V.vector.array)):
        for j in range(len(u_exact_V.vector.array)):
            A_mat[i, j] = A.getValues(i,j)

    print('A =', A_mat)

    b = dolfinx.fem.assemble_vector(L)
    dolfinx.fem.apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(b, bcs)

    print('b =', b.array)
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
    print('u_h =', u_h.vector.array)
    return u_h

def estimate(u_h):
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
    a_e = inner(grad(e), grad(v)) * dx
    
    # Linear form
    V = u_h.function_space
    f = dolfinx.Function(V) # Zero data
    r = f + div(grad(u_h))
    L_e = inner(r, v) * dx + inner(jump(grad(u_h), -n), avg(v)) * dS

    # Error form
    V_e = dolfinx.FunctionSpace(mesh, element_e)
    e_h = ufl.Coefficient(V_f)
    v_e = ufl.TestFunction(V_e)

    L_eta = inner(inner(grad(e_h), grad(e_h)), v_e) * dx

    V_f = dolfinx.FunctionSpace(mesh, element_f)

    # Function to store result
    eta_h = dolfinx.Function(V_e)
    e_h_f = dolfinx.Function(V_f)

    # Boundary conditions
    boundary_entities = dolfinx.mesh.locate_entities_boundary(
            mesh, 1, lambda x: np.ones(x.shape[1], dtype=bool))

    fenics_error_estimation.estimate(
            eta_h, e_h_f, u_h, a_e, L_e, L_eta, N, boundary_entities)

    return eta_h, e_h_f


if __name__=="__main__":
    main()
