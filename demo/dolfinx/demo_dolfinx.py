import numpy as np

import cffi
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
from dolfinx import DirichletBC, Function, FunctionSpace, RectangleMesh
from dolfinx.cpp.mesh import CellType
from dolfinx.fem import (apply_lifting, assemble_matrix, assemble_vector,
                         locate_dofs_topological, set_bc)
from dolfinx.io import XDMFFile
from dolfinx.mesh import Mesh, create_mesh, locate_entities_boundary

import ufl
from ufl import avg, cos, div, dS, dx, grad, inner, jump, pi, sin

ffi = cffi.FFI()


# Won't try to get it work with complex arithmetic at first
assert dolfinx.has_petsc_complex == False


def primal():
    mesh = RectangleMesh(
        MPI.COMM_WORLD,
        [np.array([0, 0, 0]), np.array([1, 1, 0])], [3, 3],
        CellType.triangle, dolfinx.cpp.mesh.GhostMode.shared_facet)

    element = ufl.FiniteElement("CG", ufl.triangle, 1)
    V = FunctionSpace(mesh, element)
    dx = ufl.Measure("dx", domain=mesh)

    x = ufl.SpatialCoordinate(mesh)
    f = 8.0*pi**2*sin(2.0*pi*x[0])*sin(2.0*pi*x[1])

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = inner(grad(u), grad(v))*dx
    L = inner(f, v)*dx

    u0 = Function(V)
    u0.vector.set(0.0)
    facets = locate_entities_boundary(
        mesh, 1, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = locate_dofs_topological(V, 1, facets)
    bcs = [DirichletBC(u0, dofs)]

    A = assemble_matrix(a, bcs=bcs)
    A.assemble()

    b = assemble_vector(L)
    apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)

    u = Function(V)
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setOperators(A)
    solver.solve(b, u.vector)

    with XDMFFile(mesh.mpi_comm(), "output/u.xdmf", "w") as of:
        of.write_mesh(mesh)
        of.write_function(u)

    return u


def estimate(u_h):
    ufl_mesh = ufl.Mesh(ufl.VectorElement("CG", ufl.triangle, 1))
    dx = ufl.Measure("dx", domain=ufl_mesh)
    dS = ufl.Measure("dS", domain=ufl_mesh)

    element_f = ufl.FiniteElement("DG", ufl.triangle, 2)
    # We need this for the local dof mapping. Not used for constructing a form.
    element_f_cg = ufl.FiniteElement("CG", ufl.triangle, 2)
    element_g = ufl.FiniteElement("DG", ufl.triangle, 1)
    # We will construct a dolfin.FunctionSpace for assembling the final computed estimator.
    element_e = ufl.FiniteElement("DG", ufl.triangle, 0)

    V = ufl.FunctionSpace(ufl_mesh, u_h.ufl_element())
    V_f = ufl.FunctionSpace(ufl_mesh, element_f)

    u = ufl.Coefficient(V)

    e = ufl.TrialFunction(V_f)
    v = ufl.TestFunction(V_f)

    n = ufl.FacetNormal(ufl_mesh)

    x = ufl.SpatialCoordinate(ufl_mesh)
    f = 8.0*pi**2*sin(2.0*pi*x[0])*sin(2.0*pi*x[1])

    a_e = inner(grad(e), grad(v))*dx
    L_e = inner(jump(grad(u), -n), avg(v))*dS + inner(f + div((grad(u))), v)*dx

    V_e = ufl.FunctionSpace(ufl_mesh, element_e)
    e_h = ufl.Coefficient(V_f)
    v_e = ufl.TestFunction(V_e)
    L_eta = inner(inner(grad(e_h), grad(e_h)), v_e)*dx

    N = np.load("interpolation.npy")

    a_form = dolfinx.jit.ffcx_jit(a_e)
    L_form = dolfinx.jit.ffcx_jit(L_e)
    L_eta_form = dolfinx.jit.ffcx_jit(L_eta)

    cg_dofmap = dolfinx.jit.ffcx_jit(element_f_cg)[1]

    # Cell integral, no coefficients, no constants.
    a_kernel_cell = a_form.create_cell_integral(-1).tabulate_tensor

    # Cell integral, one coefficient (CG1), no constants.
    L_kernel_cell = L_form.create_cell_integral(-1).tabulate_tensor
    # Interior facet integral, one coefficient, no constant.
    L_kernel_interior = L_form.create_interior_facet_integral(
        -1).tabulate_tensor

    # Cell integral, one coefficient (DG2), no constants.
    L_eta_kernel_cell = L_eta_form.create_cell_integral(-1).tabulate_tensor

    # Construct local entity dof map
    cg_tabulate_entity_dofs = cg_dofmap.tabulate_entity_dofs
    cg_num_entity_dofs = np.frombuffer(ffi.buffer(
        cg_dofmap.num_entity_dofs), dtype=np.intc)

    entity_dofmap = [np.zeros(i, dtype=np.intc) for i in cg_num_entity_dofs]

    # Begin unpacking data
    V_dolfin = u_h.function_space
    mesh = V_dolfin.mesh

    # DG2 space for debugging, can be removed once working
    V_f = FunctionSpace(mesh, element_f)
    e_h = Function(V_f)
    e = e_h.vector
    V_f_dofs = V_f.dofmap

    # Space for final assembly of error
    V_e = FunctionSpace(mesh, element_e)
    eta_h = Function(V_e)
    eta = eta_h.vector
    V_e_dofs = V_e.dofmap

    x = mesh.geometry.x
    x_dofs = mesh.geometry.dofmap
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local

    boundary_facets = locate_entities_boundary(
        mesh, 1, lambda x: np.ones(x.shape[1], dtype=bool))

    V_dofs = V_dolfin.dofmap
    u = u_h.vector.array

    # Output
    A_local = np.zeros((6, 6), dtype=PETSc.ScalarType)
    b_local = np.zeros(6, dtype=PETSc.ScalarType)
    # Input for cell integral
    # Geometry [restriction][num_dofs][gdim]
    coefficients = np.zeros((1, 1, 3), dtype=PETSc.ScalarType)
    geometry = np.zeros((1, 3, 2))

    # Interior facet integrals
    # Output for two adjacent cells
    b_macro = np.zeros(12, dtype=PETSc.ScalarType)
    # Input for interior facet integrals
    # Permutations (global)
    mesh.topology.create_entity_permutations()
    perms = mesh.topology.get_facet_permutations()
    cell_info = mesh.topology.get_cell_permutation_info()

    # Permutations (local)
    perm = np.zeros(2, dtype=np.uint8)

    # Local facets (local)
    local_facet = np.zeros(2, dtype=np.intc)

    # TODO: Generalise
    # Data [coefficient][restriction][dof]
    coefficients_macro = np.zeros((1, 2, 3), dtype=PETSc.ScalarType)
    # Geometry [restriction][num_dofs][gdim]
    geometry_macro = np.zeros((2, 3, 2))

    # Connectivity
    tdim = mesh.topology.dim
    f_to_c = mesh.topology.connectivity(tdim - 1, tdim)
    c_to_f = mesh.topology.connectivity(tdim, tdim - 1)
    # TODO: Check this is the right convention.
    local_f_to_v = {0: [1, 2], 1: [0, 2], 2: [0, 1]}

    # Final error estimator calculation
    # Output
    eta_local = np.zeros(1, dtype=PETSc.ScalarType)

    for i in range(0, num_cells):
        c = x_dofs.links(i)

        # Pack geometry
        for j in range(3):
            for k in range(2):
                geometry[0, j, k] = x[c[j], k]

        # Pack coefficients
        coefficients[0, 0, :] = u[V_dofs.cell_dofs(i)]

        # Compared with DOLFIN-old, looks ok.
        A_local.fill(0.0)
        # No coefficients, no constants
        a_kernel_cell(ffi.cast("double *", ffi.from_buffer(A_local)), ffi.NULL,
                      ffi.NULL,
                      ffi.cast("double *", ffi.from_buffer(geometry)), ffi.NULL,
                      ffi.NULL, 0)

        # Compared with DOLFIN-old, looks ok, although small difference at 4-5th s.f.
        b_local.fill(0.0)
        L_kernel_cell(ffi.cast("double *", ffi.from_buffer(b_local)),
                      ffi.cast("double *", ffi.from_buffer(coefficients)),
                      ffi.NULL,
                      ffi.cast("double *", ffi.from_buffer(geometry)), ffi.NULL,
                      ffi.NULL, 0)

        # NOTE: This interior facet part does not match with DOLFIN-old.
        # NOTE: Try implementing standard interior facet assembler and compare?

        # TODO: Would be nice to reimplement links for numba version.
        # Alternative seems to be manually using offsets.
        facets_for_cell = c_to_f.links(i)
        assert(len(facets_for_cell) == 3)
        for f in facets_for_cell:
            cells = f_to_c.links(f)
            assert(len(cells) == 1 or 2)
            # If there is no cell across the facet then it is an exterior facet
            if len(cells) != 2:
                continue

            # What is the local facet number [0, 1, ...] in the attached cells
            # for the facet of interest?
            for j in range(0, 2):
                facets = c_to_f.links(cells[j])
                assert(len(facets) == 3)
                index = np.where(facets == f)[0]
                local_facet[j] = index

            # Orientation
            perm[0] = perms[local_facet[0], cells[0]]
            perm[1] = perms[local_facet[1], cells[1]]

            # Pack geometry
            for j in range(0, 2):
                c = x_dofs.links(cells[j])
                for k in range(3):
                    for l in range(2):
                        geometry_macro[j, k, l] = x[c[k], l]

            # Pack coefficients.
            # TODO: Generalise.
            for j in range(0, 2):
                coefficients_macro[0, j, :] = u[V_dofs.cell_dofs(cells[j])]

            b_macro.fill(0.0)
            L_kernel_interior(ffi.from_buffer("double *", b_macro),
                              ffi.from_buffer("double *", coefficients_macro),
                              ffi.NULL,
                              ffi.cast(
                                  "double *", ffi.from_buffer(geometry_macro)),
                              ffi.cast("int *", ffi.from_buffer(local_facet)),
                              ffi.cast("uint8_t *", ffi.from_buffer(perm)),
                              0)
            # Assemble the relevant part of the macro cell tensor into the
            # local cell tensor.
            # TODO: Generalise
            index = np.where(cells == i)[0]
            offset = 0 if index == 0 else 6
            b_local += b_macro[offset:offset + 6]

        # NOTE: Not checked thoroughly passed here because previous not working.
        # Is one of the current cell's facets or vertices on the boundary?
        local_dofs = []
        for j, facet in enumerate(facets_for_cell):
            global_facet = boundary_facets[np.where(
                facet == boundary_facets)[0]]
            # If no local facet is on a boundary facet exit the loop
            if len(global_facet) == 0:
                continue

            # TODO: Would need to handle edge dofs in 3D.
            # Local facet dofs
            cg_tabulate_entity_dofs(ffi.from_buffer(
                "int *", entity_dofmap[1]), 1, j)
            local_dofs.append(np.copy(entity_dofmap[1]))

            # Local vertices attached to facet
            vertices = local_f_to_v[j]
            for k in range(2):
                cg_tabulate_entity_dofs(ffi.from_buffer(
                    "int *", entity_dofmap[0]), 0, vertices[k])
                local_dofs.append(np.copy(entity_dofmap[0]))

        local_dofs = np.unique(np.array(local_dofs).reshape(-1))

        # Set Dirichlet boundary conditions on local system
        # TODO: Need to generalise to non-zero condition?
        for j in local_dofs:
            A_local[j, :] = 0.0
            A_local[:, j] = 0.0
            A_local[j, j] = 1.0
            b_local[j] = 0.0

        # Project
        A_0 = N.T@A_local@N
        b_0 = N.T@b_local

        # Solve
        e_0 = np.linalg.solve(A_0, b_0)

        # Project back
        e_local = N@e_0

        # Assemble (temporary)
        dofs = V_f_dofs.cell_dofs(i)
        e.setValues(dofs, e_local)

        # Compute eta on cell
        eta_local.fill(0.0)
        L_eta_kernel_cell(ffi.cast("double *", ffi.from_buffer(eta_local)),
                          ffi.cast("double *", ffi.from_buffer(e_local)),
                          ffi.NULL,
                          ffi.cast(
                              "double *", ffi.from_buffer(geometry)), ffi.NULL,
                          ffi.NULL, 0)

        # Assemble
        dofs = V_e_dofs.cell_dofs(i)
        eta[dofs] = eta_local

    with XDMFFile(mesh.mpi_comm(), "output/eta.xdmf", "w") as of:
        of.write_mesh(mesh)
        of.write_function(eta_h)

def main():
    u = primal()
    estimate(u)


if __name__ == "__main__":
    main()
