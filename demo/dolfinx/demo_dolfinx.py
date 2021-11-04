# Copyright 2020, Jack S. Hale
# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np

import cffi
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
from dolfinx import cpp
from dolfinx import DirichletBC, Function, FunctionSpace, RectangleMesh
from dolfinx.cpp.mesh import CellType
from dolfinx.fem import (apply_lifting, assemble_matrix, assemble_vector, assemble_scalar, Form,
                         locate_dofs_topological, set_bc)
from dolfinx.fem.assemble import _create_cpp_form
from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities_boundary

import fenics_error_estimation.cpp
from fenics_error_estimation import estimate, create_interpolation

import ufl
from ufl import avg, cos, div, dS, dx, grad, inner, jump, pi, sin
from ufl.algorithms.elementtransformations import change_regularity

ffi = cffi.FFI()


def primal():
    mesh = RectangleMesh(
        MPI.COMM_WORLD,
        [np.array([0, 0, 0]), np.array([1, 1, 0])], [128, 128],
        CellType.triangle, dolfinx.cpp.mesh.GhostMode.shared_facet)

    element = ufl.FiniteElement("CG", ufl.triangle, 1)
    V = FunctionSpace(mesh, element)
    dx = ufl.Measure("dx", domain=mesh)

    x = ufl.SpatialCoordinate(mesh)
    f = 8.0 * pi**2 * sin(2.0 * pi * x[0]) * sin(2.0 * pi * x[1])

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx

    u0 = Function(V)
    u0.vector.set(0.0)
    facets = locate_entities_boundary(
        mesh, 1, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = locate_dofs_topological(V, 1, facets)
    bcs = [DirichletBC(u0, dofs)]

    problem = dolfinx.fem.LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    u = problem.solve()

    with XDMFFile(mesh.mpi_comm(), "output/u.xdmf", "w") as of:
        of.write_mesh(mesh)
        of.write_function(u)

    u_exact = sin(2.0 * pi * x[0]) * sin(2.0 * pi * x[1])
    error = mesh.mpi_comm().allreduce(assemble_scalar(inner(grad(u - u_exact), grad(u - u_exact)) * dx(degree=3)), op=MPI.SUM)
    print("True error: {}".format(np.sqrt(error)))

    return u


def estimate_primal(u_h):
    mesh = u_h.function_space.mesh
    dx = ufl.Measure("dx", domain=mesh.ufl_domain())
    dS = ufl.Measure("dS", domain=mesh.ufl_domain())

    element_f = ufl.FiniteElement("DG", ufl.triangle, 2)
    element_g = ufl.FiniteElement("DG", ufl.triangle, 1)
    element_e = ufl.FiniteElement("DG", ufl.triangle, 0)
    N = create_interpolation(element_f, element_g)

    V_f = ufl.FunctionSpace(mesh.ufl_domain(), element_f)
    e = ufl.TrialFunction(V_f)
    v = ufl.TestFunction(V_f)

    n = ufl.FacetNormal(mesh.ufl_domain())

    # Data
    x = ufl.SpatialCoordinate(mesh.ufl_domain())
    f = 8.0 * pi**2 * sin(2.0 * pi * x[0]) * sin(2.0 * pi * x[1])

    # Bilinear form
    a_e = inner(grad(e), grad(v)) * dx

    # Linear form
    V = ufl.FunctionSpace(mesh.ufl_domain(), u_h.ufl_element())
    L_e = inner(jump(grad(u_h), -n), avg(v)) * dS + inner(f + div((grad(u_h))), v) * dx

    # Error form
    V_e = dolfinx.FunctionSpace(mesh, element_e)
    e_h = ufl.Coefficient(V_f)
    v_e = ufl.TestFunction(V_e)
    L_eta = inner(inner(grad(e_h), grad(e_h)), v_e) * dx

    # Functions to store results
    eta_h = Function(V_e)
    V_f_dolfin = dolfinx.FunctionSpace(mesh, element_f)
    e_h = dolfinx.Function(V_f_dolfin)

    # Boundary conditions
    boundary_entities = locate_entities_boundary(
        mesh, 1, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_entities_sorted = np.sort(boundary_entities)

    V_f_dolfin = dolfinx.FunctionSpace(mesh, element_f)
    e_D = dolfinx.Function(V_f_dolfin)
    e_h = dolfinx.Function(V_f_dolfin)

    estimate(eta_h, a_e, L_e, L_eta, N, boundary_entities_sorted, e_h=e_h, e_D=e_D)

    # Ghost update is not strictly necessary on DG_0 space but left anyway
    eta_h.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    print("Bank-Weiser error from estimator: {}".format(np.sqrt(eta_h.vector.sum())))

    # Try assembling L_eta from e_h directly
    L_eta = inner(inner(grad(e_h), grad(e_h)), v_e) * dx
    eta_h_2 = assemble_vector(L_eta)
    eta_h_2.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    print("Bank-Weiser error from estimator: {}".format(np.sqrt(eta_h_2.sum())))

    with XDMFFile(mesh.mpi_comm(), "output/eta.xdmf", "w") as of:
        of.write_mesh(mesh)
        of.write_function(eta_h)


def estimate_primal_python(u_h):
    mesh = u_h.function_space.mesh
    ufl_mesh = mesh.ufl_domain()
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
    e = ufl.TrialFunction(V_f)
    v = ufl.TestFunction(V_f)

    n = ufl.FacetNormal(ufl_mesh)

    x = ufl.SpatialCoordinate(ufl_mesh)
    f = 8.0 * pi**2 * sin(2.0 * pi * x[0]) * sin(2.0 * pi * x[1])

    a_e = inner(grad(e), grad(v)) * dx
    L_e = inner(jump(grad(u_h), -n), avg(v)) * dS + inner(f + div((grad(u_h))), v) * dx

    V_e = ufl.FunctionSpace(ufl_mesh, element_e)
    e_h = ufl.Coefficient(V_f)
    v_e = ufl.TestFunction(V_e)
    L_eta = inner(inner(grad(e_h), grad(e_h)), v_e) * dx

    N = create_interpolation(element_f, element_g)

    a_form, _, _ = dolfinx.jit.ffcx_jit(mesh.mpi_comm(), a_e)
    L_form, _, _ = dolfinx.jit.ffcx_jit(mesh.mpi_comm(), L_e)
    L_eta_form, _, _ = dolfinx.jit.ffcx_jit(mesh.mpi_comm(), L_eta)

    cg_element_and_dofmap, _, _ = dolfinx.jit.ffcx_jit(mesh.mpi_comm(), element_f_cg)
    cg_dofmap = cg_element_and_dofmap[1]

    # Cell integral, no coefficients, no constants.
    a_kernel_cell = a_form.integrals(dolfinx.fem.IntegralType.cell)[0].tabulate_tensor_float64

    # Cell integral, one coefficient (CG1), no constants.
    L_kernel_cell = L_form.integrals(dolfinx.fem.IntegralType.cell)[0].tabulate_tensor_float64
    # Interior facet integral, one coefficient, no constant.
    L_kernel_interior = L_form.integrals(dolfinx.fem.IntegralType.interior_facet)[0].tabulate_tensor_float64

    # Cell integral, one coefficient (DG2), no constants.
    L_eta_kernel_cell = L_eta_form.integrals(dolfinx.fem.IntegralType.cell)[0].tabulate_tensor_float64

    # Construct local entity dof map
    cg_tabulate_entity_dofs = cg_dofmap.tabulate_entity_dofs
    cg_num_entity_dofs = np.frombuffer(ffi.buffer(
        cg_dofmap.num_entity_dofs, ffi.sizeof("int") * 4), dtype=np.intc)
    entity_dofmap = [np.zeros(i, dtype=np.intc) for i in cg_num_entity_dofs]

    # Begin unpacking data
    V_dolfin = u_h.function_space
    mesh = V_dolfin.mesh

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
    geometry = np.zeros((1, 3, 3))

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
    geometry_macro = np.zeros((2, 3, 3))

    # Connectivity
    tdim = mesh.topology.dim
    f_to_c = mesh.topology.connectivity(tdim - 1, tdim)
    c_to_f = mesh.topology.connectivity(tdim, tdim - 1)

    # Final error estimator calculation
    # Output
    eta_local = np.zeros(1, dtype=PETSc.ScalarType)

    for i in range(0, num_cells):
        c = x_dofs.links(i)

        # Pack geometry
        for j in range(3):
            for k in range(3):
                geometry[0, j, k] = x[c[j], k]

        # Pack coefficients
        coefficients[0, 0, :] = u[V_dofs.cell_dofs(i)]

        # Compared with DOLFIN-old, looks ok.
        A_local.fill(0.0)
        # No coefficients, no constants
        a_kernel_cell(ffi.cast("double *", ffi.from_buffer(A_local)), ffi.NULL,
                      ffi.NULL,
                      ffi.cast("double *", ffi.from_buffer(geometry)), ffi.NULL,
                      ffi.NULL)

        # Compared with DOLFIN-old, looks ok, although small difference at 4-5th s.f.
        b_local.fill(0.0)
        L_kernel_cell(ffi.cast("double *", ffi.from_buffer(b_local)),
                      ffi.cast("double *", ffi.from_buffer(coefficients)),
                      ffi.NULL,
                      ffi.cast("double *", ffi.from_buffer(geometry)), ffi.NULL,
                      ffi.NULL)

        # Compared with DOLFIN-old, looks ok.
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
            perm[0] = perms[cells[0] * len(facets) + local_facet[0]]
            perm[1] = perms[cells[1] * len(facets) + local_facet[1]]

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
                              ffi.cast("uint8_t *", ffi.from_buffer(perm)))
            # Assemble the relevant part of the macro cell tensor into the
            # local cell tensor.
            # TODO: Generalise
            index = np.where(cells == i)[0]
            offset = 0 if index == 0 else 6
            b_local += b_macro[offset:offset + 6]

        # Is one of the current cell's facets or vertices on the boundary?
        local_dofs = []
        for j, facet in enumerate(facets_for_cell):
            global_facet = boundary_facets[np.where(
                facet == boundary_facets)[0]]
            # If facet is not a boundary facet exit the loop
            if len(global_facet) == 0:
                continue

            # Local facet dofs
            cg_tabulate_entity_dofs(ffi.from_buffer(
                "int *", entity_dofmap[1]), 1, j)
            local_dofs.append(np.copy(entity_dofmap[1]))

        local_dofs = np.unique(np.array(local_dofs).reshape(-1))

        # Set Dirichlet boundary conditions on local system
        # TODO: Need to generalise to non-zero condition?
        for j in local_dofs:
            A_local[j, :] = 0.0
            A_local[:, j] = 0.0
            A_local[j, j] = 1.0
            b_local[j] = 0.0

        # Project
        A_0 = N.T @ A_local @ N
        b_0 = N.T @ b_local

        # Solve
        e_0 = np.linalg.solve(A_0, b_0)

        # Project back
        e_local = N @ e_0

        # Compute eta on cell
        eta_local.fill(0.0)
        L_eta_kernel_cell(ffi.cast("double *", ffi.from_buffer(eta_local)),
                          ffi.cast("double *", ffi.from_buffer(e_local)),
                          ffi.NULL,
                          ffi.cast(
                              "double *", ffi.from_buffer(geometry)),
                          ffi.cast("int *", ffi.from_buffer(local_facet)),
                          ffi.cast("uint8_t *", ffi.from_buffer(perm)))

        # Assemble
        dofs = V_e_dofs.cell_dofs(i)
        eta[dofs] = eta_local

    with XDMFFile(mesh.mpi_comm(), "output/eta_python.xdmf", "w") as of:
        of.write_mesh(mesh)
        of.write_function(eta_h)
    print("Bank-Weiser error from estimator: {}".format(np.sqrt(eta.sum())))


def main():
    u = primal()
    estimate_primal(u)
    if MPI.COMM_WORLD.size == 1:
        estimate_primal_python(u)


if __name__ == "__main__":
    main()
