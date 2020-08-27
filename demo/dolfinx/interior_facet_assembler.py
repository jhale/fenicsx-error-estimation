from ufl import avg, cos, div, dS, dx, grad, inner, jump, pi, sin
import ufl
from dolfinx.mesh import Mesh, create_mesh, locate_entities_boundary
from dolfinx.io import XDMFFile
from dolfinx.fem import (apply_lifting, assemble_matrix, assemble_vector,
                         locate_dofs_topological, set_bc)
from dolfinx.cpp.mesh import CellType
from dolfinx import DirichletBC, Function, FunctionSpace, RectangleMesh
import dolfinx
from petsc4py import PETSc
from mpi4py import MPI
import numpy as np

import cffi
ffi = cffi.FFI()

mesh = RectangleMesh(
    MPI.COMM_WORLD,
    [np.array([0, 0, 0]), np.array([1, 1, 0])], [3, 3],
    CellType.triangle, dolfinx.cpp.mesh.GhostMode.shared_facet)

V = FunctionSpace(mesh, ("DG", 1))

dS_ = dS(metadata={"quadrature_degree": 3})

v = ufl.TestFunction(V)
n = ufl.FacetNormal(mesh)

L = avg(v)*dS_

b = assemble_vector(L)
b.assemble()
print(b.array)

L_eta_form = dolfinx.jit.ffcx_jit(L)
L_kernel = L_eta_form.create_interior_facet_integral(-1).tabulate_tensor

V_dofmap = V.dofmap

tdim = mesh.topology.dim
f_to_c = mesh.topology.connectivity(tdim - 1, tdim)
c_to_f = mesh.topology.connectivity(tdim, tdim - 1)

mesh.topology.create_entity_permutations()
perms = mesh.topology.get_facet_permutations()
cell_info = mesh.topology.get_cell_permutation_info()

num_facets = mesh.topology.index_map(1).size_local

x = mesh.geometry.x
x_dofs = mesh.geometry.dofmap

local_facet = np.zeros(2, dtype=np.intc)
perm = np.zeros(2, dtype=np.uint8)
geometry_macro = np.zeros((2, 3, 2), dtype=np.float64)

b = np.zeros(V.dim, dtype=PETSc.ScalarType)
b_macro = np.zeros(6, dtype=PETSc.ScalarType)

for f in range(0, num_facets):
    cells = f_to_c.links(f)

    if len(cells) != 2:
        continue

    # Attached cells
    # Checked local_facet against C++
    for i in range(0, 2):
        facets = c_to_f.links(cells[i])
        index = np.where(facets == f)[0]
        local_facet[i] = index

    # Checked perm against C++
    perm[0] = perms[local_facet[0], cells[0]]
    perm[1] = perms[local_facet[1], cells[1]]

    # Pack geometry
    for j in range(0, 2):
        c = x_dofs.links(cells[j])
        for k in range(3):
            for l in range(2):
                geometry_macro[j, k, l] = x[c[k], l]

    b_macro.fill(0.0)
    L_kernel(ffi.cast("double *", ffi.from_buffer("double *", b_macro)),
             ffi.NULL,
             ffi.NULL,
             ffi.cast(
        "double *", ffi.from_buffer(geometry_macro)),
        ffi.cast("int *", ffi.from_buffer(local_facet)),
        ffi.cast("uint8_t *", ffi.from_buffer(perm)),
        0)

    b[V_dofmap.cell_dofs(cells[0])] += b_macro[0:3]
    b[V_dofmap.cell_dofs(cells[1])] += b_macro[3:]

print(b)
