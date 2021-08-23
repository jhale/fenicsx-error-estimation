import dolfinx
import dolfinx.io
from dolfinx.cpp.mesh import CellType
import ufl
from mpi4py import MPI
import numpy as np
import math

'''
# Single tetrahedron mesh (from: https://gist.github.com/jorgensd/21ddb252a5741e5d483206fbf32f88fa)
gdim = 3
shape = "tetrahedron"
degree = 1

cell = ufl.Cell(shape, geometric_dimension=gdim)
domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, degree))

x = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1]])
cells = np.array([[0, 1, 2, 3]], dtype=np.int32)
mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, x, domain)
mesh.topology.create_connectivity_all()
facets = np.arange(mesh.topology.index_map(mesh.topology.dim - 1).size_local, dtype=np.int32)
mt = dolfinx.MeshTags(mesh, mesh.topology.dim - 1, facets, facets)
'''
# Single triangle mesh
gdim = 2
shape = "triangle"
degree = 1
cell = ufl.Cell(shape, geometric_dimension=gdim)
domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, degree))

x = np.array([[0., 0., 0.], [0., 1., 0.], [1., 1., 0.]])
cells = np.array([[0, 1, 2]], dtype=np.int64)
mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, x, domain)

element_cg = ufl.FiniteElement('CG', cell, 2)
element_dg = ufl.FiniteElement('DG', cell, 2)
V_cg = dolfinx.FunctionSpace(mesh, element_cg)
V_dg = dolfinx.FunctionSpace(mesh, element_dg)

dof_coordinates = V_cg.tabulate_dof_coordinates()
num_dof = V_cg.dofmap.index_map.size_global
print(f'CG -> DG')
for i, dofcoord in zip(range(num_dof), dof_coordinates):
    v_cg = dolfinx.Function(V_cg)
    v_cg.vector.array[i] = 1.

    for j in range(num_dof):
        v_dg = dolfinx.Function(V_dg)
        v_dg.vector.array[j] = 1.

        if np.isclose(v_dg.eval(dofcoord, 0)[0], 1.):
            print(f' {i} -> {j}')
