# Copyright 2019-2020, Jack S. Hale, Raphaël Bulle
# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np
import scipy as sp
from scipy import linalg
from dolfin import *


def create_interpolation(element_f, element_g):
    """Construct a projection operator.

    Given local nodal finite element spaces V_f (element_f) and V_g (element_g)
    construct an operator N that takes functions V_f to the special space V_f
    with L v_T = 0, where L is the Lagrangian (nodal) interpolant between V_f
    and V_g.
    """
    gdim = element_f.cell().geometric_dimension()

    if gdim == 1:
        mesh = UnitIntervalMesh(MPI.comm_self, 1)
    elif gdim == 2:
        mesh = Mesh(MPI.comm_self)
        editor = MeshEditor()
        editor.open(mesh, "triangle", 2, 2)

        editor.init_vertices(3)
        editor.init_cells(1)

        editor.add_vertex(0, np.array([0.0, 0.0]))
        editor.add_vertex(1, np.array([1.0, 0.0]))
        editor.add_vertex(2, np.array([0.0, 1.0]))
        editor.add_cell(0, np.array([0, 1, 2], dtype=np.uintp))

        editor.close()
    elif gdim == 3:
        mesh = Mesh(MPI.comm_self)
        editor = MeshEditor()
        editor.open(mesh, "tetrahedron", 3, 3)

        editor.init_vertices(4)
        editor.init_cells(1)

        editor.add_vertex(0, np.array([0.0, 0.0, 0.0]))
        editor.add_vertex(1, np.array([1.0, 0.0, 0.0]))
        editor.add_vertex(2, np.array([0.0, 1.0, 0.0]))
        editor.add_vertex(3, np.array([0.0, 0.0, 1.0]))
        editor.add_cell(0, np.array([0, 1, 2, 3], dtype=np.uintp))

        editor.close()
    else:
        raise NotImplementedError

    assert(mesh.ordered())
    assert(mesh.num_cells() == 1)

    V_f = FunctionSpace(mesh, element_f)
    V_g = FunctionSpace(mesh, element_g)

    V_f_dim = V_f.dim()
    V_g_dim = V_g.dim()

    assert(V_f_dim > V_g_dim)

    # Using "Function" prior to create_transfer_matrix, initialises PETSc for
    # unknown reason...
    # Looks like a no-op but actually required to ensure some internal data
    # structures are setup.
    w = Function(V_f)  # noqa: F841

    # Get interpolation matrices from fine space to coarse one and conversely
    G_1 = PETScDMCollection.create_transfer_matrix(V_f, V_g).array()
    G_2 = PETScDMCollection.create_transfer_matrix(V_g, V_f).array()

    # Create a square matrix for interpolation from fine space to coarse one
    # with coarse space seen as a subspace of the fine one
    G = G_2 @ G_1

    # Change of basis to reduce N as a diagonal with only ones and zeros
    _, vals, V = linalg.svd(G)

    eps = 1000. * np.finfo(np.float64).eps
    assert(np.count_nonzero(np.less(vals, eps)) == V_f_dim - V_g_dim)
    assert(np.count_nonzero(np.logical_not(np.less(vals, eps))) == V_g_dim)

    null_mask = np.less(np.abs(vals), eps)
    # Reduce V to get a rectangular matrix in order to reduce the linear system
    # dimensions
    null_space = sp.compress(null_mask, V, axis=0)
    N = sp.transpose(null_space)
    assert(not np.all(np.iscomplex(N)))
    assert(np.linalg.matrix_rank(N) == V_f_dim - V_g_dim)
    return N
