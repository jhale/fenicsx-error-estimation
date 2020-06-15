## Copyright 2019-2020, Jack S. Hale, Raphaël Bulle
## SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np
import scipy as sp
from scipy import linalg
from dolfin import *

def create_interpolation(element_f, element_g, dof_list=None):
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

    if dof_list is None:
        dim_coarse = V_g.dim()
    else:
        dim_coarse = len(dof_list)
 
    assert(V_f_dim > dim_coarse)

    w = Function(V_f)

    # Get interpolation matrices from fine space to coarse one and conversely
    G_1 = PETScDMCollection.create_transfer_matrix(V_f, V_g).array()
    G_2 = PETScDMCollection.create_transfer_matrix(V_g, V_f).array()
    # Using "Function" prior to create_transfer_matrix, initialises PETSc for
    # unknown reason...

    if dof_list is not None:
        # Create a square matrix for interpolation from fine space to coarse
        # one with coarse space seen as a subspace of the fine one, the coarse
        # space being defined according to prescribed dof list
        R = np.eye(dim_coarse)
        new_list = np.setdiff1d(np.arange(dim_coarse), dof_list)
        R_red = np.delete(R, new_list, 0)
        G = (G_2@R_red.T)@(R_red@G_1)
    else:
        # Create a square matrix for interpolation from fine space to coarse one
        # with coarse space seen as a subspace of the fine one
        G = G_2@G_1

    #G[np.isclose(G, 0.0)] = 0.0

    # Change of basis to reduce N as a diagonal with only ones and zeros
    #eigs, P = np.linalg.eig(G)
    #P, eigs,_ = np.linalg.svd(G)
    #N_red = sp.linalg.null_space(G)
    _, eigs, P = linalg.svd(G)

    assert(np.count_nonzero(np.isclose(eigs, 0.0)) == V_f_dim - dim_coarse)
    assert(np.count_nonzero(np.logical_not(np.isclose(eigs, 0.0))) == dim_coarse)

    null_mask = np.less(np.abs(eigs), 0.5)
    # Reduce N to get a rectangular matrix in order to reduce the linear system
    # dimensions
    null_space = sp.compress(null_mask, P, axis=0)
    N_red = sp.transpose(null_space)
    assert(not np.all(np.iscomplex(N_red)))
    assert(np.linalg.matrix_rank(N_red) == V_f_dim - dim_coarse)
    return N_red
