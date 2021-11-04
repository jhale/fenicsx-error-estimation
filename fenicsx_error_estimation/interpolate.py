# Copyright 2019-2020, Jack S. Hale, Raphaël Bulle
# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np
import scipy as sp
from scipy import linalg

import basix


def create_interpolation(element_f, element_g):
    """Construct a projection operator.

    Given local nodal finite element spaces V_f (element_f) and V_g (element_g)
    construct an operator N that takes functions V_f to the special space V_f
    with L v_T = 0, where L is the Lagrangian (nodal) interpolant between V_f
    and V_g.
    """
    assert element_f.cell().cellname() == element_g.cell().cellname()
    assert element_f.degree() > element_g.degree()

    basix_cell = basix.cell.string_to_type(element_f.cell().cellname())

    basix_element_f = basix.create_element(basix.ElementFamily.P, basix_cell,
                                           element_f.degree(), basix._basixcpp.LagrangeVariant.equispaced, True)
    basix_element_g = basix.create_element(basix.ElementFamily.P, basix_cell,
                                           element_g.degree(), basix._basixcpp.LagrangeVariant.equispaced, True)

    # Interpolation element_f to element_g
    G_1 = basix.compute_interpolation_operator(basix_element_f, basix_element_g)
    # and from element_g to element_f
    G_2 = basix.compute_interpolation_operator(basix_element_g, basix_element_f)

    # Create a square matrix for interpolation from fine space to coarse one
    # with coarse space seen as a subspace of the fine one
    G = G_2 @ G_1

    # Change of basis to reduce N as a diagonal with only ones and zeros
    _, eigs, P = linalg.svd(G)

    assert(np.count_nonzero(np.isclose(eigs, 0.0)) == basix_element_f.dim - basix_element_g.dim)
    assert(np.count_nonzero(np.logical_not(np.isclose(eigs, 0.0))) == basix_element_g.dim)

    null_mask = np.less(np.abs(eigs), 0.5)
    # Reduce N to get a rectangular matrix in order to reduce the linear system
    # dimensions
    null_space = sp.compress(null_mask, P, axis=0)
    N_red = sp.transpose(null_space)
    assert(not np.all(np.iscomplex(N_red)))
    assert(np.linalg.matrix_rank(N_red) == basix_element_f.dim - basix_element_g.dim)

    return N_red
