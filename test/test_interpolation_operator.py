import itertools

import pytest

import ufl

from fenics_error_estimation import create_interpolation


@pytest.mark.parametrize('k,cell_type', itertools.product([1, 2, 3, 4], [ufl.interval, ufl.triangle, ufl.tetrahedron]))
def test_interpolation_operator(k, cell_type):
    V_f = ufl.FiniteElement("Lagrange", cell_type, k + 1)
    V_g = ufl.FiniteElement("Lagrange", cell_type, k)

    # Various assertions build into function
    create_interpolation(V_f, V_g)
