import itertools

import numpy as np
import pytest

from dolfin import *

from fenics_error_estimation import create_interpolation

@pytest.mark.parametrize('k,cell_type', itertools.product([1, 2, 3, 4], [interval, triangle, tetrahedron]))
def test_interpolation_operator(k, cell_type):
    V_f = FiniteElement("DG", cell_type, k + 1)
    V_g = FiniteElement("DG", cell_type, k)

    # Various assertions build into function
    N = create_interpolation(V_f, V_g) 
