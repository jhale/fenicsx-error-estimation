import itertools

import numpy as np
import pytest

from dolfin import *

from bank_weiser import local_interpolation_to_V0

@pytest.mark.parametrize('k,mesh', itertools.product([1, 2, 3, 4], [UnitIntervalMesh(5),
                                                      UnitSquareMesh(4, 4),
                                                      UnitCubeMesh.create(2, 2, 2, CellType.Type.tetrahedron)]))
def test_interpolation_operator(k, mesh):
    V_f = FunctionSpace(mesh, "DG", k + 1)
    V_g = FunctionSpace(mesh, "DG", k)

    # Various assertions build into function
    N = local_interpolation_to_V0(V_f, V_g) 
