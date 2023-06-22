import fenicsx_error_estimation
import numpy as np

import dolfinx
import ufl
from dolfinx.fem import (Constant,
                         Function,
                         FunctionSpace,
                         apply_lifting,
                         dirichletbc,
                         form,
                         locate_dofs_topological,
                         set_bc)
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from dolfinx.io import XDMFFile
from dolfinx.mesh import (CellType,
                          compute_incident_entities,
                          create_unit_square,
                          locate_entities_boundary)
from dolfinx.cpp.mesh import CellType
from ufl import (Coefficient,
                 Measure,
                 TestFunction,
                 TrialFunction,
                 avg,
                 div,
                 grad,
                 inner,
                 jump)

from mpi4py import MPI
from petsc4py import PETSc
import pandas as pd

import os
import shutil
import sys
sys.path.append("../")

from rational_schemes import BP_rational_approximation, BURA_rational_approximation
from FE_utils import mesh_refinement, parametric_problem

mesh_tri = create_unit_square(MPI.COMM_WORLD, 10, 10, cell_type=CellType.triangle)
mesh_quad = create_unit_square(MPI.COMM_WORLD, 10, 10, cell_type=CellType.quadrilateral)

element_tri = ufl.FiniteElement("CG", mesh_tri.ufl_cell(), 1)
V_tri = FunctionSpace(mesh_tri, element_tri)

element_quad = ufl.FiniteElement("CG", mesh_quad.ufl_cell(), 1)
V_quad = FunctionSpace(mesh_quad, element_quad)

def func(x):
  values = np.sin(2.0 * np.pi * x[0]) * np.sin(2.0 * np.pi * x[1])
  return values

f_tri = Function(V_tri)
f_quad = Function(V_quad)

f_tri.interpolate(func)
f_quad.interpolate(func)

f_tri_quad = Function(V_quad)
f_tri_quad.interpolate(f_tri)

with XDMFFile(mesh.comm, "./f_tri.xdmf", "w") as of:
  of.write_mesh(mesh_tri)
  of.write_function(f_tri)

with XDMFFile(mesh.comm, "./f_quad.xdmf", "w") as of:
  of.write_mesh(mesh_quad)
  of.write_function(f_quad)

with XDMFFile(mesh.comm, "./f_tri_quad.xdmf", "w") as of:
  of.write_mesh(mesh_quad)
  of.write_function(f_tri_quad)