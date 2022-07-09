import numpy as np

import mpi4py.MPI as MPI
import petsc4py.PETSc as PETSc

import pandas

import dolfinx
from dolfinx.io import XDMFFile

import ufl
from ufl import avg, div, grad, inner, jump

import fenicsx_error_estimation.estimate
from fenicsx_error_estimation import create_interpolation


with XDMFFile(MPI.COMM_WORLD, "fine_mesh.xdmf", 'r') as fi:
    mesh = fi.read_mesh()


def u_exact(x):
    r = np.sqrt((x[0] - 1.) * (x[0] - 1.) + (x[1] - 1.) * (x[1] - 1.))
    theta = np.arctan2((x[1] - 1.), (x[0] - 1.)) + np.pi / 2.
    values = r**(2. / 3.) * np.sin((2. / 3.) * theta)
    values[np.where(np.logical_or(np.logical_and(np.isclose((x[0] - 1.), 0., atol=1e-10), (x[1] - 1.) < 0.),
                                  np.logical_and(np.isclose((x[1] - 1.), 0., atol=1e-10), (x[0] - 1.) < 0.)))] = 0.
    return values


def weight(x):  # Gaussian function to focus the goal functional on a particular region of the domain
    eps_f = 0.1
    center_x = 0.75
    center_y = 0.75
    r2 = (((x[0] - 1.) - center_x)**2 + ((x[1] - 1.) - center_y)**2) / eps_f**2.

    values = np.zeros_like(x[0])

    values = np.exp(- r2/10.)
    return values


V = dolfinx.FunctionSpace(mesh, ("CG", 3))
dx = ufl.Measure("dx", domain=mesh)

weight_V = dolfinx.Function(V)
weight_V.interpolate(weight)
u_exact_V = dolfinx.Function(V)
u_exact_V.interpolate(u_exact)

J_exact = dolfinx.fem.assemble_scalar(inner(weight_V, u_exact_V) * dx)
print(J_exact)
