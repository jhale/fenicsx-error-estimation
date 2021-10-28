# Copyright 2020, Jack S. Hale
# SPDX-License-Identifier: LGPL-3.0-or-later
import argparse
import os
import shutil

import numpy as np
import pickle as pkl

import cffi
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
from dolfinx import cpp
from dolfinx import DirichletBC, Function, FunctionSpace, VectorFunctionSpace, UnitSquareMesh
from dolfinx.common import Timer
from dolfinx.cpp.mesh import CellType
from dolfinx.fem import (apply_lifting, assemble_matrix, assemble_vector, assemble_scalar, Form,
                         locate_dofs_topological, set_bc)
from dolfinx.fem.assemble import _create_cpp_form
from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities_boundary

import fenics_error_estimation.cpp
from fenics_error_estimation import estimate, create_interpolation

import ufl
from ufl import avg, div, dS, dx, grad, inner, jump, sqrt
from ufl.algorithms.elementtransformations import change_regularity

ffi = cffi.FFI()


# Won't try to get it work with complex arithmetic at first
assert dolfinx.has_petsc_complex == False

def projection(v, V_f):
    mesh = V_f.mesh
    ufl_domain = mesh.ufl_domain()
    dx = ufl.Measure("dx", domain=ufl_domain)

    w = ufl.TestFunction(V_f)
    u = ufl.TrialFunction(V_f)

    a = inner(u,w) * dx
    L = inner(v,w) * dx

    A = dolfinx.fem.assemble_matrix(a)
    A.assemble()
    b = dolfinx.fem.assemble_vector(L)

    options = PETSc.Options()
    options["ksp_type"] = "cg"
    options["ksp_view"] = None
    options["pc_type"] = "hypre"
    options["ksp_rtol"] = 1e-10
    options["pc_hypre_type"] = "boomeramg"
    options["ksp_monitor_true_residual"] = None

    v_f = dolfinx.Function(V_f)
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setOperators(A)
    solver.setFromOptions()
    solver.solve(b, v_f.vector)
    v_f.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                           mode=PETSc.ScatterMode.FORWARD)
    return v_f

def primal(k, V):
    mesh = V.mesh
    dx = ufl.Measure("dx", domain=mesh)

    x = ufl.SpatialCoordinate(mesh.ufl_domain())
    f = 8.0 *ufl.pi**2 *ufl.sin(2.0 *ufl.pi * x[0]) *ufl.sin(2.0 *ufl.pi * x[1])

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx

    uD = Function(V)
    facets = locate_entities_boundary(
        mesh, 1, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = locate_dofs_topological(V, mesh.topology.dim - 1, facets)
    bcs = [DirichletBC(uD, dofs)]

    A = dolfinx.fem.assemble_matrix(a)
    A.assemble()
    b = dolfinx.fem.assemble_vector(L)

    u = Function(V)
    options = PETSc.Options()
    options["ksp_type"] = "cg"
    options["ksp_view"] = None
    options["pc_type"] = "hypre"
    options["ksp_rtol"] = 1e-10
    options["pc_hypre_type"] = "boomeramg"
    options["pc_factor_mat_solver_type"] = "mumps"
    options["ksp_monitor_true_residual"] = None
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setOperators(A)
    solver.setFromOptions()
    solver.solve(b, u.vector)

    return u

def exact_error(k, u_h):
    mesh = u_h.function_space.mesh
    dx = ufl.Measure("dx", domain=mesh.ufl_domain())
    ds = ufl.Measure("ds", domain=mesh.ufl_domain())

    element_f = ufl.FiniteElement("CG", ufl.triangle, k+1)
    V_f = FunctionSpace(mesh, element_f)

    # Exact solution
    x = ufl.SpatialCoordinate(mesh.ufl_domain())
    u_exact =ufl.sin(2.0 *ufl.pi * x[0]) *ufl.sin(2.0 *ufl.pi * x[1])

    element_e = ufl.FiniteElement("DG", ufl.triangle, 0)
    V_e = FunctionSpace(mesh, element_e)
    v_e = ufl.TestFunction(V_e)

    eta_h = Function(V_e)
    eta = assemble_vector(inner(inner(grad(u_h - u_exact), grad(u_h - u_exact)), v_e) * dx(degree=k + 1))[:]

    eta_h.vector.setArray(eta)
    return eta_h


def estimate_bw(k, u_h):
    mesh = u_h.function_space.mesh
    V = u_h.function_space
    dx = ufl.Measure("dx", domain=mesh.ufl_domain())
    dS = ufl.Measure("dS", domain=mesh.ufl_domain())

    element_f = ufl.FiniteElement("DG", ufl.triangle, 2)
    element_g = ufl.FiniteElement("DG", ufl.triangle, 1)
    element_e = ufl.FiniteElement("DG", ufl.triangle, 0)
    N = create_interpolation(element_f, element_g)

    V_f = ufl.FunctionSpace(mesh.ufl_domain(), element_f)
    e = ufl.TrialFunction(V_f)
    v = ufl.TestFunction(V_f)

    n = ufl.FacetNormal(mesh.ufl_domain())

    x = ufl.SpatialCoordinate(mesh)
    f = 8.0 * ufl.pi**2 * ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])
   
    # Computation Dirichlet boundary data error
    V_f_global = FunctionSpace(mesh, element_f)     #TODO: remove this global space when the DBC is homogeneous = zero
    e_D = Function(V_f_global)

    element_dg = ufl.FiniteElement("DG", ufl.triangle, 0)
    V_dg = FunctionSpace(mesh, element_dg)

    # Bilinear form
    a_e = inner(grad(e), grad(v)) * dx

    # Linear form
    V = ufl.FunctionSpace(mesh.ufl_domain(), u_h.ufl_element())
    L_e = inner(jump(grad(u_h), -n), avg(v)) * dS + inner(div((grad(u_h))), v) * dx

    # Error form
    V_e = dolfinx.FunctionSpace(mesh, element_e)
    e_h = dolfinx.Function(V_f_global)
    v_e = ufl.TestFunction(V_e)
    L_eta = inner(inner(grad(e_h), grad(e_h)), v_e) * dx

    # Functions to store results
    eta_h = Function(V_e)
    V_f_dolfin = dolfinx.FunctionSpace(mesh, element_f)
    e_h = dolfinx.Function(V_f_dolfin)

    # Boundary conditions
    boundary_entities = locate_entities_boundary(
        mesh, 1, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_entities_sorted = np.sort(boundary_entities)

    estimate(eta_h, e_D, a_e, L_e, L_eta, N, boundary_entities_sorted)

    # Ghost update is not strictly necessary on DG_0 space but left anyway
    eta_h.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    return eta_h

def marking(eta):
    mesh = eta.function_space.mesh
    # Dorfler algorithm parameter
    theta = 0.5

    eta_global = np.sum(eta.vector.array)
    cutoff = theta*eta_global

    sorted_cells = np.argsort(eta.vector.array)[::-1]
    rolling_sum = 0.
    for i, e in enumerate(eta.vector.array[sorted_cells]):
        rolling_sum += e
        if rolling_sum > cutoff:
            breakpoint = i
            break
    
    refine_cells = sorted_cells[0:breakpoint]
    indices = np.array(np.sort(refine_cells), dtype = np.int32)
    markers = np.zeros(indices.shape, dtype = np.int8)
    markers_tag = dolfinx.MeshTags(mesh, mesh.topology.dim, indices, markers)
    return markers_tag

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_mpi_rank", default = 1, type=int, help="Number of mpi ranks (default 1).")
    parser.add_argument("--weak", default=False, action="store_true", help="Weak scaling test (default strong scaling).")

    args = parser.parse_args()
    rank = vars(args)["num_mpi_rank"]
    weak = vars(args)["weak"]

    k = 1   # FE order
    OUTPUT_DIR = f'./output/P{str(k)}/'

    if weak:
        mesh = UnitSquareMesh(MPI.COMM_WORLD, 1000*int(np.round(np.sqrt(rank), 0)), 1000*int(np.round(np.sqrt(rank), 0)))
    else:
        mesh = UnitSquareMesh(MPI.COMM_WORLD, 3000, 3000)

    try:
        os.mkdir(OUTPUT_DIR)
    except:
        pass

    if weak:
        OUTPUTFILE = 'times_weak.pkl'
    else:
        OUTPUTFILE = 'times_strong.pkl'

    try:
        with open(os.path.join(OUTPUT_DIR, OUTPUTFILE), 'rb') as fi:
            times = pkl.load(fi)
    except:
        times = {'rank': [], 'dofs': [], 'primal solve': [], 'bw estimator': [], 'bw estimator D': [], 'residual estimator': [], 'zz estimator': []}

    times['rank'].append(rank)
    element = ufl.FiniteElement("CG", ufl.triangle, k)
    V = FunctionSpace(mesh, element)
    dofs = V.dofmap.index_map.size_global
    times['dofs'].append(dofs)

    print(f'SOLVE...')
    with Timer() as t:
        u = primal(k, V)
        times['primal solve'].append(t.elapsed()[0])

    print(f'BW EST. COMPUTATION...')
    with Timer() as t:
        eta_bw = estimate_bw(k, u)
        times['bw estimator'].append(t.elapsed()[0])

    with open(os.path.join(OUTPUT_DIR, OUTPUTFILE), 'wb') as of:
        pkl.dump(times, of)

    print(times)

if __name__ == "__main__":
    main()
