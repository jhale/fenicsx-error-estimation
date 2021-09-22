# Copyright 2020, Jack S. Hale
# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import shutil

import numpy as np
import pickle as pkl

import cffi
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
from dolfinx import cpp
from dolfinx import DirichletBC, Function, FunctionSpace, VectorFunctionSpace, UnitCubeMesh
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
from ufl import avg, cos, div, dS, dx, grad, inner, jump, pi, sin
from ufl.algorithms.elementtransformations import change_regularity

ffi = cffi.FFI()


# Won't try to get it work with complex arithmetic at first
assert dolfinx.has_petsc_complex == False

def primal(k, V):
    mesh = V.mesh
    dx = ufl.Measure("dx", domain=mesh)

    x = ufl.SpatialCoordinate(mesh)
    f = 12.0 * pi**2 * sin(2.0 * pi * x[0]) * sin(2.0 * pi * x[1]) * sin(2.0 * pi * x[2])

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx(degree=k + 3)

    u0 = Function(V)
    u0.vector.set(0.0)
    facets = locate_entities_boundary(
        mesh, mesh.topology.dim - 1, lambda x: np.full(x.shape[1], True, dtype=bool))
    dofs = locate_dofs_topological(V, mesh.topology.dim - 1, facets)
    bcs = [DirichletBC(u0, dofs)]

    A = assemble_matrix(a, bcs=bcs)
    A.assemble()

    b = assemble_vector(L)
    apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)

    u = Function(V)
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    PETSc.Options()["ksp_type"] = "cg"
    PETSc.Options()["ksp_rtol"] = 1E-10
    PETSc.Options()["pc_type"] = "hypre"
    PETSc.Options()["pc_hypre_type"] = "boomeramg"
    PETSc.Options()["pc_hypre_boomeramg_strong_threshold"] = 0.5
    PETSc.Options()["pc_hypre_boomeramg_coarsen_type"] = "HMIS"
    PETSc.Options()["pc_hypre_boomeramg_interp_type"] = "ext+i"
    PETSc.Options()["pc_hypre_boomeramg_agg_nl"] = 4
    PETSc.Options()["pc_hypre_boomeramg_agg_num_paths"] = 2
    PETSc.Options()["pc_hypre_boomeramg_truncfactor"] = 0.35
    PETSc.Options()["ksp_monitor_true_residual"] = ""
    PETSc.Options()["ksp_view"] = ""
    solver.setOperators(A)
    solver.setFromOptions()
    solver.solve(b, u.vector)
    u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    return u

def exact_error(k, u_h):
    mesh = u_h.function_space.mesh
    dx = ufl.Measure("dx", domain=mesh.ufl_domain())

    element_e = ufl.FiniteElement("DG", ufl.tetrahedron, 0)
    V_e = FunctionSpace(mesh, element_e)
    v_e = ufl.TestFunction(V_e)

    # Exact solution
    x = ufl.SpatialCoordinate(mesh.ufl_domain())
    u_exact = sin(2.0 * pi * x[0]) * sin(2.0 * pi * x[1]) * sin(2.0 * pi * x[2])

    eta_h = Function(V_e)
    eta = assemble_vector(inner(inner(grad(u_h - u_exact), grad(u_h - u_exact)), v_e) * dx(degree=k + 3))[:]
    eta_h.vector.setArray(eta)

    #global_error = mesh.mpi_comm().allreduce(assemble_scalar(inner(grad(u_h - u_exact), grad(u_h - u_exact)) * dx(degree=k + 3)), op=MPI.SUM)
    return eta_h

def estimate_bw(k, k_g, k_f, u_h):
    mesh = u_h.function_space.mesh
    dx = ufl.Measure("dx", domain=mesh.ufl_domain())
    dS = ufl.Measure("dS", domain=mesh.ufl_domain())

    element_f = ufl.FiniteElement("DG", ufl.tetrahedron, k_f)
    element_g = ufl.FiniteElement("DG", ufl.tetrahedron, k_g)
    element_e = ufl.FiniteElement("DG", ufl.tetrahedron, 0)
    N = create_interpolation(element_f, element_g)

    V_f = ufl.FunctionSpace(mesh.ufl_domain(), element_f)
    e = ufl.TrialFunction(V_f)
    v = ufl.TestFunction(V_f)

    n = ufl.FacetNormal(mesh.ufl_domain())

    # Data
    x = ufl.SpatialCoordinate(mesh.ufl_domain())
    f = 12.0 * pi**2 * sin(2.0 * pi * x[0]) * sin(2.0 * pi * x[1]) * sin(2.0 * pi * x[2])

    # Bilinear form
    a_e = inner(grad(e), grad(v)) * dx

    # Linear form
    V = ufl.FunctionSpace(mesh.ufl_domain(), u_h.ufl_element())
    deg_T = max(k-2, k_f)
    deg_E = max(k-1, k_f)
    L_e = inner(jump(grad(u_h), -n), avg(v)) * dS(degree = deg_E) + inner(f + div((grad(u_h))), v) * dx(degree = deg_T)

    # Error form
    V_e = dolfinx.FunctionSpace(mesh, element_e)
    e_h = ufl.Coefficient(V_f)
    v_e = ufl.TestFunction(V_e)
    L_eta = inner(inner(grad(e_h), grad(e_h)), v_e) * dx

    # Functions to store results
    eta_h = Function(V_e)

    # Boundary conditions
    boundary_entities = locate_entities_boundary(
        mesh, mesh.topology.dim - 1, lambda x: np.full(x.shape[1], True, dtype=bool))
    boundary_entities_sorted = np.sort(boundary_entities)

    estimate(eta_h, u_h, a_e, L_e, L_eta, N, boundary_entities_sorted)

    # Ghost update is not strictly necessary on DG_0 space but left anyway
    eta_h.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    return eta_h

def estimate_residual(k, u_h):
    mesh = u_h.function_space.mesh
    ufl_domain = mesh.ufl_domain()
    
    # Data
    x = ufl.SpatialCoordinate(mesh.ufl_domain())
    f = 12.0 * pi**2 * sin(2.0 * pi * x[0]) * sin(2.0 * pi * x[1]) * sin(2.0 * pi * x[2])

    dx = ufl.Measure("dx", domain=ufl_domain)
    dS = ufl.Measure("dS", domain=ufl_domain)

    n = ufl.FacetNormal(mesh)
    h_T = ufl.CellDiameter(mesh)
    h_E = ufl.FacetArea(mesh)

    r = f + div(grad(u_h))
    J_h = jump(grad(u_h), -n)

    element_e = ufl.FiniteElement("DG", ufl.tetrahedron, 0)
    V_e = FunctionSpace(mesh, element_e)
    v_e = ufl.TestFunction(V_e)

    # Interior residual
    deg = max(0, k - 2)
    R_T = h_T**2 * inner(inner(r, r), v_e) * dx(degree = deg)
    eta_T = assemble_vector(R_T)[:]

    # Facets residual
    deg = max(0, k - 1)
    R_E = avg(h_E) * inner(inner(J_h, J_h), avg(v_e)) * dS(degree = deg)
    eta_E = assemble_vector(R_E)[:]

    eta_h = Function(V_e)
    eta = eta_T + eta_E
    eta_h.vector.setArray(eta)

    return eta_h

def estimate_zz(u_h):
    mesh = u_h.function_space.mesh
    ufl_domain = mesh.ufl_domain()

    dx = ufl.Measure("dx", domain=ufl_domain)

    # Recovered gradient construction
    W = VectorFunctionSpace(mesh, ('CG', 1))

    w_h = ufl.TrialFunction(W)
    v_h = ufl.TestFunction(W)

    a = Form(inner(w_h, v_h) * dx, {'quadrature_rule': 'vertex', 'representation': 'quadrature'})
    L = inner(grad(u_h), v_h) * dx

    A = assemble_matrix(a)
    A.assemble()
    b = assemble_vector(L)

    G_h = Function(W)
    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    PETSc.Options()["ksp_type"] = "cg"
    PETSc.Options()["ksp_rtol"] = 1E-10
    PETSc.Options()["pc_type"] = "hypre"
    PETSc.Options()["pc_hypre_type"] = "boomeramg"
    solver.setOperators(A)
    solver.setFromOptions()
    solver.solve(b, G_h.vector)
    G_h.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    # ZZ estimator
    disc_zz = grad(u_h) - G_h

    element_e = ufl.FiniteElement("DG", ufl.tetrahedron, 0)
    V_e = FunctionSpace(mesh, element_e)
    v_e = ufl.TestFunction(V_e)

    eta_h = Function(V_e)
    eta = assemble_vector(inner(inner(disc_zz, disc_zz), v_e) * dx)
    eta_h.vector.setArray(eta)
    return eta_h

def marking(eta):
    mesh = eta.function_space.mesh
    theta = 0.3

    eta_global = np.sum(eta.vector.array)
    cutoff = theta*eta_global

    sorted_cells = np.argsort(eta.vector.array)[::-1]
    rolling_sum = 0.0
    for i, e in enumerate(eta.vector.array[sorted_cells]):
        rolling_sum += e
        if rolling_sum > cutoff:
            breakpoint = i
            break

    refine_cells = sorted_cells[0:breakpoint]
    indices = np.array(np.sort(refine_cells), dtype=np.int32)
    markers = np.zeros(indices.shape, dtype=np.int8)
    markers_tag = dolfinx.MeshTags(mesh, mesh.topology.dim, indices, markers)
    
    return markers_tag

def main():
    k = 1
    OUTPUT_DIR = f'./output/P{str(k)}/'
    max_it = 15

    mesh = UnitCubeMesh(MPI.COMM_WORLD, 4, 4, 4)

    try:
        shutil.rmtree(OUTPUT_DIR)
    except:
        pass

    os.mkdir(OUTPUT_DIR)

    dirs = ['meshes', 'solutions', 'true_errors', 'bw_estimators', 'residual_estimators', 'zz_estimators']

    for d in dirs:
        os.mkdir(os.path.join(OUTPUT_DIR, d))

    results = {'dofs': [], 'true error': [], 'bw estimator': [], 'residual estimator': [], 'zz estimator': []}
    for i in range(max_it):
        times = {}
        print(f'step {i+1}')
        with XDMFFile(MPI.COMM_WORLD, os.path.join(OUTPUT_DIR, f"meshes/mesh_{str(i).zfill(4)}.xdmf"), "w") as fo:
            fo.write_mesh(mesh)

        element = ufl.FiniteElement("CG", ufl.tetrahedron, k)
        V = FunctionSpace(mesh, element)
        dofs = V.dofmap.index_map.size_global
        results['dofs'].append(dofs)

        print(f'step {i+1} SOLVE...')
        with Timer() as t:
            u = primal(k, V)
            times['primal solve'] = t.elapsed()[0]

        with XDMFFile(MPI.COMM_WORLD, os.path.join(OUTPUT_DIR, f"solutions/u_{str(i).zfill(4)}.xdmf"), "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(u)

        print(f'step {i+1} TRUE ERROR COMPUTATION...')
        with Timer() as t:
            eta = exact_error(k, u)
            times['true error'] = t.elapsed()[0]

        with XDMFFile(MPI.COMM_WORLD, os.path.join(OUTPUT_DIR, f"true_errors/u_{str(i).zfill(4)}.xdmf"), "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(eta)
        true_err = np.sqrt(sum(eta.vector.array))
        results['true error'].append(true_err)
        #results['global error'].append(np.sqrt(global_error))

        print(f'step {i+1} BW EST. COMPUTATION...')
        with Timer() as t:
            eta = estimate_bw(k, k, k + 1, u)
            times['bw estimator'] = t.elapsed()[0]

        with XDMFFile(MPI.COMM_WORLD, os.path.join(OUTPUT_DIR, f"bw_estimators/eta_{str(i).zfill(4)}.xdmf"), "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(eta)
        bw_est = np.sqrt(sum(eta.vector.array))
        results['bw estimator'].append(bw_est)

        print(f'step {i+1} RESIDUAL EST. COMPUTATION...')
        with Timer() as t:
            eta = estimate_residual(k, u)
            times['residual estimator'] = t.elapsed()[0]

        with XDMFFile(MPI.COMM_WORLD, os.path.join(OUTPUT_DIR, f"residual_estimators/eta_{str(i).zfill(4)}.xdmf"), "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(eta)
        residual_est = np.sqrt(sum(eta.vector.array))
        results['residual estimator'].append(residual_est)

        if k == 1:
            print(f'step {i+1} ZZ EST. COMPUTATION...')
            with Timer() as t:
                eta = estimate_zz(u)
                times['zz estimator'] = t.elapsed()[0]

            with XDMFFile(MPI.COMM_WORLD, os.path.join(OUTPUT_DIR, f"zz_estimators/eta_{str(i).zfill(4)}.xdmf"), "w") as fo:
                fo.write_mesh(mesh)
                fo.write_function(eta)
            zz_est = np.sqrt(sum(eta.vector.array))
            results['zz estimator'].append(zz_est)

        with open(os.path.join(OUTPUT_DIR, 'results.pkl'), 'wb') as of:
            pkl.dump(results, of)

        markers_tag = marking(eta)

        mesh.topology.create_entities(mesh.topology.dim - 1)
        refined_mesh = dolfinx.mesh.refine(mesh, cell_markers=markers_tag)

        mesh = refined_mesh

    print(times)

if __name__ == "__main__":
    main()
