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
from dolfinx import DirichletBC, Function, FunctionSpace, VectorFunctionSpace, RectangleMesh
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
#assert dolfinx.has_petsc_complex == False

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

    def u_exact(x):
        r = np.sqrt(x[0] * x[0] + x[1] * x[1])
        theta = np.arctan2(x[1], x[0]) + np.pi/2.
        values = r**(2./3.) * np.sin((2./3.) * theta)
        values[np.where(np.logical_or(np.logical_and(np.isclose(x[0], 0., atol=1e-10), x[1] < 0.), np.logical_and(np.isclose(x[1], 0., atol=1e-10), x[0] < 0.)))] = 0.
        return values

    f = dolfinx.Function(V)     # Zero data

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx

    uD = Function(V)
    uD.interpolate(u_exact)
    facets = locate_entities_boundary(
        mesh, 1, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = locate_dofs_topological(V, mesh.topology.dim - 1, facets)
    bcs = [DirichletBC(uD, dofs)]

    problem = dolfinx.fem.LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    u = problem.solve()

    return u

def exact_error(k, u_h, dirichlet_osc=False):
    mesh = u_h.function_space.mesh
    dx = ufl.Measure("dx", domain=mesh.ufl_domain())
    ds = ufl.Measure("ds", domain=mesh.ufl_domain())

    element_f = ufl.FiniteElement("CG", ufl.triangle, k+1)
    V_f = FunctionSpace(mesh, element_f)

    def u_exact(x):
        r = np.sqrt(x[0] * x[0] + x[1] * x[1])
        theta = np.arctan2(x[1], x[0]) + np.pi/2.
        values = r**(2./3.) * np.sin((2./3.) * theta)
        values[np.where(np.logical_or(np.logical_and(np.isclose(x[0], 0., atol=1e-10), x[1] < 0.), np.logical_and(np.isclose(x[1], 0., atol=1e-10), x[0] < 0.)))] = 0.
        return values

    f = dolfinx.Function(V_f)     # Zero data

    u_f = ufl.TrialFunction(V_f)
    v_f = ufl.TestFunction(V_f)
    a = inner(grad(u_f), grad(v_f)) * dx
    L = inner(f, v_f) * dx

    uD = Function(V_f)
    uD.interpolate(u_exact)
    facets = locate_entities_boundary(
        mesh, 1, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = locate_dofs_topological(V_f, mesh.topology.dim - 1, facets)
    bcs = [DirichletBC(uD, dofs)]

    problem = dolfinx.fem.LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    u_exact = problem.solve()

    element_e = ufl.FiniteElement("DG", ufl.triangle, 0)
    V_e = FunctionSpace(mesh, element_e)
    v_e = ufl.TestFunction(V_e)

    eta_h = Function(V_e)
    eta = assemble_vector(inner(inner(grad(u_h - u_exact), grad(u_h - u_exact)), v_e) * dx(degree=k + 1))[:]

    n = ufl.FacetNormal(mesh)
    h_E = ufl.FacetArea(mesh)
    if dirichlet_osc:
        # Dirichlet oscillations
        uD = dolfinx.Function(V_f)
        uD.interpolate(u_exact)
        surface_grad_uD = grad(uD) - n * inner(grad(uD), n)

        surface_grad_u_h = grad(u_h) - n * inner(grad(u_h), n)
        R_D = h_E * inner(inner(surface_grad_u_h - surface_grad_uD, surface_grad_u_h - surface_grad_uD), v_e) * ds
        eta_D = assemble_vector(R_D)[:]
    else:
        eta_D = np.zeros_like(eta)

    eta_h.vector.setArray(eta + eta_D)
    return eta_h


def estimate_bw(k, u_h, kf=2, kg=1, dirichlet_est=False):
    mesh = u_h.function_space.mesh
    V = u_h.function_space
    dx = ufl.Measure("dx", domain=mesh.ufl_domain())
    dS = ufl.Measure("dS", domain=mesh.ufl_domain())

    element_f = ufl.FiniteElement("DG", ufl.triangle, kf)
    element_g = ufl.FiniteElement("DG", ufl.triangle, kg)
    element_e = ufl.FiniteElement("DG", ufl.triangle, 0)
    N = create_interpolation(element_f, element_g)

    V_f = ufl.FunctionSpace(mesh.ufl_domain(), element_f)
    e = ufl.TrialFunction(V_f)
    v = ufl.TestFunction(V_f)

    n = ufl.FacetNormal(mesh.ufl_domain())

    x = ufl.SpatialCoordinate(mesh)
    r = ufl.sqrt(x[0]**2 + x[1]**2)
    theta = ufl.mathfunctions.Atan2(x[1], x[0])

    u_exact = r**(2./3.)*ufl.sin((2./3.)*(theta + ufl.pi/2.))

    # Computation Dirichlet boundary data error
    V_f_global = FunctionSpace(mesh, element_f)
    if dirichlet_est:
        g = projection(u_exact, V_f_global) # Must use projection rather than interpolation to get correct results
        u_f = projection(u_h, V_f_global)
        e_D = Function(V_f_global)

        e_D.vector[:] = g.vector[:] - u_f.vector[:]
        with XDMFFile(MPI.COMM_WORLD, './e_D.xdmf', 'w') as of:
            of.write_mesh(mesh)
            of.write_function(e_D)
    else:
        e_D = Function(V_f_global)

    f = dolfinx.Function(V)     # Zero data

    element_dg = ufl.FiniteElement("DG", ufl.triangle, 0)
    V_dg = FunctionSpace(mesh, element_dg)

    f_dg = projection(f, V_dg)

    # Bilinear form
    a_e = inner(grad(e), grad(v)) * dx

    # Linear form
    V = ufl.FunctionSpace(mesh.ufl_domain(), u_h.ufl_element())
    L_e = inner(jump(grad(u_h), -n), avg(v)) * dS + inner(f_dg + div((grad(u_h))), v) * dx

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

    estimate(eta_h, e_D, a_e, L_e, L_eta, N, boundary_entities_sorted, e_h=e_h)

    # Ghost update is not strictly necessary on DG_0 space but left anyway
    eta_h.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    return eta_h

def estimate_residual(k, u_h, dirichlet_osc=False):
    V = u_h.function_space
    mesh = V.mesh
    ufl_domain = mesh.ufl_domain()

    def u_exact(x):
        r = np.sqrt(x[0] * x[0] + x[1] * x[1])
        theta = np.arctan2(x[1], x[0]) + np.pi/2.
        values = r**(2./3.) * np.sin((2./3.) * theta)
        values[np.where(np.logical_or(np.logical_and(np.isclose(x[0], 0., atol=1e-10), x[1] < 0.), np.logical_and(np.isclose(x[1], 0., atol=1e-10), x[0] < 0.)))] = 0.
        return values
    
    f = dolfinx.Function(V)     # Zero data

    dx = ufl.Measure("dx", domain=ufl_domain)
    ds = ufl.Measure("ds", domain=ufl_domain)
    dS = ufl.Measure("dS", domain=ufl_domain)

    n = ufl.FacetNormal(mesh)
    h_T = ufl.CellDiameter(mesh)
    h_E = ufl.FacetArea(mesh)

    r = f + div(grad(u_h))
    J_h = jump(grad(u_h), -n)

    element_e = ufl.FiniteElement("DG", ufl.triangle, 0)
    V_e = FunctionSpace(mesh, element_e)
    v_e = ufl.TestFunction(V_e)

    # Interior residual
    d = mesh.topology.dim
    R_T = h_T**d * inner(inner(r, r), v_e) * dx
    eta_T = assemble_vector(R_T)[:]

    # Facets residual
    R_E = avg(h_E) * inner(inner(J_h, J_h), avg(v_e)) * dS
    eta_E = assemble_vector(R_E)[:]

    if dirichlet_osc:
        element_D = ufl.FiniteElement("CG", ufl.triangle, k+1)

        V_D = FunctionSpace(mesh, element_D)
        # Dirichlet oscillations
        uD = dolfinx.Function(V_D)
        uD.interpolate(u_exact)
        #uD = projection(u_exact, V_D)
        surface_grad_uD = grad(uD) - n * inner(grad(uD), n)

        surface_grad_u_h = grad(u_h) - n * inner(grad(u_h), n)
        R_D = h_E * inner(inner(surface_grad_u_h - surface_grad_uD, surface_grad_u_h - surface_grad_uD), v_e) * ds
        eta_D = assemble_vector(R_D)[:]
    else:
        eta_D = np.zeros_like(eta_E)

    eta_h = Function(V_e)
    eta = eta_T + eta_E + eta_D
    eta_h.vector.setArray(eta)

    return eta_h

def estimate_zz(u_h, dirichlet_osc=False):
    mesh = u_h.function_space.mesh
    ufl_domain = mesh.ufl_domain()

    dx = ufl.Measure("dx", domain=ufl_domain)

    def u_exact(x):
        r = np.sqrt(x[0] * x[0] + x[1] * x[1])
        theta = np.arctan2(x[1], x[0]) + np.pi/2.
        values = r**(2./3.) * np.sin((2./3.) * theta)
        values[np.where(np.logical_or(np.logical_and(np.isclose(x[0], 0., atol=1e-10), x[1] < 0.), np.logical_and(np.isclose(x[1], 0., atol=1e-10), x[0] < 0.)))] = 0.
        return values

    # Recovered gradient construction
    W = VectorFunctionSpace(mesh, ('CG', 1))

    w_h = ufl.TrialFunction(W)
    v_h = ufl.TestFunction(W)

    a = Form(inner(w_h, v_h) * dx(metadata={'quadrature_rule': 'vertex', 'representation': 'quadrature'}))
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

    element_e = ufl.FiniteElement("DG", ufl.triangle, 0)
    V_e = FunctionSpace(mesh, element_e)
    v_e = ufl.TestFunction(V_e)

    eta_h = Function(V_e)
    eta_E = assemble_vector(inner(inner(disc_zz, disc_zz), v_e) * dx)[:]
    
    if dirichlet_osc:
        n = ufl.FacetNormal(mesh)
        h_E = ufl.FacetArea(mesh)
        ds = ufl.Measure("ds", domain=ufl_domain)
        element_D = ufl.FiniteElement("CG", ufl.triangle, 2)

        V_D = FunctionSpace(mesh, element_D)
        # Dirichlet oscillations
        uD = dolfinx.Function(V_D)
        uD.interpolate(u_exact)
        #uD = projection(u_exact, V_D)
        surface_grad_uD = grad(uD) - n * inner(grad(uD), n)

        surface_grad_u_h = grad(u_h) - n * inner(grad(u_h), n)
        R_D = h_E * inner(inner(surface_grad_u_h - surface_grad_uD, surface_grad_u_h - surface_grad_uD), v_e) * ds
        eta_D = assemble_vector(R_D)[:]
    else:
        eta_D = np.zeros_like(eta_E)

    eta = eta_E + eta_D
    eta_h.vector.setArray(eta)
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
    k = 1
    OUTPUT_DIR = f'./output/P{str(k)}/'
    max_it = 15


    with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", 'r') as fi:
        mesh = fi.read_mesh(name="Grid")
    try:
        shutil.rmtree(OUTPUT_DIR)
    except:
        pass

    os.mkdir(OUTPUT_DIR)

    dirs = ['meshes', 'solutions', 'true_errors', 'bw_estimators', 'residual_estimators', 'zz_estimators']

    for d in dirs:
        os.mkdir(os.path.join(OUTPUT_DIR, d))

    results = {'dofs': [], 'true error': [], 'true error D': [], 'residual estimator': [], 'residual estimator D': [], 'zz estimator': [], 'zz estimator D': []}
    for i in range(max_it):
        times = {}
        print(f'step {i+1}')
        with XDMFFile(MPI.COMM_WORLD, os.path.join(OUTPUT_DIR, f"meshes/mesh_{str(i).zfill(4)}.xdmf"), "w") as fo:
            fo.write_mesh(mesh)

        element = ufl.FiniteElement("CG", ufl.triangle, k)
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

        print(f'step {i+1} TRUE ERROR D COMPUTATION...')
        with Timer() as t:
            eta_err_D = exact_error(k, u, dirichlet_osc=True)
            times['true error D'] = t.elapsed()[0]

        with XDMFFile(MPI.COMM_WORLD, os.path.join(OUTPUT_DIR, f"true_errors/u_{str(i).zfill(4)}.xdmf"), "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(eta_err_D)
        true_err_D = np.sqrt(sum(eta_err_D.vector.array))
        results['true error D'].append(true_err_D)
        print('true error D=', true_err_D)
        
        print(f'step {i+1} TRUE ERROR COMPUTATION...')
        with Timer() as t:
            eta_err = exact_error(k, u, dirichlet_osc=False)
            times['true error'] = t.elapsed()[0]

        with XDMFFile(MPI.COMM_WORLD, os.path.join(OUTPUT_DIR, f"true_errors/u_{str(i).zfill(4)}.xdmf"), "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(eta_err)
        true_err = np.sqrt(sum(eta_err.vector.array))
        results['true error'].append(true_err)
        print('true error=', true_err)

        print(f'step {i+1} BW EST. D COMPUTATION...')
        for kg in range(0, 3):
            for kf in range(kg+1,5):
                print(f'kf = {kf}, kg = {kg}')
                if i == 0:
                    results[f'bw estimator {str(kf)} {str(kg)} D'] = []

                with Timer() as t:
                    eta_bw_D = estimate_bw(k, u, kf=kf, kg=kg, dirichlet_est=True)
                    times[f'bw estimator {str(kf)} {str(kg)} D'] = t.elapsed()[0]

                bw_est_D = np.sqrt(sum(eta_bw_D.vector.array))
                results[f'bw estimator {str(kf)} {str(kg)} D'].append(bw_est_D)

                with XDMFFile(MPI.COMM_WORLD, os.path.join(OUTPUT_DIR, f"bw_estimators/eta_{str(kf)}_{str(kg)}_D_{str(i).zfill(4)}.xdmf"), "w") as fo:
                    fo.write_mesh(mesh)
                    fo.write_function(eta_bw_D)
                print(f'bw estimator {str(kf)} {str(kg)} D =', bw_est_D)
        
        print(f'step {i+1} BW EST. COMPUTATION...')
        for kg in range(0, 3):
            for kf in range(kg+1,5):
                print(f'kf = {kf}, kg = {kg}')
                if i == 0:
                    results[f'bw estimator {str(kf)} {str(kg)}'] = []

                with Timer() as t:
                    eta_bw = estimate_bw(k, u, kf=kf, kg=kg, dirichlet_est=False)
                    times[f'bw estimator {str(kf)} {str(kg)}'] = t.elapsed()[0]

                bw_est = np.sqrt(sum(eta_bw.vector.array))
                results[f'bw estimator {str(kf)} {str(kg)}'].append(bw_est)

                with XDMFFile(MPI.COMM_WORLD, os.path.join(OUTPUT_DIR, f"bw_estimators/eta_{str(kf)}_{str(kg)}_{str(i).zfill(4)}.xdmf"), "w") as fo:
                    fo.write_mesh(mesh)
                    fo.write_function(eta_bw)
                print(f'bw estimator {str(kf)} {str(kg)} =', bw_est)

        print(f'step {i+1} RESIDUAL EST. D COMPUTATION...')
        with Timer() as t:
            eta_res_D = estimate_residual(k, u, dirichlet_osc=True)
            times['residual estimator D'] = t.elapsed()[0]

        with XDMFFile(MPI.COMM_WORLD, os.path.join(OUTPUT_DIR, f"residual_estimators/eta_{str(i).zfill(4)}.xdmf"), "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(eta_res_D)
        residual_est_D = np.sqrt(sum(eta_res_D.vector.array))
        results['residual estimator D'].append(residual_est_D)

        print(f'step {i+1} RESIDUAL EST. COMPUTATION...')
        with Timer() as t:
            eta_res = estimate_residual(k, u, dirichlet_osc=False)
            times['residual estimator'] = t.elapsed()[0]

        with XDMFFile(MPI.COMM_WORLD, os.path.join(OUTPUT_DIR, f"residual_estimators/eta_{str(i).zfill(4)}.xdmf"), "w") as fo:
            fo.write_mesh(mesh)
            fo.write_function(eta_res)
        residual_est = np.sqrt(sum(eta_res.vector.array))
        results['residual estimator'].append(residual_est)

        if k == 1:
            print(f'step {i+1} ZZ EST. COMPUTATION...')
            with Timer() as t:
                eta_zz = estimate_zz(u, dirichlet_osc=False)
                times['zz estimator'] = t.elapsed()[0]

            with XDMFFile(MPI.COMM_WORLD, os.path.join(OUTPUT_DIR, f"zz_estimators/eta_{str(i).zfill(4)}.xdmf"), "w") as fo:
                fo.write_mesh(mesh)
                fo.write_function(eta_zz)
            zz_est = np.sqrt(sum(eta_zz.vector.array))
            results['zz estimator'].append(zz_est)
            
            print(f'step {i+1} ZZ EST. D COMPUTATION...')
            with Timer() as t:
                eta_zz_D = estimate_zz(u, dirichlet_osc=True)
                times['zz estimator D'] = t.elapsed()[0]

            with XDMFFile(MPI.COMM_WORLD, os.path.join(OUTPUT_DIR, f"zz_estimators/eta_{str(i).zfill(4)}.xdmf"), "w") as fo:
                fo.write_mesh(mesh)
                fo.write_function(eta_zz_D)
            zz_est_D = np.sqrt(sum(eta_zz_D.vector.array))
            results['zz estimator D'].append(zz_est_D)


        with open(os.path.join(OUTPUT_DIR, 'results.pkl'), 'wb') as of:
            pkl.dump(results, of)
        with open(os.path.join(OUTPUT_DIR, 'times.pkl'), 'wb') as of:
            pkl.dump(times, of)

        markers_tag = marking(eta_res)

        mesh.topology.create_entities(mesh.topology.dim - 1)
        refined_mesh = dolfinx.mesh.refine(mesh, cell_markers=markers_tag)

        mesh = refined_mesh
    print(times)

if __name__ == "__main__":
    main()
