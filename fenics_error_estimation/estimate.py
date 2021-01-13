# Copyright 2019-2020, Jack S. Hale, Raphaël Bulle
# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np
from mpi4py import MPI

import cffi

import dolfinx
import dolfinx.cpp
from ufl.algorithms.elementtransformations import change_regularity

import fenics_error_estimation.cpp

ffi = cffi.FFI()

def estimate(eta_h, u_h, a_e, L_e, L_eta, N, bc_entities):
    """Estimate the error using an implicit estimation strategy.
    """
    mesh = u_h.function_space.mesh

    a_e_ufc = dolfinx.jit.ffcx_jit(MPI.COMM_WORLD, a_e)
    a_e_dolfin = dolfinx.cpp.fem.create_form(ffi.cast("uintptr_t", a_e_ufc), [])
    assert(a_e_dolfin.num_coefficients() == 0)
    assert(len(a_e.constants()) == 0)
    a_e_dolfin.set_mesh(mesh)

    L_e_ufc = dolfinx.jit.ffcx_jit(MPI.COMM_WORLD, L_e)
    L_e_dolfin = dolfinx.cpp.fem.create_form(ffi.cast("uintptr_t", L_e_ufc), [])

    original_constants = [c._cpp_object for c in L_e.constants()]
    assert(L_e_dolfin.num_coefficients() == 1)
    L_e_dolfin.set_coefficient(0, u_h._cpp_object)
    L_e_dolfin.set_constants(original_constants)
    L_e_dolfin.set_mesh(mesh)

    L_eta_ufc = dolfinx.jit.ffcx_jit(MPI.COMM_WORLD, L_eta)
    L_eta_dolfin = dolfinx.cpp.fem.create_form(ffi.cast("uintptr_t", L_eta_ufc), [eta_h.function_space._cpp_object])
    L_eta_dolfin.set_mesh(mesh)

    element_f_cg = change_regularity(a_e.arguments()[0].ufl_element(), "CG")

    # Finite element for local solves
    element_ufc, dofmap_ufc = dolfinx.jit.ffcx_jit(MPI.COMM_WORLD, element_f_cg)
    element = dolfinx.cpp.fem.FiniteElement(ffi.cast("uintptr_t", element_ufc))
    dof_layout = dolfinx.cpp.fem.create_element_dof_layout(
        ffi.cast("uintptr_t", dofmap_ufc), mesh.topology.cell_type, [])

    fenics_error_estimation.cpp.projected_local_solver(
        eta_h._cpp_object, a_e_dolfin, L_e_dolfin, L_eta_dolfin, element, dof_layout, N, bc_entities)

def weighted_estimate(eta_uh, eta_zh):
    pass
    # eta_uh_vec = eta_uh.vector()
    # eta_zh_vec = eta_zh.vector()

    # sum_eta_uh = eta_uh_vec.sum()
    # sum_eta_zh = eta_zh_vec.sum()

    # eta_wh = Function(eta_uh.function_space(), name="eta")
    # eta_wh.vector()[:] = ((sum_eta_zh / (sum_eta_uh + sum_eta_zh)) * eta_uh_vec) + \
    #                     ((sum_eta_uh / (sum_eta_uh + sum_eta_zh)) * eta_zh_vec)

    # return eta_wh
