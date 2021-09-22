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


def _create_form(form, form_compiler_parameters: dict = {}, jit_parameters: dict = {}):
    """Create form without concrete Function Space"""
    sd = form.subdomain_data()
    subdomains, = list(sd.values())
    domain, = list(sd.keys())
    mesh = domain.ufl_cargo()
    if mesh is None:
        raise RuntimeError("Expecting to find a Mesh in the form.")

    ufc_form = dolfinx.jit.ffcx_jit(
        mesh.mpi_comm(),
        form,
        form_compiler_parameters=form_compiler_parameters,
        jit_parameters=jit_parameters)[0]

    original_coefficients = form.coefficients()

    coeffs = []
    for i in range(ufc_form.num_coefficients):
        try:
            coeffs.append(original_coefficients[ufc_form.original_coefficient_position[i]]._cpp_object)
        except AttributeError:
            coeffs.append(None)

    # For every argument in form extract its function space
    function_spaces = []
    for func in form.arguments():
        try:
            function_spaces.append(func.ufl_function_space()._cpp_object)
        except AttributeError:
            pass

    subdomains = {dolfinx.cpp.fem.IntegralType.cell: subdomains.get("cell"),
                  dolfinx.cpp.fem.IntegralType.exterior_facet: subdomains.get("exterior_facet"),
                  dolfinx.cpp.fem.IntegralType.interior_facet: subdomains.get("interior_facet"),
                  dolfinx.cpp.fem.IntegralType.vertex: subdomains.get("vertex")}

    # Prepare dolfinx.cpp.fem.Form and hold it as a member
    ffi = cffi.FFI()
    form = dolfinx.cpp.fem.create_form(ffi.cast("uintptr_t", ffi.addressof(ufc_form)),
                                       function_spaces, coeffs,
                                       [c._cpp_object for c in form.constants()], subdomains, mesh)

    return form


def estimate(eta_h, u_h, e_D, a_e, L_e, L_eta, N, bc_entities, e_h=None):
    """Estimate the error using the Bank-Weiser implicit estimation strategy.
    """
    mesh = u_h.function_space.mesh
    mpi_comm = mesh.mpi_comm()

    a_e_dolfin = _create_form(a_e)
    L_e_dolfin = _create_form(L_e)
    print('TEST1')
    L_eta_dolfin = _create_form(L_eta)
    print('TEST2')
    element_f_cg = change_regularity(a_e.arguments()[0].ufl_element(), "CG")

    # Finite element for local solves
    cg_element_and_dofmap, _, _ = dolfinx.jit.ffcx_jit(MPI.COMM_WORLD, element_f_cg)
    cg_element = cg_element_and_dofmap[0]
    cg_dofmap = cg_element_and_dofmap[1]

    element = dolfinx.cpp.fem.FiniteElement(ffi.cast("uintptr_t", ffi.addressof(cg_element)))
    dof_layout = dolfinx.cpp.fem.create_element_dof_layout(
        ffi.cast("uintptr_t", ffi.addressof(cg_dofmap)), mesh.topology.cell_type, [])

    if e_h is None:
        # This version of the function does not modify the second argument.
        fenics_error_estimation.cpp.projected_local_solver_no_error_solution(
            eta_h._cpp_object, eta_h._cpp_object, e_D._cpp_object, a_e_dolfin, L_e_dolfin, L_eta_dolfin, element, dof_layout, N, bc_entities)
    else:
        fenics_error_estimation.cpp.projected_local_solver_error_solution(
            eta_h._cpp_object, e_h._cpp_object, e_D._cpp_object, a_e_dolfin, L_e_dolfin, L_eta_dolfin, element, dof_layout, N, bc_entities)


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
