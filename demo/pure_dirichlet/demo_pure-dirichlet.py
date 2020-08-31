# Copyright 2019-2020, Jack S. Hale, Raphaël Bulle
# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np

from dolfin import *
import fenics_error_estimation

parameters["ghost_mode"] = "shared_facet"

mesh = UnitSquareMesh(16, 16)
mesh = refine(mesh, redistribute=True)
mesh = refine(mesh, redistribute=True)

k = 3
V = FunctionSpace(mesh, "CG", k)

u = TrialFunction(V)
v = TestFunction(V)

f = Expression("8.0*pi*pi*sin(2.0*pi*x[0])*sin(2.0*pi*x[1])", degree=k + 3)

a = inner(grad(u), grad(v)) * dx
L = inner(f, v) * dx


class Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


bcs = DirichletBC(V, Constant(0.0), "on_boundary")

u_h = Function(V)
A, b = assemble_system(a, L, bcs=bcs)

solver = PETScLUSolver()
solver.solve(A, u_h.vector(), b)

element_f = FiniteElement("DG", triangle, k + 1)
element_g = FiniteElement("DG", triangle, k)

N = fenics_error_estimation.create_interpolation(element_f, element_g)

V_f = FunctionSpace(mesh, element_f)
e = TrialFunction(V_f)
v = TestFunction(V_f)

bc = DirichletBC(V_f, Constant(0.0), "on_boundary", "geometric")
n = FacetNormal(mesh)
a_e = inner(grad(e), grad(v)) * dx
L_e = inner(f + div(grad(u_h)), v) * dx + \
    inner(jump(grad(u_h), -n), avg(v)) * dS

e_h = fenics_error_estimation.estimate(a_e, L_e, N, bc)
error = norm(e_h, "H10")

# Computation of local error indicator
V_e = FunctionSpace(mesh, "DG", 0)
v = TestFunction(V_e)

eta_h = Function(V_e)
eta = assemble(inner(inner(grad(e_h), grad(e_h)), v) * dx)
eta_h.vector()[:] = eta

u_exact = Expression("sin(2.0*pi*x[0])*sin(2.0*pi*x[1])", degree=k + 3)

with XDMFFile("output/u_h.xdmf") as f:
    f.write_checkpoint(u_h, "u_h")

with XDMFFile("output/e_h.xdmf") as f:
    f.write_checkpoint(e_h, "Residual solution")

with XDMFFile("output/eta.xdmf") as f:
    f.write_checkpoint(eta_h, "Element-wise estimator")

error_bw = np.sqrt(eta_h.vector().sum())
error_exact = errornorm(u_exact, u_h, "H10")

print("Exact error: {}".format(error_exact))
print("Bank-Weiser error from estimator: {}".format(error_bw))


def test():
    assert(np.allclose(error_exact, error_bw, rtol=0.1))
