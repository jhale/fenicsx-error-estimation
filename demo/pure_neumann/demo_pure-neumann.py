## Copyright 2019-2020, Jack S. Hale, Raphaël Bulle
## SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np

from dolfin import *
import fenics_error_estimation

parameters["ghost_mode"] = "shared_facet"

mesh = UnitSquareMesh(128, 128)

k = 2
element = FiniteElement("CG", triangle, k)
V = FunctionSpace(mesh, element)

u = TrialFunction(V)
v = TestFunction(V)

f = Expression(
    "(2*pow(2*pi,2)+1)*sin(2*pi*x[0]-0.5*pi)*sin(2*pi*x[1]-0.5*pi)", degree=k + 6)
g = Constant(0.0)

a = inner(grad(u), grad(v))*dx + inner(u, v)*dx
L = inner(f, v)*dx

u_h = Function(V)
A, b = assemble_system(a, L)

solver = PETScLUSolver()
solver.solve(A, u_h.vector(), b)

element_f = FiniteElement("DG", triangle, k + 1)
element_g = FiniteElement("DG", triangle, k)

N = fenics_error_estimation.create_interpolation(element_f, element_g)

V_f = FunctionSpace(mesh, element_f)
e = TrialFunction(V_f)
v = TestFunction(V_f)

n = FacetNormal(mesh)
# Local Bank-Weiser error estimation problems
a_e = inner(grad(e), grad(v))*dx
L_e = inner(f + div(grad(u_h)), v)*dx + \
      inner(jump(grad(u_h), -n), avg(v))*dS + \
      inner(g - dot(grad(u_h), n), v)*ds

e_h = fenics_error_estimation.estimate(a_e, L_e, N)
error = norm(e_h, "H10")

# Computation of local error indicator
V_e = FunctionSpace(mesh, "DG", 0)
v = TestFunction(V_e)

eta_h = Function(V_e)
eta = assemble(inner(inner(grad(e_h), grad(e_h)), v)*dx)
eta_h.vector()[:] = eta

u_exact = Expression(
    "sin(2*pi*x[0]-0.5*pi)*sin(2*pi*x[1]-0.5*pi)", degree=k + 5)

error_bw = np.sqrt(eta_h.vector().sum())
error_exact = errornorm(u_exact, u_h, "H10")

print("Exact error: {}".format(error_exact))
print("Bank-Weiser error from estimator: {}".format(error_bw))


def test():
    pass
