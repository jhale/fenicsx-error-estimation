import numpy as np

from dolfin import *
import bank_weiser

mesh = UnitSquareMesh(128, 128)

k = 1
V = FunctionSpace(mesh, "CG", k)

u = TrialFunction(V)
v = TestFunction(V)

f = Expression(
    "(2*pow(2*pi,2)+1)*sin(2*pi*x[0]-0.5*pi)*sin(2*pi*x[1]-0.5*pi)", degree=k + 3)

a = inner(grad(u), grad(v))*dx
L = inner(f, v)*dx

u_h = Function(V)
A, b = assemble_system(a, L)

solver = PETScLUSolver()
solver.solve(A, u_h.vector(), b)

V_f = FunctionSpace(mesh, "DG", k + 1)
V_g = FunctionSpace(mesh, "DG", k)

N = bank_weiser.create_interpolation(V_f, V_g)

e = TrialFunction(V_f)
v = TestFunction(V_f)

n = FacetNormal(mesh)
a_e = inner(grad(e), grad(v))*dx
L_e = inner(f + div(grad(u_h)), v)*dx + \
    inner(jump(grad(u_h), -n), avg(v))*dS

e_h = bank_weiser.estimate(a_e, L_e, N)
error = norm(e_h, "H10")

# Computation of local error indicator
V_e = FunctionSpace(mesh, "DG", 0)
v = TestFunction(V_e)

eta_h = Function(V_e)
eta = assemble(inner(inner(grad(e_h), grad(e_h)), v)*dx)
eta_h.vector()[:] = eta

u_exact = Expression(
    "sin(2*pi*x[0]-0.5*pi)*sin(2*pi*x[1]-0.5*pi)", degree=k + 3)

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
    pass
