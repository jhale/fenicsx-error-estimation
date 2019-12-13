import numpy as np

from dolfin import *
from bank_weiser import estimate, local_interpolation_to_V0

parameters["ghost_mode"] = "shared_facet"
mesh = UnitSquareMesh(120, 120)

k = 1
V = FunctionSpace(mesh, "CG", k)

u = TrialFunction(V)
v = TestFunction(V)

f = Expression("8.0*pi*pi*sin(2.0*pi*x[0])*sin(2.0*pi*x[1])", degree = k + 3)

a = inner(grad(u), grad(v))*dx
L = inner(f, v)*dx

class Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

boundary = Boundary()

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(1)
boundary.mark(boundaries, 0)

bcs = DirichletBC(V, Constant(0.0), boundaries, 0)

u_h = Function(V)
A, b = assemble_system(a, L, bcs=bcs)

solver = PETScLUSolver()
solver.solve(A, u_h.vector(), b)

V_f = FunctionSpace(mesh, "DG", k + 1)
V_g = FunctionSpace(mesh, "DG", k)

N = local_interpolation_to_V0(V_f, V_g)

e = TrialFunction(V_f)
v = TestFunction(V_f)

bcs = [DirichletBC(V_f, Constant(0.0), boundaries, 0)]
n = FacetNormal(mesh)
a_e = inner(grad(e), grad(v))*dx
L_e = inner(f + div(grad(u_h)), v)*dx + \
      inner(jump(grad(u_h), -n), avg(v))*dS

e_h = estimate(a_e, L_e, N, bcs)
error = norm(e_h, "H10")

# Computation of local error indicator

V_e = FunctionSpace(mesh, "DG", 0)
u = TrialFunction(V_e)
v = TestFunction(V_e)

vol = CellVolume(mesh)
a = inner(u, v)*dx
L = inner(inner(grad(e_h), grad(e_h))*vol, v)*dx

eta = Function(V_e)
solver = LocalSolver(a, L, solver_type=LocalSolver.SolverType.Cholesky)
solver.solve_local_rhs(eta)

u_exact = Expression("sin(2.0*pi*x[0])*sin(2.0*pi*x[1])", degree = k + 3)

with XDMFFile("output/u_h.xdmf") as f:
    f.write_checkpoint(u_h, "u_h")

with XDMFFile("output/e_h.xdmf") as f:
    f.write_checkpoint(e_h, "Residual solution")

with XDMFFile("output/eta.xdmf") as f:
    f.write_checkpoint(eta, "Element-wise estimator")

error_bw = np.sqrt(eta.vector().sum())
error_exact = errornorm(u_exact, u_h, "H10")

print("Exact error: {}".format(error_exact))
print("Bank-Weiser error from estimator: {}".format(error_bw))
