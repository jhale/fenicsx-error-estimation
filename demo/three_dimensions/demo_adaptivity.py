import numpy as np

from dolfin import *
import ufl

import bank_weiser

k = 1

def main():
    mesh = Mesh()
    try:
        with XDMFFile(MPI.comm_world, 'mesh.xdmf') as f:
            f.read(mesh)
    except:
        print("Generate the mesh using `python3 generate_mesh.py` before running this script.")
        exit()

    for i in range(0, 7):
        u_h = solve(mesh)

        eta_h = estimate(u_h)
        error_bw = np.sqrt(eta_h.vector().sum())
        print(error_bw)

        markers = mark(eta_h, 0.1)
        mesh = refine(mesh, markers)

        with XDMFFile("output/mesh_{}.xdmf".format(str(i).zfill(4))) as f:
            f.write(mesh)

def solve(mesh):
    V = FunctionSpace(mesh, "CG", k)

    u = TrialFunction(V)
    v = TestFunction(V)

    f = Constant(1.0)

    a = inner(grad(u), grad(v))*dx
    L = inner(f, v)*dx

    def all_boundary(x, on_boundary):
        return on_boundary

    bcs = DirichletBC(V, Constant(0.0), all_boundary)

    A, b = assemble_system(a, L, bcs=bcs)

    u_h = Function(V)

    PETScOptions.set("ksp_type", "cg")
    PETScOptions.set("ksp_monitor_true_residual")
    PETScOptions.set("pc_type", "hypre")
    PETScOptions.set("ksp_view")
    solver = PETScKrylovSolver()
    solver.set_from_options()
    solver.solve(A, u_h.vector(), b)

    return u_h

def estimate(u_h):
    mesh = u_h.function_space().mesh()

    V_f = FunctionSpace(mesh, "DG", k + 1)
    V_g = FunctionSpace(mesh, "DG", k)

    N = bank_weiser.local_interpolation_to_V0(V_f, V_g)

    e = TrialFunction(V_f)
    v = TestFunction(V_f)

    f = Constant(1.0)

    def all_boundary(x, on_boundary):
        return on_boundary

    bcs = DirichletBC(V_f, Constant(0.0), all_boundary, 'geometric')

    n = FacetNormal(mesh)
    a_e = inner(grad(e), grad(v))*dx
    L_e = inner(f + div(grad(u_h)), v)*dx + \
          inner(jump(grad(u_h), -n), avg(v))*dS

    e_h = bank_weiser.estimate(a_e, L_e, N, bcs)
    error = norm(e_h, "H10")

    # Computation of local error indicator
    V_e = FunctionSpace(mesh, "DG", 0)
    v = TestFunction(V_e)

    eta_h = Function(V_e)
    eta = assemble(inner(inner(grad(e_h), grad(e_h)), v)*dx)
    eta_h.vector()[:] = eta

    return eta_h

def mark(eta_h, alpha):
    etas = eta_h.vector().get_local()
    indices = etas.argsort()[::-1]
    sorted = etas[indices]

    total = sum(sorted)
    fraction = alpha*total

    mesh = eta_h.function_space().mesh()
    markers = MeshFunction("bool", mesh, mesh.geometry().dim(), False)

    v = 0.0
    for i in indices:
        if v >= fraction:
            break
        markers[int(i)] = True
        v += sorted[i]

    return markers

if __name__ == "__main__":
    main()
