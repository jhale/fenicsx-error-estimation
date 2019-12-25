import pandas as pd
import numpy as np

from dolfin import *
import ufl

import bank_weiser

k = 1
parameters["ghost_mode"] = "shared_facet"

def main():
    mesh = Mesh()
    try:
        with XDMFFile(MPI.comm_world, 'mesh.xdmf') as f:
            f.read(mesh)
    except:
        print(
            "Generate the mesh using `python3 generate_mesh.py` before running this script.")
        exit()

    results = []
    for i in range(0, 13):
        result = {}
        V = FunctionSpace(mesh, "CG", k)
        u_h = solve(V)

        eta_h = estimate(u_h)
        result["error_bw"] = np.sqrt(eta_h.vector().sum())
        result["hmin"] = mesh.hmin()
        result["hmax"] = mesh.hmax()
        result["num_dofs"] = V.dim()

        markers = bank_weiser.maximum(eta_h, 0.1)
        mesh = refine(mesh, markers)

        with XDMFFile("output/mesh_{}.xdmf".format(str(i).zfill(4))) as f:
            f.write(mesh)

        results.append(result)

    if (MPI.comm_world.rank == 0):
        df = pd.DataFrame(results)
        df.to_pickle("output/results.pkl")
        print(df)

def solve(V):
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
    PETScOptions.set("pc_hypre_type", "boomeramg")
    PETScOptions.set("pc_hypre_boomeramg_strong_threshold", 0.5)
    PETScOptions.set("pc_hypre_boomeramg_coarsen_type", "HMIS")
    PETScOptions.set("pc_hypre_boomeramg_agg_nl", 4)
    PETScOptions.set("pc_hypre_boomeramg_agg_num_paths", 2)
    PETScOptions.set("pc_hypre_boomeramg_interp_type", "ext+i")
    PETScOptions.set("pc_hypre_boomeramg_truncfactor", 0.35)
    PETScOptions.set("ksp_view")
    solver = PETScKrylovSolver()
    solver.set_from_options()
    solver.solve(A, u_h.vector(), b)

    return u_h


def estimate(u_h):
    mesh = u_h.function_space().mesh()

    V_f = FunctionSpace(mesh, "DG", k + 1)
    V_g = FunctionSpace(mesh, "DG", k)

    N = bank_weiser.create_interpolation(V_f, V_g)

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

if __name__ == "__main__":
    main()


def test():
    pass
