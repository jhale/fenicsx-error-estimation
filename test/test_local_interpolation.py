import itertools
import os

import numpy as np
import pytest

from dolfin import *
from dolfin.fem.assembling import _create_dolfin_form

from fenics_error_estimation import create_interpolation, estimate

results_dirichlet = []
results_neumann = []

@pytest.fixture(params=itertools.product([1, 2], [2**4, 2**5, 2**6]))
def mesh(request):
    d = request.param[0]
    n = request.param[1]

    if d == 1:
        return UnitIntervalMesh(n)
    elif d == 2:
        return UnitSquareMesh(n, n)
    elif d == 3:
        return UnitCubeMesh.create(n, n, n, CellType.Type.tetrahedron)
    else:
        raise NotImplementedError


@pytest.fixture
def u_and_f_dirichlet(mesh, k):
    if mesh.geometry().dim() == 1:
        return (Expression("sin(2.0*pi*x[0])", degree=k + 3), Expression("4.0*pi*pi*sin(2.0*pi*x[0])", degree=k + 3))
    elif mesh.geometry().dim() == 2:
        return (Expression("sin(2.0*pi*x[0])*sin(2.0*pi*x[1])", degree=k + 3), Expression("8.0*pi*pi*sin(2.0*pi*x[0])*sin(2.0*pi*x[1])", degree=k + 3))
    elif mesh.geometry().dim() == 3:
        return (Expression("sin(2.0*pi*x[0])*sin(2.0*pi*x[1])*sin(2.0*pi*x[2])", degree=k + 3), Expression("12.0*pi*pi*sin(2.0*pi*x[0])*sin(2.0*pi*x[1])*sin(2.0*pi*x[2])", degree=k + 3))
    else:
        raise NotImplementedError


@pytest.mark.parametrize('k', [1, 2])
def test_local_pure_dirichlet(request, mesh, k, u_and_f_dirichlet, output_xdmf=False, estimate=estimate):
    if k == 1:
        k_g = k
    else:
        k_g = k - 1

    V = FunctionSpace(mesh, "CG", k)
    element_f = FiniteElement("DG", mesh.ufl_cell(), k + 1)
    element_g = FiniteElement("DG", mesh.ufl_cell(), k)
    V_f = FunctionSpace(mesh, element_f)
    V_e = FunctionSpace(mesh, "DG", 0)

    u_exact, f = u_and_f_dirichlet

    def boundary(x, on_boundary):
        return on_boundary

    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v))*dx
    L = inner(f, v)*dx

    bcs = [DirichletBC(V, Constant(0.0), boundary)]

    u_h = Function(V)
    A, b = assemble_system(a, L, bcs=bcs)

    solver = PETScLUSolver()
    solver.solve(A, u_h.vector(), b)

    e = TrialFunction(V_f)
    v = TestFunction(V_f)

    d = mesh.geometry().dim()
    if d == 1:
        bcs = [DirichletBC(V_f, Constant(0.0), boundary, "pointwise")]
    else:
        bcs = [DirichletBC(V_f, Constant(0.0), boundary, "geometric")]

    n = FacetNormal(mesh)
    a_e = inner(grad(e), grad(v))*dx
    L_e = inner(f + div(grad(u_h)), v)*dx + \
        inner(jump(grad(u_h), -n), avg(v))*dS

    N = create_interpolation(element_f, element_g)

    e_V_f = estimate(a_e, L_e, N, bcs)

    error_bw = np.sqrt(assemble(inner(grad(e_V_f), grad(e_V_f))*dx))
    error_exact = errornorm(u_exact, u_h, "H10")

    V_e = FunctionSpace(mesh, "DG", 0)
    e_V_e = project(inner(grad(e_V_f), grad(e_V_f)), V_e)

    u_exact_V = interpolate(u_exact, V)

    if output_xdmf:
        with XDMFFile("output/pure_dirichlet/local_interpolation/e_V_e.xdmf") as f:
            f.write_checkpoint(e_V_e, "e_V_e")

        with XDMFFile("output/pure_dirichlet/local_interpolation/u_h.xdmf") as f:
            f.write_checkpoint(u_h, "u_h")

        with XDMFFile("output/pure_dirichlet/local_interpolation/u_exact_V.xdmf") as f:
            f.write_checkpoint(u_exact_V, "u_exact_V")

        with XDMFFile("output/pure_dirichlet/local_interpolation/e_V_f.xdmf") as f:
            f.write_checkpoint(e_V_f, "e_V_f")

    result = {}
    result["name"] = request.node.name
    result["hmin"] = mesh.hmin()
    result["hmax"] = mesh.hmax()
    result["k"] = k
    result["gdim"] = d
    result["num_dofs"] = V.dim()
    result["error_bank_weiser"] = error_bw
    result["error_exact"] = error_exact
    result["relative_error"] = error_bw/error_exact - 1.

    results_dirichlet.append(result)

    return result


def test_convergence_rates_pure_dirichlet():
    import pandas as pd
    import matplotlib.pyplot as plt
    from mpltools import annotation
    from marker_pos import loglog_marker_pos

    try:
        os.makedirs("output/pure_dirichlet/")
    except OSError:
        pass

    df = pd.DataFrame(results_dirichlet)
    print(df)
    df.to_csv("output/pure_dirichlet/test_convergence_rates.csv")

    for gdim in df["gdim"].unique():
        for k in df["k"].unique():
            mask = (df.k == k) & (df.gdim == gdim)
            data = df.loc[mask]

            rate = np.polyfit(np.log(data['hmax'].values), np.log(
                data['error_exact'].values), 1)[0]
            assert(rate > k - 0.1)
            rate = np.polyfit(np.log(data['hmax'].values), np.log(
                data['error_bank_weiser'].values), 1)[0]
            assert(rate > k - 0.1)

            fig = plt.figure()
            plt.loglog(data['hmax'], data['error_exact'], label="True error")
            plt.loglog(data['hmax'], data['error_bank_weiser'],
                       dashes=[6, 2], label="Bank-Weiser error")
            plt.legend()
            plt.title(r"$k = {}$, $d = {}$".format(k, gdim))
            plt.xlabel(r"$h_{\mathrm{max}}$")
            plt.ylabel(r"$H^1_0(\Omega)$ error")
            marker_x, marker_y = loglog_marker_pos(
                data['hmax'].values, data['error_exact'].values, data['error_bank_weiser'].values)
            annotation.slope_marker((marker_x, marker_y), (k, 1))
            plt.savefig(
                "output/pure_dirichlet/convergence-k-{}-gdim-{}.pdf".format(k, gdim))


@pytest.mark.parametrize('k', [1, 2])
def test_against_global_pure_dirichlet(request, mesh, k, u_and_f_dirichlet):
    from test_global_interpolation import _global_pure_dirichlet
    if (mesh.num_cells() > 256):
        pytest.skip("Problem too large for global approach...")
    result_global = _global_pure_dirichlet(request, mesh, k, u_and_f_dirichlet)
    result_local = test_local_pure_dirichlet(
        request, mesh, k, u_and_f_dirichlet)
    assert(np.allclose(result_local["error_bank_weiser"],
                       result_global["error_bank_weiser"], atol=1e-2))


@pytest.mark.parametrize('k', [1])
def test_against_lagrange_pure_dirichlet(request, mesh, k, u_and_f_dirichlet):
    from test_lagrange_multipliers import _lagrange_pure_dirichlet
    result_lagrange = _lagrange_pure_dirichlet(
        request, mesh, k, u_and_f_dirichlet)
    result_local = test_local_pure_dirichlet(
        request, mesh, k, u_and_f_dirichlet)
    assert(np.allclose(result_local["error_bank_weiser"],
                       result_lagrange["error_bank_weiser"], atol=1e-1))


@pytest.mark.parametrize('k', [1, 2])
def test_against_python_pure_dirichlet(request, mesh, k, u_and_f_dirichlet):
    from fenics_error_estimation.estimate import estimate_python
    result_python = test_local_pure_dirichlet(
        request, mesh, k, u_and_f_dirichlet, estimate=estimate_python)
    result_local = test_local_pure_dirichlet(
        request, mesh, k, u_and_f_dirichlet)
    assert(np.allclose(result_local["error_bank_weiser"],
                       result_python["error_bank_weiser"], atol=1e-2))


@pytest.fixture
def u_and_f_neumann(mesh, k):
    if mesh.geometry().dim() == 1:
        return (Expression("sin(2.0*pi*x[0]-0.5*pi)", degree=k + 3), Expression("(4.0*pi*pi + 1)*sin(2.0*pi*x[0]-0.5*pi)", degree=k + 3))
    elif mesh.geometry().dim() == 2:
        return (Expression("sin(2.0*pi*x[0]-0.5*pi)*sin(2.0*pi*x[1]-0.5*pi)", degree=k + 3), Expression("(8.0*pi*pi + 1)*sin(2.0*pi*x[0]-0.5*pi)*sin(2.0*pi*x[1]-0.5*pi)", degree=k + 3))
    elif mesh.geometry().dim() == 3:
        return (Expression("sin(2.0*pi*x[0]-0.5*pi)*sin(2.0*pi*x[1]-0.5*pi)*sin(2.0*pi*x[2]-0.5*pi)", degree=k + 3), Expression("(12.0*pi*pi + 1)*sin(2.0*pi*x[0]-0.5*pi)*sin(2.0*pi*x[1]-0.5*pi)*sin(2.0*pi*x[2]-0.5*pi)", degree=k + 3))
    else:
        raise NotImplementedError


@pytest.mark.parametrize('k', [1, 2])
def test_local_pure_neumann(request, mesh, k, u_and_f_neumann, output_xdmf=False, estimate=estimate):
    if k == 1:
        k_g = k
    else:
        k_g = k-1

    V = FunctionSpace(mesh, "CG", k)
    element_f = FiniteElement("DG", mesh.ufl_cell(), k + 1)
    element_g = FiniteElement("DG", mesh.ufl_cell(), k)
    V_f = FunctionSpace(mesh, element_f)
    V_e = FunctionSpace(mesh, "DG", 0)

    u_exact, f = u_and_f_neumann

    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v))*dx + inner(u, v)*dx
    L = inner(f, v)*dx

    u_h = Function(V)
    A, b = assemble_system(a, L)

    solver = PETScLUSolver()
    solver.solve(A, u_h.vector(), b)

    e = TrialFunction(V_f)
    v = TestFunction(V_f)

    n = FacetNormal(mesh)
    a_e = inner(grad(e), grad(v))*dx + inner(e, v)*dx
    L_e = inner(f + div(grad(u_h)), v)*dx + inner(jump(grad(u_h), -n),
                                                  avg(v))*dS - inner(inner(grad(u_h), n), v)*ds

    N = create_interpolation(element_f, element_g)

    e_V_f = estimate(a_e, L_e, N, [])

    error_bw = np.sqrt(
        assemble(inner(grad(e_V_f), grad(e_V_f))*dx + inner(e_V_f, e_V_f)*dx))
    error_exact = errornorm(u_exact, u_h, "H1")

    V_e = FunctionSpace(mesh, "DG", 0)
    e_V_e = project(inner(grad(e_V_f), grad(e_V_f)) + inner(e_V_f, e_V_f), V_e)
    u_exact_V = interpolate(u_exact, V)

    if output_xdmf:
        with XDMFFile("output/pure_neumann/local_interpolation/e_V_e.xdmf") as f:
            f.write_checkpoint(e_V_e, "e_V_e")

        with XDMFFile("output/pure_neumann/local_interpolation/u_h.xdmf") as f:
            f.write_checkpoint(u_h, "u_h")

        with XDMFFile("output/pure_neumann/local_interpolation/u_exact_V.xdmf") as f:
            f.write_checkpoint(u_exact_V, "u_exact_V")

        with XDMFFile("output/pure_neumann/local_interpolation/e_V_f.xdmf") as f:
            f.write_checkpoint(e_V_f, "e_V_f")

    result = {}
    result["name"] = request.node.name
    result["hmin"] = mesh.hmin()
    result["hmax"] = mesh.hmax()
    result["k"] = k
    result["gdim"] = mesh.geometry().dim()
    result["num_dofs"] = V.dim()
    result["error_bank_weiser"] = error_bw
    result["error_exact"] = error_exact
    result["relative_error"] = error_bw/error_exact - 1.

    results_neumann.append(result)

    return result


def test_convergence_rates_pure_neumann():
    import pandas as pd
    import matplotlib.pyplot as plt
    from mpltools import annotation
    from marker_pos import loglog_marker_pos

    try:
        os.makedirs("output/pure_neumann")
    except OSError:
        pass

    df = pd.DataFrame(results_neumann)
    print(df)
    df.to_csv("output/pure_neumann/test_convergence_rates.csv")

    for gdim in df["gdim"].unique():
        for k in df["k"].unique():
            mask = (df.k == k) & (df.gdim == gdim)
            data = df.loc[mask]

            rate = np.polyfit(np.log(data['hmax'].values), np.log(
                data['error_exact'].values), 1)[0]
            assert(rate > k - 0.1)
            rate = np.polyfit(np.log(data['hmax'].values), np.log(
                data['error_bank_weiser'].values), 1)[0]
            assert(rate > k - 0.1)

            fig = plt.figure()
            plt.loglog(data['hmax'], data['error_exact'], label="True error")
            plt.loglog(data['hmax'], data['error_bank_weiser'],
                       dashes=[6, 2], label="Bank-Weiser error")
            plt.legend()
            plt.title(r"$k = {}$, $d = {}$".format(k, gdim))
            plt.xlabel(r"$h_{\mathrm{max}}$")
            plt.ylabel(r"$H^1(\Omega)$ error")
            marker_x, marker_y = loglog_marker_pos(
                data['hmax'].values, data['error_exact'].values, data['error_bank_weiser'].values)
            annotation.slope_marker((marker_x, marker_y), (k, 1))
            plt.savefig(
                "output/pure_neumann/convergence-k-{}-gdim-{}.pdf".format(k, gdim))


@pytest.mark.parametrize('k', [1, 2])
def test_against_global_pure_neumann(request, mesh, k, u_and_f_neumann):
    from test_global_interpolation import _global_pure_neumann
    if (mesh.num_cells() > 256):
        pytest.skip("Problem too large for global approach...")
    result_global = _global_pure_neumann(request, mesh, k, u_and_f_neumann)
    result_local = test_local_pure_neumann(request, mesh, k, u_and_f_neumann)
    assert(np.allclose(result_local["error_bank_weiser"],
                       result_global["error_bank_weiser"], atol=1e-2))


@pytest.mark.parametrize('k', [1])
def test_against_lagrange_pure_neumann(request, mesh, k, u_and_f_neumann):
    from test_lagrange_multipliers import _lagrange_pure_neumann
    result_lagrange = _lagrange_pure_neumann(request, mesh, k, u_and_f_neumann)
    result_local = test_local_pure_neumann(request, mesh, k, u_and_f_neumann)
    assert(np.allclose(result_local["error_bank_weiser"],
                       result_lagrange["error_bank_weiser"], atol=1e-1))


@pytest.mark.parametrize('k', [1, 2])
def test_against_python_pure_neumann(request, mesh, k, u_and_f_neumann):
    from fenics_error_estimation.estimate import estimate_python
    result_python = test_local_pure_neumann(
        request, mesh, k, u_and_f_neumann, estimate=estimate_python)
    result_local = test_local_pure_neumann(request, mesh, k, u_and_f_neumann)
    assert(np.allclose(result_local["error_bank_weiser"],
                       result_python["error_bank_weiser"], atol=1e-2))
