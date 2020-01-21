import numpy as np

import dolfin
import fenics_error_estimation

def test_maximum():
    comm = dolfin.MPI.comm_world

    mesh = dolfin.UnitIntervalMesh(comm.size*5)
    V = dolfin.FunctionSpace(mesh, "DG", 0)
    markers = dolfin.MeshFunction("bool", mesh, mesh.geometry().dim(), False)

    eta_h = dolfin.Function(V)
    eta_h.vector()[:] = (comm.rank + 1)*np.arange(0, V.dim()/comm.size)

    theta = 0.2
    eta_max = eta_h.vector().max()

    markers = fenics_error_estimation.maximum(eta_h, theta)
    print(eta_max*theta)
    print(eta_h.vector().get_local())
    print(markers.array())
    assert(np.alltrue(markers.array()[np.where(eta_h.vector() > theta*eta_max)])) 
    assert(not(np.alltrue(markers.array()[np.where(eta_h.vector() < theta*eta_max)])))
