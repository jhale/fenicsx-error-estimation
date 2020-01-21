## Copyright 2019-2020, Jack S. Hale, Raphaël Bulle
## SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np
import mpi4py.MPI as MPI
from dolfin import MeshFunction

def dorfler(eta_h, theta):
    """Dörfler marking strategy"""
    if (eta_h.function_space().mesh().mpi_comm().size > 1):
        raise SystemExit("Does not work with with MPI size > 1")

    etas = eta_h.vector().get_local()
    indices = etas.argsort()[::-1]
    sorted = etas[indices]

    total = sum(sorted)
    fraction = theta*total

    mesh = eta_h.function_space().mesh()
    markers = MeshFunction("bool", mesh, mesh.geometry().dim(), False)

    v = 0.0
    for i in indices:
        if v >= fraction:
            break
        markers[int(i)] = True
        v += sorted[i]

    return markers

def maximum(eta_h, theta):
    """Maximum marking strategy"""
    mesh = eta_h.function_space().mesh()
    etas = eta_h.vector().get_local()

    eta_max = eta_h.vector().max()
    frac = theta*eta_max

    markers = MeshFunction("bool", mesh, mesh.geometry().dim(), False)
    marked = np.zeros_like(etas, dtype=np.bool)
    marked[np.where(etas > frac)] = True
    markers.set_values(marked)

    return markers
