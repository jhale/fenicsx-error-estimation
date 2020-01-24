## Copyright 2019-2020, Jack S. Hale, Raphaël Bulle
## SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np
import mpi4py.MPI as MPI
from dolfin import MeshFunction, cells

import itertools

def dorfler(eta_h, theta):
    """Dörfler marking strategy"""
    if (eta_h.function_space().mesh().mpi_comm().size > 1):
        raise SystemExit("Does not work with with MPI size > 1")

    etas = eta_h.vector().get_local()
    indices = etas.argsort()[::-1]

    total = sum(etas)
    fraction = theta*total

    mesh = eta_h.function_space().mesh()
    markers = MeshFunction("bool", mesh, mesh.geometry().dim(), False)

    v = 0.0
    for i in indices:
        if v >= fraction:
            break
        markers[int(i)] = True
        v += etas[i]

    return markers

def dorfler_parallel(eta_h, theta):
    mesh = eta_h.function_space().mesh()
    comm = mesh.mpi_comm()

    # NOTE: Check the necessity of this loop.
    etas = MeshFunction("double", mesh, mesh.topology().dim(), 0.0)
    for c in cells(mesh):
        etas[c] = eta_h.vector()[c.index()]

    markers = MeshFunction("bool", mesh, mesh.geometry().dim(), False)
    markers_local = np.copy(markers.array())
    markers_local_shape = comm.gather(markers_local.shape, root=0)

    # Communicate indicators on each process back to the rank 0 process
    # NOTE: Obviously suboptimal and problematic for very large problems.
    etas_local = etas.array()
    eta_global = comm.gather(etas_local, root=0)

    if (comm.rank == 0):
        eta_global = np.hstack(eta_global)
        markers_global = np.zeros_like(eta_global, dtype=np.bool)
        markers_local_shape = np.array(markers_local_shape).reshape(-1)

        # Indices biggest to smallest
        indices = np.argsort(eta_global)[::-1]
        sum_eta_global = np.sum(eta_global)
        fraction = theta*sum_eta_global

        # Find set with minimal cardinality.
        # TODO: Non-sequential memory access and tight loop.
        # TODO: Implement O(N) algorithm.
        rolling_sum = 0.0
        for i in indices:
            if rolling_sum >= fraction:
                break
            rolling_sum += eta_global[i]
            markers_global[i] = True

        send_counts = markers_local_shape
        displacements = np.zeros(comm.size)
        for i in range(1, len(displacements)):
            displacements[i] = displacements[i - 1] + send_counts[i - 1]
    else:
        markers_global = None
        displacements = None
        send_counts = None

    # Scatter back
    comm.Scatterv([markers_global, send_counts, displacements, MPI.BOOL], markers_local, root=0)
    markers.set_values(markers_local)

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
