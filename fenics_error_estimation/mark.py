## Copyright 2019-2020, Jack S. Hale, Raphaël Bulle
## SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np
import mpi4py.MPI as MPI
from dolfin import MeshFunction

import itertools

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

def dorfler_parallel(eta_h, theta):
    mesh = eta_h.function_space().mesh()
    comm = mesh.mpi_comm()

    markers = MeshFunction("bool", mesh, mesh.geometry().dim(), False)
    markers_local = markers.array()

    # Communicate indicators on each process back to the rank 0 process
    # NOTE: Obviously suboptimal and problematic for very large problems.
    eta_local = eta_h.vector().get_local()
    eta_global = np.empty(eta_h.function_space().dim(), dtype=np.float64)
    comm.Gather(eta_local, eta_global, root=0)

    # Communicate local ranges back to rank 0 process.
    # These are used for sending the markers back at the end.
    ranges = comm.gather(eta_h.vector().local_range(), root=0)

    if (comm.rank == 0):
        markers_global = np.zeros(eta_h.function_space().dim(), dtype=np.bool)

        # Indices biggest to smallest
        indices = np.argsort(eta_global)[::-1]
        sum_eta_global = np.sum(eta_global)
        fraction = theta*sum_eta_global

        # Find set with minimal cardinality.
        # TODO: Non-sequential memory access and tight loop.
        rolling_sum = 0.0
        for i in indices:
            if rolling_sum >= fraction:
                break
            rolling_sum += eta_global[i]
            markers_global[i] = True

        send_counts = [r[1] - r[0] for r in ranges]
        displacements = list(itertools.accumulate([r[0] for r in ranges]))
    else:
        markers_global = None
        send_counts = None
        displacements = None

    # Scatter back
    comm.Scatterv([markers_global, send_counts, displacements, MPI.BOOL], markers_local, root=0)
    markers.set_values(markers_local)

    eturn markers


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
