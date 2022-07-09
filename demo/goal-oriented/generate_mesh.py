# Copyright 2019-2020, Jack S. Hale, Raphaël Bulle
# SPDX-License-Identifier: LGPL-3.0-or-later
from dolfinx.io import (XDMFFile, extract_gmsh_geometry,
                        ufl_mesh_from_gmsh)
from dolfinx.mesh import create_mesh
from mpi4py import MPI

import gmsh

resolution = 0.1
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
model = gmsh.model()
model.add("L-shaped")

square = model.occ.addRectangle(0, 0, 0, 2., 2.)

small_square = model.occ.addRectangle(0, 0, 0, 1., 1.)

model.occ.cut([(2, square)], [(2, small_square)])

# Get entities of dim 0 (boundary vertices) tags
ent_tags = model.occ.get_entities(dim=0)
# Define the resolution near each entity
model.occ.mesh.setSize(ent_tags, resolution)
model.occ.synchronize()
model.mesh.generate(2)

x = extract_gmsh_geometry(model, model_name="L-shaped")[:, 0:2]

element_types, element_tags, node_tags = model.mesh.getElements(dim=2)

name, dim, order, num_nodes, local_coords, num_first_order_nodes = model.mesh.getElementProperties(element_types[0])

cells = node_tags[0].reshape(-1, num_nodes) - 1

mesh = create_mesh(MPI.COMM_SELF, cells, x, ufl_mesh_from_gmsh(element_types[0], 2))

with XDMFFile(MPI.COMM_SELF, "mesh.xdmf", "w") as of:
    of.write_mesh(mesh)
