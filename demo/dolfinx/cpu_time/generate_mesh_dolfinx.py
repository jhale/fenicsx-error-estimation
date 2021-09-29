# Copyright 2019-2020, Jack S. Hale, Raphaël Bulle
# SPDX-License-Identifier: LGPL-3.0-or-later
import pygmsh as pg
import meshio

lc = 0.01

with pg.geo.Geometry() as geom:
    geom.add_polygon(
            [
                [0., 0.],
                [0., -1.],
                [1., -1.],
                [1., 1.],
                [-1., 1.],
                [-1., 0.]
            ],
            mesh_size = lc)
    mesh = geom.generate_mesh()

mesh.points = mesh.points[:, :2]

for cell in mesh.cells:
    if cell.type == 'triangle':
        triangle_cells = cell.data

meshio.write("./mesh.xdmf", meshio.Mesh(
    points = mesh.points,
    cells={"triangle": triangle_cells}))

'''
with pg.geo.Geometry() as geom:
    geom.add_polygon(
        [
            [0.0, 0.0],
            [0.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 0.0],
        ],
        mesh_size=0.5,
    )
    mesh = geom.generate_mesh()

mesh.points = mesh.points[:, :2]
meshio.write("mesh.xdmf", mesh)
'''
