# Copyright 2019-2020, Jack S. Hale, Raphaël Bulle
# SPDX-License-Identifier: LGPL-3.0-or-later
import pygmsh as pg
import meshio

lc = 0.5

with pg.geo.Geometry() as geom:
    lcar = 0.5

    # Points
    p1 = geom.add_point([0., 0., 0.], lcar)
    p2 = geom.add_point([0., -1., 0.], lcar)
    p3 = geom.add_point([1., -1., 0.], lcar)
    p4 = geom.add_point([1., 1., 0.], lcar)
    p5 = geom.add_point([-1., 1., 0.], lcar)
    p6 = geom.add_point([-1., 0., 0.], lcar)

    # Lines
    l1 = geom.add_line(p1, p2)
    l2 = geom.add_line(p2, p3)
    l3 = geom.add_line(p3, p4)
    l4 = geom.add_line(p4, p5)
    l5 = geom.add_line(p5, p6)
    l6 = geom.add_line(p6, p1)

    # Suface
    lloop = geom.add_line_loop([l1, l2, l3, l4, l5, l6])
    surf = geom.add_plane_surface(lloop)

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
