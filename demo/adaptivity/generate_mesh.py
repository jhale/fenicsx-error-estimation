## Copyright 2019-2020, Jack S. Hale, Raphaël Bulle
## SPDX-License-Identifier: LGPL-3.0-or-later
import pygmsh as pg
import meshio

lc = 0.1
geom = pg.built_in.Geometry()

# Points
p1 = geom.add_point([0, 0, 0], lcar=lc)
p2 = geom.add_point([0, -1, 0], lcar=lc)
p3 = geom.add_point([1, -1, 0], lcar=lc)
p4 = geom.add_point([1, 1, 0], lcar=lc)
p5 = geom.add_point([-1, 1, 0], lcar=lc)
p6 = geom.add_point([-1, 0, 0], lcar=lc)

# Lines
l1 = geom.add_line(p1, p2)
l2 = geom.add_line(p2, p3)
l3 = geom.add_line(p3, p4)
l4 = geom.add_line(p4, p5)
l5 = geom.add_line(p5, p6)
l6 = geom.add_line(p6, p1)

# Surface
lloop = geom.add_line_loop([l1, l2, l3, l4, l5, l6])
surf = geom.add_plane_surface(lloop)

mesh = pg.generate_mesh(geom)

mesh.points = mesh.points[:, :2]  # Used to convert the 3D mesh into 2D

meshio.write("mesh.xdmf", meshio.Mesh(
    points=mesh.points,
    cells={"triangle": mesh.cells["triangle"]}))
