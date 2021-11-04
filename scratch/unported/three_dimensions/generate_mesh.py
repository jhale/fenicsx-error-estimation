import pygmsh as pg
import meshio

lc = 0.1
geom = pg.opencascade.Geometry()

big_box = geom.add_box([-1.0, -1.0, -1.0], [2.0, 2.0, 2.0], char_length=0.3)
corner_box = geom.add_box([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], char_length=0.3)
geom.boolean_difference([big_box], [corner_box])

mesh = pg.generate_mesh(geom)

meshio.write("mesh.xdmf", meshio.Mesh(
    points=mesh.points,
    cells={"tetra": mesh.cells["tetra"]}))
