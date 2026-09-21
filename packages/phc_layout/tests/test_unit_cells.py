import gdsfactory as gf
from phc_layout.components import (
    phc_hexagonal_unit_cell,
    phc_square_unit_cell,
)


def test_hexagonal_unit_cell():
    c = phc_hexagonal_unit_cell(pitch=1.0, radius=0.25)
    assert isinstance(c, gf.Component)
    # Ensure there is geometry in the cell
    polys = c.get_polygons()
    assert len(polys) > 0


def test_square_unit_cell():
    c = phc_square_unit_cell(pitch=1.0, radius=0.25)
    assert isinstance(c, gf.Component)
    polys = c.get_polygons()
    assert len(polys) > 0
