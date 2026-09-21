import gdsfactory as gf
import pytest
from phc_layout.components import (
    get_unit_cell,
    get_unit_cell_spec,
    list_unit_cells,
    phc_wyckoff_unit_cell,
)
from phc_layout.tech import LAYER_PHC
from phc_mpb import gds_to_mpb_geometry


def test_phc_wyckoff_unit_cell_basic():
    # Hexagonal primitive 1a
    c = phc_wyckoff_unit_cell(
        pitch=1.0,
        point_group="C6v",
        features=(("1a", 0.25),),
    )
    assert isinstance(c, gf.Component)
    assert len(c.insts) == 1

    # Hexagonal 1a + 6d (snowflake)
    c_snow = phc_wyckoff_unit_cell(
        pitch=1.0,
        point_group="C6v",
        features=(
            ("1a", 0.2),
            ("6d", 0.1, 0.35),
        ),
    )
    assert isinstance(c_snow, gf.Component)
    # 1 center + 6 satellites = 7 instances
    assert len(c_snow.insts) == 7


def test_phc_wyckoff_unit_cell_boundary():
    # Test boundary extrusion on LAYER_PHC.SLAB
    c_ws = phc_wyckoff_unit_cell(
        pitch=1.0,
        point_group="C6v",
        features=(("1a", 0.2),),
        boundary_layer=LAYER_PHC.SLAB,
        boundary_type="wigner_seitz",
    )
    assert isinstance(c_ws, gf.Component)
    # 1 hole instance + 1 slab polygon
    assert len(c_ws.insts) == 1
    polys = c_ws.get_polygons()
    assert len(polys) > 0


def test_phc_wyckoff_unit_cell_dict_features():
    c = phc_wyckoff_unit_cell(
        pitch=1.0,
        point_group="C4v",
        features=[
            {"position": "1a", "radius": 0.2},
            {"letter": "4d", "radius": 0.1, "param": 0.25},
        ],
    )
    assert isinstance(c, gf.Component)
    assert len(c.insts) == 5  # 1 + 4


def test_phc_wyckoff_unit_cell_errors():
    with pytest.raises(ValueError, match="must be positive"):
        phc_wyckoff_unit_cell(pitch=-1.0)

    with pytest.raises(ValueError, match="Unknown point group"):
        phc_wyckoff_unit_cell(point_group="Oh")

    with pytest.raises(ValueError, match="Feature tuple must have length 2"):
        phc_wyckoff_unit_cell(features=((1, 2, 3, 4),))


def test_unit_cell_database_catalog():
    all_cells = list_unit_cells()
    assert len(all_cells) == 16

    c6v_cells = list_unit_cells(point_group="C6v")
    assert len(c6v_cells) == 8
    assert "c6v_honeycomb" in c6v_cells
    assert "c6v_kagome" in c6v_cells
    assert "c6v_painter_snowflake" in c6v_cells

    c4v_cells = list_unit_cells(point_group="C4v")
    assert len(c4v_cells) == 8
    assert "c4v_checkerboard" in c4v_cells
    assert "c4v_lieb" in c4v_cells

    flatband_cells = list_unit_cells(tag="flatband")
    assert "c6v_kagome" in flatband_cells
    assert "c4v_lieb" in flatband_cells

    snowflake_cells = list_unit_cells(tag="snowflake")
    assert "c6v_snowflake_6d" in snowflake_cells
    assert "c6v_painter_snowflake" in snowflake_cells


def test_unit_cell_database_instantiation_all():
    """Verify that every single unit cell in the database instantiates cleanly."""
    expected_multiplicities = {
        "c6v_primitive": 1,
        "c6v_honeycomb": 2,
        "c6v_kagome": 3,
        "c6v_ring_6d": 6,
        "c6v_ring_6e": 6,
        "c6v_snowflake_6d": 7,
        "c6v_snowflake_6e": 7,
        "c4v_primitive": 1,
        "c4v_checkerboard": 2,
        "c4v_lieb": 2,
        "c4v_ring_4d": 4,
        "c4v_cross_4e": 4,
        "c4v_edges_4f": 4,
        "c4v_snowflake_4d": 5,
        "c4v_snowflake_4e": 5,
    }

    for name in list_unit_cells():
        spec = get_unit_cell_spec(name)
        assert spec.name == name

        c = get_unit_cell(name, pitch=0.5)
        assert isinstance(c, gf.Component)
        if name == "c6v_painter_snowflake":
            assert len(c.get_polygons()) > 0
        else:
            assert len(c.insts) == expected_multiplicities[name]


def test_get_unit_cell_with_radius_override():
    c = get_unit_cell("c6v_honeycomb", pitch=1.0, radius=0.15)
    assert isinstance(c, gf.Component)
    assert len(c.insts) == 2


def test_phc_snowflake_unit_cell():
    from phc_layout.components import phc_snowflake_unit_cell

    c = phc_snowflake_unit_cell(pitch=1.0, radius=0.4, width=0.15)
    assert isinstance(c, gf.Component)
    assert len(c.get_polygons()) > 0

    with pytest.raises(ValueError, match="must be positive"):
        phc_snowflake_unit_cell(pitch=-1.0)
    with pytest.raises(ValueError, match="must be positive"):
        phc_snowflake_unit_cell(radius=-0.2)
    with pytest.raises(ValueError, match="must be positive"):
        phc_snowflake_unit_cell(width=-0.1)


def test_mpb_geometry_interoperability():
    """Verifies that database unit cells convert cleanly to MPB geometries."""
    c_honeycomb = get_unit_cell("c6v_honeycomb", pitch=0.5)
    geom_honeycomb = gds_to_mpb_geometry(c_honeycomb, pitch=0.5, dimension="2D")
    assert len(geom_honeycomb) == 2

    c_lieb = get_unit_cell("c4v_lieb", pitch=0.5)
    geom_lieb = gds_to_mpb_geometry(c_lieb, pitch=0.5, dimension="2D")
    assert len(geom_lieb) == 2

    c_snow = get_unit_cell("c6v_painter_snowflake", pitch=0.5)
    geom_snow = gds_to_mpb_geometry(c_snow, pitch=0.5, dimension="2D")
    assert len(geom_snow) == 1
