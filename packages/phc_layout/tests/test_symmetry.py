import numpy as np
import pytest
from phc_layout.lattice import HexagonalLattice, SquareLattice
from phc_layout.symmetry import (
    WYCKOFF_POSITIONS_2D,
    C4v,
    C6v,
    WyckoffPosition,
    get_point_group,
    get_wyckoff_position,
)


def test_point_groups():
    c4v = C4v()
    c6v = C6v()

    assert c4v.name == "C4v"
    assert c4v.order == 8
    assert c6v.name == "C6v"
    assert c6v.order == 12

    assert c4v != c6v
    assert c4v == C4v()
    assert c4v == "c4v"
    assert c6v == "C6V"

    pg_set = {c4v, c6v, C4v()}
    assert len(pg_set) == 2


def test_get_point_group():
    assert get_point_group("c4v") == C4v()
    assert get_point_group("C6v") == C6v()
    assert get_point_group("4mm") == C4v()
    assert get_point_group("6mm") == C6v()
    assert get_point_group(C4v()) == C4v()

    with pytest.raises(ValueError, match="Unknown point group"):
        get_point_group("D2h")


def test_wyckoff_position_basic():
    pos = WyckoffPosition((0.2, 0.3), letter="custom")
    assert pos.multiplicity == 1
    assert len(pos) == 1
    assert pos.letter == "custom"
    assert pos.positions == ((0.2, 0.3),)

    multi = WyckoffPosition(((0.0, 0.0), (0.5, 0.5)), letter="2pts")
    assert multi.multiplicity == 2
    assert len(multi) == 2


def test_wyckoff_position_to_cartesian():
    lat = HexagonalLattice(a=1.0)
    pos = WyckoffPosition(((0.0, 0.0), (1.0 / 3.0, 2.0 / 3.0)), letter="test")

    coords = pos.to_cartesian(lat, wrap_to_cell=False)
    assert len(coords) == 2
    assert np.isclose(coords[0][0], 0.0)
    assert np.isclose(coords[0][1], 0.0)

    # 1/3 * a1 + 2/3 * a2 = 1/3(1, 0) + 2/3(-1/2, sqrt(3)/2) = (0, sqrt(3)/3)
    assert np.isclose(coords[1][0], 0.0, atol=1e-12)
    assert np.isclose(coords[1][1], 1.0 / np.sqrt(3.0), atol=1e-12)

    # With wrap_to_cell: (1/3, 2/3) wraps to (1/3, -1/3)
    coords_wrapped = pos.to_cartesian(lat, wrap_to_cell=True)
    # 1/3(1, 0) - 1/3(-1/2, sqrt(3)/2) = (1/2, -sqrt(3)/6)
    assert np.isclose(coords_wrapped[1][0], 0.5, atol=1e-12)
    assert np.isclose(coords_wrapped[1][1], -np.sqrt(3.0) / 6.0, atol=1e-12)


def test_c6v_wyckoff_positions():
    assert "C6v" in WYCKOFF_POSITIONS_2D

    # 1a, 2b, 3c (fixed)
    p1a = get_wyckoff_position("C6v", "1a")
    assert p1a.multiplicity == 1
    assert p1a.positions == ((0.0, 0.0),)

    p2b = get_wyckoff_position("C6v", "2b")
    assert p2b.multiplicity == 2

    p3c = get_wyckoff_position("C6v", "3c")
    assert p3c.multiplicity == 3

    # 6d (parameterized along axes)
    p6d = get_wyckoff_position("C6v", "6d", x=0.3)
    assert p6d.multiplicity == 6
    lat = HexagonalLattice(a=1.0)
    coords_6d = p6d.to_cartesian(lat)
    for x, y in coords_6d:
        assert np.isclose(np.hypot(x, y), 0.3, atol=1e-12)

    # 6e (parameterized along diagonals)
    p6e = get_wyckoff_position("C6v", "6e", param=0.2)
    assert p6e.multiplicity == 6
    coords_6e = p6e.to_cartesian(lat)
    for x, y in coords_6e:
        assert np.isclose(np.hypot(x, y), np.sqrt(3.0) * 0.2, atol=1e-12)

    # 12f (general)
    p12f = get_wyckoff_position("C6v", "12f", p=(0.1, 0.2))
    assert p12f.multiplicity == 12


def test_c4v_wyckoff_positions():
    assert "C4v" in WYCKOFF_POSITIONS_2D

    p1a = get_wyckoff_position("C4v", "1a")
    assert p1a.multiplicity == 1

    p1b = get_wyckoff_position("C4v", "1b")
    assert p1b.multiplicity == 1
    assert p1b.positions == ((0.5, 0.5),)

    p2c = get_wyckoff_position("C4v", "2c")
    assert p2c.multiplicity == 2

    p4d = get_wyckoff_position("C4v", "4d", x=0.25)
    assert p4d.multiplicity == 4
    lat = SquareLattice(a=1.0)
    for x, y in p4d.to_cartesian(lat):
        assert np.isclose(abs(x), 0.25)
        assert np.isclose(abs(y), 0.25)

    p4e = get_wyckoff_position("C4v", "4e", x=0.3)
    assert p4e.multiplicity == 4

    p4f = get_wyckoff_position("C4v", "4f", x=0.2)
    assert p4f.multiplicity == 4

    p8g = get_wyckoff_position("C4v", "8g", p=(0.1, 0.3))
    assert p8g.multiplicity == 8


def test_wyckoff_position_errors():
    with pytest.raises(ValueError, match="not found for point group"):
        get_wyckoff_position("C6v", "4d")

    with pytest.raises(ValueError, match="requires parameter"):
        get_wyckoff_position("C6v", "6d")
