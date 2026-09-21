import numpy as np
import pytest
from phc_layout.lattice import (
    HexagonalLattice,
    Lattice,
    SquareLattice,
    get_lattice,
)


def test_hexagonal_lattice_properties():
    pitch = 0.5
    lat = HexagonalLattice(a=pitch)
    assert lat.a == pitch
    assert lat.pitch == pitch

    a1, a2, a3 = lat.a1, lat.a2, lat.a3
    assert np.isclose(a1[0], pitch)
    assert np.isclose(a1[1], 0.0)
    assert np.isclose(a2[0], -pitch / 2.0)
    assert np.isclose(a2[1], pitch * np.sqrt(3.0) / 2.0)
    assert np.isclose(a3[2], 1.0)


def test_square_lattice_properties():
    pitch = 0.4
    lat = SquareLattice(a=pitch)
    assert lat.a == pitch
    assert lat.pitch == pitch

    a1, a2, a3 = lat.a1, lat.a2, lat.a3
    assert np.isclose(a1[0], pitch)
    assert np.isclose(a1[1], 0.0)
    assert np.isclose(a2[0], 0.0)
    assert np.isclose(a2[1], pitch)
    assert np.isclose(a3[2], 1.0)


def test_lattice_invalid_pitch():
    with pytest.raises(ValueError, match="must be positive"):
        HexagonalLattice(a=-1.0)

    with pytest.raises(ValueError, match="must be positive"):
        SquareLattice(a=0.0)


def test_lattice_singular_matrix():
    with pytest.raises(ValueError, match="linearly dependent"):
        Lattice(a1=(1.0, 0.0, 0.0), a2=(2.0, 0.0, 0.0), a3=(0.0, 0.0, 1.0))


def test_hexagonal_coordinate_roundtrip():
    lat = HexagonalLattice(a=0.45)
    test_points = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0 / 3.0, 2.0 / 3.0),
        (-0.5, 0.25),
        (0.123, -0.789),
    ]

    for u, v in test_points:
        x, y = lat.to_cartesian_2d(u, v)
        u_rec, v_rec = lat.from_cartesian_2d(x, y)
        assert np.isclose(u, u_rec, atol=1e-12)
        assert np.isclose(v, v_rec, atol=1e-12)

    # 3D roundtrip
    x, y, z = lat.to_cartesian(0.2, -0.4, 1.5)
    u_rec, v_rec, w_rec = lat.from_cartesian(x, y, z)
    assert np.isclose(u_rec, 0.2, atol=1e-12)
    assert np.isclose(v_rec, -0.4, atol=1e-12)
    assert np.isclose(w_rec, 1.5, atol=1e-12)


def test_square_coordinate_roundtrip():
    lat = SquareLattice(a=1.0)
    test_points = [(0.5, 0.5), (0.0, 0.5), (0.5, 0.0), (-0.2, 0.8)]
    for u, v in test_points:
        x, y = lat.to_cartesian_2d(u, v)
        u_rec, v_rec = lat.from_cartesian_2d(x, y)
        assert np.isclose(u, u_rec, atol=1e-12)
        assert np.isclose(v, v_rec, atol=1e-12)


def test_reciprocal_vectors_orthogonality():
    for lat in [HexagonalLattice(a=0.5), SquareLattice(a=0.8)]:
        a_vecs = [np.array(lat.a1), np.array(lat.a2), np.array(lat.a3)]
        b_vecs = [np.array(b) for b in lat.reciprocal_vectors]

        for i in range(3):
            for j in range(3):
                dot = float(np.dot(a_vecs[i], b_vecs[j]))
                expected = 2.0 * np.pi if i == j else 0.0
                assert np.isclose(dot, expected, atol=1e-10)


def test_wigner_seitz_polygons():
    a = 0.6
    hex_lat = HexagonalLattice(a=a)
    poly_hex = hex_lat.wigner_seitz_polygon()
    assert len(poly_hex) == 6
    # All vertices at distance a / sqrt(3)
    for x, y in poly_hex:
        dist = np.hypot(x, y)
        assert np.isclose(dist, a / np.sqrt(3.0), atol=1e-12)

    sq_lat = SquareLattice(a=a)
    poly_sq = sq_lat.wigner_seitz_polygon()
    assert len(poly_sq) == 4
    for x, y in poly_sq:
        dist = np.hypot(x, y)
        assert np.isclose(dist, a / np.sqrt(2.0), atol=1e-12)


def test_get_lattice_factory():
    assert isinstance(get_lattice("hexagonal", pitch=0.5), HexagonalLattice)
    assert isinstance(get_lattice("square", pitch=0.5), SquareLattice)
    assert isinstance(get_lattice("triangular", pitch=0.5), HexagonalLattice)

    with pytest.raises(ValueError, match="Unknown lattice_type"):
        get_lattice("fcc", pitch=1.0)
