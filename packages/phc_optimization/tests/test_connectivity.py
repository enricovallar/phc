"""Unit tests for dielectric geometry connectivity and neck-width checking."""

import numpy as np
import pytest
from phc_optimization.connectivity import (
    check_array_connectivity,
    check_slab_connectivity,
)


def test_uniform_solid_slab() -> None:
    """Verifies that a solid dielectric grid is recognized as continuous."""
    eps = np.full((32, 32), fill_value=12.0)
    is_conn, n_comp, _msg = check_array_connectivity(eps, epsilon_threshold=1.1)
    assert is_conn is True
    assert n_comp == 1


def test_empty_air_grid() -> None:
    """Verifies that an air grid is recognized as non-dielectric."""
    eps = np.full((32, 32), fill_value=1.0)
    is_conn, n_comp, _msg = check_array_connectivity(eps, epsilon_threshold=1.1)
    assert is_conn is False
    assert n_comp == 0


def test_connected_matrix_with_hole() -> None:
    """Verifies a centered circular hole embedded in a continuous dielectric matrix."""
    n = 40
    y, x = np.ogrid[:n, :n]
    r = np.hypot(x - n / 2, y - n / 2)

    # Dielectric matrix (eps=12) with air hole (eps=1) of radius 10 px
    eps = np.where(r < 10, 1.0, 12.0)
    is_conn, _n_comp, _msg = check_array_connectivity(
        eps, epsilon_threshold=1.1, check_pbc=True
    )
    assert is_conn is True


def test_severed_matrix_over_etched() -> None:
    """Verifies that an oversized hole severing periodic connections is flagged as disconnected."""
    n = 40
    y, x = np.ogrid[:n, :n]
    r = np.hypot(x - n / 2, y - n / 2)

    # Air hole of radius 22 px touching and severing the boundary of the 40x40 cell
    eps = np.where(r < 22, 1.0, 12.0)
    is_conn, _n_comp, _msg = check_array_connectivity(
        eps, epsilon_threshold=1.1, check_pbc=True
    )
    assert is_conn is False


def test_min_neck_width_tolerance() -> None:
    """Verifies that narrow necks below min_neck_width_px are severed by morphological opening."""
    n = 40
    eps = np.full((n, n), fill_value=1.0)
    # Cross structure of width 4 spanning both x and y
    eps[18:22, :] = 12.0
    eps[:, 18:22] = 12.0

    # With min_neck_width_px=3, width 4 survives opening
    is_conn_3, _, _ = check_array_connectivity(eps, min_neck_width_px=3, check_pbc=True)
    assert is_conn_3 is True

    # With min_neck_width_px=5, width 4 is eliminated by opening
    is_conn_5, _, _ = check_array_connectivity(eps, min_neck_width_px=5, check_pbc=True)
    assert is_conn_5 is False


def test_check_slab_connectivity_type_error() -> None:
    """Verifies TypeError on invalid input type."""
    with pytest.raises(TypeError):
        check_slab_connectivity("invalid_type")
