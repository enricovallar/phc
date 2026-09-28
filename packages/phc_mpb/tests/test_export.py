"""Unit tests for MPB ASCII table export module."""

import tempfile
from pathlib import Path

import meep as mp
import numpy as np
import pytest
from phc_mpb.export import (
    compute_kmag_list,
    export_freqs_data,
    export_irreps_data,
    format_freqs_data,
    format_irreps_data,
)


def test_compute_kmag_list() -> None:
    """Tests Cartesian wavevector magnitude calculation for square lattice."""
    lat = mp.Lattice(size=mp.Vector3(1, 1, 0))
    k_points = [
        mp.Vector3(0, 0, 0),
        mp.Vector3(0.5, 0, 0),
        [0.5, 0.5, 0],
    ]
    kmags = compute_kmag_list(k_points, lat)
    assert len(kmags) == 3
    assert np.isclose(kmags[0], 0.0)
    assert np.isclose(kmags[1], 0.5)
    assert np.isclose(kmags[2], np.sqrt(0.5**2 + 0.5**2))


def test_format_freqs_data() -> None:
    """Tests string formatting of freqs.data."""
    lat = mp.Lattice(size=mp.Vector3(1, 1, 0))
    k_points = [mp.Vector3(0, 0, 0), mp.Vector3(0.5, 0, 0)]
    freqs = np.array([[0.1, 0.2], [0.3, 0.4]])

    text = format_freqs_data(k_points, lat, freqs)
    lines = text.strip().split("\n")
    assert len(lines) == 3
    assert (
        "k1" in lines[0]
        and "kmag/2pi" in lines[0]
        and "b1" in lines[0]
        and "b2" in lines[0]
    )

    vals_row1 = lines[1].split()
    assert float(vals_row1[0]) == 0.0
    assert float(vals_row1[3]) == 0.0
    assert np.isclose(float(vals_row1[4]), 0.1)
    assert np.isclose(float(vals_row1[5]), 0.2)

    with pytest.raises(ValueError, match="k_points length"):
        format_freqs_data(k_points, lat, np.array([[0.1, 0.2]]))


def test_format_irreps_data() -> None:
    """Tests string formatting of irreps.data."""
    symmetries = [
        {"band": 1, "freq": 0.25, "irrep": "A_1"},
        {"band": 2, "freq": 0.50, "irrep": "E_1"},
    ]
    text = format_irreps_data(symmetries, kmag=0.0)
    lines = text.strip().split("\n")
    assert len(lines) == 3
    assert "kmag/2pi" in lines[0] and "band_index" in lines[0] and "parity" in lines[0]

    vals_row1 = lines[1].split()
    assert float(vals_row1[0]) == 0.0
    assert int(vals_row1[1]) == 1
    assert np.isclose(float(vals_row1[2]), 0.25)
    assert vals_row1[3] == "A_1"


def test_export_files() -> None:
    """Tests exporting freqs.data and irreps.data to disk."""
    lat = mp.Lattice(size=mp.Vector3(1, 1, 0))
    k_points = [mp.Vector3(0, 0, 0), mp.Vector3(0.5, 0, 0)]
    freqs = np.array([[0.1, 0.2], [0.3, 0.4]])
    symmetries = [{"band": 1, "freq": 0.1, "irrep": "A_1"}]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        freq_file = tmp_path / "freqs.data"
        irrep_file = tmp_path / "irreps.data"

        out_freq = export_freqs_data(freq_file, k_points, lat, freqs)
        assert out_freq.exists()
        assert "kmag/2pi" in out_freq.read_text(encoding="utf-8")

        out_irrep = export_irreps_data(irrep_file, symmetries)
        assert out_irrep.exists()
        assert "parity" in out_irrep.read_text(encoding="utf-8")
