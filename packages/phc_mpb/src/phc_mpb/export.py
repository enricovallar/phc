"""Data table exporters for MPB eigenfrequencies and point-group irreducible representations.

Provides human-readable, columnar ASCII export routines reproducing canonical MPB tables:
- `freqs.data`: Table of reciprocal k-vectors, wavevector magnitudes kmag/2pi, and eigenfrequencies b1, b2, ...
- `irreps.data`: Table of wavevector magnitude kmag/2pi, band index, eigenfrequency, and parity/irrep label.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import meep as mp
import numpy as np


def compute_kmag_list(
    k_points: Sequence[Any],
    geometry_lattice: Any,
) -> list[float]:
    """Computes Cartesian wavevector magnitudes kmag/2pi for a sequence of k-points.

    Args:
        k_points: Sequence of mp.Vector3 reciprocal k-points.
        geometry_lattice: mp.Lattice instance defining the simulation domain.

    Returns:
        List of Cartesian wavevector magnitudes in units of 2pi/a.
    """
    kmag_list: list[float] = []
    for k in k_points:
        if isinstance(k, mp.Vector3):
            k_cart = mp.reciprocal_to_cartesian(k, geometry_lattice)
            kmag = float(np.sqrt(k_cart.x**2 + k_cart.y**2 + k_cart.z**2))
        elif hasattr(k, "__len__") and len(k) >= 3:
            vec = mp.Vector3(float(k[0]), float(k[1]), float(k[2]))
            k_cart = mp.reciprocal_to_cartesian(vec, geometry_lattice)
            kmag = float(np.sqrt(k_cart.x**2 + k_cart.y**2 + k_cart.z**2))
        else:
            kmag = 0.0
        kmag_list.append(kmag)
    return kmag_list


def format_freqs_data(
    k_points: Sequence[Any],
    geometry_lattice: Any,
    freqs: np.ndarray | Sequence[Sequence[float]],
) -> str:
    """Formats eigenfrequencies across the k-path into an aligned ASCII table (freqs.data).

    Table columns:
        # k1          k2          k3          kmag/2pi    b1          b2          b3          ...

    Args:
        k_points: Sequence of mp.Vector3 k-points along the Brillouin zone path.
        geometry_lattice: mp.Lattice defining unit cell basis vectors.
        freqs: 2D array or list of shape (num_k_points, num_bands) containing normalized
            dimensionless eigenfrequencies (omega * a / 2pi c).

    Returns:
        Formatted ASCII table string.

    Raises:
        ValueError: If row counts of k_points and freqs do not match.
    """
    freq_arr = np.asarray(freqs)
    if len(k_points) != freq_arr.shape[0]:
        raise ValueError(
            f"k_points length ({len(k_points)}) does not match freqs row count ({freq_arr.shape[0]})."
        )

    num_bands = freq_arr.shape[1]
    kmags = compute_kmag_list(k_points, geometry_lattice)

    # Build header
    header_cols = ["# k1        ", "k2        ", "k3        ", "kmag/2pi  "]
    for b in range(1, num_bands + 1):
        header_cols.append(f"b{b:<9d}")
    header_line = "  ".join(header_cols)

    lines: list[str] = [header_line]
    for i, k in enumerate(k_points):
        if isinstance(k, mp.Vector3):
            k1, k2, k3 = k.x, k.y, k.z
        else:
            k1, k2, k3 = float(k[0]), float(k[1]), float(k[2])
        kmag = kmags[i]

        row_vals = [
            f"{k1:10.6f}",
            f"{k2:10.6f}",
            f"{k3:10.6f}",
            f"{kmag:10.6f}",
        ]
        for b in range(num_bands):
            row_vals.append(f"{float(freq_arr[i, b]):10.6f}")
        lines.append("  ".join(row_vals))

    return "\n".join(lines) + "\n"


def format_irreps_data(
    symmetries: Sequence[dict[str, Any]],
    gamma_freqs: Sequence[float] | None = None,
    kmag: float = 0.0,
) -> str:
    """Formats point-group irreducible representations into an aligned ASCII table (irreps.data).

    Table columns:
        # kmag/2pi    band_index  frequency   parity

    Args:
        symmetries: List of band symmetry dictionaries from compute_band_symmetries.
        gamma_freqs: Optional sequence of frequencies at the high-symmetry point.
        kmag: Wavevector magnitude in units of 2pi/a (default: 0.0 for Gamma).

    Returns:
        Formatted ASCII table string.
    """
    lines: list[str] = [
        f"{'# kmag/2pi':<12}  {'band_index':<10}  {'frequency':<12}  {'parity':<8}"
    ]

    for s in symmetries:
        b = int(s.get("band", 0))
        if gamma_freqs is not None and 0 < b <= len(gamma_freqs):
            freq = float(gamma_freqs[b - 1])
        else:
            freq = float(s.get("freq", 0.0))
        irrep = str(s.get("irrep", "N/A"))

        lines.append(f"  {kmag:<10.6f}  {b:<10d}  {freq:<12.6f}  {irrep:<8s}")

    return "\n".join(lines) + "\n"


def export_freqs_data(
    filepath: str | Path,
    k_points: Sequence[Any],
    geometry_lattice: Any,
    freqs: np.ndarray | Sequence[Sequence[float]],
) -> Path:
    """Writes the freqs.data ASCII table to disk.

    Args:
        filepath: Destination file path.
        k_points: Sequence of mp.Vector3 k-points.
        geometry_lattice: mp.Lattice instance.
        freqs: 2D array of normalized frequencies.

    Returns:
        Resolved Path to the saved file.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    table_str = format_freqs_data(k_points, geometry_lattice, freqs)
    path.write_text(table_str, encoding="utf-8")
    return path


def export_irreps_data(
    filepath: str | Path,
    symmetries: Sequence[dict[str, Any]],
    gamma_freqs: Sequence[float] | None = None,
    kmag: float = 0.0,
) -> Path:
    """Writes the irreps.data ASCII table to disk.

    Args:
        filepath: Destination file path.
        symmetries: List of band symmetry records.
        gamma_freqs: Optional frequency list at Gamma.
        kmag: Wavevector magnitude (default: 0.0).

    Returns:
        Resolved Path to the saved file.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    table_str = format_irreps_data(symmetries, gamma_freqs=gamma_freqs, kmag=kmag)
    path.write_text(table_str, encoding="utf-8")
    return path
