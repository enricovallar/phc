from typing import Literal

import meep as mp
import numpy as np


def create_lattice(
    lattice_type: Literal["hexagonal", "square"] = "hexagonal",
    pitch: float = 1.0,
    dimension: Literal["2D", "3D_slab"] = "2D",
    supercell: tuple[int, int] = (1, 1),
    supercell_z: float = 5.0,
) -> mp.Lattice:
    """Builds an mp.Lattice object for MPB simulations.

    Args:
        lattice_type: "hexagonal" (triangular) or "square".
        pitch: Physical lattice constant in microns.
        dimension: "2D" (infinite along z) or "3D_slab" (finite thickness with supercell buffer).
        supercell: (Nx, Ny) supercell multipliers in basis directions.
        supercell_z: Supercell height in units of a for 3D slabs.
    """
    nx, ny = supercell
    nz = 0.0 if dimension == "2D" else float(supercell_z)

    if lattice_type == "hexagonal":
        basis1 = (0.5 * np.sqrt(3), 0.5, 0.0)
        basis2 = (0.5 * np.sqrt(3), -0.5, 0.0)
    elif lattice_type == "square":
        basis1 = (1.0, 0.0, 0.0)
        basis2 = (0.0, 1.0, 0.0)
    else:
        raise ValueError(
            f"Unknown lattice_type: {lattice_type}. Use 'hexagonal' or 'square'."
        )

    basis3 = (0.0, 0.0, 1.0)

    return mp.Lattice(
        basis1=mp.Vector3(*basis1),
        basis2=mp.Vector3(*basis2),
        basis3=mp.Vector3(*basis3),
        size=mp.Vector3(nx, ny, nz),
    )


def get_high_symmetry_kpath(
    lattice_type: Literal["hexagonal", "square"] = "hexagonal",
    k_density: int = 20,
) -> tuple[list[mp.Vector3], list[str], list[int]]:
    """Generates an interpolated k-path across the irreducible Brillouin zone.

    Returns:
        tuple of (k_points, node_labels, node_indices)
    """
    if lattice_type == "hexagonal":
        corners = [
            (0.0, 0.0, 0.0),  # Gamma
            (0.0, 0.5, 0.0),  # M
            (-1.0 / 3.0, 1.0 / 3.0, 0.0),  # K
            (0.0, 0.0, 0.0),  # Gamma
        ]
        labels = ["Γ", "M", "K", "Γ"]
    elif lattice_type == "square":
        corners = [
            (0.0, 0.0, 0.0),  # Gamma
            (0.0, 0.5, 0.0),  # X
            (0.5, 0.5, 0.0),  # M
            (0.0, 0.0, 0.0),  # Gamma
        ]
        labels = ["Γ", "X", "M", "Γ"]
    else:
        raise ValueError(f"Unknown lattice_type: {lattice_type}")

    mp_corners = [mp.Vector3(*pt) for pt in corners]
    interpolated = mp.interpolate(k_density, mp_corners)
    node_indices = [i * (k_density + 1) for i in range(len(labels))]
    return interpolated, labels, node_indices
