"""Lattice utilities and converters for MPB band structure simulations."""

from typing import Literal

import meep as mp
import numpy as np
from phc_layout.lattice import HexagonalLattice, Lattice, SquareLattice


def lattice_to_mpb_lattice(
    lattice: Lattice,
    dimension: Literal["2D", "3D_slab"] = "2D",
    supercell: tuple[int, int] = (1, 1),
    supercell_z: float = 4.0,
    normalize: bool = True,
) -> mp.Lattice:
    """Converts a phc_layout.lattice.Lattice object into an mp.Lattice for MPB.

    Args:
        lattice: Lattice instance from phc_layout.lattice (e.g. HexagonalLattice, SquareLattice).
        dimension: "2D" (infinite along z) or "3D_slab" (finite thickness with supercell buffer).
        supercell: (Nx, Ny) supercell multipliers in basis directions.
        supercell_z: Supercell height in units of a for 3D slabs.
        normalize: If True, normalizes basis vectors by pitch a so that length units
            in MPB are dimensionless (a = 1).

    Returns:
        mp.Lattice object configured for MPB simulation.

    Raises:
        TypeError: If lattice is not an instance of Lattice.
        ValueError: If dimension is invalid.
    """
    if not isinstance(lattice, Lattice):
        raise TypeError(
            f"Expected phc_layout.lattice.Lattice instance, got {type(lattice)}."
        )

    if dimension not in ("2D", "3D_slab"):
        raise ValueError(
            f"Unknown dimension '{dimension}'. Expected '2D' or '3D_slab'."
        )

    a1 = np.array(lattice.a1, dtype=float)
    a2 = np.array(lattice.a2, dtype=float)
    a3 = np.array(lattice.a3, dtype=float)

    if normalize:
        if hasattr(lattice, "a"):
            pitch = float(lattice.a)
        elif hasattr(lattice, "pitch"):
            pitch = float(lattice.pitch)
        else:
            pitch = float(np.linalg.norm(a1[:2]))

        if pitch <= 0.0:
            raise ValueError(f"Lattice pitch must be positive, got {pitch}.")

        b1 = a1 / pitch
        b2 = a2 / pitch
        norm_a3 = float(np.linalg.norm(a3))
        b3 = (a3 / norm_a3) if norm_a3 > 0.0 else np.array([0.0, 0.0, 1.0])
    else:
        b1, b2, b3 = a1, a2, a3

    nx, ny = supercell
    nz = 0.0 if dimension == "2D" else float(supercell_z)

    return mp.Lattice(
        basis1=mp.Vector3(float(b1[0]), float(b1[1]), float(b1[2])),
        basis2=mp.Vector3(float(b2[0]), float(b2[1]), float(b2[2])),
        basis3=mp.Vector3(float(b3[0]), float(b3[1]), float(b3[2])),
        size=mp.Vector3(float(nx), float(ny), float(nz)),
    )


to_mpb_lattice = lattice_to_mpb_lattice


def create_lattice(
    lattice_type: Literal["hexagonal", "square"] | Lattice = "hexagonal",
    pitch: float = 1.0,
    dimension: Literal["2D", "3D_slab"] = "2D",
    supercell: tuple[int, int] = (1, 1),
    supercell_z: float = 4.0,
) -> mp.Lattice:
    """Builds an mp.Lattice object for MPB simulations.

    Supports both preset names ('hexagonal', 'square') and phc_layout.lattice.Lattice objects.

    Args:
        lattice_type: "hexagonal" (triangular), "square", or a phc_layout.lattice.Lattice instance.
        pitch: Physical lattice constant in microns (used when lattice_type is str).
        dimension: "2D" (infinite along z) or "3D_slab" (finite thickness with supercell buffer).
        supercell: (Nx, Ny) supercell multipliers in basis directions.
        supercell_z: Supercell height in units of a for 3D slabs.

    Returns:
        mp.Lattice object configured for MPB simulation.

    Raises:
        ValueError: If lattice_type string is unrecognized.
    """
    if isinstance(lattice_type, Lattice):
        return lattice_to_mpb_lattice(
            lattice=lattice_type,
            dimension=dimension,
            supercell=supercell,
            supercell_z=supercell_z,
        )

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
    lattice_type: Literal["hexagonal", "square"] | Lattice = "hexagonal",
    k_density: int = 20,
) -> tuple[list[mp.Vector3], list[str], list[int]]:
    """Generates an interpolated k-path across the irreducible Brillouin zone.

    Args:
        lattice_type: 'hexagonal', 'square', or a phc_layout.lattice.Lattice instance.
        k_density: Number of interpolation points between vertices.

    Returns:
        tuple of (k_points, node_labels, node_indices).

    Raises:
        ValueError: If lattice_type is not recognized or geometry cannot be inferred.
    """
    if isinstance(lattice_type, HexagonalLattice):
        # Hexagonal lattice in crystallographic basis:
        # a1 = (1, 0, 0), a2 = (-0.5, sqrt(3)/2, 0)
        # K-path convention: Γ is the second high-symmetry point (M→Γ→K→M)
        corners = [
            (0.0, 0.5, 0.0),  # M
            (0.0, 0.0, 0.0),  # Gamma
            (1.0 / 3.0, 1.0 / 3.0, 0.0),  # K
            (0.0, 0.5, 0.0),  # M
        ]
        labels = ["M", "Γ", "K", "M"]
    elif isinstance(lattice_type, SquareLattice) or lattice_type == "square":
        # K-path convention: Γ is the second high-symmetry point (X→Γ→M→X)
        corners = [
            (0.0, 0.5, 0.0),  # X
            (0.0, 0.0, 0.0),  # Gamma
            (0.5, 0.5, 0.0),  # M
            (0.0, 0.5, 0.0),  # X
        ]
        labels = ["X", "Γ", "M", "X"]
    elif lattice_type == "hexagonal":
        # Legacy hexagonal basis: basis1=(sqrt(3)/2, 1/2), basis2=(sqrt(3)/2, -1/2)
        # K-path convention: Γ is the second high-symmetry point (M→Γ→K→M)
        corners = [
            (0.0, 0.5, 0.0),  # M
            (0.0, 0.0, 0.0),  # Gamma
            (-1.0 / 3.0, 1.0 / 3.0, 0.0),  # K
            (0.0, 0.5, 0.0),  # M
        ]
        labels = ["M", "Γ", "K", "M"]
    elif isinstance(lattice_type, Lattice):
        # Infer geometry from in-plane basis angle
        a1_2d = np.array(lattice_type.a1[:2])
        a2_2d = np.array(lattice_type.a2[:2])
        cos_angle = np.dot(a1_2d, a2_2d) / (
            np.linalg.norm(a1_2d) * np.linalg.norm(a2_2d)
        )
        angle_deg = float(np.rad2deg(np.arccos(np.clip(cos_angle, -1.0, 1.0))))
        if np.isclose(angle_deg, 90.0, atol=1e-2):
            return get_high_symmetry_kpath(lattice_type="square", k_density=k_density)
        elif np.isclose(angle_deg, 120.0, atol=1e-2) or np.isclose(
            angle_deg, 60.0, atol=1e-2
        ):
            return get_high_symmetry_kpath(
                lattice_type=HexagonalLattice(a=1.0), k_density=k_density
            )
        else:
            raise ValueError(
                f"Cannot automatically infer high-symmetry k-path for lattice angle {angle_deg:.1f} degrees."
            )
    else:
        raise ValueError(f"Unknown lattice_type: {lattice_type}")

    mp_corners = [mp.Vector3(*pt) for pt in corners]
    interpolated = mp.interpolate(k_density, mp_corners)
    node_indices = [i * (k_density + 1) for i in range(len(labels))]
    return interpolated, labels, node_indices
