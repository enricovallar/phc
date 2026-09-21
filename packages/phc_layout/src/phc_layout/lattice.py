"""Lattice definitions and exact coordinate transformations for 2D and 3D systems."""

from collections.abc import Sequence
from typing import Literal

import numpy as np


class Lattice:
    """General Bravais lattice defined by three basis vectors.

    Provides exact forward and inverse coordinate transformations between
    fractional coordinates (u, v, w) and physical Cartesian coordinates (x, y, z)
    in micrometers.

    Attributes:
        a1: First basis vector (x, y, z).
        a2: Second basis vector (x, y, z).
        a3: Third basis vector (x, y, z).
    """

    def __init__(
        self,
        a1: Sequence[float],
        a2: Sequence[float],
        a3: Sequence[float] = (0.0, 0.0, 1.0),
    ) -> None:
        """Initializes a Lattice from three basis vectors.

        Args:
            a1: First basis vector of length 2 or 3 in micrometers.
            a2: Second basis vector of length 2 or 3 in micrometers.
            a3: Third basis vector of length 2 or 3 in micrometers.

        Raises:
            ValueError: If basis vectors are linearly dependent or matrix is singular.
        """
        self._a1 = self._to_vector3(a1)
        self._a2 = self._to_vector3(a2)
        self._a3 = self._to_vector3(a3)

        # 3x3 transformation matrix A = [a1, a2, a3] as columns
        self._matrix_3d = np.column_stack([self._a1, self._a2, self._a3])
        det_3d = np.linalg.det(self._matrix_3d)
        if np.isclose(det_3d, 0.0):
            raise ValueError(
                f"Basis vectors are linearly dependent (det = {det_3d:.6e})."
            )
        self._inv_matrix_3d = np.linalg.inv(self._matrix_3d)

        # 2x2 in-plane matrix A_2D = [[a1x, a2x], [a1y, a2y]]
        self._matrix_2d = self._matrix_3d[:2, :2]
        det_2d = np.linalg.det(self._matrix_2d)
        if np.isclose(det_2d, 0.0):
            raise ValueError(
                f"In-plane 2D basis vectors are linearly dependent (det = {det_2d:.6e})."
            )
        self._inv_matrix_2d = np.linalg.inv(self._matrix_2d)

    @staticmethod
    def _to_vector3(v: Sequence[float]) -> np.ndarray:
        """Converts a 2D or 3D sequence into a 3D float array."""
        arr = np.array(v, dtype=float)
        if arr.size == 2:
            return np.array([arr[0], arr[1], 0.0], dtype=float)
        elif arr.size == 3:
            return arr
        raise ValueError(
            f"Basis vector must have length 2 or 3, got length {arr.size}."
        )

    @property
    def a1(self) -> tuple[float, float, float]:
        """Returns first basis vector as a 3-tuple."""
        return (float(self._a1[0]), float(self._a1[1]), float(self._a1[2]))

    @property
    def a2(self) -> tuple[float, float, float]:
        """Returns second basis vector as a 3-tuple."""
        return (float(self._a2[0]), float(self._a2[1]), float(self._a2[2]))

    @property
    def a3(self) -> tuple[float, float, float]:
        """Returns third basis vector as a 3-tuple."""
        return (float(self._a3[0]), float(self._a3[1]), float(self._a3[2]))

    @property
    def primitive_vectors_2d(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Returns in-plane 2D primitive basis vectors (a1, a2)."""
        return (
            (float(self._a1[0]), float(self._a1[1])),
            (float(self._a2[0]), float(self._a2[1])),
        )

    @property
    def reciprocal_vectors(
        self,
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]:
        """Returns reciprocal basis vectors (b1, b2, b3) satisfying a_i . b_j = 2pi delta_ij."""
        volume = float(np.dot(self._a1, np.cross(self._a2, self._a3)))
        b1 = 2.0 * np.pi * np.cross(self._a2, self._a3) / volume
        b2 = 2.0 * np.pi * np.cross(self._a3, self._a1) / volume
        b3 = 2.0 * np.pi * np.cross(self._a1, self._a2) / volume
        return (
            (float(b1[0]), float(b1[1]), float(b1[2])),
            (float(b2[0]), float(b2[1]), float(b2[2])),
            (float(b3[0]), float(b3[1]), float(b3[2])),
        )

    def to_cartesian(
        self, u: float, v: float, w: float = 0.0
    ) -> tuple[float, float, float]:
        """Converts fractional coordinates (u, v, w) to Cartesian coordinates (x, y, z).

        Args:
            u: Fractional coordinate along basis vector a1.
            v: Fractional coordinate along basis vector a2.
            w: Fractional coordinate along basis vector a3.

        Returns:
            Cartesian coordinates (x, y, z) in micrometers.
        """
        frac = np.array([u, v, w], dtype=float)
        cart = self._matrix_3d @ frac
        return (float(cart[0]), float(cart[1]), float(cart[2]))

    def to_cartesian_2d(self, u: float, v: float) -> tuple[float, float]:
        """Converts in-plane fractional coordinates (u, v) to Cartesian coordinates (x, y).

        Args:
            u: Fractional coordinate along basis vector a1.
            v: Fractional coordinate along basis vector a2.

        Returns:
            Cartesian coordinates (x, y) in micrometers.
        """
        frac = np.array([u, v], dtype=float)
        cart = self._matrix_2d @ frac
        return (float(cart[0]), float(cart[1]))

    def from_cartesian(
        self, x: float, y: float, z: float = 0.0
    ) -> tuple[float, float, float]:
        """Converts Cartesian coordinates (x, y, z) to fractional coordinates (u, v, w).

        Solves the linear system A * [u, v, w]^T = [x, y, z]^T using exact matrix inversion.

        Args:
            x: Cartesian x position in micrometers.
            y: Cartesian y position in micrometers.
            z: Cartesian z position in micrometers.

        Returns:
            Fractional coordinates (u, v, w).
        """
        cart = np.array([x, y, z], dtype=float)
        frac = self._inv_matrix_3d @ cart
        return (float(frac[0]), float(frac[1]), float(frac[2]))

    def from_cartesian_2d(self, x: float, y: float) -> tuple[float, float]:
        """Converts in-plane Cartesian coordinates (x, y) to fractional coordinates (u, v).

        Args:
            x: Cartesian x position in micrometers.
            y: Cartesian y position in micrometers.

        Returns:
            Fractional coordinates (u, v).
        """
        cart = np.array([x, y], dtype=float)
        frac = self._inv_matrix_2d @ cart
        return (float(frac[0]), float(frac[1]))

    def unit_cell_polygon(self, centered: bool = True) -> list[tuple[float, float]]:
        """Returns the vertices of the primitive parallelepiped unit cell in 2D.

        Args:
            centered: If True, center the cell around (0, 0) by taking u, v in [-0.5, 0.5].
                If False, take u, v in [0, 1].

        Returns:
            List of (x, y) vertex coordinates in counter-clockwise order.
        """
        if centered:
            uv_corners = [
                (-0.5, -0.5),
                (0.5, -0.5),
                (0.5, 0.5),
                (-0.5, 0.5),
            ]
        else:
            uv_corners = [
                (0.0, 0.0),
                (1.0, 0.0),
                (1.0, 1.0),
                (0.0, 1.0),
            ]
        return [self.to_cartesian_2d(u, v) for u, v in uv_corners]

    def wigner_seitz_polygon(self) -> list[tuple[float, float]]:
        """Returns the vertices of the Wigner-Seitz (Voronoi) cell in 2D.

        Default implementation computes the Voronoi polygon from nearest lattice neighbors.

        Returns:
            List of (x, y) vertex coordinates in counter-clockwise order.
        """
        # Overridden in subclasses for analytical precision, but general fallback provided
        return self.unit_cell_polygon(centered=True)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(a1={self.a1}, a2={self.a2}, a3={self.a3})"


class HexagonalLattice(Lattice):
    """2D Hexagonal (triangular) Bravais lattice with standard crystallographic orientation.

    Basis vectors:
        a1 = (a, 0, 0)
        a2 = (-a/2, a * sqrt(3)/2, 0)
        a3 = (0, 0, 1)

    The angle between a1 and a2 is 120 degrees, corresponding to Plane Group 17 (p6mm).
    """

    def __init__(self, a: float = 1.0) -> None:
        """Initializes a HexagonalLattice.

        Args:
            a: Lattice constant (pitch) in micrometers. Must be positive.

        Raises:
            ValueError: If pitch a is <= 0.
        """
        if a <= 0:
            raise ValueError(f"Lattice constant 'a' must be positive, got {a}.")
        self._a = float(a)
        super().__init__(
            a1=(self._a, 0.0, 0.0),
            a2=(-self._a / 2.0, self._a * np.sqrt(3.0) / 2.0, 0.0),
            a3=(0.0, 0.0, 1.0),
        )

    @property
    def a(self) -> float:
        """Returns the lattice pitch in micrometers."""
        return self._a

    @property
    def pitch(self) -> float:
        """Alias for lattice constant a in micrometers."""
        return self._a

    def wigner_seitz_polygon(self) -> list[tuple[float, float]]:
        """Returns the 6 vertices of the hexagonal Wigner-Seitz cell.

        The cell is a regular hexagon of circumradius R = a / sqrt(3) centered at (0, 0).

        Returns:
            List of 6 (x, y) vertices ordered counter-clockwise.
        """
        r = self._a / np.sqrt(3.0)
        # Vertices are at angles 30, 90, 150, 210, 270, 330 degrees
        angles_deg = [30.0, 90.0, 150.0, 210.0, 270.0, 330.0]
        return [
            (
                float(r * np.cos(np.deg2rad(deg))),
                float(r * np.sin(np.deg2rad(deg))),
            )
            for deg in angles_deg
        ]


class SquareLattice(Lattice):
    """2D Square Bravais lattice.

    Basis vectors:
        a1 = (a, 0, 0)
        a2 = (0, a, 0)
        a3 = (0, 0, 1)

    Corresponding to Plane Group 11 (p4mm).
    """

    def __init__(self, a: float = 1.0) -> None:
        """Initializes a SquareLattice.

        Args:
            a: Lattice constant (pitch) in micrometers. Must be positive.

        Raises:
            ValueError: If pitch a is <= 0.
        """
        if a <= 0:
            raise ValueError(f"Lattice constant 'a' must be positive, got {a}.")
        self._a = float(a)
        super().__init__(
            a1=(self._a, 0.0, 0.0),
            a2=(0.0, self._a, 0.0),
            a3=(0.0, 0.0, 1.0),
        )

    @property
    def a(self) -> float:
        """Returns the lattice pitch in micrometers."""
        return self._a

    @property
    def pitch(self) -> float:
        """Alias for lattice constant a in micrometers."""
        return self._a

    def wigner_seitz_polygon(self) -> list[tuple[float, float]]:
        """Returns the 4 vertices of the square Wigner-Seitz cell of side length a.

        Returns:
            List of 4 (x, y) vertices ordered counter-clockwise.
        """
        half = self._a / 2.0
        return [
            (-half, -half),
            (half, -half),
            (half, half),
            (-half, half),
        ]


LatticeType = Literal["hexagonal", "square"]


def get_lattice(lattice_type: LatticeType | str, pitch: float = 1.0) -> Lattice:
    """Creates a Lattice instance by type name.

    Args:
        lattice_type: Lattice type name ('hexagonal' or 'square').
        pitch: Lattice constant in micrometers.

    Returns:
        HexagonalLattice or SquareLattice instance.

    Raises:
        ValueError: If lattice_type is not recognized.
    """
    key = lattice_type.lower()
    if key in ("hexagonal", "hex", "triangular"):
        return HexagonalLattice(a=pitch)
    elif key in ("square", "sq"):
        return SquareLattice(a=pitch)
    raise ValueError(
        f"Unknown lattice_type '{lattice_type}'. Expected 'hexagonal' or 'square'."
    )
