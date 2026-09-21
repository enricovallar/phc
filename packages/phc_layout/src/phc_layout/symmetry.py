"""Crystallographic point group symmetries and 2D Wyckoff positions."""

from collections.abc import Callable, Sequence
from typing import Any

from phc_layout.lattice import Lattice


class PointGroup:
    """Base class representing a crystallographic point group in 2D.

    Attributes:
        name: Standard Schoenflies / Hermann-Mauguin notation name.
        order: Group order (number of symmetry operations).
    """

    def __init__(self, name: str, order: int = 1) -> None:
        """Initializes a PointGroup.

        Args:
            name: Name of the point group (e.g., 'C4v', 'C6v').
            order: Order of the point group.
        """
        self._name = name
        self._order = order

    @property
    def name(self) -> str:
        """Returns the point group name."""
        return self._name

    @property
    def order(self) -> int:
        """Returns the order of the point group."""
        return self._order

    def __hash__(self) -> int:
        return hash(self._name)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PointGroup):
            return self._name == other._name
        if isinstance(other, str):
            return self._name.lower() == other.lower()
        return False

    def __repr__(self) -> str:
        return self._name


class C4v(PointGroup):
    """Square 4mm point group (order 8).

    Generators include 4-fold rotation C4 and mirror reflections sigma_v, sigma_d.
    """

    def __init__(self) -> None:
        super().__init__("C4v", order=8)


class C6v(PointGroup):
    """Hexagonal 6mm point group (order 12).

    Generators include 6-fold rotation C6 and mirror reflections sigma_v, sigma_d.
    """

    def __init__(self) -> None:
        super().__init__("C6v", order=12)


def get_point_group(pg: PointGroup | str) -> PointGroup:
    """Normalizes a PointGroup instance or string name to a PointGroup object.

    Args:
        pg: PointGroup instance or string ('C4v', 'C6v').

    Returns:
        PointGroup instance.

    Raises:
        ValueError: If point group is not recognized.
    """
    if isinstance(pg, PointGroup):
        return pg
    name = str(pg).strip()
    if name.lower() in ("c4v", "4mm", "c4"):
        return C4v()
    if name.lower() in ("c6v", "6mm", "c6"):
        return C6v()
    raise ValueError(f"Unknown point group '{pg}'. Expected 'C4v' or 'C6v'.")


class WyckoffPosition:
    """Represents a set of symmetry-equivalent fractional coordinates (Wyckoff site).

    Attributes:
        positions: Tuple of (u, v) fractional coordinate pairs.
        letter: Wyckoff letter label (e.g. '1a', '2b', '6d').
        point_group: Associated PointGroup instance.
    """

    def __init__(
        self,
        positions: Sequence[Sequence[float]] | Sequence[float],
        letter: str | None = None,
        point_group: PointGroup | None = None,
    ) -> None:
        """Initializes a WyckoffPosition.

        Args:
            positions: Sequence of (u, v) coordinates or single coordinate (u, v).
            letter: Optional Wyckoff letter label.
            point_group: Optional PointGroup reference.

        Raises:
            ValueError: If coordinate points do not have 2 components.
        """
        # Normalize to tuple of (u, v) float tuples
        if len(positions) == 2 and isinstance(positions[0], (int, float)):
            pts = ((float(positions[0]), float(positions[1])),)
        else:
            normalized: list[tuple[float, float]] = []
            for pt in positions:
                pt_seq = tuple(pt)  # type: ignore[arg-type]
                if len(pt_seq) != 2:
                    raise ValueError(
                        f"Each Wyckoff coordinate must have (u, v), got {pt_seq}."
                    )
                normalized.append((float(pt_seq[0]), float(pt_seq[1])))
            pts = tuple(normalized)

        self._positions = pts
        self._letter = letter
        self._point_group = point_group

    @property
    def positions(self) -> tuple[tuple[float, float], ...]:
        """Returns the tuple of fractional (u, v) coordinates."""
        return self._positions

    @property
    def letter(self) -> str | None:
        """Returns the Wyckoff letter label."""
        return self._letter

    @property
    def point_group(self) -> PointGroup | None:
        """Returns the associated PointGroup."""
        return self._point_group

    @property
    def multiplicity(self) -> int:
        """Returns the multiplicity (number of equivalent positions)."""
        return len(self._positions)

    def to_cartesian(
        self, lattice: Lattice, wrap_to_cell: bool = False
    ) -> list[tuple[float, float]]:
        """Converts all fractional coordinates to micrometer Cartesian coordinates.

        Args:
            lattice: Lattice defining the primitive basis vectors.
            wrap_to_cell: If True, wraps fractional coordinates into [-0.5, 0.5)
                before conversion to ensure all holes reside in the primary cell.

        Returns:
            List of (x, y) Cartesian positions in micrometers.
        """
        cartesian_coords: list[tuple[float, float]] = []
        for u, v in self._positions:
            if wrap_to_cell:
                u = ((u + 0.5) % 1.0) - 0.5
                v = ((v + 0.5) % 1.0) - 0.5
            cartesian_coords.append(lattice.to_cartesian_2d(u, v))
        return cartesian_coords

    def __len__(self) -> int:
        return len(self._positions)

    def __iter__(self):
        return iter(self._positions)

    def __repr__(self) -> str:
        letter_str = f"'{self._letter}', " if self._letter else ""
        return f"WyckoffPosition({letter_str}{self._positions})"

    def __str__(self) -> str:
        return f"{self._positions}"


WyckoffValue = WyckoffPosition | Callable[..., WyckoffPosition]

# Standard 2D Wyckoff positions table for C6v (p6mm) and C4v (p4mm)
WYCKOFF_POSITIONS_2D: dict[str, dict[str, WyckoffValue]] = {
    "C6v": {
        "1a": WyckoffPosition(((0.0, 0.0),), letter="1a", point_group=C6v()),
        "2b": WyckoffPosition(
            ((1.0 / 3.0, 2.0 / 3.0), (2.0 / 3.0, 1.0 / 3.0)),
            letter="2b",
            point_group=C6v(),
        ),
        "3c": WyckoffPosition(
            ((0.5, 0.0), (0.0, 0.5), (0.5, 0.5)),
            letter="3c",
            point_group=C6v(),
        ),
        "6d": lambda x: WyckoffPosition(
            (
                (float(x), 0.0),
                (0.0, float(x)),
                (-float(x), -float(x)),
                (-float(x), 0.0),
                (0.0, -float(x)),
                (float(x), float(x)),
            ),
            letter="6d",
            point_group=C6v(),
        ),
        "6e": lambda x: WyckoffPosition(
            (
                (float(x), -float(x)),
                (float(x), 2.0 * float(x)),
                (-2.0 * float(x), -float(x)),
                (-float(x), float(x)),
                (-float(x), -2.0 * float(x)),
                (2.0 * float(x), float(x)),
            ),
            letter="6e",
            point_group=C6v(),
        ),
        "12f": lambda p: WyckoffPosition(
            (
                (float(p[0]), float(p[1])),
                (-float(p[1]), float(p[0]) - float(p[1])),
                (-float(p[0]) + float(p[1]), -float(p[0])),
                (-float(p[0]), -float(p[1])),
                (float(p[1]), -float(p[0]) + float(p[1])),
                (float(p[0]) - float(p[1]), float(p[0])),
                (-float(p[1]), -float(p[0])),
                (float(p[0]) - float(p[1]), -float(p[1])),
                (float(p[0]), float(p[0]) - float(p[1])),
                (float(p[1]), float(p[0])),
                (-float(p[0]) + float(p[1]), float(p[1])),
                (-float(p[0]), -float(p[0]) + float(p[1])),
            ),
            letter="12f",
            point_group=C6v(),
        ),
    },
    "C4v": {
        "1a": WyckoffPosition(((0.0, 0.0),), letter="1a", point_group=C4v()),
        "1b": WyckoffPosition(((0.5, 0.5),), letter="1b", point_group=C4v()),
        "2c": WyckoffPosition(
            ((0.5, 0.0), (0.0, 0.5)),
            letter="2c",
            point_group=C4v(),
        ),
        "4d": lambda x: WyckoffPosition(
            (
                (float(x), float(x)),
                (-float(x), float(x)),
                (float(x), -float(x)),
                (-float(x), -float(x)),
            ),
            letter="4d",
            point_group=C4v(),
        ),
        "4e": lambda x: WyckoffPosition(
            (
                (float(x), 0.0),
                (-float(x), 0.0),
                (0.0, float(x)),
                (0.0, -float(x)),
            ),
            letter="4e",
            point_group=C4v(),
        ),
        "4f": lambda x: WyckoffPosition(
            (
                (float(x), 0.5),
                (-float(x), 0.5),
                (0.5, float(x)),
                (0.5, -float(x)),
            ),
            letter="4f",
            point_group=C4v(),
        ),
        "8g": lambda p: WyckoffPosition(
            (
                (float(p[0]), float(p[1])),
                (-float(p[0]), -float(p[1])),
                (-float(p[1]), float(p[0])),
                (float(p[1]), -float(p[0])),
                (float(p[0]), -float(p[1])),
                (-float(p[0]), float(p[1])),
                (-float(p[1]), -float(p[0])),
                (float(p[1]), float(p[0])),
            ),
            letter="8g",
            point_group=C4v(),
        ),
    },
}


def get_wyckoff_position(
    point_group: PointGroup | str,
    letter: str,
    param: float | Sequence[float] | None = None,
    **kwargs: Any,
) -> WyckoffPosition:
    """Retrieves and evaluates a Wyckoff position for a given point group.

    Args:
        point_group: PointGroup instance or string ('C4v', 'C6v').
        letter: Wyckoff position label (e.g., '1a', '2b', '6d').
        param: Numerical parameter for parameterized positions (e.g. x for 6d/4d,
            or (x, y) tuple for 8g/12f).
        **kwargs: Alternative parameter keywords (e.g. x=0.2, p=(0.1, 0.2)).

    Returns:
        Evaluated WyckoffPosition instance.

    Raises:
        ValueError: If the point group or Wyckoff position is unknown, or if required
            parameters are missing.
    """
    pg = get_point_group(point_group)
    pg_key = "C6v" if pg == C6v() else "C4v"

    positions_map = WYCKOFF_POSITIONS_2D.get(pg_key, {})
    if letter not in positions_map:
        raise ValueError(
            f"Wyckoff position '{letter}' not found for point group {pg.name}. "
            f"Available positions: {list(positions_map.keys())}."
        )

    entry = positions_map[letter]
    if isinstance(entry, WyckoffPosition):
        return entry

    # Parameterized position
    arg = param
    if arg is None:
        if "x" in kwargs:
            arg = kwargs["x"]
        elif "p" in kwargs:
            arg = kwargs["p"]
        elif len(kwargs) == 1:
            arg = next(iter(kwargs.values()))

    if arg is None:
        raise ValueError(
            f"Wyckoff position '{letter}' in {pg.name} requires parameter value(s) "
            "(e.g., param=0.2 or x=0.2)."
        )

    return entry(arg)
