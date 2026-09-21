"""phc_layout: Photonic crystal layout generation tools based on gdsfactory."""

import gdsfactory as gf

from phc_layout.tech import LAYER_PHC, LayerMapPhC

gf.gpdk.PDK.activate()

from phc_layout import components, lattice, symmetry, tech, utils
from phc_layout.lattice import (
    HexagonalLattice,
    Lattice,
    SquareLattice,
    get_lattice,
)
from phc_layout.symmetry import (
    WYCKOFF_POSITIONS_2D,
    C4v,
    C6v,
    PointGroup,
    WyckoffPosition,
    get_point_group,
    get_wyckoff_position,
)

__all__ = [
    "LAYER_PHC",
    "WYCKOFF_POSITIONS_2D",
    "C4v",
    "C6v",
    "HexagonalLattice",
    "Lattice",
    "LayerMapPhC",
    "PointGroup",
    "SquareLattice",
    "WyckoffPosition",
    "components",
    "get_lattice",
    "get_point_group",
    "get_wyckoff_position",
    "lattice",
    "symmetry",
    "tech",
    "utils",
]
