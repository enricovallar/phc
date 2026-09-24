"""Curated database of canonical photonic crystal unit cells based on Wyckoff positions."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import gdsfactory as gf

from phc_layout.components.unit_cell import (
    phc_snowflake_unit_cell,
    phc_wyckoff_unit_cell,
)
from phc_layout.tech import LAYER_PHC


@dataclass(frozen=True)
class FeatureSpec:
    """Specification for a feature placed at a Wyckoff position.

    Attributes:
        position: Wyckoff letter (e.g. '1a', '2b', '6d').
        r_over_a: Hole radius normalized to lattice pitch (r / a).
        param: Numerical parameter for parameterized positions (e.g. x).
    """

    position: str
    r_over_a: float
    param: Any = None


@dataclass(frozen=True)
class UnitCellSpec:
    """Specification for a canonical unit cell geometry.

    Attributes:
        name: Unique identifier for the unit cell.
        point_group: Point group symmetry ('C6v' or 'C4v').
        lattice_type: Bravais lattice type ('hexagonal' or 'square').
        description: Physical description and relevant optical properties.
        features: Tuple of feature specifications.
        tags: Categorization tags.
    """

    name: str
    point_group: Literal["C6v", "C4v"]
    lattice_type: Literal["hexagonal", "square"]
    description: str
    features: tuple[FeatureSpec, ...] = field(default_factory=tuple)
    tags: tuple[str, ...] = field(default_factory=tuple)


# Registry of canonical unit cells
UNIT_CELL_DATABASE: dict[str, UnitCellSpec] = {
    # -------------------------------------------------------------
    # C6v (Hexagonal / Triangular) Family
    # -------------------------------------------------------------
    "c6v_primitive": UnitCellSpec(
        name="c6v_primitive",
        point_group="C6v",
        lattice_type="hexagonal",
        description="Standard hexagonal unit cell with a single hole at Wyckoff position 1a (origin).",
        features=(FeatureSpec(position="1a", r_over_a=0.25),),
        tags=("primitive", "c6v", "hexagonal"),
    ),
    "c6v_honeycomb": UnitCellSpec(
        name="c6v_honeycomb",
        point_group="C6v",
        lattice_type="hexagonal",
        description=(
            "Honeycomb / graphene-like bipartite triangular lattice with holes at Wyckoff "
            "position 2b, exhibiting Dirac cones at the Brillouin zone K points."
        ),
        features=(FeatureSpec(position="2b", r_over_a=0.18),),
        tags=("honeycomb", "graphene", "dirac", "c6v"),
    ),
    "c6v_kagome": UnitCellSpec(
        name="c6v_kagome",
        point_group="C6v",
        lattice_type="hexagonal",
        description=(
            "Kagome tri-partite lattice with holes at edge midpoints (Wyckoff position 3c), "
            "characterized by dispersionless flat bands."
        ),
        features=(FeatureSpec(position="3c", r_over_a=0.15),),
        tags=("kagome", "flatband", "c6v"),
    ),
    "c6v_ring_6d": UnitCellSpec(
        name="c6v_ring_6d",
        point_group="C6v",
        lattice_type="hexagonal",
        description=(
            "Hexagonal ring of 6 satellite holes oriented along the principal lattice axes "
            "at Wyckoff position 6d."
        ),
        features=(FeatureSpec(position="6d", r_over_a=0.12, param=0.30),),
        tags=("ring", "satellite", "c6v"),
    ),
    "c6v_ring_6e": UnitCellSpec(
        name="c6v_ring_6e",
        point_group="C6v",
        lattice_type="hexagonal",
        description=(
            "Hexagonal ring of 6 satellite holes oriented along the diagonal reflection axes "
            "(rotated 30 degrees relative to 6d) at Wyckoff position 6e."
        ),
        features=(FeatureSpec(position="6e", r_over_a=0.12, param=0.20),),
        tags=("ring", "satellite", "c6v"),
    ),
    "c6v_snowflake_6d": UnitCellSpec(
        name="c6v_snowflake_6d",
        point_group="C6v",
        lattice_type="hexagonal",
        description=(
            "Snowflake core-satellite unit cell with center hole at 1a and 6 satellite holes "
            "along the lattice axes at 6d, commonly used in valley-Hall topological photonics."
        ),
        features=(
            FeatureSpec(position="1a", r_over_a=0.20),
            FeatureSpec(position="6d", r_over_a=0.10, param=0.35),
        ),
        tags=("snowflake", "core_satellite", "valley_hall", "topological", "c6v"),
    ),
    "c6v_snowflake_6e": UnitCellSpec(
        name="c6v_snowflake_6e",
        point_group="C6v",
        lattice_type="hexagonal",
        description=(
            "Snowflake core-satellite unit cell with center hole at 1a and 6 satellite holes "
            "along diagonal axes at 6e."
        ),
        features=(
            FeatureSpec(position="1a", r_over_a=0.20),
            FeatureSpec(position="6e", r_over_a=0.10, param=0.22),
        ),
        tags=("snowflake", "core_satellite", "topological", "c6v"),
    ),
    "c6v_painter_snowflake": UnitCellSpec(
        name="c6v_painter_snowflake",
        point_group="C6v",
        lattice_type="hexagonal",
        description=(
            "Continuous snowflake optomechanical crystal (Safavi-Naeini & Painter, 2010) "
            "with a 6-pointed slotted hole formed by three intersecting rotated arms, "
            "supporting simultaneous optical and acoustic bandgaps."
        ),
        features=(FeatureSpec(position="1a", r_over_a=0.40, param=0.15),),
        tags=("snowflake", "optomechanics", "slotted", "painter", "c6v"),
    ),

    "c6v_2b_6d": UnitCellSpec(
        name="c6v_2b_6d",
        point_group="C6v",
        lattice_type="hexagonal",
        description=(
            "Aperture in the center"
        ),
        features=(
            FeatureSpec(position="2b", r_over_a=0.15, param=0.15),
            FeatureSpec(position="6d", r_over_a=0.1, param=0.25),
        ),
        tags=("Enrico", "c6v"),
    ),
    # -------------------------------------------------------------
    # C4v (Square) Family
    # -------------------------------------------------------------
    "c4v_primitive": UnitCellSpec(
        name="c4v_primitive",
        point_group="C4v",
        lattice_type="square",
        description="Standard square unit cell with a single hole at Wyckoff position 1a (origin).",
        features=(FeatureSpec(position="1a", r_over_a=0.25),),
        tags=("primitive", "c4v", "square"),
    ),
    "c4v_checkerboard": UnitCellSpec(
        name="c4v_checkerboard",
        point_group="C4v",
        lattice_type="square",
        description=(
            "Bipartite checkerboard square lattice with holes at Wyckoff positions 1a (origin) "
            "and 1b (cell center)."
        ),
        features=(
            FeatureSpec(position="1a", r_over_a=0.20),
            FeatureSpec(position="1b", r_over_a=0.20),
        ),
        tags=("checkerboard", "bipartite", "c4v"),
    ),
    "c4v_lieb": UnitCellSpec(
        name="c4v_lieb",
        point_group="C4v",
        lattice_type="square",
        description=(
            "Lieb lattice with holes at square edge midpoints (Wyckoff position 2c), "
            "featuring coexistence of a flat band and a Dirac cone."
        ),
        features=(FeatureSpec(position="2c", r_over_a=0.18),),
        tags=("lieb", "flatband", "dirac", "c4v"),
    ),
    "c4v_ring_4d": UnitCellSpec(
        name="c4v_ring_4d",
        point_group="C4v",
        lattice_type="square",
        description="Square ring of 4 holes located along diagonal reflection axes at Wyckoff position 4d.",
        features=(FeatureSpec(position="4d", r_over_a=0.12, param=0.25),),
        tags=("ring", "diagonal", "c4v"),
    ),
    "c4v_cross_4e": UnitCellSpec(
        name="c4v_cross_4e",
        point_group="C4v",
        lattice_type="square",
        description="Cross of 4 holes oriented along Cartesian axes at Wyckoff position 4e.",
        features=(FeatureSpec(position="4e", r_over_a=0.12, param=0.30),),
        tags=("cross", "c4v"),
    ),
    "c4v_edges_4f": UnitCellSpec(
        name="c4v_edges_4f",
        point_group="C4v",
        lattice_type="square",
        description="Unit cell with 4 holes located along square edges at Wyckoff position 4f.",
        features=(FeatureSpec(position="4f", r_over_a=0.12, param=0.25),),
        tags=("edges", "c4v"),
    ),
    "c4v_snowflake_4d": UnitCellSpec(
        name="c4v_snowflake_4d",
        point_group="C4v",
        lattice_type="square",
        description=(
            "Core-satellite square unit cell with central hole at 1a and 4 diagonal satellite "
            "holes at 4d."
        ),
        features=(
            FeatureSpec(position="1a", r_over_a=0.20),
            FeatureSpec(position="4d", r_over_a=0.10, param=0.30),
        ),
        tags=("snowflake", "core_satellite", "c4v"),
    ),
    "c4v_snowflake_4e": UnitCellSpec(
        name="c4v_snowflake_4e",
        point_group="C4v",
        lattice_type="square",
        description=(
            "Core-cross square unit cell with central hole at 1a and 4 Cartesian axis holes at 4e."
        ),
        features=(
            FeatureSpec(position="1a", r_over_a=0.20),
            FeatureSpec(position="4e", r_over_a=0.10, param=0.32),
        ),
        tags=("snowflake", "core_satellite", "cross", "c4v"),
    ),
}


def list_unit_cells(
    point_group: str | None = None,
    tag: str | None = None,
) -> list[str]:
    """Lists available unit cell names in the database, optionally filtered by point group or tag.

    Args:
        point_group: Optional point group filter ('C4v', 'C6v').
        tag: Optional category tag filter (e.g. 'honeycomb', 'flatband', 'snowflake').

    Returns:
        Sorted list of matching unit cell names.
    """
    results: list[str] = []
    for name, spec in UNIT_CELL_DATABASE.items():
        if (
            point_group is not None
            and spec.point_group.lower() != point_group.strip().lower()
        ):
            continue
        if tag is not None and tag.strip().lower() not in [
            t.lower() for t in spec.tags
        ]:
            continue
        results.append(name)
    return sorted(results)


def get_unit_cell_spec(name: str) -> UnitCellSpec:
    """Retrieves the specification for a unit cell by name.

    Args:
        name: Name of the unit cell in the database.

    Returns:
        UnitCellSpec instance.

    Raises:
        KeyError: If name is not found in UNIT_CELL_DATABASE.
    """
    if name not in UNIT_CELL_DATABASE:
        raise KeyError(
            f"Unit cell '{name}' not found. Available unit cells: {list_unit_cells()}."
        )
    return UNIT_CELL_DATABASE[name]


@gf.cell
def get_unit_cell(
    name: str = "c6v_primitive",
    pitch: float = 1.0,
    radius: float | None = None,
    layer: gf.typings.LayerSpec = LAYER_PHC.ETCH,
    boundary_layer: gf.typings.LayerSpec | None = None,
    boundary_type: Literal["primitive", "wigner_seitz"] = "primitive",
    wrap_to_cell: bool = False,
    override_features: Sequence[tuple[Any, ...]] | None = None,
) -> gf.Component:
    """Instantiates a canonical unit cell from the database.

    Args:
        name: Unit cell identifier in UNIT_CELL_DATABASE.
        pitch: Lattice constant a in micrometers.
        radius: Optional global override for hole radius in micrometers.
        layer: Target GDS layer for etch holes.
        boundary_layer: Optional GDS layer to extrude unit cell boundary.
        boundary_type: 'primitive' (rhombus/parallelepiped) or 'wigner_seitz' (hexagon/square).
        wrap_to_cell: If True, wraps fractional coordinates into [-0.5, 0.5).
        override_features: Optional sequence of custom feature specifications.

    Returns:
        gf.Component representing the unit cell.

    Raises:
        KeyError: If unit cell name is not found.
    """
    spec = get_unit_cell_spec(name)

    if name in ("c6v_painter_snowflake", "snowflake_painter"):
        r = radius if radius is not None else 0.40 * pitch
        w = 0.15 * pitch
        return phc_snowflake_unit_cell(
            pitch=pitch,
            radius=r,
            width=w,
            layer=layer,
            boundary_layer=boundary_layer,
            boundary_type=boundary_type,
            wrap_to_cell=wrap_to_cell,
        )

    if override_features is not None:
        features_to_build = override_features
    else:
        built_features: list[tuple[Any, ...]] = []
        for feat in spec.features:
            r = radius if radius is not None else (feat.r_over_a * pitch)
            if feat.param is not None:
                built_features.append((feat.position, r, feat.param))
            else:
                built_features.append((feat.position, r))
        features_to_build = tuple(built_features)

    return phc_wyckoff_unit_cell(
        pitch=pitch,
        point_group=spec.point_group,
        features=features_to_build,
        layer=layer,
        boundary_layer=boundary_layer,
        boundary_type=boundary_type,
        wrap_to_cell=wrap_to_cell,
    )
