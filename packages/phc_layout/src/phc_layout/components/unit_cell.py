"""Photonic crystal unit cell component generators based on Wyckoff symmetry."""

from collections.abc import Sequence
from typing import Any, Literal

import gdsfactory as gf

from phc_layout.lattice import HexagonalLattice, Lattice, SquareLattice
from phc_layout.symmetry import get_point_group, get_wyckoff_position
from phc_layout.tech import LAYER_PHC


@gf.cell
def phc_wyckoff_unit_cell(
    pitch: float = 1.0,
    point_group: str = "C6v",
    features: Sequence[tuple[Any, ...] | dict[str, Any]] = (("1a", 0.25),),
    layer: gf.typings.LayerSpec = LAYER_PHC.ETCH,
    boundary_layer: gf.typings.LayerSpec | None = None,
    boundary_type: Literal["primitive", "wigner_seitz"] = "primitive",
    wrap_to_cell: bool = False,
) -> gf.Component:
    """Creates a photonic crystal unit cell with features placed at Wyckoff positions.

    Args:
        pitch: Lattice constant a in micrometers.
        point_group: Symmetry point group ('C6v' for hexagonal or 'C4v' for square).
        features: Sequence of feature specifications. Each feature can be:
            - tuple (letter, radius): e.g. ('1a', 0.25)
            - tuple (letter, radius, param): e.g. ('6d', 0.12, 0.3)
            - dict with keys 'position'/'letter', 'radius', optional 'param', and optional 'layer'.
        layer: Default GDS layer for etch holes.
        boundary_layer: Optional GDS layer to extrude unit cell boundary polygon (e.g. LAYER_PHC.SLAB).
        boundary_type: 'primitive' (parallelepiped/rhombus) or 'wigner_seitz' (Voronoi hexagon/square).
        wrap_to_cell: If True, wraps fractional coordinates into [-0.5, 0.5) prior to placement.

    Returns:
        gf.Component containing the placed holes and optional boundary polygon.

    Raises:
        ValueError: If pitch <= 0, unknown point group, or invalid feature specification.
    """
    if pitch <= 0:
        raise ValueError(f"Lattice pitch must be positive, got {pitch}.")

    pg = get_point_group(point_group)
    lat: Lattice
    if pg.name == "C6v":
        lat = HexagonalLattice(a=pitch)
    elif pg.name == "C4v":
        lat = SquareLattice(a=pitch)
    else:
        raise ValueError(f"Unsupported point group '{pg.name}'. Use 'C6v' or 'C4v'.")

    c = gf.Component()

    # Optional boundary polygon (e.g., slab)
    if boundary_layer is not None:
        if boundary_type == "wigner_seitz":
            pts_boundary = lat.wigner_seitz_polygon()
        else:
            pts_boundary = lat.unit_cell_polygon(centered=wrap_to_cell)
        c.add_polygon(points=pts_boundary, layer=boundary_layer)

    # Place features at Wyckoff positions
    for feat in features:
        letter: str
        radius: float
        param: Any = None
        feat_layer = layer

        if isinstance(feat, dict):
            letter = str(feat.get("position") or feat.get("letter"))
            radius = float(feat["radius"])
            param = feat.get("param")
            if "layer" in feat and feat["layer"] is not None:
                feat_layer = feat["layer"]
        elif isinstance(feat, (list, tuple)):
            if len(feat) == 2:
                letter = str(feat[0])
                radius = float(feat[1])
            elif len(feat) == 3:
                letter = str(feat[0])
                radius = float(feat[1])
                param = feat[2]
            else:
                raise ValueError(
                    f"Feature tuple must have length 2 (pos, r) or 3 (pos, r, param), got {feat}."
                )
        else:
            raise TypeError(f"Unsupported feature specification type: {type(feat)}")

        wyck = get_wyckoff_position(pg, letter, param=param)
        cart_positions = wyck.to_cartesian(lat, wrap_to_cell=wrap_to_cell)

        hole = gf.components.circle(radius=radius, layer=feat_layer)
        for x, y in cart_positions:
            ref = c << hole
            ref.move((x, y))

    return c


@gf.cell
def phc_hexagonal_unit_cell(
    pitch: float = 1.0,
    radius: float = 0.25,
    layer: gf.typings.LayerSpec = LAYER_PHC.ETCH,
) -> gf.Component:
    """Creates a primitive unit cell for a triangular/hexagonal PhC lattice.

    Contains a circular etch hole centered at Wyckoff position 1a (0, 0).

    Args:
        pitch: Lattice constant a in microns.
        radius: Hole radius in microns.
        layer: Target GDS layer for etch holes.

    Returns:
        gf.Component with hole centered at (0, 0).
    """
    return phc_wyckoff_unit_cell(
        pitch=pitch,
        point_group="C6v",
        features=(("1a", radius),),
        layer=layer,
    )


@gf.cell
def phc_square_unit_cell(
    pitch: float = 1.0,
    radius: float = 0.25,
    layer: gf.typings.LayerSpec = LAYER_PHC.ETCH,
) -> gf.Component:
    """Creates a primitive unit cell for a square PhC lattice.

    Contains a circular etch hole centered at Wyckoff position 1a (0, 0).

    Args:
        pitch: Lattice constant a in microns.
        radius: Hole radius in microns.
        layer: Target GDS layer for etch holes.

    Returns:
        gf.Component with hole centered at (0, 0).
    """
    return phc_wyckoff_unit_cell(
        pitch=pitch,
        point_group="C4v",
        features=(("1a", radius),),
        layer=layer,
    )


@gf.cell
def phc_snowflake_unit_cell(
    pitch: float = 1.0,
    radius: float = 0.40,
    width: float = 0.15,
    layer: gf.typings.LayerSpec = LAYER_PHC.ETCH,
    boundary_layer: gf.typings.LayerSpec | None = None,
    boundary_type: Literal["primitive", "wigner_seitz"] = "primitive",
    wrap_to_cell: bool = False,
) -> gf.Component:
    """Creates a snowflake optomechanical crystal unit cell on a triangular/hexagonal lattice.

    Based on the seminal optomechanical crystal design by Safavi-Naeini & Painter (2010),
    the unit cell features a 6-pointed star/snowflake slotted hole centered at the origin,
    formed by three intersecting rectangular arms rotated by 0°, 60°, and 120°. This geometry
    is widely used in quantum optomechanics and phononic-photonic crystal engineering to open
    simultaneous optical and acoustic bandgaps.

    Args:
        pitch: Lattice constant a in micrometers. Must be positive.
        radius: Radius / half-length of snowflake arms from the origin in micrometers.
            Typical values range from 0.35 * pitch to 0.45 * pitch.
        width: Width of the snowflake arms in micrometers.
            Typical values range from 0.10 * pitch to 0.20 * pitch.
        layer: Target GDS layer for the etch pattern.
        boundary_layer: Optional GDS layer to extrude unit cell boundary polygon (e.g. LAYER_PHC.SLAB).
        boundary_type: 'primitive' (rhombic unit cell) or 'wigner_seitz' (hexagonal Wigner-Seitz cell).
        wrap_to_cell: Kept for interface consistency; snowflake is centered at (0, 0).

    Returns:
        gf.Component containing the snowflake etch hole and optional boundary polygon.

    Raises:
        ValueError: If pitch <= 0, radius <= 0, or width <= 0.
    """
    if pitch <= 0:
        raise ValueError(f"Lattice pitch must be positive, got {pitch}.")
    if radius <= 0:
        raise ValueError(f"Snowflake radius must be positive, got {radius}.")
    if width <= 0:
        raise ValueError(f"Snowflake width must be positive, got {width}.")

    import shapely
    import shapely.affinity as sa
    import shapely.geometry as sg

    lat = HexagonalLattice(a=pitch)
    c = gf.Component()

    # Optional boundary polygon
    if boundary_layer is not None:
        if boundary_type == "wigner_seitz":
            pts_boundary = lat.wigner_seitz_polygon()
        else:
            pts_boundary = lat.unit_cell_polygon(centered=wrap_to_cell)
        c.add_polygon(points=pts_boundary, layer=boundary_layer)

    # Construct the snowflake polygon via boolean union of three rotated rectangular arms
    rect = sg.box(-radius, -width / 2.0, radius, width / 2.0)
    arms = [sa.rotate(rect, angle, origin=(0.0, 0.0)) for angle in (0.0, 60.0, 120.0)]
    snowflake_poly = shapely.union_all(arms)

    if snowflake_poly.geom_type == "Polygon":
        pts = list(snowflake_poly.exterior.coords)
        c.add_polygon(points=pts, layer=layer)
    elif snowflake_poly.geom_type == "MultiPolygon":
        for p in snowflake_poly.geoms:
            c.add_polygon(points=list(p.exterior.coords), layer=layer)

    return c
