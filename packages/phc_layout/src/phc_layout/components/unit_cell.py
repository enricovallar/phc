import gdsfactory as gf

from phc_layout.tech import LAYER_PHC


@gf.cell
def phc_hexagonal_unit_cell(
    pitch: float = 1.0,
    radius: float = 0.25,
    layer: gf.typings.LayerSpec = LAYER_PHC.ETCH,
) -> gf.Component:
    """Creates a primitive unit cell for a triangular/hexagonal PhC lattice.

    Contains a circular etch hole centered at (0, 0).

    Args:
        pitch: Lattice constant a in microns.
        radius: Hole radius in microns (or as a fraction of pitch).
        layer: Target GDS layer.

    Returns:
        gf.Component with hole centered at (0, 0).
    """
    c = gf.Component()
    hole = gf.components.circle(radius=radius, layer=layer)
    c << hole
    return c


@gf.cell
def phc_square_unit_cell(
    pitch: float = 1.0,
    radius: float = 0.25,
    layer: gf.typings.LayerSpec = LAYER_PHC.ETCH,
) -> gf.Component:
    """Creates a primitive unit cell for a square PhC lattice.

    Contains a circular etch hole centered at (0, 0).

    Args:
        pitch: Lattice constant a in microns.
        radius: Hole radius in microns.
        layer: Target GDS layer.

    Returns:
        gf.Component with hole centered at (0, 0).
    """
    c = gf.Component()
    hole = gf.components.circle(radius=radius, layer=layer)
    c << hole
    return c
