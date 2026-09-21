# src/phc_layout/components/basis.py
import gdsfactory as gf
import numpy as np

from phc_layout.tech import LAYER_PHC


@gf.cell
def triangular_basis(
    lattice_constant: float = 1.0,
    r_center: float = 0.2,
    r_edge: float = 0.1,
    layer: gf.typings.LayerSpec = LAYER_PHC.ETCH,
) -> gf.Component:
    """Creates a hexagonal unit cell / basis with a central hole and 6 satellite holes.

    Args:
        lattice_constant: Lattice constant a in microns.
        r_center: Radius of the central hole (normalized to a).
        r_edge: Radius of the edge holes (normalized to a).
        layer: Layer to place the holes on.

    Returns:
        Component representing the basis.
    """
    basis = gf.Component("triangular_basis")
    center_circle = gf.components.circle(
        radius=r_center * lattice_constant, layer=layer
    )
    edge_circle = gf.components.circle(radius=r_edge * lattice_constant, layer=layer)

    d = lattice_constant / 2.0
    angles_deg = [0, 60, 120, 180, 240, 300]
    edge_vectors = [
        (d * np.cos(np.deg2rad(phi)), d * np.sin(np.deg2rad(phi))) for phi in angles_deg
    ]

    basis << center_circle
    for vec in edge_vectors:
        ref = basis << edge_circle
        ref.move(vec)

    return basis
