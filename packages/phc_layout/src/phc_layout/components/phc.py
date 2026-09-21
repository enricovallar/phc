# src/phc_layout/components/phc.py

import gdsfactory as gf
import numpy as np


@gf.cell
def lattice_patch_c6v_rect(
    component: gf.typings.ComponentSpec = "circle",
    columns: int = 15,
    rows: int = 21,
    pitch: float = 0.45,
    centered: bool = True,
    add_ports: bool = True,
    n_cols: int | None = None,
    n_rows: int | None = None,
) -> gf.Component:
    """Creates a rectangular bounding patch of a triangular/hexagonal lattice."""
    columns = n_cols if n_cols is not None else columns
    rows = n_rows if n_rows is not None else rows

    c = gf.Component()
    comp = gf.get_component(component)

    dx = pitch
    dy = pitch * np.sqrt(3) / 2

    x_offset = (columns - 1) * dx / 2.0 if centered else 0.0
    y_offset = (rows - 1) * dy / 2.0 if centered else 0.0

    for r in range(rows):
        row_shift = (dx / 2.0) if (r % 2 != 0) else 0.0
        for col in range(columns):
            x = col * dx + row_shift - x_offset
            y = r * dy - y_offset
            ref = c << comp
            ref.move((x, y))

    return c


lattice_patch_phc_C6v_rect = lattice_patch_c6v_rect


@gf.cell
def array_hexagonal_patch(
    component: gf.typings.ComponentSpec = "circle",
    rings: int = 5,
    pitch: float = 1.0,
    centered: bool = True,
    add_ports: bool = False,
) -> gf.Component:
    """Creates a concentric hexagonal patch of a triangular lattice."""
    c = gf.Component()
    comp = gf.get_component(component)

    # Central element
    c << comp

    for ring in range(1, rings + 1):
        for sector in range(6):
            angle1 = sector * np.pi / 3
            angle2 = (sector + 1) * np.pi / 3
            corner1 = np.array([np.cos(angle1), np.sin(angle1)]) * (ring * pitch)
            corner2 = np.array([np.cos(angle2), np.sin(angle2)]) * (ring * pitch)

            for step in range(ring):
                pos = corner1 + (corner2 - corner1) * (step / ring)
                ref = c << comp
                ref.move(pos)

    return c


@gf.cell
def lattice_patch_c6v_hex(
    component: gf.typings.ComponentSpec = "circle",
    rings: int = 5,
    pitch: float = 1.0,
    centered: bool = True,
    add_ports: bool = False,
) -> gf.Component:
    return array_hexagonal_patch(
        component=component,
        rings=rings,
        pitch=pitch,
        centered=centered,
        add_ports=add_ports,
    )
