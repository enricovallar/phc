# src/gds_designer/utils.py
from collections.abc import Sequence

import gdsfactory as gf
import klayout.db as kdb
import numpy as np
from shapely.geometry import MultiPoint, MultiPolygon, Point, Polygon
from shapely.ops import unary_union
from shapely.prepared import prep


@gf.cell
def get_component_hull(
    component: gf.typings.ComponentSpec,
    layer: gf.typings.LayerSpec = (1, 0),
    margin: float = 0.0,
    layers_to_include: Sequence[gf.typings.LayerSpec] | None = None,
) -> gf.Component:
    """Computes the 2D convex hull of a component's polygons, with optional buffer margin.

    Args:
        component: Component to compute the convex hull for.
        layer: Layer to place the hull polygon on.
        margin: Distance to buffer/expand the hull.
        layers_to_include: Specific layers to include when extracting polygons.

    Returns:
        Component containing the hull polygon.
    """
    c_source = gf.get_component(component)
    poly_dict = c_source.get_polygons_points()

    points: list[tuple[float, float]] = []
    for lyr, polys in poly_dict.items():
        if layers_to_include is None or lyr in layers_to_include:
            for p in polys:
                points.extend(p)

    if not points:
        raise ValueError(
            f"Component '{c_source.name}' contains no polygons on the selected layers."
        )

    hull_geom = MultiPoint(points).convex_hull

    if margin > 0:
        hull_geom = hull_geom.buffer(margin)

    if not isinstance(hull_geom, Polygon):
        raise TypeError(
            f"Expected a 2D Polygon hull, but got {type(hull_geom).__name__}."
        )

    # Convert coordinates to klayout DPoints (floating-point microns)
    dpoints = [kdb.DPoint(x, y) for x, y in hull_geom.exterior.coords[:-1]]
    kpoly = kdb.DPolygon(dpoints)

    c = gf.Component(f"{c_source.name}_hull")
    c.add_polygon(kpoly, layer=layer)

    return c


@gf.cell
def deduplicate_instances(
    component: gf.typings.ComponentSpec,
    precision: int = 4,
) -> gf.Component:
    """Flattens composite cell references down to primitive instances (e.g. circles)
    and removes duplicate instances sharing the same cell and location.

    Args:
        component: The component or component spec containing composite references.
        precision: Decimal places for spatial grid rounding (4 = 0.1 nm precision).

    Returns:
        A new Component containing only unique primitive instances.
    """
    c_source = gf.get_component(component)
    clean = gf.Component()
    seen: set[tuple[str, tuple[float, float]]] = set()

    for inst in c_source.insts:
        # If the instance contains nested children (like basis -> circles), unpack them;
        # otherwise, treat the instance itself as the primitive.
        child_insts = inst.cell.insts if inst.cell.insts else [inst]

        for child in child_insts:
            # Combine parent transformation (patch lattice) and child transformation (basis offset)
            combined_trans = inst.dtrans * child.dtrans

            # Snap position to grid to avoid floating-point residuals from trigonometry
            pos = (
                round(float(combined_trans.disp.x), precision),
                round(float(combined_trans.disp.y), precision),
            )

            key = (child.cell.name, pos)
            if key not in seen:
                seen.add(key)
                ref = clean.add_ref(child.cell)
                ref.dtrans = combined_trans

    return clean


@gf.cell
def carve_cavity_hierarchical(
    component: gf.typings.ComponentSpec,
    cavity: gf.typings.ComponentSpec | Polygon | MultiPolygon,
    margin: float = 0.0,
) -> gf.Component:
    """Carves a cavity out of an array of instances without flattening the geometry.

    Filters out instances whose centers fall inside the cavity boundary.
    Preserves hierarchical cell referencing and avoids converting instances into raw polygons.

    Args:
        component: The component containing instances to be carved (e.g., mirror patch).
        cavity: A Component (such as a hull or patch) or a Shapely Polygon defining the cavity.
        margin: Optional additional margin to buffer the cavity boundary.

    Returns:
        A new Component containing only the instances outside the cavity.
    """
    c_source = gf.get_component(component)

    if isinstance(cavity, (Polygon, MultiPolygon)):
        cavity_poly = cavity
    else:
        c_cavity = gf.get_component(cavity)
        poly_dict = c_cavity.get_polygons_points()
        points: list[tuple[float, float]] = []
        for lyr_polys in poly_dict.values():
            for p in lyr_polys:
                points.extend(p)
        if not points:
            raise ValueError(
                f"Cavity component '{c_cavity.name}' contains no polygons."
            )
        cavity_poly = MultiPoint(points).convex_hull

    if margin > 0:
        cavity_poly = cavity_poly.buffer(margin)

    prep_cavity = prep(cavity_poly)
    clean = gf.Component()

    for inst in c_source.insts:
        # If the instance contains nested children, check each primitive child
        if inst.cell.insts:
            for child in inst.cell.insts:
                combined_trans = inst.dtrans * child.dtrans
                pos = Point(combined_trans.disp.x, combined_trans.disp.y)
                if not prep_cavity.contains(pos):
                    ref = clean.add_ref(child.cell)
                    ref.dtrans = combined_trans
        else:
            pos = Point(inst.dtrans.disp.x, inst.dtrans.disp.y)
            if not prep_cavity.contains(pos):
                ref = clean.add_ref(inst.cell)
                ref.dtrans = inst.dtrans

    return clean


@gf.cell
def carve_mirror_waveguides(
    mirror_patch: gf.typings.ComponentSpec,
    cavity_hull: gf.typings.ComponentSpec | Polygon | MultiPolygon,
    open_ports: Sequence[int] = (1, 2, 3),
    waveguide_width: float = 1.0,
    angle_offset: float = 0.0,
    reach: float | None = None,
    add_ports: bool = True,
    port_layer: gf.typings.LayerSpec = (1, 0),
) -> gf.Component:
    """Carves a cavity and selective waveguide corridors out of a mirror patch hierarchically.

    Args:
        mirror_patch: The component containing mirror instances (e.g. circle array).
        cavity_hull: A Component or Shapely Polygon defining the central cavity.
        open_ports: Sequence of 1-based port indices (1 to 6) to open as waveguides.
        waveguide_width: Width of the carved waveguide corridor.
        angle_offset: Rotation offset in degrees (0.0 for vertex exits, 30.0 for facet exits).
        reach: Outward length of the corridor. If None, automatically extends beyond the mirror bbox.
        add_ports: If True, adds gdsfactory optical Ports at the outer boundary exits.
        port_layer: Layer on which to attach ports.

    Returns:
        Component: The carved mirror with open waveguides, preserving cell referencing.
    """
    c_mirror = gf.get_component(mirror_patch)

    # 1. Resolve cavity polygon
    if isinstance(cavity_hull, (Polygon, MultiPolygon)):
        cavity_poly = cavity_hull
    else:
        c_cavity = gf.get_component(cavity_hull)
        poly_dict = c_cavity.get_polygons_points()
        points: list[tuple[float, float]] = []
        for lyr_polys in poly_dict.values():
            for p in lyr_polys:
                points.extend(p)
        if not points:
            raise ValueError(
                f"Cavity component '{c_cavity.name}' contains no polygons."
            )
        cavity_poly = MultiPoint(points).convex_hull

    # 2. Determine boundary reach
    bbox = c_mirror.bbox()
    max_bound = max(abs(bbox.left), abs(bbox.right), abs(bbox.bottom), abs(bbox.top))
    corridor_reach = reach if reach is not None else (max_bound + 5.0)

    # 3. Build waveguide corridors
    corridors: list[Polygon] = []
    port_exit_points: list[tuple[int, tuple[float, float], float]] = []
    half_w = waveguide_width / 2.0

    for port_idx in open_ports:
        if port_idx < 1 or port_idx > 6:
            raise ValueError(
                f"Port index {port_idx} is invalid. Port indices must be between 1 and 6."
            )

        theta_deg = angle_offset + (port_idx - 1) * 60.0
        rad = np.deg2rad(theta_deg)
        ux, uy = float(np.cos(rad)), float(np.sin(rad))
        vx, vy = float(-np.sin(rad)), float(np.cos(rad))

        # Rectangular corridor polygon extending from center out to reach
        p1 = (-half_w * vx, -half_w * vy)
        p2 = (half_w * vx, half_w * vy)
        p3 = (corridor_reach * ux + half_w * vx, corridor_reach * uy + half_w * vy)
        p4 = (corridor_reach * ux - half_w * vx, corridor_reach * uy - half_w * vy)
        corridors.append(Polygon([p1, p2, p3, p4]))

        # Calculate exit point on the mirror bounding box
        candidates: list[float] = []
        if abs(ux) > 1e-6:
            tx = (bbox.right if ux > 0 else bbox.left) / ux
            if tx > 0:
                candidates.append(tx)
        if abs(uy) > 1e-6:
            ty = (bbox.top if uy > 0 else bbox.bottom) / uy
            if ty > 0:
                candidates.append(ty)
        t_exit = min(candidates) if candidates else corridor_reach
        port_exit_points.append(
            (port_idx, (float(t_exit * ux), float(t_exit * uy)), float(theta_deg))
        )

    # 4. Merge cavity and active corridors
    shapes_to_carve = [cavity_poly] + corridors
    carve_region = unary_union(shapes_to_carve)
    prep_region = prep(carve_region)

    # 5. Filter instances hierarchically
    clean = gf.Component()
    for inst in c_mirror.insts:
        if inst.cell.insts:
            for child in inst.cell.insts:
                combined_trans = inst.dtrans * child.dtrans
                pos = Point(combined_trans.disp.x, combined_trans.disp.y)
                if not prep_region.contains(pos):
                    ref = clean.add_ref(child.cell)
                    ref.dtrans = combined_trans
        else:
            pos = Point(inst.dtrans.disp.x, inst.dtrans.disp.y)
            if not prep_region.contains(pos):
                ref = clean.add_ref(inst.cell)
                ref.dtrans = inst.dtrans

    # 6. Add ports at waveguide exits if requested
    if add_ports:
        for port_idx, exit_pos, orientation in port_exit_points:
            clean.add_port(
                name=f"o{port_idx}",
                center=exit_pos,
                width=waveguide_width,
                orientation=orientation,
                layer=port_layer,
            )

    return clean
