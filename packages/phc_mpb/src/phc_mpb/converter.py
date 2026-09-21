from pathlib import Path
from typing import Any, Literal

import klayout.db as kdb
import numpy as np
from phc_materials import to_mpb_medium


def extract_polygons_from_gds(
    gds_path: str | Path,
    layer: tuple[int, int] = (1, 0),
) -> list[list[tuple[float, float]]]:
    """Reads a GDS file using klayout.db and extracts all polygons on the specified layer."""
    gds_path = Path(gds_path)
    if not gds_path.is_file():
        raise FileNotFoundError(f"GDS file not found: {gds_path}")

    ly = kdb.Layout()
    ly.read(str(gds_path))
    top = ly.top_cell()
    if not top:
        return []

    target_layer, target_datatype = layer
    layer_idx = ly.find_layer(target_layer, target_datatype)
    if layer_idx is None:
        return []

    polygons: list[list[tuple[float, float]]] = []
    si = top.begin_shapes_rec(layer_idx)
    while not si.at_end():
        shape = si.shape()
        if shape.is_polygon():
            trans = si.trans()
            poly = shape.polygon.transformed(trans)
            pts = [(pt.x * ly.dbu, pt.y * ly.dbu) for pt in poly.each_point_hull()]
            polygons.append(pts)
        si.next()

    return polygons


def _polygon_to_points(poly: Any, dbu: float = 0.001) -> list[tuple[float, float]]:
    """Converts various polygon objects (gdstk, klayout, shapely, list) into a list of (x, y) float tuples."""
    if hasattr(poly, "points"):
        return [(float(pt[0]), float(pt[1])) for pt in poly.points]
    elif hasattr(poly, "each_point_hull"):
        dp = poly.to_dtype(dbu) if hasattr(poly, "to_dtype") else poly
        return [(float(pt.x), float(pt.y)) for pt in dp.each_point_hull()]
    elif hasattr(poly, "exterior"):
        return [(float(x), float(y)) for x, y in poly.exterior.coords]
    elif isinstance(poly, (list, tuple, np.ndarray)):
        return [(float(pt[0]), float(pt[1])) for pt in poly]
    raise TypeError(f"Cannot extract points from polygon object: {type(poly)}")


def gds_to_mpb_geometry(
    gds_source: Any,
    pitch: float = 1.0,
    dimension: Literal["2D", "3D_slab"] = "2D",
    slab_thickness: float = 0.22,
    z_center: float = 0.0,
    etch_layer: tuple[int, int] = (1, 0),
    etch_material: str = "air",
) -> list[Any]:
    """Converts GDS polygons on the etch layer into a list of MPB Prisms.

    Args:
        gds_source: Path to .gds file or a gdsfactory.Component.
        pitch: Lattice constant a in microns (used for normalization).
        dimension: "2D" (infinite along z) or "3D_slab" (finite thickness).
        slab_thickness: Thickness of slab in microns (used when dimension='3D_slab').
        z_center: Vertical center of slab in microns.
        etch_layer: (layer, datatype) of the holes.
        etch_material: Material key for the holes (defaults to "air").

    Returns:
        List of mp.Prism objects (or dict representations if meep is not installed).
    """
    # 1. Extract polygon objects in microns
    dbu = 0.001
    if hasattr(gds_source, "get_polygons"):
        if hasattr(gds_source, "kcl") and hasattr(gds_source.kcl, "dbu"):
            dbu = gds_source.kcl.dbu
        polygons = gds_source.get_polygons()
        if isinstance(polygons, dict):
            polys = polygons.get(etch_layer, [])
            if not polys and polygons:
                polys = next(iter(polygons.values()))
        else:
            polys = polygons
    else:
        polys = extract_polygons_from_gds(gds_source, layer=etch_layer)

    import meep as mp

    # 2. Compute dimensionless heights and centers
    if dimension == "2D":
        prism_height = mp.inf
        prism_center = mp.Vector3(0, 0, 0)
    else:
        prism_height = slab_thickness / pitch
        prism_center = mp.Vector3(0, 0, z_center / pitch)

    medium = to_mpb_medium(etch_material)

    # 3. Create MPB geometric objects with scaled dimensions (a = 1)
    objects = []
    for poly in polys:
        pts = _polygon_to_points(poly, dbu=dbu)
        if len(pts) < 3:
            continue

        pts_arr = np.array(pts)
        centroid = np.mean(pts_arr, axis=0)
        radii = np.sqrt(np.sum((pts_arr - centroid) ** 2, axis=1))
        r_mean = float(np.mean(radii))
        r_std = float(np.std(radii))

        is_circle = len(pts) >= 8 and r_mean > 0 and (r_std / r_mean) < 0.05

        if is_circle:
            # Native MPB Cylinder: robust, faster, and preserves exact circular symmetry
            cyl_center = mp.Vector3(
                centroid[0] / pitch,
                centroid[1] / pitch,
                0.0 if dimension == "2D" else z_center / pitch,
            )
            cyl = mp.Cylinder(
                radius=r_mean / pitch,
                height=prism_height,
                center=cyl_center,
                material=medium,
            )
            objects.append(cyl)
        else:
            # Arbitrary polygon: clean consecutive duplicate points
            unique_pts = []
            for pt in pts:
                if not unique_pts or (
                    abs(pt[0] - unique_pts[-1][0]) > 1e-7
                    or abs(pt[1] - unique_pts[-1][1]) > 1e-7
                ):
                    unique_pts.append(pt)
            if (
                len(unique_pts) > 2
                and abs(unique_pts[0][0] - unique_pts[-1][0]) < 1e-7
                and abs(unique_pts[0][1] - unique_pts[-1][1]) < 1e-7
            ):
                unique_pts.pop()

            scaled_pts = [(x / pitch, y / pitch) for x, y in unique_pts]
            v_list = [mp.Vector3(x, y, 0) for x, y in scaled_pts]
            prism = mp.Prism(
                vertices=v_list,
                height=prism_height,
                center=prism_center,
                material=medium,
            )
            objects.append(prism)

    return objects
