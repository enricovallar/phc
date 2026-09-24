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
    slab_material: str | float | Any | None = None,
    substrate_material: str | float | Any | None = None,
    substrate_thickness: float | None = None,
    z_center: float = 0.0,
    etch_layer: tuple[int, int] = (1, 0),
    etch_material: str | float | Any = "air",
    geometry_lattice: Any = None,
) -> list[Any]:
    """Converts GDS polygons on the etch layer into a list of MPB Prisms or Cylinders.

    Coordinates extracted from the GDS layout are physical Cartesian coordinates. If a
    `geometry_lattice` is provided, coordinates are transformed into the lattice vector
    basis required by MPB geometric objects via `mp.cartesian_to_lattice`.

    Args:
        gds_source: Path to .gds file or a gdsfactory.Component.
        pitch: Lattice constant a in microns (used for normalization).
        dimension: "2D" (infinite along z) or "3D_slab" (finite thickness).
        slab_thickness: Thickness of slab in microns (used when dimension='3D_slab').
        slab_material: Optional material key or numeric index for the dielectric slab core (e.g. "si" or 3.48).
            If specified and dimension='3D_slab', an mp.Block covering the unit cell
            with height slab_thickness / pitch is prepended before the etch holes.
        substrate_material: Optional material key or numeric index for the bottom substrate cladding (e.g. "sio2" or 1.44).
            If specified and dimension='3D_slab', an mp.Block is placed below the slab.
        substrate_thickness: Optional thickness of the substrate in microns. If None, fills
            the lower half of the computational cell down to the supercell lower boundary.
        z_center: Vertical center of slab in microns.
        etch_layer: (layer, datatype) of the holes.
        etch_material: Material key for the holes (defaults to "air").
        geometry_lattice: Optional mp.Lattice or phc_layout.lattice.Lattice instance
            defining the simulation basis vectors. If provided, coordinates are transformed
            from Cartesian to the lattice vector basis.

    Returns:
        List of mp.GeometricObject instances (Block, Cylinder, or Prism).
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

    # Normalize geometry_lattice if phc_layout.lattice.Lattice was passed
    geom_lat = None
    if geometry_lattice is not None:
        if not hasattr(geometry_lattice, "basis1"):
            from phc_mpb.lattice import lattice_to_mpb_lattice

            geom_lat = lattice_to_mpb_lattice(
                geometry_lattice, dimension=dimension, normalize=True
            )
        else:
            geom_lat = geometry_lattice

    # 2. Compute dimensionless heights and base z-positions
    if dimension == "2D":
        prism_height = mp.inf
        z_base = 0.0
    else:
        prism_height = slab_thickness / pitch
        z_base = (z_center - 0.5 * slab_thickness) / pitch

    medium = to_mpb_medium(etch_material)

    # 3. Create MPB geometric objects with scaled dimensions (a = 1)
    objects = []

    # Substrate block below slab
    if dimension == "3D_slab" and substrate_material is not None:
        sub_med = to_mpb_medium(substrate_material)
        z_slab_bottom = (z_center - 0.5 * slab_thickness) / pitch
        if substrate_thickness is not None:
            sub_h = substrate_thickness / pitch
            sub_cz = z_slab_bottom - 0.5 * sub_h
        elif geom_lat is not None and hasattr(geom_lat, "size") and geom_lat.size.z > 0:
            sz = float(geom_lat.size.z)
            sub_bottom = -0.5 * sz
            sub_h = max(0.0, z_slab_bottom - sub_bottom)
            sub_cz = sub_bottom + 0.5 * sub_h
        else:
            sub_h = 10.0
            sub_cz = z_slab_bottom - 0.5 * sub_h

        if sub_h > 0:
            objects.append(
                mp.Block(
                    center=mp.Vector3(0.0, 0.0, sub_cz),
                    size=mp.Vector3(mp.inf, mp.inf, sub_h),
                    material=sub_med,
                )
            )

    if dimension == "3D_slab" and slab_material is not None:
        slab_med = to_mpb_medium(slab_material)
        objects.append(
            mp.Block(
                center=mp.Vector3(0.0, 0.0, z_center / pitch),
                size=mp.Vector3(mp.inf, mp.inf, prism_height),
                material=slab_med,
            )
        )

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
            if geom_lat is not None:
                cyl_center = mp.cartesian_to_lattice(cyl_center, geom_lat)

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
            if geom_lat is not None:
                v_list = [
                    mp.cartesian_to_lattice(mp.Vector3(x, y, z_base), geom_lat)
                    for x, y in scaled_pts
                ]
            else:
                v_list = [mp.Vector3(x, y, z_base) for x, y in scaled_pts]

            # Note: In mp.Prism, omitting center keeps vertices unshifted at (x, y, z_base).
            # Passing center causes Meep to translate vertices by (center - centroid - 0.5*height*axis),
            # which for height=1e+20 causes catastrophic floating-point cancellation at -5e+19.
            prism = mp.Prism(
                vertices=v_list,
                height=prism_height,
                material=medium,
            )
            objects.append(prism)

    return objects
