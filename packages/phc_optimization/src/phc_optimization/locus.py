"""Parametric Degeneracy Locus & Manifold Extraction for 2D Optimization Landscapes.

Extracts continuous 1D optimal connected manifolds (such as accidental Dirac cone curves
or zero-gap degeneracy lines) from 2D surrogate Figure of Merit (FOM) landscapes.
Works across any two arbitrary parameter names (e.g. r1/r2, pitch/radius, w1/w2).

Algorithms:
1. Percentile thresholding and morphological closing (scipy.ndimage).
2. Connected component segmentation and ranking (scipy.ndimage.label).
3. Topological skeletonization (Zhang-Suen thinning / skimage.morphology).
4. Endpoint-to-endpoint graph trajectory ordering.
5. Parametric B-spline curve smoothing (scipy.interpolate.splprep / splev).
"""

import csv
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import scipy.ndimage as ndi
from scipy.interpolate import splev, splprep


def zhang_suen_thinning(image: np.ndarray) -> np.ndarray:
    """Performs topological skeletonization using the Zhang-Suen morphological thinning algorithm.

    Preserves 8-connectivity and endpoints to extract a 1-pixel-wide skeleton from a binary mask.

    Args:
        image: 2D binary numpy array (boolean or integer 0/1).

    Returns:
        2D boolean array representing the 1-pixel-wide morphological skeleton.
    """
    img = image.copy().astype(np.uint8)
    prev = np.zeros_like(img)

    while True:
        # Step 1
        p2 = np.roll(img, -1, axis=0)
        p3 = np.roll(np.roll(img, -1, axis=0), 1, axis=1)
        p4 = np.roll(img, 1, axis=1)
        p5 = np.roll(np.roll(img, 1, axis=0), 1, axis=1)
        p6 = np.roll(img, 1, axis=0)
        p7 = np.roll(np.roll(img, 1, axis=0), -1, axis=1)
        p8 = np.roll(img, -1, axis=1)
        p9 = np.roll(np.roll(img, -1, axis=0), -1, axis=1)

        neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
        transitions = (
            ((p2 == 0) & (p3 == 1)).astype(int)
            + ((p3 == 0) & (p4 == 1)).astype(int)
            + ((p4 == 0) & (p5 == 1)).astype(int)
            + ((p5 == 0) & (p6 == 1)).astype(int)
            + ((p6 == 0) & (p7 == 1)).astype(int)
            + ((p7 == 0) & (p8 == 1)).astype(int)
            + ((p8 == 0) & (p9 == 1)).astype(int)
            + ((p9 == 0) & (p2 == 1)).astype(int)
        )

        m1 = (
            (img == 1)
            & (neighbors >= 2)
            & (neighbors <= 6)
            & (transitions == 1)
            & (p2 * p4 * p6 == 0)
            & (p4 * p6 * p8 == 0)
        )
        img[m1] = 0

        # Step 2
        p2 = np.roll(img, -1, axis=0)
        p3 = np.roll(np.roll(img, -1, axis=0), 1, axis=1)
        p4 = np.roll(img, 1, axis=1)
        p5 = np.roll(np.roll(img, 1, axis=0), 1, axis=1)
        p6 = np.roll(img, 1, axis=0)
        p7 = np.roll(np.roll(img, 1, axis=0), -1, axis=1)
        p8 = np.roll(img, -1, axis=1)
        p9 = np.roll(np.roll(img, -1, axis=0), -1, axis=1)

        neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
        transitions = (
            ((p2 == 0) & (p3 == 1)).astype(int)
            + ((p3 == 0) & (p4 == 1)).astype(int)
            + ((p4 == 0) & (p5 == 1)).astype(int)
            + ((p5 == 0) & (p6 == 1)).astype(int)
            + ((p6 == 0) & (p7 == 1)).astype(int)
            + ((p7 == 0) & (p8 == 1)).astype(int)
            + ((p8 == 0) & (p9 == 1)).astype(int)
            + ((p9 == 0) & (p2 == 1)).astype(int)
        )

        m2 = (
            (img == 1)
            & (neighbors >= 2)
            & (neighbors <= 6)
            & (transitions == 1)
            & (p2 * p4 * p8 == 0)
            & (p2 * p6 * p8 == 0)
        )
        img[m2] = 0

        if np.array_equal(img, prev):
            break
        prev = img.copy()

    return img.astype(bool)


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """Thins a binary 2D mask to a 1-pixel-wide skeleton.

    Args:
        mask: 2D binary numpy mask.

    Returns:
        Skeletonized boolean mask.
    """
    try:
        from skimage.morphology import skeletonize

        return skeletonize(mask)
    except (ImportError, ModuleNotFoundError):
        return zhang_suen_thinning(mask)


def order_skeleton_points(
    pts_x: np.ndarray, pts_y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Orders 2D skeleton pixel coordinates into a continuous sequential trajectory.

    Handles both open branches (endpoint-to-endpoint) and closed circular loops.

    Args:
        pts_x: 1D array of x-coordinates.
        pts_y: 1D array of y-coordinates.

    Returns:
        Tuple of (ordered_x, ordered_y) arrays.
    """
    if len(pts_x) <= 2:
        return pts_x, pts_y

    n = len(pts_x)

    # Estimate 8-connectivity distance threshold
    diffs_x = np.abs(np.subtract.outer(pts_x, pts_x))
    diffs_y = np.abs(np.subtract.outer(pts_y, pts_y))
    nonzero_x = diffs_x[diffs_x > 1e-9]
    nonzero_y = diffs_y[diffs_y > 1e-9]
    dx = float(np.min(nonzero_x)) if nonzero_x.size > 0 else 1.0
    dy = float(np.min(nonzero_y)) if nonzero_y.size > 0 else 1.0
    diag_step = 1.5 * np.hypot(dx, dy)

    # Build adjacency list
    adj: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(pts_x[i] - pts_x[j], pts_y[i] - pts_y[j])
            if d <= diag_step:
                adj[i].append(j)
                adj[j].append(i)

    # Identify degree-1 endpoints (for open branches)
    deg1 = [i for i, nbrs in enumerate(adj) if len(nbrs) == 1]
    start = deg1[0] if deg1 else 0

    visited = {start}
    path = [start]
    curr = start
    while len(visited) < n:
        unvisited_nbrs = [nbr for nbr in adj[curr] if nbr not in visited]
        if unvisited_nbrs:
            next_node = min(
                unvisited_nbrs,
                key=lambda j: np.hypot(pts_x[curr] - pts_x[j], pts_y[curr] - pts_y[j]),
            )
        else:
            unvisited_all = [i for i in range(n) if i not in visited]
            next_node = min(
                unvisited_all,
                key=lambda j: np.hypot(pts_x[curr] - pts_x[j], pts_y[curr] - pts_y[j]),
            )
        visited.add(next_node)
        path.append(next_node)
        curr = next_node

    if not deg1:
        path.append(start)

    return pts_x[path], pts_y[path]


def compute_curve_normals(
    x1: np.ndarray | list[float],
    x2: np.ndarray | list[float],
) -> np.ndarray:
    """Computes 2D unit normal vectors along a planar parameter trajectory.

    For a 2D curve parameterized by arc length or point index, computes the tangent
    vector via central differences, and rotates it by 90 degrees counter-clockwise
    to obtain the perpendicular unit normal: n = (-t_y, t_x) / ||t||.

    Args:
        x1: Array or sequence of coordinates for the first parameter.
        x2: Array or sequence of coordinates for the second parameter.

    Returns:
        Array of shape (N, 2) containing normalized 2D normal vectors [n_x1, n_x2].
    """
    p1 = np.asarray(x1, dtype=float)
    p2 = np.asarray(x2, dtype=float)
    n = len(p1)
    if n == 0:
        return np.empty((0, 2), dtype=float)
    if n == 1:
        return np.array([[0.0, 1.0]], dtype=float)

    is_closed = (n >= 4) and (np.hypot(p1[0] - p1[-1], p2[0] - p2[-1]) < 1e-4)

    dx1 = np.zeros(n, dtype=float)
    dx2 = np.zeros(n, dtype=float)

    if is_closed:
        dx1 = np.roll(p1, -1) - np.roll(p1, 1)
        dx2 = np.roll(p2, -1) - np.roll(p2, 1)
    else:
        # Central difference for interior points
        dx1[1:-1] = (p1[2:] - p1[:-2]) / 2.0
        dx2[1:-1] = (p2[2:] - p2[:-2]) / 2.0
        # Forward/backward differences at boundaries
        dx1[0] = p1[1] - p1[0]
        dx2[0] = p2[1] - p2[0]
        dx1[-1] = p1[-1] - p1[-2]
        dx2[-1] = p2[-1] - p2[-2]

    # Rotate tangent by 90 degrees counter-clockwise: (-dy, dx)
    normals = np.column_stack([-dx2, dx1])
    lengths = np.hypot(normals[:, 0], normals[:, 1])

    valid_mask = lengths > 1e-12
    normals[valid_mask] = normals[valid_mask] / lengths[valid_mask, np.newaxis]
    normals[~valid_mask] = [0.0, 1.0]

    return normals


def extract_polar_ring_locus(
    grid_x1: np.ndarray,
    grid_x2: np.ndarray,
    fom_2d: np.ndarray,
    threshold_percentile: float = 85.0,
    sample_points: int = 50,
    smoothness: float = 0.0001,
    spline_degree: int = 3,
    center: tuple[float, float] | list[float] | str = "auto",
    p1_name: str = "x1",
    p2_name: str = "x2",
) -> list[dict[str, Any]]:
    """Extracts a continuous closed ring locus using polar radial ray maximum-ridge tracing.

    Shoots radial rays outward from the high-FOM centroid at uniform angular intervals,
    identifies the maximum-FOM radius along each ray, and fits a smooth periodic cubic
    B-spline. Ideal for annular/ring degeneracy manifolds.

    Args:
        grid_x1: 1D coordinate array for first parameter.
        grid_x2: 1D coordinate array for second parameter.
        fom_2d: 2D array of Figure of Merit values corresponding to (grid_x2, grid_x1).
        threshold_percentile: Percentile defining high-FOM candidate regions.
        sample_points: Number of points sampled along the smooth output curve.
        smoothness: B-spline smoothing regularization factor.
        spline_degree: Degree of B-spline interpolation.
        center: Coordinates of the ring center [x1c, x2c] or "auto" (centroid of top FOM).
        p1_name: Name of the first parameter dimension.
        p2_name: Name of the second parameter dimension.

    Returns:
        List containing a single dictionary representing the closed ring locus.
    """
    finite_fom = fom_2d[np.isfinite(fom_2d)]
    if finite_fom.size == 0:
        return []

    x1_min, x1_max = float(grid_x1[0]), float(grid_x1[-1])
    x2_min, x2_max = float(grid_x2[0]), float(grid_x2[-1])
    n1, n2 = len(grid_x1), len(grid_x2)

    # 1. Determine ring center
    if isinstance(center, (list, tuple)) and len(center) == 2:
        x1c, x2c = float(center[0]), float(center[1])
    else:
        cutoff = float(np.percentile(finite_fom, threshold_percentile))
        mask = (fom_2d >= cutoff) & np.isfinite(fom_2d)
        y_idx, x_idx = np.where(mask)
        if len(x_idx) == 0:
            x1c = 0.5 * (x1_min + x1_max)
            x2c = 0.5 * (x2_min + x2_max)
        else:
            x1c = float(np.mean(grid_x1[x_idx]))
            x2c = float(np.mean(grid_x2[y_idx]))

    # Max radial reach
    max_r = min(x1c - x1_min, x1_max - x1c, x2c - x2_min, x2_max - x2c) * 0.95
    if max_r <= 0.005:
        max_r = min(x1_max - x1_min, x2_max - x2_min) * 0.45

    # 2. Polar rays
    thetas = np.linspace(0, 2 * np.pi, max(sample_points, 30), endpoint=False)
    r_scan = np.linspace(0.005, max_r, 250)

    ring_x1 = []
    ring_x2 = []
    for th in thetas:
        ux, uy = np.cos(th), np.sin(th)
        px = x1c + r_scan * ux
        py = x2c + r_scan * uy
        idx_x = (
            (np.clip(px, x1_min, x1_max) - x1_min)
            / max(x1_max - x1_min, 1e-12)
            * (n1 - 1)
        )
        idx_y = (
            (np.clip(py, x2_min, x2_max) - x2_min)
            / max(x2_max - x2_min, 1e-12)
            * (n2 - 1)
        )
        fom_ray = ndi.map_coordinates(fom_2d, [idx_y, idx_x], order=3, mode="nearest")
        best_r = r_scan[int(np.argmax(fom_ray))]
        ring_x1.append(float(x1c + best_r * ux))
        ring_x2.append(float(x2c + best_r * uy))

    # 3. Fit smooth periodic spline
    k = min(spline_degree, len(ring_x1) - 1, 3)
    try:
        tck, _ = splprep([ring_x1, ring_x2], s=smoothness, k=k, per=True)
        u_fine = np.linspace(0, 1, sample_points, endpoint=False)
        curve_x1, curve_x2 = splev(u_fine, tck)
    except (ValueError, TypeError, RuntimeError):
        u_fine = np.linspace(0, 1, sample_points, endpoint=False)
        curve_x1 = np.interp(u_fine, np.linspace(0, 1, len(ring_x1)), ring_x1)
        curve_x2 = np.interp(u_fine, np.linspace(0, 1, len(ring_x2)), ring_x2)

    # Resample FOM along fitted closed curve
    idx_x = (
        (np.clip(curve_x1, x1_min, x1_max) - x1_min)
        / max(x1_max - x1_min, 1e-12)
        * (n1 - 1)
    )
    idx_y = (
        (np.clip(curve_x2, x2_min, x2_max) - x2_min)
        / max(x2_max - x2_min, 1e-12)
        * (n2 - 1)
    )
    fom_along_curve = ndi.map_coordinates(
        fom_2d, [idx_y, idx_x], order=3, mode="nearest"
    )

    return [
        {
            "locus_id": 1,
            "p1_name": p1_name,
            "p2_name": p2_name,
            "x1": [float(v) for v in curve_x1],
            "x2": [float(v) for v in curve_x2],
            "fom": [float(v) for v in fom_along_curve],
            "max_fom": float(np.max(fom_along_curve)),
            "mean_fom": float(np.mean(fom_along_curve)),
            "is_closed": True,
            "center": [x1c, x2c],
        }
    ]


def extract_optimal_loci(
    grid_x1: np.ndarray,
    grid_x2: np.ndarray,
    fom_2d: np.ndarray,
    threshold_percentile: float = 85.0,
    min_locus_area_px: int = 15,
    max_loci: int = 1,
    sample_points: int = 50,
    smoothness: float = 0.001,
    spline_degree: int = 3,
    mode: Literal["auto", "cartesian", "polar"] = "auto",
    p1_name: str = "x1",
    p2_name: str = "x2",
) -> list[dict[str, Any]]:
    """Extracts continuous 1D degeneracy locus curves from a 2D Figure of Merit landscape.

    Isolates the highest-FOM ridges (where band splitting cost approaches zero) and fits
    smooth parametric B-spline curves across the 2D parameter space. Supports Cartesian
    skeletonization for open branches, polar radial ray tracing for closed loops, and auto-detection.

    Args:
        grid_x1: 1D coordinate array for first parameter.
        grid_x2: 1D coordinate array for second parameter.
        fom_2d: 2D array of Figure of Merit values corresponding to (grid_x2, grid_x1).
        threshold_percentile: Cutoff percentile (e.g. 85%) defining high-FOM candidate regions.
        min_locus_area_px: Minimum connected component pixel area to qualify as a valid locus.
        max_loci: Maximum number of distinct connected locus ridges to extract (defaults to 1).
        sample_points: Number of points along the sampled smooth output curve.
        smoothness: B-spline smoothing parameter s.
        spline_degree: Degree of B-spline interpolation (default: 3).
        mode: Extraction geometry mode: 'polar' (closed ring), 'cartesian' (open), or 'auto'.
        p1_name: Name of the first parameter.
        p2_name: Name of the second parameter.

    Returns:
        List of dictionaries per detected locus, each containing:
            - 'locus_id': 1-based locus index.
            - 'p1_name': Name of parameter 1.
            - 'p2_name': Name of parameter 2.
            - 'x1': List of float coordinates along first parameter.
            - 'x2': List of float coordinates along second parameter.
            - 'fom': List of resampled FOM values along the locus curve.
            - 'max_fom': Maximum FOM achieved along this curve.
            - 'mean_fom': Mean FOM along this curve.
            - 'is_closed': True if the locus forms a closed loop.
    """
    finite_fom = fom_2d[np.isfinite(fom_2d)]
    if finite_fom.size == 0:
        return []

    if mode == "polar":
        return extract_polar_ring_locus(
            grid_x1=grid_x1,
            grid_x2=grid_x2,
            fom_2d=fom_2d,
            threshold_percentile=threshold_percentile,
            sample_points=sample_points,
            smoothness=smoothness,
            spline_degree=spline_degree,
            p1_name=p1_name,
            p2_name=p2_name,
        )

    cutoff = float(np.percentile(finite_fom, threshold_percentile))
    mask = (fom_2d >= cutoff) & np.isfinite(fom_2d)
    mask = ndi.binary_closing(mask)

    if mode == "auto":
        filled = ndi.binary_fill_holes(mask)
        # If filling holes adds a significant interior region, it is an annular ring
        if np.sum(filled) > (np.sum(mask) + 15):
            return extract_polar_ring_locus(
                grid_x1=grid_x1,
                grid_x2=grid_x2,
                fom_2d=fom_2d,
                threshold_percentile=threshold_percentile,
                sample_points=sample_points,
                smoothness=smoothness,
                spline_degree=spline_degree,
                p1_name=p1_name,
                p2_name=p2_name,
            )

    x1_min, x1_max = float(grid_x1[0]), float(grid_x1[-1])
    x2_min, x2_max = float(grid_x2[0]), float(grid_x2[-1])
    n1, n2 = len(grid_x1), len(grid_x2)

    labeled, num_features = ndi.label(mask)
    if num_features == 0:
        return []

    # Rank connected components by maximum FOM value
    components = []
    for lbl in range(1, num_features + 1):
        comp_mask = labeled == lbl
        area = int(np.sum(comp_mask))
        if area < min_locus_area_px:
            continue
        max_val = float(np.max(fom_2d[comp_mask]))
        mean_val = float(np.mean(fom_2d[comp_mask]))
        components.append((lbl, area, max_val, mean_val, comp_mask))

    if not components:
        return []

    components.sort(key=lambda c: c[2], reverse=True)
    selected_components = components[:max_loci]

    loci_results = []
    for locus_idx, (_lbl, _area, _max_val, _mean_val, comp_mask) in enumerate(
        selected_components, start=1
    ):
        skel = skeletonize_mask(comp_mask)
        y_idx, x_idx = np.where(skel)

        if len(x_idx) < 3:
            continue

        raw_x1 = grid_x1[x_idx]
        raw_x2 = grid_x2[y_idx]

        ord_x1, ord_x2 = order_skeleton_points(raw_x1, raw_x2)

        # Remove duplicate adjacent points
        dists = np.hypot(np.diff(ord_x1), np.diff(ord_x2))
        valid_steps = np.r_[True, dists > 1e-8]
        ord_x1 = ord_x1[valid_steps]
        ord_x2 = ord_x2[valid_steps]

        if len(ord_x1) < 4:
            u_fine = np.linspace(0, 1, sample_points)
            curve_x1 = np.interp(u_fine, np.linspace(0, 1, len(ord_x1)), ord_x1)
            curve_x2 = np.interp(u_fine, np.linspace(0, 1, len(ord_x2)), ord_x2)
            is_closed = False
        else:
            is_closed = (
                len(ord_x1) >= 6
                and np.hypot(ord_x1[0] - ord_x1[-1], ord_x2[0] - ord_x2[-1]) < 0.03
            )
            k = min(spline_degree, len(ord_x1) - 1, 3)
            try:
                tck, _u = splprep([ord_x1, ord_x2], s=smoothness, k=k, per=is_closed)
                u_fine = np.linspace(0, 1, sample_points)
                curve_x1, curve_x2 = splev(u_fine, tck)
            except (ValueError, TypeError, RuntimeError):
                u_fine = np.linspace(0, 1, sample_points)
                curve_x1 = np.interp(u_fine, np.linspace(0, 1, len(ord_x1)), ord_x1)
                curve_x2 = np.interp(u_fine, np.linspace(0, 1, len(ord_x2)), ord_x2)

        # Resample FOM along curve
        idx_x = (
            (np.clip(curve_x1, x1_min, x1_max) - x1_min)
            / max(x1_max - x1_min, 1e-12)
            * (n1 - 1)
        )
        idx_y = (
            (np.clip(curve_x2, x2_min, x2_max) - x2_min)
            / max(x2_max - x2_min, 1e-12)
            * (n2 - 1)
        )
        fom_along_curve = ndi.map_coordinates(
            fom_2d, [idx_y, idx_x], order=3, mode="nearest"
        )

        loci_results.append(
            {
                "locus_id": locus_idx,
                "p1_name": p1_name,
                "p2_name": p2_name,
                "x1": [float(v) for v in curve_x1],
                "x2": [float(v) for v in curve_x2],
                "fom": [float(v) for v in fom_along_curve],
                "max_fom": float(np.max(fom_along_curve)),
                "mean_fom": float(np.mean(fom_along_curve)),
                "is_closed": bool(is_closed),
            }
        )

    return loci_results


def refine_locus_points(
    locus: dict[str, Any],
    param_names: list[str],
    evaluate_point_fn: Callable[[dict[str, float]], tuple[float, float]],
    tolerance: float = 1e-5,
    max_steps: int = 8,
    exclude_unrefined: bool = True,
    max_residual_gap: float = 1e-4,
    param_bounds: dict[str, tuple[float, float]] | None = None,
    step_mag: float = 0.002,
    delta_max: float = 0.035,
) -> dict[str, Any]:
    """Refines sampled locus points to exact degeneracy using a 1D normal line search.

    For each point along the extracted locus, computes the perpendicular normal vector
    to the curve trajectory and performs a 1D secant root-finding search along the normal
    vector until the frequency gap cost between target bands drops below `tolerance`.

    Args:
        locus: Locus dictionary containing 'x1' and 'x2' coordinate lists.
        param_names: List of parameter names (e.g. ['r1', 'r2']).
        evaluate_point_fn: Callable returning (signed_gap, cost) for a given parameter dictionary.
        tolerance: Target cost threshold (e.g. 1e-5) defining exact degeneracy.
        max_steps: Maximum secant iterations per point.
        exclude_unrefined: Whether to prune points that cannot reach max_residual_gap.
        max_residual_gap: Cutoff cost above which unrefined points are flagged or pruned.
        param_bounds: Optional parameter bounds dict for clipping coordinate proposals.
        step_mag: Initial probing displacement along the normal vector.
        delta_max: Maximum allowed displacement from the initial spline point.

    Returns:
        Refined locus dictionary with updated 'x1', 'x2', 'residual_gap', 'costs',
        and preserved 'x1_unrefined', 'x2_unrefined'.
    """
    x1_pts = np.asarray(locus["x1"], dtype=float)
    x2_pts = np.asarray(locus["x2"], dtype=float)
    n_pts = len(x1_pts)

    locus["x1_unrefined"] = list(x1_pts)
    locus["x2_unrefined"] = list(x2_pts)

    normals = compute_curve_normals(x1_pts, x2_pts)
    p1_n = param_names[0] if len(param_names) >= 1 else "x1"
    p2_n = param_names[1] if len(param_names) >= 2 else "x2"

    r1_ref = []
    r2_ref = []
    gaps_ref = []
    costs_ref = []
    is_valid_list = []

    for i in range(n_pts):
        norm_vec = normals[i]
        p_init = {p1_n: float(x1_pts[i]), p2_n: float(x2_pts[i])}

        def _get_coords(
            delta: float,
            base_p: dict[str, float] = p_init,
            n_v: np.ndarray = norm_vec,
        ) -> dict[str, float]:
            cur = {
                p1_n: float(base_p[p1_n] + delta * n_v[0]),
                p2_n: float(base_p[p2_n] + delta * n_v[1]),
            }
            if param_bounds:
                if p1_n in param_bounds:
                    b1 = param_bounds[p1_n]
                    cur[p1_n] = float(np.clip(cur[p1_n], b1[0], b1[1]))
                if p2_n in param_bounds:
                    b2 = param_bounds[p2_n]
                    cur[p2_n] = float(np.clip(cur[p2_n], b2[0], b2[1]))
            return cur

        # Step 0: delta = 0
        gap0, cost0 = evaluate_point_fn(_get_coords(0.0))
        best_p = _get_coords(0.0)
        best_gap = gap0
        best_cost = cost0

        if cost0 >= tolerance:
            # Probe positive
            gap_pos, cost_pos = evaluate_point_fn(_get_coords(step_mag))
            if cost_pos < best_cost:
                best_p = _get_coords(step_mag)
                best_gap = gap_pos
                best_cost = cost_pos

            if cost_pos >= tolerance:
                if cost_pos >= cost0 or cost_pos >= 0.99:
                    gap_neg, cost_neg = evaluate_point_fn(_get_coords(-step_mag))
                    if cost_neg < best_cost:
                        best_p = _get_coords(-step_mag)
                        best_gap = gap_neg
                        best_cost = cost_neg
                    deltas = [0.0, -step_mag]
                    gaps = [gap0, gap_neg]
                    costs = [cost0, cost_neg]
                else:
                    deltas = [0.0, step_mag]
                    gaps = [gap0, gap_pos]
                    costs = [cost0, cost_pos]

                # Secant loop
                for _step in range(2, max(4, max_steps)):
                    d_prev, d_curr = deltas[-2], deltas[-1]
                    g_prev, g_curr = gaps[-2], gaps[-1]
                    denom = g_curr - g_prev
                    if abs(denom) < 1e-10:
                        d_next = d_curr + (
                            step_mag * 0.5 if _step % 2 == 0 else -step_mag * 0.5
                        )
                    else:
                        d_next = d_curr - g_curr * (d_curr - d_prev) / denom

                    d_next = float(np.clip(d_next, -delta_max, delta_max))
                    if any(abs(d_next - d) < 1e-5 for d in deltas):
                        d_next = float(
                            np.clip(
                                d_curr
                                + (step_mag * 0.25 if g_curr > 0 else -step_mag * 0.25),
                                -delta_max,
                                delta_max,
                            )
                        )

                    pt_next = _get_coords(d_next)
                    g_next, c_next = evaluate_point_fn(pt_next)
                    deltas.append(d_next)
                    gaps.append(g_next)
                    costs.append(c_next)

                    if c_next < best_cost:
                        best_p = pt_next
                        best_gap = g_next
                        best_cost = c_next

                    if c_next < tolerance:
                        break

        r1_ref.append(float(best_p[p1_n]))
        r2_ref.append(float(best_p[p2_n]))
        gaps_ref.append(float(best_gap))
        costs_ref.append(float(best_cost))
        is_valid_list.append(bool(best_cost <= max_residual_gap))

    if exclude_unrefined:
        valid_indices = [i for i, v in enumerate(is_valid_list) if v]
        num_pruned = n_pts - len(valid_indices)
        if valid_indices and num_pruned > 0:
            gap_cut = None
            for k in range(len(valid_indices) - 1):
                if valid_indices[k + 1] > valid_indices[k] + 1:
                    gap_cut = k + 1
                    break
            if gap_cut is not None:
                valid_indices = valid_indices[gap_cut:] + valid_indices[:gap_cut]
            locus["is_closed"] = False

            locus["x1"] = [r1_ref[i] for i in valid_indices]
            locus["x2"] = [r2_ref[i] for i in valid_indices]
            locus["residual_gap"] = [costs_ref[i] for i in valid_indices]
            locus["costs"] = [costs_ref[i] for i in valid_indices]
            locus["gaps"] = [gaps_ref[i] for i in valid_indices]
            locus["is_valid"] = [is_valid_list[i] for i in valid_indices]
            if "fom" in locus and len(locus["fom"]) == n_pts:
                locus["fom"] = [locus["fom"][i] for i in valid_indices]
        else:
            locus["x1"] = r1_ref
            locus["x2"] = r2_ref
            locus["residual_gap"] = costs_ref
            locus["costs"] = costs_ref
            locus["gaps"] = gaps_ref
            locus["is_valid"] = is_valid_list
    else:
        locus["x1"] = r1_ref
        locus["x2"] = r2_ref
        locus["residual_gap"] = costs_ref
        locus["costs"] = costs_ref
        locus["gaps"] = gaps_ref
        locus["is_valid"] = is_valid_list

    return locus


def evaluate_locus_group_velocities(
    locus: dict[str, Any],
    param_names: list[str],
    compute_vg_fn: Callable[[dict[str, float]], float],
) -> dict[str, Any]:
    """Evaluates group velocities at an adjacent k-point for each refined locus point.

    Args:
        locus: Locus dictionary containing refined coordinate arrays 'x1' and 'x2'.
        param_names: Names of the two parameters (e.g. ['r1', 'r2']).
        compute_vg_fn: Callable that accepts a parameter dictionary and returns
            the group velocity (dimensionless v_g / c).

    Returns:
        Updated locus dictionary with 'group_velocity' and 'vg' float lists.
    """
    x1_pts = locus["x1"]
    x2_pts = locus["x2"]
    vgs = []
    p1_name = param_names[0] if len(param_names) >= 1 else "x1"
    p2_name = param_names[1] if len(param_names) >= 2 else "x2"

    for p1, p2 in zip(x1_pts, x2_pts, strict=False):
        pt_params = {p1_name: float(p1), p2_name: float(p2)}
        vg_val = compute_vg_fn(pt_params)
        vgs.append(float(vg_val))

    locus["group_velocity"] = vgs
    locus["vg"] = vgs
    return locus


def export_locus_to_csv(
    locus: dict[str, Any],
    output_path: str | Path,
    param_names: list[str] | None = None,
) -> Path:
    """Exports a single locus trajectory with residual gaps and group velocities to CSV.

    Args:
        locus: Locus dictionary with coordinates and physical metrics.
        output_path: Target path for the output CSV file.
        param_names: Optional explicit parameter names.

    Returns:
        Resolved Path to the saved CSV file.
    """
    p = Path(output_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)

    p1_n = (
        param_names[0]
        if param_names and len(param_names) >= 1
        else locus.get("p1_name", "x1")
    )
    p2_n = (
        param_names[1]
        if param_names and len(param_names) >= 2
        else locus.get("p2_name", "x2")
    )

    x1_pts = locus.get("x1", [])
    x2_pts = locus.get("x2", [])
    n = len(x1_pts)
    foms = locus.get("fom", [0.0] * n)
    gaps = locus.get("residual_gap", locus.get("costs", [0.0] * n))
    vgs = locus.get("group_velocity", locus.get("vg", [0.0] * n))
    is_valid = locus.get("is_valid", [True] * n)

    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "point_idx",
                "t_normalized",
                p1_n,
                p2_n,
                "residual_gap",
                "fom",
                "group_velocity",
                "is_valid",
            ]
        )
        for i in range(n):
            t_norm = float(i) / max(n - 1, 1)
            writer.writerow(
                [
                    i + 1,
                    round(t_norm, 5),
                    round(float(x1_pts[i]), 6),
                    round(float(x2_pts[i]), 6),
                    round(float(gaps[i]), 8) if i < len(gaps) else 0.0,
                    round(float(foms[i]), 4) if i < len(foms) else 0.0,
                    round(float(vgs[i]), 6) if i < len(vgs) else 0.0,
                    bool(is_valid[i]) if i < len(is_valid) else True,
                ]
            )
    return p


def export_loci_to_json(
    loci: list[dict[str, Any]],
    output_path: str | Any,
) -> Any:
    """Exports extracted 1D degeneracy loci manifolds to a structured JSON file.

    Args:
        loci: List of locus dictionaries produced by extract_optimal_loci.
        output_path: Target path to write JSON.

    Returns:
        Resolved Path to the written JSON file.
    """
    import json
    from pathlib import Path

    def _json_default(obj: Any) -> Any:
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    p = Path(output_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(loci, f, indent=2, default=_json_default)
    return p
