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

from typing import Any

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


def extract_optimal_loci(
    grid_x1: np.ndarray,
    grid_x2: np.ndarray,
    fom_2d: np.ndarray,
    threshold_percentile: float = 85.0,
    min_locus_area_px: int = 15,
    max_loci: int = 2,
    sample_points: int = 50,
    smoothness: float = 0.001,
    spline_degree: int = 3,
) -> list[dict[str, Any]]:
    """Extracts continuous 1D degeneracy locus curves from a 2D Figure of Merit landscape.

    Isolates the highest-FOM ridges (where band splitting cost approaches zero) and fits
    smooth parametric B-spline curves across the 2D parameter space. Works for ANY two
    arbitrary parameter dimensions.

    Args:
        grid_x1: 1D coordinate array for first parameter.
        grid_x2: 1D coordinate array for second parameter.
        fom_2d: 2D array of Figure of Merit values corresponding to (grid_x2, grid_x1).
        threshold_percentile: Cutoff percentile (e.g. 85%) defining high-FOM candidate regions.
        min_locus_area_px: Minimum connected component pixel area to qualify as a valid locus.
        max_loci: Maximum number of distinct connected locus ridges to extract.
        sample_points: Number of points along the sampled smooth output curve.
        smoothness: B-spline smoothing parameter s.
        spline_degree: Degree of B-spline interpolation (default: 3).

    Returns:
        List of dictionaries per detected locus, each containing:
            - 'locus_id': 1-based locus index.
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

    x1_min, x1_max = float(grid_x1[0]), float(grid_x1[-1])
    x2_min, x2_max = float(grid_x2[0]), float(grid_x2[-1])
    n1, n2 = len(grid_x1), len(grid_x2)

    cutoff = float(np.percentile(finite_fom, threshold_percentile))
    mask = (fom_2d >= cutoff) & np.isfinite(fom_2d)
    mask = ndi.binary_closing(mask)

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
                "x1": [float(v) for v in curve_x1],
                "x2": [float(v) for v in curve_x2],
                "fom": [float(v) for v in fom_along_curve],
                "max_fom": float(np.max(fom_along_curve)),
                "mean_fom": float(np.mean(fom_along_curve)),
                "is_closed": bool(is_closed),
            }
        )

    return loci_results


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
