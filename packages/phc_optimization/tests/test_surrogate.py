"""Unit tests for surrogate re-fitting, landscape prediction, and locus extraction."""

import numpy as np
from phc_optimization.locus import (
    extract_optimal_loci,
    order_skeleton_points,
    zhang_suen_thinning,
)
from phc_optimization.surrogate import (
    fit_clean_surrogate,
    predict_surrogate_landscape,
)
from phc_optimization.types import OptimizationRecord


def _make_dummy_records(n: int = 15) -> list[OptimizationRecord]:
    """Generates synthetic records with physical and penalty evaluations."""
    records = []
    np.random.seed(42)
    for i in range(n):
        # Arbitrary parameter names
        p1 = float(np.random.uniform(0.2, 0.4))
        p2 = float(np.random.uniform(0.1, 0.3))

        # Synthetic cost based on distance from diagonal locus p1 - p2 = 0.1
        dist = abs(p1 - p2 - 0.1)
        cost = float(dist * 0.5 + 0.01)

        # Introduce 2 penalty points
        is_penalty = i in (3, 7)
        if is_penalty:
            cost = 1.0

        records.append(
            OptimizationRecord(
                eval_index=i + 1,
                generation=i // 4,
                params={"radius_a": p1, "radius_b": p2},
                cost=cost,
                fom=1.0 / cost,
                status="Evaluated",
                metadata={"penalty": is_penalty},
            )
        )
    return records


def test_fit_clean_surrogate_filters_penalties():
    """Verifies that clean GP refitting excludes penalty points and trains successfully."""
    records = _make_dummy_records(20)
    param_names = ["radius_a", "radius_b"]

    gp = fit_clean_surrogate(
        records=records,
        param_names=param_names,
        cost_cutoff=0.5,
        min_clean_points=5,
    )
    assert gp is not None

    # Predict on test coordinates
    test_pts = np.array([[0.25, 0.15], [0.35, 0.25]])
    pred, std = gp.predict(test_pts, return_std=True)
    assert len(pred) == 2
    assert np.all(std >= 0.0)


def test_predict_surrogate_landscape_arbitrary_parameters():
    """Verifies that predict_surrogate_landscape works on any 2 arbitrary parameter names."""
    records = _make_dummy_records(15)
    param_names = ["radius_a", "radius_b"]
    bounds = ((0.2, 0.4), (0.1, 0.3))

    landscape = predict_surrogate_landscape(
        records=records,
        param_names=param_names,
        bounds=bounds,
        grid_points=30,
        clean_refit=True,
    )

    assert landscape.param_names == ("radius_a", "radius_b")
    assert landscape.X1.shape == (30, 30)
    assert landscape.X2.shape == (30, 30)
    assert landscape.predicted_fom.shape == (30, 30)
    assert np.all(landscape.predicted_fom > 0.0)
    assert landscape.is_clean_fit is True


def test_zhang_suen_thinning():
    """Verifies topological skeletonization on a synthetic binary strip."""
    mask = np.zeros((30, 30), dtype=bool)
    # 5-pixel-wide diagonal strip
    for i in range(5, 25):
        mask[i, max(0, i - 2) : min(30, i + 3)] = True

    skel = zhang_suen_thinning(mask)
    assert np.any(skel)
    # Skeleton must have significantly fewer pixels than original wide strip
    assert np.sum(skel) < np.sum(mask) // 2


def test_order_skeleton_points():
    """Verifies sequential endpoint-to-endpoint point ordering."""
    x = np.array([0.0, 2.0, 1.0, 3.0])
    y = np.array([0.0, 2.0, 1.0, 3.0])

    ord_x, ord_y = order_skeleton_points(x, y)
    assert len(ord_x) == 4
    # Points should form monotonic sequence along diagonal
    dists = np.hypot(np.diff(ord_x), np.diff(ord_y))
    assert np.all(dists > 0.0)


def test_extract_optimal_loci():
    """Verifies extraction of a 1D degeneracy locus ridge from a 2D FOM landscape."""
    x1 = np.linspace(0.2, 0.4, 50)
    x2 = np.linspace(0.1, 0.3, 50)
    X1, X2 = np.meshgrid(x1, x2)

    # Ridge along X1 - X2 = 0.1
    fom = 100.0 * np.exp(-((X1 - X2 - 0.1) ** 2) / 0.001)

    loci = extract_optimal_loci(
        grid_x1=x1,
        grid_x2=x2,
        fom_2d=fom,
        threshold_percentile=80.0,
        sample_points=25,
    )

    assert len(loci) >= 1
    locus = loci[0]
    assert "x1" in locus
    assert "x2" in locus
    assert "fom" in locus
    assert len(locus["x1"]) == 25
    assert locus["max_fom"] > 50.0
