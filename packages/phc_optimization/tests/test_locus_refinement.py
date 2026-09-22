"""Unit tests for degeneracy locus extraction, normal computation, secant refinement, and profiling."""

from pathlib import Path

import numpy as np
from phc_optimization.locus import (
    compute_curve_normals,
    evaluate_locus_group_velocities,
    export_locus_to_csv,
    extract_optimal_loci,
    extract_polar_ring_locus,
    refine_locus_points,
)
from phc_optimization.plotting import plot_locus_profile


def test_compute_curve_normals_line():
    """Verifies that normal vectors along a straight diagonal line are unit length and perpendicular."""
    x1 = np.linspace(0.0, 1.0, 20)
    x2 = np.linspace(0.0, 1.0, 20)

    normals = compute_curve_normals(x1, x2)
    assert normals.shape == (20, 2)

    # Unit length check
    lengths = np.hypot(normals[:, 0], normals[:, 1])
    np.testing.assert_allclose(lengths, 1.0, atol=1e-6)

    # Normal to (1, 1) direction should be (-1/sqrt(2), 1/sqrt(2))
    expected_normal = np.array([-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)])
    for n in normals:
        np.testing.assert_allclose(np.abs(np.dot(n, expected_normal)), 1.0, atol=1e-5)


def test_compute_curve_normals_circle():
    """Verifies normal vectors along a circular loop point radially."""
    thetas = np.linspace(0.0, 2.0 * np.pi, 30, endpoint=True)
    r = 0.25
    x1 = r * np.cos(thetas)
    x2 = r * np.sin(thetas)

    normals = compute_curve_normals(x1, x2)
    lengths = np.hypot(normals[:, 0], normals[:, 1])
    np.testing.assert_allclose(lengths, 1.0, atol=1e-6)

    # Tangent is counter-clockwise -> normal rotated 90 deg CCW points inward or outward
    # In either case, normal is parallel to radial unit vector (x1/r, x2/r)
    rad_u = np.column_stack([np.cos(thetas), np.sin(thetas)])
    dot_prods = np.abs(np.sum(normals * rad_u, axis=1))
    np.testing.assert_allclose(dot_prods, 1.0, atol=1e-2)


def test_extract_polar_ring_locus():
    """Verifies polar ray tracing extraction on a synthetic circular ridge landscape."""
    x1 = np.linspace(-0.5, 0.5, 60)
    x2 = np.linspace(-0.5, 0.5, 60)
    X1, X2 = np.meshgrid(x1, x2)

    r_target = 0.28
    dist = np.hypot(X1, X2)
    fom_2d = 100.0 * np.exp(-((dist - r_target) ** 2) / 0.002)

    loci = extract_polar_ring_locus(
        grid_x1=x1,
        grid_x2=x2,
        fom_2d=fom_2d,
        threshold_percentile=70.0,
        sample_points=35,
        p1_name="radius_x",
        p2_name="radius_y",
    )

    assert len(loci) == 1
    locus = loci[0]
    assert locus["is_closed"] is True
    assert len(locus["x1"]) == 35
    assert len(locus["x2"]) == 35
    assert locus["max_fom"] > 80.0

    # Coordinates should lie on circle of radius r_target
    extracted_radii = np.hypot(locus["x1"], locus["x2"])
    np.testing.assert_allclose(extracted_radii, r_target, atol=0.015)


def test_extract_optimal_loci_single_selection():
    """Verifies that max_loci=1 extracts exactly one dominant ridge from multiple candidates."""
    x1 = np.linspace(0.0, 1.0, 50)
    x2 = np.linspace(0.0, 1.0, 50)
    X1, _ = np.meshgrid(x1, x2)

    # Ridge 1 (stronger FOM)
    fom1 = 200.0 * np.exp(-((X1 - 0.3) ** 2) / 0.002)
    # Ridge 2 (weaker FOM)
    fom2 = 50.0 * np.exp(-((X1 - 0.8) ** 2) / 0.002)
    fom_2d = fom1 + fom2

    loci_single = extract_optimal_loci(
        grid_x1=x1,
        grid_x2=x2,
        fom_2d=fom_2d,
        threshold_percentile=70.0,
        max_loci=1,
        sample_points=20,
    )
    assert len(loci_single) == 1
    assert loci_single[0]["max_fom"] > 150.0


def test_refine_locus_points_synthetic():
    """Verifies that 1D secant refinement along curve normals fine-tunes points to exact degeneracy."""
    # Synthetic target manifold: circle of radius R_0 = 0.30
    # True gap: g(p1, p2) = p1^2 + p2^2 - R_0^2
    r_target = 0.30

    def mock_eval(params: dict[str, float]) -> tuple[float, float]:
        p1 = params["r1"]
        p2 = params["r2"]
        gap = float(p1**2 + p2**2 - r_target**2)
        cost = abs(gap) / 0.5
        return gap, cost

    # Initialize unrefined points with slight offset: radius ~ 0.285
    thetas = np.linspace(0.2, 1.2, 10)
    r_init = 0.285
    init_x1 = r_init * np.cos(thetas)
    init_x2 = r_init * np.sin(thetas)

    locus = {
        "locus_id": 1,
        "x1": list(init_x1),
        "x2": list(init_x2),
        "fom": [10.0] * 10,
        "is_closed": False,
    }

    refined_locus = refine_locus_points(
        locus=locus,
        param_names=["r1", "r2"],
        evaluate_point_fn=mock_eval,
        tolerance=1e-5,
        max_steps=8,
    )

    assert "x1_unrefined" in refined_locus
    assert len(refined_locus["x1_unrefined"]) == 10
    assert "residual_gap" in refined_locus

    # Check that residual costs are below tolerance
    for c in refined_locus["costs"]:
        assert c < 1e-5

    # Check refined radii are now close to r_target
    refined_radii = np.hypot(refined_locus["x1"], refined_locus["x2"])
    np.testing.assert_allclose(refined_radii, r_target, atol=1e-4)


def test_evaluate_locus_group_velocities():
    """Verifies that adjacent-point group velocity evaluation populates locus dictionary."""
    locus = {
        "locus_id": 1,
        "x1": [0.25, 0.26, 0.27],
        "x2": [0.20, 0.21, 0.22],
    }

    def mock_vg(params: dict[str, float]) -> float:
        return float(params["r1"] * 0.5 + params["r2"] * 0.2)

    updated = evaluate_locus_group_velocities(
        locus=locus,
        param_names=["r1", "r2"],
        compute_vg_fn=mock_vg,
    )

    assert "group_velocity" in updated
    assert len(updated["group_velocity"]) == 3
    expected_vgs = [
        0.25 * 0.5 + 0.20 * 0.2,
        0.26 * 0.5 + 0.21 * 0.2,
        0.27 * 0.5 + 0.22 * 0.2,
    ]
    np.testing.assert_allclose(updated["group_velocity"], expected_vgs, atol=1e-6)


def test_plot_locus_profile_and_export_csv(tmp_path: Path):
    """Verifies that 3-panel locus profile plot and CSV export execute cleanly without error."""
    n = 15
    locus = {
        "locus_id": 1,
        "p1_name": "r1",
        "p2_name": "r2",
        "x1": list(np.linspace(0.20, 0.35, n)),
        "x2": list(np.linspace(0.20, 0.30, n)),
        "group_velocity": list(np.linspace(0.12, 0.28, n)),
        "fom": list(np.logspace(2, 4, n)),
        "residual_gap": list(np.logspace(-4, -2, n)),
        "is_valid": [True] * n,
    }

    fig_path = tmp_path / "test_locus_profile.png"
    fig = plot_locus_profile(
        locus=locus,
        param_names=["r1", "r2"],
        output_path=fig_path,
        title="Test Locus Profile",
    )
    assert fig is not None
    assert fig_path.is_file()

    csv_path = tmp_path / "test_locus_points.csv"
    saved_csv = export_locus_to_csv(locus, csv_path, param_names=["r1", "r2"])
    assert saved_csv.is_file()

    csv_content = csv_path.read_text()
    assert "point_idx" in csv_content
    assert "group_velocity" in csv_content
    assert "r1" in csv_content
    assert "r2" in csv_content


def test_bayesian_optimizer_analyze_locus_runs(tmp_path: Path):
    """Verifies that BayesianOptimizer.analyze_locus runs cleanly without parameter mismatch."""
    import gdsfactory as gf
    from phc_layout.components.unit_cell import phc_wyckoff_unit_cell
    from phc_optimization import BayesianOptimizer

    def mock_cell(
        r1: float = 0.25, r2: float = 0.20, pitch: float = 1.0
    ) -> gf.Component:
        return phc_wyckoff_unit_cell(
            pitch=pitch,
            point_group="C4v",
            features=[("1a", r1), ("1b", r2)],
        )

    opt = BayesianOptimizer(
        cell_factory=mock_cell,
        parameters={"r1": (0.22, 0.32), "r2": (0.20, 0.30)},
        fixed_parameters={"pitch": 1.0},
        objective="dirac_degeneracy",
        objective_kwargs={
            "symmetry_group": "C4v",
            "polarization": "te",
            "bypass_irrep_identification": True,
            "mode_indices": [3, 4, 5],
        },
        batch_size=1,
        num_workers=1,
        resolution=12,
        num_bands=6,
        initial_points=2,
        max_iterations=1,
        output_dir=tmp_path,
        show_progress=False,
    )

    opt.run(show_progress=False)
    # Call analyze_locus and ensure no TypeError is raised
    res = opt.analyze_locus(sample_points=5, max_refine_steps=2)
    assert isinstance(res, list)
