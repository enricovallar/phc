"""Unit tests for degeneracy locus extraction, normal computation, secant refinement, and profiling."""

from pathlib import Path

import numpy as np
from phc_optimization.locus import (
    compute_curve_normals,
    evaluate_locus_dirac_frequencies,
    evaluate_locus_group_velocities,
    export_locus_to_csv,
    extract_optimal_loci,
    extract_polar_ring_locus,
    find_target_locus_point,
    refine_locus_points,
)
from phc_optimization.plotting import plot_locus_dirac_frequency, plot_locus_profile


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
        fixed_parameters={"pitch": 1.0, "supercell_z": 4.0, "slab_thickness": 0.5},
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


def test_evaluate_locus_dirac_frequencies():
    """Verifies that evaluate_locus_dirac_frequencies computes and populates Dirac frequencies."""
    locus = {
        "locus_id": 1,
        "x1": [0.25, 0.28, 0.30],
        "x2": [0.20, 0.22, 0.24],
        "p1_name": "r1",
        "p2_name": "r2",
    }

    def mock_freq_fn(params: dict[str, float]) -> float:
        # Mock linear dispersion with parameters
        return 0.30 + 0.2 * params["r1"] + 0.1 * params["r2"]

    updated = evaluate_locus_dirac_frequencies(
        locus=locus,
        param_names=["r1", "r2"],
        compute_freq_fn=mock_freq_fn,
    )
    assert "dirac_frequency" in updated
    assert len(updated["dirac_frequency"]) == 3
    assert updated["dirac_frequency"][0] == 0.30 + 0.2 * 0.25 + 0.1 * 0.20
    assert "omega_d" in updated
    assert updated["omega_d"] == updated["dirac_frequency"]


def test_plot_locus_dirac_frequency(tmp_path: Path):
    """Verifies that plot_locus_dirac_frequency generates a 2-panel figure with target highlighting."""
    n_pts = 10
    x1_vals = np.linspace(0.20, 0.30, n_pts)
    x2_vals = np.linspace(0.15, 0.25, n_pts)
    # Frequency monotonically increases from 0.28 to 0.42
    omega_d_vals = np.linspace(0.28, 0.42, n_pts)

    locus = {
        "locus_id": 1,
        "p1_name": "r1",
        "p2_name": "r2",
        "x1": list(x1_vals),
        "x2": list(x2_vals),
        "dirac_frequency": list(omega_d_vals),
    }

    target_wavelength = 1.55
    target_thickness = 0.25
    slab_thickness = 0.5
    pitch = 1.0

    fig_path = tmp_path / "test_dirac_freq.png"
    fig = plot_locus_dirac_frequency(
        locus=locus,
        param_names=["r1", "r2"],
        target_wavelength=target_wavelength,
        target_thickness=target_thickness,
        slab_thickness=slab_thickness,
        pitch=pitch,
        output_path=fig_path,
        title="Test Dirac Frequency Plot",
    )

    assert fig is not None
    assert fig_path.is_file()

    # Verify 2 subplots exist
    axes = fig.get_axes()
    # At least ax1, ax2 (and possibly inset)
    assert len(axes) >= 2
    # Verify metadata on locus
    assert "optimal_match" in locus
    opt_info = locus["optimal_match"]
    assert "omega_d" in opt_info
    assert "pitch" in opt_info
    assert "thickness" in opt_info
    assert "frequency_thz" in opt_info
    assert "thickness_error" in opt_info
    assert opt_info["target_wavelength"] == target_wavelength
    assert opt_info["target_thickness"] == target_thickness

    # Ideal omega_d = target_thickness / ( (slab_thickness / pitch) * target_wavelength )
    expected_ideal = 0.25 / (0.5 * 1.55)
    np.testing.assert_allclose(opt_info["ideal_omega_d"], expected_ideal, rtol=1e-5)

    # Check closest point was selected
    errors = [abs(0.5 * w * target_wavelength - target_thickness) for w in omega_d_vals]
    expected_best_idx = int(np.argmin(errors))
    assert opt_info["index"] == expected_best_idx


def test_refine_locus_points_records_frequencies():
    """Verifies that refine_locus_points captures Dirac frequencies when evaluate_point_fn returns 3 elements."""
    locus = {
        "locus_id": 1,
        "x1": [0.25, 0.28],
        "x2": [0.20, 0.22],
        "fom": [10.0, 10.0],
        "is_closed": False,
    }

    def mock_eval_3tuple(params: dict[str, float]) -> tuple[float, float, float]:
        gap = abs(params["r1"] - 0.25)
        cost = gap
        freq = 0.35 + 0.1 * params["r1"]
        return gap, cost, freq

    refined = refine_locus_points(
        locus=locus,
        param_names=["r1", "r2"],
        evaluate_point_fn=mock_eval_3tuple,
        tolerance=1e-5,
        max_steps=4,
        exclude_unrefined=False,
    )

    assert "dirac_frequency" in refined
    assert len(refined["dirac_frequency"]) == 2
    assert "omega_d" in refined
    assert refined["dirac_frequency"][0] > 0.0


def test_find_target_locus_point():
    """Verifies that find_target_locus_point correctly identifies closest points for target thickness and target frequency."""
    locus = {
        "locus_id": 1,
        "p1_name": "r1",
        "p2_name": "r2",
        "x1": [0.20, 0.25, 0.30],
        "x2": [0.15, 0.20, 0.25],
        "dirac_frequency": [0.35, 0.40, 0.45],
    }

    # Case 1: Target frequency
    idx, match = find_target_locus_point(
        locus=locus,
        target_frequency=0.402,
        pitch=1.0,
        slab_thickness=0.5,
    )
    assert idx == 1
    assert match["point_idx"] == 2
    assert match["omega_d"] == 0.40
    assert match["ideal_omega_d"] == 0.402
    assert match["r1"] == 0.25
    assert match["r2"] == 0.20

    # Case 2: Target wavelength and thickness
    # eta = 0.5 / 1.0 = 0.5
    # h = eta * omega_d * lambda = 0.5 * omega_d * 1.55
    # For omega_d = 0.35 -> h = 0.27125
    # For omega_d = 0.40 -> h = 0.310
    # For omega_d = 0.45 -> h = 0.34875
    # Target thickness = 0.31 -> closest is index 1
    idx2, match2 = find_target_locus_point(
        locus=locus,
        target_wavelength=1.55,
        target_thickness=0.31,
        pitch=1.0,
        slab_thickness=0.5,
    )
    assert idx2 == 1
    assert match2["point_idx"] == 2
    assert abs(match2["thickness"] - 0.31) < 1e-4


def test_load_loci_from_json_and_csv(tmp_path: Path):
    """Verifies saving and re-loading loci from JSON and CSV formats."""
    from phc_optimization.locus import (
        export_loci_to_json,
        export_locus_to_csv,
        load_loci_from_json,
        load_locus_from_csv,
    )

    locus = {
        "locus_id": 1,
        "p1_name": "r1",
        "p2_name": "r2",
        "x1": [0.22, 0.28],
        "x2": [0.18, 0.24],
        "dirac_frequency": [0.38, 0.42],
        "group_velocity": [0.15, 0.18],
        "residual_gap": [1e-6, 2e-6],
        "fom": [95.0, 98.0],
        "is_valid": [True, True],
    }

    # Test JSON export and load
    json_path = tmp_path / "optimal_loci.json"
    export_loci_to_json([locus], json_path)
    assert json_path.is_file()

    # Load from direct file
    loaded_from_file = load_loci_from_json(json_path)
    assert len(loaded_from_file) == 1
    assert loaded_from_file[0]["x1"] == locus["x1"]
    assert loaded_from_file[0]["dirac_frequency"] == locus["dirac_frequency"]

    # Load from directory
    loaded_from_dir = load_loci_from_json(tmp_path)
    assert len(loaded_from_dir) == 1
    assert loaded_from_dir[0]["x2"] == locus["x2"]

    # Test CSV export and load
    csv_path = tmp_path / "locus_points.csv"
    export_locus_to_csv(locus, csv_path, param_names=["r1", "r2"])
    assert csv_path.is_file()

    loaded_csv = load_locus_from_csv(csv_path)
    assert loaded_csv["p1_name"] == "r1"
    assert loaded_csv["p2_name"] == "r2"
    np.testing.assert_allclose(loaded_csv["x1"], locus["x1"])
    np.testing.assert_allclose(loaded_csv["dirac_frequency"], locus["dirac_frequency"])
    np.testing.assert_allclose(loaded_csv["group_velocity"], locus["group_velocity"])


def test_find_target_locus_point_nanometers():
    """Verifies that find_target_locus_point correctly converts nanometer inputs and returns nm metadata."""
    locus = {
        "locus_id": 1,
        "p1_name": "r1",
        "p2_name": "r2",
        "x1": [0.20, 0.25, 0.30],
        "x2": [0.15, 0.20, 0.25],
        "dirac_frequency": [0.35, 0.40, 0.45],
    }

    idx, match = find_target_locus_point(
        locus=locus,
        target_wavelength_nm=1550.0,
        target_thickness_nm=310.0,
        pitch=1.0,
        slab_thickness=0.5,
    )
    assert idx == 1
    assert match["target_wavelength"] == 1.55
    assert match["target_wavelength_nm"] == 1550.0
    assert match["target_thickness"] == 0.31
    assert match["target_thickness_nm"] == 310.0
    assert abs(match["thickness_nm"] - 310.0) < 0.1
    assert "pitch_nm" in match


def test_find_latest_locus_path(tmp_path: Path):
    """Verifies finding the newest optimal_loci.json in a directory tree."""
    import time

    from phc_optimization.locus import find_latest_locus_path, load_loci_from_json

    dir1 = tmp_path / "run_old"
    dir2 = tmp_path / "run_new"
    dir1.mkdir()
    dir2.mkdir()

    f1 = dir1 / "optimal_loci.json"
    f2 = dir2 / "optimal_loci.json"
    f1.write_text('[{"locus_id": 1, "x1": [0.2], "x2": [0.2]}]', encoding="utf-8")
    time.sleep(0.05)
    f2.write_text('[{"locus_id": 2, "x1": [0.3], "x2": [0.3]}]', encoding="utf-8")

    latest_path = find_latest_locus_path(base_dir=tmp_path)
    assert latest_path == f2.resolve()

    # Test load_loci_from_json with 'latest'
    loaded = load_loci_from_json(path=None, geometry=None)
    assert isinstance(loaded, list)
