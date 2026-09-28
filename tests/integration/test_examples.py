from pathlib import Path

import numpy as np
import pytest

from examples.demo_hex_2d_mpb import run_hex_2d_pipeline


@pytest.mark.integration
def test_hex_2d_mpb_pipeline_smoke(tmp_path: Path):
    """End-to-end smoke test verifying layout -> materials -> mpb interconnection."""
    results = run_hex_2d_pipeline(
        pitch=0.45,
        radius_ratio=0.25,
        quick=True,
        output_dir=tmp_path,
    )

    # 1. Assert result payload structure
    assert "results" in results
    assert "freqs" in results["results"]
    assert "te" in results["results"]["freqs"]

    freqs = results["results"]["freqs"]["te"]
    assert freqs.shape[0] > 0  # k-points
    assert freqs.shape[1] == 4  # 4 bands in quick mode

    # 2. Assert generated files in tmp_path
    assert (tmp_path / "phc_hex_unit_cell.gds").is_file()
    assert (tmp_path / "epsilon_hex_2d.png").is_file()
    assert (tmp_path / "band_diagram_hex_te.png").is_file()


@pytest.mark.integration
def test_unit_cell_plot_pipeline_smoke(tmp_path: Path):
    """End-to-end smoke test verifying unit cell layout generation and .plot() pipeline."""
    from examples.plot_unit_cells import run_unit_cell_plot_pipeline

    results = run_unit_cell_plot_pipeline(quick=True, output_dir=tmp_path)

    assert "plots" in results
    assert "cells" in results
    assert len(results["plots"]) > 0

    for name in results["cells"]:
        ws_file = tmp_path / f"{name}_wigner_seitz.png"
        prim_file = tmp_path / f"{name}_primitive.png"
        assert ws_file.is_file()
        assert prim_file.is_file()


@pytest.mark.integration
def test_run_all_band_diagrams_pipeline_smoke(tmp_path: Path):
    """End-to-end smoke test verifying batch band diagram pipeline execution."""
    from examples.run_all_band_diagrams import run_all_band_diagrams_pipeline

    out = run_all_band_diagrams_pipeline(quick=True, output_dir=tmp_path, verbose=False)

    assert "summary" in out
    assert "results" in out
    assert "output_dir" in out
    assert "summary_file" in out

    # Verify summary JSON file created
    summary_file = Path(out["summary_file"])
    assert summary_file.is_file()

    # Verify geometries were processed
    summary = out["summary"]
    assert summary["total_geometries"] == 2
    geoms = summary["geometries"]
    assert len(geoms) == 2

    for g in geoms:
        cell_dir = tmp_path / g["name"]
        assert cell_dir.is_dir()
        assert (cell_dir / "unit_cell.gds").is_file()
        assert (cell_dir / "band_structure.png").is_file()
        assert (cell_dir / "epsilon_map.png").is_file()
        assert (cell_dir / "simulation_results.json").is_file()


@pytest.mark.integration
def test_demo_slab_3d_parallel_mpb_smoke(tmp_path: Path):
    """End-to-end smoke test verifying 3D PhC slab parallel MPB simulation pipeline."""
    from examples.demo_slab_3d_parallel_mpb import run_slab_3d_pipeline

    out = run_slab_3d_pipeline(quick=True, output_dir=tmp_path, verbose=False)

    assert "results" in out
    assert "freqs" in out["results"]
    assert "te_like" in out["results"]["freqs"]
    assert "light_line" in out["results"]

    freqs = out["results"]["freqs"]["te_like"]
    assert freqs.shape[0] > 0
    assert freqs.shape[1] == 3  # 3 bands in quick mode

    # Canonical artifacts (phc_hydra SSOT standard)
    assert (tmp_path / "unit_cell.gds").is_file()
    assert (tmp_path / "band_structure.png").is_file()
    assert (tmp_path / "epsilon_map.png").is_file()
    assert (tmp_path / "simulation_results.json").is_file()

    # Legacy aliases for backward-compatibility
    assert (tmp_path / "slab_unit_cell.gds").is_file()
    assert (tmp_path / "epsilon_slab_profiles.png").is_file()
    assert (tmp_path / "epsilon_slab_midplane.png").is_file()
    assert (tmp_path / "band_diagram_slab_te_like.png").is_file()
    assert (tmp_path / "slab_simulation_results.json").is_file()


@pytest.mark.integration
def test_demo_slab_3d_substrate_mpb_smoke(tmp_path: Path):
    """End-to-end smoke test verifying 3D asymmetric slab with substrate and polarization fractions."""
    from examples.demo_slab_3d_substrate_mpb import run_slab_3d_substrate_pipeline

    out = run_slab_3d_substrate_pipeline(quick=True, output_dir=tmp_path, verbose=False)

    assert "results" in out
    assert "freqs" in out["results"]
    assert "all" in out["results"]["freqs"]
    assert "light_line" in out["results"]
    assert "te_fractions" in out["results"]

    freqs = out["results"]["freqs"]["all"]
    assert freqs.shape[0] > 0  # k-points
    assert freqs.shape[1] == 4  # 4 bands in quick mode

    te_fracs = out["results"]["te_fractions"]
    assert te_fracs.shape == freqs.shape
    assert te_fracs.min() >= 0.0
    assert te_fracs.max() <= 1.0

    # Canonical artifacts (phc_hydra SSOT standard)
    assert (tmp_path / "unit_cell.gds").is_file()
    assert (tmp_path / "band_structure.png").is_file()
    assert (tmp_path / "epsilon_map.png").is_file()
    assert (tmp_path / "simulation_results.json").is_file()


@pytest.mark.integration
def test_demo_slab_3d_mode_parity_mpb_smoke(tmp_path: Path):
    """End-to-end smoke test verifying 3D slab mode parity (TE-like vs TM-like) and TE fraction analysis."""
    from examples.demo_slab_3d_mode_parity_te_fraction_mpb import (
        run_slab_3d_mode_parity_pipeline,
    )

    out = run_slab_3d_mode_parity_pipeline(
        quick=True,
        output_dir=tmp_path,
        verbose=False,
    )

    # 1. Assert result structures
    assert "results_unified" in out
    assert "results_parity" in out
    assert "results_all" in out
    assert "te_like" in out["results_parity"]["freqs"]
    assert "tm_like" in out["results_parity"]["freqs"]
    assert "all" in out["results_all"]["freqs"]
    assert "te_fractions" in out["results_all"]
    assert "comparison" in out

    freqs_te = out["results_parity"]["freqs"]["te_like"]
    freqs_tm = out["results_parity"]["freqs"]["tm_like"]
    freqs_all = out["results_all"]["freqs"]["all"]
    te_fracs = out["results_all"]["te_fractions"]

    assert freqs_te.shape[0] > 0
    assert freqs_tm.shape[0] > 0
    assert freqs_all.shape[0] > 0
    assert te_fracs.shape == freqs_all.shape
    assert te_fracs.min() >= 0.0
    assert te_fracs.max() <= 1.0

    assert "confinements" in out["results_all"]
    conf_all = out["results_all"]["confinements"]
    assert conf_all is not None
    assert conf_all.shape == freqs_all.shape
    assert conf_all.min() >= 0.0
    assert conf_all.max() <= 1.0

    # 2. Assert canonical and comparison artifacts
    assert (tmp_path / "unit_cell.gds").is_file()
    assert (tmp_path / "band_structure.png").is_file()
    assert (tmp_path / "band_structure_parity.png").is_file()
    assert (tmp_path / "band_structure_te_fraction.png").is_file()
    assert (tmp_path / "band_structure_comparison.png").is_file()
    assert (tmp_path / "epsilon_map.png").is_file()
    assert (tmp_path / "simulation_results.json").is_file()


@pytest.mark.integration
def test_demo_optimization_2d_smoke(tmp_path: Path):
    """End-to-end smoke test verifying 2D Bayesian Optimization workflow."""
    from examples.demo_optimization_2d import run_optimization_2d_pipeline

    out = run_optimization_2d_pipeline(
        quick=True,
        output_dir=tmp_path,
    )

    assert "best_params" in out
    assert "r1" in out["best_params"]
    assert "r2" in out["best_params"]
    assert "best_fom" in out
    assert out["best_fom"] > 0.0
    assert out["total_evaluations"] == 3

    assert (tmp_path / "unit_cell.gds").is_file()
    assert (tmp_path / "simulation_results.json").is_file()
    assert (tmp_path / "bo_trajectory.csv").is_file()
    assert (tmp_path / "bo_convergence.png").is_file()
    assert (tmp_path / "bo_surrogate_map.png").is_file()
    assert (tmp_path / "bo_evaluations.log").is_file()
    assert (tmp_path / "bo_evaluations.jsonl").is_file()
    assert (tmp_path / "bo_evaluations.json").is_file()


@pytest.mark.integration
def test_demo_optimization_3d_smoke(tmp_path: Path):
    """End-to-end smoke test verifying 3D PhC Slab Bayesian Optimization workflow."""
    from examples.demo_optimization_3d import run_optimization_3d_pipeline

    out = run_optimization_3d_pipeline(
        quick=True,
        output_dir=tmp_path,
    )

    assert "best_params" in out
    assert "r1" in out["best_params"]
    assert "r2" in out["best_params"]
    assert "best_fom" in out
    assert out["best_fom"] > 0.0
    assert out["total_evaluations"] == 3

    assert (tmp_path / "unit_cell.gds").is_file()
    assert (tmp_path / "simulation_results.json").is_file()
    assert (tmp_path / "bo_trajectory.csv").is_file()
    assert (tmp_path / "bo_convergence.png").is_file()
    assert (tmp_path / "bo_surrogate_map.png").is_file()
    assert (tmp_path / "bo_evaluations.log").is_file()
    assert (tmp_path / "bo_evaluations.jsonl").is_file()
    assert (tmp_path / "bo_evaluations.json").is_file()


@pytest.mark.integration
def test_demo_slab_3d_cladding_sweep_smoke(tmp_path: Path):
    """End-to-end smoke test verifying 3D PhC slab cladding refractive index sweep and mode tracking."""
    from examples.demo_slab_3d_cladding_sweep_mpb import run_cladding_sweep_pipeline

    out = run_cladding_sweep_pipeline(
        quick=True,
        output_dir=tmp_path,
        verbose=False,
    )

    # 1. Assert result payload structure
    assert "n_clad_values" in out
    assert "baseline_te_freqs" in out
    assert "baseline_tm_freqs" in out
    assert "freqs" in out
    assert "te_fractions" in out
    assert "geometry" in out
    assert "artifacts" in out

    n_clad = out["n_clad_values"]
    assert len(n_clad) == 3

    baseline_te = out["baseline_te_freqs"]
    assert len(baseline_te) == 10

    baseline_tm = out["baseline_tm_freqs"]
    assert len(baseline_tm) == 10

    freqs = out["freqs"]
    assert freqs.shape == (3, 20)

    te_fractions = out["te_fractions"]
    assert te_fractions.shape == (3, 20)
    assert te_fractions.min() >= 0.0
    assert te_fractions.max() <= 1.0

    # 2. Assert canonical, epsilon, and band structure artifacts
    assert (tmp_path / "unit_cell.gds").is_file()
    assert (tmp_path / "cladding_sweep.png").is_file()
    assert (tmp_path / "band_structure_baseline_reference.png").is_file()
    assert (tmp_path / "band_structure_overlapped_with_reference.png").is_file()
    assert (tmp_path / "band_structure.png").is_file()
    assert (tmp_path / "epsilon_map.png").is_file()
    assert (tmp_path / "epsilon_vertical_profiles_comparison.png").is_file()
    assert (tmp_path / "simulation_results.json").is_file()

    assert "epsilon_plots" in out
    assert len(out["epsilon_plots"]) == 3
    for p_str in out["epsilon_plots"]:
        assert Path(p_str).is_file()

    assert "band_plots" in out
    assert len(out["band_plots"]) == 3
    for p_str in out["band_plots"]:
        assert Path(p_str).is_file()


@pytest.mark.integration
def test_c6v_mode_tracking_pipeline_smoke(tmp_path: Path):
    """End-to-end smoke test verifying C6v 2b-6d modal overlap and mode tracking pipeline."""
    from analysis.test_c6v_mode_tracking import run_mode_tracking_pipeline

    out = run_mode_tracking_pipeline(
        quick=True,
        output_dir=tmp_path,
        verbose=False,
    )

    assert "results_summary" in out
    assert "overlap_matrix" in out
    assert "best_matches" in out
    assert "top_target_bands" in out

    overlap_mat = out["overlap_matrix"]
    assert overlap_mat.ndim == 2
    assert np.all(overlap_mat >= 0.0)
    assert np.all(overlap_mat <= 1.0)

    # Canonical artifacts
    assert (tmp_path / "unit_cell.gds").is_file()
    assert (tmp_path / "epsilon_map.png").is_file()
    assert (tmp_path / "mode_tracking.png").is_file()
    assert (tmp_path / "simulation_results.json").is_file()


@pytest.mark.integration
def test_c6v_substrate_optimization_pipeline_smoke(tmp_path: Path):
    """End-to-end smoke test verifying C6v 2b-6d accidental degeneracy optimization on SiO2."""
    from analysis.optimize_c6v_substrate_degeneracy import (
        run_c6v_substrate_optimization_pipeline,
    )

    out = run_c6v_substrate_optimization_pipeline(
        quick=True,
        output_dir=tmp_path,
        verbose=False,
    )

    assert "target_design" in out
    assert "primary_locus" in out
    assert "optimization_result" in out
    assert "manifest" in out

    td = out["target_design"]
    assert "parameters" in td
    assert "r1_um" in td["parameters"]
    assert "r2_um" in td["parameters"]
    assert "operating_point" in td
    assert "omega_d" in td["operating_point"]

    # Canonical artifacts
    assert (tmp_path / "unit_cell.gds").is_file()
    assert (tmp_path / "optimal_loci.png").is_file()
    assert (tmp_path / "locus_refined.csv").is_file()
    assert (tmp_path / "target_design_summary.json").is_file()
    assert (tmp_path / "simulation_results.json").is_file()


@pytest.mark.integration
def test_optimal_c6v_substrate_bands_smoke(tmp_path: Path):
    """End-to-end smoke test verifying optimal C6v band structure simulation on SiO2."""
    from analysis.simulate_optimal_c6v_substrate_bands import (
        run_optimal_band_structure_simulation,
    )

    out = run_optimal_band_structure_simulation(
        quick=True,
        output_dir=tmp_path,
        verbose=False,
    )

    assert "results" in out
    assert "freqs" in out["results"]
    assert "all" in out["results"]["freqs"]
    assert len(out["results"]["freqs"]["all"]) > 0

    assert (tmp_path / "unit_cell.gds").is_file()
    assert (tmp_path / "epsilon_map.png").is_file()
    assert (tmp_path / "band_structure.png").is_file()
    assert (tmp_path / "band_structure_wavelength.png").is_file()
    assert (tmp_path / "simulation_results.json").is_file()


@pytest.mark.integration
def test_normal_te_symmetry_analysis_smoke():
    """Smoke test verifying normal TE implementation and Gamma symmetry classification."""
    from analysis.simulate_optimal_c6v_substrate_bands import (
        run_normal_te_symmetry_analysis,
    )

    out = run_normal_te_symmetry_analysis(
        resolution=12,
        resolution_z=8,
        num_bands=6,
        verbose=False,
    )

    assert "air_frequencies" in out
    assert "air_symmetries" in out
    assert "substrate_frequencies" in out
    assert "substrate_symmetries" in out
    assert len(out["air_symmetries"]) > 0
    assert len(out["substrate_symmetries"]) > 0


