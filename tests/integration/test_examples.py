from pathlib import Path

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
