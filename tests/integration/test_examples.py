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
