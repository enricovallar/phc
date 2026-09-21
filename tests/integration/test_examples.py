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
