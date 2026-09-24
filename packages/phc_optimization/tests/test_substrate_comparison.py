"""Unit tests for substrate band comparison and comparative plotting."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from phc_optimization.plotting import plot_band_structure_comparison


def test_plot_band_structure_comparison(tmp_path: Path):
    """Verifies that plot_band_structure_comparison correctly formats a 2-panel figure with shared y-limits."""
    k_len = 25
    bands_air = np.linspace(0.2, 0.6, k_len * 4).reshape(k_len, 4)
    bands_sub = np.linspace(0.15, 0.65, k_len * 8).reshape(k_len, 8)
    ll_air = np.linspace(0.4, 0.8, k_len)
    ll_sub = np.linspace(0.28, 0.55, k_len)

    res_air = {
        "freqs": {"te_like": bands_air},
        "light_line": ll_air,
    }
    res_sub = {
        "freqs": {"all": bands_sub},
        "light_line": ll_sub,
    }

    labels = ["M", "Γ", "X", "M"]
    indices = [0, 8, 16, 24]
    out_path = tmp_path / "band_structure_substrate_comparison.png"

    fig = plot_band_structure_comparison(
        results_air=res_air,
        results_substrate=res_sub,
        node_labels=labels,
        node_indices=indices,
        target_frequency=0.38,
        title="Test Band Comparison",
        output_path=out_path,
    )

    assert fig is not None
    assert out_path.is_file()

    axes = fig.get_axes()
    assert len(axes) == 2

    # Left and right axes should share y-limits
    y_lim_left = axes[0].get_ylim()
    y_lim_right = axes[1].get_ylim()
    np.testing.assert_allclose(y_lim_left, y_lim_right, atol=1e-5)

    plt.close(fig)


def test_run_substrate_band_comparison_smoke(tmp_path: Path):
    """Smoke test verifying run_substrate_band_comparison calculates 2.5x bands for the substrate."""
    from phc_layout.components.unit_cell import phc_wyckoff_unit_cell
    from phc_optimization.analysis import run_substrate_band_comparison

    def cell_builder(r1: float = 0.25, r2: float = 0.15):
        return phc_wyckoff_unit_cell(
            pitch=1.0,
            point_group="C4v",
            features=[("1a", r1), ("1b", r2)],
        )

    out = run_substrate_band_comparison(
        cell_factory=cell_builder,
        params={"r1": 0.25, "r2": 0.20},
        target_frequency=0.40,
        pitch=1.0,
        slab_thickness=0.5,
        supercell_z=3.0,
        resolution=(8, 8, 4),
        num_bands=2,
        k_density=2,
        output_dir=tmp_path,
        verbose=False,
    )

    assert "results_air" in out
    assert "results_substrate" in out
    assert "fig_comparison" in out

    # Verify air membrane has num_bands = 2
    freqs_air = out["results_air"]["freqs"]["te_like"]
    assert freqs_air.shape[1] == 2

    # Verify substrate has ceil(2.5 * 2) = 5 bands
    freqs_sub = out["results_substrate"]["freqs"]["all"]
    assert freqs_sub.shape[1] == 5

    assert (tmp_path / "band_structure_substrate_comparison.png").is_file()
    plt.close("all")
