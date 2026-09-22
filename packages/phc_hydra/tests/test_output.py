"""Unit tests for SimulationOutputManager and directory resolution in phc_hydra."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from omegaconf import OmegaConf
from phc_hydra.output import (
    CANONICAL_GDS_NAME,
    CANONICAL_RESULTS_JSON_NAME,
    SimulationOutputManager,
    resolve_simulation_output_dir,
)


class MockLayoutComponent:
    """Mock layout component implementing write_gds protocol."""

    def __init__(self, name: str = "mock_cell") -> None:
        self.name = name

    def write_gds(self, filepath: str) -> None:
        Path(filepath).write_text("dummy gds bytes")


def test_resolve_simulation_output_dir_default(tmp_path: Path):
    """Verifies that the canonical output path matches outputs/<solver>/<sim_type>/<geom>/<ts>."""
    out = resolve_simulation_output_dir(
        solver="mpb",
        sim_type="slab_band_diagram",
        geometry="c6v_primitive",
        timestamp="2026-01-01_12-00-00",
        base_dir=tmp_path,
    )
    expected = (
        tmp_path / "mpb" / "slab_band_diagram" / "c6v_primitive" / "2026-01-01_12-00-00"
    )
    assert out == expected
    assert out.is_dir()


def test_resolve_simulation_output_dir_override(tmp_path: Path):
    """Verifies that explicit override_dir takes precedence."""
    override = tmp_path / "custom_run"
    out = resolve_simulation_output_dir(
        solver="mpb",
        sim_type="band_diagram",
        geometry="c6v_primitive",
        override_dir=override,
    )
    assert out == override
    assert out.is_dir()


def test_resolve_simulation_output_dir_empty_raises():
    """Verifies that empty parameters raise ValueError."""
    with pytest.raises(ValueError, match="solver cannot be empty"):
        resolve_simulation_output_dir(solver="")

    with pytest.raises(ValueError, match="sim_type cannot be empty"):
        resolve_simulation_output_dir(sim_type="")

    with pytest.raises(ValueError, match="geometry cannot be empty"):
        resolve_simulation_output_dir(geometry="")


def test_simulation_output_manager_mandatory_gds_enforcement(tmp_path: Path):
    """Verifies that save_results_json raises RuntimeError if no GDS was exported."""
    manager = SimulationOutputManager(output_dir=tmp_path)
    assert not manager.has_gds()

    with pytest.raises(RuntimeError, match="Mandatory GDS layout missing"):
        manager.save_results_json(
            geometry_cfg={"name": "test"},
            simulation_cfg={"solver": "mpb"},
            validate_gds=True,
        )


def test_simulation_output_manager_full_lifecycle(tmp_path: Path):
    """Verifies end-to-end artifact saving: GDS, plot, data, and summary JSON."""
    manager = SimulationOutputManager(
        output_dir=tmp_path,
        solver="mpb",
        sim_type="band_diagram",
        geometry_name="c6v_primitive",
    )

    # 1. Save dummy GDS
    cell = MockLayoutComponent("UNIT_CELL")
    gds_path = manager.save_gds(cell, filename=CANONICAL_GDS_NAME)
    assert gds_path.is_file()
    assert manager.has_gds()
    assert manager.files["gds"] == gds_path

    # 2. Save figure
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    plot_path = manager.save_figure(fig, artifact_key="band_plot", close=True)
    assert plot_path.is_file()
    assert manager.files["band_plot"] == plot_path

    # 3. Save data array
    arr = np.array([1.0, 2.0, 3.0])
    data_path = manager.save_data(arr, artifact_key="data", filename="eigenvalues.npy")
    assert data_path.is_file()
    assert manager.files["data"] == data_path

    # 4. Save results JSON (mandatory GDS rule satisfied)
    json_path = manager.save_results_json(
        geometry_cfg={"name": "c6v_primitive", "pitch_um": 0.5},
        simulation_cfg={"solver": "mpb", "resolution": 32},
        band_gaps=[{"bands": [1, 2], "gap_pct": 15.2}],
        extra_data={"speedup": 3.8},
    )
    assert json_path.is_file()
    assert json_path.name == CANONICAL_RESULTS_JSON_NAME

    # 5. Check manifest
    manifest = manager.manifest
    assert "gds" in manifest
    assert "band_plot" in manifest
    assert "data" in manifest
    assert "results_json" in manifest


def test_simulation_output_manager_from_config(tmp_path: Path):
    """Verifies creation of output manager from DictConfig."""
    cfg = OmegaConf.create(
        {
            "geometry": {"name": "c4v_snowflake"},
            "simulation": {"solver": "mpb", "sim_type": "band_diagram"},
        }
    )
    manager = SimulationOutputManager.from_config(cfg, output_dir=tmp_path)
    assert manager.geometry_name == "c4v_snowflake"
    assert manager.solver == "mpb"
    assert manager.sim_type == "band_diagram"
    assert manager.output_dir == tmp_path


def test_find_latest_simulation(tmp_path: Path):
    """Verifies that find_latest_simulation correctly resolves the most recent simulation run."""
    from phc_hydra.output import find_latest_simulation

    # 1. Empty directory raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        find_latest_simulation(base_dir=tmp_path)

    # 2. Create mock hierarchical directories
    run1 = (
        tmp_path / "mpb" / "slab_mode_parity" / "c6v_primitive" / "2026-01-01_10-00-00"
    )
    run2 = (
        tmp_path / "mpb" / "slab_mode_parity" / "c6v_primitive" / "2026-01-01_12-00-00"
    )
    run1.mkdir(parents=True)
    run2.mkdir(parents=True)

    (run1 / CANONICAL_RESULTS_JSON_NAME).write_text('{"run": 1}')
    (run2 / CANONICAL_RESULTS_JSON_NAME).write_text('{"run": 2}')

    # Should pick run2 (latest timestamp)
    latest = find_latest_simulation(
        solver="mpb",
        sim_type="slab_mode_parity",
        geometry="c6v_primitive",
        base_dir=tmp_path,
    )
    assert latest == run2

    # Auto-discovery with partial filters
    latest_auto = find_latest_simulation(
        sim_type="slab_mode_parity",
        base_dir=tmp_path,
    )
    assert latest_auto == run2

    # Unmatched filter raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        find_latest_simulation(solver="nonexistent", base_dir=tmp_path)
