from pathlib import Path

import pytest

from examples.simulate_unit_cell import run_unit_cell_simulation


@pytest.mark.integration
def test_simulate_c6v_primitive_quick(tmp_path: Path):
    """Verifies that c6v_primitive unit cell executes cleanly in quick mode."""
    cfg = {
        "geometry": {
            "name": "c6v_primitive",
            "lattice_type": "hexagonal",
            "point_group": "C6v",
            "pitch_um": 0.5,
            "radius_ratio": 0.25,
            "background_material": "si",
            "etch_material": "air",
            "dimension": "2D",
        },
        "simulation": {
            "solver": "mpb",
            "sim_type": "band_diagram",
            "polarization": "te",
        },
        "verbose": False,
    }

    out = run_unit_cell_simulation(cfg, quick=True, output_dir=tmp_path)

    assert "results" in out
    assert "te" in out["results"]["freqs"]
    freqs = out["results"]["freqs"]["te"]
    assert freqs.shape[1] == 4  # 4 bands in quick mode

    # Verify generated files
    assert (tmp_path / "unit_cell.gds").is_file()
    assert (tmp_path / "band_structure.png").is_file()
    assert (tmp_path / "epsilon_map.png").is_file()
    assert (tmp_path / "simulation_results.json").is_file()


@pytest.mark.integration
def test_simulate_c6v_honeycomb_quick(tmp_path: Path):
    """Verifies that c6v_honeycomb unit cell executes cleanly in quick mode."""
    cfg = {
        "geometry": {
            "name": "c6v_honeycomb",
            "lattice_type": "hexagonal",
            "point_group": "C6v",
            "pitch_um": 0.5,
            "background_material": "si",
            "etch_material": "air",
            "dimension": "2D",
        },
        "simulation": {
            "solver": "mpb",
            "sim_type": "band_diagram",
            "polarization": "te",
        },
        "verbose": False,
    }

    out = run_unit_cell_simulation(cfg, quick=True, output_dir=tmp_path)
    assert (tmp_path / "band_structure.png").is_file()
    assert out["results"]["freqs"]["te"].shape[1] == 4


@pytest.mark.integration
def test_simulate_c6v_kagome_quick(tmp_path: Path):
    """Verifies that c6v_kagome unit cell executes cleanly in quick mode."""
    cfg = {
        "geometry": {
            "name": "c6v_kagome",
            "lattice_type": "hexagonal",
            "point_group": "C6v",
            "pitch_um": 0.5,
            "background_material": "si",
            "etch_material": "air",
            "dimension": "2D",
        },
        "simulation": {
            "solver": "mpb",
            "sim_type": "band_diagram",
            "polarization": "te",
        },
        "verbose": False,
    }

    run_unit_cell_simulation(cfg, quick=True, output_dir=tmp_path)
    assert (tmp_path / "unit_cell.gds").is_file()
    assert (tmp_path / "simulation_results.json").is_file()


@pytest.mark.integration
def test_simulate_c4v_lieb_quick(tmp_path: Path):
    """Verifies that c4v_lieb unit cell executes cleanly in quick mode."""
    cfg = {
        "geometry": {
            "name": "c4v_lieb",
            "lattice_type": "square",
            "point_group": "C4v",
            "pitch_um": 0.5,
            "background_material": "si",
            "etch_material": "air",
            "dimension": "2D",
        },
        "simulation": {
            "solver": "mpb",
            "sim_type": "band_diagram",
            "polarization": "te",
        },
        "verbose": False,
    }

    run_unit_cell_simulation(cfg, quick=True, output_dir=tmp_path)
    assert (tmp_path / "unit_cell.gds").is_file()
    assert (tmp_path / "band_structure.png").is_file()


@pytest.mark.integration
def test_hierarchical_output_path(tmp_path: Path):
    """Verifies that the output directory follows outputs/<solver>/<sim_type>/<geometry>/..."""
    hierarchical_dir = (
        tmp_path / "outputs" / "mpb" / "band_diagram" / "c6v_primitive" / "test_run"
    )
    cfg = {
        "geometry": {
            "name": "c6v_primitive",
            "lattice_type": "hexagonal",
            "point_group": "C6v",
            "pitch_um": 0.5,
            "background_material": "si",
            "etch_material": "air",
            "dimension": "2D",
        },
        "simulation": {
            "solver": "mpb",
            "sim_type": "band_diagram",
            "polarization": "te",
        },
        "verbose": False,
    }

    out = run_unit_cell_simulation(cfg, quick=True, output_dir=hierarchical_dir)
    out_dir = Path(out["output_dir"])

    # Path should contain mpb/band_diagram/c6v_primitive/test_run
    parts = out_dir.parts
    assert "outputs" in parts
    assert "mpb" in parts
    assert "band_diagram" in parts
    assert "c6v_primitive" in parts


@pytest.mark.integration
def test_simulate_c6v_snowflake_6d_quick(tmp_path: Path):
    """Verifies that c6v_snowflake_6d executes cleanly in quick mode."""
    cfg = {
        "geometry": {
            "name": "c6v_snowflake_6d",
            "lattice_type": "hexagonal",
            "point_group": "C6v",
            "pitch_um": 0.5,
            "background_material": "si",
            "etch_material": "air",
            "dimension": "2D",
        },
        "simulation": {
            "solver": "mpb",
            "sim_type": "band_diagram",
            "polarization": "te",
        },
        "verbose": False,
    }

    out = run_unit_cell_simulation(cfg, quick=True, output_dir=tmp_path)
    assert (tmp_path / "unit_cell.gds").is_file()
    assert (tmp_path / "band_structure.png").is_file()
    assert (tmp_path / "simulation_results.json").is_file()
    assert out["results"]["freqs"]["te"].shape[1] == 4


@pytest.mark.integration
def test_simulate_c6v_painter_snowflake_quick(tmp_path: Path):
    """Verifies that c6v_painter_snowflake executes cleanly in quick mode."""
    cfg = {
        "geometry": {
            "name": "c6v_painter_snowflake",
            "lattice_type": "hexagonal",
            "point_group": "C6v",
            "pitch_um": 0.5,
            "radius_ratio": 0.40,
            "width_ratio": 0.15,
            "background_material": "si",
            "etch_material": "air",
            "dimension": "2D",
        },
        "simulation": {
            "solver": "mpb",
            "sim_type": "band_diagram",
            "polarization": "te",
        },
        "verbose": False,
    }

    out = run_unit_cell_simulation(cfg, quick=True, output_dir=tmp_path)
    assert (tmp_path / "unit_cell.gds").is_file()
    assert (tmp_path / "band_structure.png").is_file()
    assert (tmp_path / "simulation_results.json").is_file()
    assert out["results"]["freqs"]["te"].shape[1] == 4
