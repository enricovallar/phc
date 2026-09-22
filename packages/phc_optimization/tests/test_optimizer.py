"""Unit and fast integration tests for BayesianOptimizer engine."""

import tempfile

import gdsfactory as gf
from phc_layout.components.unit_cell import phc_wyckoff_unit_cell
from phc_optimization.optimizer import BayesianOptimizer


def dummy_cell_factory(
    r1: float = 0.25, r2: float = 0.15, pitch: float = 1.0
) -> gf.Component:
    """Parametric unit cell generator using phc_layout Wyckoff positions."""
    return phc_wyckoff_unit_cell(
        pitch=pitch,
        point_group="C4v",
        features=[("1a", r1), ("1b", r2)],
    )


def test_bayesian_optimizer_fast_run() -> None:
    """Verifies end-to-end execution of BayesianOptimizer with arbitrary and fixed parameters."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        opt = BayesianOptimizer(
            cell_factory=dummy_cell_factory,
            parameters={
                "r1": (0.20, 0.35),
                "r2": (0.10, 0.25),
            },
            fixed_parameters={"pitch": 1.0},
            objective="dirac_degeneracy",
            objective_kwargs={
                "bypass_irrep_identification": True,
                "mode_indices": [2, 3, 4],
            },
            lattice_type="square",
            pitch=1.0,
            dimension="2D",
            resolution=16,
            num_bands=4,
            initial_points=2,
            max_iterations=1,
            batch_size=1,
            enforce_connectivity=True,
            output_dir=tmp_dir,
            random_state=42,
        )

        res = opt.run()

        assert len(res.records) == 3  # 2 initial + 1 iteration
        assert res.best_fom > 0.0
        assert "r1" in res.best_params
        assert "r2" in res.best_params

        # Verify generated output files
        out_path = res.output_dir
        assert out_path is not None
        assert (out_path / "unit_cell.gds").is_file()
        assert (out_path / "simulation_results.json").is_file()
        assert (out_path / "bo_trajectory.csv").is_file()
        assert (out_path / "bo_convergence.png").is_file()
        assert (out_path / "bo_surrogate_map.png").is_file()

        # Verify per-evaluation log and data files
        assert (out_path / "bo_evaluations.log").is_file()
        assert (out_path / "bo_evaluations.jsonl").is_file()
        assert (out_path / "bo_evaluations.json").is_file()

        # Validate comprehensive JSON structure
        import json

        with open(out_path / "bo_evaluations.json") as f:
            data = json.load(f)
        assert "run_metadata" in data
        assert "evaluations" in data
        assert len(data["evaluations"]) == 3
        first = data["evaluations"][0]
        assert "params" in first
        assert "k_points" in first
        assert "freqs" in first["k_points"][0]
        assert "cost" in first
        assert "fom" in first
