"""Unit tests for objective functions and exact FOM calculations."""

import gdsfactory as gf
import numpy as np
from phc_optimization.objectives import (
    DiracDegeneracyObjective,
    get_objective,
)


def test_get_objective_factory() -> None:
    """Tests resolving objectives from registry by string name."""
    obj = get_objective("dirac_degeneracy", target_irreps=["A_1", "E", "E"])
    assert isinstance(obj, DiracDegeneracyObjective)
    assert obj.target_irreps == ["A_1", "E", "E"]


def test_dirac_degeneracy_exact_fom() -> None:
    """Verifies exact FOM calculation matching the legacy formula."""
    obj = DiracDegeneracyObjective(
        target_irreps=["A_1", "E", "E"],
        bypass_irrep_identification=True,
        mode_indices=[2, 3, 4],
        target_cost=0.001,
    )

    component = gf.Component()
    # Gamma frequencies for bands 1..4: band 2=0.500, band 3=0.502, band 4=0.504
    freqs_gamma = np.array([[0.10, 0.500, 0.502, 0.504]])
    solver_results = {"freqs": {"te": freqs_gamma}}

    eval_res = obj.evaluate(
        component, ms=None, solver_results=solver_results, params={}
    )

    # High = 0.504, Low = 0.500
    # raw_cost = 0.004
    # freq_middle = 0.502
    # normalized_cost = 0.004 / 0.502 = 0.00796812749003984
    # expected_fom = 1.0 / normalized_cost = 125.5
    assert not eval_res.is_penalty
    assert abs(eval_res.metadata["raw_cost"] - 0.004) < 1e-6
    expected_norm_cost = 0.004 / 0.502
    assert abs(eval_res.cost - expected_norm_cost) < 1e-6
    assert abs(eval_res.fom - (1.0 / expected_norm_cost)) < 1e-4


def test_dirac_degeneracy_target_cost_clamp() -> None:
    """Verifies that cost is clamped to target_cost floor when splitting is smaller."""
    obj = DiracDegeneracyObjective(
        bypass_irrep_identification=True,
        mode_indices=[2, 3, 4],
        target_cost=0.01,
    )

    component = gf.Component()
    # Very small split: high=0.5001, low=0.5000 -> norm_cost ~ 0.0002 < target_cost (0.01)
    freqs_gamma = np.array([[0.10, 0.5000, 0.50005, 0.5001]])
    solver_results = {"freqs": {"te": freqs_gamma}}

    eval_res = obj.evaluate(
        component, ms=None, solver_results=solver_results, params={}
    )
    assert eval_res.cost == 0.01
