"""Unit tests for objective functions and exact FOM calculations."""

import gdsfactory as gf
import numpy as np
from phc_optimization.objectives import (
    DiracDegeneracyObjective,
    ModalOverlapDegeneracyObjective,
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


def test_modal_overlap_degeneracy_evaluation() -> None:
    """Verifies modal overlap degeneracy objective with synthetic reference fields."""
    nx, ny, nz = 8, 8, 4
    # Create 3 orthogonal reference modes
    f1 = np.zeros((nx, ny, nz, 3), dtype=complex)
    f2 = np.zeros((nx, ny, nz, 3), dtype=complex)
    f3 = np.zeros((nx, ny, nz, 3), dtype=complex)
    f1[2, 2, :, 0] = 1.0
    f2[4, 4, :, 0] = 1.0
    f3[6, 6, :, 0] = 1.0

    ref_fields = {
        16: (f1, f1),
        17: (f2, f2),
        18: (f3, f3),
    }
    ref_freqs = {16: 0.865, 17: 0.865, 18: 0.865}

    obj = get_objective(
        "modal_overlap_degeneracy",
        ref_fields=ref_fields,
        ref_frequencies=ref_freqs,
        ref_bands=[16, 17, 18],
        slab_thickness=None,
        target_cost=0.0001,
    )

    # In target solver, mode order is slightly permuted or split:
    # Target band 10 matches ref 18, 11 matches ref 17, 12 matches ref 16
    t_fields = {
        10: (f3, f3),
        11: (f2, f2),
        12: (f1, f1),
        13: (
            np.ones((nx, ny, nz, 3), dtype=complex),
            np.ones((nx, ny, nz, 3), dtype=complex),
        ),
    }

    # Dummy ModeSolver mock or container for get_efield/get_dfield
    class MockModeSolver:
        def __init__(self) -> None:
            self.num_bands = 13
            self.all_freqs = [[0.0] * 13]

        def get_efield(self, band: int) -> np.ndarray:
            return t_fields.get(band, (np.zeros((nx, ny, nz, 3), dtype=complex), None))[
                0
            ]

        def get_dfield(self, band: int) -> np.ndarray:
            return t_fields.get(band, (None, np.zeros((nx, ny, nz, 3), dtype=complex)))[
                1
            ]

    ms_tar = MockModeSolver()
    component = gf.Component()
    # Target frequencies: band 10 = 0.870, band 11 = 0.865, band 12 = 0.860
    gamma_freqs = [0.0] * 9 + [0.870, 0.865, 0.860, 0.900]
    solver_results = {"freqs": {"te": [gamma_freqs]}}

    eval_res = obj.evaluate(
        component=component,
        ms=ms_tar,
        solver_results=solver_results,
        params={"pitch": 0.381},
    )

    assert not eval_res.is_penalty
    # Should identify bands 10, 11, 12 as top matching modes
    tracked = sorted(eval_res.metadata["target_bands"])
    assert tracked == [10, 11, 12]
    # Frequencies: low=0.860, mid=0.865, high=0.870
    assert abs(eval_res.metadata["freq_high"] - 0.870) < 1e-6
    assert abs(eval_res.metadata["freq_middle"] - 0.865) < 1e-6
    assert abs(eval_res.metadata["freq_low"] - 0.860) < 1e-6
    # raw_cost = 0.010
    assert abs(eval_res.metadata["raw_cost"] - 0.010) < 1e-6
    # signed_gap = (0.870 - 0.865) - (0.865 - 0.860) = 0.0
    assert abs(eval_res.metadata["signed_gap"]) < 1e-6
    # Overlap should be ~1.0
    assert eval_res.metadata["mean_overlap"] > 0.99


def test_modal_overlap_degeneracy_tracking_strategy() -> None:
    """Verifies that ModalOverlapDegeneracyObjective with cluster strategy prevents distant mode jumping."""
    nx, ny, nz = 4, 4, 4
    # 2 reference modes
    ref_fields = {
        1: (
            np.ones((nx, ny, nz, 3), dtype=complex),
            np.ones((nx, ny, nz, 3), dtype=complex),
        ),
        2: (
            np.ones((nx, ny, nz, 3), dtype=complex) * 1j,
            np.ones((nx, ny, nz, 3), dtype=complex) * 1j,
        ),
    }

    # Target fields for 4 modes:
    # Mode 1: matches ref 1
    # Mode 2: matches ref 2 slightly (0.3)
    # Mode 3: distant frequency mode matching ref 2 strongly (0.9)
    # Mode 4: empty
    f1 = np.ones((nx, ny, nz, 3), dtype=complex)
    f2 = np.ones((nx, ny, nz, 3), dtype=complex) * 0.3j
    f3 = np.ones((nx, ny, nz, 3), dtype=complex) * 0.9j

    t_fields = {
        1: (f1, f1),
        2: (f2, f2),
        3: (f3, f3),
        4: (np.zeros((nx, ny, nz, 3)), np.zeros((nx, ny, nz, 3))),
    }

    class MockMS:
        def __init__(self) -> None:
            self.num_bands = 4
            self.all_freqs = [[0.500, 0.502, 0.800, 0.900]]

        def get_efield(self, band: int) -> np.ndarray:
            return t_fields.get(band, (np.zeros((nx, ny, nz, 3)), None))[0]

        def get_dfield(self, band: int) -> np.ndarray:
            return t_fields.get(band, (None, np.zeros((nx, ny, nz, 3))))[1]

    ms_mock = MockMS()
    solver_results = {"freqs": {"te": [[0.500, 0.502, 0.800, 0.900]]}}
    comp = gf.Component()

    # 1. Cluster strategy: avoids jumping to mode 3 at 0.800, selects cohesive cluster [1, 2]
    obj_cluster = ModalOverlapDegeneracyObjective(
        ref_fields=ref_fields,
        ref_frequencies={1: 0.500, 2: 0.502},
        ref_bands=[1, 2],
        tracking_strategy="cluster",
    )
    res_cluster = obj_cluster.evaluate(comp, ms_mock, solver_results, params={})
    assert res_cluster.metadata["target_bands"] == [1, 2]
    assert abs(res_cluster.metadata["raw_cost"] - 0.002) < 1e-6

    # 2. Greedy strategy: jumps to mode 3 (higher overlap) despite large frequency disparity
    obj_greedy = ModalOverlapDegeneracyObjective(
        ref_fields=ref_fields,
        ref_frequencies={1: 0.500, 2: 0.502},
        ref_bands=[1, 2],
        tracking_strategy="greedy",
    )
    res_greedy = obj_greedy.evaluate(comp, ms_mock, solver_results, params={})
    assert 3 in res_greedy.metadata["target_bands"]


def test_modal_overlap_degeneracy_enforce_irreps() -> None:
    """Verifies that ModalOverlapDegeneracyObjective correctly penalizes alien irreps and filters valid multiplets."""
    nx, ny, nz = 4, 4, 4
    ref_fields = {
        1: (
            np.ones((nx, ny, nz, 3), dtype=complex),
            np.ones((nx, ny, nz, 3), dtype=complex),
        ),
        2: (
            np.ones((nx, ny, nz, 3), dtype=complex) * 1j,
            np.ones((nx, ny, nz, 3), dtype=complex) * 1j,
        ),
        3: (
            np.ones((nx, ny, nz, 3), dtype=complex) * -1,
            np.ones((nx, ny, nz, 3), dtype=complex) * -1,
        ),
    }

    # Target fields:
    # Bands 1, 2, 3: A_2 + E_1 + E_1 target triplet (overlap ~ 0.7)
    # Bands 4, 5, 6: E_2 + B_1 accidental crossing (overlap ~ 0.95, tiny splitting)
    f_target = {
        1: (
            np.ones((nx, ny, nz, 3), dtype=complex) * 0.7,
            np.ones((nx, ny, nz, 3), dtype=complex) * 0.7,
        ),
        2: (
            np.ones((nx, ny, nz, 3), dtype=complex) * 0.7j,
            np.ones((nx, ny, nz, 3), dtype=complex) * 0.7j,
        ),
        3: (
            np.ones((nx, ny, nz, 3), dtype=complex) * -0.7,
            np.ones((nx, ny, nz, 3), dtype=complex) * -0.7,
        ),
        4: (
            np.ones((nx, ny, nz, 3), dtype=complex) * 0.95,
            np.ones((nx, ny, nz, 3), dtype=complex) * 0.95,
        ),
        5: (
            np.ones((nx, ny, nz, 3), dtype=complex) * 0.95j,
            np.ones((nx, ny, nz, 3), dtype=complex) * 0.95j,
        ),
        6: (
            np.ones((nx, ny, nz, 3), dtype=complex) * -0.95,
            np.ones((nx, ny, nz, 3), dtype=complex) * -0.95,
        ),
    }

    class MockMS:
        def __init__(self) -> None:
            self.num_bands = 6
            self.all_freqs = [[0.740, 0.750, 0.751, 0.8100, 0.8101, 0.8102]]

        def get_efield(self, band: int) -> np.ndarray:
            return f_target.get(band, (np.zeros((nx, ny, nz, 3)), None))[0]

        def get_dfield(self, band: int) -> np.ndarray:
            return f_target.get(band, (None, np.zeros((nx, ny, nz, 3))))[1]

    ms_mock = MockMS()
    solver_results = {
        "freqs": {"te": [[0.740, 0.750, 0.751, 0.8100, 0.8101, 0.8102]]},
        "symmetries": {
            "te": [
                {"band": 1, "irrep": "A_2", "confidence": 0.95},
                {"band": 2, "irrep": "E_1", "confidence": 0.95},
                {"band": 3, "irrep": "E_1", "confidence": 0.95},
                {"band": 4, "irrep": "E_2", "confidence": 0.95},
                {"band": 5, "irrep": "B_1", "confidence": 0.95},
                {"band": 6, "irrep": "E_1", "confidence": 0.95},
            ]
        },
    }
    comp = gf.Component()

    # 1. With enforce_irreps=True: Must select the genuine A_2 + E_1 + E_1 cluster [1, 2, 3]
    obj_enforced = ModalOverlapDegeneracyObjective(
        ref_fields=ref_fields,
        ref_frequencies={1: 0.750, 2: 0.750, 3: 0.750},
        ref_bands=[1, 2, 3],
        target_irreps=["A_2", "E_1", "E_1"],
        enforce_irreps=True,
        tracking_strategy="cluster",
    )
    res_enforced = obj_enforced.evaluate(comp, ms_mock, solver_results, params={})
    assert res_enforced.metadata["target_bands"] == [1, 2, 3]
    assert res_enforced.metadata["irrep_match"] is True
    assert res_enforced.metadata["tracked_irreps"] == ["A_2", "E_1", "E_1"]
    assert res_enforced.is_penalty is False

    # 2. If target_irreps is mismatched, it must be penalized
    obj_mismatched = ModalOverlapDegeneracyObjective(
        ref_fields=ref_fields,
        ref_frequencies={1: 0.750, 2: 0.750, 3: 0.750},
        ref_bands=[1, 2, 3],
        target_irreps=["B_2", "B_2", "B_2"],
        enforce_irreps=True,
        tracking_strategy="cluster",
        irrep_penalty_weight=50.0,
    )
    res_mismatched = obj_mismatched.evaluate(comp, ms_mock, solver_results, params={})
    assert res_mismatched.metadata["irrep_match"] is False
    assert res_mismatched.is_penalty is True
    assert res_mismatched.cost >= 50.0
    assert res_mismatched.fom <= 0.02
