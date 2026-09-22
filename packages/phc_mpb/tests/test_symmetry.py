"""Unit tests for point-group symmetry calculation, character projection, and degenerate mode resolution."""

import meep as mp
import pytest
from phc_mpb.solver import create_mode_solver, run_band_solver
from phc_mpb.symmetry import (
    compute_projections,
    compute_subspace_trace_projection,
    failsafe_irrep_mapping,
    find_bands_from_irreps,
    get_symmetry_operators,
    identify_irrep,
)


def test_symmetry_operators() -> None:
    """Verifies construction of symmetry operator matrices for C4v and C6v."""
    ops_c4v = get_symmetry_operators("C4v")
    assert "C4" in ops_c4v
    assert "C2" in ops_c4v
    assert "sv" in ops_c4v
    assert "sd" in ops_c4v

    ops_c6v = get_symmetry_operators("C6v")
    assert "C6" in ops_c6v
    assert "C3" in ops_c6v
    assert "C2" in ops_c6v
    assert "sv" in ops_c6v
    assert "sd" in ops_c6v

    with pytest.raises(ValueError, match="Unsupported point group"):
        get_symmetry_operators("C2v")


def test_character_projections_c4v() -> None:
    """Verifies projection formulas for ideal C4v character tables."""
    # A_1: C4=1, C2=1, sv=1, sd=1
    chars_a1 = {"C4": 1.0, "C2": 1.0, "sv": 1.0, "sd": 1.0}
    projs = compute_projections(chars_a1, group="C4v")
    irrep, conf = identify_irrep(projs)
    assert irrep == "A_1"
    assert abs(conf - 1.0) < 1e-4

    # A_2: C4=1, C2=1, sv=-1, sd=-1
    chars_a2 = {"C4": 1.0, "C2": 1.0, "sv": -1.0, "sd": -1.0}
    projs = compute_projections(chars_a2, group="C4v")
    irrep, conf = identify_irrep(projs)
    assert irrep == "A_2"
    assert abs(conf - 1.0) < 1e-4

    # B_1: C4=-1, C2=1, sv=1, sd=-1
    chars_b1 = {"C4": -1.0, "C2": 1.0, "sv": 1.0, "sd": -1.0}
    projs = compute_projections(chars_b1, group="C4v")
    irrep, conf = identify_irrep(projs)
    assert irrep == "B_1"
    assert abs(conf - 1.0) < 1e-4

    # B_2: C4=-1, C2=1, sv=-1, sd=1
    chars_b2 = {"C4": -1.0, "C2": 1.0, "sv": -1.0, "sd": 1.0}
    projs = compute_projections(chars_b2, group="C4v")
    irrep, conf = identify_irrep(projs)
    assert irrep == "B_2"
    assert abs(conf - 1.0) < 1e-4

    # E (single state): C4=0, C2=-1, sv=0, sd=0
    chars_e = {"C4": 0.0, "C2": -1.0, "sv": 0.0, "sd": 0.0}
    projs = compute_projections(chars_e, group="C4v")
    irrep, conf = identify_irrep(projs)
    assert irrep == "E"
    assert abs(conf - 1.0) < 1e-4


def test_character_projections_c6v() -> None:
    """Verifies projection formulas for C6v character tables."""
    # E_1 in C6v: C6=1, C3=-1, C2=-2, sv=0, sd=0
    chars_e1 = {"C6": 0.5, "C3": -0.5, "C2": -1.0, "sv": 0.0, "sd": 0.0}
    projs = compute_projections(chars_e1, group="C6v")
    irrep, _conf = identify_irrep(projs)
    assert irrep == "E_1"


def test_degenerate_subspace_trace_projection() -> None:
    """Tests invariant trace projection when numerical eigensolvers mix states in an E doublet."""
    # Suppose two degenerate numerical states psi1 and psi2 have rotated mixed reflections
    rec1 = {
        "band": 3,
        "characters": {"C4": 0.0, "C2": -1.0, "sv": 0.707, "sd": -0.707},
    }
    rec2 = {
        "band": 4,
        "characters": {"C4": 0.0, "C2": -1.0, "sv": -0.707, "sd": 0.707},
    }

    mult = compute_subspace_trace_projection([rec1, rec2], symmetry_group="C4v")
    # Tr(E)=2, Tr(C4)=0, Tr(C2)=-2, Tr(sv)=0, Tr(sd)=0 -> exactly E irrep with multiplicity 1.0
    assert abs(mult["E"] - 1.0) < 1e-4
    assert abs(mult["A_1"]) < 1e-4
    assert abs(mult["A_2"]) < 1e-4


def test_failsafe_irrep_mapping() -> None:
    """Tests failsafe relabeling on a scrambled degenerate cluster."""
    full_map = {
        2: ("A_1", 0.98, 0.450),
        3: ("Unknown", 0.52, 0.451),
        4: ("Unknown", 0.49, 0.451),
    }
    corrections = failsafe_irrep_mapping(
        target_irreps=["A_1", "E", "E"],
        degeneracy_tol=0.005,
        full_irrep_map=full_map,
        bands_to_check=[2, 3, 4],
    )
    assert len(corrections) == 2
    assert full_map[3][0] == "E"
    assert full_map[4][0] == "E"


def test_find_bands_from_irreps() -> None:
    """Tests dynamic mapping of target irreps to band indices."""
    symmetries = [
        {"band": 1, "irrep": "A_1", "confidence": 1.0, "freq": 0.10},
        {"band": 2, "irrep": "A_2", "confidence": 0.99, "freq": 0.35},
        {"band": 3, "irrep": "E", "confidence": 0.95, "freq": 0.38},
        {"band": 4, "irrep": "E", "confidence": 0.95, "freq": 0.38},
    ]
    bands, _full_map, err, _corr = find_bands_from_irreps(
        symmetries=symmetries,
        target_irreps=["A_2", "E", "E"],
        min_band=2,
        degeneracy_tol=0.005,
    )
    assert err is None
    assert bands == [2, 3, 4]


def test_live_mpb_symmetry_and_group_velocity() -> None:
    """Tests running MPB with compute_symmetries and compute_group_velocities."""
    lattice = mp.Lattice(size=mp.Vector3(1, 1))
    geom = [mp.Cylinder(radius=0.2, material=mp.Medium(index=3.45))]
    k_points = [mp.Vector3(0, 0, 0)]

    ms = create_mode_solver(
        geometry_lattice=lattice,
        geometry=geom,
        k_points=k_points,
        resolution=16,
        num_bands=4,
        default_material="air",
    )

    results = run_band_solver(
        ms,
        polarization="te",
        dimension="2D",
        compute_group_velocities=True,
        compute_symmetries=True,
        symmetry_group="C4v",
    )

    assert "symmetries" in results
    assert "te" in results["symmetries"]
    sym_recs = results["symmetries"]["te"]
    assert len(sym_recs) == 4
    # Band 1 and 2 should be A_2
    assert sym_recs[0]["irrep"] == "A_2"

    assert "group_velocities" in results
    assert "te" in results["group_velocities"]
    vgs = results["group_velocities"]["te"]
    assert vgs.shape == (1, 4, 3)
