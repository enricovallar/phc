#!/usr/bin/env python3
"""Bayesian Optimization of the C6v 2b-6d Photonic Crystal Unit Cell.

Optimizes the geometric parameters of the C6v hexagonal lattice unit cell
(Wyckoff 2b holes with radius r1 and Wyckoff 6d holes with radius r2 and parameter p2)
etched in an hBN membrane (or 2D periodic sheet / 3D substrate-clad slab) for accidental
Dirac cone degeneracy at Γ:

1. Defines a parametric C6v hexagonal lattice unit cell with Wyckoff 2b and 6d holes.
2. Connects to the BayesianOptimizer engine with either DiracDegeneracyObjective (irrep-based)
   or ModalOverlapDegeneracyObjective (tracking 3 modes from an unperturbed reference simulation).
3. Supports asymmetric cladding substrates (e.g. SiO2 substrate beneath hBN slab).
4. Automatically enforces physical dielectric continuity and connectivity.
5. Solves all eigenbands at Γ and tracks mode degeneracies (phc_mpb).
6. Supports scale-invariant formulation (e.g. h/a = 0.25) and physical mapping to target
   wavelength (e.g. λ = 436 nm) and thickness (e.g. h = 100 nm).
7. Refines continuous degeneracy locus curves and evaluates adjacent group velocities.
8. Solves and plots full band diagrams (both normalized and physical wavelength) for both
   the global optimum and the closest physical match point on the refined locus.
9. Exports optimal layout GDS (unit_cell.gds), convergence plots, and optimal loci into
   the canonical output directory.

Command-Line Usage:
    # 100-point initial exploration (h/a=0.25, te_like, 16 workers, target A2 + E1 + E1):
    python analysis/optimize_c6v_2b_6d.py --initial-points 100 --slab-thickness 0.25 --workers 16

    # Rapid smoke test (< 5 seconds, low resolution, 2 initial points, 1 iteration):
    python analysis/optimize_c6v_2b_6d.py --quick

    # 3D slab on SiO2 substrate matching 3 reference modes from unperturbed air-clad membrane:
    python analysis/optimize_c6v_2b_6d.py --substrate-material sio2 --match-reference-modes --workers 16

    # Refine continuous degeneracy locus #1 and plot closest match full band diagram:
    python analysis/optimize_c6v_2b_6d.py --analyze-locus

    # Detect 2 loci on GP surrogate map and refine only locus #2:
    python analysis/optimize_c6v_2b_6d.py --find-loci 2 --analyze-locus 2

    # Analyze locus #2 from an existing optimization run without re-running optimization:
    python analysis/optimize_c6v_2b_6d.py --from-run <DIR> --analyze-locus 2 --threshold-percentile 80

CLI Options:
    --quick                     Run in rapid smoke-test mode with minimal resolution and evaluations.
    --slab-thickness H          Membrane slab thickness in units of a (default: 0.25 for h/a=0.25).
    --supercell-z Z             Supercell height in units of a (default: 4.0).
    --vary-p2                   Optimize 6d position parameter p2 in addition to radii r1 and r2.
    --substrate-material MAT    Substrate cladding material key from phc_materials (e.g. 'sio2', 'air').
    --substrate-thickness SUB_H Substrate buffer thickness in units of a (default: 2.0).
    --match-reference-modes     Identify target degeneracy by tracking 3 reference modes via modal overlap.
    --reference-data DIR        Path to pre-saved reference dataset directory in 'saved_data/<name>'.
    --reference-run DIR         Path to previous unperturbed simulation directory.
    --reference-bands B1 B2 B3  1-based band indices of 3 reference modes at Gamma (default: 9 10 11).
    --reference-r1 R1           Nominal r1 for unperturbed reference cell.
    --reference-r2 R2           Nominal r2 for unperturbed reference cell.
    --reference-p2 P2           Nominal p2 for unperturbed reference cell (default: 0.25).
    --polarization POL          Polarization mode: 'te_like', 'tm_like', or 'all' (default: 'all' if substrate).
    --target-irreps IRREPS      Target irreducible representations at Gamma (default: A_2 E_1 E_1).
    --enforce-irreps / --no-enforce-irreps Enforce that tracked modes match target point-group irreps (default: True).
    --irrep-occurrences OCC     Occurrence counts above min_band (default: 1 4 4).
    --target-wavelength NM      Target physical wavelength in nm (default: 436.0).
    --target-thickness NM       Target physical slab thickness in nm (default: 100.0).
    --r1-bounds MIN MAX         Search bounds for primary hole radius r1 in units of a (default: 0.15 0.25).
    --r2-bounds MIN MAX         Search bounds for satellite hole radius r2 in units of a (default: 0.05 0.10).
    --resolution RES            In-plane MPB computational mesh resolution per pitch a (default: 18, quick: 12).
    --resolution-z RESZ         Vertical MPB mesh resolution for 3D slabs (default: 16, quick: 6).
    --num-bands N               Number of eigenbands to compute at Gamma (default: 15 / 28 on substrate).
    --initial-points N          Number of initial quasi-random exploration points (default: 100, quick: 2).
    --max-iterations N          Number of Bayesian optimization active learning generations (default: 0, quick: 1).
    --workers W                 Number of parallel worker processes (default: 20, quick: 1).
    --batch-size B              Candidate points evaluated per generation (default: matches --workers).
    --output-dir PATH           Custom output directory override (default: auto-resolved by phc_hydra).
    --no-progress               Disable the real-time tqdm progress bar.
    --find-loci N               Number of degeneracy loci to detect on surrogate map: int or 'auto' (default: 1).
    --analyze-locus [IDX ...]   Refine continuous degeneracy locus curve(s) (defaults to [1] if passed without args).
    --locus-mode MODE           Mode for degeneracy locus refinement ('cartesian', 'polar', or 'auto').
    --force-open / --no-force-open Force degeneracy locus manifold to be an open curve (default: True).
    --overlap-mode MODE         Spatial overlap formulation ('midplane' [default], 'slab', or 'full').
    --max-refine-steps N         Maximum number of refinement steps for the degeneracy locus (default: 5).
    --tracking-strategy STRATEGY Target mode selection algorithm ('cluster', 'bipartite', or 'greedy', default: 'cluster').
    --kpath-type TYPE           K-path trajectory for band diagram plotting ('gamma_centered' or 'standard', default: 'gamma_centered').
    --k-max KMAX                Maximum wavevector radius |k|/(2π) for Gamma-centered k-path (default: 0.1).
    --lam-min NM                Minimum wavelength limit in nm for physical wavelength band diagrams (default: 420.0 nm).
    --lam-max NM                Maximum wavelength limit in nm for physical wavelength band diagrams (default: 450.0 nm).
    --from-run DIR              Path to existing run directory to analyze without re-running optimization.
    --threshold-percentile P    Cutoff percentile for surrogate locus extraction (default: 85.0).
    --save-match-reference NAME Reference name to save locus match point dataset to 'saved_data/<name>'.
"""

import argparse
import json
import select
import shutil
import sys
import warnings
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

# Ensure repository root is in sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import matplotlib

matplotlib.use("Agg")
import gdsfactory as gf
import meep as mp
import numpy as np
from phc_layout.components.unit_cell import phc_wyckoff_unit_cell
from phc_layout.lattice import HexagonalLattice
from phc_mpb import (
    create_lattice,
    create_mode_solver,
    extract_eigenmode_fields,
    gds_to_mpb_geometry,
    run_band_solver,
)
from phc_optimization import (
    BayesianOptimizer,
    ModalOverlapDegeneracyObjective,
    OptimizationRecord,
    OptimizationResult,
)
from phc_utils import export_gds, silence_c_stdout

# Suppress MPB solver chatter and repetitive Sobol balance warnings
mp.verbosity(0)
warnings.filterwarnings("ignore", category=UserWarning, module="skopt")
warnings.filterwarnings("ignore", message=".*balance properties of Sobol.*")


def make_c6v_2b_6d_unit_cell(
    r1: float = 0.15,
    r2: float = 0.10,
    p2: float = 0.25,
    pitch: float = 1.0,
    **kwargs: Any,
) -> gf.Component:
    """Generates a C6v hexagonal lattice unit cell with Wyckoff 2b and 6d holes.

    Args:
        r1: Radius of primary holes at Wyckoff position 2b in micrometers.
        r2: Radius of satellite holes at Wyckoff position 6d in micrometers.
        p2: Coordinate parameter for Wyckoff position 6d.
        pitch: Lattice pitch a in micrometers.
        **kwargs: Extra unused keyword arguments passed by generic runners.

    Returns:
        gf.Component containing the placed holes on the etch layer.
    """
    return phc_wyckoff_unit_cell(
        pitch=pitch,
        point_group="C6v",
        features=[("2b", r1), ("6d", r2, p2)],
    )


def solve_unperturbed_reference_modes(
    r1: float = 0.20,
    r2: float = 0.07,
    p2: float = 0.25,
    pitch: float = 1.0,
    slab_thickness: float = 0.25,
    supercell_z: float = 4.0,
    matrix_material: str = "hBN",
    cladding_material: str = "air",
    resolution: int = 18,
    resolution_z: int = 16,
    num_bands: int = 15,
    target_bands: Sequence[int] = (9, 10, 11),
    overlap_mode: Literal["midplane", "slab", "full"] = "midplane",
    quick: bool = False,
    verbose: bool = True,
) -> tuple[
    dict[int, tuple[np.ndarray, np.ndarray | None]], dict[int, float], list[int]
]:
    """Solves the unperturbed symmetric air-clad slab at Γ to extract 3 reference eigenfields.

    Args:
        r1: Radius of primary holes at Wyckoff position 2b in micrometers.
        r2: Radius of satellite holes at Wyckoff position 6d in micrometers.
        p2: Coordinate parameter for Wyckoff position 6d.
        pitch: Lattice pitch a in micrometers.
        slab_thickness: Normalized slab thickness in units of pitch a.
        supercell_z: Supercell height in units of pitch a.
        slab_material: Slab core material key (default: "hBN").
        cladding_material: Upper and lower cladding material key (default: "air").
        resolution: In-plane MPB computational grid resolution.
        resolution_z: Vertical MPB computational grid resolution.
        num_bands: Number of eigenbands computed at Γ.
        target_bands: Sequence of 3 target reference band indices to extract.
        overlap_mode: Spatial domain extraction formulation ('midplane', 'slab', or 'full').
        quick: If True, uses low resolution and lower band indices for smoke tests.
        verbose: If True, logs progress and extracted eigenfrequencies.

    Returns:
        Tuple of (ref_fields_dict, ref_freqs_dict, resolved_target_bands).
    """
    if verbose:
        print(
            "\n--- Solving Unperturbed Reference System (Air Cladding, Symmetric) at Γ ---"
        )

    ref_comp = make_c6v_2b_6d_unit_cell(r1=r1, r2=r2, p2=p2, pitch=pitch)
    hex_lat = HexagonalLattice(a=pitch)
    lattice = create_lattice(
        lattice_type=hex_lat,
        pitch=pitch,
        dimension="3D_slab",
        supercell_z=supercell_z,
    )

    norm_h = slab_thickness / max(pitch, 1e-12)
    geometry = gds_to_mpb_geometry(
        gds_source=ref_comp,
        pitch=pitch,
        dimension="3D_slab",
        slab_thickness=slab_thickness,
        slab_material=matrix_material,
        substrate_material=None,
        etch_material=cladding_material,
        geometry_lattice=lattice,
    )

    k_points = [mp.Vector3(0.0, 0.0, 0.0)]
    res = (resolution, resolution, resolution_z)

    with silence_c_stdout():
        ms_ref = create_mode_solver(
            geometry_lattice=lattice,
            geometry=geometry,
            k_points=k_points,
            default_material=cladding_material,
            resolution=res,
            num_bands=num_bands,
        )
        res_ref = run_band_solver(
            ms=ms_ref,
            polarization="te_like",
            dimension="3D_slab",
            verbose=False,
        )

    all_freqs = res_ref.get("freqs", {}).get(
        "te_like",
        res_ref.get("freqs", {}).get("all", ms_ref.all_freqs),
    )[0]

    # Resolve target bands
    resolved_bands = [b for b in target_bands if b <= len(all_freqs)]
    if len(resolved_bands) < len(target_bands):
        if quick or len(all_freqs) < max(target_bands):
            resolved_bands = [
                min(len(all_freqs), i + 2) for i in range(len(target_bands))
            ]
            resolved_bands = sorted(set(resolved_bands))
            while len(resolved_bands) < len(target_bands) and len(resolved_bands) < len(
                all_freqs
            ):
                resolved_bands.append(resolved_bands[-1] + 1)
        else:
            resolved_bands = list(target_bands)

    ref_fields: dict[int, tuple[np.ndarray, np.ndarray | None]] = {}
    ref_freqs: dict[int, float] = {}

    for b in resolved_bands:
        f = float(all_freqs[b - 1])
        ref_freqs[b] = f
        ref_fields[b] = extract_eigenmode_fields(
            ms_ref,
            band=b,
            field="electric_displacement",
            slab_thickness=norm_h,
            z_center=0.0,
            overlap_mode=overlap_mode,
        )
        if verbose:
            lam_nm = (pitch / f * 1000.0) if f > 0 else 0.0
            print(f"  Reference Band {b:2d}: ω~ = {f:.5f} (λ = {lam_nm:.1f} nm)")

    return ref_fields, ref_freqs, resolved_bands


def timed_input(prompt: str, timeout: float = 60.0) -> str | None:
    """Prompts for input with a timeout in seconds.

    If stdin is not a TTY or timeout expires without input, returns None.

    Args:
        prompt: Display string prompting user for input.
        timeout: Duration in seconds to wait before timing out (default: 60.0).

    Returns:
        Stripped string input if received before timeout, otherwise None.
    """
    if not sys.stdin.isatty():
        return None
    print(prompt, end="", flush=True)
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    if rlist:
        return sys.stdin.readline().strip()
    print(
        f"\n[Timer] No input received within {int(timeout)}s. Skipping reference save."
    )
    return None


def extract_optimal_point_fields(
    params: dict[str, Any],
    pitch: float = 1.0,
    slab_thickness: float = 0.25,
    supercell_z: float = 4.0,
    resolution: int = 18,
    resolution_z: int = 16,
    matrix_material: str = "hBN",
    cladding_material: str = "air",
    substrate_material: str | None = None,
    substrate_thickness: float | None = None,
    polarization: str = "all",
    target_bands: Sequence[int] = (2, 3, 4),
    overlap_mode: Literal["midplane", "slab", "full"] = "midplane",
    verbose: bool = True,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray | None]], dict[int, float]]:
    """Solves the optimal unit cell at Γ and extracts spatial eigenmode fields (E and D).

    Args:
        params: Geometric parameters for unit cell (r1, r2, p2, pitch).
        pitch: Unit cell lattice pitch in micrometers (default: 1.0).
        slab_thickness: Normalized slab thickness h in units of pitch a.
        supercell_z: Vertical supercell height in units of pitch a.
        resolution: In-plane MPB computational grid resolution.
        resolution_z: Vertical MPB computational grid resolution.
        matrix_material: Slab core material key.
        cladding_material: Top cladding material key.
        substrate_material: Bottom substrate cladding material key (or None).
        substrate_thickness: Substrate buffer thickness in units of a (or None).
        polarization: MPB polarization mode ('all', 'te_like', etc.).
        target_bands: Sequence of 1-based band indices to extract.
        overlap_mode: Spatial domain extraction formulation ('midplane', 'slab', or 'full').
        verbose: Whether to log progress to stdout.

    Returns:
        Tuple of (fields_dict, freqs_dict) where fields_dict maps band index to (e_field, d_field).
    """
    if verbose:
        print("\n--- Solving Optimal Design at Γ to Extract Modal Fields ---")

    r1_val = float(params.get("r1", 0.2))
    r2_val = float(params.get("r2", 0.08))
    p2_val = float(params.get("p2", 0.25))

    comp = make_c6v_2b_6d_unit_cell(r1=r1_val, r2=r2_val, p2=p2_val, pitch=pitch)
    lattice = create_lattice(
        lattice_type=HexagonalLattice(a=pitch),
        pitch=pitch,
        dimension="3D_slab",
        supercell_z=supercell_z,
    )
    norm_h = slab_thickness / max(pitch, 1e-12)
    geometry = gds_to_mpb_geometry(
        gds_source=comp,
        pitch=pitch,
        dimension="3D_slab",
        slab_thickness=slab_thickness,
        slab_material=matrix_material,
        substrate_material=substrate_material,
        substrate_thickness=float(substrate_thickness)
        if substrate_thickness is not None
        else None,
        etch_material=cladding_material,
        geometry_lattice=lattice,
    )

    k_points = [mp.Vector3(0.0, 0.0, 0.0)]
    res = (resolution, resolution, resolution_z)
    max_b = max(target_bands) if target_bands else 15
    num_b = max(max_b + 4, 15)

    default_mat = (
        substrate_material
        if (substrate_material and substrate_material != "air")
        else cladding_material
    )

    with silence_c_stdout():
        ms = create_mode_solver(
            geometry_lattice=lattice,
            geometry=geometry,
            k_points=k_points,
            default_material=default_mat,
            resolution=res,
            num_bands=num_b,
        )
        res_solver = run_band_solver(
            ms=ms,
            polarization=polarization,
            dimension="3D_slab",
            verbose=False,
        )

    all_f = res_solver.get("freqs", {}).get(
        polarization.lower(),
        res_solver.get("freqs", {}).get("all", ms.all_freqs),
    )[0]

    fields_dict: dict[int, tuple[np.ndarray, np.ndarray | None]] = {}
    freqs_dict: dict[int, float] = {}

    for b in target_bands:
        if b <= len(all_f):
            f_val = float(all_f[b - 1])
            freqs_dict[b] = f_val
            e_field, d_field = extract_eigenmode_fields(
                ms,
                band=b,
                field="electric_displacement",
                slab_thickness=norm_h,
                z_center=0.0,
                overlap_mode=overlap_mode,
            )
            fields_dict[b] = (e_field, d_field)
            if verbose:
                lam_nm = (pitch / f_val * 1000.0) if f_val > 0 else 0.0
                print(
                    f"  Optimal Mode Band {b:2d}: ω~ = {f_val:.5f} (λ = {lam_nm:.1f} nm)"
                )

    return fields_dict, freqs_dict


def save_optimal_field_reference(
    name: str,
    params: dict[str, Any],
    pitch: float = 1.0,
    slab_thickness: float = 0.25,
    supercell_z: float = 4.0,
    resolution: int = 18,
    resolution_z: int = 16,
    matrix_material: str = "hBN",
    cladding_material: str = "air",
    substrate_material: str | None = None,
    substrate_thickness: float | None = None,
    polarization: str = "all",
    tracked_bands: Sequence[int] = (2, 3, 4),
    overlap_mode: Literal["midplane", "slab", "full"] = "midplane",
    base_dir: Path | str = "saved_data",
) -> Path:
    """Saves the optimal point fields, geometry parameters, and GDS layout to saved_data/<name>/.

    Args:
        name: Subdirectory name for the saved reference dataset.
        params: Geometric parameters dictionary (r1, r2, p2, pitch).
        pitch: Unit cell lattice pitch in micrometers (default: 1.0).
        slab_thickness: Normalized slab thickness h in units of pitch a.
        supercell_z: Vertical supercell height in units of pitch a.
        resolution: In-plane MPB computational grid resolution.
        resolution_z: Vertical MPB computational grid resolution.
        matrix_material: Slab core material key.
        cladding_material: Top cladding material key.
        substrate_material: Bottom substrate cladding material key (or None).
        substrate_thickness: Substrate buffer thickness in units of a (or None).
        polarization: MPB polarization mode ('all', 'te_like', etc.).
        tracked_bands: Sequence of 1-based band indices to save.
        base_dir: Base directory where reference folders are stored (default: 'saved_data').

    Returns:
        Path to the created saved_data/<name>/ directory.
    """
    save_dir = Path(base_dir).resolve() / name
    save_dir.mkdir(parents=True, exist_ok=True)

    fields_dict, freqs_dict = extract_optimal_point_fields(
        params=params,
        pitch=pitch,
        slab_thickness=slab_thickness,
        supercell_z=supercell_z,
        resolution=resolution,
        resolution_z=resolution_z,
        matrix_material=matrix_material,
        cladding_material=cladding_material,
        substrate_material=substrate_material,
        substrate_thickness=substrate_thickness,
        polarization=polarization,
        target_bands=tracked_bands,
        overlap_mode=overlap_mode,
        verbose=True,
    )

    npz_data: dict[str, np.ndarray] = {}
    for b, (e_arr, d_arr) in fields_dict.items():
        if e_arr is not None:
            npz_data[f"e_{b}"] = e_arr
        if d_arr is not None:
            npz_data[f"d_{b}"] = d_arr

    fields_file = save_dir / "fields.npz"
    np.savez_compressed(fields_file, **npz_data)

    meta = {
        "name": name,
        "timestamp": datetime.now(UTC).isoformat(),
        "geometry": {
            "r1": float(params.get("r1", 0.0)),
            "r2": float(params.get("r2", 0.0)),
            "p": float(params.get("pitch", pitch)),
            "p2": float(params.get("p2", 0.25)),
            "slab_thickness": float(slab_thickness),
            "supercell_thickness": float(supercell_z),
            "resolution": int(resolution),
            "resolution_z": int(resolution_z),
            "substrate_material": substrate_material,
            "substrate_thickness": float(substrate_thickness)
            if substrate_thickness is not None
            else None,
            "matrix_material": matrix_material,
            "cladding_material": cladding_material,
        },
        "polarization": polarization,
        "bands": list(tracked_bands),
        "overlap_mode": overlap_mode,
        "frequencies": {str(b): f for b, f in freqs_dict.items()},
        "files": {
            "fields_npz": "fields.npz",
            "unit_cell_gds": "unit_cell.gds",
        },
    }
    meta_file = save_dir / "metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    comp = make_c6v_2b_6d_unit_cell(
        r1=float(params.get("r1", 0.2)),
        r2=float(params.get("r2", 0.08)),
        p2=float(params.get("p2", 0.25)),
        pitch=pitch,
    )
    gds_file = save_dir / "unit_cell.gds"
    export_gds(comp, gds_file, overwrite=True)

    print(f"\n[Saved] Reference data successfully saved to '{save_dir}':")
    print(f"  - Fields:   {fields_file}")
    print(f"  - Metadata: {meta_file}")
    print(f"  - Layout:   {gds_file}")
    return save_dir


def save_match_point_reference(
    name: str,
    match_meta: dict[str, Any],
    output_dir: Path | str,
    base_dir: Path | str = "saved_data",
    matrix_material: str = "hBN",
    cladding_material: str = "air",
    substrate_material: str | None = None,
    substrate_thickness: float | None = None,
    supercell_z: float = 4.0,
    resolution: int = 18,
    resolution_z: int = 16,
    polarization: str = "all",
    tracked_bands: Sequence[int] = (9, 10, 11),
    p2: float = 0.25,
    slab_thickness: float = 0.25,
    overlap_mode: Literal["midplane", "slab", "full"] = "midplane",
) -> Path:
    """Saves the locus closest match point dataset, dispersion, fields, and metadata to saved_data/<name>/.

    Args:
        name: Subdirectory name for the saved reference dataset.
        match_meta: Target match dictionary from locus analysis containing coordinates,
            pitch, target wavelength, thickness, and solver results.
        output_dir: Simulation output directory containing locus artifacts.
        base_dir: Base directory where reference folders are stored (default: 'saved_data').
        matrix_material: Slab core material key.
        cladding_material: Top cladding material key.
        substrate_material: Bottom substrate cladding material key (or None).
        substrate_thickness: Substrate buffer thickness in units of a (or None).
        supercell_z: Vertical supercell height in units of pitch a.
        resolution: In-plane MPB computational grid resolution for field extraction.
        resolution_z: Vertical MPB computational grid resolution for field extraction.
        polarization: MPB polarization mode.
        tracked_bands: Sequence of 1-based band indices to extract fields for.
        p2: Coordinate parameter for 6d satellite holes.
        slab_thickness: Normalized slab thickness h in units of pitch a.
        overlap_mode: Spatial domain extraction formulation ('midplane', 'slab', or 'full').

    Returns:
        Path to the created saved_data/<name>/ directory.
    """
    save_dir = Path(base_dir).resolve() / name
    save_dir.mkdir(parents=True, exist_ok=True)

    r1 = float(match_meta.get("r1", 0.2))
    r2 = float(match_meta.get("r2", 0.08))
    pitch_nm = float(match_meta.get("pitch_nm", 1000.0))
    pitch = float(match_meta.get("pitch", pitch_nm / 1000.0))
    th_nm = float(match_meta.get("thickness_nm", 100.0))
    norm_h = (
        float(slab_thickness)
        if slab_thickness is not None
        else (th_nm / max(pitch_nm, 1e-12))
    )

    # 1. Save band dispersion array (copy from locus run if saved or build from solver_results)
    disp_file = save_dir / "dispersion.npz"
    disp_copied = False
    out_p = Path(output_dir)
    for cand_disp in [
        out_p / "locus_match_dispersion.npz",
        out_p / "locus_01" / "locus_01_match_dispersion.npz",
    ]:
        if cand_disp.exists():
            shutil.copy2(cand_disp, disp_file)
            disp_copied = True
            break

    if not disp_copied:
        solver_results = match_meta.get("solver_results")
        if solver_results:
            disp_data: dict[str, Any] = {}
            freqs = solver_results.get("freqs", {})
            if isinstance(freqs, dict):
                for k, v in freqs.items():
                    if v is not None:
                        disp_data[f"freqs_{k}"] = np.asarray(v)
            elif freqs is not None:
                disp_data["freqs"] = np.asarray(freqs)
            if solver_results.get("k_points") is not None:
                disp_data["k_points"] = np.asarray(solver_results["k_points"])
            if solver_results.get("k_labels") is not None:
                disp_data["k_labels"] = np.asarray(solver_results["k_labels"])
            if solver_results.get("k_indices") is not None:
                disp_data["k_indices"] = np.asarray(solver_results["k_indices"])
            if solver_results.get("light_line") is not None:
                disp_data["light_line"] = np.asarray(solver_results["light_line"])
            if solver_results.get("te_fractions") is not None:
                disp_data["te_fractions"] = np.asarray(solver_results["te_fractions"])
            disp_data["pitch"] = np.asarray(pitch)
            disp_data["pitch_nm"] = np.asarray(pitch_nm)
            np.savez_compressed(disp_file, **disp_data)

    # 2. Extract and save modal fields at Gamma
    fields_dict: dict[int, tuple[np.ndarray, np.ndarray | None]] = {}
    freqs_dict: dict[int, float] = {}
    try:
        match_params = {"r1": r1, "r2": r2, "p2": p2, "pitch": pitch}
        fields_dict, freqs_dict = extract_optimal_point_fields(
            params=match_params,
            pitch=pitch,
            slab_thickness=norm_h,
            supercell_z=supercell_z,
            resolution=resolution,
            resolution_z=resolution_z,
            matrix_material=matrix_material,
            cladding_material=cladding_material,
            substrate_material=substrate_material,
            substrate_thickness=substrate_thickness,
            polarization=polarization,
            target_bands=tracked_bands,
            overlap_mode=overlap_mode,
            verbose=True,
        )
        npz_fields: dict[str, np.ndarray] = {}
        for b, (e_arr, d_arr) in fields_dict.items():
            if e_arr is not None:
                npz_fields[f"e_{b}"] = e_arr
            if d_arr is not None:
                npz_fields[f"d_{b}"] = d_arr
        fields_file = save_dir / "fields.npz"
        np.savez_compressed(fields_file, **npz_fields)
    except (RuntimeError, ValueError, KeyError, OSError) as e:
        print(f"  Warning: Could not extract modal fields for match point: {e}")

    # 3. Export layout GDS
    comp = make_c6v_2b_6d_unit_cell(
        r1=r1,
        r2=r2,
        p2=p2,
        pitch=pitch,
    )
    gds_file = save_dir / "unit_cell.gds"
    export_gds(comp, gds_file, overwrite=True)

    # 4. Copy locus match figures if present
    copied_files: dict[str, str] = {}
    out_p = Path(output_dir)
    for cand_dir in [out_p / "locus_01", out_p]:
        if not cand_dir.exists():
            continue
        for f_name in [
            "locus_01_match_band_structure.png",
            "locus_01_match_band_structure_wavelength.png",
            "locus_01_match_epsilon.png",
            "locus_match_band_structure.png",
            "locus_match_band_structure_wavelength.png",
            "locus_match_epsilon.png",
            "locus_dirac_frequency.png",
            "locus_wavelength.png",
            "locus_profile.png",
        ]:
            src_f = cand_dir / f_name
            if src_f.exists() and not (save_dir / src_f.name).exists():
                shutil.copy2(src_f, save_dir / src_f.name)
                copied_files[src_f.stem] = src_f.name

    # 5. Save metadata.json
    meta = {
        "name": name,
        "timestamp": datetime.now(UTC).isoformat(),
        "type": "locus_match_point",
        "geometry": {
            "r1": r1,
            "r2": r2,
            "p": pitch,
            "p2": p2,
            "pitch_nm": pitch_nm,
            "slab_thickness": norm_h,
            "thickness_nm": th_nm,
            "supercell_thickness": float(supercell_z),
            "resolution": int(resolution),
            "resolution_z": int(resolution_z),
            "substrate_material": substrate_material,
            "substrate_thickness": float(substrate_thickness)
            if substrate_thickness is not None
            else None,
            "matrix_material": matrix_material,
            "cladding_material": cladding_material,
        },
        "locus_characterization": {
            "target_wavelength_nm": match_meta.get("target_wavelength_nm"),
            "target_thickness_nm": match_meta.get("target_thickness_nm"),
            "residual_thickness_error_nm": match_meta.get(
                "residual_thickness_error_nm"
            ),
            "omega_d": match_meta.get("omega_d"),
            "v_g_delta_omega": match_meta.get("v_g_delta_omega"),
            "residual_gap": match_meta.get("residual_gap"),
        },
        "polarization": polarization,
        "bands": list(tracked_bands),
        "frequencies": {str(b): f for b, f in freqs_dict.items()},
        "files": {
            "dispersion_npz": "dispersion.npz"
            if (save_dir / "dispersion.npz").exists()
            else None,
            "fields_npz": "fields.npz" if (save_dir / "fields.npz").exists() else None,
            "unit_cell_gds": "unit_cell.gds",
            **copied_files,
        },
    }
    meta_file = save_dir / "metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[Saved] Locus match point data successfully saved to '{save_dir}':")
    if (save_dir / "dispersion.npz").exists():
        print(f"  - Dispersion: {save_dir / 'dispersion.npz'}")
    if (save_dir / "fields.npz").exists():
        print(f"  - Fields:     {save_dir / 'fields.npz'}")
    print(f"  - Metadata:   {meta_file}")
    print(f"  - Layout:     {gds_file}")
    return save_dir


def load_saved_reference_data(
    reference_data_path: Path | str,
) -> tuple[
    dict[int, tuple[np.ndarray, np.ndarray | None]],
    dict[int, float],
    list[int],
    dict[str, Any],
]:
    """Loads pre-saved reference modal fields, frequencies, and geometry metadata.

    Args:
        reference_data_path: Path to directory (or file) containing fields.npz and metadata.json.

    Returns:
        Tuple of (fields_dict, freqs_dict, bands_list, metadata_dict).

    Raises:
        FileNotFoundError: If fields.npz or metadata.json is missing in reference_data_path.
    """
    ref_p = Path(reference_data_path).resolve()
    if ref_p.is_file():
        ref_dir = ref_p.parent
    else:
        ref_dir = ref_p

    fields_file = ref_dir / "fields.npz"
    meta_file = ref_dir / "metadata.json"

    if not fields_file.is_file() or not meta_file.is_file():
        raise FileNotFoundError(
            f"Reference directory '{ref_dir}' must contain both 'fields.npz' and 'metadata.json'."
        )

    with open(meta_file, encoding="utf-8") as f:
        meta = json.load(f)

    bands = [int(b) for b in meta.get("bands", [])]
    freqs = {int(b): float(f) for b, f in meta.get("frequencies", {}).items()}

    npz = np.load(fields_file)
    fields: dict[int, tuple[np.ndarray, np.ndarray | None]] = {}
    for b in bands:
        e_key = f"e_{b}"
        d_key = f"d_{b}"
        e_arr = npz.get(e_key, None)
        d_arr = npz.get(d_key, None)
        fields[b] = (e_arr, d_arr)

    return fields, freqs, bands, meta


def load_run_records(
    run_dir: Path | str,
    enforce_irreps: bool = False,
    target_irreps: Sequence[str] | None = None,
) -> tuple[list[OptimizationRecord], dict[str, Any]]:
    """Loads optimization evaluation records and simulation metadata from an existing run directory.

    When enforce_irreps is True and target_irreps is provided, audits each evaluation's
    tracked modes against the point-group representation requirements. If an evaluation
    tracked alien or mismatched modes (e.g. accidental E2/B1 crossings), it resolves the
    correct target multiplet from the evaluation's symmetry data or penalizes the evaluation.

    Args:
        run_dir: Path to the existing run directory containing bo_evaluations.json.
        enforce_irreps: If True, audits and reconciles evaluation irreps against target_irreps.
        target_irreps: Target point-group representations forming the multiplet (e.g. ['A_2', 'E_1', 'E_1']).

    Returns:
        Tuple of (records_list, summary_metadata_dict).

    Raises:
        FileNotFoundError: If bo_evaluations.json is missing in run_dir.
    """
    p = Path(run_dir).resolve()
    eval_file = p / "bo_evaluations.json"
    sim_file = p / "simulation_results.json"
    if not eval_file.is_file():
        raise FileNotFoundError(
            f"Run directory '{p}' does not contain 'bo_evaluations.json'."
        )

    with open(eval_file, encoding="utf-8") as f:
        data = json.load(f)

    sim_meta: dict[str, Any] = {}
    if sim_file.is_file():
        with open(sim_file, encoding="utf-8") as f:
            sim_meta = json.load(f)

    import itertools
    from collections import Counter

    from phc_mpb.classification import resolve_target_irrep_counts

    target_counts = (
        resolve_target_irrep_counts(target_irreps, len(target_irreps))
        if enforce_irreps and target_irreps
        else None
    )

    records: list[OptimizationRecord] = []
    evals = data.get("evaluations", []) if isinstance(data, dict) else data
    corrected_count = 0

    for ev in evals:
        cost = float(ev.get("cost", 1.0))
        fom = float(ev.get("fom", 1.0))
        status = str(ev.get("status", ""))
        metadata = dict(ev)

        if target_counts:
            syms = ev.get(
                "symmetries",
                ev.get("metadata", {}).get("solver_data", {}).get("symmetries", []),
            )
            if syms and target_irreps:
                from phc_mpb.symmetry import resolve_multiplet_symmetries

                syms = resolve_multiplet_symmetries(
                    syms, target_irreps=list(target_irreps)
                )

            tb = ev.get("target_bands", ev.get("metadata", {}).get("target_bands", []))
            band_irreps = {
                s["band"]: s["irrep"] for s in syms if "band" in s and "irrep" in s
            }
            band_freqs = {
                s["band"]: s["freq"] for s in syms if "band" in s and "freq" in s
            }

            current_irreps = Counter([band_irreps.get(b) for b in tb])
            if band_irreps and current_irreps != target_counts:
                # Find valid cluster matching target_counts
                k_target = sum(target_counts.values())
                valid_clusters = []
                bands = sorted(band_irreps.keys())
                for i in range(len(bands)):
                    for combo in itertools.combinations(bands[i : i + 5], k_target):
                        if (
                            combo[-1] - combo[0] <= 4
                            and Counter([band_irreps[b] for b in combo])
                            == target_counts
                        ):
                            f_vals = [band_freqs[b] for b in combo]
                            f_mid = sum(f_vals) / max(len(f_vals), 1)
                            if f_mid > 0.5:
                                c = (max(f_vals) - min(f_vals)) / max(f_mid, 1e-12)
                                valid_clusters.append(
                                    (combo, f_mid, c, 1.0 / max(c, 1e-12))
                                )

                if valid_clusters:
                    valid_clusters.sort(key=lambda x: x[2])
                    best = valid_clusters[0]
                    cost = float(best[2])
                    fom = float(best[3])
                    status = f"RECONCILED (Bands {list(best[0])} matching {list(target_irreps or [])})"
                    metadata["target_bands"] = list(best[0])
                    metadata["cost"] = cost
                    metadata["fom"] = fom
                    metadata["irrep_match"] = True
                else:
                    cost = 1.0
                    fom = 1.0
                    status = f"PENALIZED (Irrep mismatch: no valid {list(target_irreps or [])} cluster)"
                    metadata["cost"] = cost
                    metadata["fom"] = fom
                    metadata["irrep_match"] = False
                corrected_count += 1

        rec = OptimizationRecord(
            eval_index=ev.get("eval_index", len(records) + 1),
            generation=ev.get("generation", 0),
            params=ev.get("params", {}),
            cost=cost,
            fom=fom,
            status=status,
            connectivity=str(ev.get("connectivity", "PASSED")),
            timing=ev.get("timing", {}),
            metadata=metadata,
        )
        records.append(rec)

    if corrected_count > 0:
        print(
            f"\n--- Enforce Irreps: Reconciled {corrected_count}/{len(records)} evaluations with mismatched irreps ---"
        )

    return records, sim_meta


def run_c6v_optimization_pipeline(
    quick: bool = False,
    slab_thickness: float | None = 0.25,
    supercell_z: float = 4.0,
    p2: float = 0.25,
    vary_p2: bool = False,
    polarization: str | None = None,
    resolution: int | None = None,
    resolution_z: int | None = None,
    num_bands: int | None = None,
    initial_points: int | None = None,
    max_iterations: int | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    matrix_material: str = "hBN",
    cladding_material: str = "air",
    substrate_material: str | None = None,
    substrate_thickness: float = 2.0,
    match_reference_modes: bool = False,
    reference_data: str | Path | None = None,
    reference_run: str | Path | None = None,
    reference_bands: Sequence[int] = (9, 10, 11),
    reference_r1: float | None = None,
    reference_r2: float | None = None,
    reference_p2: float = 0.25,
    output_dir: Path | str | None = None,
    show_progress: bool = True,
    find_loci: int | str = 1,
    analyze_locus: bool | Sequence[int | str] | int | str | None = None,
    threshold_percentile: float = 85.0,
    from_run: str | Path | None = None,
    mode_indices: tuple[int, int, int] = (9, 10, 11),
    target_irreps: Sequence[str] = ("A_2", "E_1", "E_1"),
    irrep_occurrences: Sequence[int] = (1, 4, 4),
    enforce_irreps: bool = True,
    target_wavelength_nm: float = 436.0,
    target_thickness_nm: float = 100.0,
    r1_bounds: tuple[float, float] = (0.15, 0.25),
    r2_bounds: tuple[float, float] = (0.05, 0.10),
    locus_mode: str = "cartesian",
    force_open: bool = True,
    overlap_mode: Literal["midplane", "slab", "full"] = "midplane",
    interpolate: bool = True,
    max_refine_steps: int = 5,
    tracking_strategy: str = "cluster",
    kpath_type: str = "gamma_centered",
    k_max: float = 0.1,
    lam_min_nm: float = 420.0,
    lam_max_nm: float = 450.0,
) -> dict[str, Any]:
    """Runs the Bayesian Optimization pipeline for the C6v 2b-6d unit cell.

    Args:
        quick: If True, executes a rapid low-resolution smoke test (<5 s).
        slab_thickness: Normalized membrane thickness h/a (default: 0.25, None for 2D PhC).
        supercell_z: Vertical supercell height in units of lattice constant a (default: 4.0).
        p2: Coordinate parameter for Wyckoff position 6d satellite holes (default: 0.25).
        vary_p2: If True, includes 6d radial position parameter p2 in optimization bounds.
        polarization: Target mode polarization ('te_like', 'tm_like', 'all', or None for auto).
        resolution: In-plane MPB computational mesh resolution per pitch a.
        resolution_z: Vertical MPB computational mesh resolution along z for slabs.
        num_bands: Number of eigenbands computed at Gamma.
        initial_points: Number of initial quasi-random exploration points.
        max_iterations: Number of active learning generations.
        batch_size: Candidate points evaluated per generation.
        num_workers: Number of parallel worker processes.
        matrix_material: Slab/matrix dielectric material key (default: "hBN").
        cladding_material: Upper cladding material key (default: "air").
        substrate_material: Substrate cladding material key (e.g. "sio2", default: None / "air").
        substrate_thickness: Substrate buffer thickness in units of pitch a (default: 2.0).
        match_reference_modes: If True, tracks 3 unperturbed reference modes via spatial overlap.
        reference_data: Path to pre-saved reference dataset directory in 'saved_data/<name>'.
        reference_run: Optional directory of previous unperturbed run to load reference parameters.
        reference_bands: Sequence of 3 reference band indices at Gamma (default: (9, 10, 11)).
        reference_r1: Optional explicit r1 for unperturbed reference cell.
        reference_r2: Optional explicit r2 for unperturbed reference cell.
        reference_p2: Coordinate parameter p2 for unperturbed reference cell (default: 0.25).
        output_dir: Custom output directory or None to auto-resolve via phc_hydra.
        show_progress: Whether to display a real-time progress bar.
        find_loci: Number of degeneracy loci to detect on surrogate map: int or 'auto' (default: 1).
        analyze_locus: Locus ID(s) to refine (e.g. 1, [1, 2], True for [1], or None to skip).
        mode_indices: 1-based indices of target degenerate bands at Gamma (default: [9, 10, 11]).
        target_irreps: Target irreducible representations at Gamma (default: ('A_2', 'E_1', 'E_1')).
        irrep_occurrences: Occurrence index per target irrep above min_band (default: (1, 4, 4)).
        target_wavelength_nm: Target physical wavelength in nanometers (default: 436.0).
        target_thickness_nm: Target physical slab thickness in nanometers (default: 100.0).
        r1_bounds: Search range for primary hole radius r1 in units of a (default: (0.15, 0.25)).
        r2_bounds: Search range for satellite hole radius r2 in units of a (default: (0.05, 0.10)).
        max_refine_steps: Maximum number of refinement steps for the degeneracy locus (default: 5).
        locus_mode: Mode for degeneracy locus refinement ('cartesian' or 'polar' or 'auto').
        force_open: If True (default), forces the degeneracy locus curve to remain an open path.
        overlap_mode: Spatial domain extraction formulation ('midplane' [default], 'slab', or 'full').
        interpolate: If True (default), enables spatial field interpolation across mismatched mesh resolutions.
        tracking_strategy: Mode tracking algorithm ('cluster', 'bipartite', or 'greedy'). Default: 'cluster'.
        kpath_type: K-path trajectory for band diagram: 'gamma_centered' (default) or 'standard'.
        k_max: Maximum Cartesian wavevector radius |k|/(2π) for Gamma-centered k-path (default: 0.1).
        lam_min_nm: Minimum physical wavelength limit in nm for band diagrams (default: 420.0 nm).
        lam_max_nm: Maximum physical wavelength limit in nm for band diagrams (default: 450.0 nm).

    Returns:
        Dictionary containing best parameters, best FOM, residual cost, optimal loci,
        physical scaling analysis, and output directory.
    """
    is_3d = slab_thickness is not None
    dim = "3D_slab" if is_3d else "2D"
    has_substrate = (
        is_3d
        and substrate_material is not None
        and str(substrate_material).strip().lower() not in ("none", "air", "")
    )
    sub_mat_key = str(substrate_material).strip() if has_substrate else None

    # Map polarization key for 2D vs 3D slab
    if polarization is not None:
        pol_key = polarization.lower()
    elif has_substrate or match_reference_modes:
        # Asymmetric substrate slabs lack horizontal parity; solve all modes
        pol_key = "all"
    elif is_3d:
        pol_key = "te_like"
    else:
        pol_key = "te"

    if quick:
        default_resolution = 12
        default_resolution_z = 6
        default_num_bands = 8 if has_substrate else 6
        default_initial_points = 2
        default_max_iterations = 1
        default_batch_size = 1
        default_num_workers = 1
        bypass_irrep = True
    else:
        default_resolution = 18
        default_resolution_z = 16
        default_num_bands = 28 if has_substrate else 15
        default_initial_points = 100
        default_max_iterations = 0
        default_num_workers = 20
        default_batch_size = None
        bypass_irrep = False

    res_val = resolution if resolution is not None else default_resolution
    res_z_val = resolution_z if resolution_z is not None else default_resolution_z
    bands_val = num_bands if num_bands is not None else default_num_bands
    init_pts = initial_points if initial_points is not None else default_initial_points
    max_iters = max_iterations if max_iterations is not None else default_max_iterations
    n_workers = num_workers if num_workers is not None else default_num_workers
    b_size = batch_size if batch_size is not None else (default_batch_size or n_workers)

    # Configure parameter search bounds
    if vary_p2:
        search_params = {
            "r1": r1_bounds,
            "r2": r2_bounds,
            "p2": (0.15, 0.30),
        }
        fixed_params: dict[str, Any] = {"pitch": 1.0}
    else:
        search_params = {
            "r1": r1_bounds,
            "r2": r2_bounds,
        }
        fixed_params = {"pitch": 1.0, "p2": float(p2)}

    if is_3d:
        fixed_params["slab_thickness"] = float(slab_thickness)  # type: ignore[arg-type]
        fixed_params["supercell_z"] = float(supercell_z)
        if has_substrate and sub_mat_key:
            fixed_params["substrate_material"] = sub_mat_key
            fixed_params["substrate_thickness"] = float(substrate_thickness)

    # -------------------------------------------------------------
    # Objective Configuration: Irrep-based vs Modal Overlap Tracking
    # -------------------------------------------------------------
    actual_ref_bands: list[int] = []
    if match_reference_modes:
        if reference_data is not None:
            ref_fields, ref_freqs, actual_ref_bands, _ref_meta = (
                load_saved_reference_data(reference_data)
            )
            if show_progress:
                print(
                    f"\n--- Loaded Pre-Saved Reference Modal Fields from '{reference_data}' ---"
                )
                for b in actual_ref_bands:
                    f_val = ref_freqs.get(b, 0.0)
                    lam_nm = (1.0 / f_val * 1000.0) if f_val > 0 else 0.0
                    print(
                        f"  Reference Band {b:2d}: ω~ = {f_val:.5f} (λ = {lam_nm:.1f} nm)"
                    )
        else:
            ref_r1_val = reference_r1
            ref_r2_val = reference_r2
            if reference_run is not None:
                ref_p = Path(reference_run).resolve()
                if ref_p.is_file():
                    ref_p = ref_p.parent
                sim_json = ref_p / "simulation_results.json"
                loci_json = ref_p / "optimal_loci.json"
                if sim_json.is_file():
                    try:
                        with open(sim_json, encoding="utf-8") as f:
                            s_data = json.load(f)
                            bp = s_data.get("best_params", {})
                            if ref_r1_val is None and "r1" in bp:
                                ref_r1_val = float(bp["r1"])
                            if ref_r2_val is None and "r2" in bp:
                                ref_r2_val = float(bp["r2"])
                    except (json.JSONDecodeError, OSError):
                        pass
                elif loci_json.is_file():
                    try:
                        with open(loci_json, encoding="utf-8") as f:
                            l_data = json.load(f)
                            if (
                                isinstance(l_data, list)
                                and l_data
                                and "x1" in l_data[0]
                            ):
                                mid_idx = len(l_data[0]["x1"]) // 2
                                if ref_r1_val is None:
                                    ref_r1_val = float(l_data[0]["x1"][mid_idx])
                                if ref_r2_val is None:
                                    ref_r2_val = float(l_data[0]["x2"][mid_idx])
                    except (json.JSONDecodeError, OSError):
                        pass

            if ref_r1_val is None:
                ref_r1_val = float((r1_bounds[0] + r1_bounds[1]) / 2.0)
            if ref_r2_val is None:
                ref_r2_val = float((r2_bounds[0] + r2_bounds[1]) / 2.0)

            resolved_ref_bands = list(reference_bands)
            if quick or bands_val < max(resolved_ref_bands):
                resolved_ref_bands = [2, 3, 4]

            ref_fields, ref_freqs, actual_ref_bands = solve_unperturbed_reference_modes(
                r1=ref_r1_val,
                r2=ref_r2_val,
                p2=float(reference_p2),
                pitch=1.0,
                slab_thickness=float(slab_thickness)
                if slab_thickness is not None
                else 0.25,
                supercell_z=float(supercell_z),
                matrix_material=matrix_material,
                cladding_material=cladding_material,
                resolution=res_val,
                resolution_z=res_z_val,
                num_bands=max(15, max(resolved_ref_bands) + 4) if not quick else 6,
                target_bands=resolved_ref_bands,
                overlap_mode=overlap_mode,
                quick=quick,
                verbose=show_progress,
            )

        obj_instance = ModalOverlapDegeneracyObjective(
            ref_fields=ref_fields,
            ref_frequencies=ref_freqs,
            ref_bands=actual_ref_bands,
            target_band_candidates=range(1, bands_val + 1),
            polarization=pol_key,
            field="electric_displacement",
            slab_thickness=float(slab_thickness)
            if slab_thickness is not None
            else 0.25,
            pitch=1.0,
            overlap_mode=overlap_mode,
            interpolate=interpolate,
            min_overlap_threshold=0.25 if not quick else 0.05,
            symmetry_group="C6v",
            tracking_strategy=tracking_strategy,
            target_irreps=list(target_irreps),
            enforce_irreps=enforce_irreps,
        )
        obj_arg: Any = obj_instance
        obj_kwargs: dict[str, Any] | None = None
    else:
        resolved_indices = list(mode_indices)
        if quick or bands_val < max(resolved_indices):
            resolved_indices = [2, 3, 4]
            bypass_irrep = True

        obj_arg = "dirac_degeneracy"
        obj_kwargs = {
            "symmetry_group": "C6v",
            "polarization": pol_key,
            "target_irreps": list(target_irreps),
            "irrep_occurrences": list(irrep_occurrences),
            "bypass_irrep_identification": bypass_irrep,
            "mode_indices": resolved_indices,
            "min_band": 2,
            "degeneracy_tol": 0.005,
        }

    sub_tag = f"_{sub_mat_key.lower()}" if has_substrate and sub_mat_key else ""
    geo_name = f"c6v_2b_6d_{dim.lower()}{sub_tag}"

    loaded_records = None
    if from_run is not None:
        run_p = Path(from_run).resolve()
        loaded_records, _sim_meta = load_run_records(
            run_p,
            enforce_irreps=enforce_irreps,
            target_irreps=list(target_irreps),
        )
        print(
            f"\n--- Loaded {len(loaded_records)} evaluations from existing run '{run_p.name}' ---"
        )
        if output_dir is None:
            output_dir = run_p

    opt = BayesianOptimizer(
        cell_factory=make_c6v_2b_6d_unit_cell,
        parameters=search_params,
        fixed_parameters=fixed_params,
        objective=obj_arg,
        objective_kwargs=obj_kwargs,
        batch_size=b_size,
        num_workers=n_workers,
        strategy="cl_min",
        lattice_type=HexagonalLattice(a=1.0),
        pitch=1.0,
        dimension=dim,
        resolution=(res_val, res_val, res_z_val) if is_3d else res_val,
        num_bands=bands_val,
        matrix_material=matrix_material,
        background_material=cladding_material,
        substrate_material=sub_mat_key,
        substrate_thickness=float(substrate_thickness) if has_substrate else None,
        enforce_connectivity=True,
        epsilon_threshold=1.1,
        min_neck_width_px=1,
        initial_points=init_pts,
        max_iterations=max_iters,
        output_dir=output_dir,
        geometry_name=geo_name,
        random_state=42,
        show_progress=show_progress,
        find_loci=find_loci,
    )

    if loaded_records is not None:
        opt.records = loaded_records
        opt.output_dir = Path(output_dir).resolve()
        valid_recs = [r for r in loaded_records if r.connectivity != "FAILED"]
        best_rec = (
            max(valid_recs, key=lambda r: r.fom) if valid_recs else loaded_records[0]
        )
        result = OptimizationResult(
            best_params=best_rec.params,
            best_fom=best_rec.fom,
            best_cost=best_rec.cost,
            records=loaded_records,
            gp_model=None,
            output_dir=opt.output_dir,
        )
        if len(opt.param_names) == 2:
            from phc_optimization.plotting import plot_bo_surrogate_map

            plot_bo_surrogate_map(
                optimizer=opt.optimizer,
                records=loaded_records,
                param_names=opt.param_names,
                output_path=opt.output_dir / "bo_surrogate_map.png",
                max_loci=find_loci,
            )
    else:
        result = opt.run(show_progress=show_progress)
        if not quick:
            opt.run_best(
                plot_eps=True,
                plot_bands=True,
                save_plots=True,
                best_params=None,
                rectify=True,
                periods=3,
                grid_resolution=32,
                k_density=20,
                num_workers=n_workers,
                kpath_type=kpath_type,
                k_max=k_max,
                lam_min=lam_min_nm / 1000.0,
                lam_max=lam_max_nm / 1000.0,
            )

    locus_results = []
    if analyze_locus:
        refine_ids: int | str | list[int]
        if analyze_locus is True:
            refine_ids = [1]
        elif isinstance(analyze_locus, str) and analyze_locus.lower() in (
            "all",
            "auto",
        ):
            refine_ids = "all"
        elif isinstance(analyze_locus, int):
            refine_ids = [analyze_locus]
        elif isinstance(analyze_locus, (list, tuple, set)) and any(
            str(x).lower() in ("all", "auto") for x in analyze_locus
        ):
            refine_ids = "all"
        else:
            refine_ids = [int(x) for x in analyze_locus]
        locus_results = opt.analyze_locus(
            refine_loci=refine_ids,
            max_loci=find_loci,
            threshold_percentile=threshold_percentile,
            delta_k=0.01,
            exclude_unrefined=False,
            max_residual_gap=1e-4,
            max_refine_steps=max_refine_steps,
            mode=locus_mode,
            force_open=force_open,
            num_workers=n_workers,
            target_wavelength_nm=target_wavelength_nm,
            target_thickness_nm=target_thickness_nm,
            sample_points=20 if not quick else 3,
            plot_match_bands=not quick,
            k_density_match=20 if not quick else 6,
            kpath_type=kpath_type,
            k_max=k_max,
            lam_min=lam_min_nm / 1000.0,
            lam_max=lam_max_nm / 1000.0,
        )

    loci_file = result.output_dir / "optimal_loci.json"
    loci_data = []
    if loci_file.is_file():
        with open(loci_file, encoding="utf-8") as f:
            loci_data = json.load(f)

    # -------------------------------------------------------------
    # Physical Analysis: Map Normalized Degeneracies to Physical Scale
    # -------------------------------------------------------------
    physical_analysis: dict[str, Any] | None = None
    if len(result.records) > 0:
        valid_records = [
            r
            for r in result.records
            if r.connectivity != "FAILED"
            and r.metadata
            and "freq_middle" in r.metadata
            and float(r.metadata.get("freq_middle", 0.0)) > 0
        ]
        if valid_records:
            h_over_a = fixed_params.get("slab_thickness", 0.1) if is_3d else 0.1
            a_fixed_h_nm = target_thickness_nm / h_over_a

            sorted_by_cost = sorted(valid_records, key=lambda r: r.cost)
            candidate_records = [r for r in sorted_by_cost if r.cost <= 0.05]
            if not candidate_records:
                candidate_records = sorted_by_cost[:10]

            # 1. Best point under fixed h (a = h / (h/a)):
            best_cand_fixed_h = min(
                candidate_records,
                key=lambda r: abs(
                    (a_fixed_h_nm / float(r.metadata["freq_middle"]))
                    - target_wavelength_nm
                ),
            )
            w_h = float(best_cand_fixed_h.metadata["freq_middle"])
            lam_at_fixed_h = a_fixed_h_nm / w_h
            r1_at_fixed_h = best_cand_fixed_h.params["r1"] * a_fixed_h_nm
            r2_at_fixed_h = best_cand_fixed_h.params["r2"] * a_fixed_h_nm

            # 2. Best point under fixed lambda (a = omega_mid * lambda):
            best_cand_fixed_lam = min(
                candidate_records,
                key=lambda r: abs(
                    h_over_a * (float(r.metadata["freq_middle"]) * target_wavelength_nm)
                    - target_thickness_nm
                ),
            )
            w_lam = float(best_cand_fixed_lam.metadata["freq_middle"])
            a_at_fixed_lam = w_lam * target_wavelength_nm
            h_at_fixed_lam = h_over_a * a_at_fixed_lam
            r1_at_fixed_lam = best_cand_fixed_lam.params["r1"] * a_at_fixed_lam
            r2_at_fixed_lam = best_cand_fixed_lam.params["r2"] * a_at_fixed_lam

            physical_analysis = {
                "h_over_a": float(h_over_a),
                "target_wavelength_nm": float(target_wavelength_nm),
                "target_thickness_nm": float(target_thickness_nm),
                "fixed_thickness_scale": {
                    "description": (
                        f"Fixed thickness h={target_thickness_nm}nm fixes pitch "
                        f"a={a_fixed_h_nm:.1f}nm (h/a={h_over_a}); wavelength closest to target."
                    ),
                    "best_params_normalized": best_cand_fixed_h.params,
                    "normalized_frequency": w_h,
                    "cost": float(best_cand_fixed_h.cost),
                    "fom": float(best_cand_fixed_h.fom),
                    "physical_pitch_nm": a_fixed_h_nm,
                    "physical_thickness_nm": target_thickness_nm,
                    "physical_wavelength_nm": lam_at_fixed_h,
                    "physical_r1_nm": r1_at_fixed_h,
                    "physical_r2_nm": r2_at_fixed_h,
                },
                "fixed_wavelength_scale": {
                    "description": (
                        f"Fixed wavelength lambda={target_wavelength_nm}nm sets pitch "
                        f"a=omega*lambda; thickness closest to target."
                    ),
                    "best_params_normalized": best_cand_fixed_lam.params,
                    "normalized_frequency": w_lam,
                    "cost": float(best_cand_fixed_lam.cost),
                    "fom": float(best_cand_fixed_lam.fom),
                    "physical_pitch_nm": a_at_fixed_lam,
                    "physical_thickness_nm": h_at_fixed_lam,
                    "physical_wavelength_nm": target_wavelength_nm,
                    "physical_r1_nm": r1_at_fixed_lam,
                    "physical_r2_nm": r2_at_fixed_lam,
                },
            }

            analysis_file = result.output_dir / "physical_scaling_analysis.json"
            with open(analysis_file, "w", encoding="utf-8") as f:
                json.dump(physical_analysis, f, indent=2)

    opt_tracked_bands: list[int] | None = None
    if len(result.records) > 0:
        valid_records = [
            r for r in result.records if not r.metadata.get("penalty", False)
        ]
        best_r = (
            max(valid_records, key=lambda r: r.fom)
            if valid_records
            else min(result.records, key=lambda r: r.cost)
        )
        if best_r.metadata and "tracked_bands" in best_r.metadata:
            opt_tracked_bands = [int(b) for b in best_r.metadata["tracked_bands"]]
        elif hasattr(obj_arg, "ref_bands"):
            opt_tracked_bands = [int(b) for b in obj_arg.ref_bands]
        elif hasattr(obj_arg, "mode_indices"):
            opt_tracked_bands = [int(b) for b in obj_arg.mode_indices]

    return {
        "best_params": result.best_params,
        "best_fom": result.best_fom,
        "best_cost": result.best_cost,
        "total_evaluations": len(result.records),
        "optimal_loci": loci_data,
        "refined_loci": locus_results,
        "physical_analysis": physical_analysis,
        "output_dir": str(result.output_dir),
        "substrate_material": sub_mat_key,
        "match_reference_modes": match_reference_modes,
        "tracked_bands": opt_tracked_bands,
        "polarization": pol_key,
    }


def main() -> None:
    """CLI entrypoint for running the C6v 2b-6d Bayesian Optimization."""
    parser = argparse.ArgumentParser(
        description="Bayesian Optimization of C6v 2b-6d Photonic Crystal Unit Cell for Dirac Cone Degeneracy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run in rapid smoke-test mode with low resolution and minimal iterations.",
    )
    parser.add_argument(
        "--slab-thickness",
        type=float,
        default=0.25,
        help="Membrane slab thickness h in units of pitch a (default: 0.25 for h/a=0.25).",
    )
    parser.add_argument(
        "--supercell-z",
        type=float,
        default=4.0,
        help="Vertical supercell height in units of lattice constant a (default: 4.0).",
    )
    parser.add_argument(
        "--p2",
        type=float,
        default=0.25,
        help="Wyckoff 6d satellite hole position parameter (default: 0.25).",
    )
    parser.add_argument(
        "--vary-p2",
        action="store_true",
        help="Optimize 6d position parameter p2 in addition to radii r1 and r2.",
    )
    parser.add_argument(
        "--substrate-material",
        type=str,
        default=None,
        help="Substrate cladding material key from phc_materials (e.g. 'sio2', 'air').",
    )
    parser.add_argument(
        "--substrate-thickness",
        type=float,
        default=2.0,
        help="Substrate buffer thickness in units of pitch a (default: 2.0).",
    )
    parser.add_argument(
        "--matrix-material",
        type=str,
        default="hBN",
        help="Slab dielectric core material key from phc_materials (default: 'hBN').",
    )
    parser.add_argument(
        "--cladding-material",
        type=str,
        default="air",
        help="Top cladding dielectric material key from phc_materials (default: 'air').",
    )
    parser.add_argument(
        "--match-reference-modes",
        action="store_true",
        help="Identify target degeneracy by tracking 3 reference modes via modal overlap.",
    )
    parser.add_argument(
        "--reference-data",
        type=str,
        default=None,
        help="Path to pre-saved reference dataset directory in 'saved_data/<name>' containing fields.npz and metadata.json.",
    )
    parser.add_argument(
        "--save-reference",
        type=str,
        default=None,
        help="Reference name to save optimal point modal fields directly to 'saved_data/<name>' without prompting.",
    )
    parser.add_argument(
        "--save-match-reference",
        type=str,
        default=None,
        help="Reference name to save locus match point dataset directly to 'saved_data/<name>' without prompting.",
    )
    parser.add_argument(
        "--reference-run",
        type=str,
        default=None,
        help="Path to previous unperturbed simulation directory to extract reference cell from.",
    )
    parser.add_argument(
        "--reference-bands",
        type=int,
        nargs=3,
        default=[9, 10, 11],
        help="1-based band indices of 3 reference modes at Gamma (default: 9 10 11).",
    )
    parser.add_argument(
        "--reference-r1",
        type=float,
        default=None,
        help="Nominal r1 for unperturbed reference cell (default: auto).",
    )
    parser.add_argument(
        "--reference-r2",
        type=float,
        default=None,
        help="Nominal r2 for unperturbed reference cell (default: auto).",
    )
    parser.add_argument(
        "--reference-p2",
        type=float,
        default=0.25,
        help="Nominal p2 for unperturbed reference cell (default: 0.25).",
    )
    parser.add_argument(
        "--polarization",
        type=str,
        default=None,
        choices=["tm", "te", "tm_like", "te_like", "all", "no_parity"],
        help="Target mode polarization (default: 'all' when substrate is used, else 'te_like').",
    )
    parser.add_argument(
        "--target-irreps",
        nargs="+",
        default=["A_2", "E_1", "E_1"],
        help="Target irreducible representations at Gamma (default: A_2 E_1 E_1).",
    )
    parser.add_argument(
        "--enforce-irreps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enforce that tracked bands match target point-group irreps, penalizing alien representations (default: True).",
    )
    parser.add_argument(
        "--irrep-occurrences",
        type=int,
        nargs="+",
        default=[1, 4, 4],
        help="Occurrence count per target irrep above min_band (default: 1 4 4).",
    )
    parser.add_argument(
        "--target-wavelength",
        type=float,
        default=436.0,
        help="Target physical wavelength in nanometers (default: 436.0 nm).",
    )
    parser.add_argument(
        "--target-thickness",
        type=float,
        default=100.0,
        help="Target physical membrane thickness in nanometers (default: 100.0 nm).",
    )
    parser.add_argument(
        "--r1-bounds",
        type=float,
        nargs=2,
        default=[0.15, 0.25],
        help="Search range for primary hole radius r1 in units of a (default: 0.15 0.25).",
    )
    parser.add_argument(
        "--r2-bounds",
        type=float,
        nargs=2,
        default=[0.05, 0.10],
        help="Search range for satellite hole radius r2 in units of a (default: 0.05 0.10).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="In-plane MPB computational mesh resolution per pitch a (default: 18, quick: 12).",
    )
    parser.add_argument(
        "--resolution-z",
        type=int,
        default=None,
        help="Vertical MPB mesh resolution for 3D slabs (default: 16, quick: 6).",
    )
    parser.add_argument(
        "--num-bands",
        type=int,
        default=None,
        help="Number of eigenbands to compute at Gamma (default: 15 / 28 on substrate).",
    )
    parser.add_argument(
        "--initial-points",
        type=int,
        default=None,
        help="Number of initial quasi-random exploration points (default: 100, quick: 2).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Number of Bayesian optimization active learning generations (default: 0, quick: 1).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes (default: 20, quick: 1).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of candidate points evaluated per BO generation (default: matches --workers).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory override (default: auto-resolved by phc_hydra).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the real-time progress bar.",
    )
    parser.add_argument(
        "--find-loci",
        default="1",
        help="Number of degeneracy loci to detect on the GP surrogate map: an integer (e.g. 1, 2) or 'auto' (default: 1).",
    )
    parser.add_argument(
        "--analyze-locus",
        nargs="*",
        default=None,
        metavar="IDX",
        help=(
            "Refine continuous degeneracy locus curve(s) and compute adjacent group velocity. "
            "Accepts one or more locus IDs (e.g. '--analyze-locus', '--analyze-locus 2', '--analyze-locus 1 2'). "
            "Defaults to [1] if passed without arguments; omitted means do not refine."
        ),
    )
    parser.add_argument(
        "--locus-mode",
        type=str,
        default="cartesian",
        choices=["cartesian", "polar", "auto"],
        help="Mode for degeneracy locus refinement (default: cartesian).",
    )
    parser.add_argument(
        "--force-open",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force degeneracy locus manifold to be an open curve (default: True).",
    )
    parser.add_argument(
        "--max-refine-steps",
        type=int,
        default=5,
        help="Maximum number of refinement steps for the degeneracy locus (default: 5).",
    )
    parser.add_argument(
        "--tracking-strategy",
        type=str,
        default="cluster",
        choices=["cluster", "bipartite", "greedy"],
        help="Target mode tracking algorithm: 'cluster' (multiplet cohesion, default), 'bipartite' (Hungarian matching), or 'greedy' (subspace projection).",
    )
    parser.add_argument(
        "--kpath-type",
        type=str,
        default="gamma_centered",
        choices=["gamma_centered", "standard"],
        help="K-path trajectory for band diagram plotting: 'gamma_centered' (default) or 'standard'.",
    )
    parser.add_argument(
        "--k-max",
        type=float,
        default=0.1,
        help="Maximum Cartesian wavevector radius |k|/(2π) for Gamma-centered k-path (default: 0.1).",
    )
    parser.add_argument(
        "--lam-min",
        type=float,
        default=420.0,
        help="Minimum wavelength limit in nm for physical wavelength band diagrams (default: 420.0 nm).",
    )
    parser.add_argument(
        "--lam-max",
        type=float,
        default=450.0,
        help="Maximum wavelength limit in nm for physical wavelength band diagrams (default: 450.0 nm).",
    )
    parser.add_argument(
        "--overlap-mode",
        type=str,
        default="midplane",
        choices=["midplane", "slab", "full"],
        help=(
            "Spatial overlap formulation: 'midplane' (2D z=0 mid-plane slice, "
            "invariant to slab thickness h/a, default), 'slab' (3D masked slab core |z| <= h/2), "
            "or 'full' (full 3D supercell)."
        ),
    )
    parser.add_argument(
        "--interpolate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable automatic field interpolation across mismatched mesh resolutions for mode matching (default: True).",
    )
    parser.add_argument(
        "--from-run",
        type=str,
        default=None,
        help="Path to an existing optimization run directory to analyze without re-running optimization.",
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=85.0,
        help="Cutoff percentile (e.g. 80.0 or 85.0) for surrogate locus ridge extraction (default: 85.0).",
    )
    args = parser.parse_args()

    sub_mat = args.substrate_material
    if args.match_reference_modes and sub_mat is None:
        sub_mat = "sio2"

    print("=" * 72)
    print(" C6v 2b-6d Photonic Crystal Bayesian Optimization ")
    dim_str = (
        f"3D Slab (h/a={args.slab_thickness})"
        if args.slab_thickness is not None
        else "2D Periodic Sheet"
    )
    pol_display = (
        args.polarization.upper()
        if args.polarization is not None
        else ("ALL" if (sub_mat or args.match_reference_modes) else "TE_LIKE")
    )
    print(f" Structure: {dim_str} in hBN, Polarization: {pol_display}")
    if sub_mat and sub_mat.lower() != "air":
        print(f" Substrate: {sub_mat} (thickness = {args.substrate_thickness} a)")
    if args.match_reference_modes:
        print(
            f" Mode Matching: Modal overlap tracking 3 reference modes (bands {args.reference_bands}, overlap_mode: {args.overlap_mode})"
        )
    else:
        print(
            f" Target Irreps: {args.target_irreps} (occurrences: {args.irrep_occurrences})"
        )
    print(
        f" Search Bounds: r1 in [{args.r1_bounds[0]}, {args.r1_bounds[1]}], r2 in [{args.r2_bounds[0]}, {args.r2_bounds[1]}]"
    )
    print(" Domain Connectivity at z=0: ENFORCED (PBC-spanning check)")
    print(
        f" Physical Targets: λ = {args.target_wavelength} nm, h = {args.target_thickness} nm"
    )
    print("=" * 72)

    find_loci_raw = str(args.find_loci).strip()
    find_loci_val: int | str = (
        "auto" if find_loci_raw.lower() == "auto" else int(find_loci_raw)
    )

    if args.analyze_locus is None:
        analyze_locus_arg = None
    elif len(args.analyze_locus) == 0:
        analyze_locus_arg = [1]
    elif any(str(x).lower() in ("all", "auto") for x in args.analyze_locus):
        analyze_locus_arg = "all"
        if find_loci_val == 1:
            find_loci_val = "auto"
    else:
        analyze_locus_arg = [int(x) for x in args.analyze_locus]

    res = run_c6v_optimization_pipeline(
        quick=args.quick,
        slab_thickness=args.slab_thickness,
        supercell_z=args.supercell_z,
        vary_p2=args.vary_p2,
        polarization=args.polarization,
        resolution=args.resolution,
        resolution_z=args.resolution_z,
        num_bands=args.num_bands,
        initial_points=args.initial_points,
        max_iterations=args.max_iterations,
        batch_size=args.batch_size,
        num_workers=args.workers,
        matrix_material=args.matrix_material,
        cladding_material=args.cladding_material,
        substrate_material=sub_mat,
        substrate_thickness=args.substrate_thickness,
        match_reference_modes=args.match_reference_modes,
        reference_data=args.reference_data,
        reference_run=args.reference_run,
        reference_bands=tuple(args.reference_bands),
        reference_r1=args.reference_r1,
        reference_r2=args.reference_r2,
        reference_p2=args.reference_p2,
        output_dir=args.output_dir,
        show_progress=not args.no_progress,
        find_loci=find_loci_val,
        analyze_locus=analyze_locus_arg,
        threshold_percentile=args.threshold_percentile,
        from_run=args.from_run,
        locus_mode=args.locus_mode,
        force_open=args.force_open,
        overlap_mode=args.overlap_mode,
        interpolate=args.interpolate,
        target_irreps=args.target_irreps,
        irrep_occurrences=args.irrep_occurrences,
        enforce_irreps=args.enforce_irreps,
        target_wavelength_nm=args.target_wavelength,
        target_thickness_nm=args.target_thickness,
        r1_bounds=tuple(args.r1_bounds),
        r2_bounds=tuple(args.r2_bounds),
        tracking_strategy=args.tracking_strategy,
        kpath_type=args.kpath_type,
        k_max=args.k_max,
        lam_min_nm=args.lam_min,
        lam_max_nm=args.lam_max,
    )

    print("\n" + "=" * 72)
    print(" Optimization Complete! ")
    print(f"  Best Parameters: {res['best_params']}")
    print(f"  Best FOM:        {res['best_fom']:.2f}")
    print(f"  Residual Cost:   {res['best_cost']:.6f}")
    print(f"  Evaluations:     {res['total_evaluations']}")
    if res.get("match_reference_modes"):
        print(
            "  Mode Matching:   Tracked 3 unperturbed reference modes via modal overlap"
        )
    if res.get("substrate_material"):
        print(f"  Substrate:       {res['substrate_material']}")
    if res.get("optimal_loci"):
        print(
            f"  Degeneracy Loci: {len(res['optimal_loci'])} manifold curve(s) extracted (optimal_loci.json)"
        )
    if res.get("refined_loci"):
        print(
            f"  Refined Loci:    {len(res['refined_loci'])} refined locus manifold(s) analyzed with group velocity"
        )
        for l_idx, loc in enumerate(res["refined_loci"]):
            if "target_match" in loc:
                tm = loc["target_match"]
                p1_n = loc.get("p1_name", "r1")
                p2_n = loc.get("p2_name", "r2")
                print(f"\n  Closest Target Match on Locus #{l_idx + 1}:")
                print(
                    f"    Normalized Radii: {p1_n} = {tm.get(p1_n, 0.0):.4f} a, {p2_n} = {tm.get(p2_n, 0.0):.4f} a"
                )
                print(f"    Physical Pitch:   a = {tm.get('pitch_nm', 0.0):.1f} nm")
                print(
                    f"    Slab Thickness:   h = {tm.get('thickness_nm', 0.0):.1f} nm (target: {tm.get('target_thickness_nm', 0.0):.1f} nm)"
                )
                print(
                    f"    Target λ:         λ = {tm.get('target_wavelength_nm', 0.0):.1f} nm (ω~_D = {tm.get('omega_d', 0.0):.4f})"
                )
                print(
                    "    Band Diagrams:    locus_match_band_structure.png (normalized)"
                )
                print(
                    "                      locus_match_band_structure_wavelength.png (physical)"
                )
                print(
                    "    Locus Diagrams:   locus_dirac_frequency.png (ω~_D vs s & samples)"
                )
                print(
                    "                      locus_wavelength.png (λ vs samples: match h vs target h)"
                )
    if res.get("physical_analysis"):
        pa = res["physical_analysis"]
        f_h = pa.get("fixed_thickness_scale", {})
        f_lam = pa.get("fixed_wavelength_scale", {})
        print(f"\n  Physical Parameter Matching (h/a = {pa.get('h_over_a', 0.1)}):")
        print(
            f"    [Scale 1: Fixed h = {pa['target_thickness_nm']} nm -> Pitch a = {f_h.get('physical_pitch_nm'):.1f} nm]"
        )
        print(
            f"      Resulting λ = {f_h.get('physical_wavelength_nm'):.1f} nm (target: {pa['target_wavelength_nm']} nm)"
        )
        print(
            f"      Params: r1 = {f_h.get('physical_r1_nm'):.1f} nm, r2 = {f_h.get('physical_r2_nm'):.1f} nm (ω~ = {f_h.get('normalized_frequency'):.4f})"
        )
        print(
            f"    [Scale 2: Fixed λ = {pa['target_wavelength_nm']} nm -> Pitch a = {f_lam.get('physical_pitch_nm'):.1f} nm]"
        )
        print(
            f"      Resulting h = {f_lam.get('physical_thickness_nm'):.1f} nm (target: {pa['target_thickness_nm']} nm)"
        )
        print(
            f"      Params: r1 = {f_lam.get('physical_r1_nm'):.1f} nm, r2 = {f_lam.get('physical_r2_nm'):.1f} nm (ω~ = {f_lam.get('normalized_frequency'):.4f})"
        )
    print(f"\n  Output Folder:   {res['output_dir']}")
    print("=" * 72)

    # -------------------------------------------------------------
    # Interactive Reference Field Data Storage Prompt (60s timer)
    # -------------------------------------------------------------
    save_ref_name = args.save_reference
    if save_ref_name is None and sys.stdin.isatty():
        print("\n" + "=" * 72)
        print(" Save Optimal Reference Field Data ")
        print("=" * 72)
        ans = timed_input(
            "Do you want to store the field of the optimal point to 'saved_data'? [y/N] (Auto-skips in 60s): ",
            timeout=60.0,
        )
        if ans and ans.lower() in ("y", "yes"):
            ts_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            default_name = f"c6v_opt_{ts_str}"
            user_name = timed_input(
                f"Enter reference name [default: {default_name}] (Auto-skips in 60s): ",
                timeout=60.0,
            )
            save_ref_name = (
                user_name.strip() if user_name and user_name.strip() else default_name
            )

    if save_ref_name:
        opt_bands = res.get("tracked_bands") or args.reference_bands
        res_val = (
            args.resolution
            if args.resolution is not None
            else (12 if args.quick else 18)
        )
        res_z_val = (
            args.resolution_z
            if args.resolution_z is not None
            else (6 if args.quick else 16)
        )
        save_optimal_field_reference(
            name=save_ref_name,
            params=res["best_params"],
            pitch=1.0,
            slab_thickness=float(args.slab_thickness)
            if args.slab_thickness is not None
            else 0.25,
            supercell_z=float(args.supercell_z),
            resolution=res_val,
            resolution_z=res_z_val,
            matrix_material=args.matrix_material,
            cladding_material=args.cladding_material,
            substrate_material=sub_mat,
            substrate_thickness=float(args.substrate_thickness) if sub_mat else None,
            polarization=res.get("polarization", "all" if sub_mat else "te_like"),
            tracked_bands=opt_bands,
            overlap_mode=args.overlap_mode,
        )

    # -------------------------------------------------------------
    # Interactive Locus Match Point Reference Storage Prompt (60s timer)
    # -------------------------------------------------------------
    first_match_meta = None
    if res.get("refined_loci"):
        for loc in res["refined_loci"]:
            if "target_match" in loc:
                first_match_meta = loc["target_match"]
                break

    save_match_name = args.save_match_reference
    if first_match_meta is not None:
        if save_match_name is None and sys.stdin.isatty():
            print("\n" + "=" * 72)
            print(" Save Locus Match Point Data ")
            print("=" * 72)
            ans_m = timed_input(
                "Do you want to store the match point data of the locus to 'saved_data'? [y/N] (Auto-skips in 60s): ",
                timeout=60.0,
            )
            if ans_m and ans_m.lower() in ("y", "yes"):
                ts_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
                default_match_name = f"c6v_match_{ts_str}"
                user_match_name = timed_input(
                    f"Enter match reference name [default: {default_match_name}] (Auto-skips in 60s): ",
                    timeout=60.0,
                )
                save_match_name = (
                    user_match_name.strip()
                    if user_match_name and user_match_name.strip()
                    else default_match_name
                )

        if save_match_name:
            res_val = (
                args.resolution
                if args.resolution is not None
                else (12 if args.quick else 18)
            )
            res_z_val = (
                args.resolution_z
                if args.resolution_z is not None
                else (6 if args.quick else 16)
            )
            opt_bands = res.get("tracked_bands") or args.reference_bands
            save_match_point_reference(
                name=save_match_name,
                match_meta=first_match_meta,
                output_dir=res["output_dir"],
                base_dir="saved_data",
                matrix_material=args.matrix_material,
                cladding_material=args.cladding_material,
                substrate_material=sub_mat,
                substrate_thickness=float(args.substrate_thickness)
                if sub_mat
                else None,
                supercell_z=float(args.supercell_z),
                resolution=res_val,
                resolution_z=res_z_val,
                polarization=res.get("polarization", "all" if sub_mat else "te_like"),
                tracked_bands=opt_bands,
                p2=float(args.reference_p2) if args.reference_p2 is not None else 0.25,
                slab_thickness=float(args.slab_thickness)
                if args.slab_thickness is not None
                else 0.25,
                overlap_mode=args.overlap_mode,
            )


if __name__ == "__main__":
    main()
