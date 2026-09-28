#!/usr/bin/env python3
"""Accidental Degeneracy Optimization for 3D C6v 2b-6d Slab with SiO2 Substrate.

Starting from the TE-like degenerate/near-degenerate triplet at Γ in an unperturbed
symmetric membrane (air-clad hBN slab, λ ≈ 440 nm), optimizes Wyckoff hole radii R1 (2b)
and R2 (6d) on an asymmetric silica (SiO2, n_sub ≈ 1.444) substrate.

Physics & Methodology:
1. Reference Simulation: Solves the unperturbed air-clad membrane (n_sub = 1.0) at Γ and
   extracts reference modal fields for Bands 16, 17, 18.
2. Modal Overlap Degeneracy Objective: Evaluates candidate structures on the SiO2 substrate
   by projecting substrate eigenmodes onto the unperturbed triplet manifold via spatial
   overlap integrals, tracking the 3 corresponding modes and minimizing their frequency splitting.
3. Bayesian Optimization: Searches the (R1, R2) parameter space to map the accidental
   degeneracy landscape and locate the global optimum.
4. Locus Extraction & Normal-Line Refinement: Traces the continuous 1D degeneracy locus
   ridge across the Gaussian Process surrogate, refining sampled points with MPB normal line
   secant root finding.
5. Target Design Identification: Pinpoints the optimal locus point matching target physical
   thickness h = 100 nm and operating wavelength λ0 = 440 nm on SiO2.
6. Artifact Export: Saves unit_cell.gds, optimal_loci.png, locus_refined.csv,
   target_design_summary.json, and simulation_results.json into canonical Hydra output:
   outputs/mpb/optimization/c6v_2b_6d_substrate/<timestamp>/.

Command-Line Usage:
    # Full optimization run (nominal resolution, 10 GP generations)
    python analysis/optimize_c6v_substrate_degeneracy.py

    # Quick smoke test execution (< 15 seconds)
    python analysis/optimize_c6v_substrate_degeneracy.py --quick

    # Custom resolution, iterations, and parallel workers
    python analysis/optimize_c6v_substrate_degeneracy.py --resolution 28 --iterations 15 --workers 4

    # Custom output directory
    python analysis/optimize_c6v_substrate_degeneracy.py --output-dir outputs/custom_c6v_opt
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import meep as mp
import numpy as np

# Ensure repository root is in sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from phc_hydra import SimulationOutputManager, resolve_simulation_output_dir
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
    export_locus_to_csv,
    find_target_locus_point,
    plot_bo_surrogate_map,
)
from phc_utils import export_gds, silence_c_stdout

# Suppress Meep/MPB C-level verbosity by default
mp.verbosity(0)


def c6v_cell_factory(
    r1: float,
    r2: float,
    p2: float = 0.25,
    pitch: float = 0.381,
) -> Any:
    """Instantiates a C6v unit cell component with Wyckoff 2b and 6d holes.

    Args:
        r1: Radius of inner Wyckoff 2b hole in micrometers.
        r2: Radius of satellite Wyckoff 6d holes in micrometers.
        p2: Radial coordinate of Wyckoff 6d holes (default: 0.25).
        pitch: Hexagonal lattice pitch constant a in micrometers (default: 0.381).

    Returns:
        gdsfactory.Component containing the unit cell layout.
    """
    return phc_wyckoff_unit_cell(
        pitch=float(pitch),
        features=[("2b", float(r1)), ("6d", float(r2), float(p2))],
    )


def solve_unperturbed_reference_modes(
    pitch: float = 0.381,
    slab_thickness: float = 0.100,
    r1_nom: float = 0.089,
    r2_nom: float = 0.038,
    p2: float = 0.25,
    supercell_z: float = 4.0,
    slab_material: str = "hBN",
    cladding_material: str = "air",
    resolution: int = 24,
    num_bands: int = 22,
    target_bands: tuple[int, ...] = (16, 17, 18),
    verbose: bool = True,
) -> tuple[
    dict[int, tuple[np.ndarray, np.ndarray | None]], dict[int, float], list[int]
]:
    """Solves the unperturbed symmetric air-clad slab at Γ to extract reference fields.

    Args:
        pitch: Lattice constant a in micrometers.
        slab_thickness: Slab thickness in micrometers.
        r1_nom: Nominal Wyckoff 2b hole radius in micrometers.
        r2_nom: Nominal Wyckoff 6d hole radius in micrometers.
        p2: Wyckoff 6d radial coordinate.
        supercell_z: Supercell height in units of lattice constant a.
        slab_material: Slab core material key (e.g. 'hBN').
        cladding_material: Upper and lower cladding material key (e.g. 'air').
        resolution: Grid resolution per unit pitch a.
        num_bands: Number of eigenbands to compute at Γ.
        target_bands: Desired reference band indices at Γ (default: (16, 17, 18)).
        verbose: If True, prints progress details.

    Returns:
        Tuple of (ref_fields_dict, ref_freqs_dict, resolved_target_bands).
    """
    if verbose:
        print(
            "\n--- 1. Solving Unperturbed Reference System (Air Cladding, n_sub = 1.0) ---"
        )

    ref_comp = c6v_cell_factory(r1=r1_nom, r2=r2_nom, p2=p2, pitch=pitch)
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
        slab_material=slab_material,
        substrate_material=None,
        etch_material=cladding_material,
        geometry_lattice=lattice,
    )

    k_points = [mp.Vector3(0.0, 0.0, 0.0)]

    with silence_c_stdout():
        ms_ref = create_mode_solver(
            geometry_lattice=lattice,
            geometry=geometry,
            k_points=k_points,
            default_material=cladding_material,
            resolution=resolution,
            num_bands=num_bands,
        )
        res_ref = run_band_solver(
            ms=ms_ref,
            polarization="all",
            dimension="3D_slab",
            verbose=False,
        )

    all_freqs = res_ref.get("freqs", {}).get("all", ms_ref.all_freqs)[0]

    # Resolve target bands
    avail_bands = [b for b in target_bands if b <= len(all_freqs)]
    if len(avail_bands) < len(target_bands):
        # Fallback to top modes near target frequency if band count is lower
        f_target = pitch / 0.440
        diffs = [abs(f - f_target) for f in all_freqs]
        sorted_indices = np.argsort(diffs)
        avail_bands = sorted(
            [int(idx + 1) for idx in sorted_indices[: len(target_bands)]]
        )

    ref_fields: dict[int, tuple[np.ndarray, np.ndarray | None]] = {}
    ref_freqs: dict[int, float] = {}

    for b in avail_bands:
        f = float(all_freqs[b - 1])
        lam_nm = (pitch / f * 1000.0) if f > 0 else 0.0
        ref_freqs[b] = f
        ref_fields[b] = extract_eigenmode_fields(
            ms_ref,
            band=b,
            field="electric_displacement",
            slab_thickness=norm_h,
            z_center=0.0,
        )
        if verbose:
            print(f"  Ref Band {b:2d}: ω~ = {f:.5f} | λ = {lam_nm:.2f} nm")

    return ref_fields, ref_freqs, avail_bands


def run_c6v_substrate_optimization_pipeline(
    pitch: float = 0.381,
    slab_thickness: float = 0.100,
    p2: float = 0.25,
    r1_bounds: tuple[float, float] = (0.060, 0.120),
    r2_bounds: tuple[float, float] = (0.020, 0.060),
    slab_material: str = "hBN",
    cladding_material: str = "air",
    substrate_material: str = "sio2",
    supercell_z: float = 4.0,
    target_wavelength_nm: float = 440.0,
    target_thickness_nm: float = 100.0,
    resolution: int = 24,
    num_bands: int = 22,
    initial_points: int = 8,
    max_iterations: int = 10,
    batch_size: int = 2,
    num_workers: int = 2,
    sample_points: int = 8,
    max_refine_steps: int = 5,
    refine_tolerance: float = 1e-4,
    quick: bool = False,
    output_dir: Path | str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Executes the end-to-end accidental degeneracy optimization pipeline on SiO2 substrate.

    Args:
        pitch: Hexagonal lattice pitch a in micrometers (default: 0.381).
        slab_thickness: Membrane thickness h in micrometers (default: 0.100).
        p2: Wyckoff 6d satellite radial position (default: 0.25).
        r1_bounds: Parameter search bounds for Wyckoff 2b hole radius (um).
        r2_bounds: Parameter search bounds for Wyckoff 6d hole radius (um).
        slab_material: Slab core material key (default: 'hBN').
        cladding_material: Top cladding material key (default: 'air').
        substrate_material: Substrate cladding material key (default: 'sio2').
        supercell_z: Supercell height in units of lattice constant a (default: 4.0).
        target_wavelength_nm: Desired accidental degeneracy wavelength in nm (default: 440.0).
        target_thickness_nm: Desired physical slab thickness in nm (default: 100.0).
        resolution: Grid resolution per unit pitch a (default: 24).
        num_bands: Number of eigenbands to compute at Γ (default: 22).
        initial_points: Initial Sobol exploratory evaluations.
        max_iterations: Number of guided Bayesian optimization generations.
        batch_size: Candidates evaluated per generation.
        num_workers: Worker processes for candidate evaluations.
        sample_points: Number of points along extracted locus spline to refine.
        max_refine_steps: Maximum secant iterations per locus point.
        refine_tolerance: Degeneracy cost threshold defining convergence.
        quick: If True, executes rapid smoke test (< 15 seconds).
        output_dir: Optional explicit output directory.
        verbose: If True, prints pipeline progress.

    Returns:
        Structured dictionary containing target design point, refined locus,
        and simulation artifact manifest.
    """
    t_start = time.perf_counter()

    # 1. Quick mode parameter overrides
    if quick:
        resolution = 14
        num_bands = 20
        initial_points = 3
        max_iterations = 2
        batch_size = 1
        num_workers = 1
        sample_points = 4
        max_refine_steps = 2
        refine_tolerance = 1e-3

    # 2. Output directory coordination via phc_hydra
    resolved_dir = (
        Path(output_dir)
        if output_dir is not None
        else resolve_simulation_output_dir(
            solver="mpb",
            sim_type="optimization",
            geometry="c6v_2b_6d_substrate",
        )
    )
    out_mgr = SimulationOutputManager(
        output_dir=resolved_dir,
        solver="mpb",
        sim_type="optimization",
        geometry_name="c6v_2b_6d_substrate",
    )
    canonical_out = out_mgr.output_dir

    if verbose:
        print("=" * 72)
        print("  C6v 2b-6d Substrate Accidental Degeneracy Optimization Pipeline")
        print("=" * 72)
        print(f"  Output Directory: {canonical_out}")
        print(f"  Slab Material: {slab_material} | Substrate: {substrate_material}")
        print(
            f"  Target: λ0 = {target_wavelength_nm:.1f} nm, h = {target_thickness_nm:.1f} nm"
        )
        print(f"  Resolution: {resolution} | Num Bands: {num_bands} | Quick: {quick}")

    # 3. Solve unperturbed reference system
    ref_fields, ref_freqs, ref_bands = solve_unperturbed_reference_modes(
        pitch=pitch,
        slab_thickness=slab_thickness,
        r1_nom=0.089,
        r2_nom=0.038,
        p2=p2,
        supercell_z=supercell_z,
        slab_material=slab_material,
        cladding_material=cladding_material,
        resolution=resolution,
        num_bands=num_bands,
        target_bands=(16, 17, 18),
        verbose=verbose,
    )

    # 4. Construct Modal Overlap Degeneracy Objective
    min_b = max(1, min(ref_bands) - 4)
    max_b = min(num_bands, max(ref_bands) + 4)
    candidate_window = list(range(min_b, max_b + 1))

    objective = ModalOverlapDegeneracyObjective(
        ref_fields=ref_fields,
        ref_frequencies=ref_freqs,
        ref_bands=ref_bands,
        target_band_candidates=candidate_window,
        polarization="all",
        field="electric_displacement",
        slab_thickness=slab_thickness / pitch,
        pitch=pitch,
        min_overlap_threshold=0.35,
        target_cost=0.0005,
    )

    # 5. Bayesian Optimization Controller
    if verbose:
        print("\n--- 2. Executing Bayesian Optimization on SiO2 Substrate ---")

    optimizer = BayesianOptimizer(
        cell_factory=c6v_cell_factory,
        parameters={
            "r1": r1_bounds,
            "r2": r2_bounds,
        },
        fixed_parameters={
            "p2": p2,
            "pitch": pitch,
            "slab_thickness": slab_thickness,
            "supercell_z": supercell_z,
        },
        objective=objective,
        lattice_type="hexagonal",
        pitch=pitch,
        dimension="3D_slab",
        supercell_z=supercell_z,
        resolution=resolution,
        num_bands=num_bands,
        background_material=cladding_material,
        matrix_material=slab_material,
        substrate_material=substrate_material,
        max_iterations=max_iterations,
        initial_points=initial_points,
        batch_size=batch_size,
        num_workers=num_workers,
        output_dir=canonical_out,
        geometry_name="c6v_2b_6d_substrate",
        show_progress=not quick,
    )

    opt_result = optimizer.run()

    if verbose:
        print(f"  ✓ Optimization finished: {len(opt_result.records)} evaluations")
        print(f"  Best Parameters: {opt_result.best_params}")
        print(
            f"  Best Cost: {opt_result.best_cost:.5e} (FOM: {opt_result.best_fom:.1f})"
        )

    # 6. Extract Degeneracy Locus & Refine with Normal Line Root-Finding
    if verbose:
        print("\n--- 3. Extracting & Refining Degeneracy Locus Ridge ---")

    loci = optimizer.analyze_locus(
        mode="ridge",
        delta_k=0.01,
        sample_points=sample_points,
        refine=True,
        refine_tolerance=refine_tolerance,
        max_refine_steps=max_refine_steps,
        exclude_unrefined=False,
        save_artifacts=True,
        plot_profile=True,
        plot_dirac_freq=True,
        target_wavelength=target_wavelength_nm / 1000.0,
        target_thickness=target_thickness_nm / 1000.0,
        compare_substrate=False,
        verbose=verbose,
    )

    if not loci:
        raise RuntimeError(
            "Failed to extract any degeneracy loci from surrogate landscape."
        )

    primary_locus = loci[0]

    # Export canonical locus_refined.csv
    locus_csv_path = canonical_out / "locus_refined.csv"
    export_locus_to_csv(
        primary_locus,
        locus_csv_path,
        param_names=optimizer.param_names,
    )
    if verbose:
        print(f"  ✓ Exported canonical locus CSV: {locus_csv_path.name}")

    # 7. Pinpoint Target Design matching h = 100 nm and λ = 440 nm
    if verbose:
        print("\n--- 4. Pinpointing Target Design (h = 100 nm, λ = 440 nm) ---")

    best_idx, target_match = find_target_locus_point(
        locus=primary_locus,
        target_wavelength_nm=target_wavelength_nm,
        target_thickness_nm=target_thickness_nm,
        slab_thickness=slab_thickness,
        pitch=pitch,
    )

    t_r1 = float(primary_locus["x1"][best_idx])
    t_r2 = float(primary_locus["x2"][best_idx])
    omega_d = float(target_match["omega_d"])
    ideal_omega_d = float(target_match["ideal_omega_d"])
    scaled_pitch_nm = float(target_match["pitch_nm"])
    scaled_thickness_nm = float(target_match["thickness_nm"])

    target_summary = {
        "index": best_idx,
        "parameters": {
            "r1_um": round(t_r1, 5),
            "r2_um": round(t_r2, 5),
            "p2": p2,
            "r1_over_a": round(t_r1 / pitch, 5),
            "r2_over_a": round(t_r2 / pitch, 5),
        },
        "target_spec": {
            "target_wavelength_nm": target_wavelength_nm,
            "target_thickness_nm": target_thickness_nm,
            "substrate": substrate_material,
            "slab": slab_material,
        },
        "operating_point": {
            "omega_d": round(omega_d, 6),
            "ideal_omega_d": round(ideal_omega_d, 6),
            "frequency_discrepancy_pct": round(
                abs(omega_d - ideal_omega_d) / ideal_omega_d * 100, 3
            ),
            "physical_pitch_nm": round(scaled_pitch_nm, 2),
            "physical_thickness_nm": round(scaled_thickness_nm, 2),
            "residual_gap": float(primary_locus["residual_gap"][best_idx])
            if "residual_gap" in primary_locus
            else None,
        },
    }

    target_json_path = canonical_out / "target_design_summary.json"
    with open(target_json_path, "w", encoding="utf-8") as f:
        json.dump(target_summary, f, indent=2)

    if verbose:
        print(f"  ✓ Target Design Point #{best_idx + 1}:")
        print(f"    R1 = {t_r1:.4f} μm (R1/a = {t_r1 / pitch:.4f})")
        print(f"    R2 = {t_r2:.4f} μm (R2/a = {t_r2 / pitch:.4f})")
        print(f"    ω~_D = {omega_d:.5f} (Ideal: {ideal_omega_d:.5f})")
        print(f"    Physical pitch: a = {scaled_pitch_nm:.1f} nm")
        print(f"    Physical thickness: h = {scaled_thickness_nm:.1f} nm")
        print(f"  ✓ Saved target design summary: {target_json_path.name}")

    # 8. Export physical unit cell GDS and optimal loci surrogate map
    target_comp = c6v_cell_factory(r1=t_r1, r2=t_r2, p2=p2, pitch=pitch)
    gds_path = canonical_out / "unit_cell.gds"
    export_gds(target_comp, gds_path, overwrite=True)
    if verbose:
        print(f"  ✓ Exported target unit cell GDS: {gds_path.name}")

    loci_fig_path = canonical_out / "optimal_loci.png"
    plot_bo_surrogate_map(
        optimizer=optimizer.optimizer,
        records=optimizer.records,
        param_names=optimizer.param_names,
        output_path=loci_fig_path,
        overlay_locus=True,
    )
    if verbose:
        print(f"  ✓ Exported optimal loci surrogate map: {loci_fig_path.name}")

    # 9. Register simulation manifest and save results JSON
    elapsed = time.perf_counter() - t_start
    sim_manifest = {
        "geometry": "c6v_2b_6d_substrate",
        "solver": "mpb",
        "sim_type": "optimization",
        "quick": quick,
        "elapsed_seconds": round(elapsed, 2),
        "target_design": target_summary,
        "best_bo_params": opt_result.best_params,
        "best_bo_cost": float(opt_result.best_cost),
        "num_evaluations": len(opt_result.records),
        "num_loci_found": len(loci),
    }

    out_mgr.save_results_json(
        geometry_cfg={
            "name": "c6v_2b_6d_substrate",
            "pitch": pitch,
            "slab_thickness": slab_thickness,
            "target_design": target_summary["parameters"],
        },
        simulation_cfg={
            "solver": "mpb",
            "sim_type": "optimization",
            "substrate_material": substrate_material,
            "resolution": resolution,
            "num_bands": num_bands,
            "quick": quick,
        },
        extra_data=sim_manifest,
    )

    if verbose:
        print(f"\nPipeline completed in {elapsed:.2f} s.")
        print(f"Canonical outputs saved to: {canonical_out}")

    return {
        "output_dir": canonical_out,
        "target_design": target_summary,
        "primary_locus": primary_locus,
        "optimization_result": opt_result,
        "manifest": sim_manifest,
    }


def main() -> None:
    """CLI Entrypoint for C6v 2b-6d Substrate Optimization."""
    parser = argparse.ArgumentParser(
        description="Optimize C6v 2b-6d PhC slab on SiO2 substrate for accidental degeneracy near 440 nm."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Execute fast smoke test with low resolution and minimal iterations.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=24,
        help="MPB grid resolution per pitch a (default: 24).",
    )
    parser.add_argument(
        "--num-bands",
        type=int,
        default=22,
        help="Number of eigenbands to compute at Γ (default: 22).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of guided Bayesian optimization generations (default: 10).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of concurrent worker processes for BO evaluation (default: 2).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Explicit output directory (defaults to canonical Hydra path).",
    )

    args = parser.parse_args()

    run_c6v_substrate_optimization_pipeline(
        quick=args.quick,
        resolution=args.resolution,
        num_bands=args.num_bands,
        max_iterations=args.iterations,
        num_workers=args.workers,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
