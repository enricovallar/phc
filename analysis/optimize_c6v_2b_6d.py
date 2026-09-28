#!/usr/bin/env python3
"""Bayesian Optimization of the C6v 2b-6d Photonic Crystal Unit Cell.

Optimizes the geometric parameters of the C6v hexagonal lattice unit cell
(Wyckoff 2b holes with radius r1 and Wyckoff 6d holes with radius r2 and parameter p2)
etched in an hBN membrane (or 2D periodic sheet) for accidental Dirac cone degeneracy at Γ:

1. Defines a parametric C6v hexagonal lattice unit cell with Wyckoff 2b and 6d holes.
2. Connects to the BayesianOptimizer engine with the DiracDegeneracyObjective.
3. Automatically enforces physical dielectric continuity and connectivity.
4. Solves eigenbands at Γ and tracks mode degeneracies (phc_mpb).
5. Supports scale-invariant formulation (e.g. h/a = 0.1) and physical mapping to target
   wavelength (e.g. λ = 436 nm) and thickness (e.g. h = 100 nm).
6. Exports optimal layout GDS (unit_cell.gds), convergence plots, and optimal loci into
   the canonical output directory.

Command-Line Usage:
    # 100-point initial exploration (h/a=0.1, te_like, 16 workers, target A2 + E1 + E1):
    python analysis/optimize_c6v_2b_6d.py --initial-points 100 --slab-thickness 0.1 --workers 16

    # Rapid smoke test (< 5 seconds, low resolution, 2 initial points, 1 iteration):
    python analysis/optimize_c6v_2b_6d.py --quick

    # 3D slab membrane optimization with custom target irreps and occurrences:
    python analysis/optimize_c6v_2b_6d.py --target-irreps A_2 E_1 E_1 --irrep-occurrences 1 4 4

    # Refine continuous degeneracy locus and compute group velocities:
    python analysis/optimize_c6v_2b_6d.py --analyze-locus

CLI Options:
    --quick                 Run in rapid smoke-test mode with minimal resolution and evaluations.
    --slab-thickness H      Membrane slab thickness in units of a (default: 0.1 for h/a=0.1).
    --supercell-z Z         Supercell height in units of a (default: 4.0).
    --vary-p2               Optimize 6d position parameter p2 in addition to radii r1 and r2.
    --polarization POL      Polarization mode: 'te_like' or 'tm_like' (default: 'te_like').
    --target-irreps IRREPS  Target irreducible representations at Gamma (default: A_2 E_1 E_1).
    --irrep-occurrences OCC Occurrence counts above min_band (default: 1 4 4).
    --target-wavelength NM  Target physical wavelength in nm (default: 436.0).
    --r1-bounds MIN MAX     Search bounds for primary hole radius r1 in units of a (default: 0.05 0.25).
    --r2-bounds MIN MAX     Search bounds for satellite hole radius r2 in units of a (default: 0.02 0.10).
    --resolution RES        In-plane MPB computational mesh resolution per pitch a (default: 18, quick: 12).
    --resolution-z RESZ     Vertical MPB mesh resolution for 3D slabs (default: 16, quick: 6).
    --num-bands N           Number of eigenbands to compute at Gamma (default: 15, quick: 6).
    --initial-points N      Number of initial quasi-random exploration points (default: 100, quick: 2).
    --max-iterations N      Number of Bayesian optimization active learning generations (default: 0, quick: 1).
    --workers W             Number of parallel worker processes (default: 20, quick: 1).
    --output-dir PATH       Custom output directory override (default: auto-resolved by phc_hydra).
    --no-progress           Disable the real-time tqdm progress bar.
    --analyze-locus         Refine continuous degeneracy locus curve and compute adjacent group velocity.
"""

import argparse
import json
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Ensure repository root is in sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import gdsfactory as gf
import meep as mp
from phc_layout.components.unit_cell import phc_wyckoff_unit_cell
from phc_layout.lattice import HexagonalLattice
from phc_optimization import BayesianOptimizer

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


def run_c6v_optimization_pipeline(
    quick: bool = False,
    slab_thickness: float | None = 0.3,
    supercell_z: float = 4.0,
    p2: float = 0.25,
    vary_p2: bool = False,
    polarization: str = "te_like",
    resolution: int | None = None,
    resolution_z: int | None = None,
    num_bands: int | None = None,
    initial_points: int | None = None,
    max_iterations: int | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    matrix_material: str = "hBN",
    cladding_material: str = "air",
    output_dir: Path | str | None = None,
    show_progress: bool = True,
    analyze_locus: bool = False,
    mode_indices: tuple[int, int, int] = (9, 10, 11),
    target_irreps: Sequence[str] = ("A_2", "E_1", "E_1"),
    irrep_occurrences: Sequence[int] = (1, 4, 4),
    target_wavelength_nm: float = 436.0,
    target_thickness_nm: float = 100.0,
    r1_bounds: tuple[float, float] = (0.15, 0.25),
    r2_bounds: tuple[float, float] = (0.05, 0.10),
    locus_mode: str = "cartesian",
) -> dict[str, Any]:
    """Runs the Bayesian Optimization pipeline for the C6v 2b-6d unit cell.

    Args:
        quick: If True, executes a rapid low-resolution smoke test (<5 s).
        slab_thickness: Normalized membrane thickness h/a (default: 0.3, None for 2D PhC).
        supercell_z: Vertical supercell height in units of lattice constant a (default: 4.0).
        p2: Coordinate parameter for Wyckoff position 6d satellite holes (default: 0.25).
        vary_p2: If True, includes 6d radial position parameter p2 in optimization bounds.
        polarization: Target mode polarization ('te_like', 'tm_like', 'te', or 'tm').
        resolution: In-plane MPB computational mesh resolution per pitch a.
        resolution_z: Vertical MPB computational mesh resolution along z for slabs.
        num_bands: Number of eigenbands computed at Gamma.
        initial_points: Number of initial quasi-random exploration points (default: 50, quick: 2).
        max_iterations: Number of active learning generations.
        batch_size: Candidate points evaluated per generation.
        num_workers: Number of parallel worker processes.
        matrix_material: Slab/matrix dielectric material key (default: "hBN").
        cladding_material: Cladding material key (default: "air").
        output_dir: Custom output directory or None to auto-resolve via phc_hydra.
        show_progress: Whether to display a real-time progress bar.
        analyze_locus: If True, refines the continuous degeneracy locus manifold.
        mode_indices: 1-based indices of target degenerate bands at Gamma (default: [9, 10, 11]).
        target_irreps: Target irreducible representations at Gamma (default: ('A_2', 'E_1', 'E_1')).
        irrep_occurrences: Occurrence index per target irrep above min_band (default: (1, 4, 4)).
        target_wavelength_nm: Target physical wavelength in nanometers (default: 436.0).
        target_thickness_nm: Target physical slab thickness in nanometers (default: 100.0).
        r1_bounds: Search range for primary hole radius r1 in units of a (default: (0.15, 0.25)).
        r2_bounds: Search range for satellite hole radius r2 in units of a (default: (0.05, 0.10)).
        locus_mode: Mode for degeneracy locus refinement ('cartesian' or 'polar' or 'auto').


    Returns:
        Dictionary containing best parameters, best FOM, residual cost, optimal loci,
        physical scaling analysis, and output directory.
    """
    is_3d = slab_thickness is not None
    dim = "3D_slab" if is_3d else "2D"

    # Map polarization key for 2D vs 3D slab
    if is_3d:
        pol_key = "tm_like" if polarization.lower().startswith("tm") else "te_like"
    else:
        pol_key = "tm" if polarization.lower().startswith("tm") else "te"

    if quick:
        default_resolution = 12
        default_resolution_z = 6
        default_num_bands = 6
        default_initial_points = 2
        default_max_iterations = 1
        default_batch_size = 1
        default_num_workers = 1
        bypass_irrep = True
    else:
        default_resolution = 18
        default_resolution_z = 16
        default_num_bands = 15
        default_initial_points = 100
        default_max_iterations = 0
        default_batch_size = 4
        default_num_workers = 20
        bypass_irrep = False

    res_val = resolution if resolution is not None else default_resolution
    res_z_val = resolution_z if resolution_z is not None else default_resolution_z
    bands_val = num_bands if num_bands is not None else default_num_bands
    init_pts = initial_points if initial_points is not None else default_initial_points
    max_iters = max_iterations if max_iterations is not None else default_max_iterations
    b_size = batch_size if batch_size is not None else default_batch_size
    n_workers = num_workers if num_workers is not None else default_num_workers

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

    resolved_indices = list(mode_indices)
    if quick or bands_val < max(resolved_indices):
        resolved_indices = [2, 3, 4]
        bypass_irrep = True

    opt = BayesianOptimizer(
        cell_factory=make_c6v_2b_6d_unit_cell,
        parameters=search_params,
        fixed_parameters=fixed_params,
        objective="dirac_degeneracy",
        objective_kwargs={
            "symmetry_group": "C6v",
            "polarization": pol_key,
            "target_irreps": list(target_irreps),
            "irrep_occurrences": list(irrep_occurrences),
            "bypass_irrep_identification": bypass_irrep,
            "mode_indices": resolved_indices,
            "min_band": 2,
            "degeneracy_tol": 0.005,
        },
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
        enforce_connectivity=True,
        epsilon_threshold=1.1,
        min_neck_width_px=1,
        initial_points=init_pts,
        max_iterations=max_iters,
        output_dir=output_dir,
        geometry_name=f"c6v_2b_6d_{dim.lower()}",
        random_state=42,
        show_progress=show_progress,
    )

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
        )

    locus_results = []
    if analyze_locus:
        locus_results = opt.analyze_locus(
            delta_k=0.01,
            exclude_unrefined=False,
            max_residual_gap=1e-4,
            max_refine_steps=12 if not quick else 2,
            mode=locus_mode,
            num_workers=n_workers,
            target_wavelength_nm=target_wavelength_nm,
            target_thickness_nm=target_thickness_nm,
            sample_points=20 if not quick else 3,
            plot_match_bands=not quick,
            k_density_match=20 if not quick else 6,
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

    return {
        "best_params": result.best_params,
        "best_fom": result.best_fom,
        "best_cost": result.best_cost,
        "total_evaluations": len(result.records),
        "optimal_loci": loci_data,
        "refined_loci": locus_results,
        "physical_analysis": physical_analysis,
        "output_dir": str(result.output_dir),
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
        help="Membrane slab thickness h in units of pitch a (default: 0.3 for h/a=0.3).",
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
        "--polarization",
        type=str,
        default="te_like",
        choices=["tm", "te", "tm_like", "te_like"],
        help="Target mode polarization.",
    )
    parser.add_argument(
        "--target-irreps",
        nargs="+",
        default=["A_2", "E_1", "E_1"],
        help="Target irreducible representations at Gamma (default: A_2 E_1 E_1).",
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
        help="Number of eigenbands to compute at Gamma (default: 15, quick: 6).",
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
        "--analyze-locus",
        action="store_true",
        help="Refine continuous degeneracy locus curve and compute adjacent group velocity.",
    )

    parser.add_argument(
        "--locus-mode",
        type=str,
        default="cartesian",
        choices=["cartesian", "polar", "auto"],
        help="Mode for degeneracy locus refinement (default: cartesian).",
    )
    args = parser.parse_args()

    print("=" * 72)
    print(" C6v 2b-6d Photonic Crystal Bayesian Optimization ")
    dim_str = (
        f"3D Slab (h/a={args.slab_thickness})"
        if args.slab_thickness is not None
        else "2D Periodic Sheet"
    )
    print(f" Structure: {dim_str} in hBN, Polarization: {args.polarization.upper()}")
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
        num_workers=args.workers,
        output_dir=args.output_dir,
        show_progress=not args.no_progress,
        analyze_locus=args.analyze_locus,
        locus_mode=args.locus_mode,
        target_irreps=args.target_irreps,
        irrep_occurrences=args.irrep_occurrences,
        target_wavelength_nm=args.target_wavelength,
        target_thickness_nm=args.target_thickness,
        r1_bounds=tuple(args.r1_bounds),
        r2_bounds=tuple(args.r2_bounds),
    )

    print("\n" + "=" * 72)
    print(" Optimization Complete! ")
    print(f"  Best Parameters: {res['best_params']}")
    print(f"  Best FOM:        {res['best_fom']:.2f}")
    print(f"  Residual Cost:   {res['best_cost']:.6f}")
    print(f"  Evaluations:     {res['total_evaluations']}")
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


if __name__ == "__main__":
    main()
