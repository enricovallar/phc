#!/usr/bin/env python3
"""End-to-End Demonstration: 3D Photonic Crystal Slab Bayesian Optimization.

Demonstrates the modular optimization architecture for 3D slab membranes:
1. Defines a parametric 3D C4v square lattice PhC slab via GDSFactory (phc_layout)
   with arbitrary parameter names (r1, r2) and fixed parameters (pitch, slab_thickness).
2. Connects to the BayesianOptimizer engine (phc_optimization) with DiracDegeneracyObjective
   in 3D_slab mode with guided TE-like mode parity (even mirror symmetry sigma_z = +1).
3. Automatically enforces 3D physical dielectric slab continuity and minimum neck width across PBC.
4. Solves eigenbands at Gamma and analyzes point-group symmetries and group velocities (phc_mpb).
5. Exports optimal layout GDS (unit_cell.gds), convergence curves, surrogate map, per-evaluation
   log files (bo_evaluations.log, bo_evaluations.jsonl, bo_evaluations.json), and post-run
   optimal 3D permittivity cross-sections (best_epsilon.png) and band diagram (best_band_structure.png).

Command-Line Usage:
    # Run full 3D optimization with batched active learning and multi-worker parallelism:
    python examples/demo_optimization_3d.py

    # Rapid smoke test (< 3 seconds, low resolution, 3 evaluations):
    python examples/demo_optimization_3d.py --quick

    # Custom in-plane and vertical mesh resolution:
    python examples/demo_optimization_3d.py --resolution 20 --resolution-z 10

    # Custom iteration and worker limits:
    python examples/demo_optimization_3d.py --initial-points 6 --max-iterations 4 --workers 4

    # Custom output directory:
    python examples/demo_optimization_3d.py --output-dir outputs/custom_opt_3d

    # Disable real-time progress bar:
    python examples/demo_optimization_3d.py --no-progress

    # Refine continuous degeneracy locus and compute adjacent group velocities:
    python examples/demo_optimization_3d.py --analyze-locus

CLI Options:
    --quick              Run in rapid smoke-test mode with minimal resolution and evaluations.
    --resolution RES     In-plane (x, y) mesh resolution per unit pitch a (default: 20, quick: 12).
    --resolution-z RESZ  Vertical (z) mesh resolution per unit pitch a (default: 10, quick: 6).
    --supercell-z HEIGHT Vertical supercell height in units of pitch a (default: 4.0).
    --num-bands N        Number of eigenbands to compute at Gamma (default: 8, quick: 4).
    --initial-points N   Number of quasi-random initial exploration points (default: 6, quick: 2).
    --max-iterations N   Number of Bayesian optimization active learning generations (default: 4, quick: 1).
    --batch-size B       Candidate points proposed per generation via Constant Liar (default: 4, quick: 1).
    --workers W          Number of parallel worker processes for candidate evaluations (default: 4, quick: 1).
    --output-dir PATH    Custom output directory override (default: auto-resolved by phc_hydra).
    --no-progress        Disable the real-time tqdm progress bar.
    --analyze-locus      Refine continuous degeneracy locus curve and compute adjacent group velocity.
"""

import argparse
import warnings
from pathlib import Path
from typing import Any

import gdsfactory as gf
import meep as mp
from phc_layout.components.unit_cell import phc_wyckoff_unit_cell
from phc_optimization import BayesianOptimizer

# Suppress MPB solver chatter and repetitive Sobol balance warnings
mp.verbosity(0)
warnings.filterwarnings("ignore", category=UserWarning, module="skopt")
warnings.filterwarnings("ignore", message=".*balance properties of Sobol.*")


def make_c4v_slab_unit_cell(
    r1: float = 0.25,
    r2: float = 0.15,
    pitch: float = 1.0,
    slab_thickness: float = 0.5,
) -> gf.Component:
    """Generates a C4v square lattice photonic crystal unit cell for a 3D slab membrane.

    Holes are placed at Wyckoff position 1a (cell origin) with radius r1,
    and Wyckoff position 1b (cell center) with radius r2.

    Args:
        r1: Radius of primary hole at origin (micrometers).
        r2: Radius of secondary hole at center (micrometers).
        pitch: Lattice pitch a (micrometers).
        slab_thickness: Slab membrane thickness (micrometers).

    Returns:
        GDSFactory Component containing the physical mask layout.
    """
    return phc_wyckoff_unit_cell(
        pitch=pitch,
        point_group="C4v",
        features=[("1a", r1), ("1b", r2)],
    )


def run_optimization_3d_pipeline(
    quick: bool = False,
    resolution: int | None = None,
    resolution_z: int | None = None,
    supercell_z: float | None = None,
    num_bands: int | None = None,
    initial_points: int | None = None,
    max_iterations: int | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    output_dir: Path | str | None = None,
    show_progress: bool = True,
    analyze_locus: bool = False,
) -> dict[str, Any]:
    """Runs the 3D PhC slab Bayesian Optimization pipeline for Dirac cone engineering.

    Args:
        quick: If True, executes a low-overhead smoke test (few points, low resolution)
            suitable for automated integration testing in < 3 seconds.
        resolution: In-plane MPB computational mesh resolution per pitch unit a.
        resolution_z: Vertical MPB computational mesh resolution along z.
        supercell_z: Vertical supercell height in units of lattice constant a (default: 4.0).
        num_bands: Number of eigenbands computed at Gamma.
        initial_points: Number of initial quasi-random exploration points.
        max_iterations: Number of active learning generations.
        batch_size: Candidate points evaluated per generation.
        num_workers: Number of parallel worker processes for candidate evaluation.
        output_dir: Custom output directory or None to auto-resolve via phc_hydra.
        show_progress: Whether to display a real-time tqdm progress bar during search.
        analyze_locus: If True, executes 1D secant locus refinement and adjacent-point group velocity evaluation.

    Returns:
        Dictionary containing best parameters, best FOM, residual cost, and output directory.
    """
    if quick:
        default_res_xy = 12
        default_res_z = 6
        default_num_bands = 4
        default_initial_points = 2
        default_max_iterations = 1
        default_batch_size = 1
        default_num_workers = 1
        bypass_irrep = True
        mode_indices = [2, 3, 4]
        target_irreps = ["A_2", "E", "E"]
        irrep_occurrences = [1, 1, 1]
    else:
        default_res_xy = 16
        default_res_z = 20
        default_num_bands = 10
        default_initial_points = 40
        default_max_iterations = 3
        default_batch_size = 4
        default_num_workers = 4
        bypass_irrep = True
        mode_indices = [8, 9, 10]
        target_irreps = ["A_2", "E", "E"]
        irrep_occurrences = [1, 1, 1]

    r_xy = resolution if resolution is not None else default_res_xy
    r_z = resolution_z if resolution_z is not None else default_res_z
    sz = supercell_z if supercell_z is not None else 4.0
    bands_val = num_bands if num_bands is not None else default_num_bands
    init_pts = initial_points if initial_points is not None else default_initial_points
    max_iters = max_iterations if max_iterations is not None else default_max_iterations
    b_sz = batch_size if batch_size is not None else default_batch_size
    n_workers = num_workers if num_workers is not None else default_num_workers

    # 3D grid resolution tuple (rx, ry, rz)
    grid_resolution = (r_xy, r_xy, r_z)

    opt = BayesianOptimizer(
        cell_factory=make_c4v_slab_unit_cell,
        parameters={
            "r1": (0.2, 0.35),
            "r2": (0.2, 0.35),
        },
        fixed_parameters={
            "pitch": 1.0,
            "slab_thickness": 0.5,
            "supercell_z": sz,
        },
        objective="dirac_degeneracy",
        objective_kwargs={
            "symmetry_group": "C4v",
            "polarization": "te_like",
            "target_irreps": target_irreps,
            "irrep_occurrences": irrep_occurrences,
            "bypass_irrep_identification": bypass_irrep,
            "mode_indices": mode_indices,
            "min_band": 2,
            "degeneracy_tol": 0.005,
        },
        batch_size=b_sz,
        num_workers=n_workers,
        strategy="cl_min",
        lattice_type="square",
        pitch=1.0,
        dimension="3D_slab",
        supercell_z=sz,
        resolution=grid_resolution,
        num_bands=bands_val,
        matrix_material="inp",
        background_material="air",
        enforce_connectivity=True,
        epsilon_threshold=1.1,
        min_neck_width_px=1,
        initial_points=init_pts,
        max_iterations=max_iters,
        output_dir=output_dir,
        geometry_name="c4v_dirac_3d",
        random_state=42,
        show_progress=show_progress,
    )

    result = opt.run(show_progress=show_progress)

    # Post-optimization characterization for full runs
    if not quick:
        opt.run_best(num_workers=n_workers, k_density=12)

    locus_results = []
    if analyze_locus:
        locus_results = opt.analyze_locus(delta_k=0.01)

    loci_file = result.output_dir / "optimal_loci.json"
    loci_data = []
    if loci_file.is_file():
        import json

        with open(loci_file, encoding="utf-8") as f:
            loci_data = json.load(f)

    return {
        "best_params": result.best_params,
        "best_fom": result.best_fom,
        "best_cost": result.best_cost,
        "total_evaluations": len(result.records),
        "optimal_loci": loci_data,
        "refined_loci": locus_results,
        "output_dir": str(result.output_dir),
    }


def main() -> None:
    """CLI entrypoint for running the 3D PhC Slab Bayesian Optimization demonstration."""
    parser = argparse.ArgumentParser(
        description="3D Photonic Crystal Slab Bayesian Optimization — Accidental Dirac Cone Engineering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run in rapid smoke-test mode with low resolution and minimal iterations.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="In-plane MPB mesh resolution per pitch unit a (default: 20, quick: 12).",
    )
    parser.add_argument(
        "--resolution-z",
        type=int,
        default=None,
        help="Vertical MPB mesh resolution along z (default: 10, quick: 6).",
    )
    parser.add_argument(
        "--supercell-z",
        type=float,
        default=None,
        help="Vertical supercell height in units of pitch a (default: 4.0).",
    )
    parser.add_argument(
        "--num-bands",
        type=int,
        default=None,
        help="Number of eigenbands to compute at Gamma (default: 8, quick: 4).",
    )
    parser.add_argument(
        "--initial-points",
        type=int,
        default=None,
        help="Number of initial quasi-random exploration points (default: 6, quick: 2).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Number of Bayesian optimization active learning generations (default: 4, quick: 1).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Candidate points proposed per generation (default: 4, quick: 1).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes for candidate evaluation (default: 4, quick: 1).",
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
        help="Disable the real-time tqdm progress bar.",
    )
    parser.add_argument(
        "--analyze-locus",
        action="store_true",
        help="Refine continuous degeneracy locus curve and compute adjacent group velocity.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print(" 3D Photonic Crystal Slab Bayesian Optimization Demo ")
    print("=" * 70)
    res = run_optimization_3d_pipeline(
        quick=args.quick,
        resolution=args.resolution,
        resolution_z=args.resolution_z,
        supercell_z=args.supercell_z,
        num_bands=args.num_bands,
        initial_points=args.initial_points,
        max_iterations=args.max_iterations,
        batch_size=args.batch_size,
        num_workers=args.workers,
        output_dir=args.output_dir,
        show_progress=not args.no_progress,
        analyze_locus=args.analyze_locus,
    )
    print("\n" + "=" * 70)
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
    print(f"  Output Folder:   {res['output_dir']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
