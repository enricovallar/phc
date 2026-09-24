#!/usr/bin/env python3
"""End-to-End Demonstration: 2D Photonic Crystal Bayesian Optimization.

Demonstrates the modular optimization architecture:
1. Defines a parametric 2D C4v square lattice unit cell via GDSFactory (phc_layout)
   with arbitrary parameter names (r1, r2) and fixed parameters (pitch).
2. Connects to the BayesianOptimizer engine (phc_optimization) with the DiracDegeneracyObjective.
3. Automatically enforces physical dielectric slab continuity and minimum neck width.
4. Solves eigenbands at Gamma and analyzes point-group symmetries and group velocities (phc_mpb).
5. Exports optimal layout GDS (unit_cell.gds) and convergence plots into the canonical output directory.

Command-Line Usage:
    # Run full optimization with active learning and real-time progress bar:
    python examples/demo_optimization_2d.py

    # Rapid smoke test (< 3 seconds, low resolution, 3 evaluations):
    python examples/demo_optimization_2d.py --quick

    # Custom resolution and iteration limits:
    python examples/demo_optimization_2d.py --resolution 32 --initial-points 6 --max-iterations 8

    # Custom output directory:
    python examples/demo_optimization_2d.py --output-dir outputs/custom_optimization_run

    # Disable real-time progress bar:
    python examples/demo_optimization_2d.py --no-progress

    # Refine continuous degeneracy locus and compute adjacent group velocities:
    python examples/demo_optimization_2d.py --analyze-locus

CLI Options:
    --quick              Run in rapid smoke-test mode with minimal resolution and evaluations.
    --resolution RES     MPB computational mesh grid resolution per pitch unit a (default: 32, quick: 16).
    --num-bands N        Number of eigenbands to compute at the Gamma point (default: 8, quick: 4).
    --initial-points N   Number of quasi-random initial exploration evaluations (default: 6, quick: 2).
    --max-iterations N   Number of Bayesian optimization active learning generations (default: 4, quick: 1).
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


def make_c4v_unit_cell(
    r1: float = 0.25,
    r2: float = 0.15,
    pitch: float = 1.0,
    **kwargs: Any,
) -> gf.Component:
    """Generates a C4v square lattice photonic crystal unit cell with two hole radii.

    Holes are placed at Wyckoff position 1a (cell origin) with radius r1,
    and Wyckoff position 1b (cell center) with radius r2.

    Args:
        r1: Radius of primary hole at origin (micrometers).
        r2: Radius of secondary hole at center (micrometers).
        pitch: Lattice pitch a (micrometers).
        **kwargs: Extra unused keyword arguments passed by generic runners.

    Returns:
        GDSFactory Component containing the physical mask layout.
    """
    return phc_wyckoff_unit_cell(
        pitch=pitch,
        point_group="C4v",
        features=[("1a", r1), ("1b", r2)],
    )


def run_optimization_2d_pipeline(
    quick: bool = False,
    resolution: int | None = None,
    num_bands: int | None = None,
    initial_points: int | None = None,
    max_iterations: int | None = None,
    output_dir: Path | str | None = None,
    show_progress: bool = True,
    analyze_locus: bool = False,
) -> dict[str, Any]:
    """Runs the 2D Bayesian Optimization pipeline for accidental Dirac cone engineering.

    Args:
        quick: If True, executes a low-overhead smoke test (few points, low resolution)
            suitable for automated integration testing in < 3 seconds.
        resolution: MPB computational mesh resolution per pitch unit a (override default).
        num_bands: Number of eigenbands computed at Gamma (override default).
        initial_points: Number of initial quasi-random exploration points (override default).
        max_iterations: Number of active learning generations (override default).
        output_dir: Custom output directory or None to auto-resolve via phc_hydra.
        show_progress: Whether to display a real-time tqdm progress bar during search.
        analyze_locus: If True, executes 1D secant locus refinement and adjacent-point group velocity evaluation.

    Returns:
        Dictionary containing best parameters, best FOM, residual cost, and output directory.
    """
    if quick:
        default_resolution = 16
        default_num_bands = 8
        default_initial_points = 2
        default_max_iterations = 1
        default_batch_size = 1
        default_num_workers = 1
        bypass_irrep = True
        mode_indices = [4, 5, 6]
    else:
        default_resolution = 18
        default_num_bands = 10
        default_initial_points = 100
        default_max_iterations = 10
        default_batch_size = 4
        default_num_workers = 4
        bypass_irrep = False
        mode_indices = [4, 5, 6]

    res_val = resolution if resolution is not None else default_resolution
    bands_val = num_bands if num_bands is not None else default_num_bands
    init_pts = initial_points if initial_points is not None else default_initial_points
    max_iters = max_iterations if max_iterations is not None else default_max_iterations

    opt = BayesianOptimizer(
        cell_factory=make_c4v_unit_cell,
        parameters={
            "r1": (0.20, 0.35),
            "r2": (0.20, 0.35),
        },
        fixed_parameters={"pitch": 1.0},
        objective="dirac_degeneracy",
        objective_kwargs={
            "symmetry_group": "C4v",
            "polarization": "te",
            "target_irreps": ["A_2", "E", "E"],
            "irrep_occurrences": [2, 1, 1],
            "bypass_irrep_identification": bypass_irrep,
            "mode_indices": mode_indices,
            "min_band": 2,
            "degeneracy_tol": 0.005,
        },
        batch_size=default_batch_size,
        num_workers=default_num_workers,
        strategy="cl_min",
        lattice_type="square",
        pitch=1.0,
        dimension="2D",
        resolution=res_val,
        num_bands=bands_val,
        matrix_material="inp",
        background_material="air",
        enforce_connectivity=True,
        epsilon_threshold=1.1,
        min_neck_width_px=1,
        initial_points=init_pts,
        max_iterations=max_iters,
        output_dir=output_dir,
        geometry_name="c4v_dirac_2d",
        random_state=42,
        show_progress=show_progress,
    )

    result = opt.run(show_progress=show_progress)
    if not quick:
        opt.run_best()

    locus_results = []
    if analyze_locus:
        locus_results = opt.analyze_locus(
            delta_k=0.01,
            exclude_unrefined=False,  # Keeps all 40 refined points
            max_residual_gap=1e-4,
            max_refine_steps=12,
        )

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
    """CLI entrypoint for running the 2D PhC Bayesian Optimization demonstration."""
    parser = argparse.ArgumentParser(
        description="2D Photonic Crystal Bayesian Optimization — Accidental Dirac Cone Engineering.",
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
        help="MPB mesh resolution per pitch unit a (default: 32, quick: 16).",
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
    print(" 2D Photonic Crystal Bayesian Optimization Demo ")
    print("=" * 70)
    res = run_optimization_2d_pipeline(
        quick=args.quick,
        resolution=args.resolution,
        num_bands=args.num_bands,
        initial_points=args.initial_points,
        max_iterations=args.max_iterations,
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
