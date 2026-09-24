#!/usr/bin/env python3
"""Bayesian Optimization of the C6v 2b-6d Photonic Crystal Unit Cell.

Optimizes the geometric parameters of the C6v hexagonal lattice unit cell
(Wyckoff 2b holes with radius r1 and Wyckoff 6d holes with radius r2 and parameter p2)
etched in an hBN membrane (or 2D periodic sheet) for accidental Dirac cone degeneracy at Γ:

1. Defines a parametric C6v hexagonal lattice unit cell with Wyckoff 2b and 6d holes.
2. Connects to the BayesianOptimizer engine with the DiracDegeneracyObjective.
3. Automatically enforces physical dielectric continuity and connectivity.
4. Solves eigenbands at Γ and tracks mode degeneracies (phc_mpb).
5. Exports optimal layout GDS (unit_cell.gds), convergence plots, and optimal loci into
   the canonical output directory.

Command-Line Usage:
    # Standard optimization run (2D, 10 initial points, 10 iterations):
    python analysis/optimize_c6v_2b_6d.py

    # Rapid smoke test (< 5 seconds, low resolution, 2 initial points, 1 iteration):
    python analysis/optimize_c6v_2b_6d.py --quick

    # 3D slab membrane optimization (with finite thickness, e.g. 0.5 um):
    python analysis/optimize_c6v_2b_6d.py --slab-thickness 0.5

    # Vary also the 6d radial position parameter p2 (3-parameter search: r1, r2, p2):
    python analysis/optimize_c6v_2b_6d.py --vary-p2

    # Custom resolution and iteration limits:
    python analysis/optimize_c6v_2b_6d.py --resolution 24 --initial-points 15 --max-iterations 15

    # Refine continuous degeneracy locus and compute group velocities:
    python analysis/optimize_c6v_2b_6d.py --analyze-locus

CLI Options:
    --quick              Run in rapid smoke-test mode with minimal resolution and evaluations.
    --slab-thickness H   Membrane slab thickness in micrometers (default: None for 2D).
    --vary-p2            Optimize 6d position parameter p2 in addition to radii r1 and r2.
    --polarization POL   Polarization mode: 'tm' or 'te' (default: 'tm').
    --resolution RES     In-plane MPB computational mesh resolution per pitch a (default: 20, quick: 12).
    --resolution-z RESZ  Vertical MPB mesh resolution for 3D slabs (default: 16, quick: 6).
    --num-bands N        Number of eigenbands to compute at Gamma (default: 8, quick: 6).
    --initial-points N   Number of initial quasi-random exploration points (default: 10, quick: 2).
    --max-iterations N   Number of Bayesian optimization active learning generations (default: 10, quick: 1).
    --workers W          Number of parallel worker processes (default: 4, quick: 1).
    --output-dir PATH    Custom output directory override (default: auto-resolved by phc_hydra).
    --no-progress        Disable the real-time tqdm progress bar.
    --analyze-locus      Refine continuous degeneracy locus curve and compute adjacent group velocity.
"""

import argparse
import sys
import warnings
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
    p2: float = 0.15,
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
    slab_thickness: float | None = None,
    supercell_z: float = 4.0,
    vary_p2: bool = False,
    polarization: str = "tm",
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
    mode_indices: tuple[int, int, int] = (2, 3, 4),
) -> dict[str, Any]:
    """Runs the Bayesian Optimization pipeline for the C6v 2b-6d unit cell.

    Args:
        quick: If True, executes a rapid low-resolution smoke test (<5 s).
        slab_thickness: Physical membrane thickness in micrometers (None for 2D PhC).
        supercell_z: Vertical supercell height in units of lattice constant a (default: 4.0).
        vary_p2: If True, includes 6d radial position parameter p2 in optimization bounds.
        polarization: Target mode polarization ('tm' or 'te').
        resolution: In-plane MPB computational mesh resolution per pitch a.
        resolution_z: Vertical MPB computational mesh resolution along z for slabs.
        num_bands: Number of eigenbands computed at Gamma.
        initial_points: Number of initial quasi-random exploration points.
        max_iterations: Number of active learning generations.
        batch_size: Candidate points evaluated per generation.
        num_workers: Number of parallel worker processes.
        matrix_material: Slab/matrix dielectric material key (default: "hBN").
        cladding_material: Cladding material key (default: "air").
        output_dir: Custom output directory or None to auto-resolve via phc_hydra.
        show_progress: Whether to display a real-time progress bar.
        analyze_locus: If True, refines the continuous degeneracy locus manifold.
        mode_indices: 1-based indices of target degenerate bands at Gamma (default: [2, 3, 4]).

    Returns:
        Dictionary containing best parameters, best FOM, residual cost, and output directory.
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
        default_num_bands = 8
        default_initial_points = 10
        default_max_iterations = 10
        default_batch_size = 4
        default_num_workers = 4
        bypass_irrep = True

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
            "r1": (0.08, 0.22),
            "r2": (0.05, 0.16),
            "p2": (0.12, 0.25),
        }
        fixed_params: dict[str, Any] = {"pitch": 1.0}
    else:
        search_params = {
            "r1": (0.08, 0.22),
            "r2": (0.05, 0.16),
        }
        fixed_params = {"pitch": 1.0, "p2": 0.15}

    if is_3d:
        fixed_params["slab_thickness"] = float(slab_thickness)  # type: ignore[arg-type]
        fixed_params["supercell_z"] = float(supercell_z)

    opt = BayesianOptimizer(
        cell_factory=make_c6v_2b_6d_unit_cell,
        parameters=search_params,
        fixed_parameters=fixed_params,
        objective="dirac_degeneracy",
        objective_kwargs={
            "symmetry_group": "C6v",
            "polarization": pol_key,
            "target_irreps": ["A_1", "E_1", "E_1"],
            "irrep_occurrences": [1, 1, 1],
            "bypass_irrep_identification": bypass_irrep,
            "mode_indices": list(mode_indices),
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
        opt.run_best()

    locus_results = []
    if analyze_locus:
        locus_results = opt.analyze_locus(
            delta_k=0.01,
            exclude_unrefined=False,
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
        default=None,
        help="Membrane slab thickness in micrometers (default: None for 2D PhC).",
    )
    parser.add_argument(
        "--vary-p2",
        action="store_true",
        help="Optimize 6d position parameter p2 in addition to radii r1 and r2.",
    )
    parser.add_argument(
        "--polarization",
        type=str,
        default="tm",
        choices=["tm", "te", "tm_like", "te_like"],
        help="Target mode polarization.",
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
        help="Number of eigenbands to compute at Gamma (default: 8, quick: 6).",
    )
    parser.add_argument(
        "--initial-points",
        type=int,
        default=None,
        help="Number of initial quasi-random exploration points (default: 10, quick: 2).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Number of Bayesian optimization active learning generations (default: 10, quick: 1).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes (default: 4, quick: 1).",
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
    args = parser.parse_args()

    print("=" * 72)
    print(" C6v 2b-6d Photonic Crystal Bayesian Optimization ")
    dim_str = (
        f"3D Slab (h={args.slab_thickness} μm)"
        if args.slab_thickness is not None
        else "2D Periodic Sheet"
    )
    print(f" Structure: {dim_str} in hBN, Polarization: {args.polarization.upper()}")
    print("=" * 72)

    res = run_c6v_optimization_pipeline(
        quick=args.quick,
        slab_thickness=args.slab_thickness,
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
    print(f"  Output Folder:   {res['output_dir']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
