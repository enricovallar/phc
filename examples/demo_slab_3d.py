#!/usr/bin/env python3
"""End-to-End Demonstration: Simple 3D Photonic Crystal Slab Simulation in MPB.

Demonstrates the canonical workflow for a 3D suspended photonic crystal slab membrane,
using the identical slab configuration and default parameters from the 3D Dirac
optimization pipeline (C4v square lattice, InP slab membrane, r1=0.25, r2=0.15, h=0.5):

1. Generates the 3D C4v square lattice PhC unit cell with gdsfactory (``phc_layout``).
2. Exports the physical GDSII layout mask (``unit_cell.gds``) via ``phc_hydra``.
3. Queries optical material properties from ``phc_materials`` (InP slab, n ≈ 3.17; Air cladding, n = 1.0).
4. Extrudes the 3D dielectric slab membrane and etch hole cylinders in ``phc_mpb``.
5. Solves for TE-like guided slab modes along the irreducible Brillouin zone path (X -> Γ -> M -> X).
6. Plots the 3D dielectric permittivity map (in-plane xy mid-plane and vertical xz cross-section).
7. Plots the photonic band structure featuring discrete dots, cladding light line, and band gaps.
8. Exports structured simulation metadata to ``simulation_results.json``.

Command-Line Usage:
    # Standard execution with default parameters (no arguments required):
    python examples/demo_slab_3d.py

    # Rapid smoke test (< 2 seconds):
    python examples/demo_slab_3d.py --quick

    # Custom mesh resolution and worker count:
    python examples/demo_slab_3d.py --resolution 20 --resolution-z 20 --workers 4

    # Interactive visualization preview:
    python examples/demo_slab_3d.py --show

CLI Options:
    --quick              Run in rapid smoke-test mode with minimal resolution and bands.
    --resolution RES     In-plane (x, y) mesh resolution per unit pitch a (default: 16, quick: 12).
    --resolution-z RESZ  Vertical (z) mesh resolution per unit pitch a (default: 16, quick: 6).
    --num-bands N        Number of eigenbands to compute at each k-point (default: 8, quick: 4).
    --k-density K        Interpolation density between high-symmetry vertices (default: 6, quick: 2).
    --workers W          Number of concurrent worker processes for MPB (default: 4, quick: 1).
    --output-dir PATH    Custom output directory override (default: auto-resolved by phc_hydra).
    --show               Display interactive Matplotlib preview windows after simulation.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import meep as mp

# Ensure repository root is in sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from phc_hydra import SimulationOutputManager, resolve_simulation_output_dir
from phc_layout.components.unit_cell import phc_wyckoff_unit_cell
from phc_materials import get_material
from phc_mpb import (
    create_lattice,
    create_mode_solver,
    gds_to_mpb_geometry,
    get_epsilon_grid,
    get_high_symmetry_kpath,
    plot_band_structure,
    plot_epsilon,
    run_band_solver,
)
from phc_utils import silence_c_stdout

# Suppress Meep/MPB C-level verbosity by default
mp.verbosity(0)


def run_slab_3d_pipeline(
    pitch: float = 1.0,
    slab_thickness: float = 0.5,
    r1: float = 0.25,
    r2: float = 0.15,
    supercell_z: float = 4.0,
    slab_material: str = "inp",
    cladding_material: str = "air",
    resolution: int = 16,
    resolution_z: int = 16,
    num_bands: int = 8,
    k_density: int = 6,
    num_workers: int = 4,
    quick: bool = False,
    output_dir: Path | str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Runs the end-to-end 3D PhC slab simulation pipeline connecting layout, materials, and MPB.

    Uses the identical slab geometry and default parameters as the 3D Dirac optimization example:
    C4v square lattice unit cell with Wyckoff 1a hole (r1) at the origin and Wyckoff 1b hole (r2)
    at the cell center, etched into an InP membrane suspended in air.

    Args:
        pitch: Lattice constant a in micrometers (default: 1.0 μm).
        slab_thickness: Physical slab membrane thickness in micrometers (default: 0.5 μm, h/a = 0.5).
        r1: Radius of primary hole at Wyckoff position 1a (origin) in micrometers (default: 0.25 μm).
        r2: Radius of secondary hole at Wyckoff position 1b (center) in micrometers (default: 0.15 μm).
        supercell_z: Vertical supercell height in units of lattice constant a (default: 4.0).
        slab_material: Material key for dielectric slab core from phc_materials (default: "inp").
        cladding_material: Material key for background cladding from phc_materials (default: "air").
        resolution: In-plane (x, y) mesh resolution per unit pitch a (default: 16).
        resolution_z: Vertical (z) mesh resolution per unit pitch a (default: 16).
        num_bands: Number of eigenbands to compute at each k-point (default: 8).
        k_density: Interpolation density between high-symmetry vertices (default: 6).
        num_workers: Number of concurrent worker processes for MPB (default: 4).
        quick: If True, overrides resolution, k_density, and num_bands to minimal settings
            (res=12, res_z=6, num_bands=4, k_density=2, workers=1) for rapid smoke testing (<2 s).
        output_dir: Custom directory override. If None, auto-resolved via phc_hydra:
            outputs/mpb/slab_band_diagram/c4v_dirac_3d/<timestamp>/.
        verbose: If True, prints formatted progress updates to stdout.

    Returns:
        Dictionary containing:
            - 'results': MPB solver results including frequencies, gaps, and light line.
            - 'ms': Initialized MPB ModeSolver instance.
            - 'labels': High-symmetry k-path point labels.
            - 'indices': High-symmetry k-path tick indices.
            - 'eps_grid': Extracted 3D dielectric permittivity array.
            - 'output_dir': Resolved Path to the output directory.
            - 'files': Dict mapping artifact keys ('gds', 'band_plot', 'eps_plot', 'results_json')
              to saved file paths.
            - 'summary': Complete structured simulation results summary.

    Raises:
        RuntimeError: If MPB solver fails to produce frequency results.
    """
    t_start = time.time()

    if quick:
        resolution = 12
        resolution_z = 6
        num_bands = 4
        k_density = 2
        num_workers = 1

    # Resolve output directory according to phc_hydra SSOT standard
    out_path = resolve_simulation_output_dir(
        solver="mpb",
        sim_type="slab_band_diagram",
        geometry="c4v_dirac_3d",
        override_dir=output_dir,
    )
    output_mgr = SimulationOutputManager(
        output_dir=out_path,
        solver="mpb",
        sim_type="slab_band_diagram",
        geometry_name="c4v_dirac_3d",
    )

    if verbose:
        print("=" * 72)
        print(" Simple 3D Photonic Crystal Slab Simulation (C4v InP Membrane) ")
        print("=" * 72)
        print(f"Lattice Pitch:       a = {pitch:.3f} μm (Square lattice)")
        print(
            f"Slab Thickness:     h = {slab_thickness:.3f} μm (h/a = {slab_thickness / pitch:.3f})"
        )
        print(f"Hole Radii:         r₁ = {r1:.3f} μm (1a), r₂ = {r2:.3f} μm (1b)")
        print(f"Supercell Height:   s_z = {supercell_z:.1f} a")
        print(f"Slab Material:      {slab_material.upper()} (phc_materials)")
        print(f"Cladding:           {cladding_material.upper()} (phc_materials)")
        print(f"Mesh Resolution:    r_xy = {resolution}, r_z = {resolution_z}")
        print(f"Eigenbands:         {num_bands} bands (TE-like / even parity)")
        print(f"Output Directory:   {out_path}")
        print("=" * 72 + "\n")

    # Step 1: Generate 3D Slab Unit Cell Layout (phc_layout)
    if verbose:
        print("[1/6] Generating C4v square slab unit cell layout...")
    unit_cell = phc_wyckoff_unit_cell(
        pitch=pitch,
        point_group="C4v",
        features=[("1a", r1), ("1b", r2)],
    )

    # Step 2: Export Physical GDSII Mask (phc_hydra SSOT)
    gds_path = output_mgr.save_gds(unit_cell, filename="unit_cell.gds")
    if verbose:
        print(f"      -> Exported GDSII binary: {gds_path}")

    # Step 3: Query Material Registry (phc_materials SSOT)
    mat_slab = get_material(slab_material)
    mat_clad = get_material(cladding_material)
    if verbose:
        print(
            f"[2/6] Material Registry: Slab='{mat_slab.name}' (n={mat_slab.index:.3f}), "
            f"Cladding='{mat_clad.name}' (n={mat_clad.index:.3f})"
        )

    # Step 4: Setup 3D Slab Lattice & Convert GDS to MPB Geometry
    if verbose:
        print("[3/6] Setting up 3D MPB square lattice & extruding geometry...")
    lattice = create_lattice(
        lattice_type="square",
        pitch=pitch,
        dimension="3D_slab",
        supercell_z=supercell_z,
    )
    geometry = gds_to_mpb_geometry(
        gds_source=gds_path,
        pitch=pitch,
        dimension="3D_slab",
        slab_thickness=slab_thickness,
        slab_material=slab_material,
        etch_material=cladding_material,
        geometry_lattice=lattice,
    )
    if verbose:
        print(
            f"      -> Extracted {len(geometry)} 3D geometric objects (Slab + Hole Cylinders)."
        )

    # High-symmetry path: X -> Γ -> M -> X (Γ is second point per convention)
    k_points, labels, indices = get_high_symmetry_kpath(
        lattice_type="square",
        k_density=k_density,
    )
    if verbose:
        print(
            f"      -> Brillouin Zone Path: {' -> '.join(labels)} ({len(k_points)} k-points)"
        )

    # Step 5: Execute MPB Band Solver
    if verbose:
        print(
            f"\n[4/6] Solving TE-like guided modes (resolution={resolution}x{resolution}x{resolution_z}, workers={num_workers})..."
        )
    ms = create_mode_solver(
        geometry_lattice=lattice,
        geometry=geometry,
        k_points=k_points,
        default_material=mat_clad,
        resolution=(resolution, resolution, resolution_z),
        num_bands=num_bands,
    )

    t_solve_start = time.time()
    with silence_c_stdout():
        results = run_band_solver(
            ms=ms,
            polarization="te_like",
            dimension="3D_slab",
            cladding_index=mat_clad.index,
            num_workers=num_workers,
            verbose=False,
        )
    solve_duration = time.time() - t_solve_start

    freqs = results.get("freqs", {}).get("te_like")
    if freqs is None or len(freqs) == 0:
        raise RuntimeError("MPB failed to compute TE-like band frequencies.")
    if verbose:
        print(
            f"      -> Computed {freqs.shape[0]} k-points for {freqs.shape[1]} bands in {solve_duration:.2f} s."
        )

    # Step 6: Extract & Plot MPB 3D Dielectric Permittivity
    if verbose:
        print("\n[5/6] Extracting & plotting 3D dielectric permittivity profile...")
    with silence_c_stdout():
        eps_grid = get_epsilon_grid(
            ms=ms,
            rectify=True,
            periods=2,
            periods_z=1,
            resolution=32 if not quick else 16,
        )

    fig_eps = plot_epsilon(
        epsilon=eps_grid,
        title=f"3D PhC Slab Permittivity — {mat_slab.name.upper()} Membrane in {mat_clad.name.upper()}",
    )
    eps_file = output_mgr.save_figure(
        fig_eps, artifact_key="eps_plot", filename="epsilon_map.png"
    )
    plt.close(fig_eps)
    if verbose:
        print(f"      -> Saved dual-plane dielectric epsilon plot: {eps_file}")

    # Step 7: Plot Photonic Band Structure
    if verbose:
        print("\n[6/6] Rendering photonic band structure diagram...")
    fig_band = plot_band_structure(
        results=results,
        node_labels=labels,
        node_indices=indices,
        title=f"3D Square PhC Slab ({mat_slab.name.upper()}, h/a={slab_thickness / pitch:.2f}, r₁={r1:.2f}, r₂={r2:.2f}) — TE-like",
    )
    band_file = output_mgr.save_figure(
        fig_band, artifact_key="band_plot", filename="band_structure.png"
    )
    plt.close(fig_band)
    if verbose:
        print(f"      -> Saved band structure diagram: {band_file}")

    # Check for omnidirectional band gaps
    gaps = results.get("gaps", {}).get("te_like", [])
    if verbose and gaps:
        for idx, g in enumerate(gaps, start=1):
            if isinstance(g, dict):
                bands_str = f"{g.get('bands')}"
                pct = float(g.get("gap_pct", 0.0))
            elif isinstance(g, (int, float)):
                bands_str = f"[{idx}, {idx + 1}]"
                pct = float(g)
            else:
                continue
            if pct > 0:
                print(f"      -> Band Gap (Bands {bands_str}): {pct:.2f}%")

    # Save structured results JSON summary
    t_total = time.time() - t_start
    summary_data = {
        "geometry": {
            "name": "c4v_dirac_3d",
            "lattice_type": "square",
            "point_group": "C4v",
            "pitch_um": pitch,
            "slab_thickness_um": slab_thickness,
            "r1_um": r1,
            "r2_um": r2,
            "supercell_z": supercell_z,
            "slab_material": slab_material,
            "cladding_material": cladding_material,
        },
        "simulation": {
            "solver": "mpb",
            "dimension": "3D_slab",
            "polarization": "te_like",
            "resolution_xy": resolution,
            "resolution_z": resolution_z,
            "num_bands": num_bands,
            "k_density": k_density,
        },
        "bands": {
            "k_points": len(k_points),
            "frequencies": {
                pol: arr.tolist() for pol, arr in results.get("freqs", {}).items()
            },
            "gaps": results.get("gaps", {}),
            "light_line": results.get("light_line", []),
        },
        "elapsed_time_s": t_total,
    }
    json_path = output_mgr.save_results_json(summary_data)
    if verbose:
        print(f"      -> Saved simulation results JSON: {json_path}")
        print(f"\nCompleted in {t_total:.2f} seconds.")
        print("=" * 72)

    return {
        "results": results,
        "ms": ms,
        "labels": labels,
        "indices": indices,
        "eps_grid": eps_grid,
        "output_dir": out_path,
        "files": {
            "gds": gds_path,
            "band_plot": band_file,
            "eps_plot": eps_file,
            "results_json": json_path,
        },
        "summary": summary_data,
    }


def main() -> None:
    """CLI entrypoint for running the simple 3D PhC slab demonstration."""
    parser = argparse.ArgumentParser(
        description="Simple 3D Photonic Crystal Slab Simulation (C4v InP Slab Membrane).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run in rapid smoke-test mode (<2 seconds) for fast testing.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=16,
        help="In-plane mesh resolution per unit pitch a (default: 16, quick: 12).",
    )
    parser.add_argument(
        "--resolution-z",
        type=int,
        default=16,
        help="Vertical (z) mesh resolution per unit pitch a (default: 16, quick: 6).",
    )
    parser.add_argument(
        "--num-bands",
        type=int,
        default=8,
        help="Number of eigenbands to compute at each k-point (default: 8, quick: 4).",
    )
    parser.add_argument(
        "--k-density",
        type=int,
        default=6,
        help="Number of k-points between high-symmetry vertices (default: 6, quick: 2).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        dest="num_workers",
        help="Number of concurrent worker processes for MPB (default: 4, quick: 1).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Custom output directory override (default: auto-resolved by phc_hydra).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display interactive Matplotlib preview windows after simulation.",
    )
    args = parser.parse_args()

    run_slab_3d_pipeline(
        resolution=args.resolution,
        resolution_z=args.resolution_z,
        num_bands=args.num_bands,
        k_density=args.k_density,
        num_workers=args.num_workers,
        quick=args.quick,
        output_dir=args.output_dir,
    )

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
