#!/usr/bin/env python3
"""End-to-End Demonstration: 3D Photonic Crystal Slab with Concurrent Multi-Worker MPB.

Demonstrates the Single Source of Truth (SSOT) architecture for 3D Photonic Crystal Slabs:
1. Generates a hexagonal unit cell using `phc_layout`.
2. Exports the physical GDSII layout mask via `phc_utils.export_gds`.
3. Queries optical material properties from `phc_materials` (Silicon slab n=3.48, Air holes n=1.0).
4. Extrudes the 3D dielectric slab membrane and etch hole cylinders/prisms in `phc_mpb`.
5. Computes high-symmetry k-path (M -> Γ -> K -> M) across the Brillouin zone.
6. Solves for TE-like guided slab modes (even parity, zeven) concurrently across multiple worker
   processes using `run_parallel_band_solver` with anisotropic mesh resolution (rx, ry vs rz).
7. Renders the photonic band structure featuring the cladding light line and guided band gaps.
8. Extracts the 3D dielectric permittivity distribution at the slab mid-plane z=0.
9. Exports structured simulation metadata and parallel execution speedup metrics.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

# Ensure repository root is in sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from phc_hydra import SimulationOutputManager, resolve_simulation_output_dir
from phc_layout.components import get_unit_cell
from phc_layout.lattice import HexagonalLattice
from phc_materials import get_material
from phc_mpb import (
    create_mode_solver,
    gds_to_mpb_geometry,
    get_epsilon_grid,
    get_high_symmetry_kpath,
    lattice_to_mpb_lattice,
    plot_band_structure,
    plot_epsilon,
    run_parallel_band_solver,
)
from phc_utils import export_gds


def run_slab_3d_pipeline(
    pitch: float = 0.45,
    slab_thickness_um: float = 0.22,
    radius_ratio: float = 0.25,
    resolution: int = 24,
    resolution_z: int = 8,
    num_bands: int = 6,
    k_density: int = 12,
    supercell_z: float = 5.0,
    num_workers: int = 4,
    quick: bool = False,
    output_dir: Path | str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Runs the end-to-end 3D PhC slab simulation pipeline connecting layout, materials, and MPB.

    Args:
        pitch: Lattice constant a in micrometers (default: 0.45 um).
        slab_thickness_um: Physical slab membrane thickness in micrometers (default: 0.22 um = 220 nm).
        radius_ratio: Hole radius normalized to pitch (r / a, default: 0.25).
        resolution: In-plane (x, y) mesh resolution per unit pitch a (default: 24).
        resolution_z: Vertical (z) mesh resolution per unit pitch a (default: 8).
        num_bands: Number of eigenbands to compute at each k-point (default: 6).
        k_density: Number of interpolated k-points between high-symmetry vertices (default: 12).
        supercell_z: Vertical supercell height in units of pitch a including cladding buffer (default: 5.0).
        num_workers: Number of concurrent worker processes for k-point parallelization (default: 4).
        quick: If True, overrides simulation parameters to minimal settings (resolution=12,
            resolution_z=4, num_bands=3, k_density=2, supercell_z=3.0) for rapid smoke testing (<=1.0s).
        output_dir: Optional directory override. If None, resolves canonical hierarchy:
            outputs/mpb/slab_band_diagram/c6v_primitive/<timestamp>/.
        verbose: If True, prints formatted progress updates and timing analysis.

    Returns:
        Dictionary containing:
            - 'results': MPB solver results including frequencies, gaps, and light line.
            - 'gaps': List of detected omnidirectional band gaps.
            - 'output_dir': Resolved path to output directory.
            - 'files': Dict mapping artifact keys ('gds', 'band_plot', 'eps_plot', 'results_json')
              to saved file paths.
            - 'elapsed_time_s': Total pipeline execution time in seconds.
    """
    t_start = time.time()

    if quick:
        resolution = 12
        resolution_z = 4
        num_bands = 3
        k_density = 1
        supercell_z = 3.0
        num_workers = 4

    out_path = resolve_simulation_output_dir(
        solver="mpb",
        sim_type="slab_band_diagram",
        geometry="c6v_primitive",
        override_dir=output_dir,
    )
    output_mgr = SimulationOutputManager(
        output_dir=out_path,
        solver="mpb",
        sim_type="slab_band_diagram",
        geometry_name="c6v_primitive",
    )

    if verbose:
        print("\n" + "=" * 65)
        print("3D Photonic Crystal Slab Simulation Pipeline (TE-like Modes)")
        print(f"Lattice Pitch:       a = {pitch:.3f} um")
        print(
            f"Slab Thickness:     h = {slab_thickness_um:.3f} um (h/a = {slab_thickness_um / pitch:.3f})"
        )
        print(
            f"Hole Radius:        r = {radius_ratio * pitch:.3f} um (r/a = {radius_ratio:.2f})"
        )
        print(f"Supercell Height:   s_z = {supercell_z:.1f} a")
        print(f"Mesh Resolution:    r_xy = {resolution}, r_z = {resolution_z}")
        print(f"Parallel Workers:   {num_workers} concurrent processes")
        print(f"Output Directory:   {out_path}")
        print("=" * 65 + "\n")

    # -------------------------------------------------------------
    # Step 1: Layout Generation (phc_layout)
    # -------------------------------------------------------------
    if verbose:
        print("[1/6] Generating 2D hexagonal unit cell layout...")
    hole_radius_um = radius_ratio * pitch
    unit_cell = get_unit_cell(
        name="c6v_primitive",
        pitch=pitch,
        radius=hole_radius_um,
        wrap_to_cell=True,
    )

    # -------------------------------------------------------------
    # Step 2: GDSII Mask Export (phc_hydra SSOT)
    # -------------------------------------------------------------
    gds_path = output_mgr.save_gds(unit_cell, filename="unit_cell.gds")
    legacy_gds = out_path / "slab_unit_cell.gds"
    if legacy_gds != gds_path:
        export_gds(unit_cell, legacy_gds, overwrite=True)
    if verbose:
        print(f"      -> Exported GDSII mask: {gds_path}")

    # -------------------------------------------------------------
    # Step 3: Material Properties Query (phc_materials SSOT)
    # -------------------------------------------------------------
    mat_slab = get_material("si")
    mat_air = get_material("air")
    if verbose:
        print(
            f"[2/6] Material SSOT: Slab='{mat_slab.name}' (n={mat_slab.index:.3f}), "
            f"Cladding/Holes='{mat_air.name}' (n={mat_air.index:.3f})"
        )

    # -------------------------------------------------------------
    # Step 4: 3D Slab Lattice & Geometry Extrusion (phc_mpb)
    # -------------------------------------------------------------
    if verbose:
        print("[3/6] Building 3D slab lattice and extruding geometry...")
    layout_lat = HexagonalLattice(a=pitch)
    mpb_lat = lattice_to_mpb_lattice(
        layout_lat,
        dimension="3D_slab",
        supercell=(1, 1),
        supercell_z=supercell_z,
        normalize=True,
    )

    mpb_geom = gds_to_mpb_geometry(
        gds_source=gds_path,
        pitch=pitch,
        dimension="3D_slab",
        slab_thickness=slab_thickness_um,
        slab_material="si",
        etch_material="air",
        geometry_lattice=mpb_lat,
    )
    if verbose:
        print(
            f"      -> Generated {len(mpb_geom)} 3D geometric objects (Slab Block + Hole Prisms)."
        )

    # -------------------------------------------------------------
    # Step 5: High-Symmetry K-Path & Concurrent MPB Band Solve
    # -------------------------------------------------------------
    k_pts, labels, indices = get_high_symmetry_kpath(layout_lat, k_density=k_density)
    if verbose:
        print(
            f"[4/6] Solving TE-like modes across {len(k_pts)} k-points "
            f"({' -> '.join(labels)}) using {num_workers} parallel workers..."
        )

    t_solve_start = time.time()
    results = run_parallel_band_solver(
        geometry_lattice=mpb_lat,
        geometry=mpb_geom,
        k_points=k_pts,
        default_material="air",
        resolution=resolution,
        resolution_z=resolution_z,
        num_bands=num_bands,
        polarization="te_like",
        dimension="3D_slab",
        num_workers=num_workers,
        cladding_index=1.0,
        verbose=False,
    )
    t_solve_end = time.time()
    solve_duration = t_solve_end - t_solve_start
    if verbose:
        print(f"      -> Band solve completed in {solve_duration:.2f} seconds.")

    # -------------------------------------------------------------
    # Step 6: Band Gap Analysis, Dielectric Grid & Plotting
    # -------------------------------------------------------------
    if verbose:
        print("[5/6] Extracting dielectric profile and rendering figures...")

    # A. Midplane Dielectric Permittivity Map
    ms_eps = create_mode_solver(
        geometry_lattice=mpb_lat,
        geometry=mpb_geom,
        k_points=[k_pts[0]],
        default_material="air",
        resolution=resolution,
        resolution_z=resolution_z,
        num_bands=1,
    )
    eps_grid = get_epsilon_grid(
        ms_eps, rectify=True, periods=2, periods_z=1, resolution=resolution
    )

    fig_eps = plot_epsilon(
        eps_grid,
        title=r"3D PhC Slab Dielectric Distribution $\varepsilon(\mathbf{r})$",
    )
    eps_plot_path = output_mgr.save_figure(
        fig_eps, artifact_key="eps_plot", filename="epsilon_map.png"
    )
    # Also save to legacy profile and midplane aliases for backward compatibility
    fig_eps.savefig(out_path / "epsilon_slab_profiles.png")
    fig_eps.savefig(out_path / "epsilon_slab_midplane.png")
    plt.close(fig_eps)

    # B. Photonic Band Structure Plot
    fig_band = plot_band_structure(
        results=results,
        node_labels=labels,
        node_indices=indices,
        title="3D PhC Slab Band Structure (TE-like Even Modes)",
    )
    band_plot_path = output_mgr.save_figure(
        fig_band, artifact_key="band_plot", filename="band_structure.png"
    )
    fig_band.savefig(out_path / "band_diagram_slab_te_like.png")
    plt.close(fig_band)

    # -------------------------------------------------------------
    # Step 7: Results Compilation & JSON Serialization
    # -------------------------------------------------------------
    if verbose:
        print("[6/6] Compiling summary report...")

    freqs_arr = results["freqs"]["te_like"]
    light_line = results.get("light_line", [])

    detected_gaps: list[dict[str, Any]] = []
    for b in range(num_bands - 1):
        top_lower = float(results["freqs"]["te_like"][:, b].max())
        bot_upper = float(results["freqs"]["te_like"][:, b + 1].min())
        if bot_upper > top_lower:
            gap_pct = 200.0 * (bot_upper - top_lower) / (bot_upper + top_lower)
            gap_entry = {
                "bands": [b + 1, b + 2],
                "lower_freq": top_lower,
                "upper_freq": bot_upper,
                "gap_pct": gap_pct,
            }
            detected_gaps.append(gap_entry)
            if verbose:
                print(
                    f"      -> Complete TE-like Gap Bands {b + 1}-{b + 2}: "
                    f"[{top_lower:.4f}, {bot_upper:.4f}] (Gap: {gap_pct:.2f}%)"
                )

    guided_gaps = results.get("guided_gaps", {}).get("te_like", [])
    if verbose and guided_gaps:
        for gg in guided_gaps:
            print(
                f"      -> Guided Gap (Below Light Line) Bands {gg['bands'][0]}-{gg['bands'][1]}: "
                f"[{gg['lower_freq']:.4f}, {gg['upper_freq']:.4f}] (Gap: {gg['gap_pct']:.2f}%)"
            )

    t_total = time.time() - t_start

    summary_data = {
        "geometry": {
            "name": "c6v_primitive",
            "lattice_type": "hexagonal",
            "pitch_um": pitch,
            "slab_thickness_um": slab_thickness_um,
            "radius_ratio": radius_ratio,
            "supercell_z": supercell_z,
        },
        "simulation": {
            "solver": "mpb",
            "dimension": "3D_slab",
            "polarization": "te_like",
            "resolution_xy": resolution,
            "resolution_z": resolution_z,
            "num_bands": num_bands,
            "k_density": k_density,
            "num_k_points": len(k_pts),
            "num_workers": results["num_workers"],
        },
        "timing": {
            "solve_duration_s": round(solve_duration, 3),
            "total_duration_s": round(t_total, 3),
        },
        "gaps": detected_gaps,
        "guided_gaps": guided_gaps,
        "band_frequencies": freqs_arr.tolist(),
        "light_line": light_line,
    }

    results_json_path = output_mgr.save_results_json(
        geometry_cfg=summary_data["geometry"],
        simulation_cfg=summary_data["simulation"],
        band_gaps=detected_gaps,
        extra_data={
            "timing": summary_data["timing"],
            "guided_gaps": guided_gaps,
            "band_frequencies": freqs_arr.tolist(),
            "light_line": light_line,
        },
        filename="simulation_results.json",
    )
    # Also write legacy JSON alias for backward compatibility
    legacy_json_path = out_path / "slab_simulation_results.json"
    if legacy_json_path != results_json_path:
        with open(legacy_json_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

    if verbose:
        print("\n" + "-" * 65)
        print("Summary of Generated Artifacts:")
        print(f"  * GDS Layout:       {gds_path}")
        print(f"  * Permittivity Map: {eps_plot_path}")
        print(f"  * Band Diagram:     {band_plot_path}")
        print(f"  * Results JSON:     {results_json_path}")
        print(f"  * Total Run Time:   {t_total:.2f}s (Solve: {solve_duration:.2f}s)")
        print("-" * 65 + "\n")

    return {
        "results": results,
        "gaps": detected_gaps,
        "guided_gaps": guided_gaps,
        "output_dir": str(out_path),
        "files": {
            "gds": str(gds_path),
            "eps_plot": str(eps_plot_path),
            "band_plot": str(band_plot_path),
            "results_json": str(results_json_path),
            "manifest": output_mgr.manifest,
        },
        "elapsed_time_s": t_total,
    }


def main() -> None:
    """CLI entrypoint for running the 3D PhC slab demonstration."""
    parser = argparse.ArgumentParser(
        description="3D Photonic Crystal Slab MPB Band Structure Simulation with Parallel k-Points Wrapper."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run in rapid smoke-test mode with low resolution (<=1.0s).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent worker processes (default: 4).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=24,
        help="In-plane mesh resolution (default: 24).",
    )
    parser.add_argument(
        "--resolution-z",
        type=int,
        default=8,
        help="Vertical mesh resolution along z (default: 8).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory override (defaults to outputs/mpb/slab_band_diagram/c6v_primitive/<timestamp>).",
    )
    args = parser.parse_args()

    run_slab_3d_pipeline(
        resolution=args.resolution,
        resolution_z=args.resolution_z,
        num_workers=args.workers,
        quick=args.quick,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
