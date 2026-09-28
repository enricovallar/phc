#!/usr/bin/env python3
"""Parametric Lower Cladding Refractive Index Sweep at the Gamma Point for 3D C6v 2b-6d hBN Slab.

Evaluates eigenmode shifts at the high-symmetry Γ point (k = [0, 0, 0]) as the
lower substrate cladding refractive index is swept from air (n = 1.0) to silica (SiO2, n ≈ 1.444):

1. Generates the C6v hexagonal lattice unit cell with Wyckoff 2b holes (radius r1)
   and Wyckoff 6d holes (radius r2, position parameter p2) etched into an hBN slab membrane.
2. Exports the physical GDSII layout mask (unit_cell.gds).
3. Extracts and plots the 3D dielectric permittivity cross-sections (epsilon_map.png).
4. Reference simulations at symmetric air cladding (n_sub = 1.0):
   - Solves baseline TE-like modes (z-even parity).
   - Solves baseline TM-like modes (z-odd parity).
5. Parametric sweep:
   - Sweeps lower cladding index n_sub from 1.0 (air) to 1.444 (silica SiO2).
   - Solves ALL bands (no parity constraint) at the Γ point at each step.
6. Unified wavelength visualization:
   - Plots free-space wavelength lambda_0 = pitch / omega_tilde in the range 350 nm to 500 nm (0.35 - 0.50 um).
   - Dual y-axis: primary axis in micrometers (um) and secondary right axis in nanometers (nm).
   - Overlays horizontal reference lines for unperturbed TE-like (blue dashed) and
     TM-like (red dotted) modes to visualize mode shifts, crossings, and pull-down.
7. Saves artifacts (GDS, epsilon map, sweep figure, structured results JSON) to
   outputs/mpb/cladding_sweep_gamma/c6v_2b_6d_cladding_sweep/<timestamp>/.

Command-Line Usage:
    # Standard run (wavelength 350 nm - 500 nm, 9 sweep steps)
    python analysis/test_c6v_cladding_sweep.py

    # Quick smoke test execution
    python analysis/test_c6v_cladding_sweep.py --quick

    # Custom sweep points and wavelength limits
    python analysis/test_c6v_cladding_sweep.py --num-points 15 --lam-min 0.35 --lam-max 0.50
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import meep as mp
import numpy as np

# Ensure repository root is in sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from phc_hydra import SimulationOutputManager, resolve_simulation_output_dir
from phc_layout.components.unit_cell import phc_wyckoff_unit_cell
from phc_layout.lattice import HexagonalLattice
from phc_materials import get_material
from phc_mpb import (
    create_lattice,
    create_mode_solver,
    gds_to_mpb_geometry,
    get_epsilon_grid,
    plot_epsilon,
    run_band_solver,
)
from phc_utils import silence_c_stdout

# Suppress Meep/MPB C-level verbosity by default
mp.verbosity(0)


def run_gamma_simulation(
    gds_path: Path | str,
    pitch: float = 0.381,
    slab_thickness: float = 0.1,
    supercell_z: float = 4.0,
    slab_material: str = "hBN",
    cladding_material: str = "air",
    substrate_index: float = 1.0,
    resolution: int = 25,
    resolution_z: int = 20,
    num_bands: int = 24,
    polarization: str = "all",
    verbose: bool = False,
) -> dict[str, Any]:
    """Runs a single 3D PhC slab MPB simulation at the Γ point (k = [0, 0, 0]).

    Args:
        gds_path: Path to the exported GDSII unit cell file.
        pitch: Lattice pitch constant a in micrometers (default: 0.381 μm).
        slab_thickness: Slab membrane thickness in micrometers (default: 0.1 μm).
        supercell_z: Supercell height in units of lattice constant a (default: 4.0).
        slab_material: Dielectric slab core material key (default: "hBN").
        cladding_material: Top and background cladding material key (default: "air").
        substrate_index: Refractive index of the lower substrate cladding (default: 1.0).
        resolution: In-plane (x, y) mesh resolution per unit pitch a (default: 25).
        resolution_z: Vertical (z) mesh resolution per unit pitch a (default: 20).
        num_bands: Number of eigenbands to compute at Γ (default: 24).
        polarization: Solver mode: 'te_like', 'tm_like', or 'all' (default: 'all').
        verbose: If True, streams solver progress to stdout.

    Returns:
        Dictionary containing solver 'results', 'ms' ModeSolver instance,
        and extracted 1D array of eigenfrequencies at Γ.
    """
    mat_clad = get_material(cladding_material)
    hex_lat = HexagonalLattice(a=pitch)

    lattice = create_lattice(
        lattice_type=hex_lat,
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
        substrate_material=float(substrate_index) if substrate_index > 1.0 else None,
        etch_material=cladding_material,
        geometry_lattice=lattice,
    )

    # Gamma point only
    k_points = [mp.Vector3(0.0, 0.0, 0.0)]

    ms = create_mode_solver(
        geometry_lattice=lattice,
        geometry=geometry,
        k_points=k_points,
        default_material=mat_clad,
        resolution=(resolution, resolution, resolution_z),
        num_bands=num_bands,
    )

    with silence_c_stdout():
        results = run_band_solver(
            ms=ms,
            polarization=polarization,
            dimension="3D_slab",
            cladding_index=mat_clad.index,
            num_workers=1,
            verbose=verbose,
        )

    # Extract 1D frequency array at Gamma (k-index 0)
    freqs_dict = results.get("freqs", {})
    freq_arr = freqs_dict.get(polarization)
    if freq_arr is None and f"{polarization}_like" in freqs_dict:
        freq_arr = freqs_dict[f"{polarization}_like"]
    if freq_arr is None and freqs_dict:
        freq_arr = next(iter(freqs_dict.values()))

    if freq_arr is None or len(freq_arr) == 0:
        raise RuntimeError(
            f"MPB failed to compute eigenfrequencies for polarization='{polarization}'."
        )

    return {
        "results": results,
        "ms": ms,
        "freqs": np.copy(freq_arr[0]),
    }


def run_c6v_cladding_sweep(
    pitch: float = 0.381,
    slab_thickness: float = 0.1,
    r1: float = 0.081,
    r2: float = 0.034,
    p2: float = 0.25,
    supercell_z: float = 4.0,
    slab_material: str = "hBN",
    cladding_material: str = "air",
    n_start: float = 1.0,
    n_end: float = 1.444,
    num_points: int = 9,
    resolution: int = 25,
    resolution_z: int = 20,
    num_bands_ref: int = 20,
    num_bands_sweep: int = 24,
    lam_min: float = 0.35,
    lam_max: float = 0.50,
    normalize: bool = False,
    output_dir: Path | str | None = None,
    verbose: bool = True,
    quick: bool = False,
    sim_name: str = "c6v_2b_6d_cladding_sweep",
) -> dict[str, Any]:
    """Sweeps the lower cladding refractive index at Γ for C6v 2b-6d hBN slab and plots wavelength vs n_sub.

    Args:
        pitch: Lattice pitch constant a in micrometers (default: 0.381 μm).
        slab_thickness: Slab membrane thickness in micrometers (default: 0.1 μm).
        r1: Radius of primary hole Wyckoff 2b in micrometers (default: 0.081 μm).
        r2: Radius of satellite hole Wyckoff 6d in micrometers (default: 0.034 μm).
        p2: Coordinate parameter for Wyckoff position 6d (default: 0.25).
        supercell_z: Supercell height in units of lattice constant a (default: 4.0).
        slab_material: Core dielectric slab material key (default: "hBN").
        cladding_material: Top and background cladding material key (default: "air").
        n_start: Starting lower cladding refractive index (default: 1.0, air).
        n_end: Ending lower cladding refractive index (default: 1.444, silica SiO2).
        num_points: Number of refractive index points across the sweep (default: 9).
        resolution: In-plane (x, y) mesh resolution per unit pitch a (default: 25).
        resolution_z: Vertical (z) mesh resolution per unit pitch a (default: 20).
        num_bands_ref: Number of baseline TE/TM eigenbands to compute at air (default: 20).
        num_bands_sweep: Number of eigenbands computed at each sweep step (default: 24).
        lam_min: Minimum wavelength in micrometers for the plot (default: 0.35 μm = 350 nm).
        lam_max: Maximum wavelength in micrometers for the plot (default: 0.50 μm = 500 nm).
        normalize: If True, plots normalized frequency instead of physical wavelength (default: False).
        output_dir: Custom output directory override. If None, auto-resolved via phc_hydra.
        verbose: If True, prints formatted progress updates to stdout.
        quick: If True, uses fast minimal settings for smoke testing.
        sim_name: Geometry name for output naming hierarchy (default: "c6v_2b_6d_cladding_sweep").

    Returns:
        Dictionary containing sweep data, reference frequencies, saved file paths, and summary.
    """
    t_start = time.time()

    if quick:
        resolution = 12
        resolution_z = 10
        num_bands_ref = 8
        num_bands_sweep = 12
        num_points = 3

    # 1. Resolve output directory and manager via phc_hydra SSOT standard
    out_path = resolve_simulation_output_dir(
        solver="mpb",
        sim_type="cladding_sweep_gamma",
        geometry=sim_name,
        override_dir=output_dir,
    )
    output_mgr = SimulationOutputManager(
        output_dir=out_path,
        solver="mpb",
        sim_type="cladding_sweep_gamma",
        geometry_name=sim_name,
    )

    if verbose:
        print("\n" + "=" * 72)
        print("3D C6v 2b-6d hBN Slab Lower Cladding Index Sweep at Γ")
        print(f"Lattice pitch a:       {pitch:.3f} μm")
        print(
            f"Slab thickness h:      {slab_thickness:.3f} μm (h/a = {slab_thickness / pitch:.3f})"
        )
        print(
            f"Hole radii:            r1 = {r1:.3f} μm (2b), r2 = {r2:.3f} μm (6d, p2={p2:.2f})"
        )
        print(f"Slab core material:    {slab_material.upper()}")
        print(
            f"Lower cladding sweep:  n_sub = [{n_start:.3f} -> {n_end:.3f}] ({num_points} steps)"
        )
        print(
            f"Wavelength range:      [{lam_min * 1000:.0f} nm -> {lam_max * 1000:.0f} nm]"
        )
        print(f"Output directory:      {out_path}")
        print("=" * 72)

    # 2. Step 1: Layout generation & GDS export
    if verbose:
        print("\n[1/4] Generating C6v 2b-6d hexagonal unit cell and exporting GDSII...")
    unit_cell = phc_wyckoff_unit_cell(
        pitch=pitch,
        point_group="C6v",
        features=[("2b", r1), ("6d", r2, p2)],
    )
    gds_path = output_mgr.save_gds(unit_cell, filename="unit_cell.gds")
    if verbose:
        print(f"      -> Exported layout mask: {gds_path}")

    # 3. Step 2: Dielectric permittivity visualization
    if verbose:
        print("\n[2/4] Computing & saving dielectric permittivity cross-sections...")
    hex_lat = HexagonalLattice(a=pitch)
    lattice_vis = create_lattice(
        lattice_type=hex_lat,
        pitch=pitch,
        dimension="3D_slab",
        supercell_z=supercell_z,
    )
    geom_vis = gds_to_mpb_geometry(
        gds_source=gds_path,
        pitch=pitch,
        dimension="3D_slab",
        slab_thickness=slab_thickness,
        slab_material=slab_material,
        substrate_material=float(
            n_end
        ),  # Visualize with SiO2 substrate to confirm stack profile
        etch_material=cladding_material,
        geometry_lattice=lattice_vis,
    )
    ms_vis = create_mode_solver(
        geometry_lattice=lattice_vis,
        geometry=geom_vis,
        k_points=[mp.Vector3(0.0, 0.0, 0.0)],
        default_material=get_material(cladding_material),
        resolution=(resolution, resolution, resolution_z),
        num_bands=1,
    )
    eps_grid = get_epsilon_grid(
        ms_vis,
        rectify=True,
        periodicity=3,
        resolution=resolution,
    )
    fig_eps = plot_epsilon(
        eps_grid,
        title=f"C6v 2b-6d hBN Slab Permittivity (n_sub={n_end:.2f})",
    )
    eps_file = output_mgr.save_figure(
        fig_eps, artifact_key="eps_plot", filename="epsilon_map.png"
    )
    plt.close(fig_eps)
    if verbose:
        print(f"      -> Saved permittivity map: {eps_file}")

    # 4. Step 3: Reference symmetric air-clad baseline simulations (n_sub = 1.0)
    if verbose:
        print(
            "\n[3/4] Solving reference baseline modes at symmetric air cladding (n_sub = 1.0)..."
        )

    t_ref_start = time.time()
    res_te = run_gamma_simulation(
        gds_path=gds_path,
        pitch=pitch,
        slab_thickness=slab_thickness,
        supercell_z=supercell_z,
        slab_material=slab_material,
        cladding_material=cladding_material,
        substrate_index=1.0,
        resolution=resolution,
        resolution_z=resolution_z,
        num_bands=num_bands_ref,
        polarization="te_like",
        verbose=False,
    )
    freqs_ref_te = res_te["freqs"]

    res_tm = run_gamma_simulation(
        gds_path=gds_path,
        pitch=pitch,
        slab_thickness=slab_thickness,
        supercell_z=supercell_z,
        slab_material=slab_material,
        cladding_material=cladding_material,
        substrate_index=1.0,
        resolution=resolution,
        resolution_z=resolution_z,
        num_bands=num_bands_ref,
        polarization="tm_like",
        verbose=False,
    )
    freqs_ref_tm = res_tm["freqs"]
    if verbose:
        print(
            f"      -> Solved reference TE/TM baseline modes in {time.time() - t_ref_start:.2f} s"
        )

    # 5. Step 4: Parametric sweep across lower cladding refractive index
    n_values = np.linspace(n_start, n_end, num_points)
    sweep_freqs_list: list[np.ndarray] = []

    if verbose:
        print(
            f"\n[4/4] Executing parametric sweep across {num_points} substrate indices [{n_start:.3f} -> {n_end:.3f}]..."
        )

    t_sweep_start = time.time()
    for idx, n_sub in enumerate(n_values):
        t_step = time.time()
        sim_step = run_gamma_simulation(
            gds_path=gds_path,
            pitch=pitch,
            slab_thickness=slab_thickness,
            supercell_z=supercell_z,
            slab_material=slab_material,
            cladding_material=cladding_material,
            substrate_index=float(n_sub),
            resolution=resolution,
            resolution_z=resolution_z,
            num_bands=num_bands_sweep,
            polarization="all",
            verbose=False,
        )
        sweep_freqs_list.append(sim_step["freqs"])
        if verbose:
            print(
                f"      [{idx + 1:2d}/{num_points:2d}] n_sub = {n_sub:.4f} "
                f"-> solved {num_bands_sweep} bands in {time.time() - t_step:.2f} s"
            )

    sweep_freqs_matrix = np.array(
        sweep_freqs_list
    )  # shape: (num_points, num_bands_sweep)
    t_sweep_elapsed = time.time() - t_sweep_start

    # 6. Plotting: Unified wavelength diagram with horizontal reference lines
    if verbose:
        print("\nGenerating sweep diagram with baseline references...")

    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=150)

    # Filter out DC electrostatic/null zero modes (f < 1e-4) for reference lines
    phys_ref_te = freqs_ref_te[freqs_ref_te > 1e-4]
    phys_ref_tm = freqs_ref_tm[freqs_ref_tm > 1e-4]

    if normalize:
        y_ref_te = phys_ref_te
        y_ref_tm = phys_ref_tm
        y_sweep = sweep_freqs_matrix
        y_label = (
            r"Normalized Frequency $\tilde{\omega} = \omega a / 2\pi c = a / \lambda$"
        )
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            y_ref_te = np.where(phys_ref_te > 0, pitch / phys_ref_te, np.nan)
            y_ref_tm = np.where(phys_ref_tm > 0, pitch / phys_ref_tm, np.nan)
            y_sweep = np.where(
                sweep_freqs_matrix > 0, pitch / sweep_freqs_matrix, np.nan
            )
        y_label = r"Wavelength $\lambda$ [$\mu$m]"

    # Draw horizontal reference lines for TE-like baseline modes
    for i, y_te in enumerate(y_ref_te):
        if not np.isfinite(y_te):
            continue
        ax.axhline(
            y=y_te,
            color="tab:blue",
            linestyle="--",
            linewidth=1.2,
            alpha=0.75,
            label="Ref TE-like (n=1.0)" if i == 0 else None,
        )

    # Draw horizontal reference lines for TM-like baseline modes
    for i, y_tm in enumerate(y_ref_tm):
        if not np.isfinite(y_tm):
            continue
        ax.axhline(
            y=y_tm,
            color="tab:red",
            linestyle=":",
            linewidth=1.4,
            alpha=0.75,
            label="Ref TM-like (n=1.0)" if i == 0 else None,
        )

    # Plot swept bands (all bands without parity constraint)
    for b in range(num_bands_sweep):
        if np.max(sweep_freqs_matrix[:, b]) < 1e-4:
            continue
        valid_mask = np.isfinite(y_sweep[:, b])
        if not np.any(valid_mask):
            continue
        ax.plot(
            n_values[valid_mask],
            y_sweep[valid_mask, b],
            marker="o",
            markersize=3.5,
            linestyle="-",
            linewidth=1.0,
            color="black",
            alpha=0.85,
            label="All bands (swept)" if b == 0 else None,
        )

    ax.set_xlabel(
        r"Lower Cladding Refractive Index $n_{\mathrm{sub}}$",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_ylabel(
        y_label,
        fontsize=11,
        fontweight="bold",
    )
    ax.set_title(
        f"C6v 2b-6d hBN Slab — Lower Cladding Index Sweep at Γ\n"
        f"(a = {pitch * 1e3:.0f} nm, h = {slab_thickness * 1e3:.0f} nm, r₁ = {r1 * 1e3:.0f} nm, r₂ = {r2 * 1e3:.0f} nm)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlim(n_start - 0.02, n_end + 0.02)
    if normalize:
        ax.set_ylim(0.70, 1.15)
    else:
        ax.set_ylim(lam_min, lam_max)
        # Add secondary right axis in nanometers
        secax = ax.secondary_yaxis(
            "right",
            functions=(lambda x: x * 1000.0, lambda x: x / 1000.0),
        )
        secax.set_ylabel(r"Wavelength $\lambda$ [nm]", fontsize=11, fontweight="bold")

    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=10)
    plt.tight_layout()

    sweep_plot_file = output_mgr.save_figure(
        fig, artifact_key="cladding_sweep_plot", filename="cladding_sweep_gamma.png"
    )
    plt.close(fig)

    # 7. Save structured JSON simulation results
    t_total = time.time() - t_start
    summary_data = {
        "geometry": {
            "name": sim_name,
            "lattice_type": "hexagonal",
            "point_group": "C6v",
            "pitch_um": pitch,
            "slab_thickness_um": slab_thickness,
            "r1_um": r1,
            "r2_um": r2,
            "p2": p2,
            "supercell_z": supercell_z,
            "slab_material": slab_material,
            "cladding_material": cladding_material,
        },
        "simulation": {
            "solver": "mpb",
            "dimension": "3D_slab",
            "resolution": resolution,
            "resolution_z": resolution_z,
            "num_bands_ref": num_bands_ref,
            "num_bands_sweep": num_bands_sweep,
            "n_start": n_start,
            "n_end": n_end,
            "num_points": num_points,
            "lam_min_um": lam_min,
            "lam_max_um": lam_max,
        },
        "results": {
            "n_values": n_values.tolist(),
            "ref_te_frequencies": freqs_ref_te.tolist(),
            "ref_tm_frequencies": freqs_ref_tm.tolist(),
            "sweep_frequencies": sweep_freqs_matrix.tolist(),
            "sweep_elapsed_time_s": t_sweep_elapsed,
        },
        "elapsed_time_s": t_total,
    }
    json_path = output_mgr.save_results_json(summary_data)

    if verbose:
        print(f"\n      -> Saved sweep plot: {sweep_plot_file}")
        print(f"      -> Saved results JSON: {json_path}")
        print(f"\nCompleted in {t_total:.2f} seconds.")
        print("=" * 72)

    return {
        "n_values": n_values,
        "ref_te": freqs_ref_te,
        "ref_tm": freqs_ref_tm,
        "sweep_freqs": sweep_freqs_matrix,
        "output_dir": out_path,
        "files": {
            "gds": gds_path,
            "sweep_plot": sweep_plot_file,
            "eps_plot": eps_file,
            "results_json": json_path,
        },
        "summary": summary_data,
    }


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the C6v cladding index sweep."""
    parser = argparse.ArgumentParser(
        description="Parametric lower substrate cladding index sweep at Γ for C6v 2b-6d hBN slab."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick execution mode with minimal resolution and sweep points for smoke testing.",
    )
    parser.add_argument(
        "--pitch",
        type=float,
        default=0.381,
        help="Lattice constant pitch a in micrometers (default: 0.381).",
    )
    parser.add_argument(
        "--slab-thickness",
        type=float,
        default=0.1,
        help="Slab membrane thickness in micrometers (default: 0.1).",
    )
    parser.add_argument(
        "--r1",
        type=float,
        default=0.081,
        help="Radius of primary Wyckoff 2b holes in micrometers (default: 0.081).",
    )
    parser.add_argument(
        "--r2",
        type=float,
        default=0.034,
        help="Radius of satellite Wyckoff 6d holes in micrometers (default: 0.034).",
    )
    parser.add_argument(
        "--p2",
        type=float,
        default=0.25,
        help="Coordinate parameter for Wyckoff 6d position (default: 0.25).",
    )
    parser.add_argument(
        "--n-start",
        type=float,
        default=1.0,
        help="Starting lower cladding refractive index (default: 1.0, air).",
    )
    parser.add_argument(
        "--n-end",
        type=float,
        default=1.444,
        help="Ending lower cladding refractive index (default: 1.444, silica SiO2).",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=9,
        help="Number of sweep index steps between n_start and n_end (default: 9).",
    )
    parser.add_argument(
        "--lam-min",
        type=float,
        default=0.35,
        help="Minimum wavelength limit in micrometers (default: 0.35 μm = 350 nm).",
    )
    parser.add_argument(
        "--lam-max",
        type=float,
        default=0.50,
        help="Maximum wavelength limit in micrometers (default: 0.50 μm = 500 nm).",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Plot normalized frequency instead of physical wavelength.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory path override.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display interactive matplotlib window after completion.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = run_c6v_cladding_sweep(
        pitch=args.pitch,
        slab_thickness=args.slab_thickness,
        r1=args.r1,
        r2=args.r2,
        p2=args.p2,
        n_start=args.n_start,
        n_end=args.n_end,
        num_points=args.num_points,
        lam_min=args.lam_min,
        lam_max=args.lam_max,
        normalize=args.normalize,
        quick=args.quick,
        output_dir=args.output_dir,
    )
    if not args.no_show:
        plot_path = results["files"]["sweep_plot"]
        print(f"Sweep figure available at: {plot_path}")
