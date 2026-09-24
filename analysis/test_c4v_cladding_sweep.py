#!/usr/bin/env python3
"""Parametric Lower Cladding Refractive Index Sweep at the Gamma Point for a 3D C4v PhC Slab.

Evaluates eigenmode shifts at the high-symmetry Γ point (k = [0, 0, 0]) as the
lower substrate cladding refractive index is swept from air (n = 1.0) to silica (SiO2, n ≈ 1.444):

1. Generates the C4v square lattice unit cell (Wyckoff 1a hole at origin, 1b hole at center)
   etched into an InP membrane suspended in air.
2. Exports the physical GDSII layout mask (unit_cell.gds).
3. Reference simulations at symmetric air cladding (n_sub = 1.0):
   - Solves baseline TE-like modes (z-even parity).
   - Solves baseline TM-like modes (z-odd parity).
4. Parametric sweep:
   - Sweeps lower cladding index n_sub from 1.0 to 1.444 across multiple steps.
   - Solves ALL bands (no parity constraint) at the Γ point at each step.
5. Unified visualization:
   - Plots eigenfrequencies ωa / 2πc at Γ vs. lower cladding index n_sub.
   - Overlays horizontal reference lines for unperturbed TE-like (blue dashed) and
     TM-like (red dotted) modes to visualize mode shifts, crossings, and pull-down.
6. Saves artifacts (GDS, epsilon map, sweep figure, structured results JSON) to
   outputs/mpb/cladding_sweep_gamma/c4v_cladding_sweep/<timestamp>/.
"""

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
    pitch: float = 1.0,
    slab_thickness: float = 0.5,
    supercell_z: float = 4.0,
    slab_material: str = "inp",
    cladding_material: str = "air",
    substrate_index: float = 1.0,
    resolution: int = 16,
    resolution_z: int = 16,
    num_bands: int = 8,
    polarization: str = "all",
    verbose: bool = False,
) -> dict[str, Any]:
    """Runs a single 3D PhC slab MPB simulation at the Γ point (k = [0, 0, 0]).

    Args:
        gds_path: Path to the exported GDSII unit cell file.
        pitch: Lattice constant a in micrometers (default: 1.0 μm).
        slab_thickness: Slab membrane thickness in micrometers (default: 0.5 μm).
        supercell_z: Supercell height in units of lattice constant a (default: 4.0).
        slab_material: Dielectric slab core material key (default: "inp").
        cladding_material: Top and background cladding material key (default: "air").
        substrate_index: Refractive index of the lower substrate cladding (default: 1.0).
        resolution: In-plane (x, y) mesh resolution per unit pitch a (default: 16).
        resolution_z: Vertical (z) mesh resolution per unit pitch a (default: 16).
        num_bands: Number of eigenbands to compute at Γ (default: 8).
        polarization: Solver mode: 'te_like', 'tm_like', or 'all' (default: 'all').
        verbose: If True, streams solver progress to stdout.

    Returns:
        Dictionary containing solver 'results', 'ms' ModeSolver instance,
        and extracted 1D array of eigenfrequencies at Γ.
    """
    mat_clad = get_material(cladding_material)

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


def run_cladding_sweep(
    pitch: float = 1.0,
    slab_thickness: float = 0.5,
    r1: float = 0.25,
    r2: float = 0.15,
    supercell_z: float = 4.0,
    slab_material: str = "inp",
    cladding_material: str = "air",
    n_start: float = 1.0,
    n_end: float = 1.444,
    num_points: int = 9,
    resolution: int = 16,
    resolution_z: int = 16,
    num_bands_ref: int = 8,
    num_bands_sweep: int = 16,
    output_dir: Path | str | None = None,
    verbose: bool = True,
    sim_name: str = "c4v_cladding_sweep",
) -> dict[str, Any]:
    """Sweeps the lower cladding refractive index at Γ and plots results with baseline references.

    Args:
        pitch: Lattice constant a in micrometers (default: 1.0 μm).
        slab_thickness: Slab membrane thickness in micrometers (default: 0.5 μm).
        r1: Radius of primary hole at origin Wyckoff 1a in micrometers (default: 0.25 μm).
        r2: Radius of secondary hole at center Wyckoff 1b in micrometers (default: 0.15 μm).
        supercell_z: Supercell height in units of lattice constant a (default: 4.0).
        slab_material: Core dielectric slab material key (default: "inp").
        cladding_material: Top and background cladding material key (default: "air").
        n_start: Starting lower cladding refractive index (default: 1.0, air).
        n_end: Ending lower cladding refractive index (default: 1.444, silica SiO2).
        num_points: Number of refractive index points across the sweep (default: 9).
        resolution: In-plane (x, y) mesh resolution per unit pitch a (default: 16).
        resolution_z: Vertical (z) mesh resolution per unit pitch a (default: 16).
        num_bands_ref: Number of baseline TE/TM eigenbands to compute at air (default: 8).
        num_bands_sweep: Number of eigenbands computed at each sweep step (default: 16).
        output_dir: Custom output directory override. If None, auto-resolved via phc_hydra.
        verbose: If True, prints formatted progress updates to stdout.
        sim_name: Geometry name for output naming hierarchy (default: "c4v_cladding_sweep").

    Returns:
        Dictionary containing sweep data, reference frequencies, saved file paths, and summary.
    """
    t_start = time.time()

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
        print("3D PhC Slab Lower Cladding Index Sweep at Γ (k = [0, 0, 0])")
        print(f"Lattice pitch a:       {pitch:.3f} μm")
        print(
            f"Slab thickness h:      {slab_thickness:.3f} μm (h/a = {slab_thickness / pitch:.3f})"
        )
        print(f"Hole radii:            r1 = {r1:.3f} μm, r2 = {r2:.3f} μm")
        print(f"Slab core material:    {slab_material.upper()}")
        print(
            f"Lower cladding sweep:  n_sub = [{n_start:.3f} -> {n_end:.3f}] ({num_points} steps)"
        )
        print(f"Mesh resolution:       xy={resolution}, z={resolution_z}")
        print(f"Output directory:      {out_path}")
        print("=" * 72)

    # 2. Generate unit cell layout & export GDSII layout mask
    unit_cell = phc_wyckoff_unit_cell(
        pitch=pitch,
        point_group="C4v",
        features=[("1a", r1), ("1b", r2)],
    )
    gds_path = output_mgr.save_gds(unit_cell, filename="unit_cell.gds")
    if verbose:
        print(f"[1/4] Exported GDSII layout mask: {gds_path}")

    # 3. Reference simulations at symmetric air cladding (n_sub = 1.0)
    if verbose:
        print("\n[2/4] Computing baseline references at air cladding (n_sub = 1.0)...")

    # TE-like reference (z-even)
    t_ref_start = time.time()
    res_ref_te = run_gamma_simulation(
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
    freqs_ref_te = res_ref_te["freqs"]

    # TM-like reference (z-odd)
    res_ref_tm = run_gamma_simulation(
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
    freqs_ref_tm = res_ref_tm["freqs"]
    t_ref_elapsed = time.time() - t_ref_start

    if verbose:
        print(f"      -> Baseline references computed in {t_ref_elapsed:.2f} s.")
        print(f"      -> TE-like frequencies at Γ (n=1.0): {np.round(freqs_ref_te, 4)}")
        print(f"      -> TM-like frequencies at Γ (n=1.0): {np.round(freqs_ref_tm, 4)}")

    # 4. Save permittivity map for silica substrate (last point)
    with silence_c_stdout():
        eps_grid = get_epsilon_grid(
            ms=res_ref_te["ms"],
            rectify=True,
            periods=2,
            periods_z=1,
            resolution=32,
        )
    fig_eps = plot_epsilon(
        epsilon=eps_grid,
        title=f"3D PhC Slab Permittivity — {slab_material.upper()} Membrane in Air",
    )
    eps_file = output_mgr.save_figure(
        fig_eps, artifact_key="eps_plot", filename="epsilon_map.png"
    )
    plt.close(fig_eps)

    # 5. Sweep lower cladding index n_sub for all bands
    if verbose:
        print(
            f"\n[3/4] Sweeping lower cladding index n_sub across {num_points} points..."
        )

    n_values = np.linspace(n_start, n_end, num_points)
    sweep_freqs_list: list[np.ndarray] = []

    t_sweep_start = time.time()
    for idx, n_sub in enumerate(n_values, start=1):
        t_step = time.time()
        res_step = run_gamma_simulation(
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
        step_freqs = res_step["freqs"]
        sweep_freqs_list.append(step_freqs)
        if verbose:
            print(
                f"      [{idx:02d}/{num_points:02d}] n_sub = {n_sub:.3f} "
                f"-> solved {num_bands_sweep} bands in {time.time() - t_step:.2f} s"
            )

    sweep_freqs_matrix = np.array(
        sweep_freqs_list
    )  # shape: (num_points, num_bands_sweep)
    t_sweep_elapsed = time.time() - t_sweep_start

    # 6. Plotting: Unified figure with horizontal reference lines
    if verbose:
        print("\n[4/4] Generating sweep diagram with horizontal reference lines...")

    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=150)

    # Filter out DC electrostatic/null zero modes (f < 1e-4) for reference lines
    phys_ref_te = freqs_ref_te[freqs_ref_te > 1e-4]
    phys_ref_tm = freqs_ref_tm[freqs_ref_tm > 1e-4]

    # Draw horizontal reference lines for TE-like baseline modes
    for i, f_te in enumerate(phys_ref_te):
        ax.axhline(
            y=f_te,
            color="tab:blue",
            linestyle="--",
            linewidth=1.2,
            alpha=0.75,
            label="Ref TE-like (n=1.0)" if i == 0 else None,
        )

    # Draw horizontal reference lines for TM-like baseline modes
    for i, f_tm in enumerate(phys_ref_tm):
        ax.axhline(
            y=f_tm,
            color="tab:red",
            linestyle=":",
            linewidth=1.4,
            alpha=0.75,
            label="Ref TM-like (n=1.0)" if i == 0 else None,
        )

    # Plot swept bands (all bands without parity constraint)
    for b in range(num_bands_sweep):
        # Exclude DC zero-frequency null bands
        if np.max(sweep_freqs_matrix[:, b]) < 1e-4:
            continue
        ax.plot(
            n_values,
            sweep_freqs_matrix[:, b],
            marker="o",
            markersize=3.5,
            linestyle="-",
            linewidth=1.0,
            color="black",
            alpha=0.85,
            label="All bands (swept)" if b == 2 else None,
        )

    ax.set_xlabel(
        r"Lower Cladding Refractive Index $n_{\mathrm{sub}}$",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_ylabel(
        r"Normalized Frequency $\tilde{\omega} = \omega a / 2\pi c = a / \lambda$",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_title(
        f"C4v PhC Slab — Lower Cladding Index Sweep at Γ\n"
        f"(h/a = {slab_thickness / pitch:.2f}, r₁ = {r1:.2f} μm, r₂ = {r2:.2f} μm, {slab_material.upper()} in air)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlim(n_start - 0.02, n_end + 0.02)
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
            "sim_type": "cladding_sweep_gamma",
            "k_point": [0.0, 0.0, 0.0],
            "n_start": n_start,
            "n_end": n_end,
            "num_points": num_points,
            "resolution_xy": resolution,
            "resolution_z": resolution_z,
            "num_bands_ref": num_bands_ref,
            "num_bands_sweep": num_bands_sweep,
        },
        "reference_frequencies": {
            "te_like": freqs_ref_te.tolist(),
            "tm_like": freqs_ref_tm.tolist(),
        },
        "sweep_results": {
            "n_values": n_values.tolist(),
            "frequencies": sweep_freqs_matrix.tolist(),
        },
        "elapsed_time_s": {
            "reference_solve_s": t_ref_elapsed,
            "sweep_solve_s": t_sweep_elapsed,
            "total_s": t_total,
        },
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


if __name__ == "__main__":
    # Parameters matching the C4v slab unit cell
    res = run_cladding_sweep(
        pitch=1.0,
        slab_thickness=0.5,
        r1=0.25,
        r2=0.15,
        supercell_z=4.0,
        slab_material="inp",
        cladding_material="air",
        n_start=1.0,
        n_end=1.444,
        num_points=9,
        resolution=16,
        resolution_z=16,
        num_bands_ref=8,
        num_bands_sweep=16,
        sim_name="c4v_cladding_sweep",
    )
    plt.show()
