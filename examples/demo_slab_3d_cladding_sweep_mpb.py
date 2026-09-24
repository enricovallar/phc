#!/usr/bin/env python3
"""End-to-End Demonstration: 3D Photonic Crystal Slab Cladding Index Sweep.

Demonstrates tracking electromagnetic eigenmode frequencies as the cladding
refractive index is swept from air (n = 1.0) to silica (SiO2, n ≈ 1.444) for
an optimal 3D photonic crystal slab membrane:

1. Loads the optimal C4v square lattice unit cell geometry (r1, r2, pitch, slab_thickness)
   engineered in ``demo_optimization_3d``.
2. Exports the physical GDSII layout mask via ``phc_hydra.SimulationOutputManager``.
3. Step 1 (Dual Baselines Reference): Solves baseline unperturbed TE-like modes (zeven parity)
   and TM-like modes (zodd parity) at the unperturbed starting point (symmetric air cladding,
   n = 1.0) across the irreducible Brillouin zone (X -> Γ -> M -> X). Plots the full baseline
   reference band diagram (``band_structure_baseline_reference.png``) with TE (blue) and TM (red) modes.
4. Step 2 (Parametric Cladding Sweep): Sweeps the cladding refractive index n from 1.0 to 1.444.
   For each index n, solves ALL modes (polarization='all', no parity constraint) across the full k-path
   with sufficient bands (num_bands_sweep = 3 * num_bands) to track both TE and TM mode families,
   computes continuous modal TE/TM electric polarization fractions (f_TE) using
   dielectric slab core masking (|z - z_center| <= slab_thickness / 2), and renders the full
   band diagram for each simulation point (``band_structure_step_*.png``).
5. Step 3 (Final Overlapped Band Diagram): For the last simulation point (highest cladding index,
   n ≈ 1.444), overlaps the baseline reference parity modes (TE blue, TM red) and the current
   modes (colored by f_TE) on a single diagram (``band_structure_overlapped_with_reference.png``
   and canonical ``band_structure.png``) to visualize mode shifts, crossings, and hybridization.
6. Step 4 (Cladding Sweep Dispersion & Permittivity Profiling): Renders eigenfrequencies vs. refractive
   index at Γ, dual-plane permittivity distributions (``epsilon_step_*.png``), and a side-by-side
   vertical profile comparison (``epsilon_vertical_profiles_comparison.png``).
7. Step 5 (Structured Metadata Export): Exports results and artifacts to the canonical
   hierarchy ``outputs/mpb/cladding_sweep_3d/c4v_dirac_3d/<timestamp>/``.

Command-Line Usage:
    # Standard execution across 9 refractive index points:
    python examples/demo_slab_3d_cladding_sweep_mpb.py

    # Rapid smoke test (< 5 seconds, low resolution, 3 points, k-density=2):
    python examples/demo_slab_3d_cladding_sweep_mpb.py --quick

    # Custom index range and number of sweep steps:
    python examples/demo_slab_3d_cladding_sweep_mpb.py --n-start 1.0 --n-end 1.45 --num-points 10

    # Custom sweep bands:
    python examples/demo_slab_3d_cladding_sweep_mpb.py --num-bands 8 --num-bands-sweep 24

    # Symmetric cladding sweep (both top and bottom swept):
    python examples/demo_slab_3d_cladding_sweep_mpb.py --cladding-mode symmetric

    # Custom geometric parameters:
    python examples/demo_slab_3d_cladding_sweep_mpb.py --r1 0.2452 --r2 0.2326 --pitch 1.0 --thickness 0.5 --matrix-material inp

    # Auto-load optimal geometry from the latest 3D optimization run:
    python examples/demo_slab_3d_cladding_sweep_mpb.py --load-locus

CLI Options:
    --quick                   Run in rapid smoke-test mode with minimal resolution and sweep points.
    --r1 R1                   Primary hole radius at origin Wyckoff 1a (default: 0.2452 μm).
    --r2 R2                   Secondary hole radius at center Wyckoff 1b (default: 0.2326 μm).
    --pitch P                 Lattice pitch a in micrometers (default: 1.0 μm).
    --thickness H             Slab membrane thickness in micrometers (default: 0.5 μm).
    --supercell-z HEIGHT      Supercell vertical height in units of pitch a (default: 4.0).
    --matrix-material MAT     Slab core material key (default: 'inp' for InP, n ≈ 3.167).
    --n-start N               Starting cladding refractive index (default: 1.0 for air).
    --n-end N                 Ending cladding refractive index (default: 1.444 for SiO2).
    --num-points M            Number of sweep points between n-start and n-end (default: 9, quick: 3).
    --k-density K             K-points per high-symmetry segment (default: 8, quick: 2).
    --resolution RES          In-plane (x, y) mesh resolution per unit pitch a (default: 16, quick: 12).
    --resolution-z RESZ       Vertical (z) mesh resolution per unit pitch a (default: 20, quick: 6).
    --num-bands N             Number of baseline eigenbands computed for each polarization (default: 10, quick: 10).
    --num-bands-sweep NSWEEP  Number of eigenbands computed during cladding sweep (default: 25, quick: 20).
    --cladding-mode MODE      'substrate' (bottom cladding swept, top air) or 'symmetric' (both swept).
    --workers W               Number of parallel worker processes (default: 4, quick: 2).
    --output-dir PATH         Custom output directory override (default: auto-resolved by phc_hydra).
    --load-locus [PATH]       Auto-load or load optimal parameters from optimal_loci.json.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np

# Ensure repository root is in sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from phc_hydra import SimulationOutputManager, resolve_simulation_output_dir
from phc_layout.components.unit_cell import phc_wyckoff_unit_cell
from phc_layout.lattice import SquareLattice
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


def run_cladding_sweep_pipeline(
    r1: float = 0.2452,
    r2: float = 0.2326,
    pitch: float = 1.0,
    slab_thickness: float = 0.5,
    supercell_z: float = 4.0,
    matrix_material: str = "inp",
    n_start: float = 1.0,
    n_end: float = 1.444,
    num_index_points: int = 9,
    k_density: int = 8,
    resolution: int = 16,
    resolution_z: int = 20,
    num_bands: int = 10,
    num_bands_sweep: int | None = None,
    k_point: tuple[float, float, float] = (0.0, 0.0, 0.0),
    cladding_mode: Literal["substrate", "symmetric"] = "substrate",
    num_workers: int = 4,
    quick: bool = False,
    output_dir: Path | str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Runs the 3D PhC slab cladding refractive index sweep pipeline.

    Calculates unperturbed TE-like and TM-like eigenmode frequencies at the starting
    point (symmetric air membrane, n=1.0) to establish dual baseline horizontal reference lines,
    sweeps the cladding refractive index towards silica (n ≈ 1.444), solves all eigenmodes without
    parity constraints (polarization='all'), and plots frequency shifts and mode crossings colored
    by modal TE polarization fraction. Also computes and renders full band diagrams across the
    irreducible Brillouin zone (X -> Γ -> M -> X) for baseline reference, individual sweep steps,
    and an overlapped comparison for the final substrate point.

    Args:
        r1: Primary hole radius at Wyckoff position 1a (origin) in micrometers (default: 0.2452 μm).
        r2: Secondary hole radius at Wyckoff position 1b (center) in micrometers (default: 0.2326 μm).
        pitch: Lattice constant a in micrometers (default: 1.0 μm).
        slab_thickness: Slab membrane thickness h in micrometers (default: 0.5 μm).
        supercell_z: Vertical supercell height in units of pitch a (default: 4.0).
        matrix_material: Slab core material key (default: 'inp' for InP, n ≈ 3.167).
        n_start: Starting cladding refractive index (default: 1.0).
        n_end: Ending cladding refractive index (default: 1.444 for SiO2).
        num_index_points: Number of refractive index points along the sweep (default: 9).
        k_density: Number of k-points per high-symmetry segment (default: 8, quick: 2).
        resolution: In-plane (x, y) grid resolution per unit pitch a (default: 16).
        resolution_z: Vertical (z) grid resolution per unit pitch a (default: 20).
        num_bands: Number of eigenbands computed for each baseline polarization (default: 10).
        num_bands_sweep: Number of eigenbands computed during cladding sweep (default: 25).
        k_point: 3-tuple (kx, ky, kz) in reciprocal lattice coordinates (default: (0, 0, 0) for Gamma).
        cladding_mode: Cladding sweep configuration: 'substrate' (bottom cladding swept, top air)
            or 'symmetric' (both top and bottom claddings swept symmetrically).
        num_workers: Number of concurrent worker processes for MPB execution (default: 4).
        quick: If True, executes rapid smoke test with minimal settings (num_index_points=3,
            resolution=12, resolution_z=6, num_bands=10, num_bands_sweep=20, k_density=2).
        output_dir: Optional directory override. If None, resolves standard hierarchy:
            outputs/mpb/cladding_sweep_3d/c4v_dirac_3d/<timestamp>/.
        verbose: If True, prints formatted progress updates and execution timing.

    Returns:
        Dictionary containing:
            - 'base_te_freqs': 1D numpy array of baseline unperturbed TE eigenfrequencies at n_start.
            - 'base_tm_freqs': 1D numpy array of baseline unperturbed TM eigenfrequencies at n_start.
            - 'swept_indices': 1D numpy array of swept refractive index values.
            - 'freqs_vs_index': 2D numpy array of shape (num_index_points, num_bands_sweep) containing frequencies.
            - 'te_fracs_vs_index': 2D numpy array of shape (num_index_points, num_bands_sweep) with TE fractions.
            - 'band_plots': List of file paths to full band diagram plots for each sweep step.
            - 'band_structure_baseline': File path to baseline reference band diagram.
            - 'band_structure_overlapped': File path to overlapped band diagram for final step.
            - 'epsilon_plots': List of file paths to the individual dual-plane permittivity plots.
            - 'geometry': Dict with unit cell geometric parameters.
            - 'simulation': Dict with solver parameters.
            - 'output_dir': Path to resolved output directory.
            - 'files': Dict mapping artifact keys to file paths.
            - 'artifacts': Dict mapping artifact keys to file paths.
            - 'elapsed_time_s': Total pipeline execution time in seconds.
    """
    import meep as mp

    t_start = time.time()

    if quick:
        num_index_points = 3
        resolution = 12
        resolution_z = 6
        num_bands = 10
        k_density = 2
        if num_bands_sweep is None:
            num_bands_sweep = 20

    if num_bands_sweep is None:
        num_bands_sweep = 25

    out_path = resolve_simulation_output_dir(
        solver="mpb",
        sim_type="cladding_sweep_3d",
        geometry="c4v_dirac_3d",
        override_dir=output_dir,
    )
    output_mgr = SimulationOutputManager(
        output_dir=out_path,
        solver="mpb",
        sim_type="cladding_sweep_3d",
        geometry_name="c4v_dirac_3d",
    )

    slab_mat_spec = get_material(matrix_material)
    norm_slab_thickness = slab_thickness / pitch

    if verbose:
        print("\n" + "=" * 75)
        print("3D PhC Slab Cladding Refractive Index Sweep — Eigenmode Tracking")
        print(f"Lattice Pitch:       a = {pitch:.3f} μm")
        print(
            f"Slab Thickness:     h = {slab_thickness:.3f} μm (h/a = {norm_slab_thickness:.3f})"
        )
        print(f"Hole Radii:         r1 = {r1:.4f} μm, r2 = {r2:.4f} μm")
        print(
            f"Slab Material:      '{slab_mat_spec.name}' (n = {slab_mat_spec.index:.3f})"
        )
        print(
            f"Cladding Sweep:     n = [{n_start:.3f} -> {n_end:.3f}] across {num_index_points} points "
            f"(mode: {cladding_mode})"
        )
        print(f"Mesh Resolution:    r_xy = {resolution}, r_z = {resolution_z}")
        print(f"Bands at Γ:         {num_bands} eigenbands")
        print(f"Output Directory:   {out_path}")
        print("=" * 75 + "\n")

    # ------------------------------------------------------------------
    # Step 1: Layout Generation (phc_layout) & GDS Export (phc_hydra)
    # ------------------------------------------------------------------
    if verbose:
        print("[1/4] Generating C4v square lattice unit cell layout...")
    unit_cell = phc_wyckoff_unit_cell(
        pitch=pitch,
        point_group="C4v",
        features=[("1a", r1), ("1b", r2)],
    )
    gds_path = output_mgr.save_gds(unit_cell, filename="unit_cell.gds")
    if verbose:
        print(f"      -> Exported GDSII mask: {gds_path}")

    # Set up 3D MPB Lattice and high-symmetry k-path
    layout_lat = SquareLattice(a=pitch)
    mpb_lat = lattice_to_mpb_lattice(
        layout_lat,
        dimension="3D_slab",
        supercell=(1, 1),
        supercell_z=supercell_z,
        normalize=True,
    )
    k_pts, k_labels, k_indices = get_high_symmetry_kpath(
        layout_lat, k_density=k_density
    )
    if "Γ" in k_labels:
        gamma_idx = k_indices[k_labels.index("Γ")]
    elif "Gamma" in k_labels:
        gamma_idx = k_indices[k_labels.index("Gamma")]
    else:
        gamma_idx = 0

    # ------------------------------------------------------------------
    # Step 2: Baseline Dual Parity Simulation at n = n_start (Air Membrane)
    # ------------------------------------------------------------------
    if verbose:
        print(
            f"[2/4] Solving baseline unperturbed TE-like and TM-like modes at starting index n = {n_start:.3f}..."
        )
    mpb_geom_base = gds_to_mpb_geometry(
        gds_source=gds_path,
        pitch=pitch,
        dimension="3D_slab",
        slab_thickness=slab_thickness,
        slab_material=matrix_material,
        substrate_material=None,  # pure membrane with symmetric air background
        etch_material="air",
        geometry_lattice=mpb_lat,
    )

    # A. Baseline TE-like modes across full k-path (z-odd H / z-even E)
    res_base_te = run_parallel_band_solver(
        geometry_lattice=mpb_lat,
        geometry=mpb_geom_base,
        k_points=k_pts,
        default_material=mp.Medium(index=float(n_start)),
        resolution=resolution,
        resolution_z=resolution_z,
        num_bands=num_bands,
        polarization="te_like",
        dimension="3D_slab",
        cladding_index=float(n_start),
        num_workers=num_workers,
        verbose=False,
    )
    base_te_freqs = np.copy(res_base_te["freqs"]["te_like"][gamma_idx, :])

    # B. Baseline TM-like modes across full k-path (z-even H / z-odd E)
    res_base_tm = run_parallel_band_solver(
        geometry_lattice=mpb_lat,
        geometry=mpb_geom_base,
        k_points=k_pts,
        default_material=mp.Medium(index=float(n_start)),
        resolution=resolution,
        resolution_z=resolution_z,
        num_bands=num_bands,
        polarization="tm_like",
        dimension="3D_slab",
        cladding_index=float(n_start),
        num_workers=num_workers,
        verbose=False,
    )
    base_tm_freqs = np.copy(res_base_tm["freqs"]["tm_like"][gamma_idx, :])

    # C. Full baseline reference band diagram (TE-like and TM-like modes)
    results_base_combined = {
        "freqs": {
            "te_like": res_base_te["freqs"]["te_like"],
            "tm_like": res_base_tm["freqs"]["tm_like"],
        },
        "light_line": res_base_te.get("light_line", []),
        "dimension": "3D_slab",
    }
    fig_base = plot_band_structure(
        results_base_combined,
        k_labels=k_labels,
        k_indices=k_indices,
        title=rf"Baseline Reference ($n_{{\mathrm{{clad}}}} = {n_start:.3f}$, Air Membrane) — TE & TM Modes",
    )
    base_band_path = output_mgr.save_figure(
        fig_base,
        artifact_key="band_structure_baseline_reference",
        filename="band_structure_baseline_reference.png",
    )
    plt.close(fig_base)

    if verbose:
        print(
            f"      -> Computed {len(base_te_freqs)} baseline TE-like eigenfrequencies at Γ:"
        )
        for idx, f_val in enumerate(base_te_freqs, start=1):
            print(f"         Band {idx} (TE): ω̃ = {f_val:.5f}")
        print(
            f"      -> Computed {len(base_tm_freqs)} baseline TM-like eigenfrequencies at Γ:"
        )
        for idx, f_val in enumerate(base_tm_freqs, start=1):
            print(f"         Band {idx} (TM): ω̃ = {f_val:.5f}")
        print(f"      -> Exported baseline band diagram: {base_band_path}")

    # ------------------------------------------------------------------
    # Step 3: Parametric Cladding Refractive Index Sweep (polarization='all')
    # ------------------------------------------------------------------
    if verbose:
        print(
            f"[3/4] Sweeping cladding refractive index across {num_index_points} points "
            f"(solving {num_bands_sweep} bands per point across full k-path)..."
        )

    n_values = np.linspace(n_start, n_end, num_index_points)
    all_freqs_list: list[np.ndarray] = []
    all_te_fracs_list: list[np.ndarray] = []
    band_plot_paths: list[str] = []
    eps_plot_paths: list[str] = []
    xz_slices: list[tuple[float, np.ndarray]] = []
    last_eps_grid: np.ndarray | None = None
    res_last: dict[str, Any] | None = None

    t_sweep_start = time.time()
    for step_i, n_clad in enumerate(n_values, start=1):
        # Configure geometry based on cladding_mode
        if cladding_mode == "symmetric":
            bg_medium: Any = mp.Medium(index=float(n_clad))
            mpb_geom = gds_to_mpb_geometry(
                gds_source=gds_path,
                pitch=pitch,
                dimension="3D_slab",
                slab_thickness=slab_thickness,
                slab_material=matrix_material,
                substrate_material=None,
                etch_material=float(n_clad),
                geometry_lattice=mpb_lat,
            )
            clad_idx_for_ll = float(n_clad)
        else:
            # Substrate mode: bottom substrate swept, top cladding is air (n=1.0)
            bg_medium = "air"
            mpb_geom = gds_to_mpb_geometry(
                gds_source=gds_path,
                pitch=pitch,
                dimension="3D_slab",
                slab_thickness=slab_thickness,
                slab_material=matrix_material,
                substrate_material=float(n_clad),
                etch_material="air",
                geometry_lattice=mpb_lat,
            )
            clad_idx_for_ll = float(n_clad)

        # Mode solver instance for epsilon grid extraction
        ms_step = create_mode_solver(
            geometry_lattice=mpb_lat,
            geometry=mpb_geom,
            k_points=[k_pts[gamma_idx]],
            default_material=bg_medium,
            resolution=resolution,
            resolution_z=resolution_z,
            num_bands=num_bands_sweep,
        )

        res_step = run_parallel_band_solver(
            geometry_lattice=mpb_lat,
            geometry=mpb_geom,
            k_points=k_pts,
            default_material=bg_medium,
            resolution=resolution,
            resolution_z=resolution_z,
            num_bands=num_bands_sweep,
            polarization="all",
            dimension="3D_slab",
            cladding_index=clad_idx_for_ll,
            compute_polarization_fractions=True,
            polarization_method="slab",
            slab_thickness=norm_slab_thickness,
            z_center=0.0,
            num_workers=num_workers,
            verbose=False,
        )

        step_freqs = res_step["freqs"]["all"][gamma_idx, :]
        step_te = res_step["te_fractions"][gamma_idx, :]

        all_freqs_list.append(step_freqs)
        all_te_fracs_list.append(step_te)

        # Full band diagram for this step
        band_fig_step = plot_band_structure(
            res_step,
            k_labels=k_labels,
            k_indices=k_indices,
            title=rf"3D PhC Slab ($n_{{\mathrm{{clad}}}} = {n_clad:.3f}$) — Full Band Structure",
        )
        band_filename = f"band_structure_step_{step_i:02d}_n_{n_clad:.3f}.png"
        band_path = output_mgr.save_figure(
            band_fig_step,
            artifact_key=f"band_step_{step_i}",
            filename=band_filename,
        )
        plt.close(band_fig_step)
        band_plot_paths.append(str(band_path))

        # Retrieve dielectric permittivity grid using default MPB method
        eps_step = get_epsilon_grid(
            ms_step,
            rectify=True,
            periods=2,
            periods_z=1,
            resolution=resolution,
        )
        last_eps_grid = eps_step

        # Plot dual-plane permittivity distribution using default method
        eps_fig = plot_epsilon(
            eps_step,
            title=rf"3D PhC Slab ($n_{{\mathrm{{clad}}}} = {n_clad:.3f}$, $\varepsilon_{{\mathrm{{clad}}}} = {n_clad**2:.2f}$) — $\varepsilon(\mathbf{{r}})$",
            extent=(-pitch, pitch, -pitch, pitch),
            extent_z=(-supercell_z * pitch / 2, supercell_z * pitch / 2),
            xlabel=r"$x$ (μm)",
            ylabel=r"$y$ (μm)",
            zlabel=r"$z$ (μm)",
            title_xy=rf"In-Plane Mid-Plane ($z=0,\ n_{{\mathrm{{clad}}}}={n_clad:.3f}$)",
            title_xz=rf"Vertical Cross-Section ($y=0,\ n_{{\mathrm{{clad}}}}={n_clad:.3f}$)",
            interpolation="none",
        )
        eps_filename = f"epsilon_step_{step_i:02d}_n_{n_clad:.3f}.png"
        eps_path = output_mgr.save_figure(
            eps_fig,
            artifact_key=f"eps_step_{step_i}",
            filename=eps_filename,
        )
        plt.close(eps_fig)
        eps_plot_paths.append(str(eps_path))

        mid_y = eps_step.shape[1] // 2
        xz_slices.append((n_clad, eps_step[:, mid_y, :]))

        if step_i == num_index_points:
            res_last = res_step

        if verbose:
            print(
                f"      [{step_i}/{num_index_points}] n_clad = {n_clad:.3f} | "
                f"Bands at Γ: [{step_freqs[0]:.4f} .. {step_freqs[-1]:.4f}] | "
                f"f_TE range: [{step_te.min():.2f} .. {step_te.max():.2f}] | "
                f"Band plot: {band_filename} | Eps: {eps_filename}"
            )

    t_sweep_end = time.time()
    sweep_duration = t_sweep_end - t_sweep_start
    if verbose:
        print(f"      -> Sweep completed in {sweep_duration:.2f} seconds.")

    freqs_arr = np.array(all_freqs_list)  # (M, num_bands_sweep)
    te_fracs_arr = np.array(all_te_fracs_list)  # (M, num_bands_sweep)

    # ------------------------------------------------------------------
    # Step 4: Publication-Quality Plotting & Artifacts Export
    # ------------------------------------------------------------------
    if verbose:
        print(
            "[4/4] Rendering frequency vs. refractive index diagram and saving artifacts..."
        )

    fig, ax = plt.subplots(figsize=(9.5, 6.5), dpi=150)

    # A. Draw horizontal dotted lines for baseline TE-like eigenfrequencies (Blue)
    for b_idx, w_ref in enumerate(base_te_freqs):
        label_str = "Baseline TE-like (Air Membrane)" if b_idx == 0 else None
        ax.axhline(
            w_ref,
            color="#1f77b4",  # Blue
            linestyle=":",
            linewidth=1.4,
            alpha=0.8,
            label=label_str,
            zorder=1,
        )

    # B. Draw horizontal dotted lines for baseline TM-like eigenfrequencies (Red)
    for b_idx, w_ref in enumerate(base_tm_freqs):
        label_str = "Baseline TM-like (Air Membrane)" if b_idx == 0 else None
        ax.axhline(
            w_ref,
            color="#d62728",  # Red
            linestyle=":",
            linewidth=1.4,
            alpha=0.8,
            label=label_str,
            zorder=1,
        )

    # C. Thin lines connecting eigenbands across refractive index points
    for b in range(num_bands_sweep):
        ax.plot(
            n_values,
            freqs_arr[:, b],
            color="#bbbbbb",
            linewidth=0.8,
            alpha=0.6,
            zorder=2,
        )

    # D. Scatter points colored continuously by modal TE fraction
    cmap = plt.get_cmap("coolwarm_r")  # 1.0 (TE) = Blue, 0.0 (TM) = Red
    sc = ax.scatter(
        np.repeat(n_values, num_bands_sweep),
        freqs_arr.flatten(),
        c=te_fracs_arr.flatten(),
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        s=36,
        edgecolors="#222222",
        linewidths=0.5,
        zorder=4,
    )

    # Format colorbar with physical annotations
    cbar = fig.colorbar(sc, ax=ax, pad=0.03, aspect=25)
    cbar.set_label(
        r"Modal TE Polarization Fraction $f_{\mathrm{TE}}$", fontsize=11, labelpad=8
    )
    cbar.ax.tick_params(labelsize=9)
    cbar.ax.text(
        1.3,
        0.05,
        "TM-like",
        transform=cbar.ax.transAxes,
        color="#b22222",
        va="bottom",
        ha="left",
        fontweight="bold",
        fontsize=9,
    )
    cbar.ax.text(
        1.3,
        0.95,
        "TE-like",
        transform=cbar.ax.transAxes,
        color="#1f4e79",
        va="top",
        ha="left",
        fontweight="bold",
        fontsize=9,
    )

    # Formatting plot aesthetics
    ax.set_xlabel(
        rf"Cladding Refractive Index $n_{{\mathrm{{clad}}}}$ ({cladding_mode.capitalize()} Mode)",
        fontsize=12,
        labelpad=6,
    )
    ax.set_ylabel(
        r"Normalized Frequency $\tilde{\omega} = \omega a / 2\pi c = a / \lambda$",
        fontsize=12,
        labelpad=6,
    )
    ax.set_title(
        r"3D PhC Slab: Eigenmode Frequency vs. Cladding Refractive Index"
        f"\n($C_{{4v}}$ Square Lattice, $r_1={r1:.3f}a$, $r_2={r2:.3f}a$, $h={norm_slab_thickness:.2f}a$)",
        fontsize=12,
        pad=10,
    )
    ax.set_xlim(n_start - 0.02, n_end + 0.02)
    y_top = max(float(base_te_freqs.max()), float(base_tm_freqs.max())) * 1.15
    ax.set_ylim(-0.01, min(float(freqs_arr.max()) + 0.02, y_top))
    ax.grid(True, linestyle=":", alpha=0.5, color="gray")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=10)

    plt.tight_layout()
    plot_path = output_mgr.save_figure(
        fig, artifact_key="sweep_plot", filename="cladding_sweep.png"
    )
    plt.close(fig)

    # B. Canonical epsilon_map.png (representing the endpoint with full substrate)
    can_eps_path: Path | None = None
    if last_eps_grid is not None:
        can_eps_fig = plot_epsilon(
            last_eps_grid,
            title=rf"3D PhC Slab ($n_{{\mathrm{{clad}}}} = {n_values[-1]:.3f}$) — Canonical $\varepsilon(\mathbf{{r}})$",
            extent=(-pitch, pitch, -pitch, pitch),
            extent_z=(-supercell_z * pitch / 2, supercell_z * pitch / 2),
            xlabel=r"$x$ (μm)",
            ylabel=r"$y$ (μm)",
            zlabel=r"$z$ (μm)",
            title_xy=rf"In-Plane Mid-Plane ($z=0,\ n_{{\mathrm{{clad}}}}={n_values[-1]:.3f}$)",
            title_xz=rf"Vertical Cross-Section ($y=0,\ n_{{\mathrm{{clad}}}}={n_values[-1]:.3f}$)",
            interpolation="none",
        )
        can_eps_path = output_mgr.save_figure(
            can_eps_fig,
            artifact_key="eps_plot",
            filename="epsilon_map.png",
        )
        plt.close(can_eps_fig)

    # C. Multi-Panel Vertical Permittivity Profiles Comparison Across All Simulation Points
    vert_comp_path: Path | None = None
    if xz_slices:
        fig_vert, axes_vert = plt.subplots(
            1,
            len(n_values),
            figsize=(max(3.6 * len(n_values), 8.0), 4.6),
            dpi=150,
            sharey=True,
        )
        if len(n_values) == 1:
            axes_vert = np.array([axes_vert])
        extent_xz = (
            -pitch,
            pitch,
            -supercell_z * pitch / 2,
            supercell_z * pitch / 2,
        )
        h_half = slab_thickness / 2.0

        for ax_v, (n_c, xz_data) in zip(axes_vert, xz_slices):
            ax_v.imshow(
                xz_data.T,
                origin="lower",
                extent=extent_xz,
                cmap="viridis",
                interpolation="none",
                aspect="auto",
            )
            ax_v.axhline(h_half, color="red", linestyle="--", linewidth=1.0, alpha=0.8)
            ax_v.axhline(-h_half, color="red", linestyle="--", linewidth=1.0, alpha=0.8)
            ax_v.set_title(
                rf"$n_{{\mathrm{{clad}}}} = {n_c:.3f}$",
                fontsize=11,
                fontweight="bold",
            )
            ax_v.set_xlabel(r"$x$ (μm)", fontsize=10)
            ax_v.grid(True, linestyle=":", alpha=0.35, color="gray")

        axes_vert[0].set_ylabel(r"$z$ (μm)", fontsize=10)
        fig_vert.suptitle(
            r"3D PhC Slab: Vertical Permittivity Profiles $\varepsilon(x, y=0, z)$ Across Sweep"
            rf"\n(Slab core $z \in [\pm {h_half:.2f}\,\mu\mathrm{{m}}]$ highlighted by dashed red lines)",
            fontsize=12,
            fontweight="bold",
            y=0.99,
        )
        plt.tight_layout()
        vert_comp_path = output_mgr.save_figure(
            fig_vert,
            artifact_key="vertical_comparison",
            filename="epsilon_vertical_profiles_comparison.png",
        )
        plt.close(fig_vert)

    # D. Overlapped Band Diagram: Baseline Reference (Air Membrane) vs. Final Substrate Point
    overlap_path: Path | None = None
    can_band_path: Path | None = None
    if res_last is not None:
        results_overlap = {
            "freqs": {
                "all": res_last["freqs"]["all"],
                "te_like": res_base_te["freqs"]["te_like"],
                "tm_like": res_base_tm["freqs"]["tm_like"],
            },
            "te_fractions": res_last.get("te_fractions"),
            "light_line": res_last.get("light_line", res_base_te.get("light_line", [])),
            "dimension": "3D_slab",
        }
        fig_overlap = plot_band_structure(
            results_overlap,
            k_labels=k_labels,
            k_indices=k_indices,
            title=rf"Overlapped Bands: Baseline Reference vs. Substrate ($n_{{\mathrm{{clad}}}} = {n_values[-1]:.3f}$)",
            hollow_fractions=True,
        )
        overlap_path = output_mgr.save_figure(
            fig_overlap,
            artifact_key="band_structure_overlapped",
            filename="band_structure_overlapped_with_reference.png",
        )
        # Canonical band_structure.png per AGENTS.md rule 3
        can_band_path = output_mgr.save_figure(
            fig_overlap,
            artifact_key="band_structure",
            filename="band_structure.png",
        )
        plt.close(fig_overlap)

    # JSON Metadata summary
    t_total = time.time() - t_start
    summary_data = {
        "geometry": {
            "point_group": "C4v",
            "lattice_type": "square",
            "pitch_um": pitch,
            "slab_thickness_um": slab_thickness,
            "r1_um": r1,
            "r2_um": r2,
            "supercell_z": supercell_z,
            "matrix_material": matrix_material,
        },
        "simulation": {
            "solver": "mpb",
            "dimension": "3D_slab",
            "k_point": list(k_point),
            "k_density": k_density,
            "k_labels": k_labels,
            "k_indices": k_indices,
            "resolution_xy": resolution,
            "resolution_z": resolution_z,
            "num_bands_baseline": num_bands,
            "num_bands_sweep": num_bands_sweep,
            "cladding_mode": cladding_mode,
            "n_start": n_start,
            "n_end": n_end,
            "num_index_points": num_index_points,
        },
        "timing": {
            "sweep_duration_s": round(sweep_duration, 3),
            "total_duration_s": round(t_total, 3),
        },
        "baseline_te_frequencies": base_te_freqs.tolist(),
        "baseline_tm_frequencies": base_tm_freqs.tolist(),
        "swept_refractive_indices": n_values.tolist(),
        "frequencies_vs_index": freqs_arr.tolist(),
        "te_fractions_vs_index": te_fracs_arr.tolist(),
        "band_structure_baseline": str(base_band_path),
        "band_structure_overlapped": str(overlap_path) if overlap_path else None,
        "band_plots": band_plot_paths,
        "epsilon_plots": eps_plot_paths,
        "epsilon_vertical_comparison": str(vert_comp_path) if vert_comp_path else None,
    }

    json_path = output_mgr.save_results_json(
        geometry_cfg=summary_data["geometry"],
        simulation_cfg=summary_data["simulation"],
        band_gaps=[],
        extra_data={
            "timing": summary_data["timing"],
            "baseline_te_frequencies": summary_data["baseline_te_frequencies"],
            "baseline_tm_frequencies": summary_data["baseline_tm_frequencies"],
            "swept_refractive_indices": summary_data["swept_refractive_indices"],
            "frequencies_vs_index": summary_data["frequencies_vs_index"],
            "te_fractions_vs_index": summary_data["te_fractions_vs_index"],
            "band_structure_baseline": str(base_band_path),
            "band_structure_overlapped": str(overlap_path) if overlap_path else None,
            "band_plots": band_plot_paths,
            "epsilon_plots": eps_plot_paths,
            "epsilon_vertical_comparison": str(vert_comp_path)
            if vert_comp_path
            else None,
        },
        filename="simulation_results.json",
    )

    if verbose:
        print("\n" + "-" * 75)
        print("Summary of Generated Artifacts:")
        print(f"  * GDS Layout:            {gds_path}")
        print(f"  * Cladding Sweep Plot:   {plot_path}")
        print(f"  * Baseline Band Diagram: {base_band_path}")
        if overlap_path:
            print(f"  * Overlapped Band Plot:  {overlap_path}")
        if can_band_path:
            print(f"  * Canonical Band Plot:   {can_band_path}")
        print(f"  * Step Band Diagrams:    {len(band_plot_paths)} files saved")
        if can_eps_path:
            print(f"  * Canonical Epsilon Map: {can_eps_path}")
        if vert_comp_path:
            print(f"  * Vertical Comparison:   {vert_comp_path}")
        print(f"  * Point Permittivity Maps: {len(eps_plot_paths)} files saved")
        print(f"  * Metadata JSON:         {json_path}")
        print(
            f"  * Total Run Time:        {t_total:.2f}s (Sweep: {sweep_duration:.2f}s)"
        )
        print("-" * 75 + "\n")

    return {
        "base_te_freqs": base_te_freqs,
        "baseline_te_freqs": base_te_freqs,
        "base_tm_freqs": base_tm_freqs,
        "baseline_tm_freqs": base_tm_freqs,
        "swept_indices": n_values,
        "n_clad_values": n_values,
        "freqs_vs_index": freqs_arr,
        "freqs": freqs_arr,
        "te_fracs_vs_index": te_fracs_arr,
        "te_fractions": te_fracs_arr,
        "band_plots": band_plot_paths,
        "band_structure_baseline": str(base_band_path),
        "band_structure_overlapped": str(overlap_path) if overlap_path else None,
        "band_structure": str(can_band_path) if can_band_path else None,
        "epsilon_plots": eps_plot_paths,
        "epsilon_vertical_comparison": str(vert_comp_path) if vert_comp_path else None,
        "geometry": summary_data["geometry"],
        "simulation": summary_data["simulation"],
        "output_dir": str(out_path),
        "files": {
            "gds": str(gds_path),
            "sweep_plot": str(plot_path),
            "band_structure": str(can_band_path) if can_band_path else None,
            "band_structure_baseline": str(base_band_path),
            "band_structure_overlapped": str(overlap_path) if overlap_path else None,
            "epsilon_map": str(can_eps_path) if can_eps_path else None,
            "vertical_comparison": str(vert_comp_path) if vert_comp_path else None,
            "results_json": str(json_path),
            "manifest": output_mgr.manifest,
        },
        "artifacts": {
            "gds": str(gds_path),
            "sweep_plot": str(plot_path),
            "band_structure": str(can_band_path) if can_band_path else None,
            "band_structure_baseline": str(base_band_path),
            "band_structure_overlapped": str(overlap_path) if overlap_path else None,
            "epsilon_map": str(can_eps_path) if can_eps_path else None,
            "vertical_comparison": str(vert_comp_path) if vert_comp_path else None,
            "results_json": str(json_path),
        },
        "elapsed_time_s": t_total,
    }


def main() -> None:
    """CLI entrypoint for running the 3D PhC slab cladding refractive index sweep."""
    parser = argparse.ArgumentParser(
        description="3D Photonic Crystal Slab — Cladding Refractive Index Sweep and Eigenmode Tracking."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run in rapid smoke-test mode with minimal resolution and sweep points.",
    )
    parser.add_argument(
        "--r1",
        type=float,
        default=0.2452,
        help="Primary hole radius at origin 1a (default: 0.2452 μm).",
    )
    parser.add_argument(
        "--r2",
        type=float,
        default=0.2326,
        help="Secondary hole radius at center 1b (default: 0.2326 μm).",
    )
    parser.add_argument(
        "--pitch",
        type=float,
        default=1.0,
        help="Lattice pitch a in micrometers (default: 1.0 μm).",
    )
    parser.add_argument(
        "--thickness",
        type=float,
        default=0.5,
        help="Slab membrane thickness in micrometers (default: 0.5 μm).",
    )
    parser.add_argument(
        "--supercell-z",
        type=float,
        default=4.0,
        help="Supercell vertical height in units of pitch a (default: 4.0).",
    )
    parser.add_argument(
        "--matrix-material",
        type=str,
        default="inp",
        help="Slab core dielectric material key (default: 'inp', n ≈ 3.167).",
    )
    parser.add_argument(
        "--n-start",
        type=float,
        default=1.0,
        help="Starting cladding refractive index (default: 1.0).",
    )
    parser.add_argument(
        "--n-end",
        type=float,
        default=1.444,
        help="Ending cladding refractive index (default: 1.444 for SiO2).",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=9,
        help="Number of sweep points between n-start and n-end (default: 9).",
    )
    parser.add_argument(
        "--k-density",
        type=int,
        default=8,
        help="Number of k-points per high-symmetry segment (default: 8, quick: 2).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=16,
        help="In-plane mesh resolution per unit pitch a (default: 16).",
    )
    parser.add_argument(
        "--resolution-z",
        type=int,
        default=20,
        help="Vertical mesh resolution along z (default: 20).",
    )
    parser.add_argument(
        "--num-bands",
        type=int,
        default=10,
        help="Number of baseline eigenbands computed for each polarization at Gamma (default: 10).",
    )
    parser.add_argument(
        "--num-bands-sweep",
        type=int,
        default=25,
        help="Number of eigenbands computed during cladding sweep (default: 25).",
    )
    parser.add_argument(
        "--cladding-mode",
        choices=["substrate", "symmetric"],
        default="substrate",
        help="Cladding configuration: 'substrate' (bottom swept, top air) or 'symmetric' (both swept).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker processes (default: 4).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory override.",
    )
    parser.add_argument(
        "--load-locus",
        nargs="?",
        const="latest",
        default=None,
        help="Auto-load or load optimal parameters from optimal_loci.json.",
    )
    args = parser.parse_args()

    r1_val = args.r1
    r2_val = args.r2

    # Optionally load optimal parameters from locus
    if args.load_locus is not None:
        from phc_optimization import (
            find_latest_locus_path,
            find_target_locus_point,
            load_loci_from_json,
        )

        locus_p = (
            find_latest_locus_path(geometry="c4v_dirac_3d")
            if args.load_locus in ("latest", "auto", "true")
            else Path(args.load_locus)
        )
        if locus_p and locus_p.is_file():
            loci = load_loci_from_json(locus_p)
            if loci and "x1" in loci[0] and "x2" in loci[0]:
                locus = loci[0]
                # Target Point 5 (omega_D ≈ 0.5809, as evaluated in 3D substrate comparison)
                _idx, target_info = find_target_locus_point(
                    locus=locus,
                    target_frequency=0.5809,
                    slab_thickness=args.thickness,
                    pitch=args.pitch,
                )
                r1_val = float(target_info.get("r1", locus["x1"][0]))
                r2_val = float(target_info.get("r2", locus["x2"][0]))
                print(
                    f"Loaded optimal target design from '{locus_p}' (Point #{target_info.get('point_idx', '?')}): "
                    f"r1={r1_val:.4f}, r2={r2_val:.4f} (omega_D={target_info.get('omega_d', 0):.4f})"
                )

    run_cladding_sweep_pipeline(
        r1=r1_val,
        r2=r2_val,
        pitch=args.pitch,
        slab_thickness=args.thickness,
        supercell_z=args.supercell_z,
        matrix_material=args.matrix_material,
        n_start=args.n_start,
        n_end=args.n_end,
        num_index_points=args.num_points,
        k_density=args.k_density,
        resolution=args.resolution,
        resolution_z=args.resolution_z,
        num_bands=args.num_bands,
        num_bands_sweep=args.num_bands_sweep,
        cladding_mode=args.cladding_mode,
        num_workers=args.workers,
        quick=args.quick,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
