#!/usr/bin/env python3
"""Modal Overlap Analysis and Mode Tracking for 3D C6v 2b-6d hBN Slab under Substrate Perturbation.

Evaluates how the triplet of degenerate/near-degenerate TE-like eigenmodes at the zone-center
Γ point (k = [0, 0, 0]) near λ ≈ 440 nm (pitch a = 0.381 μm) splits and redshifts when the
lower cladding is transitioned from symmetric air (n_sub = 1.0) to a dielectric substrate
(e.g., silica SiO2, n_sub ≈ 1.444).

Physics & Methodology:
1. Generates the C6v hexagonal lattice unit cell layout with Wyckoff 2b holes (inner radius r1)
   and Wyckoff 6d holes (satellite radius r2, radial position p2) etched into an hBN slab.
2. Exports the physical GDSII layout mask (unit_cell.gds) and permittivity cross-section (epsilon_map.png).
3. Solves the unperturbed symmetric membrane (n_sub = 1.0) at Γ and identifies the 3 TE-like modes
   at λ ≈ 440 nm.
4. Solves the perturbed asymmetric membrane with lower substrate (e.g. n_sub = 1.444).
5. Computes the modal overlap matrix:
       η_ij = |<E_i,ref | D_j,sub> * <D_i,ref | E_j,sub>| / (U_i,ref * U_j,sub)
   using phc_mpb.classification.compute_mode_overlap_matrix restricted to the slab core.
6. Evaluates the degenerate subspace projection:
       P_subspace(target_j) = sum_{i in triplet} η_ij
   identifying which substrate modes capture the original manifold and quantifying their
   splitting and redshift.
7. Generates unified visualization (mode_tracking.png):
   - Overlap matrix heatmap between reference and substrate eigenmodes.
   - Subspace projection bar chart showing energy inheritance.
   - Mode connection diagram showing frequency pull-down and wavelength shift (nm).
8. Exports structured JSON summary (simulation_results.json) to canonical Hydra path:
   outputs/mpb/mode_tracking/c6v_2b_6d/<timestamp>/.

Command-Line Usage:
    # Standard run (nominal resolution, SiO2 substrate n = 1.444)
    python analysis/test_c6v_mode_tracking.py

    # Quick smoke test execution (< 2 seconds)
    python analysis/test_c6v_mode_tracking.py --quick

    # Custom substrate index and target wavelength
    python analysis/test_c6v_mode_tracking.py --substrate-index 1.50 --target-wavelength 0.44

    # Custom mesh resolution and band search range
    python analysis/test_c6v_mode_tracking.py --resolution 30 --num-bands 28
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
    compute_modal_metrics,
    create_lattice,
    create_mode_solver,
    gds_to_mpb_geometry,
    get_epsilon_grid,
    plot_epsilon,
    run_band_solver,
    track_modes_by_overlap,
)
from phc_utils import export_gds, silence_c_stdout

# Suppress Meep/MPB C-level verbosity by default
mp.verbosity(0)


def run_gamma_mode_solver(
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
    """Runs a 3D PhC slab MPB simulation at the Γ point (k = [0, 0, 0]).

    Args:
        gds_path: Path to the exported GDSII unit cell file.
        pitch: Lattice pitch constant a in micrometers (default: 0.381 μm).
        slab_thickness: Slab membrane thickness in micrometers (default: 0.1 μm).
        supercell_z: Supercell height in units of lattice constant a (default: 4.0).
        slab_material: Dielectric slab core material key (default: "hBN").
        cladding_material: Top and background cladding material key (default: "air").
        substrate_index: Refractive index of lower substrate cladding (default: 1.0).
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

    k_points = [mp.Vector3(0.0, 0.0, 0.0)]

    with silence_c_stdout():
        ms = create_mode_solver(
            geometry_lattice=lattice,
            geometry=geometry,
            k_points=k_points,
            default_material=mat_clad,
            resolution=(resolution, resolution, resolution_z),
            num_bands=num_bands,
        )

        results = run_band_solver(
            ms=ms,
            polarization=polarization,
            dimension="3D_slab",
            cladding_index=mat_clad.index,
            num_workers=1,
            verbose=verbose,
        )

    freqs = (
        np.array(ms.all_freqs[0]) if hasattr(ms, "all_freqs") else np.zeros(num_bands)
    )

    return {
        "results": results,
        "ms": ms,
        "frequencies": freqs,
    }


def find_modes_near_wavelength(
    frequencies: np.ndarray,
    pitch: float,
    target_wavelength_um: float = 0.440,
    count: int = 3,
) -> list[int]:
    """Identifies the 1-based band indices closest to a target free-space wavelength.

    Args:
        frequencies: 1D array of normalized frequencies omega_tilde = a / lambda.
        pitch: Lattice pitch constant a in micrometers.
        target_wavelength_um: Target wavelength in micrometers (default: 0.440 μm).
        count: Number of closest modes to select (default: 3).

    Returns:
        Sorted list of 1-based band indices.
    """
    wavelengths = np.zeros_like(frequencies)
    valid = frequencies > 0
    wavelengths[valid] = pitch / frequencies[valid]

    diffs = np.abs(wavelengths - target_wavelength_um)
    # Zero frequency bands have diffs = target_wavelength_um, ensure we only pick positive freqs
    diffs[~valid] = np.inf

    best_idx = np.argsort(diffs)[:count]
    return sorted(int(b + 1) for b in best_idx)


def plot_mode_tracking_summary(
    overlap_matrix: np.ndarray,
    ref_bands: list[int],
    target_bands: list[int],
    ref_freqs: np.ndarray,
    target_freqs: np.ndarray,
    subspace_projection: np.ndarray,
    best_matches: list[dict[str, Any]],
    pitch: float,
    substrate_index: float,
    output_path: Path | str,
    lam_min_nm: float = 350.0,
    lam_max_nm: float = 550.0,
) -> None:
    """Plots a comprehensive multi-panel diagnostic figure for mode tracking.

    Panels:
    1. Overlap Matrix Heatmap: Coupling between each reference mode and target bands.
    2. Subspace Projection Bar Chart: Total overlap of each substrate mode with the reference triplet.
    3. Mode Redshift & Splitting: Energy levels at n=1.0 vs n=n_sub with connection lines.

    Args:
        overlap_matrix: 2D array of shape (len(ref_bands), len(target_bands)).
        ref_bands: List of 1-based reference band indices.
        target_bands: List of 1-based target band indices.
        ref_freqs: 1D array of reference frequencies at Γ.
        target_freqs: 1D array of target frequencies at Γ.
        subspace_projection: 1D array of subspace overlap values per target band.
        best_matches: List of match summaries for each reference band.
        pitch: Lattice pitch constant in micrometers.
        substrate_index: Refractive index of target substrate.
        output_path: Filepath where the plot image will be saved.
        lam_min_nm: Minimum wavelength display limit in nm (default: 350 nm).
        lam_max_nm: Maximum wavelength display limit in nm (default: 550 nm).
    """
    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 1.2], wspace=0.32)

    # -------------------------------------------------------------
    # Panel 1: Overlap Matrix Heatmap
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.imshow(
        overlap_matrix,
        cmap="Blues",
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        origin="upper",
    )
    plt.colorbar(
        im,
        ax=ax1,
        label="Normalized Overlap Integral $\\eta$",
        fraction=0.046,
        pad=0.04,
    )

    ref_labels = []
    for b in ref_bands:
        f = ref_freqs[b - 1]
        lam = (pitch / f * 1000.0) if f > 0 else 0.0
        ref_labels.append(f"Band {b}\n({lam:.1f} nm)")

    ax1.set_yticks(range(len(ref_bands)))
    ax1.set_yticklabels(ref_labels, fontsize=9)
    ax1.set_ylabel(
        "Air Reference Modes ($n_{\\mathrm{sub}}=1.0$)", fontsize=10, fontweight="bold"
    )

    step_x = max(1, len(target_bands) // 8)
    x_ticks = list(range(0, len(target_bands), step_x))
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([f"B{target_bands[i]}" for i in x_ticks], fontsize=9)
    ax1.set_xlabel(
        f"Substrate Modes ($n_{{\\mathrm{{sub}}}}={substrate_index:.3f}$)",
        fontsize=10,
        fontweight="bold",
    )
    ax1.set_title("(a) Modal Overlap Matrix", fontsize=11, fontweight="bold")

    # Annotate significant cells
    for i in range(len(ref_bands)):
        for j in range(len(target_bands)):
            val = overlap_matrix[i, j]
            if val >= 0.15:
                text_color = "white" if val > 0.6 else "black"
                ax1.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                )

    # -------------------------------------------------------------
    # Panel 2: Subspace Projection Bar Chart
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    colors = [
        "#1f77b4" if p < 0.25 else "#ff7f0e" if p < 0.60 else "#d62728"
        for p in subspace_projection
    ]
    ax2.bar(
        range(len(target_bands)),
        subspace_projection,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
    )
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels([f"{target_bands[i]}" for i in x_ticks], fontsize=9)
    ax2.set_xlabel(
        f"Target Band Index ($n_{{\\mathrm{{sub}}}}={substrate_index:.3f}$)",
        fontsize=10,
        fontweight="bold",
    )
    ax2.set_ylabel(
        "Degenerate Subspace Overlap $\\sum_i \\eta_{ij}$",
        fontsize=10,
        fontweight="bold",
    )
    ax2.set_title("(b) 440nm Manifold Projection", fontsize=11, fontweight="bold")
    ax2.set_ylim(0.0, max(1.1, float(np.max(subspace_projection)) * 1.15))
    ax2.grid(True, linestyle=":", alpha=0.5, axis="y")

    # Annotate top 3 target modes
    top_indices = np.argsort(subspace_projection)[-3:]
    for idx in top_indices:
        val = subspace_projection[idx]
        if val > 0.1:
            ax2.text(
                idx,
                val + 0.03,
                f"B{target_bands[idx]}\n{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
                color="#b30000",
            )

    # -------------------------------------------------------------
    # Panel 3: Mode Splitting & Redshift Diagram
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_xlim(-0.3, 1.3)
    ax3.set_xticks([0.0, 1.0])
    ax3.set_xticklabels(
        ["Air ($n=1.0$)", f"Substrate ($n={substrate_index:.3f}$)"],
        fontsize=10,
        fontweight="bold",
    )
    ax3.set_ylabel("Wavelength $\\lambda_0$ [nm]", fontsize=10, fontweight="bold")
    ax3.set_title("(c) Mode Splitting & Redshift", fontsize=11, fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.5)

    # Secondary y-axis in micrometers
    ax3_r = ax3.twinx()
    ax3_r.set_ylabel("Wavelength $\\lambda_0$ [$\\mu$m]", fontsize=10)

    # Plot all target modes as background gray markers
    for tb in target_bands:
        f = target_freqs[tb - 1]
        if f > 0:
            lam = pitch / f * 1000.0
            if lam_min_nm <= lam <= lam_max_nm:
                ax3.plot(1.0, lam, marker="_", color="#aaaaaa", markersize=10, mew=1.5)

    # Reference modes at x=0
    ref_lams = []
    for rb in ref_bands:
        f = ref_freqs[rb - 1]
        lam = (pitch / f * 1000.0) if f > 0 else 0.0
        ref_lams.append(lam)
        ax3.plot(0.0, lam, marker="o", color="#1f77b4", markersize=7, zorder=5)
        ax3.text(
            -0.06,
            lam,
            f"B{rb} ({lam:.1f}nm)",
            ha="right",
            va="center",
            fontsize=8,
            color="#1f77b4",
            fontweight="bold",
        )

    # Connect matched target modes
    matched_colors = ["#2ca02c", "#d62728", "#9467bd", "#e377c2"]
    for k, match in enumerate(best_matches):
        c = matched_colors[k % len(matched_colors)]
        lam_ref = match["ref_wavelength_nm"]
        lam_tar = match["target_wavelength_nm"]
        tb = match["target_band"]
        ov = match["max_overlap"]
        delta_lam = match["delta_wavelength_nm"]

        if lam_ref and lam_tar and (lam_min_nm <= lam_tar <= lam_max_nm):
            ax3.plot(1.0, lam_tar, marker="s", color=c, markersize=7, zorder=5)
            ax3.plot(
                [0.0, 1.0],
                [lam_ref, lam_tar],
                linestyle="--",
                color=c,
                linewidth=1.2 + 2.0 * ov,
                alpha=0.85,
                zorder=4,
            )
            ax3.text(
                1.06,
                lam_tar,
                f"B{tb} ({lam_tar:.1f}nm)\n$\\eta={ov * 100:.1f}\\%$\n$\\Delta\\lambda={delta_lam:+.1f}$nm",
                ha="left",
                va="center",
                fontsize=7.5,
                color=c,
                fontweight="bold",
            )

    ax3.set_ylim(lam_min_nm, lam_max_nm)
    ax3_r.set_ylim(lam_min_nm / 1000.0, lam_max_nm / 1000.0)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_mode_tracking_pipeline(
    pitch: float = 0.381,
    slab_thickness: float = 0.1,
    r1: float = 0.081,
    r2: float = 0.034,
    p2: float = 0.25,
    supercell_z: float = 4.0,
    slab_material: str = "hBN",
    cladding_material: str = "air",
    substrate_index: float = 1.444,
    target_wavelength_um: float = 0.440,
    ref_bands: list[int] | None = None,
    resolution: int = 25,
    resolution_z: int = 20,
    num_bands: int = 24,
    quick: bool = False,
    output_dir: Path | str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Runs full modal overlap and mode tracking analysis for the C6v 2b-6d unit cell.

    Args:
        pitch: Lattice pitch constant a in micrometers (default: 0.381 μm).
        slab_thickness: Membrane thickness in micrometers (default: 0.1 μm).
        r1: Wyckoff 2b inner hole radius in micrometers (default: 0.081 μm).
        r2: Wyckoff 6d satellite hole radius in micrometers (default: 0.034 μm).
        p2: Wyckoff 6d radial position parameter (default: 0.25).
        supercell_z: Supercell height in units of pitch a (default: 4.0).
        slab_material: Membrane material key (default: "hBN").
        cladding_material: Cladding material key (default: "air").
        substrate_index: Lower cladding substrate refractive index (default: 1.444).
        target_wavelength_um: Target wavelength to track in micrometers (default: 0.440 μm).
        ref_bands: Explicit list of reference band indices (None for auto-detection).
        resolution: In-plane mesh resolution (default: 25, or 10 in quick mode).
        resolution_z: Out-of-plane mesh resolution (default: 20, or 8 in quick mode).
        num_bands: Number of bands to compute (default: 24, or 6 in quick mode).
        quick: If True, uses coarse mesh for sub-second verification.
        output_dir: Optional explicit output directory.
        verbose: If True, streams solver trace to stdout.

    Returns:
        Structured dictionary containing geometry parameters, reference and substrate
        frequencies, overlap matrix, best matches, and artifact file paths.
    """
    if quick:
        resolution = 10
        resolution_z = 8
        num_bands = 8

    t0 = time.time()
    print("=" * 72)
    print(" C6v 2b-6d Unit Cell Modal Overlap & Mode Tracking Analysis")
    print(f" Mode: {'QUICK SMOKE TEST' if quick else 'FULL HIGH-RESOLUTION'}")
    print(f" Pitch: a = {pitch:.4f} μm | Thickness: h = {slab_thickness:.4f} μm")
    print(f" Radii: r1 = {r1:.4f} μm (2b), r2 = {r2:.4f} μm (6d, p2 = {p2:.2f})")
    print(f" Substrate Refractive Index: n_sub = {substrate_index:.4f}")
    print(f" Mesh Resolution: {resolution} x {resolution} x {resolution_z}")
    print("=" * 72)

    # 1. Resolve Output Directory
    resolved_output_dir = (
        Path(output_dir)
        if output_dir is not None
        else resolve_simulation_output_dir(
            solver="mpb",
            sim_type="mode_tracking",
            geometry="c6v_2b_6d",
        )
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    manager = SimulationOutputManager(resolved_output_dir)

    # 2. Layout Generation & GDS Export
    gds_path = resolved_output_dir / "unit_cell.gds"
    cell = phc_wyckoff_unit_cell(
        point_group="C6v",
        pitch=pitch,
        features=[("2b", r1), ("6d", r2, p2)],
    )
    export_gds(cell, gds_path, overwrite=True)
    print(f"  ✓ Exported unit cell layout: {gds_path.name}")

    # 3. Reference Simulation: Symmetric Air Cladding (n_sub = 1.0)
    print("\n--- 1. Solving Unperturbed Reference System (n_sub = 1.0, Air) ---")
    sim_ref = run_gamma_mode_solver(
        gds_path=gds_path,
        pitch=pitch,
        slab_thickness=slab_thickness,
        supercell_z=supercell_z,
        slab_material=slab_material,
        cladding_material=cladding_material,
        substrate_index=1.0,
        resolution=resolution,
        resolution_z=resolution_z,
        num_bands=num_bands,
        polarization="all",
        verbose=verbose,
    )
    ms_ref = sim_ref["ms"]
    freqs_ref = sim_ref["frequencies"]

    # Identify Reference Bands near Target Wavelength (440 nm)
    if ref_bands is None:
        target_ref_bands = find_modes_near_wavelength(
            frequencies=freqs_ref,
            pitch=pitch,
            target_wavelength_um=target_wavelength_um,
            count=3 if num_bands >= 8 else 2,
        )
    else:
        target_ref_bands = [b for b in ref_bands if 1 <= b <= num_bands]

    print(f"  Tracked Reference Bands: {target_ref_bands}")
    for b in target_ref_bands:
        f = freqs_ref[b - 1]
        lam = (pitch / f * 1000.0) if f > 0 else 0.0
        with silence_c_stdout():
            m = compute_modal_metrics(
                ms_ref, band_idx=b, slab_thickness=slab_thickness / pitch
            )
        te_pct = m["te"] * 100.0 if isinstance(m, dict) else 0.0
        print(f"    Band {b:2d}: ω~ = {f:.5f} | λ = {lam:.2f} nm | TE = {te_pct:.1f}%")

    # 4. Perturbed Simulation: Asymmetric Substrate (n_sub = substrate_index)
    print(f"\n--- 2. Solving Perturbed System (n_sub = {substrate_index:.3f}) ---")
    sim_sub = run_gamma_mode_solver(
        gds_path=gds_path,
        pitch=pitch,
        slab_thickness=slab_thickness,
        supercell_z=supercell_z,
        slab_material=slab_material,
        cladding_material=cladding_material,
        substrate_index=substrate_index,
        resolution=resolution,
        resolution_z=resolution_z,
        num_bands=num_bands,
        polarization="all",
        verbose=verbose,
    )
    ms_sub = sim_sub["ms"]
    freqs_sub = sim_sub["frequencies"]

    # 5. Overlap Integration & Mode Tracking
    print(
        "\n--- 3. Computing Modal Overlap Integrals & Degenerate Subspace Projection ---"
    )
    norm_thickness = slab_thickness / pitch
    with silence_c_stdout():
        tracking = track_modes_by_overlap(
            ms_ref=ms_ref,
            ms_target=ms_sub,
            ref_bands=target_ref_bands,
            target_bands=list(range(1, num_bands + 1)),
            field="electric_displacement",
            slab_thickness=norm_thickness,
            z_center=0.0,
            pitch=pitch,
        )

    overlap_mat = tracking["overlap_matrix"]
    subspace_proj = tracking["subspace_projection"]
    best_matches = tracking["best_matches"]

    print("\n  Tracking Results:")
    for match in best_matches:
        rb = match["ref_band"]
        tb = match["target_band"]
        ov = match["max_overlap"]
        lam_ref = match["ref_wavelength_nm"]
        lam_tar = match["target_wavelength_nm"]
        delta_lam = match["delta_wavelength_nm"]
        print(
            f"    Air Band {rb:2d} ({lam_ref:.1f} nm) -> Substrate Band {tb:2d} ({lam_tar:.1f} nm) | "
            f"Overlap: {ov * 100:.1f}% | Shift: {delta_lam:+.1f} nm"
        )

    top_target_bands = tracking["ranked_target_bands"][: len(target_ref_bands)]
    print(f"\n  Top Substrate Bands by 440nm Manifold Projection: {top_target_bands}")
    for tb in top_target_bands:
        p_val = float(subspace_proj[tb - 1])
        f_val = freqs_sub[tb - 1]
        lam_val = (pitch / f_val * 1000.0) if f_val > 0 else 0.0
        with silence_c_stdout():
            m_sub = compute_modal_metrics(
                ms_sub, band_idx=tb, slab_thickness=norm_thickness
            )
        te_pct = m_sub["te"] * 100.0 if isinstance(m_sub, dict) else 0.0
        print(
            f"    Substrate Band {tb:2d}: λ = {lam_val:.2f} nm | Manifold Overlap: {p_val * 100:.1f}% | TE: {te_pct:.1f}%"
        )

    # 6. Visualization Export
    eps_path = resolved_output_dir / "epsilon_map.png"
    with silence_c_stdout():
        eps_grid = get_epsilon_grid(ms_ref)
    plot_epsilon(
        epsilon=eps_grid,
        save_path=eps_path,
    )
    print(f"  ✓ Exported permittivity cross-section: {eps_path.name}")

    plot_path = resolved_output_dir / "mode_tracking.png"
    plot_mode_tracking_summary(
        overlap_matrix=overlap_mat,
        ref_bands=target_ref_bands,
        target_bands=list(range(1, num_bands + 1)),
        ref_freqs=freqs_ref,
        target_freqs=freqs_sub,
        subspace_projection=subspace_proj,
        best_matches=best_matches,
        pitch=pitch,
        substrate_index=substrate_index,
        output_path=plot_path,
        lam_min_nm=350.0,
        lam_max_nm=550.0,
    )
    print(f"  ✓ Exported mode tracking summary figure: {plot_path.name}")

    # 7. Structured JSON Summary
    elapsed_time = time.time() - t0
    results_summary = {
        "geometry": {
            "name": "c6v_2b_6d",
            "pitch_um": pitch,
            "slab_thickness_um": slab_thickness,
            "r1_um": r1,
            "r2_um": r2,
            "p2": p2,
            "supercell_z": supercell_z,
            "slab_material": slab_material,
            "cladding_material": cladding_material,
            "substrate_index": substrate_index,
        },
        "simulation": {
            "resolution": resolution,
            "resolution_z": resolution_z,
            "num_bands": num_bands,
            "quick": quick,
            "target_wavelength_um": target_wavelength_um,
            "elapsed_seconds": elapsed_time,
        },
        "results": {
            "ref_bands": target_ref_bands,
            "ref_frequencies": freqs_ref.tolist(),
            "target_frequencies": freqs_sub.tolist(),
            "overlap_matrix": overlap_mat.tolist(),
            "subspace_projection": subspace_proj.tolist(),
            "best_matches": best_matches,
            "ranked_target_bands": top_target_bands,
        },
    }

    json_path = manager.save_results_json(
        geometry_cfg=results_summary["geometry"],
        simulation_cfg=results_summary["simulation"],
        extra_data=results_summary["results"],
    )
    print(f"  ✓ Saved structured simulation results: {json_path.name}")
    print(f"\nExecution completed in {elapsed_time:.2f} seconds.")
    print("=" * 72)

    return {
        "output_dir": resolved_output_dir,
        "results_summary": results_summary,
        "overlap_matrix": overlap_mat,
        "best_matches": best_matches,
        "top_target_bands": top_target_bands,
    }


def main() -> None:
    """CLI entrypoint for modal overlap and mode tracking analysis."""
    parser = argparse.ArgumentParser(
        description="Modal Overlap Analysis and Mode Tracking for 3D C6v 2b-6d hBN Slab under Substrate Perturbation."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run fast low-resolution verification (< 2 seconds).",
    )
    parser.add_argument(
        "--pitch",
        type=float,
        default=0.381,
        help="Lattice pitch a in μm (default: 0.381).",
    )
    parser.add_argument(
        "--slab-thickness",
        type=float,
        default=0.1,
        help="Membrane thickness in μm (default: 0.1).",
    )
    parser.add_argument(
        "--r1",
        type=float,
        default=0.081,
        help="Wyckoff 2b hole radius in μm (default: 0.081).",
    )
    parser.add_argument(
        "--r2",
        type=float,
        default=0.034,
        help="Wyckoff 6d hole radius in μm (default: 0.034).",
    )
    parser.add_argument(
        "--p2",
        type=float,
        default=0.25,
        help="Wyckoff 6d radial position parameter (default: 0.25).",
    )
    parser.add_argument(
        "--substrate-index",
        type=float,
        default=1.444,
        help="Substrate refractive index (default: 1.444).",
    )
    parser.add_argument(
        "--target-wavelength",
        type=float,
        default=0.440,
        help="Target wavelength in μm (default: 0.440).",
    )
    parser.add_argument(
        "--ref-bands",
        type=int,
        nargs="+",
        default=None,
        help="Explicit reference band indices to track.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=25,
        help="In-plane mesh resolution (default: 25).",
    )
    parser.add_argument(
        "--resolution-z",
        type=int,
        default=20,
        help="Vertical mesh resolution (default: 20).",
    )
    parser.add_argument(
        "--num-bands",
        type=int,
        default=24,
        help="Number of bands to solve at Γ (default: 24).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Explicit output directory path."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Stream solver output to stdout."
    )

    args = parser.parse_args()

    run_mode_tracking_pipeline(
        pitch=args.pitch,
        slab_thickness=args.slab_thickness,
        r1=args.r1,
        r2=args.r2,
        p2=args.p2,
        substrate_index=args.substrate_index,
        target_wavelength_um=args.target_wavelength,
        ref_bands=args.ref_bands,
        resolution=args.resolution,
        resolution_z=args.resolution_z,
        num_bands=args.num_bands,
        quick=args.quick,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
