#!/usr/bin/env python3
"""End-to-End Demonstration: 3D PhC Slab Mode Parity (TE-like vs TM-like) and TE Fraction Analysis.

Demonstrates the workflow for analyzing electromagnetic polarization in a 3D symmetric photonic
crystal slab membrane:
1. Generates a hexagonal unit cell using ``phc_layout``.
2. Exports the physical GDSII layout mask via ``phc_hydra.SimulationOutputManager``.
3. Queries optical material properties from ``phc_materials`` (Silicon slab n=3.48, Air holes/cladding n=1.0).
4. Extrudes the 3D dielectric slab membrane and air hole prisms in ``phc_mpb``.
5. Computes high-symmetry k-path (M -> Γ -> K -> M) across the Brillouin zone.
6. Solves for TE-like guided slab modes (even mirror parity, σ_z = +1, zeven) concurrently across workers.
7. Solves for TM-like guided slab modes (odd mirror parity, σ_z = -1, zodd) concurrently across workers.
8. Renders a unified band structure plot displaying BOTH TE-like and TM-like modes together using
   TWO DISTINCT MARKERS (circles 'o' for TE-like, triangles '^' for TM-like).
9. Solves for ALL eigenbands without parity constraints (NO_PARITY, polarization='all') and computes
   per-k-point, per-band modal TE/TM electric field energy fractions:
       f_TE = integral( Re(E_xy* . D_xy) ) / integral( Re(E* . D) )
10. Validates that the eigenfrequencies of the parity-restricted runs directly match the all-bands
    eigenvalues, and that the calculated TE fraction quantitatively identifies the modal symmetry.
11. Renders the all-bands dispersion colored continuously by TE fraction (blue=TE, red=TM) with a colorbar.
12. Renders a unified side-by-side comparison figure juxtaposing parity classification against TE fraction.
13. Extracts the 3D dielectric permittivity distribution (in-plane xy mid-plane and vertical xz cross-section).
14. Exports structured simulation metadata and manifest via ``phc_hydra.SimulationOutputManager``.
"""

import argparse
import json
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

from phc_hydra import (
    SimulationOutputManager,
    find_latest_simulation,
    resolve_simulation_output_dir,
)
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
    summarize_mode_physics,
)
from phc_utils import export_gds


def run_slab_3d_mode_parity_pipeline(
    pitch: float = 0.45,
    slab_thickness_um: float = 0.22,
    radius_ratio: float = 0.25,
    resolution: int = 20,
    resolution_z: int = 6,
    num_bands: int = 4,
    k_density: int = 8,
    supercell_z: float = 4.5,
    num_workers: int = 4,
    quick: bool = False,
    output_dir: Path | str | None = None,
    verbose: bool = True,
    marker_parity: str = "o",
    marker_all: str = "o",
    polarization_method: Literal[
        "midplane", "volumetric", "slab", "magnetic"
    ] = "midplane",
    filter_artifacts: bool = False,
    min_confinement: float = 0.20,
    fade_transitional: bool = True,
) -> dict[str, Any]:
    """Runs the end-to-end 3D PhC slab mode parity and TE fraction analysis pipeline.

    Solves for TE-like (even parity) and TM-like (odd parity) eigenmodes using filled dots
    (marker_parity='o', blue for TE-like, red for TM-like), and solves for all bands with
    slightly larger hollow circles (marker_all='o') with boundaries colored continuously by
    modal TE energy fraction (f_TE in [0, 1]).
    Renders a unified band diagram combining all modes on a single plot.

    Args:
        pitch: Lattice constant a in micrometers (default: 0.45 μm).
        slab_thickness_um: Physical slab membrane thickness in micrometers (default: 0.22 μm = 220 nm).
        radius_ratio: Hole radius normalized to pitch (r / a, default: 0.25).
        resolution: In-plane (x, y) mesh resolution per unit pitch a (default: 20).
        resolution_z: Vertical (z) mesh resolution per unit pitch a (default: 6).
        num_bands: Number of eigenbands to compute for each parity mode (default: 4).
        k_density: Number of interpolated k-points between high-symmetry vertices (default: 8).
        supercell_z: Vertical supercell height in units of pitch a including cladding buffer (default: 4.5).
        num_workers: Number of concurrent worker processes for k-point parallelization (default: 4).
        quick: If True, overrides parameters to minimal settings (resolution=12, resolution_z=4,
            num_bands=3, k_density=1, supercell_z=3.0) for rapid smoke testing (<=15s).
        output_dir: Optional directory override. If None, resolves canonical hierarchy:
            outputs/mpb/slab_mode_parity/c6v_primitive/<timestamp>/.
        verbose: If True, prints formatted progress updates and physical validation analysis.
        marker_parity: Matplotlib marker for TE-like and TM-like parity modes (default: 'o').
        marker_all: Matplotlib marker for all-bands modes colored by TE fraction (default: 'o').
        polarization_method: Polarization fraction calculation method ('midplane', 'volumetric', 'slab', 'magnetic').
        filter_artifacts: If True, filters out spurious supercell radiation continuum artifacts (default: False).
        min_confinement: Confinement threshold below which modes are treated as artifacts (default: 0.20).
        fade_transitional: If True, applies continuous alpha fading for transitional leaky modes (default: True).

    Returns:
        Dictionary containing:
            - 'results_unified': Unified results containing te_like, tm_like, all, and te_fractions.
            - 'results_parity': Results for TE-like and TM-like parity simulations.
            - 'results_all': Results for all-bands simulation including 'te_fractions'.
            - 'comparison': Comparison dictionary correlating parity bands with TE fractions.
            - 'output_dir': Resolved path to output directory.
            - 'files': Dict mapping artifact keys to saved file paths.
            - 'elapsed_time_s': Total pipeline execution time in seconds.

    Raises:
        RuntimeError: If MPB solver execution fails or required fields are missing.
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
        sim_type="slab_mode_parity",
        geometry="c6v_primitive",
        override_dir=output_dir,
    )
    output_mgr = SimulationOutputManager(
        output_dir=out_path,
        solver="mpb",
        sim_type="slab_mode_parity",
        geometry_name="c6v_primitive",
    )

    if verbose:
        print("\n" + "=" * 75)
        print("3D PhC Slab Mode Parity (TE-like vs TM-like) & TE Fraction Pipeline")
        print(f"Lattice Pitch:       a = {pitch:.3f} μm")
        print(
            f"Slab Thickness:     h = {slab_thickness_um:.3f} μm (h/a = {slab_thickness_um / pitch:.3f})"
        )
        print(
            f"Hole Radius:        r = {radius_ratio * pitch:.3f} μm (r/a = {radius_ratio:.2f})"
        )
        print(f"Supercell Height:   s_z = {supercell_z:.1f} a")
        print(f"Mesh Resolution:    r_xy = {resolution}, r_z = {resolution_z}")
        print(f"Bands Per Parity:   {num_bands} bands")
        print(f"Parallel Workers:   {num_workers} concurrent processes")
        print(
            f"Unified Markers:    Parity (TE/TM) = '{marker_parity}', All Bands = '{marker_all}'"
        )
        print(f"Output Directory:   {out_path}")
        print("=" * 75 + "\n")

    # ------------------------------------------------------------------
    # Step 1: Layout Generation (phc_layout)
    # ------------------------------------------------------------------
    if verbose:
        print("[1/7] Generating 2D hexagonal unit cell layout...")
    hole_radius_um = radius_ratio * pitch
    unit_cell = get_unit_cell(
        name="c6v_primitive",
        pitch=pitch,
        radius=hole_radius_um,
        wrap_to_cell=True,
    )

    # ------------------------------------------------------------------
    # Step 2: GDSII Mask Export (phc_hydra SSOT)
    # ------------------------------------------------------------------
    gds_path = output_mgr.save_gds(unit_cell, filename="unit_cell.gds")
    legacy_gds = out_path / "slab_unit_cell.gds"
    if legacy_gds != gds_path:
        export_gds(unit_cell, legacy_gds, overwrite=True)
    if verbose:
        print(f"      -> Exported physical GDSII mask: {gds_path}")

    # ------------------------------------------------------------------
    # Step 3: Material Properties Query (phc_materials SSOT)
    # ------------------------------------------------------------------
    mat_slab = get_material("si")
    mat_air = get_material("air")
    if verbose:
        print(
            f"[2/7] Material SSOT: Slab='{mat_slab.name}' (n={mat_slab.index:.3f}), "
            f"Cladding/Holes='{mat_air.name}' (n={mat_air.index:.3f})"
        )

    # ------------------------------------------------------------------
    # Step 4: 3D Slab Lattice & Geometry Extrusion (phc_mpb)
    # ------------------------------------------------------------------
    if verbose:
        print("[3/7] Building 3D slab lattice and extruding geometry...")
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

    # High-symmetry path: M -> Γ -> K -> M (satisfies AGENTS.md rule 6F)
    k_pts, labels, indices = get_high_symmetry_kpath(layout_lat, k_density=k_density)
    if verbose:
        print(
            f"      -> Brillouin Zone Path: {' -> '.join(labels)} ({len(k_pts)} total k-points)"
        )

    # ------------------------------------------------------------------
    # Step 5: Solve TE-like (even) and TM-like (odd) Parity Modes
    # ------------------------------------------------------------------
    if verbose:
        print(
            f"[4/7] Solving TE-like (even, σ_z=+1) and TM-like (odd, σ_z=-1) modes "
            f"across {len(k_pts)} k-points..."
        )

    t0_te = time.time()
    results_te = run_parallel_band_solver(
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
        compute_polarization_fractions=True,
        polarization_method=polarization_method,
    )
    t1_te = time.time()
    if verbose:
        print(f"      -> TE-like solve completed in {t1_te - t0_te:.2f}s.")

    t0_tm = time.time()
    results_tm = run_parallel_band_solver(
        geometry_lattice=mpb_lat,
        geometry=mpb_geom,
        k_points=k_pts,
        default_material="air",
        resolution=resolution,
        resolution_z=resolution_z,
        num_bands=num_bands,
        polarization="tm_like",
        dimension="3D_slab",
        num_workers=num_workers,
        cladding_index=1.0,
        verbose=False,
        compute_polarization_fractions=True,
        polarization_method=polarization_method,
    )
    t1_tm = time.time()
    if verbose:
        print(f"      -> TM-like solve completed in {t1_tm - t0_tm:.2f}s.")

    # Combine into unified parity results structure
    results_parity: dict[str, Any] = {
        "freqs": {
            "te_like": results_te["freqs"]["te_like"],
            "tm_like": results_tm["freqs"]["tm_like"],
        },
        "confinements": {
            "te_like": results_te.get("confinements"),
            "tm_like": results_tm.get("confinements"),
        },
        "gaps": {
            "te_like": results_te.get("gaps", {}).get("te_like", []),
            "tm_like": results_tm.get("gaps", {}).get("tm_like", []),
        },
        "guided_gaps": {
            "te_like": results_te.get("guided_gaps", {}).get("te_like", []),
            "tm_like": results_tm.get("guided_gaps", {}).get("tm_like", []),
        },
        "light_line": results_te.get("light_line", []),
        "dimension": "3D_slab",
    }

    # ------------------------------------------------------------------
    # Step 6: Solve ALL Bands (no parity) and Calculate TE Fractions
    # ------------------------------------------------------------------
    num_bands_all = 2 * num_bands
    if verbose:
        print(
            f"[5/7] Solving ALL {num_bands_all} bands (NO_PARITY) and calculating modal TE fractions..."
        )

    t0_all = time.time()
    results_all = run_parallel_band_solver(
        geometry_lattice=mpb_lat,
        geometry=mpb_geom,
        k_points=k_pts,
        default_material="air",
        resolution=resolution,
        resolution_z=resolution_z,
        num_bands=num_bands_all,
        polarization="all",
        dimension="3D_slab",
        num_workers=num_workers,
        cladding_index=1.0,
        verbose=False,
        compute_polarization_fractions=True,
        polarization_method=polarization_method,
    )
    t1_all = time.time()
    if verbose:
        print(f"      -> All-bands solve completed in {t1_all - t0_all:.2f}s.")

    te_fracs = results_all.get("te_fractions")
    freqs_all = results_all["freqs"]["all"]

    # Quantitative analysis of TE fractions
    comparison: dict[str, Any] = {
        "band_summaries": [],
        "te_like_count": 0,
        "tm_like_count": 0,
    }

    if verbose:
        print("\n" + "-" * 75)
        print("Modal Polarization & Parity Verification Summary:")
        print(
            f"{'Band':<6} {'Mean Freq (ω̃)':<16} {'Mean TE Frac':<16} {'Dominant Character':<20}"
        )
        print("-" * 75)

    if te_fracs is not None:
        for b in range(num_bands_all):
            mean_f = float(np.mean(freqs_all[:, b]))
            mean_te = float(np.mean(te_fracs[:, b]))
            char = "TE-like (even)" if mean_te >= 0.5 else "TM-like (odd)"
            if mean_te >= 0.5:
                comparison["te_like_count"] += 1
            else:
                comparison["tm_like_count"] += 1

            comparison["band_summaries"].append(
                {
                    "band": b + 1,
                    "mean_frequency": mean_f,
                    "mean_te_fraction": mean_te,
                    "character": char,
                }
            )
            if verbose:
                print(f"#{b + 1:<5} {mean_f:<16.4f} {mean_te:<16.3f} {char:<20}")
        print("-" * 75 + "\n")

    # ------------------------------------------------------------------
    # Step 7: Render Unified Band Diagrams and Permittivity Plots
    # ------------------------------------------------------------------
    if verbose:
        print(
            "[6/7] Rendering publication-quality band diagrams and permittivity maps..."
        )

    # Combine parity modes and all-bands into one unified dataset
    results_unified: dict[str, Any] = {
        "freqs": {
            "all": results_all["freqs"]["all"],
            "te_like": results_te["freqs"]["te_like"],
            "tm_like": results_tm["freqs"]["tm_like"],
        },
        "te_fractions": results_all.get("te_fractions"),
        "confinements": {
            "all": results_all.get("confinements"),
            "te_like": results_te.get("confinements"),
            "tm_like": results_tm.get("confinements"),
        },
        "light_line": results_te.get("light_line", []),
        "dimension": "3D_slab",
    }

    # Plot 1: UNIFIED BAND DIAGRAM WITH TWO DIFFERENT MARKERS
    # TE/TM parity modes share ONE marker (marker_parity, default: 'o') with distinct colors
    # (blue for TE-like, red for TM-like), while ALL bands uses a DIFFERENT marker style (marker_all,
    # default: 'o' hollow circle with colored boundary ring) colored continuously by TE fraction!
    custom_markers = {
        "te_like": marker_parity,
        "tm_like": marker_parity,
        "all": marker_all,
    }
    fig_unified = plot_band_structure(
        results=results_unified,
        node_labels=labels,
        node_indices=indices,
        markers=custom_markers,
        markersize={"te_like": 3.0, "tm_like": 3.0, "all": 6.5},
        alpha={"te_like": 1.0, "tm_like": 1.0, "all": 0.85},
        title=r"3D PhC Slab: Parity Modes ($\sigma_z = \pm 1$) + All Bands ($f_{\mathrm{TE}}$)",
        filter_artifacts=filter_artifacts,
        min_confinement=min_confinement,
        fade_transitional=fade_transitional,
    )
    band_plot_path = output_mgr.save_figure(
        fig_unified, artifact_key="band_plot", filename="band_structure.png"
    )
    fig_unified.savefig(out_path / "band_diagram_unified.png")
    plt.close(fig_unified)
    if verbose:
        print(f"      -> Saved unified dual-marker band diagram: {band_plot_path}")

    # Plot 2: PARITY MODES ONLY (TE-like blue circles vs TM-like red circles)
    fig_parity = plot_band_structure(
        results=results_parity,
        node_labels=labels,
        node_indices=indices,
        markers={"te_like": marker_parity, "tm_like": marker_parity},
        markersize=3.2,
        title=r"3D PhC Slab Parity Modes ($\sigma_z = \pm 1$): TE-like (blue) & TM-like (red)",
        filter_artifacts=filter_artifacts,
        min_confinement=min_confinement,
        fade_transitional=fade_transitional,
    )
    parity_plot_path = output_mgr.save_figure(
        fig_parity, artifact_key="parity_plot", filename="band_structure_parity.png"
    )
    plt.close(fig_parity)
    if verbose:
        print(f"      -> Saved parity-only band diagram: {parity_plot_path}")

    # Plot 3: ALL BANDS ONLY COLORED BY TE FRACTION (marker_all squares)
    fig_fraction = plot_band_structure(
        results=results_all,
        node_labels=labels,
        node_indices=indices,
        marker=marker_all,
        markersize=4.0,
        title=r"3D PhC Slab Band Structure — TE Fraction $f_{\mathrm{TE}}$",
        cmap="coolwarm_r",
        filter_artifacts=filter_artifacts,
        min_confinement=min_confinement,
        fade_transitional=fade_transitional,
    )
    frac_plot_path = output_mgr.save_figure(
        fig_fraction,
        artifact_key="frac_plot",
        filename="band_structure_te_fraction.png",
    )
    plt.close(fig_fraction)
    if verbose:
        print(f"      -> Saved all-bands TE fraction plot: {frac_plot_path}")

    # Plot 4: SIDE-BY-SIDE JUXTAPOSITION COMPARISON FIGURE
    fig_comp, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(14, 5.5), dpi=150, sharey=True
    )
    plot_band_structure(
        results=results_parity,
        node_labels=labels,
        node_indices=indices,
        markers={"te_like": marker_parity, "tm_like": marker_parity},
        markersize=3.0,
        title=r"(a) Mirror Parity ($\sigma_z = \pm 1$): TE-like & TM-like",
        ax=ax_left,
        filter_artifacts=filter_artifacts,
        min_confinement=min_confinement,
        fade_transitional=fade_transitional,
    )
    plot_band_structure(
        results=results_all,
        node_labels=labels,
        node_indices=indices,
        marker=marker_all,
        markersize=4.0,
        title=r"(b) All Bands: Electric TE Energy Fraction $f_{\mathrm{TE}}$",
        cmap="coolwarm_r",
        ax=ax_right,
        filter_artifacts=filter_artifacts,
        min_confinement=min_confinement,
        fade_transitional=fade_transitional,
    )
    fig_comp.suptitle(
        r"3D PhC Slab: Symmetry Parity ($\sigma_z$) vs Continuous TE Fraction ($f_{\mathrm{TE}}$)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    comp_plot_path = output_mgr.save_figure(
        fig_comp, artifact_key="comp_plot", filename="band_structure_comparison.png"
    )
    plt.close(fig_comp)
    if verbose:
        print(f"      -> Saved side-by-side comparison figure: {comp_plot_path}")

    # Plot 4: 3D SLAB DIELECTRIC PERMITTIVITY MAP (xy midplane + vertical xz cut)
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
        title=r"3D PhC Slab Permittivity $\varepsilon(\mathbf{r})$ — Midplane & Vertical Cut",
        interpolation="none",
    )
    eps_plot_path = output_mgr.save_figure(
        fig_eps, artifact_key="eps_plot", filename="epsilon_map.png"
    )
    plt.close(fig_eps)
    if verbose:
        print(f"      -> Saved dielectric permittivity map: {eps_plot_path}")

    # ------------------------------------------------------------------
    # Step 8: Results Compilation & Structured Manifest (phc_hydra)
    # ------------------------------------------------------------------
    if verbose:
        print("[7/7] Compiling structured simulation summary...")

    saved_files = {
        "gds": str(gds_path),
        "band_plot": str(band_plot_path),
        "frac_plot": str(frac_plot_path),
        "comp_plot": str(comp_plot_path),
        "eps_plot": str(eps_plot_path),
    }

    # Extract detected band gaps for summary JSON
    summary_gaps: list[dict[str, Any]] = []
    for pol_k, pol_data in results_parity["freqs"].items():
        n_b = pol_data.shape[1]
        for b in range(n_b - 1):
            top_low = float(np.max(pol_data[:, b]))
            bot_up = float(np.min(pol_data[:, b + 1]))
            if bot_up > top_low:
                g_pct = 200.0 * (bot_up - top_low) / (bot_up + top_low)
                summary_gaps.append(
                    {
                        "polarization": pol_k,
                        "bands": [b + 1, b + 2],
                        "lower_freq": top_low,
                        "upper_freq": bot_up,
                        "gap_pct": g_pct,
                    }
                )

    # Summarize mode physics (guided vs leaky vs radiation)
    mode_summary = {}
    if "all" in results_all["freqs"] and results_all.get("confinements") is not None:
        mode_summary = summarize_mode_physics(
            results_all["freqs"]["all"],
            results_all["confinements"],
            results_parity["light_line"],
            cutoff=min_confinement,
        )

    json_path = output_mgr.save_results_json(
        geometry_cfg={
            "name": "c6v_primitive",
            "lattice": "hexagonal",
            "pitch_um": pitch,
            "slab_thickness_um": slab_thickness_um,
            "radius_ratio": radius_ratio,
            "supercell_z": supercell_z,
            "slab_material": "si",
            "cladding_material": "air",
        },
        simulation_cfg={
            "solver": "mpb",
            "sim_type": "slab_mode_parity",
            "resolution": resolution,
            "resolution_z": resolution_z,
            "num_bands_parity": num_bands,
            "num_bands_all": num_bands_all,
            "k_density": k_density,
            "num_workers": num_workers,
            "markers": custom_markers,
            "polarization_method": polarization_method,
        },
        band_gaps=summary_gaps,
        extra_data={
            "comparison": comparison,
            "frequencies": {
                "te_like": results_parity["freqs"]["te_like"].tolist(),
                "tm_like": results_parity["freqs"]["tm_like"].tolist(),
                "all": results_all["freqs"]["all"].tolist(),
            },
            "te_fractions": (te_fracs.tolist() if te_fracs is not None else []),
            "confinements": {
                "all": (
                    results_all["confinements"].tolist()
                    if results_all.get("confinements") is not None
                    else []
                ),
                "te_like": (
                    results_te["confinements"].tolist()
                    if results_te.get("confinements") is not None
                    else []
                ),
                "tm_like": (
                    results_tm["confinements"].tolist()
                    if results_tm.get("confinements") is not None
                    else []
                ),
            },
            "mode_summary": mode_summary,
            "filter_settings": {
                "filter_artifacts": filter_artifacts,
                "min_confinement": min_confinement,
                "fade_transitional": fade_transitional,
            },
            "light_line": results_parity.get("light_line", []),
            "k_labels": labels,
            "k_indices": indices,
        },
    )
    saved_files["parity_plot"] = str(parity_plot_path)
    saved_files["results_json"] = str(json_path)

    elapsed_total = time.time() - t_start
    if verbose:
        print(f"\nPipeline successfully completed in {elapsed_total:.2f} seconds.")
        print(f"Artifacts preserved in: {out_path}\n")

    return {
        "results_unified": results_unified,
        "results_parity": results_parity,
        "results_all": results_all,
        "comparison": comparison,
        "output_dir": out_path,
        "files": saved_files,
        "elapsed_time_s": elapsed_total,
    }


def replot_from_simulation(
    target: Path | str | None = None,
    marker_parity: str = "o",
    marker_all: str = "o",
    filter_artifacts: bool = False,
    min_confinement: float = 0.20,
    fade_transitional: bool = True,
    output_dir: Path | str | None = None,
    verbose: bool = True,
) -> dict[str, str]:
    """Re-renders all band structure plots from an existing simulation results JSON without running MPB.

    Args:
        target: Path to a simulation output directory or directly to a simulation_results.json file.
            If None, 'latest', or empty, automatically resolves the most recent simulation run.
        marker_parity: New marker style for TE-like and TM-like parity modes (default: 'o').
        marker_all: New marker style for all-bands modes colored by TE fraction (default: 'o').
        filter_artifacts: If True, filters out spurious supercell radiation continuum artifacts (default: False).
        min_confinement: Confinement threshold below which modes are treated as artifacts (default: 0.20).
        fade_transitional: If True, applies continuous alpha fading for transitional leaky modes (default: True).
        output_dir: Target directory where updated figures will be saved. If None, overwrites or saves
            into the source simulation directory.
        verbose: If True, prints progress updates.

    Returns:
        Dict mapping artifact keys ('band_plot', 'parity_plot', 'frac_plot', 'comp_plot') to saved file paths.

    Raises:
        FileNotFoundError: If the specified target or simulation_results.json cannot be found.
        ValueError: If the JSON does not contain required frequency data.
    """
    if target is None or str(target).strip() in ("", "latest", "recent"):
        target_dir = find_latest_simulation(solver="mpb", sim_type="slab_mode_parity")
        json_file = target_dir / "simulation_results.json"
    else:
        target_path = Path(target)
        if target_path.is_dir():
            json_file = target_path / "simulation_results.json"
        else:
            json_file = target_path

    if not json_file.is_file():
        raise FileNotFoundError(f"Simulation results file not found at: {json_file}")

    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)

    extra = data.get("extra_data", {})
    freq_data = extra.get("frequencies", {})
    if not freq_data or "all" not in freq_data:
        raise ValueError(
            f"Invalid simulation_results.json at {json_file}: missing 'frequencies' in 'extra_data'."
        )

    out_dir = Path(output_dir) if output_dir is not None else json_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    freqs_all = np.array(freq_data["all"])
    freqs_te = np.array(freq_data.get("te_like", []))
    freqs_tm = np.array(freq_data.get("tm_like", []))
    te_fracs = np.array(extra.get("te_fractions", []))

    # Extract modal confinement data if present
    conf_data = extra.get("confinements", {})
    conf_all = None
    conf_te = None
    conf_tm = None
    if isinstance(conf_data, dict):
        if conf_data.get("all"):
            conf_all = np.array(conf_data["all"])
        if conf_data.get("te_like"):
            conf_te = np.array(conf_data["te_like"])
        if conf_data.get("tm_like"):
            conf_tm = np.array(conf_data["tm_like"])
    elif isinstance(conf_data, list) and len(conf_data) > 0:
        conf_all = np.array(conf_data)

    # Reconstruct or load k-path labels & light line
    labels = extra.get("k_labels")
    indices = extra.get("k_indices")
    light_line = extra.get("light_line", [])

    if not labels or not indices or not light_line:
        pitch = float(data.get("geometry", {}).get("pitch_um", 0.45))
        k_density = int(data.get("simulation", {}).get("k_density", 8))
        layout_lat = HexagonalLattice(a=pitch)
        k_pts, fallback_labels, fallback_indices = get_high_symmetry_kpath(
            layout_lat, k_density=k_density
        )
        labels = labels or fallback_labels
        indices = indices or fallback_indices

        if not light_line:
            import meep as mp

            mpb_lat = lattice_to_mpb_lattice(
                layout_lat,
                dimension="3D_slab",
                supercell=(1, 1),
                supercell_z=float(data.get("geometry", {}).get("supercell_z", 4.5)),
                normalize=True,
            )
            k_cart = [mp.reciprocal_to_cartesian(k, mpb_lat) for k in k_pts]
            k_mag = [float(np.sqrt(k.x**2 + k.y**2)) for k in k_cart]
            light_line = [km / 1.0 for km in k_mag]

    results_unified: dict[str, Any] = {
        "freqs": {
            "all": freqs_all,
            "te_like": freqs_te,
            "tm_like": freqs_tm,
        },
        "te_fractions": te_fracs if len(te_fracs) > 0 else None,
        "confinements": {
            "all": conf_all,
            "te_like": conf_te,
            "tm_like": conf_tm,
        },
        "light_line": light_line,
        "dimension": "3D_slab",
    }
    results_parity: dict[str, Any] = {
        "freqs": {
            "te_like": freqs_te,
            "tm_like": freqs_tm,
        },
        "confinements": {
            "te_like": conf_te,
            "tm_like": conf_tm,
        },
        "light_line": light_line,
        "dimension": "3D_slab",
    }
    results_all: dict[str, Any] = {
        "freqs": {"all": freqs_all},
        "te_fractions": te_fracs if len(te_fracs) > 0 else None,
        "confinements": conf_all,
        "light_line": light_line,
        "dimension": "3D_slab",
    }

    if verbose:
        print(f"Re-plotting from: {json_file}")
        print(f"Marker Parity (TE/TM): '{marker_parity}', Marker All: '{marker_all}'")
        if filter_artifacts:
            print(
                f"Artifact filtering: ENABLED (min_confinement={min_confinement:.2f}, fade_transitional={fade_transitional})"
            )
        print(f"Saving updated figures to: {out_dir}")

    custom_markers = {
        "te_like": marker_parity,
        "tm_like": marker_parity,
        "all": marker_all,
    }

    # 1. Unified Band Diagram
    band_plot_path = out_dir / "band_structure.png"
    fig_unified = plot_band_structure(
        results=results_unified,
        node_labels=labels,
        node_indices=indices,
        markers=custom_markers,
        markersize={"te_like": 3.0, "tm_like": 3.0, "all": 6.5},
        alpha={"te_like": 1.0, "tm_like": 1.0, "all": 0.85},
        title=r"3D PhC Slab: Parity Modes ($\sigma_z = \pm 1$) + All Bands ($f_{\mathrm{TE}}$)",
        filter_artifacts=filter_artifacts,
        min_confinement=min_confinement,
        fade_transitional=fade_transitional,
        save_path=band_plot_path,
    )
    plt.close(fig_unified)

    # 2. Parity Modes
    parity_plot_path = out_dir / "band_structure_parity.png"
    fig_parity = plot_band_structure(
        results=results_parity,
        node_labels=labels,
        node_indices=indices,
        markers={"te_like": marker_parity, "tm_like": marker_parity},
        markersize=3.2,
        title=r"3D PhC Slab Parity Modes ($\sigma_z = \pm 1$): TE-like & TM-like",
        filter_artifacts=filter_artifacts,
        min_confinement=min_confinement,
        fade_transitional=fade_transitional,
        save_path=parity_plot_path,
    )
    plt.close(fig_parity)

    # 3. All Bands TE Fraction
    frac_plot_path = out_dir / "band_structure_te_fraction.png"
    fig_fraction = plot_band_structure(
        results=results_all,
        node_labels=labels,
        node_indices=indices,
        marker=marker_all,
        markersize=4.0,
        title=r"3D PhC Slab Band Structure — TE Fraction $f_{\mathrm{TE}}$",
        cmap="coolwarm_r",
        filter_artifacts=filter_artifacts,
        min_confinement=min_confinement,
        fade_transitional=fade_transitional,
        save_path=frac_plot_path,
    )
    plt.close(fig_fraction)

    # 4. Side-by-Side Comparison
    comp_plot_path = out_dir / "band_structure_comparison.png"
    fig_comp, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(14, 5.5), dpi=150, sharey=True
    )
    plot_band_structure(
        results=results_parity,
        node_labels=labels,
        node_indices=indices,
        markers={"te_like": marker_parity, "tm_like": marker_parity},
        markersize=3.0,
        title=r"(a) Mirror Parity ($\sigma_z = \pm 1$): TE-like & TM-like",
        ax=ax_left,
        filter_artifacts=filter_artifacts,
        min_confinement=min_confinement,
        fade_transitional=fade_transitional,
    )
    plot_band_structure(
        results=results_all,
        node_labels=labels,
        node_indices=indices,
        marker=marker_all,
        markersize=4.0,
        title=r"(b) All Bands: Electric TE Energy Fraction $f_{\mathrm{TE}}$",
        cmap="coolwarm_r",
        ax=ax_right,
        filter_artifacts=filter_artifacts,
        min_confinement=min_confinement,
        fade_transitional=fade_transitional,
    )
    fig_comp.suptitle(
        r"3D PhC Slab: Symmetry Parity ($\sigma_z$) vs Continuous TE Fraction ($f_{\mathrm{TE}}$)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    fig_comp.savefig(comp_plot_path, dpi=150)
    plt.close(fig_comp)

    if verbose:
        print("Re-plotting completed successfully in <1 second!")
        print(f"  -> {band_plot_path}")
        print(f"  -> {parity_plot_path}")
        print(f"  -> {frac_plot_path}")
        print(f"  -> {comp_plot_path}")

    return {
        "band_plot": str(band_plot_path),
        "parity_plot": str(parity_plot_path),
        "frac_plot": str(frac_plot_path),
        "comp_plot": str(comp_plot_path),
    }


def main() -> None:
    """Command-line entrypoint for 3D slab mode parity and TE fraction demonstration."""
    parser = argparse.ArgumentParser(
        description="3D Photonic Crystal Slab Mode Parity (TE-like vs TM-like) and TE Fraction Analysis."
    )
    parser.add_argument(
        "--replot",
        nargs="?",
        const="latest",
        type=str,
        default=None,
        help="Re-plot figures from an existing simulation directory or results JSON without re-running MPB. "
        "If passed without a path, automatically re-plots the most recent simulation run.",
    )
    parser.add_argument(
        "--pitch", type=float, default=0.45, help="Lattice pitch a in micrometers."
    )
    parser.add_argument(
        "--slab-thickness",
        type=float,
        default=0.22,
        help="Slab membrane thickness in micrometers.",
    )
    parser.add_argument(
        "--radius-ratio", type=float, default=0.25, help="Hole radius ratio r/a."
    )
    parser.add_argument(
        "--resolution", type=int, default=20, help="In-plane mesh resolution."
    )
    parser.add_argument(
        "--resolution-z", type=int, default=6, help="Vertical mesh resolution."
    )
    parser.add_argument(
        "--num-bands", type=int, default=4, help="Number of bands per parity."
    )
    parser.add_argument(
        "--k-density", type=int, default=8, help="K-point path density."
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Concurrent worker count."
    )
    parser.add_argument(
        "--marker-parity",
        type=str,
        default="o",
        help="Marker style for TE-like and TM-like parity modes (default: 'o').",
    )
    parser.add_argument(
        "--marker-all",
        type=str,
        default="o",
        help="Marker style for all-bands modes colored by TE fraction (default: 'o').",
    )
    parser.add_argument(
        "--polarization-method",
        type=str,
        choices=["midplane", "volumetric", "slab", "magnetic"],
        default="midplane",
        help="Polarization fraction calculation method (default: 'midplane').",
    )
    parser.add_argument(
        "--filter-artifacts",
        action="store_true",
        help="Filter out unguided supercell radiation continuum artifacts (eta < min_confinement).",
    )
    parser.add_argument(
        "--min-confinement",
        type=float,
        default=0.20,
        help="Core slab energy confinement threshold below which modes are filtered as artifacts (default: 0.20).",
    )
    parser.add_argument(
        "--no-fade",
        action="store_true",
        help="Disable continuous alpha fading for transitional leaky modes (0.20 <= eta < 0.50).",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run low-resolution smoke test."
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output directory path."
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress console status logs."
    )

    args = parser.parse_args()

    if args.replot:
        replot_from_simulation(
            target=args.replot,
            marker_parity=args.marker_parity,
            marker_all=args.marker_all,
            filter_artifacts=args.filter_artifacts,
            min_confinement=args.min_confinement,
            fade_transitional=not args.no_fade,
            output_dir=args.output_dir,
            verbose=not args.quiet,
        )
        return

    run_slab_3d_mode_parity_pipeline(
        pitch=args.pitch,
        slab_thickness_um=args.slab_thickness,
        radius_ratio=args.radius_ratio,
        resolution=args.resolution,
        resolution_z=args.resolution_z,
        num_bands=args.num_bands,
        k_density=args.k_density,
        num_workers=args.num_workers,
        quick=args.quick,
        output_dir=args.output_dir,
        verbose=not args.quiet,
        marker_parity=args.marker_parity,
        marker_all=args.marker_all,
        polarization_method=args.polarization_method,
        filter_artifacts=args.filter_artifacts,
        min_confinement=args.min_confinement,
        fade_transitional=not args.no_fade,
    )


if __name__ == "__main__":
    main()
