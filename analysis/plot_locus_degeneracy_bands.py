#!/usr/bin/env python3
"""Plot Degeneracy Bands (λ_min, λ_max vs Slab Thickness) for Photonic Crystal Loci.

Analyzes the scale-invariant accidental Dirac degeneracies along an extracted 1D
degeneracy locus (by default Locus #1 from C6v optimization run 2026-10-01_10-13-16)
as a function of physical slab thickness h (in nm) under a fixed h/a aspect ratio.

Physical Scaling Principles:
    In electromagnetic eigensolvers (MPB), Maxwell's equations without material dispersion
    are scale-invariant. Dimensionless unit cell simulations fix the slab thickness ratio:
        η = h / a  (e.g. η = 0.25)
    For any physical slab thickness h (nm), the required lattice pitch is:
        a(h) = h / η = h / 0.25 = 4 * h
    For any point along the degeneracy locus with normalized Dirac frequency ω̃_D = a / λ_D,
    the corresponding physical operating wavelength is:
        λ_D(h) = a(h) / ω̃_D = h / (η * ω̃_D)

    Along the continuous 1D locus, the normalized Dirac frequency spans [ω̃_min, ω̃_max].
    Therefore, for any physical membrane thickness h, accidental Dirac degeneracies span
    a continuous "Degeneracy Band" bounded by:
        λ_min(h) = h / (η * ω̃_max)
        λ_max(h) = h / (η * ω̃_min)
    with spectral bandwidth Δλ(h) = λ_max(h) - λ_min(h) = h/η * (1/ω̃_min - 1/ω̃_max).

Command-Line Usage:
    # Standard analysis of Locus 1 from the default 2026-10-01_10-13-16 run:
    python analysis/plot_locus_degeneracy_bands.py

    # Analyze with custom slab thickness sweep (e.g. 50 to 160 nm):
    python analysis/plot_locus_degeneracy_bands.py --h-min 50 --h-max 160

    # Analyze a different locus or run directory:
    python analysis/plot_locus_degeneracy_bands.py --run-dir outputs/mpb/optimization/... --locus-id 2

    # Quick smoke test:
    python analysis/plot_locus_degeneracy_bands.py --quick

CLI Options:
    --run-dir PATH          Path to optimization run directory (default: 2026-10-01_10-13-16).
    --locus-id ID           Locus ID to analyze (default: 1).
    --h-min H_MIN           Minimum physical slab thickness in nm (default: 60.0 nm).
    --h-max H_MAX           Maximum physical slab thickness in nm (default: 140.0 nm).
    --h-points N            Number of thickness sweep sample points (default: 100, quick: 10).
    --target-wavelength LAM Target physical wavelength in nm (default: 436.0 nm).
    --target-thickness H    Target physical slab thickness in nm (default: 100.0 nm).
    --h-over-a ETA          Override fixed h/a aspect ratio if not in metadata (default: 0.25).
    --output-path PATH      Custom output figure path override.
    --quick                 Run rapid smoke test with minimal sweep points.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# Ensure repository root is in sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

DEFAULT_RUN_DIR = Path(
    "/zhome/2f/7/202918/phc/outputs/mpb/optimization/c6v_2b_6d_3d_slab_sio2/2026-10-01_10-13-16"
)


def load_locus_data(run_dir: Path | str, locus_id: int = 1) -> dict[str, Any]:
    """Loads locus coordinates, frequencies, and physical scaling metadata from run directory.

    Args:
        run_dir: Path to the optimization run output directory.
        locus_id: Integer identifier of the locus to extract (default: 1).

    Returns:
        Dictionary containing locus point arrays ('x1', 'x2', 'omega_d', 'vg'),
        locus metadata, target scaling values, and fixed ratio h/a.

    Raises:
        FileNotFoundError: If optimal_loci.json or required metadata files are missing.
        ValueError: If the requested locus_id is not found in the run data.
    """
    run_path = Path(run_dir).resolve()
    loci_file = run_path / "optimal_loci.json"
    if not loci_file.is_file():
        raise FileNotFoundError(
            f"optimal_loci.json not found in run directory: {run_path}"
        )

    with open(loci_file, encoding="utf-8") as f:
        all_loci = json.load(f)

    target_locus = None
    for loc in all_loci:
        if loc.get("locus_id") == locus_id:
            target_locus = loc
            break

    if target_locus is None:
        available_ids = [loc.get("locus_id") for loc in all_loci]
        raise ValueError(
            f"Locus ID {locus_id} not found in {loci_file}. Available locus IDs: {available_ids}"
        )

    # Resolve fixed h/a aspect ratio
    h_over_a = 0.25
    target_wavelength_nm = 436.0
    target_thickness_nm = 100.0

    ps_file = run_path / "physical_scaling_analysis.json"
    if ps_file.is_file():
        try:
            with open(ps_file, encoding="utf-8") as f:
                ps_data = json.load(f)
                h_over_a = float(ps_data.get("h_over_a", h_over_a))
                target_wavelength_nm = float(
                    ps_data.get("target_wavelength_nm", target_wavelength_nm)
                )
                target_thickness_nm = float(
                    ps_data.get("target_thickness_nm", target_thickness_nm)
                )
        except (KeyError, ValueError, json.JSONDecodeError, OSError):
            h_over_a = 0.25
    else:
        sim_file = run_path / "simulation_results.json"
        if sim_file.is_file():
            try:
                with open(sim_file, encoding="utf-8") as f:
                    sim_data = json.load(f)
                    fixed_params = sim_data.get("fixed_parameters", {})
                    if "slab_thickness" in fixed_params:
                        h_over_a = float(fixed_params["slab_thickness"])
            except (KeyError, ValueError, json.JSONDecodeError, OSError):
                h_over_a = 0.25

    omega_d = target_locus.get("dirac_frequency") or target_locus.get("omega_d")
    if not omega_d:
        raise ValueError(
            f"Locus {locus_id} in {loci_file} has no 'dirac_frequency' or 'omega_d' field."
        )

    return {
        "locus": target_locus,
        "locus_id": locus_id,
        "run_dir": run_path,
        "h_over_a": h_over_a,
        "target_wavelength_nm": target_wavelength_nm,
        "target_thickness_nm": target_thickness_nm,
        "omega_d": np.array(omega_d, dtype=float),
        "x1": np.array(target_locus.get("x1", []), dtype=float),
        "x2": np.array(target_locus.get("x2", []), dtype=float),
        "p1_name": target_locus.get("p1_name", "r1"),
        "p2_name": target_locus.get("p2_name", "r2"),
        "group_velocity": np.array(
            target_locus.get("group_velocity", target_locus.get("vg", [])),
            dtype=float,
        ),
        "target_match": target_locus.get("target_match")
        or target_locus.get("optimal_match"),
    }


def compute_degeneracy_bands(
    omega_d: np.ndarray,
    h_over_a: float,
    h_min_nm: float = 60.0,
    h_max_nm: float = 140.0,
    h_points: int = 100,
    target_wavelength_nm: float = 436.0,
    target_thickness_nm: float = 100.0,
) -> dict[str, Any]:
    """Calculates degeneracy wavelength bounds and physical dimensions across a slab thickness sweep.

    Args:
        omega_d: 1D array of normalized Dirac frequencies along the locus.
        h_over_a: Fixed dimensionless slab thickness aspect ratio η = h/a.
        h_min_nm: Minimum physical slab thickness in nm (default: 60.0 nm).
        h_max_nm: Maximum physical slab thickness in nm (default: 140.0 nm).
        h_points: Number of discrete sweep points for slab thickness (default: 100).
        target_wavelength_nm: Target operating wavelength in nm (default: 436.0 nm).
        target_thickness_nm: Target slab thickness in nm (default: 100.0 nm).

    Returns:
        Dictionary containing thickness sweep array, minimum/maximum wavelength curves,
        bandwidth array, physical pitch curve, and target intersection analyses.

    Raises:
        ValueError: If omega_d is empty, h_over_a <= 0, or h_min_nm >= h_max_nm.
    """
    if len(omega_d) == 0:
        raise ValueError("omega_d array must not be empty.")
    if h_over_a <= 0.0:
        raise ValueError(f"h_over_a must be strictly positive, got {h_over_a}.")
    if h_min_nm >= h_max_nm:
        raise ValueError(
            f"h_min_nm ({h_min_nm}) must be strictly less than h_max_nm ({h_max_nm})."
        )

    w_min = float(np.min(omega_d))
    w_max = float(np.max(omega_d))

    h_arr = np.linspace(h_min_nm, h_max_nm, h_points)
    pitch_arr = h_arr / h_over_a

    # Slopes: dλ/dh = 1 / (η * ω)
    slope_lam_min = 1.0 / (h_over_a * w_max)
    slope_lam_max = 1.0 / (h_over_a * w_min)

    lam_min_arr = h_arr * slope_lam_min
    lam_max_arr = h_arr * slope_lam_max
    delta_lam_arr = lam_max_arr - lam_min_arr

    # Evaluate all intermediate locus tracks: λ_i(h) = h / (η * ω_i)
    locus_tracks = np.outer(h_arr, 1.0 / (h_over_a * omega_d))

    # Intersections at target wavelength λ_target:
    # h = λ_target * η * ω
    h_at_lam_target_min = target_wavelength_nm * h_over_a * w_min
    h_at_lam_target_max = target_wavelength_nm * h_over_a * w_max

    # Intersections at target thickness h_target:
    lam_at_h_target_min = target_thickness_nm * slope_lam_min
    lam_at_h_target_max = target_thickness_nm * slope_lam_max

    return {
        "h_arr": h_arr,
        "pitch_arr": pitch_arr,
        "lam_min_arr": lam_min_arr,
        "lam_max_arr": lam_max_arr,
        "delta_lam_arr": delta_lam_arr,
        "locus_tracks": locus_tracks,
        "w_min": w_min,
        "w_max": w_max,
        "slope_lam_min": slope_lam_min,
        "slope_lam_max": slope_lam_max,
        "target_wavelength_nm": target_wavelength_nm,
        "target_thickness_nm": target_thickness_nm,
        "h_at_lam_target_min": h_at_lam_target_min,
        "h_at_lam_target_max": h_at_lam_target_max,
        "lam_at_h_target_min": lam_at_h_target_min,
        "lam_at_h_target_max": lam_at_h_target_max,
    }


def plot_degeneracy_bands_figure(
    data: dict[str, Any],
    bands: dict[str, Any],
    output_path: Path | str | None = None,
) -> plt.Figure:
    """Renders a comprehensive publication-quality 2-panel figure of degeneracy bands vs slab thickness.

    Panel (a) displays the physical operating wavelength bounds [λ_min, λ_max]
    versus slab thickness h (nm), highlighting the shaded accidental Dirac degeneracy band,
    individual locus sample trajectories, and intersections with design targets.
    Panel (b) displays the required lattice pitch a and physical hole radii R1, R2
    versus slab thickness h (nm) for the bounding geometries.

    Args:
        data: Loaded locus dictionary from `load_locus_data`.
        bands: Computed band sweep dictionary from `compute_degeneracy_bands`.
        output_path: Optional file path to save the generated figure.

    Returns:
        Matplotlib Figure containing the 2-panel analysis.
    """
    h_arr = bands["h_arr"]
    lam_min = bands["lam_min_arr"]
    lam_max = bands["lam_max_arr"]
    locus_tracks = bands["locus_tracks"]
    h_over_a = data["h_over_a"]
    p1_name = data["p1_name"]
    p2_name = data["p2_name"]
    x1 = data["x1"]
    x2 = data["x2"]
    target_match = data.get("target_match") or {}

    idx_min = int(np.argmin(data["omega_d"]))
    idx_max = int(np.argmax(data["omega_d"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.2), dpi=200)

    # -------------------------------------------------------------
    # Panel (a): Degeneracy Bands: λ_min, λ_max vs slab_thickness_nm
    # -------------------------------------------------------------
    # Shaded degeneracy continuum
    ax1.fill_between(
        h_arr,
        lam_min,
        lam_max,
        color="#2b7bba",
        alpha=0.25,
        label="Accidental Degeneracy Band",
        zorder=2,
    )

    # Individual locus sample tracks (streamlines)
    num_samples = locus_tracks.shape[1]
    for j in range(num_samples):
        # Color by normalized progress along locus
        frac = j / max(1, num_samples - 1)
        color = plt.cm.viridis(frac)
        ax1.plot(
            h_arr,
            locus_tracks[:, j],
            color=color,
            alpha=0.45,
            linewidth=0.8,
            linestyle="-",
            zorder=3,
        )

    # Bounding curves
    ax1.plot(
        h_arr,
        lam_min,
        color="#084081",
        linewidth=2.2,
        label=rf"$\lambda_{{\min}}(h)$ (Pt {idx_max + 1}: {p1_name}={x1[idx_max]:.3f}, {p2_name}={x2[idx_max]:.3f})",
        zorder=4,
    )
    ax1.plot(
        h_arr,
        lam_max,
        color="#e31a1c",
        linewidth=2.2,
        label=rf"$\lambda_{{\max}}(h)$ (Pt {idx_min + 1}: {p1_name}={x1[idx_min]:.3f}, {p2_name}={x2[idx_min]:.3f})",
        zorder=4,
    )

    # Target wavelength horizontal reference line
    target_lam = bands["target_wavelength_nm"]
    target_h = bands["target_thickness_nm"]
    ax1.axhline(
        target_lam,
        color="#d95f02",
        linestyle="--",
        linewidth=1.6,
        alpha=0.85,
        label=rf"Target $\lambda = {target_lam:.1f}\,$nm",
        zorder=5,
    )

    # Target thickness vertical reference line
    ax1.axvline(
        target_h,
        color="#7570b3",
        linestyle=":",
        linewidth=1.6,
        alpha=0.85,
        label=rf"Target $h = {target_h:.1f}\,$nm",
        zorder=5,
    )

    # Highlight intersections
    # 1. Target thickness intersection band [λ_min(h_target), λ_max(h_target)]
    lam_at_ht_min = bands["lam_at_h_target_min"]
    lam_at_ht_max = bands["lam_at_h_target_max"]
    ax1.plot(
        [target_h, target_h],
        [lam_at_ht_min, lam_at_ht_max],
        color="#7570b3",
        linewidth=3.5,
        marker="o",
        markersize=6,
        zorder=6,
    )
    ax1.annotate(
        f"At $h = {target_h:.0f}\\,$nm:\n[{lam_at_ht_min:.1f}, {lam_at_ht_max:.1f}] nm",
        xy=(target_h, 0.5 * (lam_at_ht_min + lam_at_ht_max)),
        xytext=(target_h + 3.0, 0.5 * (lam_at_ht_min + lam_at_ht_max) - 15),
        arrowprops={"arrowstyle": "->", "color": "#7570b3", "lw": 1.2},
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.25",
            "fc": "white",
            "ec": "#7570b3",
            "alpha": 0.9,
        },
        zorder=7,
    )

    # 2. Target wavelength intersection thickness range [h_min(λ_target), h_max(λ_target)]
    h_at_lt_min = bands["h_at_lam_target_min"]
    h_at_lt_max = bands["h_at_lam_target_max"]
    ax1.plot(
        [h_at_lt_min, h_at_lt_max],
        [target_lam, target_lam],
        color="#d95f02",
        linewidth=3.5,
        marker="s",
        markersize=6,
        zorder=6,
    )
    ax1.annotate(
        f"For $\\lambda = {target_lam:.0f}\\,$nm:\n$h \\in [{h_at_lt_min:.1f}, {h_at_lt_max:.1f}]\\,$nm",
        xy=(0.5 * (h_at_lt_min + h_at_lt_max), target_lam),
        xytext=(0.5 * (h_at_lt_min + h_at_lt_max) - 22, target_lam + 35),
        arrowprops={"arrowstyle": "->", "color": "#d95f02", "lw": 1.2},
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.25",
            "fc": "white",
            "ec": "#d95f02",
            "alpha": 0.9,
        },
        zorder=7,
    )

    # Highlight optimal match point if available
    if target_match and "thickness_nm" in target_match:
        h_match = float(target_match["thickness_nm"])
        lam_match = float(target_match.get("target_wavelength_nm", target_lam))
        p_match = float(target_match.get("pitch_nm", h_match / h_over_a))
        ax1.plot(
            h_match,
            lam_match,
            marker="*",
            color="#ffd700",
            markeredgecolor="#000000",
            markeredgewidth=1.2,
            markersize=14,
            label=rf"Design Match: $h={h_match:.1f}\,$nm, $a={p_match:.1f}\,$nm",
            zorder=8,
        )

    ax1.set_xlabel("Physical Slab Thickness $h$ (nm)", fontsize=11, fontweight="bold")
    ax1.set_ylabel(
        r"Operating Wavelength $\lambda$ (nm)", fontsize=11, fontweight="bold"
    )
    ax1.set_title(
        rf"(a) Degeneracy Bands $[\lambda_{{\min}}, \lambda_{{\max}}]$ vs Slab Thickness ($h/a = {h_over_a:.2f}$)",
        fontsize=12,
        fontweight="bold",
    )
    ax1.grid(True, linestyle="--", alpha=0.5, zorder=1)
    ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.95)

    # -------------------------------------------------------------
    # Panel (b): Physical Dimensions (Pitch a, Radii R1, R2) vs h
    # -------------------------------------------------------------
    pitch_arr = bands["pitch_arr"]
    r1_min_arr = x1[idx_min] * pitch_arr
    r1_max_arr = x1[idx_max] * pitch_arr
    r2_min_arr = x2[idx_min] * pitch_arr
    r2_max_arr = x2[idx_max] * pitch_arr

    ax2.plot(
        h_arr,
        pitch_arr,
        color="#2b7bba",
        linewidth=2.2,
        label=rf"Lattice Pitch $a(h) = h / {h_over_a:.2f} = {1.0 / h_over_a:.1f}\,h$",
        zorder=4,
    )

    # R1 shaded range and bounds
    ax2.fill_between(
        h_arr,
        r1_min_arr,
        r1_max_arr,
        color="#33a02c",
        alpha=0.25,
        label=f"Hole Radius $R_1(h)$ Range ({x1[idx_min]:.3f} to {x1[idx_max]:.3f} $a$)",
        zorder=2,
    )
    ax2.plot(
        h_arr,
        r1_min_arr,
        color="#33a02c",
        linestyle="--",
        linewidth=1.4,
        zorder=3,
    )
    ax2.plot(
        h_arr,
        r1_max_arr,
        color="#33a02c",
        linestyle="-",
        linewidth=1.8,
        zorder=3,
    )

    # R2 shaded range and bounds
    ax2.fill_between(
        h_arr,
        r2_max_arr,
        r2_min_arr,
        color="#ff7f00",
        alpha=0.25,
        label=f"Hole Radius $R_2(h)$ Range ({x2[idx_max]:.3f} to {x2[idx_min]:.3f} $a$)",
        zorder=2,
    )
    ax2.plot(
        h_arr,
        r2_max_arr,
        color="#ff7f00",
        linestyle="-",
        linewidth=1.8,
        zorder=3,
    )
    ax2.plot(
        h_arr,
        r2_min_arr,
        color="#ff7f00",
        linestyle="--",
        linewidth=1.4,
        zorder=3,
    )

    # Target thickness vertical line
    ax2.axvline(
        target_h,
        color="#7570b3",
        linestyle=":",
        linewidth=1.6,
        alpha=0.85,
        label=rf"Target $h = {target_h:.0f}\,$nm",
        zorder=5,
    )

    # Annotate dimensions at target thickness
    pitch_at_ht = target_h / h_over_a
    ax2.plot(
        [target_h],
        [pitch_at_ht],
        marker="o",
        color="#2b7bba",
        markersize=6,
        zorder=6,
    )
    ax2.annotate(
        f"At $h = {target_h:.0f}\\,$nm:\n$a = {pitch_at_ht:.1f}\\,$nm",
        xy=(target_h, pitch_at_ht),
        xytext=(target_h - 26, pitch_at_ht + 25),
        arrowprops={"arrowstyle": "->", "color": "#2b7bba", "lw": 1.2},
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.25",
            "fc": "white",
            "ec": "#2b7bba",
            "alpha": 0.9,
        },
        zorder=7,
    )

    ax2.set_xlabel("Physical Slab Thickness $h$ (nm)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Physical Dimension (nm)", fontsize=11, fontweight="bold")
    ax2.set_title(
        r"(b) Physical Pitch $a$ and Hole Radii $R_1, R_2$ vs Slab Thickness",
        fontsize=12,
        fontweight="bold",
    )
    ax2.grid(True, linestyle="--", alpha=0.5, zorder=1)
    ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.95)

    fig.suptitle(
        f"Scale-Invariant Degeneracy Analysis along Locus #{data['locus_id']} (Fixed $h/a = {h_over_a:.2f}$)\n"
        f"Slopes: $d\\lambda_{{\\min}}/dh = {bands['slope_lam_min']:.3f}$, "
        f"$d\\lambda_{{\\max}}/dh = {bands['slope_lam_max']:.3f}$ | "
        f"Dirac Frequencies: $\\tilde{{\\omega}}_D \\in [{bands['w_min']:.4f}, {bands['w_max']:.4f}]$",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()

    if output_path is not None:
        out_file = Path(output_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file, bbox_inches="tight")
        print(f"Saved degeneracy bands figure to: {out_file}")

    return fig


def run_degeneracy_bands_pipeline(
    run_dir: Path | str = DEFAULT_RUN_DIR,
    locus_id: int = 1,
    h_min_nm: float = 60.0,
    h_max_nm: float = 140.0,
    h_points: int = 100,
    target_wavelength_nm: float = 436.0,
    target_thickness_nm: float = 100.0,
    h_over_a: float | None = None,
    output_path: Path | str | None = None,
    quick: bool = False,
) -> dict[str, Any]:
    """Coordinates locus loading, thickness sweep calculation, figure rendering, and summary export.

    Args:
        run_dir: Directory containing optimization run outputs (default: 2026-10-01_10-13-16).
        locus_id: Locus ID to evaluate (default: 1).
        h_min_nm: Minimum physical slab thickness in nm (default: 60.0 nm).
        h_max_nm: Maximum physical slab thickness in nm (default: 140.0 nm).
        h_points: Number of discrete thickness evaluation points (default: 100, quick: 10).
        target_wavelength_nm: Operating target wavelength in nm (default: 436.0 nm).
        target_thickness_nm: Fabrication target slab thickness in nm (default: 100.0 nm).
        h_over_a: Optional override for fixed h/a aspect ratio.
        output_path: Custom destination path for the saved figure.
        quick: Fast smoke-test mode with reduced sweep density.

    Returns:
        Structured dictionary containing loaded data, sweep arrays, and output paths.
    """
    if quick:
        h_points = 10

    data = load_locus_data(run_dir=run_dir, locus_id=locus_id)
    eta = h_over_a if h_over_a is not None else data["h_over_a"]
    data["h_over_a"] = eta

    bands = compute_degeneracy_bands(
        omega_d=data["omega_d"],
        h_over_a=eta,
        h_min_nm=h_min_nm,
        h_max_nm=h_max_nm,
        h_points=h_points,
        target_wavelength_nm=target_wavelength_nm,
        target_thickness_nm=target_thickness_nm,
    )

    # Determine default output figure locations
    run_path = data["run_dir"]
    locus_dir = run_path / f"locus_{locus_id:02d}"
    if output_path is None:
        if locus_dir.is_dir():
            target_fig = locus_dir / "locus_degeneracy_bands_vs_thickness.png"
        else:
            target_fig = run_path / "locus_degeneracy_bands_vs_thickness.png"
    else:
        target_fig = Path(output_path).resolve()

    fig = plot_degeneracy_bands_figure(data=data, bands=bands, output_path=target_fig)
    plt.close(fig)

    # Also save a canonical copy in main run directory if saved into locus subdirectory
    if target_fig.parent == locus_dir:
        canonical_fig = run_path / "locus_degeneracy_bands_vs_thickness.png"
        import shutil

        shutil.copyfile(target_fig, canonical_fig)
        print(f"Copied figure to run root: {canonical_fig}")

    # Export structured summary JSON and CSV
    summary_json_path = target_fig.with_suffix(".json")
    summary_data = {
        "locus_id": locus_id,
        "h_over_a": eta,
        "omega_d_min": bands["w_min"],
        "omega_d_max": bands["w_max"],
        "slope_lam_min": bands["slope_lam_min"],
        "slope_lam_max": bands["slope_lam_max"],
        "target_wavelength_nm": target_wavelength_nm,
        "target_thickness_nm": target_thickness_nm,
        "thickness_range_for_target_wavelength_nm": [
            bands["h_at_lam_target_min"],
            bands["h_at_lam_target_max"],
        ],
        "wavelength_range_at_target_thickness_nm": [
            bands["lam_at_h_target_min"],
            bands["lam_at_h_target_max"],
        ],
        "num_locus_points": len(data["omega_d"]),
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    return {
        "data": data,
        "bands": bands,
        "figure_path": target_fig,
        "summary_json_path": summary_json_path,
    }


def main() -> None:
    """CLI entrypoint for degeneracy bands scaling analysis."""
    parser = argparse.ArgumentParser(
        description="Plot degeneracy bands (λ_min, λ_max vs slab_thickness_nm) for scale-invariant photonic crystal loci."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=str(DEFAULT_RUN_DIR),
        help=f"Optimization run directory path (default: {DEFAULT_RUN_DIR}).",
    )
    parser.add_argument(
        "--locus-id",
        type=int,
        default=1,
        help="Locus ID to analyze (default: 1).",
    )
    parser.add_argument(
        "--h-min",
        type=float,
        default=60.0,
        help="Minimum physical slab thickness in nm (default: 60.0 nm).",
    )
    parser.add_argument(
        "--h-max",
        type=float,
        default=140.0,
        help="Maximum physical slab thickness in nm (default: 140.0 nm).",
    )
    parser.add_argument(
        "--h-points",
        type=int,
        default=100,
        help="Number of thickness sweep points (default: 100).",
    )
    parser.add_argument(
        "--target-wavelength",
        type=float,
        default=436.0,
        help="Target physical operating wavelength in nm (default: 436.0 nm).",
    )
    parser.add_argument(
        "--target-thickness",
        type=float,
        default=100.0,
        help="Target physical slab thickness in nm (default: 100.0 nm).",
    )
    parser.add_argument(
        "--h-over-a",
        type=float,
        default=None,
        help="Override fixed h/a aspect ratio (default: auto-detected as 0.25).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Custom output figure path override.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run in rapid smoke-test mode with coarse sampling.",
    )

    args = parser.parse_args()

    print("=" * 72)
    print(" Photonic Crystal Degeneracy Bands Scaling Analysis")
    print(f" Run Directory: {args.run_dir}")
    print(f" Locus ID: #{args.locus_id}")
    print(f" Slab Thickness Sweep: [{args.h_min:.1f}, {args.h_max:.1f}] nm")
    print(
        f" Design Targets: λ = {args.target_wavelength:.1f} nm, h = {args.target_thickness:.1f} nm"
    )
    print("=" * 72)

    res = run_degeneracy_bands_pipeline(
        run_dir=args.run_dir,
        locus_id=args.locus_id,
        h_min_nm=args.h_min,
        h_max_nm=args.h_max,
        h_points=args.h_points,
        target_wavelength_nm=args.target_wavelength,
        target_thickness_nm=args.target_thickness,
        h_over_a=args.h_over_a,
        output_path=args.output_path,
        quick=args.quick,
    )

    bands = res["bands"]
    print("\nScaling Analysis Results:")
    print(
        f"  Dirac frequency range: ω̃_D ∈ [{bands['w_min']:.5f}, {bands['w_max']:.5f}]"
    )
    print(f"  Fixed aspect ratio:    h/a = {res['data']['h_over_a']:.4f}")
    print(f"  Band boundaries slope: dλ_min/dh = {bands['slope_lam_min']:.4f}")
    print(f"                         dλ_max/dh = {bands['slope_lam_max']:.4f}")
    print(
        f"  For target λ = {args.target_wavelength:.1f} nm: slab thickness h ∈ [{bands['h_at_lam_target_min']:.1f}, {bands['h_at_lam_target_max']:.1f}] nm"
    )
    print(
        f"  At target h = {args.target_thickness:.1f} nm:  wavelength λ ∈ [{bands['lam_at_h_target_min']:.1f}, {bands['lam_at_h_target_max']:.1f}] nm"
    )
    print(f"\nFigure saved: {res['figure_path']}")
    print(f"Summary JSON saved: {res['summary_json_path']}")


if __name__ == "__main__":
    main()
