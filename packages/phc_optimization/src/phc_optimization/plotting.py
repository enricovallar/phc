"""Visualization routines for Bayesian optimization trajectories, convergence, and surrogate maps."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter, LogFormatterSciNotation, LogLocator

from phc_optimization.locus import extract_optimal_loci
from phc_optimization.surrogate import predict_surrogate_landscape
from phc_optimization.types import OptimizationRecord


def plot_bo_convergence(
    records: Sequence[OptimizationRecord],
    output_path: Path | str | None = None,
    title: str = "Bayesian Optimization Convergence",
) -> plt.Figure:
    """Plots optimization convergence: evaluated points and best-so-far FOM.

    Args:
        records: Sequence of evaluated OptimizationRecord objects.
        output_path: Optional path to save the generated figure.
        title: Plot title.

    Returns:
        matplotlib Figure instance.
    """
    eval_indices = [r.eval_index for r in records]
    foms = [r.fom for r in records]
    best_foms = np.maximum.accumulate(foms)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    # Panel 1: FOM vs Evaluation Index
    ax1.plot(
        eval_indices, foms, "o", color="#1f77b4", alpha=0.6, label="Evaluated Sample"
    )
    ax1.plot(
        eval_indices,
        best_foms,
        "-",
        color="#d62728",
        linewidth=2.0,
        label="Best-so-far FOM",
    )
    ax1.set_yscale("log")
    ax1.set_xlabel("Evaluation Index")
    ax1.set_ylabel("Figure of Merit (FOM)")
    ax1.set_title("FOM vs. Evaluations")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right")

    # Panel 2: Cost vs Evaluation Index
    costs = [r.cost for r in records]
    min_costs = np.minimum.accumulate(costs)
    ax2.plot(
        eval_indices, costs, "s", color="#2ca02c", alpha=0.6, label="Evaluated Cost"
    )
    ax2.plot(
        eval_indices, min_costs, "-", color="#ff7f0e", linewidth=2.0, label="Min Cost"
    )
    ax2.set_yscale("log")
    ax2.set_xlabel("Evaluation Index")
    ax2.set_ylabel("Normalized Cost (Split / Omega_mid)")
    ax2.set_title("Cost Minimization")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()

    if output_path is not None:
        p = Path(output_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")

    return fig


def plot_bo_surrogate_map(
    optimizer: Any,
    records: Sequence[OptimizationRecord],
    param_names: list[str],
    output_path: Path | str | None = None,
    grid_points: int = 150,
    clean_refit: bool = True,
    overlay_locus: bool = True,
    colorbar_limits: tuple[float, float] | None = None,
) -> plt.Figure | None:
    """Plots a high-fidelity 2-panel surrogate map for any 2-parameter optimization run.

    Uses `phc_optimization.surrogate` for penalty-filtered clean GP re-fitting and
    `phc_optimization.locus` for 1D degeneracy manifold extraction and overlay.

    Layout:
    - Plot 1 (Left / a): Raw optimization evaluated points colored by FOM with LogNorm.
    - Plot 2 (Right / b): GP surrogate predicted FOM landscape with log-spaced contours,
      overlaid with explored points, optimal point star, and 1D degeneracy locus curve.
    - Dedicated right strip (cax): Shared colorbar with inward log ticks and scientific notation.

    Args:
        optimizer: skopt.Optimizer instance with parameter space definition.
        records: Evaluated candidate records.
        param_names: List of parameter names (must have length 2).
        output_path: Optional path to save the generated figure.
        grid_points: Resolution per dimension for surrogate evaluation (default: 150).
        clean_refit: Whether to fit a clean penalty-filtered GP to eliminate boundary cliff artifacts.
        overlay_locus: Whether to extract and overlay the 1D degeneracy ridge curve in green.
        colorbar_limits: Optional manual (vmin, vmax) colorbar bounds.

    Returns:
        matplotlib Figure instance, or None if parameter count != 2.
    """
    if len(param_names) != 2:
        return None

    p1_name, p2_name = param_names[0], param_names[1]
    p1_vals = np.array([float(r.params[p1_name]) for r in records])
    p2_vals = np.array([float(r.params[p2_name]) for r in records])
    fom_vals = np.array([float(r.fom) for r in records])

    # Extract bounds from optimizer dimensions or record extrema
    if hasattr(optimizer, "space") and len(optimizer.space.dimensions) >= 2:
        dim1 = optimizer.space.dimensions[0]
        dim2 = optimizer.space.dimensions[1]
        b1 = (float(dim1.low), float(dim1.high))
        b2 = (float(dim2.low), float(dim2.high))
    else:
        b1 = (float(np.min(p1_vals)), float(np.max(p1_vals)))
        b2 = (float(np.min(p2_vals)), float(np.max(p2_vals)))

    active_model = (
        optimizer.models[-1]
        if hasattr(optimizer, "models") and optimizer.models
        else None
    )

    # 1. Predict surrogate landscape via dedicated surrogate module
    landscape = predict_surrogate_landscape(
        records=records,
        param_names=[p1_name, p2_name],
        bounds=(b1, b2),
        optimizer_model=active_model,
        grid_points=grid_points,
        clean_refit=clean_refit,
    )

    # 2. Determine shared LogNorm color scale bounds
    if colorbar_limits and len(colorbar_limits) == 2:
        vmin, vmax = float(colorbar_limits[0]), float(colorbar_limits[1])
    else:
        all_foms = np.r_[fom_vals, landscape.predicted_fom.ravel()]
        finite_pos = all_foms[np.isfinite(all_foms) & (all_foms > 0)]
        if finite_pos.size > 0:
            vmin = max(float(np.percentile(finite_pos, 1.0)), 1e-12)
            vmax = float(np.percentile(finite_pos, 99.0))
            if vmax <= vmin:
                vmax = vmin * 10
        else:
            vmin, vmax = 1.0, 1000.0

    log_norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    # 3. Identify optimal point
    valid_recs = [r for r in records if not r.metadata.get("penalty", False)]
    if valid_recs:
        best_r = max(valid_recs, key=lambda r: r.fom)
    else:
        best_r = min(records, key=lambda r: r.cost)
    best_x = float(best_r.params[p1_name])
    best_y = float(best_r.params[p2_name])

    # 4. Extract 1D degeneracy loci via dedicated locus module
    loci = []
    if overlay_locus:
        try:
            loci = extract_optimal_loci(
                grid_x1=landscape.x1_grid,
                grid_x2=landscape.x2_grid,
                fom_2d=landscape.predicted_fom,
                threshold_percentile=85.0,
            )
        except (ValueError, RuntimeError, TypeError):
            loci = []

    # 5. Build 3-panel GridSpec layout
    fig = plt.figure(figsize=(13.0, 5.8), dpi=150)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.04], wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    p1_label = f"${p1_name}/a$" if not p1_name.startswith("$") else p1_name
    p2_label = f"${p2_name}/a$" if not p2_name.startswith("$") else p2_name

    # Panel 1: Evaluated Points
    ax1.scatter(
        p1_vals,
        p2_vals,
        c=fom_vals,
        cmap="cool",
        norm=log_norm,
        s=35,
        edgecolors="black",
        linewidths=0.5,
        marker="o",
        zorder=4,
    )
    ax1.scatter(
        best_x,
        best_y,
        c="gold",
        edgecolors="black",
        marker="*",
        s=220,
        zorder=6,
        label="Optimal Point",
    )
    ax1.set_xlabel(p1_label, fontsize=11)
    ax1.set_ylabel(p2_label, fontsize=11)
    ax1.set_title(
        f"(a) Evaluated Samples ({p1_name}, {p2_name})", fontsize=12, fontweight="bold"
    )
    ax1.set_xlim(b1[0], b1[1])
    ax1.set_ylim(b2[0], b2[1])
    ax1.set_aspect("equal", adjustable="box")
    ax1.grid(alpha=0.4, linestyle="--")

    # Panel 2: GP Surrogate Expectation Map
    levels = np.logspace(np.log10(vmin), np.log10(vmax), 150)
    heatmap = ax2.contourf(
        landscape.X1,
        landscape.X2,
        landscape.predicted_fom,
        levels=levels,
        cmap="cool",
        norm=log_norm,
        extend="both",
    )

    # Overlay loci ridges in green
    if loci:
        for idx, locus in enumerate(loci):
            lbl_locus = (
                "Degeneracy Locus" if idx == 0 else f"Locus #{locus['locus_id']}"
            )
            ax2.plot(
                locus["x1"],
                locus["x2"],
                color="#00FF66",
                linestyle="--",
                linewidth=2.0,
                alpha=0.9,
                label=lbl_locus,
                zorder=7,
            )

    ax2.scatter(
        p1_vals,
        p2_vals,
        c="white",
        edgecolors="black",
        s=20,
        alpha=0.7,
        label="Explored points",
        zorder=5,
    )
    ax2.scatter(
        best_x,
        best_y,
        c="gold",
        edgecolors="black",
        marker="*",
        s=220,
        zorder=6,
        label="Optimal Point",
    )
    ax2.set_xlabel(p1_label, fontsize=11)
    ax2.set_ylabel(p2_label, fontsize=11)
    clean_tag = " (Clean GP)" if landscape.is_clean_fit else ""
    ax2.set_title(
        f"(b) Surrogate FOM Landscape{clean_tag}", fontsize=12, fontweight="bold"
    )
    ax2.set_xlim(b1[0], b1[1])
    ax2.set_ylim(b2[0], b2[1])
    ax2.set_aspect("equal", adjustable="box")
    ax2.grid(alpha=0.4, linestyle="--")

    # Shared axis tick formatting
    for ax in (ax1, ax2):
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    # Dedicated colorbar axis with inward ticks
    cbar = fig.colorbar(
        heatmap, cax=cax, extend="both", format=LogFormatterSciNotation()
    )
    cbar.set_label(r"$\mathbb{E}[C]^{-1}$ (Figure of Merit)", fontsize=12)
    cbar.locator = LogLocator(base=10.0, numticks=6)
    cbar.update_ticks()
    cbar.ax.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10), numticks=12)
    )
    cbar.ax.yaxis.set_minor_formatter(plt.NullFormatter())
    cbar.ax.tick_params(which="major", direction="in", length=5)
    cbar.ax.tick_params(which="minor", direction="in", length=2.5)

    # Unified bottom legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels1 + labels2, handles1 + handles2, strict=False))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=4,
        facecolor="white",
        edgecolor="black",
        framealpha=0.9,
    )

    if output_path is not None:
        p = Path(output_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")
        if loci:
            from phc_optimization.locus import export_loci_to_json

            export_loci_to_json(loci, p.parent / "optimal_loci.json")

    return fig
