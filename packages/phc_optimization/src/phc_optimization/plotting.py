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


def plot_locus_profile(
    locus: dict[str, Any],
    param_names: Sequence[str] | None = None,
    param_bounds: dict[str, tuple[float, float]] | None = None,
    output_path: Path | str | None = None,
    title: str = "Degeneracy Locus Analysis",
) -> plt.Figure:
    """Plots a 3-panel figure analyzing the extracted optimal locus and group velocity.

    - Panel (a): Parameter Space trajectory (p1 vs p2) color-coded by group velocity v_g / c.
    - Panel (b): Group Velocity profile v_g / c vs normalized arc length t in [0, 1].
    - Panel (c): Figure of Merit profile FOM vs normalized arc length t (log scale).

    Args:
        locus: Locus dictionary containing coordinate arrays and physical metrics.
        param_names: Optional sequence of parameter names (e.g. ['r1', 'r2']).
        param_bounds: Optional parameter bounds dict for axis limits.
        output_path: Optional path to save the generated figure.
        title: Overall plot title.

    Returns:
        Matplotlib Figure object containing the 3-panel locus profile.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    p1_name = (
        param_names[0]
        if param_names and len(param_names) >= 1
        else locus.get("p1_name", "x1")
    )
    p2_name = (
        param_names[1]
        if param_names and len(param_names) >= 2
        else locus.get("p2_name", "x2")
    )

    p1_label = f"${p1_name}/a$" if not p1_name.startswith("$") else p1_name
    p2_label = f"${p2_name}/a$" if not p2_name.startswith("$") else p2_name

    x1_vals = np.asarray(locus.get("x1", []), dtype=float)
    x2_vals = np.asarray(locus.get("x2", []), dtype=float)
    n_pts = len(x1_vals)
    t_vals = np.linspace(0, 1, n_pts) if n_pts > 0 else np.array([])

    vg_vals = np.asarray(locus.get("group_velocity", locus.get("vg", [])), dtype=float)
    fom_vals = np.asarray(locus.get("fom", []), dtype=float)
    has_vg = len(vg_vals) == n_pts and n_pts > 0 and np.any(vg_vals > 0)
    has_fom = len(fom_vals) == n_pts and n_pts > 0

    fig = plt.figure(figsize=(15.5, 4.6), dpi=150)
    gs = gridspec.GridSpec(1, 3, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # -------------------------------------------------------------
    # Panel (a): Parameter Space Trajectory
    # -------------------------------------------------------------
    if n_pts > 0:
        if "x1_unrefined" in locus and "x2_unrefined" in locus:
            u_x1 = locus["x1_unrefined"]
            u_x2 = locus["x2_unrefined"]
            if len(u_x1) > 0:
                ax1.plot(
                    u_x1,
                    u_x2,
                    color="#B0B0B0",
                    linestyle=":",
                    linewidth=1.8,
                    zorder=2,
                    label="Surrogate Initial",
                )

        ax1.plot(
            x1_vals,
            x2_vals,
            color="#888888",
            linestyle="--",
            linewidth=1.5,
            zorder=3,
        )
        if has_vg:
            sc = ax1.scatter(
                x1_vals,
                x2_vals,
                c=vg_vals,
                cmap="plasma",
                s=45,
                edgecolors="black",
                linewidths=0.5,
                zorder=4,
            )
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("right", size="5%", pad=0.08)
            cbar = fig.colorbar(sc, cax=cax)
            cbar.set_label(r"$v_g / c$", fontsize=10, fontweight="bold")
            cbar.ax.tick_params(labelsize=8.5)
        else:
            ax1.scatter(
                x1_vals,
                x2_vals,
                color="#0070C0",
                s=35,
                edgecolors="black",
                linewidths=0.5,
                zorder=4,
            )

        # Mark Start (t=0) and End (t=1)
        ax1.scatter(
            x1_vals[0],
            x2_vals[0],
            c="#00FF00",
            edgecolors="black",
            marker="o",
            s=90,
            zorder=6,
            label=r"Start ($t=0$)",
        )
        ax1.scatter(
            x1_vals[-1],
            x2_vals[-1],
            c="#FF0000",
            edgecolors="black",
            marker="s",
            s=90,
            zorder=6,
            label=r"End ($t=1$)",
        )

    ax1.set_xlabel(p1_label, fontsize=11, fontweight="bold")
    ax1.set_ylabel(p2_label, fontsize=11, fontweight="bold")
    ax1.set_title("(a) Parameter Space Trajectory", fontsize=11.5, fontweight="bold")
    ax1.set_aspect("equal", adjustable="box")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    if n_pts > 0:
        ax1.legend(loc="best", frameon=True, framealpha=0.85, fontsize=8.5)

    if param_bounds:
        if p1_name in param_bounds:
            b1 = param_bounds[p1_name]
            ax1.set_xlim(float(b1[0]), float(b1[1]))
        if p2_name in param_bounds:
            b2 = param_bounds[p2_name]
            ax1.set_ylim(float(b2[0]), float(b2[1]))

    # -------------------------------------------------------------
    # Panel (b): Group Velocity Profile
    # -------------------------------------------------------------
    if has_vg:
        ax2.plot(
            t_vals,
            vg_vals,
            "-o",
            color="#0070C0",
            linewidth=2.0,
            markersize=4.5,
            label=r"$v_g(t) / c$",
        )
        ax2.set_ylabel(r"Group Velocity $v_g / c$", fontsize=11, fontweight="bold")
        ax2.set_xlabel(
            r"Normalized Trajectory $t \in [0, 1]$", fontsize=11, fontweight="bold"
        )
        ax2.set_title(r"(b) Group Velocity $v_g / c$", fontsize=11.5, fontweight="bold")
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="upper right", frameon=True, fontsize=9)
    else:
        ax2.text(
            0.5,
            0.5,
            "Group Velocity\nNot Evaluated",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=11,
            color="#666666",
        )
        ax2.set_title(r"(b) Group Velocity $v_g / c$", fontsize=11.5, fontweight="bold")
        ax2.set_xlabel(r"Normalized Trajectory $t \in [0, 1]$", fontsize=11)

    # -------------------------------------------------------------
    # Panel (c): Figure of Merit / Residual Gap Profile
    # -------------------------------------------------------------
    if has_fom:
        ax3.plot(
            t_vals,
            fom_vals,
            "-s",
            color="#00B050",
            linewidth=1.8,
            markersize=4.0,
            label=r"$\mathrm{FOM}(t)$",
        )
        ax3.set_yscale("log")
        ax3.set_ylabel(
            r"$\mathrm{FOM} = \mathbb{E}[C]^{-1}$", fontsize=11, fontweight="bold"
        )
        ax3.set_xlabel(
            r"Normalized Trajectory $t \in [0, 1]$", fontsize=11, fontweight="bold"
        )
        ax3.set_title(
            r"(c) $\mathrm{FOM} = \mathbb{E}[C]^{-1}$",
            fontsize=11.5,
            fontweight="bold",
        )
        ax3.grid(True, linestyle=":", alpha=0.6)
        ax3.legend(loc="upper right", frameon=True, fontsize=9)
    else:
        gaps = locus.get("residual_gap", locus.get("costs", []))
        if gaps:
            ax3.plot(
                t_vals[: len(gaps)],
                gaps,
                "-s",
                color="#E30613",
                linewidth=1.8,
                markersize=4.0,
                label=r"Residual Gap",
            )
            ax3.set_yscale("log")
            ax3.set_ylabel("Residual Gap", fontsize=11, fontweight="bold")
            ax3.set_xlabel(
                r"Normalized Trajectory $t \in [0, 1]$",
                fontsize=11,
                fontweight="bold",
            )
            ax3.set_title("Residual Gap Profile", fontsize=11.5, fontweight="bold")
            ax3.grid(True, linestyle=":", alpha=0.6)
            ax3.legend(loc="upper right", frameon=True, fontsize=9)
        else:
            ax3.text(
                0.5,
                0.5,
                "FOM Profile\nNot Evaluated",
                ha="center",
                va="center",
                transform=ax3.transAxes,
                fontsize=11,
                color="#666666",
            )
            ax3.set_title("Figure of Merit Profile", fontsize=11.5, fontweight="bold")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)

    if output_path is not None:
        p = Path(output_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")

    return fig


def plot_locus_dirac_frequency(
    locus: dict[str, Any],
    param_names: Sequence[str] | None = None,
    target_wavelength: float = 1.55,
    target_thickness: float | None = None,
    target_frequency: float | None = None,
    target_wavelength_nm: float | None = None,
    target_thickness_nm: float | None = None,
    slab_thickness: float = 0.5,
    pitch: float = 1.0,
    output_path: Path | str | None = None,
    title: str = "Dirac Frequency along Optimal Degeneracy Locus",
    show_inset: bool = True,
) -> plt.Figure:
    """Plots a 2-panel figure analyzing the Dirac frequency along the degeneracy locus.

    - Panel (a) on left: Dirac frequency omega_D vs arc length s(x1, x2) along the locus curve,
      including an optional parameter space (x1 vs x2) trajectory inset.
    - Panel (b) on right: Dirac frequency omega_D vs discrete sample point index.

    In both panels, highlights the design point closest to the specified target wavelength
    and target slab thickness, annotating the exact normalized Dirac frequency, required
    physical pitch, resulting physical thickness, and equivalent optical frequency (THz).

    Dimensional Scaling:
        Normalized frequency omega_D = a / lambda.
        For a target wavelength lambda_target and unit-cell thickness ratio eta = h_sim / a_sim,
        the required pitch is a = omega_D * lambda_target, and the physical thickness is
        h = eta * a = eta * omega_D * lambda_target.
        The design point closest to both targets minimizes |h - target_thickness|.

    Args:
        locus: Locus dictionary containing 'x1', 'x2', and 'dirac_frequency' (or 'omega_d').
        param_names: Optional sequence of parameter names (e.g. ['r1', 'r2']).
        target_wavelength: Desired operating wavelength lambda_target in micrometers (default: 1.55).
        target_thickness: Desired slab membrane thickness h_target in micrometers.
            If None, defaults to slab_thickness.
        slab_thickness: Slab thickness h_sim in the simulation (or ratio h/a, default: 0.5).
        pitch: Lattice pitch a_sim in the simulation (default: 1.0).
        output_path: Optional path to save the generated figure.
        title: Overall plot figure title.
        show_inset: Whether to display a parameter space trajectory inset in panel (a).

    Returns:
        Matplotlib Figure object containing the 2-panel Dirac frequency analysis.

    Raises:
        ValueError: If locus lacks valid coordinate or Dirac frequency arrays.
    """
    if target_wavelength_nm is not None:
        target_wavelength = float(target_wavelength_nm) / 1000.0
    if target_thickness_nm is not None:
        target_thickness = float(target_thickness_nm) / 1000.0

    p1_name = (
        param_names[0]
        if param_names and len(param_names) >= 1
        else locus.get("p1_name", "x1")
    )
    p2_name = (
        param_names[1]
        if param_names and len(param_names) >= 2
        else locus.get("p2_name", "x2")
    )
    p1_label = f"${p1_name}/a$" if not p1_name.startswith("$") else p1_name
    p2_label = f"${p2_name}/a$" if not p2_name.startswith("$") else p2_name

    x1_vals = np.asarray(locus.get("x1", []), dtype=float)
    x2_vals = np.asarray(locus.get("x2", []), dtype=float)
    n_pts = len(x1_vals)
    if n_pts == 0:
        raise ValueError("Locus contains no points ('x1' is empty).")

    df_raw = locus.get(
        "dirac_frequency",
        locus.get("omega_d", locus.get("freq_middle", [])),
    )
    if not df_raw or len(df_raw) == 0:
        raise ValueError(
            "Locus does not contain 'dirac_frequency' or 'omega_d'. "
            "Ensure Dirac frequencies are evaluated before plotting."
        )
    omega_d = np.asarray(df_raw, dtype=float)
    if len(omega_d) != n_pts:
        raise ValueError(
            f"Dimension mismatch: 'x1' has {n_pts} points, but 'dirac_frequency' has {len(omega_d)} values."
        )

    if n_pts > 1:
        dx1 = np.diff(x1_vals)
        dx2 = np.diff(x2_vals)
        arc_lengths = np.concatenate(([0.0], np.cumsum(np.hypot(dx1, dx2))))
    else:
        arc_lengths = np.array([0.0])

    sample_indices = np.arange(1, n_pts + 1)

    target_h = (
        float(target_thickness)
        if target_thickness is not None
        else float(slab_thickness)
    )
    eta = float(slab_thickness) / max(float(pitch), 1e-12)

    if target_frequency is not None:
        ideal_omega_d = float(target_frequency)
        errors = np.abs(omega_d - ideal_omega_d)
    else:
        # Ideal Dirac frequency if both targets are matched exactly:
        ideal_omega_d = target_h / max(eta * target_wavelength, 1e-12)
        h_vals = eta * omega_d * target_wavelength
        errors = np.abs(h_vals - target_h)

    opt_idx = int(np.argmin(errors))
    opt_omega_d = float(omega_d[opt_idx])
    opt_s = float(arc_lengths[opt_idx])
    opt_pt_idx = int(sample_indices[opt_idx])
    opt_a = float(opt_omega_d * target_wavelength)
    opt_h = float(eta * opt_a)
    opt_x1 = float(x1_vals[opt_idx])
    opt_x2 = float(x2_vals[opt_idx])

    # Physical frequency in THz: f = c / lambda
    c_light = 299792458.0
    f_thz = (c_light / (target_wavelength * 1e-6)) * 1e-12

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.2), dpi=150)
    fig.subplots_adjust(top=0.88, bottom=0.14, wspace=0.30)

    # -------------------------------------------------------------
    # Panel (a): omega_D vs s(x1, x2)
    # -------------------------------------------------------------
    ax1.plot(
        arc_lengths,
        omega_d,
        "-o",
        color="#005A9E",
        linewidth=2.0,
        markersize=5,
        label=r"$\tilde{\omega}_D(s)$",
        zorder=3,
    )

    min_w = float(np.min(omega_d))
    max_w = float(np.max(omega_d))
    span_w = max_w - min_w if max_w > min_w else 0.05
    if (min_w - 0.5 * span_w) <= ideal_omega_d <= (max_w + 0.5 * span_w):
        ax1.axhline(
            ideal_omega_d,
            color="#D9534F",
            linestyle="--",
            linewidth=1.5,
            alpha=0.85,
            label=rf"Target $\tilde{{\omega}}_D = {ideal_omega_d:.4f}$",
            zorder=2,
        )

    # Highlight optimal point
    ax1.scatter(
        opt_s,
        opt_omega_d,
        marker="*",
        s=260,
        facecolor="#FFD700",
        edgecolor="#8B0000",
        linewidth=1.5,
        zorder=6,
        label=f"Closest Match (Pt #{opt_pt_idx})",
    )

    callout_str = (
        f"Closest Match (Pt #{opt_pt_idx}):\n"
        f"  $\\tilde{{\\omega}}_D = {opt_omega_d:.5f}$\n"
        f"  $f = {f_thz:.2f}$ THz ($\\lambda = {target_wavelength:.3f}\\,\\mu$m)\n"
        f"  $a = {opt_a * 1e3:.1f}$ nm, $h = {opt_h * 1e3:.1f}$ nm\n"
        f"  ${p1_name} = {opt_x1:.4f}$, ${p2_name} = {opt_x2:.4f}$"
    )
    near_top = (opt_omega_d - min_w) / max(span_w, 1e-12) > 0.65
    y_off_a = -65 if near_top else 20
    x_off_a = 20 if opt_idx < n_pts // 2 else -140
    xy_text_offset_a = (x_off_a, y_off_a)
    ax1.annotate(
        callout_str,
        xy=(opt_s, opt_omega_d),
        xytext=xy_text_offset_a,
        textcoords="offset points",
        fontsize=8.5,
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "#FFFFE0",
            "edgecolor": "#B8860B",
            "alpha": 0.9,
        },
        arrowprops={
            "arrowstyle": "->",
            "connectionstyle": "arc3,rad=0.2" if not near_top else "arc3,rad=-0.2",
            "color": "#8B0000",
            "lw": 1.2,
        },
        zorder=7,
    )

    ax1.set_xlabel(
        rf"Arc Length $s({p1_name}, {p2_name})$ [$\mu$m]",
        fontsize=11,
        fontweight="bold",
    )
    ax1.set_ylabel(
        r"Dirac Frequency $\tilde{\omega}_D = \omega a / (2\pi c)$",
        fontsize=11,
        fontweight="bold",
    )
    ax1.set_title(
        rf"(a) $\tilde{{\omega}}_D$ vs $s({p1_name}, {p2_name})$",
        fontsize=11.5,
        fontweight="bold",
    )
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    ax1.margins(y=0.18)
    ax1.legend(
        loc="lower left" if near_top else "upper left",
        frameon=True,
        framealpha=0.85,
        fontsize=8.5,
    )

    # Inset on Left Panel
    if show_inset and n_pts >= 2:
        ax_ins = ax1.inset_axes([0.58, 0.12, 0.38, 0.35])
        ax_ins.plot(x1_vals, x2_vals, "--", color="#888888", linewidth=1.2)
        ax_ins.scatter(
            x1_vals,
            x2_vals,
            c=omega_d,
            cmap="viridis",
            s=20,
            zorder=3,
        )
        ax_ins.scatter(
            opt_x1,
            opt_x2,
            marker="*",
            s=130,
            facecolor="#FFD700",
            edgecolor="#8B0000",
            linewidth=1.2,
            zorder=5,
        )
        ax_ins.set_xlabel(p1_label, fontsize=7.5)
        ax_ins.set_ylabel(p2_label, fontsize=7.5)
        ax_ins.tick_params(labelsize=6.5)
        ax_ins.set_title(
            rf"Trajectory $({p1_name}, {p2_name})$",
            fontsize=7.5,
            fontweight="bold",
        )

    # -------------------------------------------------------------
    # Panel (b): omega_D vs sample point
    # -------------------------------------------------------------
    ax2.plot(
        sample_indices,
        omega_d,
        "-s",
        color="#2E7D32",
        linewidth=1.8,
        markersize=5,
        label=r"$\tilde{\omega}_D(i)$",
        zorder=3,
    )
    if (min_w - 0.5 * span_w) <= ideal_omega_d <= (max_w + 0.5 * span_w):
        ax2.axhline(
            ideal_omega_d,
            color="#D9534F",
            linestyle="--",
            linewidth=1.5,
            alpha=0.85,
            label=rf"Target $\tilde{{\omega}}_D = {ideal_omega_d:.4f}$",
            zorder=2,
        )

    # Highlight optimal point
    ax2.scatter(
        opt_pt_idx,
        opt_omega_d,
        marker="*",
        s=260,
        facecolor="#FFD700",
        edgecolor="#8B0000",
        linewidth=1.5,
        zorder=6,
        label=f"Closest Match (Pt #{opt_pt_idx})",
    )

    y_off_b = -40 if near_top else 20
    x_off_b = -80 if opt_idx > n_pts // 2 else 15
    xy_text_offset_b = (x_off_b, y_off_b)
    ax2.annotate(
        f"Pt #{opt_pt_idx}: $\\tilde{{\\omega}}_D = {opt_omega_d:.5f}$\n$f = {f_thz:.2f}$ THz",
        xy=(opt_pt_idx, opt_omega_d),
        xytext=xy_text_offset_b,
        textcoords="offset points",
        fontsize=8.5,
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "#E8F5E9",
            "edgecolor": "#2E7D32",
            "alpha": 0.9,
        },
        arrowprops={
            "arrowstyle": "->",
            "connectionstyle": "arc3,rad=-0.2",
            "color": "#2E7D32",
            "lw": 1.2,
        },
        zorder=7,
    )

    ax2.set_xlabel("Sample Point Index", fontsize=11, fontweight="bold")
    ax2.set_ylabel(
        r"Dirac Frequency $\tilde{\omega}_D = \omega a / (2\pi c)$",
        fontsize=11,
        fontweight="bold",
    )
    ax2.set_title(
        r"(b) $\tilde{\omega}_D$ vs Sample Point",
        fontsize=11.5,
        fontweight="bold",
    )
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    ax2.margins(y=0.18)
    if n_pts <= 20:
        ax2.set_xticks(sample_indices)
    ax2.legend(
        loc="lower left" if near_top else "upper left",
        frameon=True,
        framealpha=0.85,
        fontsize=8.5,
    )

    fig.suptitle(title, fontsize=12.5, fontweight="bold", y=0.98)

    best_match_info = {
        "index": opt_idx,
        "point_idx": opt_pt_idx,
        "arc_length": opt_s,
        "omega_d": opt_omega_d,
        "ideal_omega_d": ideal_omega_d,
        "target_wavelength": float(target_wavelength),
        "target_thickness": float(target_h),
        "pitch": opt_a,
        "thickness": opt_h,
        "frequency_thz": f_thz,
        p1_name: opt_x1,
        p2_name: opt_x2,
        "thickness_error": float(abs(opt_h - target_h)),
        "frequency_error": float(abs(opt_omega_d - ideal_omega_d)),
    }
    locus["optimal_match"] = best_match_info
    fig._optimal_match = best_match_info  # type: ignore[attr-defined]

    if output_path is not None:
        p = Path(output_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")

    return fig


def plot_band_structure_comparison(
    results_air: dict[str, Any],
    results_substrate: dict[str, Any],
    node_labels: Sequence[str] | None = None,
    node_indices: Sequence[int] | None = None,
    target_frequency: float | None = None,
    gamma_index: int | None = None,
    title: str = "Photonic Band Structure Comparison: Air vs SiO₂ Substrate Cladding",
    output_path: Path | str | None = None,
    air_title: str = "(a) Symmetric Membrane (Air Cladding, TE-like)",
    substrate_title: str = "(b) Asymmetric Slab (SiO₂ Substrate, All Modes)",
    share_ylim: bool = True,
    max_freq_crop: float | None = None,
) -> plt.Figure:
    """Plots a side-by-side band diagram comparison between symmetric air-clad membrane and asymmetric substrate-clad slab.

    - Panel (a): Air cladding (z-mirror symmetry, TE-like parity modes, air light line).
    - Panel (b): SiO2 substrate cladding (broken vertical symmetry, all modes, SiO2 light line).

    Visualizes computed eigenfrequencies as discrete dots without artificial continuity
    interpolation, displays omnidirectional band gaps, and plots the light line and light cone.
    If target_frequency is provided, draws a horizontal dashed reference line across both panels
    and marks the target Dirac frequency at the Gamma point.

    Args:
        results_air: Solver results dictionary from the air-clad membrane simulation.
        results_substrate: Solver results dictionary from the substrate-clad simulation.
        node_labels: Sequence of high-symmetry k-point labels (e.g. ['X', 'Γ', 'M', 'X']).
        node_indices: Sequence of k-point indices for the high-symmetry vertices.
        target_frequency: Optional normalized Dirac frequency omega_D to highlight.
        gamma_index: Optional index of the Gamma point along the k-path. If None, automatically
            detected from node_labels.
        title: Overall super-title for the comparison figure.
        output_path: Optional path to save the generated figure.
        air_title: Subplot title for panel (a).
        substrate_title: Subplot title for panel (b).
        share_ylim: Whether to force both subplots to have identical frequency y-limits.
        max_freq_crop: Optional maximum frequency ceiling for y-axis limits.

    Returns:
        Matplotlib Figure object containing the 2-panel band structure comparison.

    Raises:
        ValueError: If results dictionaries do not contain frequency arrays.
    """
    from phc_mpb.plotting import plot_band_structure

    labels_list = list(node_labels) if node_labels is not None else None
    indices_list = [int(i) for i in node_indices] if node_indices is not None else None

    # Detect Gamma index if not explicitly provided
    if gamma_index is None and labels_list is not None and indices_list is not None:
        for lbl, idx in zip(labels_list, indices_list, strict=False):
            if lbl in ("Γ", "Gamma", "G"):
                gamma_index = idx
                break

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.2, 5.5), dpi=150)
    fig.subplots_adjust(top=0.88, bottom=0.12, wspace=0.28)

    # Panel (a): Air-clad symmetric membrane
    plot_band_structure(
        results=results_air,
        node_labels=labels_list,
        node_indices=indices_list,
        title=air_title,
        ax=ax1,
        plot_gaps=True,
    )

    # Panel (b): Substrate-clad asymmetric slab
    plot_band_structure(
        results=results_substrate,
        node_labels=labels_list,
        node_indices=indices_list,
        title=substrate_title,
        ax=ax2,
        plot_gaps=False,
    )

    # Determine uniform y-limit if requested
    if share_ylim:
        freqs_air = results_air.get("freqs", {})
        max_f_air = 0.5
        for arr in freqs_air.values():
            if len(arr) > 0:
                max_f_air = max(max_f_air, float(np.nanmax(arr)))

        freqs_sub = results_substrate.get("freqs", {})
        max_f_sub = 0.5
        for arr in freqs_sub.values():
            if len(arr) > 0:
                max_f_sub = max(max_f_sub, float(np.nanmax(arr)))

        # Top limit based on air modes (to zoom in on the physical range of interest)
        if max_freq_crop is not None:
            y_top = float(max_freq_crop)
        else:
            y_top = 1.15 * max_f_air

        ax1.set_ylim(bottom=0.0, top=y_top)
        ax2.set_ylim(bottom=0.0, top=y_top)

    # Highlight target frequency if provided
    if target_frequency is not None:
        f_target = float(target_frequency)
        for ax in (ax1, ax2):
            ax.axhline(
                f_target,
                color="#D9534F",
                linestyle="--",
                linewidth=1.5,
                alpha=0.85,
                label=rf"Target $\tilde{{\omega}}_D = {f_target:.4f}$",
                zorder=4,
            )
            if gamma_index is not None:
                ax.scatter(
                    gamma_index,
                    f_target,
                    marker="*",
                    s=220,
                    facecolor="#FFD700",
                    edgecolor="#8B0000",
                    linewidth=1.4,
                    zorder=7,
                    label="Dirac Crossing",
                )
            # Re-generate legend with updated handles
            handles, leg_labels = ax.get_legend_handles_labels()
            # Deduplicate labels while preserving order
            by_label = dict(zip(leg_labels, handles, strict=False))
            ax.legend(
                by_label.values(),
                by_label.keys(),
                loc="upper right",
                framealpha=0.9,
                fontsize=8.5,
            )

    fig.suptitle(title, fontsize=12.5, fontweight="bold", y=0.98)

    if output_path is not None:
        p = Path(output_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")

    return fig
