"""Visualization routines for Bayesian optimization trajectories, convergence, and surrogate maps."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

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
    grid_points: int = 60,
) -> plt.Figure | None:
    """Plots a 2-panel surrogate map for 2-parameter optimization runs.

    - Panel 1: Evaluated sample coordinates colored by FOM.
    - Panel 2: GP surrogate model predicted FOM landscape.

    Args:
        optimizer: skopt.Optimizer instance with trained GP model.
        records: Evaluated candidate records.
        param_names: List of parameter names (must have length 2).
        output_path: Optional path to save the plot.
        grid_points: Number of resolution points per dimension for surrogate meshgrid.

    Returns:
        matplotlib Figure instance, or None if parameter count != 2.
    """
    if len(param_names) != 2:
        return None

    p1_name, p2_name = param_names[0], param_names[1]
    p1_vals = [r.params[p1_name] for r in records]
    p2_vals = [r.params[p2_name] for r in records]
    fom_vals = np.array([r.fom for r in records])

    # Extract bounds from optimizer dimensions
    dim1 = optimizer.space.dimensions[0]
    dim2 = optimizer.space.dimensions[1]
    x1_lin = np.linspace(dim1.low, dim1.high, grid_points)
    x2_lin = np.linspace(dim2.low, dim2.high, grid_points)
    X1, X2 = np.meshgrid(x1_lin, x2_lin)
    grid_coords = np.column_stack([X1.ravel(), X2.ravel()])

    # Predict with GP model
    gp = (
        optimizer.models[-1]
        if hasattr(optimizer, "models") and optimizer.models
        else None
    )
    if gp is None:
        return None

    mu, std = gp.predict(grid_coords, return_std=True)
    # Log-normal expectation for y = log10(C): E[C] = 10^(mu + (ln(10)/2) * std^2)
    exp_cost = 10 ** (mu + (np.log(10) / 2.0) * (std**2))
    predicted_fom = (1.0 / np.maximum(exp_cost, 1e-12)).reshape(X1.shape)

    vmin = max(1.0, float(np.percentile(fom_vals, 5)))
    vmax = max(vmin * 10, float(np.percentile(fom_vals, 95)))
    norm = LogNorm(vmin=vmin, vmax=vmax)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    # Panel 1: Samples
    sc = ax1.scatter(
        p1_vals,
        p2_vals,
        c=fom_vals,
        cmap="cool",
        norm=norm,
        s=45,
        edgecolors="black",
        linewidths=0.5,
    )
    ax1.set_xlabel(p1_name)
    ax1.set_ylabel(p2_name)
    ax1.set_title("(a) Evaluated Samples")
    ax1.grid(True, linestyle=":", alpha=0.5)
    plt.colorbar(sc, ax=ax1, label="FOM")

    # Panel 2: Surrogate landscape
    cf = ax2.contourf(X1, X2, predicted_fom, levels=50, cmap="cool", norm=norm)
    ax2.scatter(
        p1_vals,
        p2_vals,
        color="white",
        s=15,
        edgecolors="black",
        linewidths=0.4,
        alpha=0.7,
    )
    ax2.set_xlabel(p1_name)
    ax2.set_ylabel(p2_name)
    ax2.set_title("(b) GP Predicted FOM Landscape")
    ax2.grid(True, linestyle=":", alpha=0.5)
    plt.colorbar(cf, ax=ax2, label="Surrogate Predicted FOM")

    fig.tight_layout()
    if output_path is not None:
        p = Path(output_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")

    return fig
