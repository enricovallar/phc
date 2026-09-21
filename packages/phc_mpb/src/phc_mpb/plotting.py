from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_band_structure(
    results: dict[str, Any],
    node_labels: list[str],
    node_indices: list[int],
    title: str = "Photonic Band Structure",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plots the photonic band structure for computed polarizations."""
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)

    freqs_dict = results.get("freqs", {})

    color_map = {
        "te": "tab:blue",
        "tm": "tab:red",
        "te_like": "tab:blue",
        "tm_like": "tab:red",
    }

    x_max = 0
    for pol_key, freq_arr in freqs_dict.items():
        if len(freq_arr) == 0:
            continue
        num_k, num_bands = freq_arr.shape
        x = np.arange(num_k)
        x_max = max(x_max, num_k - 1)
        color = color_map.get(pol_key, "black")
        label_prefix = pol_key.upper()

        for band_idx in range(num_bands):
            lbl = label_prefix if band_idx == 0 else None
            ax.plot(x, freq_arr[:, band_idx], color=color, lw=1.5, label=lbl)

        # Highlight complete band gaps (if any)
        for band_idx in range(num_bands - 1):
            bot = float(np.max(freq_arr[:, band_idx]))
            top = float(np.min(freq_arr[:, band_idx + 1]))
            if top > bot:
                gap_pct = 200.0 * (top - bot) / (top + bot)
                ax.axhspan(
                    bot,
                    top,
                    color="gold",
                    alpha=0.25,
                    label=f"{label_prefix} Gap {band_idx + 1}-{band_idx + 2} ({gap_pct:.1f}%)",
                )

    # Plot light line if available (for 3D slabs)
    if "light_line" in results and len(results["light_line"]) > 0:
        ll = results["light_line"]
        x_ll = np.arange(len(ll))
        ax.plot(x_ll, ll, color="black", linestyle="--", lw=1.2, label="Light Line")
        ax.fill_between(
            x_ll, ll, np.max(ll) * 1.5, color="gray", alpha=0.15, label="Light Cone"
        )

    # High-symmetry labels & vertical grid lines
    if node_indices and len(node_indices) == len(node_labels):
        ax.set_xticks(node_indices)
        ax.set_xticklabels(node_labels, fontsize=12)
        for idx in node_indices:
            ax.axvline(x=idx, color="gray", linestyle=":", lw=0.8)

    ax.set_xlim(0, x_max if x_max > 0 else 1)
    ax.set_ylim(bottom=0)
    ax.set_ylabel(
        r"Normalized Frequency $\tilde{\omega} = \omega a / 2\pi c = a / \lambda$",
        fontsize=11,
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)

    return fig


def plot_epsilon(
    epsilon: np.ndarray,
    title: str = r"MPB Dielectric Permittivity $\varepsilon(\mathbf{r})$",
    cmap: str = "viridis",
    colorbar: bool = True,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plots the 2D dielectric function grid retrieved from MPB."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

    # If 3D, take mid-plane slice
    if epsilon.ndim == 3:
        mid_z = epsilon.shape[2] // 2
        data = epsilon[:, :, mid_z]
    else:
        data = epsilon

    im = ax.imshow(
        data,
        origin="lower",
        cmap=cmap,
        interpolation="bilinear",
        aspect="equal",
    )

    if colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"Permittivity $\varepsilon$", fontsize=11)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Grid X", fontsize=10)
    ax.set_ylabel("Grid Y", fontsize=10)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)

    return fig
