from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_band_structure(
    results: dict[str, Any],
    node_labels: list[str] | None = None,
    node_indices: list[int] | None = None,
    title: str = "Photonic Band Structure",
    save_path: str | Path | None = None,
    *,
    k_labels: list[str] | None = None,
    k_indices: list[int] | None = None,
) -> plt.Figure:
    """Plots the photonic crystal band structure across the Brillouin zone k-path.

    Visualizes computed eigenfrequencies for TE and TM polarizations as a function of the
    unfolded 1D wavevector path. Highlights complete omnidirectional band gaps with gold
    shading and percentage labels. If light line data is present in results, plots the light
    line and shades the radiative light cone.

    Args:
        results: Dictionary returned by MPB solver runners (e.g. `run_band_solver`), containing:
            - 'freqs': Mapping of polarization keys ('te', 'tm', 'te_like', 'tm_like') to
              2D numpy arrays of shape `(num_k_points, num_bands)` containing normalized
              dimensionless frequencies `omega * a / (2 * pi * c) = a / lambda`.
            - 'light_line' (optional): 1D array of light line frequencies for 3D slab modes.
        node_labels: List of string labels for high-symmetry k-points along the path
            (e.g., `["Γ", "M", "K", "Γ"]`). Also accepts keyword argument `k_labels`.
        node_indices: List of integer indices corresponding to the positions of the
            high-symmetry points in the k-path (e.g., `[0, 16, 32, 48]`). Also accepts
            keyword argument `k_indices`.
        title: Plot title displayed at top of figure.
        save_path: Optional file path (PNG, PDF, SVG) where figure will be saved.
            Parent directories are created automatically if they do not exist.
        k_labels: Alias for `node_labels`.
        k_indices: Alias for `node_indices`.

    Returns:
        The matplotlib Figure object containing the rendered band diagram.

    Raises:
        ValueError: If results dictionary does not contain a valid 'freqs' map.
    """
    labels = k_labels if k_labels is not None else (node_labels or [])
    indices = k_indices if k_indices is not None else (node_indices or [])

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
    if indices and len(indices) == len(labels):
        ax.set_xticks(indices)
        ax.set_xticklabels(labels, fontsize=12)
        for idx in indices:
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
    extent: tuple[float, float, float, float] | None = None,
    xlabel: str = "Grid X",
    ylabel: str = "Grid Y",
) -> plt.Figure:
    """Plots the 2D dielectric permittivity distribution grid retrieved from MPB.

    If a 3D permittivity array is passed, extracts and plots the 2D cross-section at the
    mid-plane (z = N_z // 2). Transposes the 2D array prior to rendering with `imshow`
    because MPBData returns dimension 0 along Cartesian X and dimension 1 along Cartesian Y,
    whereas Matplotlib maps array dimension 0 to vertical rows (Y) and dimension 1 to horizontal
    columns (X).

    Args:
        epsilon: 2D or 3D numpy array containing dielectric permittivity values epsilon(r).
        title: Title displayed at top of the figure.
        cmap: Matplotlib colormap name (default: "viridis").
        colorbar: If True, adds a labeled colorbar showing permittivity scale.
        save_path: Optional file path (PNG, PDF, SVG) where figure will be saved.
            Parent directories are created automatically if they do not exist.
        extent: Optional 4-tuple (xmin, xmax, ymin, ymax) setting the physical plot coordinates.
        xlabel: Label for horizontal axis (default: "Grid X").
        ylabel: Label for vertical axis (default: "Grid Y").

    Returns:
        The matplotlib Figure object containing the rendered permittivity plot.

    Raises:
        ValueError: If the input epsilon array is not 2D or 3D.
    """
    if epsilon.ndim not in (2, 3):
        raise ValueError(
            f"Expected 2D or 3D epsilon array, got shape {epsilon.shape} (ndim={epsilon.ndim})."
        )
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

    # If 3D, take mid-plane slice
    if epsilon.ndim == 3:
        mid_z = epsilon.shape[2] // 2
        data = epsilon[:, :, mid_z]
    else:
        data = epsilon

    # Transpose data so that dim 0 (X) is along columns (horizontal)
    # and dim 1 (Y) is along rows (vertical).
    im = ax.imshow(
        data.T,
        origin="lower",
        extent=extent,
        cmap=cmap,
        interpolation="bilinear",
        aspect="equal",
    )

    if colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"Permittivity $\varepsilon$", fontsize=11)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)

    return fig
