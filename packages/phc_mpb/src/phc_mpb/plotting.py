from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_POLARIZATION_MARKERS: dict[str, str] = {
    "te": "o",
    "te_like": "o",
    "even": "o",
    "tm": "o",
    "tm_like": "o",
    "odd": "o",
    "all": "o",
    "no_parity": "o",
}

DEFAULT_POLARIZATION_MARKERSIZES: dict[str, float] = {
    "te": 3.0,
    "te_like": 3.0,
    "even": 3.0,
    "tm": 3.0,
    "tm_like": 3.0,
    "odd": 3.0,
    "all": 6.5,
    "no_parity": 6.5,
}

DEFAULT_POLARIZATION_COLORS: dict[str, str] = {
    "te": "tab:blue",
    "te_like": "tab:blue",
    "even": "tab:blue",
    "tm": "tab:red",
    "tm_like": "tab:red",
    "odd": "tab:red",
    "all": "black",
    "no_parity": "black",
}

DEFAULT_POLARIZATION_LABELS: dict[str, str] = {
    "te": "TE",
    "te_like": "TE-like (even)",
    "even": "Even (TE-like)",
    "tm": "TM",
    "tm_like": "TM-like (odd)",
    "odd": "Odd (TM-like)",
    "all": "All bands (TE fraction)",
    "no_parity": "All bands (TE fraction)",
}


def plot_band_structure(
    results: dict[str, Any],
    node_labels: list[str] | None = None,
    node_indices: list[int] | None = None,
    title: str = "Photonic Band Structure",
    save_path: str | Path | None = None,
    *,
    k_labels: list[str] | None = None,
    k_indices: list[int] | None = None,
    marker: str | dict[str, str] | None = None,
    markers: dict[str, str] | None = None,
    markersize: float | dict[str, float] | None = None,
    colors: dict[str, str] | None = None,
    te_fractions: np.ndarray | dict[str, np.ndarray] | None = None,
    cmap: str = "coolwarm_r",
    ax: plt.Axes | None = None,
    plot_gaps: bool = True,
    alpha: float | dict[str, float] = 1.0,
    hollow_fractions: bool = True,
    edge_linewidth: float = 1.6,
    filter_artifacts: bool = False,
    min_confinement: float = 0.20,
    fade_transitional: bool = True,
    confinements: np.ndarray | dict[str, np.ndarray] | None = None,
) -> plt.Figure:
    """Plots the photonic crystal band structure across the Brillouin zone k-path.

    Visualizes computed eigenfrequencies as discrete dots (one dot per k-point, with
    no continuous lines connecting them) as a function of the unfolded 1D wavevector path.
    Supports distinct marker styles per polarization (e.g., circles for TE/TE-like and
    triangles or squares for TM/TM-like) in unified multi-polarization plots.
    If polarization fractions are present in `results['te_fractions']` or provided via
    `te_fractions`, each discrete eigenfrequency dot is colored continuously according to its
    TE fraction (1.0 = in-plane dominant / TE, 0.0 = out-of-plane dominant / TM), and a
    polarization colorbar is rendered.

    Highlights complete omnidirectional band gaps with gold shading and percentage labels.
    If light line data is present in results, plots the light line and shades the radiative light cone.

    Args:
        results: Dictionary returned by MPB solver runners (e.g. `run_band_solver`), containing:
            - 'freqs': Mapping of polarization keys ('te', 'tm', 'te_like', 'tm_like', 'all') to
              2D numpy arrays of shape `(num_k_points, num_bands)` containing normalized
              dimensionless frequencies `omega * a / (2 * pi * c) = a / lambda`.
            - 'light_line' (optional): 1D array of light line frequencies for 3D slab modes.
            - 'te_fractions' (optional): 2D array of shape `(num_k_points, num_bands)` or dict
              containing modal TE energy fractions in [0.0, 1.0].
            - 'confinements' (optional): 2D array or dict containing modal slab core energy
              confinements eta_slab in [0.0, 1.0].
        node_labels: List of string labels for high-symmetry k-points along the path
            (e.g., `["M", "Γ", "K", "M"]`). Also accepts keyword argument `k_labels`.
        node_indices: List of integer indices corresponding to the positions of the
            high-symmetry points in the k-path (e.g., `[0, 16, 32, 48]`). Also accepts
            keyword argument `k_indices`.
        title: Plot title displayed at top of figure.
        save_path: Optional file path (PNG, PDF, SVG) where figure will be saved.
            Parent directories are created automatically if they do not exist.
        k_labels: Alias for `node_labels`.
        k_indices: Alias for `node_indices`.
        marker: Matplotlib marker style for eigenfrequency dots. Can be a single marker string
            (e.g., "o", "^", "s") applied to all polarizations, a dictionary mapping polarization
            keys to marker styles (e.g., `{"te_like": "o", "tm_like": "^"}`), or None to use
            canonical defaults (TE/TE-like='o', TM/TM-like='^').
        markers: Optional dictionary mapping polarization keys to marker styles, overriding `marker`.
        markersize: Size of discrete eigenfrequency marker dots in points (default: 3.0), or dict
            mapping polarization keys to sizes.
        colors: Optional dictionary mapping polarization keys to Matplotlib colors.
        te_fractions: Optional override array or dict of modal TE fractions of shape (num_k, num_bands).
            If None, attempts to extract from results['te_fractions'].
        cmap: Matplotlib colormap for TE fraction dots (default: "coolwarm_r", where 1.0=TE=blue
            and 0.0=TM=red).
        ax: Optional existing Matplotlib Axes to draw into. If None, a new Figure and Axes are created.
        plot_gaps: Whether to calculate and shade complete omnidirectional band gaps (default: True).
        alpha: Opacity for discrete marker dots (default: 1.0), or dict per polarization.
        hollow_fractions: If True, renders fraction-colored modes as empty/hollow circles with colored
            boundaries instead of filled markers, enabling clear concentric comparison with parity modes (default: True).
        edge_linewidth: Boundary line width in points for hollow fraction markers (default: 1.6).
        filter_artifacts: If True, filters out spurious supercell radiation continuum artifacts (eta < min_confinement)
            while preserving both guided modes and leaky resonances (default: False).
        min_confinement: Cutoff threshold for artifact filtering (default: 0.20).
        fade_transitional: If True, scales opacity continuously for transitional leaky modes (0.20 <= eta < 0.50) (default: True).
        confinements: Optional explicit confinement array or dictionary overriding results['confinements'].

    Returns:
        The matplotlib Figure object containing the rendered band diagram.

    Raises:
        ValueError: If results dictionary does not contain a valid 'freqs' map.
    """
    labels = k_labels if k_labels is not None else (node_labels or [])
    indices = k_indices if k_indices is not None else (node_indices or [])

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150)
        is_standalone = True
    else:
        fig = ax.figure
        is_standalone = False

    freqs_dict = results.get("freqs", {})
    if not isinstance(freqs_dict, dict) or not freqs_dict:
        raise ValueError(
            "Results dictionary must contain a non-empty 'freqs' dictionary."
        )

    fracs_arr = (
        te_fractions if te_fractions is not None else results.get("te_fractions")
    )
    conf_data = (
        confinements if confinements is not None else results.get("confinements")
    )

    def _resolve_marker(pol_key: str) -> str:
        if markers is not None and pol_key in markers:
            return markers[pol_key]
        if isinstance(marker, dict) and pol_key in marker:
            return marker[pol_key]
        if isinstance(marker, str):
            return marker
        return DEFAULT_POLARIZATION_MARKERS.get(pol_key, "o")

    def _resolve_markersize(pol_key: str) -> float:
        if isinstance(markersize, dict) and pol_key in markersize:
            return float(markersize[pol_key])
        if markersize is not None:
            return float(markersize)
        return DEFAULT_POLARIZATION_MARKERSIZES.get(pol_key, 3.0)

    def _resolve_color(pol_key: str) -> str:
        if colors is not None and pol_key in colors:
            return colors[pol_key]
        return DEFAULT_POLARIZATION_COLORS.get(pol_key, "black")

    def _resolve_alpha(pol_key: str) -> float:
        if isinstance(alpha, dict) and pol_key in alpha:
            return float(alpha[pol_key])
        return float(alpha) if isinstance(alpha, (int, float)) else 1.0

    last_scatter = None
    x_max = 0
    for pol_key, freq_arr in freqs_dict.items():
        if len(freq_arr) == 0:
            continue
        num_k, num_bands = freq_arr.shape
        x = np.arange(num_k)
        x_max = max(x_max, num_k - 1)
        color = _resolve_color(pol_key)
        pol_marker = _resolve_marker(pol_key)
        msize = _resolve_markersize(pol_key)
        pol_alpha = _resolve_alpha(pol_key)
        label_prefix = DEFAULT_POLARIZATION_LABELS.get(pol_key, pol_key.upper())

        # Determine if this polarization key has associated TE fraction data
        if isinstance(fracs_arr, dict):
            pol_fracs = fracs_arr.get(pol_key)
        elif pol_key in ("all", "no_parity") or len(freqs_dict) == 1:
            pol_fracs = fracs_arr
        else:
            pol_fracs = None

        has_valid_fracs = (
            pol_fracs is not None
            and isinstance(pol_fracs, np.ndarray)
            and pol_fracs.shape == freq_arr.shape
        )

        # Determine if this polarization key has associated confinement data
        if isinstance(conf_data, dict):
            pol_conf = conf_data.get(pol_key)
        elif pol_key in ("all", "no_parity") or len(freqs_dict) == 1:
            pol_conf = conf_data
        else:
            pol_conf = None

        has_valid_conf = (
            pol_conf is not None
            and isinstance(pol_conf, np.ndarray)
            and pol_conf.shape == freq_arr.shape
        )

        # Filter artifacts (eta < min_confinement) while preserving guided and leaky modes
        if filter_artifacts and has_valid_conf:
            from phc_mpb.filtering import (
                compute_confinement_alphas,
                filter_band_data,
            )

            freq_to_plot, fracs_to_plot = filter_band_data(
                freq_arr,
                pol_conf,
                cutoff=min_confinement,
                te_fractions=pol_fracs,
            )
            if fade_transitional:
                conf_alphas = compute_confinement_alphas(
                    pol_conf,
                    cutoff=min_confinement,
                    full_threshold=0.50,
                    base_alpha=pol_alpha,
                )
            else:
                conf_alphas = None
        else:
            freq_to_plot = freq_arr
            fracs_to_plot = pol_fracs
            conf_alphas = None

        for band_idx in range(num_bands):
            y_band = freq_to_plot[:, band_idx]
            valid_mask = ~np.isnan(y_band)
            if not np.any(valid_mask):
                continue

            x_pts = x[valid_mask]
            y_pts = y_band[valid_mask]

            if has_valid_fracs and fracs_to_plot is not None:
                # Color discrete dots by TE fraction
                frac_vals = fracs_to_plot[valid_mask, band_idx]
                sc = ax.scatter(
                    x_pts,
                    y_pts,
                    c=frac_vals,
                    cmap=cmap,
                    vmin=0.0,
                    vmax=1.0,
                    s=msize**2 * 4,
                    marker=pol_marker,
                    zorder=2,
                    alpha=pol_alpha,
                    label=label_prefix if band_idx == 0 else None,
                )
                if hollow_fractions:
                    cmap_obj = plt.get_cmap(cmap)
                    rgba = cmap_obj(frac_vals)
                    if conf_alphas is not None:
                        rgba[:, 3] = conf_alphas[valid_mask, band_idx]
                    elif pol_alpha is not None and pol_alpha < 1.0:
                        rgba[:, 3] = pol_alpha
                    sc.set_facecolor("none")
                    sc.set_edgecolor(rgba)
                    sc.set_linewidth(edge_linewidth)
                else:
                    sc.set_edgecolor("none")
                last_scatter = sc
            else:
                lbl = label_prefix if band_idx == 0 else None
                ax.plot(
                    x_pts,
                    y_pts,
                    marker=pol_marker,
                    linestyle="none",
                    markersize=msize,
                    color=color,
                    alpha=pol_alpha,
                    zorder=3,
                    label=lbl,
                )

        # Highlight complete band gaps (if any)
        if plot_gaps:
            for band_idx in range(num_bands - 1):
                bot = float(np.max(freq_arr[:, band_idx]))
                top = float(np.min(freq_arr[:, band_idx + 1]))
                if top > bot:
                    gap_pct = 200.0 * (top - bot) / (top + bot)
                    gap_label = (
                        f"{label_prefix} Gap {band_idx + 1}-{band_idx + 2} ({gap_pct:.1f}%)"
                        if len(freqs_dict) > 1
                        else f"Gap {band_idx + 1}-{band_idx + 2} ({gap_pct:.1f}%)"
                    )
                    ax.axhspan(
                        bot,
                        top,
                        color="gold",
                        alpha=0.25,
                        label=gap_label,
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

    # Add colorbar if dots are colored by polarization fraction
    if last_scatter is not None:
        cbar = fig.colorbar(last_scatter, ax=ax, pad=0.02)
        cbar.set_label(r"Polarization: TE Fraction $f_{\mathrm{TE}}$", fontsize=10)
        cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(["0.0 (TM)", "0.25", "0.50", "0.75", "1.0 (TE)"])

    ax.set_xlim(0, x_max if x_max > 0 else 1)
    ax.set_ylim(bottom=0)
    ax.set_ylabel(
        r"Normalized Frequency $\tilde{\omega} = \omega a / 2\pi c = a / \lambda$",
        fontsize=11,
    )
    ax.set_title(title, fontsize=13, fontweight="bold")

    handles, _legend_labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.3)

    if is_standalone:
        plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)

    return fig


def replot_band_structure_from_results(
    results_path: Path | str | None = None,
    solver: str = "mpb",
    sim_type: str | None = None,
    geometry: str | None = None,
    base_dir: Path | str = "outputs",
    save_path: Path | str | None = None,
    *,
    marker: str | dict[str, str] | None = None,
    markers: dict[str, str] | None = None,
    markersize: float | dict[str, float] | None = None,
    colors: dict[str, str] | None = None,
    title: str | None = None,
    cmap: str = "coolwarm_r",
    ax: plt.Axes | None = None,
    hollow_fractions: bool = True,
    edge_linewidth: float = 1.6,
    filter_artifacts: bool = False,
    min_confinement: float = 0.20,
    fade_transitional: bool = True,
) -> plt.Figure:
    """Re-plots a band structure diagram from saved simulation results JSON without re-running solvers.

    If `results_path` is None or 'latest', automatically resolves the most recent simulation run
    across the workspace using `phc_hydra.find_latest_simulation`.

    Args:
        results_path: Path to simulation_results.json or directory containing it. If None or 'latest',
            automatically locates the latest simulation matching the optional filters.
        solver: Solver filter when auto-discovering latest run (default: 'mpb').
        sim_type: Simulation type filter when auto-discovering (e.g. 'slab_mode_parity', 'band_diagram').
        geometry: Geometry filter when auto-discovering (e.g. 'c6v_primitive').
        base_dir: Base output directory for auto-discovery (default: 'outputs').
        save_path: Optional file path to save the generated figure.
        marker: Single marker or marker dictionary for polarization modes.
        markers: Optional dictionary mapping polarization keys to marker styles.
        markersize: Size of markers in points or dict per polarization.
        colors: Optional dictionary mapping polarization keys to colors.
        title: Plot title override.
        cmap: Colormap for continuous TE fraction dots (default: 'coolwarm_r').
        ax: Optional Matplotlib Axes to draw into.
        hollow_fractions: If True, renders fraction-colored modes as empty/hollow circles with colored boundaries (default: True).
        edge_linewidth: Boundary line width in points for hollow fraction markers (default: 1.6).
        filter_artifacts: If True, filters out spurious supercell radiation continuum artifacts (eta < min_confinement) (default: False).
        min_confinement: Confinement threshold below which modes are filtered (default: 0.20).
        fade_transitional: If True, scales opacity continuously for transitional leaky modes (default: True).

    Returns:
        Matplotlib Figure object containing the re-rendered band structure.

    Raises:
        FileNotFoundError: If no matching simulation results JSON file can be found.
        ValueError: If results JSON does not contain valid frequency data.
    """
    import json

    from phc_hydra import find_latest_simulation

    if results_path is None or str(results_path).strip() in ("", "latest", "recent"):
        target_dir = find_latest_simulation(
            solver=solver, sim_type=sim_type, geometry=geometry, base_dir=base_dir
        )
        json_file = target_dir / "simulation_results.json"
    else:
        p = Path(results_path)
        json_file = p / "simulation_results.json" if p.is_dir() else p

    if not json_file.is_file():
        raise FileNotFoundError(f"Simulation results JSON not found: {json_file}")

    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)

    extra = data.get("extra_data", {})
    freq_data = extra.get("frequencies", {})
    if not freq_data:
        raise ValueError(
            f"No 'frequencies' dictionary found in extra_data of {json_file}."
        )

    freqs_dict: dict[str, np.ndarray] = {}
    for k, v in freq_data.items():
        if v:
            freqs_dict[k] = np.array(v)

    te_fracs_raw = extra.get("te_fractions")
    te_fracs = np.array(te_fracs_raw) if te_fracs_raw else None

    conf_raw = extra.get("confinements")
    if isinstance(conf_raw, dict):
        confinements_data = {k: np.array(v) for k, v in conf_raw.items() if v}
    elif conf_raw:
        confinements_data = np.array(conf_raw)
    else:
        confinements_data = None

    results: dict[str, Any] = {
        "freqs": freqs_dict,
        "te_fractions": te_fracs,
        "confinements": confinements_data,
        "light_line": extra.get("light_line", []),
        "dimension": data.get("geometry", {}).get("dimension", "3D_slab"),
    }

    labels = extra.get("k_labels")
    indices = extra.get("k_indices")

    plot_title = title or (
        f"Re-plotted Band Structure — {data.get('geometry', {}).get('name', 'PhC')}"
    )

    return plot_band_structure(
        results=results,
        node_labels=labels,
        node_indices=indices,
        title=plot_title,
        save_path=save_path,
        marker=marker,
        markers=markers,
        markersize=markersize,
        colors=colors,
        cmap=cmap,
        ax=ax,
        hollow_fractions=hollow_fractions,
        edge_linewidth=edge_linewidth,
        filter_artifacts=filter_artifacts,
        min_confinement=min_confinement,
        fade_transitional=fade_transitional,
    )


def plot_epsilon(
    epsilon: np.ndarray,
    title: str = r"MPB Dielectric Permittivity $\varepsilon(\mathbf{r})$",
    cmap: str = "viridis",
    colorbar: bool = True,
    save_path: str | Path | None = None,
    extent: tuple[float, float, float, float] | None = None,
    extent_z: tuple[float, float] | None = None,
    xlabel: str = "Grid X",
    ylabel: str = "Grid Y",
    zlabel: str = "Grid Z",
    slice_z: int | None = None,
    slice_y: int | None = None,
    title_xy: str = r"In-Plane Mid-Plane $\varepsilon(x, y, z=0)$",
    title_xz: str = r"Vertical Cross-Section $\varepsilon(x, y=0, z)$",
    interpolation: str | None = "none",
) -> plt.Figure:
    """Plots the dielectric permittivity distribution grid retrieved from MPB.

    For 2D arrays, renders a single in-plane permittivity distribution.
    For 3D slab arrays, renders a unified single figure containing two subplots side by side:
      1. Left subplot: In-plane mid-plane cross-section (xy plane at z = z_mid).
      2. Right subplot: Vertical cross-section (xz plane at y = y_mid) showing slab thickness,
         cladding boundaries, and vertical hole profile.

    By default, uses `interpolation="none"` to display raw discrete numerical grid voxels
    without artificial spatial pixel smoothing.

    Transposes 2D slice arrays prior to rendering with `imshow` because MPB/MPBData returns
    dimension 0 along Cartesian X and dimension 1 along Cartesian Y (or Z), whereas Matplotlib
    maps array dimension 0 to vertical rows and dimension 1 to horizontal columns.

    Args:
        epsilon: 2D or 3D numpy array containing dielectric permittivity values epsilon(r).
        title: Title displayed at top of the overall figure.
        cmap: Matplotlib colormap name (default: "viridis").
        colorbar: If True, adds labeled colorbars showing the permittivity scale.
        save_path: Optional file path (PNG, PDF, SVG) where the figure will be saved.
            Parent directories are created automatically if they do not exist.
        extent: Optional 4-tuple (xmin, xmax, ymin, ymax) setting the in-plane plot coordinates.
        extent_z: Optional 2-tuple (zmin, zmax) setting vertical plot coordinates for the xz cut.
        xlabel: Label for horizontal axis (default: "Grid X").
        ylabel: Label for in-plane vertical axis (default: "Grid Y").
        zlabel: Label for out-of-plane vertical axis (default: "Grid Z").
        slice_z: Optional grid index along z for the xy mid-plane slice (default: Nz // 2).
        slice_y: Optional grid index along y for the xz cross-section (default: Ny // 2).
        title_xy: Subplot title for the xy in-plane slice (default: "In-Plane Mid-Plane...").
        title_xz: Subplot title for the xz cross-section (default: "Vertical Cross-Section...").
        interpolation: Matplotlib imshow interpolation scheme (default: "none" to show raw discrete voxels).

    Returns:
        The matplotlib Figure object containing the rendered permittivity plot(s).

    Raises:
        ValueError: If the input epsilon array is not 2D or 3D.
    """
    if epsilon.ndim not in (2, 3):
        raise ValueError(
            f"Expected 2D or 3D epsilon array, got shape {epsilon.shape} (ndim={epsilon.ndim})."
        )

    if epsilon.ndim == 3:
        # 3D Slab Dual-Plane Visualization: xy midplane and xz cross-section in one figure
        fig, (ax_xy, ax_xz) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

        mid_z = slice_z if slice_z is not None else epsilon.shape[2] // 2
        mid_y = slice_y if slice_y is not None else epsilon.shape[1] // 2

        xy_data = epsilon[:, :, mid_z]
        xz_data = epsilon[:, mid_y, :]

        # 1. Left Subplot: In-plane xy mid-plane
        im_xy = ax_xy.imshow(
            xy_data.T,
            origin="lower",
            extent=extent,
            cmap=cmap,
            interpolation=interpolation,
            aspect="equal",
        )
        ax_xy.set_title(title_xy, fontsize=11, fontweight="bold")
        ax_xy.set_xlabel(xlabel, fontsize=10)
        ax_xy.set_ylabel(ylabel, fontsize=10)
        if colorbar:
            cbar_xy = fig.colorbar(im_xy, ax=ax_xy, fraction=0.046, pad=0.04)
            cbar_xy.set_label(r"Permittivity $\varepsilon$", fontsize=10)

        # 2. Right Subplot: Vertical xz cross-section cut
        extent_xz = (
            (extent[0], extent[1], extent_z[0], extent_z[1])
            if extent is not None and extent_z is not None
            else None
        )
        im_xz = ax_xz.imshow(
            xz_data.T,
            origin="lower",
            extent=extent_xz,
            cmap=cmap,
            interpolation=interpolation,
            aspect="auto",
        )
        ax_xz.set_title(title_xz, fontsize=11, fontweight="bold")
        ax_xz.set_xlabel(xlabel, fontsize=10)
        ax_xz.set_ylabel(zlabel, fontsize=10)
        if colorbar:
            cbar_xz = fig.colorbar(im_xz, ax=ax_xz, fraction=0.046, pad=0.04)
            cbar_xz.set_label(r"Permittivity $\varepsilon$", fontsize=10)

        fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)
        plt.tight_layout()

    else:
        # 2D Periodic System: Single In-Plane Visualization
        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

        im = ax.imshow(
            epsilon.T,
            origin="lower",
            extent=extent,
            cmap=cmap,
            interpolation=interpolation,
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
