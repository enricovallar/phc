"""Comparative electromagnetic analysis routines for photonic crystal slabs.

Provides automated multi-solver characterization comparing symmetric air-clad membranes
against asymmetric substrate-clad slabs at optimal locus design points.
"""

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import gdsfactory as gf
from phc_materials import get_material
from phc_mpb import (
    create_lattice,
    create_mode_solver,
    gds_to_mpb_geometry,
    get_high_symmetry_kpath,
    run_band_solver,
)
from phc_utils import silence_c_stdout

from phc_optimization.plotting import plot_band_structure_comparison


def _instantiate_cell(
    cell_factory: Callable[..., gf.Component],
    params: dict[str, Any],
) -> gf.Component:
    """Helper to instantiate cell component with signature filtering."""
    import inspect

    try:
        sig = inspect.signature(cell_factory)
        has_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if has_var_kw:
            cell_kwargs = params
        else:
            cell_kwargs = {k: v for k, v in params.items() if k in sig.parameters}
    except (ValueError, TypeError):
        cell_kwargs = params
    return cell_factory(**cell_kwargs)


def run_substrate_band_comparison(
    cell_factory: Callable[..., gf.Component],
    params: dict[str, Any],
    pitch: float = 1.0,
    slab_thickness: float = 0.5,
    supercell_z: float = 4.0,
    lattice_type: str = "square",
    matrix_material: str = "inp",
    background_material: str = "air",
    substrate_material: str = "sio2",
    resolution: int | tuple[int, int, int] = 16,
    num_bands: int = 10,
    num_bands_substrate: int | None = None,
    k_density: int = 12,
    num_workers: int = 1,
    target_frequency: float | None = None,
    output_dir: Path | str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Runs a comparative band structure analysis between an air-clad membrane and a SiO₂ substrate-clad slab.

    Evaluates the design at the specified parameters along the high-symmetry k-path:
    1. Symmetric Membrane (Air cladding): Solves guided TE-like modes (σ_z = +1 parity)
       with the Air light line (n_clad = 1.0).
    2. Asymmetric Slab (SiO₂ bottom substrate): Breaks vertical mirror symmetry σ_z,
       solving ALL modes (polarization='all') with increased band count and the SiO₂
       substrate light line (n_clad ≈ 1.44).
    3. Renders and saves a unified side-by-side band comparison figure.

    Args:
        cell_factory: Callable generating the GDSFactory unit cell layout.
        params: Dictionary of geometric parameters for the cell (e.g. {'r1': 0.25, 'r2': 0.20}).
        pitch: Lattice constant a in micrometers (default: 1.0).
        slab_thickness: Slab membrane thickness h in micrometers (default: 0.5).
        supercell_z: Vertical supercell height in units of pitch a (default: 4.0).
        lattice_type: Bravais lattice geometry ('square' or 'hexagonal').
        matrix_material: Slab core material key (default: 'inp').
        background_material: Top cladding material key (default: 'air').
        substrate_material: Bottom substrate cladding material key (default: 'sio2').
        resolution: Computational grid resolution per pitch unit a (default: 16).
        num_bands: Number of eigenbands computed for the air-clad membrane (default: 10).
        num_bands_substrate: Number of eigenbands computed for the substrate slab (default: 2.5 * num_bands).
        k_density: Number of interpolated k-points between high-symmetry vertices (default: 12).
        num_workers: Number of parallel worker processes for k-point solving (default: 1).
        target_frequency: Optional normalized Dirac frequency to mark as a reference line.
        output_dir: Optional directory where the comparison figure will be saved.
        verbose: Whether to print progress information to stdout.

    Returns:
        Dictionary containing:
            - 'results_air': Solver output dict for the symmetric air-clad membrane.
            - 'results_substrate': Solver output dict for the asymmetric substrate-clad slab.
            - 'fig_comparison': Matplotlib Figure object of the 2-panel comparison.
            - 'output_path': Path to the saved comparison figure (or None).
            - 'target_frequency': The evaluated target Dirac frequency.
    """
    n_sub_bands = (
        int(num_bands_substrate)
        if num_bands_substrate is not None
        else math.ceil(2.5 * num_bands)
    )
    h_val = float(params.get("slab_thickness", slab_thickness))
    sz_val = float(params.get("supercell_z", supercell_z))

    if verbose:
        param_str = ", ".join(f"{k}={v:.4f}" for k, v in params.items())
        print("\n" + "=" * 70)
        print(" Comparative Band Structure Analysis: Air vs SiO₂ Substrate")
        print(f"  Design Parameters:   {param_str}")
        print(f"  Pitch:               a = {pitch:.3f} μm")
        print(f"  Slab Thickness:      h = {h_val:.3f} μm (h/a = {h_val / pitch:.3f})")
        print(f"  Supercell Height:    s_z = {sz_val:.1f} a")
        print(f"  Air Membrane Bands:  {num_bands} (TE-like guided)")
        print(f"  Substrate Bands:     {n_sub_bands} (All modes, no parity)")
        print(f"  Substrate Material:  {substrate_material}")
        print("=" * 70)

    # 1. Instantiate layout and MPB lattice
    component = _instantiate_cell(cell_factory, params)
    mpb_lat = create_lattice(
        lattice_type=lattice_type,
        pitch=pitch,
        dimension="3D_slab",
        supercell_z=sz_val,
    )

    # 2. High-symmetry k-path
    k_pts, k_labels, k_indices = get_high_symmetry_kpath(
        lattice_type=lattice_type,
        k_density=k_density,
    )

    # 3. Simulate Symmetric Air-Clad Membrane (TE-like modes)
    if verbose:
        print("[1/2] Computing air-clad symmetric membrane band structure...")
    mpb_geom_air = gds_to_mpb_geometry(
        gds_source=component,
        pitch=pitch,
        dimension="3D_slab",
        slab_thickness=h_val,
        slab_material=matrix_material,
        etch_material=background_material,
        geometry_lattice=mpb_lat,
    )
    with silence_c_stdout():
        ms_air = create_mode_solver(
            geometry_lattice=mpb_lat,
            geometry=mpb_geom_air,
            k_points=k_pts,
            default_material=background_material,
            resolution=resolution,
            num_bands=num_bands,
        )
        res_air = run_band_solver(
            ms=ms_air,
            polarization="te_like",
            dimension="3D_slab",
            cladding_index=1.0,
            num_workers=num_workers,
            verbose=False,
        )

    # 4. Simulate Asymmetric SiO₂ Substrate-Clad Slab (All modes)
    if verbose:
        print(
            f"[2/2] Computing SiO₂ substrate-clad slab band structure ({n_sub_bands} bands, all modes)..."
        )
    mat_sub = get_material(substrate_material)
    mpb_geom_sub = gds_to_mpb_geometry(
        gds_source=component,
        pitch=pitch,
        dimension="3D_slab",
        slab_thickness=h_val,
        slab_material=matrix_material,
        substrate_material=substrate_material,
        etch_material=background_material,
        geometry_lattice=mpb_lat,
    )
    with silence_c_stdout():
        ms_sub = create_mode_solver(
            geometry_lattice=mpb_lat,
            geometry=mpb_geom_sub,
            k_points=k_pts,
            default_material=background_material,
            resolution=resolution,
            num_bands=n_sub_bands,
        )
        res_sub = run_band_solver(
            ms=ms_sub,
            polarization="all",
            dimension="3D_slab",
            cladding_index=mat_sub.index,
            num_workers=num_workers,
            verbose=False,
        )

    # 5. Render side-by-side comparison figure
    out_fig_path = None
    if output_dir is not None:
        p_dir = Path(output_dir).resolve()
        p_dir.mkdir(parents=True, exist_ok=True)
        out_fig_path = p_dir / "band_structure_substrate_comparison.png"

    param_summary = ", ".join(f"{k}={v:.3f}" for k, v in params.items())
    fig_title = (
        f"Photonic Band Structure Comparison: Air vs SiO₂ Substrate ({param_summary})"
    )
    fig = plot_band_structure_comparison(
        results_air=res_air,
        results_substrate=res_sub,
        node_labels=k_labels,
        node_indices=k_indices,
        target_frequency=target_frequency,
        title=fig_title,
        output_path=out_fig_path,
    )

    if verbose and out_fig_path is not None:
        print(f"  -> Saved band diagram comparison to '{out_fig_path}'")

    return {
        "results_air": res_air,
        "results_substrate": res_sub,
        "fig_comparison": fig,
        "output_path": out_fig_path,
        "target_frequency": target_frequency,
        "params": params,
    }
