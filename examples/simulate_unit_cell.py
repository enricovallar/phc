#!/usr/bin/env python3
"""Simulation Pipeline Template for Photonic Crystal Unit Cells.

Integrates:
1. Layout generation (phc_layout) from canonical database or custom Wyckoff features.
2. GDS layout export (phc_utils).
3. Optical material properties query (phc_materials).
4. Geometry and lattice conversion (phc_mpb).
5. High-symmetry k-path generation and MPB band solve (phc_mpb).
6. Band gap analysis, epsilon grid extraction, and artifact plotting.
7. Hydra-managed hierarchical outputs (outputs/<solver>/<sim_type>/<geometry>/<timestamp>/).
"""

from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from phc_hydra import SimulationOutputManager
from phc_layout.components import (
    UNIT_CELL_DATABASE,
    get_unit_cell,
    phc_wyckoff_unit_cell,
)
from phc_layout.lattice import get_lattice
from phc_materials import get_material
from phc_mpb import (
    create_mode_solver,
    gds_to_mpb_geometry,
    get_epsilon_grid,
    get_high_symmetry_kpath,
    lattice_to_mpb_lattice,
    plot_band_structure,
    plot_epsilon,
    run_band_solver,
)


def run_unit_cell_simulation(
    cfg: DictConfig | dict[str, Any],
    quick: bool = False,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Runs end-to-end unit cell band structure simulation connecting layout, materials, and MPB.

    Args:
        cfg: Configuration dictionary or Hydra DictConfig defining geometry, simulation,
            and output settings.
        quick: If True, overrides simulation parameters to minimal values for fast smoke testing.
        output_dir: Optional override for the output directory. If None, resolves from
            cfg or uses the hierarchical default: outputs/<solver>/<sim_type>/<geometry>/<timestamp>/.

    Returns:
        Dictionary containing:
            - 'results': Solver band data (frequencies, gaps, light line).
            - 'gaps': List of detected omnidirectional band gaps.
            - 'output_dir': Path to saved artifacts directory.
            - 'files': Dict of generated file paths (gds, band_plot, eps_plot, results_json).
            - 'mode_solver': Initialized MPB ModeSolver instance.

    Raises:
        ValueError: If configuration values are invalid or geometry is not recognized.
    """
    # Convert to plain dict for flexible extraction
    if isinstance(cfg, DictConfig):
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    else:
        cfg_dict = dict(cfg)

    geom_cfg = cfg_dict.get("geometry", {})
    sim_cfg = cfg_dict.get("simulation", {})

    # Extract geometric parameters
    geom_name = geom_cfg.get("name", "c6v_primitive")
    pitch_um = float(geom_cfg.get("pitch_um", 0.5))
    radius_ratio = geom_cfg.get("radius_ratio")
    radius_um = (float(radius_ratio) * pitch_um) if radius_ratio is not None else None
    point_group = geom_cfg.get("point_group", "C6v")
    lattice_type = geom_cfg.get("lattice_type", "hexagonal")
    dimension = geom_cfg.get("dimension", "2D")
    slab_thickness_um = float(geom_cfg.get("slab_thickness_um", 0.22))
    bg_mat_key = geom_cfg.get("background_material", "si")
    etch_mat_key = geom_cfg.get("etch_material", "air")
    custom_features = geom_cfg.get("features")

    # Extract simulation parameters
    solver = sim_cfg.get("solver", "mpb")
    sim_type = sim_cfg.get("sim_type", "band_diagram")
    polarization = sim_cfg.get("polarization", "te")
    num_bands = int(sim_cfg.get("num_bands", 8))
    resolution = int(sim_cfg.get("resolution", 32))
    k_density = int(sim_cfg.get("k_density", 16))
    verbose = bool(cfg_dict.get("verbose", True))

    is_quick = quick or bool(sim_cfg.get("quick", False))
    if is_quick:
        resolution = 8
        k_density = 2
        num_bands = 4

    # Resolve output directory via phc_hydra
    output_manager = SimulationOutputManager.from_config(cfg, output_dir=output_dir)
    out_path = output_manager.output_dir

    if verbose:
        print("\n=======================================================")
        print("Photonic Crystal Unit Cell Simulation Pipeline")
        print(f"Geometry:    {geom_name} (Lattice: {lattice_type}, PG: {point_group})")
        print(f"Materials:   Background='{bg_mat_key}', Holes='{etch_mat_key}'")
        print(f"Solver:      {solver} ({sim_type}, Pol: {polarization})")
        print(f"Resolution:  {resolution}, Bands: {num_bands}, K-Density: {k_density}")
        print(f"Output Path: {out_path}")
        print("=======================================================\n")

    # -------------------------------------------------------------
    # Step 1: Layout Generation
    # -------------------------------------------------------------
    if verbose:
        print(f"[1/6] Generating unit cell layout for '{geom_name}'...")

    wrap_cell = bool(geom_cfg.get("wrap_to_cell", False))
    if geom_name in ("c6v_painter_snowflake", "snowflake_painter", "snowflake"):
        radius_ratio = float(geom_cfg.get("radius_ratio", 0.40))
        float(geom_cfg.get("width_ratio", 0.15))
        unit_cell = get_unit_cell(
            "c6v_painter_snowflake",
            pitch=pitch_um,
            radius=radius_ratio * pitch_um,
            wrap_to_cell=wrap_cell,
        )
    elif custom_features is not None:
        feature_tuples = []
        for feat in custom_features:
            pos = feat["position"]
            r = (
                float(feat["radius_ratio"]) * pitch_um
                if "radius_ratio" in feat
                else float(feat["radius"])
            )
            param = feat.get("param")
            if param is not None:
                feature_tuples.append((pos, r, param))
            else:
                feature_tuples.append((pos, r))
        if geom_name in UNIT_CELL_DATABASE:
            unit_cell = get_unit_cell(
                geom_name,
                pitch=pitch_um,
                radius=radius_um,
                wrap_to_cell=wrap_cell,
                override_features=tuple(feature_tuples),
            )
        else:
            unit_cell = phc_wyckoff_unit_cell(
                pitch=pitch_um,
                point_group=point_group,
                features=tuple(feature_tuples),
                wrap_to_cell=wrap_cell,
            )
    elif geom_name in UNIT_CELL_DATABASE:
        unit_cell = get_unit_cell(
            geom_name, pitch=pitch_um, radius=radius_um, wrap_to_cell=wrap_cell
        )
    else:
        raise ValueError(
            f"Unknown geometry '{geom_name}'. Specify a valid UNIT_CELL_DATABASE name "
            "or custom features in configuration."
        )

    # -------------------------------------------------------------
    # Step 2: GDS Export (phc_hydra)
    # -------------------------------------------------------------
    gds_path = output_manager.save_gds(unit_cell, filename="unit_cell.gds")
    if verbose:
        print(f"      -> Exported GDS layout: {gds_path}")

    # -------------------------------------------------------------
    # Step 3: Material Query & SSOT
    # -------------------------------------------------------------
    bg_mat = get_material(bg_mat_key)
    etch_mat = get_material(etch_mat_key)
    if verbose:
        print(
            f"[2/6] Material SSOT: Background '{bg_mat.name}' (n={bg_mat.index:.3f}, eps={bg_mat.epsilon:.2f}), "
            f"Holes '{etch_mat.name}' (n={etch_mat.index:.3f})"
        )

    # -------------------------------------------------------------
    # Step 4: Lattice & Geometry Conversion
    # -------------------------------------------------------------
    if verbose:
        print("[3/6] Converting layout lattice to MPB domain...")
    layout_lat = get_lattice(lattice_type, pitch=pitch_um)
    mpb_lat = lattice_to_mpb_lattice(
        layout_lat,
        dimension=dimension,
        supercell=(1, 1),
    )

    mpb_geom = gds_to_mpb_geometry(
        gds_source=gds_path,
        pitch=pitch_um,
        dimension=dimension,
        slab_thickness=slab_thickness_um,
        etch_material=etch_mat_key,
        geometry_lattice=mpb_lat,
    )
    if verbose:
        print(f"      -> Extracted {len(mpb_geom)} physical hole prism(s) from GDS.")

    # -------------------------------------------------------------
    # Step 5: High-Symmetry K-Path & Band Solve
    # -------------------------------------------------------------
    k_pts, labels, indices = get_high_symmetry_kpath(layout_lat, k_density=k_density)
    if verbose:
        print(
            f"[4/6] Initializing MPB ModeSolver across Brillouin path: {' -> '.join(labels)} ({len(k_pts)} points)..."
        )

    ms = create_mode_solver(
        geometry_lattice=mpb_lat,
        geometry=mpb_geom,
        k_points=k_pts,
        default_material=bg_mat_key,
        resolution=resolution,
        num_bands=num_bands,
    )

    results = run_band_solver(
        ms=ms,
        polarization=polarization,
        dimension=dimension,
    )

    # -------------------------------------------------------------
    # Step 6: Analysis, Plotting & Output Generation
    # -------------------------------------------------------------
    if verbose:
        print("[5/6] Analyzing photonic band gaps...")

    detected_gaps = []
    freqs_dict = results.get("freqs", {})
    for pol_key, freqs_arr in freqs_dict.items():
        if freqs_arr is None or freqs_arr.size == 0:
            continue
        n_bands = freqs_arr.shape[1]
        for b in range(n_bands - 1):
            top_lower = float(np.max(freqs_arr[:, b]))
            bot_upper = float(np.min(freqs_arr[:, b + 1]))
            if bot_upper > top_lower:
                gap_width = bot_upper - top_lower
                midgap = (bot_upper + top_lower) / 2.0
                gap_ratio = (gap_width / midgap) * 100.0 if midgap > 0 else 0.0
                gap_entry = {
                    "polarization": pol_key,
                    "bands": [b + 1, b + 2],
                    "lower_freq": top_lower,
                    "upper_freq": bot_upper,
                    "gap_width": gap_width,
                    "midgap": midgap,
                    "gap_pct": gap_ratio,
                }
                detected_gaps.append(gap_entry)
                if verbose:
                    print(
                        f"      -> Complete {pol_key.upper()} Gap between Bands {b + 1}-{b + 2}: "
                        f"[{top_lower:.4f}, {bot_upper:.4f}], Delta_omega/omega_0 = {gap_ratio:.2f}%"
                    )

    if verbose and not detected_gaps:
        print("      -> No complete omnidirectional band gaps detected.")

    # Generate plots
    if verbose:
        print("[6/6] Generating visualization artifacts...")
    fig_band = plot_band_structure(
        results=results,
        node_labels=labels,
        node_indices=indices,
        title=f"Band Structure: {geom_name} ({polarization.upper()})",
    )
    output_manager.save_figure(
        fig_band,
        artifact_key="band_structure_plot",
        filename="band_structure.png",
        close=True,
    )

    plot_periods = int(sim_cfg.get("plot_periods", 3))
    eps_grid = get_epsilon_grid(
        ms, rectify=True, periodicity=plot_periods, resolution=resolution * 2
    )
    fig_eps = plot_epsilon(
        eps_grid,
        title=f"Dielectric Permittivity: {geom_name}",
    )
    output_manager.save_figure(
        fig_eps,
        artifact_key="epsilon_plot",
        filename="epsilon_map.png",
        close=True,
    )

    # Save summary JSON (enforces mandatory GDS layout check)
    output_manager.save_results_json(
        geometry_cfg={
            "name": geom_name,
            "lattice_type": lattice_type,
            "point_group": point_group,
            "pitch_um": pitch_um,
            "dimension": dimension,
            "background_material": bg_mat_key,
            "etch_material": etch_mat_key,
        },
        simulation_cfg={
            "solver": solver,
            "sim_type": sim_type,
            "polarization": polarization,
            "resolution": resolution,
            "num_bands": num_bands,
            "k_density": k_density,
        },
        band_gaps=detected_gaps,
    )

    if verbose:
        print(f"\nSimulation complete! All artifacts saved to: {out_path}\n")

    return {
        "results": results,
        "gaps": detected_gaps,
        "output_dir": out_path,
        "files": output_manager.manifest,
        "mode_solver": ms,
    }


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra CLI entrypoint for running unit cell simulations."""
    run_unit_cell_simulation(cfg)


if __name__ == "__main__":
    main()
