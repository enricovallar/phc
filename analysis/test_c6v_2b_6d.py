#!/usr/bin/env python3
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import meep as mp

# Ensure repository root is in sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from phc_hydra import SimulationOutputManager, resolve_simulation_output_dir
from phc_layout.components.unit_cell import phc_wyckoff_unit_cell
from phc_layout.lattice import HexagonalLattice
from phc_materials import get_material
from phc_mpb import (
    create_lattice,
    create_mode_solver,
    format_freqs_data,
    format_irreps_data,
    gds_to_mpb_geometry,
    get_epsilon_grid,
    get_high_symmetry_kpath,
    plot_band_structure,
    plot_epsilon,
    resolve_multiplet_symmetries,
    run_band_solver,
)
from phc_utils import silence_c_stdout

# Suppress Meep/MPB C-level verbosity by default
mp.verbosity(0)


def run_pipeline(
    pitch: float = 1.0,
    slab_thickness: float | None = None,
    r1: float = 0.15,
    r2: float = 0.10,
    p2: float = 0.25,
    supercell_z: float = 4.0,
    slab_material: str = "hBN",
    cladding_material: str = "air",
    resolution: int = 16,
    resolution_z: int = 16,
    num_bands: int = 8,
    k_density: int = 6,
    num_workers: int = 4,
    quick: bool = False,
    polarization: str = "te",
    compute_symmetries: bool = True,
    symmetry_group: str = "C6v",
    normalize: bool | None = None,
    lam_min: float = 0.3,
    lam_max: float = 0.6,
    output_dir: Path | str | None = None,
    verbose: bool = True,
    sim_name: str = "c6v_2b_6d",
) -> dict[str, Any]:
    """Runs the end-to-end simulation pipeline connecting layout, materials, and MPB.

    C6v hexagonal lattice unit cell with Wyckoff 2b holes (r1) and Wyckoff 6d holes (r2, param p2),
    etched into a hBN membrane or 2D periodic sheet.

    Note: it supports both 2D and 3D slab simulations.

    Args:
        pitch: Lattice constant a in micrometers (default: 1.0 μm).
        slab_thickness: Physical slab membrane thickness in micrometers (default: None).
            If None, simulates a 2D PhC.
        r1: Radius of primary holes at Wyckoff position 2b in micrometers (default: 0.15 μm).
        r2: Radius of satellite holes at Wyckoff position 6d in micrometers (default: 0.10 μm).
        p2: Coordinate parameter for Wyckoff position 6d (default: 0.15).
        supercell_z: Vertical supercell height in units of lattice constant a (default: 4.0).
        slab_material: Material key for dielectric slab core from phc_materials (default: "hBN").
        cladding_material: Material key for background cladding from phc_materials (default: "air").
        resolution: In-plane (x, y) mesh resolution per unit pitch a (default: 16).
        resolution_z: Vertical (z) mesh resolution per unit pitch a (default: 16).
        num_bands: Number of eigenbands to compute at each k-point (default: 8).
        k_density: Interpolation density between high-symmetry vertices (default: 6).
        num_workers: Number of concurrent worker processes for MPB (default: 4).
        quick: If True, overrides resolution, k_density, and num_bands to minimal settings.
        output_dir: Custom directory override. If None, auto-resolved via phc_hydra.
        verbose: If True, prints formatted progress updates to stdout.
        sim_name: Name of the simulation to use as the geometry name (default: "c6v_2b_6d").

    Returns:
        Dictionary containing:
            - 'results': MPB solver results including frequencies, gaps, and light line.
            - 'ms': Initialized MPB ModeSolver instance.
            - 'labels': High-symmetry k-path point labels.
            - 'indices': High-symmetry k-path tick indices.
            - 'eps_grid': Extracted 3D dielectric permittivity array.
            - 'output_dir': Resolved Path to the output directory.
            - 'files': Dict mapping artifact keys ('gds', 'band_plot', 'eps_plot', 'results_json')
              to saved file paths.
            - 'summary': Complete structured simulation results summary.

    Raises:
        RuntimeError: If MPB solver fails to produce frequency results.
    """
    t_start = time.time()

    # Resolve output directory according to phc_hydra SSOT standard
    out_path = resolve_simulation_output_dir(
        solver="mpb",
        sim_type="band_diagram",
        geometry=sim_name,
        override_dir=output_dir,
    )
    output_mgr = SimulationOutputManager(
        output_dir=out_path,
        solver="mpb",
        sim_type="slab_band_diagram",
        geometry_name=sim_name,  # pyright: ignore[reportArgumentType]
    )

    unit_cell = phc_wyckoff_unit_cell(
        pitch=pitch,
        point_group="C6v",
        features=[("2b", r1), ("6d", r2, p2)],
    )
    gds_path = output_mgr.save_gds(unit_cell, filename="unit_cell.gds")
    mat_slab = get_material(slab_material)
    mat_clad = get_material(cladding_material)
    hex_lat = HexagonalLattice(a=pitch)
    lattice = create_lattice(
        lattice_type=hex_lat,
        pitch=pitch,
        dimension="3D_slab" if slab_thickness is not None else "2D",
        supercell_z=supercell_z,
    )
    geometry = gds_to_mpb_geometry(
        gds_source=gds_path,
        pitch=pitch,
        dimension="3D_slab" if slab_thickness is not None else "2D",
        slab_thickness=slab_thickness if slab_thickness is not None else 0.5,
        slab_material=slab_material if slab_thickness is not None else None,
        geometry_lattice=lattice,
    )
    # High-symmetry path: M -> Γ -> K -> M (Γ is second point per convention)
    k_points, labels, indices = get_high_symmetry_kpath(
        lattice_type=hex_lat,
        k_density=k_density,
    )

    ms = create_mode_solver(
        geometry_lattice=lattice,
        geometry=geometry,
        k_points=k_points,
        default_material=mat_clad if slab_thickness is not None else mat_slab,
        resolution=(resolution, resolution, resolution_z)
        if slab_thickness is not None
        else resolution,
        num_bands=num_bands,
    )
    print("Running MPB band solver...")
    t_solve_start = time.time()
    with silence_c_stdout():
        results = run_band_solver(
            ms=ms,
            polarization=polarization,
            dimension="3D_slab" if slab_thickness is not None else "2D",
            cladding_index=mat_clad.index,
            num_workers=num_workers,
            compute_symmetries=compute_symmetries,
            symmetry_group=symmetry_group,
            verbose=False,
        )
    solve_duration = time.time() - t_solve_start
    print(f"Solved in {solve_duration:.2f} s.")

    print(f"Available results: {results.keys()}")
    print(f"Available frequencies: {results.get('freqs', {}).keys()}")
    freqs = results.get("freqs", {}).get(polarization)
    if freqs is None or len(freqs) == 0:
        raise RuntimeError(f"MPB failed to extract {polarization} band frequencies.")

    with silence_c_stdout():
        eps_grid = get_epsilon_grid(
            ms=ms,
            rectify=True,
            periods=2,
            periods_z=1,
            resolution=32 if not quick else 16,
        )

    fig_eps = plot_epsilon(
        epsilon=eps_grid,
        title=f"Permittivity — {mat_slab.name.upper()} Membrane in {mat_clad.name.upper()}",
    )
    eps_file = output_mgr.save_figure(
        fig_eps, artifact_key="eps_plot", filename="epsilon_map.png"
    )
    plt.close(fig_eps)
    if slab_thickness is None:
        title = f"2D C6v (r₁={r1:.2f}, r₂={r2:.2f}, p₂={p2:.2f})"
    else:
        title = f"C6v slab (h/a={slab_thickness / pitch:.2f}, r₁={r1:.2f}, r₂={r2:.2f}, p₂={p2:.2f})"
    norm_val = normalize if normalize is not None else (slab_thickness is None)
    fig_band = plot_band_structure(
        results=results,
        node_labels=labels,
        node_indices=indices,
        title=title,
        normalize=norm_val,
        pitch=pitch,
        lam_min=lam_min,
        lam_max=lam_max,
    )
    band_file = output_mgr.save_figure(
        fig_band, artifact_key="band_plot", filename="band_structure.png"
    )
    plt.close(fig_band)

    # Format and save freqs.data and irreps.data tables
    freqs_table_str = format_freqs_data(
        k_points=k_points,
        geometry_lattice=lattice,
        freqs=freqs,
    )
    freqs_file = output_mgr.save_data(
        data=freqs_table_str,
        artifact_key="freqs_data",
        filename="freqs.data",
    )

    sym_list = results.get("symmetries", {}).get(polarization, [])
    if not sym_list and "te_like" in results.get("symmetries", {}):
        sym_list = results["symmetries"]["te_like"]
    elif not sym_list and "all" in results.get("symmetries", {}):
        sym_list = results["symmetries"]["all"]

    gamma_idx = indices[1] if len(indices) > 1 else 0
    gamma_freqs = freqs[gamma_idx] if len(freqs) > gamma_idx else None
    irreps_table_str = format_irreps_data(
        symmetries=sym_list,
        gamma_freqs=gamma_freqs,
        kmag=0.0,
    )
    irreps_file = output_mgr.save_data(
        data=irreps_table_str,
        artifact_key="irreps_data",
        filename="irreps.data",
    )

    raw_syms = results.get("symmetries", {})
    serializable_syms: dict[str, list[dict[str, Any]]] = {}
    for p_k, recs in raw_syms.items():
        clean_recs = []
        for r in recs:
            r_copy = dict(r)
            if "characters" in r_copy:
                r_copy["characters"] = {
                    op: {
                        "real": round(float(v.real), 6),
                        "imag": round(float(v.imag), 6),
                    }
                    for op, v in r_copy["characters"].items()
                }
            clean_recs.append(r_copy)
        serializable_syms[p_k] = clean_recs

    t_total = time.time() - t_start
    summary_data = {
        "geometry": {
            "name": sim_name,
            "lattice_type": "hexagonal",
            "point_group": "C6v",
            "pitch_um": pitch,
            "slab_thickness_um": slab_thickness,
            "r1_um": r1,
            "r2_um": r2,
            "p2": p2,
            "supercell_z": supercell_z,
            "slab_material": slab_material,
            "cladding_material": cladding_material,
        },
        "simulation": {
            "solver": "mpb",
            "dimension": "3D_slab" if slab_thickness is not None else "2D",
            "polarization": polarization,
            "resolution_xy": resolution,
            "resolution_z": resolution_z,
            "num_bands": num_bands,
            "k_density": k_density,
        },
        "bands": {
            "k_points": len(k_points),
            "frequencies": {
                pol: arr.tolist() for pol, arr in results.get("freqs", {}).items()
            },
            "gaps": results.get("gaps", {}),
            "light_line": results.get("light_line", []),
            "symmetries": serializable_syms,
        },
        "elapsed_time_s": t_total,
    }
    json_path = output_mgr.save_results_json(summary_data)
    if verbose:
        print(f"      -> Saved simulation results JSON: {json_path}")
        print(f"\nCompleted in {t_total:.2f} seconds.")
        print("=" * 72)

    return {
        "results": results,
        "ms": ms,
        "labels": labels,
        "indices": indices,
        "eps_grid": eps_grid,
        "symmetries": results.get("symmetries", {}),
        "output_dir": out_path,
        "files": {
            "gds": gds_path,
            "band_plot": band_file,
            "eps_plot": eps_file,
            "freqs_data": freqs_file,
            "irreps_data": irreps_file,
            "results_json": json_path,
        },
        "summary": summary_data,
    }


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from phc_mpb import plot_band_structure

    num_bands = 15
    k_density = 20
    slab_thickness = 0.1
    normalize = slab_thickness is None
    num_workers = 20
    pol1 = "te_like" if slab_thickness is not None else "tm"
    pol2 = "tm_like" if slab_thickness is not None else "te"
    pitch = 0.381
    r1 = 0.081
    r2 = 0.034
    lam_min = 0.3
    lam_max = 0.6
    resolution = 25
    resolution_z = 20
    print(f"Running {pol1.upper()}...")
    res_pol1 = run_pipeline(
        polarization=pol1,
        sim_name=f"c6v_2b_6d_{pol1}",
        num_bands=num_bands,
        pitch=pitch,
        r1=r1,
        r2=r2,
        k_density=k_density,
        slab_thickness=slab_thickness,
        num_workers=num_workers,
        normalize=normalize,
        lam_min=lam_min,
        lam_max=lam_max,
    )

    print(f"Running {pol2.upper()}...")
    res_pol2 = run_pipeline(
        polarization=pol2,
        sim_name=f"c6v_2b_6d_{pol2}",
        num_bands=num_bands,
        k_density=k_density,
        slab_thickness=slab_thickness,
        num_workers=num_workers,
        r1=r1,
        r2=r2,
        pitch=pitch,
        normalize=normalize,
        lam_min=lam_min,
        lam_max=lam_max,
    )

    print("Running No Parity (All)...")
    # Note: Use "all" as the key so line 162 matches results["freqs"]["all"]
    res_all = run_pipeline(
        polarization="all",
        sim_name="c6v_2b_6d_all",
        num_bands=num_bands * 2,
        k_density=k_density,
        slab_thickness=slab_thickness,
        num_workers=num_workers,
        r1=r1,
        r2=r2,
        pitch=pitch,
        normalize=normalize,
        lam_min=lam_min,
        lam_max=lam_max,
    )

    # Combine freqs, gaps, and symmetries into a single results dictionary
    combined_results = {
        "freqs": {
            "all": res_all["results"]["freqs"]["all"],
            pol1: res_pol1["results"]["freqs"][pol1],
            pol2: res_pol2["results"]["freqs"][pol2],
        },
        "gaps": {
            **res_pol1["results"].get("gaps", {}),
            **res_pol2["results"].get("gaps", {}),
        },
        "symmetries": {
            pol1: res_pol1["results"].get("symmetries", {}).get(pol1, []),
            pol2: res_pol2["results"].get("symmetries", {}).get(pol2, []),
            "all": res_all["results"].get("symmetries", {}).get("all", []),
        },
    }

    labels = res_pol1["labels"]
    indices = res_pol1["indices"]

    # Print formatted table of irreducible representations at Gamma for each polarization
    print("\n" + "=" * 80)
    print("C6v Irreducible Representations at Γ for Each Polarization")
    print("=" * 80)
    gamma_idx = indices[1] if len(indices) > 1 else 0
    for pol_name in (pol1, pol2, "all"):
        sym_list = combined_results["symmetries"].get(pol_name, [])
        freq_mat = combined_results["freqs"].get(pol_name)
        gamma_freqs = (
            freq_mat[gamma_idx]
            if freq_mat is not None and len(freq_mat) > gamma_idx
            else []
        )
        sym_list = resolve_multiplet_symmetries(
            sym_list, symmetry_group="C6v", gamma_freqs=gamma_freqs
        )
        combined_results["symmetries"][pol_name] = sym_list
        print(f"\n--- Polarization: {pol_name.upper()} ---")
        print(
            f"{'Band':<6} | {'omega~':<10} | {'lambda (nm)':<12} | {'Irrep':<6} | {'Confidence':<10}"
        )
        print("-" * 52)
        for s in sym_list:
            b = s.get("band", 0)
            f = (
                float(gamma_freqs[b - 1])
                if len(gamma_freqs) >= b
                else s.get("freq", 0.0)
            )
            lam = (pitch / f * 1000.0) if f > 0 else 0.0
            irrep = s.get("irrep", "N/A")
            conf = s.get("confidence", 0.0)
            print(f"{b:<6d} | {f:<10.5f} | {lam:<12.2f} | {irrep:<6s} | {conf:<10.3f}")
    print("=" * 80 + "\n")

    fig = plot_band_structure(
        results=combined_results,
        node_labels=labels,
        node_indices=indices,
        title="C6v 2b 6d PhC Band Structure — TE, TM & All Modes",
        markers={"all": "s", pol1: "o", pol2: "^"},
        markersize={"all": 5.0, pol1: 3.0, pol2: 3.0},
        colors={"all": "gray", pol1: "tab:blue", pol2: "tab:red"},
        alpha={"all": 0.4, pol1: 0.9, pol2: 0.9},
        plot_gaps=True,
        normalize=normalize,
        pitch=pitch,
        lam_min=lam_min,
        lam_max=lam_max,
    )

    # Save to disk or display
    combined_img_path = Path("outputs/combined_c6v_2b_6d_band_structure.png")
    combined_img_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(combined_img_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    import json

    from phc_mpb import export_freqs_data, export_irreps_data

    # Export freqs.data and irreps.data tables requested by user
    k_pts = res_pol1["ms"].k_points
    geom_lat = res_pol1["ms"].geometry_lattice

    # 1. Export canonical freqs.data (TE by default, plus per-pol files)
    export_freqs_data(
        "outputs/freqs.data", k_pts, geom_lat, res_pol1["results"]["freqs"][pol1]
    )
    export_freqs_data(
        "outputs/freqs_te.data", k_pts, geom_lat, res_pol1["results"]["freqs"][pol1]
    )
    export_freqs_data(
        "outputs/freqs_tm.data", k_pts, geom_lat, res_pol2["results"]["freqs"][pol2]
    )
    export_freqs_data(
        "outputs/freqs_all.data", k_pts, geom_lat, res_all["results"]["freqs"]["all"]
    )

    # 2. Export canonical irreps.data (TE by default, plus per-pol files)
    export_irreps_data(
        "outputs/irreps.data",
        combined_results["symmetries"].get(pol1, []),
        gamma_freqs=combined_results["freqs"][pol1][gamma_idx],
    )
    export_irreps_data(
        "outputs/irreps_te.data",
        combined_results["symmetries"].get(pol1, []),
        gamma_freqs=combined_results["freqs"][pol1][gamma_idx],
    )
    export_irreps_data(
        "outputs/irreps_tm.data",
        combined_results["symmetries"].get(pol2, []),
        gamma_freqs=combined_results["freqs"][pol2][gamma_idx],
    )
    export_irreps_data(
        "outputs/irreps_all.data",
        combined_results["symmetries"].get("all", []),
        gamma_freqs=combined_results["freqs"]["all"][gamma_idx],
    )
    print(
        "Exported data tables: outputs/freqs.data, outputs/irreps.data (and per-polarization tables)"
    )

    combined_json_path = Path("outputs/combined_c6v_2b_6d_results.json")
    with open(combined_json_path, "w") as f_json:
        json.dump(
            {
                "pitch_um": pitch,
                "slab_thickness_um": slab_thickness,
                "r1_um": r1,
                "r2_um": r2,
                "frequencies": {
                    k: v.tolist() for k, v in combined_results["freqs"].items()
                },
                "symmetries": combined_results["symmetries"],
            },
            f_json,
            indent=2,
            default=str,
        )
    print(f"Saved combined results JSON: {combined_json_path}")
