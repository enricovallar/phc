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
from phc_materials import get_material
from phc_mpb import (
    create_lattice,
    create_mode_solver,
    gds_to_mpb_geometry,
    get_epsilon_grid,
    get_high_symmetry_kpath,
    plot_band_structure,
    plot_epsilon,
    run_band_solver,
)
from phc_utils import silence_c_stdout

# Suppress Meep/MPB C-level verbosity by default
mp.verbosity(0)


def run_pipeline(
    pitch: float = 1.0,
    slab_thickness: float | None = None,
    r1: float = 0.25,
    r2: float = 0.15,
    supercell_z: float = 4.0,
    slab_material: str = "inp",
    cladding_material: str = "air",
    resolution: int = 16,
    resolution_z: int = 16,
    num_bands: int = 8,
    k_density: int = 6,
    num_workers: int = 4,
    quick: bool = False,
    polarization: str = "te",
    output_dir: Path | str | None = None,
    verbose: bool = True,
    sim_name: str = "c4v_test"
) -> dict[str, Any]:
    """Runs the end-to-end  simulation pipeline connecting layout, materials, and MPB.

    Uses the identical slab geometry and default parameters as the 3D Dirac optimization example:
    C4v square lattice unit cell with Wyckoff 1a hole (r1) at the origin and Wyckoff 1b hole (r2)
    at the cell center, etched into an InP membrane suspended in air.

    Note: it should support 2D simulations

    Args:
        pitch: Lattice constant a in micrometers (default: 1.0 μm).
        slab_thickness: Physical slab membrane thickness in micrometers (default: None)
            If None, simulate a 2D phc.
        r1: Radius of primary hole at Wyckoff position 1a (origin) in micrometers (default: 0.25 μm).
        r2: Radius of secondary hole at Wyckoff position 1b (center) in micrometers (default: 0.15 μm).
        supercell_z: Vertical supercell height in units of lattice constant a (default: 4.0).
        slab_material: Material key for dielectric slab core from phc_materials (default: "inp").
        cladding_material: Material key for background cladding from phc_materials (default: "air").
        resolution: In-plane (x, y) mesh resolution per unit pitch a (default: 16).
        resolution_z: Vertical (z) mesh resolution per unit pitch a (default: 16).
        num_bands: Number of eigenbands to compute at each k-point (default: 8).
        k_density: Interpolation density between high-symmetry vertices (default: 6).
        num_workers: Number of concurrent worker processes for MPB (default: 4).
        quick: If True, overrides resolution, k_density, and num_bands to minimal settings
            (res=12, res_z=6, num_bands=4, k_density=2, workers=1) for rapid smoke testing (<2 s).
        output_dir: Custom directory override. If None, auto-resolved via phc_hydra:
            outputs/mpb/slab_band_diagram/c4v_dirac_3d/<timestamp>/.
        verbose: If True, prints formatted progress updates to stdout.
        sim_name: Name of the simulation to use as the geometry name (default: "c4v_test").

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
        point_group="C4v",
        features=[("1a", r1), ("1b", r2)],
    )
    gds_path = output_mgr.save_gds(unit_cell, filename="unit_cell.gds")
    mat_slab = get_material(slab_material)
    mat_clad = get_material(cladding_material)
    lattice = create_lattice(
        lattice_type="square",
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
    # High-symmetry path: X -> Γ -> M -> X (Γ is second point per convention)
    k_points, labels, indices = get_high_symmetry_kpath(
        lattice_type="square",
        k_density=k_density,
    )

    ms = create_mode_solver(
        geometry_lattice=lattice,
        geometry=geometry,
        k_points=k_points,
        default_material=mat_clad if slab_thickness is not None else mat_slab,
        resolution=(resolution, resolution, resolution_z) if slab_thickness is not None else resolution,
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
        title = f"r₁={r1:.2f}, r₂={r2:.2f}"
    else:
        title = f"h/a={slab_thickness  / pitch:.2f}, r₁={r1:.2f}, r₂={r2:.2f}"
    fig_band = plot_band_structure(
        results=results,
        node_labels=labels,
        node_indices=indices,
        title=title,
    )
    band_file = output_mgr.save_figure(
        fig_band, artifact_key="band_plot", filename="band_structure.png"
    )
    plt.close(fig_band)


    # Save structured results JSON summary
    t_total = time.time() - t_start
    summary_data = {
        "geometry": {
            "name": sim_name,
            "lattice_type": "square",
            "point_group": "C4v",
            "pitch_um": pitch,
            "slab_thickness_um": slab_thickness,
            "r1_um": r1,
            "r2_um": r2,
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
        "output_dir": out_path,
        "files": {
            "gds": gds_path,
            "band_plot": band_file,
            "eps_plot": eps_file,
            "results_json": json_path,
        },
        "summary": summary_data,
    }

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from phc_mpb import plot_band_structure
    num_bands = 8
    k_density = 5
    slab_thickness = 0.5
    num_workers = 8
    pol1 = "te_like" if slab_thickness is not None else "tm"
    pol2 = "tm_like" if slab_thickness is not None else "te"

    print(f"Running {pol1.upper()}...")
    res_pol1 = run_pipeline(polarization=pol1, sim_name=f"c4v_{pol1}", num_bands=num_bands,
        k_density=k_density, slab_thickness=slab_thickness, num_workers=num_workers)

    print(f"Running {pol2.upper()}...")
    res_pol2 = run_pipeline(polarization=pol2, sim_name=f"c4v_{pol2}", num_bands=num_bands,
        k_density=k_density, slab_thickness=slab_thickness, num_workers=num_workers)

    print("Running No Parity (All)...")
    # Note: Use "all" as the key so line 162 matches results["freqs"]["all"]
    res_all = run_pipeline(polarization="all", sim_name="c4v_all", num_bands=num_bands*2,
        k_density=k_density, slab_thickness=slab_thickness, num_workers=num_workers)

    # Combine freqs and gaps into a single results dictionary
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
    }

    labels = res_pol1["labels"]
    indices = res_pol1["indices"]

    fig = plot_band_structure(
        results=combined_results,
        node_labels=labels,
        node_indices=indices,
        title="2D C4v PhC Band Structure — TE, TM & All Modes",
        markers={"all": "s", pol1: "o", pol2: "^"},
        markersize={"all": 5.0, pol1: 3.0, pol2: 3.0},
        colors={"all": "gray", pol1: "tab:blue", pol2: "tab:red"},
        alpha={"all": 0.4, pol1: 0.9, pol2: 0.9},
        plot_gaps=True,

    )

    # Save to disk or display
    fig.savefig("outputs/combined_c4v_band_structure.png", dpi=150, bbox_inches="tight")
    plt.show()
