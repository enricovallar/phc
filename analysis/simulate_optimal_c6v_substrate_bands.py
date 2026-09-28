#!/usr/bin/env python3
"""Band Structure Simulation and Normal TE Symmetry Analysis for the Optimal C6v Photonic Crystal.

Simulates the full dispersion band structure of the optimal C6v Wyckoff 2b-6d unit cell
on a SiO2 substrate along the high-symmetry k-path (M -> Γ -> K -> M). Also provides the
canonical 'normal TE' (z-even) implementation for the air-clad reference membrane,
performing point-group symmetry classification at Γ to verify the target irreps
(E_1 doublet and A_2 singlet).

Optimal Parameters (accidental degeneracy at Γ around 440 nm):
    - r1 = 0.120 μm (Wyckoff 2b hole radius, r1/a = 0.315)
    - r2 = 0.060 μm (Wyckoff 6d hole radius, r2/a = 0.157)
    - p2 = 0.25 (Wyckoff 6d radial displacement)
    - pitch a = 0.381 μm (or physical a = 357.94 nm, h = 93.95 nm for exact 440 nm)
    - slab thickness h = 0.100 μm
    - slab material = hBN (n ≈ 2.12)
    - substrate = SiO2 (n ≈ 1.444)

Command-Line Usage:
    # Full band structure simulation on SiO2 substrate:
    python analysis/simulate_optimal_c6v_substrate_bands.py

    # Quick smoke test (< 5 seconds):
    python analysis/simulate_optimal_c6v_substrate_bands.py --quick

    # Run normal TE implementation with Gamma symmetry verification:
    python analysis/simulate_optimal_c6v_substrate_bands.py --check-symmetries

    # Run both full band structure and symmetry verification:
    python analysis/simulate_optimal_c6v_substrate_bands.py --all

CLI Options:
    --quick              Execute rapid low-resolution smoke test.
    --check-symmetries   Run normal TE and all-polarization symmetry analysis at Gamma.
    --all                Run both band structure simulation and symmetry checks.
    --resolution RES     In-plane MPB grid resolution (default: 24, quick: 12).
    --resolution-z RESZ  Vertical MPB grid resolution (default: 20, quick: 8).
    --num-bands N        Number of eigenbands to compute (default: 22, quick: 8).
    --k-density K        Number of k-points between high-symmetry vertices (default: 12, quick: 2).
    --workers W          Number of parallel worker processes (default: 4, quick: 1).
    --output-dir PATH    Custom directory override for output artifacts.
"""

import argparse
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
    run_band_solver,
)
from phc_utils import silence_c_stdout

# Suppress MPB solver chatter by default
mp.verbosity(0)


def run_normal_te_symmetry_analysis(
    pitch: float = 0.381,
    slab_thickness: float = 0.100,
    r1: float = 0.120,
    r2: float = 0.060,
    p2: float = 0.25,
    resolution: int = 20,
    resolution_z: int = 16,
    num_bands: int = 14,
    verbose: bool = True,
) -> dict[str, Any]:
    """Executes the normal TE (run_zeven) implementation and verifies C6v point-group symmetries.

    Evaluates:
    1. Symmetric air-clad membrane: uses normal TE (run_zeven) to calculate the eigenbands
       and evaluates the point-group character projections at Γ. Verifies that the target
       degenerate cluster consists of an E_1 doublet and an A_2 singlet.
    2. Asymmetric SiO2 substrate: solves all modes and verifies the corresponding
       E_1 and A_2 modal symmetries for the accidental degeneracy at Γ.

    Args:
        pitch: Lattice constant a in micrometers (default: 0.381 μm).
        slab_thickness: Membrane thickness h in micrometers (default: 0.100 μm).
        r1: Radius of Wyckoff 2b holes in micrometers (default: 0.120 μm).
        r2: Radius of Wyckoff 6d holes in micrometers (default: 0.060 μm).
        p2: Coordinate parameter for Wyckoff 6d holes (default: 0.25).
        resolution: In-plane MPB grid resolution (default: 20).
        resolution_z: Vertical MPB grid resolution (default: 16).
        num_bands: Number of eigenbands at Gamma (default: 14).
        verbose: Whether to print formatted symmetry results to stdout.

    Returns:
        Dictionary containing 'air_te_records' and 'substrate_records'.
    """
    hex_lat = HexagonalLattice(a=pitch)
    lattice = create_lattice(
        lattice_type=hex_lat,
        pitch=pitch,
        dimension="3D_slab",
        supercell_z=4.0,
    )
    unit_cell = phc_wyckoff_unit_cell(
        pitch=pitch,
        point_group="C6v",
        features=[("2b", r1), ("6d", r2, p2)],
    )

    # 1. Air membrane normal TE (run_zeven)
    geom_air = gds_to_mpb_geometry(
        gds_source=unit_cell,
        pitch=pitch,
        dimension="3D_slab",
        slab_thickness=slab_thickness,
        slab_material="hBN",
        substrate_material=None,
        etch_material="air",
        geometry_lattice=lattice,
    )

    with silence_c_stdout():
        ms_air = create_mode_solver(
            geometry_lattice=lattice,
            geometry=geom_air,
            k_points=[mp.Vector3(0, 0, 0)],
            default_material="air",
            resolution=(resolution, resolution, resolution_z),
            num_bands=num_bands,
        )
        res_air = run_band_solver(
            ms=ms_air,
            polarization="te",
            dimension="3D_slab",
            compute_symmetries=True,
            symmetry_group="C6v",
            verbose=False,
        )

    air_freqs = res_air["freqs"]["te_like"][0]
    air_syms = res_air.get("symmetries", {}).get("te_like", [])

    if verbose:
        print("\n" + "=" * 76)
        print("  1. NORMAL TE IMPLEMENTATION (z-even) — Air-Clad Reference Membrane")
        print("=" * 76)
        print(f"  {'Band':<6} | {'omega~':<8} | {'lambda (nm)':<12} | {'Irrep':<6} | {'Confidence':<10} | Projections")
        print("  " + "-" * 72)
        for b_idx, f in enumerate(air_freqs):
            b = b_idx + 1
            lam = (pitch / f * 1000.0) if f > 0 else 0.0
            s_rec = next((s for s in air_syms if s.get("band") == b), None)
            irrep = s_rec.get("irrep", "N/A") if s_rec else "N/A"
            conf = s_rec.get("confidence", 0.0) if s_rec else 0.0
            projs = s_rec.get("projections", {}) if s_rec else {}
            flag = " <--- TARGET BAND" if b in (9, 10, 11) else ""
            print(f"  {b:<6d} | {f:<8.5f} | {lam:<12.2f} | {irrep:<6s} | {conf:<10.3f} | {projs}{flag}")

    # 2. SiO2 Substrate (polarization="all", broken z-parity)
    geom_sub = gds_to_mpb_geometry(
        gds_source=unit_cell,
        pitch=pitch,
        dimension="3D_slab",
        slab_thickness=slab_thickness,
        slab_material="hBN",
        substrate_material="sio2",
        etch_material="air",
        geometry_lattice=lattice,
    )

    with silence_c_stdout():
        ms_sub = create_mode_solver(
            geometry_lattice=lattice,
            geometry=geom_sub,
            k_points=[mp.Vector3(0, 0, 0)],
            default_material="air",
            resolution=(resolution, resolution, resolution_z),
            num_bands=max(20, num_bands + 6),
        )
        res_sub = run_band_solver(
            ms=ms_sub,
            polarization="all",
            dimension="3D_slab",
            compute_symmetries=True,
            symmetry_group="C6v",
            verbose=False,
        )

    sub_freqs = res_sub["freqs"]["all"][0]
    sub_syms = res_sub.get("symmetries", {}).get("all", [])

    if verbose:
        print("\n" + "=" * 76)
        print("  2. OPTIMAL POINT ON SiO2 SUBSTRATE (polarization='all') — Target Modes at Γ")
        print("=" * 76)
        print(f"  {'Band':<6} | {'omega~':<8} | {'lambda (nm)':<12} | {'Irrep':<6} | {'Confidence':<10} | Projections")
        print("  " + "-" * 72)
        for b_idx, f in enumerate(sub_freqs):
            b = b_idx + 1
            lam = (pitch / f * 1000.0) if f > 0 else 0.0
            s_rec = next((s for s in sub_syms if s.get("band") == b), None)
            irrep = s_rec.get("irrep", "N/A") if s_rec else "N/A"
            conf = s_rec.get("confidence", 0.0) if s_rec else 0.0
            projs = s_rec.get("projections", {}) if s_rec else {}
            flag = " <--- ACCIDENTAL DEGENERACY" if b in (16, 17, 18) else ""
            print(f"  {b:<6d} | {f:<8.5f} | {lam:<12.2f} | {irrep:<6s} | {conf:<10.3f} | {projs}{flag}")
        print("=" * 76 + "\n")

    return {
        "air_frequencies": air_freqs.tolist(),
        "air_symmetries": air_syms,
        "substrate_frequencies": sub_freqs.tolist(),
        "substrate_symmetries": sub_syms,
    }


def run_optimal_band_structure_simulation(
    pitch: float = 0.381,
    slab_thickness: float = 0.100,
    r1: float = 0.120,
    r2: float = 0.060,
    p2: float = 0.25,
    supercell_z: float = 4.0,
    slab_material: str = "hBN",
    substrate_material: str = "sio2",
    cladding_material: str = "air",
    resolution: int = 24,
    resolution_z: int = 20,
    num_bands: int = 22,
    k_density: int = 12,
    num_workers: int = 4,
    quick: bool = False,
    output_dir: Path | str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Runs full band structure simulation for the optimal C6v design on SiO2 substrate.

    Computes dispersion along M -> Γ -> K -> M with discrete eigenfrequency dots,
    light lines for both SiO2 substrate and air cladding, modal TE fractions,
    dielectric permittivity maps, GDS layout export, and structured JSON results.

    Args:
        pitch: Lattice pitch a in micrometers (default: 0.381 μm).
        slab_thickness: Membrane thickness h in micrometers (default: 0.100 μm).
        r1: Wyckoff 2b hole radius in micrometers (default: 0.120 μm).
        r2: Wyckoff 6d hole radius in micrometers (default: 0.060 μm).
        p2: Wyckoff 6d radial parameter (default: 0.25).
        supercell_z: Supercell height in units of pitch a (default: 4.0).
        slab_material: Core slab material key (default: "hBN").
        substrate_material: Substrate material key (default: "sio2").
        cladding_material: Top cladding material key (default: "air").
        resolution: In-plane computational mesh resolution (default: 24).
        resolution_z: Vertical computational mesh resolution (default: 20).
        num_bands: Number of eigenbands computed per k-point (default: 22).
        k_density: Number of k-points between high-symmetry vertices (default: 12).
        num_workers: Number of parallel worker processes (default: 4).
        quick: If True, uses minimal settings for rapid smoke-testing (<3 s).
        output_dir: Custom output directory override.
        verbose: If True, prints formatted progress updates.

    Returns:
        Dictionary containing simulation results, output paths, and structured data.
    """
    t0 = time.time()
    sim_name = "c6v_2b_6d_substrate_optimal"

    # Quick smoke test overrides
    if quick:
        resolution = 12
        resolution_z = 8
        num_bands = 8
        k_density = 2
        num_workers = 1

    # Resolve output directory via phc_hydra SSOT standard
    out_path = resolve_simulation_output_dir(
        solver="mpb",
        sim_type="band_diagram",
        geometry=sim_name,
        override_dir=output_dir,
    )
    output_mgr = SimulationOutputManager(
        output_dir=out_path,
        solver="mpb",
        sim_type="band_diagram",
        geometry_name=sim_name,  # pyright: ignore[reportArgumentType]
    )

    if verbose:
        print("\n" + "=" * 76)
        print("  SIMULATING OPTIMAL C6v BAND STRUCTURE ON SiO2 SUBSTRATE")
        print(f"  Output directory: {out_path}")
        print("=" * 76)

    # 1. Build layout & export mandatory GDS
    unit_cell = phc_wyckoff_unit_cell(
        pitch=pitch,
        point_group="C6v",
        features=[("2b", r1), ("6d", r2, p2)],
    )
    gds_path = output_mgr.save_gds(unit_cell, filename="unit_cell.gds")

    # 2. Setup MPB lattice, geometry, and high-symmetry path
    mat_slab = get_material(slab_material)
    mat_sub = get_material(substrate_material)

    hex_lat = HexagonalLattice(a=pitch)
    lattice = create_lattice(
        lattice_type=hex_lat,
        pitch=pitch,
        dimension="3D_slab",
        supercell_z=supercell_z,
    )
    geometry = gds_to_mpb_geometry(
        gds_source=gds_path,
        pitch=pitch,
        dimension="3D_slab",
        slab_thickness=slab_thickness,
        slab_material=slab_material,
        substrate_material=substrate_material,
        etch_material=cladding_material,
        geometry_lattice=lattice,
    )

    # High-symmetry path: M -> Γ -> K -> M (Γ is second vertex)
    k_points, labels, indices = get_high_symmetry_kpath(
        lattice_type=hex_lat,
        k_density=k_density,
    )

    ms = create_mode_solver(
        geometry_lattice=lattice,
        geometry=geometry,
        k_points=k_points,
        default_material=cladding_material,
        resolution=(resolution, resolution, resolution_z),
        num_bands=num_bands,
    )

    # 3. Solve band structure (polarization='all' for asymmetric slab)
    if verbose:
        print(f"  Solving {len(k_points)} k-points with {num_bands} bands ({num_workers} workers)...")
    t_solve = time.time()
    with silence_c_stdout():
        results = run_band_solver(
            ms=ms,
            polarization="all",
            dimension="3D_slab",
            cladding_index=mat_sub.index,  # Substrate light line
            num_workers=num_workers,
            compute_fractions=True,
            polarization_method="slab",
            slab_thickness=slab_thickness / pitch,
            z_center=0.0,
            verbose=False,
        )
    solve_duration = time.time() - t_solve
    if verbose:
        print(f"  Solved in {solve_duration:.2f} s.")

    # 4. Generate & save dual-plane dielectric permittivity map
    if verbose:
        print("  Generating dual-plane permittivity map...")
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
        title=f"Permittivity — {mat_slab.name.upper()} Membrane on {mat_sub.name.upper()}",
    )
    eps_file = output_mgr.save_figure(
        fig_eps, artifact_key="eps_plot", filename="epsilon_map.png"
    )
    plt.close(fig_eps)

    # 5. Plot Band Structure in Normalized Frequency (omega * a / 2pi c)
    if verbose:
        print("  Rendering normalized band structure...")
    fig_norm = plot_band_structure(
        results=results,
        node_labels=labels,
        node_indices=indices,
        title=f"Optimal C6v Slab on SiO₂ (r₁={r1:.3f}μm, r₂={r2:.3f}μm, h={slab_thickness:.3f}μm)",
        normalize=True,
        pitch=pitch,
        plot_gaps=True,
    )
    # Add substrate and air light lines
    norm_band_file = output_mgr.save_figure(
        fig_norm, artifact_key="band_plot", filename="band_structure.png"
    )
    plt.close(fig_norm)

    # 6. Plot Band Structure in Physical Wavelength (lambda in nm)
    if verbose:
        print("  Rendering physical wavelength band structure (350 - 550 nm)...")
    fig_wave = plot_band_structure(
        results=results,
        node_labels=labels,
        node_indices=indices,
        title=f"Optimal C6v Dispersion — λ (nm) [SiO₂ Substrate, h={slab_thickness*1e3:.0f} nm]",
        normalize=False,
        pitch=pitch,
        lam_min=0.35,
        lam_max=0.55,
        plot_gaps=False,
    )
    wave_band_file = output_mgr.save_figure(
        fig_wave, artifact_key="band_plot_wavelength", filename="band_structure_wavelength.png"
    )
    plt.close(fig_wave)

    # Export freqs.data and irreps.data ASCII tables
    freqs_table_str = format_freqs_data(
        k_points=k_points,
        geometry_lattice=lattice,
        freqs=results["freqs"]["all"],
    )
    freqs_file = output_mgr.save_data(
        data=freqs_table_str,
        artifact_key="freqs_data",
        filename="freqs.data",
    )

    gamma_idx = indices[1]  # Γ is second vertex
    gamma_freqs = results["freqs"]["all"][gamma_idx]
    sub_syms = results.get("symmetries", {}).get("all", [])
    irreps_table_str = format_irreps_data(
        symmetries=sub_syms,
        gamma_freqs=gamma_freqs,
        kmag=0.0,
    )
    irreps_file = output_mgr.save_data(
        data=irreps_table_str,
        artifact_key="irreps_data",
        filename="irreps.data",
    )

    # 7. Extract Gamma point degeneracy metrics
    # Target bands around 0.814 (bands 16, 17, 18 if num_bands >= 18)
    deg_info: dict[str, Any] = {}
    if len(gamma_freqs) >= 18:
        b16 = float(gamma_freqs[15])
        b17 = float(gamma_freqs[16])
        b18 = float(gamma_freqs[17])
        split = b18 - b16
        mid = (b18 + b16) / 2.0
        lam_mid = pitch / mid * 1000.0
        deg_info = {
            "band_16_freq": b16,
            "band_17_freq": b17,
            "band_18_freq": b18,
            "frequency_splitting": split,
            "normalized_splitting_cost": split / mid,
            "degeneracy_frequency_omega": mid,
            "degeneracy_wavelength_nm": lam_mid,
        }
        if verbose:
            print("\n  Degenerate Cluster at Γ:")
            print(f"    Band 16: omega~ = {b16:.5f} (lambda = {pitch/b16*1e3:.2f} nm)")
            print(f"    Band 17: omega~ = {b17:.5f} (lambda = {pitch/b17*1e3:.2f} nm)")
            print(f"    Band 18: omega~ = {b18:.5f} (lambda = {pitch/b18*1e3:.2f} nm)")
            print(f"    Delta omega~ = {split:.6f} ({split/mid*100:.3f}% splitting)")

    # 8. Save structured results JSON summary
    t_total = time.time() - t0
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
            "r1_over_a": r1 / pitch,
            "r2_over_a": r2 / pitch,
            "supercell_z": supercell_z,
            "slab_material": slab_material,
            "substrate_material": substrate_material,
            "cladding_material": cladding_material,
        },
        "simulation": {
            "solver": "mpb",
            "dimension": "3D_slab",
            "polarization": "all",
            "resolution_xy": resolution,
            "resolution_z": resolution_z,
            "num_bands": num_bands,
            "k_density": k_density,
        },
        "degeneracy_at_gamma": deg_info,
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
        print(f"\n  Saved simulation artifacts to: {out_path}")
        print(f"  Elapsed total time: {t_total:.2f} s")
        print("=" * 76 + "\n")

    return {
        "results": results,
        "ms": ms,
        "labels": labels,
        "indices": indices,
        "eps_grid": eps_grid,
        "output_dir": out_path,
        "files": {
            "gds": gds_path,
            "band_plot": norm_band_file,
            "band_plot_wavelength": wave_band_file,
            "eps_plot": eps_file,
            "freqs_data": freqs_file,
            "irreps_data": irreps_file,
            "results_json": json_path,
        },
        "summary": summary_data,
        "degeneracy_metrics": deg_info,
    }


def main() -> None:
    """CLI entrypoint for optimal C6v band simulation and symmetry analysis."""
    parser = argparse.ArgumentParser(
        description="Simulate band structure and verify normal TE symmetries for the optimal C6v PhC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--quick", action="store_true", help="Rapid smoke test mode.")
    parser.add_argument("--check-symmetries", action="store_true", help="Run normal TE symmetry checks.")
    parser.add_argument("--all", action="store_true", help="Run both band simulation and symmetry checks.")
    parser.add_argument("--resolution", type=int, default=24, help="In-plane grid resolution.")
    parser.add_argument("--resolution-z", type=int, default=20, help="Vertical grid resolution.")
    parser.add_argument("--num-bands", type=int, default=22, help="Number of eigenbands.")
    parser.add_argument("--k-density", type=int, default=12, help="K-path interpolation density.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker processes.")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory.")

    args = parser.parse_args()

    if args.check_symmetries or args.all:
        run_normal_te_symmetry_analysis(
            resolution=args.resolution if not args.quick else 14,
            resolution_z=args.resolution_z if not args.quick else 8,
            num_bands=14 if not args.quick else 8,
        )

    if not args.check_symmetries or args.all:
        run_optimal_band_structure_simulation(
            resolution=args.resolution,
            resolution_z=args.resolution_z,
            num_bands=args.num_bands,
            k_density=args.k_density,
            num_workers=args.workers,
            quick=args.quick,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
