#!/usr/bin/env python3
"""End-to-end Demonstration: 2D Hexagonal Photonic Crystal in MPB.

Demonstrates the Single Source of Truth (SSOT) architecture:
1. Generates a 2D hexagonal PhC unit cell with gdsfactory (phc_layout).
2. Exports the physical GDSII file (outputs/phc_hex_unit_cell.gds).
3. Queries optical material properties from phc_materials (Silicon, n = 3.48).
4. Converts the GDS file directly into MPB geometric prisms (phc_mpb).
5. Runs MPB ModeSolver across the irreducible Brillouin zone (Γ -> M -> K -> Γ).
6. Plots and saves the photonic band structure (outputs/band_diagram_hex_te.png).
"""

from pathlib import Path
from typing import Any

from phc_layout.components import phc_hexagonal_unit_cell
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
from phc_utils import export_gds


def run_hex_2d_pipeline(
    pitch: float = 0.45,
    radius_ratio: float = 0.25,
    resolution: int = 32,
    num_bands: int = 8,
    k_density: int = 16,
    quick: bool = False,
    output_dir: Path | str | None = "outputs",
) -> dict[str, Any]:
    """Runs the end-to-end 2D hexagonal PhC pipeline connecting layout, materials, and MPB.

    Args:
        pitch: Lattice pitch in micrometers (default: 0.45 um).
        radius_ratio: Hole radius as a fraction of pitch (r/a, default: 0.25).
        resolution: MPB mesh grid resolution per unit distance a.
        num_bands: Number of eigenbands to compute.
        k_density: Interpolation density between high-symmetry k-points.
        quick: If True, uses low-resolution parameters for fast integration smoke testing.
        output_dir: Directory to save generated GDS and plot artifacts (or None to skip disk writes).

    Returns:
        Dictionary containing MPB simulation results, ModeSolver instance, and extracted band data.
    """
    if quick:
        resolution = 8
        k_density = 2
        num_bands = 4

    out_path = Path(output_dir) if output_dir is not None else None
    if out_path:
        out_path.mkdir(parents=True, exist_ok=True)
        gds_file = out_path / "phc_hex_unit_cell.gds"
        plot_file = out_path / "band_diagram_hex_te.png"
        eps_plot_file = out_path / "epsilon_hex_2d.png"
    else:
        gds_file = None
        plot_file = None
        eps_plot_file = None

    radius = radius_ratio * pitch

    # Step 1: Generate 2D Layout with GDSfactory
    print(
        f"\n[1/6] Generating 2D Hexagonal Unit Cell layout (pitch={pitch * 1e3:.0f}nm, r={radius * 1e3:.1f}nm)..."
    )
    unit_cell = phc_hexagonal_unit_cell(pitch=pitch, radius=radius)
    gds_source: Any = unit_cell
    if gds_file:
        gds_source = export_gds(unit_cell, gds_file, overwrite=True)
        print(f"      -> Exported GDSII binary: {gds_source}")

    # Step 2: Query Material Registry
    print("\n[2/6] Querying Material Registry...")
    mat = get_material("si")
    print(
        f"      -> Selected: {mat.description} (index n = {mat.index}, eps = {mat.epsilon:.2f})"
    )
    print(f"      -> Lumerical Database Name: '{mat.lumerical_name}'")

    # Step 3: Setup MPB Simulation Domain & Convert GDS Geometry
    print("\n[3/6] Setting up MPB Hexagonal Lattice & Converting GDS to Prisms...")
    lattice = create_lattice(
        lattice_type="hexagonal",
        pitch=pitch,
        dimension="2D",
    )
    geometry = gds_to_mpb_geometry(
        gds_source=gds_source,
        pitch=pitch,
        dimension="2D",
        etch_material="air",
    )
    print(f"      -> Extracted {len(geometry)} geometric prism(s) from GDS.")

    k_points, labels, indices = get_high_symmetry_kpath(
        lattice_type="hexagonal",
        k_density=k_density,
    )
    print(
        f"      -> Brillouin Zone Path: {' -> '.join(labels)} ({len(k_points)} total k-points)"
    )

    # Step 4: Execute MPB Band Solver
    print(
        f"\n[4/6] Initializing MPB ModeSolver (resolution={resolution}, num_bands={num_bands})..."
    )
    ms = create_mode_solver(
        geometry_lattice=lattice,
        geometry=geometry,
        k_points=k_points,
        default_material=mat,
        resolution=resolution,
        num_bands=num_bands,
    )

    print("      -> Running TE Polarization (in-plane E)...")
    results = run_band_solver(
        ms=ms,
        polarization="te",
        dimension="2D",
    )

    # Step 5: Extract & Plot MPB Dielectric Permittivity (epsilon)
    print("\n[5/6] Extracting & Plotting MPB Dielectric Permittivity (epsilon)...")
    eps_grid = get_epsilon_grid(
        ms=ms,
        rectify=True,
        periodicity=3,
        resolution=32,
    )
    if eps_plot_file:
        plot_epsilon(
            epsilon=eps_grid,
            title=r"MPB Hexagonal PhC Permittivity $\varepsilon(\mathbf{r})$ - 3 Periods",
            cmap="viridis",
            colorbar=True,
            save_path=eps_plot_file,
        )
        print(f"      -> Saved dielectric epsilon plot: {eps_plot_file}")

    # Step 6: Plotting & Band Gap Analysis
    print("\n[6/6] Analyzing Results & Generating Band Diagram Plot...")
    freqs = results.get("freqs", {}).get("te")
    if freqs is None or len(freqs) == 0:
        raise RuntimeError("MPB failed to compute TE band frequencies.")

    print(f"      -> Computed {freqs.shape[0]} k-points for {freqs.shape[1]} bands.")

    # Check for bandgap between band 1 and band 2
    gap = ms.retrieve_gap(1)
    if gap > 0:
        print(f"      -> Bandgap (Band 1-2): {gap:.2f}%")

    if plot_file:
        plot_band_structure(
            results=results,
            node_labels=labels,
            node_indices=indices,
            title="2D Hexagonal PhC (Air Holes in Si, r/a = 0.25) - TE Modes",
            save_path=plot_file,
        )
        print(f"      -> Saved band diagram: {plot_file}")

    return {
        "results": results,
        "ms": ms,
        "labels": labels,
        "indices": indices,
        "gap_band1_2": gap,
        "eps_grid": eps_grid,
    }


def main() -> None:
    print("=" * 70)
    print(" 2D Hexagonal Photonic Crystal Band Structure Demo (SSOT Pipeline) ")
    print("=" * 70)

    run_hex_2d_pipeline(
        pitch=0.45,
        radius_ratio=0.25,
        resolution=32,
        num_bands=8,
        k_density=16,
        quick=False,
        output_dir="outputs",
    )

    print("\n" + "=" * 70)
    print(" Demo Run Completed Successfully! ")
    print("=" * 70)


if __name__ == "__main__":
    main()
