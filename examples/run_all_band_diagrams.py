#!/usr/bin/env python3
"""Batch execution pipeline for all canonical photonic crystal unit cell band diagrams.

Automates:
1. Iterating through all 16 canonical unit cells in the database (or specified subsets).
2. Loading geometry presets from configs/geometry/*.yaml.
3. Running end-to-end band structure calculations using MPB.
4. Detecting and logging complete omnidirectional band gaps.
5. Exporting GDS layouts, epsilon permittivity maps, and band diagram plots.
6. Generating an aggregated summary report (JSON and formatted ASCII table).
"""

import argparse
import json
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Ensure repository root is in sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from omegaconf import DictConfig, OmegaConf
from phc_layout.components import UNIT_CELL_DATABASE, list_unit_cells

from examples.simulate_unit_cell import run_unit_cell_simulation


def run_all_band_diagrams_pipeline(
    quick: bool = False,
    output_dir: Path | str | None = None,
    geometries: Sequence[str] | None = None,
    solver: str = "mpb",
    polarization: str = "te",
    resolution: int | None = None,
    num_bands: int | None = None,
    k_density: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Runs band structure simulation pipelines for a batch of photonic crystal unit cells.

    Args:
        quick: If True, uses low resolution and minimal k-points on a 2-cell subset
            for rapid execution (<2s total) in automated smoke testing.
        output_dir: Directory where batch results and individual cell directories will be stored.
            If None, defaults to 'outputs/batch_band_diagrams/<timestamp>/'.
        geometries: Optional list of unit cell names to simulate. If None and quick is False,
            all 16 canonical unit cells in UNIT_CELL_DATABASE are simulated.
        solver: Solver engine ('mpb').
        polarization: Polarization state ('te' or 'tm').
        resolution: MPB spatial mesh grid resolution per unit pitch. If None, uses preset default.
        num_bands: Number of eigenbands to compute. If None, uses preset default.
        k_density: Number of k-points interpolated along each high-symmetry path segment.
        verbose: If True, prints formatted console logs and ASCII progress summary.

    Returns:
        Dictionary containing:
            - 'summary': Aggregated metadata including gap metrics for each geometry.
            - 'results': Mapping of geometry name to individual simulation result dictionaries.
            - 'output_dir': Path to the root batch output directory.
            - 'summary_file': Path to the saved batch_summary.json report.

    Raises:
        ValueError: If an unknown geometry name is specified.
    """
    configs_geom_dir = Path(__file__).resolve().parent.parent / "configs" / "geometry"

    # 1. Resolve list of geometries to simulate
    if geometries is not None:
        cells_to_run = list(geometries)
        for name in cells_to_run:
            if name not in UNIT_CELL_DATABASE:
                raise ValueError(
                    f"Unknown geometry '{name}'. Available database unit cells: {list_unit_cells()}."
                )
    elif quick:
        # Fast smoke-test subset: one hexagonal, one square
        cells_to_run = ["c6v_primitive", "c4v_primitive"]
    else:
        # Complete database suite (all 16 unit cells)
        cells_to_run = list(UNIT_CELL_DATABASE.keys())

    # 2. Resolve root output directory
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
    if output_dir is not None:
        batch_out_path = Path(output_dir)
    else:
        batch_out_path = Path("outputs") / "batch_band_diagrams" / timestamp

    batch_out_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n" + "=" * 80)
        print("  PHOTONIC CRYSTAL BATCH BAND DIAGRAM SIMULATION PIPELINE")
        print("=" * 80)
        print(f"  Geometries ({len(cells_to_run)}): {', '.join(cells_to_run)}")
        print(f"  Solver:           {solver} ({polarization.upper()} polarization)")
        print(
            f"  Mode:             {'QUICK SMOKE-TEST' if quick else 'FULL RESOLUTION'}"
        )
        print(f"  Output Directory: {batch_out_path}")
        print("=" * 80 + "\n")

    batch_start_time = time.time()
    results_map: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []

    # 3. Iterate and simulate each unit cell
    for idx, geom_name in enumerate(cells_to_run, start=1):
        spec = UNIT_CELL_DATABASE[geom_name]
        cell_start_time = time.time()

        if verbose:
            print(
                f"[{idx}/{len(cells_to_run)}] Simulating '{geom_name}' ({spec.point_group}, {spec.lattice_type})..."
            )

        # Load YAML geometry configuration if available, otherwise construct from spec
        geom_yaml = configs_geom_dir / f"{geom_name}.yaml"
        if geom_yaml.is_file():
            loaded_cfg = OmegaConf.load(geom_yaml)
            geom_dict = OmegaConf.to_container(loaded_cfg, resolve=True)
            if not isinstance(geom_dict, dict):
                geom_dict = {"name": geom_name}
        else:
            geom_dict = {
                "name": geom_name,
                "point_group": spec.point_group,
                "lattice_type": spec.lattice_type,
                "pitch_um": 0.5,
                "background_material": "si",
                "etch_material": "air",
                "dimension": "2D",
                "slab_thickness_um": 0.22,
            }

        # Build simulation dictionary
        sim_dict: dict[str, Any] = {
            "solver": solver,
            "sim_type": "band_diagram",
            "polarization": polarization,
            "quick": quick,
        }
        if resolution is not None:
            sim_dict["resolution"] = resolution
        if num_bands is not None:
            sim_dict["num_bands"] = num_bands
        if k_density is not None:
            sim_dict["k_density"] = k_density

        full_cfg = DictConfig(
            {
                "geometry": geom_dict,
                "simulation": sim_dict,
                "verbose": False,
            }
        )

        cell_out_dir = batch_out_path / geom_name
        cell_out_dir.mkdir(parents=True, exist_ok=True)

        sim_out = run_unit_cell_simulation(
            cfg=full_cfg,
            quick=quick,
            output_dir=cell_out_dir,
        )

        cell_elapsed = time.time() - cell_start_time
        results_map[geom_name] = sim_out
        gaps = sim_out.get("gaps", [])

        # Format gap summary for table
        gap_desc_list = []
        for g in gaps:
            bands = g.get("bands", ["?", "?"])
            pct = g.get("gap_pct", 0.0)
            gap_desc_list.append(f"Bands {bands[0]}-{bands[1]} ({pct:.1f}%)")
        gap_str = ", ".join(gap_desc_list) if gap_desc_list else "None"

        summary_rows.append(
            {
                "name": geom_name,
                "point_group": spec.point_group,
                "lattice_type": spec.lattice_type,
                "gaps": gaps,
                "gap_summary": gap_str,
                "elapsed_s": round(cell_elapsed, 2),
                "output_dir": str(cell_out_dir),
                "files": sim_out.get("files", {}),
            }
        )

        if verbose:
            print(f"      -> Complete in {cell_elapsed:.2f}s | Gaps: {gap_str}")

    total_elapsed = time.time() - batch_start_time

    # 4. Save batch summary JSON report
    batch_summary_doc = {
        "timestamp": timestamp,
        "quick": quick,
        "solver": solver,
        "polarization": polarization,
        "total_geometries": len(cells_to_run),
        "total_elapsed_seconds": round(total_elapsed, 2),
        "geometries": summary_rows,
    }

    summary_json_path = batch_out_path / "batch_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(batch_summary_doc, f, indent=2)

    # 5. Print ASCII Summary Table
    if verbose:
        print("\n" + "=" * 96)
        print("                               BATCH SIMULATION SUMMARY")
        print("=" * 96)
        print(
            f"{'Geometry':<24} {'Lattice':<12} {'PG':<6} {'Elapsed':<10} {'Omnidirectional Band Gaps':<40}"
        )
        print("-" * 96)
        for row in summary_rows:
            print(
                f"{row['name']:<24} {row['lattice_type']:<12} {row['point_group']:<6} "
                f"{row['elapsed_s']:>5.2f}s     {row['gap_summary']:<40}"
            )
        print("=" * 96)
        print(
            f"Total Wall Time: {total_elapsed:.2f}s | Summary JSON saved to: {summary_json_path}\n"
        )

    return {
        "summary": batch_summary_doc,
        "results": results_map,
        "output_dir": batch_out_path,
        "summary_file": summary_json_path,
    }


def main() -> None:
    """CLI entrypoint for batch unit cell band diagram simulations."""
    parser = argparse.ArgumentParser(
        description="Run batch band diagram simulations across photonic crystal unit cells in MPB."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run fast smoke-test mode with low resolution on 2 cells (<2s total).",
    )
    parser.add_argument(
        "--geometries",
        nargs="+",
        default=None,
        help="Optional list of specific geometry names to simulate (e.g. c6v_honeycomb c6v_kagome).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory override (defaults to outputs/batch_band_diagrams/<timestamp>).",
    )
    parser.add_argument(
        "--polarization",
        type=str,
        default="te",
        choices=["te", "tm"],
        help="Simulation polarization mode ('te' or 'tm', default: 'te').",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Mesh grid resolution override per pitch a.",
    )
    parser.add_argument(
        "--num-bands",
        type=int,
        default=None,
        help="Number of eigenbands to compute override.",
    )
    parser.add_argument(
        "--k-density",
        type=int,
        default=None,
        help="Number of k-points interpolated per path segment override.",
    )

    args = parser.parse_args()

    run_all_band_diagrams_pipeline(
        quick=args.quick,
        output_dir=args.output_dir,
        geometries=args.geometries,
        polarization=args.polarization,
        resolution=args.resolution,
        num_bands=args.num_bands,
        k_density=args.k_density,
    )


if __name__ == "__main__":
    main()
