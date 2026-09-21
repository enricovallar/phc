#!/usr/bin/env python3
"""Example demonstrating photonic crystal unit cell generation and visualization.

Generates GDS layouts for canonical unit cells from `phc_layout.components`
and renders them using gdsfactory's `.plot()` method to verify geometric correctness.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from phc_layout.components import UNIT_CELL_DATABASE, get_unit_cell
from phc_layout.lattice import HexagonalLattice, SquareLattice
from phc_layout.tech import LAYER_PHC


def run_unit_cell_plot_pipeline(
    quick: bool = False,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Generates unit cell layouts from phc_layout and renders them using Component.plot().

    Args:
        quick: If True, only generates a minimal subset of 2 unit cells for fast smoke testing.
        output_dir: Directory where generated plot artifacts will be saved.
            Defaults to 'outputs/layout_plots/'.

    Returns:
        Dictionary containing:
            - 'plots': Mapping of cell names to saved image paths.
            - 'output_dir': Path to output directory.
            - 'cells': List of processed unit cell names.
    """
    out_path = (
        Path(output_dir) if output_dir is not None else Path("outputs/layout_plots")
    )
    out_path.mkdir(parents=True, exist_ok=True)

    cells_to_plot = (
        ["c6v_honeycomb", "c4v_lieb"] if quick else list(UNIT_CELL_DATABASE.keys())
    )

    saved_plots: dict[str, str] = {}

    print("\n=======================================================")
    print("PHC Layout Unit Cell Generation & Plotting Pipeline")
    print(f"Output directory: {out_path}")
    print(f"Cells to process: {len(cells_to_plot)}")
    print("=======================================================\n")

    # -------------------------------------------------------------
    # 1. Individual Unit Cell Plots (.plot())
    # -------------------------------------------------------------
    for name in cells_to_plot:
        spec = UNIT_CELL_DATABASE[name]
        pitch = 1.0

        # Generate with Wigner-Seitz cell boundary
        cell_ws = get_unit_cell(
            name=name,
            pitch=pitch,
            boundary_layer=LAYER_PHC.SLAB,
            boundary_type="wigner_seitz",
            wrap_to_cell=True,
        )
        cell_ws.plot(return_fig=True)
        ws_plot_path = out_path / f"{name}_wigner_seitz.png"
        plt.title(f"{name} (Wigner-Seitz boundary)", fontsize=11, color="white")
        plt.savefig(ws_plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        # Generate with Primitive cell boundary
        cell_prim = get_unit_cell(
            name=name,
            pitch=pitch,
            boundary_layer=LAYER_PHC.SLAB,
            boundary_type="primitive",
            wrap_to_cell=False,
        )
        cell_prim.plot(return_fig=True)
        prim_plot_path = out_path / f"{name}_primitive.png"
        plt.title(f"{name} (Primitive boundary)", fontsize=11, color="white")
        plt.savefig(prim_plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        saved_plots[f"{name}_ws"] = str(ws_plot_path)
        saved_plots[f"{name}_prim"] = str(prim_plot_path)
        print(f"  [+] Plotted '{name}' ({spec.point_group}, {spec.lattice_type})")

    # -------------------------------------------------------------
    # 2. Multi-Cell Gallery Overview Plot
    # -------------------------------------------------------------
    if not quick:
        gallery_keys = [
            "c6v_primitive",
            "c6v_honeycomb",
            "c6v_kagome",
            "c6v_snowflake_6d",
            "c4v_primitive",
            "c4v_checkerboard",
            "c4v_lieb",
            "c4v_snowflake_4d",
        ]
        fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=150)
        fig.patch.set_facecolor("#1a1a1a")

        for idx, key in enumerate(gallery_keys):
            ax = axes[idx // 4, idx % 4]
            ax.set_facecolor("black")
            c = get_unit_cell(
                key,
                pitch=1.0,
                boundary_layer=LAYER_PHC.SLAB,
                boundary_type="wigner_seitz" if "c6v" in key else "primitive",
                wrap_to_cell=True,
            )
            # Render into matplotlib axis using .plot(ax=...)
            c.plot(ax=ax, show_labels=False, show_ruler=False)
            ax.set_title(key, fontsize=12, fontweight="bold", color="white", pad=8)
            ax.set_aspect("equal")

        plt.suptitle(
            "Canonical Photonic Crystal Unit Cells (phc_layout Database)",
            fontsize=15,
            fontweight="bold",
            color="white",
            y=0.98,
        )
        plt.tight_layout()
        gallery_path = out_path / "unit_cell_gallery.png"
        plt.savefig(gallery_path, facecolor=fig.get_facecolor(), dpi=150)
        plt.close()
        saved_plots["gallery"] = str(gallery_path)
        print(f"\n  [+] Saved multi-cell gallery: {gallery_path}")

        # -------------------------------------------------------------
        # 3. Periodic Lattice Tiling Verification (3x3 Supercells)
        # -------------------------------------------------------------
        tiling_keys = ["c6v_primitive", "c6v_honeycomb", "c6v_kagome", "c4v_lieb"]
        fig, axes = plt.subplots(2, 2, figsize=(12, 12), dpi=150)
        fig.patch.set_facecolor("#1a1a1a")

        for idx, key in enumerate(tiling_keys):
            ax = axes[idx // 2, idx % 2]
            ax.set_facecolor("black")

            # Create 3x3 tiled lattice in GDS
            import gdsfactory as gf

            supercell = gf.Component(f"supercell_{key}")
            unit = get_unit_cell(key, pitch=1.0, wrap_to_cell=False)

            lat = HexagonalLattice(a=1.0) if "c6v" in key else SquareLattice(a=1.0)

            for u in range(-1, 2):
                for v in range(-1, 2):
                    pos = lat.to_cartesian_2d(u, v)
                    ref = supercell << unit
                    ref.move(pos)

            supercell.plot(ax=ax, show_labels=False, show_ruler=False)
            ax.set_title(
                f"{key} (3x3 Tiled Array)",
                fontsize=13,
                fontweight="bold",
                color="white",
                pad=8,
            )
            ax.set_aspect("equal")

        plt.suptitle(
            "Periodic Lattice Tiling Verification (phc_layout -> GDS)",
            fontsize=15,
            fontweight="bold",
            color="white",
            y=0.98,
        )
        plt.tight_layout()
        tiling_path = out_path / "lattice_tiling_verification.png"
        plt.savefig(tiling_path, facecolor=fig.get_facecolor(), dpi=150)
        plt.close()
        saved_plots["tiling"] = str(tiling_path)
        print(f"  [+] Saved lattice tiling verification: {tiling_path}")

    print(f"\nCompleted! Generated {len(saved_plots)} plot files in {out_path}\n")
    return {
        "plots": saved_plots,
        "output_dir": out_path,
        "cells": cells_to_plot,
    }


if __name__ == "__main__":
    run_unit_cell_plot_pipeline(quick=False)
