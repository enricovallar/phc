# Solved Problems & Root-Cause Analyses

This directory serves as a persistent, tracked knowledge base of solved bugs, numerical anomalies, physical edge-cases, and their permanent resolutions within the `phc` workspace.

## Index of Solved Problems

| ID | Date | Component | Summary |
| :--- | :--- | :--- | :--- |
| [001](001_mpb_prism_infinite_height_snowflake.md) | 2026-09-21 | `phc_mpb` | Floating-point cancellation in MPB 2D Prism extrusion causing inverted permittivity maps for snowflake unit cells |
| [002](002_slab_3d_polarization_fraction_cladding_dilution.md) | 2026-09-22 | `phc_mpb` | 3D slab polarization fraction cladding dilution causing TM-like modes to render as blue/white instead of red |
| [003](003_mpb_degeneracy_multiplet_irrep_resolution.md) | 2026-09-25 | `phc_mpb` | Degenerate mode mixing and single-state reflection projection causing spurious irrep labeling and split doublets/triplets |
| [004](004_asymmetric_slab_subspace_irrep_mixing.md) | 2026-10-01 | `phc_mpb`, `phc_optimization` | Substrate asymmetry and eigensolver mode mixing causing false irrep penalty in accidental triplet optimization |

