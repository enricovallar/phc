# Running Photonic Crystal Unit Cell Simulations & Layouts

This folder contains end-to-end runnable pipelines for generating GDS layouts, querying optical material databases, running electromagnetic band solvers (MPB), and analyzing photonic band structures.

---

## 1. Quick Start Guide

All simulation pipelines are integrated with **Hydra** for parameter configuration and command-line overrides.

### Activate Virtual Environment
```bash
# In the repository root
source .venv/bin/activate
```

### Run Default Simulation (`c6v_primitive`)
```bash
python examples/simulate_unit_cell.py
```

### Fast Smoke-Test Mode (<2 seconds)
Append `simulation.quick=true` for rapid iteration at reduced numerical resolution:
```bash
python examples/simulate_unit_cell.py simulation.quick=true
```

---

## 2. Simulating Unit Cells from the Database

You can simulate any canonical unit cell simply by overriding `geometry=<name>`.

### Hexagonal ($C_{6v}$) Unit Cells

| Geometry Name | Point Group | Lattice | Description |
| :--- | :---: | :---: | :--- |
| **`c6v_primitive`** | $C_{6v}$ | Hexagonal | Standard triangular lattice unit cell with single hole at Wyckoff position $1a$. |
| **`c6v_honeycomb`** | $C_{6v}$ | Hexagonal | Bipartite graphene-like lattice ($2b$ holes) exhibiting Dirac cones at $K$. |
| **`c6v_kagome`** | $C_{6v}$ | Hexagonal | Tri-partite lattice ($3c$ holes) exhibiting flat bands and wide band gaps. |
| **`c6v_snowflake_6d`** | $C_{6v}$ | Hexagonal | Core-satellite snowflake with center hole at $1a$ and 6 satellite holes along lattice axes at $6d$. Used in valley-Hall topological photonics. |
| **`c6v_snowflake_6e`** | $C_{6v}$ | Hexagonal | Core-satellite snowflake with center hole at $1a$ and 6 satellite holes along diagonal axes at $6e$. |
| **`c6v_painter_snowflake`** | $C_{6v}$ | Hexagonal | Safavi-Naeini & Painter (2010) continuous optomechanical snowflake with 3 intersecting rotated rectangular slots, opening large simultaneous phononic & photonic band gaps. |
| **`c6v_ring_6d`** | $C_{6v}$ | Hexagonal | Ring of 6 satellite holes oriented along the principal lattice axes at $6d$. |
| **`c6v_ring_6e`** | $C_{6v}$ | Hexagonal | Ring of 6 satellite holes oriented along diagonal reflection axes at $6e$. |

### Square ($C_{4v}$) Unit Cells

| Geometry Name | Point Group | Lattice | Description |
| :--- | :---: | :---: | :--- |
| **`c4v_primitive`** | $C_{4v}$ | Square | Standard square lattice unit cell with single hole at $1a$. |
| **`c4v_checkerboard`** | $C_{4v}$ | Square | Bipartite checkerboard square lattice with holes at $1a$ (origin) and $1b$ (center). |
| **`c4v_lieb`** | $C_{4v}$ | Square | Lieb lattice with edge midpoint holes at $2c$, featuring flat bands and Dirac crossings. |
| **`c4v_snowflake_4d`** | $C_{4v}$ | Square | Core-satellite square snowflake with central hole at $1a$ and 4 diagonal satellite holes at $4d$. |
| **`c4v_snowflake_4e`** | $C_{4v}$ | Square | Core-cross square snowflake with central hole at $1a$ and 4 Cartesian axis holes at $4e$. |
| **`c4v_ring_4d`** | $C_{4v}$ | Square | Ring of 4 holes along diagonal reflection axes at $4d$. |
| **`c4v_cross_4e`** | $C_{4v}$ | Square | Cross of 4 holes along Cartesian axes at $4e$. |
| **`c4v_edges_4f`** | $C_{4v}$ | Square | 4 holes located along square edges at $4f$. |

---

## 3. Example Commands

### Run Snowflake Simulations
```bash
# 1. Valley-Hall core-satellite snowflake (C6v)
python examples/simulate_unit_cell.py geometry=c6v_snowflake_6d

# 2. Painter optomechanical continuous snowflake (41% TE bandgap!)
python examples/simulate_unit_cell.py geometry=c6v_painter_snowflake

# 3. Square core-satellite snowflake (C4v)
python examples/simulate_unit_cell.py geometry=c4v_snowflake_4d
```

### Run Honeycomb & Kagome Simulations
```bash
# Honeycomb (Dirac cones at K)
python examples/simulate_unit_cell.py geometry=c6v_honeycomb

# Kagome (Complete TE bandgaps & flat bands)
python examples/simulate_unit_cell.py geometry=c6v_kagome
```

### Run Lieb Lattice Simulation
```bash
python examples/simulate_unit_cell.py geometry=c4v_lieb
```

---

## 4. Customizing Parameters via CLI

You can override any parameter in the YAML configurations directly on the command line:

### Physical Geometry Overrides
```bash
# Change pitch and radius
python examples/simulate_unit_cell.py geometry=c6v_primitive geometry.pitch_um=0.6 geometry.radius_ratio=0.30

# Adjust Painter snowflake arm length and width
python examples/simulate_unit_cell.py geometry=c6v_painter_snowflake geometry.radius_ratio=0.42 geometry.width_ratio=0.12
```

### Numerical Solver Overrides
```bash
# High-resolution simulation (resolution=64, 12 bands, 32 k-points per segment)
python examples/simulate_unit_cell.py geometry=c6v_kagome simulation.resolution=64 simulation.num_bands=12 simulation.k_density=32

# Switch polarization to TM (or both)
python examples/simulate_unit_cell.py geometry=c4v_primitive simulation.polarization=tm
```

### Optical Material Overrides (SSOT)
All materials are queried from `phc_materials` by key:
```bash
# GaAs slab with air holes
python examples/simulate_unit_cell.py geometry.background_material=gaas geometry.etch_material=air

# Silicon slab with SiO2 filled holes
python examples/simulate_unit_cell.py geometry.background_material=si geometry.etch_material=sio2
```

---

## 5. Hydra Multi-Run Parameter Sweeps

Hydra natively supports multi-run sweeps with the `-m` (or `--multirun`) flag:

### Sweep Across Different Unit Cells
```bash
python examples/simulate_unit_cell.py -m geometry=c6v_primitive,c6v_honeycomb,c6v_kagome,c6v_painter_snowflake simulation.quick=true
```

### Sweep Hole Radius
```bash
python examples/simulate_unit_cell.py -m geometry=c6v_primitive geometry.radius_ratio=0.20,0.25,0.30,0.35
```

---

## 6. Batch Execution of All Band Diagrams (`run_all_band_diagrams.py`)

To simulate band structures across all unit cells in an automated batch run, use [`examples/run_all_band_diagrams.py`](run_all_band_diagrams.py):

```bash
# Run batch simulation across all 16 canonical unit cells
python examples/run_all_band_diagrams.py

# Fast batch smoke test (runs 2 representative cells in ~1 second)
python examples/run_all_band_diagrams.py --quick

# Simulate a chosen subset of geometries
python examples/run_all_band_diagrams.py --geometries c6v_honeycomb c6v_kagome c6v_painter_snowflake c4v_lieb

# TM polarization batch
python examples/run_all_band_diagrams.py --polarization tm

# Specify a custom output directory
python examples/run_all_band_diagrams.py --output-dir outputs/my_batch_run/
```

### Batch Output Report:
The batch runner outputs a formatted summary table in your terminal:
```text
================================================================================================
                               BATCH SIMULATION SUMMARY
================================================================================================
Geometry                 Lattice      PG     Elapsed    Omnidirectional Band Gaps               
------------------------------------------------------------------------------------------------
c6v_primitive            hexagonal    C6v     0.71s     Bands 1-2 (16.5%)                       
c6v_honeycomb            hexagonal    C6v     0.85s     None (Dirac crossings at K)
c6v_kagome               hexagonal    C6v     0.82s     Bands 4-5 (7.1%), Bands 6-7 (7.4%)
c6v_painter_snowflake    hexagonal    C6v     0.96s     Bands 1-2 (41.0%)
...
================================================================================================
```
In addition, it saves an aggregated `batch_summary.json` containing complete metrics and links to all generated GDS layouts, band plots, and epsilon permittivity maps.

---

## 7. Output Files & Directory Structure

All artifacts are saved into a clean, hierarchical directory structure:
```text
outputs/
└── <solver>/               # e.g., mpb/
    └── <sim_type>/         # e.g., band_diagram/
        └── <geometry>/     # e.g., c6v_painter_snowflake/
            └── <timestamp>/
                ├── unit_cell.gds           # GDSII layout of the unit cell
                ├── band_structure.png      # Publication band diagram with gap shading
                ├── epsilon_map.png         # Rectified real-space dielectric permittivity map
                └── simulation_results.json # Machine-readable frequencies and gap analysis
```

### Output Artifacts Explained:
1. **`unit_cell.gds`**: Physical CAD file containing the exact polygon shapes and boundaries.
2. **`band_structure.png`**: High-symmetry band diagram ($\Gamma \to M \to K \to \Gamma$ or $\Gamma \to X \to M \to \Gamma$), displaying the light line and shading all omnidirectional band gaps with their fractional gap sizes ($\Delta\omega / \omega_0$).
3. **`epsilon_map.png`**: Real-space map of the dielectric constant $\varepsilon(\mathbf{r})$ discretized on the solver grid.
4. **`simulation_results.json`**: JSON record containing eigenfrequencies at each $k$-point, detected band gaps (lower bound, upper bound, gap size), and file paths.

---

## 8. Visualizing Layouts Before Simulating

To visually inspect the unit cell GDS geometries without running the electromagnetic solver, run:
```bash
# Fast mode: plots canonical honeycomb and Lieb
python examples/plot_unit_cells.py quick=true

# Full mode: plots all 16 database unit cells, generates a multi-cell gallery,
# and verifies 3x3 periodic lattice tiling in GDS:
python examples/plot_unit_cells.py
```
Generated plot artifacts will be stored in `outputs/layout_plots/`.

