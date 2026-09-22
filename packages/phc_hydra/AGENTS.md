# `phc_hydra` Guidelines & Rules

## Scope
`phc_hydra` is the Single Source of Truth (SSOT) for Hydra configuration coordination, hierarchical simulation output directory resolution, and simulation artifact manifest management across the `phc` workspace.

---

## Core Rules

### 1. Hierarchical Output Directory Standard
All simulation output directories must strictly follow the canonical hierarchy:
```
outputs/<solver>/<sim_type>/<geometry>/<timestamp>/
```
- `<solver>`: Lowercase solver name (e.g., `mpb`, `lumerical`).
- `<sim_type>`: Specific simulation task (e.g., `band_diagram`, `slab_band_diagram`, `cavity_q`, `transmission`).
- `<geometry>`: Canonical geometry identifier (e.g., `c6v_primitive`, `c4v_snowflake_4e`, `l3_cavity`).
- `<timestamp>`: Formatted timestamp `YYYY-MM-DD_HH-MM-SS` (or `multirun_<timestamp>/<job_id>` for parameter sweeps).
- Never invent ad-hoc un-namespaced output paths (such as `outputs/demo_slab_3d` or `outputs/temp`).

### 2. Mandatory GDS Layout Rule
**Every simulation run across all physics domains MUST export its physical layout GDS file.**
- The GDS file must be saved into the simulation output directory before the run is considered complete.
- Canonically named `unit_cell.gds` (or `layout.gds`).
- `SimulationOutputManager.save_results_json(...)` enforces this rule by default, raising an explicit `RuntimeError` if no GDS layout exists in the output directory.

### 3. Separation of Concerns & Clean Dependencies
- `phc_hydra` must **never** invoke electromagnetic solvers (e.g. `meep`, `mpb`, `lumapi`).
- `phc_hydra` must **never** define photonic crystal geometry algorithms (leave layout to `phc_layout`).
- It depends strictly on `phc_utils` for safe file writing (`export_gds`), `hydra-core`, `omegaconf`, `matplotlib`, `numpy`, and the Python standard library.

### 4. Canonical Artifact Names for Band Diagram Simulations
For band structure simulations, the output directory must contain:
- `unit_cell.gds`: Physical layout mask.
- `band_structure.png`: Dispersion band diagram (dots per $k$-point, light line, band gap highlights).
- `epsilon_map.png`: Dielectric permittivity distribution map (2D slice or 3D slab dual-plane cross-section).
- `simulation_results.json`: Complete structured JSON summary containing geometry, simulation parameters, band gaps, and file manifest.

Other simulation types (`sim_type`) may include additional simulation-specific artifacts (e.g., `transmission.png`, `s_parameters.s2p`, `field_profile.png`), but must always include the GDS file and `simulation_results.json`.

### 5. Programmatic & CLI Duality
- Workflows must support both `@hydra.main` CLI execution and headless programmatic Python execution.
- Programmatic callers can pass an explicit `output_dir` (e.g. pytest `tmp_path`), which `SimulationOutputManager` respects while maintaining identical internal artifact structure.
