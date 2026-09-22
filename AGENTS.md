# Antigravity Workspace Guidelines for `phc`

## 1. Workspace Overview & Architecture

`phc` is a modular Python workspace designed for photonic crystal (PhC) and integrated photonics research, layout generation, and multi-solver electromagnetic simulation.

The workspace follows a strict monorepo architecture with distinct packages located under `packages/`:
- **`phc_layout`**: Photonic crystal unit cell and supercell layout generation using GDSFactory.
- **`phc_materials`**: Single Source of Truth (SSOT) for optical material specifications (isotropic and anisotropic refractive indices) and adapters for simulation tools.
- **`phc_mpb`**: General MPB (MIT Photonic Bands) solver interface for band structures, gap analysis, and dielectric distributions. Used for band structure calculations and optimization.
- **`phc_lumerical`**: Planned Ansys Lumerical FDTD / MODE simulation automation and GDS import scripts.
- **`phc_hydra`**: Single Source of Truth (SSOT) for Hydra configuration coordination, hierarchical simulation output directory resolution, and simulation artifact manifest management.
- **`phc_utils`**: Shared cross-package interfuse utilities, safe I/O operations, unit conversions, and path validators.

### Modularity & Separation of Concerns
- Never leak solver logic (e.g. MPB or Lumerical calls) into `phc_layout` or `phc_hydra`.
- Never hardcode material refractive indices in layout or simulation scripts; always query `phc_materials`.
- Keep packages independently testable.
- Test packages without needing a running solver.

---

## 2. Non-Negotiable Core Rules

### A. Strictly No Mock or Synthetic Fallbacks
- Never fall back to synthetic sinusoids, mock bands, or placeholder data when an import, tool, or calculation fails.
- If a dependency, tool, or calculation fails, **fail fast and raise an explicit, descriptive exception** immediately so issues can be diagnosed and fixed at the root.

### B. Single Direct Code Paths (No Fallback Chains)
- Do not chain imports or provide nested `try...except ImportError` fallback ladders across geometry or simulation engines.
- Standardize on direct, canonical libraries (e.g., `klayout.db` for layout parsing, `meep.mpb` for band solving).

### C. Generalized Physics & Solvers
- Solver packages like `phc_mpb` must not be hardcoded to a single lattice or dimension.
- Solvers must support:
  - Both **hexagonal** and **square** lattices.
  - Both **2D periodic** systems and **3D slab / membrane** structures.
  - Both **TE** and **TM** (or TE-like / TM-like) polarizations.
  - Arbitrary unit cell definitions.

### D. Suppressed / Reduced MPB Solver Verbosity by Default
- **Always suppress MPB and Meep solver output verbosity by default** across all workflows, benchmarks, examples, and optimization routines.
- Meep and MPB C/C++ engines print voluminous per-iteration trace lines and band outputs directly to C-level `stdout`. Workflows must suppress this solver chatter by default using `mp.verbosity(0)` and `phc_utils.silence_c_stdout()` (or `verbose: bool = False`), ensuring clean terminal output and progress reporting unless explicit debugging verbosity is requested.

---

## 3. Hydra Configuration & Standard Simulation Output File Tree

The workspace uses **Hydra** (`hydra-core` and `omegaconf`) for experiment configuration, parametric sweeps, and reproducibility, coordinated by **`phc_hydra`**:
- **Location**: All presets reside in `configs/`:
  - `configs/config.yaml`: Root configuration declaring default composition (`defaults: ...`).
  - `configs/geometry/`: Geometric parameters (pitch, radius ratios, lattice type).
  - `configs/simulation/`: Solver settings (resolution, num_bands, k-point density, polarization).
  - `configs/stack/`: Technology `LayerStack` contracts (thicknesses, $z$-positions, material keys).
- **Decoupled Materials**: In Hydra YAML files, specify materials using **string keys** (`material: "si"` or `slab_material: "si"`), **never** hardcoded numeric refractive indices. Resolvers must query `phc_materials.get_material(key)`.
- **Standard Output Directory Hierarchy**:
  All simulation outputs across the workspace must strictly follow the canonical hierarchy:
  ```
  outputs/<solver>/<sim_type>/<geometry>/<timestamp>/
  ```
  (or `multirun_<timestamp>/<job_id>` for parameter sweeps).
  Never invent ad-hoc un-namespaced output paths (e.g. `outputs/demo_slab_3d` or `outputs/temp`).
- **Mandatory GDS Layout Rule**:
  **Every physical simulation run across all physics domains MUST export its physical layout GDS file into its output directory** (canonically `unit_cell.gds` or `layout.gds`). A simulation run without an exported GDS layout is invalid. `SimulationOutputManager.save_results_json` enforces this rule by default.
- **Canonical Artifact Names for Band Diagram Simulations**:
  - `unit_cell.gds`: Exported physical layout mask.
  - `band_structure.png`: Dispersion band diagram (dots per $k$-point, light line, band gap highlights).
  - `epsilon_map.png`: Dielectric permittivity distribution map (2D slice or 3D slab dual-plane cross-sections, `interpolation="none"`).
  - `simulation_results.json`: Complete structured JSON summary containing geometry, simulation parameters, band gaps, and file manifest.
  Other simulation types (`sim_type`) save simulation-specific artifacts (e.g., `transmission.png`, `q_factor.png`, `s_parameters.s2p`, `fields.npz`), but must always include the GDS layout file and `simulation_results.json`.
- **Pipeline Interface**: Multi-step simulation runners should accept `DictConfig` configurations and support command-line overrides (e.g. `simulation.resolution=64`). All runners must manage output directories and artifacts through `phc_hydra.SimulationOutputManager`.


---

## 4. Integration Examples & Interconnection Testing

Cross-module workflows demonstrate and test how packages interoperate (`phc_layout` $\to$ `phc_materials` $\to$ `phc_mpb` $\to$ `outputs/`):
- **Location**: Human-facing runnable workflows belong in `examples/` (e.g., `examples/demo_hex_2d_mpb.py`).
- **Dual-Mode Pattern**: Every script in `examples/` must provide a parameterized entrypoint with a `quick: bool = False` argument:
  ```python
  def run_*_pipeline(..., quick: bool = False, output_dir: Path | str | None = None) -> dict: ...
  ```
  - `quick=False`: High physical resolution, plots saved to disk, full analysis.
  - `quick=True`: Low resolution (e.g. `resolution=16`, `num_bands=4`, `k_density=2`), executes in $<2$ seconds.
- **Mandatory CLI Entrypoints & Documentation**:
  Every script in `examples/` must be directly executable from the command line:
  1. **CLI Argument Parser**: Must implement a clean CLI parser using `argparse` in `main()` supporting flags for common overrides (e.g. `--quick`, `--resolution`, `--num-bands`, `--output-dir`, `--workers`, etc.) with informative help strings.
  2. **Header Usage Documentation**: The module-level docstring at the top of the file must document **Command-Line Usage** and **CLI Options**, giving copy-pasteable terminal commands (e.g., standard execution, quick smoke test, custom resolution/iteration overrides) so users immediately know how to invoke the script from the shell.
- **Automated Smoke Testing**:
  - All examples must have an automated test under `tests/integration/` (tagged with `@pytest.mark.integration`).
  - Integration tests execute each example with `quick=True` and assert that expected outputs/data structures are returned without errors.
- **Contract Enforcement**: Whenever modifying public APIs in `phc_layout`, `phc_materials`, or `phc_mpb`, the agent **must** execute `pytest tests/integration/` to verify that all cross-package interconnections remain intact.

---

## 5. Tooling & Environment Standards

- **Python Version**: `>=3.11, <3.12`.
- **Virtual Environment**: `.venv/bin/python`.
  - Meep / MPB is linked from the conda environment via site-packages `.pth`.
  - Workspace packages are linked via `.venv/lib/python3.11/site-packages/phc_workspace.pth`.
- **Linting & Formatting**:
  - Linter: `.venv/bin/ruff check --fix --unsafe-fixes .`
  - Formatter: `.venv/bin/ruff format .`
  - All modified code must pass `ruff check .` with **0 errors and 0 warnings**.
- **Testing**:
  - Test Runner: `.venv/bin/pytest`
  - Pytest is configured with `addopts = "--import-mode=importlib"` and workspace `pythonpath`.
  - Fast unit tests: `pytest -m "not integration"`
  - Integration tests: `pytest -m integration`
  - Always run `pytest` before concluding code changes.

---

## 6. Coding & Docstring Conventions

### A. Mandatory Google-Style Docstrings
All public modules, classes, functions, and methods must have comprehensive, structured Google-style docstrings. Never write single-line or vague docstrings for non-trivial APIs. Every docstring must include:
1. **Summary**: Concise one-line imperative description of what the function/class does.
2. **Extended Explanation** (if non-trivial): Details on algorithms, coordinate transformations, or physical assumptions.
3. **`Args:`**: Every argument must be documented with:
   - Name and expected type/structure.
   - Physical meaning, coordinate convention, and units (e.g. micrometers $\mu\text{m}$, normalized frequency $\tilde{\omega}$).
   - Default value behavior.
4. **`Returns:`**: Explicit description of the return type and data structure, including dictionary keys, array dimensions/shapes, and units.
5. **`Raises:`**: Specific exception classes (e.g., `ValueError`, `FileNotFoundError`, `RuntimeError`) and the exact conditions triggering them. Never catch or raise blind `Exception`.

### B. Shared Interfuse Utilities (`phc_utils`)
- Common cross-package operations (such as safe GDS file export, unit conversions $\tilde{\omega} \leftrightarrow \lambda_0$, path validation, and shared data serializers) must live in `packages/phc_utils`.
- Never duplicate GDS file writing boilerplate (`if exists: unlink()`, try/except ladders) across simulation scripts; use `phc_utils.export_gds(component, filepath, overwrite=True)`.

### C. Type Annotations & Units
- **Type Annotations**: All public classes, methods, and functions must have complete type annotations (`typing.Any`, `typing.Literal`, `typing.Sequence`, etc.).
- **Units**: 
  - Layout dimensions: micrometers ($\mu\text{m}$).
  - Frequencies in MPB: normalized dimensionless units ($\tilde{\omega} = \omega a / 2\pi c = a / \lambda$).

### D. 3D Photonic Crystal Slab Permittivity Visualization
For 3D slab geometries, dielectric permittivity ($\varepsilon(\mathbf{r})$) plotting must generate a single unified figure containing two subplots side by side:
1. **In-plane $xy$ mid-plane slice** ($z = z_{\text{mid}}$ / $z = 0$): Displays the periodic unit cell or lattice cross-section in the slab plane.
2. **Vertical $xz$ plane cross-section** ($y = y_{\text{mid}}$): Displays the vertical slab profile, slab thickness, air/substrate cladding, and etch hole vertical cross-section.
- **Vertical Periodicity Standard ($z$-periodicity = 1)**: For 3D slab permittivity profiling, vertical periodicity along $z$ must strictly remain 1 (`periods_z=1`). The supercell height already includes the physical membrane and top/bottom cladding buffer; repeating $z$ would artificially portray an unphysical multilayer stack.
Never plot only a single isolated 2D slice for 3D slab structures.

### E. Band Diagram & Permittivity Visualization Standards
- **Band Diagrams as Discrete Dots**: Band diagrams must always display calculated eigenfrequencies as discrete dots (one dot per $k$-point, e.g. `marker="o"`, `linestyle="none"`). Never connect computed eigenfrequencies with continuous lines, as discrete eigenvalues across the Brillouin zone should not imply artificial continuity across band anti-crossings or between coarse $k$-points. Reference thresholds (such as the light line $\omega = c k_{\parallel}$) remain analytical dashed lines.
- **Dielectric Permittivity Maps Without Interpolation**: Permittivity grid plots (`plot_epsilon`) must use `interpolation="none"` by default so that the discrete numerical Yee/MPB grid and physical material boundaries are visualized faithfully without artificial pixel smoothing (such as bilinear or bicubic filtering).

### F. High-Symmetry K-Path Convention
- **Γ as the Second High-Symmetry Point**: By default, the zone-center $\Gamma$ point must be the **second** vertex in the irreducible Brillouin zone k-path, not the first. This ensures that band diagrams begin from a zone-boundary point, pass through $\Gamma$, and return to a zone-boundary point:
  - **Hexagonal** ($C_{6v}$): $M \to \Gamma \to K \to M$
  - **Square** ($C_{4v}$): $X \to \Gamma \to M \to X$
- Never start a k-path at $\Gamma$ by default.
