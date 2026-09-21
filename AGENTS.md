# Antigravity Workspace Guidelines for `phc`

## 1. Workspace Overview & Architecture

`phc` is a modular Python workspace designed for photonic crystal (PhC) and integrated photonics research, layout generation, and multi-solver electromagnetic simulation.

The workspace follows a strict monorepo architecture with distinct packages located under `packages/`:
- **`phc_layout`**: Photonic crystal unit cell and supercell layout generation using GDSFactory.
- **`phc_materials`**: Single Source of Truth (SSOT) for optical material specifications (isotropic and anisotropic refractive indices) and adapters for simulation tools.
- **`phc_mpb`**: General MPB (MIT Photonic Bands) solver interface for band structures, gap analysis, and dielectric distributions. Used for band structure calculations and optimization.
- **`phc_lumerical`**: Planned Ansys Lumerical FDTD / MODE simulation automation and GDS import scripts.

### Modularity & Separation of Concerns
- Never leak solver logic (e.g. MPB or Lumerical calls) into `phc_layout`.
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

---

## 3. Hydra Configuration Standards

The workspace uses **Hydra** (`hydra-core` and `omegaconf`) for experiment configuration, parametric sweeps, and reproducibility:
- **Location**: All presets reside in `configs/`:
  - `configs/config.yaml`: Root configuration declaring default composition (`defaults: ...`).
  - `configs/geometry/`: Geometric parameters (pitch, radius ratios, lattice type).
  - `configs/simulation/`: Solver settings (resolution, num_bands, k-point density, polarization).
  - `configs/stack/`: Technology `LayerStack` contracts (thicknesses, $z$-positions, material keys).
- **Decoupled Materials**: In Hydra YAML files, specify materials using **string keys** (`material: "si"` or `slab_material: "si"`), **never** hardcoded numeric refractive indices. Resolvers must query `phc_materials.get_material(key)`.
- **Pipeline Interface**: Multi-step simulation runners should accept `DictConfig` configurations and support command-line overrides (e.g. `simulation.resolution=64`).
- **Outputs**: Hydra run directories belong in `outputs/` and are gitignored.

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
