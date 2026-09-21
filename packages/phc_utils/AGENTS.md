# `phc_utils` Guidelines & Rules

## Scope
`phc_utils` provides shared, cross-package interfuse utilities, safe I/O operations, unit conversions, and helper tools used across `phc_layout`, `phc_materials`, `phc_mpb`, and `phc_lumerical`.

## Core Rules

1. **Lightweight & Modular**:
   - `phc_utils` must remain lightweight and dependency-minimal (Standard library and `numpy`).
   - Do not introduce heavy dependencies like `meep`, `lumapi`, or GUI packages into `phc_utils`.

2. **Safe I/O**:
   - Always ensure robust file and directory handling (e.g. `export_gds` handles atomic directory creation and clean error handling).

3. **Physics & Unit Conversions**:
   - Maintain exact, bidirectionally tested conversions between normalized dimensionless frequencies ($\tilde{\omega} = \omega a / 2\pi c = a / \lambda$) and physical wavelengths ($\lambda_0$) or frequencies ($f$).

4. **Docstrings & Types**:
   - Every function and class must include complete type annotations and Google-style docstrings (`Args:`, `Returns:`, `Raises:`).
