# `phc_materials` Guidelines & Rules

## Scope
`phc_materials` is the Single Source of Truth (SSOT) for optical material parameters, refractive indices, anisotropy, and simulator medium translation across the workspace.

## Core Rules

1. **Material Specifications**:
   - `IsotropicMaterial`: Defined by scalar refractive index $n$ (or $\epsilon = n^2$).
   - `AnisotropicMaterial`: Defined by principal indices $[n_x, n_y, n_z]$ (or $[\epsilon_x, \epsilon_y, \epsilon_z]$) and optical axis rotation angle in degrees.
   - All materials must declare standard metadata:
     - `name`: Unique lookup identifier (e.g., `'si'`, `'sio2'`, `'sin'`, `'lnoi_xcut'`).
     - `lumerical_name`: Exact string matching the Ansys Lumerical material database entry (e.g., `'Si (Silicon) - Palik'`, `'SiO2 (Glass) - Palik'`).
     - `description`: Human-readable description including reference wavelength (default $1550\,\text{nm}$).

2. **Registry Interface**:
   - `MaterialRegistry` must implement Python container conventions:
     - `__contains__(self, key: str) -> bool` (`'si' in reg`).
     - `__iter__(self)` (`for key in reg:`).
     - `keys() -> list[str]` and `get(key: str) -> MaterialSpec`.
   - Support loading from external YAML configurations via `MaterialRegistry.from_yaml(path)`.

3. **Simulator Adapters**:
   - **MPB Adapter** (`phc_materials.mpb.to_mpb_medium`):
     - For isotropic materials: `mp.Medium(index=mat.index)` or `mp.Medium(epsilon=mat.epsilon)`.
     - For anisotropic materials: `mp.Medium(epsilon_diag=mp.Vector3(eps_x, eps_y, eps_z))`.
     - Fail fast if `meep` is missing; never return a mock dictionary.
   - **Lumerical Adapter**:
     - Provide mappings to Lumerical material names or `.mdf` (Material Data Format) file generators.

4. **No Direct Layout or Simulation Code**:
   - `phc_materials` must not generate geometries or execute solvers. It is strictly an optical data provider.
