# phc-materials

Centralized optical material registry and solver adapters for the Photonic Crystal Workspace.

## Features
- **Isotropic & Anisotropic Material Models**:
  - Scalar index $n$ and diagonal permittivity tensors $\boldsymbol{\varepsilon} = \text{diag}(n_x^2, n_y^2, n_z^2)$.
  - Crystal rotation angles (in-plane).
- **MPB Adapter**:
  - Automatically translates material specifications into `meep.Medium(epsilon_diag=...)` with `.rotate()`.
- **Lumerical Method A Persistence**:
  - Script utilities to export and import `.mdf` material database files (`custom_materials.mdf`).
