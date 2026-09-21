# Material Databases & Anisotropy Architecture

This document serves as the persistent specification and implementation guide for handling optical materials, material databases, and optical anisotropy across layout tools and simulation solvers (MPB, Meep, and Ansys Lumerical).

---

## 1. Overview & Architectural Goals

1. **Decoupling**: Geometry definitions (`gdsfactory.Component` $\to$ `.gds`) do not hardcode material models. They only reference material keys (e.g. `"si"`, `"lnoi_xcut"`, `"sio2"`).
2. **Multi-Solver Consistency**: The same material key resolves correctly to:
   - **MPB**: scalar or diagonal tensor dielectric constant $\boldsymbol{\varepsilon} = (n_x^2, n_y^2, n_z^2)$ at the simulation frequency.
   - **Meep**: `mp.Medium` with diagonal anisotropy and optional dispersion/loss.
   - **Lumerical (FDTD/MODE)**: Dispersive multi-coefficient models or anisotropic entries in Lumerical's Material Database.
3. **Reproducibility (Method A)**: All custom and anisotropic materials for Lumerical are tracked directly inside the repository as version-controlled `.mdf` files and loaded programmatically, eliminating dependency on machine-local GUI settings.

---

## 2. Directory Structure

```text
phc/
├── materials/
│   ├── materials.yaml            # Master declarative specification (human-readable SSOT)
│   └── custom_materials.mdf      # Lumerical Material Database file (version-controlled)
├── packages/phc_layout/
│   └── src/phc_layout/
│       └── materials.py          # Python registry & solver conversion adapters
├── SSOT.md                       # Core geometry & simulation contract
└── MATERIALS.md                  # This document
```

---

## 3. Handling Anisotropic Materials

### Physical Representation
Optical anisotropy arises in birefringent materials where the permittivity tensor $\boldsymbol{\varepsilon}$ is non-scalar. Along the crystal principal axes:
$$\boldsymbol{\varepsilon} = \begin{pmatrix} \varepsilon_x & 0 & 0 \\ 0 & \varepsilon_y & 0 \\ 0 & 0 & \varepsilon_z \end{pmatrix} = \begin{pmatrix} n_x^2 & 0 & 0 \\ 0 & n_y^2 & 0 \\ 0 & 0 & n_z^2 \end{pmatrix}$$

#### Common Crystal Cuts (e.g., Lithium Niobate $\text{LiNbO}_3$ at 1550 nm: $n_o \approx 2.286$, $n_e \approx 2.203$)
| Crystal Platform | Optical Axis ($c$-axis) | $n_x$ | $n_y$ | $n_z$ | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **$X$-cut LNOI** | Along $x$ (in-plane) | $n_e$ (2.203) | $n_o$ (2.286) | $n_o$ (2.286) | High in-plane electro-optic $r_{33}$ coupling |
| **$Y$-cut LNOI** | Along $y$ (in-plane) | $n_o$ (2.286) | $n_e$ (2.203) | $n_o$ (2.286) | In-plane optical axis |
| **$Z$-cut LNOI** | Along $z$ (out-of-plane) | $n_o$ (2.286) | $n_o$ (2.286) | $n_e$ (2.203) | Symmetric in $(x, y)$ plane |

---

### Solver Implementation: MPB / Meep

Meep and MPB represent anisotropy through the `epsilon_diag` vector in `mp.Medium`:

```python
import meep as mp


def get_mpb_medium(material_cfg: dict) -> mp.Medium:
    """Translates material config entry into a Meep/MPB Medium."""
    if material_cfg.get("type") == "anisotropic":
        nx, ny, nz = material_cfg["indices"]
        medium = mp.Medium(epsilon_diag=mp.Vector3(nx**2, ny**2, nz**2))

        # Handle arbitrary in-plane rotation (e.g. angled waveguides)
        rotation_deg = material_cfg.get("rotation_deg", 0.0)
        if rotation_deg != 0.0:
            import numpy as np

            medium = medium.rotate(mp.Vector3(0, 0, 1), np.deg2rad(rotation_deg))
        return medium
    else:
        index = material_cfg["index"]
        return mp.Medium(index=index)
```

---

### Solver Implementation: Ansys Lumerical

#### Method A: Version-Controlled `.mdf` Database (Mandatory Standard)
Never rely on the local Lumerical GUI's *"Save as default"* feature, as it saves only to the local user's profile (`~/.config/Lumerical/`) and fails silently on Linux HPC compute nodes, Docker containers, or teammates' computers.

**Implementation Workflow**:
1. Keep the compiled `.mdf` database in the repository at `materials/custom_materials.mdf`.
2. When launching any Lumerical session (FDTD or MODE) via `lumapi`, import the database immediately:
   ```python
   import os
   import lumapi


   def init_lumerical_session(project_root: str) -> lumapi.FDTD:
       fdtd = lumapi.FDTD(hide=True)
       mdf_path = os.path.join(project_root, "materials", "custom_materials.mdf")
       if os.path.isfile(mdf_path):
           fdtd.importmaterialdb(mdf_path)
       else:
           raise FileNotFoundError(f"Material database not found at {mdf_path}")
       return fdtd
   ```

#### Script-Based Anisotropic Material Generation (Generating / Updating `.mdf`)
To programmatically generate or update custom materials inside the database without the GUI:
```python
def add_anisotropic_dielectric(fdtd, name: str, nx: float, ny: float, nz: float):
    """Creates a diagonal anisotropic dielectric material in Lumerical."""
    if not fdtd.materialexists(name):
        mat = fdtd.addmaterial("Dielectric")
        fdtd.setmaterial(mat, "name", name)

    # Configure diagonal anisotropy
    fdtd.setmaterial(name, "Anisotropy", 1)  # 1 = diagonal anisotropy
    fdtd.setmaterial(name, "Permittivity", [nx**2, ny**2, nz**2])


def export_workspace_materials(fdtd, export_path: str):
    """Saves the current database to a version-controlled .mdf file."""
    fdtd.exportmaterialdb(export_path)
```

#### Rotated Crystal Orientations in Lumerical
When the optical axis does not align with the Cartesian grid axes $(x, y, z)$, use Lumerical's **Permittivity Rotation** grid attribute:
```python
def assign_crystal_rotation(fdtd, structure_name: str, euler_z1_deg: float):
    """Assigns an in-plane crystal rotation to a geometric object."""
    grid_attr = fdtd.addgridattribute("permittivity rotation")
    fdtd.setnamed(grid_attr, "name", f"{structure_name}_rot")
    fdtd.setnamed(
        f"{structure_name}_rot", "rotation 1", euler_z1_deg
    )  # Z-axis rotation

    # Link grid attribute to the target structure
    fdtd.setnamed(structure_name, "grid attribute name", f"{structure_name}_rot")
```

---

## 4. Master Declarative Schema (`materials/materials.yaml`)

This YAML file serves as the SSOT for all project materials:

```yaml
# materials/materials.yaml
version: "1.0"

materials:
  # Isotropic standard materials
  air:
    type: isotropic
    index: 1.0
    lumerical_name: "etch"

  si:
    type: isotropic
    index: 3.48
    lumerical_name: "Si (Silicon) - Palik"

  sio2:
    type: isotropic
    index: 1.444
    lumerical_name: "SiO2 (Glass) - Palik"

  sin:
    type: isotropic
    index: 2.00
    lumerical_name: "Si3N4 - Philip"

  # Anisotropic platforms
  lnoi_xcut:
    type: anisotropic
    description: "X-cut Lithium Niobate (optical axis along X)"
    indices: [2.203, 2.286, 2.286]   # [nx=ne, ny=no, nz=no] at 1550nm
    lumerical_name: "LN_Xcut"

  lnoi_zcut:
    type: anisotropic
    description: "Z-cut Lithium Niobate (optical axis along Z)"
    indices: [2.286, 2.286, 2.203]   # [nx=no, ny=no, nz=ne] at 1550nm
    lumerical_name: "LN_Zcut"

  batio3:
    type: anisotropic
    description: "Barium Titanate thin film"
    indices: [2.35, 2.35, 2.30]
    lumerical_name: "BaTiO3_Custom"
```

---

## 5. Summary Checklist

- [ ] Every new material is recorded first in `materials/materials.yaml`.
- [ ] For anisotropic materials, specify principal indices $[n_x, n_y, n_z]$ along with crystal cut context.
- [ ] Any custom Lumerical model is exported to `materials/custom_materials.mdf` and committed to git.
- [ ] Simulation runners always call `fdtd.importmaterialdb(...)` upon initialization.
- [ ] Solvers query the centralized registry rather than defining raw permittivity values in script files.
