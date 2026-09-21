# Photonic Crystal & Integrated Photonics Workspace

This repository combines **Miniconda** (for native non-PyPI dependencies like MIT Photonic Bands `mpb`) and **`uv`** (for fast Python package resolution, workspaces, and lockfile versioning).

## Architecture

* **Conda (`environment.yml`)**: Provides the base Python 3.11 runtime and compiled C/Fortran tools (`mpb`).
* **`uv` (`pyproject.toml`)**: Manages project packages, dependencies (`gdsfactory`, `ansys-lumerical-core`), and multi-package workspaces under `packages/`.

---

## Getting Started

### 1. Install Miniconda (if not already installed)

If you do not have Miniconda installed:
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

### 2. Create and Activate the Conda Environment

```bash
conda env create -f environment.yml
conda activate phc
```

### 3. Sync Packages & Lock Dependencies with `uv`

Once activated, `uv` will synchronize all packages in the workspace and generate `uv.lock`:
```bash
uv sync
```

---

## Adding Packages to `packages/`

To add a new sub-package (for example, `packages/phc-layout`):
1. Create a directory: `packages/phc-layout`
2. Add a `pyproject.toml` inside it:
   ```toml
   [project]
   name = "phc-layout"
   version = "0.1.0"
   dependencies = [
       "gdsfactory>=8.0.0",
   ]
   ```
3. Run `uv sync` from the repository root to automatically link it across the workspace.

---

## Running Commands and Tests

Always run python scripts or tests using `uv run` to ensure dependencies and workspace packages are properly loaded:
```bash
uv run pytest
```

---

## Configuration Management with Hydra

Configuration presets live in the `configs/` directory:
```text
configs/
├── config.yaml               # Default base configuration
├── simulation/               # Solver configurations (mpb, lumerical)
└── geometry/                 # Photonic crystal geometries
```

### Examples with Hydra & `uv run`:

* **Run default MPB simulation**:
  ```bash
  uv run python run_sim.py
  ```
* **Switch solver to Lumerical from CLI**:
  ```bash
  uv run python run_sim.py simulation=lumerical
  ```
* **Override geometry parameters**:
  ```bash
  uv run python run_sim.py geometry.radius_over_a=0.30 geometry.lattice_constant_nm=450
  ```
* **Multi-run parameter sweep** (Hydra sweeps):
  ```bash
  uv run python run_sim.py -m geometry.radius_over_a=0.20,0.25,0.30
  ```

