# Hydra Configuration Directory (`configs/`)

This directory contains hierarchical Hydra configurations for photonic crystal simulation experiments, geometries, and solver settings.

---

## Directory Structure

```text
configs/
├── config.yaml               # Root entrypoint config declaring defaults and output path
├── geometry/                 # Geometry presets for all 16 canonical unit cells
│   ├── c6v_primitive.yaml
│   ├── c6v_honeycomb.yaml
│   ├── c6v_kagome.yaml
│   ├── c6v_ring_6d.yaml
│   ├── c6v_ring_6e.yaml
│   ├── c6v_snowflake_6d.yaml
│   ├── c6v_snowflake_6e.yaml
│   ├── c6v_painter_snowflake.yaml
│   ├── c4v_primitive.yaml
│   ├── c4v_checkerboard.yaml
│   ├── c4v_lieb.yaml
│   ├── c4v_ring_4d.yaml
│   ├── c4v_cross_4e.yaml
│   ├── c4v_edges_4f.yaml
│   ├── c4v_snowflake_4d.yaml
│   ├── c4v_snowflake_4e.yaml
│   └── custom_wyckoff.yaml
├── simulation/               # Solver configurations
│   ├── mpb.yaml              # MPB band solver settings
│   └── lumerical.yaml        # Ansys Lumerical settings (planned)
└── stack/                    # Physical layer stack technology contracts
```

---

## Running Simulations with Configs

To run simulations using these configuration presets:
```bash
# Default configuration
python examples/simulate_unit_cell.py

# Switch geometry preset
python examples/simulate_unit_cell.py geometry=c6v_painter_snowflake

# Override individual parameters
python examples/simulate_unit_cell.py geometry=c6v_kagome simulation.resolution=64

# Parameter sweep across geometries
python examples/simulate_unit_cell.py -m geometry=c6v_primitive,c6v_honeycomb,c6v_kagome,c6v_painter_snowflake
```

For complete usage instructions and tables of all supported unit cells, see [`examples/README.md`](../examples/README.md).
