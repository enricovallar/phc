# `phc_hydra`

Hydra experiment configuration, hierarchical output directory resolution, and simulation artifact tracking for the `phc` workspace.

## Features
- **Hierarchical Output Directory SSOT**: Standardizes all simulation outputs to `outputs/<solver>/<sim_type>/<geometry>/<timestamp>/`.
- **SimulationOutputManager**: Manages saving simulation artifacts (plots, data, summary JSON) with manifest tracking.
- **Mandatory GDS Layout Enforcement**: Ensures every electromagnetic simulation run exports its physical GDS layout mask.
- **Programmatic Configuration**: Helpers for composing Hydra configs without CLI overhead.
