"""`phc_hydra`: Hydra experiment configuration, output hierarchy, and artifact management."""

from phc_hydra.config import load_simulation_config, to_plain_dict
from phc_hydra.output import (
    CANONICAL_BAND_PLOT_NAME,
    CANONICAL_EPSILON_PLOT_NAME,
    CANONICAL_GDS_NAME,
    CANONICAL_RESULTS_JSON_NAME,
    SimulationOutputManager,
    find_latest_simulation,
    resolve_simulation_output_dir,
)

__all__ = [
    "CANONICAL_BAND_PLOT_NAME",
    "CANONICAL_EPSILON_PLOT_NAME",
    "CANONICAL_GDS_NAME",
    "CANONICAL_RESULTS_JSON_NAME",
    "SimulationOutputManager",
    "find_latest_simulation",
    "load_simulation_config",
    "resolve_simulation_output_dir",
    "to_plain_dict",
]
