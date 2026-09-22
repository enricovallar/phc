"""Simulation output directory resolution and artifact management.

Provides the Single Source of Truth (SSOT) for workspace simulation output hierarchies:
    outputs/<solver>/<sim_type>/<geometry>/<timestamp>/
and enforces the mandatory inclusion of physical GDS layout masks across all simulation runs.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
from phc_utils import export_gds

CANONICAL_GDS_NAME = "unit_cell.gds"
CANONICAL_BAND_PLOT_NAME = "band_structure.png"
CANONICAL_EPSILON_PLOT_NAME = "epsilon_map.png"
CANONICAL_RESULTS_JSON_NAME = "simulation_results.json"


def resolve_simulation_output_dir(
    solver: str = "mpb",
    sim_type: str = "band_diagram",
    geometry: str = "c6v_primitive",
    timestamp: str | None = None,
    base_dir: Path | str = "outputs",
    override_dir: Path | str | None = None,
    cfg: DictConfig | dict[str, Any] | None = None,
) -> Path:
    """Resolves the canonical hierarchical simulation output directory.

    Evaluates directory targets in priority order:
    1. Explicit `override_dir` (e.g. pytest temporary directory `tmp_path`).
    2. Config `output_dir` property if defined in `cfg`.
    3. Canonical hierarchy: `<base_dir>/<solver>/<sim_type>/<geometry>/<timestamp>/`.

    Parent directories are created automatically and atomically.

    Args:
        solver: Simulation engine name (e.g., 'mpb', 'lumerical').
        sim_type: Simulation task type (e.g., 'band_diagram', 'slab_band_diagram', 'cavity_q').
        geometry: Geometry or unit cell identifier (e.g., 'c6v_primitive', 'l3_cavity').
        timestamp: Optional formatted timestamp string. Defaults to UTC '%Y-%m-%d_%H-%M-%S'.
        base_dir: Root output directory path (default: 'outputs').
        override_dir: Direct path override bypassing the standard hierarchy.
        cfg: Optional Hydra DictConfig or configuration dictionary.

    Returns:
        Resolved absolute or workspace-relative Path to the created output directory.

    Raises:
        ValueError: If solver, sim_type, or geometry strings are empty.
    """
    if not solver.strip():
        raise ValueError("solver cannot be empty.")
    if not sim_type.strip():
        raise ValueError("sim_type cannot be empty.")
    if not geometry.strip():
        raise ValueError("geometry cannot be empty.")

    if override_dir is not None:
        out_path = Path(override_dir)
    elif cfg is not None:
        cfg_dict = (
            OmegaConf.to_container(cfg, resolve=True)
            if isinstance(cfg, DictConfig)
            else cfg
        )
        if isinstance(cfg_dict, dict) and cfg_dict.get("output_dir"):
            out_path = Path(cfg_dict["output_dir"])
        else:
            ts = timestamp or datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
            out_path = Path(base_dir) / solver / sim_type / geometry / ts
    else:
        ts = timestamp or datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
        out_path = Path(base_dir) / solver / sim_type / geometry / ts

    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def find_latest_simulation(
    solver: str | None = None,
    sim_type: str | None = None,
    geometry: str | None = None,
    base_dir: Path | str = "outputs",
) -> Path:
    """Finds the most recent simulation run directory containing a simulation_results.json.

    Searches hierarchical simulation outputs in `base_dir` and resolves the most recent
    run directory matching the optional filters.

    Args:
        solver: Optional solver name filter (e.g., 'mpb', 'lumerical').
        sim_type: Optional simulation task type filter (e.g., 'slab_mode_parity', 'band_diagram').
        geometry: Optional geometry identifier filter (e.g., 'c6v_primitive').
        base_dir: Base directory containing simulation runs (default: 'outputs').

    Returns:
        Path to the most recent simulation directory containing simulation_results.json.

    Raises:
        FileNotFoundError: If base_dir does not exist or no matching simulation run is found.
    """
    root = Path(base_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Base output directory does not exist: {root}")

    # If all three specific levels are given, search directly in that subdirectory
    if solver and sim_type and geometry:
        target_dir = root / solver / sim_type / geometry
        if target_dir.is_dir():
            candidates = [
                d
                for d in target_dir.iterdir()
                if d.is_dir() and (d / CANONICAL_RESULTS_JSON_NAME).is_file()
            ]
            if candidates:
                candidates.sort(key=lambda d: d.name)
                return candidates[-1]

    # Otherwise recursively search for all simulation_results.json files
    json_files = list(root.rglob(CANONICAL_RESULTS_JSON_NAME))
    matched_dirs: list[Path] = []
    for jf in json_files:
        p = jf.parent
        try:
            rel_parts = p.relative_to(root).parts
        except ValueError:
            rel_parts = p.parts

        if solver and solver not in rel_parts:
            continue
        if sim_type and sim_type not in rel_parts:
            continue
        if geometry and geometry not in rel_parts:
            continue
        matched_dirs.append(p)

    if not matched_dirs:
        filter_str = f"solver={solver}, sim_type={sim_type}, geometry={geometry}"
        raise FileNotFoundError(
            f"No matching simulation runs found under '{root}' matching ({filter_str})."
        )

    # Sort candidates by directory name (timestamp) and file modification time
    matched_dirs.sort(
        key=lambda d: (d.name, (d / CANONICAL_RESULTS_JSON_NAME).stat().st_mtime)
    )
    return matched_dirs[-1]


class SimulationOutputManager:
    """Manages artifact file creation, saving, and manifest tracking for a simulation run.

    Enforces workspace output standards:
    1. Standard directory tree: outputs/<solver>/<sim_type>/<geometry>/<timestamp>/
    2. Mandatory GDS Layout Rule: Every physical simulation run MUST export its GDS layout mask.
    3. Structured JSON reporting with complete file manifests.

    Attributes:
        output_dir: Resolved Path where all artifacts are written.
        solver: Solver engine name (e.g. 'mpb', 'lumerical').
        sim_type: Simulation category (e.g. 'band_diagram', 'slab_band_diagram').
        geometry_name: Geometry identifier.
        timestamp: Timestamp of the simulation run.
        files: Internal mapping of artifact keys to resolved file Paths.
    """

    def __init__(
        self,
        output_dir: Path | str,
        solver: str = "mpb",
        sim_type: str = "band_diagram",
        geometry_name: str = "c6v_primitive",
        timestamp: str | None = None,
    ) -> None:
        """Initializes the output manager and ensures output directory exists.

        Args:
            output_dir: Root directory for this simulation's artifacts.
            solver: Electromagnetic solver name (default: 'mpb').
            sim_type: Simulation task type (default: 'band_diagram').
            geometry_name: Geometry identifier (default: 'c6v_primitive').
            timestamp: Optional timestamp string. Defaults to current UTC time.
        """
        self.output_dir = Path(output_dir)
        self.solver = solver
        self.sim_type = sim_type
        self.geometry_name = geometry_name
        self.timestamp = timestamp or datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
        self.files: dict[str, Path] = {}

        self.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(
        cls,
        cfg: DictConfig | dict[str, Any],
        output_dir: Path | str | None = None,
        base_dir: Path | str = "outputs",
    ) -> "SimulationOutputManager":
        """Factory creating a SimulationOutputManager from a configuration object.

        Args:
            cfg: Hydra DictConfig or dict containing 'geometry' and 'simulation' sections.
            output_dir: Optional explicit override directory.
            base_dir: Root directory if output_dir is not specified (default: 'outputs').

        Returns:
            Configured SimulationOutputManager instance with output directory created.
        """
        cfg_dict = (
            OmegaConf.to_container(cfg, resolve=True)
            if isinstance(cfg, DictConfig)
            else dict(cfg)
        )
        geom_cfg = cfg_dict.get("geometry", {})
        sim_cfg = cfg_dict.get("simulation", {})

        geom_name = geom_cfg.get("name", "c6v_primitive")
        solver = sim_cfg.get("solver", "mpb")
        sim_type = sim_cfg.get("sim_type", "band_diagram")

        resolved_dir = resolve_simulation_output_dir(
            solver=solver,
            sim_type=sim_type,
            geometry=geom_name,
            base_dir=base_dir,
            override_dir=output_dir,
            cfg=cfg,
        )

        return cls(
            output_dir=resolved_dir,
            solver=solver,
            sim_type=sim_type,
            geometry_name=geom_name,
        )

    def save_gds(
        self,
        component: Any,
        filename: str = CANONICAL_GDS_NAME,
        artifact_key: str = "gds",
    ) -> Path:
        """Exports layout geometry to GDSII in the simulation output directory.

        Satisfies the workspace Mandatory GDS Layout Rule.

        Args:
            component: Layout component (GDSFactory Component, gdstk.Cell, or Path to source GDS).
            filename: Target GDS filename within output_dir (default: 'unit_cell.gds').
            artifact_key: Key under which the path is registered in the file manifest (default: 'gds').

        Returns:
            Resolved Path to the saved GDS file.

        Raises:
            RuntimeError: If GDS file creation fails.
        """
        target_path = self.output_dir / filename
        export_gds(component, target_path, overwrite=True)
        self.files[artifact_key] = target_path
        return target_path

    def save_figure(
        self,
        fig: plt.Figure,
        artifact_key: str,
        filename: str | None = None,
        dpi: int = 150,
        close: bool = False,
    ) -> Path:
        """Saves a Matplotlib figure into the simulation output directory.

        Args:
            fig: Matplotlib Figure instance to save.
            artifact_key: Manifest key identifying this plot (e.g. 'band_structure_plot').
            filename: Target image filename. If None, derived as f"{artifact_key}.png".
            dpi: Dots per inch image resolution (default: 150).
            close: If True, closes the figure via plt.close(fig) after saving.

        Returns:
            Resolved Path to the saved figure file.
        """
        if filename is None:
            filename = f"{artifact_key}.png"
        target_path = self.output_dir / filename
        target_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target_path, dpi=dpi)
        if close:
            plt.close(fig)
        self.files[artifact_key] = target_path
        return target_path

    def save_data(
        self,
        data: Any,
        artifact_key: str,
        filename: str,
    ) -> Path:
        """Saves arbitrary raw simulation data (NumPy array, text, or binary) to output directory.

        Args:
            data: Data object to save. NumPy arrays are saved with np.save/savez;
                strings are written as UTF-8 text; bytes are written in binary mode.
            artifact_key: Manifest key identifying this data artifact.
            filename: Target filename within output_dir.

        Returns:
            Resolved Path to the saved data file.

        Raises:
            TypeError: If data type is not supported for automatic serialization.
        """
        target_path = self.output_dir / filename
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(data, np.ndarray):
            if filename.endswith(".npz"):
                np.savez_compressed(target_path, data=data)
            else:
                np.save(target_path, data)
        elif isinstance(data, str):
            target_path.write_text(data, encoding="utf-8")
        elif isinstance(data, bytes):
            target_path.write_bytes(data)
        else:
            raise TypeError(
                f"Unsupported data type '{type(data).__name__}' for save_data. "
                "Provide np.ndarray, str, or bytes."
            )

        self.files[artifact_key] = target_path
        return target_path

    def has_gds(self) -> bool:
        """Checks if a GDS layout file has been saved or exists in the output directory.

        Returns:
            True if a GDS layout is registered in self.files or present in output_dir.
        """
        if "gds" in self.files and self.files["gds"].is_file():
            return True
        gds_files = list(self.output_dir.glob("*.gds"))
        return len(gds_files) > 0

    def save_results_json(
        self,
        geometry_cfg: dict[str, Any] | None = None,
        simulation_cfg: dict[str, Any] | None = None,
        band_gaps: list[dict[str, Any]] | None = None,
        extra_data: dict[str, Any] | None = None,
        filename: str = CANONICAL_RESULTS_JSON_NAME,
        validate_gds: bool = True,
    ) -> Path:
        """Writes the standardized simulation summary report in JSON format.

        Enforces the workspace Mandatory GDS Layout Rule before writing.

        Args:
            geometry_cfg: Geometric parameters dictionary.
            simulation_cfg: Solver settings dictionary.
            band_gaps: Optional list of identified band gap dictionaries.
            extra_data: Optional dictionary for solver-specific metrics (speedup, Q-factors, etc.).
            filename: Output JSON filename (default: 'simulation_results.json').
            validate_gds: If True, raises RuntimeError if no GDS file is found in output_dir.

        Returns:
            Resolved Path to the written JSON summary file.

        Raises:
            RuntimeError: If validate_gds is True and no GDS layout is present in output_dir.
        """
        if validate_gds and not self.has_gds():
            raise RuntimeError(
                f"Mandatory GDS layout missing in simulation output directory: {self.output_dir}. "
                "Workspace rules require every simulation to include its physical GDS layout mask. "
                "Call output_manager.save_gds(...) prior to saving results."
            )

        target_path = self.output_dir / filename

        # Ensure gds file is in manifest if not explicitly registered
        if "gds" not in self.files:
            gds_files = list(self.output_dir.glob("*.gds"))
            if gds_files:
                self.files["gds"] = gds_files[0]

        summary_payload = {
            "geometry": geometry_cfg or {},
            "simulation": simulation_cfg or {},
            "band_gaps": band_gaps or [],
            "files": {k: str(v) for k, v in self.files.items()},
            "metadata": {
                "solver": self.solver,
                "sim_type": self.sim_type,
                "geometry_name": self.geometry_name,
                "timestamp": self.timestamp,
                "created_at": datetime.now(UTC).isoformat(),
            },
        }
        if extra_data:
            summary_payload["extra_data"] = extra_data

        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)

        self.files["results_json"] = target_path
        return target_path

    @property
    def manifest(self) -> dict[str, str]:
        """Returns a string mapping of all registered artifact file paths."""
        return {k: str(v) for k, v in self.files.items()}
