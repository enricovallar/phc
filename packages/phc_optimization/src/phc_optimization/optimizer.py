"""Bayesian Optimization Engine for Photonic Crystal and Integrated Photonics Design.

Coordinates GDSFactory parametric layout generation, dielectric connectivity validation,
MPB electromagnetic eigensolving, and Gaussian Process surrogate active learning.
"""

import inspect
import json
import time
import warnings
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Literal

import gdsfactory as gf
import meep as mp
import numpy as np

# Suppress verbose C-level eigensolver chatter and repetitive sampling warnings
mp.verbosity(0)
warnings.filterwarnings("ignore", category=UserWarning, module="skopt")
warnings.filterwarnings("ignore", message=".*balance properties of Sobol.*")
from phc_hydra import SimulationOutputManager
from phc_mpb import (
    create_lattice,
    create_mode_solver,
    gds_to_mpb_geometry,
    get_epsilon_grid,
    get_high_symmetry_kpath,
    plot_band_structure,
    plot_epsilon,
    run_band_solver,
)
from phc_utils import export_gds, silence_c_stdout
from skopt import Optimizer
from skopt.space import Integer, Real

from phc_optimization.connectivity import check_slab_connectivity
from phc_optimization.locus import (
    evaluate_locus_group_velocities,
    export_loci_to_json,
    export_locus_to_csv,
    extract_optimal_loci,
    refine_locus_points,
)
from phc_optimization.objectives import BaseObjective, get_objective
from phc_optimization.plotting import (
    plot_bo_convergence,
    plot_bo_surrogate_map,
    plot_locus_profile,
)
from phc_optimization.surrogate import fit_clean_surrogate
from phc_optimization.types import OptimizationRecord, OptimizationResult, ParameterSpec


def _extract_solver_data(
    solver_results: dict[str, Any],
    pol_key: str,
    num_bands: int,
) -> dict[str, Any]:
    """Extracts serializable physics data from solver results for per-evaluation logging.

    Args:
        solver_results: Dictionary returned by run_band_solver containing frequencies,
            symmetries, group velocities, and k-point information.
        pol_key: Polarization key to look up in solver results (e.g. 'te', 'tm').
        num_bands: Number of bands solved.

    Returns:
        Dictionary with serializable physics data including k-points, frequencies,
        symmetry characters, irreps, and group velocities per band.
    """
    data: dict[str, Any] = {"polarization": pol_key, "num_bands": num_bands}

    # Frequencies per k-point
    freqs_dict = solver_results.get("freqs", {})
    actual_key = pol_key
    if actual_key not in freqs_dict:
        for alias in ("te", "tm", "te_like", "tm_like", "all"):
            if alias in freqs_dict:
                actual_key = alias
                break

    if actual_key in freqs_dict:
        freq_arr = freqs_dict[actual_key]
        bands_data: list[dict[str, Any]] = []
        for ki, k_freqs in enumerate(freq_arr):
            k_entry: dict[str, Any] = {
                "k_index": ki,
                "k1": 0.0,
                "k2": 0.0,
                "k3": 0.0,
                "kmag_over_2pi": 0.0,
                "freqs": [float(f) for f in k_freqs],
            }
            bands_data.append(k_entry)
        data["k_points"] = bands_data

    # Symmetry data (per band at Gamma)
    symmetries = solver_results.get("symmetries", {}).get(actual_key, [])
    if symmetries:
        sym_records = []
        for sym_rec in symmetries:
            sr: dict[str, Any] = {
                "band": sym_rec.get("band"),
                "freq": sym_rec.get("freq"),
                "irrep": sym_rec.get("irrep"),
                "confidence": sym_rec.get("confidence"),
                "point_group": sym_rec.get("point_group"),
            }
            # Serialize complex characters to real parts
            raw_chars = sym_rec.get("characters", {})
            sr["characters"] = {
                k: round(float(v.real), 6) if isinstance(v, complex) else float(v)
                for k, v in raw_chars.items()
            }
            sr["projections"] = sym_rec.get("projections", {})
            sym_records.append(sr)
        data["symmetries"] = sym_records

    # Group velocities (per k-point, per band, xyz components)
    vg_dict = solver_results.get("group_velocities", {})
    if actual_key in vg_dict:
        vg_arr = vg_dict[actual_key]  # shape (num_k, num_bands, 3)
        vg_data = []
        for ki in range(vg_arr.shape[0]):
            k_vgs = []
            for bi in range(vg_arr.shape[1]):
                vx, vy, vz = (
                    float(vg_arr[ki, bi, 0]),
                    float(vg_arr[ki, bi, 1]),
                    float(vg_arr[ki, bi, 2]),
                )
                k_vgs.append(
                    {
                        "band": bi + 1,
                        "vg_x": round(vx, 8),
                        "vg_y": round(vy, 8),
                        "vg_z": round(vz, 8),
                        "vg_mag": round(float(np.sqrt(vx**2 + vy**2 + vz**2)), 8),
                    }
                )
            vg_data.append({"k_index": ki, "velocities": k_vgs})
        data["group_velocities"] = vg_data

    # Gaps
    gaps = solver_results.get("gaps", {}).get(actual_key, [])
    if gaps:
        data["gaps"] = [
            {"gap_pct": float(g[0]), "low": float(g[1]), "high": float(g[2])}
            if isinstance(g, (list, tuple)) and len(g) >= 3
            else {"gap_pct": float(g[0]) if isinstance(g, (list, tuple)) else 0.0}
            for g in gaps
        ]

    return data


def _evaluate_candidate_worker(
    args: tuple[Any, ...],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    str,
    float,
    float,
    str,
    dict[str, float],
]:
    """Standalone worker executing one candidate evaluation in an isolated process.

    Args:
        args: Tuple containing candidate_values, param_names, fixed_params,
            cell_factory, lattice_type, pitch, dimension, matrix_material,
            background_material, resolution, num_bands, enforce_connectivity,
            epsilon_threshold, min_neck_width_px, and objective.

    Returns:
        Tuple of (param_dict, enriched_meta, status, cost, fom, conn_status, timing).
    """
    (
        candidate_values,
        param_names,
        fixed_params,
        cell_factory,
        lattice_type,
        pitch,
        dimension,
        matrix_material,
        background_material,
        resolution,
        num_bands,
        enforce_connectivity,
        epsilon_threshold,
        min_neck_width_px,
        objective,
        supercell_z,
    ) = args

    # Suppress MPB C-level eigensolver stdout chatter
    mp.verbosity(0)

    t_start = time.perf_counter()
    t_geom = 0.0
    t_solver = 0.0

    param_dict = dict(zip(param_names, candidate_values, strict=False))
    combined_params = {**fixed_params, **param_dict}

    # 1. Generate GDSFactory component (filter kwargs to match cell_factory signature)
    t_geom_0 = time.perf_counter()
    try:
        sig = inspect.signature(cell_factory)
        has_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if has_var_kw:
            cell_kwargs = combined_params
        else:
            cell_kwargs = {
                k: v for k, v in combined_params.items() if k in sig.parameters
            }
    except (ValueError, TypeError):
        cell_kwargs = combined_params
    component = cell_factory(**cell_kwargs)
    t_geom = time.perf_counter() - t_geom_0

    # 2. Setup and run MPB ModeSolver under C-level stdout suppression
    with silence_c_stdout():
        sz_val = float(combined_params.get("supercell_z", supercell_z))
        mpb_lattice = create_lattice(
            lattice_type=lattice_type,
            pitch=pitch,
            dimension=dimension,
            supercell_z=sz_val,
        )

        h_val = float(
            combined_params.get(
                "slab_thickness",
                combined_params.get("thickness", combined_params.get("h", 0.5)),
            )
        )
        mpb_geom = gds_to_mpb_geometry(
            gds_source=component,
            pitch=pitch,
            dimension=dimension,
            slab_thickness=h_val,
            slab_material=matrix_material,
            etch_material=background_material,
            geometry_lattice=mpb_lattice,
        )

        default_mat = matrix_material if dimension == "2D" else background_material

        ms = create_mode_solver(
            geometry_lattice=mpb_lattice,
            geometry=mpb_geom,
            k_points=[mp.Vector3(0, 0, 0)],
            default_material=default_mat,
            resolution=resolution,
            num_bands=num_bands,
        )

        # 3. Check dielectric connectivity (if enabled)
        conn_status = "NOT_CHECKED"
        if enforce_connectivity:
            is_conn, _n_comp, msg = check_slab_connectivity(
                ms_or_epsilon=ms,
                epsilon_threshold=epsilon_threshold,
                check_pbc=True,
                min_neck_width_px=min_neck_width_px,
            )
            if not is_conn:
                conn_status = f"FAILED ({msg})"
                t_total = time.perf_counter() - t_start
                return (
                    param_dict,
                    {"penalty": True},
                    f"FAILED: Disconnected dielectric ({msg})",
                    1.0,
                    1.0,
                    conn_status,
                    {"t_geom": t_geom, "t_solver": 0.0, "t_total": t_total},
                )
            conn_status = f"PASSED ({msg})"

        # 4. Run MPB solver at Gamma
        t_solv_0 = time.perf_counter()
        pol = getattr(objective, "polarization", "te")
        sym_group = getattr(objective, "symmetry_group", "C4v")

        solver_res = run_band_solver(
            ms=ms,
            polarization=pol,
            dimension=dimension,
            compute_symmetries=True,
            symmetry_group=sym_group,
            compute_group_velocities=True,
        )
        t_solver = time.perf_counter() - t_solv_0

        # 5. Evaluate target objective
        obj_eval = objective.evaluate(
            component=component,
            ms=ms,
            solver_results=solver_res,
            params=combined_params,
        )

    t_total = time.perf_counter() - t_start

    # 6. Build enriched metadata with full solver physics data
    enriched_meta = dict(obj_eval.metadata)
    enriched_meta["solver_data"] = _extract_solver_data(
        solver_res, pol_key=pol, num_bands=num_bands
    )
    if obj_eval.group_velocity is not None:
        enriched_meta["vg_top_band"] = obj_eval.group_velocity

    timing = {
        "t_geom": t_geom,
        "t_solver": t_solver,
        "t_total": t_total,
    }

    return (
        param_dict,
        enriched_meta,
        obj_eval.status,
        obj_eval.cost,
        obj_eval.fom,
        conn_status,
        timing,
    )


class BayesianOptimizer:
    """Bayesian Optimization Controller for Photonic Crystal Design.

    Searches across an arbitrary number of geometric parameters to optimize a target
    objective (such as accidental Dirac cone degeneracy or bandgap maximization),
    integrating GDSFactory components and MPB eigensolving.
    """

    def __init__(
        self,
        cell_factory: Callable[..., gf.Component],
        parameters: dict[str, tuple[float, float] | ParameterSpec],
        fixed_parameters: dict[str, Any] | None = None,
        objective: str | BaseObjective = "dirac_degeneracy",
        objective_kwargs: dict[str, Any] | None = None,
        lattice_type: Literal["square", "hexagonal"] = "square",
        pitch: float = 1.0,
        dimension: Literal["2D", "3D_slab"] = "2D",
        supercell_z: float = 4.0,
        resolution: int | tuple[int, int, int] = 32,
        num_bands: int = 8,
        background_material: str = "air",
        matrix_material: str = "si",
        layer_stack: Any = None,
        enforce_connectivity: bool = False,
        epsilon_threshold: float = 1.1,
        min_neck_width_px: int = 1,
        max_iterations: int = 15,
        batch_size: int = 1,
        num_workers: int | None = None,
        initial_points: int = 8,
        initial_sampling: str = "sobol",
        acq_func: str = "LCB",
        acq_func_kwargs: dict[str, Any] | None = None,
        strategy: str = "cl_min",
        random_state: int = 42,
        show_progress: bool = True,
        output_dir: Path | str | None = None,
        geometry_name: str = "unit_cell",
    ):
        """Initializes the BayesianOptimizer.

        Args:
            cell_factory: Callable taking parameter kwargs and returning a gf.Component.
            parameters: Dictionary mapping parameter names to bounds (min, max) or ParameterSpec.
            fixed_parameters: Fixed parameters passed to cell_factory on every evaluation.
            objective: Target objective identifier or BaseObjective instance.
            objective_kwargs: Keyword arguments passed to objective if specified as string.
            lattice_type: Bravais lattice geometry ('square' or 'hexagonal').
            pitch: Lattice constant a in micrometers.
            dimension: Simulation domain ('2D' or '3D_slab').
            resolution: Grid resolution per unit distance a.
            num_bands: Number of eigenbands to compute at Gamma.
            background_material: Background cladding material key (e.g. 'air').
            matrix_material: Slab / matrix dielectric material key (e.g. 'si').
            layer_stack: Optional Technology LayerStack for 3D slab extrusion.
            enforce_connectivity: If True, tests dielectric continuity across PBC before solving.
            epsilon_threshold: Permittivity threshold for connectivity check.
            min_neck_width_px: Minimum neck feature width in pixels for connectivity check.
            max_iterations: Number of guided optimization generations.
            batch_size: Candidate points evaluated per generation.
            num_workers: Number of parallel worker processes for candidate evaluation.
                Defaults to batch_size if None.
            initial_points: Initial exploration points before GP guidance.
            initial_sampling: Initial generator ('sobol', 'lhs', 'halton', 'random').
            acq_func: Acquisition function ('LCB', 'EI', 'PI', 'gp_hedge').
            acq_func_kwargs: Arguments for acquisition function (e.g. {'kappa': 3.5}).
            strategy: Constant liar batch strategy ('cl_min', 'cl_mean', 'cl_max').
            random_state: Random seed for reproducible surrogate sampling.
            show_progress: Whether to display a real-time tqdm progress bar.
            output_dir: Explicit output directory or None to auto-resolve via phc_hydra.
            geometry_name: Name used in standard output hierarchy.
        """
        self.cell_factory = cell_factory
        self.fixed_params = dict(fixed_parameters or {})
        self.lattice_type = lattice_type
        self.pitch = float(pitch)
        self.dimension = dimension
        self.supercell_z = float(supercell_z)
        self.resolution = resolution
        self.num_bands = int(num_bands)
        self.background_material = background_material
        self.matrix_material = matrix_material
        self.layer_stack = layer_stack

        self.enforce_connectivity = enforce_connectivity
        self.epsilon_threshold = float(epsilon_threshold)
        self.min_neck_width_px = int(min_neck_width_px)

        self.max_iterations = int(max_iterations)
        self.batch_size = max(1, int(batch_size))
        self.num_workers = (
            int(num_workers) if num_workers is not None else self.batch_size
        )
        self.initial_points = int(initial_points)
        self.initial_sampling = initial_sampling
        self.acq_func = acq_func
        self.acq_func_kwargs = acq_func_kwargs or {"kappa": 3.5}
        self.strategy = strategy
        self.random_state = int(random_state)
        self.show_progress = bool(show_progress)
        self.geometry_name = geometry_name

        # Parse parameter search space
        self.param_specs: list[ParameterSpec] = []
        for name, spec in parameters.items():
            if isinstance(spec, ParameterSpec):
                self.param_specs.append(spec)
            elif isinstance(spec, (tuple, list)) and len(spec) == 2:
                self.param_specs.append(
                    ParameterSpec(name=name, bounds=(float(spec[0]), float(spec[1])))
                )
            else:
                raise ValueError(
                    f"Invalid parameter specification for '{name}': {spec}"
                )

        self.param_names = [p.name for p in self.param_specs]

        # Resolve objective
        obj_kw = dict(objective_kwargs or {})
        self.objective: BaseObjective = get_objective(objective, **obj_kw)

        # Output directory management via phc_hydra
        if output_dir is not None:
            self.output_dir = Path(output_dir).resolve()
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.output_mgr = SimulationOutputManager(
                output_dir=self.output_dir,
                solver="mpb",
                sim_type="optimization",
                geometry_name=self.geometry_name,
            )
        else:
            from phc_hydra import resolve_simulation_output_dir

            resolved = resolve_simulation_output_dir(
                solver="mpb",
                sim_type="optimization",
                geometry=self.geometry_name,
            )
            self.output_dir = resolved
            self.output_mgr = SimulationOutputManager(
                output_dir=self.output_dir,
                solver="mpb",
                sim_type="optimization",
                geometry_name=self.geometry_name,
            )

        # Initialize skopt space
        self.dimensions = []
        for p in self.param_specs:
            if p.param_type == "integer":
                self.dimensions.append(
                    Integer(int(p.bounds[0]), int(p.bounds[1]), name=p.name)
                )
            else:
                self.dimensions.append(
                    Real(float(p.bounds[0]), float(p.bounds[1]), name=p.name)
                )

        self.optimizer = Optimizer(
            dimensions=self.dimensions,
            base_estimator="GP",
            n_initial_points=self.initial_points,
            initial_point_generator=self.initial_sampling,
            acq_func=self.acq_func,
            acq_func_kwargs=self.acq_func_kwargs,
            random_state=self.random_state,
        )

        self.records: list[OptimizationRecord] = []
        self.eval_counter = 0
        self._log_path = self.output_dir / "bo_evaluations.log"
        self._jsonl_path = self.output_dir / "bo_evaluations.jsonl"

    def _record_to_json_dict(self, rec: OptimizationRecord) -> dict[str, Any]:
        """Converts an OptimizationRecord to a fully serializable dictionary.

        Args:
            rec: Evaluation record to serialize.

        Returns:
            JSON-serializable dictionary with all per-evaluation physics data.
        """
        entry: dict[str, Any] = {
            "eval_index": rec.eval_index,
            "generation": rec.generation,
            "params": rec.params,
            "cost": rec.cost,
            "fom": rec.fom,
            "status": rec.status,
            "connectivity": rec.connectivity,
            "timing": rec.timing,
        }

        # Flatten solver_data into top level for easy access
        solver_data = rec.metadata.get("solver_data", {})
        if solver_data:
            # Per-k-point frequency table
            k_points = solver_data.get("k_points", [])
            if k_points:
                entry["k_points"] = k_points

            # Symmetry per band
            symmetries = solver_data.get("symmetries", [])
            if symmetries:
                entry["symmetries"] = symmetries

            # Group velocities
            group_velocities = solver_data.get("group_velocities", [])
            if group_velocities:
                entry["group_velocities"] = group_velocities

            # Gaps
            gaps = solver_data.get("gaps", [])
            if gaps:
                entry["gaps"] = gaps

        # Objective-specific metadata (target bands, raw/norm cost, corrections)
        for key in (
            "target_bands",
            "raw_cost",
            "normalized_cost",
            "effective_cost",
            "freq_middle",
            "freq_high",
            "freq_low",
            "vg_top_band",
        ):
            if key in rec.metadata:
                entry[key] = rec.metadata[key]

        # Irrep mapping and corrections
        if "full_map" in rec.metadata:
            fm = rec.metadata["full_map"]
            entry["irrep_map"] = (
                {
                    str(band): {"irrep": v[0], "confidence": v[1], "freq": v[2]}
                    for band, v in fm.items()
                }
                if fm
                else {}
            )
        if rec.metadata.get("corrections"):
            entry["irrep_corrections"] = [
                {
                    "band": c[0],
                    "original": c[1],
                    "confidence": c[2],
                    "corrected_to": c[3],
                }
                for c in rec.metadata["corrections"]
            ]
        if rec.metadata.get("penalty"):
            entry["penalty"] = True

        return entry

    def _write_eval_log(self, rec: OptimizationRecord) -> None:
        """Appends a human-readable log entry and a JSON Lines record for this evaluation.

        Writes two files in the output directory:
            - ``bo_evaluations.log``: Human-readable text log with formatted physics data.
            - ``bo_evaluations.jsonl``: One JSON object per line for machine-readable streaming.

        Args:
            rec: Completed optimization record to log.
        """
        # --- JSON Lines record ---
        json_entry = self._record_to_json_dict(rec)
        with open(self._jsonl_path, "a") as jf:
            jf.write(json.dumps(json_entry) + "\n")

        # --- Human-readable text log ---
        lines: list[str] = []
        sep = "-" * 72
        lines.append(sep)
        lines.append(
            f"Eval #{rec.eval_index:04d}  |  Gen {rec.generation}  |  "
            f"Status: {rec.status}"
        )
        lines.append(sep)

        # Parameters
        param_strs = [f"  {k} = {v:.6f}" for k, v in rec.params.items()]
        lines.append("Parameters:")
        lines.extend(param_strs)

        # Cost / FOM
        lines.append(f"Cost:          {rec.cost:.8f}")
        lines.append(f"FOM:           {rec.fom:.4f}")
        lines.append(f"Connectivity:  {rec.connectivity}")

        # Frequencies at Gamma
        solver_data = rec.metadata.get("solver_data", {})
        k_points = solver_data.get("k_points", [])
        if k_points:
            gamma = k_points[0]
            freqs = gamma.get("freqs", [])
            lines.append(
                f"k = ({gamma['k1']}, {gamma['k2']}, {gamma['k3']}), "
                f"kmag/2pi = {gamma['kmag_over_2pi']}"
            )
            freq_str = ", ".join(f"{f:.6f}" for f in freqs)
            lines.append(f"Freqs:  [{freq_str}]")

        # Symmetry table
        symmetries = solver_data.get("symmetries", [])
        if symmetries:
            lines.append("Symmetry at Gamma:")
            lines.append(
                f"  {'Band':>4s}  {'Freq':>10s}  {'Irrep':>6s}  "
                f"{'Conf':>6s}  Characters"
            )
            for sr in symmetries:
                chars_str = "  ".join(
                    f"{k}={v:+.3f}" for k, v in sr.get("characters", {}).items()
                )
                lines.append(
                    f"  {sr['band']:4d}  {sr['freq']:10.6f}  {sr['irrep']:>6s}  "
                    f"{sr['confidence']:6.3f}  {chars_str}"
                )

        # Group velocities
        group_velocities = solver_data.get("group_velocities", [])
        if group_velocities:
            lines.append("Group velocities at Gamma:")
            lines.append(
                f"  {'Band':>4s}  {'vg_x':>12s}  {'vg_y':>12s}  {'vg_z':>12s}  {'|vg|':>12s}"
            )
            for k_vg in group_velocities:
                for v in k_vg.get("velocities", []):
                    lines.append(
                        f"  {v['band']:4d}  {v['vg_x']:12.8f}  {v['vg_y']:12.8f}  "
                        f"{v['vg_z']:12.8f}  {v['vg_mag']:12.8f}"
                    )

        # Timing
        lines.append(
            f"Timing: geom={rec.timing.get('t_geom', 0):.3f}s  "
            f"solver={rec.timing.get('t_solver', 0):.3f}s  "
            f"total={rec.timing.get('t_total', 0):.3f}s"
        )
        lines.append("")

        with open(self._log_path, "a") as lf:
            lf.write("\n".join(lines) + "\n")

    def _write_evaluations_json(self) -> None:
        """Writes comprehensive JSON file with all evaluation records and run metadata.

        Generates ``bo_evaluations.json`` containing the full optimization trajectory
        with per-evaluation parameters, frequencies, symmetry characters, irreps,
        group velocities, cost, FOM, timing, and connectivity status.
        """
        all_entries = [self._record_to_json_dict(r) for r in self.records]

        summary: dict[str, Any] = {
            "run_metadata": {
                "objective": self.objective.name,
                "lattice_type": self.lattice_type,
                "pitch": self.pitch,
                "dimension": self.dimension,
                "resolution": self.resolution,
                "num_bands": self.num_bands,
                "matrix_material": self.matrix_material,
                "background_material": self.background_material,
                "parameter_names": self.param_names,
                "parameter_bounds": {p.name: list(p.bounds) for p in self.param_specs},
                "fixed_parameters": self.fixed_params,
                "initial_points": self.initial_points,
                "max_iterations": self.max_iterations,
                "batch_size": self.batch_size,
                "acq_func": self.acq_func,
                "strategy": self.strategy,
                "random_state": self.random_state,
                "total_evaluations": len(self.records),
            },
            "evaluations": all_entries,
        }

        json_file = self.output_dir / "bo_evaluations.json"
        with open(json_file, "w") as f:
            json.dump(summary, f, indent=2)

    def _evaluate_candidate(
        self,
        candidate_values: list[float],
        gen: int,
    ) -> OptimizationRecord:
        """Evaluates a single parameter combination through layout and solver."""
        worker_args = (
            candidate_values,
            self.param_names,
            self.fixed_params,
            self.cell_factory,
            self.lattice_type,
            self.pitch,
            self.dimension,
            self.matrix_material,
            self.background_material,
            self.resolution,
            self.num_bands,
            self.enforce_connectivity,
            self.epsilon_threshold,
            self.min_neck_width_px,
            self.objective,
        )
        (
            param_dict,
            enriched_meta,
            status,
            cost,
            fom,
            conn_status,
            timing,
        ) = _evaluate_candidate_worker(worker_args)

        self.eval_counter += 1
        record = OptimizationRecord(
            eval_index=self.eval_counter,
            generation=gen,
            params=param_dict,
            cost=cost,
            fom=fom,
            status=status,
            connectivity=conn_status,
            timing=timing,
            metadata=enriched_meta,
        )
        self._write_eval_log(record)
        return record

    def run(self, show_progress: bool | None = None) -> OptimizationResult:
        """Executes the Bayesian Optimization search loop.

        Args:
            show_progress: Whether to display a real-time tqdm progress bar.
                Defaults to self.show_progress if None.

        Returns:
            OptimizationResult containing best parameters, FOM, and trajectory records.
        """
        from tqdm import tqdm

        progress_enabled = (
            self.show_progress if show_progress is None else bool(show_progress)
        )
        total_evals = self.initial_points + (self.max_iterations * self.batch_size)
        curr_eval = 0
        gen = 0

        pbar = tqdm(
            total=total_evals,
            desc="Bayesian Optimization",
            unit="eval",
            disable=not progress_enabled,
        )

        try:
            while curr_eval < total_evals:
                batch_sz = min(self.batch_size, total_evals - curr_eval)
                candidates = self.optimizer.ask(
                    n_points=batch_sz, strategy=self.strategy
                )

                # Prepare inputs for each candidate in the batch
                task_args = [
                    (
                        x_cand,
                        self.param_names,
                        self.fixed_params,
                        self.cell_factory,
                        self.lattice_type,
                        self.pitch,
                        self.dimension,
                        self.matrix_material,
                        self.background_material,
                        self.resolution,
                        self.num_bands,
                        self.enforce_connectivity,
                        self.epsilon_threshold,
                        self.min_neck_width_px,
                        self.objective,
                        self.supercell_z,
                    )
                    for x_cand in candidates
                ]

                # Concurrent evaluation across worker processes when num_workers > 1
                if self.num_workers > 1 and len(candidates) > 1:
                    with ProcessPoolExecutor(
                        max_workers=min(self.num_workers, len(candidates))
                    ) as executor:
                        worker_results = list(
                            executor.map(_evaluate_candidate_worker, task_args)
                        )
                else:
                    worker_results = [
                        _evaluate_candidate_worker(args) for args in task_args
                    ]

                batch_costs: list[float] = []

                for (
                    param_dict,
                    enriched_meta,
                    status,
                    cost,
                    fom,
                    conn_status,
                    timing,
                ) in worker_results:
                    self.eval_counter += 1
                    rec = OptimizationRecord(
                        eval_index=self.eval_counter,
                        generation=gen,
                        params=param_dict,
                        cost=cost,
                        fom=fom,
                        status=status,
                        connectivity=conn_status,
                        timing=timing,
                        metadata=enriched_meta,
                    )
                    self.records.append(rec)
                    self._write_eval_log(rec)

                    # Objective mode: log10 or linear
                    c_val = rec.cost
                    skopt_y = float(np.log10(max(c_val, 1e-12)))
                    batch_costs.append(skopt_y)
                    curr_eval += 1

                    best_so_far = max(r.fom for r in self.records)
                    pbar.set_postfix(
                        {
                            "best_fom": f"{best_so_far:.2f}",
                            "cost": f"{rec.cost:.4f}",
                            "conn": rec.connectivity[:6],
                        }
                    )
                    pbar.update(1)

                self.optimizer.tell(candidates, batch_costs)
                gen += 1
        finally:
            pbar.close()

        # Identify best result
        valid_records = [
            r for r in self.records if not r.metadata.get("penalty", False)
        ]
        if valid_records:
            best_rec = max(valid_records, key=lambda r: r.fom)
        else:
            best_rec = min(self.records, key=lambda r: r.cost)

        # Export mandatory GDS layout for optimal design
        best_combined = {**self.fixed_params, **best_rec.params}
        try:
            sig = inspect.signature(self.cell_factory)
            has_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            if has_var_kw:
                cell_kwargs = best_combined
            else:
                cell_kwargs = {
                    k: v for k, v in best_combined.items() if k in sig.parameters
                }
        except (ValueError, TypeError):
            cell_kwargs = best_combined
        best_component = self.cell_factory(**cell_kwargs)
        optimal_gds = self.output_dir / "unit_cell.gds"
        export_gds(best_component, optimal_gds, overwrite=True)

        # Generate plots
        conv_png = self.output_dir / "bo_convergence.png"
        plot_bo_convergence(self.records, output_path=conv_png)

        if len(self.param_names) == 2:
            surr_png = self.output_dir / "bo_surrogate_map.png"
            plot_bo_surrogate_map(
                optimizer=self.optimizer,
                records=self.records,
                param_names=self.param_names,
                output_path=surr_png,
            )

        # Export tabular trajectory CSV
        csv_file = self.output_dir / "bo_trajectory.csv"
        with open(csv_file, "w") as f:
            hdr = [
                "eval_index",
                "generation",
                *self.param_names,
                "cost",
                "fom",
                "connectivity",
                "status",
            ]
            f.write(",".join(hdr) + "\n")
            for r in self.records:
                p_vals = [f"{r.params.get(k, 0.0):.6f}" for k in self.param_names]
                row = [
                    str(r.eval_index),
                    str(r.generation),
                    *p_vals,
                    f"{r.cost:.6f}",
                    f"{r.fom:.6f}",
                    r.connectivity,
                    f'"{r.status}"',
                ]
                f.write(",".join(row) + "\n")

        # Export simulation results JSON manifest
        results_summary = {
            "best_params": best_rec.params,
            "best_fom": best_rec.fom,
            "best_cost": best_rec.cost,
            "best_eval_index": best_rec.eval_index,
            "total_evaluations": len(self.records),
            "objective": self.objective.name,
            "lattice_type": self.lattice_type,
            "pitch": self.pitch,
            "dimension": self.dimension,
            "resolution": self.resolution,
            "fixed_parameters": self.fixed_params,
            "files": {
                "unit_cell_gds": str(optimal_gds.name),
                "trajectory_csv": str(csv_file.name),
                "convergence_plot": str(conv_png.name),
                "evaluations_json": "bo_evaluations.json",
                "evaluations_jsonl": "bo_evaluations.jsonl",
                "evaluations_log": "bo_evaluations.log",
            },
        }

        json_file = self.output_dir / "simulation_results.json"
        with open(json_file, "w") as f:
            json.dump(results_summary, f, indent=2)

        # Write comprehensive per-evaluation JSON data file
        self._write_evaluations_json()

        gp_model = self.optimizer.models[-1] if self.optimizer.models else None

        return OptimizationResult(
            best_params=best_rec.params,
            best_fom=best_rec.fom,
            best_cost=best_rec.cost,
            records=self.records,
            gp_model=gp_model,
            output_dir=self.output_dir,
        )

    def run_best(
        self,
        plot_eps: bool = True,
        plot_bands: bool = True,
        save_plots: bool = True,
        best_params: dict[str, float] | None = None,
        rectify: bool = True,
        periods: int = 3,
        grid_resolution: int = 64,
        k_density: int = 20,
        num_workers: int = 1,
    ) -> dict[str, Any]:
        """Runs the best parameter set from the optimization history.

        Reconstructs the optimal unit cell, extracts and plots the dielectric
        permittivity profile, solves full band dispersion along the irreducible
        Brillouin zone path, and generates the band diagram plot.

        Args:
            plot_eps: Whether to render and return the permittivity figure.
            plot_bands: Whether to solve dispersion and render the band structure.
            save_plots: If True, saves figures to `self.output_dir`.
            best_params: Optional explicit parameter overrides. If None, resolves
                the highest FOM candidate from `self.records`.
            rectify: Whether to rectify non-orthogonal unit cells for epsilon plotting.
            periods: In-plane unit cell tiling periods for the epsilon plot.
            grid_resolution: Resolution (pixels per pitch) for the epsilon grid.
            k_density: Interpolation density between high-symmetry k-points.
            num_workers: Number of parallel worker processes for the band solve.

        Returns:
            Dictionary containing:
                - "params": Resolved best parameters.
                - "component": GDSFactory Component instance.
                - "ms": Initialized MPB ModeSolver.
                - "epsilon": Extracted permittivity array.
                - "solver_results": Band dispersion results dictionary.
                - "fig_eps": Permittivity Matplotlib Figure (or None).
                - "fig_bands": Band diagram Matplotlib Figure (or None).
        """
        if not self.records and best_params is None:
            raise RuntimeError(
                "No evaluations found. Call opt.run() first or provide `best_params`."
            )

        # 1. Resolve optimal parameters
        if best_params is None:
            valid_records = [
                r for r in self.records if not r.metadata.get("penalty", False)
            ]
            if valid_records:
                best_rec = max(valid_records, key=lambda r: r.fom)
            else:
                best_rec = min(self.records, key=lambda r: r.cost)
            params = {**self.fixed_params, **best_rec.params}
        else:
            params = {**self.fixed_params, **best_params}

        # 2. Build GDS layout and MPB geometry
        try:
            sig = inspect.signature(self.cell_factory)
            has_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            if has_var_kw:
                cell_kwargs = params
            else:
                cell_kwargs = {k: v for k, v in params.items() if k in sig.parameters}
        except (ValueError, TypeError):
            cell_kwargs = params
        component = self.cell_factory(**cell_kwargs)
        sz_val = float(params.get("supercell_z", self.supercell_z))
        mpb_lattice = create_lattice(
            lattice_type=self.lattice_type,
            pitch=self.pitch,
            dimension=self.dimension,
            supercell_z=sz_val,
        )
        h_val = float(
            params.get(
                "slab_thickness",
                params.get("thickness", params.get("h", 0.5)),
            )
        )
        mpb_geom = gds_to_mpb_geometry(
            gds_source=component,
            pitch=self.pitch,
            dimension=self.dimension,
            slab_thickness=h_val,
            slab_material=self.matrix_material,
            etch_material=self.background_material,
            geometry_lattice=mpb_lattice,
        )
        default_mat = (
            self.matrix_material if self.dimension == "2D" else self.background_material
        )

        # 3. Generate high-symmetry k-path for the band diagram
        k_pts, k_labels, k_indices = get_high_symmetry_kpath(
            lattice_type=self.lattice_type,
            k_density=k_density,
        )

        # 4. Construct MPB ModeSolver initialized along the k-path
        ms = create_mode_solver(
            geometry_lattice=mpb_lattice,
            geometry=mpb_geom,
            k_points=k_pts,
            default_material=default_mat,
            resolution=self.resolution,
            num_bands=self.num_bands,
        )

        # 5. Extract permittivity grid
        eps_array = get_epsilon_grid(
            ms,
            rectify=rectify,
            periods=periods,
            periods_z=1,
            resolution=grid_resolution,
        )

        fig_eps = None
        if plot_eps:
            eps_path = self.output_dir / "best_epsilon.png" if save_plots else None
            param_str = ", ".join(f"{k}={v:.4f}" for k, v in params.items())
            fig_eps = plot_epsilon(
                epsilon=eps_array,
                title=f"Optimal Dielectric Profile ({param_str})",
                save_path=eps_path,
            )

        # 6. Compute band structure and render band diagram
        fig_bands = None
        solver_res: dict[str, Any] = {}
        if plot_bands:
            pol = getattr(self.objective, "polarization", "te")
            solver_res = run_band_solver(
                ms=ms,
                polarization=pol,
                dimension=self.dimension,
                num_workers=num_workers,
            )

            bands_path = (
                self.output_dir / "best_band_structure.png" if save_plots else None
            )
            param_str = ", ".join(f"{k}={v:.4f}" for k, v in params.items())
            fig_bands = plot_band_structure(
                results=solver_res,
                node_labels=k_labels,
                node_indices=k_indices,
                title=f"Optimal Photonic Band Structure ({param_str})",
                save_path=bands_path,
            )

        return {
            "params": params,
            "component": component,
            "ms": ms,
            "epsilon": eps_array,
            "solver_results": solver_res,
            "fig_eps": fig_eps,
            "fig_bands": fig_bands,
        }

    def analyze_locus(
        self,
        max_loci: int = 1,
        mode: Literal["auto", "cartesian", "polar"] = "auto",
        refine: bool = True,
        refine_tolerance: float = 1e-5,
        max_refine_steps: int = 8,
        exclude_unrefined: bool = True,
        max_residual_gap: float = 1e-4,
        delta_k: float = 0.001,
        sample_points: int = 40,
        plot_profile: bool = True,
        save_artifacts: bool = True,
        verbose: bool = True,
    ) -> list[dict[str, Any]]:
        """Extracts, refines, and characterizes optimal 1D degeneracy manifolds.

        Fits a penalty-filtered Gaussian Process surrogate to the evaluated optimization
        landscape, extracts continuous 1D degeneracy loci, refines sampled coordinates
        to exact degeneracy using fast Gamma-only 1D secant line searches, computes group
        velocities at an adjacent k-point k = (delta_k, 0, 0), renders a 3-panel locus
        profile figure, and saves tabular CSV / JSON manifests.

        Args:
            max_loci: Maximum number of distinct locus ridges to extract (defaults to 1).
            mode: Extraction geometry mode: 'polar' (closed ring), 'cartesian' (open), or 'auto'.
            refine: Whether to execute local secant refinement along curve normal vectors.
            refine_tolerance: Cost threshold (e.g. 1e-5) defining exact degeneracy.
            max_refine_steps: Maximum secant iterations per locus point.
            exclude_unrefined: Whether to prune points that cannot reach max_residual_gap.
            max_residual_gap: Maximum allowed residual cost for valid refined points.
            delta_k: Offset from Gamma along k_x for Hellmann-Feynman group velocity evaluation.
            sample_points: Number of points sampled along the smooth locus curve.
            plot_profile: If True, renders and saves a 3-panel locus profile plot.
            save_artifacts: If True, saves locus_points.csv and optimal_loci.json to disk.
            verbose: Whether to print progress information to stdout.

        Returns:
            List of processed locus dictionaries containing refined coordinates,
            residual gaps, and adjacent-point group velocities.

        Raises:
            RuntimeError: If called before any evaluations have been completed.
            ValueError: If the optimization problem does not have exactly 2 search parameters.
        """
        if not self.records:
            raise RuntimeError(
                "No evaluations found. Call opt.run() before analyzing loci."
            )
        if len(self.param_names) != 2:
            raise ValueError(
                f"Locus manifold analysis requires exactly 2 search parameters, got {len(self.param_names)}: {self.param_names}"
            )

        # 1. Fit clean surrogate landscape
        landscape = fit_clean_surrogate(
            records=self.records,
            param_specs=self.param_specs,
            grid_resolution=80,
        )

        p1_n, p2_n = self.param_names[0], self.param_names[1]

        # 2. Extract loci from surrogate FOM
        loci = extract_optimal_loci(
            grid_x1=landscape.x1_grid,
            grid_x2=landscape.x2_grid,
            fom_2d=landscape.predicted_fom,
            threshold_percentile=85.0,
            max_loci=max_loci,
            sample_points=sample_points,
            mode=mode,
            p1_name=p1_n,
            p2_name=p2_n,
        )

        if not loci:
            if verbose:
                print("No valid degeneracy loci found in surrogate landscape.")
            return []

        # Build reusable geometry lattice & material settings
        default_mat = (
            self.matrix_material if self.dimension == "2D" else self.background_material
        )
        pol = getattr(self.objective, "polarization", "te")

        def _build_mode_solver_at_k(
            pt_params: dict[str, float], k_pt: mp.Vector3
        ) -> tuple[Any, dict[str, Any]]:
            combined = {**self.fixed_params, **pt_params}
            try:
                sig = inspect.signature(self.cell_factory)
                has_var_kw = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in sig.parameters.values()
                )
                cell_kwargs = (
                    combined
                    if has_var_kw
                    else {k: v for k, v in combined.items() if k in sig.parameters}
                )
            except (ValueError, TypeError):
                cell_kwargs = combined
            comp = self.cell_factory(**cell_kwargs)
            sz_val = float(combined.get("supercell_z", self.supercell_z))
            lat = create_lattice(
                lattice_type=self.lattice_type,
                pitch=self.pitch,
                dimension=self.dimension,
                supercell_z=sz_val,
            )
            h_val = float(
                combined.get(
                    "slab_thickness",
                    combined.get("thickness", combined.get("h", 0.5)),
                )
            )
            geom = gds_to_mpb_geometry(
                gds_source=comp,
                pitch=self.pitch,
                dimension=self.dimension,
                slab_thickness=h_val,
                slab_material=self.matrix_material,
                etch_material=self.background_material,
                geometry_lattice=lat,
            )
            ms = create_mode_solver(
                geometry_lattice=lat,
                geometry=geom,
                k_points=[k_pt],
                default_material=default_mat,
                resolution=self.resolution,
                num_bands=self.num_bands,
            )
            return ms, combined

        def _evaluate_gamma_gap(
            pt_params: dict[str, float],
        ) -> tuple[float, float]:
            with silence_c_stdout():
                ms_gamma, combined = _build_mode_solver_at_k(
                    pt_params, mp.Vector3(0, 0, 0)
                )
                solver_res = run_band_solver(
                    ms=ms_gamma,
                    polarization=pol,
                    dimension=self.dimension,
                    num_workers=1,
                )
                obj_eval = self.objective.evaluate(solver_res, combined)
                f_high = obj_eval.metadata.get("freq_high", 0.0)
                f_low = obj_eval.metadata.get("freq_low", 0.0)
                signed_gap = float(f_high - f_low)
                return signed_gap, float(obj_eval.cost)

        def _evaluate_adjacent_vg(pt_params: dict[str, float]) -> float:
            with silence_c_stdout():
                k_adj = mp.Vector3(delta_k, 0, 0)
                ms_vg, _ = _build_mode_solver_at_k(pt_params, k_adj)
                solver_res = run_band_solver(
                    ms=ms_vg,
                    polarization=pol,
                    dimension=self.dimension,
                    compute_group_velocities=True,
                    num_workers=1,
                )
                vg_dict = solver_res.get("group_velocities", {})
                pol_k = pol.lower()
                actual_k = pol_k
                if actual_k not in vg_dict:
                    for alias in ("te", "tm", "te_like", "tm_like", "all"):
                        if alias in vg_dict:
                            actual_k = alias
                            break
                if actual_k in vg_dict:
                    vg_arr = vg_dict[actual_k]  # shape (1, num_bands, 3)
                    t_bands = getattr(self.objective, "target_bands", None) or [
                        max(1, self.num_bands // 2)
                    ]
                    vgs_list = [
                        float(np.linalg.norm(vg_arr[0, b - 1]))
                        for b in t_bands
                        if b <= vg_arr.shape[1]
                    ]
                    if vgs_list:
                        return float(max(vgs_list))
                return 0.0

        bounds_dict = {p.name: p.bounds for p in self.param_specs}

        for locus in loci:
            l_id = locus["locus_id"]
            if verbose:
                print(f"Processing Locus #{l_id} ({len(locus['x1'])} points)...")

            if refine:
                if verbose:
                    print(
                        f"  Refining Locus #{l_id} at Gamma (tol={refine_tolerance:.1e})..."
                    )
                refine_locus_points(
                    locus=locus,
                    param_names=self.param_names,
                    evaluate_point_fn=_evaluate_gamma_gap,
                    tolerance=refine_tolerance,
                    max_steps=max_refine_steps,
                    exclude_unrefined=exclude_unrefined,
                    max_residual_gap=max_residual_gap,
                    param_bounds=bounds_dict,
                )

            if verbose:
                print(
                    f"  Evaluating group velocity at adjacent point delta_k={delta_k}..."
                )
            evaluate_locus_group_velocities(
                locus=locus,
                param_names=self.param_names,
                compute_vg_fn=_evaluate_adjacent_vg,
            )

            if save_artifacts:
                locus_dir = self.output_dir / f"locus_{l_id:02d}"
                locus_dir.mkdir(parents=True, exist_ok=True)
                csv_path = locus_dir / "locus_points.csv"
                export_locus_to_csv(locus, csv_path, param_names=self.param_names)
                if verbose:
                    print(f"  Saved locus points table to '{csv_path}'")

                if plot_profile:
                    fig_path = locus_dir / "locus_profile.png"
                    plot_locus_profile(
                        locus=locus,
                        param_names=self.param_names,
                        param_bounds=bounds_dict,
                        output_path=fig_path,
                        title=f"Degeneracy Locus #{l_id} ($v_g$ at $\\Delta k={delta_k}$)",
                    )
                    if verbose:
                        print(f"  Saved 3-panel locus profile to '{fig_path}'")

        if save_artifacts and loci:
            export_loci_to_json(loci, self.output_dir / "optimal_loci.json")

        return loci
