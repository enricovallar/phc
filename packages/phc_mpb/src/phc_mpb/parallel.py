"""Concurrent multi-worker k-point wrapper for MPB band structure calculations.

Enables embarrassingly parallel band structure solving across the irreducible Brillouin
zone by partitioning k-points across multiple worker processes using ProcessPoolExecutor.
Supports anisotropic mesh resolution (rx, ry, rz) and 3D slab parity modes (zeven / zodd).
"""

import concurrent.futures
import contextlib
import os
from collections.abc import Sequence
from typing import Any, Literal

import meep as mp
import numpy as np
from phc_materials import to_mpb_medium

# Set temporary Matplotlib cache directory to avoid multiprocessing permission warnings
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_mpb_parallel")


def _worker_solve_k_chunk(task_args: tuple[Any, ...]) -> dict[str, Any]:
    """Worker entrypoint solving a contiguous chunk of k-points in an independent process.

    Args:
        task_args: Tuple containing:
            - chunk_idx: Worker/task chunk integer identifier.
            - k_chunk: List of mp.Vector3 instances for this chunk.
            - geometry_lattice: mp.Lattice instance.
            - geometry: List of mp.GeometricObject instances.
            - default_material: mp.Medium instance or material key.
            - resolution: mp.Vector3 or int resolution specification.
            - num_bands: Number of eigenbands to compute.
            - polarization: Mode polarization ('te', 'tm', 'te_like', 'tm_like', 'both', etc.).
            - dimension: '2D' or '3D_slab'.
            - tolerance: Numerical convergence tolerance for conjugate-gradient solver.
            - verbose: Whether worker prints MPB solver progress to stdout.

    Returns:
        Dict containing chunk_idx and mapping of polarization names to computed frequency arrays.

    Raises:
        RuntimeError: If MPB solver execution fails.
    """
    (
        chunk_idx,
        k_chunk,
        geometry_lattice,
        geometry,
        default_material,
        resolution,
        num_bands,
        polarization,
        dimension,
        tolerance,
        verbose,
        *rest,
    ) = task_args
    compute_fractions = rest[0] if rest else False
    polarization_method = rest[1] if len(rest) > 1 else "midplane"

    from meep import mpb

    bg_medium: Any
    if isinstance(default_material, str) or hasattr(default_material, "index"):
        bg_medium = to_mpb_medium(default_material)
    elif isinstance(default_material, (int, float)):
        bg_medium = mp.Medium(index=float(default_material))
    else:
        bg_medium = default_material

    ms = mpb.ModeSolver(
        geometry_lattice=geometry_lattice,
        geometry=geometry,
        k_points=k_chunk,
        default_material=bg_medium,
        resolution=resolution,
        num_bands=num_bands,
        tolerance=tolerance,
    )

    results: dict[str, np.ndarray] = {}
    pol = polarization.lower()
    should_compute_fracs = compute_fractions or pol in ("all", "no_parity")
    te_fracs_chunk: list[list[float]] = []
    confinements_chunk: list[list[float]] = []

    def _metrics_callback(solver: Any) -> None:
        from phc_mpb.classification import compute_modal_metrics

        metrics = compute_modal_metrics(solver, polarization_method=polarization_method)
        if isinstance(metrics, list):
            te_fracs_chunk.append([m["te"] for m in metrics])
            confinements_chunk.append([m["confinement"] for m in metrics])

    band_funcs = [_metrics_callback] if should_compute_fracs else []

    # Suppress worker stdout unless verbose is explicitly requested
    from phc_utils import silence_c_stdout

    ctx = contextlib.nullcontext() if verbose else silence_c_stdout()

    with ctx:
        if dimension == "2D":
            if pol in ("te", "both"):
                ms.run_te(*band_funcs)
                results["te"] = np.copy(ms.all_freqs)
            if pol in ("tm", "both"):
                ms.run_tm(*band_funcs)
                results["tm"] = np.copy(ms.all_freqs)
            if pol in ("all", "no_parity"):
                ms.run(*band_funcs)
                results["all"] = np.copy(ms.all_freqs)
        else:
            # 3D Slab modes
            if pol in ("te_like", "even", "both", "te"):
                ms.run_zeven(*band_funcs)
                results["te_like"] = np.copy(ms.all_freqs)
            if pol in ("tm_like", "odd", "both", "tm"):
                ms.run_zodd(*band_funcs)
                results["tm_like"] = np.copy(ms.all_freqs)
            if pol in ("all", "no_parity"):
                ms.run(*band_funcs)
                results["all"] = np.copy(ms.all_freqs)

    ret: dict[str, Any] = {
        "chunk_idx": chunk_idx,
        "freqs": results,
    }
    if should_compute_fracs and te_fracs_chunk:
        ret["te_fractions"] = np.array(te_fracs_chunk)
        ret["confinements"] = np.array(confinements_chunk)
    return ret


def run_parallel_band_solver(
    ms: Any = None,
    *,
    geometry_lattice: Any = None,
    geometry: list[Any] | None = None,
    k_points: Sequence[Any] | None = None,
    default_material: Any = "air",
    resolution: int | tuple[int, int, int] | mp.Vector3 = 32,
    resolution_z: int | None = None,
    num_bands: int = 8,
    polarization: Literal[
        "te", "tm", "te_like", "tm_like", "even", "odd", "both", "all", "no_parity"
    ] = "te_like",
    dimension: Literal["2D", "3D_slab"] = "3D_slab",
    num_workers: int | None = None,
    cladding_index: float = 1.0,
    tolerance: float = 1e-7,
    verbose: bool = False,
    compute_polarization_fractions: bool = False,
    polarization_method: Literal[
        "midplane", "volumetric", "slab", "magnetic"
    ] = "midplane",
) -> dict[str, Any]:
    """Executes MPB band structure calculations in parallel across k-points using multiple worker processes.

    Distributes k-points into contiguous slices across independent worker processes, each
    running a dedicated MPB ModeSolver. Automatically reassembles eigenfrequencies, computes
    omnidirectional and guided band gaps, and calculates the cladding light line for 3D slabs.

    Args:
        ms: Optional pre-configured mpb.ModeSolver instance. If provided, geometry_lattice,
            geometry, k_points, default_material, resolution, and num_bands are extracted from ms.
        geometry_lattice: mp.Lattice defining unit cell basis vectors and domain size.
        geometry: List of mp.GeometricObject primitives forming the structure.
        k_points: Sequence of mp.Vector3 k-points along the high-symmetry path.
        default_material: Background medium key (e.g. 'air') or mp.Medium instance.
        resolution: Grid resolution per unit distance a. Can be an integer (isotropic),
            a 3-tuple (rx, ry, rz), or an mp.Vector3.
        resolution_z: Optional vertical resolution per unit distance a along z.
            If specified, forms an anisotropic resolution Vector3(resolution, resolution, resolution_z).
        num_bands: Number of eigenbands to solve at each k-point.
        polarization: Mode polarization: 'te' or 'tm' (2D), 'te_like'/'even' or 'tm_like'/'odd' (3D slab),
            or 'both'.
        dimension: Dimensionality of simulation ('2D' or '3D_slab').
        num_workers: Number of concurrent worker processes. If None or <=0, defaults to
            min(len(k_points), os.cpu_count() or 4).
        cladding_index: Refractive index of background cladding for light line (dimension='3D_slab').
        tolerance: Conjugate-gradient solver convergence tolerance (default: 1e-7).
        verbose: If True, worker processes stream MPB iteration output to stdout.

    Returns:
        Dictionary containing:
            - 'freqs': Mapping of polarization strings ('te', 'tm', 'te_like', 'tm_like') to
              2D numpy arrays of shape (num_k_points, num_bands) in normalized dimensionless units.
            - 'gaps': Mapping of polarization strings to list of gap percentages between consecutive bands.
            - 'guided_gaps': Mapping of polarization strings to list of guided mode gaps below light line.
            - 'light_line': List of light cone frequencies (omega = |k| / n_clad) if dimension='3D_slab'.
            - 'polarization': Active polarization mode string.
            - 'dimension': Simulation dimension string.
            - 'num_workers': Actual number of worker processes utilized.

    Raises:
        ValueError: If k_points is empty, parameters are missing, or dimension is unsupported.
        TypeError: If input lattice or geometry cannot be parsed.
    """
    # 1. Extract parameters from ms if provided
    if ms is not None:
        if hasattr(ms, "geometry_lattice"):
            geometry_lattice = ms.geometry_lattice
        if hasattr(ms, "geometry"):
            geometry = ms.geometry
        if hasattr(ms, "k_points"):
            k_points = ms.k_points
        if hasattr(ms, "default_material"):
            default_material = ms.default_material
        if hasattr(ms, "resolution"):
            ms_res = ms.resolution
            if isinstance(ms_res, (list, tuple)) and len(ms_res) == 3:
                resolution = (int(ms_res[0]), int(ms_res[1]), int(ms_res[2]))
            else:
                resolution = ms_res
        if hasattr(ms, "num_bands"):
            num_bands = ms.num_bands

    # Validate essential inputs
    if geometry_lattice is None:
        raise ValueError("geometry_lattice must be provided directly or via ms.")
    if geometry is None:
        geometry = []
    if k_points is None or len(k_points) == 0:
        raise ValueError("k_points must be a non-empty sequence of mp.Vector3 objects.")
    if dimension not in ("2D", "3D_slab"):
        raise ValueError(
            f"Unsupported dimension '{dimension}'. Expected '2D' or '3D_slab'."
        )

    # 2. Normalize resolution into Vector3 or scalar
    res_arg: Any
    if resolution_z is not None:
        r_xy = (
            float(resolution[0])
            if isinstance(resolution, (tuple, list, mp.Vector3))
            else float(resolution)
        )
        res_arg = mp.Vector3(r_xy, r_xy, float(resolution_z))
    elif isinstance(resolution, (tuple, list)) and len(resolution) == 3:
        res_arg = mp.Vector3(
            float(resolution[0]), float(resolution[1]), float(resolution[2])
        )
    elif isinstance(resolution, mp.Vector3):
        res_arg = resolution
    else:
        res_arg = int(resolution)

    # Convert k_points to picklable mp.Vector3 list
    k_list: list[mp.Vector3] = []
    for kp in k_points:
        if isinstance(kp, mp.Vector3):
            k_list.append(kp)
        elif isinstance(kp, (list, tuple)) and len(kp) >= 3:
            k_list.append(mp.Vector3(float(kp[0]), float(kp[1]), float(kp[2])))
        else:
            raise TypeError(f"Cannot convert k-point {kp} to mp.Vector3.")

    total_k = len(k_list)

    # 3. Determine worker count & partition k-points into contiguous slices
    cpu_cores = os.cpu_count() or 4
    if num_workers is None or num_workers <= 0:
        workers = min(total_k, cpu_cores)
    else:
        workers = min(total_k, num_workers)

    workers = max(1, workers)

    # Contiguous chunk distribution
    chunk_size = (total_k + workers - 1) // workers
    chunks: list[list[mp.Vector3]] = []
    for i in range(workers):
        start = i * chunk_size
        end = min(start + chunk_size, total_k)
        if start < total_k:
            chunks.append(k_list[start:end])

    actual_workers = len(chunks)

    # 4. Dispatch tasks across ProcessPoolExecutor
    tasks = [
        (
            idx,
            chunk,
            geometry_lattice,
            geometry,
            default_material,
            res_arg,
            num_bands,
            polarization,
            dimension,
            tolerance,
            verbose,
            compute_polarization_fractions,
            polarization_method,
        )
        for idx, chunk in enumerate(chunks)
    ]

    worker_outputs: list[dict[str, Any]]
    if actual_workers == 1:
        # Avoid process pool overhead for single worker
        worker_outputs = [_worker_solve_k_chunk(tasks[0])]
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=actual_workers
        ) as executor:
            worker_outputs = list(executor.map(_worker_solve_k_chunk, tasks))

    # Sort outputs by chunk_idx to guarantee original ordering
    worker_outputs.sort(key=lambda o: o["chunk_idx"])

    # 5. Reassemble frequency and polarization fraction arrays
    pol_keys: set[str] = set()
    for wo in worker_outputs:
        pol_keys.update(wo["freqs"].keys())

    assembled_freqs: dict[str, np.ndarray] = {}
    for key in sorted(pol_keys):
        arrays_to_stack = [
            wo["freqs"][key] for wo in worker_outputs if key in wo["freqs"]
        ]
        if arrays_to_stack:
            assembled_freqs[key] = np.vstack(arrays_to_stack)

    te_frac_arrays = [
        wo["te_fractions"] for wo in worker_outputs if "te_fractions" in wo
    ]
    assembled_te_fracs = np.vstack(te_frac_arrays) if te_frac_arrays else None

    conf_arrays = [wo["confinements"] for wo in worker_outputs if "confinements" in wo]
    assembled_confinements = np.vstack(conf_arrays) if conf_arrays else None

    # 6. Compute band gaps
    gap_info: dict[str, list[float]] = {}
    guided_gaps: dict[str, list[dict[str, Any]]] = {}

    # Cladding light line for 3D slabs
    light_line: list[float] = []
    if dimension == "3D_slab":
        k_cart = [mp.reciprocal_to_cartesian(k, geometry_lattice) for k in k_list]
        k_mag = [float(np.sqrt(k.x**2 + k.y**2)) for k in k_cart]
        light_line = [km / cladding_index for km in k_mag]

    for pol_key, freq_arr in assembled_freqs.items():
        n_bands = freq_arr.shape[1]
        gaps: list[float] = []
        guided_list: list[dict[str, Any]] = []

        for b in range(n_bands - 1):
            top_lower = float(np.max(freq_arr[:, b]))
            bot_upper = float(np.min(freq_arr[:, b + 1]))
            if bot_upper > top_lower:
                gap_pct = 200.0 * (bot_upper - top_lower) / (bot_upper + top_lower)
                gaps.append(gap_pct)
            else:
                gaps.append(0.0)

            # Guided mode gap analysis (modes strictly below light line)
            if light_line:
                ll_arr = np.array(light_line)
                # Guided mask for lower band
                guided_mask_lower = freq_arr[:, b] <= ll_arr
                # Guided mask for upper band
                guided_mask_upper = freq_arr[:, b + 1] <= ll_arr

                if np.any(guided_mask_lower) and np.any(guided_mask_upper):
                    guided_top_lower = float(np.max(freq_arr[guided_mask_lower, b]))
                    guided_bot_upper = float(np.min(freq_arr[guided_mask_upper, b + 1]))
                    if guided_bot_upper > guided_top_lower:
                        g_gap_pct = (
                            200.0
                            * (guided_bot_upper - guided_top_lower)
                            / (guided_bot_upper + guided_top_lower)
                        )
                        guided_list.append(
                            {
                                "bands": [b + 1, b + 2],
                                "lower_freq": guided_top_lower,
                                "upper_freq": guided_bot_upper,
                                "gap_pct": g_gap_pct,
                            }
                        )

        gap_info[pol_key] = gaps
        if guided_list:
            guided_gaps[pol_key] = guided_list

    # Update ms.all_freqs if an ms instance was passed
    if ms is not None and assembled_freqs:
        # Default primary array to first available polarization
        primary_key = next(iter(assembled_freqs))
        with contextlib.suppress(Exception):
            ms.all_freqs = assembled_freqs[primary_key]

    ret: dict[str, Any] = {
        "freqs": assembled_freqs,
        "gaps": gap_info,
        "guided_gaps": guided_gaps,
        "light_line": light_line,
        "polarization": polarization,
        "dimension": dimension,
        "num_workers": actual_workers,
        "cladding_index": cladding_index,
    }
    if assembled_te_fracs is not None:
        ret["te_fractions"] = assembled_te_fracs
    if assembled_confinements is not None:
        ret["confinements"] = assembled_confinements

    return ret
