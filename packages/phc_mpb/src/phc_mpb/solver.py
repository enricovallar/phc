import contextlib
from typing import Any, Literal

import numpy as np
from phc_materials import to_mpb_medium
from phc_utils import silence_c_stdout

from phc_mpb.parallel import run_parallel_band_solver


def create_mode_solver(
    geometry_lattice: Any,
    geometry: list[Any],
    k_points: list[Any],
    default_material: Any = "si",
    resolution: int | tuple[int, int, int] | Any = 32,
    resolution_z: int | None = None,
    num_bands: int = 8,
) -> Any:
    """Initializes an mpb.ModeSolver instance with optional anisotropic resolution.

    Args:
        geometry_lattice: mp.Lattice defining unit cell basis vectors and domain size.
        geometry: List of mp.GeometricObject primitives forming the structure.
        k_points: List of mp.Vector3 k-points along the high-symmetry path.
        default_material: Background medium key (e.g. 'si', 'air') or mp.Medium instance.
        resolution: Spatial grid resolution per unit distance a. Can be an integer (isotropic),
            a 3-tuple (rx, ry, rz), or an mp.Vector3.
        resolution_z: Optional vertical resolution per unit distance a along z. If specified,
            forms an anisotropic resolution Vector3(resolution, resolution, resolution_z).
        num_bands: Number of eigenbands to solve at each k-point.

    Returns:
        mpb.ModeSolver initialized instance.
    """
    import meep as mp
    from meep import mpb

    if isinstance(default_material, (int, float)):
        bg_medium = mp.Medium(index=float(default_material))
    elif not isinstance(default_material, mp.Medium):
        bg_medium = to_mpb_medium(default_material)
    else:
        bg_medium = default_material

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

    return mpb.ModeSolver(
        geometry_lattice=geometry_lattice,
        geometry=geometry,
        k_points=k_points,
        default_material=bg_medium,
        resolution=res_arg,
        num_bands=num_bands,
    )


from phc_mpb.classification import compute_polarization_fractions


def run_band_solver(
    ms: Any,
    polarization: Literal[
        "te", "tm", "te_like", "tm_like", "even", "odd", "both", "all", "no_parity"
    ] = "te",
    dimension: Literal["2D", "3D_slab"] = "2D",
    cladding_index: float = 1.0,
    num_workers: int = 1,
    resolution_z: int | None = None,
    compute_fractions: bool = False,
    compute_group_velocities: bool = False,
    compute_symmetries: bool = False,
    symmetry_group: str = "C4v",
    verbose: bool = False,
    slab_thickness: float | None = None,
    z_center: float = 0.0,
    polarization_method: Literal[
        "midplane", "volumetric", "slab", "magnetic"
    ] = "midplane",
) -> dict[str, Any]:
    """Executes the MPB ModeSolver with the specified polarization mode and optional parallelism.

    Args:
        ms: mpb.ModeSolver instance.
        polarization: "te" or "tm" (2D), "te_like"/"even" or "tm_like"/"odd" (slab), "both",
            or "all"/"no_parity" (solve all modes without parity constraint, e.g. asymmetric slabs).
        dimension: "2D" or "3D_slab".
        cladding_index: Refractive index of background cladding for light line.
        num_workers: Number of concurrent worker processes for parallel k-point execution.
        resolution_z: Optional vertical resolution override when running in parallel.
        compute_fractions: Whether to compute modal TE/TM energy fractions at each k-point.
            Automatically enabled when polarization is 'all' or 'no_parity'.
        compute_group_velocities: Whether to compute group velocity vectors at each k-point.
        compute_symmetries: Whether to analyze point-group symmetries and irreps at Gamma (k=0).
        symmetry_group: Point group name ('C4v' or 'C6v') when compute_symmetries is True.
        verbose: If True, streams MPB C-level iteration and band output to stdout.
            If False (default), silences solver chatter for clean execution.
        slab_thickness: Optional normalized slab thickness in units of lattice constant a.
            When provided, restricts field integration along z to the dielectric slab core
            (|z - z_center| <= slab_thickness / 2), preventing dilution from substrate cladding.
        z_center: Vertical center coordinate of the slab core in units of lattice constant a (default: 0.0).
        polarization_method: Evaluation method ('midplane', 'volumetric', 'slab', or 'magnetic'). Default is 'midplane'.

    Returns:
        Dict containing freqs (dict of arrays), gaps, light_line, polarization, dimension,
        and optionally 'te_fractions', 'group_velocities' (dict of arrays), and 'symmetries' (dict).
    """
    if num_workers > 1:
        return run_parallel_band_solver(
            ms=ms,
            polarization=polarization,
            dimension=dimension,
            cladding_index=cladding_index,
            num_workers=num_workers,
            resolution_z=resolution_z,
            compute_polarization_fractions=compute_fractions,
            verbose=verbose,
            polarization_method=polarization_method,
            slab_thickness=slab_thickness,
            z_center=z_center,
        )
    if not hasattr(ms, "run_te"):
        raise TypeError(f"Expected an mpb.ModeSolver instance, got {type(ms).__name__}")

    from phc_mpb.symmetry import compute_band_symmetries

    results: dict[str, Any] = {
        "freqs": {},
        "gaps": {},
        "polarization": polarization,
        "dimension": dimension,
    }
    if compute_group_velocities:
        results["group_velocities"] = {}
    if compute_symmetries:
        results["symmetries"] = {}

    pol = polarization.lower()
    should_compute_fracs = compute_fractions or pol in ("all", "no_parity")

    def _execute_run(run_fn: Any, pol_key: str) -> None:
        te_fractions_list: list[list[float]] = []
        vg_list: list[list[list[float]]] = []
        sym_records: list[dict[str, Any]] = []

        callbacks = []

        if should_compute_fracs:

            def _fraction_callback(solver: Any) -> None:
                fracs = compute_polarization_fractions(
                    solver,
                    method=polarization_method,
                    slab_thickness=slab_thickness,
                    z_center=z_center,
                )
                if isinstance(fracs, list):
                    te_fractions_list.append([f["te"] for f in fracs])

            callbacks.append(_fraction_callback)

        if compute_group_velocities:

            def _vg_callback(solver: Any) -> None:
                vgs = solver.compute_group_velocities()
                vg_list.append([[float(v.x), float(v.y), float(v.z)] for v in vgs])

            callbacks.append(_vg_callback)

        if compute_symmetries:

            def _sym_callback(solver: Any) -> None:
                if hasattr(solver, "current_k") and solver.current_k.norm() < 1e-4:
                    records = compute_band_symmetries(
                        solver, symmetry_group=symmetry_group
                    )
                    sym_records.extend(records)

            callbacks.append(_sym_callback)

        if not verbose:
            with silence_c_stdout():
                run_fn(*callbacks)
        else:
            run_fn(*callbacks)

        results["freqs"][pol_key] = np.copy(ms.all_freqs)
        gaps_list = []
        for b in range(1, ms.num_bands):
            try:
                g = ms.retrieve_gap(b)
            except ZeroDivisionError:
                g = 0.0
            gaps_list.append(g)
        results["gaps"][pol_key] = gaps_list
        if should_compute_fracs and te_fractions_list:
            results["te_fractions"] = np.array(te_fractions_list)
        if compute_group_velocities and vg_list:
            results["group_velocities"][pol_key] = np.array(vg_list)
        if compute_symmetries and sym_records:
            results["symmetries"][pol_key] = sym_records

    if dimension == "2D":
        if pol in ("te", "both"):
            _execute_run(ms.run_te, "te")
        if pol in ("tm", "both"):
            _execute_run(ms.run_tm, "tm")
        if pol in ("all", "no_parity"):
            _execute_run(ms.run, "all")
    else:
        # 3D Slab modes
        if pol in ("te_like", "even", "both", "te"):
            _execute_run(ms.run_zeven, "te_like")
        if pol in ("tm_like", "odd", "both", "tm"):
            _execute_run(ms.run_zodd, "tm_like")
        if pol in ("all", "no_parity"):
            _execute_run(ms.run, "all")

    # Compute cladding light line for 3D slabs
    if dimension == "3D_slab" and hasattr(ms, "k_points"):
        import meep as mp

        lat = ms.geometry_lattice
        k_cart = [mp.reciprocal_to_cartesian(k, lat) for k in ms.k_points]
        k_mag = [np.sqrt(k.x**2 + k.y**2) for k in k_cart]
        results["light_line"] = [k / cladding_index for k in k_mag]

    return results


def get_epsilon_grid(
    ms: Any,
    rectify: bool = True,
    periods: int = 3,
    periodicity: int | None = None,
    periods_z: int | None = None,
    resolution: int = 64,
) -> np.ndarray:
    """Retrieves the dielectric function grid from an MPB ModeSolver instance.

    Args:
        ms: MPB ModeSolver instance.
        rectify: Whether to rectify non-orthogonal unit cells into a Cartesian grid (default: True).
        periods: Number of in-plane unit cell periods to tile along x and y axes (default: 3).
        periodicity: Alias for `periods`. If provided, takes precedence over `periods`.
        periods_z: Number of unit cell periods along z. For 3D slab arrays, defaults to 1 so
            that the single finite slab membrane and cladding buffer are not repeated vertically.
        resolution: Target resolution (grid points per unit distance a, default: 64).

    Returns:
        Converted Cartesian numpy array of dielectric permittivity.
    """
    if not hasattr(ms, "get_epsilon"):
        raise TypeError(f"Expected an mpb.ModeSolver instance, got {type(ms).__name__}")

    if periodicity is not None:
        periods = periodicity

    import meep as mp
    from meep import mpb

    with contextlib.suppress(RuntimeError, ValueError):
        ms.init_params(mp.NO_PARITY, True)

    eps = ms.get_epsilon()

    if rectify:
        if eps.ndim == 3:
            pz = periods_z if periods_z is not None else 1
            md = mpb.MPBData(
                rectify=rectify, x=periods, y=periods, z=pz, resolution=resolution
            )
        else:
            md = mpb.MPBData(rectify=rectify, periods=periods, resolution=resolution)
        return md.convert(eps)
    return np.copy(eps)
