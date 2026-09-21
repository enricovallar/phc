import contextlib
from typing import Any, Literal

import numpy as np
from phc_materials import to_mpb_medium


def create_mode_solver(
    geometry_lattice: Any,
    geometry: list[Any],
    k_points: list[Any],
    default_material: Any = "si",
    resolution: int = 32,
    num_bands: int = 8,
):
    """Initializes an mpb.ModeSolver instance."""
    import meep as mp
    from meep import mpb

    if isinstance(default_material, str) or hasattr(default_material, "index"):
        bg_medium = to_mpb_medium(default_material)
    elif isinstance(default_material, (int, float)):
        bg_medium = mp.Medium(index=float(default_material))
    else:
        bg_medium = default_material

    return mpb.ModeSolver(
        geometry_lattice=geometry_lattice,
        geometry=geometry,
        k_points=k_points,
        default_material=bg_medium,
        resolution=resolution,
        num_bands=num_bands,
    )


def run_band_solver(
    ms: Any,
    polarization: Literal[
        "te", "tm", "te_like", "tm_like", "even", "odd", "both"
    ] = "te",
    dimension: Literal["2D", "3D_slab"] = "2D",
    cladding_index: float = 1.0,
) -> dict[str, Any]:
    """Executes the MPB ModeSolver with the specified polarization mode.

    Args:
        ms: mpb.ModeSolver instance.
        polarization: "te" or "tm" (2D), "te_like"/"even" or "tm_like"/"odd" (slab), or "both".
        dimension: "2D" or "3D_slab".
        cladding_index: Refractive index of background cladding for light line.

    Returns:
        Dict containing freqs (dict of arrays), gap_info, and light_line.
    """
    if not hasattr(ms, "run_te"):
        raise TypeError(f"Expected an mpb.ModeSolver instance, got {type(ms).__name__}")

    results: dict[str, Any] = {
        "freqs": {},
        "gaps": {},
        "polarization": polarization,
        "dimension": dimension,
    }

    pol = polarization.lower()

    if dimension == "2D":
        if pol in ("te", "both"):
            ms.run_te()
            results["freqs"]["te"] = np.copy(ms.all_freqs)
            results["gaps"]["te"] = [ms.retrieve_gap(b) for b in range(1, ms.num_bands)]
        if pol in ("tm", "both"):
            ms.run_tm()
            results["freqs"]["tm"] = np.copy(ms.all_freqs)
            results["gaps"]["tm"] = [ms.retrieve_gap(b) for b in range(1, ms.num_bands)]
    else:
        # 3D Slab Mirror parity modes
        if pol in ("te_like", "even", "both", "te"):
            ms.run_zeven()
            results["freqs"]["te_like"] = np.copy(ms.all_freqs)
            results["gaps"]["te_like"] = [
                ms.retrieve_gap(b) for b in range(1, ms.num_bands)
            ]
        if pol in ("tm_like", "odd", "both", "tm"):
            ms.run_zodd()
            results["freqs"]["tm_like"] = np.copy(ms.all_freqs)
            results["gaps"]["tm_like"] = [
                ms.retrieve_gap(b) for b in range(1, ms.num_bands)
            ]

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
    resolution: int = 64,
) -> np.ndarray:
    """Retrieves the dielectric function grid from an MPB ModeSolver instance.

    Args:
        ms: MPB ModeSolver instance.
        rectify: Whether to rectify non-orthogonal unit cells into a Cartesian grid (default: True).
        periods: Number of unit cell periods to tile along each axis (default: 3).
        periodicity: Alias for `periods`. If provided, takes precedence over `periods`.
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
        md = mpb.MPBData(rectify=rectify, periods=periods, resolution=resolution)
        return md.convert(eps)
    return np.copy(eps)
