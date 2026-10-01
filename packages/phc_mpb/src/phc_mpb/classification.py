"""Physical mode classification and modal field metrics for photonic band solvers.

This module provides tools for physical characterization of electromagnetic eigenmodes:
- Polarization energy fractions: in-plane (TE) vs. out-of-plane (TM).
- Modal energy confinement: core slab energy vs. total supercell energy.
- Mode regime classification: Guided (TIR bound) vs. Leaky (resonant) vs. Radiation (continuum artifact).
- Unified single-pass modal metrics extraction from ModeSolver eigenfields.
"""

import itertools
from collections import Counter
from collections.abc import Sequence
from enum import StrEnum
from typing import Any, Literal

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import linear_sum_assignment


class ModeClassification(StrEnum):
    """Enumeration of physical mode confinement regimes in photonic crystal slabs."""

    GUIDED = "guided"
    """Bound state below the light line (omega < omega_light) with high core confinement."""

    LEAKY = "leaky"
    """Resonant state inside the light cone (omega >= omega_light) localized in the slab."""

    RADIATION = "radiation"
    """Unguided continuum state localized predominantly in the artificial supercell buffer."""


def compute_polarization_fractions(
    ms: Any,
    band_idx: int | None = None,
    method: Literal["midplane", "volumetric", "slab", "magnetic"] = "midplane",
    slab_thickness: float | None = None,
    z_center: float = 0.0,
) -> dict[str, float] | list[dict[str, float]]:
    """Calculates the TE and TM electromagnetic energy fractions for eigenmodes at the current k-point.

    The TE fraction evaluates the degree of in-plane (transverse) polarization relative to
    out-of-plane (longitudinal) polarization:

        f_TE = u_TE / (u_TE + u_TM)
        f_TM = 1 - f_TE

    For 3D slabs, four evaluation methods are supported:
    - 'midplane' (default): Evaluates electric field energy at the slab midplane z = z_mid (z=0).
      For symmetric and asymmetric slabs, in-plane electric fields of TM-like modes vanish or are
      drastically suppressed at the core midplane, yielding f_TE ~ 1.0 (pure TE) and f_TE ~ 0.0 (pure TM).
      Avoids the unphysical dilution caused by integrating over tall empty air cladding.
    - 'volumetric': Integrates electric field energy Re(E* . D) across the entire 3D computational volume.
    - 'slab': Restricts electric energy integration to the high-index slab region (epsilon > 1.5).
    - 'magnetic': Evaluates the magnetic field energy components Hz^2 vs (Hx^2 + Hy^2). Note that
      near Gamma (k_parallel -> 0), Hz -> 0 for normal-incidence waves, so 'midplane' is preferred.

    When `slab_thickness` is provided, field integration is strictly restricted along z to the
    dielectric slab core (|z - z_center| <= slab_thickness / 2), preventing field dilution from
    substrate cladding or unguided radiative components in asymmetric slabs.

    For 2D simulations, 'midplane' and 'volumetric' are identical.

    Args:
        ms: mpb.ModeSolver instance with solved eigenfields at the current k-point.
        band_idx: Optional 1-based band index (1 to num_bands). If specified, calculates
            fractions for this single band. If None, calculates fractions for all solved bands.
        method: Evaluation method ('midplane', 'volumetric', 'slab', or 'magnetic'). Default is 'midplane'.
        slab_thickness: Optional normalized slab thickness in units of lattice constant a.
            When provided, restricts field integration to |z - z_center| <= slab_thickness / 2.
            If None, falls back to unmasked evaluation according to `method`.
        z_center: Vertical center coordinate of the slab core in units of lattice constant a (default: 0.0).

    Returns:
        If band_idx is provided: Dict with keys 'te' and 'tm' (floats in [0.0, 1.0]).
        If band_idx is None: List of dicts for each band from 1 to ms.num_bands.

    Raises:
        ValueError: If band_idx is outside [1, ms.num_bands] or method is unrecognized.
        RuntimeError: If fields are not available or solver has not run.
    """
    metrics = compute_modal_metrics(
        ms,
        band_idx=band_idx,
        polarization_method=method,
        slab_thickness=slab_thickness,
        z_center=z_center,
    )
    if band_idx is not None and isinstance(metrics, dict):
        return {"te": metrics["te"], "tm": metrics["tm"]}

    if isinstance(metrics, list):
        return [{"te": m["te"], "tm": m["tm"]} for m in metrics]

    raise RuntimeError("Unexpected modal metrics return type.")


def compute_slab_confinement(
    ms: Any,
    band_idx: int | None = None,
    min_eps: float = 1.5,
    slab_thickness: float | None = None,
    z_center: float = 0.0,
) -> float | list[float]:
    """Calculates the fraction of modal electric energy confined within the high-index core slab.

    Evaluates:
        eta_slab = integral_{slab} Re(E* . D) dV / integral_{cell} Re(E* . D) dV

    Guided modes typically exhibit eta_slab in [0.70, 0.98], leaky resonant modes exhibit
    eta_slab in [0.20, 0.85], while spurious supercell radiation artifacts have eta_slab < 0.20.

    Args:
        ms: mpb.ModeSolver instance with solved eigenfields.
        band_idx: Optional 1-based band index (1 to num_bands). If None, computes for all bands.
        min_eps: Permittivity threshold identifying the core slab (default: 1.5).
        slab_thickness: Optional normalized slab thickness in units of lattice constant a.
            When provided, restricts the core slab region along z to |z - z_center| <= slab_thickness / 2.
        z_center: Vertical center coordinate of the slab core in units of lattice constant a (default: 0.0).

    Returns:
        Float in [0.0, 1.0] if band_idx is specified, or list of floats for all bands.

    Raises:
        ValueError: If band_idx is outside valid range.
        RuntimeError: If eigenfields cannot be retrieved.
    """
    metrics = compute_modal_metrics(
        ms,
        band_idx=band_idx,
        min_eps=min_eps,
        slab_thickness=slab_thickness,
        z_center=z_center,
    )
    if band_idx is not None and isinstance(metrics, dict):
        return metrics["confinement"]

    if isinstance(metrics, list):
        return [m["confinement"] for m in metrics]

    raise RuntimeError("Unexpected modal metrics return type.")


def compute_modal_metrics(
    ms: Any,
    band_idx: int | None = None,
    polarization_method: Literal[
        "midplane", "volumetric", "slab", "magnetic"
    ] = "midplane",
    min_eps: float = 1.5,
    slab_thickness: float | None = None,
    z_center: float = 0.0,
) -> dict[str, float] | list[dict[str, float]]:
    """Calculates both TE/TM polarization fractions and slab core energy confinement in a single pass.

    Retrieves eigenfields once per band, evaluating modal polarization and core confinement
    simultaneously to avoid redundant field queries and memory allocations.

    When `slab_thickness` is provided, field integration is strictly restricted along z to the
    dielectric slab core (|z - z_center| <= slab_thickness / 2), preventing field dilution from
    substrate cladding or unguided radiative components in asymmetric slabs.

    Args:
        ms: mpb.ModeSolver instance with solved eigenfields at the current k-point.
        band_idx: Optional 1-based band index (1 to num_bands). If None, computes for all bands.
        polarization_method: Method for polarization fraction calculation
            ('midplane', 'volumetric', 'slab', or 'magnetic'). Default is 'midplane'.
        min_eps: Permittivity threshold defining the core slab region (default: 1.5).
        slab_thickness: Optional normalized slab thickness in units of lattice constant a.
            When provided, restricts the z-domain: |z - z_center| <= slab_thickness / 2.
            If None, falls back to unmasked evaluation according to `polarization_method`.
        z_center: Vertical center coordinate of the slab core in units of lattice constant a (default: 0.0).

    Returns:
        If band_idx is provided: Dict with keys 'te', 'tm', and 'confinement' (floats in [0.0, 1.0]).
        If band_idx is None: List of dicts for each band from 1 to ms.num_bands.

    Raises:
        ValueError: If band_idx is outside [1, ms.num_bands] or polarization_method is unrecognized.
        RuntimeError: If fields are not available or solver has not run.
    """
    valid_methods = ("midplane", "volumetric", "slab", "magnetic")
    if polarization_method not in valid_methods:
        raise ValueError(
            f"Invalid polarization_method '{polarization_method}'. Must be one of {valid_methods}."
        )

    num_bands = getattr(ms, "num_bands", 1)

    def _calc_single_band(b: int) -> dict[str, float]:
        if b < 1 or b > num_bands:
            raise ValueError(f"band_idx {b} out of range [1, {num_bands}]")

        # 1. Retrieve Electric & Displacement Fields
        try:
            efield = ms.get_efield(b)
            dfield = ms.get_dfield(b)
        except (AttributeError, RuntimeError, TypeError) as exc:
            raise RuntimeError(
                f"Failed to retrieve eigenfields for band {b}. Ensure ModeSolver has solved this k-point."
            ) from exc

        # 2. Confinement: energy in high-index core (eps > min_eps) vs total volume
        u_pt = np.real(np.conj(efield) * dfield).sum(axis=-1)
        u_cell = float(u_pt.sum())

        nz = efield.shape[2] if efield.ndim >= 3 else 1
        has_core_mask = slab_thickness is not None and nz > 1
        z_mask = None
        if has_core_mask:
            assert slab_thickness is not None
            sz = 1.0
            if hasattr(ms, "geometry_lattice") and hasattr(ms.geometry_lattice, "size"):
                lat_sz = getattr(ms.geometry_lattice.size, "z", 0.0)
                if lat_sz > 0:
                    sz = float(lat_sz)
            z_coords = (np.arange(nz) / nz - 0.5) * sz
            z_mask = np.abs(z_coords - z_center) <= (0.5 * slab_thickness + 1e-9 * sz)
            if not np.any(z_mask):
                z_mask[int(np.argmin(np.abs(z_coords - z_center)))] = True

        try:
            eps = ms.get_epsilon()
            if has_core_mask and z_mask is not None:
                in_slab = (eps > min_eps) & z_mask[np.newaxis, np.newaxis, :]
            else:
                in_slab = eps > min_eps
            u_slab = float(u_pt[in_slab].sum())
        except (AttributeError, RuntimeError, ValueError):
            eps = None
            if has_core_mask and z_mask is not None:
                u_slab = float(u_pt[:, :, z_mask].sum())
            else:
                u_slab = u_cell

        confinement = float(np.clip(u_slab / u_cell, 0.0, 1.0)) if u_cell > 0 else 0.0

        # 3. Polarization Fractions
        if polarization_method == "magnetic":
            try:
                hfield = ms.get_hfield(b)
            except (AttributeError, RuntimeError, TypeError) as exc:
                raise RuntimeError(f"Failed to retrieve H-field for band {b}.") from exc
            if has_core_mask and z_mask is not None:
                hf = hfield[:, :, z_mask, :]
            else:
                hf = hfield
            u_hxy = float(np.sum(np.abs(hf[..., 0]) ** 2 + np.abs(hf[..., 1]) ** 2))
            u_hz = float(np.sum(np.abs(hf[..., 2]) ** 2))
            u_tot_h = u_hxy + u_hz
            f_te = float(np.clip(u_hz / u_tot_h, 0.0, 1.0)) if u_tot_h > 0 else 0.5
            return {"te": f_te, "tm": 1.0 - f_te, "confinement": confinement}

        if has_core_mask and z_mask is not None:
            ef_core = efield[:, :, z_mask, :]
            if eps is not None:
                eps_core = eps[:, :, z_mask]
                u_inplane = float(
                    np.sum(
                        eps_core
                        * (np.abs(ef_core[..., 0]) ** 2 + np.abs(ef_core[..., 1]) ** 2)
                    )
                )
                u_z = float(np.sum(eps_core * (np.abs(ef_core[..., 2]) ** 2)))
            else:
                df_core = dfield[:, :, z_mask, :]
                u_x = float(np.real(np.conj(ef_core[..., 0]) * df_core[..., 0]).sum())
                u_y = float(np.real(np.conj(ef_core[..., 1]) * df_core[..., 1]).sum())
                u_z = float(np.real(np.conj(ef_core[..., 2]) * df_core[..., 2]).sum())
                u_inplane = u_x + u_y
            u_tot_e = u_inplane + u_z
        else:
            if polarization_method == "midplane" and nz > 1:
                mid = nz // 2
                ef = efield[:, :, mid : mid + 1, :]
                df = dfield[:, :, mid : mid + 1, :]
            elif polarization_method == "slab" and nz > 1 and eps is not None:
                ef = efield[in_slab, :]
                df = dfield[in_slab, :]
            else:
                ef = efield
                df = dfield

            u_x = float(np.real(np.conj(ef[..., 0]) * df[..., 0]).sum())
            u_y = float(np.real(np.conj(ef[..., 1]) * df[..., 1]).sum())
            u_z = float(np.real(np.conj(ef[..., 2]) * df[..., 2]).sum())
            u_inplane = u_x + u_y
            u_tot_e = u_inplane + u_z

        if u_tot_e <= 0 or not np.isfinite(u_tot_e):
            f_te = 0.5
            f_tm = 0.5
        else:
            f_te = float(np.clip(u_inplane / u_tot_e, 0.0, 1.0))
            f_tm = float(np.clip(u_z / u_tot_e, 0.0, 1.0))

        return {"te": f_te, "tm": f_tm, "confinement": confinement}

    if band_idx is not None:
        return _calc_single_band(band_idx)

    return [_calc_single_band(b) for b in range(1, num_bands + 1)]


def classify_modes(
    freqs: np.ndarray,
    confinements: np.ndarray,
    light_line: Sequence[float],
    cutoff: float = 0.20,
    full_threshold: float = 0.50,
) -> np.ndarray:
    """Classifies every eigenmode across k-points and bands into its physical regime.

    Classification rules:
    - GUIDED: Lies below the light line (freq < light_line[k]) and has high slab confinement (eta >= full_threshold).
    - LEAKY: Resonates inside the light cone (freq >= light_line[k]) with slab confinement (eta >= cutoff).
    - RADIATION: Supercell continuum mode localized in the artificial air buffer (eta < cutoff).

    Args:
        freqs: 2D array of shape (num_k, num_bands) containing normalized eigenfrequencies.
        confinements: 2D array of shape (num_k, num_bands) containing slab energy confinement factors in [0, 1].
        light_line: 1D sequence of length num_k containing light line cutoff frequencies.
        cutoff: Confinement threshold below which states are considered radiation artifacts (default: 0.20).
        full_threshold: Confinement threshold above which modes are fully localized (default: 0.50).

    Returns:
        2D numpy array of shape (num_k, num_bands) with string values ('guided', 'leaky', 'radiation').

    Raises:
        ValueError: If freqs and confinements shapes do not match, or light_line length differs from num_k.
    """
    if freqs.shape != confinements.shape:
        raise ValueError(
            f"Shape mismatch: freqs {freqs.shape} vs confinements {confinements.shape}."
        )
    num_k, num_bands = freqs.shape
    if len(light_line) != num_k:
        raise ValueError(
            f"Light line length {len(light_line)} does not match num_k {num_k}."
        )

    ll_arr = np.array(light_line)[:, np.newaxis]  # shape (num_k, 1)
    classifications = np.empty((num_k, num_bands), dtype=object)

    below_light_line = freqs < ll_arr
    is_artifact = confinements < cutoff
    is_guided = below_light_line & (confinements >= cutoff)
    is_leaky = (~below_light_line) & (~is_artifact)

    classifications[is_guided] = ModeClassification.GUIDED.value
    classifications[is_leaky] = ModeClassification.LEAKY.value
    classifications[is_artifact] = ModeClassification.RADIATION.value

    return classifications


def summarize_mode_physics(
    freqs: np.ndarray,
    confinements: np.ndarray,
    light_line: Sequence[float],
    cutoff: float = 0.20,
    full_threshold: float = 0.50,
) -> dict[str, Any]:
    """Generates summary counts and statistics for physical mode regimes in a simulation.

    Args:
        freqs: 2D array of eigenfrequencies of shape (num_k, num_bands).
        confinements: 2D array of modal confinements of shape (num_k, num_bands).
        light_line: Sequence of light line frequencies along the k-path.
        cutoff: Confinement threshold for radiation artifacts (default: 0.20).
        full_threshold: Confinement threshold for full localization (default: 0.50).

    Returns:
        Dict containing total counts and ratios for guided, leaky, and radiation modes:
        {
            'total_modes': int,
            'num_guided': int,
            'num_leaky': int,
            'num_radiation': int,
            'guided_ratio': float,
            'leaky_ratio': float,
            'radiation_ratio': float,
            'mean_confinement_guided': float,
            'mean_confinement_leaky': float,
        }
    """
    classes = classify_modes(
        freqs=freqs,
        confinements=confinements,
        light_line=light_line,
        cutoff=cutoff,
        full_threshold=full_threshold,
    )
    total = int(classes.size)
    n_guided = int(np.sum(classes == ModeClassification.GUIDED.value))
    n_leaky = int(np.sum(classes == ModeClassification.LEAKY.value))
    n_rad = int(np.sum(classes == ModeClassification.RADIATION.value))

    conf_guided = (
        confinements[classes == ModeClassification.GUIDED.value] if n_guided > 0 else []
    )
    conf_leaky = (
        confinements[classes == ModeClassification.LEAKY.value] if n_leaky > 0 else []
    )

    return {
        "total_modes": total,
        "num_guided": n_guided,
        "num_leaky": n_leaky,
        "num_radiation": n_rad,
        "guided_ratio": float(n_guided / total) if total > 0 else 0.0,
        "leaky_ratio": float(n_leaky / total) if total > 0 else 0.0,
        "radiation_ratio": float(n_rad / total) if total > 0 else 0.0,
        "mean_confinement_guided": (
            float(np.mean(conf_guided)) if len(conf_guided) > 0 else 0.0
        ),
        "mean_confinement_leaky": (
            float(np.mean(conf_leaky)) if len(conf_leaky) > 0 else 0.0
        ),
        "classification_matrix": classes.tolist(),
    }


def _get_supercell_z(ms: Any) -> float:
    """Extracts supercell height along the z-axis from ModeSolver lattice."""
    if hasattr(ms, "geometry_lattice") and hasattr(ms.geometry_lattice, "size"):
        lat_sz = getattr(ms.geometry_lattice.size, "z", 0.0)
        if lat_sz > 0:
            return float(lat_sz)
    return 1.0


def _compute_z_mask(
    nz: int,
    supercell_z: float = 1.0,
    slab_thickness: float | None = None,
    z_center: float = 0.0,
) -> np.ndarray | None:
    """Computes a 1D boolean mask along the z-axis identifying the slab core."""
    if slab_thickness is None or nz <= 1:
        return None

    z_coords = (np.arange(nz) / nz - 0.5) * supercell_z
    z_mask = np.abs(z_coords - z_center) <= (0.5 * slab_thickness + 1e-9 * supercell_z)
    if not np.any(z_mask):
        z_mask[int(np.argmin(np.abs(z_coords - z_center)))] = True
    return z_mask


def extract_midplane_field(field_arr: np.ndarray | None) -> np.ndarray | None:
    """Extracts the 2D mid-plane (z = 0) spatial slice from a 3D/4D eigenmode field array.

    The symmetry of photonic crystal slab modes (e.g. C6v A2 vortex and E1 dipoles)
    is fully encoded in the 2D mid-plane slice at z = 0. Extracting this 2D slice
    ensures field dimension compatibility (shape: (Nx, Ny, 3)) across simulations
    with different slab thicknesses h/a or vertical discretization resolutions.

    Args:
        field_arr: Complex or real eigenmode field array of shape (Nx, Ny, Nz, 3),
            or 2D slice array of shape (Nx, Ny, 3), or None.

    Returns:
        2D mid-plane field array of shape (Nx, Ny, 3), or None if field_arr is None.
    """
    if field_arr is None:
        return None
    if field_arr.ndim == 4:
        nz = field_arr.shape[2]
        return field_arr[:, :, nz // 2, :]
    if field_arr.ndim == 3 and field_arr.shape[-1] == 3:
        return field_arr
    if field_arr.ndim == 3:
        nz = field_arr.shape[2]
        return field_arr[:, :, nz // 2]
    return field_arr


def extract_eigenmode_fields(
    ms: Any,
    band: int,
    field: Literal[
        "electric_displacement", "electric", "displacement", "magnetic"
    ] = "electric_displacement",
    slab_thickness: float | None = None,
    z_center: float = 0.0,
    overlap_mode: Literal["midplane", "slab", "full"] = "midplane",
) -> tuple[np.ndarray, np.ndarray | None]:
    """Retrieves and optionally masks eigenfield arrays for a specific band.

    Args:
        ms: mpb.ModeSolver instance with solved eigenfields.
        band: 1-based band index.
        field: Type of fields to retrieve ('electric_displacement', 'electric',
            'displacement', or 'magnetic').
        slab_thickness: Optional normalized slab thickness in units of lattice constant a.
            Used when overlap_mode='slab'.
        z_center: Vertical center coordinate of slab core (default: 0.0).
        overlap_mode: Spatial domain extraction formulation:
            - 'midplane' (default): 2D mid-plane slice at z = z_center (shape: (Nx, Ny, 3)),
              invariant to slab thickness h/a and vertical grid resolution differences.
            - 'slab': 3D masked slab core |z - z_center| <= slab_thickness / 2.
            - 'full': Full 3D supercell without vertical cropping.

    Returns:
        Tuple of (primary_field, secondary_field_or_None).

    Raises:
        ValueError: If overlap_mode is unrecognized or field type is invalid.
        RuntimeError: If field arrays cannot be retrieved from ModeSolver.
    """
    if overlap_mode not in ("midplane", "slab", "full"):
        raise ValueError(
            f"Unknown overlap_mode '{overlap_mode}'. Supported modes: 'midplane', 'slab', 'full'."
        )

    if isinstance(ms, dict):
        val = ms[band]
        if isinstance(val, tuple):
            e_or_h, d = val
        else:
            e_or_h, d = val, None
        if overlap_mode == "midplane":
            e_or_h = extract_midplane_field(e_or_h)
            d = extract_midplane_field(d) if d is not None else None
        return e_or_h, d

    sz = _get_supercell_z(ms)

    if field == "electric_displacement":
        try:
            e = ms.get_efield(band)
            d = ms.get_dfield(band)
        except (AttributeError, RuntimeError, TypeError) as exc:
            raise RuntimeError(
                f"Failed to retrieve E/D fields for band {band}."
            ) from exc
        if overlap_mode == "midplane":
            e = extract_midplane_field(e)
            d = extract_midplane_field(d)
        elif (
            overlap_mode == "slab"
            and slab_thickness is not None
            and e.ndim >= 3
            and e.shape[2] > 1
        ):
            z_mask = _compute_z_mask(
                nz=e.shape[2],
                supercell_z=sz,
                slab_thickness=slab_thickness,
                z_center=z_center,
            )
            if z_mask is not None:
                e = e[:, :, z_mask, :]
                d = d[:, :, z_mask, :]
        return e, d

    if field == "electric":
        try:
            e = ms.get_efield(band)
        except (AttributeError, RuntimeError, TypeError) as exc:
            raise RuntimeError(f"Failed to retrieve E-field for band {band}.") from exc
        if overlap_mode == "midplane":
            e = extract_midplane_field(e)
        elif (
            overlap_mode == "slab"
            and slab_thickness is not None
            and e.ndim >= 3
            and e.shape[2] > 1
        ):
            z_mask = _compute_z_mask(
                nz=e.shape[2],
                supercell_z=sz,
                slab_thickness=slab_thickness,
                z_center=z_center,
            )
            if z_mask is not None:
                e = e[:, :, z_mask, :]
        return e, None

    if field == "displacement":
        try:
            d = ms.get_dfield(band)
        except (AttributeError, RuntimeError, TypeError) as exc:
            raise RuntimeError(f"Failed to retrieve D-field for band {band}.") from exc
        if overlap_mode == "midplane":
            d = extract_midplane_field(d)
        elif (
            overlap_mode == "slab"
            and slab_thickness is not None
            and d.ndim >= 3
            and d.shape[2] > 1
        ):
            z_mask = _compute_z_mask(
                nz=d.shape[2],
                supercell_z=sz,
                slab_thickness=slab_thickness,
                z_center=z_center,
            )
            if z_mask is not None:
                d = d[:, :, z_mask, :]
        return d, None

    if field == "magnetic":
        try:
            h = ms.get_hfield(band)
        except (AttributeError, RuntimeError, TypeError) as exc:
            raise RuntimeError(f"Failed to retrieve H-field for band {band}.") from exc
        if overlap_mode == "midplane":
            h = extract_midplane_field(h)
        elif (
            overlap_mode == "slab"
            and slab_thickness is not None
            and h.ndim >= 3
            and h.shape[2] > 1
        ):
            z_mask = _compute_z_mask(
                nz=h.shape[2],
                supercell_z=sz,
                slab_thickness=slab_thickness,
                z_center=z_center,
            )
            if z_mask is not None:
                h = h[:, :, z_mask, :]
        return h, None

    raise ValueError(
        f"Unknown field type '{field}'. Must be 'electric_displacement', 'electric', 'displacement', or 'magnetic'."
    )


_extract_mode_fields = extract_eigenmode_fields


def interpolate_field_to_grid(
    field: np.ndarray,
    target_shape: tuple[int, ...],
) -> np.ndarray:
    """Interpolates a 2D or 3D scalar or complex vector field to match a target spatial mesh grid.

    Maps periodic in-plane dimensions across unit cell boundaries using periodic boundary
    padding, and resamples vertical slab/supercell coordinates to the target grid dimensions.
    Enables cross-simulation modal overlaps and mode tracking between different numerical mesh
    resolutions or vertical slice counts.

    Args:
        field: Input field numpy array of shape (Nx, Ny, 3) or (Nx, Ny, Nz, 3) for vector fields,
            or (Nx, Ny) / (Nx, Ny, Nz) for scalar fields. May be float or complex.
        target_shape: Desired output array shape tuple (e.g. (Nx_tar, Ny_tar, 3) or (Nx_tar, Ny_tar, Nz_tar, 3)).

    Returns:
        Interpolated numpy array of exact shape `target_shape` with identical dtype to `field`.

    Raises:
        ValueError: If `field` and `target_shape` have incompatible spatial rank dimensions.
    """
    if field.shape == target_shape:
        return field

    if field.ndim == 4 and len(target_shape) == 3:
        field = extract_midplane_field(field)
        if field.shape == target_shape:
            return field

    if field.ndim != len(target_shape):
        raise ValueError(
            f"Cannot interpolate field of shape {field.shape} to incompatible target shape {target_shape}"
        )

    is_vector = field.ndim >= 2 and field.shape[-1] == 3 and target_shape[-1] == 3
    spatial_src = field.shape[:-1] if is_vector else field.shape
    spatial_tar = target_shape[:-1] if is_vector else target_shape
    ndim = len(spatial_src)

    # Pad periodic dimensions (axes 0 and 1) to handle periodic boundary conditions
    pad_width = [(0, 1) if d < 2 else (0, 0) for d in range(ndim)]
    if is_vector:
        pad_width.append((0, 0))

    if np.iscomplexobj(field):
        field_pad = np.pad(field.real, pad_width, mode="wrap") + 1j * np.pad(
            field.imag, pad_width, mode="wrap"
        )
    else:
        field_pad = np.pad(field, pad_width, mode="wrap")

    coords_1d = []
    for d in range(ndim):
        Ns = spatial_src[d]
        Nt = spatial_tar[d]
        if d < 2:
            # In-plane periodic: [0, 1) mapped to continuous source indices [0, Ns)
            u = np.linspace(0.0, 1.0, Nt, endpoint=False)
            coords_1d.append(u * Ns)
        else:
            # Vertical dimension: [0, 1] mapped across slab/supercell
            if Nt <= 1:
                coords_1d.append(np.full(Nt, (Ns - 1) / 2.0))
            elif Ns <= 1:
                coords_1d.append(np.zeros(Nt))
            else:
                u = np.linspace(0.0, 1.0, Nt, endpoint=True)
                coords_1d.append(u * (Ns - 1))

    grids = np.meshgrid(*coords_1d, indexing="ij")
    coords = np.array([g.ravel() for g in grids])

    out = np.zeros(target_shape, dtype=field.dtype)
    if is_vector:
        for c in range(3):
            fc = field_pad[..., c]
            if np.iscomplexobj(fc):
                r = map_coordinates(fc.real, coords, order=1, mode="nearest").reshape(
                    spatial_tar
                )
                im = map_coordinates(fc.imag, coords, order=1, mode="nearest").reshape(
                    spatial_tar
                )
                out[..., c] = r + 1j * im
            else:
                out[..., c] = map_coordinates(
                    fc, coords, order=1, mode="nearest"
                ).reshape(spatial_tar)
    else:
        if np.iscomplexobj(field_pad):
            r = map_coordinates(
                field_pad.real, coords, order=1, mode="nearest"
            ).reshape(spatial_tar)
            im = map_coordinates(
                field_pad.imag, coords, order=1, mode="nearest"
            ).reshape(spatial_tar)
            out = r + 1j * im
        else:
            out = map_coordinates(field_pad, coords, order=1, mode="nearest").reshape(
                spatial_tar
            )

    return out


def compute_mode_overlap(
    ms1: Any,
    band1: int,
    ms2: Any,
    band2: int,
    field: Literal[
        "electric_displacement", "electric", "displacement", "magnetic"
    ] = "electric_displacement",
    slab_thickness: float | None = None,
    z_center: float = 0.0,
    overlap_mode: Literal["midplane", "slab", "full"] = "midplane",
    interpolate: bool = True,
) -> float:
    """Computes the normalized spatial overlap integral between two electromagnetic eigenmodes.

    Evaluates the spatial correlation between mode (ms1, band1) and mode (ms2, band2).
    For 'electric_displacement', the overlap is defined using macroscopic electromagnetic energy:
        eta = |<E1 | D2> * <D1 | E2>| / (U1 * U2)
    where U_i = integral Re(E_i* . D_i) dV is the total stored electric energy. When ms1 == ms2
    and band1 == band2, eta = 1.0 identically. For orthogonal eigenmodes of the same solver run,
    eta = 0.0.

    For 'electric', evaluates:
        eta = |integral E1* . E2 dV|^2 / (integral |E1|^2 dV * integral |E2|^2 dV)

    For 'magnetic', evaluates:
        eta = |integral H1* . H2 dV|^2 / (integral |H1|^2 dV * integral |H2|^2 dV)

    Args:
        ms1: First mpb.ModeSolver instance with solved eigenfields.
        band1: 1-based band index in ms1.
        ms2: Second mpb.ModeSolver instance with solved eigenfields.
        band2: 1-based band index in ms2.
        field: Field formulation to evaluate ('electric_displacement', 'electric',
            'displacement', or 'magnetic'). Default is 'electric_displacement'.
        slab_thickness: Optional normalized slab thickness in units of lattice constant a.
            When provided and overlap_mode='slab', restricts integration along z strictly
            to |z - z_center| <= slab_thickness / 2.
        z_center: Vertical center coordinate of the slab core in units of lattice constant a (default: 0.0).
        overlap_mode: Spatial domain extraction formulation:
            - 'midplane' (default): 2D mid-plane slice at z = z_center (shape: (Nx, Ny, 3)),
              invariant to slab thickness h/a and vertical grid resolution differences.
            - 'slab': 3D masked slab core |z - z_center| <= slab_thickness / 2.
            - 'full': Full 3D supercell without vertical cropping.
        interpolate: If True (default), resamples mode 1 fields onto mode 2 spatial grid
            when mesh resolutions differ. If False, raises ValueError on dimension mismatch.

    Returns:
        Normalized overlap scalar in the range [0.0, 1.0].

    Raises:
        ValueError: If band indices are out of range, overlap_mode is invalid, or field grid dimensions mismatch.
        RuntimeError: If fields cannot be retrieved from either solver.
    """
    if overlap_mode not in ("midplane", "slab", "full"):
        raise ValueError(
            f"Unknown overlap_mode '{overlap_mode}'. Supported modes: 'midplane', 'slab', 'full'."
        )

    f1_a, f1_b = _extract_mode_fields(
        ms1,
        band1,
        field=field,
        slab_thickness=slab_thickness,
        z_center=z_center,
        overlap_mode=overlap_mode,
    )
    f2_a, f2_b = _extract_mode_fields(
        ms2,
        band2,
        field=field,
        slab_thickness=slab_thickness,
        z_center=z_center,
        overlap_mode=overlap_mode,
    )

    if overlap_mode == "midplane":
        f1_a = extract_midplane_field(f1_a)
        f1_b = extract_midplane_field(f1_b)
        f2_a = extract_midplane_field(f2_a)
        f2_b = extract_midplane_field(f2_b)

    if f1_a.shape != f2_a.shape:
        if interpolate:
            f1_a = interpolate_field_to_grid(f1_a, f2_a.shape)
            if f1_b is not None:
                f1_b = interpolate_field_to_grid(f1_b, f2_a.shape)
        else:
            raise ValueError(
                f"Field grid dimension mismatch: mode1 {f1_a.shape} vs mode2 {f2_a.shape}. "
                f"Both simulations must use identical mesh resolution (overlap_mode='{overlap_mode}')."
            )

    if field == "electric_displacement":
        assert f1_b is not None and f2_b is not None
        # Energy norms
        u1 = float(np.real(np.conj(f1_a) * f1_b).sum())
        u2 = float(np.real(np.conj(f2_a) * f2_b).sum())
        if u1 <= 0 or u2 <= 0 or not (np.isfinite(u1) and np.isfinite(u2)):
            return 0.0

        i12 = (np.conj(f1_a) * f2_b).sum()
        i21 = (np.conj(f1_b) * f2_a).sum()
        overlap = float(np.abs(i12 * i21) / (u1 * u2))
        return float(np.clip(overlap, 0.0, 1.0))

    # Standard vector field inner product
    n1 = float(np.real(np.conj(f1_a) * f1_a).sum())
    n2 = float(np.real(np.conj(f2_a) * f2_a).sum())
    if n1 <= 0 or n2 <= 0 or not (np.isfinite(n1) and np.isfinite(n2)):
        return 0.0

    inner = (np.conj(f1_a) * f2_a).sum()
    overlap = float((np.abs(inner) ** 2) / (n1 * n2))
    return float(np.clip(overlap, 0.0, 1.0))


def compute_mode_overlap_matrix(
    ms_ref: Any,
    ms_target: Any,
    bands_ref: Sequence[int] | None = None,
    bands_target: Sequence[int] | None = None,
    field: Literal[
        "electric_displacement", "electric", "displacement", "magnetic"
    ] = "electric_displacement",
    slab_thickness: float | None = None,
    z_center: float = 0.0,
    overlap_mode: Literal["midplane", "slab", "full"] = "midplane",
    interpolate: bool = True,
) -> np.ndarray:
    """Computes the 2D overlap matrix between sequences of eigenbands from two solvers.

    Pre-extracts eigenfields for each band once, avoiding redundant field queries and
    achieving O(M + N) field extraction overhead instead of O(M * N).

    Args:
        ms_ref: Reference mpb.ModeSolver instance or pre-extracted field dict.
        ms_target: Target / perturbed mpb.ModeSolver instance or pre-extracted field dict.
        bands_ref: Sequence of 1-based reference band indices (default: all bands in ms_ref).
        bands_target: Sequence of 1-based target band indices (default: all bands in ms_target).
        field: Field formulation to evaluate ('electric_displacement', 'electric',
            'displacement', or 'magnetic'). Default is 'electric_displacement'.
        slab_thickness: Optional normalized slab thickness in units of lattice constant a.
            When provided and overlap_mode='slab', restricts integration along z strictly
            to |z - z_center| <= slab_thickness / 2.
        z_center: Vertical center coordinate of the slab core in units of lattice constant a (default: 0.0).
        overlap_mode: Spatial domain extraction formulation:
            - 'midplane' (default): 2D mid-plane slice at z = z_center (shape: (Nx, Ny, 3)),
              invariant to slab thickness h/a and vertical grid resolution differences.
            - 'slab': 3D masked slab core |z - z_center| <= slab_thickness / 2.
            - 'full': Full 3D supercell without vertical cropping.
        interpolate: If True (default), resamples reference fields onto target grid when
            mesh resolutions differ. If False, raises ValueError on dimension mismatch.

    Returns:
        2D numpy array of shape (len(bands_ref), len(bands_target)) with values in [0.0, 1.0].

    Raises:
        ValueError: If overlap_mode is invalid or field dimensions mismatch.
    """
    if overlap_mode not in ("midplane", "slab", "full"):
        raise ValueError(
            f"Unknown overlap_mode '{overlap_mode}'. Supported modes: 'midplane', 'slab', 'full'."
        )

    # 1. Pre-extract reference fields
    if isinstance(ms_ref, dict):
        b_ref = list(bands_ref) if bands_ref is not None else list(ms_ref.keys())
        ref_fields = [ms_ref[b] for b in b_ref]
    else:
        n_ref_total = getattr(ms_ref, "num_bands", 1)
        b_ref = (
            list(bands_ref)
            if bands_ref is not None
            else list(range(1, n_ref_total + 1))
        )
        ref_fields = [
            extract_eigenmode_fields(
                ms_ref,
                b,
                field=field,
                slab_thickness=slab_thickness,
                z_center=z_center,
                overlap_mode=overlap_mode,
            )
            for b in b_ref
        ]

    # 2. Pre-extract target fields
    if isinstance(ms_target, dict):
        b_tar = (
            list(bands_target) if bands_target is not None else list(ms_target.keys())
        )
        tar_fields = [ms_target[b] for b in b_tar]
    else:
        n_tar_total = getattr(ms_target, "num_bands", 1)
        b_tar = (
            list(bands_target)
            if bands_target is not None
            else list(range(1, n_tar_total + 1))
        )
        tar_fields = [
            extract_eigenmode_fields(
                ms_target,
                b,
                field=field,
                slab_thickness=slab_thickness,
                z_center=z_center,
                overlap_mode=overlap_mode,
            )
            for b in b_tar
        ]

    if overlap_mode == "midplane":
        ref_fields = [
            (extract_midplane_field(fa), extract_midplane_field(fb))
            for fa, fb in ref_fields
        ]
        tar_fields = [
            (extract_midplane_field(fa), extract_midplane_field(fb))
            for fa, fb in tar_fields
        ]

    # Check shape consistency and interpolate if enabled
    if ref_fields and tar_fields:
        ref_shape = ref_fields[0][0].shape
        tar_shape = tar_fields[0][0].shape
        if ref_shape != tar_shape:
            if interpolate:
                ref_fields = [
                    (
                        interpolate_field_to_grid(fa, tar_shape),
                        interpolate_field_to_grid(fb, tar_shape)
                        if fb is not None
                        else None,
                    )
                    for fa, fb in ref_fields
                ]
            else:
                raise ValueError(
                    f"Field grid dimension mismatch: ref {ref_shape} vs target {tar_shape}. "
                    f"Both simulations must use identical mesh resolution (overlap_mode='{overlap_mode}')."
                )

    matrix = np.zeros((len(b_ref), len(b_tar)), dtype=float)

    if field == "electric_displacement":
        u_ref = [
            float(np.real(np.conj(fa) * fb).sum())
            if (fa is not None and fb is not None)
            else 0.0
            for fa, fb in ref_fields
        ]
        u_tar = [
            float(np.real(np.conj(fa) * fb).sum())
            if (fa is not None and fb is not None)
            else 0.0
            for fa, fb in tar_fields
        ]

        for i, (fa_ref, fb_ref) in enumerate(ref_fields):
            ur = u_ref[i]
            if ur <= 0 or not np.isfinite(ur):
                continue
            assert fb_ref is not None
            for j, (fa_tar, fb_tar) in enumerate(tar_fields):
                ut = u_tar[j]
                if ut <= 0 or not np.isfinite(ut):
                    continue
                assert fb_tar is not None
                i12 = (np.conj(fa_ref) * fb_tar).sum()
                i21 = (np.conj(fb_ref) * fa_tar).sum()
                ov = float(np.abs(i12 * i21) / (ur * ut))
                matrix[i, j] = float(np.clip(ov, 0.0, 1.0))
    else:
        n_ref = [
            float(np.real(np.conj(fa) * fa).sum()) if fa is not None else 0.0
            for fa, _ in ref_fields
        ]
        n_tar = [
            float(np.real(np.conj(fa) * fa).sum()) if fa is not None else 0.0
            for fa, _ in tar_fields
        ]

        for i, (fa_ref, _) in enumerate(ref_fields):
            nr = n_ref[i]
            if nr <= 0 or not np.isfinite(nr):
                continue
            for j, (fa_tar, _) in enumerate(tar_fields):
                nt = n_tar[j]
                if nt <= 0 or not np.isfinite(nt):
                    continue
                inner = (np.conj(fa_ref) * fa_tar).sum()
                ov = float((np.abs(inner) ** 2) / (nr * nt))
                matrix[i, j] = float(np.clip(ov, 0.0, 1.0))

    return matrix


def track_modes_by_overlap(
    ms_ref: Any,
    ms_target: Any,
    ref_bands: Sequence[int],
    target_bands: Sequence[int] | None = None,
    field: Literal[
        "electric_displacement", "electric", "displacement", "magnetic"
    ] = "electric_displacement",
    slab_thickness: float | None = None,
    z_center: float = 0.0,
    overlap_mode: Literal["midplane", "slab", "full"] = "midplane",
    interpolate: bool = True,
    pitch: float | None = None,
    ref_frequencies: dict[int, float] | Sequence[float] | None = None,
    target_frequencies: dict[int, float] | Sequence[float] | None = None,
    tracking_strategy: Literal["cluster", "bipartite", "greedy"] = "cluster",
    band_irreps: dict[int, str] | None = None,
    target_irreps: Sequence[str] | None = None,
    enforce_irreps: bool = False,
    **strategy_kwargs: Any,
) -> dict[str, Any]:
    """Tracks reference eigenmodes in a perturbed / target ModeSolver using overlap integrals.

    For each reference band (or degenerate band group), identifies the target eigenmode(s)
    with maximal modal overlap, computing frequency/wavelength shifts and degenerate
    subspace projections:
        P_subspace(target_j) = sum_{i in ref_bands} Overlap(ref_i, target_j)

    This is particularly suited for tracking degenerate multiplets (e.g. C6v triplets at Gamma)
    that split under substrate-induced vertical symmetry breaking.

    Args:
        ms_ref: Reference mpb.ModeSolver instance (e.g. unperturbed air-clad slab) or pre-extracted field dict.
        ms_target: Target mpb.ModeSolver instance (e.g. asymmetric substrate slab) or pre-extracted field dict.
        ref_bands: Sequence of 1-based reference band indices to track.
        target_bands: Optional sequence of 1-based target band indices to search over
            (default: all bands in ms_target).
        field: Field formulation to evaluate ('electric_displacement', 'electric',
            'displacement', or 'magnetic'). Default is 'electric_displacement'.
        slab_thickness: Optional normalized slab thickness in units of lattice constant a.
            When provided and overlap_mode='slab', restricts integration along z strictly
            to |z - z_center| <= slab_thickness / 2.
        z_center: Vertical center coordinate of the slab core in units of lattice constant a (default: 0.0).
        overlap_mode: Spatial domain extraction formulation:
            - 'midplane' (default): 2D mid-plane slice at z = z_center (shape: (Nx, Ny, 3)),
              invariant to slab thickness h/a and vertical grid resolution differences.
            - 'slab': 3D masked slab core |z - z_center| <= slab_thickness / 2.
            - 'full': Full 3D supercell without vertical cropping.
        interpolate: If True (default), resamples reference fields onto target grid when
            mesh resolutions differ. If False, raises ValueError on dimension mismatch.
        pitch: Optional lattice pitch constant a in micrometers (um). When provided, calculates
            free-space wavelengths lambda_0 = pitch / omega and shifts in nanometers (nm).
        ref_frequencies: Optional dictionary or sequence of reference frequencies (omega_tilde).
        target_frequencies: Optional dictionary or sequence of target frequencies (omega_tilde).
        tracking_strategy: Target mode selection strategy: 'cluster' (multiplet cohesion, default),
            'bipartite' (1-to-1 matching via Hungarian algorithm), or 'greedy' (baseline subspace projection).
        **strategy_kwargs: Additional keyword arguments forwarded to the selected tracking strategy.

    Returns:
        Structured dictionary containing:
        - 'overlap_matrix': 2D np.ndarray of shape (len(ref_bands), len(target_bands)).
        - 'ref_bands': List of reference band indices.
        - 'target_bands': List of target band indices evaluated.
        - 'best_matches': List of dicts for each reference band with keys:
            - 'ref_band': int
            - 'target_band': int
            - 'max_overlap': float in [0, 1]
            - 'ref_frequency': float (omega_tilde)
            - 'target_frequency': float (omega_tilde)
            - 'delta_frequency': float (target - ref)
            - 'ref_wavelength_nm': float | None
            - 'target_wavelength_nm': float | None
            - 'delta_wavelength_nm': float | None
        - 'subspace_projection': 1D np.ndarray of length len(target_bands) giving the total
            overlap of each target mode with the subspace spanned by ref_bands.
        - 'ranked_target_bands': List of target band indices with tracked multiplet bands placed first.
        - 'tracked_bands': List of the top k_modes selected target bands.
        - 'tracking_strategy': The name of the tracking strategy applied.
        - 'overlap_mode': The overlap spatial formulation used ('midplane', 'slab', or 'full').
        - 'interpolate': Whether field interpolation was enabled.
    """
    n_tar_total = getattr(ms_target, "num_bands", 1)
    if isinstance(ms_target, dict):
        n_tar_total = max(ms_target.keys()) if ms_target else 1
    b_tar = (
        list(target_bands)
        if target_bands is not None
        else list(range(1, n_tar_total + 1))
    )
    b_ref = list(ref_bands)

    if not b_ref:
        raise ValueError("ref_bands sequence must not be empty.")

    overlap_mat = compute_mode_overlap_matrix(
        ms_ref=ms_ref,
        ms_target=ms_target,
        bands_ref=b_ref,
        bands_target=b_tar,
        field=field,
        slab_thickness=slab_thickness,
        z_center=z_center,
        overlap_mode=overlap_mode,
        interpolate=interpolate,
    )

    # Extract frequencies
    if ref_frequencies is not None:
        if isinstance(ref_frequencies, dict):
            freq_ref_fn = lambda b: float(ref_frequencies.get(b, 0.0))
        else:
            freq_ref_fn = lambda b: (
                float(ref_frequencies[b - 1]) if b - 1 < len(ref_frequencies) else 0.0
            )
    else:
        freqs_ref = getattr(ms_ref, "all_freqs", [[0.0] * 100])[0]
        freq_ref_fn = lambda b: (
            float(freqs_ref[b - 1]) if b - 1 < len(freqs_ref) else 0.0
        )

    if target_frequencies is not None:
        if isinstance(target_frequencies, dict):
            freq_tar_fn = lambda b: float(target_frequencies.get(b, 0.0))
        else:
            freq_tar_fn = lambda b: (
                float(target_frequencies[b - 1])
                if b - 1 < len(target_frequencies)
                else 0.0
            )
    else:
        freqs_tar = getattr(ms_target, "all_freqs", [[0.0] * 100])[0]
        freq_tar_fn = lambda b: (
            float(freqs_tar[b - 1]) if b - 1 < len(freqs_tar) else 0.0
        )

    best_matches: list[dict[str, Any]] = []
    for i, rb in enumerate(b_ref):
        j_max = int(np.argmax(overlap_mat[i, :]))
        tb = b_tar[j_max]
        max_ov = float(overlap_mat[i, j_max])

        f_ref = freq_ref_fn(rb)
        f_tar = freq_tar_fn(tb)
        delta_f = f_tar - f_ref

        lam_ref_nm = (pitch / f_ref * 1000.0) if (pitch and f_ref > 0) else None
        lam_tar_nm = (pitch / f_tar * 1000.0) if (pitch and f_tar > 0) else None
        delta_lam_nm = (
            (lam_tar_nm - lam_ref_nm)
            if (lam_ref_nm is not None and lam_tar_nm is not None)
            else None
        )

        best_matches.append(
            {
                "ref_band": rb,
                "target_band": tb,
                "max_overlap": max_ov,
                "ref_frequency": f_ref,
                "target_frequency": f_tar,
                "delta_frequency": delta_f,
                "ref_wavelength_nm": lam_ref_nm,
                "target_wavelength_nm": lam_tar_nm,
                "delta_wavelength_nm": delta_lam_nm,
            }
        )

    # Subspace projection: sum of overlaps across all reference bands in the manifold
    subspace_proj = np.sum(overlap_mat, axis=0)  # shape (len(target_bands),)

    tracked_bands, ranked_target_bands = select_tracked_modes(
        strategy=tracking_strategy,
        overlap_mat=overlap_mat,
        target_bands=b_tar,
        k_modes=len(b_ref),
        target_frequencies=[freq_tar_fn(b) for b in b_tar],
        ref_frequencies=[freq_ref_fn(b) for b in b_ref],
        band_irreps=band_irreps,
        target_irreps=target_irreps,
        enforce_irreps=enforce_irreps,
        **strategy_kwargs,
    )

    return {
        "overlap_matrix": overlap_mat,
        "ref_bands": b_ref,
        "target_bands": b_tar,
        "best_matches": best_matches,
        "subspace_projection": subspace_proj,
        "ranked_target_bands": ranked_target_bands,
        "tracked_bands": tracked_bands,
        "tracking_strategy": tracking_strategy,
        "overlap_mode": overlap_mode,
    }


def select_tracked_modes_greedy(
    overlap_mat: np.ndarray,
    target_bands: Sequence[int],
    k_modes: int,
) -> tuple[list[int], list[int]]:
    """Selects top target modes via greedy subspace projection ranking.

    Args:
        overlap_mat: 2D overlap array of shape (len(ref_bands), len(target_bands)).
        target_bands: Sequence of target band indices.
        k_modes: Number of tracked modes to select.

    Returns:
        Tuple of (tracked_bands, ranked_target_bands).
    """
    subspace_proj = np.sum(overlap_mat, axis=0)
    ranked_indices = np.argsort(subspace_proj)[::-1]
    ranked_target_bands = [target_bands[idx] for idx in ranked_indices]
    tracked_bands = ranked_target_bands[:k_modes]
    return tracked_bands, ranked_target_bands


def resolve_target_irrep_counts(
    target_irreps: Sequence[str] | None, k_modes: int
) -> Counter[str]:
    """Resolves target irrep requirements into a multiset Counter, expanding 2D representations if needed.

    Args:
        target_irreps: Sequence of desired irrep labels (e.g. ['A_2', 'E_1', 'E_1'] or ['A_2', 'E_1']).
        k_modes: Number of modes forming the target multiplet.

    Returns:
        Counter mapping irrep name to expected count in the candidate cluster.
    """
    if not target_irreps:
        return Counter()
    counts: Counter[str] = Counter(target_irreps)
    total = sum(counts.values())
    if total < k_modes:
        for irrep in list(counts.keys()):
            if irrep.upper().startswith("E") and counts[irrep] == 1:
                counts[irrep] += 1
                total += 1
                if total >= k_modes:
                    break
    return counts


def select_tracked_modes_cluster(
    overlap_mat: np.ndarray,
    target_bands: Sequence[int],
    k_modes: int,
    target_frequencies: Sequence[float] | dict[int, float] | np.ndarray | None = None,
    ref_frequencies: Sequence[float] | dict[int, float] | np.ndarray | None = None,
    max_band_gap: int = 2,
    spread_penalty_weight: float = 50.0,
    max_frequency_spread: float | None = 0.025,
    span_penalty_weight: float = 0.05,
    band_irreps: dict[int, str] | None = None,
    target_irreps: Sequence[str] | None = None,
    enforce_irreps: bool = False,
    irrep_penalty_weight: float = 1e5,
) -> tuple[list[int], list[int]]:
    """Selects target modes by maximizing cluster modal overlap while penalizing frequency dispersion.

    Enforces multiplet cohesion to prevent non-contiguous mode-tracking jumps between distant
    higher-order bands when tracking degenerate manifolds (such as accidental Dirac cone triplets at Γ),
    while preserving the ability to skip intervening alien bands of different symmetries.

    Args:
        overlap_mat: 2D overlap array of shape (len(ref_bands), len(target_bands)).
        target_bands: Sequence of 1-based target band indices.
        k_modes: Number of modes forming the target multiplet.
        target_frequencies: Sequence or mapping of eigenfrequencies for each target band.
        ref_frequencies: Optional sequence or mapping of reference eigenfrequencies.
        max_band_gap: Maximum band index gap between adjacent members of a candidate cluster
            (default: 2, allowing intervening alien bands to be skipped).
        spread_penalty_weight: Multiplier penalizing relative frequency spread across the cluster.
        max_frequency_spread: Maximum relative frequency spread (Δω/ω_mid) allowed before severe penalty (default: 0.025 = 2.5%).
        span_penalty_weight: Multiplier penalizing total band index span.
        band_irreps: Optional mapping of 1-based band index to point-group irreducible representation label.
        target_irreps: Optional target irreps forming the multiplet (e.g. ['A_2', 'E_1', 'E_1']).
        enforce_irreps: If True, penalizes candidate clusters that do not match target irreps.
        irrep_penalty_weight: Penalty subtracted from candidate cluster score on irrep mismatch.

    Returns:
        Tuple of (tracked_bands, ranked_target_bands) where tracked_bands has length k_modes.
    """
    n_tar = len(target_bands)
    if n_tar <= k_modes:
        t_bands = list(target_bands)
        return t_bands, t_bands

    subspace_proj = np.sum(overlap_mat, axis=0)
    greedy_order = [target_bands[i] for i in np.argsort(subspace_proj)[::-1]]

    # If frequencies are missing or all zero, fall back to greedy
    if target_frequencies is None:
        return select_tracked_modes_greedy(overlap_mat, target_bands, k_modes)

    if isinstance(target_frequencies, dict):
        freq_arr = np.array(
            [float(target_frequencies.get(b, 0.0)) for b in target_bands]
        )
    else:
        freq_arr = np.array([float(f) for f in target_frequencies])

    if np.all(freq_arr <= 0.0):
        return select_tracked_modes_greedy(overlap_mat, target_bands, k_modes)

    candidate_clusters: list[tuple[tuple[int, ...], float]] = []

    for start_i in range(n_tar):
        max_end = min(n_tar, start_i + k_modes + max_band_gap)
        window_indices = list(range(start_i, max_end))
        if len(window_indices) < k_modes:
            continue
        for combo_indices in itertools.combinations(window_indices, k_modes):
            if combo_indices[0] != start_i:
                continue
            combo_bands = tuple(target_bands[idx] for idx in combo_indices)
            f_vals = freq_arr[list(combo_indices)]
            f_mid = float(np.mean(f_vals))
            dispersion = (
                float((np.max(f_vals) - np.min(f_vals)) / f_mid) if f_mid > 0 else 0.0
            )

            # Subspace coverage: every ref mode finds a representative in the cluster
            sub_ov = overlap_mat[:, list(combo_indices)]
            coverage_score = float(np.sum(np.max(sub_ov, axis=1)))
            total_overlap = float(np.sum(sub_ov))
            idx_span = combo_indices[-1] - combo_indices[0] + 1 - k_modes

            if max_frequency_spread is not None and dispersion > max_frequency_spread:
                score = -1e6 - 1000.0 * dispersion
            else:
                score = (
                    coverage_score
                    + 0.1 * total_overlap
                    - spread_penalty_weight * dispersion
                    - 200.0 * (dispersion**2)
                    - span_penalty_weight * idx_span
                )

            # Irrep constraint evaluation
            if band_irreps is not None and (
                enforce_irreps or target_irreps is not None
            ):
                target_counts = resolve_target_irrep_counts(target_irreps, k_modes)
                combo_irreps = [band_irreps.get(b, "Unknown") for b in combo_bands]
                cluster_counts = Counter(combo_irreps)

                if target_counts:
                    # Penalize any alien irrep (not in target_counts or Unknown)
                    alien_count = sum(
                        count
                        for irr, count in cluster_counts.items()
                        if irr not in target_counts or irr == "Unknown"
                    )
                    if alien_count > 0:
                        score -= irrep_penalty_weight * alien_count

                    # Penalize multiset mismatch with target
                    if cluster_counts != target_counts:
                        score -= irrep_penalty_weight
                elif enforce_irreps:
                    if any(irr == "Unknown" for irr in combo_irreps):
                        score -= irrep_penalty_weight

            candidate_clusters.append((combo_bands, score))

    if not candidate_clusters:
        return greedy_order[:k_modes], greedy_order

    candidate_clusters.sort(key=lambda x: x[1], reverse=True)
    best_cluster = sorted(candidate_clusters[0][0])
    ranked_target_bands = best_cluster + [
        b for b in greedy_order if b not in best_cluster
    ]
    return best_cluster, ranked_target_bands


def select_tracked_modes_bipartite(
    overlap_mat: np.ndarray,
    target_bands: Sequence[int],
    k_modes: int,
    target_frequencies: Sequence[float] | dict[int, float] | np.ndarray | None = None,
    ref_frequencies: Sequence[float] | dict[int, float] | np.ndarray | None = None,
    cluster_window: int | None = 6,
    spread_penalty_weight: float = 10.0,
) -> tuple[list[int], list[int]]:
    """Selects target modes via maximum-weight bipartite matching (Hungarian algorithm).

    Guarantees a strict 1-to-1 matching between reference modes and target modes, evaluated
    within coherent local frequency windows to prevent unphysical mode-jumping.

    Args:
        overlap_mat: 2D overlap array of shape (len(ref_bands), len(target_bands)).
        target_bands: Sequence of 1-based target band indices.
        k_modes: Number of modes forming the target multiplet.
        target_frequencies: Sequence or mapping of eigenfrequencies for each target band.
        ref_frequencies: Optional sequence or mapping of reference eigenfrequencies.
        cluster_window: Optional sliding window size restricting candidate span.
        spread_penalty_weight: Multiplier penalizing relative frequency spread across matched modes.

    Returns:
        Tuple of (tracked_bands, ranked_target_bands).
    """
    n_tar = len(target_bands)
    if n_tar <= k_modes:
        t_bands = list(target_bands)
        return t_bands, t_bands

    subspace_proj = np.sum(overlap_mat, axis=0)
    greedy_order = [target_bands[i] for i in np.argsort(subspace_proj)[::-1]]

    if target_frequencies is not None:
        if isinstance(target_frequencies, dict):
            freq_arr = np.array(
                [float(target_frequencies.get(b, 0.0)) for b in target_bands]
            )
        else:
            freq_arr = np.array([float(f) for f in target_frequencies])
    else:
        freq_arr = np.zeros(n_tar)

    win_size = cluster_window if cluster_window is not None else n_tar
    win_size = max(k_modes, min(n_tar, win_size))

    best_match: list[int] | None = None
    best_score = -1e9

    for start_i in range(n_tar - k_modes + 1):
        end_i = min(n_tar, start_i + win_size)
        win_indices = list(range(start_i, end_i))
        if len(win_indices) < k_modes:
            continue

        sub_ov = overlap_mat[:, win_indices]
        row_ind, col_ind = linear_sum_assignment(-sub_ov)
        matched_target_indices = [win_indices[c] for c in col_ind]
        matched_bands = [target_bands[i] for i in matched_target_indices]

        ov_sum = float(np.sum(overlap_mat[row_ind, matched_target_indices]))
        if np.any(freq_arr > 0.0):
            f_vals = freq_arr[matched_target_indices]
            f_mid = float(np.mean(f_vals))
            disp = (
                float((np.max(f_vals) - np.min(f_vals)) / f_mid) if f_mid > 0 else 0.0
            )
        else:
            disp = 0.0

        score = ov_sum - spread_penalty_weight * disp
        if score > best_score:
            best_score = score
            best_match = matched_bands

    best_cluster = sorted(best_match) if best_match else greedy_order[:k_modes]
    ranked_target_bands = best_cluster + [
        b for b in greedy_order if b not in best_cluster
    ]
    return best_cluster, ranked_target_bands


def select_tracked_modes(
    strategy: Literal["cluster", "bipartite", "greedy"],
    overlap_mat: np.ndarray,
    target_bands: Sequence[int],
    k_modes: int,
    target_frequencies: Sequence[float] | dict[int, float] | np.ndarray | None = None,
    ref_frequencies: Sequence[float] | dict[int, float] | np.ndarray | None = None,
    **strategy_kwargs: Any,
) -> tuple[list[int], list[int]]:
    """Modular selector dispatching target mode identification across available algorithms.

    Args:
        strategy: Algorithm name ('cluster', 'bipartite', or 'greedy').
        overlap_mat: 2D overlap array of shape (len(ref_bands), len(target_bands)).
        target_bands: Sequence of 1-based target band indices.
        k_modes: Number of target modes to track.
        target_frequencies: Optional target eigenfrequencies.
        ref_frequencies: Optional reference eigenfrequencies.
        **strategy_kwargs: Strategy-specific parameters (e.g. max_band_gap, spread_penalty_weight).

    Returns:
        Tuple of (tracked_bands, ranked_target_bands).

    Raises:
        ValueError: If strategy is not recognized.
    """
    strat = strategy.lower()
    if strat == "cluster":
        return select_tracked_modes_cluster(
            overlap_mat=overlap_mat,
            target_bands=target_bands,
            k_modes=k_modes,
            target_frequencies=target_frequencies,
            ref_frequencies=ref_frequencies,
            **strategy_kwargs,
        )
    elif strat == "bipartite":
        return select_tracked_modes_bipartite(
            overlap_mat=overlap_mat,
            target_bands=target_bands,
            k_modes=k_modes,
            target_frequencies=target_frequencies,
            ref_frequencies=ref_frequencies,
            **strategy_kwargs,
        )
    elif strat == "greedy":
        return select_tracked_modes_greedy(
            overlap_mat=overlap_mat,
            target_bands=target_bands,
            k_modes=k_modes,
        )
    else:
        raise ValueError(
            f"Unknown tracking strategy '{strategy}'. Supported strategies: 'cluster', 'bipartite', 'greedy'."
        )
