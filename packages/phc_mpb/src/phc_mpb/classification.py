"""Physical mode classification and modal field metrics for photonic band solvers.

This module provides tools for physical characterization of electromagnetic eigenmodes:
- Polarization energy fractions: in-plane (TE) vs. out-of-plane (TM).
- Modal energy confinement: core slab energy vs. total supercell energy.
- Mode regime classification: Guided (TIR bound) vs. Leaky (resonant) vs. Radiation (continuum artifact).
- Unified single-pass modal metrics extraction from ModeSolver eigenfields.
"""

from collections.abc import Sequence
from enum import StrEnum
from typing import Any, Literal

import numpy as np


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


def extract_eigenmode_fields(
    ms: Any,
    band: int,
    field: Literal[
        "electric_displacement", "electric", "displacement", "magnetic"
    ] = "electric_displacement",
    slab_thickness: float | None = None,
    z_center: float = 0.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Retrieves and optionally masks eigenfield arrays for a specific band.

    Args:
        ms: mpb.ModeSolver instance with solved eigenfields.
        band: 1-based band index.
        field: Type of fields to retrieve ('electric_displacement', 'electric',
            'displacement', or 'magnetic').
        slab_thickness: Optional normalized slab thickness in units of lattice constant a.
        z_center: Vertical center coordinate of slab core (default: 0.0).

    Returns:
        Tuple of (primary_field, secondary_field_or_None).
    """
    sz = _get_supercell_z(ms)

    if field == "electric_displacement":
        try:
            e = ms.get_efield(band)
            d = ms.get_dfield(band)
        except (AttributeError, RuntimeError, TypeError) as exc:
            raise RuntimeError(
                f"Failed to retrieve E/D fields for band {band}."
            ) from exc
        if slab_thickness is not None and e.ndim >= 3 and e.shape[2] > 1:
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
        if slab_thickness is not None and e.ndim >= 3 and e.shape[2] > 1:
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
        if slab_thickness is not None and d.ndim >= 3 and d.shape[2] > 1:
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
        if slab_thickness is not None and h.ndim >= 3 and h.shape[2] > 1:
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
            When provided, restricts integration along z strictly to |z - z_center| <= slab_thickness / 2.
        z_center: Vertical center coordinate of the slab core in units of lattice constant a (default: 0.0).

    Returns:
        Normalized overlap scalar in the range [0.0, 1.0].

    Raises:
        ValueError: If band indices are out of range or field grid dimensions mismatch.
        RuntimeError: If fields cannot be retrieved from either solver.
    """
    f1_a, f1_b = _extract_mode_fields(
        ms1,
        band1,
        field=field,
        slab_thickness=slab_thickness,
        z_center=z_center,
    )
    f2_a, f2_b = _extract_mode_fields(
        ms2,
        band2,
        field=field,
        slab_thickness=slab_thickness,
        z_center=z_center,
    )

    if f1_a.shape != f2_a.shape:
        raise ValueError(
            f"Field grid dimension mismatch: mode1 {f1_a.shape} vs mode2 {f2_a.shape}. "
            "Both simulations must use identical mesh resolution."
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
) -> np.ndarray:
    """Computes the 2D overlap matrix between sequences of eigenbands from two solvers.

    Pre-extracts eigenfields for each band once, avoiding redundant field queries and
    achieving O(M + N) field extraction overhead instead of O(M * N).

    Args:
        ms_ref: Reference mpb.ModeSolver instance.
        ms_target: Target / perturbed mpb.ModeSolver instance.
        bands_ref: Sequence of 1-based reference band indices (default: all bands in ms_ref).
        bands_target: Sequence of 1-based target band indices (default: all bands in ms_target).
        field: Field formulation to evaluate ('electric_displacement', 'electric',
            'displacement', or 'magnetic'). Default is 'electric_displacement'.
        slab_thickness: Optional normalized slab thickness in units of lattice constant a.
            When provided, restricts integration along z strictly to |z - z_center| <= slab_thickness / 2.
        z_center: Vertical center coordinate of the slab core in units of lattice constant a (default: 0.0).

    Returns:
        2D numpy array of shape (len(bands_ref), len(bands_target)) with values in [0.0, 1.0].
    """
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
            )
            for b in b_tar
        ]

    # Check shape consistency
    if ref_fields and tar_fields and ref_fields[0][0].shape != tar_fields[0][0].shape:
        raise ValueError(
            f"Field grid dimension mismatch: ref {ref_fields[0][0].shape} vs target {tar_fields[0][0].shape}. "
            "Both simulations must use identical mesh resolution."
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
    pitch: float | None = None,
    ref_frequencies: dict[int, float] | Sequence[float] | None = None,
    target_frequencies: dict[int, float] | Sequence[float] | None = None,
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
            When provided, restricts integration along z strictly to |z - z_center| <= slab_thickness / 2.
        z_center: Vertical center coordinate of the slab core in units of lattice constant a (default: 0.0).
        pitch: Optional lattice pitch constant a in micrometers (um). When provided, calculates
            free-space wavelengths lambda_0 = pitch / omega and shifts in nanometers (nm).
        ref_frequencies: Optional dictionary or sequence of reference frequencies (omega_tilde).
        target_frequencies: Optional dictionary or sequence of target frequencies (omega_tilde).

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
        - 'ranked_target_bands': List of target band indices sorted by descending subspace overlap.
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
    ranked_indices = np.argsort(subspace_proj)[::-1]
    ranked_target_bands = [b_tar[idx] for idx in ranked_indices]

    return {
        "overlap_matrix": overlap_mat,
        "ref_bands": b_ref,
        "target_bands": b_tar,
        "best_matches": best_matches,
        "subspace_projection": subspace_proj,
        "ranked_target_bands": ranked_target_bands,
    }
