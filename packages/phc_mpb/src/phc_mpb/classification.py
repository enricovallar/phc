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

    For 2D simulations, 'midplane' and 'volumetric' are identical.

    Args:
        ms: mpb.ModeSolver instance with solved eigenfields at the current k-point.
        band_idx: Optional 1-based band index (1 to num_bands). If specified, calculates
            fractions for this single band. If None, calculates fractions for all solved bands.
        method: Evaluation method ('midplane', 'volumetric', 'slab', or 'magnetic'). Default is 'midplane'.

    Returns:
        If band_idx is provided: Dict with keys 'te' and 'tm' (floats in [0.0, 1.0]).
        If band_idx is None: List of dicts for each band from 1 to ms.num_bands.

    Raises:
        ValueError: If band_idx is outside [1, ms.num_bands] or method is unrecognized.
        RuntimeError: If fields are not available or solver has not run.
    """
    metrics = compute_modal_metrics(ms, band_idx=band_idx, polarization_method=method)
    if band_idx is not None and isinstance(metrics, dict):
        return {"te": metrics["te"], "tm": metrics["tm"]}

    if isinstance(metrics, list):
        return [{"te": m["te"], "tm": m["tm"]} for m in metrics]

    raise RuntimeError("Unexpected modal metrics return type.")


def compute_slab_confinement(
    ms: Any,
    band_idx: int | None = None,
    min_eps: float = 1.5,
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

    Returns:
        Float in [0.0, 1.0] if band_idx is specified, or list of floats for all bands.

    Raises:
        ValueError: If band_idx is outside valid range.
        RuntimeError: If eigenfields cannot be retrieved.
    """
    metrics = compute_modal_metrics(ms, band_idx=band_idx, min_eps=min_eps)
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
) -> dict[str, float] | list[dict[str, float]]:
    """Calculates both TE/TM polarization fractions and slab core energy confinement in a single pass.

    Retrieves eigenfields once per band, evaluating modal polarization and core confinement
    simultaneously to avoid redundant field queries and memory allocations.

    Args:
        ms: mpb.ModeSolver instance with solved eigenfields at the current k-point.
        band_idx: Optional 1-based band index (1 to num_bands). If None, computes for all bands.
        polarization_method: Method for polarization fraction calculation
            ('midplane', 'volumetric', 'slab', or 'magnetic'). Default is 'midplane'.
        min_eps: Permittivity threshold defining the core slab region (default: 1.5).

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

        try:
            eps = ms.get_epsilon()
            in_slab = eps > min_eps
            u_slab = float(u_pt[in_slab].sum())
        except (AttributeError, RuntimeError, ValueError):
            # Fallback if epsilon cannot be retrieved
            u_slab = u_cell

        confinement = float(np.clip(u_slab / u_cell, 0.0, 1.0)) if u_cell > 0 else 0.0

        # 3. Polarization Fractions
        if polarization_method == "magnetic":
            try:
                hfield = ms.get_hfield(b)
            except (AttributeError, RuntimeError, TypeError) as exc:
                raise RuntimeError(f"Failed to retrieve H-field for band {b}.") from exc
            u_hxy = float(
                np.sum(np.abs(hfield[..., 0]) ** 2 + np.abs(hfield[..., 1]) ** 2)
            )
            u_hz = float(np.sum(np.abs(hfield[..., 2]) ** 2))
            u_tot_h = u_hxy + u_hz
            f_te = float(np.clip(u_hz / u_tot_h, 0.0, 1.0)) if u_tot_h > 0 else 0.5
            return {"te": f_te, "tm": 1.0 - f_te, "confinement": confinement}

        nz = efield.shape[2] if efield.ndim >= 3 else 1
        if polarization_method == "midplane" and nz > 1:
            mid = nz // 2
            ef = efield[:, :, mid : mid + 1, :]
            df = dfield[:, :, mid : mid + 1, :]
        elif polarization_method == "slab" and nz > 1:
            ef = efield[in_slab, :]
            df = dfield[in_slab, :]
        else:
            ef = efield
            df = dfield

        u_x = float(np.real(np.conj(ef[..., 0]) * df[..., 0]).sum())
        u_y = float(np.real(np.conj(ef[..., 1]) * df[..., 1]).sum())
        u_z = float(np.real(np.conj(ef[..., 2]) * df[..., 2]).sum())
        u_te = u_x + u_y
        u_tot_e = u_te + u_z

        if u_tot_e <= 0:
            f_te = 0.5
            f_tm = 0.5
        else:
            f_te = float(np.clip(u_te / u_tot_e, 0.0, 1.0))
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
