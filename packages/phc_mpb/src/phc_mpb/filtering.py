"""Post-processing filtering, artifact masking, and opacity scaling for photonic band structures.

This module is completely solver-independent and operates purely on NumPy arrays.
It provides mathematical utilities to:
- Generate continuous confinement-weighted alpha transparency (fading transitional leaky modes).
- Mask out spurious supercell radiation continuum artifacts (eta < cutoff).
- Filter eigenfrequency and polarization fraction grids for clean dispersion rendering.
- Selectively keep or omit modes based on physical classification (guided vs. leaky vs. radiation).
"""

from collections.abc import Sequence

import numpy as np


def compute_confinement_alphas(
    confinements: np.ndarray,
    cutoff: float = 0.20,
    full_threshold: float = 0.50,
    base_alpha: float = 1.0,
) -> np.ndarray:
    """Calculates continuous opacity alpha weights based on slab core energy confinement.

    Applies a smooth piecewise linear ramp to prevent artificial discontinuities across bands:
    - eta >= full_threshold (guided & high-Q leaky): Full opacity (base_alpha).
    - cutoff <= eta < full_threshold (transitional leaky / near-cutoff): Smooth linear fade.
    - eta < cutoff (pure supercell radiation continuum artifacts): Zero opacity (0.0).

    Args:
        confinements: Array of modal energy confinement factors in [0.0, 1.0].
        cutoff: Lower threshold below which opacity drops to 0.0 (default: 0.20).
        full_threshold: Upper threshold above which opacity is base_alpha (default: 0.50).
        base_alpha: Maximum opacity for fully confined modes (default: 1.0).

    Returns:
        NumPy array of identical shape with float values in [0.0, base_alpha].

    Raises:
        ValueError: If cutoff >= full_threshold or base_alpha < 0.
    """
    if cutoff >= full_threshold:
        raise ValueError(
            f"Cutoff threshold ({cutoff}) must be strictly less than full_threshold ({full_threshold})."
        )
    if base_alpha < 0:
        raise ValueError(f"base_alpha ({base_alpha}) must be non-negative.")

    conf = np.asarray(confinements, dtype=float)
    alphas = np.zeros_like(conf)

    # Fully localized regime
    alphas[conf >= full_threshold] = base_alpha

    # Fading transitional regime
    transitional_mask = (conf >= cutoff) & (conf < full_threshold)
    ramp = (conf[transitional_mask] - cutoff) / (full_threshold - cutoff)
    alphas[transitional_mask] = ramp * base_alpha

    return np.clip(alphas, 0.0, base_alpha)


def create_confinement_mask(
    confinements: np.ndarray,
    cutoff: float = 0.20,
) -> np.ndarray:
    """Creates a boolean mask indicating physical slab-confined modes.

    Args:
        confinements: Array of modal confinement values.
        cutoff: Minimum confinement threshold (default: 0.20).

    Returns:
        Boolean NumPy array where True indicates a physical mode (guided or leaky) and
        False indicates a supercell continuum artifact.
    """
    return np.asarray(confinements) >= cutoff


def filter_band_data(
    freqs: np.ndarray,
    confinements: np.ndarray,
    cutoff: float = 0.20,
    te_fractions: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Filters band structure arrays by replacing unguided radiation artifacts with NaN.

    Preserves both guided modes and leaky resonances while eliminating unphysical supercell
    air continuum modes. Does not mutate the input arrays.

    Args:
        freqs: 2D array of eigenfrequencies of shape (num_k, num_bands).
        confinements: 2D array of modal confinements of matching shape.
        cutoff: Confinement threshold below which modes are masked out (default: 0.20).
        te_fractions: Optional 2D array of polarization fractions to mask alongside freqs.

    Returns:
        Tuple (filtered_freqs, filtered_te_fractions), where artifact values are replaced by np.nan.

    Raises:
        ValueError: If array shapes do not match.
    """
    f_copy = np.array(freqs, copy=True, dtype=float)
    mask = create_confinement_mask(confinements, cutoff=cutoff)

    f_copy[~mask] = np.nan

    fracs_copy = None
    if te_fractions is not None:
        if te_fractions.shape != freqs.shape:
            raise ValueError(
                f"Shape mismatch: freqs {freqs.shape} vs te_fractions {te_fractions.shape}."
            )
        fracs_copy = np.array(te_fractions, copy=True, dtype=float)
        fracs_copy[~mask] = np.nan

    return f_copy, fracs_copy


def filter_by_classification(
    freqs: np.ndarray,
    classifications: np.ndarray,
    keep_guided: bool = True,
    keep_leaky: bool = True,
    keep_radiation: bool = False,
) -> np.ndarray:
    """Filters eigenfrequencies based on categorical mode classifications.

    Args:
        freqs: 2D array of eigenfrequencies of shape (num_k, num_bands).
        classifications: 2D array of matching shape containing mode classification strings
            ('guided', 'leaky', 'radiation').
        keep_guided: Whether to retain guided modes (default: True).
        keep_leaky: Whether to retain leaky resonant modes (default: True).
        keep_radiation: Whether to retain supercell radiation artifacts (default: False).

    Returns:
        Filtered frequency array where omitted modes are replaced by np.nan.

    Raises:
        ValueError: If freqs and classifications shapes do not match.
    """
    if freqs.shape != classifications.shape:
        raise ValueError(
            f"Shape mismatch: freqs {freqs.shape} vs classifications {classifications.shape}."
        )

    f_out = np.array(freqs, copy=True, dtype=float)
    keep_mask = np.zeros(freqs.shape, dtype=bool)

    if keep_guided:
        keep_mask |= classifications == "guided"
    if keep_leaky:
        keep_mask |= classifications == "leaky"
    if keep_radiation:
        keep_mask |= classifications == "radiation"

    f_out[~keep_mask] = np.nan
    return f_out


def filter_light_cone(
    freqs: np.ndarray,
    light_line: Sequence[float],
) -> np.ndarray:
    """Masks all eigenmodes lying inside the light cone (above the light line).

    Args:
        freqs: 2D array of eigenfrequencies of shape (num_k, num_bands).
        light_line: Sequence of light line frequencies along the k-path.

    Returns:
        Copy of freqs where values above the light line are replaced by np.nan.

    Raises:
        ValueError: If light_line length does not match num_k.
    """
    num_k, _ = freqs.shape
    if len(light_line) != num_k:
        raise ValueError(
            f"Light line length {len(light_line)} does not match num_k {num_k}."
        )

    f_out = np.array(freqs, copy=True, dtype=float)
    ll_arr = np.array(light_line)[:, np.newaxis]
    above_light_line = f_out >= ll_arr
    f_out[above_light_line] = np.nan
    return f_out
