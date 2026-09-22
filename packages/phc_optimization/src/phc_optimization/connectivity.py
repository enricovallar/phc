"""Dielectric geometry connectivity and neck-width topology checking engine.

Verifies that the high-dielectric matrix in a unit cell or slab forms a continuous,
mechanically connected domain spanning across periodic boundary conditions (PBC),
and respects a minimum feature neck width tolerance to ensure physical fabricability.
"""

from typing import Any

import numpy as np
from scipy.ndimage import binary_opening, label


def check_array_connectivity(
    epsilon: np.ndarray,
    epsilon_threshold: float = 1.1,
    check_pbc: bool = True,
    min_neck_width_px: int = 1,
) -> tuple[bool, int, str]:
    """Checks whether the high-dielectric matrix forms a continuous connected domain across PBC.

    Applies periodic boundary condition (PBC) tiling and morphological binary opening
    to test both physical continuity and minimum neck width constraints:

    1. Thresholds the permittivity array: mask = (epsilon > epsilon_threshold).
    2. Tiles mask across a 3x3 periodic grid to simulate infinite periodicity.
    3. Performs morphological binary opening with a structuring element of size
       min_neck_width_px to sever unphysical sub-resolution necks.
    4. Evaluates connected component labels to verify that the central unit cell domain
       connects to its neighboring periodic images across both x and y boundaries.

    Args:
        epsilon: 2D or 3D numpy array of permittivity values epsilon(r).
        epsilon_threshold: Permittivity cutoff distinguishing high-index matrix from air/holes.
            Default: 1.1.
        check_pbc: If True, tests connectivity across periodic boundary conditions via 3x3 tiling.
            Default: True.
        min_neck_width_px: Minimum neck feature width in grid pixels. If > 1, applies
            morphological binary opening to ensure features below this width are severed.
            Default: 1.

    Returns:
        Tuple of (is_connected: bool, num_features: int, description: str).
    """
    mask = epsilon > epsilon_threshold

    if not np.any(mask):
        return False, 0, "No dielectric material above threshold."

    if np.all(mask):
        return True, 1, "Uniform solid dielectric slab."

    if check_pbc:
        if mask.ndim == 2:
            tiled_mask = np.tile(mask, (3, 3))
        elif mask.ndim == 3:
            tiled_mask = np.tile(mask, (3, 3, 1))
        else:
            tiled_mask = mask
    else:
        tiled_mask = mask

    if min_neck_width_px > 1:
        if mask.ndim == 2:
            structure = np.ones((min_neck_width_px, min_neck_width_px), dtype=bool)
        else:
            structure = np.ones((min_neck_width_px, min_neck_width_px, 1), dtype=bool)
        tiled_mask = binary_opening(tiled_mask, structure=structure)
        if not np.any(tiled_mask):
            return (
                False,
                0,
                f"Dielectric connections severed after opening (neck < {min_neck_width_px} px).",
            )

    labeled_array, num_features = label(tiled_mask)
    if num_features == 0:
        return False, 0, "No connected dielectric component found."

    if check_pbc:
        if mask.ndim == 2:
            ny, nx = mask.shape
            center_labels = set(np.unique(labeled_array[ny : 2 * ny, nx : 2 * nx])) - {
                0
            }
            left_labels = set(np.unique(labeled_array[ny : 2 * ny, :nx])) - {0}
            right_labels = set(np.unique(labeled_array[ny : 2 * ny, 2 * nx :])) - {0}
            bottom_labels = set(np.unique(labeled_array[:ny, nx : 2 * nx])) - {0}
            top_labels = set(np.unique(labeled_array[2 * ny :, nx : 2 * nx])) - {0}

            spans_x = bool(center_labels & left_labels & right_labels)
            spans_y = bool(center_labels & bottom_labels & top_labels)

            if spans_x and spans_y:
                return (
                    True,
                    num_features,
                    "Continuous dielectric matrix spanning both x and y boundaries.",
                )
            return (
                False,
                num_features,
                f"Disconnected dielectric matrix (spans_x={spans_x}, spans_y={spans_y}).",
            )

        if mask.ndim == 3:
            ny, nx, _nz = mask.shape
            center_labels = set(
                np.unique(labeled_array[ny : 2 * ny, nx : 2 * nx, :])
            ) - {0}
            left_labels = set(np.unique(labeled_array[ny : 2 * ny, :nx, :])) - {0}
            right_labels = set(np.unique(labeled_array[ny : 2 * ny, 2 * nx :, :])) - {0}
            bottom_labels = set(np.unique(labeled_array[:ny, nx : 2 * nx, :])) - {0}
            top_labels = set(np.unique(labeled_array[2 * ny :, nx : 2 * nx, :])) - {0}

            spans_x = bool(center_labels & left_labels & right_labels)
            spans_y = bool(center_labels & bottom_labels & top_labels)

            if spans_x and spans_y:
                return (
                    True,
                    num_features,
                    "Continuous dielectric slab spanning across x and y boundaries.",
                )
            return (
                False,
                num_features,
                f"Disconnected dielectric slab (spans_x={spans_x}, spans_y={spans_y}).",
            )

    return (num_features == 1), num_features, f"Dielectric components: {num_features}"


def check_slab_connectivity(
    ms_or_epsilon: Any,
    epsilon_threshold: float = 1.1,
    check_pbc: bool = True,
    min_neck_width_px: int = 1,
) -> tuple[bool, int, str]:
    """Inspects an MPB ModeSolver or numpy permittivity array and evaluates continuity.

    Args:
        ms_or_epsilon: Either an mpb.ModeSolver instance with initialized geometry/epsilon,
            or a numpy.ndarray containing permittivity values.
        epsilon_threshold: Permittivity threshold separating high-index matrix from low-index holes.
        check_pbc: Whether to verify continuous spanning across periodic unit cell boundaries.
        min_neck_width_px: Minimum feature neck width in grid pixels.

    Returns:
        Tuple of (is_connected: bool, num_features: int, description: str).
    """
    import contextlib

    import meep as mp

    if hasattr(ms_or_epsilon, "get_epsilon"):
        with contextlib.suppress(Exception):
            ms_or_epsilon.init_params(mp.NO_PARITY, True)
        eps_array = ms_or_epsilon.get_epsilon()
    elif isinstance(ms_or_epsilon, np.ndarray):
        eps_array = ms_or_epsilon
    else:
        raise TypeError(
            f"Expected ModeSolver or np.ndarray, got {type(ms_or_epsilon).__name__}"
        )

    return check_array_connectivity(
        epsilon=eps_array,
        epsilon_threshold=epsilon_threshold,
        check_pbc=check_pbc,
        min_neck_width_px=min_neck_width_px,
    )
