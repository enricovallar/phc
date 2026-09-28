"""Point-group symmetry classification, character projection, and degenerate mode resolution.

This module provides tools to analyze the point-group transformation properties of
photonic crystal eigenmodes at high-symmetry k-points (most notably the zone-center
Gamma point k = (0, 0, 0)).

Supported point groups:
- C4v (square lattice, order 8): representations A_1, A_2, B_1, B_2, E
- C6v (hexagonal lattice, order 12): representations A_1, A_2, B_1, B_2, E_1, E_2

Features:
- Direct evaluation of symmetry operator expectation values <psi_b | R | psi_b> in MPB.
- Character projection onto irreducible representations (irreps) with confidence metrics.
- Basis-invariant subspace trace projection Tr_B(R) = sum_{b in B} <psi_b | R | psi_b>
  for resolving degenerate multiplets (such as accidental Dirac crossings and E doublets)
  mixed by numerical eigensolvers.
- Failsafe multiplet relabeling ensuring exact reproducibility with legacy workflows.
"""

from collections.abc import Sequence
from typing import Any

import meep as mp

# Point Group Character Tables & Class Weights
# Format:
#   "g": group order
#   "weights": number of elements in each conjugacy class
#   "irreps": character values chi_i(R) for each irrep
CHARACTER_TABLES: dict[str, dict[str, Any]] = {
    "C4v": {
        "g": 8,
        "weights": {"E": 1, "C4": 2, "C2": 1, "sv": 2, "sd": 2},
        "irreps": {
            "A_1": {"E": 1, "C4": 1, "C2": 1, "sv": 1, "sd": 1},
            "A_2": {"E": 1, "C4": 1, "C2": 1, "sv": -1, "sd": -1},
            "B_1": {"E": 1, "C4": -1, "C2": 1, "sv": 1, "sd": -1},
            "B_2": {"E": 1, "C4": -1, "C2": 1, "sv": -1, "sd": 1},
            "E": {"E": 2, "C4": 0, "C2": -2, "sv": 0, "sd": 0},
        },
    },
    "C6v": {
        "g": 12,
        "weights": {"E": 1, "C6": 2, "C3": 2, "C2": 1, "sv": 3, "sd": 3},
        "irreps": {
            "A_1": {"E": 1, "C6": 1, "C3": 1, "C2": 1, "sv": 1, "sd": 1},
            "A_2": {"E": 1, "C6": 1, "C3": 1, "C2": 1, "sv": -1, "sd": -1},
            "B_1": {"E": 1, "C6": -1, "C3": 1, "C2": -1, "sv": 1, "sd": -1},
            "B_2": {"E": 1, "C6": -1, "C3": 1, "C2": -1, "sv": -1, "sd": 1},
            "E_1": {"E": 2, "C6": 1, "C3": -1, "C2": -2, "sv": 0, "sd": 0},
            "E_2": {"E": 2, "C6": -1, "C3": -1, "C2": 2, "sv": 0, "sd": 0},
        },
    },
}


def get_symmetry_operators(
    group: str = "C4v",
) -> dict[str, mp.Matrix]:
    """Constructs representative rotation and reflection transformation matrices in lattice coordinates.

    In MPB, the symmetry operation matrix W is defined in reciprocal-lattice basis coordinates.
    For standard square and triangular lattices, coordinate representations are:

    Square lattice (C4v):
        basis1 = (1, 0), basis2 = (0, 1)
        C4:  (x, y) -> (-y, x)     => [ [0, -1, 0], [1, 0, 0], [0, 0, 1] ]
        C2:  (x, y) -> (-x, -y)    => [ [-1, 0, 0], [0, -1, 0], [0, 0, 1] ]
        sv:  (x, y) -> (x, -y)     => [ [1, 0, 0], [0, -1, 0], [0, 0, 1] ]
        sd:  (x, y) -> (y, x)      => [ [0, 1, 0], [1, 0, 0], [0, 0, 1] ]

    Hexagonal lattice (C6v):
        basis1 = (1, 0), basis2 = (0.5, sqrt(3)/2)
        C6:  basis1 -> basis2, basis2 -> basis2 - basis1
        C3:  C6^2
        C2:  C6^3 = -I
        sv:  reflection across x
        sd:  reflection across y

    Args:
        group: Symmetry point group ('C4v' or 'C6v').

    Returns:
        Dictionary mapping conjugacy class representative names to meep.Matrix instances.

    Raises:
        ValueError: If group is not 'C4v' or 'C6v'.
    """
    g = group.strip()
    if g == "C4v":
        return {
            "C4": mp.Matrix(
                mp.Vector3(0, 1, 0), mp.Vector3(-1, 0, 0), mp.Vector3(0, 0, 1)
            ),
            "C2": mp.Matrix(
                mp.Vector3(-1, 0, 0), mp.Vector3(0, -1, 0), mp.Vector3(0, 0, 1)
            ),
            "sv": mp.Matrix(
                mp.Vector3(1, 0, 0), mp.Vector3(0, -1, 0), mp.Vector3(0, 0, 1)
            ),
            "sd": mp.Matrix(
                mp.Vector3(0, 1, 0), mp.Vector3(1, 0, 0), mp.Vector3(0, 0, 1)
            ),
        }
    if g == "C6v":
        return {
            "C6": mp.Matrix(
                mp.Vector3(0, 1, 0), mp.Vector3(-1, 1, 0), mp.Vector3(0, 0, 1)
            ),
            "C3": mp.Matrix(
                mp.Vector3(-1, 1, 0), mp.Vector3(-1, 0, 0), mp.Vector3(0, 0, 1)
            ),
            "C2": mp.Matrix(
                mp.Vector3(-1, 0, 0), mp.Vector3(0, -1, 0), mp.Vector3(0, 0, 1)
            ),
            "sv": mp.Matrix(
                mp.Vector3(1, 0, 0), mp.Vector3(1, -1, 0), mp.Vector3(0, 0, 1)
            ),
            "sd": mp.Matrix(
                mp.Vector3(0, 1, 0), mp.Vector3(1, 0, 0), mp.Vector3(0, 0, 1)
            ),
        }
    raise ValueError(
        f"Unsupported point group '{group}'. Supported groups: 'C4v', 'C6v'."
    )


def compute_projections(
    chars: dict[str, complex | float],
    group: str = "C4v",
) -> dict[str, float]:
    """Computes single-state character projections onto all irreducible representations.

    Uses the projection formula:
        a_i = (d_i / g) * sum_{R} w_R * Re[ chi_i(R)* * chi_obs(R) ]

    where g is the group order, d_i = chi_i(E) is the irrep dimension, and w_R is
    the class weight.

    Args:
        chars: Observed character expectations per symmetry class (e.g. {'C4': 1.0, 'C2': 1.0, ...}).
        group: Symmetry point group ('C4v' or 'C6v').

    Returns:
        Dictionary mapping each irrep name (e.g. 'A_1', 'E') to its projection score in [0.0, 1.0].

    Raises:
        ValueError: If group is not recognized in CHARACTER_TABLES.
    """
    if group not in CHARACTER_TABLES:
        raise ValueError(
            f"Unsupported point group '{group}'. Supported groups: {list(CHARACTER_TABLES.keys())}"
        )

    group_data = CHARACTER_TABLES[group]
    g = group_data["g"]
    weights = group_data["weights"]
    irreps = group_data["irreps"]

    obs = dict(chars)
    obs["E"] = complex(1.0, 0.0)

    projections: dict[str, float] = {}
    for irrep, target_chars in irreps.items():
        d_i = target_chars["E"]
        scalar_prod = 0.0 + 0.0j

        for op, w in weights.items():
            if op in obs:
                chi_target = target_chars[op]
                chi_obs = complex(obs[op])
                scalar_prod += w * (chi_target * chi_obs)

        proj_val = (d_i * scalar_prod / g).real
        projections[irrep] = float(max(0.0, proj_val))

    return projections


def identify_irrep(
    projections: dict[str, float],
) -> tuple[str, float]:
    """Selects the best-matching irreducible representation from projection scores.

    Args:
        projections: Mapping of irrep names to projection scores.

    Returns:
        Tuple of (best_irrep_name, confidence_score). If empty, returns ('Unknown', 0.0).
    """
    if not projections:
        return "Unknown", 0.0

    best_irrep = max(projections, key=projections.get)  # type: ignore[arg-type]
    return best_irrep, float(projections[best_irrep])


def compute_band_symmetries(
    ms: Any,
    symmetry_group: str = "C4v",
    bands: Sequence[int] | None = None,
    origin: mp.Vector3 | None = None,
) -> list[dict[str, Any]]:
    """Evaluates point-group symmetry characters and projects eigenmodes onto irreducible representations.

    Must be called when the MPB ModeSolver is positioned at a high-symmetry k-point
    (typically Gamma k = (0, 0, 0)).

    Args:
        ms: mpb.ModeSolver instance with solved eigenfields at the current k-point.
        symmetry_group: Point group name ('C4v' or 'C6v'). Default 'C4v'.
        bands: Optional 1-based band indices to evaluate. If None, evaluates all bands 1..ms.num_bands.
        origin: Spatial center of symmetry. Defaults to mp.Vector3(0, 0, 0).

    Returns:
        List of dicts per band, each containing:
            - 'band': 1-based band index.
            - 'freq': Eigenfrequency in normalized units (omega * a / 2pi c).
            - 'irrep': Best-matching irrep label.
            - 'confidence': Projection confidence score in [0.0, 1.0].
            - 'point_group': Evaluated point group name.
            - 'characters': Raw complex character expectations per operator.
            - 'projections': Projection values for all irreps.

    Raises:
        RuntimeError: If ms does not provide compute_symmetry or eigenfields are unavailable.
        ValueError: If symmetry_group is unsupported.
    """
    if not hasattr(ms, "compute_symmetry"):
        raise RuntimeError("ModeSolver instance does not support compute_symmetry.")

    if origin is None:
        origin = mp.Vector3(0, 0, 0)

    operators = get_symmetry_operators(symmetry_group)
    target_bands = (
        list(bands) if bands is not None else list(range(1, ms.num_bands + 1))
    )

    # Retrieve frequencies at current k-point
    current_freqs: list[float] = []
    if hasattr(ms, "freqs") and len(ms.freqs) > 0:
        current_freqs = [float(f) for f in ms.freqs]
    elif hasattr(ms, "get_freqs"):
        try:
            current_freqs = [float(f) for f in ms.get_freqs()]
        except (AttributeError, RuntimeError, TypeError):
            current_freqs = []
    if not current_freqs and hasattr(ms, "all_freqs") and len(ms.all_freqs) > 0:
        current_freqs = [float(f) for f in ms.all_freqs[-1]]
    if not current_freqs:
        current_freqs = [0.0] * getattr(ms, "num_bands", len(target_bands))

    records: list[dict[str, Any]] = []
    for b in target_bands:
        chars: dict[str, complex] = {}
        for op_name, op_mat in operators.items():
            val = ms.compute_symmetry(b, op_mat, origin)
            chars[op_name] = complex(val)

        projs = compute_projections(chars, group=symmetry_group)
        irrep, conf = identify_irrep(projs)

        freq_val = (
            float(current_freqs[b - 1]) if 0 < b <= len(current_freqs) else float("nan")
        )

        records.append(
            {
                "band": b,
                "freq": freq_val,
                "irrep": irrep,
                "confidence": round(conf, 4),
                "point_group": symmetry_group,
                "characters": {k: complex(v) for k, v in chars.items()},
                "projections": {k: round(v, 4) for k, v in projs.items()},
            }
        )

    return resolve_multiplet_symmetries(
        records, symmetry_group=symmetry_group, gamma_freqs=current_freqs
    )


def _to_complex(val: Any) -> complex:
    """Converts a scalar or stringified complex representation to complex."""
    if isinstance(val, (int, float, complex)):
        return complex(val)
    val_str = str(val).strip().strip("()")
    return complex(val_str)


def resolve_multiplet_symmetries(
    records: Sequence[dict[str, Any]],
    symmetry_group: str = "C6v",
    degeneracy_tol: float = 0.015,
    gamma_freqs: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Resolves degenerate mode mixing, accidental multiplets, and false singlet irrep assignments.

    In numerical eigensolvers (e.g. MPB), degenerate subspaces (such as 2D E doublets)
    are returned in arbitrary basis orientations. When evaluated on a single-band basis,
    reflection characters can take arbitrary values (e.g. +1 and -1), causing false matches
    with 1D irreps (such as B_1). Furthermore, isolated non-degenerate singlet modes may be
    falsely assigned to 2D irreps due to raw projection formulas.

    This function:
    1. Clusters adjacent bands by frequency at Gamma according to `degeneracy_tol`.
    2. For singlet clusters (size 1), restricts classification to 1D irreducible representations.
    3. For doublet clusters (size 2), computes basis-invariant subspace trace projections across
       both bands, assigning both to the matching 2D irrep (e.g. E_1, E_2, or E).
    4. For accidental triplet clusters (size 3, e.g. E_1 + A_2), partitions the cluster into
       the candidate singlet and doublet pair that maximizes total subspace projection fidelity.

    Args:
        records: Sequence of band symmetry dictionaries from compute_band_symmetries.
        symmetry_group: Point group name ('C4v' or 'C6v'). Default 'C6v'.
        degeneracy_tol: Relative frequency tolerance (delta_omega / omega) used to cluster
            nearly degenerate bands (default: 0.015).
        gamma_freqs: Optional sequence of frequencies at Gamma corresponding to bands.

    Returns:
        New list of band symmetry dictionaries with resolved irrep labels and subspace confidence scores.

    Raises:
        ValueError: If symmetry_group is not recognized in CHARACTER_TABLES.
    """
    if not records:
        return []

    if symmetry_group not in CHARACTER_TABLES:
        raise ValueError(
            f"Unsupported point group '{symmetry_group}'. Supported groups: {list(CHARACTER_TABLES.keys())}"
        )

    group_data = CHARACTER_TABLES[symmetry_group]
    irreps_data = group_data["irreps"]

    out_records = [dict(r) for r in records]
    if gamma_freqs is not None and len(gamma_freqs) > 0:
        for r in out_records:
            b = int(r.get("band", 1))
            if 0 < b <= len(gamma_freqs):
                r["freq"] = float(gamma_freqs[b - 1])

    freqs = [float(r.get("freq", 0.0)) for r in out_records]
    clusters: list[list[int]] = []
    curr: list[int] = [0]
    for i in range(1, len(out_records)):
        f_prev = freqs[curr[-1]]
        f_curr = freqs[i]
        denom = max(abs(f_prev), 1e-6)
        if (
            abs(f_curr - f_prev) / denom < degeneracy_tol
            or abs(f_curr - f_prev) < 0.005
        ):
            curr.append(i)
        else:
            clusters.append(curr)
            curr = [i]
    clusters.append(curr)

    out_records = [dict(r) for r in records]
    for c in clusters:
        c_recs = [out_records[i] for i in c]
        if len(c) == 1:
            # Singlet: strictly 1D irrep (exclude 2D irreps like E, E_1, E_2)
            r = c_recs[0]
            projs = r.get("projections", {})
            p_1d = {
                k: v
                for k, v in projs.items()
                if irreps_data.get(k, {}).get("E", 1) == 1
            }
            if p_1d:
                best = max(p_1d, key=p_1d.get)
                r["irrep"] = best
                r["confidence"] = round(float(p_1d[best]), 4)
        elif len(c) == 2:
            # Doublet: evaluate subspace trace across both states
            mults = compute_subspace_trace_projection(
                c_recs, symmetry_group=symmetry_group
            )
            mults_2d = {
                k: v
                for k, v in mults.items()
                if irreps_data.get(k, {}).get("E", 1) == 2
            }
            if mults_2d and max(mults_2d.values()) >= 0.5:
                best_2d = max(mults_2d, key=mults_2d.get)
                conf = round(float(mults_2d[best_2d]), 4)
                for r in c_recs:
                    r["irrep"] = best_2d
                    r["confidence"] = conf
        elif len(c) == 3:
            # Triplet: partition into 1 singlet + 1 doublet
            best_partition = None
            best_score = -1.0
            for s_idx, s_rec in enumerate(c_recs):
                pair_recs = [r for j, r in enumerate(c_recs) if j != s_idx]
                p_s = s_rec.get("projections", {})
                p_s_1d = {
                    k: v
                    for k, v in p_s.items()
                    if irreps_data.get(k, {}).get("E", 1) == 1
                }
                s_irrep = max(p_s_1d, key=p_s_1d.get) if p_s_1d else "Unknown"
                s_score = p_s_1d.get(s_irrep, 0.0)

                mults_p = compute_subspace_trace_projection(
                    pair_recs, symmetry_group=symmetry_group
                )
                mults_2d = {
                    k: v
                    for k, v in mults_p.items()
                    if irreps_data.get(k, {}).get("E", 1) == 2
                }
                p_irrep = max(mults_2d, key=mults_2d.get) if mults_2d else "Unknown"
                p_score = mults_2d.get(p_irrep, 0.0)

                total = s_score + p_score
                if total > best_score:
                    best_score = total
                    best_partition = (
                        s_idx,
                        s_irrep,
                        s_score,
                        pair_recs,
                        p_irrep,
                        p_score,
                    )

            if best_partition:
                (
                    s_idx,
                    s_irrep,
                    s_score,
                    pair_recs,
                    p_irrep,
                    p_score,
                ) = best_partition
                c_recs[s_idx]["irrep"] = s_irrep
                c_recs[s_idx]["confidence"] = round(float(s_score), 4)
                for r in pair_recs:
                    r["irrep"] = p_irrep
                    r["confidence"] = round(float(p_score), 4)

    return out_records


def compute_subspace_trace_projection(
    cluster_records: Sequence[dict[str, Any]],
    symmetry_group: str = "C4v",
) -> dict[str, float]:
    """Computes basis-invariant character projection for a degenerate subspace cluster.

    When numerical eigensolvers mix states in a multidimensional representation (such as
    E doublets in C4v/C6v), single-band character expectations vary with basis rotation.
    However, the subspace trace:
        Tr_B(R) = sum_{b in B} <psi_b | R | psi_b>
    is strictly unitary-invariant. The irrep multiplicity in the cluster is:
        m_i = (1 / g) * sum_{R} w_R * chi_i(R)* * Tr_B(R)

    Args:
        cluster_records: List of band symmetry record dictionaries for the degenerate cluster.
        symmetry_group: Point group name ('C4v' or 'C6v').

    Returns:
        Dictionary mapping irrep names to multiplicity projections m_i.
    """
    if symmetry_group not in CHARACTER_TABLES:
        raise ValueError(f"Unsupported point group '{symmetry_group}'.")

    group_data = CHARACTER_TABLES[symmetry_group]
    g = group_data["g"]
    weights = group_data["weights"]
    irreps = group_data["irreps"]

    # Sum traces across cluster bands
    cluster_dim = len(cluster_records)
    trace_chars: dict[str, complex] = {"E": complex(cluster_dim, 0.0)}

    for op in weights:
        if op == "E":
            continue
        op_sum = sum(
            _to_complex(r.get("characters", {}).get(op, 0.0 + 0.0j))
            for r in cluster_records
        )
        trace_chars[op] = complex(op_sum)

    multiplicities: dict[str, float] = {}
    for irrep, target_chars in irreps.items():
        scalar_prod = 0.0 + 0.0j
        for op, w in weights.items():
            if op in trace_chars:
                chi_target = target_chars[op]
                chi_obs = trace_chars[op]
                scalar_prod += w * (chi_target * chi_obs)

        mult_val = (scalar_prod / g).real
        multiplicities[irrep] = float(max(0.0, mult_val))

    return multiplicities


def find_bands_from_irreps(
    symmetries: Sequence[dict[str, Any]],
    target_irreps: Sequence[str],
    irrep_occurrences: Sequence[int] | None = None,
    min_band: int = 2,
    degeneracy_tol: float | None = 0.005,
) -> tuple[
    list[int] | None,
    dict[int, tuple[str, float, float]],
    str | None,
    list[tuple[int, str, float, str]],
]:
    """Dynamically maps requested target irreps to band indices, correcting degenerate mode mixing.

    Args:
        symmetries: List of band symmetry dictionaries from compute_band_symmetries.
        target_irreps: Desired irrep labels (e.g. ['A_1', 'E', 'E']).
        irrep_occurrences: 1-based occurrence index for each target irrep (default [1, 1, 1]).
        min_band: Lowest band index to consider (default 2 to skip Band 1 acoustic branch).
        degeneracy_tol: Frequency threshold for degenerate subspace resolution.

    Returns:
        Tuple of:
            - dynamic_bands: List of selected band indices matching target_irreps, or None on failure.
            - full_map: Dict mapping band -> (irrep, confidence, freq).
            - error_message: Descriptive failure string, or None if successful.
            - corrections: List of relabeling corrections applied.
    """
    filtered = [r for r in symmetries if int(r.get("band", 0)) >= min_band]
    if not filtered:
        return None, {}, f"No bands found at or above min_band={min_band}.", []

    if degeneracy_tol is not None:
        point_group = str(filtered[0].get("point_group", "C6v"))
        filtered = resolve_multiplet_symmetries(
            filtered, symmetry_group=point_group, degeneracy_tol=degeneracy_tol
        )

    full_irrep_map: dict[int, tuple[str, float, float]] = {
        int(r["band"]): (
            str(r["irrep"]),
            float(r.get("confidence", 0.0)),
            float(r.get("freq", float("nan"))),
        )
        for r in filtered
    }
    corrections: list[tuple[int, str, float, str]] = []

    # Group bands by assigned irrep
    global_bands_by_irrep: dict[str, list[int]] = {}
    for band, (irrep, _, _) in full_irrep_map.items():
        if irrep and irrep != "Unknown":
            global_bands_by_irrep.setdefault(irrep, []).append(band)

    # Slice bands into modes (1D for A/B, 2D for E)
    modes_by_irrep: dict[str, list[list[int]]] = {}
    for irrep, b_list in global_bands_by_irrep.items():
        sorted_b = sorted(b_list)
        dim = 2 if irrep.startswith("E") else 1
        modes_by_irrep[irrep] = [
            sorted_b[k : k + dim] for k in range(0, len(sorted_b), dim)
        ]

    occurrences = list(irrep_occurrences or [1] * len(target_irreps))
    dynamic_bands: list[int] = []
    used_bands: set[int] = set()

    for irrep, occ in zip(target_irreps, occurrences, strict=False):
        if irrep not in modes_by_irrep or len(modes_by_irrep[irrep]) < occ:
            err = f"Missing occurrence {occ} for irrep '{irrep}'. Available: {modes_by_irrep}"
            return None, full_irrep_map, err, corrections

        cluster = modes_by_irrep[irrep][occ - 1]
        assigned_band: int | None = None
        for b in cluster:
            if b not in used_bands:
                assigned_band = b
                break

        if assigned_band is None:
            err = f"No unassigned band left in occurrence {occ} of '{irrep}'. Cluster: {cluster}"
            return None, full_irrep_map, err, corrections

        dynamic_bands.append(assigned_band)
        used_bands.add(assigned_band)

    return dynamic_bands, full_irrep_map, None, corrections
