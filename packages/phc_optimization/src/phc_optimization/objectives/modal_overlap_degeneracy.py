"""Modal overlap degeneracy objective function for symmetry-broken photonic crystal slabs.

Tracks reference eigenmodes (such as accidental Dirac cone triplets at Gamma) from
an unperturbed symmetric reference system (e.g. air cladding) into a perturbed or
asymmetric system (e.g. SiO2 substrate cladding) using rigorous electromagnetic
overlap integrals. Minimizes frequency splitting across the tracked mode manifold.
"""

from collections.abc import Sequence
from typing import Any, Literal

import gdsfactory as gf
import numpy as np
from phc_mpb.classification import (
    extract_eigenmode_fields,
    track_modes_by_overlap,
)

from phc_optimization.objectives.base import BaseObjective
from phc_optimization.types import ObjectiveEvaluation


class ModalOverlapDegeneracyObjective(BaseObjective):
    """Evaluates degeneracy splitting of eigenmodes tracked via spatial overlap integrals.

    Computes:
        P_subspace(target_j) = sum_{i in ref_bands} Overlap(ref_i, target_j)
        target_bands = top K target modes with maximal subspace projection
        raw_cost = max(omega) - min(omega)
        omega_mid = median(omega) or mean(omega)
        normalized_cost = raw_cost / omega_mid
        FOM = 1.0 / max(normalized_cost, 1e-12)

    For 3-mode multiplets (such as accidental Dirac cones in C6v), computes the
    signed frequency gap for normal-line root-finding:
        signed_gap = (omega_high - omega_mid) - (omega_mid - omega_low)

    Attributes:
        ref_bands: Sequence of 1-based reference band indices forming the target multiplet.
        ref_fields: Pre-extracted eigenfield dictionary mapping band -> (fa, fb).
        ref_frequencies: Dictionary mapping reference band -> omega_tilde.
        field: Electromagnetic field formulation ('electric_displacement', 'electric', etc.).
        polarization: Target parity / polarization ('te' or 'tm').
    """

    def __init__(
        self,
        ref_ms: Any | None = None,
        ref_fields: dict[int, tuple[np.ndarray, np.ndarray | None]] | None = None,
        ref_frequencies: dict[int, float] | Sequence[float] | None = None,
        ref_bands: Sequence[int] = (16, 17, 18),
        target_band_candidates: Sequence[int] | None = None,
        polarization: str = "te",
        field: Literal[
            "electric_displacement", "electric", "displacement", "magnetic"
        ] = "electric_displacement",
        slab_thickness: float | None = None,
        z_center: float = 0.0,
        pitch: float | None = None,
        min_overlap_threshold: float = 0.4,
        overlap_penalty_weight: float = 0.0,
        target_cost: float = 0.0025,
        symmetry_group: str = "C6v",
        bypass_irrep_identification: bool = True,
    ):
        """Initializes the ModalOverlapDegeneracyObjective.

        Args:
            ref_ms: Optional reference mpb.ModeSolver instance with solved eigenfields.
                Fields are immediately extracted and ref_ms is not retained to guarantee picklability.
            ref_fields: Optional dictionary mapping reference band indices to pre-extracted fields.
            ref_frequencies: Optional dictionary or sequence of reference eigenfrequencies (omega_tilde).
            ref_bands: Sequence of 1-based reference band indices to track.
            target_band_candidates: Optional subset of target bands to evaluate overlap over.
            polarization: Parity polarization ('te' or 'tm').
            field: Field formulation to evaluate ('electric_displacement', 'electric',
                'displacement', or 'magnetic'). Default is 'electric_displacement'.
            slab_thickness: Optional normalized slab thickness in units of lattice constant a.
            z_center: Vertical center coordinate of the slab core (default: 0.0).
            pitch: Optional lattice constant a in micrometers (um).
            min_overlap_threshold: Minimum average subspace projection required before penalizing.
            overlap_penalty_weight: Multiplier for overlap penalty when below threshold.
            target_cost: Target cost floor below which optimization stops penalizing noise.
            symmetry_group: Point group symmetry tag ('C6v' or 'C4v').
            bypass_irrep_identification: Always True for overlap tracking.

        Raises:
            ValueError: If neither ref_ms nor ref_fields is provided, or ref_bands is empty.
        """
        self.ref_bands = [int(b) for b in ref_bands]
        if not self.ref_bands:
            raise ValueError("ref_bands sequence must not be empty.")

        self.polarization = polarization.lower()
        self.field = field
        self.slab_thickness = slab_thickness
        self.z_center = float(z_center)
        self.pitch = float(pitch) if pitch is not None else None
        self.target_band_candidates = (
            [int(b) for b in target_band_candidates]
            if target_band_candidates is not None
            else None
        )
        self.min_overlap_threshold = float(min_overlap_threshold)
        self.overlap_penalty_weight = float(overlap_penalty_weight)
        self.target_cost = float(target_cost) if target_cost is not None else None
        self.symmetry_group = symmetry_group
        self.bypass_irrep_identification = bool(bypass_irrep_identification)

        # 1. Resolve reference fields and frequencies
        if ref_fields is not None:
            self.ref_fields = dict(ref_fields)
        elif ref_ms is not None:
            self.ref_fields = {
                b: extract_eigenmode_fields(
                    ref_ms,
                    b,
                    field=field,
                    slab_thickness=slab_thickness,
                    z_center=z_center,
                )
                for b in self.ref_bands
            }
        else:
            raise ValueError(
                "Either ref_ms or ref_fields must be provided to ModalOverlapDegeneracyObjective."
            )

        if ref_frequencies is not None:
            if isinstance(ref_frequencies, dict):
                self.ref_frequencies = {
                    int(k): float(v) for k, v in ref_frequencies.items()
                }
            else:
                self.ref_frequencies = {
                    b: float(ref_frequencies[b - 1])
                    for b in self.ref_bands
                    if b - 1 < len(ref_frequencies)
                }
        elif ref_ms is not None and hasattr(ref_ms, "all_freqs"):
            f_arr = ref_ms.all_freqs[0]
            self.ref_frequencies = {
                b: float(f_arr[b - 1]) for b in self.ref_bands if b - 1 < len(f_arr)
            }
        else:
            self.ref_frequencies = {}

    @property
    def name(self) -> str:
        """Descriptive identifier."""
        return "modal_overlap_degeneracy"

    def evaluate(
        self,
        component: gf.Component,
        ms: Any,
        solver_results: dict[str, Any],
        params: dict[str, Any],
    ) -> ObjectiveEvaluation:
        """Evaluates tracked band degeneracy splitting and Figure of Merit at Gamma.

        Args:
            component: Evaluated GDSFactory unit cell component.
            ms: mpb.ModeSolver instance with solved eigenfields.
            solver_results: Dictionary of outputs from run_band_solver.
            params: Dictionary of parameters.

        Returns:
            ObjectiveEvaluation instance containing normalized cost and FOM.
        """
        # 1. Extract frequencies at Gamma
        pol_key = self.polarization
        freqs_dict = solver_results.get("freqs", {})
        if pol_key not in freqs_dict:
            alias_map = {
                "te": ["te_like", "te", "all"],
                "tm": ["tm_like", "tm", "all"],
                "te_like": ["te", "te_like", "all"],
                "tm_like": ["tm", "tm_like", "all"],
            }
            candidates = alias_map.get(pol_key, ["all"])
            for alias in candidates:
                if alias in freqs_dict:
                    pol_key = alias
                    break

        if pol_key in freqs_dict:
            gamma_freqs = freqs_dict[pol_key][0]
        elif hasattr(ms, "all_freqs"):
            gamma_freqs = ms.all_freqs[0]
        else:
            gamma_freqs = [0.0] * 100

        pitch_val = float(params.get("pitch", params.get("a", self.pitch or 1.0)))
        if self.slab_thickness is not None:
            norm_h = self.slab_thickness
        else:
            h_val = params.get(
                "slab_thickness", params.get("thickness", params.get("h", None))
            )
            norm_h = float(h_val) / max(pitch_val, 1e-12) if h_val is not None else None

        tracking_data = track_modes_by_overlap(
            ms_ref=self.ref_fields,
            ms_target=ms,
            ref_bands=self.ref_bands,
            target_bands=self.target_band_candidates,
            field=self.field,
            slab_thickness=norm_h,
            z_center=self.z_center,
            pitch=pitch_val,
            ref_frequencies=self.ref_frequencies,
            target_frequencies=gamma_freqs,
        )

        ranked_targets = tracking_data["ranked_target_bands"]
        k_modes = len(self.ref_bands)
        tracked_bands = ranked_targets[:k_modes]

        # 3. Retrieve frequencies for tracked bands
        band_freq_map = {
            b: float(gamma_freqs[b - 1]) if b - 1 < len(gamma_freqs) else 0.0
            for b in tracked_bands
        }
        sorted_freqs = sorted(band_freq_map.values())
        freq_low = sorted_freqs[0]
        freq_high = sorted_freqs[-1]

        if k_modes >= 3:
            freq_middle = sorted_freqs[1]
            signed_gap = (freq_high - freq_middle) - (freq_middle - freq_low)
        else:
            freq_middle = (freq_high + freq_low) / 2.0
            signed_gap = freq_high - freq_low

        if freq_middle <= 0:
            freq_middle = 1.0

        raw_cost = abs(freq_high - freq_low)
        normalized_cost = raw_cost / freq_middle if freq_middle > 0 else raw_cost

        # 4. Compute subspace projection / overlap quality
        subspace_proj = tracking_data["subspace_projection"]
        t_bands_list = tracking_data["target_bands"]
        tracked_overlaps = [
            float(subspace_proj[t_bands_list.index(b)])
            for b in tracked_bands
            if b in t_bands_list
        ]
        mean_overlap = float(np.mean(tracked_overlaps)) if tracked_overlaps else 0.0

        # Optional overlap penalty
        effective_cost = normalized_cost
        if (
            mean_overlap < self.min_overlap_threshold
            and self.overlap_penalty_weight > 0.0
        ):
            effective_cost += self.overlap_penalty_weight * (
                self.min_overlap_threshold - mean_overlap
            )

        if self.target_cost is not None and effective_cost < self.target_cost:
            final_cost = self.target_cost
        else:
            final_cost = effective_cost

        fom = 1.0 / max(normalized_cost, 1e-12)

        # Optional group velocity extraction
        vg_top_band: float | None = None
        if (
            "group_velocities" in solver_results
            and pol_key in solver_results["group_velocities"]
        ):
            vgs = solver_results["group_velocities"][pol_key]
            target_vgs = [
                float(np.linalg.norm(vgs[0, b - 1]))
                for b in tracked_bands
                if b <= vgs.shape[1]
            ]
            if target_vgs:
                vg_top_band = max(target_vgs)

        metadata = {
            "target_bands": tracked_bands,
            "raw_cost": raw_cost,
            "normalized_cost": normalized_cost,
            "effective_cost": final_cost,
            "freq_middle": freq_middle,
            "freq_high": freq_high,
            "freq_low": freq_low,
            "signed_gap": signed_gap,
            "mean_overlap": mean_overlap,
            "tracked_overlaps": dict(
                zip(tracked_bands, tracked_overlaps, strict=False)
            ),
            "best_matches": tracking_data["best_matches"],
        }

        status_msg = (
            f"Tracked bands {sorted(tracked_bands)} (splitting: {normalized_cost:.4e}, "
            f"overlap: {mean_overlap:.3f})"
        )

        return ObjectiveEvaluation(
            cost=final_cost,
            fom=fom,
            status=status_msg,
            metadata=metadata,
            group_velocity=vg_top_band,
        )
