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
        overlap_mode: Literal["midplane", "slab", "full"] = "midplane",
        interpolate: bool = True,
        pitch: float | None = None,
        min_overlap_threshold: float = 0.4,
        overlap_penalty_weight: float = 0.0,
        target_cost: float = 0.0025,
        symmetry_group: str = "C6v",
        bypass_irrep_identification: bool = True,
        tracking_strategy: Literal["cluster", "bipartite", "greedy"] = "cluster",
        tracking_kwargs: dict[str, Any] | None = None,
        target_irreps: Sequence[str] | None = None,
        enforce_irreps: bool = False,
        irrep_penalty_weight: float = 100.0,
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
            overlap_mode: Spatial domain extraction formulation:
                - 'midplane' (default): 2D mid-plane slice at z = z_center (shape: (Nx, Ny, 3)),
                  invariant to slab thickness h/a and vertical grid resolution differences.
                - 'slab': 3D masked slab core |z - z_center| <= slab_thickness / 2.
                - 'full': Full 3D supercell without vertical cropping.
            interpolate: If True (default), resamples reference fields onto target grid when
                mesh resolutions differ.
            pitch: Optional lattice constant a in micrometers (um).
            min_overlap_threshold: Minimum average subspace projection required before penalizing.
            overlap_penalty_weight: Multiplier for overlap penalty when below threshold.
            target_cost: Target cost floor below which optimization stops penalizing noise.
            symmetry_group: Point group symmetry tag ('C6v' or 'C4v').
            bypass_irrep_identification: If False or if enforce_irreps is True, enables symmetry computation.
            tracking_strategy: Target mode selection strategy ('cluster', 'bipartite', or 'greedy').
                Default is 'cluster' (multiplet cohesion).
            tracking_kwargs: Optional dictionary of keyword arguments passed to the mode tracker.
            target_irreps: Optional target irrep labels forming the target multiplet (e.g. ['A_2', 'E_1', 'E_1']).
            enforce_irreps: If True, filters and penalizes tracked modes that do not match target irreps.
            irrep_penalty_weight: Multiplier penalizing frequency cost when tracked modes fail irrep matching.

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
        self.overlap_mode = overlap_mode
        self.interpolate = interpolate
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
        self.enforce_irreps = bool(enforce_irreps)
        if target_irreps is not None:
            self.target_irreps: list[str] | None = [str(irr) for irr in target_irreps]
        elif (
            self.enforce_irreps
            and self.symmetry_group == "C6v"
            and len(self.ref_bands) == 3
        ):
            self.target_irreps = ["A_2", "E_1", "E_1"]
        else:
            self.target_irreps = None
        self.irrep_penalty_weight = float(irrep_penalty_weight)
        self.bypass_irrep_identification = (
            False if self.enforce_irreps else bool(bypass_irrep_identification)
        )
        self.tracking_strategy = tracking_strategy.lower()
        self.tracking_kwargs = dict(tracking_kwargs) if tracking_kwargs else {}

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
                    overlap_mode=overlap_mode,
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

        self._last_tracked_bands: list[int] | None = None

    @property
    def target_bands(self) -> list[int]:
        """Returns the active target bands (tracked if evaluated, else ref_bands)."""
        if self._last_tracked_bands:
            return list(self._last_tracked_bands)
        return list(self.ref_bands)

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

        band_irreps: dict[int, str] = {}
        if self.enforce_irreps:
            symmetries_data = solver_results.get("symmetries", {})
            sym_records = []
            if isinstance(symmetries_data, dict):
                sym_records = symmetries_data.get(
                    pol_key, symmetries_data.get("all", [])
                )
            elif isinstance(symmetries_data, list):
                sym_records = symmetries_data

            if not sym_records and hasattr(ms, "compute_symmetry"):
                from phc_mpb.symmetry import compute_band_symmetries

                sym_records = compute_band_symmetries(
                    ms,
                    symmetry_group=self.symmetry_group,
                    target_irreps=self.target_irreps,
                )
            elif sym_records and self.target_irreps:
                from phc_mpb.symmetry import resolve_multiplet_symmetries

                sym_records = resolve_multiplet_symmetries(
                    sym_records,
                    symmetry_group=self.symmetry_group,
                    target_irreps=self.target_irreps,
                )

            if sym_records:
                band_irreps = {
                    int(s["band"]): str(s.get("irrep", "Unknown"))
                    for s in sym_records
                    if "band" in s
                }

        tracking_data = track_modes_by_overlap(
            ms_ref=self.ref_fields,
            ms_target=ms,
            ref_bands=self.ref_bands,
            target_bands=self.target_band_candidates,
            field=self.field,
            slab_thickness=norm_h,
            z_center=self.z_center,
            overlap_mode=self.overlap_mode,
            interpolate=self.interpolate,
            pitch=pitch_val,
            ref_frequencies=self.ref_frequencies,
            target_frequencies=gamma_freqs,
            tracking_strategy=self.tracking_strategy,
            band_irreps=band_irreps if band_irreps else None,
            target_irreps=self.target_irreps,
            enforce_irreps=self.enforce_irreps,
            **self.tracking_kwargs,
        )

        k_modes = len(self.ref_bands)
        tracked_bands = (
            tracking_data.get("tracked_bands")
            or tracking_data["ranked_target_bands"][:k_modes]
        )
        self._last_tracked_bands = list(tracked_bands)

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

        # 5. Check irrep compliance of tracked modes
        irrep_match = True
        tracked_irreps: list[str] = []
        if self.enforce_irreps and band_irreps and self.target_irreps:
            from collections import Counter

            from phc_mpb.classification import resolve_target_irrep_counts

            target_counts = resolve_target_irrep_counts(self.target_irreps, k_modes)
            tracked_irreps = [band_irreps.get(b, "Unknown") for b in tracked_bands]
            if Counter(tracked_irreps) != target_counts:
                irrep_match = False
                effective_cost += self.irrep_penalty_weight
        elif band_irreps:
            tracked_irreps = [band_irreps.get(b, "Unknown") for b in tracked_bands]

        if self.target_cost is not None and effective_cost < self.target_cost:
            final_cost = self.target_cost
        else:
            final_cost = effective_cost

        fom = 1.0 / max(final_cost, 1e-12)

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
            "tracked_irreps": tracked_irreps,
            "irrep_match": irrep_match,
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

        if not irrep_match:
            status_msg = (
                f"PENALIZED (Irrep mismatch: {tracked_irreps} != {list(self.target_irreps or [])}, "
                f"splitting: {normalized_cost:.4e})"
            )
        else:
            status_msg = (
                f"Tracked bands {sorted(tracked_bands)} "
                f"({'+'.join(tracked_irreps) if tracked_irreps else 'matched'}, "
                f"splitting: {normalized_cost:.4e}, overlap: {mean_overlap:.3f})"
            )

        return ObjectiveEvaluation(
            cost=final_cost,
            fom=fom,
            status=status_msg,
            metadata=metadata,
            group_velocity=vg_top_band,
            is_penalty=(not irrep_match),
        )
