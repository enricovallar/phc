"""Accidental Dirac cone and degenerate multiplet objective function.

Reproduces the exact Figure of Merit (FOM), normalized band-splitting cost,
dynamic irrep matching, and degenerate mode failsafe correction from the legacy
phc_optimizer_old implementation.
"""

from collections.abc import Sequence
from typing import Any

import gdsfactory as gf
import numpy as np
from phc_mpb.symmetry import find_bands_from_irreps

from phc_optimization.objectives.base import BaseObjective
from phc_optimization.types import ObjectiveEvaluation


class DiracDegeneracyObjective(BaseObjective):
    """Evaluates the normalized frequency splitting between target degenerate bands at Gamma.

    Computes:
        raw_cost = |omega_high - omega_low|
        omega_mid = (omega_high + omega_low) / 2
        normalized_cost = raw_cost / omega_mid
        effective_cost = max(normalized_cost, target_cost)   (when normalized_cost < target_cost)
        FOM = 1.0 / max(normalized_cost, 1e-12)

    Supports:
        - Dynamic irreducible representation matching at Gamma using point-group symmetries.
        - Degenerate mode mixing correction via failsafe cluster relabeling.
        - Direct static mode indices bypass (e.g. tracking bands [2, 3, 4] directly).
    """

    def __init__(
        self,
        symmetry_group: str = "C4v",
        polarization: str = "te",
        target_irreps: Sequence[str] = ("A_1", "E", "E"),
        irrep_occurrences: Sequence[int] = (1, 1, 1),
        min_band: int = 2,
        degeneracy_tol: float = 0.005,
        target_cost: float = 0.0025,
        bypass_irrep_identification: bool = False,
        mode_indices: Sequence[int] = (2, 3, 4),
        objective_mode: str = "log",
    ):
        """Initializes the DiracDegeneracyObjective.

        Args:
            symmetry_group: Point group symmetry ('C4v' or 'C6v').
            polarization: Parity polarization ('te' or 'tm').
            target_irreps: Irrep multiplet forming the target Dirac crossing.
            irrep_occurrences: Occurrence index per target irrep above min_band.
            min_band: Lowest band index to inspect (default 2 to skip acoustic branch).
            degeneracy_tol: Frequency splitting threshold delta_omega triggering failsafe relabeling.
            target_cost: Floor below which optimization stops penalizing noise.
            bypass_irrep_identification: If True, bypasses symmetry identification and uses mode_indices.
            mode_indices: Static 1-based band indices when bypass is True.
            objective_mode: 'log' (log10 cost) or 'linear'.
        """
        self.symmetry_group = symmetry_group
        self.polarization = polarization.lower()
        self.target_irreps = list(target_irreps)
        self.irrep_occurrences = list(irrep_occurrences)
        self.min_band = min_band
        self.degeneracy_tol = degeneracy_tol
        self.target_cost = target_cost
        self.bypass_irrep_identification = bypass_irrep_identification
        self.mode_indices = [int(b) for b in mode_indices]
        self.objective_mode = objective_mode

    @property
    def name(self) -> str:
        """Descriptive identifier."""
        return "dirac_degeneracy"

    def evaluate(
        self,
        component: gf.Component,
        ms: Any,
        solver_results: dict[str, Any],
        params: dict[str, Any],
    ) -> ObjectiveEvaluation:
        """Evaluates band degeneracy splitting and Figure of Merit at the Gamma point.

        Args:
            component: Evaluated GDSFactory unit cell component.
            ms: ModeSolver instance.
            solver_results: Dictionary of outputs from run_band_solver.
            params: Dictionary of parameters.

        Returns:
            ObjectiveEvaluation instance containing normalized cost and FOM.
        """
        pol_key = self.polarization
        # Map 3D vs 2D parity aliases
        freqs_dict = solver_results.get("freqs", {})
        if pol_key not in freqs_dict:
            alias_map = {
                "te": "te_like",
                "tm": "tm_like",
                "te_like": "te",
                "tm_like": "tm",
            }
            alt = alias_map.get(pol_key)
            if alt and alt in freqs_dict:
                pol_key = alt

        if pol_key not in freqs_dict or len(freqs_dict[pol_key]) == 0:
            return ObjectiveEvaluation(
                cost=1.0,
                fom=1.0,
                status=f"FAILED: No frequency data found for polarization '{pol_key}'",
                is_penalty=True,
            )

        # Frequencies at Gamma (first k-point or closest to Gamma)
        all_freqs = freqs_dict[pol_key]  # shape (num_k, num_bands)
        gamma_freqs = all_freqs[0]

        # 1. Resolve target bands (dynamic or static)
        target_bands: list[int]
        full_map: dict[int, tuple[str, float, float]] = {}
        corrections: list[tuple[int, str, float, str]] = []
        tracking_label: str

        if not self.bypass_irrep_identification and self.target_irreps:
            symmetries = solver_results.get("symmetries", {}).get(pol_key, [])
            dynamic_bands, full_map, error_msg, corrections = find_bands_from_irreps(
                symmetries=symmetries,
                target_irreps=self.target_irreps,
                irrep_occurrences=self.irrep_occurrences,
                min_band=self.min_band,
                degeneracy_tol=self.degeneracy_tol,
            )
            if error_msg or not dynamic_bands:
                return ObjectiveEvaluation(
                    cost=1.0,
                    fom=1.0,
                    status=f"FAILED: {error_msg}",
                    metadata={"full_map": full_map, "corrections": corrections},
                    is_penalty=True,
                )
            target_bands = dynamic_bands
            tracking_label = f"Mapped to bands {target_bands}"
        else:
            target_bands = self.mode_indices
            tracking_label = f"Static mode indices {target_bands}"

        # 2. Compute cost at Gamma
        band_freqs = {
            b: float(gamma_freqs[b - 1]) for b in range(1, len(gamma_freqs) + 1)
        }

        idx_high = max(target_bands)
        idx_low = min(target_bands)
        freq_high = band_freqs.get(idx_high, 0.0)
        freq_low = band_freqs.get(idx_low, 0.0)

        freq_middle = (freq_high + freq_low) / 2.0
        if freq_middle <= 0:
            sorted_target = sorted(target_bands)
            idx_central = (
                sorted_target[1] if len(sorted_target) >= 3 else sorted_target[0]
            )
            freq_middle = band_freqs.get(idx_central, 1.0)

        raw_cost = abs(freq_high - freq_low)
        normalized_cost = raw_cost / freq_middle if freq_middle > 0 else raw_cost

        effective_cost = (
            self.target_cost
            if (self.target_cost is not None and normalized_cost < self.target_cost)
            else normalized_cost
        )

        fom = 1.0 / max(normalized_cost, 1e-12)

        # Optional group velocity extraction if present in solver_results
        vg_top_band: float | None = None
        if (
            "group_velocities" in solver_results
            and pol_key in solver_results["group_velocities"]
        ):
            vgs = solver_results["group_velocities"][
                pol_key
            ]  # shape (num_k, num_bands, 3)
            # Find group velocity for target bands at k=0 or k=1
            target_vgs = [
                float(np.linalg.norm(vgs[0, b - 1]))
                for b in target_bands
                if b <= vgs.shape[1]
            ]
            if target_vgs:
                vg_top_band = max(target_vgs)

        metadata = {
            "target_bands": target_bands,
            "raw_cost": raw_cost,
            "normalized_cost": normalized_cost,
            "effective_cost": effective_cost,
            "freq_middle": freq_middle,
            "freq_high": freq_high,
            "freq_low": freq_low,
            "corrections": corrections,
            "full_map": full_map,
        }

        return ObjectiveEvaluation(
            cost=effective_cost,
            fom=fom,
            status=tracking_label,
            metadata=metadata,
            group_velocity=vg_top_band,
            is_penalty=False,
        )
