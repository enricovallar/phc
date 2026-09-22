"""Abstract base class for modular optimization objectives."""

from abc import ABC, abstractmethod
from typing import Any

import gdsfactory as gf

from phc_optimization.types import ObjectiveEvaluation


class BaseObjective(ABC):
    """Abstract protocol for photonic optimization objectives.

    Implementations evaluate an electromagnetic simulation state (MPB ModeSolver,
    band frequencies, field metrics) alongside the physical GDSFactory component,
    returning a standardized ObjectiveEvaluation containing scalar cost (to minimize),
    Figure of Merit (to maximize), status string, and metadata.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Descriptive identifier of the objective function."""
        ...

    @abstractmethod
    def evaluate(
        self,
        component: gf.Component,
        ms: Any,
        solver_results: dict[str, Any],
        params: dict[str, Any],
    ) -> ObjectiveEvaluation:
        """Evaluates the objective function on the solved simulation state.

        Args:
            component: Evaluated GDSFactory layout component.
            ms: mpb.ModeSolver instance with solved eigenfields and frequencies.
            solver_results: Dictionary of solver outputs returned by run_band_solver
                (containing 'freqs', 'symmetries', 'group_velocities', etc.).
            params: Full dictionary of evaluated parameter values (search + fixed).

        Returns:
            ObjectiveEvaluation instance containing cost, fom, status, and metadata.
        """
        ...
