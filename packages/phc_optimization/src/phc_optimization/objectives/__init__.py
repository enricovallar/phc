"""Modular objective functions for photonic crystal optimization."""

from phc_optimization.objectives.base import BaseObjective
from phc_optimization.objectives.dirac_degeneracy import DiracDegeneracyObjective

OBJECTIVE_REGISTRY: dict[str, type[BaseObjective]] = {
    "dirac_degeneracy": DiracDegeneracyObjective,
}


def get_objective(objective: str | BaseObjective, **kwargs) -> BaseObjective:
    """Resolves an objective instance from a string name or returns the instance.

    Args:
        objective: String identifier in registry, or a BaseObjective instance.
        **kwargs: Arguments forwarded to objective constructor if string identifier.

    Returns:
        BaseObjective instance.

    Raises:
        ValueError: If objective identifier is not found in registry.
        TypeError: If objective is neither str nor BaseObjective.
    """
    if isinstance(objective, BaseObjective):
        return objective

    if isinstance(objective, str):
        key = objective.lower().strip()
        if key not in OBJECTIVE_REGISTRY:
            raise ValueError(
                f"Unknown objective '{objective}'. Available: {list(OBJECTIVE_REGISTRY.keys())}"
            )
        return OBJECTIVE_REGISTRY[key](**kwargs)

    raise TypeError(f"Expected str or BaseObjective, got {type(objective).__name__}")


__all__ = [
    "OBJECTIVE_REGISTRY",
    "BaseObjective",
    "DiracDegeneracyObjective",
    "get_objective",
]
