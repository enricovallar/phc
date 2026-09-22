"""phc_optimization: Modular Bayesian Optimization Framework for Photonic Crystals."""

from phc_optimization.connectivity import (
    check_array_connectivity,
    check_slab_connectivity,
)
from phc_optimization.objectives import (
    BaseObjective,
    DiracDegeneracyObjective,
    get_objective,
)
from phc_optimization.optimizer import BayesianOptimizer
from phc_optimization.plotting import plot_bo_convergence, plot_bo_surrogate_map
from phc_optimization.types import (
    ObjectiveEvaluation,
    OptimizationRecord,
    OptimizationResult,
    ParameterSpec,
)

__all__ = [
    "BaseObjective",
    "BayesianOptimizer",
    "DiracDegeneracyObjective",
    "ObjectiveEvaluation",
    "OptimizationRecord",
    "OptimizationResult",
    "ParameterSpec",
    "check_array_connectivity",
    "check_slab_connectivity",
    "get_objective",
    "plot_bo_convergence",
    "plot_bo_surrogate_map",
]
