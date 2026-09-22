"""phc_optimization: Modular Bayesian Optimization Framework for Photonic Crystals."""

from phc_optimization.connectivity import (
    check_array_connectivity,
    check_slab_connectivity,
)
from phc_optimization.locus import (
    extract_optimal_loci,
    order_skeleton_points,
    zhang_suen_thinning,
)
from phc_optimization.objectives import (
    BaseObjective,
    DiracDegeneracyObjective,
    get_objective,
)
from phc_optimization.optimizer import BayesianOptimizer
from phc_optimization.plotting import plot_bo_convergence, plot_bo_surrogate_map
from phc_optimization.surrogate import (
    SurrogateLandscape,
    fit_clean_surrogate,
    predict_surrogate_landscape,
)
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
    "SurrogateLandscape",
    "check_array_connectivity",
    "check_slab_connectivity",
    "extract_optimal_loci",
    "fit_clean_surrogate",
    "get_objective",
    "order_skeleton_points",
    "plot_bo_convergence",
    "plot_bo_surrogate_map",
    "predict_surrogate_landscape",
    "zhang_suen_thinning",
]
