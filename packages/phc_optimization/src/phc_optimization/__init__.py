"""phc_optimization: Modular Bayesian Optimization Framework for Photonic Crystals."""

from phc_optimization.connectivity import (
    check_array_connectivity,
    check_slab_connectivity,
)
from phc_optimization.locus import (
    compute_curve_normals,
    evaluate_locus_group_velocities,
    export_loci_to_json,
    export_locus_to_csv,
    extract_optimal_loci,
    extract_polar_ring_locus,
    order_skeleton_points,
    refine_locus_points,
    zhang_suen_thinning,
)
from phc_optimization.objectives import (
    BaseObjective,
    DiracDegeneracyObjective,
    get_objective,
)
from phc_optimization.optimizer import BayesianOptimizer
from phc_optimization.plotting import (
    plot_bo_convergence,
    plot_bo_surrogate_map,
    plot_locus_profile,
)
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
    "compute_curve_normals",
    "evaluate_locus_group_velocities",
    "export_loci_to_json",
    "export_locus_to_csv",
    "extract_optimal_loci",
    "extract_polar_ring_locus",
    "fit_clean_surrogate",
    "get_objective",
    "order_skeleton_points",
    "plot_bo_convergence",
    "plot_bo_surrogate_map",
    "plot_locus_profile",
    "predict_surrogate_landscape",
    "refine_locus_points",
    "zhang_suen_thinning",
]
