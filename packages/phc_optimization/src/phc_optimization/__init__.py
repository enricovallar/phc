"""phc_optimization: Modular Bayesian Optimization Framework for Photonic Crystals."""

from phc_optimization.analysis import run_substrate_band_comparison
from phc_optimization.connectivity import (
    check_array_connectivity,
    check_slab_connectivity,
)
from phc_optimization.locus import (
    compute_curve_normals,
    evaluate_locus_dirac_frequencies,
    evaluate_locus_group_velocities,
    export_loci_to_json,
    export_locus_to_csv,
    extract_optimal_loci,
    extract_polar_ring_locus,
    find_latest_locus_path,
    find_target_locus_point,
    load_loci_from_json,
    load_locus_from_csv,
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
    plot_band_structure_comparison,
    plot_bo_convergence,
    plot_bo_surrogate_map,
    plot_locus_dirac_frequency,
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
    "evaluate_locus_dirac_frequencies",
    "evaluate_locus_group_velocities",
    "export_loci_to_json",
    "export_locus_to_csv",
    "extract_optimal_loci",
    "extract_polar_ring_locus",
    "find_latest_locus_path",
    "find_target_locus_point",
    "fit_clean_surrogate",
    "get_objective",
    "load_loci_from_json",
    "load_locus_from_csv",
    "order_skeleton_points",
    "plot_band_structure_comparison",
    "plot_bo_convergence",
    "plot_bo_surrogate_map",
    "plot_locus_dirac_frequency",
    "plot_locus_profile",
    "predict_surrogate_landscape",
    "refine_locus_points",
    "run_substrate_band_comparison",
    "zhang_suen_thinning",
]
