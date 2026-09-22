"""phc_mpb: Photonic Crystal band structure solver and GDS converter based on MPB."""

from phc_mpb.classification import (
    ModeClassification,
    classify_modes,
    compute_modal_metrics,
    compute_polarization_fractions,
    compute_slab_confinement,
    summarize_mode_physics,
)
from phc_mpb.converter import (
    extract_polygons_from_gds,
    gds_to_mpb_geometry,
)
from phc_mpb.filtering import (
    compute_confinement_alphas,
    create_confinement_mask,
    filter_band_data,
    filter_by_classification,
    filter_light_cone,
)
from phc_mpb.lattice import (
    create_lattice,
    get_high_symmetry_kpath,
    lattice_to_mpb_lattice,
    to_mpb_lattice,
)
from phc_mpb.parallel import run_parallel_band_solver
from phc_mpb.plotting import (
    DEFAULT_POLARIZATION_COLORS,
    DEFAULT_POLARIZATION_LABELS,
    DEFAULT_POLARIZATION_MARKERS,
    DEFAULT_POLARIZATION_MARKERSIZES,
    plot_band_structure,
    plot_epsilon,
    replot_band_structure_from_results,
)
from phc_mpb.solver import (
    create_mode_solver,
    get_epsilon_grid,
    run_band_solver,
)
from phc_mpb.symmetry import (
    CHARACTER_TABLES,
    compute_band_symmetries,
    compute_projections,
    compute_subspace_trace_projection,
    failsafe_irrep_mapping,
    find_bands_from_irreps,
    get_symmetry_operators,
    identify_irrep,
)

__all__ = [
    "CHARACTER_TABLES",
    "DEFAULT_POLARIZATION_COLORS",
    "DEFAULT_POLARIZATION_LABELS",
    "DEFAULT_POLARIZATION_MARKERS",
    "DEFAULT_POLARIZATION_MARKERSIZES",
    "ModeClassification",
    "classify_modes",
    "compute_band_symmetries",
    "compute_confinement_alphas",
    "compute_modal_metrics",
    "compute_polarization_fractions",
    "compute_projections",
    "compute_slab_confinement",
    "compute_subspace_trace_projection",
    "create_confinement_mask",
    "create_lattice",
    "create_mode_solver",
    "extract_polygons_from_gds",
    "failsafe_irrep_mapping",
    "filter_band_data",
    "filter_by_classification",
    "filter_light_cone",
    "find_bands_from_irreps",
    "gds_to_mpb_geometry",
    "get_epsilon_grid",
    "get_high_symmetry_kpath",
    "get_symmetry_operators",
    "identify_irrep",
    "lattice_to_mpb_lattice",
    "plot_band_structure",
    "plot_epsilon",
    "replot_band_structure_from_results",
    "run_band_solver",
    "run_parallel_band_solver",
    "summarize_mode_physics",
    "to_mpb_lattice",
]
