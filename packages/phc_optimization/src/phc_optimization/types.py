"""Data contracts, type definitions, and results structures for photonic optimization."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class ParameterSpec:
    """Specification of an optimization parameter.

    Attributes:
        name: Name of the parameter matching a keyword argument in the layout cell factory.
        bounds: Lower and upper search bounds (min_val, max_val).
        param_type: 'real' (continuous) or 'integer' (discrete).
        default: Optional initial/default value within bounds.
    """

    name: str
    bounds: tuple[float, float]
    param_type: Literal["real", "integer"] = "real"
    default: float | None = None


@dataclass
class ObjectiveEvaluation:
    """Evaluation result produced by a target objective.

    Attributes:
        cost: Objective scalar cost to minimize (>= 0.0). For Dirac search, normalized gap splitting.
        fom: Figure of merit to maximize (typically inverse cost).
        status: Tracking message describing evaluation outcome (e.g. 'Mapped to bands [2, 3, 4]').
        metadata: Arbitrary structured metadata (e.g. target bands, irrep maps, corrections).
        group_velocity: Optional group velocity metric (e.g. top band vg at delta_k).
        is_penalty: True if evaluation failed physical checks and incurred penalty cost.
    """

    cost: float
    fom: float
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)
    group_velocity: float | None = None
    is_penalty: bool = False


@dataclass
class OptimizationRecord:
    """Recorded outcome of a single candidate evaluation.

    Attributes:
        eval_index: 1-based sequential evaluation index.
        generation: 0-based optimization generation (batch iteration).
        params: Evaluated parameter values.
        cost: Evaluated objective cost.
        fom: Evaluated figure of merit.
        status: Status string.
        connectivity: Connectivity check result ('PASSED', 'FAILED', or 'NOT_CHECKED').
        timing: Breakdown of computation times (seconds).
        metadata: Detailed objective-specific metadata.
    """

    eval_index: int
    generation: int
    params: dict[str, float]
    cost: float
    fom: float
    status: str
    connectivity: str = "NOT_CHECKED"
    timing: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """Comprehensive result of an optimization run.

    Attributes:
        best_params: Parameter set yielding maximum FOM / minimum cost.
        best_fom: Maximum achieved Figure of Merit.
        best_cost: Minimum achieved objective cost.
        records: Full list of all evaluated candidate records.
        gp_model: Trained Gaussian Process surrogate model (if fit).
        output_dir: Resolved output directory containing artifacts and manifests.
    """

    best_params: dict[str, float]
    best_fom: float
    best_cost: float
    records: list[OptimizationRecord] = field(default_factory=list)
    gp_model: Any = None
    output_dir: Path | None = None
