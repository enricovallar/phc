"""Surrogate modeling and clean Gaussian Process re-fitting engine.

Provides modular surrogate landscape prediction and outlier/penalty filtering for
Bayesian optimization trajectories. Filters out artificial boundary penalties
(such as disconnected-slab penalties) and fits smooth Gaussian Processes to evaluate
the posterior expected Figure of Merit landscape across arbitrary parameter pairs.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import Matern, WhiteKernel

from phc_optimization.types import OptimizationRecord


@dataclass
class SurrogateLandscape:
    """Computed 2D surrogate model landscape across a parameter pair.

    Attributes:
        param_names: Names of the 2 evaluated parameters (x1_name, x2_name).
        x1_grid: 1D coordinate array along first parameter.
        x2_grid: 1D coordinate array along second parameter.
        X1: 2D meshgrid matrix for first parameter.
        X2: 2D meshgrid matrix for second parameter.
        predicted_fom: 2D array of surrogate-predicted Figure of Merit (FOM).
        mu_grid: 2D array of surrogate predicted mean values.
        std_grid: 2D array of surrogate posterior standard deviations.
        is_clean_fit: True if landscape was predicted from a penalty-filtered GP.
    """

    param_names: tuple[str, str]
    x1_grid: np.ndarray
    x2_grid: np.ndarray
    X1: np.ndarray
    X2: np.ndarray
    predicted_fom: np.ndarray
    mu_grid: np.ndarray
    std_grid: np.ndarray
    is_clean_fit: bool


def fit_clean_surrogate(
    records: Sequence[OptimizationRecord],
    param_names: Sequence[str],
    objective_mode: str = "log",
    cost_cutoff: float = 0.5,
    min_clean_points: int = 8,
    random_state: int = 42,
) -> GaussianProcessRegressor | None:
    """Fits a clean Gaussian Process regressor exclusively on unpenalized physical points.

    Evaluated points that violate physical fabricability or connectivity receive
    maximum penalty costs (e.g. C = 1.0). Fitting directly on these boundary cliffs
    creates severe artificial distortion on the smooth physical dispersion manifold.
    This function isolates valid physical evaluations (cost < cost_cutoff and no penalty)
    and fits a clean GP with a Matérn (nu=2.5) kernel to recover smooth dispersion gradients.

    Args:
        records: Sequence of optimization evaluation records.
        param_names: Names of the parameters to use as inputs (e.g. arbitrary 2 parameters).
        objective_mode: 'log' (fits log10(cost)) or 'linear'.
        cost_cutoff: Cost ceiling below which points are considered valid physical states.
        min_clean_points: Minimum number of physical points required to perform re-fitting.
        random_state: Random seed for hyperparameter optimization restarts.

    Returns:
        Fitted GaussianProcessRegressor instance, or None if insufficient clean points exist.
    """
    if len(param_names) < 1:
        return None

    clean_x: list[list[float]] = []
    clean_y: list[float] = []

    for r in records:
        if r.metadata.get("penalty", False):
            continue
        c_val = float(r.cost)
        if c_val >= cost_cutoff:
            continue

        pt = [float(r.params[k]) for k in param_names]
        y_val = float(np.log10(max(c_val, 1e-12))) if objective_mode == "log" else c_val
        clean_x.append(pt)
        clean_y.append(y_val)

    if len(clean_x) < min_clean_points:
        return None

    try:
        kernel = Matern(
            length_scale=0.035, length_scale_bounds=(0.01, 0.25), nu=2.5
        ) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e-1))
        gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=3,
            random_state=random_state,
        )
        gp.fit(np.array(clean_x), np.array(clean_y))
        return gp
    except (ValueError, RuntimeError, TypeError):
        return None


def predict_surrogate_landscape(
    records: Sequence[OptimizationRecord],
    param_names: Sequence[str],
    bounds: tuple[tuple[float, float], tuple[float, float]],
    optimizer_model: Any = None,
    grid_points: int = 150,
    clean_refit: bool = True,
    cost_cutoff: float = 0.5,
    min_clean_points: int = 8,
    include_uncertainty: bool = True,
    objective_mode: str = "log",
    random_state: int = 42,
) -> SurrogateLandscape:
    """Evaluates the surrogate Figure of Merit landscape across any 2 arbitrary parameters.

    Constructs a 2D meshgrid across the specified parameter bounds, computes GP posterior
    mean and standard deviation (using either a clean penalty-filtered GP or the active
    optimizer surrogate model), and evaluates the Log-Normal expected cost and FOM.

    Args:
        records: Sequence of optimization evaluation records.
        param_names: Names of the 2 target parameters (must have length 2).
        bounds: Coordinate bounds ((min_1, max_1), (min_2, max_2)) for the 2 parameters.
        optimizer_model: Fallback trained GP model from the skopt Optimizer.
        grid_points: Resolution per dimension for the 2D evaluation meshgrid (default: 150).
        clean_refit: Whether to attempt fitting a clean GP on penalty-filtered points.
        cost_cutoff: Cost threshold for clean GP filtering (default: 0.5).
        min_clean_points: Minimum unpenalized points to fit clean GP (default: 8).
        include_uncertainty: If True, evaluates expected cost under Log-Normal distribution
            E[C] = 10^(mu + (ln(10)/2) * sigma^2). If False, evaluates deterministic mean 10^mu.
        objective_mode: 'log' or 'linear'.
        random_state: Random seed for clean GP fitting.

    Returns:
        SurrogateLandscape instance containing 2D meshgrids, predicted FOM, mu, and sigma.

    Raises:
        ValueError: If param_names does not contain exactly 2 parameter names.
    """
    if len(param_names) != 2:
        raise ValueError(
            f"Expected exactly 2 parameter names, got {len(param_names)}: {param_names}"
        )

    p1_name, p2_name = param_names[0], param_names[1]
    b1, b2 = bounds[0], bounds[1]

    x1_lin = np.linspace(float(b1[0]), float(b1[1]), grid_points)
    x2_lin = np.linspace(float(b2[0]), float(b2[1]), grid_points)
    X1, X2 = np.meshgrid(x1_lin, x2_lin)
    grid_coords = np.column_stack([X1.ravel(), X2.ravel()])

    # 1. Attempt clean GP fit if enabled
    clean_gp = None
    if clean_refit:
        clean_gp = fit_clean_surrogate(
            records=records,
            param_names=[p1_name, p2_name],
            objective_mode=objective_mode,
            cost_cutoff=cost_cutoff,
            min_clean_points=min_clean_points,
            random_state=random_state,
        )

    is_clean = clean_gp is not None
    active_model = clean_gp if is_clean else optimizer_model

    # 2. Predict mean and standard deviation across meshgrid
    if active_model is not None and hasattr(active_model, "predict"):
        try:
            mu_pred, std_pred = active_model.predict(grid_coords, return_std=True)
        except (ValueError, RuntimeError, TypeError):
            try:
                mu_pred = active_model.predict(grid_coords)
                std_pred = np.zeros_like(mu_pred)
            except (ValueError, RuntimeError, TypeError):
                mu_pred = np.zeros(len(grid_coords))
                std_pred = np.zeros_like(mu_pred)
    else:
        mu_pred = np.zeros(len(grid_coords))
        std_pred = np.zeros_like(mu_pred)

    mu_grid = mu_pred.reshape(X1.shape)
    std_grid = std_pred.reshape(X1.shape)

    # 3. Compute Log-Normal expected cost E[C] and inverse FOM
    if objective_mode == "log":
        if include_uncertainty:
            exp_cost = 10 ** (mu_grid + (np.log(10) / 2.0) * (std_grid**2))
        else:
            exp_cost = 10**mu_grid
    else:
        exp_cost = mu_grid

    predicted_fom = 1.0 / np.maximum(exp_cost, 1e-12)

    return SurrogateLandscape(
        param_names=(p1_name, p2_name),
        x1_grid=x1_lin,
        x2_grid=x2_lin,
        X1=X1,
        X2=X2,
        predicted_fom=predicted_fom,
        mu_grid=mu_grid,
        std_grid=std_grid,
        is_clean_fit=is_clean,
    )
