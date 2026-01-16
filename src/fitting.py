"""Profile likelihood optimization for GDP growth curve fitting.

The model has both non-linear (g) and linear (h, j, k) components.
We use profile likelihood optimization:
1. For each candidate alpha, solve the full linear least squares for h, j, k
2. This gives the true objective as a function of alpha alone
3. Minimize over alpha using grid search + Brent's method
4. GDP0 is fixed to population-weighted mean GDP

Three model variants are supported:
- "base":         growth = g*h + j + k           (5 h params: h0, h1, h2, h3, h4)
- "g_scales_hj":  growth = g*(h + j) + k         (4 h params: h1, h2, h3, h4; h0 absorbed into j0)
- "g_scales_all": growth = g*(h + j + k)         (4 h params: h1, h2, h3, h4; h0 absorbed into j0)

This approach is superior to alternating estimation because it always
evaluates the true objective for each alpha candidate.
"""

import numpy as np
from scipy import optimize
from scipy.sparse import csr_matrix, hstack
from scipy.sparse.linalg import lsqr
from typing import Tuple, Optional
from dataclasses import dataclass

from .model import (
    ModelParams, g_func, h_func, h_func_constrained, j_func, k_func,
    predict_from_data, constrained_to_unconstrained_h,
)
from .data_loader import FittingData


# Valid model variants
MODEL_VARIANTS = ["base", "g_scales_hj", "g_scales_all"]


@dataclass
class FitResult:
    """Container for fitting results."""
    params: ModelParams
    residuals: np.ndarray
    r_squared: float
    rmse: float
    aic: float
    bic: float
    n_iterations: int
    converged: bool
    objective_history: list
    model_variant: str  # "base", "g_scales_hj", or "g_scales_all"
    # Grid search results (if performed)
    grid_search_alphas: Optional[np.ndarray] = None
    grid_search_objectives: Optional[np.ndarray] = None
    # Constrained model parameters (when constrained=True)
    constrained: bool = False
    T_opt: Optional[float] = None  # Optimal temperature (constrained model)
    P_opt: Optional[float] = None  # Optimal precipitation (constrained model)


def build_linear_design_matrix(
    data: FittingData,
    g_values: np.ndarray,
    model_variant: str,
) -> np.ndarray:
    """Build design matrix for linear subproblem given fixed g values.

    Model variants:
        "base":         y = g*h0 + g*h1*T + g*h2*T^2 + g*h3*P + g*h4*P^2 + j + k
        "g_scales_hj":  y = g*h1*T + g*h2*T^2 + g*h3*P + g*h4*P^2 + g*j + k
        "g_scales_all": y = g*h1*T + g*h2*T^2 + g*h3*P + g*h4*P^2 + g*j + g*k

    For g_scales_hj and g_scales_all, h0 is absorbed into j0.
    We drop k[0] = 0 for identifiability (first year is reference).

    Returns design matrix X.
    """
    n = data.n_obs
    n_countries = data.n_countries
    n_years = data.n_years

    # h parameters
    if model_variant == "base":
        # 5 columns: g, g*T, g*T^2, g*P, g*P^2
        X_h = np.column_stack([
            g_values,
            g_values * data.temp,
            g_values * data.temp**2,
            g_values * data.precp,
            g_values * data.precp**2,
        ])
    else:
        # 4 columns: g*T, g*T^2, g*P, g*P^2 (no h0)
        X_h = np.column_stack([
            g_values * data.temp,
            g_values * data.temp**2,
            g_values * data.precp,
            g_values * data.precp**2,
        ])

    # j parameters: 3 * n_countries columns
    # For model variants, j may be scaled by g
    scale_j = g_values if model_variant in ["g_scales_hj", "g_scales_all"] else np.ones(n)

    X_j = np.zeros((n, 3 * n_countries))
    for obs_idx in range(n):
        i = data.country_idx[obs_idx]
        t = data.time[obs_idx]
        s = scale_j[obs_idx]
        X_j[obs_idx, i] = s                      # j0[i] (possibly scaled by g)
        X_j[obs_idx, n_countries + i] = s * t    # j1[i] * t
        X_j[obs_idx, 2 * n_countries + i] = s * t**2  # j2[i] * t^2

    # k parameters: n_years - 1 columns (drop k[0] for identifiability)
    # For g_scales_all, k is scaled by g
    scale_k = g_values if model_variant == "g_scales_all" else np.ones(n)

    X_k = np.zeros((n, n_years - 1))
    for obs_idx in range(n):
        y = data.year_idx[obs_idx]
        if y > 0:
            X_k[obs_idx, y - 1] = scale_k[obs_idx]

    # Combine all columns
    X = np.hstack([X_h, X_j, X_k])

    return X


def solve_linear_subproblem(
    data: FittingData,
    g_values: np.ndarray,
    model_variant: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve for h, j, k parameters given fixed g values.

    Uses ordinary least squares.

    Returns:
        h_params: array of shape (5,) for base, (4,) for other variants
                  For non-base variants, h0 is set to 0.0 and h_params = [0, h1, h2, h3, h4]
        j0, j1, j2: arrays of shape (n_countries,)
        k: array of shape (n_years,) with k[0] = 0
    """
    from scipy import linalg

    X = build_linear_design_matrix(data, g_values, model_variant)
    y = data.growth_pcGDP

    beta, residuals, rank, s = linalg.lstsq(X, y, cond=None, lapack_driver='gelsy')

    n_countries = data.n_countries
    n_years = data.n_years

    if model_variant == "base":
        n_h = 5
        h_params = beta[:5]
    else:
        n_h = 4
        # Return h_params as [0, h1, h2, h3, h4] for consistency
        h_params = np.concatenate([[0.0], beta[:4]])

    j0 = beta[n_h:n_h + n_countries]
    j1 = beta[n_h + n_countries:n_h + 2 * n_countries]
    j2 = beta[n_h + 2 * n_countries:n_h + 3 * n_countries]

    k = np.zeros(n_years)
    k[1:] = beta[n_h + 3 * n_countries:]

    return h_params, j0, j1, j2, k


def objective_alpha_only(alpha: float, GDP0: float, data: FittingData,
                         h_params: np.ndarray, j0: np.ndarray, j1: np.ndarray,
                         j2: np.ndarray, k: np.ndarray) -> float:
    """Compute sum of squared residuals for given alpha (GDP0 fixed).

    Args:
        alpha: Convergence parameter in [0, 1]
        GDP0: Fixed reference GDP level
        data: FittingData object
        h_params: [h0, h1, h2, h3, h4]
        j0, j1, j2, k: linear parameters

    Returns:
        Sum of squared residuals
    """
    # Avoid invalid values
    if alpha < 0 or alpha > 1:
        return 1e20

    g = g_func(data.pcGDP, GDP0, alpha)
    h = h_func(data.temp, data.precp, *h_params)
    j = j_func(data.country_idx, data.time, j0, j1, j2)
    k_vals = k_func(data.year_idx, k)

    pred = g * h + j + k_vals
    residuals = data.growth_pcGDP - pred

    return np.sum(residuals**2)


def compute_predictions(
    data: FittingData,
    g_values: np.ndarray,
    h_params: np.ndarray,
    j0: np.ndarray,
    j1: np.ndarray,
    j2: np.ndarray,
    k: np.ndarray,
    model_variant: str,
) -> np.ndarray:
    """Compute predictions for a given model variant.

    Model variants:
        "base":         pred = g*h + j + k
        "g_scales_hj":  pred = g*(h + j) + k
        "g_scales_all": pred = g*(h + j + k)
    """
    h_vals = h_func(data.temp, data.precp, *h_params)
    j_vals = j_func(data.country_idx, data.time, j0, j1, j2)
    k_vals = k_func(data.year_idx, k)

    if model_variant == "base":
        return g_values * h_vals + j_vals + k_vals
    elif model_variant == "g_scales_hj":
        return g_values * (h_vals + j_vals) + k_vals
    else:  # g_scales_all
        return g_values * (h_vals + j_vals + k_vals)


def objective_for_alpha(
    alpha: float,
    GDP0: float,
    data: FittingData,
    model_variant: str,
) -> float:
    """Compute the best possible objective for a given alpha.

    This solves the full linear subproblem for h, j, k given alpha,
    then returns the sum of squared residuals.
    """
    if alpha <= 0 or alpha >= 1:
        return 1e20

    g_values = g_func(data.pcGDP, GDP0, alpha)
    h_params, j0, j1, j2, k = solve_linear_subproblem(data, g_values, model_variant)

    pred = compute_predictions(data, g_values, h_params, j0, j1, j2, k, model_variant)
    residuals = data.growth_pcGDP - pred

    return np.sum(residuals**2)


def fit_model(
    data: FittingData,
    model_variant: str,
    verbose: bool,
) -> FitResult:
    """Fit the full model by optimizing alpha with linear subproblem solved exactly.

    For each candidate alpha, we solve the full linear least squares problem
    for h, j, k parameters. This gives the true objective as a function of alpha,
    which we then minimize using scipy's bounded optimizer (Brent's method).

    GDP0 is fixed to the population-weighted mean GDP from the data.

    Args:
        data: FittingData object
        model_variant: One of "base", "g_scales_hj", or "g_scales_all"
        verbose: Print progress

    Returns:
        FitResult object
    """
    # GDP0 is fixed to population-weighted mean GDP
    GDP0 = data.pop_weighted_mean_gdp

    # Model variant descriptions
    variant_desc = {
        "base": "g*h + j + k",
        "g_scales_hj": "g*(h + j) + k",
        "g_scales_all": "g*(h + j + k)",
    }

    if verbose:
        print(f"Fitting model...")
        print(f"  Model variant: {model_variant} [{variant_desc[model_variant]}]")
        print(f"  N={data.n_obs}, countries={data.n_countries}, years={data.n_years}")
        print(f"  GDP0 (fixed) = {GDP0:.2f} (pop-weighted mean GDP, {data.gdp0_reference_year})")

    # Optimize alpha using Brent's method on [0.01, 0.99]
    if verbose:
        print(f"\n  Optimizing alpha with Brent's method...")

    # Track all function evaluations
    eval_history = []

    def tracked_objective(alpha):
        obj = objective_for_alpha(alpha, GDP0, data, model_variant)
        eval_history.append((alpha, obj))
        if verbose:
            print(f"    alpha={alpha:.8f}: obj={obj:.8f}")
        return obj

    result = optimize.minimize_scalar(
        tracked_objective,
        bounds=(0.01, 0.99),
        method='bounded',
        options={'xatol': 1e-8}
    )

    alpha = result.x
    final_obj = result.fun

    if verbose:
        print(f"  Optimization converged: alpha={alpha:.8f}, obj={final_obj:.8f}")
        print(f"  Function evaluations: {len(eval_history)}")

    # Get final parameters at optimal alpha
    if verbose:
        print(f"\n  Computing final parameters...")

    converged = result.success
    objective_history = [obj for _, obj in eval_history]

    # Sort evaluations by alpha for output (shows the search path)
    eval_history_sorted = sorted(eval_history, key=lambda x: x[0])
    grid_search_alphas = np.array([a for a, _ in eval_history_sorted])
    grid_search_objectives = np.array([obj for _, obj in eval_history_sorted])

    # Final solve for linear parameters with converged alpha
    g_values = g_func(data.pcGDP, GDP0, alpha)
    h_params, j0, j1, j2, k = solve_linear_subproblem(data, g_values, model_variant)

    # Compute final predictions and residuals
    params = ModelParams(
        GDP0=GDP0, alpha=alpha,
        h0=h_params[0], h1=h_params[1], h2=h_params[2],
        h3=h_params[3], h4=h_params[4],
        j0=j0, j1=j1, j2=j2, k=k,
    )

    predictions = compute_predictions(data, g_values, h_params, j0, j1, j2, k, model_variant)
    residuals = data.growth_pcGDP - predictions

    # Compute fit statistics
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((data.growth_pcGDP - np.mean(data.growth_pcGDP))**2)
    r_squared = 1 - ss_res / ss_tot
    rmse = np.sqrt(ss_res / data.n_obs)

    # AIC/BIC (assuming Gaussian errors)
    # For non-base variants, h0 is absorbed so we have 4 h params instead of 5
    n_h_params = 5 if model_variant == "base" else 4
    n_params = 1 + n_h_params + 3 * data.n_countries + (data.n_years - 1)
    log_likelihood = -data.n_obs / 2 * (np.log(2 * np.pi * ss_res / data.n_obs) + 1)
    aic = 2 * n_params - 2 * log_likelihood
    bic = np.log(data.n_obs) * n_params - 2 * log_likelihood

    if verbose:
        print(f"\nFit complete:")
        print(f"  R² = {r_squared:.4f}")
        print(f"  RMSE = {rmse:.6f}")
        print(f"  AIC = {aic:.1f}, BIC = {bic:.1f}")

    # Compute standard errors
    if verbose:
        print(f"\nComputing standard errors via Hessian (this may take a while)...")
    params = compute_standard_errors(data, params, model_variant)

    return FitResult(
        params=params,
        residuals=residuals,
        r_squared=r_squared,
        rmse=rmse,
        aic=aic,
        bic=bic,
        n_iterations=len(eval_history),
        converged=converged,
        objective_history=objective_history,
        model_variant=model_variant,
        grid_search_alphas=grid_search_alphas,
        grid_search_objectives=grid_search_objectives,
    )


# ============================================================================
# Constrained model fitting: h(T*, P*) = 0
# ============================================================================

def build_constrained_design_matrix_sparse(
    data: FittingData,
    g_values: np.ndarray,
    T_opt: float,
    P_opt: float,
):
    """Build sparse design matrix for constrained model with fixed T_opt, P_opt.

    Model: y = g*h2*(T - T_opt)^2 + g*h4*(P - P_opt)^2 + j + k

    Uses sparse representation for j and k columns (which are mostly zeros)
    for much faster least squares solving.

    Args:
        data: FittingData object
        g_values: Precomputed g(GDP) values for each observation
        T_opt: Optimal temperature
        P_opt: Optimal precipitation

    Returns:
        Sparse CSR design matrix X
    """
    from scipy import sparse

    n = data.n_obs
    n_countries = data.n_countries
    n_years = data.n_years

    # h parameters: 2 dense columns for h2 and h4
    X_h = np.column_stack([
        g_values * (data.temp - T_opt)**2,   # h2 column
        g_values * (data.precp - P_opt)**2,  # h4 column
    ])

    # j parameters: 3 * n_countries sparse columns
    row_idx = np.arange(n)
    col_j0 = data.country_idx
    col_j1 = n_countries + data.country_idx
    col_j2 = 2 * n_countries + data.country_idx
    data_j0 = np.ones(n)
    data_j1 = data.time
    data_j2 = data.time**2

    J_sparse = sparse.csr_matrix(
        (np.concatenate([data_j0, data_j1, data_j2]),
         (np.tile(row_idx, 3), np.concatenate([col_j0, col_j1, col_j2]))),
        shape=(n, 3 * n_countries)
    )

    # k parameters: n_years - 1 sparse columns (drop k[0])
    mask = data.year_idx > 0
    row_k = np.arange(n)[mask]
    col_k = data.year_idx[mask] - 1

    K_sparse = sparse.csr_matrix(
        (np.ones(len(row_k)), (row_k, col_k)),
        shape=(n, n_years - 1)
    )

    # Combine: X_h (as sparse) | J_sparse | K_sparse
    X_sparse = sparse.hstack([sparse.csr_matrix(X_h), J_sparse, K_sparse])

    return X_sparse


def solve_constrained_subproblem(
    data: FittingData,
    g_values: np.ndarray,
    T_opt: float,
    P_opt: float,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve for h2, h4, j, k given fixed alpha, T_opt, P_opt.

    Uses sparse LSQR for efficiency (design matrix is ~1% dense).

    Args:
        data: FittingData object
        g_values: Precomputed g(GDP) values
        T_opt: Optimal temperature
        P_opt: Optimal precipitation

    Returns:
        h2, h4, j0, j1, j2, k
    """
    from scipy.sparse.linalg import lsqr

    X = build_constrained_design_matrix_sparse(data, g_values, T_opt, P_opt)
    y = data.growth_pcGDP

    # LSQR is an iterative solver efficient for sparse least squares
    # Use tight tolerances to ensure consistent objective values
    result = lsqr(X, y, atol=1e-14, btol=1e-14, iter_lim=10000)
    beta = result[0]

    n_countries = data.n_countries
    n_years = data.n_years

    h2 = beta[0]
    h4 = beta[1]

    j0 = beta[2:2 + n_countries]
    j1 = beta[2 + n_countries:2 + 2 * n_countries]
    j2 = beta[2 + 2 * n_countries:2 + 3 * n_countries]

    k = np.zeros(n_years)
    k[1:] = beta[2 + 3 * n_countries:]

    return h2, h4, j0, j1, j2, k


def compute_constrained_predictions(
    data: FittingData,
    g_values: np.ndarray,
    T_opt: float,
    P_opt: float,
    h2: float,
    h4: float,
    j0: np.ndarray,
    j1: np.ndarray,
    j2: np.ndarray,
    k: np.ndarray,
) -> np.ndarray:
    """Compute predictions for constrained model.

    pred = g * h2 * (T - T_opt)^2 + g * h4 * (P - P_opt)^2 + j + k
    """
    h_vals = h_func_constrained(data.temp, data.precp, T_opt, P_opt, h2, h4)
    j_vals = j_func(data.country_idx, data.time, j0, j1, j2)
    k_vals = k_func(data.year_idx, k)

    return g_values * h_vals + j_vals + k_vals


def objective_constrained(
    params: np.ndarray,
    GDP0: float,
    data: FittingData,
) -> float:
    """Objective function for constrained 3-parameter optimization.

    Args:
        params: [alpha, T_opt, P_opt]
        GDP0: Fixed reference GDP
        data: FittingData object

    Returns:
        Sum of squared residuals
    """
    alpha, T_opt, P_opt = params

    # Bounds checking (should be handled by optimizer, but be safe)
    if alpha <= 0 or alpha >= 1:
        return 1e20

    g_values = g_func(data.pcGDP, GDP0, alpha)
    h2, h4, j0, j1, j2, k = solve_constrained_subproblem(data, g_values, T_opt, P_opt)

    pred = compute_constrained_predictions(
        data, g_values, T_opt, P_opt, h2, h4, j0, j1, j2, k
    )
    residuals = data.growth_pcGDP - pred

    return np.sum(residuals**2)


def fit_model_constrained(
    data: FittingData,
    verbose: bool,
) -> FitResult:
    """Fit model with h(T*, P*) = 0 constraint.

    Uses reparameterization: h(T, P) = h2*(T - T*)^2 + h4*(P - P*)^2

    Optimizes 3 parameters (alpha, T_opt, P_opt) with linear subproblem
    for (h2, h4, j, k).

    Args:
        data: FittingData object
        verbose: Print progress

    Returns:
        FitResult object with constrained=True
    """
    GDP0 = data.pop_weighted_mean_gdp

    # Bounds for optimization
    T_min, T_max = data.temp.min(), data.temp.max()
    P_min, P_max = data.precp.min(), data.precp.max()

    if verbose:
        print(f"Fitting constrained model...")
        print(f"  Model: h(T, P) = h2*(T - T*)^2 + h4*(P - P*)^2")
        print(f"  N={data.n_obs}, countries={data.n_countries}, years={data.n_years}")
        print(f"  GDP0 (fixed) = {GDP0:.2f} (pop-weighted mean GDP, {data.gdp0_reference_year})")
        print(f"  T* bounds: [{T_min:.1f}, {T_max:.1f}]°C")
        print(f"  P* bounds: [{P_min:.2f}, {P_max:.2f}]")

    # Initial guess: alpha=0.4, T/P at median values
    x0 = np.array([0.4, np.median(data.temp), np.median(data.precp)])
    bounds = [(0.01, 0.99), (T_min, T_max), (P_min, P_max)]

    # Track evaluations
    eval_history = []

    def tracked_objective(params):
        obj = objective_constrained(params, GDP0, data)
        eval_history.append((params.copy(), obj))
        if verbose and len(eval_history) % 1 == 0:
            print(f"    Iteration {len(eval_history)}: "
                  f"alpha={params[0]:.6f}, T*={params[1]:.2f}, P*={params[2]:.2f}, "
                  f"obj={obj:.6f}")
        return obj

    if verbose:
        print(f"\n  Optimizing with L-BFGS-B...")

    # Use L-BFGS-B with moderate eps for finite-difference gradients
    # The sparse solver is fast enough that this works well
    result = optimize.minimize(
        tracked_objective,
        x0=x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 200, 'ftol': 1e-8, 'eps': 1e-3}
    )

    alpha, T_opt, P_opt = result.x
    final_obj = result.fun
    converged = result.success

    if verbose:
        print(f"  Optimization {'converged' if converged else 'stopped'}: "
              f"alpha={alpha:.6f}, T*={T_opt:.2f}°C, P*={P_opt:.4f}")
        print(f"  Final objective: {final_obj:.6f}")
        print(f"  Function evaluations: {len(eval_history)}")

    # Get final parameters
    g_values = g_func(data.pcGDP, GDP0, alpha)
    h2, h4, j0, j1, j2, k = solve_constrained_subproblem(data, g_values, T_opt, P_opt)

    # Convert to unconstrained form for ModelParams
    h0, h1, h2_out, h3, h4_out = constrained_to_unconstrained_h(T_opt, P_opt, h2, h4)

    params = ModelParams(
        GDP0=GDP0, alpha=alpha,
        h0=h0, h1=h1, h2=h2, h3=h3, h4=h4,
        j0=j0, j1=j1, j2=j2, k=k,
    )

    # Compute predictions and residuals
    predictions = compute_constrained_predictions(
        data, g_values, T_opt, P_opt, h2, h4, j0, j1, j2, k
    )
    residuals = data.growth_pcGDP - predictions

    # Fit statistics
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((data.growth_pcGDP - np.mean(data.growth_pcGDP))**2)
    r_squared = 1 - ss_res / ss_tot
    rmse = np.sqrt(ss_res / data.n_obs)

    # AIC/BIC: 3 nonlinear params (alpha, T_opt, P_opt) + 2 h params (h2, h4)
    # + 3*n_countries j params + (n_years-1) k params
    # But h2 and h4 are from linear solve, so effectively 3 nonlinear params
    # For consistency with unconstrained, count all linear params
    n_params = 1 + 2 + 3 * data.n_countries + (data.n_years - 1)  # alpha + h2,h4 + j + k
    log_likelihood = -data.n_obs / 2 * (np.log(2 * np.pi * ss_res / data.n_obs) + 1)
    aic = 2 * n_params - 2 * log_likelihood
    bic = np.log(data.n_obs) * n_params - 2 * log_likelihood

    if verbose:
        print(f"\nFit complete:")
        print(f"  R² = {r_squared:.4f}")
        print(f"  RMSE = {rmse:.6f}")
        print(f"  AIC = {aic:.1f}, BIC = {bic:.1f}")
        print(f"\nConstrained parameters:")
        print(f"  T* = {T_opt:.2f}°C")
        print(f"  P* = {P_opt:.4f}")
        print(f"  h2 = {h2:.6e}")
        print(f"  h4 = {h4:.6e}")
        print(f"\nDerived unconstrained parameters:")
        print(f"  h0 = {h0:.6e}")
        print(f"  h1 = {h1:.6e}")
        print(f"  h3 = {h3:.6e}")

    objective_history = [obj for _, obj in eval_history]

    return FitResult(
        params=params,
        residuals=residuals,
        r_squared=r_squared,
        rmse=rmse,
        aic=aic,
        bic=bic,
        n_iterations=len(eval_history),
        converged=converged,
        objective_history=objective_history,
        model_variant="base",  # Constrained only for base model
        constrained=True,
        T_opt=T_opt,
        P_opt=P_opt,
    )


def compute_standard_errors(
    data: FittingData,
    params: ModelParams,
    model_variant: str,
) -> ModelParams:
    """Compute standard errors via numerical Hessian.

    Uses finite differences to approximate the Hessian of the negative
    log-likelihood, then inverts to get the covariance matrix.

    GDP0 is treated as a known constant (not estimated), so it is excluded
    from the Hessian calculation. For non-base variants, h0 is also excluded
    (it's absorbed into j0).
    """
    param_vec = pack_params_for_hessian(params, data.n_countries, data.n_years, model_variant)
    GDP0 = params.GDP0

    def neg_log_likelihood(p):
        h_params, j0, j1, j2, k = unpack_params_for_hessian(
            p, data.n_countries, data.n_years, model_variant
        )
        alpha = p[0]
        g_values = g_func(data.pcGDP, GDP0, alpha)
        pred = compute_predictions(data, g_values, h_params, j0, j1, j2, k, model_variant)
        residuals = data.growth_pcGDP - pred
        ss_res = np.sum(residuals**2)
        return data.n_obs / 2 * np.log(ss_res / data.n_obs)

    try:
        hessian = compute_hessian(neg_log_likelihood, param_vec, eps=1e-5)
        cov = np.linalg.inv(hessian)
        se = np.sqrt(np.maximum(np.diag(cov), 0))
        params = unpack_standard_errors(params, se, data.n_countries, data.n_years, model_variant)
    except (np.linalg.LinAlgError, ValueError) as e:
        print(f"Warning: Could not compute standard errors: {e}")

    return params


def pack_params_for_hessian(
    params: ModelParams,
    n_countries: int,
    n_years: int,
    model_variant: str,
) -> np.ndarray:
    """Pack estimated parameters into a vector for Hessian computation.

    Excludes GDP0. For non-base variants, h0 is also excluded (absorbed into j0).
    """
    if model_variant == "base":
        h_part = [params.h0, params.h1, params.h2, params.h3, params.h4]
    else:
        h_part = [params.h1, params.h2, params.h3, params.h4]

    return np.concatenate([
        [params.alpha],
        h_part,
        params.j0,
        params.j1,
        params.j2,
        params.k,
    ])


def unpack_params_for_hessian(
    vec: np.ndarray,
    n_countries: int,
    n_years: int,
    model_variant: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unpack parameter vector for computing predictions.

    Returns: h_params (always length 5), j0, j1, j2, k
    """
    idx = 1  # Skip alpha (index 0)

    if model_variant == "base":
        h_params = vec[idx:idx + 5]
        idx += 5
    else:
        h_params = np.concatenate([[0.0], vec[idx:idx + 4]])
        idx += 4

    j0 = vec[idx:idx + n_countries]; idx += n_countries
    j1 = vec[idx:idx + n_countries]; idx += n_countries
    j2 = vec[idx:idx + n_countries]; idx += n_countries
    k = vec[idx:idx + n_years]

    return h_params, j0, j1, j2, k


def unpack_standard_errors(
    params: ModelParams,
    se: np.ndarray,
    n_countries: int,
    n_years: int,
    model_variant: str,
) -> ModelParams:
    """Add standard errors to ModelParams.

    GDP0 has no standard error since it is a fixed known value.
    For non-base variants, h0 SE is None (h0 is not estimated).
    """
    idx = 0

    params.se_GDP0 = None
    params.se_alpha = se[idx]; idx += 1

    if model_variant == "base":
        params.se_h0 = se[idx]; idx += 1
    else:
        params.se_h0 = None  # h0 absorbed into j0

    params.se_h1 = se[idx]; idx += 1
    params.se_h2 = se[idx]; idx += 1
    params.se_h3 = se[idx]; idx += 1
    params.se_h4 = se[idx]; idx += 1

    params.se_j0 = se[idx:idx + n_countries]; idx += n_countries
    params.se_j1 = se[idx:idx + n_countries]; idx += n_countries
    params.se_j2 = se[idx:idx + n_countries]; idx += n_countries

    params.se_k = se[idx:idx + n_years]

    return params


def compute_hessian(f, x, eps=1e-5):
    """Compute Hessian matrix using finite differences."""
    n = len(x)
    hessian = np.zeros((n, n))
    f0 = f(x)

    for i in range(n):
        x_plus_i = x.copy()
        x_plus_i[i] += eps
        x_minus_i = x.copy()
        x_minus_i[i] -= eps

        for j in range(i, n):
            if i == j:
                # Diagonal: second derivative
                f_plus = f(x_plus_i)
                f_minus = f(x_minus_i)
                hessian[i, i] = (f_plus - 2 * f0 + f_minus) / (eps**2)
            else:
                # Off-diagonal: mixed partial
                x_pp = x.copy()
                x_pp[i] += eps
                x_pp[j] += eps

                x_pm = x.copy()
                x_pm[i] += eps
                x_pm[j] -= eps

                x_mp = x.copy()
                x_mp[i] -= eps
                x_mp[j] += eps

                x_mm = x.copy()
                x_mm[i] -= eps
                x_mm[j] -= eps

                hessian[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * eps**2)
                hessian[j, i] = hessian[i, j]

    return hessian
