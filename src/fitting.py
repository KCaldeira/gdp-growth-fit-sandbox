"""Profile likelihood optimization for GDP growth curve fitting.

The model has both non-linear (g) and linear (h, j, k) components.
We use profile likelihood optimization:
1. For each candidate alpha, solve the full linear least squares for h, j, k
2. This gives the true objective as a function of alpha alone
3. Minimize over alpha using grid search + Brent's method
4. GDP0 is fixed to population-weighted mean GDP

This approach is superior to alternating estimation because it always
evaluates the true objective for each alpha candidate.
"""

import numpy as np
from scipy import optimize
from scipy.sparse import csr_matrix, hstack
from scipy.sparse.linalg import lsqr
from typing import Tuple, Optional
from dataclasses import dataclass

from .model import ModelParams, g_func, h_func, j_func, k_func, predict_from_data
from .data_loader import FittingData


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
    # Grid search results (if performed)
    grid_search_alphas: Optional[np.ndarray] = None
    grid_search_objectives: Optional[np.ndarray] = None


def build_linear_design_matrix(data: FittingData, g_values: np.ndarray) -> np.ndarray:
    """Build design matrix for linear subproblem given fixed g values.

    The linear model is:
        y = g*h0 + g*h1*T + g*h2*T^2 + g*h3*P + g*h4*P^2
            + j0[i] + j1[i]*t + j2[i]*t^2 + k[t]

    We drop k[0] = 0 for identifiability (first year is reference).

    Returns design matrix X and mapping info.
    """
    n = data.n_obs
    n_countries = data.n_countries
    n_years = data.n_years

    # h parameters: 5 columns
    # Columns: g, g*T, g*T^2, g*P, g*P^2
    X_h = np.column_stack([
        g_values,
        g_values * data.temp,
        g_values * data.temp**2,
        g_values * data.precp,
        g_values * data.precp**2,
    ])

    # j parameters: 3 * n_countries columns
    # For each country i: indicator, indicator*t, indicator*t^2
    X_j = np.zeros((n, 3 * n_countries))
    for obs_idx in range(n):
        i = data.country_idx[obs_idx]
        t = data.time[obs_idx]
        X_j[obs_idx, i] = 1.0                    # j0[i]
        X_j[obs_idx, n_countries + i] = t       # j1[i]
        X_j[obs_idx, 2 * n_countries + i] = t**2  # j2[i]

    # k parameters: n_years - 1 columns (drop k[0] for identifiability)
    # For year index y > 0: indicator
    X_k = np.zeros((n, n_years - 1))
    for obs_idx in range(n):
        y = data.year_idx[obs_idx]
        if y > 0:
            X_k[obs_idx, y - 1] = 1.0

    # Combine all columns
    X = np.hstack([X_h, X_j, X_k])

    return X


def solve_linear_subproblem(data: FittingData, g_values: np.ndarray
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve for h, j, k parameters given fixed g values.

    Uses ordinary least squares.

    Returns:
        h_params: array of shape (5,) for h0, h1, h2, h3, h4
        j0, j1, j2: arrays of shape (n_countries,)
        k: array of shape (n_years,) with k[0] = 0
    """
    X = build_linear_design_matrix(data, g_values)
    y = data.growth_pcGDP

    # Solve least squares: X @ beta = y
    # Use scipy.linalg.lstsq which is more robust than numpy
    from scipy import linalg
    beta, residuals, rank, s = linalg.lstsq(X, y, cond=None, lapack_driver='gelsy')

    # Extract parameters
    n_countries = data.n_countries
    n_years = data.n_years

    h_params = beta[:5]

    j0 = beta[5:5 + n_countries]
    j1 = beta[5 + n_countries:5 + 2 * n_countries]
    j2 = beta[5 + 2 * n_countries:5 + 3 * n_countries]

    # k with k[0] = 0
    k = np.zeros(n_years)
    k[1:] = beta[5 + 3 * n_countries:]

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


def objective_for_alpha(alpha: float, GDP0: float, data: FittingData) -> float:
    """Compute the best possible objective for a given alpha.

    This solves the full linear subproblem for h, j, k given alpha,
    then returns the sum of squared residuals.
    """
    if alpha <= 0 or alpha >= 1:
        return 1e20

    g_values = g_func(data.pcGDP, GDP0, alpha)
    h_params, j0, j1, j2, k = solve_linear_subproblem(data, g_values)

    # Compute predictions and residuals
    h_vals = h_func(data.temp, data.precp, *h_params)
    j_vals = j_func(data.country_idx, data.time, j0, j1, j2)
    k_vals = k_func(data.year_idx, k)

    pred = g_values * h_vals + j_vals + k_vals
    residuals = data.growth_pcGDP - pred

    return np.sum(residuals**2)


def fit_model(data: FittingData, verbose: bool = True) -> FitResult:
    """Fit the full model by optimizing alpha with linear subproblem solved exactly.

    For each candidate alpha, we solve the full linear least squares problem
    for h, j, k parameters. This gives the true objective as a function of alpha,
    which we then minimize using scipy's bounded optimizer (Brent's method).

    GDP0 is fixed to the population-weighted mean GDP from the data.

    Args:
        data: FittingData object
        verbose: Print progress

    Returns:
        FitResult object
    """
    # GDP0 is fixed to population-weighted mean GDP
    GDP0 = data.pop_weighted_mean_gdp

    if verbose:
        print(f"Fitting model...")
        print(f"  N={data.n_obs}, countries={data.n_countries}, years={data.n_years}")
        print(f"  GDP0 (fixed) = {GDP0:.2f} (pop-weighted mean GDP, {data.gdp0_reference_year})")

    # Optimize alpha using Brent's method on [0.01, 0.99]
    if verbose:
        print(f"\n  Optimizing alpha with Brent's method...")

    # Track all function evaluations
    eval_history = []

    def tracked_objective(alpha):
        obj = objective_for_alpha(alpha, GDP0, data)
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
    h_params, j0, j1, j2, k = solve_linear_subproblem(data, g_values)

    # Compute final predictions and residuals
    params = ModelParams(
        GDP0=GDP0, alpha=alpha,
        h0=h_params[0], h1=h_params[1], h2=h_params[2],
        h3=h_params[3], h4=h_params[4],
        j0=j0, j1=j1, j2=j2, k=k,
    )

    predictions = predict_from_data(data, params)
    residuals = data.growth_pcGDP - predictions

    # Compute fit statistics
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((data.growth_pcGDP - np.mean(data.growth_pcGDP))**2)
    r_squared = 1 - ss_res / ss_tot
    rmse = np.sqrt(ss_res / data.n_obs)

    # AIC/BIC (assuming Gaussian errors)
    # 1 (alpha) + 5 (h params) + 3*n_countries (j params) + (n_years - 1) (k params)
    n_params = 1 + 5 + 3 * data.n_countries + (data.n_years - 1)
    log_likelihood = -data.n_obs / 2 * (np.log(2 * np.pi * ss_res / data.n_obs) + 1)
    aic = 2 * n_params - 2 * log_likelihood
    bic = np.log(data.n_obs) * n_params - 2 * log_likelihood

    if verbose:
        print(f"\nFit complete:")
        print(f"  R² = {r_squared:.4f}")
        print(f"  RMSE = {rmse:.6f}")
        print(f"  AIC = {aic:.1f}, BIC = {bic:.1f}")

    # Compute standard errors
    params = compute_standard_errors(data, params)

    return FitResult(
        params=params,
        residuals=residuals,
        r_squared=r_squared,
        rmse=rmse,
        aic=aic,
        bic=bic,
        n_iterations=len(eval_history),  # Number of objective function evaluations
        converged=converged,
        objective_history=objective_history,
        grid_search_alphas=grid_search_alphas,
        grid_search_objectives=grid_search_objectives,
    )


def compute_standard_errors(data: FittingData, params: ModelParams) -> ModelParams:
    """Compute standard errors via numerical Hessian.

    Uses finite differences to approximate the Hessian of the negative
    log-likelihood, then inverts to get the covariance matrix.

    GDP0 is treated as a known constant (not estimated), so it is excluded
    from the Hessian calculation.
    """
    # Pack estimated parameters (excluding GDP0 which is fixed)
    param_vec = pack_params_for_hessian(params, data.n_countries, data.n_years)
    GDP0 = params.GDP0  # Fixed value

    def neg_log_likelihood(p):
        params_temp = unpack_params_for_hessian(p, GDP0, data.n_countries, data.n_years)
        pred = predict_from_data(data, params_temp)
        residuals = data.growth_pcGDP - pred
        ss_res = np.sum(residuals**2)
        # Negative log likelihood (up to constant)
        return data.n_obs / 2 * np.log(ss_res / data.n_obs)

    # Compute Hessian numerically
    try:
        hessian = compute_hessian(neg_log_likelihood, param_vec, eps=1e-5)

        # Invert to get covariance matrix
        cov = np.linalg.inv(hessian)
        se = np.sqrt(np.maximum(np.diag(cov), 0))

        # Unpack standard errors (GDP0 has no SE since it's fixed)
        params = unpack_standard_errors(params, se, data.n_countries, data.n_years)
    except (np.linalg.LinAlgError, ValueError) as e:
        print(f"Warning: Could not compute standard errors: {e}")

    return params


def pack_params_for_hessian(params: ModelParams, n_countries: int, n_years: int) -> np.ndarray:
    """Pack estimated parameters into a vector for Hessian computation.

    Excludes GDP0 since it is treated as a known constant.
    """
    return np.concatenate([
        [params.alpha],
        [params.h0, params.h1, params.h2, params.h3, params.h4],
        params.j0,
        params.j1,
        params.j2,
        params.k,
    ])


def unpack_params_for_hessian(vec: np.ndarray, GDP0: float,
                               n_countries: int, n_years: int) -> ModelParams:
    """Unpack parameter vector into ModelParams, with GDP0 provided separately."""
    idx = 0

    alpha = vec[idx]; idx += 1

    h0 = vec[idx]; idx += 1
    h1 = vec[idx]; idx += 1
    h2 = vec[idx]; idx += 1
    h3 = vec[idx]; idx += 1
    h4 = vec[idx]; idx += 1

    j0 = vec[idx:idx + n_countries]; idx += n_countries
    j1 = vec[idx:idx + n_countries]; idx += n_countries
    j2 = vec[idx:idx + n_countries]; idx += n_countries

    k = vec[idx:idx + n_years]; idx += n_years

    return ModelParams(
        GDP0=GDP0, alpha=alpha,
        h0=h0, h1=h1, h2=h2, h3=h3, h4=h4,
        j0=j0, j1=j1, j2=j2, k=k,
    )


def unpack_standard_errors(params: ModelParams, se: np.ndarray,
                           n_countries: int, n_years: int) -> ModelParams:
    """Add standard errors to ModelParams.

    GDP0 has no standard error since it is a fixed known value.
    """
    idx = 0

    params.se_GDP0 = None  # GDP0 is fixed, no SE
    params.se_alpha = se[idx]; idx += 1

    params.se_h0 = se[idx]; idx += 1
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
