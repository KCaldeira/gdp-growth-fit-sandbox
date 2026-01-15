"""Alternating estimation algorithm for GDP growth curve fitting.

The model has both non-linear (g) and linear (h, j, k) components.
We use alternating estimation:
1. Fix GDP0, alpha -> solve linear problem for h, j, k
2. Fix h, j, k -> optimize over GDP0, alpha
3. Iterate until convergence
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
    beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

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


def fit_model(data: FittingData, max_iter: int = 100, tol: float = 1e-6,
              alpha_init: float = 0.1, verbose: bool = True) -> FitResult:
    """Fit the full model using alternating estimation.

    GDP0 is fixed to the population-weighted mean GDP from the data.

    Args:
        data: FittingData object
        max_iter: Maximum number of alternating iterations
        tol: Convergence tolerance (relative change in objective)
        alpha_init: Initial value for alpha
        verbose: Print progress

    Returns:
        FitResult object
    """
    # GDP0 is fixed to population-weighted mean GDP
    GDP0 = data.pop_weighted_mean_gdp
    alpha = alpha_init
    objective_history = []

    if verbose:
        print(f"Starting alternating estimation...")
        print(f"  N={data.n_obs}, countries={data.n_countries}, years={data.n_years}")
        print(f"  GDP0 (fixed) = {GDP0:.2f} (pop-weighted mean GDP, {data.gdp0_reference_year})")

    for iteration in range(max_iter):
        # Step 1: Fix g, solve for h, j, k
        g_values = g_func(data.pcGDP, GDP0, alpha)
        h_params, j0, j1, j2, k = solve_linear_subproblem(data, g_values)

        # Step 2: Fix h, j, k, optimize over alpha only (GDP0 is fixed)
        result = optimize.minimize_scalar(
            objective_alpha_only,
            args=(GDP0, data, h_params, j0, j1, j2, k),
            bounds=(0.0, 1.0),
            method='bounded',
        )

        alpha_new = result.x
        obj = result.fun
        objective_history.append(obj)

        # Check convergence
        if iteration > 0:
            rel_change = abs(objective_history[-1] - objective_history[-2]) / (
                abs(objective_history[-2]) + 1e-10
            )
            if verbose and iteration % 10 == 0:
                print(f"  Iter {iteration}: obj={obj:.4f}, "
                      f"alpha={alpha_new:.4f}, rel_change={rel_change:.2e}")
            if rel_change < tol:
                if verbose:
                    print(f"  Converged at iteration {iteration}")
                alpha = alpha_new
                break
        else:
            if verbose:
                print(f"  Iter {iteration}: obj={obj:.4f}, alpha={alpha_new:.4f}")

        alpha = alpha_new

    converged = iteration < max_iter - 1

    # Final solve for linear parameters with converged g
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
        n_iterations=iteration + 1,
        converged=converged,
        objective_history=objective_history,
    )


def compute_standard_errors(data: FittingData, params: ModelParams) -> ModelParams:
    """Compute standard errors via numerical Hessian.

    Uses finite differences to approximate the Hessian of the negative
    log-likelihood, then inverts to get the covariance matrix.
    """
    # Pack all parameters into a single vector
    param_vec = pack_params(params, data.n_countries, data.n_years)

    def neg_log_likelihood(p):
        params_temp = unpack_params(p, data.n_countries, data.n_years)
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

        # Unpack standard errors
        params = unpack_standard_errors(params, se, data.n_countries, data.n_years)
    except (np.linalg.LinAlgError, ValueError) as e:
        print(f"Warning: Could not compute standard errors: {e}")

    return params


def pack_params(params: ModelParams, n_countries: int, n_years: int) -> np.ndarray:
    """Pack model parameters into a single vector."""
    return np.concatenate([
        [params.GDP0, params.alpha],
        [params.h0, params.h1, params.h2, params.h3, params.h4],
        params.j0,
        params.j1,
        params.j2,
        params.k,
    ])


def unpack_params(vec: np.ndarray, n_countries: int, n_years: int) -> ModelParams:
    """Unpack parameter vector into ModelParams."""
    idx = 0

    GDP0 = vec[idx]; idx += 1
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
    """Add standard errors to ModelParams."""
    idx = 0

    params.se_GDP0 = se[idx]; idx += 1
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
