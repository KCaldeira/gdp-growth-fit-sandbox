"""Monte Carlo cluster bootstrap uncertainty analysis.

This module implements cluster bootstrap (resampling countries with replacement)
for computing uncertainty bands on h(T, P_const), g(GDP), and g(GDP)*h(T, P_const)
without assuming normality of errors.

The cluster bootstrap approach:
1. Fit the original model to get point estimates
2. For B bootstrap iterations:
   - Resample n_countries countries with replacement
   - Build a new dataset from the selected countries (with re-indexed country IDs)
   - Refit the model to get bootstrap parameter estimates
3. Use the empirical distribution of parameters across bootstrap samples
   to compute percentile-based confidence bands

This preserves within-country correlation structure across years.
"""

import numpy as np
from dataclasses import dataclass
from scipy import optimize
from typing import Tuple

from .model import (
    ModelParams, g_func, h_func, h_func_constrained, j_func, k_func,
    predict_from_data, constrained_to_unconstrained_h,
)
from .data_loader import FittingData
from .fitting import (
    build_linear_design_matrix,
    solve_linear_subproblem,
    compute_predictions,
    solve_constrained_subproblem,
    compute_constrained_predictions,
)


@dataclass
class BootstrapResult:
    """Container for bootstrap results."""
    # Point estimates (from original fit)
    params: ModelParams

    # Bootstrap parameter samples
    alpha_samples: np.ndarray  # shape (B,)
    h_samples: np.ndarray      # shape (B, 5) for h0, h1, h2, h3, h4

    # Optional: full parameter samples (not stored by default to save memory)
    # j and k parameters vary in dimension across bootstrap samples due to
    # potentially different numbers of unique countries/years

    # Metadata
    n_bootstrap: int
    n_successful: int  # number of successful bootstrap fits
    model_variant: str  # "base", "g_scales_hj", or "g_scales_all"

    # Constrained model parameters (optional, when constrained=True)
    constrained: bool = False
    T_opt_samples: np.ndarray = None  # shape (B,) for constrained model
    P_opt_samples: np.ndarray = None  # shape (B,) for constrained model
    T_opt_point: float = None  # Point estimate of T*
    P_opt_point: float = None  # Point estimate of P*


def create_bootstrap_data(
    data: FittingData,
    selected_country_indices: np.ndarray,
) -> FittingData:
    """Create a bootstrap dataset by selecting countries with replacement.

    Args:
        data: Original FittingData
        selected_country_indices: Array of country indices to include (with replacement)

    Returns:
        New FittingData with resampled countries (re-indexed)
    """
    # Build list of observations to include, with new country indices
    obs_indices = []
    new_country_idx = []

    for new_idx, orig_country_idx in enumerate(selected_country_indices):
        # Find all observations for this original country
        country_obs = np.where(data.country_idx == orig_country_idx)[0]
        obs_indices.extend(country_obs)
        new_country_idx.extend([new_idx] * len(country_obs))

    obs_indices = np.array(obs_indices)
    new_country_idx = np.array(new_country_idx, dtype=np.int32)

    # Extract data for selected observations
    growth_pcGDP = data.growth_pcGDP[obs_indices]
    pcGDP = data.pcGDP[obs_indices]
    temp = data.temp[obs_indices]
    precp = data.precp[obs_indices]
    time = data.time[obs_indices]
    year_idx = data.year_idx[obs_indices]
    pop = data.pop[obs_indices]

    # Remap year indices to be contiguous (in case some years are missing)
    unique_years_in_sample = np.unique(year_idx)
    year_remap = {old: new for new, old in enumerate(unique_years_in_sample)}
    new_year_idx = np.array([year_remap[y] for y in year_idx], dtype=np.int32)

    # Create new FittingData
    # Note: mappings are not fully populated since we don't need them for fitting
    return FittingData(
        growth_pcGDP=growth_pcGDP,
        pcGDP=pcGDP,
        temp=temp,
        precp=precp,
        time=time,
        country_idx=new_country_idx,
        year_idx=new_year_idx,
        pop=pop,
        iso_to_idx={},  # Not needed for fitting
        idx_to_iso={},
        year_to_idx={},
        idx_to_year={},
        n_obs=len(growth_pcGDP),
        n_countries=len(selected_country_indices),
        n_years=len(unique_years_in_sample),
        pop_weighted_mean_gdp=data.pop_weighted_mean_gdp,
        pop_weighted_mean_precp=data.pop_weighted_mean_precp,
        gdp0_reference_year=data.gdp0_reference_year,
    )


def fit_bootstrap_sample(
    boot_data: FittingData,
    GDP0: float,
    model_variant: str,
) -> Tuple[float, np.ndarray]:
    """Fit model to a bootstrap sample.

    Returns:
        alpha, h_params (array of 5 values, with h0=0 for non-base variants)
    """
    def objective_for_alpha(alpha: float) -> float:
        """Compute best SSR for given alpha."""
        if alpha <= 0 or alpha >= 1:
            return 1e20

        g_values = g_func(boot_data.pcGDP, GDP0, alpha)
        h_params, j0, j1, j2, k = solve_linear_subproblem(boot_data, g_values, model_variant)

        pred = compute_predictions(boot_data, g_values, h_params, j0, j1, j2, k, model_variant)
        residuals = boot_data.growth_pcGDP - pred

        return np.sum(residuals**2)

    # Optimize alpha
    result = optimize.minimize_scalar(
        objective_for_alpha,
        bounds=(0.01, 0.99),
        method='bounded',
        options={'xatol': 1e-6}
    )

    alpha = result.x

    # Get h parameters at optimal alpha
    g_values = g_func(boot_data.pcGDP, GDP0, alpha)
    h_params, _, _, _, _ = solve_linear_subproblem(boot_data, g_values, model_variant)

    return alpha, h_params


def fit_bootstrap_sample_constrained(
    boot_data: FittingData,
    GDP0: float,
    T_bounds: Tuple[float, float],
    P_bounds: Tuple[float, float],
) -> Tuple[float, float, float, float, float]:
    """Fit constrained model to a bootstrap sample.

    Uses 3-parameter optimization over (alpha, T_opt, P_opt) with linear
    subproblem for (h2, h4, j, k).

    Args:
        boot_data: Bootstrap sample data
        GDP0: Fixed reference GDP
        T_bounds: (T_min, T_max) bounds for optimal temperature
        P_bounds: (P_min, P_max) bounds for optimal precipitation

    Returns:
        alpha, T_opt, P_opt, h2, h4
    """
    def objective_for_params(params: np.ndarray) -> float:
        """Compute best SSR for given (alpha, T_opt, P_opt)."""
        alpha, T_opt, P_opt = params

        if alpha <= 0 or alpha >= 1:
            return 1e20

        g_values = g_func(boot_data.pcGDP, GDP0, alpha)
        h2, h4, j0, j1, j2, k = solve_constrained_subproblem(
            boot_data, g_values, T_opt, P_opt
        )

        pred = compute_constrained_predictions(
            boot_data, g_values, T_opt, P_opt, h2, h4, j0, j1, j2, k
        )
        residuals = boot_data.growth_pcGDP - pred

        return np.sum(residuals**2)

    # Initial guess
    x0 = np.array([0.4, np.median(boot_data.temp), np.median(boot_data.precp)])
    bounds = [(0.01, 0.99), T_bounds, P_bounds]

    # Optimize
    result = optimize.minimize(
        objective_for_params,
        x0=x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-8}
    )

    alpha, T_opt, P_opt = result.x

    # Get h2, h4 at optimal parameters
    g_values = g_func(boot_data.pcGDP, GDP0, alpha)
    h2, h4, _, _, _, _ = solve_constrained_subproblem(
        boot_data, g_values, T_opt, P_opt
    )

    return alpha, T_opt, P_opt, h2, h4


def run_bootstrap(
    data: FittingData,
    params: ModelParams,
    n_bootstrap: int,
    random_seed: int,
    model_variant: str,
    verbose: bool,
    constrained: bool,
    T_opt_point: float,
    P_opt_point: float,
) -> BootstrapResult:
    """Run cluster bootstrap to get parameter uncertainty.

    Resamples countries with replacement, preserving within-country
    correlation structure.

    Args:
        data: FittingData object
        params: Fitted model parameters (point estimates)
        n_bootstrap: Number of bootstrap iterations
        random_seed: Random seed for reproducibility
        model_variant: One of "base", "g_scales_hj", or "g_scales_all"
        verbose: Print progress
        constrained: Whether to use constrained model (h(T*, P*) = 0)
        T_opt_point: Point estimate of T* (used for constrained=True, ignored otherwise)
        P_opt_point: Point estimate of P* (used for constrained=True, ignored otherwise)

    Returns:
        BootstrapResult with parameter samples
    """
    rng = np.random.default_rng(random_seed)
    GDP0 = params.GDP0

    # Storage for bootstrap samples
    alpha_samples = np.zeros(n_bootstrap)
    h_samples = np.zeros((n_bootstrap, 5))

    # Constrained-specific storage
    T_opt_samples = np.zeros(n_bootstrap) if constrained else None
    P_opt_samples = np.zeros(n_bootstrap) if constrained else None

    # Bounds for constrained optimization
    T_bounds = (data.temp.min(), data.temp.max())
    P_bounds = (data.precp.min(), data.precp.max())

    n_successful = 0

    if verbose:
        print(f"Running cluster bootstrap with {n_bootstrap} iterations...")
        if constrained:
            print(f"  Mode: Constrained (h(T*, P*) = 0)")
        else:
            print(f"  Model variant: {model_variant}")
        print(f"  Resampling {data.n_countries} countries with replacement")

    for b in range(n_bootstrap):
        if verbose and (b + 1) % 100 == 0:
            print(f"  Bootstrap iteration {b + 1}/{n_bootstrap}")

        # Sample countries with replacement
        selected_countries = rng.integers(0, data.n_countries, size=data.n_countries)

        # Create bootstrap dataset
        boot_data = create_bootstrap_data(data, selected_countries)

        # Fit model to bootstrap sample
        try:
            if constrained:
                alpha_b, T_opt_b, P_opt_b, h2_b, h4_b = fit_bootstrap_sample_constrained(
                    boot_data, GDP0, T_bounds, P_bounds
                )
                # Convert to unconstrained h parameters for storage
                h0_b, h1_b, _, h3_b, _ = constrained_to_unconstrained_h(
                    T_opt_b, P_opt_b, h2_b, h4_b
                )
                h_b = np.array([h0_b, h1_b, h2_b, h3_b, h4_b])

                T_opt_samples[b] = T_opt_b
                P_opt_samples[b] = P_opt_b
            else:
                alpha_b, h_b = fit_bootstrap_sample(boot_data, GDP0, model_variant)

            alpha_samples[b] = alpha_b
            h_samples[b, :] = h_b
            n_successful += 1

        except Exception as e:
            if verbose:
                print(f"  Warning: Bootstrap iteration {b} failed: {e}")
            alpha_samples[b] = np.nan
            h_samples[b, :] = np.nan
            if constrained:
                T_opt_samples[b] = np.nan
                P_opt_samples[b] = np.nan

    if verbose:
        print(f"  Completed {n_successful}/{n_bootstrap} bootstrap iterations")

    return BootstrapResult(
        params=params,
        alpha_samples=alpha_samples,
        h_samples=h_samples,
        n_bootstrap=n_bootstrap,
        n_successful=n_successful,
        model_variant=model_variant,
        constrained=constrained,
        T_opt_samples=T_opt_samples,
        P_opt_samples=P_opt_samples,
        T_opt_point=T_opt_point if constrained else None,
        P_opt_point=P_opt_point if constrained else None,
    )


def compute_h_uncertainty_bands(
    bootstrap_result: BootstrapResult,
    T_values: np.ndarray,
    P_const: float,
    percentiles: Tuple[float, float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute uncertainty bands for h(T, P_const) as a function of T.

    Args:
        bootstrap_result: Result from run_bootstrap
        T_values: Array of temperature values to evaluate
        P_const: Fixed precipitation value
        percentiles: Tuple of (lower, median, upper) percentiles, e.g., (2.5, 50, 97.5)

    Returns:
        Tuple of (lower_band, median, upper_band), each shape (len(T_values),)
    """
    valid_mask = ~np.isnan(bootstrap_result.alpha_samples)
    h_samples = bootstrap_result.h_samples[valid_mask]

    n_T = len(T_values)
    n_samples = len(h_samples)

    # Compute h(T, P_const) for each bootstrap sample
    # h = h0 + h1*T + h2*T^2 + h3*P + h4*P^2
    h_values = np.zeros((n_samples, n_T))
    for i in range(n_samples):
        h0, h1, h2, h3, h4 = h_samples[i]
        h_values[i, :] = h0 + h1 * T_values + h2 * T_values**2 + h3 * P_const + h4 * P_const**2

    lower = np.percentile(h_values, percentiles[0], axis=0)
    median = np.percentile(h_values, percentiles[1], axis=0)
    upper = np.percentile(h_values, percentiles[2], axis=0)

    return lower, median, upper


def compute_g_uncertainty_bands(
    bootstrap_result: BootstrapResult,
    GDP_values: np.ndarray,
    percentiles: Tuple[float, float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute uncertainty bands for g(GDP) as a function of GDP.

    Args:
        bootstrap_result: Result from run_bootstrap
        GDP_values: Array of GDP values to evaluate
        percentiles: Tuple of (lower, median, upper) percentiles

    Returns:
        Tuple of (lower_band, median, upper_band), each shape (len(GDP_values),)
    """
    valid_mask = ~np.isnan(bootstrap_result.alpha_samples)
    alpha_samples = bootstrap_result.alpha_samples[valid_mask]
    GDP0 = bootstrap_result.params.GDP0

    n_GDP = len(GDP_values)
    n_samples = len(alpha_samples)

    # Compute g(GDP) for each bootstrap sample
    g_values = np.zeros((n_samples, n_GDP))
    for i in range(n_samples):
        g_values[i, :] = g_func(GDP_values, GDP0, alpha_samples[i])

    lower = np.percentile(g_values, percentiles[0], axis=0)
    median = np.percentile(g_values, percentiles[1], axis=0)
    upper = np.percentile(g_values, percentiles[2], axis=0)

    return lower, median, upper


def compute_gh_uncertainty_bands(
    bootstrap_result: BootstrapResult,
    GDP_values: np.ndarray,
    T_values: np.ndarray,
    P_const: float,
    percentiles: Tuple[float, float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute uncertainty bands for g(GDP)*h(T, P_const).

    This computes the full climate effect including GDP scaling.

    Args:
        bootstrap_result: Result from run_bootstrap
        GDP_values: Array of GDP values (length n_GDP)
        T_values: Array of temperature values (length n_T)
        P_const: Fixed precipitation value
        percentiles: Tuple of (lower, median, upper) percentiles

    Returns:
        Tuple of (lower_band, median, upper_band), each shape (n_GDP, n_T)
    """
    valid_mask = ~np.isnan(bootstrap_result.alpha_samples)
    alpha_samples = bootstrap_result.alpha_samples[valid_mask]
    h_samples = bootstrap_result.h_samples[valid_mask]
    GDP0 = bootstrap_result.params.GDP0

    n_GDP = len(GDP_values)
    n_T = len(T_values)
    n_samples = len(alpha_samples)

    # Compute g(GDP)*h(T) for each bootstrap sample
    # Result is 3D: (n_samples, n_GDP, n_T)
    gh_values = np.zeros((n_samples, n_GDP, n_T))

    for i in range(n_samples):
        # g(GDP) for this sample
        g_i = g_func(GDP_values, GDP0, alpha_samples[i])  # shape (n_GDP,)

        # h(T, P_const) for this sample
        h0, h1, h2, h3, h4 = h_samples[i]
        h_i = h0 + h1 * T_values + h2 * T_values**2 + h3 * P_const + h4 * P_const**2  # shape (n_T,)

        # Outer product: g_i[:, None] * h_i[None, :]
        gh_values[i, :, :] = g_i[:, np.newaxis] * h_i[np.newaxis, :]

    lower = np.percentile(gh_values, percentiles[0], axis=0)
    median = np.percentile(gh_values, percentiles[1], axis=0)
    upper = np.percentile(gh_values, percentiles[2], axis=0)

    return lower, median, upper


def compute_gh_at_fixed_gdp(
    bootstrap_result: BootstrapResult,
    GDP_value: float,
    T_values: np.ndarray,
    P_const: float,
    percentiles: Tuple[float, float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute uncertainty bands for g(GDP)*h(T, P_const) at a single GDP value.

    Convenience function for plotting climate response at a specific GDP level.

    Args:
        bootstrap_result: Result from run_bootstrap
        GDP_value: Single GDP value
        T_values: Array of temperature values
        P_const: Fixed precipitation value
        percentiles: Tuple of percentiles

    Returns:
        Tuple of (lower_band, median, upper_band), each shape (len(T_values),)
    """
    valid_mask = ~np.isnan(bootstrap_result.alpha_samples)
    alpha_samples = bootstrap_result.alpha_samples[valid_mask]
    h_samples = bootstrap_result.h_samples[valid_mask]
    GDP0 = bootstrap_result.params.GDP0

    n_T = len(T_values)
    n_samples = len(alpha_samples)

    gh_values = np.zeros((n_samples, n_T))

    for i in range(n_samples):
        g_i = g_func(np.array([GDP_value]), GDP0, alpha_samples[i])[0]
        h0, h1, h2, h3, h4 = h_samples[i]
        h_i = h0 + h1 * T_values + h2 * T_values**2 + h3 * P_const + h4 * P_const**2
        gh_values[i, :] = g_i * h_i

    lower = np.percentile(gh_values, percentiles[0], axis=0)
    median = np.percentile(gh_values, percentiles[1], axis=0)
    upper = np.percentile(gh_values, percentiles[2], axis=0)

    return lower, median, upper


def compute_optimal_T_distribution(
    bootstrap_result: BootstrapResult,
) -> np.ndarray:
    """Get the bootstrap distribution of optimal temperature T*.

    The optimal temperature maximizes h(T), i.e., T* = -h1 / (2*h2).
    (Note: h2 < 0 for a maximum, so this is where dh/dT = 0)

    Returns:
        Array of optimal T values from each bootstrap sample
    """
    valid_mask = ~np.isnan(bootstrap_result.alpha_samples)
    h_samples = bootstrap_result.h_samples[valid_mask]

    h1 = h_samples[:, 1]
    h2 = h_samples[:, 2]

    T_optimal = -h1 / (2 * h2)

    return T_optimal


def compute_dhdT_uncertainty_bands(
    bootstrap_result: BootstrapResult,
    T_values: np.ndarray,
    percentiles: Tuple[float, float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute uncertainty bands for dh/dT as a function of T.

    dh/dT = h1 + 2*h2*T

    Args:
        bootstrap_result: Result from run_bootstrap
        T_values: Array of temperature values
        percentiles: Tuple of percentiles

    Returns:
        Tuple of (lower_band, median, upper_band)
    """
    valid_mask = ~np.isnan(bootstrap_result.alpha_samples)
    h_samples = bootstrap_result.h_samples[valid_mask]

    n_T = len(T_values)
    n_samples = len(h_samples)

    dhdT_values = np.zeros((n_samples, n_T))
    for i in range(n_samples):
        h1 = h_samples[i, 1]
        h2 = h_samples[i, 2]
        dhdT_values[i, :] = h1 + 2 * h2 * T_values

    lower = np.percentile(dhdT_values, percentiles[0], axis=0)
    median = np.percentile(dhdT_values, percentiles[1], axis=0)
    upper = np.percentile(dhdT_values, percentiles[2], axis=0)

    return lower, median, upper
