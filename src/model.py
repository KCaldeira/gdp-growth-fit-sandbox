"""Model component functions for GDP growth curve fitting.

Model: growth_pcGDP(i,t) = g(pcGDP[i,t]) * h(T[i,t], P[i,t]) + j(i,t) + k(t)

Where:
    g(GDP) = (GDP / GDP0)^(-alpha)
    h(T, P) = h0 + h1*T + h2*T^2 + h3*P + h4*P^2
    j(i, t) = j0[i] + j1[i]*t + j2[i]*t^2
    k(t) = k[t]
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelParams:
    """Container for all model parameters."""
    # g function parameters
    GDP0: float
    alpha: float

    # h function parameters (climate response)
    h0: float
    h1: float
    h2: float
    h3: float
    h4: float

    # j function parameters (country-specific trends)
    # Shape: (n_countries,) for each
    j0: np.ndarray
    j1: np.ndarray
    j2: np.ndarray

    # k function parameters (year fixed effects)
    # Shape: (n_years,)
    k: np.ndarray

    # Mean-centered j and adjusted k (optional, for alternative parameterization)
    # In this parameterization: sum(j0_mc) = sum(j1_mc) = sum(j2_mc) = 0
    # and k_mc is adjusted so predictions are numerically identical
    j0_mc: Optional[np.ndarray] = None
    j1_mc: Optional[np.ndarray] = None
    j2_mc: Optional[np.ndarray] = None
    k_mc: Optional[np.ndarray] = None

    # Standard errors (optional, populated after fitting)
    se_GDP0: Optional[float] = None
    se_alpha: Optional[float] = None
    se_h0: Optional[float] = None
    se_h1: Optional[float] = None
    se_h2: Optional[float] = None
    se_h3: Optional[float] = None
    se_h4: Optional[float] = None
    se_j0: Optional[np.ndarray] = None
    se_j1: Optional[np.ndarray] = None
    se_j2: Optional[np.ndarray] = None
    se_k: Optional[np.ndarray] = None


def g_func(GDP: np.ndarray, GDP0: float, alpha: float) -> np.ndarray:
    """Compute g(GDP) = (GDP / GDP0)^(-alpha).

    This represents the GDP convergence term - poorer countries
    have higher potential growth rates.

    Args:
        GDP: Per capita GDP values
        GDP0: Reference GDP level
        alpha: Convergence parameter

    Returns:
        g values for each observation
    """
    return np.power(GDP / GDP0, -alpha)


def h_func(T: np.ndarray, P: np.ndarray,
           h0: float, h1: float, h2: float, h3: float, h4: float) -> np.ndarray:
    """Compute h(T, P) = h0 + h1*T + h2*T^2 + h3*P + h4*P^2.

    This represents the climate response function.

    Args:
        T: Temperature values
        P: Log precipitation values
        h0-h4: Climate response coefficients

    Returns:
        h values for each observation
    """
    return h0 + h1 * T + h2 * T**2 + h3 * P + h4 * P**2


def h_func_constrained(T: np.ndarray, P: np.ndarray,
                       T_opt: float, P_opt: float, h2: float, h4: float) -> np.ndarray:
    """Compute constrained h(T, P) = h2*(T - T_opt)^2 + h4*(P - P_opt)^2.

    This parameterization enforces h(T_opt, P_opt) = 0, normalizing the
    climate response function to pass through zero at optimal temperature
    and precipitation.

    The equivalent unconstrained parameters are:
        h1 = -2*h2*T_opt
        h3 = -2*h4*P_opt
        h0 = h2*T_opt^2 + h4*P_opt^2

    Args:
        T: Temperature values
        P: Log precipitation values
        T_opt: Optimal temperature (where h is minimized in T)
        P_opt: Optimal precipitation (where h is minimized in P)
        h2: Quadratic coefficient for temperature (typically < 0 for maximum)
        h4: Quadratic coefficient for precipitation (typically < 0 for maximum)

    Returns:
        h values for each observation
    """
    return h2 * (T - T_opt)**2 + h4 * (P - P_opt)**2


def constrained_to_unconstrained_h(T_opt: float, P_opt: float,
                                    h2: float, h4: float) -> tuple:
    """Convert constrained h parameters to unconstrained form.

    Args:
        T_opt: Optimal temperature
        P_opt: Optimal precipitation
        h2: Quadratic temperature coefficient
        h4: Quadratic precipitation coefficient

    Returns:
        Tuple of (h0, h1, h2, h3, h4)
    """
    h1 = -2 * h2 * T_opt
    h3 = -2 * h4 * P_opt
    h0 = h2 * T_opt**2 + h4 * P_opt**2
    return h0, h1, h2, h3, h4


def j_func(country_idx: np.ndarray, time: np.ndarray,
           j0: np.ndarray, j1: np.ndarray, j2: np.ndarray) -> np.ndarray:
    """Compute j(i, t) = j0[i] + j1[i]*t + j2[i]*t^2.

    This represents country-specific time trends.

    Args:
        country_idx: Country index for each observation
        time: Time values for each observation
        j0, j1, j2: Country-specific coefficients (shape: n_countries)

    Returns:
        j values for each observation
    """
    return j0[country_idx] + j1[country_idx] * time + j2[country_idx] * time**2


def k_func(year_idx: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Compute k(t) = k[t].

    This represents year fixed effects.

    Args:
        year_idx: Year index for each observation
        k: Year fixed effects (shape: n_years)

    Returns:
        k values for each observation
    """
    return k[year_idx]


def predict(pcGDP: np.ndarray, temp: np.ndarray, precp: np.ndarray,
            time: np.ndarray, country_idx: np.ndarray, year_idx: np.ndarray,
            params: ModelParams) -> np.ndarray:
    """Compute full model prediction.

    growth_pcGDP(i,t) = g(pcGDP[i,t]) * h(T[i,t], P[i,t]) + j(i,t) + k(t)

    Args:
        pcGDP: Per capita GDP values
        temp: Temperature values
        precp: Log precipitation values
        time: Time values
        country_idx: Country indices
        year_idx: Year indices
        params: Model parameters

    Returns:
        Predicted growth_pcGDP values
    """
    g = g_func(pcGDP, params.GDP0, params.alpha)
    h = h_func(temp, precp, params.h0, params.h1, params.h2, params.h3, params.h4)
    j = j_func(country_idx, time, params.j0, params.j1, params.j2)
    k = k_func(year_idx, params.k)

    return g * h + j + k


def predict_from_data(data, params: ModelParams) -> np.ndarray:
    """Convenience function to predict using FittingData object."""
    return predict(
        data.pcGDP, data.temp, data.precp, data.time,
        data.country_idx, data.year_idx, params
    )


def compute_mean_centered_params(params: ModelParams, n_years: int) -> None:
    """Compute mean-centered j parameters and adjusted k parameters.

    This transformation re-parameterizes the model so that:
    - sum(j0_mc) = sum(j1_mc) = sum(j2_mc) = 0
    - k_mc is adjusted to give numerically identical predictions

    The transformation is:
        j0_mc[i] = j0[i] - mean(j0)
        j1_mc[i] = j1[i] - mean(j1)
        j2_mc[i] = j2[i] - mean(j2)
        k_mc[t] = k[t] + mean(j0) + mean(j1)*t + mean(j2)*t^2

    This modifies params in place, adding j0_mc, j1_mc, j2_mc, k_mc.

    Args:
        params: ModelParams object (modified in place)
        n_years: Number of years (for generating time indices)
    """
    m0 = np.mean(params.j0)
    m1 = np.mean(params.j1)
    m2 = np.mean(params.j2)

    params.j0_mc = params.j0 - m0
    params.j1_mc = params.j1 - m1
    params.j2_mc = params.j2 - m2

    t = np.arange(n_years)
    params.k_mc = params.k + m0 + m1 * t + m2 * t**2
