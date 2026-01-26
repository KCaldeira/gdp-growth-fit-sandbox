"""Data loading and preprocessing for GDP growth curve fitting."""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class FittingData:
    """Container for data needed by the fitting algorithm."""
    # Observation arrays (length N)
    growth_pcGDP: np.ndarray  # Target variable
    pcGDP: np.ndarray         # Per capita GDP
    temp: np.ndarray          # Temperature
    precp: np.ndarray         # Log precipitation (P)
    time: np.ndarray          # Time index
    country_idx: np.ndarray   # Country index for each observation
    year_idx: np.ndarray      # Year index for each observation
    pop: np.ndarray           # Population

    # Lagged values (for "level" model variant)
    # These are None when compute_lags=False, populated when compute_lags=True
    pcGDP_lag: Optional[np.ndarray] = None   # Per capita GDP lagged by 1 year
    temp_lag: Optional[np.ndarray] = None    # Temperature lagged by 1 year
    precp_lag: Optional[np.ndarray] = None   # Log precipitation lagged by 1 year

    # Mappings
    iso_to_idx: Dict[str, int] = field(default_factory=dict)   # iso_id -> country index
    idx_to_iso: Dict[int, str] = field(default_factory=dict)   # country index -> iso_id
    year_to_idx: Dict[int, int] = field(default_factory=dict)  # year -> year index
    idx_to_year: Dict[int, int] = field(default_factory=dict)  # year index -> year

    # Dimensions
    n_obs: int = 0
    n_countries: int = 0
    n_years: int = 0

    # Derived values
    pop_weighted_mean_gdp: float = 0.0     # Population-weighted mean of pcGDP (5-year country means)
    pop_weighted_mean_precp: float = 0.0   # Population-weighted mean of log-precipitation (5-year country means)
    gdp0_reference_years: tuple = (2018, 2022)  # Years used for GDP0 and P0 calculation


def load_data(csv_path: str, compute_lags: bool) -> FittingData:
    """Load and preprocess data from CSV file.

    Args:
        csv_path: Path to the input CSV file
        compute_lags: If True, compute lagged values (pcGDP_lag, temp_lag, precp_lag)
                      for the "level" model variant and drop first observation per country.

    Returns:
        FittingData object containing all arrays and mappings
    """
    df = pd.read_csv(csv_path)

    # Create country index mapping (sorted for reproducibility)
    unique_countries = sorted(df['iso_id'].unique())
    iso_to_idx = {iso: i for i, iso in enumerate(unique_countries)}
    idx_to_iso = {i: iso for iso, i in iso_to_idx.items()}

    # Create year index mapping (sorted)
    unique_years = sorted(df['year'].unique())
    year_to_idx = {year: i for i, year in enumerate(unique_years)}
    idx_to_year = {i: year for year, i in year_to_idx.items()}

    # Extract arrays
    growth_pcGDP = df['growth_pcGDP'].values.astype(np.float64)
    pcGDP = df['pcGDP'].values.astype(np.float64)
    temp = df['temp'].values.astype(np.float64)
    precp = df['precp'].values.astype(np.float64)
    time = df['time'].values.astype(np.float64)
    pop = df['Pop'].values.astype(np.float64)

    # Map to indices
    country_idx = df['iso_id'].map(iso_to_idx).values.astype(np.int32)
    year_idx = df['year'].map(year_to_idx).values.astype(np.int32)

    # Compute population-weighted means using 5-year (2018-2022) country averages
    # Algorithm:
    # 1. For each country, compute 5-year mean population
    # 2. For each country, compute 5-year mean pcGDP and precipitation
    # 3. GDP0 = weighted mean of country mean GDPs, weighted by country mean populations
    reference_years = (2018, 2022)
    ref_mask = (df['year'] >= reference_years[0]) & (df['year'] <= reference_years[1])
    df_ref = df.loc[ref_mask]

    # Compute country-level 5-year means
    country_means = df_ref.groupby('iso_id').agg({
        'Pop': 'mean',
        'pcGDP': 'mean',
        'precp': 'mean'
    })

    # Compute population-weighted means across countries
    country_pop = country_means['Pop'].values
    country_gdp = country_means['pcGDP'].values
    country_precp = country_means['precp'].values
    pop_weighted_mean_gdp = np.sum(country_gdp * country_pop) / np.sum(country_pop)
    pop_weighted_mean_precp = np.sum(country_precp * country_pop) / np.sum(country_pop)

    # Initialize lagged values
    pcGDP_lag = None
    temp_lag = None
    precp_lag = None

    if compute_lags:
        # Sort by country and year to ensure proper lag computation
        df_sorted = df.sort_values(['iso_id', 'year']).reset_index(drop=True)

        # Compute lagged values within each country using groupby.shift(1)
        df_sorted['pcGDP_lag'] = df_sorted.groupby('iso_id')['pcGDP'].shift(1)
        df_sorted['temp_lag'] = df_sorted.groupby('iso_id')['temp'].shift(1)
        df_sorted['precp_lag'] = df_sorted.groupby('iso_id')['precp'].shift(1)

        # Drop first observation per country (where lag is NaN)
        df_sorted = df_sorted.dropna(subset=['pcGDP_lag', 'temp_lag', 'precp_lag'])

        # Re-extract arrays from filtered dataframe
        growth_pcGDP = df_sorted['growth_pcGDP'].values.astype(np.float64)
        pcGDP = df_sorted['pcGDP'].values.astype(np.float64)
        temp = df_sorted['temp'].values.astype(np.float64)
        precp = df_sorted['precp'].values.astype(np.float64)
        time = df_sorted['time'].values.astype(np.float64)
        pop = df_sorted['Pop'].values.astype(np.float64)
        country_idx = df_sorted['iso_id'].map(iso_to_idx).values.astype(np.int32)
        year_idx = df_sorted['year'].map(year_to_idx).values.astype(np.int32)

        # Extract lagged values
        pcGDP_lag = df_sorted['pcGDP_lag'].values.astype(np.float64)
        temp_lag = df_sorted['temp_lag'].values.astype(np.float64)
        precp_lag = df_sorted['precp_lag'].values.astype(np.float64)

    return FittingData(
        growth_pcGDP=growth_pcGDP,
        pcGDP=pcGDP,
        temp=temp,
        precp=precp,
        time=time,
        country_idx=country_idx,
        year_idx=year_idx,
        pop=pop,
        pcGDP_lag=pcGDP_lag,
        temp_lag=temp_lag,
        precp_lag=precp_lag,
        iso_to_idx=iso_to_idx,
        idx_to_iso=idx_to_iso,
        year_to_idx=year_to_idx,
        idx_to_year=idx_to_year,
        n_obs=len(growth_pcGDP),
        n_countries=len(unique_countries),
        n_years=len(unique_years),
        pop_weighted_mean_gdp=pop_weighted_mean_gdp,
        pop_weighted_mean_precp=pop_weighted_mean_precp,
        gdp0_reference_years=reference_years,
    )
