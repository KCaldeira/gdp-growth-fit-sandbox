"""Data loading and preprocessing for GDP growth curve fitting."""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict


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

    # Mappings
    iso_to_idx: Dict[str, int]   # iso_id -> country index
    idx_to_iso: Dict[int, str]   # country index -> iso_id
    year_to_idx: Dict[int, int]  # year -> year index
    idx_to_year: Dict[int, int]  # year index -> year

    # Dimensions
    n_obs: int
    n_countries: int
    n_years: int

    # Derived values
    pop_weighted_mean_gdp: float     # Population-weighted mean of pcGDP (5-year country means)
    pop_weighted_mean_precp: float   # Population-weighted mean of log-precipitation (5-year country means)
    gdp0_reference_years: tuple      # Years used for GDP0 and P0 calculation (e.g., (2018, 2022))


def load_data(csv_path: str) -> FittingData:
    """Load and preprocess data from CSV file.

    Args:
        csv_path: Path to the input CSV file

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

    return FittingData(
        growth_pcGDP=growth_pcGDP,
        pcGDP=pcGDP,
        temp=temp,
        precp=precp,
        time=time,
        country_idx=country_idx,
        year_idx=year_idx,
        pop=pop,
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
