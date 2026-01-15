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
    pop_weighted_mean_gdp: float  # Population-weighted mean of pcGDP


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

    # Compute population-weighted mean GDP
    pop_weighted_mean_gdp = np.sum(pcGDP * pop) / np.sum(pop)

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
    )
