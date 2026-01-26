#!/usr/bin/env python3
"""Run a single constrained model fit (no bootstrap).

Usage:
    python run_constrained_fit.py [--model-variant growth|level]
"""

import argparse

from src.data_loader import load_data
from src.fitting import fit_model_constrained


def main():
    parser = argparse.ArgumentParser(description="Run constrained model fit")
    parser.add_argument(
        "--model-variant", "-m",
        default="growth",
        choices=["growth", "level"],
        help="Model variant: growth (g[t]*h[t] + j + k), "
             "level (g[t]*h[t] - g[t-1]*h[t-1] + j + k) (default: growth)"
    )
    args = parser.parse_args()

    # Load data (compute lags for "level" model variant)
    compute_lags = (args.model_variant == "level")
    print(f"Loading data...")
    data = load_data('data/input/df_base_withPop.csv', compute_lags=compute_lags)
    print(f"  N={data.n_obs}, countries={data.n_countries}, years={data.n_years}")
    if compute_lags:
        print(f"  (Lagged data computed for 'level' model variant)")

    result = fit_model_constrained(
        data, model_variant=args.model_variant, verbose=True
    )

    print(f'\n=== Results ===')
    print(f'Model variant: {args.model_variant}')
    print(f'T* = {result.T_opt:.2f}°C')
    print(f'P* = {result.P_opt:.4f}')
    print(f'alpha = {result.params.alpha:.6f}')
    print(f'h2 = {result.params.h2:.6e}')
    print(f'h4 = {result.params.h4:.6e}')
    print(f'R² = {result.r_squared:.4f}')


if __name__ == "__main__":
    main()
