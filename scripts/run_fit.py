#!/usr/bin/env python3
"""Main script to run the GDP growth curve fitting."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_data
from src.fitting import fit_model
from src.output import save_all_outputs, create_output_directory


def main():
    parser = argparse.ArgumentParser(
        description="Fit non-linear GDP growth model"
    )
    parser.add_argument(
        "--input", "-i",
        default="data/input/df_base_withPop.csv",
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory (default: timestamped subdirectory of data/output)"
    )
    parser.add_argument(
        "--model-variant", "-m",
        default="growth",
        choices=["growth", "level"],
        help="Model variant: growth (g[t]*h[t] + j + k), "
             "level (g[t]*h[t] - g[t-1]*h[t-1] + j + k) (default: growth)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )
    parser.add_argument(
        "--compute-se",
        action="store_true",
        help="Compute standard errors via numerical Hessian (slow, off by default)"
    )

    args = parser.parse_args()

    # Load data (compute lags for "level" model variant)
    compute_lags = (args.model_variant == "level")
    print(f"Loading data from {args.input}...")
    data = load_data(args.input, compute_lags=compute_lags)
    print(f"  Loaded {data.n_obs} observations, "
          f"{data.n_countries} countries, {data.n_years} years")
    if compute_lags:
        print(f"  (Lagged data computed for 'level' model variant)")

    # Fit model
    print("\nFitting model...")
    result = fit_model(
        data,
        model_variant=args.model_variant,
        verbose=not args.quiet,
        compute_se=args.compute_se,
    )

    # Save outputs
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = create_output_directory()

    print(f"\nSaving results...")
    save_all_outputs(result, data, output_dir)

    print(f"\nDone! Results saved to {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
