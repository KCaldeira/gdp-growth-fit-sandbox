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
        "--max-iter",
        type=int,
        default=100,
        help="Maximum iterations for alternating estimation"
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="Convergence tolerance"
    )
    parser.add_argument(
        "--alpha-init",
        type=float,
        default=0.1,
        help="Initial value for alpha parameter (GDP0 is fixed to pop-weighted mean GDP)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.input}...")
    data = load_data(args.input)
    print(f"  Loaded {data.n_obs} observations, "
          f"{data.n_countries} countries, {data.n_years} years")

    # Fit model
    print("\nFitting model...")
    result = fit_model(
        data,
        max_iter=args.max_iter,
        tol=args.tol,
        alpha_init=args.alpha_init,
        verbose=not args.quiet,
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
