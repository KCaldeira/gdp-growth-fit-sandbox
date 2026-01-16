#!/usr/bin/env python3
"""Run cluster bootstrap uncertainty analysis and generate plots.

This script:
1. Loads the data and fits the model
2. Runs cluster bootstrap (resampling countries with replacement)
3. Generates uncertainty plots for h(T), g(GDP), and g(GDP)*h(T)

Usage:
    python run_bootstrap.py [--n-bootstrap N] [--seed S] [--output-dir DIR]
"""

import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import load_data
from src.fitting import fit_model
from src.bootstrap import (
    run_bootstrap,
    compute_h_uncertainty_bands,
    compute_g_uncertainty_bands,
    compute_gh_at_fixed_gdp,
    compute_optimal_T_distribution,
    compute_dhdT_uncertainty_bands,
    BootstrapResult,
)
from src.model import h_func, g_func


def plot_h_temperature_response(
    bootstrap_result: BootstrapResult,
    T_range: np.ndarray,
    P_const: float,
    output_dir: Path,
) -> None:
    """Plot h(T, P_const) with bootstrap uncertainty bands."""
    params = bootstrap_result.params

    # Point estimate
    h_point = h_func(T_range, np.full_like(T_range, P_const),
                     params.h0, params.h1, params.h2, params.h3, params.h4)

    # Bootstrap bands
    h_lower, h_median, h_upper = compute_h_uncertainty_bands(
        bootstrap_result, T_range, P_const, percentiles=(2.5, 50, 97.5)
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    # Shade bootstrap CI
    ax.fill_between(T_range, h_lower, h_upper, alpha=0.3, color='blue',
                    label='95% Bootstrap CI')

    # Bootstrap median
    ax.plot(T_range, h_median, 'b--', linewidth=1.5, label='Bootstrap median')

    # Point estimate
    ax.plot(T_range, h_point, 'b-', linewidth=2, label='Point estimate')

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)

    # Mark optimal temperature from point estimate
    T_opt = -params.h1 / (2 * params.h2)
    if T_range.min() <= T_opt <= T_range.max():
        ax.axvline(x=T_opt, color='red', linestyle=':', linewidth=2,
                   label=f'T* = {T_opt:.1f}°C (point est.)')

    ax.set_xlabel('Temperature (°C)', fontsize=12)
    ax.set_ylabel('h(T, P)', fontsize=12)
    ax.set_title(f'Climate Response h(T) at P = {P_const:.2f}\n'
                 f'(Cluster Bootstrap, n={bootstrap_result.n_successful})', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "bootstrap_h_temperature.png", dpi=150)
    plt.close()
    print(f"  Saved: bootstrap_h_temperature.png")


def plot_g_gdp_response(
    bootstrap_result: BootstrapResult,
    GDP_range: np.ndarray,
    output_dir: Path,
) -> None:
    """Plot g(GDP) with bootstrap uncertainty bands."""
    params = bootstrap_result.params
    GDP0 = params.GDP0

    # Point estimate
    g_point = g_func(GDP_range, GDP0, params.alpha)

    # Bootstrap bands
    g_lower, g_median, g_upper = compute_g_uncertainty_bands(
        bootstrap_result, GDP_range, percentiles=(2.5, 50, 97.5)
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    # Shade bootstrap CI
    ax.fill_between(GDP_range, g_lower, g_upper, alpha=0.3, color='green',
                    label='95% Bootstrap CI')

    # Bootstrap median
    ax.plot(GDP_range, g_median, 'g--', linewidth=1.5, label='Bootstrap median')

    # Point estimate
    ax.plot(GDP_range, g_point, 'g-', linewidth=2, label='Point estimate')

    # Reference line at GDP0
    ax.axvline(x=GDP0, color='red', linestyle=':', linewidth=2,
               label=f'GDP0 = {GDP0:.0f}')
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1,
               label='g = 1 (at GDP0)')

    ax.set_xscale('log')
    ax.set_xlabel('Per Capita GDP', fontsize=12)
    ax.set_ylabel('g(GDP) = (GDP/GDP0)^(-α)', fontsize=12)
    ax.set_title(f'GDP Convergence Factor g(GDP)\n'
                 f'(Cluster Bootstrap, n={bootstrap_result.n_successful})', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "bootstrap_g_gdp.png", dpi=150)
    plt.close()
    print(f"  Saved: bootstrap_g_gdp.png")


def plot_gh_combined(
    bootstrap_result: BootstrapResult,
    T_range: np.ndarray,
    P_const: float,
    GDP_values: list,
    output_dir: Path,
) -> None:
    """Plot g(GDP)*h(T) for multiple GDP levels with bootstrap uncertainty."""
    params = bootstrap_result.params
    GDP0 = params.GDP0

    # Colors for different GDP levels
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(GDP_values)))

    fig, ax = plt.subplots(figsize=(12, 7))

    for GDP_val, color in zip(GDP_values, colors):
        # Point estimate
        g_point = g_func(np.array([GDP_val]), GDP0, params.alpha)[0]
        h_point = h_func(T_range, np.full_like(T_range, P_const),
                         params.h0, params.h1, params.h2, params.h3, params.h4)
        gh_point = g_point * h_point

        # Bootstrap bands
        gh_lower, gh_median, gh_upper = compute_gh_at_fixed_gdp(
            bootstrap_result, GDP_val, T_range, P_const, percentiles=(2.5, 50, 97.5)
        )

        # Label for legend
        label = f'GDP = ${GDP_val:,.0f}'

        # Shade CI
        ax.fill_between(T_range, gh_lower, gh_upper, alpha=0.2, color=color)

        # Point estimate line
        ax.plot(T_range, gh_point, '-', linewidth=2, color=color, label=label)

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)

    ax.set_xlabel('Temperature (°C)', fontsize=12)
    ax.set_ylabel('g(GDP) × h(T, P)', fontsize=12)
    ax.set_title(f'Climate Effect on Growth by GDP Level\n'
                 f'(P = {P_const:.2f}, Cluster Bootstrap, n={bootstrap_result.n_successful})',
                 fontsize=14)
    ax.legend(loc='best', title='Per Capita GDP')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "bootstrap_gh_combined.png", dpi=150)
    plt.close()
    print(f"  Saved: bootstrap_gh_combined.png")


def plot_optimal_temperature_distribution(
    bootstrap_result: BootstrapResult,
    output_dir: Path,
) -> None:
    """Plot the bootstrap distribution of optimal temperature T*."""
    params = bootstrap_result.params

    # Point estimate
    T_opt_point = -params.h1 / (2 * params.h2)

    # Bootstrap distribution
    T_opt_samples = compute_optimal_T_distribution(bootstrap_result)

    # Remove any extreme outliers (e.g., if h2 ≈ 0 in some samples)
    T_opt_clean = T_opt_samples[np.isfinite(T_opt_samples)]
    q1, q99 = np.percentile(T_opt_clean, [1, 99])
    T_opt_clean = T_opt_clean[(T_opt_clean >= q1) & (T_opt_clean <= q99)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Histogram
    ax = axes[0]
    ax.hist(T_opt_clean, bins=50, density=True, alpha=0.7, color='purple',
            edgecolor='black', linewidth=0.5)

    # Mark point estimate and percentiles
    p2_5, p50, p97_5 = np.percentile(T_opt_clean, [2.5, 50, 97.5])
    ax.axvline(x=T_opt_point, color='red', linestyle='-', linewidth=2,
               label=f'Point est: {T_opt_point:.2f}°C')
    ax.axvline(x=p50, color='blue', linestyle='--', linewidth=2,
               label=f'Median: {p50:.2f}°C')
    ax.axvline(x=p2_5, color='gray', linestyle=':', linewidth=1.5,
               label=f'95% CI: [{p2_5:.2f}, {p97_5:.2f}]°C')
    ax.axvline(x=p97_5, color='gray', linestyle=':', linewidth=1.5)

    ax.set_xlabel('Optimal Temperature T* (°C)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Bootstrap Distribution of Optimal Temperature', fontsize=14)
    ax.legend(loc='best')

    # Right: Box plot + swarm
    ax = axes[1]
    bp = ax.boxplot(T_opt_clean, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')

    # Overlay point estimate
    ax.scatter([1], [T_opt_point], color='red', s=100, zorder=5,
               label=f'Point est: {T_opt_point:.2f}°C', marker='D')

    ax.set_ylabel('Optimal Temperature T* (°C)', fontsize=12)
    ax.set_title('Bootstrap Distribution Summary', fontsize=14)
    ax.set_xticks([])
    ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig(output_dir / "bootstrap_optimal_temperature.png", dpi=150)
    plt.close()
    print(f"  Saved: bootstrap_optimal_temperature.png")


def plot_dhdT_marginal_effect(
    bootstrap_result: BootstrapResult,
    T_range: np.ndarray,
    output_dir: Path,
) -> None:
    """Plot dh/dT with bootstrap uncertainty bands."""
    params = bootstrap_result.params

    # Point estimate: dh/dT = h1 + 2*h2*T
    dhdT_point = params.h1 + 2 * params.h2 * T_range

    # Bootstrap bands
    dhdT_lower, dhdT_median, dhdT_upper = compute_dhdT_uncertainty_bands(
        bootstrap_result, T_range, percentiles=(2.5, 50, 97.5)
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    # Shade bootstrap CI
    ax.fill_between(T_range, dhdT_lower, dhdT_upper, alpha=0.3, color='orange',
                    label='95% Bootstrap CI')

    # Bootstrap median
    ax.plot(T_range, dhdT_median, color='orange', linestyle='--', linewidth=1.5,
            label='Bootstrap median')

    # Point estimate
    ax.plot(T_range, dhdT_point, color='darkorange', linestyle='-', linewidth=2,
            label='Point estimate')

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)

    # Mark optimal temperature
    T_opt = -params.h1 / (2 * params.h2)
    if T_range.min() <= T_opt <= T_range.max():
        ax.axvline(x=T_opt, color='red', linestyle=':', linewidth=2,
                   label=f'T* = {T_opt:.1f}°C')

    ax.set_xlabel('Temperature (°C)', fontsize=12)
    ax.set_ylabel('dh/dT (Marginal Effect)', fontsize=12)
    ax.set_title(f'Marginal Effect of Temperature\n'
                 f'(Cluster Bootstrap, n={bootstrap_result.n_successful})', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "bootstrap_dhdT.png", dpi=150)
    plt.close()
    print(f"  Saved: bootstrap_dhdT.png")


def plot_alpha_distribution(
    bootstrap_result: BootstrapResult,
    output_dir: Path,
) -> None:
    """Plot the bootstrap distribution of alpha."""
    params = bootstrap_result.params

    # Get valid samples
    valid_mask = ~np.isnan(bootstrap_result.alpha_samples)
    alpha_samples = bootstrap_result.alpha_samples[valid_mask]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(alpha_samples, bins=50, density=True, alpha=0.7, color='teal',
            edgecolor='black', linewidth=0.5)

    # Mark point estimate and percentiles
    p2_5, p50, p97_5 = np.percentile(alpha_samples, [2.5, 50, 97.5])
    ax.axvline(x=params.alpha, color='red', linestyle='-', linewidth=2,
               label=f'Point est: {params.alpha:.4f}')
    ax.axvline(x=p50, color='blue', linestyle='--', linewidth=2,
               label=f'Median: {p50:.4f}')
    ax.axvline(x=p2_5, color='gray', linestyle=':', linewidth=1.5,
               label=f'95% CI: [{p2_5:.4f}, {p97_5:.4f}]')
    ax.axvline(x=p97_5, color='gray', linestyle=':', linewidth=1.5)

    ax.set_xlabel('α (Convergence Parameter)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Bootstrap Distribution of α\n'
                 f'(Cluster Bootstrap, n={bootstrap_result.n_successful})', fontsize=14)
    ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig(output_dir / "bootstrap_alpha.png", dpi=150)
    plt.close()
    print(f"  Saved: bootstrap_alpha.png")


def save_bootstrap_coefficients(
    bootstrap_result: BootstrapResult,
    output_dir: Path,
) -> None:
    """Save all bootstrap coefficient samples to CSV."""
    valid_mask = ~np.isnan(bootstrap_result.alpha_samples)
    alpha_samples = bootstrap_result.alpha_samples[valid_mask]
    h_samples = bootstrap_result.h_samples[valid_mask]

    # Create DataFrame with all global (time/country-independent) coefficients
    df = pd.DataFrame({
        'iteration': np.arange(1, len(alpha_samples) + 1),
        'alpha': alpha_samples,
        'h0': h_samples[:, 0],
        'h1': h_samples[:, 1],
        'h2': h_samples[:, 2],
        'h3': h_samples[:, 3],
        'h4': h_samples[:, 4],
    })

    # Also compute derived quantities
    df['T_optimal'] = -df['h1'] / (2 * df['h2'])

    csv_path = output_dir / "bootstrap_coefficients.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: bootstrap_coefficients.csv ({len(df)} samples)")


def save_bootstrap_summary(
    bootstrap_result: BootstrapResult,
    output_dir: Path,
) -> None:
    """Save bootstrap summary statistics to file."""
    valid_mask = ~np.isnan(bootstrap_result.alpha_samples)
    alpha_samples = bootstrap_result.alpha_samples[valid_mask]
    h_samples = bootstrap_result.h_samples[valid_mask]

    params = bootstrap_result.params

    with open(output_dir / "bootstrap_summary.txt", "w") as f:
        f.write("Cluster Bootstrap Uncertainty Analysis\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Bootstrap iterations: {bootstrap_result.n_bootstrap}\n")
        f.write(f"Successful fits: {bootstrap_result.n_successful}\n\n")

        f.write("Parameter Estimates (Point Est, [95% Bootstrap CI])\n")
        f.write("-" * 50 + "\n\n")

        # Alpha
        p2_5, p50, p97_5 = np.percentile(alpha_samples, [2.5, 50, 97.5])
        f.write(f"alpha:  {params.alpha:.6f}  [{p2_5:.6f}, {p97_5:.6f}]\n\n")

        # h parameters
        h_names = ['h0', 'h1', 'h2', 'h3', 'h4']
        h_point = [params.h0, params.h1, params.h2, params.h3, params.h4]
        for i, name in enumerate(h_names):
            p2_5, p50, p97_5 = np.percentile(h_samples[:, i], [2.5, 50, 97.5])
            f.write(f"{name}:  {h_point[i]:.6e}  [{p2_5:.6e}, {p97_5:.6e}]\n")

        f.write("\n")

        # Optimal temperature
        T_opt_point = -params.h1 / (2 * params.h2)
        T_opt_samples = compute_optimal_T_distribution(bootstrap_result)
        T_opt_clean = T_opt_samples[np.isfinite(T_opt_samples)]
        q1, q99 = np.percentile(T_opt_clean, [1, 99])
        T_opt_clean = T_opt_clean[(T_opt_clean >= q1) & (T_opt_clean <= q99)]
        p2_5, p50, p97_5 = np.percentile(T_opt_clean, [2.5, 50, 97.5])

        f.write(f"\nOptimal Temperature T*:\n")
        f.write(f"  Point estimate: {T_opt_point:.2f}°C\n")
        f.write(f"  Bootstrap median: {p50:.2f}°C\n")
        f.write(f"  95% Bootstrap CI: [{p2_5:.2f}, {p97_5:.2f}]°C\n")

    print(f"  Saved: bootstrap_summary.txt")


def main():
    parser = argparse.ArgumentParser(description="Run cluster bootstrap uncertainty analysis")
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                        help="Number of bootstrap iterations (default: 1000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: ./data/output/bootstrap_TIMESTAMP)")
    parser.add_argument("--data-file", type=str, default="data/input/df_base_withPop.csv",
                        help="Input data file")
    parser.add_argument("--model-variant", type=str, default="base",
                        choices=["base", "g_scales_hj", "g_scales_all"],
                        help="Model variant: base (g*h + j + k), "
                             "g_scales_hj (g*(h+j) + k), "
                             "g_scales_all (g*(h+j+k)) (default: base)")
    args = parser.parse_args()

    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("data/output") / f"bootstrap_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Cluster Bootstrap Uncertainty Analysis")
    print("=" * 60)

    # Load data
    print(f"\nLoading data from {args.data_file}...")
    data = load_data(args.data_file)
    print(f"  N={data.n_obs}, countries={data.n_countries}, years={data.n_years}")

    # Fit model
    print("\nFitting model...")
    fit_result = fit_model(data, model_variant=args.model_variant, verbose=True)

    # Run bootstrap
    print(f"\nRunning cluster bootstrap (n={args.n_bootstrap}, seed={args.seed})...")
    bootstrap_result = run_bootstrap(
        data,
        fit_result.params,
        n_bootstrap=args.n_bootstrap,
        random_seed=args.seed,
        model_variant=args.model_variant,
        verbose=True,
    )

    # Temperature range: 0°C to 30°C (standard for climate-growth plots)
    T_range = np.linspace(0, 30, 200)

    GDP_min, GDP_max = data.pcGDP.min(), data.pcGDP.max()
    GDP_range = np.logspace(np.log10(GDP_min), np.log10(GDP_max), 100)

    # Use population-weighted mean log-precipitation (similar to GDP0)
    P_const = data.pop_weighted_mean_precp
    print(f"\nUsing P_const = {P_const:.4f} (population-weighted mean log-precipitation)")

    # Representative GDP levels for comparison plots
    GDP_percentiles = [10, 25, 50, 75, 90]
    GDP_values = [np.percentile(data.pcGDP, p) for p in GDP_percentiles]

    # Generate plots
    print(f"\nGenerating plots (output: {output_dir})...")

    plot_h_temperature_response(bootstrap_result, T_range, P_const, output_dir)
    plot_g_gdp_response(bootstrap_result, GDP_range, output_dir)
    plot_gh_combined(bootstrap_result, T_range, P_const, GDP_values, output_dir)
    plot_optimal_temperature_distribution(bootstrap_result, output_dir)
    plot_dhdT_marginal_effect(bootstrap_result, T_range, output_dir)
    plot_alpha_distribution(bootstrap_result, output_dir)
    save_bootstrap_coefficients(bootstrap_result, output_dir)
    save_bootstrap_summary(bootstrap_result, output_dir)

    print("\n" + "=" * 60)
    print("Bootstrap analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
