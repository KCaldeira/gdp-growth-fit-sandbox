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
from src.fitting import fit_model, fit_model_constrained
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
from src.output import save_all_outputs


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

    if bootstrap_result.constrained:
        # Constrained model: T* is directly estimated
        T_opt_point = bootstrap_result.T_opt_point
        valid_mask = ~np.isnan(bootstrap_result.T_opt_samples)
        T_opt_samples = bootstrap_result.T_opt_samples[valid_mask]
        T_opt_clean = T_opt_samples  # No need to remove outliers, bounded by optimization
        title_suffix = "(Constrained Model)"
    else:
        # Unconstrained model: T* derived from h1, h2
        T_opt_point = -params.h1 / (2 * params.h2)
        T_opt_samples = compute_optimal_T_distribution(bootstrap_result)

        # Remove any extreme outliers (e.g., if h2 ≈ 0 in some samples)
        T_opt_clean = T_opt_samples[np.isfinite(T_opt_samples)]
        q1, q99 = np.percentile(T_opt_clean, [1, 99])
        T_opt_clean = T_opt_clean[(T_opt_clean >= q1) & (T_opt_clean <= q99)]
        title_suffix = "(Derived from h1, h2)"

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
    ax.set_title(f'Bootstrap Distribution of T* {title_suffix}', fontsize=14)
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


def plot_optimal_precipitation_distribution(
    bootstrap_result: BootstrapResult,
    output_dir: Path,
) -> None:
    """Plot the bootstrap distribution of optimal precipitation P* (constrained only)."""
    if not bootstrap_result.constrained:
        return  # Only for constrained model

    P_opt_point = bootstrap_result.P_opt_point
    valid_mask = ~np.isnan(bootstrap_result.P_opt_samples)
    P_opt_samples = bootstrap_result.P_opt_samples[valid_mask]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Histogram
    ax = axes[0]
    ax.hist(P_opt_samples, bins=50, density=True, alpha=0.7, color='teal',
            edgecolor='black', linewidth=0.5)

    # Mark point estimate and percentiles
    p2_5, p50, p97_5 = np.percentile(P_opt_samples, [2.5, 50, 97.5])
    ax.axvline(x=P_opt_point, color='red', linestyle='-', linewidth=2,
               label=f'Point est: {P_opt_point:.4f}')
    ax.axvline(x=p50, color='blue', linestyle='--', linewidth=2,
               label=f'Median: {p50:.4f}')
    ax.axvline(x=p2_5, color='gray', linestyle=':', linewidth=1.5,
               label=f'95% CI: [{p2_5:.4f}, {p97_5:.4f}]')
    ax.axvline(x=p97_5, color='gray', linestyle=':', linewidth=1.5)

    ax.set_xlabel('Optimal Log-Precipitation P*', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Bootstrap Distribution of P* (Constrained Model)', fontsize=14)
    ax.legend(loc='best')

    # Right: Box plot
    ax = axes[1]
    bp = ax.boxplot(P_opt_samples, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')

    # Overlay point estimate
    ax.scatter([1], [P_opt_point], color='red', s=100, zorder=5,
               label=f'Point est: {P_opt_point:.4f}', marker='D')

    ax.set_ylabel('Optimal Log-Precipitation P*', fontsize=12)
    ax.set_title('Bootstrap Distribution Summary', fontsize=14)
    ax.set_xticks([])
    ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig(output_dir / "bootstrap_optimal_precipitation.png", dpi=150)
    plt.close()
    print(f"  Saved: bootstrap_optimal_precipitation.png")


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
    """Save bootstrap coefficient samples to two CSV files.

    Creates:
    - bootstrap_coefficients_simple.csv: alpha, h0, h1, h2, h3, h4
    - bootstrap_coefficients_complete.csv: all parameters including j and k
    """
    valid_mask = ~np.isnan(bootstrap_result.alpha_samples)
    alpha_samples = bootstrap_result.alpha_samples[valid_mask]
    h_samples = bootstrap_result.h_samples[valid_mask]

    # Simple CSV: just alpha and h parameters
    if bootstrap_result.constrained:
        # Constrained model: include T_opt and P_opt
        T_opt_samples = bootstrap_result.T_opt_samples[valid_mask]
        P_opt_samples = bootstrap_result.P_opt_samples[valid_mask]

        df_simple = pd.DataFrame({
            'iteration': np.arange(1, len(alpha_samples) + 1),
            'alpha': alpha_samples,
            'T_opt': T_opt_samples,
            'P_opt': P_opt_samples,
            'h0': h_samples[:, 0],
            'h1': h_samples[:, 1],
            'h2': h_samples[:, 2],
            'h3': h_samples[:, 3],
            'h4': h_samples[:, 4],
        })
    else:
        # Unconstrained model
        df_simple = pd.DataFrame({
            'iteration': np.arange(1, len(alpha_samples) + 1),
            'alpha': alpha_samples,
            'h0': h_samples[:, 0],
            'h1': h_samples[:, 1],
            'h2': h_samples[:, 2],
            'h3': h_samples[:, 3],
            'h4': h_samples[:, 4],
        })

    csv_simple_path = output_dir / "bootstrap_coefficients_simple.csv"
    df_simple.to_csv(csv_simple_path, index=False)
    print(f"  Saved: bootstrap_coefficients_simple.csv ({len(df_simple)} samples)")

    # Complete CSV: include j and k parameters
    if not bootstrap_result.constrained:
        j0_samples = bootstrap_result.j0_samples[valid_mask]
        j1_samples = bootstrap_result.j1_samples[valid_mask]
        j2_samples = bootstrap_result.j2_samples[valid_mask]
        k_samples = bootstrap_result.k_samples[valid_mask]

        # Build complete dataframe with j and k columns
        data_dict = {
            'iteration': np.arange(1, len(alpha_samples) + 1),
            'alpha': alpha_samples,
            'h0': h_samples[:, 0],
            'h1': h_samples[:, 1],
            'h2': h_samples[:, 2],
            'h3': h_samples[:, 3],
            'h4': h_samples[:, 4],
        }

        # Add j0 columns (one per country)
        for i in range(bootstrap_result.n_countries):
            data_dict[f'j0_{i}'] = j0_samples[:, i]

        # Add j1 columns (one per country)
        for i in range(bootstrap_result.n_countries):
            data_dict[f'j1_{i}'] = j1_samples[:, i]

        # Add j2 columns (one per country)
        for i in range(bootstrap_result.n_countries):
            data_dict[f'j2_{i}'] = j2_samples[:, i]

        # Add k columns (one per year)
        for i in range(bootstrap_result.n_years):
            data_dict[f'k_{i}'] = k_samples[:, i]

        df_complete = pd.DataFrame(data_dict)
        csv_complete_path = output_dir / "bootstrap_coefficients_complete.csv"
        df_complete.to_csv(csv_complete_path, index=False)
        print(f"  Saved: bootstrap_coefficients_complete.csv ({len(df_complete)} samples, "
              f"{bootstrap_result.n_countries} countries, {bootstrap_result.n_years} years)")
    else:
        print("  Note: Complete CSV not saved for constrained model (j/k not tracked)")


def save_bootstrap_summary(
    bootstrap_result: BootstrapResult,
    output_dir: Path,
) -> None:
    """Save bootstrap summary statistics to file."""
    valid_mask = ~np.isnan(bootstrap_result.alpha_samples)
    alpha_samples = bootstrap_result.alpha_samples[valid_mask]
    h_samples = bootstrap_result.h_samples[valid_mask]

    params = bootstrap_result.params

    # Define percentile ranges to report
    # 1 SD of normal = 68.27%, so [15.87%, 84.13%]
    # 2 SD of normal = 95.45%, so [2.28%, 97.72%]
    percentile_ranges = [
        ("2 SD (95.45%)", 2.275, 97.725),
        ("95%", 2.5, 97.5),
        ("90%", 5.0, 95.0),
        ("1 SD (68.27%)", 15.865, 84.135),
        ("50% (IQR)", 25.0, 75.0),
    ]

    def format_ci_table(samples, point_est, fmt=":.6f"):
        """Format a table of confidence intervals for a parameter."""
        lines = []
        lines.append(f"  Point estimate: {point_est:{fmt[1:]}}")
        median_val = np.percentile(samples, 50)
        # SE of median = sqrt(pi/2) * SD / sqrt(n) for normal distribution
        se_median = np.sqrt(np.pi / 2) * np.std(samples) / np.sqrt(len(samples))
        lines.append(f"  Bootstrap median: {median_val:{fmt[1:]}} (SE: {se_median:{fmt[1:]}})")
        for name, lo, hi in percentile_ranges:
            p_lo, p_hi = np.percentile(samples, [lo, hi])
            lines.append(f"  {name:20s} CI: [{p_lo:{fmt[1:]}}, {p_hi:{fmt[1:]}}]")
        return "\n".join(lines)

    with open(output_dir / "bootstrap_summary.txt", "w") as f:
        f.write("Cluster Bootstrap Uncertainty Analysis\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Bootstrap iterations: {bootstrap_result.n_bootstrap}\n")
        f.write(f"Successful fits: {bootstrap_result.n_successful}\n")
        if bootstrap_result.constrained:
            f.write(f"Model: Constrained (h(T*, P*) = 0)\n")
        else:
            f.write(f"Model variant: {bootstrap_result.model_variant}\n")
        f.write("\n")

        f.write("Percentile ranges reported:\n")
        f.write("  2 SD       = [2.28%, 97.72%] (95.45% of normal distribution)\n")
        f.write("  95%        = [2.5%, 97.5%]\n")
        f.write("  90%        = [5%, 95%]\n")
        f.write("  1 SD       = [15.87%, 84.13%] (68.27% of normal distribution)\n")
        f.write("  50% (IQR)  = [25%, 75%]\n")
        f.write("\n")
        f.write("=" * 60 + "\n\n")

        # Alpha
        f.write("ALPHA (convergence parameter)\n")
        f.write("-" * 40 + "\n")
        f.write(format_ci_table(alpha_samples, params.alpha, ":.6f"))
        f.write("\n\n")

        if bootstrap_result.constrained:
            # Constrained model: show T*, P*, h2, h4 first
            T_opt_samples = bootstrap_result.T_opt_samples[valid_mask]
            P_opt_samples = bootstrap_result.P_opt_samples[valid_mask]

            f.write("T* (optimal temperature, °C)\n")
            f.write("-" * 40 + "\n")
            f.write(format_ci_table(T_opt_samples, bootstrap_result.T_opt_point, ":.2f"))
            f.write("\n\n")

            f.write("P* (optimal log-precipitation)\n")
            f.write("-" * 40 + "\n")
            f.write(format_ci_table(P_opt_samples, bootstrap_result.P_opt_point, ":.4f"))
            f.write("\n\n")

            f.write("h2 (temperature quadratic)\n")
            f.write("-" * 40 + "\n")
            f.write(format_ci_table(h_samples[:, 2], params.h2, ":.6e"))
            f.write("\n\n")

            f.write("h4 (precipitation quadratic)\n")
            f.write("-" * 40 + "\n")
            f.write(format_ci_table(h_samples[:, 4], params.h4, ":.6e"))
            f.write("\n\n")

        else:
            # Unconstrained model: show h parameters
            h_names = ['h0', 'h1', 'h2', 'h3', 'h4']
            h_descriptions = [
                'h0 (intercept)',
                'h1 (temperature linear)',
                'h2 (temperature quadratic)',
                'h3 (precipitation linear)',
                'h4 (precipitation quadratic)',
            ]
            h_point = [params.h0, params.h1, params.h2, params.h3, params.h4]

            for i, (name, desc) in enumerate(zip(h_names, h_descriptions)):
                f.write(f"{desc}\n")
                f.write("-" * 40 + "\n")
                f.write(format_ci_table(h_samples[:, i], h_point[i], ":.6e"))
                f.write("\n\n")

            # Derived optimal temperature (for unconstrained only)
            T_opt_point = -params.h1 / (2 * params.h2)
            T_opt_samples = compute_optimal_T_distribution(bootstrap_result)
            T_opt_clean = T_opt_samples[np.isfinite(T_opt_samples)]
            q1, q99 = np.percentile(T_opt_clean, [1, 99])
            T_opt_clean = T_opt_clean[(T_opt_clean >= q1) & (T_opt_clean <= q99)]

            f.write("T* = -h1/(2*h2) (derived optimal temperature, °C)\n")
            f.write("-" * 40 + "\n")
            f.write(format_ci_table(T_opt_clean, T_opt_point, ":.2f"))
            f.write("\n\n")

            # Derived optimal precipitation (for unconstrained only)
            P_opt_point = -params.h3 / (2 * params.h4)
            h3_samples = h_samples[:, 3]
            h4_samples = h_samples[:, 4]
            P_opt_samples = -h3_samples / (2 * h4_samples)
            P_opt_clean = P_opt_samples[np.isfinite(P_opt_samples)]
            q1, q99 = np.percentile(P_opt_clean, [1, 99])
            P_opt_clean = P_opt_clean[(P_opt_clean >= q1) & (P_opt_clean <= q99)]

            f.write("P* = -h3/(2*h4) (derived optimal log-precipitation)\n")
            f.write("-" * 40 + "\n")
            f.write(format_ci_table(P_opt_clean, P_opt_point, ":.4f"))
            f.write("\n")

    print(f"  Saved: bootstrap_summary.txt")


def compute_stats_from_csv(csv_path: Path, output_path: Path) -> None:
    """Compute bootstrap statistics from an existing CSV file.

    Args:
        csv_path: Path to bootstrap_coefficients_simple.csv
        output_path: Path to write the summary output
    """
    # Define percentile ranges
    percentile_ranges = [
        ("2 SD (95.45%)", 2.275, 97.725),
        ("95%", 2.5, 97.5),
        ("90%", 5.0, 95.0),
        ("1 SD (68.27%)", 15.865, 84.135),
        ("50% (IQR)", 25.0, 75.0),
    ]

    def format_ci_table(samples, fmt=":.6f"):
        """Format a table of confidence intervals for a parameter."""
        lines = []
        median_val = np.percentile(samples, 50)
        # SE of median = sqrt(pi/2) * SD / sqrt(n) for normal distribution
        se_median = np.sqrt(np.pi / 2) * np.std(samples) / np.sqrt(len(samples))
        lines.append(f"  Bootstrap median: {median_val:{fmt[1:]}} (SE: {se_median:{fmt[1:]}})")
        for name, lo, hi in percentile_ranges:
            p_lo, p_hi = np.percentile(samples, [lo, hi])
            lines.append(f"  {name:20s} CI: [{p_lo:{fmt[1:]}}, {p_hi:{fmt[1:]}}]")
        return "\n".join(lines)

    # Read CSV
    df = pd.read_csv(csv_path)
    n_samples = len(df)

    print(f"Read {n_samples} bootstrap samples from {csv_path}")

    with open(output_path, "w") as f:
        f.write("Bootstrap Statistics from CSV\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Source file: {csv_path}\n")
        f.write(f"Number of samples: {n_samples}\n\n")

        f.write("Percentile ranges reported:\n")
        f.write("  2 SD       = [2.28%, 97.72%] (95.45% of normal distribution)\n")
        f.write("  95%        = [2.5%, 97.5%]\n")
        f.write("  90%        = [5%, 95%]\n")
        f.write("  1 SD       = [15.87%, 84.13%] (68.27% of normal distribution)\n")
        f.write("  50% (IQR)  = [25%, 75%]\n")
        f.write("\n")
        f.write("=" * 60 + "\n\n")

        # Check which columns are present
        if 'alpha' in df.columns:
            f.write("ALPHA (convergence parameter)\n")
            f.write("-" * 40 + "\n")
            f.write(format_ci_table(df['alpha'].values, ":.6f"))
            f.write("\n\n")

        # h parameters
        h_cols = ['h0', 'h1', 'h2', 'h3', 'h4']
        h_descriptions = [
            'h0 (intercept)',
            'h1 (temperature linear)',
            'h2 (temperature quadratic)',
            'h3 (precipitation linear)',
            'h4 (precipitation quadratic)',
        ]
        for col, desc in zip(h_cols, h_descriptions):
            if col in df.columns:
                f.write(f"{desc}\n")
                f.write("-" * 40 + "\n")
                f.write(format_ci_table(df[col].values, ":.6e"))
                f.write("\n\n")

        # Derived T* and P* if h1, h2, h3, h4 are present
        if all(col in df.columns for col in ['h1', 'h2']):
            T_opt_samples = -df['h1'].values / (2 * df['h2'].values)
            T_opt_clean = T_opt_samples[np.isfinite(T_opt_samples)]
            if len(T_opt_clean) > 0:
                q1, q99 = np.percentile(T_opt_clean, [1, 99])
                T_opt_clean = T_opt_clean[(T_opt_clean >= q1) & (T_opt_clean <= q99)]
                f.write("T* = -h1/(2*h2) (derived optimal temperature, °C)\n")
                f.write("-" * 40 + "\n")
                f.write(format_ci_table(T_opt_clean, ":.2f"))
                f.write("\n\n")

        if all(col in df.columns for col in ['h3', 'h4']):
            P_opt_samples = -df['h3'].values / (2 * df['h4'].values)
            P_opt_clean = P_opt_samples[np.isfinite(P_opt_samples)]
            if len(P_opt_clean) > 0:
                q1, q99 = np.percentile(P_opt_clean, [1, 99])
                P_opt_clean = P_opt_clean[(P_opt_clean >= q1) & (P_opt_clean <= q99)]
                f.write("P* = -h3/(2*h4) (derived optimal log-precipitation)\n")
                f.write("-" * 40 + "\n")
                f.write(format_ci_table(P_opt_clean, ":.4f"))
                f.write("\n")

        # T_opt and P_opt if directly present (constrained model)
        if 'T_opt' in df.columns:
            f.write("T* (optimal temperature, °C) - directly estimated\n")
            f.write("-" * 40 + "\n")
            f.write(format_ci_table(df['T_opt'].values, ":.2f"))
            f.write("\n\n")

        if 'P_opt' in df.columns:
            f.write("P* (optimal log-precipitation) - directly estimated\n")
            f.write("-" * 40 + "\n")
            f.write(format_ci_table(df['P_opt'].values, ":.4f"))
            f.write("\n")

    print(f"Saved statistics to: {output_path}")


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
    parser.add_argument("--constrained", action="store_true",
                        help="Use constrained model with h(T*, P*) = 0. "
                             "This normalizes the climate response to pass through "
                             "zero at optimal temperature and precipitation. "
                             "Only available for base model variant.")
    parser.add_argument("--from-csv", type=str, default=None,
                        help="Compute statistics from existing bootstrap CSV file "
                             "(e.g., bootstrap_coefficients_simple.csv). "
                             "When specified, skips bootstrap and just computes stats.")
    args = parser.parse_args()

    # Handle --from-csv mode
    if args.from_csv:
        csv_path = Path(args.from_csv).expanduser()
        if not csv_path.exists():
            print(f"Error: File not found: {csv_path}")
            return

        # Determine output path
        if args.output_dir:
            output_path = Path(args.output_dir) / "bootstrap_stats_from_csv.txt"
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        else:
            output_path = csv_path.parent / "bootstrap_stats_from_csv.txt"

        compute_stats_from_csv(csv_path, output_path)
        return

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

    # Validate constrained flag
    if args.constrained and args.model_variant != "base":
        print("Error: --constrained is only available for base model variant")
        return

    # Load data
    print(f"\nLoading data from {args.data_file}...")
    data = load_data(args.data_file)
    print(f"  N={data.n_obs}, countries={data.n_countries}, years={data.n_years}")

    # Fit model
    print("\nFitting model...")
    if args.constrained:
        fit_result = fit_model_constrained(data, verbose=True)
        T_opt_point = fit_result.T_opt
        P_opt_point = fit_result.P_opt
    else:
        fit_result = fit_model(data, model_variant=args.model_variant, verbose=True)
        T_opt_point = None
        P_opt_point = None

    # Run bootstrap
    print(f"\nRunning cluster bootstrap (n={args.n_bootstrap}, seed={args.seed})...")
    bootstrap_result = run_bootstrap(
        data,
        fit_result.params,
        n_bootstrap=args.n_bootstrap,
        random_seed=args.seed,
        model_variant=args.model_variant,
        verbose=True,
        constrained=args.constrained,
        T_opt_point=T_opt_point,
        P_opt_point=P_opt_point,
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

    # Save point estimate outputs (same format as run_fit.py)
    print(f"\nSaving point estimate outputs...")
    save_all_outputs(fit_result, data, output_dir)

    # Generate bootstrap-specific plots and outputs
    print(f"\nGenerating bootstrap plots and outputs...")

    plot_h_temperature_response(bootstrap_result, T_range, P_const, output_dir)
    plot_g_gdp_response(bootstrap_result, GDP_range, output_dir)
    plot_gh_combined(bootstrap_result, T_range, P_const, GDP_values, output_dir)
    plot_optimal_temperature_distribution(bootstrap_result, output_dir)
    plot_optimal_precipitation_distribution(bootstrap_result, output_dir)  # Only plots for constrained
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
