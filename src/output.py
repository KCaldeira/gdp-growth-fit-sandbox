"""Output handling for GDP growth curve fitting results."""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .data_loader import FittingData
from .fitting import FitResult
from .model import predict_from_data


def create_output_directory(base_path: str = "data/output") -> Path:
    """Create a timestamped output directory.

    Args:
        base_path: Base directory for outputs

    Returns:
        Path to the created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_path) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_parameters(result: FitResult, data: FittingData,
                    output_dir: Path) -> None:
    """Save fitted parameters to files.

    Creates:
        - global_params.json: GDP0, alpha, h0-h4 with SEs
        - country_params.csv: j0, j1, j2 for each country
        - year_effects.csv: k for each year
    """
    params = result.params

    # Global parameters
    global_params = {
        "GDP0": {"estimate": params.GDP0, "se": params.se_GDP0},
        "alpha": {"estimate": params.alpha, "se": params.se_alpha},
        "h0": {"estimate": params.h0, "se": params.se_h0},
        "h1": {"estimate": params.h1, "se": params.se_h1},
        "h2": {"estimate": params.h2, "se": params.se_h2},
        "h3": {"estimate": params.h3, "se": params.se_h3},
        "h4": {"estimate": params.h4, "se": params.se_h4},
    }

    with open(output_dir / "global_params.json", "w") as f:
        json.dump(global_params, f, indent=2, default=_json_serializer)

    # Country parameters
    country_data = []
    for i in range(data.n_countries):
        iso = data.idx_to_iso[i]
        row = {
            "iso_id": iso,
            "j0": params.j0[i],
            "j1": params.j1[i],
            "j2": params.j2[i],
        }
        if params.se_j0 is not None:
            row["se_j0"] = params.se_j0[i]
            row["se_j1"] = params.se_j1[i]
            row["se_j2"] = params.se_j2[i]
        country_data.append(row)

    pd.DataFrame(country_data).to_csv(output_dir / "country_params.csv", index=False)

    # Year effects
    year_data = []
    for i in range(data.n_years):
        year = data.idx_to_year[i]
        row = {"year": year, "k": params.k[i]}
        if params.se_k is not None:
            row["se_k"] = params.se_k[i]
        year_data.append(row)

    pd.DataFrame(year_data).to_csv(output_dir / "year_effects.csv", index=False)


def save_summary(result: FitResult, data: FittingData, output_dir: Path) -> None:
    """Save model summary statistics."""
    summary = {
        "n_observations": data.n_obs,
        "n_countries": data.n_countries,
        "n_years": data.n_years,
        "n_parameters": 2 + 5 + 3 * data.n_countries + data.n_years,
        "r_squared": result.r_squared,
        "rmse": result.rmse,
        "aic": result.aic,
        "bic": result.bic,
        "n_iterations": result.n_iterations,
        "converged": result.converged,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=_json_serializer)

    # Also save as readable text
    with open(output_dir / "summary.txt", "w") as f:
        f.write("GDP Growth Model Fit Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Observations: {data.n_obs}\n")
        f.write(f"Countries: {data.n_countries}\n")
        f.write(f"Years: {data.n_years}\n")
        f.write(f"Parameters: {summary['n_parameters']}\n\n")
        f.write(f"R²: {result.r_squared:.4f}\n")
        f.write(f"RMSE: {result.rmse:.6f}\n")
        f.write(f"AIC: {result.aic:.1f}\n")
        f.write(f"BIC: {result.bic:.1f}\n\n")
        f.write(f"Iterations: {result.n_iterations}\n")
        f.write(f"Converged: {result.converged}\n\n")

        # Global parameters
        f.write("Global Parameters:\n")
        f.write("-" * 40 + "\n")
        params = result.params
        f.write(f"  GDP0:  {params.GDP0:12.4f}  (fixed)\n")
        f.write(f"  alpha: {params.alpha:12.6f}")
        if params.se_alpha:
            f.write(f"  (SE: {params.se_alpha:.6f})")
        f.write("\n\n")

        f.write("Climate Response (h):\n")
        f.write(f"  h0: {params.h0:12.4e}")
        if params.se_h0:
            f.write(f"  (SE: {params.se_h0:.4e})")
        f.write("\n")
        f.write(f"  h1: {params.h1:12.4e}")
        if params.se_h1:
            f.write(f"  (SE: {params.se_h1:.4e})")
        f.write("\n")
        f.write(f"  h2: {params.h2:12.4e}")
        if params.se_h2:
            f.write(f"  (SE: {params.se_h2:.4e})")
        f.write("\n")
        f.write(f"  h3: {params.h3:12.4e}")
        if params.se_h3:
            f.write(f"  (SE: {params.se_h3:.4e})")
        f.write("\n")
        f.write(f"  h4: {params.h4:12.4e}")
        if params.se_h4:
            f.write(f"  (SE: {params.se_h4:.4e})")
        f.write("\n")


def save_diagnostic_plots(result: FitResult, data: FittingData,
                          output_dir: Path) -> None:
    """Generate and save diagnostic plots."""
    predictions = predict_from_data(data, result.params)
    residuals = result.residuals

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Fitted vs Actual
    ax = axes[0, 0]
    ax.scatter(data.growth_pcGDP, predictions, alpha=0.3, s=5)
    lims = [min(data.growth_pcGDP.min(), predictions.min()),
            max(data.growth_pcGDP.max(), predictions.max())]
    ax.plot(lims, lims, 'r--', linewidth=1)
    ax.set_xlabel("Actual growth_pcGDP")
    ax.set_ylabel("Fitted growth_pcGDP")
    ax.set_title("Fitted vs Actual")

    # 2. Residuals vs Fitted
    ax = axes[0, 1]
    ax.scatter(predictions, residuals, alpha=0.3, s=5)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted")

    # 3. Residual histogram
    ax = axes[1, 0]
    ax.hist(residuals, bins=50, density=True, alpha=0.7)
    ax.set_xlabel("Residuals")
    ax.set_ylabel("Density")
    ax.set_title("Residual Distribution")

    # 4. Q-Q plot
    ax = axes[1, 1]
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot")

    plt.tight_layout()
    plt.savefig(output_dir / "diagnostics.png", dpi=150)
    plt.close()

    # Additional plot: Convergence
    if result.objective_history:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(result.objective_history, marker='o', markersize=3)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Sum of Squared Residuals")
        ax.set_title("Convergence History")
        ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig(output_dir / "convergence.png", dpi=150)
        plt.close()

    # Year effects plot
    fig, ax = plt.subplots(figsize=(10, 5))
    years = [data.idx_to_year[i] for i in range(data.n_years)]
    ax.plot(years, result.params.k, marker='o', markersize=4)
    if result.params.se_k is not None:
        ax.fill_between(years,
                        result.params.k - 1.96 * result.params.se_k,
                        result.params.k + 1.96 * result.params.se_k,
                        alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel("Year")
    ax.set_ylabel("Year Effect k(t)")
    ax.set_title("Year Fixed Effects")
    plt.tight_layout()
    plt.savefig(output_dir / "year_effects.png", dpi=150)
    plt.close()


def save_residuals(result: FitResult, data: FittingData,
                   output_dir: Path) -> None:
    """Save residuals to CSV for further analysis."""
    predictions = predict_from_data(data, result.params)

    df = pd.DataFrame({
        "country_idx": data.country_idx,
        "iso_id": [data.idx_to_iso[i] for i in data.country_idx],
        "year_idx": data.year_idx,
        "year": [data.idx_to_year[i] for i in data.year_idx],
        "actual": data.growth_pcGDP,
        "fitted": predictions,
        "residual": result.residuals,
    })

    df.to_csv(output_dir / "residuals.csv", index=False)


def save_climate_response_plots(result: FitResult, data: FittingData,
                                output_dir: Path) -> None:
    """Generate climate response plots with uncertainty.

    Creates two plots:
    1. Climate response scaling with GDP - shows how g(GDP)*h scales with GDP
    2. Climate response at mean GDP - shows h(T,P) response surface
    """
    params = result.params
    GDP0 = params.GDP0

    # =========================================================================
    # Plot 1: Climate response scaling with GDP
    # Shows g(GDP) * h for different GDP levels, with uncertainty from alpha
    # =========================================================================

    # Use median temperature and precipitation for the h value
    T_median = np.median(data.temp)
    P_median = np.median(data.precp)
    h_at_median = params.h0 + params.h1 * T_median + params.h2 * T_median**2 + \
                  params.h3 * P_median + params.h4 * P_median**2

    # GDP range from data
    gdp_min, gdp_max = data.pcGDP.min(), data.pcGDP.max()
    gdp_range = np.logspace(np.log10(gdp_min), np.log10(gdp_max), 100)

    # g(GDP) for the estimated alpha
    g_values = (gdp_range / GDP0) ** (-params.alpha)
    climate_response = g_values * h_at_median

    fig, ax = plt.subplots(figsize=(10, 6))

    # Main line
    ax.plot(gdp_range, climate_response, 'b-', linewidth=2,
            label=f'$\\alpha$ = {params.alpha:.4f}')

    # Uncertainty bands from alpha SE
    if params.se_alpha is not None:
        alpha_lo = max(0, params.alpha - 1.96 * params.se_alpha)
        alpha_hi = min(1, params.alpha + 1.96 * params.se_alpha)

        g_lo = (gdp_range / GDP0) ** (-alpha_lo)
        g_hi = (gdp_range / GDP0) ** (-alpha_hi)

        response_lo = g_lo * h_at_median
        response_hi = g_hi * h_at_median

        # Fill between (handle case where lo and hi might swap)
        ax.fill_between(gdp_range,
                        np.minimum(response_lo, response_hi),
                        np.maximum(response_lo, response_hi),
                        alpha=0.3, color='blue',
                        label=f'95% CI ($\\alpha$ = {alpha_lo:.4f} to {alpha_hi:.4f})')

    ax.axhline(y=h_at_median, color='gray', linestyle='--', linewidth=1,
               label=f'h at GDP0 (={h_at_median:.4f})')
    ax.axvline(x=GDP0, color='red', linestyle=':', linewidth=1,
               label=f'GDP0 = {GDP0:.0f}')

    ax.set_xscale('log')
    ax.set_xlabel('Per Capita GDP')
    ax.set_ylabel('Climate Response: g(GDP) × h(T,P)')
    ax.set_title(f'Climate Response Scaling with GDP\n(at T={T_median:.1f}°C, P={P_median:.2f})')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "climate_response_vs_gdp.png", dpi=150)
    plt.close()

    # =========================================================================
    # Plot 2: Climate response at mean GDP (h(T,P) surface)
    # At GDP = GDP0, g(GDP0/GDP0) = 1, so response is just h(T,P)
    # =========================================================================

    # Temperature range from data
    T_min, T_max = data.temp.min(), data.temp.max()
    T_range = np.linspace(T_min, T_max, 50)

    # Compute h(T) at median precipitation
    h_values = params.h0 + params.h1 * T_range + params.h2 * T_range**2 + \
               params.h3 * P_median + params.h4 * P_median**2

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: h(T) at median P with uncertainty
    ax = axes[0]
    ax.plot(T_range, h_values, 'b-', linewidth=2, label='Estimate')

    # Uncertainty propagation for h(T) = h0 + h1*T + h2*T^2 + h3*P + h4*P^2
    # Variance of h = sum of (partial derivative * SE)^2 assuming independence
    if params.se_h0 is not None:
        var_h = (params.se_h0**2 +
                 (T_range * params.se_h1)**2 +
                 (T_range**2 * params.se_h2)**2 +
                 (P_median * params.se_h3)**2 +
                 (P_median**2 * params.se_h4)**2)
        se_h = np.sqrt(var_h)

        ax.fill_between(T_range, h_values - 1.96 * se_h, h_values + 1.96 * se_h,
                        alpha=0.3, color='blue', label='95% CI')

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Climate Response h(T,P)')
    ax.set_title(f'Temperature Response at GDP0\n(P = {P_median:.2f})')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Right plot: 2D contour of h(T,P)
    ax = axes[1]
    P_min, P_max = data.precp.min(), data.precp.max()
    P_range = np.linspace(P_min, P_max, 50)

    T_grid, P_grid = np.meshgrid(T_range, P_range)
    h_grid = params.h0 + params.h1 * T_grid + params.h2 * T_grid**2 + \
             params.h3 * P_grid + params.h4 * P_grid**2

    contour = ax.contourf(T_grid, P_grid, h_grid, levels=20, cmap='RdBu_r')
    plt.colorbar(contour, ax=ax, label='h(T,P)')

    # Mark the optimal temperature (where dh/dT = 0)
    if params.h2 != 0:
        T_opt = -params.h1 / (2 * params.h2)
        if T_min <= T_opt <= T_max:
            ax.axvline(x=T_opt, color='black', linestyle='--', linewidth=1,
                       label=f'Optimal T = {T_opt:.1f}°C')
            ax.legend(loc='best')

    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Log Precipitation (P)')
    ax.set_title('Climate Response Surface h(T,P) at GDP0')

    plt.tight_layout()
    plt.savefig(output_dir / "climate_response_surface.png", dpi=150)
    plt.close()


def save_optimization_results(result: FitResult, data: FittingData,
                              output_dir: Path) -> None:
    """Save alpha optimization results (plot and CSV) if available."""
    if result.grid_search_alphas is None:
        return

    # Save CSV (sorted by alpha)
    df = pd.DataFrame({
        'alpha': result.grid_search_alphas,
        'objective': result.grid_search_objectives,
    })
    df.to_csv(output_dir / "alpha_optimization.csv", index=False)

    # Save plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(result.grid_search_alphas, result.grid_search_objectives,
            'b-', linewidth=2, marker='o', markersize=6)
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Objective (Sum of Squared Residuals)')
    ax.set_title(f'Alpha Optimization Path (Brent\'s Method)\n(GDP0 = {data.pop_weighted_mean_gdp:.0f}, {data.gdp0_reference_year})')
    ax.grid(True, alpha=0.3)

    # Mark minimum
    min_idx = np.argmin(result.grid_search_objectives)
    ax.axvline(x=result.grid_search_alphas[min_idx], color='red', linestyle='--',
               label=f'Optimum: alpha={result.grid_search_alphas[min_idx]:.4f}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "alpha_optimization.png", dpi=150)
    plt.close()


def save_all_outputs(result: FitResult, data: FittingData,
                     output_dir: Optional[Path] = None) -> Path:
    """Save all outputs to a timestamped directory.

    Args:
        result: FitResult from fitting
        data: FittingData used in fitting
        output_dir: Optional specific output directory

    Returns:
        Path to the output directory
    """
    if output_dir is None:
        output_dir = create_output_directory()

    print(f"Saving outputs to {output_dir}")

    save_parameters(result, data, output_dir)
    save_summary(result, data, output_dir)
    save_diagnostic_plots(result, data, output_dir)
    save_residuals(result, data, output_dir)
    save_climate_response_plots(result, data, output_dir)
    save_optimization_results(result, data, output_dir)

    print("  - global_params.json")
    print("  - country_params.csv")
    print("  - year_effects.csv")
    print("  - summary.json / summary.txt")
    print("  - diagnostics.png")
    print("  - convergence.png")
    print("  - year_effects.png")
    print("  - residuals.csv")
    print("  - climate_response_vs_gdp.png")
    print("  - climate_response_surface.png")
    if result.grid_search_alphas is not None:
        print("  - alpha_optimization.png")
        print("  - alpha_optimization.csv")

    return output_dir


def _json_serializer(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if obj is None:
        return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
