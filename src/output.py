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
        f.write(f"  GDP0:  {params.GDP0:12.4f}")
        if params.se_GDP0:
            f.write(f"  (SE: {params.se_GDP0:.4f})")
        f.write("\n")
        f.write(f"  alpha: {params.alpha:12.6f}")
        if params.se_alpha:
            f.write(f"  (SE: {params.se_alpha:.6f})")
        f.write("\n\n")

        f.write("Climate Response (h):\n")
        f.write(f"  h0: {params.h0:12.6f}")
        if params.se_h0:
            f.write(f"  (SE: {params.se_h0:.6f})")
        f.write("\n")
        f.write(f"  h1: {params.h1:12.6f}")
        if params.se_h1:
            f.write(f"  (SE: {params.se_h1:.6f})")
        f.write("\n")
        f.write(f"  h2: {params.h2:12.6f}")
        if params.se_h2:
            f.write(f"  (SE: {params.se_h2:.6f})")
        f.write("\n")
        f.write(f"  h3: {params.h3:12.6f}")
        if params.se_h3:
            f.write(f"  (SE: {params.se_h3:.6f})")
        f.write("\n")
        f.write(f"  h4: {params.h4:12.6f}")
        if params.se_h4:
            f.write(f"  (SE: {params.se_h4:.6f})")
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

    print("  - global_params.json")
    print("  - country_params.csv")
    print("  - year_effects.csv")
    print("  - summary.json / summary.txt")
    print("  - diagnostics.png")
    print("  - convergence.png")
    print("  - year_effects.png")
    print("  - residuals.csv")

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
