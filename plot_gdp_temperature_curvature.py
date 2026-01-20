#!/usr/bin/env python
"""
2x5 panel plot of g(GDP) * h2 * (T - Topt)^2 and its derivative
for multiple GDP levels with 90% bootstrap confidence bands.

Top row: f(T) = g(GDP) * h2 * (T - Topt)^2
Bottom row: df/dT = 2 * g(GDP) * h2 * (T - Topt)

Usage:
    python plot_gdp_temperature_curvature.py [BOOTSTRAP_DIR]

If no directory is provided, uses the most recent ./data/output/bootstrap* directory.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# GDP percentiles (2018-2022 country means, unweighted)
GDP_PERCENTILES = {
    10: 795.92,
    30: 2462.25,
    50: 5488.50,
    70: 13671.19,
    90: 43957.55,
}

# Temperature range
T_RANGE = np.linspace(0, 30, 301)


def find_most_recent_bootstrap_dir():
    """Find the most recent bootstrap directory in ./data/output/.

    Only considers directories matching bootstrap_YYYYMMDD_HHMMSS pattern.
    """
    output_dir = Path("data/output")
    pattern = re.compile(r"^bootstrap_\d{8}_\d{6}$")
    bootstrap_dirs = sorted(
        d for d in output_dir.glob("bootstrap_*") if pattern.match(d.name)
    )
    if not bootstrap_dirs:
        raise FileNotFoundError(
            "No bootstrap directories found in ./data/output/ "
            "(looking for bootstrap_YYYYMMDD_HHMMSS pattern)"
        )
    return bootstrap_dirs[-1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot GDP-scaled temperature curvature effect with bootstrap CI"
    )
    parser.add_argument(
        "bootstrap_dir",
        nargs="?",
        type=Path,
        help="Bootstrap output directory (default: most recent ./data/output/bootstrap*)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of bootstrap samples to use (default: all)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine bootstrap directory
    if args.bootstrap_dir is not None:
        bootstrap_dir = args.bootstrap_dir
    else:
        bootstrap_dir = find_most_recent_bootstrap_dir()
        print(f"Using most recent bootstrap directory: {bootstrap_dir}")

    # Load point estimates
    with open(bootstrap_dir / "global_params.json") as f:
        params = json.load(f)

    GDP0 = params["GDP0"]["estimate"]
    alpha_point = params["alpha"]["estimate"]
    h1_point = params["h1"]["estimate"]
    h2_point = params["h2"]["estimate"]

    # Load bootstrap samples
    df_boot = pd.read_csv(bootstrap_dir / "bootstrap_coefficients_simple.csv")

    # Limit number of samples if requested
    if args.max_samples is not None and args.max_samples < len(df_boot):
        df_boot = df_boot.iloc[: args.max_samples]
        n_samples_used = args.max_samples
        output_file = bootstrap_dir / f"gdp_temperature_curvature_n{n_samples_used}.png"
    else:
        n_samples_used = len(df_boot)
        output_file = bootstrap_dir / "gdp_temperature_curvature.png"

    alpha_samples = df_boot["alpha"].values
    h1_samples = df_boot["h1"].values
    h2_samples = df_boot["h2"].values
    n_boot = len(df_boot)

    # Compute point estimate
    Topt_point = -h1_point / (2 * h2_point)

    # Pre-compute all curves for each GDP percentile
    gdp_data = {}
    for pct, gdp in GDP_PERCENTILES.items():
        # Point estimate curves
        g_point = (gdp / GDP0) ** (-alpha_point)
        func_point = g_point * h2_point * (T_RANGE - Topt_point) ** 2
        deriv_point = 2 * g_point * h2_point * (T_RANGE - Topt_point)

        # Bootstrap curves
        func_boot = np.zeros((n_boot, len(T_RANGE)))
        deriv_boot = np.zeros((n_boot, len(T_RANGE)))
        for i in range(n_boot):
            Topt_i = -h1_samples[i] / (2 * h2_samples[i])
            g_i = (gdp / GDP0) ** (-alpha_samples[i])
            func_boot[i, :] = g_i * h2_samples[i] * (T_RANGE - Topt_i) ** 2
            deriv_boot[i, :] = 2 * g_i * h2_samples[i] * (T_RANGE - Topt_i)

        # 90% CI (5th and 95th percentiles)
        gdp_data[pct] = {
            "gdp": gdp,
            "func_point": func_point,
            "func_lower": np.percentile(func_boot, 5, axis=0),
            "func_upper": np.percentile(func_boot, 95, axis=0),
            "deriv_point": deriv_point,
            "deriv_lower": np.percentile(deriv_boot, 5, axis=0),
            "deriv_upper": np.percentile(deriv_boot, 95, axis=0),
        }

    # Determine shared y-axis limits for each row
    func_min = min(d["func_lower"].min() for d in gdp_data.values())
    func_max = max(d["func_upper"].max() for d in gdp_data.values())
    func_margin = (func_max - func_min) * 0.05
    func_ylim = (func_min - func_margin, func_max + func_margin)

    deriv_min = min(d["deriv_lower"].min() for d in gdp_data.values())
    deriv_max = max(d["deriv_upper"].max() for d in gdp_data.values())
    deriv_margin = (deriv_max - deriv_min) * 0.05
    deriv_ylim = (deriv_min - deriv_margin, deriv_max + deriv_margin)

    # Create 2x5 figure (A4 landscape: 297mm x 210mm = 11.69" x 8.27")
    fig, axes = plt.subplots(2, 5, figsize=(11.69, 8.27), sharey="row")

    # Color for all panels
    color = "steelblue"

    for col, (pct, data) in enumerate(gdp_data.items()):
        gdp = data["gdp"]

        # Top row: function
        ax_top = axes[0, col]
        ax_top.fill_between(
            T_RANGE, data["func_lower"], data["func_upper"], alpha=0.3, color=color
        )
        ax_top.plot(T_RANGE, data["func_point"], color=color, linewidth=1.5)
        ax_top.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax_top.axvline(Topt_point, color="red", linestyle=":", linewidth=1)
        ax_top.set_xlim(0, 30)
        ax_top.set_ylim(func_ylim)
        ax_top.set_title(f"{pct}th pct\n${gdp:,.0f}", fontsize=10)
        ax_top.grid(True, alpha=0.3)
        if col == 0:
            ax_top.set_ylabel(r"$g \cdot h_2 \cdot (T - T^*)^2$", fontsize=10)

        # Bottom row: derivative
        ax_bot = axes[1, col]
        ax_bot.fill_between(
            T_RANGE, data["deriv_lower"], data["deriv_upper"], alpha=0.3, color=color
        )
        ax_bot.plot(T_RANGE, data["deriv_point"], color=color, linewidth=1.5)
        ax_bot.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax_bot.axvline(Topt_point, color="red", linestyle=":", linewidth=1)
        ax_bot.set_xlim(0, 30)
        ax_bot.set_ylim(deriv_ylim)
        ax_bot.set_xlabel("Temperature (°C)", fontsize=9)
        ax_bot.grid(True, alpha=0.3)
        if col == 0:
            ax_bot.set_ylabel(r"$\frac{d}{dT}[g \cdot h_2 \cdot (T - T^*)^2]$", fontsize=10)

    # Add overall title
    fig.suptitle(
        f"GDP-scaled Temperature Curvature Effect (90% bootstrap CI, n={n_samples_used})\n"
        f"Red line: T* = {Topt_point:.1f}°C",
        fontsize=12,
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_file, dpi=150)
    print(f"Saved: {output_file}")
    plt.close()


if __name__ == "__main__":
    main()
