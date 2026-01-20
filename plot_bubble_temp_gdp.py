#!/usr/bin/env python
"""
Bubble plot of temperature vs log10(per-capita GDP) with population as bubble size.

Usage:
    python plot_bubble_temp_gdp.py [--year YEAR] [--animate]

Options:
    --year YEAR    Year to plot (default: 2020)
    --animate      Create animation through all available years
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_FILE = Path("data/input/df_base_withPop.csv")
OUTPUT_DIR = Path("data/output")


def load_data():
    """Load and prepare the data."""
    df = pd.read_csv(DATA_FILE)
    # Add log10 of per-capita GDP
    df["log10_pcGDP"] = np.log10(df["pcGDP"])
    return df


def plot_bubble_year(df, year, ax):
    """Plot bubble chart for a single year."""
    df_year = df[df["year"] == year].copy()

    # Scale bubble sizes (population in millions, scaled for visibility)
    pop_millions = df_year["Pop"] / 1e6
    # Scale so largest bubble is reasonable size
    bubble_sizes = (pop_millions / pop_millions.max()) * 6000

    scatter = ax.scatter(
        df_year["temp"],
        df_year["log10_pcGDP"],
        s=bubble_sizes,
        alpha=0.6,
        edgecolors="black",
        linewidth=0.8,
    )

    ax.set_xlabel("Mean Temperature (°C)", fontsize=12)
    ax.set_ylabel("Per-capita GDP (2015$)", fontsize=12)
    ax.set_title(f"Temperature vs GDP by Country ({year})", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Set y-axis ticks at $100, $1,000, $10,000, $100,000
    ax.set_yticks([2, 3, 4, 5])
    ax.set_yticklabels(["$100", "$1,000", "$10,000", "$100,000"])

    return scatter


def create_static_plot(df, year):
    """Create a static bubble plot for a single year."""
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_bubble_year(df, year, ax)
    ax.set_xlim(0, 30)
    ax.set_ylim(2, 5.2)  # $100 to ~$150,000
    plt.tight_layout()

    output_file = OUTPUT_DIR / f"bubble_temp_gdp_{year}.png"
    plt.savefig(output_file, dpi=150)
    print(f"Saved: {output_file}")
    plt.close()


def create_animation(df, low_res=False):
    """Create an animated bubble plot through all years."""
    from PIL import Image
    import io

    years = sorted(df["year"].unique())

    # Set consistent axis limits across all frames
    x_min, x_max = 0, 30
    y_min, y_max = 2, 5.2  # $100 to ~$150,000

    if low_res:
        output_file = OUTPUT_DIR / "bubble_temp_gdp_animated_lowres.gif"
        dpi = 30
        n_colors = 16
    else:
        output_file = OUTPUT_DIR / "bubble_temp_gdp_animated.gif"
        dpi = 100
        n_colors = 256

    frames = []
    for year in years:
        # Create fresh figure for each frame to ensure consistency
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        plot_bubble_year(df, year, ax)
        plt.tight_layout()

        # Save frame to buffer
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        img = Image.open(buf).copy()  # Copy to detach from buffer
        buf.close()
        plt.close(fig)

        # Convert to palette mode for compression
        img = img.convert("P", palette=Image.ADAPTIVE, colors=n_colors)
        frames.append(img)

    # Ensure all frames have identical size (use first frame as reference)
    target_size = frames[0].size
    frames = [f.resize(target_size, Image.LANCZOS) if f.size != target_size else f for f in frames]

    # Save as animated GIF - don't use optimize which can cause inconsistencies
    frames[0].save(
        output_file,
        save_all=True,
        append_images=frames[1:],
        duration=100,  # ms per frame
        loop=0,
    )
    print(f"Saved: {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bubble plot of temperature vs GDP with population size"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2020,
        help="Year to plot (default: 2020)",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Create animation through all available years",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = load_data()

    # Check if requested year exists
    available_years = sorted(df["year"].unique())
    print(f"Available years: {available_years[0]} - {available_years[-1]}")

    if args.animate:
        create_animation(df, low_res=False)
        create_animation(df, low_res=True)
    else:
        if args.year not in available_years:
            print(f"Year {args.year} not in data. Using {available_years[-1]} instead.")
            args.year = available_years[-1]
        create_static_plot(df, args.year)


if __name__ == "__main__":
    main()
