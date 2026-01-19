# GDP Growth Fit Sandbox

A project for fitting non-linear models relating economic growth to GDP levels and climate variables (temperature and precipitation), with country-specific trends and year fixed effects. Based on Burke, Hsiang & Miguel (2015) methodology.

## Current Status

The project now supports:
- **Three model variants** with different assumptions about how GDP scaling affects components
- **Cluster bootstrap uncertainty analysis** (resampling countries with replacement)
- **Monte Carlo-based confidence intervals** that don't assume normality of errors
- **Hessian-based standard errors** (assuming normality) for comparison

## Model Specification

### Base Model (default)

```
growth_pcGDP(i,t) = g(pcGDP[i,t]) * h(T[i,t], P[i,t]) + j(i,t) + k(t)
```

### Alternative Model Variants

Two alternative specifications are available via the `--model-variant` flag:

| Variant | Equation | Description |
|---------|----------|-------------|
| `base` | `g*h + j + k` | GDP scaling applies only to climate response |
| `g_scales_hj` | `g*(h + j) + k` | GDP scaling applies to climate + country trends |
| `g_scales_all` | `g*(h + j + k)` | GDP scaling applies to everything |

For `g_scales_hj` and `g_scales_all`, h0 is absorbed into j0 (4 h parameters instead of 5).

### Component Functions

**g(GDP)** - GDP convergence term:
```
g(GDP) = (GDP / GDP0)^(-alpha)
```
- `GDP0` is **fixed** using 5-year (2018-2022) country averages: for each country, compute the mean population and mean GDP over these 5 years, then compute the population-weighted mean GDP across countries
- `alpha` is estimated, constrained to [0.01, 0.99]

**h(T, P)** - Climate response function:
```
h(T, P) = h0 + h1*T + h2*T^2 + h3*P + h4*P^2
```
Parameters: `h0`, `h1`, `h2`, `h3`, `h4` (h0 omitted for non-base variants)

**j(i, t)** - Country-specific time trends:
```
j(i, t) = j0[i] + j1[i]*t + j2[i]*t^2
```
For identifiability, the first country is the reference: `j0[0] = j1[0] = j2[0] = 0`

**k(t)** - Year fixed effects:
```
k(t) = k[t]   (all years estimated)
```

### Alternative Parameterization (Mean-Centered)

An alternative parameterization is also computed where the j parameters sum to zero across all countries:
- `j0_mc`, `j1_mc`, `j2_mc`: Mean-centered country effects (sum to zero)
- `k_mc`: Adjusted year effects to maintain identical predictions

This transformation makes k represent the global mean trend, with j representing country-specific deviations from that mean.

## Uncertainty Analysis

### Cluster Bootstrap (Recommended)

The cluster bootstrap resamples **countries with replacement**, preserving within-country correlation across years. This approach:
- Does not assume normality of errors
- Captures the full parameter correlation structure
- Provides percentile-based confidence intervals at multiple levels

**Reported percentile ranges:**
| Range | Percentiles | Description |
|-------|-------------|-------------|
| 2 SD | [2.28%, 97.72%] | 95.45% of normal distribution |
| 95% | [2.5%, 97.5%] | Standard 95% CI |
| 90% | [5%, 95%] | 90% CI |
| 1 SD | [15.87%, 84.13%] | 68.27% of normal distribution |
| 50% (IQR) | [25%, 75%] | Interquartile range |

**Derived quantities:**
- **T\*** = -h1/(2×h2): Optimal temperature
- **P\*** = -h3/(2×h4): Optimal log-precipitation

```bash
python run_bootstrap.py --n-bootstrap 1000 --model-variant base
```

### Hessian-Based Standard Errors

Also computed automatically via numerical Hessian of the negative log-likelihood. These assume Gaussian errors but account for correlations between all parameters.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Dependencies:** numpy, pandas, matplotlib, scipy, scikit-learn, openpyxl

## Usage

### Bootstrap Uncertainty Analysis

```bash
# Run with defaults (1000 bootstrap iterations, base model)
python run_bootstrap.py

# Specify model variant and iterations
python run_bootstrap.py --model-variant g_scales_hj --n-bootstrap 500

# Full options
python run_bootstrap.py \
  --n-bootstrap 1000 \
  --seed 42 \
  --model-variant base \
  --output-dir ./data/output/my_analysis \
  --data-file data/input/df_base_withPop.csv
```

### Compute Statistics from Existing Results

You can compute summary statistics from a previously-generated bootstrap file (CSV or XLSX) without re-running the bootstrap:

```bash
# Compute stats from existing file (output goes to same directory)
python run_bootstrap.py --from-file ./data/output/bootstrap_20260118_065142/bootstrap_coefficients_simple.csv

# Also works with XLSX files
python run_bootstrap.py --from-file ./path/to/bootstrap_coefficients.xlsx

# Specify different output directory
python run_bootstrap.py --from-file ./path/to/bootstrap_coefficients_simple.csv --output-dir ./my_stats
```

Note: `--from-csv` is still supported as an alias for backward compatibility.

### Plot GDP-Temperature Curvature Effect

Generate a 2×5 panel plot showing the GDP-scaled temperature curvature effect `g(GDP) × h₂ × (T - T*)²` and its derivative for different GDP percentiles:

```bash
# Use most recent bootstrap directory (default)
python plot_gdp_temperature_curvature.py

# Specify a specific bootstrap directory
python plot_gdp_temperature_curvature.py data/output/bootstrap_20260119_084350
```

Output: `gdp_temperature_curvature.png` saved to the bootstrap directory.

### Command Line Options (run_bootstrap.py)

| Option | Default | Description |
|--------|---------|-------------|
| `--n-bootstrap` | 1000 | Number of bootstrap iterations |
| `--seed` | 42 | Random seed for reproducibility |
| `--model-variant` | base | Model variant: base, g_scales_hj, g_scales_all |
| `--output-dir` | auto | Output directory (default: ./data/output/bootstrap_{timestamp}) |
| `--data-file` | data/input/df_base_withPop.csv | Input data file |
| `--from-file` | none | Compute stats from existing file (CSV or XLSX, skips bootstrap) |
| `--constrained` | false | Use constrained model with h(T\*, P\*) = 0 |

## Output Files

### Bootstrap Analysis Output

The bootstrap now produces both point estimate outputs and bootstrap-specific outputs:

```
data/output/bootstrap_{timestamp}/
├── # Point estimate outputs (same as run_fit.py)
├── global_params.json               # GDP0, alpha, h0-h4 with Hessian SEs
├── country_params.xlsx              # j0, j1, j2 (Sheet 1: original, Sheet 2: mean_centered)
├── year_effects.xlsx                # k[t] (Sheet 1: original, Sheet 2: mean_centered)
├── summary.json/txt                 # Fit statistics (R², RMSE, AIC, BIC)
├── diagnostics.png                  # Residual diagnostic plots
├── climate_response_*.png           # Climate response visualizations
├── residuals.csv                    # Fitted values and residuals
│
├── # Bootstrap coefficient samples
├── bootstrap_coefficients_simple.csv    # alpha, h0-h4 for each iteration
├── bootstrap_coefficients_complete.csv  # All parameters including j and k
│
├── # Bootstrap summary (multiple CI levels, SE on median)
├── bootstrap_summary.txt            # Statistics with 2SD/95%/90%/1SD/IQR CIs
├── bootstrap_stats_from_file.txt    # (if using --from-file)
│
├── # Bootstrap plots
├── bootstrap_h_temperature.png      # h(T) with bootstrap 95% CI
├── bootstrap_g_gdp.png              # g(GDP) with bootstrap 95% CI
├── bootstrap_gh_combined.png        # g(GDP)*h(T) for multiple GDP levels
├── bootstrap_dhdT.png               # Marginal effect dh/dT with CI
├── bootstrap_alpha.png              # Bootstrap distribution of α
└── bootstrap_optimal_temperature.png # Distribution of optimal T*
```

**bootstrap_coefficients_simple.csv** columns:
- `iteration`, `alpha`, `h0`, `h1`, `h2`, `h3`, `h4`

**bootstrap_coefficients_complete.csv** columns:
- `iteration`, `alpha`, `h0`-`h4`, `j0_0`...`j0_N`, `j1_0`...`j1_N`, `j2_0`...`j2_N`, `k_0`...`k_M`

### Standard Fit Output (via run_fit.py)

```
data/output/{timestamp}/
├── global_params.json           # GDP0, alpha, h0-h4 with Hessian SEs
├── country_params.xlsx          # j0, j1, j2 (Sheet 1: original, Sheet 2: mean_centered)
├── year_effects.xlsx            # k[t] (Sheet 1: original, Sheet 2: mean_centered)
├── summary.json/txt             # Fit statistics (R², RMSE, AIC, BIC)
├── diagnostics.png              # Residual diagnostic plots
├── climate_response_*.png       # Climate response visualizations
└── residuals.csv                # Fitted values and residuals
```

**XLSX File Structure:**
- **country_params.xlsx**:
  - Sheet "original": `iso_id`, `j0`, `j1`, `j2`, `se_j0`, `se_j1`, `se_j2`
  - Sheet "mean_centered": `iso_id`, `j0_mc`, `j1_mc`, `j2_mc` (sum to zero)
- **year_effects.xlsx**:
  - Sheet "original": `year`, `k`, `se_k`
  - Sheet "mean_centered": `year`, `k_mc` (adjusted for mean-centering)

## Project Structure

```
gdp-growth-fit-sandbox/
├── data/
│   ├── input/                   # Source CSV data (DO NOT MODIFY)
│   └── output/                  # Timestamped results
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Load CSV, compute population-weighted means
│   ├── model.py                 # Model component functions (g, h, j, k)
│   ├── fitting.py               # Profile likelihood optimization + model variants
│   ├── bootstrap.py             # Cluster bootstrap uncertainty analysis
│   └── output.py                # Save results and diagnostic plots
├── scripts/
│   └── run_fit.py               # Basic fitting entry point
├── run_bootstrap.py             # Bootstrap analysis entry point
├── plot_gdp_temperature_curvature.py  # GDP-scaled curvature effect plot
├── requirements.txt
├── CLAUDE.md                    # Coding style guide
└── README.md
```

## Fitting Algorithm

Uses **profile likelihood optimization**:

1. **Brent's method** searches for optimal α in [0.01, 0.99]
2. For each candidate α, solve the **full linear least squares** for h, j, k
3. The objective (sum of squared residuals) is evaluated exactly

This ensures reliable convergence by always evaluating the true objective.

## Reference Values

Following Burke et al. methodology:
- **GDP0**: Population-weighted mean per capita GDP using 5-year (2018-2022) country averages
- **P_const**: Population-weighted mean log-precipitation using 5-year (2018-2022) country averages
- **Temperature range**: 0°C to 30°C (standard for climate-growth plots)

## Data

Input CSV columns:
- `iso_id`: Country ISO code
- `year`: Year of observation
- `pcGDP`: Per capita GDP
- `growth_pcGDP`: Growth rate (target variable)
- `temp`: Temperature (°C)
- `precp`: Log precipitation (already transformed)
- `time`: Time index for trends
- `Pop`: Population
