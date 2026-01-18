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
- `GDP0` is **fixed** to the population-weighted mean GDP for the most recent year
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

**k(t)** - Year fixed effects:
```
k(t) = k[t]   (with k[first_year] = 0 for identifiability)
```

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

You can compute summary statistics from a previously-generated bootstrap CSV without re-running the bootstrap:

```bash
# Compute stats from existing CSV (output goes to same directory)
python run_bootstrap.py --from-csv ./data/output/bootstrap_20260118_065142/bootstrap_coefficients_simple.csv

# Specify different output directory
python run_bootstrap.py --from-csv ./path/to/bootstrap_coefficients_simple.csv --output-dir ./my_stats
```

### Command Line Options (run_bootstrap.py)

| Option | Default | Description |
|--------|---------|-------------|
| `--n-bootstrap` | 1000 | Number of bootstrap iterations |
| `--seed` | 42 | Random seed for reproducibility |
| `--model-variant` | base | Model variant: base, g_scales_hj, g_scales_all |
| `--output-dir` | auto | Output directory (default: ./data/output/bootstrap_{timestamp}) |
| `--data-file` | data/input/df_base_withPop.csv | Input data file |
| `--from-csv` | none | Compute stats from existing CSV (skips bootstrap) |
| `--constrained` | false | Use constrained model with h(T\*, P\*) = 0 |

## Output Files

### Bootstrap Analysis Output

The bootstrap now produces both point estimate outputs and bootstrap-specific outputs:

```
data/output/bootstrap_{timestamp}/
├── # Point estimate outputs (same as run_fit.py)
├── global_params.json               # GDP0, alpha, h0-h4 with Hessian SEs
├── country_params.csv               # j0, j1, j2 for each country
├── year_effects.csv                 # k[t] for each year
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
├── bootstrap_stats_from_csv.txt     # (if using --from-csv)
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
├── country_params.csv           # j0, j1, j2 for each country
├── year_effects.csv             # k[t] for each year
├── summary.json/txt             # Fit statistics (R², RMSE, AIC, BIC)
├── diagnostics.png              # Residual diagnostic plots
├── climate_response_*.png       # Climate response visualizations
└── residuals.csv                # Fitted values and residuals
```

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
- **GDP0**: Population-weighted mean per capita GDP (most recent year)
- **P_const**: Population-weighted mean log-precipitation (for plotting h(T) curves)
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
