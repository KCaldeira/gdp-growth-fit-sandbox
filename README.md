# GDP Growth Fit Sandbox

A project for testing non-linear curve fitting algorithms on GDP, temperature, and precipitation data.

## Project Overview

This project fits a non-linear model relating economic growth to GDP levels and climate variables (temperature and precipitation), with country-specific trends and year fixed effects.

## Model Specification

The model to be fitted is:

```
growth_pcGDP(i,t) = g(pcGDP[i,t]) * h(T[i,t], P[i,t]) + j(i,t) + k(t)
```

Where:
- `i` = country index (integer mapping of iso_id)
- `t` = year
- `T` = temperature (temp)
- `P` = log(precipitation) = precp (already log-transformed in input data)

### Component Functions

**g(GDP)** - GDP convergence term:
```
g(GDP) = (GDP / GDP0)^(-alpha)
```
- `GDP0` is **fixed** to the population-weighted mean GDP for the most recent year in the data
- `alpha` is estimated, constrained to [0, 1]

**h(T, P)** - Climate response function:
```
h(T, P) = h0 + h1*T + h2*T^2 + h3*P + h4*P^2
```
Parameters: `h0`, `h1`, `h2`, `h3`, `h4`

**j(i, t)** - Country-specific time trends:
```
j(i, t) = j0[i] + j1[i]*t + j2[i]*t^2
```
Parameters: `j0[i]`, `j1[i]`, `j2[i]` for each country i

**k(t)** - Year fixed effects:
```
k(t) = k[t]
```
Parameters: `k[t]` for each year t (with k[first_year] = 0 for identifiability)

## Fitting Algorithm

Uses **alternating estimation** to handle the non-linear structure:

1. Fix `alpha` → solve linear least squares for h, j, k parameters
2. Fix h, j, k → optimize over `alpha` using bounded scalar optimization
3. Iterate until convergence

Standard errors are computed via numerical Hessian of the log-likelihood.

## Data

Input data is stored in `./data/input/` and contains:
- **iso_id**: Country ISO code
- **year**: Year of observation
- **pcGDP**: Per capita GDP
- **growth_pcGDP**: Growth rate of per capita GDP
- **temp**: Temperature
- **precp**: Log precipitation (already transformed)
- **time/time2**: Time indices for trend analysis
- **Pop**: Population

## Output

Results from each test run are saved to timestamped subdirectories in `./data/output/`.

### Output Files
- `global_params.json` - GDP0, alpha, h0-h4 with standard errors
- `country_params.csv` - j0, j1, j2 for each country
- `year_effects.csv` - k[t] for each year
- `summary.json/txt` - Fit statistics (R², RMSE, AIC, BIC)
- `diagnostics.png` - Residual diagnostic plots
- `convergence.png` - Optimization convergence history
- `year_effects.png` - Year fixed effects with 95% CI
- `climate_response_vs_gdp.png` - Climate response scaling with GDP (with uncertainty)
- `climate_response_surface.png` - h(T,P) response surface at GDP0
- `residuals.csv` - Fitted values and residuals

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Run with default settings
python scripts/run_fit.py

# Custom options
python scripts/run_fit.py --max-iter 100 --alpha-init 0.5

# Specify output directory
python scripts/run_fit.py --output data/output/my_test
```

### Command Line Options
- `--input, -i`: Path to input CSV file (default: data/input/df_base_withPop.csv)
- `--output, -o`: Output directory (default: timestamped subdirectory)
- `--max-iter`: Maximum iterations for alternating estimation (default: 100)
- `--tol`: Convergence tolerance (default: 1e-6)
- `--alpha-init`: Initial value for alpha parameter (default: 0.1)
- `--quiet, -q`: Suppress progress output

## Project Structure

```
gdp-growth-fit-sandbox/
├── data/
│   ├── input/              # Source CSV data
│   └── output/             # Timestamped test results
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Load CSV, create mappings
│   ├── model.py            # Model component functions
│   ├── fitting.py          # Alternating estimation algorithm
│   └── output.py           # Save results and plots
├── scripts/
│   └── run_fit.py          # Main entry point
├── requirements.txt
└── README.md
```
