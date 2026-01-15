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
- `P` = log(precipitation) = log(precp)

### Component Functions

**g(GDP)** - GDP convergence term:
```
g(GDP) = (GDP / GDP0)^(-alpha)
```
Parameters: `GDP0`, `alpha`

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
Parameters: `k[t]` for each year t

## Data

Input data is stored in `./data/input/` and contains:
- **iso_id**: Country ISO code
- **year**: Year of observation
- **pcGDP**: Per capita GDP
- **growth_pcGDP**: Growth rate of per capita GDP
- **temp**: Temperature
- **precp**: Precipitation
- **time/time2**: Time indices for trend analysis
- **Pop**: Population

## Output

Results from each test run are saved to timestamped subdirectories in `./data/output/`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

(To be added as fitting algorithms are implemented)

## Project Structure

```
gdp-growth-fit-sandbox/
├── data/
│   ├── input/          # Source CSV data
│   └── output/         # Timestamped test results
├── requirements.txt
└── README.md
```
