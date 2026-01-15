# GDP Growth Fit Sandbox

A project for testing non-linear curve fitting algorithms on GDP, temperature, and precipitation data.

## Project Overview

This project explores relationships between economic growth and climate variables using various non-linear regression techniques.

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
