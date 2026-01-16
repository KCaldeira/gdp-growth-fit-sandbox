#!/usr/bin/env python3
"""Run a single constrained model fit (no bootstrap).

Usage:
    python run_constrained_fit.py
"""

from src.data_loader import load_data
from src.fitting import fit_model_constrained

data = load_data('data/input/df_base_withPop.csv')
result = fit_model_constrained(data, verbose=True)

print(f'\n=== Results ===')
print(f'T* = {result.T_opt:.2f}°C')
print(f'P* = {result.P_opt:.4f}')
print(f'alpha = {result.params.alpha:.6f}')
print(f'h2 = {result.params.h2:.6e}')
print(f'h4 = {result.params.h4:.6e}')
print(f'R² = {result.r_squared:.4f}')
