"""
Vector Autoregression (VAR) Model — Complete Example
=====================================================
Demonstrates multivariate time series modeling with VAR, including:
  1. Simulating a bivariate VAR(2) process
  2. Stationarity testing and lag order selection
  3. Model fitting and parameter interpretation
  4. Granger causality testing
  5. Impulse Response Functions (IRF)
  6. Forecast Error Variance Decomposition (FEVD)
  7. Forecasting with confidence intervals

Dependencies: numpy, pandas, matplotlib, statsmodels
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM


# =============================================================================
# 1. Simulate a Bivariate VAR(2) Process
# =============================================================================
# Why simulate? We know the true parameters, so we can verify that the
# estimator recovers them. This builds intuition before applying VAR to
# real data where ground truth is unknown.

np.random.seed(42)
n = 500
K = 2  # Number of variables

# True coefficient matrices for VAR(2):
#   Y_t = A1 * Y_{t-1} + A2 * Y_{t-2} + u_t
A1_true = np.array([
    [0.5,  0.1],   # GDP_growth depends on its own lag (0.5) and Inflation lag (0.1)
    [0.2,  0.3],   # Inflation depends on GDP_growth lag (0.2) and its own lag (0.3)
])
A2_true = np.array([
    [-0.2,  0.0],
    [ 0.1, -0.1],
])

# Error covariance — off-diagonal captures contemporaneous correlation
Sigma_u = np.array([
    [1.0, 0.3],
    [0.3, 0.8],
])

# Generate the process
y = np.zeros((n, K))
for t in range(2, n):
    y[t] = (A1_true @ y[t - 1]
            + A2_true @ y[t - 2]
            + np.random.multivariate_normal([0, 0], Sigma_u))

df = pd.DataFrame(y, columns=['GDP_growth', 'Inflation'])

print("=" * 60)
print("1. Simulated VAR(2) Data — First 5 Rows")
print("=" * 60)
print(df.head())
print(f"\nShape: {df.shape}")


# =============================================================================
# 2. Stationarity Testing
# =============================================================================
# Why test stationarity? VAR assumes all input series are stationary.
# Non-stationary series require differencing or a VECM approach.

print("\n" + "=" * 60)
print("2. Stationarity Testing (ADF)")
print("=" * 60)

for col in df.columns:
    adf_stat, adf_p, _, _, _, _ = adfuller(df[col])
    status = "Stationary" if adf_p < 0.05 else "NON-STATIONARY"
    print(f"  {col}: ADF stat = {adf_stat:.3f}, p-value = {adf_p:.4f} -> {status}")


# =============================================================================
# 3. Lag Order Selection and Model Fitting
# =============================================================================
# Why use information criteria for lag selection? In VAR, the number of
# parameters grows as K^2 * p. Over-parameterization wastes degrees of
# freedom and increases forecast variance. AIC/BIC find the sweet spot.

print("\n" + "=" * 60)
print("3. Lag Order Selection")
print("=" * 60)

model = VAR(df)
lag_order_results = model.select_order(maxlags=8)
print(lag_order_results.summary())

# Fit using AIC-selected lag order
result = model.fit(maxlags=8, ic='aic')
print(f"\nSelected lag order (AIC): {result.k_ar}")
print(f"Number of estimated parameters per equation: "
      f"{result.k_ar * K + 1}")  # K lags * K vars + intercept

# Print estimated coefficients
print("\n--- Estimated A1 (lag 1) ---")
print(result.coefs[0].round(3))
print("--- True A1 ---")
print(A1_true)

if result.k_ar >= 2:
    print("\n--- Estimated A2 (lag 2) ---")
    print(result.coefs[1].round(3))
    print("--- True A2 ---")
    print(A2_true)


# =============================================================================
# 4. Granger Causality Testing
# =============================================================================
# Granger causality tests whether lagged values of X contain information
# that helps predict Y beyond what Y's own lags provide.
# IMPORTANT: Granger causality is about predictive precedence, NOT true
# causal mechanisms. "X Granger-causes Y" means X's past helps forecast Y.

print("\n" + "=" * 60)
print("4. Granger Causality Tests")
print("=" * 60)

# Test: Does Inflation Granger-cause GDP_growth?
print("\n--- H0: Inflation does NOT Granger-cause GDP_growth ---")
gc1 = grangercausalitytests(df[['GDP_growth', 'Inflation']], maxlag=4, verbose=False)
for lag, test_results in gc1.items():
    f_stat = test_results[0]['ssr_ftest'][0]
    p_val = test_results[0]['ssr_ftest'][1]
    sig = "*" if p_val < 0.05 else ""
    print(f"  Lag {lag}: F = {f_stat:.3f}, p = {p_val:.4f} {sig}")

# Test: Does GDP_growth Granger-cause Inflation?
print("\n--- H0: GDP_growth does NOT Granger-cause Inflation ---")
gc2 = grangercausalitytests(df[['Inflation', 'GDP_growth']], maxlag=4, verbose=False)
for lag, test_results in gc2.items():
    f_stat = test_results[0]['ssr_ftest'][0]
    p_val = test_results[0]['ssr_ftest'][1]
    sig = "*" if p_val < 0.05 else ""
    print(f"  Lag {lag}: F = {f_stat:.3f}, p = {p_val:.4f} {sig}")


# =============================================================================
# 5. Impulse Response Functions (IRF)
# =============================================================================
# IRFs trace how a one-standard-deviation shock to one variable propagates
# through the system over time. Orthogonalized IRFs use Cholesky
# decomposition to isolate "structural" shocks.

print("\n" + "=" * 60)
print("5. Impulse Response Functions")
print("=" * 60)

irf = result.irf(periods=20)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Why plot all 4 combinations? Each cell shows how a shock to the
# column variable affects the row variable over time.
responses = [
    (0, 0, 'GDP_growth -> GDP_growth'),
    (0, 1, 'Inflation -> GDP_growth'),
    (1, 0, 'GDP_growth -> Inflation'),
    (1, 1, 'Inflation -> Inflation'),
]

for ax, (resp_idx, imp_idx, title) in zip(axes.flat, responses):
    # Orthogonalized IRF values
    irf_vals = irf.orth_irfs[:, resp_idx, imp_idx]
    lower = irf.orth_ci[:, resp_idx, imp_idx, 0]  # Lower CI
    upper = irf.orth_ci[:, resp_idx, imp_idx, 1]  # Upper CI

    periods = range(len(irf_vals))
    ax.plot(periods, irf_vals, 'b-', linewidth=2)
    ax.fill_between(periods, lower, upper, alpha=0.2, color='blue')
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel('Periods')
    ax.grid(True, alpha=0.3)

plt.suptitle('Orthogonalized Impulse Response Functions (95% CI)', fontsize=14)
plt.tight_layout()
plt.savefig('var_irf.png', dpi=100, bbox_inches='tight')
plt.show()
print("  Saved: var_irf.png")


# =============================================================================
# 6. Forecast Error Variance Decomposition (FEVD)
# =============================================================================
# FEVD quantifies what fraction of each variable's forecast uncertainty
# at horizon h is attributable to shocks from each variable.
# Why FEVD matters: It reveals whether a variable is primarily "self-driven"
# or heavily influenced by other variables in the system.

print("\n" + "=" * 60)
print("6. Forecast Error Variance Decomposition")
print("=" * 60)

fevd = result.fevd(periods=20)

# Print decomposition at selected horizons
for h in [1, 5, 10, 20]:
    print(f"\n  Horizon {h}:")
    for i, col in enumerate(df.columns):
        decomp = fevd.decomp[h - 1, i, :]
        parts = ", ".join(f"{df.columns[j]}: {decomp[j]:.1%}" for j in range(K))
        print(f"    {col} variance due to -> {parts}")

# Plot FEVD
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for i, col in enumerate(df.columns):
    # Stack the contributions
    bottom = np.zeros(20)
    for j in range(K):
        vals = fevd.decomp[:, i, j]
        axes[i].bar(range(1, 21), vals, bottom=bottom,
                    label=df.columns[j], alpha=0.8)
        bottom += vals
    axes[i].set_title(f'FEVD: {col}')
    axes[i].set_xlabel('Horizon')
    axes[i].set_ylabel('Fraction of Variance')
    axes[i].legend()
    axes[i].set_ylim(0, 1.05)
    axes[i].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('var_fevd.png', dpi=100, bbox_inches='tight')
plt.show()
print("  Saved: var_fevd.png")


# =============================================================================
# 7. Forecasting with Confidence Intervals
# =============================================================================
print("\n" + "=" * 60)
print("7. Forecasting")
print("=" * 60)

forecast_steps = 15
# forecast_interval returns (point_forecast, lower_ci, upper_ci)
point_fc, lower_ci, upper_ci = result.forecast_interval(
    df.values[-result.k_ar:], steps=forecast_steps, alpha=0.05
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for i, col in enumerate(df.columns):
    # Plot last 80 observed values for context
    obs_range = range(n - 80, n)
    axes[i].plot(obs_range, df[col].values[-80:], 'b-', label='Observed')

    # Forecast
    fc_range = range(n, n + forecast_steps)
    axes[i].plot(fc_range, point_fc[:, i], 'r-', linewidth=2, label='Forecast')
    axes[i].fill_between(fc_range, lower_ci[:, i], upper_ci[:, i],
                         color='red', alpha=0.2, label='95% CI')

    axes[i].axvline(n, color='k', linestyle='--', alpha=0.5)
    axes[i].set_title(f'{col} — {forecast_steps}-step Forecast')
    axes[i].legend(loc='upper left')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('var_forecast.png', dpi=100, bbox_inches='tight')
plt.show()
print("  Saved: var_forecast.png")

# Print forecast table
fc_df = pd.DataFrame(
    point_fc, columns=[f'{c}_forecast' for c in df.columns],
    index=range(1, forecast_steps + 1)
)
fc_df.index.name = 'Step'
print("\nForecast values:")
print(fc_df.round(3))


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("""
VAR (Vector Autoregression) Key Concepts:

1. VAR(p) models K time series jointly — each variable depends on
   its own lags AND the lags of all other variables.

2. Granger causality tests whether one variable's past helps predict
   another (predictive precedence, not true causation).

3. Impulse Response Functions (IRF) trace how a shock to one variable
   propagates through the system over time.

4. Forecast Error Variance Decomposition (FEVD) quantifies what
   fraction of forecast uncertainty is due to each variable's shocks.

5. If series are non-stationary but cointegrated, use VECM instead
   of VAR on differences — VECM preserves long-run equilibrium info.

6. Parameter count grows as K^2 * p, so information criteria (AIC/BIC)
   are essential for lag selection to avoid overfitting.
""")
