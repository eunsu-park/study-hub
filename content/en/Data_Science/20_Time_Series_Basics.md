# 20. Time Series Analysis Fundamentals

[Previous: Bayesian Inference](./19_Bayesian_Inference.md) | [Next: Time Series Models](./21_Time_Series_Models.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Identify and describe the four components of a time series: trend, seasonality, cycle, and residual
2. Distinguish between stationary and non-stationary series and explain why stationarity matters for modeling
3. Apply the ADF and KPSS unit root tests and interpret their combined results
4. Transform non-stationary series into stationary ones using first differencing and seasonal differencing
5. Compute and interpret the ACF and PACF to identify candidate time series models (AR, MA, ARMA)
6. Perform additive, multiplicative, and STL decomposition and assess seasonal strength
7. Conduct a comprehensive exploratory time series analysis combining visualization, stationarity tests, and decomposition

---

Most statistical methods assume that observations are independent, but time series data is inherently ordered and autocorrelated -- today's value depends on yesterday's. Ignoring this temporal structure leads to misleading conclusions and poor forecasts. This lesson introduces the fundamental concepts and diagnostic tools you need before fitting any time series model: understanding the components hidden in a series, testing whether it is stationary, and decomposing it into interpretable parts.

---

## 1. Components of Time Series

### 1.1 Four Components

| Component | Description | Example |
|----------|------|------|
| **Trend** | Long-term increase/decrease pattern | Population growth, technological advancement |
| **Seasonality** | Pattern repeating at fixed intervals | Summer air conditioner sales, year-end shopping |
| **Cycle** | Repeating pattern with non-fixed intervals | Business cycle (irregular) |
| **Residual/Noise** | Unexplained random variation | Measurement error, unpredictable factors |

### 1.2 Time Series Data Generation and Visualization

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

np.random.seed(42)

# Generate time series data
n_points = 365 * 3  # 3 years of daily data
dates = pd.date_range(start='2021-01-01', periods=n_points, freq='D')
t = np.arange(n_points)

# Components
trend = 0.05 * t  # Linear trend
# 2π/365 is the angular frequency for a yearly cycle: one full sine wave = 2π radians,
# and we want exactly one cycle per 365 days.  Dividing by a different period (e.g., 7)
# gives a different cycle length — the denominator *is* the period in time units.
seasonal = 10 * np.sin(2 * np.pi * t / 365)  # Annual seasonality
weekly = 3 * np.sin(2 * np.pi * t / 7)  # Weekly pattern
noise = np.random.normal(0, 2, n_points)  # Noise

# Synthetic time series
y = 100 + trend + seasonal + weekly + noise

# Create DataFrame
ts_data = pd.DataFrame({
    'date': dates,
    'value': y,
    'trend': 100 + trend,
    'seasonal': seasonal,
    'weekly': weekly,
    'noise': noise
})
# Setting 'date' as a DatetimeIndex enables powerful time-based operations that are
# unavailable on a plain integer index: .resample('W') for resampling, .rolling(7)
# for moving windows, and intuitive date slicing like ts_data['2022-06':'2022-12'].
ts_data.set_index('date', inplace=True)

# Visualization
fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

axes[0].plot(ts_data.index, ts_data['value'], 'b-', alpha=0.7)
axes[0].set_ylabel('Value')
axes[0].set_title('Synthetic Time Series')
axes[0].grid(True, alpha=0.3)

axes[1].plot(ts_data.index, ts_data['trend'], 'g-')
axes[1].set_ylabel('Trend')
axes[1].set_title('Trend')
axes[1].grid(True, alpha=0.3)

axes[2].plot(ts_data.index, ts_data['seasonal'], 'orange')
axes[2].set_ylabel('Seasonal')
axes[2].set_title('Seasonality (Annual)')
axes[2].grid(True, alpha=0.3)

axes[3].plot(ts_data.index, ts_data['weekly'], 'purple')
axes[3].set_ylabel('Weekly')
axes[3].set_title('Weekly Pattern')
axes[3].grid(True, alpha=0.3)

axes[4].plot(ts_data.index, ts_data['noise'], 'gray', alpha=0.7)
axes[4].set_ylabel('Noise')
axes[4].set_title('Residual (Noise)')
axes[4].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 1.3 Real Data Examples

```python
# statsmodels built-in datasets
import statsmodels.api as sm
from statsmodels.datasets import co2, sunspots

# CO2 data
co2_data = co2.load_pandas().data
co2_data = co2_data.resample('M').mean()  # Monthly average

# Sunspot data
sunspots_data = sunspots.load_pandas().data
sunspots_data.index = pd.date_range(start='1700', periods=len(sunspots_data), freq='Y')

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# CO2: Clear trend + seasonality
axes[0].plot(co2_data.index, co2_data['co2'])
axes[0].set_xlabel('Year')
axes[0].set_ylabel('CO2 (ppm)')
axes[0].set_title('Mauna Loa CO2: Trend + Seasonality')
axes[0].grid(True, alpha=0.3)

# Sunspots: Cyclic pattern (~11 years)
axes[1].plot(sunspots_data.index, sunspots_data['SUNACTIVITY'])
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Sunspot Activity')
axes[1].set_title('Sunspot Activity: ~11-year cycle')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 2. Stationarity

### 2.1 Concept of Stationarity

**Strict Stationarity**:
Joint distribution of all orders invariant to time shift

**Weak Stationarity**:
1. Mean constant over time: E[Yₜ] = μ
2. Variance constant over time: Var(Yₜ) = σ²
3. Autocovariance depends only on lag: Cov(Yₜ, Yₜ₊ₕ) = γ(h)

```python
def demonstrate_stationarity():
    """Compare stationary and non-stationary series"""
    np.random.seed(42)
    n = 500

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Stationary: White noise
    white_noise = np.random.normal(0, 1, n)
    axes[0, 0].plot(white_noise)
    axes[0, 0].set_title('Stationary: White Noise')
    axes[0, 0].axhline(0, color='r', linestyle='--')
    axes[0, 0].set_ylim(-4, 4)

    # Stationary: AR(1) |φ| < 1
    ar1_stationary = np.zeros(n)
    phi = 0.7
    for t in range(1, n):
        ar1_stationary[t] = phi * ar1_stationary[t-1] + np.random.normal(0, 1)
    axes[0, 1].plot(ar1_stationary)
    axes[0, 1].set_title(f'Stationary: AR(1), φ={phi}')
    axes[0, 1].axhline(0, color='r', linestyle='--')

    # Stationary: MA(1)
    ma1 = np.zeros(n)
    errors = np.random.normal(0, 1, n)
    theta = 0.6
    for t in range(1, n):
        ma1[t] = errors[t] + theta * errors[t-1]
    axes[0, 2].plot(ma1)
    axes[0, 2].set_title(f'Stationary: MA(1), θ={theta}')
    axes[0, 2].axhline(0, color='r', linestyle='--')

    # Non-stationary: Linear trend
    trend_ts = 0.1 * np.arange(n) + np.random.normal(0, 2, n)
    axes[1, 0].plot(trend_ts)
    axes[1, 0].set_title('Non-stationary: Linear Trend')

    # Non-stationary: Random walk (unit root)
    random_walk = np.cumsum(np.random.normal(0, 1, n))
    axes[1, 1].plot(random_walk)
    axes[1, 1].set_title('Non-stationary: Random Walk (Unit Root)')

    # Non-stationary: Heteroscedastic
    heteroscedastic = np.zeros(n)
    for t in range(n):
        heteroscedastic[t] = np.random.normal(0, 1 + 0.01 * t)
    axes[1, 2].plot(heteroscedastic)
    axes[1, 2].set_title('Non-stationary: Heteroscedastic (Increasing Variance)')

    for ax in axes.flatten():
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

demonstrate_stationarity()
```

### 2.2 Why Stationarity Matters

```python
print("=== Why Stationarity Matters ===")
print()
print("1. Foundation for Statistical Inference")
print("   - Sample statistics (mean, variance) provide consistent estimates")
print("   - For non-stationary series, sample mean doesn't estimate population mean")
print()
print("2. Predictability")
print("   - Stationary series: Past patterns repeat in the future")
print("   - Non-stationary series: Long-term forecasting impossible or unstable")
print()
print("3. Model Application")
print("   - ARMA models assume stationarity")
print("   - Non-stationary series require transformation before modeling (differencing, log, etc.)")
```

---

## 3. Unit Root Tests

### 3.1 ADF Test (Augmented Dickey-Fuller)

**Hypotheses**:
- H₀: Unit root exists (non-stationary)
- H₁: No unit root (stationary)

```python
from statsmodels.tsa.stattools import adfuller

def adf_test(series, title=''):
    """
    Perform ADF unit root test and print results

    Parameters:
    -----------
    series : array-like
        Time series data
    title : str
        Name of the series
    """
    result = adfuller(series, autolag='AIC')

    print(f"=== ADF Test: {title} ===")
    print(f"Test Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    print(f"Lags Used: {result[2]}")
    print(f"Observations: {result[3]}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"   {key}: {value:.4f}")

    if result[1] < 0.05:
        print("Conclusion: p < 0.05 → No unit root (Stationary)")
    else:
        print("Conclusion: p >= 0.05 → Unit root exists (Non-stationary)")
    print()

    return result[1]  # Return p-value

# Test
np.random.seed(42)

# Stationary series
stationary = np.random.normal(0, 1, 500)
adf_test(stationary, 'White Noise (Stationary)')

# Non-stationary series: Random walk
random_walk = np.cumsum(np.random.normal(0, 1, 500))
adf_test(random_walk, 'Random Walk (Non-stationary)')

# Non-stationary series: Trend
trend = 0.1 * np.arange(500) + np.random.normal(0, 1, 500)
adf_test(trend, 'Linear Trend (Non-stationary)')
```

### 3.2 KPSS Test

**Hypotheses** (opposite of ADF):
- H₀: Stationary (trend stationary)
- H₁: Non-stationary (unit root)

```python
from statsmodels.tsa.stattools import kpss

def kpss_test(series, title='', regression='c'):
    """
    Perform KPSS test

    Parameters:
    -----------
    regression : str
        'c' - constant only (level stationarity)
        'ct' - constant + trend (trend stationarity)
    """
    result = kpss(series, regression=regression, nlags='auto')

    print(f"=== KPSS Test: {title} ===")
    print(f"Test Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    print(f"Lags Used: {result[2]}")
    print("Critical Values:")
    for key, value in result[3].items():
        print(f"   {key}: {value:.4f}")

    if result[1] < 0.05:
        print("Conclusion: p < 0.05 → Non-stationary")
    else:
        print("Conclusion: p >= 0.05 → Stationary")
    print()

    return result[1]

# Combined ADF and KPSS interpretation
np.random.seed(42)

test_series = {
    'White Noise': np.random.normal(0, 1, 500),
    'Random Walk': np.cumsum(np.random.normal(0, 1, 500)),
    'Trend + Noise': 0.1 * np.arange(500) + np.random.normal(0, 1, 500),
}

print("=== Combined ADF + KPSS Interpretation ===")
print("ADF p < 0.05 AND KPSS p >= 0.05 → Stationary")
print("ADF p >= 0.05 AND KPSS p < 0.05 → Non-stationary (differencing needed)")
print("Both p < 0.05 → Trend stationary (stationary after detrending)")
print("Both p >= 0.05 → Inconclusive (further analysis needed)")
print()

for name, series in test_series.items():
    print(f"--- {name} ---")
    adf_p = adf_test(series, f'{name} (ADF)')
    kpss_p = kpss_test(series, f'{name} (KPSS)')
    print()
```

---

## 4. Differencing

### 4.1 Concept of Differencing

**First differencing**: ∇Yₜ = Yₜ - Yₜ₋₁

**Second differencing**: ∇²Yₜ = ∇(∇Yₜ) = (Yₜ - Yₜ₋₁) - (Yₜ₋₁ - Yₜ₋₂)

**Seasonal differencing**: ∇ₛYₜ = Yₜ - Yₜ₋ₛ (s = seasonal period)

```python
def demonstrate_differencing(y, title=''):
    """Visualize the effect of differencing"""

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    # Original
    axes[0, 0].plot(y)
    axes[0, 0].set_title(f'Original: {title}')
    axes[0, 0].grid(True, alpha=0.3)

    # First difference
    diff1 = np.diff(y)
    axes[1, 0].plot(diff1)
    axes[1, 0].set_title('First Difference')
    axes[1, 0].grid(True, alpha=0.3)

    # Second difference
    diff2 = np.diff(y, n=2)
    axes[2, 0].plot(diff2)
    axes[2, 0].set_title('Second Difference')
    axes[2, 0].grid(True, alpha=0.3)

    # ADF test results
    adf_original = adfuller(y)[1]
    adf_diff1 = adfuller(diff1)[1]
    adf_diff2 = adfuller(diff2)[1]

    # Histograms
    axes[0, 1].hist(y, bins=30, density=True, alpha=0.7)
    axes[0, 1].set_title(f'Original Distribution (ADF p={adf_original:.4f})')

    axes[1, 1].hist(diff1, bins=30, density=True, alpha=0.7)
    axes[1, 1].set_title(f'First Difference Distribution (ADF p={adf_diff1:.4f})')

    axes[2, 1].hist(diff2, bins=30, density=True, alpha=0.7)
    axes[2, 1].set_title(f'Second Difference Distribution (ADF p={adf_diff2:.4f})')

    plt.tight_layout()
    plt.show()

    return adf_original, adf_diff1, adf_diff2

# Random walk
np.random.seed(42)
random_walk = np.cumsum(np.random.normal(0, 1, 500))
demonstrate_differencing(random_walk, 'Random Walk')

# Quadratic trend
t = np.arange(500)
quadratic_trend = 0.001 * t**2 + np.random.normal(0, 2, 500)
demonstrate_differencing(quadratic_trend, 'Quadratic Trend')
```

### 4.2 Seasonal Differencing

```python
# Data with seasonality
np.random.seed(42)
n = 365 * 3
t = np.arange(n)

# Trend + annual seasonality + noise
seasonal_ts = 0.01 * t + 10 * np.sin(2 * np.pi * t / 365) + np.random.normal(0, 1, n)

fig, axes = plt.subplots(4, 1, figsize=(14, 12))

# Original
axes[0].plot(seasonal_ts)
axes[0].set_title(f'Original (ADF p={adfuller(seasonal_ts)[1]:.4f})')
axes[0].grid(True, alpha=0.3)

# Regular first difference
diff1 = np.diff(seasonal_ts)
axes[1].plot(diff1)
axes[1].set_title(f'First Difference (ADF p={adfuller(diff1)[1]:.4f})')
axes[1].grid(True, alpha=0.3)

# Seasonal difference (lag=365)
# Regular first differencing removes a linear trend but leaves seasonal structure intact,
# because subtracting yesterday from today doesn't cancel a yearly cycle.
# Seasonal differencing (lag=s) removes the seasonality by comparing each observation
# to the same point exactly one season ago — eliminating any pattern with period s.
seasonal_diff = seasonal_ts[365:] - seasonal_ts[:-365]
axes[2].plot(seasonal_diff)
axes[2].set_title(f'Seasonal Difference (lag=365) (ADF p={adfuller(seasonal_diff)[1]:.4f})')
axes[2].grid(True, alpha=0.3)

# Seasonal difference + first difference
both_diff = np.diff(seasonal_diff)
axes[3].plot(both_diff)
axes[3].set_title(f'Seasonal + First Difference (ADF p={adfuller(both_diff)[1]:.4f})')
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 5. ACF and PACF

### 5.1 Autocorrelation Function (ACF)

**Autocovariance**: γ(h) = Cov(Yₜ, Yₜ₊ₕ)

**Autocorrelation**: ρ(h) = γ(h) / γ(0) = Corr(Yₜ, Yₜ₊ₕ)

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf

def analyze_acf_pacf(y, lags=40, title=''):
    """ACF/PACF analysis"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Time series
    axes[0, 0].plot(y[:200])  # First 200 points only
    axes[0, 0].set_title(f'Time Series: {title}')
    axes[0, 0].grid(True, alpha=0.3)

    # Histogram
    axes[0, 1].hist(y, bins=30, density=True, alpha=0.7)
    axes[0, 1].set_title('Distribution')
    axes[0, 1].grid(True, alpha=0.3)

    # ACF
    plot_acf(y, lags=lags, ax=axes[1, 0], alpha=0.05)
    axes[1, 0].set_title('ACF (Autocorrelation Function)')

    # PACF
    plot_pacf(y, lags=lags, ax=axes[1, 1], alpha=0.05, method='ywm')
    axes[1, 1].set_title('PACF (Partial Autocorrelation Function)')

    plt.tight_layout()
    plt.show()

# Analyze various patterns
np.random.seed(42)
n = 500

# White noise
white_noise = np.random.normal(0, 1, n)
analyze_acf_pacf(white_noise, title='White Noise')

# AR(1) process
ar1 = np.zeros(n)
phi = 0.8
for t in range(1, n):
    ar1[t] = phi * ar1[t-1] + np.random.normal(0, 1)
analyze_acf_pacf(ar1, title=f'AR(1), φ={phi}')

# MA(1) process
ma1 = np.zeros(n)
errors = np.random.normal(0, 1, n)
theta = 0.6
for t in range(1, n):
    ma1[t] = errors[t] + theta * errors[t-1]
analyze_acf_pacf(ma1, title=f'MA(1), θ={theta}')
```

### 5.2 PACF (Partial Autocorrelation Function)

**Concept**: Correlation after removing the effect of intermediate lags

```python
def pacf_intuition():
    """Intuitive understanding of PACF"""

    print("=== PACF Intuition ===")
    print()
    print("ACF(lag 2) = Corr(Yₜ, Yₜ₋₂)")
    print("  → Includes the effect of lag 1")
    print()
    print("PACF(lag 2) = Corr(Yₜ, Yₜ₋₂ | Yₜ₋₁)")
    print("  → Pure effect of lag 2 after removing lag 1's influence")
    print()
    print("Applications:")
    print("  - AR(p) model: PACF cuts off after lag p")
    print("  - MA(q) model: ACF cuts off after lag q")
    print("  - ARMA: Both decay exponentially/oscillate")

pacf_intuition()
```

### 5.3 ACF/PACF Patterns for Model Identification

```python
def acf_pacf_patterns():
    """ACF/PACF patterns and model correspondence"""

    patterns = """
    | Model | ACF Pattern | PACF Pattern |
    |------|----------|-----------|
    | AR(p) | Exponential/oscillating decay | Cuts off after lag p |
    | MA(q) | Cuts off after lag q | Exponential/oscillating decay |
    | ARMA(p,q) | Exponential/oscillating decay | Exponential/oscillating decay |
    | AR(1) φ>0 | Exponential decay | Significant only at lag 1 |
    | AR(1) φ<0 | Alternating decay | Significant only at lag 1 (negative) |
    | AR(2) | Damped sine wave or mixed exponentials | Cuts off at lag 2 |
    | MA(1) θ>0 | Significant only at lag 1 (negative) | Exponential decay |
    | MA(1) θ<0 | Significant only at lag 1 (positive) | Alternating decay |
    """
    print(patterns)

acf_pacf_patterns()

# Visual examples
fig, axes = plt.subplots(4, 3, figsize=(15, 12))

np.random.seed(42)
n = 500
errors = np.random.normal(0, 1, n)

# AR(1) φ = 0.8
ar1_pos = np.zeros(n)
for t in range(1, n):
    ar1_pos[t] = 0.8 * ar1_pos[t-1] + errors[t]

# AR(1) φ = -0.8
ar1_neg = np.zeros(n)
for t in range(1, n):
    ar1_neg[t] = -0.8 * ar1_neg[t-1] + errors[t]

# MA(1) θ = 0.8
ma1_pos = np.zeros(n)
for t in range(1, n):
    ma1_pos[t] = errors[t] + 0.8 * errors[t-1]

# AR(2)
ar2 = np.zeros(n)
for t in range(2, n):
    ar2[t] = 0.5 * ar2[t-1] + 0.3 * ar2[t-2] + errors[t]

models = [
    (ar1_pos, 'AR(1) φ=0.8'),
    (ar1_neg, 'AR(1) φ=-0.8'),
    (ma1_pos, 'MA(1) θ=0.8'),
    (ar2, 'AR(2) φ₁=0.5, φ₂=0.3'),
]

for i, (y, title) in enumerate(models):
    axes[i, 0].plot(y[:200])
    axes[i, 0].set_title(title)
    axes[i, 0].grid(True, alpha=0.3)

    plot_acf(y, lags=20, ax=axes[i, 1], alpha=0.05)
    axes[i, 1].set_title(f'{title} - ACF')

    plot_pacf(y, lags=20, ax=axes[i, 2], alpha=0.05, method='ywm')
    axes[i, 2].set_title(f'{title} - PACF')

plt.tight_layout()
plt.show()
```

---

## 6. Time Series Decomposition

### 6.1 Additive vs Multiplicative Models

**Additive Model**:
$$Y_t = T_t + S_t + R_t$$

**Multiplicative Model**:
$$Y_t = T_t \times S_t \times R_t$$

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Compare additive and multiplicative
np.random.seed(42)
n = 365 * 3
t = np.arange(n)

# Additive time series
trend = 100 + 0.1 * t
seasonal_add = 10 * np.sin(2 * np.pi * t / 365)
noise_add = np.random.normal(0, 5, n)
additive_ts = trend + seasonal_add + noise_add

# Multiplicative time series (seasonal variation proportional to level)
trend = 100 + 0.1 * t
seasonal_mult = 1 + 0.1 * np.sin(2 * np.pi * t / 365)
noise_mult = np.random.normal(1, 0.05, n)
multiplicative_ts = trend * seasonal_mult * noise_mult

fig, axes = plt.subplots(2, 1, figsize=(14, 6))

dates = pd.date_range(start='2021-01-01', periods=n, freq='D')

axes[0].plot(dates, additive_ts)
axes[0].set_title('Additive Model: Constant seasonal variation')
axes[0].grid(True, alpha=0.3)

axes[1].plot(dates, multiplicative_ts)
axes[1].set_title('Multiplicative Model: Seasonal variation proportional to level')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Model selection criteria:")
print("- Constant seasonal variation → Additive")
print("- Seasonal variation proportional to level → Multiplicative")
print("- If uncertain, try log transformation then additive model")
```

### 6.2 Classical Decomposition

```python
# Example: decompose monthly data
# Generate airline passenger-like data
np.random.seed(42)
n_months = 144  # 12 years
t = np.arange(n_months)

# Multiplicative pattern data
trend = 100 + 2 * t + 0.01 * t**2
seasonal = 1 + 0.2 * np.sin(2 * np.pi * t / 12) + 0.1 * np.cos(4 * np.pi * t / 12)
noise = np.random.normal(1, 0.03, n_months)
airline_like = trend * seasonal * noise

dates = pd.date_range(start='2010-01', periods=n_months, freq='M')
ts_series = pd.Series(airline_like, index=dates)

# Additive decomposition
decomposition_add = seasonal_decompose(ts_series, model='additive', period=12)

# Multiplicative decomposition
decomposition_mult = seasonal_decompose(ts_series, model='multiplicative', period=12)

# Visualization
fig, axes = plt.subplots(4, 2, figsize=(14, 12))

# Additive decomposition
axes[0, 0].plot(ts_series)
axes[0, 0].set_title('Original')
axes[0, 0].set_ylabel('Value')

axes[1, 0].plot(decomposition_add.trend)
axes[1, 0].set_title('Additive - Trend')
axes[1, 0].set_ylabel('Trend')

axes[2, 0].plot(decomposition_add.seasonal)
axes[2, 0].set_title('Additive - Seasonality')
axes[2, 0].set_ylabel('Seasonality')

axes[3, 0].plot(decomposition_add.resid)
axes[3, 0].set_title('Additive - Residual')
axes[3, 0].set_ylabel('Residual')

# Multiplicative decomposition
axes[0, 1].plot(ts_series)
axes[0, 1].set_title('Original')

axes[1, 1].plot(decomposition_mult.trend)
axes[1, 1].set_title('Multiplicative - Trend')

axes[2, 1].plot(decomposition_mult.seasonal)
axes[2, 1].set_title('Multiplicative - Seasonality')

axes[3, 1].plot(decomposition_mult.resid)
axes[3, 1].set_title('Multiplicative - Residual')

for ax in axes.flatten():
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 6.3 STL Decomposition (Seasonal and Trend decomposition using Loess)

```python
from statsmodels.tsa.seasonal import STL

# STL decomposition (more flexible)
stl = STL(ts_series, period=12, robust=True)
result = stl.fit()

fig, axes = plt.subplots(4, 1, figsize=(12, 10))

result.observed.plot(ax=axes[0])
axes[0].set_ylabel('Observed')
axes[0].set_title('STL Decomposition')

result.trend.plot(ax=axes[1])
axes[1].set_ylabel('Trend')

result.seasonal.plot(ax=axes[2])
axes[2].set_ylabel('Seasonal')

result.resid.plot(ax=axes[3])
axes[3].set_ylabel('Residual')

for ax in axes:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Advantages of STL
print("=== Advantages of STL Decomposition ===")
print("1. Seasonal pattern can change over time")
print("2. Robust to outliers (robust=True option)")
print("3. Flexibility in trend extraction control")
print("4. Can handle asymmetric seasonal patterns")
```

---

## 7. Real Data Analysis Example

### 7.1 Comprehensive Analysis Function

```python
def comprehensive_time_series_analysis(series, period=None, title=''):
    """
    Comprehensive time series analysis

    Parameters:
    -----------
    series : pd.Series
        Time series data (requires DatetimeIndex)
    period : int
        Seasonal period (tries auto-detection if None)
    title : str
        Title
    """
    print(f"=== Comprehensive Time Series Analysis: {title} ===\n")

    # Basic statistics
    print("1. Basic Statistics")
    print(f"   Number of observations: {len(series)}")
    print(f"   Mean: {series.mean():.4f}")
    print(f"   Standard deviation: {series.std():.4f}")
    print(f"   Min/Max: {series.min():.4f} / {series.max():.4f}")
    print(f"   Period: {series.index.min()} ~ {series.index.max()}")
    print()

    # Stationarity tests
    print("2. Stationarity Tests")
    adf_result = adfuller(series.dropna())
    print(f"   ADF statistic: {adf_result[0]:.4f}")
    print(f"   ADF p-value: {adf_result[1]:.4f}")

    kpss_result = kpss(series.dropna(), regression='c')
    print(f"   KPSS statistic: {kpss_result[0]:.4f}")
    print(f"   KPSS p-value: {kpss_result[1]:.4f}")

    if adf_result[1] < 0.05 and kpss_result[1] >= 0.05:
        print("   Conclusion: Stationary series")
        is_stationary = True
    elif adf_result[1] >= 0.05:
        print("   Conclusion: Non-stationary (differencing needed)")
        is_stationary = False
    else:
        print("   Conclusion: Further analysis needed")
        is_stationary = False
    print()

    # Visualization
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # Original time series
    series.plot(ax=axes[0, 0])
    axes[0, 0].set_title(f'Original Time Series: {title}')
    axes[0, 0].grid(True, alpha=0.3)

    # Distribution
    series.hist(bins=30, ax=axes[0, 1], density=True, alpha=0.7)
    axes[0, 1].set_title('Distribution')
    axes[0, 1].grid(True, alpha=0.3)

    # ACF
    plot_acf(series.dropna(), lags=min(40, len(series)//4), ax=axes[1, 0], alpha=0.05)
    axes[1, 0].set_title('ACF')

    # PACF
    plot_pacf(series.dropna(), lags=min(40, len(series)//4), ax=axes[1, 1],
              alpha=0.05, method='ywm')
    axes[1, 1].set_title('PACF')

    # Differencing
    diff_series = series.diff().dropna()
    diff_series.plot(ax=axes[2, 0])
    axes[2, 0].set_title(f'First Difference (ADF p={adfuller(diff_series)[1]:.4f})')
    axes[2, 0].grid(True, alpha=0.3)

    # Differenced ACF
    plot_acf(diff_series, lags=min(40, len(diff_series)//4), ax=axes[2, 1], alpha=0.05)
    axes[2, 1].set_title('ACF after Differencing')

    plt.tight_layout()
    plt.show()

    # Decomposition (if period is given)
    if period is not None and len(series) >= 2 * period:
        print("3. Time Series Decomposition")
        try:
            stl = STL(series.dropna(), period=period, robust=True)
            result = stl.fit()

            fig, axes = plt.subplots(4, 1, figsize=(12, 10))

            result.observed.plot(ax=axes[0])
            axes[0].set_ylabel('Observed')
            axes[0].set_title('STL Decomposition')

            result.trend.plot(ax=axes[1])
            axes[1].set_ylabel('Trend')

            result.seasonal.plot(ax=axes[2])
            axes[2].set_ylabel('Seasonal')

            result.resid.plot(ax=axes[3])
            axes[3].set_ylabel('Residual')

            for ax in axes:
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

            # Seasonal strength
            var_seasonal = result.seasonal.var()
            var_resid = result.resid.var()
            seasonal_strength = max(0, 1 - var_resid / (var_seasonal + var_resid))
            print(f"   Seasonal strength: {seasonal_strength:.3f}")

        except Exception as e:
            print(f"   Decomposition failed: {e}")

    return is_stationary

# Real data analysis
# CO2 data analysis
co2_data = co2.load_pandas().data['co2'].resample('M').mean().dropna()
comprehensive_time_series_analysis(co2_data, period=12, title='Mauna Loa CO2')
```

---

## 8. Practice Problems

### Problem 1: Determining Stationarity
Determine whether the following time series are stationary and explain why:
1. yₜ = 0.5yₜ₋₁ + εₜ
2. yₜ = yₜ₋₁ + εₜ
3. yₜ = t + εₜ
4. yₜ = sin(2πt/12) + εₜ

### Problem 2: Determining Differencing Order
Use ADF test to determine the number of differences needed to make the following data stationary:
- Random walk with drift: yₜ = 0.1 + yₜ₋₁ + εₜ

### Problem 3: ACF/PACF Interpretation
Identify the models corresponding to the following ACF/PACF patterns:
1. ACF: significant up to lag 3, then 0  |  PACF: exponential decay
2. ACF: exponential decay  |  PACF: significant up to lag 2, then 0

### Problem 4: Time Series Decomposition
When monthly data shows:
- Seasonal variation that increases over time
- Long-term upward trend

Which decomposition model is appropriate, and how should you preprocess the data?

---

## 9. Key Summary

### Time Series Analysis Checklist

1. **Visualization**: Check data patterns (trend, seasonality, outliers)
2. **Stationarity Test**: ADF + KPSS
3. **Transformation/Differencing**: Non-stationary → stationary
4. **ACF/PACF Analysis**: Model identification
5. **Decomposition**: Separate and understand components

### Main Tests and Interpretation

| Test | H₀ | Conclusion Criterion |
|------|-----|----------|
| ADF | Unit root exists | p < 0.05 → stationary |
| KPSS | Stationary | p < 0.05 → non-stationary |

### ACF/PACF Patterns

| Model | ACF | PACF |
|------|-----|------|
| AR(p) | Gradual decay | Cuts off after lag p |
| MA(q) | Cuts off after lag q | Gradual decay |
| ARMA | Gradual decay | Gradual decay |

### Next Chapter Preview

Chapter 11 **Time Series Models** will cover:
- AR, MA, ARMA models
- ARIMA model and differencing
- Model identification and estimation
- Forecasting and evaluation
