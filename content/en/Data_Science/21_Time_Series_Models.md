# 21. Time Series Models

[Previous: Time Series Basics](./20_Time_Series_Basics.md) | [Next: Multivariate Analysis](./22_Multivariate_Analysis.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the structure and stationarity conditions of AR(p) models
2. Describe how MA(q) models represent a time series as a weighted sum of past shocks
3. Distinguish between AR, MA, ARMA, and ARIMA models based on ACF/PACF patterns
4. Apply the Box-Jenkins methodology to identify, estimate, and diagnose time series models
5. Compare candidate models using AIC and BIC information criteria
6. Implement rolling and fixed-horizon forecasts and evaluate their performance with RMSE and MAE
7. Trace the complete ARIMA modeling pipeline from stationarity testing through forecasting
8. Apply Vector Autoregression (VAR) for multivariate time series, including Granger causality, impulse response analysis, and variance decomposition

---

Time series data is everywhere -- stock prices, sensor readings, web traffic -- yet a single snapshot tells you nothing about what comes next. AR, MA, ARMA, and ARIMA models give you a principled framework for capturing the temporal structure hidden in sequential observations, enabling you to forecast future values and quantify the uncertainty of those forecasts. Mastering the Box-Jenkins methodology for identifying and diagnosing these models is a foundational skill for any data scientist working with temporal data.

---

## 1. AR Model (AutoRegressive Model)

### 1.1 AR(p) Model Definition

**AR(p) Model**:
$$Y_t = c + \phi_1 Y_{t-1} + \phi_2 Y_{t-2} + ... + \phi_p Y_{t-p} + \epsilon_t$$

Where:
- c: constant (intercept)
- φᵢ: AR coefficients
- εₜ: white noise, εₜ ~ WN(0, σ²)

**Stationarity condition**: All roots of the characteristic equation 1 - φ₁z - φ₂z² - ... - φₚzᵖ = 0 must be outside the unit circle

### 1.2 AR(1) Model

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm

np.random.seed(42)

def simulate_ar1(phi, n=500, c=0, sigma=1):
    """Simulate AR(1): Y_t = c + φ*Y_{t-1} + ε_t"""
    y = np.zeros(n)
    y[0] = np.random.normal(0, sigma / np.sqrt(1 - phi**2))  # Stationary initial value

    for t in range(1, n):
        y[t] = c + phi * y[t-1] + np.random.normal(0, sigma)

    return y

# Simulate AR(1) with various φ values
phi_values = [0.9, 0.5, -0.5, -0.9]
fig, axes = plt.subplots(len(phi_values), 3, figsize=(15, 12))

for i, phi in enumerate(phi_values):
    y = simulate_ar1(phi, n=500)

    # Time series
    axes[i, 0].plot(y[:200])
    axes[i, 0].set_title(f'AR(1), φ = {phi}')
    axes[i, 0].grid(True, alpha=0.3)

    # ACF
    plot_acf(y, lags=20, ax=axes[i, 1], alpha=0.05)
    axes[i, 1].set_title(f'ACF (φ = {phi})')

    # PACF
    plot_pacf(y, lags=20, ax=axes[i, 2], alpha=0.05, method='ywm')
    axes[i, 2].set_title(f'PACF (φ = {phi})')

plt.tight_layout()
plt.show()

# Theoretical ACF for AR(1)
print("Theoretical ACF for AR(1):")
for phi in phi_values:
    acf_theory = [phi**k for k in range(6)]
    print(f"  φ = {phi}: ρ(k) = {acf_theory}")
```

### 1.3 AR(2) Model

```python
def simulate_ar2(phi1, phi2, n=500, c=0, sigma=1):
    """Simulate AR(2)"""
    y = np.zeros(n)

    for t in range(2, n):
        y[t] = c + phi1 * y[t-1] + phi2 * y[t-2] + np.random.normal(0, sigma)

    return y

# AR(2) examples
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Real roots case: φ₁ = 0.5, φ₂ = 0.3
y1 = simulate_ar2(0.5, 0.3, n=500)
axes[0, 0].plot(y1[:200])
axes[0, 0].set_title('AR(2): φ₁=0.5, φ₂=0.3 (real roots)')
axes[0, 0].grid(True, alpha=0.3)
plot_acf(y1, lags=20, ax=axes[0, 1], alpha=0.05)
plot_pacf(y1, lags=20, ax=axes[0, 2], alpha=0.05, method='ywm')

# Complex roots case: φ₁ = 1.0, φ₂ = -0.5 (oscillating pattern)
y2 = simulate_ar2(1.0, -0.5, n=500)
axes[1, 0].plot(y2[:200])
axes[1, 0].set_title('AR(2): φ₁=1.0, φ₂=-0.5 (complex roots, oscillating)')
axes[1, 0].grid(True, alpha=0.3)
plot_acf(y2, lags=20, ax=axes[1, 1], alpha=0.05)
plot_pacf(y2, lags=20, ax=axes[1, 2], alpha=0.05, method='ywm')

plt.tight_layout()
plt.show()

# Check stationarity condition
def check_ar2_stationarity(phi1, phi2):
    """Check AR(2) stationarity conditions"""
    # Conditions: |φ₂| < 1, φ₂ + φ₁ < 1, φ₂ - φ₁ < 1
    cond1 = abs(phi2) < 1
    cond2 = phi2 + phi1 < 1
    cond3 = phi2 - phi1 < 1

    print(f"AR(2) φ₁={phi1}, φ₂={phi2}")
    print(f"  |φ₂| < 1: {cond1} ({abs(phi2):.2f})")
    print(f"  φ₂ + φ₁ < 1: {cond2} ({phi2 + phi1:.2f})")
    print(f"  φ₂ - φ₁ < 1: {cond3} ({phi2 - phi1:.2f})")
    print(f"  Stationarity: {cond1 and cond2 and cond3}")
    return cond1 and cond2 and cond3

check_ar2_stationarity(0.5, 0.3)
check_ar2_stationarity(1.0, -0.5)
check_ar2_stationarity(0.6, 0.5)  # Non-stationary
```

### 1.4 Estimating AR with statsmodels

```python
# Generate and estimate AR(1) data
np.random.seed(42)
true_phi = 0.7
y_ar1 = simulate_ar1(true_phi, n=300)

# Fit ARIMA(1,0,0) = AR(1)
model = ARIMA(y_ar1, order=(1, 0, 0))
result = model.fit()

print("=== AR(1) Model Estimation Results ===")
print(f"True φ = {true_phi}")
print(result.summary())

# Generate and estimate AR(2) data
true_phi1, true_phi2 = 0.5, 0.3
y_ar2 = simulate_ar2(true_phi1, true_phi2, n=300)

model_ar2 = ARIMA(y_ar2, order=(2, 0, 0))
result_ar2 = model_ar2.fit()

print("\n=== AR(2) Model Estimation Results ===")
print(f"True φ₁ = {true_phi1}, φ₂ = {true_phi2}")
print(result_ar2.summary())
```

---

## 2. MA Model (Moving Average Model)

### 2.1 MA(q) Model Definition

**MA(q) Model**:
$$Y_t = c + \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + ... + \theta_q \epsilon_{t-q}$$

Where:
- c: constant (mean)
- θᵢ: MA coefficients
- εₜ: white noise

**Characteristic**: MA model is always stationary (finite combination of past shocks)

### 2.2 MA(1) Model

```python
def simulate_ma(theta_list, n=500, c=0, sigma=1):
    """Simulate MA(q)"""
    q = len(theta_list)
    epsilon = np.random.normal(0, sigma, n + q)
    y = np.zeros(n)

    for t in range(n):
        y[t] = c + epsilon[t + q]
        for i, theta in enumerate(theta_list):
            y[t] += theta * epsilon[t + q - i - 1]

    return y

# Simulate MA(1)
theta_values = [0.8, 0.3, -0.3, -0.8]
fig, axes = plt.subplots(len(theta_values), 3, figsize=(15, 12))

for i, theta in enumerate(theta_values):
    y = simulate_ma([theta], n=500)

    axes[i, 0].plot(y[:200])
    axes[i, 0].set_title(f'MA(1), θ = {theta}')
    axes[i, 0].grid(True, alpha=0.3)

    plot_acf(y, lags=20, ax=axes[i, 1], alpha=0.05)
    axes[i, 1].set_title(f'ACF (θ = {theta})')

    plot_pacf(y, lags=20, ax=axes[i, 2], alpha=0.05, method='ywm')
    axes[i, 2].set_title(f'PACF (θ = {theta})')

plt.tight_layout()
plt.show()

# Theoretical ACF for MA(1)
print("Theoretical ACF for MA(1):")
for theta in theta_values:
    rho1 = theta / (1 + theta**2)
    print(f"  θ = {theta}: ρ(1) = {rho1:.3f}, ρ(k>1) = 0")
```

### 2.3 MA(2) Model

```python
# Simulate MA(2)
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# MA(2) example 1
y1 = simulate_ma([0.5, 0.3], n=500)
axes[0, 0].plot(y1[:200])
axes[0, 0].set_title('MA(2): θ₁=0.5, θ₂=0.3')
axes[0, 0].grid(True, alpha=0.3)
plot_acf(y1, lags=20, ax=axes[0, 1], alpha=0.05)
plot_pacf(y1, lags=20, ax=axes[0, 2], alpha=0.05, method='ywm')

# MA(2) example 2
y2 = simulate_ma([1.0, -0.5], n=500)
axes[1, 0].plot(y2[:200])
axes[1, 0].set_title('MA(2): θ₁=1.0, θ₂=-0.5')
axes[1, 0].grid(True, alpha=0.3)
plot_acf(y2, lags=20, ax=axes[1, 1], alpha=0.05)
plot_pacf(y2, lags=20, ax=axes[1, 2], alpha=0.05, method='ywm')

plt.tight_layout()
plt.show()
```

### 2.4 Estimating MA with statsmodels

```python
# Generate and estimate MA(1) data
np.random.seed(42)
true_theta = 0.6
y_ma1 = simulate_ma([true_theta], n=300)

# Fit ARIMA(0,0,1) = MA(1)
model_ma1 = ARIMA(y_ma1, order=(0, 0, 1))
result_ma1 = model_ma1.fit()

print("=== MA(1) Model Estimation Results ===")
print(f"True θ = {true_theta}")
print(result_ma1.summary())

# Estimate MA(2)
true_theta1, true_theta2 = 0.5, 0.3
y_ma2 = simulate_ma([true_theta1, true_theta2], n=300)

model_ma2 = ARIMA(y_ma2, order=(0, 0, 2))
result_ma2 = model_ma2.fit()

print("\n=== MA(2) Model Estimation Results ===")
print(f"True θ₁ = {true_theta1}, θ₂ = {true_theta2}")
print(result_ma2.summary())
```

---

## 3. ARMA Model

### 3.1 ARMA(p,q) Definition

**ARMA(p,q) Model**:
$$Y_t = c + \sum_{i=1}^{p} \phi_i Y_{t-i} + \epsilon_t + \sum_{j=1}^{q} \theta_j \epsilon_{t-j}$$

### 3.2 ARMA(1,1) Example

```python
# Use statsmodels' arma_generate_sample
np.random.seed(42)

# ARMA(1,1): φ=0.7, θ=0.4
ar = [1, -0.7]  # [1, -φ₁]
ma = [1, 0.4]   # [1, θ₁]

y_arma11 = arma_generate_sample(ar, ma, nsample=500, scale=1, burnin=100)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(y_arma11[:200])
axes[0].set_title('ARMA(1,1): φ=0.7, θ=0.4')
axes[0].grid(True, alpha=0.3)

plot_acf(y_arma11, lags=20, ax=axes[1], alpha=0.05)
axes[1].set_title('ACF')

plot_pacf(y_arma11, lags=20, ax=axes[2], alpha=0.05, method='ywm')
axes[2].set_title('PACF')

plt.tight_layout()
plt.show()

# Estimate ARMA(1,1)
model_arma = ARIMA(y_arma11, order=(1, 0, 1))
result_arma = model_arma.fit()

print("=== ARMA(1,1) Model Estimation Results ===")
print("True: φ=0.7, θ=0.4")
print(result_arma.summary())
```

---

## 4. ARIMA Model

### 4.1 ARIMA(p,d,q) Definition

**ARIMA(p,d,q)**: Integrated ARMA - includes differencing

$$\nabla^d Y_t = c + \sum_{i=1}^{p} \phi_i \nabla^d Y_{t-i} + \epsilon_t + \sum_{j=1}^{q} \theta_j \epsilon_{t-j}$$

Where ∇ᵈ denotes d-th order differencing

### 4.2 Box-Jenkins Methodology

```python
def box_jenkins_methodology():
    """Explain Box-Jenkins methodology"""
    print("""
    =================================================
    Box-Jenkins Methodology (ARIMA Modeling)
    =================================================

    Step 1: Identification
    ───────────────────────────
    - Examine time series plot for patterns
    - Test for stationarity (ADF, KPSS)
    - Difference if necessary to achieve stationarity → determine d
    - Analyze ACF/PACF of stationary series → determine p, q candidates

    Step 2: Estimation
    ───────────────────────────
    - Estimate parameters of candidate models (MLE)
    - Compare information criteria (AIC, BIC)
    - Check parameter significance

    Step 3: Diagnostic Checking
    ───────────────────────────
    - Residual analysis: Check if white noise
    - Ljung-Box test
    - Residual ACF/PACF
    - Normality test

    Step 4: Forecasting
    ───────────────────────────
    - Point forecasts
    - Prediction intervals
    - Evaluate forecast performance
    """)

box_jenkins_methodology()
```

### 4.3 Model Identification: ACF/PACF Patterns

```python
def identify_model_from_acf_pacf():
    """Guide to model identification based on ACF/PACF patterns"""
    guide = """
    =================================================
    Model Identification Based on ACF/PACF Patterns
    =================================================

    Pattern 1: ACF gradual decay, PACF cuts off after lag p
    → AR(p) model

    Pattern 2: ACF cuts off after lag q, PACF gradual decay
    → MA(q) model

    Pattern 3: Both ACF and PACF gradually decay
    → ARMA(p,q) model

    Pattern 4: ACF decays very slowly (linearly)
    → Non-stationary, differencing needed (increase d)

    Pattern 5: ACF spikes at seasonal lags
    → Consider SARIMA

    Practical Tips:
    - Cut-off: Abruptly becomes 0
    - Gradual decay: Exponentially or oscillating decay
    - Outside 95% confidence band = significant
    """
    print(guide)

identify_model_from_acf_pacf()
```

### 4.4 ARIMA Practice: Non-stationary Time Series

```python
# Generate non-stationary time series: random walk + noise
np.random.seed(42)
n = 300

# ARIMA(1,1,0) process: ΔYₜ = 0.5 * ΔYₜ₋₁ + εₜ
# i.e., Yₜ - Yₜ₋₁ = 0.5*(Yₜ₋₁ - Yₜ₋₂) + εₜ
ar = [1, -0.5]
ma = [1]
y_diff = arma_generate_sample(ar, ma, nsample=n, scale=1, burnin=100)
y_arima110 = np.cumsum(y_diff)  # Cumulate to create non-stationary series

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Original (non-stationary)
axes[0, 0].plot(y_arima110)
axes[0, 0].set_title('Original (Non-stationary)')
axes[0, 0].grid(True, alpha=0.3)

plot_acf(y_arima110, lags=20, ax=axes[0, 1], alpha=0.05)
axes[0, 1].set_title('Original ACF (slow decay)')

plot_pacf(y_arima110, lags=20, ax=axes[0, 2], alpha=0.05, method='ywm')
axes[0, 2].set_title('Original PACF')

# After first differencing (stationary)
diff1 = np.diff(y_arima110)
axes[1, 0].plot(diff1)
axes[1, 0].set_title('After First Difference (Stationary)')
axes[1, 0].grid(True, alpha=0.3)

plot_acf(diff1, lags=20, ax=axes[1, 1], alpha=0.05)
axes[1, 1].set_title('Differenced ACF')

plot_pacf(diff1, lags=20, ax=axes[1, 2], alpha=0.05, method='ywm')
axes[1, 2].set_title('Differenced PACF (AR(1) pattern)')

plt.tight_layout()
plt.show()

# Estimate ARIMA(1,1,0)
from statsmodels.tsa.stattools import adfuller

print("=== Stationarity Tests ===")
print(f"Original ADF p-value: {adfuller(y_arima110)[1]:.4f}")
print(f"After differencing ADF p-value: {adfuller(diff1)[1]:.4f}")

model = ARIMA(y_arima110, order=(1, 1, 0))
result = model.fit()

print("\n=== ARIMA(1,1,0) Estimation Results ===")
print("True: φ=0.5")
print(result.summary())
```

---

## 5. Model Selection

### 5.1 Information Criteria (AIC, BIC)

```python
def compare_arima_models(y, max_p=3, max_d=2, max_q=3):
    """
    Grid search for optimal ARIMA model
    """
    results = []

    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0:
                    continue  # Exclude meaningless model
                try:
                    model = ARIMA(y, order=(p, d, q))
                    fit = model.fit()
                    results.append({
                        'order': (p, d, q),
                        'aic': fit.aic,
                        'bic': fit.bic,
                        'llf': fit.llf
                    })
                except:
                    pass

    # Sort results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('aic')

    return results_df

# Generate data (ARIMA(1,1,1))
np.random.seed(42)
ar = [1, -0.6]
ma = [1, 0.4]
y_diff = arma_generate_sample(ar, ma, nsample=300, scale=1, burnin=100)
y = np.cumsum(y_diff)

# Compare models
print("=== ARIMA Model Comparison (by AIC) ===")
comparison = compare_arima_models(y, max_p=3, max_d=2, max_q=3)
print(comparison.head(10))

# Best model
best_order = comparison.iloc[0]['order']
print(f"\nBest model: ARIMA{best_order}")
```

### 5.2 auto_arima (pmdarima)

```python
# Install pmdarima: pip install pmdarima
try:
    from pmdarima import auto_arima

    # Automatic ARIMA
    auto_model = auto_arima(
        y,
        start_p=0, max_p=3,
        start_q=0, max_q=3,
        d=None,  # Auto-determine
        seasonal=False,
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )

    print("\n=== auto_arima Results ===")
    print(auto_model.summary())

except ImportError:
    print("pmdarima is not installed.")
    print("Install: pip install pmdarima")
```

---

## 6. Model Diagnostics

### 6.1 Residual Analysis

```python
def diagnose_arima_model(result, title=''):
    """
    ARIMA model diagnostics
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    residuals = result.resid

    # Residual time series
    axes[0, 0].plot(residuals)
    axes[0, 0].axhline(0, color='r', linestyle='--')
    axes[0, 0].set_title('Residual Time Series')
    axes[0, 0].grid(True, alpha=0.3)

    # Residual histogram
    axes[0, 1].hist(residuals, bins=30, density=True, alpha=0.7)
    x = np.linspace(residuals.min(), residuals.max(), 100)
    from scipy.stats import norm
    axes[0, 1].plot(x, norm.pdf(x, residuals.mean(), residuals.std()), 'r-', lw=2)
    axes[0, 1].set_title('Residual Distribution')
    axes[0, 1].grid(True, alpha=0.3)

    # Residual ACF
    plot_acf(residuals, lags=20, ax=axes[1, 0], alpha=0.05)
    axes[1, 0].set_title('Residual ACF')

    # Q-Q Plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot')

    plt.suptitle(f'Model Diagnostics: {title}', y=1.02)
    plt.tight_layout()
    plt.show()

    # Ljung-Box test
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb_result = acorr_ljungbox(residuals, lags=[10, 20, 30], return_df=True)
    print("\n=== Ljung-Box Test ===")
    print(lb_result)
    print("\np-value > 0.05: Residuals are white noise (model adequate)")

# Run diagnostics
model_final = ARIMA(y, order=(1, 1, 1))
result_final = model_final.fit()
diagnose_arima_model(result_final, 'ARIMA(1,1,1)')
```

### 6.2 Standardized Residuals

```python
def standardized_residual_analysis(result):
    """Standardized residual analysis"""
    residuals = result.resid
    std_residuals = residuals / residuals.std()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Standardized residuals
    axes[0].plot(std_residuals)
    axes[0].axhline(0, color='k', linestyle='-')
    axes[0].axhline(2, color='r', linestyle='--', label='±2σ')
    axes[0].axhline(-2, color='r', linestyle='--')
    axes[0].axhline(3, color='orange', linestyle=':', label='±3σ')
    axes[0].axhline(-3, color='orange', linestyle=':')
    axes[0].set_title('Standardized Residuals')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Outlier proportion
    outliers_2sigma = np.sum(np.abs(std_residuals) > 2) / len(std_residuals)
    outliers_3sigma = np.sum(np.abs(std_residuals) > 3) / len(std_residuals)

    text = f"|z| > 2: {outliers_2sigma*100:.1f}% (expected: 4.6%)\n"
    text += f"|z| > 3: {outliers_3sigma*100:.1f}% (expected: 0.3%)"
    axes[1].text(0.1, 0.5, text, transform=axes[1].transAxes, fontsize=12,
                 verticalalignment='center')
    axes[1].axis('off')
    axes[1].set_title('Outlier Proportion')

    plt.tight_layout()
    plt.show()

standardized_residual_analysis(result_final)
```

---

## 7. Forecasting

### 7.1 Point Forecast and Prediction Interval

```python
# Forecasting
forecast_steps = 30

# Perform forecast
forecast = result_final.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int(alpha=0.05)

# Visualization
fig, ax = plt.subplots(figsize=(14, 6))

# Original data (last 100 points)
ax.plot(range(len(y)-100, len(y)), y[-100:], 'b-', label='Observed')

# Forecast
forecast_index = range(len(y), len(y) + forecast_steps)
ax.plot(forecast_index, forecast_mean, 'r-', label='Forecast')

# Confidence interval
ax.fill_between(forecast_index,
                forecast_ci.iloc[:, 0],
                forecast_ci.iloc[:, 1],
                color='r', alpha=0.2, label='95% Prediction Interval')

ax.axvline(len(y), color='k', linestyle='--', alpha=0.5)
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.set_title('ARIMA Forecast')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Forecast results
print("=== Forecast Results (first 10) ===")
forecast_df = pd.DataFrame({
    'Forecast': forecast_mean[:10],
    'Lower': forecast_ci.iloc[:10, 0],
    'Upper': forecast_ci.iloc[:10, 1]
})
print(forecast_df)
```

### 7.2 Forecast Performance Evaluation

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_forecast(y_train, y_test, order):
    """
    Evaluate forecast performance

    Parameters:
    -----------
    y_train : array
        Training data
    y_test : array
        Test data
    order : tuple
        ARIMA order (p, d, q)
    """
    # Fit model
    model = ARIMA(y_train, order=order)
    result = model.fit()

    # Forecast
    forecast = result.get_forecast(steps=len(y_test))
    y_pred = forecast.predicted_mean

    # Evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    print(f"=== Forecast Performance ({order}) ===")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")

    # Visualization
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(len(y_train)-50, len(y_train)), y_train[-50:], 'b-', label='Train')
    ax.plot(range(len(y_train), len(y_train)+len(y_test)), y_test, 'g-', label='Actual')
    ax.plot(range(len(y_train), len(y_train)+len(y_test)), y_pred, 'r--', label='Forecast')
    ax.axvline(len(y_train), color='k', linestyle=':', alpha=0.5)
    ax.legend()
    ax.set_title(f'Forecast vs Actual (ARIMA{order})')
    ax.grid(True, alpha=0.3)
    plt.show()

    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'mape': mape}

# Split data
train_size = int(len(y) * 0.8)
y_train = y[:train_size]
y_test = y[train_size:]

# Evaluate
evaluate_forecast(y_train, y_test, (1, 1, 1))
```

### 7.3 Rolling Forecast

```python
def rolling_forecast(y, order, initial_train_size, horizon=1):
    """
    Rolling window forecast

    Parameters:
    -----------
    y : array
        Full time series
    order : tuple
        ARIMA order
    initial_train_size : int
        Initial training data size
    horizon : int
        Forecast horizon
    """
    predictions = []
    actuals = []

    for i in range(initial_train_size, len(y) - horizon + 1):
        train = y[:i]
        actual = y[i:i+horizon]

        model = ARIMA(train, order=order)
        result = model.fit()

        pred = result.get_forecast(steps=horizon).predicted_mean.values
        predictions.append(pred[0])
        actuals.append(actual[0])

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Evaluation
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)

    print(f"=== Rolling Forecast (ARIMA{order}) ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")

    # Visualization
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(actuals, 'b-', label='Actual', alpha=0.7)
    ax.plot(predictions, 'r--', label='Forecast', alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Rolling 1-step Forecast')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()

    return predictions, actuals

predictions, actuals = rolling_forecast(y, (1, 1, 1), initial_train_size=200)
```

---

## 8. Real Data Application

### 8.1 Comprehensive Analysis Example

```python
def complete_arima_analysis(y, title=''):
    """
    Complete ARIMA modeling pipeline
    """
    print(f"{'='*60}")
    print(f"Time Series Analysis: {title}")
    print(f"{'='*60}")

    # 1. Visualization and stationarity test
    print("\n[Step 1] Stationarity Analysis")
    from statsmodels.tsa.stattools import adfuller, kpss

    adf_p = adfuller(y)[1]
    kpss_p = kpss(y, regression='c')[1]

    print(f"ADF p-value: {adf_p:.4f}")
    print(f"KPSS p-value: {kpss_p:.4f}")

    # Determine differencing
    d = 0
    y_diff = y.copy()
    while adf_p > 0.05 and d < 2:
        d += 1
        y_diff = np.diff(y_diff)
        adf_p = adfuller(y_diff)[1]
        print(f"After {d} difference(s) ADF p-value: {adf_p:.4f}")

    print(f"Determined d: {d}")

    # 2. ACF/PACF analysis
    print("\n[Step 2] ACF/PACF Analysis")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(y_diff if d > 0 else y, lags=20, ax=axes[0], alpha=0.05)
    plot_pacf(y_diff if d > 0 else y, lags=20, ax=axes[1], alpha=0.05, method='ywm')
    plt.suptitle(f'ACF/PACF (after {d} difference(s))')
    plt.tight_layout()
    plt.show()

    # 3. Model comparison
    print("\n[Step 3] Model Comparison (AIC)")
    results = []
    for p in range(4):
        for q in range(4):
            if p == 0 and q == 0:
                continue
            try:
                model = ARIMA(y, order=(p, d, q))
                fit = model.fit()
                results.append({'p': p, 'd': d, 'q': q, 'aic': fit.aic, 'bic': fit.bic})
            except:
                pass

    results_df = pd.DataFrame(results).sort_values('aic')
    print(results_df.head(5))

    # Best model
    best = results_df.iloc[0]
    best_order = (int(best['p']), int(best['d']), int(best['q']))
    print(f"\nBest model: ARIMA{best_order}")

    # 4. Final model fitting and diagnostics
    print(f"\n[Step 4] Final Model Fitting: ARIMA{best_order}")
    final_model = ARIMA(y, order=best_order)
    final_result = final_model.fit()
    print(final_result.summary())

    # Diagnostics
    diagnose_arima_model(final_result, f'ARIMA{best_order}')

    # 5. Forecasting
    print("\n[Step 5] Forecasting")
    forecast = final_result.get_forecast(steps=20)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(y[-100:], label='Observed')
    forecast_index = range(len(y), len(y) + 20)
    ax.plot(forecast_index, forecast.predicted_mean, 'r-', label='Forecast')
    ci = forecast.conf_int()
    ax.fill_between(forecast_index, ci.iloc[:, 0], ci.iloc[:, 1],
                    color='r', alpha=0.2, label='95% CI')
    ax.legend()
    ax.set_title(f'ARIMA{best_order} Forecast')
    ax.grid(True, alpha=0.3)
    plt.show()

    return final_result

# Example execution
np.random.seed(42)
ar = [1, -0.7, 0.2]
ma = [1, 0.5]
y_sample = arma_generate_sample(ar, ma, nsample=300, scale=1, burnin=100)
y_sample = np.cumsum(y_sample)  # Make non-stationary

result = complete_arima_analysis(y_sample, 'Sample Time Series')
```

---

## 9. Vector Autoregression (VAR)

All the models above -- AR, MA, ARMA, ARIMA -- handle a single time series. But in practice, economic and scientific variables move together: GDP and unemployment, temperature and pressure, stock prices across sectors. **Vector Autoregression (VAR)** extends the AR framework to multiple time series simultaneously, letting each variable depend on its own past *and* the past of every other variable in the system.

### 9.1 VAR(p) Model Formulation

A VAR(p) model for a K-dimensional time series **Y**ₜ = (Y₁ₜ, Y₂ₜ, ..., Yₖₜ)' is:

$$\mathbf{Y}_t = \mathbf{c} + \mathbf{A}_1 \mathbf{Y}_{t-1} + \mathbf{A}_2 \mathbf{Y}_{t-2} + \cdots + \mathbf{A}_p \mathbf{Y}_{t-p} + \mathbf{u}_t$$

Where:
- **c**: K x 1 vector of intercepts
- **A**ᵢ: K x K coefficient matrices (one per lag)
- **u**ₜ: K x 1 white noise vector with E(**u**ₜ) = **0** and covariance matrix Σ

For a bivariate VAR(1) with variables Y and X:

$$\begin{pmatrix} Y_t \\ X_t \end{pmatrix} = \begin{pmatrix} c_1 \\ c_2 \end{pmatrix} + \begin{pmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{pmatrix} \begin{pmatrix} Y_{t-1} \\ X_{t-1} \end{pmatrix} + \begin{pmatrix} u_{1t} \\ u_{2t} \end{pmatrix}$$

The off-diagonal elements (a₁₂, a₂₁) capture **cross-variable dynamics** -- this is what makes VAR powerful compared to running separate AR models for each variable.

### 9.2 Fitting a VAR Model in Python

```python
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

np.random.seed(42)

# --- Simulate a bivariate VAR(2) process ---
# Why simulate? It lets us verify that the estimator recovers the true parameters.
n = 500
K = 2
y = np.zeros((n, K))

# True coefficient matrices
A1 = np.array([[0.5, 0.1],
               [0.2, 0.3]])
A2 = np.array([[-0.2, 0.0],
               [0.1, -0.1]])

for t in range(2, n):
    y[t] = A1 @ y[t-1] + A2 @ y[t-2] + np.random.multivariate_normal(
        [0, 0], [[1.0, 0.3], [0.3, 0.8]]
    )

df_var = pd.DataFrame(y, columns=['GDP_growth', 'Inflation'])

# Check stationarity -- VAR requires all series to be stationary
for col in df_var.columns:
    adf_p = adfuller(df_var[col])[1]
    print(f"{col}: ADF p-value = {adf_p:.4f} ({'Stationary' if adf_p < 0.05 else 'Non-stationary'})")

# Fit VAR with automatic lag selection
# Why use information criteria? They balance fit against overfitting,
# which is especially important in VAR because the number of parameters
# grows as K^2 * p (quadratically in the number of variables).
model = VAR(df_var)
lag_order = model.select_order(maxlags=8)
print("\n=== Lag Order Selection ===")
print(lag_order.summary())

# Fit the selected model
result = model.fit(maxlags=8, ic='aic')
print(f"\nSelected lag order: {result.k_ar}")
print(result.summary())
```

### 9.3 Granger Causality

Granger causality asks: does knowing the past of variable X improve the forecast of variable Y beyond what Y's own past provides? It is a statistical test of *predictive precedence*, not true causation.

```python
from statsmodels.tsa.stattools import grangercausalitytests

# Test whether Inflation Granger-causes GDP_growth
# Why Granger causality matters: If X Granger-causes Y, including X's lags
# in the model significantly reduces Y's forecast error. This guides
# which variables belong in your VAR system.
print("=== Granger Causality: Inflation -> GDP_growth ===")
gc_result = grangercausalitytests(df_var[['GDP_growth', 'Inflation']], maxlag=4)

print("\n=== Granger Causality: GDP_growth -> Inflation ===")
gc_result = grangercausalitytests(df_var[['Inflation', 'GDP_growth']], maxlag=4)

# Interpret: if p-value < 0.05 at any lag, reject the null hypothesis
# that the second variable does NOT Granger-cause the first.
```

### 9.4 Impulse Response Functions (IRF)

An IRF traces how a one-unit shock to one variable propagates through the system over time. This is arguably the most important tool for interpreting VAR results -- it reveals the *dynamic multiplier effects* between variables.

```python
# Compute impulse response functions
# Why orthogonalize? Raw shocks are typically correlated across variables.
# Orthogonalized IRFs use Cholesky decomposition of the error covariance
# matrix to isolate the effect of a "pure" shock to one variable.
irf = result.irf(periods=20)

# Plot IRFs with confidence bands
fig = irf.plot(orth=True, impulse='GDP_growth', response='Inflation')
plt.suptitle('Orthogonalized IRF: Shock to GDP_growth -> Inflation')
plt.tight_layout()
plt.show()

fig = irf.plot(orth=True, impulse='Inflation', response='GDP_growth')
plt.suptitle('Orthogonalized IRF: Shock to Inflation -> GDP_growth')
plt.tight_layout()
plt.show()

# Cumulative IRF -- shows the long-run (accumulated) effect
fig = irf.plot_cum_effects(orth=True)
plt.suptitle('Cumulative Impulse Response Functions')
plt.tight_layout()
plt.show()
```

### 9.5 Forecast Error Variance Decomposition (FEVD)

FEVD answers: at forecast horizon h, what fraction of variable Y's forecast uncertainty is attributable to shocks from variable X? It quantifies the relative importance of each shock source over time.

```python
# Compute FEVD
fevd = result.fevd(periods=20)
print("=== Forecast Error Variance Decomposition ===")
print(fevd.summary())

# Plot FEVD
fig = fevd.plot()
plt.suptitle('Forecast Error Variance Decomposition')
plt.tight_layout()
plt.show()

# Why FEVD matters: In macroeconomics, FEVD helps answer questions like
# "What fraction of output volatility is driven by monetary policy shocks
# vs. supply shocks?" It moves beyond point estimates to show the
# structural importance of each variable in the system.
```

### 9.6 Forecasting with VAR

```python
# Forecast the next 10 periods
forecast_steps = 10
forecast = result.forecast(df_var.values[-result.k_ar:], steps=forecast_steps)

# Forecast with confidence intervals
forecast_ci = result.forecast_interval(
    df_var.values[-result.k_ar:], steps=forecast_steps, alpha=0.05
)
point_forecast, lower_ci, upper_ci = forecast_ci

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for i, col in enumerate(df_var.columns):
    # Last 50 observed values
    axes[i].plot(range(n-50, n), df_var[col].values[-50:], 'b-', label='Observed')

    # Forecasts
    fcast_idx = range(n, n + forecast_steps)
    axes[i].plot(fcast_idx, point_forecast[:, i], 'r-', label='Forecast')
    axes[i].fill_between(fcast_idx, lower_ci[:, i], upper_ci[:, i],
                         color='r', alpha=0.2, label='95% CI')
    axes[i].axvline(n, color='k', linestyle='--', alpha=0.5)
    axes[i].set_title(f'{col} Forecast')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 9.7 VAR vs VECM: When Series Are Cointegrated

VAR assumes all series are stationary. If your series are non-stationary but share a common stochastic trend (i.e., they are **cointegrated**), you should use a **Vector Error Correction Model (VECM)** instead. VECM is essentially a VAR in first differences plus error correction terms that capture the long-run equilibrium relationship.

| Aspect | VAR | VECM |
|--------|-----|------|
| Input series | Stationary (or differenced) | Non-stationary but cointegrated |
| Long-run relationship | Not modeled | Explicitly modeled via cointegrating equations |
| When to use | Variables move independently | Variables share a common long-run trend |
| Differencing | Difference first, then fit VAR | No need to difference -- VECM handles it |
| Test first | ADF/KPSS for stationarity | Johansen test for cointegration rank |

```python
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM

# Simulate cointegrated series (Y and X share a common trend)
np.random.seed(42)
n = 300
trend = np.cumsum(np.random.normal(0, 1, n))  # Common stochastic trend
x = trend + np.random.normal(0, 0.5, n)
y_coint = 0.8 * trend + np.random.normal(0, 0.5, n)  # Cointegrated with x

df_coint = pd.DataFrame({'Y': y_coint, 'X': x})

# Johansen cointegration test
# Why Johansen? Unlike Engle-Granger (limited to 2 variables),
# Johansen handles any number of variables and tests for the
# number of cointegrating relationships (rank).
johansen_result = coint_johansen(df_coint, det_order=0, k_ar_diff=2)
print("=== Johansen Cointegration Test ===")
print(f"Trace statistic: {johansen_result.lr1}")
print(f"Critical values (90%, 95%, 99%): {johansen_result.cvt}")

# Fit VECM if cointegrated
vecm = VECM(df_coint, k_ar_diff=2, coint_rank=1)
vecm_result = vecm.fit()
print("\n=== VECM Results ===")
print(vecm_result.summary())
```

**Rule of thumb**: Always test for cointegration before fitting a VAR on levels. If you find cointegration, VECM preserves valuable long-run information that differencing would destroy.

---

## 10. Practice Problems

### Problem 1: Model Identification
Identify the model corresponding to the following ACF/PACF patterns:
1. ACF: only lag 1 significant (0.4), then 0
   PACF: exponential decay

2. ACF: exponential decay (0.9, 0.81, 0.73, ...)
   PACF: only lag 1 significant (0.9)

### Problem 2: ARIMA Estimation
Generate and estimate data from the following model:
- ARIMA(2,1,1): φ₁=0.5, φ₂=0.2, θ₁=0.3

### Problem 3: Model Comparison
How should you decide when AIC and BIC select different models?

### Problem 4: Forecast Evaluation
Compare advantages and disadvantages of train-test split vs rolling forecast.

---

## 11. Key Summary

### ARIMA Model Framework

| Model | Form | Characteristics |
|------|------|------|
| AR(p) | Yₜ = φ₁Yₜ₋₁ + ... + φₚYₜ₋ₚ + εₜ | PACF cuts off after lag p |
| MA(q) | Yₜ = εₜ + θ₁εₜ₋₁ + ... + θₑεₜ₋ₑ | ACF cuts off after lag q |
| ARMA(p,q) | AR + MA | Both decay gradually |
| ARIMA(p,d,q) | ARMA after d differences | For non-stationary series |

### Box-Jenkins Checklist

1. [ ] Time series plot and stationarity test
2. [ ] Difference if necessary (determine d)
3. [ ] ACF/PACF analysis (candidate p, q)
4. [ ] Fit candidate models and compare AIC/BIC
5. [ ] Residual diagnostics (white noise, Ljung-Box)
6. [ ] Forecasting and performance evaluation

### Information Criteria

- **AIC**: Emphasizes forecast performance, may select larger models
- **BIC**: Emphasizes parsimony, prefers smaller models

### Next Chapter Preview

Chapter 12 **Multivariate Analysis** will cover:
- Principal Component Analysis (PCA)
- Factor Analysis
- Discriminant Analysis (LDA, QDA)
- Cluster Validation
