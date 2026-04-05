# 10. Bayesian Time Series

[Previous: Gaussian Processes](./09_Gaussian_Processes.md) | [Next: Bayesian Optimization](./11_Bayesian_Optimization.md)

---

> **Framework Note**: This lesson uses PyMC 5.x and Prophet for Bayesian time series modeling.
>
> Installation: `pip install pymc arviz numpy matplotlib pandas prophet`

## Learning Objectives

- Build structural time series models with Bayesian priors
- Understand state-space models and the Kalman filter
- Use Facebook Prophet for decomposable time series
- Model trend, seasonality, and changepoints probabilistically
- Quantify forecast uncertainty with posterior predictive distributions

---

## 1. Bayesian Approach to Time Series

Traditional time series methods (ARIMA) treat parameters as fixed. The Bayesian approach quantifies uncertainty in all components: trend, seasonality, noise, and changepoints.

### 1.1 Structural Time Series Decomposition

$$y_t = \mu_t + s_t + \epsilon_t$$

Where $\mu_t$ is trend, $s_t$ is seasonality, and $\epsilon_t$ is observation noise.

```python
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd

# Generate synthetic time series
np.random.seed(42)
T = 365
t = np.arange(T)
trend = 0.05 * t
seasonal = 3 * np.sin(2 * np.pi * t / 7) + 1.5 * np.sin(2 * np.pi * t / 365)
noise = np.random.normal(0, 1, T)
y = 50 + trend + seasonal + noise

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
axes[0].plot(t, y, 'k-', alpha=0.7)
axes[0].set_title("Observed")
axes[1].plot(t, 50 + trend, 'b-')
axes[1].set_title("Trend")
axes[2].plot(t, seasonal, 'g-')
axes[2].set_title("Seasonality")
axes[3].plot(t, noise, 'r-', alpha=0.5)
axes[3].set_title("Noise")
plt.tight_layout()
plt.savefig("time_series_decomposition.png", dpi=100)
plt.show()
```

---

## 2. Local Linear Trend Model

```python
with pm.Model() as llt_model:
    # Trend parameters
    sigma_level = pm.HalfNormal("sigma_level", sigma=1)
    sigma_slope = pm.HalfNormal("sigma_slope", sigma=0.1)
    sigma_obs = pm.HalfNormal("sigma_obs", sigma=2)

    # Initial level and slope
    level_init = pm.Normal("level_init", mu=y[0], sigma=10)
    slope_init = pm.Normal("slope_init", mu=0, sigma=1)

    # Random walk for level and slope
    level_innovations = pm.Normal("level_innov", mu=0, sigma=sigma_level, shape=T-1)
    slope_innovations = pm.Normal("slope_innov", mu=0, sigma=sigma_slope, shape=T-1)

    # Build level and slope sequences
    levels = pm.math.concatenate([[level_init],
        level_init + slope_init + pm.math.cumsum(level_innovations + slope_innovations)])
    # Simplified: use scan or manual recursion in practice

    # Observation model
    y_obs = pm.Normal("y_obs", mu=levels[:T], sigma=sigma_obs, observed=y)

    trace_llt = pm.sample(2000, tune=1000, chains=4, random_seed=42)
```

---

## 3. Bayesian Seasonal Model

```python
# Fourier series for seasonality
def fourier_features(t, period, n_fourier):
    """Create Fourier features for seasonal patterns."""
    features = []
    for i in range(1, n_fourier + 1):
        features.append(np.sin(2 * np.pi * i * t / period))
        features.append(np.cos(2 * np.pi * i * t / period))
    return np.column_stack(features)

# Weekly and yearly seasonality
X_weekly = fourier_features(t, period=7, n_fourier=3)
X_yearly = fourier_features(t, period=365.25, n_fourier=5)
X_seasonal = np.column_stack([X_weekly, X_yearly])

with pm.Model() as seasonal_model:
    # Trend
    intercept = pm.Normal("intercept", mu=50, sigma=20)
    slope = pm.Normal("slope", mu=0, sigma=1)

    # Seasonal coefficients with shrinkage
    sigma_seasonal = pm.HalfNormal("sigma_seasonal", sigma=2)
    beta_seasonal = pm.Normal("beta_seasonal", mu=0, sigma=sigma_seasonal,
                               shape=X_seasonal.shape[1])

    # Observation noise
    sigma = pm.HalfNormal("sigma", sigma=5)

    mu = intercept + slope * t + pm.math.dot(X_seasonal, beta_seasonal)
    y_obs = pm.Normal("y", mu=mu, sigma=sigma, observed=y)

    trace_seasonal = pm.sample(3000, tune=1000, chains=4, random_seed=42)

summary = az.summary(trace_seasonal, var_names=["intercept", "slope", "sigma"])
print(summary)
```

---

## 4. Changepoint Detection

```python
# Bayesian changepoint model
np.random.seed(42)
T_cp = 200
y_cp = np.concatenate([
    np.random.normal(5, 1, 80),
    np.random.normal(8, 1.5, 60),
    np.random.normal(3, 0.8, 60),
])

with pm.Model() as changepoint_model:
    # Two changepoints
    tau1 = pm.DiscreteUniform("tau1", lower=20, upper=120)
    tau2 = pm.DiscreteUniform("tau2", lower=tau1 + 10, upper=180)

    # Segment means
    mu1 = pm.Normal("mu1", mu=5, sigma=5)
    mu2 = pm.Normal("mu2", mu=5, sigma=5)
    mu3 = pm.Normal("mu3", mu=5, sigma=5)

    sigma = pm.HalfNormal("sigma", sigma=3)

    # Build piecewise mean
    idx = np.arange(T_cp)
    mu = pm.math.switch(idx < tau1, mu1,
         pm.math.switch(idx < tau2, mu2, mu3))

    y_obs = pm.Normal("y", mu=mu, sigma=sigma, observed=y_cp)

    trace_cp = pm.sample(5000, tune=2000, chains=4, random_seed=42,
                         step=pm.Metropolis())  # discrete params need Metropolis

print(az.summary(trace_cp, var_names=["tau1", "tau2", "mu1", "mu2", "mu3"]))
```

---

## 5. Facebook Prophet

Prophet implements a decomposable time series model with automatic changepoint detection.

```python
from prophet import Prophet

# Prepare data in Prophet format
df = pd.DataFrame({
    'ds': pd.date_range('2023-01-01', periods=T, freq='D'),
    'y': y,
})

# Fit Prophet model
m = Prophet(
    changepoint_prior_scale=0.05,   # flexibility of trend changes
    seasonality_prior_scale=10.0,   # strength of seasonality
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    interval_width=0.95,            # uncertainty interval width
    mcmc_samples=300,               # use MCMC for full uncertainty
)
m.fit(df)

# Forecast
future = m.make_future_dataframe(periods=60)
forecast = m.predict(future)

# Plot
fig = m.plot(forecast)
plt.title("Prophet Forecast with Uncertainty")
plt.savefig("prophet_forecast.png", dpi=100)
plt.show()

# Component decomposition
fig2 = m.plot_components(forecast)
plt.savefig("prophet_components.png", dpi=100)
plt.show()
```

---

## 6. State-Space Models

### 6.1 Kalman Filter

```python
class KalmanFilter:
    """Simple Kalman filter for local level model."""

    def __init__(self, sigma_state, sigma_obs, mu_0=0, P_0=1.0):
        self.Q = sigma_state**2
        self.R = sigma_obs**2
        self.mu = mu_0
        self.P = P_0

    def filter(self, observations):
        """Run Kalman filter forward pass."""
        filtered_means = []
        filtered_vars = []

        for y in observations:
            # Predict
            mu_pred = self.mu
            P_pred = self.P + self.Q

            # Update
            K = P_pred / (P_pred + self.R)  # Kalman gain
            self.mu = mu_pred + K * (y - mu_pred)
            self.P = (1 - K) * P_pred

            filtered_means.append(self.mu)
            filtered_vars.append(self.P)

        return np.array(filtered_means), np.array(filtered_vars)


kf = KalmanFilter(sigma_state=0.5, sigma_obs=1.5)
filtered_mu, filtered_var = kf.filter(y[:100])

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(y[:100], 'k.', alpha=0.5, label='Observations')
ax.plot(filtered_mu, 'b-', linewidth=2, label='Filtered estimate')
ax.fill_between(range(100),
                filtered_mu - 2*np.sqrt(filtered_var),
                filtered_mu + 2*np.sqrt(filtered_var),
                alpha=0.2, label='±2σ')
ax.legend()
ax.set_title("Kalman Filter: Local Level Model")
plt.tight_layout()
plt.savefig("kalman_filter.png", dpi=100)
plt.show()
```

### 6.2 Bayesian State-Space in PyMC

```python
with pm.Model() as ssm_model:
    sigma_state = pm.HalfNormal("sigma_state", sigma=1)
    sigma_obs = pm.HalfNormal("sigma_obs", sigma=2)

    # Latent states (random walk)
    states = pm.GaussianRandomWalk("states", sigma=sigma_state, shape=100,
                                    init_dist=pm.Normal.dist(mu=y[0], sigma=5))

    y_obs = pm.Normal("y", mu=states, sigma=sigma_obs, observed=y[:100])

    trace_ssm = pm.sample(2000, tune=1000, chains=4, random_seed=42)

states_mean = trace_ssm.posterior["states"].values.mean(axis=(0, 1))
states_std = trace_ssm.posterior["states"].values.std(axis=(0, 1))
```

---

## 7. Autoregressive Models

```python
# Bayesian AR(p) model
p_order = 3

with pm.Model() as ar_model:
    # AR coefficients
    phi = pm.Normal("phi", mu=0, sigma=0.5, shape=p_order)
    sigma = pm.HalfNormal("sigma", sigma=2)
    mu_const = pm.Normal("mu", mu=0, sigma=10)

    # Build AR likelihood
    y_ar = y[:200]
    for t_step in range(p_order, len(y_ar)):
        ar_mean = mu_const + sum(phi[j] * y_ar[t_step - j - 1] for j in range(p_order))
        pm.Normal(f"y_{t_step}", mu=ar_mean, sigma=sigma, observed=y_ar[t_step])

    trace_ar = pm.sample(2000, tune=1000, chains=4, random_seed=42)

print(az.summary(trace_ar, var_names=["phi", "mu", "sigma"]))
```

---

## 8. Forecasting with Uncertainty

```python
def bayesian_forecast(trace, y_history, n_forecast=30):
    """Generate forecast with full posterior uncertainty."""
    n_samples = 1000
    intercept = trace.posterior["intercept"].values.flatten()[:n_samples]
    slope = trace.posterior["slope"].values.flatten()[:n_samples]
    sigma = trace.posterior["sigma"].values.flatten()[:n_samples]

    T_hist = len(y_history)
    t_future = np.arange(T_hist, T_hist + n_forecast)

    forecasts = np.zeros((n_samples, n_forecast))
    for i in range(n_samples):
        mu_f = intercept[i] + slope[i] * t_future
        forecasts[i] = mu_f + np.random.normal(0, sigma[i], n_forecast)

    return forecasts

# Plot forecast fan chart
forecasts = bayesian_forecast(trace_seasonal, y, n_forecast=60)
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(t, y, 'k-', alpha=0.7, label='Observed')
t_future = np.arange(T, T + 60)
for q in [5, 25, 50, 75, 95]:
    ax.fill_between(t_future,
                    np.percentile(forecasts, 50 - (q-50) if q < 50 else 0, axis=0),
                    np.percentile(forecasts, q, axis=0),
                    alpha=0.15, color='blue')
ax.plot(t_future, forecasts.mean(axis=0), 'b-', linewidth=2, label='Forecast mean')
ax.legend()
ax.set_title("Bayesian Forecast with Uncertainty Fan Chart")
plt.tight_layout()
plt.savefig("forecast_fan.png", dpi=100)
plt.show()
```

---

## Summary

| Model | Components | Inference | Best For |
|-------|-----------|-----------|----------|
| Local Linear Trend | Level + slope | MCMC/Kalman | Short-term, non-seasonal |
| Structural TS | Trend + seasonality | MCMC | Decomposable series |
| Prophet | Trend + holidays + seasonality | MAP/MCMC | Business forecasting |
| State-Space | Latent states | Kalman/MCMC | Real-time filtering |
| Bayesian AR | Autoregressive | MCMC | Stationary series |

---

## References

1. Durbin, J. & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods*. Oxford.
2. Harvey, A. C. (1990). *Forecasting, Structural Time Series Models and the Kalman Filter*. Cambridge.
3. Taylor, S. J. & Letham, B. (2018). "Forecasting at Scale." *The American Statistician*.
4. Hyndman, R. J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*. OTexts.

---

[Previous: Gaussian Processes](./09_Gaussian_Processes.md) | [Next: Bayesian Optimization →](./11_Bayesian_Optimization.md)
