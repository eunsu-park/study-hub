"""
Bayesian Time Series Examples
- Kalman filter, seasonal decomposition, forecasting
"""
import numpy as np


class KalmanFilter:
    """Local level Kalman filter."""
    def __init__(self, sigma_state, sigma_obs, mu_0=0, P_0=1.0):
        self.Q, self.R = sigma_state**2, sigma_obs**2
        self.mu, self.P = mu_0, P_0

    def filter(self, obs):
        means, variances = [], []
        for y in obs:
            mu_pred, P_pred = self.mu, self.P + self.Q
            K = P_pred / (P_pred + self.R)
            self.mu = mu_pred + K * (y - mu_pred)
            self.P = (1 - K) * P_pred
            means.append(self.mu)
            variances.append(self.P)
        return np.array(means), np.array(variances)


def fourier_features(t, period, n_fourier):
    feats = []
    for i in range(1, n_fourier + 1):
        feats.extend([np.sin(2*np.pi*i*t/period), np.cos(2*np.pi*i*t/period)])
    return np.column_stack(feats)


if __name__ == "__main__":
    np.random.seed(42)
    T = 200
    t = np.arange(T)
    trend = 0.05 * t
    seasonal = 3 * np.sin(2 * np.pi * t / 7)
    y = 50 + trend + seasonal + np.random.normal(0, 1, T)

    # Kalman filter
    kf = KalmanFilter(sigma_state=0.5, sigma_obs=2.0, mu_0=y[0])
    mu_f, var_f = kf.filter(y)
    print(f"Kalman filter: final estimate={mu_f[-1]:.2f}, true≈{50+trend[-1]+seasonal[-1]:.2f}")

    # Fourier seasonal regression
    X = fourier_features(t, 7, 3)
    X_full = np.column_stack([np.ones(T), t, X])
    beta = np.linalg.lstsq(X_full, y, rcond=None)[0]
    print(f"Seasonal regression: intercept={beta[0]:.2f}, trend={beta[1]:.4f}")
    print(f"  (true: intercept=50, trend=0.05)")
