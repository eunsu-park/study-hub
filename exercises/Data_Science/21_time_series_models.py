"""
Exercises for Lesson 21: Time Series Models
Topic: Data_Science

Solutions to practice problems from the lesson.
"""
import numpy as np


# === Exercise 1: AR(1) Model Simulation and Estimation ===
# Problem: Simulate an AR(1) process with phi=0.8, then estimate phi
#   using the Yule-Walker equation and least squares. Compare the
#   estimates and verify stationarity.
def exercise_1():
    """Solution for AR(1) simulation and parameter estimation.

    AR(1): Y_t = c + phi * Y_{t-1} + epsilon_t

    Yule-Walker estimate: phi_hat = rho(1) (sample autocorrelation at lag 1)
    OLS estimate: regress Y_t on Y_{t-1}

    Stationarity requires |phi| < 1.
    """
    np.random.seed(42)
    n = 500
    phi_true = 0.8
    c_true = 2.0
    sigma = 1.0

    # Simulate AR(1)
    y = np.zeros(n)
    # Stationary initial value: E[Y] = c / (1 - phi)
    y[0] = c_true / (1 - phi_true) + np.random.normal(0, sigma / np.sqrt(1 - phi_true**2))

    for t in range(1, n):
        y[t] = c_true + phi_true * y[t - 1] + np.random.normal(0, sigma)

    print(f"Simulated AR(1): Y_t = {c_true} + {phi_true}*Y_{{t-1}} + e_t")
    print(f"  n = {n}")
    print(f"  Theoretical mean: {c_true / (1 - phi_true):.4f}")
    print(f"  Theoretical variance: {sigma**2 / (1 - phi_true**2):.4f}")
    print(f"  Sample mean: {y.mean():.4f}")
    print(f"  Sample variance: {y.var(ddof=1):.4f}")

    # Yule-Walker estimate: phi_hat = rho(1)
    y_centered = y - y.mean()
    rho_1 = np.sum(y_centered[1:] * y_centered[:-1]) / np.sum(y_centered**2)
    c_yw = y.mean() * (1 - rho_1)

    print(f"\nYule-Walker estimates:")
    print(f"  phi_hat = rho(1) = {rho_1:.4f}  (true: {phi_true})")
    print(f"  c_hat = mean*(1-phi) = {c_yw:.4f}  (true: {c_true})")

    # OLS estimate: regress Y_t on Y_{t-1}
    Y = y[1:]           # dependent variable
    X = y[:-1]           # lagged variable

    # Add intercept: X_aug = [1, Y_{t-1}]
    X_aug = np.column_stack([np.ones(len(X)), X])
    # beta = (X'X)^{-1} X'Y
    XtX = X_aug.T @ X_aug
    XtY = X_aug.T @ Y
    beta_ols = np.linalg.solve(XtX, XtY)

    print(f"\nOLS estimates:")
    print(f"  c_hat   = {beta_ols[0]:.4f}  (true: {c_true})")
    print(f"  phi_hat = {beta_ols[1]:.4f}  (true: {phi_true})")

    # Residual analysis
    residuals = Y - X_aug @ beta_ols
    resid_std = residuals.std(ddof=2)
    print(f"\nResidual analysis:")
    print(f"  Residual std: {resid_std:.4f} (true sigma: {sigma})")
    resid_acf1 = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
    print(f"  Residual lag-1 ACF: {resid_acf1:.4f} (should be ~0)")

    # Stationarity check
    print(f"\nStationarity: |phi_hat| = {abs(beta_ols[1]):.4f} < 1? "
          f"{'Yes (stationary)' if abs(beta_ols[1]) < 1 else 'No (non-stationary)'}")


# === Exercise 2: MA(1) Model Simulation and ACF ===
# Problem: Simulate an MA(1) process with theta=0.6 and verify that the
#   ACF cuts off after lag 1. Estimate theta from the sample ACF.
def exercise_2():
    """Solution for MA(1) simulation and identification.

    MA(1): Y_t = mu + epsilon_t + theta * epsilon_{t-1}

    Theoretical ACF:
        rho(0) = 1
        rho(1) = theta / (1 + theta^2)
        rho(k) = 0 for k >= 2

    The ACF cutoff after lag 1 is the identifying signature of MA(1).
    """
    np.random.seed(42)
    n = 1000
    theta_true = 0.6
    mu = 5.0
    sigma = 1.0

    # Simulate MA(1)
    epsilon = np.random.normal(0, sigma, n + 1)
    y = np.zeros(n)
    for t in range(n):
        y[t] = mu + epsilon[t + 1] + theta_true * epsilon[t]

    print(f"Simulated MA(1): Y_t = {mu} + e_t + {theta_true}*e_{{t-1}}, n={n}")
    print(f"  Theoretical: mean={mu:.4f}, var={sigma**2 * (1 + theta_true**2):.4f}")
    print(f"  Sample:      mean={y.mean():.4f}, var={y.var(ddof=1):.4f}")

    # Compute sample ACF for lags 0-5
    max_lag = 5
    y_c = y - y.mean()
    gamma_0 = np.var(y, ddof=0)
    sig_bound = 1.96 / np.sqrt(n)

    print(f"\nACF comparison (bound=+/-{sig_bound:.4f}):")
    print(f"  {'Lag':>4s}  {'Sample':>8s}  {'Theory':>8s}")
    print(f"  {'-'*24}")

    for k in range(max_lag + 1):
        if k == 0:
            acf_sample, acf_theory = 1.0, 1.0
        else:
            acf_sample = np.mean(y_c[k:] * y_c[:-k]) / gamma_0
            acf_theory = theta_true / (1 + theta_true**2) if k == 1 else 0.0
        print(f"  {k:4d}  {acf_sample:8.4f}  {acf_theory:8.4f}")

    # Estimate theta from rho(1)
    # rho(1) = theta / (1 + theta^2)
    # Solving: theta^2 * rho(1) - theta + rho(1) = 0
    cov_1 = np.mean(y_c[1:] * y_c[:-1])
    rho_1 = cov_1 / gamma_0

    # Quadratic formula: theta = (1 +/- sqrt(1 - 4*rho_1^2)) / (2*rho_1)
    discriminant = 1 - 4 * rho_1**2
    if discriminant >= 0:
        theta_est_1 = (1 - np.sqrt(discriminant)) / (2 * rho_1)
        theta_est_2 = (1 + np.sqrt(discriminant)) / (2 * rho_1)
        # Choose the invertible solution: |theta| < 1
        theta_est = theta_est_1 if abs(theta_est_1) < abs(theta_est_2) else theta_est_2
        print(f"\nTheta estimation from rho(1) = {rho_1:.4f}:")
        print(f"  Solutions: {theta_est_1:.4f}, {theta_est_2:.4f}")
        print(f"  Invertible estimate: {theta_est:.4f} (true: {theta_true})")
    else:
        print(f"\nNo real solution (discriminant < 0). rho(1) = {rho_1:.4f}")


# === Exercise 3: ARIMA Model Identification ===
# Problem: Given a non-stationary series (random walk with drift),
#   apply differencing to achieve stationarity and identify the appropriate
#   ARIMA(p,d,q) order from the ACF/PACF of the differenced series.
def exercise_3():
    """Solution for ARIMA model identification via Box-Jenkins methodology.

    Steps:
    1. Check stationarity of the original series
    2. Difference until stationary (determine d)
    3. Examine ACF/PACF of differenced series to identify p and q
    4. Verify that the identified model makes sense
    """
    np.random.seed(42)
    n = 300

    # Generate ARIMA(1,1,0): difference is AR(1)
    # First generate AR(1) innovations
    phi = 0.6
    drift = 0.5
    ar_innovations = np.zeros(n)
    for t in range(1, n):
        ar_innovations[t] = drift + phi * ar_innovations[t - 1] + np.random.normal(0, 1)

    # Integrate (cumsum) to get non-stationary series
    y = 100 + np.cumsum(ar_innovations)

    print("ARIMA Model Identification")
    print(f"True model: ARIMA(1,1,0) with phi={phi}, drift={drift}")
    print(f"  n = {n}")
    print(f"  y[0]={y[0]:.2f}, y[-1]={y[-1]:.2f}")

    # Step 1: Check stationarity via segment comparison
    n_seg = 4
    seg_len = n // n_seg
    print(f"\nStep 1: Stationarity check (original series)")
    print(f"  {'Segment':>8s}  {'Mean':>10s}  {'Std':>8s}")
    print(f"  {'-'*28}")
    for i in range(n_seg):
        seg = y[i * seg_len:(i + 1) * seg_len]
        print(f"  {i+1:8d}  {seg.mean():10.2f}  {seg.std():8.2f}")
    print(f"  -> Means change substantially: non-stationary (need differencing)")

    # Step 2: First differencing
    dy = np.diff(y)  # d=1
    print(f"\nStep 2: First differencing (d=1)")
    print(f"  {'Segment':>8s}  {'Mean':>10s}  {'Std':>8s}")
    print(f"  {'-'*28}")
    seg_len_d = len(dy) // n_seg
    for i in range(n_seg):
        seg = dy[i * seg_len_d:(i + 1) * seg_len_d]
        print(f"  {i+1:8d}  {seg.mean():10.4f}  {seg.std():8.4f}")
    print(f"  -> Means and stds are roughly constant: stationary")

    # Step 3: ACF/PACF of differenced series
    dy_c = dy - dy.mean()
    gamma_0 = np.var(dy, ddof=0)

    max_lag = 5
    acf_vals = np.zeros(max_lag + 1)
    for k in range(max_lag + 1):
        if k == 0:
            acf_vals[k] = 1.0
        else:
            acf_vals[k] = np.mean(dy_c[k:] * dy_c[:-k]) / gamma_0

    sig = 1.96 / np.sqrt(len(dy))
    print(f"\nStep 3: ACF/PACF of differenced series (bound=+/-{sig:.4f})")
    print(f"  {'Lag':>4s}  {'ACF':>8s}  {'Sig':>5s}")
    print(f"  {'-'*20}")
    for k in range(max_lag + 1):
        is_sig = "***" if abs(acf_vals[k]) > sig and k > 0 else ""
        print(f"  {k:4d}  {acf_vals[k]:8.4f}  {is_sig:>5s}")

    # PACF at lag 1 via Yule-Walker: for identifying AR order
    pacf_1 = acf_vals[1]

    print(f"\n  PACF(1) = {pacf_1:.4f} (significant)")
    print(f"  ACF decays gradually, PACF cuts off after lag 1")
    print(f"  -> Differenced series is AR(1) -> original is ARIMA(1,1,0)")
    print(f"  Estimated phi: {pacf_1:.4f} (true: {phi})")


# === Exercise 4: Forecast Evaluation ===
# Problem: Split an AR(1) series into train and test sets, generate
#   one-step-ahead forecasts, and compute RMSE and MAE.
def exercise_4():
    """Solution for time series forecast evaluation.

    One-step-ahead forecast for AR(1):
        Y_hat_{t+1} = c + phi * Y_t

    We evaluate using:
    - RMSE (Root Mean Squared Error): penalizes large errors
    - MAE (Mean Absolute Error): robust to outliers
    - MAPE (Mean Absolute Percentage Error): scale-independent
    """
    np.random.seed(42)
    n = 300
    phi = 0.7
    c = 3.0
    sigma = 1.0

    # Generate AR(1)
    y = np.zeros(n)
    y[0] = c / (1 - phi)
    for t in range(1, n):
        y[t] = c + phi * y[t - 1] + np.random.normal(0, sigma)

    # Train/test split
    train_size = 250
    y_train = y[:train_size]
    y_test = y[train_size:]
    n_test = len(y_test)

    print(f"AR(1) Forecast Evaluation")
    print(f"  True parameters: c={c}, phi={phi}, sigma={sigma}")
    print(f"  Train size: {train_size}, Test size: {n_test}")

    # Estimate parameters from training data (OLS)
    Y_tr = y_train[1:]
    X_tr = np.column_stack([np.ones(train_size - 1), y_train[:-1]])
    beta = np.linalg.solve(X_tr.T @ X_tr, X_tr.T @ Y_tr)
    c_hat, phi_hat = beta[0], beta[1]

    print(f"\nEstimated parameters:")
    print(f"  c_hat:   {c_hat:.4f} (true: {c})")
    print(f"  phi_hat: {phi_hat:.4f} (true: {phi})")

    # One-step-ahead forecasts on test set
    forecasts = np.zeros(n_test)
    for i in range(n_test):
        if i == 0:
            y_prev = y_train[-1]
        else:
            y_prev = y_test[i - 1]  # Use actual value (rolling forecast)
        forecasts[i] = c_hat + phi_hat * y_prev

    # Compute error metrics
    errors = y_test - forecasts
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(np.abs(errors))
    mape = np.mean(np.abs(errors / y_test)) * 100

    print(f"\nForecast Evaluation Metrics:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  Theoretical RMSE (sigma): {sigma:.4f}")

    # Compare with naive forecast (Y_hat = Y_{t-1})
    naive_fc = np.concatenate([[y_train[-1]], y_test[:-1]])
    naive_rmse = np.sqrt(np.mean((y_test - naive_fc)**2))

    print(f"\nNaive forecast RMSE: {naive_rmse:.4f}")
    print(f"AR(1) vs Naive RMSE improvement: {(1 - rmse/naive_rmse)*100:.1f}%")

    # Residual diagnostics
    resid_acf1 = np.corrcoef(errors[:-1], errors[1:])[0, 1]
    print(f"\nResidual mean: {errors.mean():.4f}, lag-1 ACF: {resid_acf1:.4f}")


# === Exercise 5: Information Criteria for Model Selection ===
# Problem: Compute AIC and BIC for AR models of different orders
#   fitted to the same data, and select the best order.
def exercise_5():
    """Solution for model selection using AIC and BIC.

    AIC = -2 * log_likelihood + 2 * k
    BIC = -2 * log_likelihood + k * log(n)

    where k = number of parameters, n = number of observations.
    Lower values indicate better model fit (balancing complexity).
    BIC penalizes complexity more heavily than AIC for n >= 8.
    """
    np.random.seed(42)
    n = 300

    # True model: AR(2) with phi1=0.5, phi2=0.3
    phi1_true = 0.5
    phi2_true = 0.3
    sigma_true = 1.0

    y = np.zeros(n)
    for t in range(2, n):
        y[t] = phi1_true * y[t-1] + phi2_true * y[t-2] + np.random.normal(0, sigma_true)

    print(f"True model: AR(2) with phi1={phi1_true}, phi2={phi2_true}")
    print(f"  n = {n}\n")

    def fit_ar(y, p):
        """Fit AR(p) model via OLS and return log-likelihood, AIC, BIC."""
        n_total = len(y)
        n_eff = n_total - p  # effective sample size

        # Build regression: Y_t on [1, Y_{t-1}, ..., Y_{t-p}]
        Y = y[p:]
        X_cols = [np.ones(n_eff)]
        for lag in range(1, p + 1):
            X_cols.append(y[p - lag:n_total - lag])
        X = np.column_stack(X_cols)

        # OLS
        beta = np.linalg.solve(X.T @ X, X.T @ Y)
        residuals = Y - X @ beta
        sigma2 = np.sum(residuals**2) / n_eff

        # Log-likelihood (Gaussian)
        log_lik = -0.5 * n_eff * (np.log(2 * np.pi * sigma2) + 1)

        # Number of parameters: p AR coefficients + intercept + sigma
        k = p + 2
        aic = -2 * log_lik + 2 * k
        bic = -2 * log_lik + k * np.log(n_eff)

        return beta, sigma2, log_lik, aic, bic

    print(f"{'Order':>6s}  {'AIC':>10s}  {'BIC':>10s}  {'sigma^2':>10s}")
    print("-" * 40)

    results = []
    for p in range(1, 6):
        beta, sigma2, ll, aic, bic = fit_ar(y, p)
        results.append((p, ll, aic, bic, sigma2, beta))
        print(f"{p:6d}  {aic:10.2f}  {bic:10.2f}  {sigma2:10.4f}")

    best_aic = min(results, key=lambda r: r[2])
    best_bic = min(results, key=lambda r: r[3])
    print(f"\nBest by AIC: AR({best_aic[0]}), Best by BIC: AR({best_bic[0]})")

    best = best_bic
    beta = best[5]
    print(f"\nAR({best[0]}) estimates: intercept={beta[0]:.4f}, "
          f"phi_1={beta[1]:.4f} (true {phi1_true}), "
          f"phi_2={beta[2]:.4f} (true {phi2_true})")


if __name__ == "__main__":
    print("=== Exercise 1: AR(1) Model Simulation and Estimation ===")
    exercise_1()
    print("\n=== Exercise 2: MA(1) Model Simulation and ACF ===")
    exercise_2()
    print("\n=== Exercise 3: ARIMA Model Identification ===")
    exercise_3()
    print("\n=== Exercise 4: Forecast Evaluation ===")
    exercise_4()
    print("\n=== Exercise 5: Information Criteria for Model Selection ===")
    exercise_5()
    print("\nAll exercises completed!")
