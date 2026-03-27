"""
Exercises for Lesson 20: Time Series Analysis Fundamentals
Topic: Data_Science

Solutions to practice problems from the lesson.
"""
import numpy as np


# === Exercise 1: Time Series Components ===
# Problem: Generate a synthetic time series with trend, seasonality, and
#   noise components. Then decompose it by subtracting the known components
#   and verify the residual is approximately white noise.
def exercise_1():
    """Solution generating and decomposing a time series into components.

    A time series y(t) can be modeled as:
        Additive:       y(t) = Trend(t) + Seasonal(t) + Noise(t)
        Multiplicative: y(t) = Trend(t) * Seasonal(t) * Noise(t)

    We generate an additive series with known components, then recover
    them to verify correctness.
    """
    np.random.seed(42)
    n = 365 * 2  # 2 years of daily data
    t = np.arange(n)

    # Components
    trend = 50 + 0.05 * t                            # linear trend
    seasonal = 10 * np.sin(2 * np.pi * t / 365)      # annual cycle
    noise = np.random.normal(0, 2, n)                 # Gaussian noise

    # Combined series
    y = trend + seasonal + noise

    print("Synthetic Time Series Components:")
    print(f"  Length: {n} observations (2 years daily)")
    print(f"  Trend: 50 + 0.05*t (linear, slope=0.05 per day)")
    print(f"  Seasonal: amplitude=10, period=365 days")
    print(f"  Noise: N(0, 2)")

    # Basic statistics
    print(f"\nOverall statistics:")
    print(f"  Mean:  {y.mean():.2f}")
    print(f"  Std:   {y.std():.2f}")
    print(f"  Min:   {y.min():.2f}")
    print(f"  Max:   {y.max():.2f}")

    # Decompose by subtracting known trend and seasonal
    detrended = y - trend
    residual = detrended - seasonal

    print(f"\nDecomposition verification:")
    print(f"  Residual mean: {residual.mean():.4f} (should be ~0)")
    print(f"  Residual std:  {residual.std():.4f} (should be ~2)")

    # Check residual autocorrelation at lag 1
    r1 = np.corrcoef(residual[:-1], residual[1:])[0, 1]
    print(f"  Residual lag-1 autocorrelation: {r1:.4f} (should be ~0)")
    print(f"  Residual is {'likely white noise' if abs(r1) < 0.1 else 'autocorrelated'}")

    # Moving average as an alternative trend estimate
    window = 365
    # Pad to handle edges: we only compute where full window fits
    n_valid = n - window + 1
    ma_trend = np.convolve(y, np.ones(window) / window, mode='valid')

    print(f"\nMoving average trend estimation (window={window}):")
    print(f"  First MA value:  {ma_trend[0]:.2f}  (true trend at midpoint: {trend[window//2]:.2f})")
    print(f"  Last MA value:   {ma_trend[-1]:.2f}  (true trend at midpoint: {trend[n - window//2 - 1]:.2f})")
    print(f"  MA captures the linear trend well.")


# === Exercise 2: Stationarity Testing ===
# Problem: Generate a stationary AR(1) process and a non-stationary
#   random walk. Implement a simple stationarity check by comparing
#   the mean and variance of different segments of the series.
def exercise_2():
    """Solution demonstrating stationarity vs non-stationarity.

    A stationary series has constant mean and variance over time.
    A random walk Y_t = Y_{t-1} + e_t is non-stationary because its
    variance grows linearly with time: Var(Y_t) = t * sigma^2.

    We verify this by splitting each series into segments and comparing
    their statistics.
    """
    np.random.seed(42)
    n = 1000

    # Stationary: AR(1) with phi = 0.7
    phi = 0.7
    ar1 = np.zeros(n)
    for i in range(1, n):
        ar1[i] = phi * ar1[i - 1] + np.random.normal(0, 1)

    # Non-stationary: Random walk
    rw = np.cumsum(np.random.normal(0, 1, n))

    def segment_stats(series, n_segments=4):
        """Compute mean and std for each segment."""
        seg_len = len(series) // n_segments
        results = []
        for i in range(n_segments):
            start = i * seg_len
            end = start + seg_len
            seg = series[start:end]
            results.append((seg.mean(), seg.std()))
        return results

    print("Stationarity Check via Segment Comparison\n")

    # AR(1) process
    print(f"AR(1) process (phi={phi}, stationary):")
    print(f"  Theoretical mean: 0")
    print(f"  Theoretical std:  {1/np.sqrt(1 - phi**2):.4f}")
    print(f"  {'Segment':>8s}  {'Mean':>8s}  {'Std':>8s}")
    print(f"  {'-'*28}")
    ar1_stats = segment_stats(ar1)
    for i, (m, s) in enumerate(ar1_stats):
        print(f"  {i+1:8d}  {m:8.4f}  {s:8.4f}")
    means = [m for m, _ in ar1_stats]
    stds = [s for _, s in ar1_stats]
    print(f"  Range of means: {max(means) - min(means):.4f} (small = stationary)")
    print(f"  Range of stds:  {max(stds) - min(stds):.4f} (small = stationary)")

    print()

    # Random walk
    print("Random walk (non-stationary):")
    print(f"  Var(Y_t) = t * sigma^2 (grows over time)")
    print(f"  {'Segment':>8s}  {'Mean':>8s}  {'Std':>8s}")
    print(f"  {'-'*28}")
    rw_stats = segment_stats(rw)
    for i, (m, s) in enumerate(rw_stats):
        print(f"  {i+1:8d}  {m:8.4f}  {s:8.4f}")
    means_rw = [m for m, _ in rw_stats]
    stds_rw = [s for _, s in rw_stats]
    print(f"  Range of means: {max(means_rw) - min(means_rw):.4f} (large = non-stationary)")
    print(f"  Range of stds:  {max(stds_rw) - min(stds_rw):.4f}")

    # First differencing to make random walk stationary
    rw_diff = np.diff(rw)
    print(f"\nFirst difference of random walk:")
    print(f"  Mean: {rw_diff.mean():.4f} (should be ~0)")
    print(f"  Std:  {rw_diff.std():.4f} (should be ~1)")
    print(f"  Lag-1 autocorrelation: {np.corrcoef(rw_diff[:-1], rw_diff[1:])[0,1]:.4f}")
    print(f"  Differencing transforms the random walk into white noise.")


# === Exercise 3: Autocorrelation Function ===
# Problem: Compute the sample ACF for an AR(1) process and compare
#   with the theoretical ACF. Implement the PACF using the Durbin-Levinson
#   recursion.
def exercise_3():
    """Solution computing ACF and PACF from scratch.

    For AR(1) with parameter phi:
        ACF:  rho(k) = phi^k  (exponential decay)
        PACF: alpha(1) = phi, alpha(k) = 0 for k >= 2

    The ACF and PACF patterns help identify model order:
        AR(p):  ACF decays gradually, PACF cuts off after lag p
        MA(q):  ACF cuts off after lag q, PACF decays gradually
    """
    np.random.seed(42)
    phi = 0.8
    n = 500

    # Generate AR(1) process
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = phi * y[i - 1] + np.random.normal(0, 1)

    # Compute sample ACF
    max_lag = 15
    y_centered = y - y.mean()
    var_y = np.var(y, ddof=0)

    acf_values = np.zeros(max_lag + 1)
    for k in range(max_lag + 1):
        if k == 0:
            acf_values[k] = 1.0
        else:
            cov_k = np.mean(y_centered[k:] * y_centered[:-k])
            acf_values[k] = cov_k / var_y

    # Theoretical ACF for AR(1)
    acf_theoretical = np.array([phi**k for k in range(max_lag + 1)])

    print(f"ACF for AR(1) with phi={phi}")
    print(f"  {'Lag':>4s}  {'Sample':>8s}  {'Theory':>8s}  {'Diff':>8s}")
    print(f"  {'-'*32}")
    for k in range(max_lag + 1):
        diff = abs(acf_values[k] - acf_theoretical[k])
        print(f"  {k:4d}  {acf_values[k]:8.4f}  {acf_theoretical[k]:8.4f}  {diff:8.4f}")

    # Compute PACF using Durbin-Levinson recursion
    # PACF(k) is the last coefficient in the AR(k) model fit via Yule-Walker
    pacf_values = np.zeros(max_lag + 1)
    pacf_values[0] = 1.0
    pacf_values[1] = acf_values[1]

    phi_prev = np.array([acf_values[1]])

    for k in range(2, max_lag + 1):
        # Durbin-Levinson recursion
        numerator = acf_values[k] - np.sum(phi_prev * acf_values[k-1:0:-1])
        denominator = 1.0 - np.sum(phi_prev * acf_values[1:k])

        if abs(denominator) < 1e-12:
            pacf_values[k] = 0.0
            break

        pacf_k = numerator / denominator
        pacf_values[k] = pacf_k

        # Update coefficients
        phi_new = np.zeros(k)
        phi_new[k - 1] = pacf_k
        for j in range(k - 1):
            phi_new[j] = phi_prev[j] - pacf_k * phi_prev[k - 2 - j]
        phi_prev = phi_new

    print(f"\nPACF for AR(1) with phi={phi}")
    print(f"  {'Lag':>4s}  {'Sample PACF':>12s}  {'Expected':>10s}")
    print(f"  {'-'*30}")
    for k in range(max_lag + 1):
        expected = phi if k == 1 else (1.0 if k == 0 else 0.0)
        print(f"  {k:4d}  {pacf_values[k]:12.4f}  {expected:10.4f}")

    # Significance bounds (approximate)
    se = 1.0 / np.sqrt(n)
    bound = 1.96 * se
    print(f"\n  95% significance bound: +/- {bound:.4f}")
    print(f"  PACF cuts off after lag 1, confirming AR(1) structure.")


# === Exercise 4: Time Series Decomposition ===
# Problem: Implement additive decomposition using a centered moving average
#   for trend and period-averaging for the seasonal component.
def exercise_4():
    """Solution implementing classical additive decomposition.

    Steps:
    1. Estimate trend using a centered moving average of length = period
    2. Remove trend to get detrended series
    3. Average the detrended values by season position to get seasonal component
    4. Residual = original - trend - seasonal
    """
    np.random.seed(42)
    period = 12  # monthly data with yearly seasonality
    n_years = 5
    n = period * n_years

    t = np.arange(n)
    trend_true = 100 + 0.5 * t
    seasonal_true = 8 * np.sin(2 * np.pi * t / period)
    noise = np.random.normal(0, 2, n)
    y = trend_true + seasonal_true + noise

    print(f"Synthetic monthly data: {n_years} years, {n} observations")
    print(f"  True trend: 100 + 0.5*t")
    print(f"  True seasonal: amplitude=8, period={period}")
    print(f"  Noise: N(0, 2)")

    # Step 1: Centered moving average for trend
    # For even period, use a 2x moving average: first MA(12), then MA(2)
    half = period // 2

    # Simple centered MA: average of period values centered at each point
    trend_est = np.full(n, np.nan)
    for i in range(half, n - half):
        # For even period: (0.5*y[i-6] + y[i-5] + ... + y[i+5] + 0.5*y[i+6]) / 12
        window = y[i - half:i + half + 1].copy()
        # Adjust endpoints for even period
        window[0] *= 0.5
        window[-1] *= 0.5
        trend_est[i] = np.sum(window) / period

    # Find valid range
    valid = ~np.isnan(trend_est)
    valid_idx = np.where(valid)[0]

    print(f"\nStep 1: Trend estimation (centered MA, period={period})")
    print(f"  Valid range: indices {valid_idx[0]} to {valid_idx[-1]}")
    # Compare at a few points
    for idx in [12, 24, 36, 48]:
        if idx < len(trend_est) and valid[idx]:
            err = abs(trend_est[idx] - trend_true[idx])
            print(f"  t={idx}: estimated={trend_est[idx]:.2f}, true={trend_true[idx]:.2f}, error={err:.2f}")

    # Step 2: Detrended series
    detrended = np.where(valid, y - trend_est, np.nan)

    # Step 3: Seasonal component (average by month position)
    seasonal_est = np.zeros(period)
    for m in range(period):
        month_vals = []
        for yr in range(n_years):
            idx = yr * period + m
            if idx < n and valid[idx]:
                month_vals.append(detrended[idx])
        if month_vals:
            seasonal_est[m] = np.mean(month_vals)

    # Center the seasonal component (subtract its mean)
    seasonal_est -= seasonal_est.mean()

    print(f"\nStep 3: Seasonal component estimates")
    print(f"  {'Month':>6s}  {'Estimated':>10s}  {'True':>8s}  {'Error':>8s}")
    print(f"  {'-'*36}")
    for m in range(period):
        true_s = 8 * np.sin(2 * np.pi * m / period)
        err = abs(seasonal_est[m] - true_s)
        print(f"  {m+1:6d}  {seasonal_est[m]:10.4f}  {true_s:8.4f}  {err:8.4f}")

    # Step 4: Residual
    seasonal_full = np.tile(seasonal_est, n_years)
    residual = np.where(valid, y - trend_est - seasonal_full, np.nan)
    resid_valid = residual[valid]

    print(f"\nStep 4: Residual statistics (valid observations)")
    print(f"  Mean: {resid_valid.mean():.4f} (should be ~0)")
    print(f"  Std:  {resid_valid.std():.4f} (should be ~2)")
    print(f"  Min:  {resid_valid.min():.4f}")
    print(f"  Max:  {resid_valid.max():.4f}")


if __name__ == "__main__":
    print("=== Exercise 1: Time Series Components ===")
    exercise_1()
    print("\n=== Exercise 2: Stationarity Testing ===")
    exercise_2()
    print("\n=== Exercise 3: Autocorrelation Function ===")
    exercise_3()
    print("\n=== Exercise 4: Time Series Decomposition ===")
    exercise_4()
    print("\nAll exercises completed!")
