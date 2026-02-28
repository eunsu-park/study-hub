"""
Time Series ML - Exercise Solutions
=====================================
Lesson 18: Time Series ML

Exercises cover:
  1. Energy demand forecasting with lag/rolling/calendar features
  2. Stock price direction classification
  3. Time series classification (AR, random walk, periodic)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    accuracy_score, f1_score, confusion_matrix, classification_report
)


# ============================================================
# Exercise 1: Energy Demand Forecasting
# Create features and predict hourly energy demand.
# ============================================================
def exercise_1_energy_demand():
    """Energy demand forecasting with temporal feature engineering.

    Key principles:
    - ALWAYS use temporal splits (no random CV for time series!)
    - ALWAYS shift before rolling to prevent look-ahead bias
    - Calendar features capture human behavior patterns (weekday vs weekend)
    - Cyclical encoding (sin/cos) ensures hour 23 and hour 0 are close
    """
    print("=" * 60)
    print("Exercise 1: Energy Demand Forecasting")
    print("=" * 60)

    # Generate 2 years of hourly energy demand data
    np.random.seed(42)
    n_hours = 365 * 2 * 24
    dates = pd.date_range("2022-01-01", periods=n_hours, freq="h")

    hours = dates.hour
    day_of_week = dates.dayofweek
    month = dates.month

    # Realistic demand pattern: daily cycle + weekly cycle + seasonal + trend
    daily_pattern = 50 * np.sin(2 * np.pi * hours / 24 - np.pi / 2) + 100
    weekly_effect = 20 * (day_of_week < 5).astype(float)  # weekday premium
    seasonal = 30 * np.sin(2 * np.pi * np.arange(n_hours) / (365 * 24))
    trend = np.linspace(0, 20, n_hours)
    noise = np.random.randn(n_hours) * 10

    demand = daily_pattern + weekly_effect + seasonal + trend + noise
    demand = np.clip(demand, 20, None)

    df = pd.DataFrame({"date": dates, "demand": demand})

    # --- Feature Engineering ---
    # Calendar features
    df["hour"] = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Cyclical encoding -- hour 23 and hour 0 are adjacent
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Lag features (shift prevents look-ahead leakage)
    for lag in [1, 24, 168]:  # 1 hour, 1 day, 1 week
        df[f"lag_{lag}"] = df["demand"].shift(lag)

    # Rolling statistics (shift(1) before rolling to avoid using current value)
    for window in [24, 168]:  # 1 day, 1 week
        df[f"rolling_mean_{window}"] = df["demand"].shift(1).rolling(window).mean()
        df[f"rolling_std_{window}"] = df["demand"].shift(1).rolling(window).std()

    df = df.dropna().reset_index(drop=True)

    # Temporal split: last 30 days for testing
    split_idx = len(df) - 30 * 24
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    feature_cols = [c for c in df.columns if c not in ["date", "demand"]]
    X_train, y_train = train[feature_cols].values, train["demand"].values
    X_test, y_test = test[feature_cols].values, test["demand"].values

    # Train model
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)

    # Naive baseline: use demand from 1 week ago
    naive_pred = test["lag_168"].values
    naive_mae = mean_absolute_error(y_test, naive_pred)
    naive_rmse = np.sqrt(mean_squared_error(y_test, naive_pred))

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"Features: {len(feature_cols)}")
    print(f"\n{'Metric':<10} {'GB Model':>10} {'Naive (lag_168)':>15}")
    print("-" * 38)
    print(f"{'MAE':<10} {mae:>10.2f} {naive_mae:>15.2f}")
    print(f"{'RMSE':<10} {rmse:>10.2f} {naive_rmse:>15.2f}")

    # Top features
    importances = gb.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print(f"\nTop 10 features:")
    for i in range(10):
        print(f"  {feature_cols[sorted_idx[i]]:<25s} {importances[sorted_idx[i]]:.4f}")


# ============================================================
# Exercise 2: Stock Price Direction Classification
# Predict next-day return direction with walk-forward validation.
# ============================================================
def exercise_2_stock_direction():
    """Stock price direction prediction with walk-forward CV.

    Walk-forward validation is essential for financial data:
    - Train on past, predict future (never peek ahead)
    - Gap between train and test prevents information leakage from
      autocorrelated features

    Features include:
    - Returns at multiple horizons (momentum)
    - Volatility (risk indicator)
    - RSI (relative strength index -- mean-reversion indicator)
    """
    print("\n" + "=" * 60)
    print("Exercise 2: Stock Price Direction Classification")
    print("=" * 60)

    # Generate synthetic daily stock price
    np.random.seed(42)
    n_days = 1000
    returns = np.random.randn(n_days) * 0.015 + 0.0003  # slight positive drift
    prices = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        "price": prices,
        "return_1d": np.append([0], np.diff(np.log(prices))),
    })

    # Features
    for window in [5, 20]:
        df[f"return_{window}d"] = df["return_1d"].rolling(window).sum()
        df[f"volatility_{window}d"] = df["return_1d"].rolling(window).std()

    # RSI (14-day) -- momentum oscillator
    delta = df["return_1d"]
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Volume proxy (synthetic)
    df["volume_ratio"] = np.random.lognormal(0, 0.3, n_days)

    # Target: next-day direction (1 if positive return, else 0)
    df["target"] = (df["return_1d"].shift(-1) > 0).astype(int)

    df = df.dropna().reset_index(drop=True)
    df = df.iloc[:-1]  # remove last row (no target)

    feature_cols = ["return_1d", "return_5d", "return_20d",
                    "volatility_5d", "volatility_20d", "rsi_14", "volume_ratio"]

    # Walk-forward validation with gap
    print("\nWalk-forward validation (expanding window, 5-day gap):")
    fold_size = 100
    gap = 5
    results = []

    for start in range(500, len(df) - fold_size, fold_size):
        train_end = start - gap
        test_start = start
        test_end = min(start + fold_size, len(df))

        X_train = df.loc[:train_end, feature_cols].values
        y_train = df.loc[:train_end, "target"].values
        X_test = df.loc[test_start:test_end-1, feature_cols].values
        y_test = df.loc[test_start:test_end-1, "target"].values

        clf = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        results.append({"acc": acc, "f1": f1})

    mean_acc = np.mean([r["acc"] for r in results])
    mean_f1 = np.mean([r["f1"] for r in results])

    print(f"  Number of folds: {len(results)}")
    print(f"  Mean accuracy: {mean_acc:.4f}")
    print(f"  Mean F1:       {mean_f1:.4f}")
    print(f"  Random baseline accuracy: ~0.50")

    # Profitability simulation
    train_end = len(df) - 200
    X_train = df.loc[:train_end, feature_cols].values
    y_train = df.loc[:train_end, "target"].values
    X_test = df.loc[train_end+gap:, feature_cols].values
    y_test = df.loc[train_end+gap:, "target"].values
    returns_test = df.loc[train_end+gap:, "return_1d"].shift(-1).values[:-1]

    clf = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
    clf.fit(X_train, y_train)
    signals = clf.predict(X_test[:-1])

    strategy_returns = signals * returns_test
    cumulative = np.cumsum(strategy_returns)
    buy_hold = np.cumsum(returns_test)

    print(f"\n  Strategy cumulative return: {cumulative[-1]*100:.2f}%")
    print(f"  Buy-and-hold return:       {buy_hold[-1]*100:.2f}%")


# ============================================================
# Exercise 3: Time Series Classification
# Classify AR, random walk, and periodic time series.
# ============================================================
def exercise_3_ts_classification():
    """Classify time series into 3 types using statistical features.

    Three classes:
    - AR(1): mean-reverting (phi=0.5), stationary
    - Random Walk: non-stationary, variance grows with time
    - Periodic: deterministic seasonal pattern with noise

    Feature extraction converts variable-length time series into
    fixed-length feature vectors for standard classifiers.
    """
    print("\n" + "=" * 60)
    print("Exercise 3: Time Series Classification")
    print("=" * 60)

    np.random.seed(42)
    n_series = 100
    length = 200
    series_list = []
    labels = []

    for _ in range(n_series):
        # Class 0: AR(1) process (mean-reverting)
        ar = np.zeros(length)
        for t in range(1, length):
            ar[t] = 0.5 * ar[t-1] + np.random.randn()
        series_list.append(ar)
        labels.append(0)

        # Class 1: Random walk (non-stationary)
        rw = np.cumsum(np.random.randn(length))
        series_list.append(rw)
        labels.append(1)

        # Class 2: Periodic (seasonal)
        t = np.arange(length)
        periodic = 3 * np.sin(2 * np.pi * t / 30) + np.random.randn(length) * 0.5
        series_list.append(periodic)
        labels.append(2)

    y = np.array(labels)

    # Extract statistical features from each time series
    def extract_features(ts):
        """Extract discriminative features from a time series."""
        returns = np.diff(ts)
        return [
            np.mean(ts),                      # level
            np.std(ts),                        # overall variability
            np.mean(returns),                  # drift
            np.std(returns),                   # return volatility
            np.corrcoef(ts[:-1], ts[1:])[0, 1],  # lag-1 autocorrelation
            np.max(ts) - np.min(ts),           # range
            np.mean(np.abs(np.diff(np.sign(returns)))),  # zero-crossing rate
            # Spectral feature: dominant frequency energy
            np.max(np.abs(np.fft.rfft(ts - np.mean(ts)))[1:]) / length,
            # Trend strength: R^2 of linear fit
            np.corrcoef(np.arange(length), ts)[0, 1] ** 2,
            # Stationarity proxy: variance ratio (first half vs second half)
            np.var(ts[:length//2]) / (np.var(ts[length//2:]) + 1e-8),
        ]

    feature_names = [
        "mean", "std", "drift", "return_std", "autocorr_1",
        "range", "zero_crossing", "spectral_peak", "trend_r2", "var_ratio"
    ]

    X = np.array([extract_features(s) for s in series_list])

    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Classify
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)

    class_names = ["AR(1)", "Random Walk", "Periodic"]
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Feature importance
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("Feature importance:")
    for i in range(len(feature_names)):
        print(f"  {feature_names[sorted_idx[i]]:<20s} {importances[sorted_idx[i]]:.4f}")


if __name__ == "__main__":
    exercise_1_energy_demand()
    exercise_2_stock_direction()
    exercise_3_ts_classification()
