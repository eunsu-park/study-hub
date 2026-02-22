# Time Series Machine Learning

## Overview

This lesson covers machine learning approaches to time series forecasting and classification. Unlike classical statistical methods (ARIMA, SARIMA — see Data_Science L20-21), ML approaches treat forecasting as a supervised learning problem by engineering temporal features and using tree-based models, Prophet, or specialized classifiers.

---

## 1. ML vs Classical Time Series

### 1.1 Two Paradigms

```python
"""
Classical Time Series (Data_Science L20-21):
  - ARIMA, SARIMA, ETS
  - Model the time series directly (autocorrelation, trend, seasonality)
  - Single series at a time (local model)
  - Interpretable parameters (p, d, q)

ML Time Series (this lesson):
  - XGBoost, LightGBM, Random Forest, Prophet
  - Transform time series into tabular features (lag, rolling, calendar)
  - Can model multiple series simultaneously (global model)
  - More flexible: handles non-linear relationships, exogenous variables

When to use ML:
  - Multiple related time series (global model)
  - Complex non-linear patterns
  - Many exogenous features available
  - Need to combine with other ML pipelines

When to use classical:
  - Single univariate series
  - Need interpretable model
  - Short series with limited data
  - Well-understood seasonal patterns
"""
```

### 1.2 Time Series as Supervised Learning

```python
import numpy as np
import pandas as pd

# Original time series
dates = pd.date_range('2023-01-01', periods=10, freq='D')
sales = [100, 110, 105, 120, 115, 130, 125, 140, 135, 150]
ts = pd.DataFrame({'date': dates, 'sales': sales})

# Transform to supervised: features = past values, target = future value
def create_supervised(df, target_col, lags):
    """Convert time series to supervised learning format."""
    result = df.copy()
    for lag in lags:
        result[f'{target_col}_lag_{lag}'] = result[target_col].shift(lag)
    result = result.dropna()
    return result

supervised = create_supervised(ts, 'sales', lags=[1, 2, 3])
print("Time series as supervised learning:")
print(supervised.to_string(index=False))
print("\nFeatures: lag_1, lag_2, lag_3  →  Target: sales")
```

---

## 2. Feature Engineering for Time Series

### 2.1 Lag Features

```python
import numpy as np
import pandas as pd

# Generate realistic daily sales data
np.random.seed(42)
n_days = 730  # 2 years
dates = pd.date_range('2022-01-01', periods=n_days, freq='D')

# Trend + weekly seasonality + yearly seasonality + noise
trend = np.linspace(100, 150, n_days)
weekly = 15 * np.sin(2 * np.pi * np.arange(n_days) / 7)
yearly = 30 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)
noise = np.random.normal(0, 5, n_days)
sales = trend + weekly + yearly + noise

df = pd.DataFrame({'date': dates, 'sales': sales})

# Lag features
for lag in [1, 7, 14, 28, 365]:
    df[f'lag_{lag}'] = df['sales'].shift(lag)

print("Lag features (last 5 rows):")
print(df[['date', 'sales', 'lag_1', 'lag_7', 'lag_28', 'lag_365']].tail())
```

### 2.2 Rolling and Expanding Statistics

```python
# Rolling statistics (always shift to avoid leakage!)
for window in [7, 14, 30]:
    shifted = df['sales'].shift(1)  # Exclude current value
    df[f'rolling_mean_{window}'] = shifted.rolling(window).mean()
    df[f'rolling_std_{window}'] = shifted.rolling(window).std()
    df[f'rolling_min_{window}'] = shifted.rolling(window).min()
    df[f'rolling_max_{window}'] = shifted.rolling(window).max()

# Expanding statistics
df['expanding_mean'] = df['sales'].shift(1).expanding().mean()

# EWM (Exponentially Weighted Moving Average)
df['ewm_7'] = df['sales'].shift(1).ewm(span=7).mean()
df['ewm_30'] = df['sales'].shift(1).ewm(span=30).mean()

print("Rolling features:")
print(df[['date', 'sales', 'rolling_mean_7', 'rolling_std_7', 'ewm_7']].tail())
```

### 2.3 Calendar Features

```python
# Date/time components
df['day_of_week'] = df['date'].dt.dayofweek
df['day_of_month'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['quarter'] = df['date'].dt.quarter
df['day_of_year'] = df['date'].dt.dayofyear
df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
df['is_month_end'] = df['date'].dt.is_month_end.astype(int)

# Cyclical encoding (see L15 Feature Engineering)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

print("Calendar features:")
print(df[['date', 'day_of_week', 'month', 'is_weekend', 'month_sin', 'dow_sin']].head(10))
```

### 2.4 Fourier Features

```python
def add_fourier_features(df, period, n_harmonics, prefix='fourier'):
    """Add Fourier features to capture periodic patterns."""
    t = np.arange(len(df))
    for k in range(1, n_harmonics + 1):
        df[f'{prefix}_sin_{k}'] = np.sin(2 * np.pi * k * t / period)
        df[f'{prefix}_cos_{k}'] = np.cos(2 * np.pi * k * t / period)
    return df

# Weekly pattern (period=7, 3 harmonics)
df = add_fourier_features(df, period=7, n_harmonics=3, prefix='weekly')

# Yearly pattern (period=365.25, 5 harmonics)
df = add_fourier_features(df, period=365.25, n_harmonics=5, prefix='yearly')

fourier_cols = [c for c in df.columns if c.startswith('weekly') or c.startswith('yearly')]
print(f"Added {len(fourier_cols)} Fourier features")
```

---

## 3. Cross-Validation for Time Series

### 3.1 Why Random CV Fails

```python
"""
Standard K-Fold CV randomly shuffles data → future data leaks into training!

Time Series CV must respect temporal order:
  - Training data is always BEFORE validation data
  - No shuffling allowed

Methods:
1. TimeSeriesSplit: Expanding window
   Train: [----]        Val: [-]
   Train: [------]      Val: [-]
   Train: [--------]    Val: [-]

2. Sliding Window:
   Train: [----]        Val: [-]
      Train: [----]     Val: [-]
         Train: [----]  Val: [-]

3. Purged CV: Add gap between train and val to prevent leakage
"""
```

### 3.2 TimeSeriesSplit Implementation

```python
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

# Prepare data (drop rows with NaN from lag/rolling features)
feature_cols = [c for c in df.columns if c not in ['date', 'sales']]
df_clean = df.dropna().copy()
X = df_clean[feature_cols]
y = df_clean['sales']

# TimeSeriesSplit (expanding window)
tscv = TimeSeriesSplit(n_splits=5)

fig, ax = plt.subplots(figsize=(12, 4))
for i, (train_idx, val_idx) in enumerate(tscv.split(X)):
    ax.scatter(train_idx, [i]*len(train_idx), c='blue', s=1, label='Train' if i==0 else None)
    ax.scatter(val_idx, [i]*len(val_idx), c='red', s=1, label='Val' if i==0 else None)

ax.set_xlabel('Sample Index')
ax.set_ylabel('CV Fold')
ax.set_title('TimeSeriesSplit (Expanding Window)')
ax.legend()
plt.tight_layout()
plt.savefig('time_series_cv.png', dpi=150)
plt.show()
```

### 3.3 Custom Sliding Window CV

```python
def sliding_window_cv(n_samples, train_size, val_size, step=None):
    """Generate sliding window train/val indices."""
    if step is None:
        step = val_size
    indices = []
    start = 0
    while start + train_size + val_size <= n_samples:
        train_idx = list(range(start, start + train_size))
        val_idx = list(range(start + train_size, start + train_size + val_size))
        indices.append((train_idx, val_idx))
        start += step
    return indices

# Example: 365-day training window, 30-day validation, step 30 days
splits = sliding_window_cv(len(X), train_size=365, val_size=30, step=30)
print(f"Sliding window: {len(splits)} folds")
print(f"First fold: train {splits[0][0][0]}-{splits[0][0][-1]}, val {splits[0][1][0]}-{splits[0][1][-1]}")
```

---

## 4. Tree-Based Time Series Forecasting

### 4.1 XGBoost / LightGBM for Forecasting

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Use only non-date features
feature_cols = [c for c in df.columns if c not in ['date', 'sales']]
df_model = df.dropna().reset_index(drop=True)
X = df_model[feature_cols]
y = df_model['sales']

# Time-based train/test split (last 60 days for testing)
split_point = len(X) - 60
X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
dates_test = df_model['date'].iloc[split_point:]

# Train Gradient Boosting
gb = GradientBoostingRegressor(
    n_estimators=200, max_depth=5, learning_rate=0.1,
    subsample=0.8, random_state=42
)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
r2 = r2_score(y_test, y_pred)

print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"R²:   {r2:.4f}")

# Plot predictions vs actual
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(dates_test, y_test, 'b-', label='Actual', linewidth=2)
ax.plot(dates_test, y_pred, 'r--', label='Predicted', linewidth=2)
ax.fill_between(dates_test, y_pred - rmse, y_pred + rmse, alpha=0.2, color='red')
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
ax.set_title(f'Gradient Boosting Forecast (MAE={mae:.2f}, MAPE={mape:.1f}%)')
ax.legend()
plt.tight_layout()
plt.savefig('gb_forecast.png', dpi=150)
plt.show()
```

### 4.2 Feature Importance for Time Series

```python
# Which temporal features matter most?
importances = pd.Series(gb.feature_importances_, index=feature_cols)
top_20 = importances.nlargest(20)

fig, ax = plt.subplots(figsize=(10, 8))
top_20.sort_values().plot(kind='barh', ax=ax, color='steelblue')
ax.set_title('Top 20 Feature Importances (Time Series GB)')
ax.set_xlabel('Importance')
plt.tight_layout()
plt.savefig('ts_feature_importance.png', dpi=150)
plt.show()
```

### 4.3 Multi-Step Forecasting Strategies

```python
"""
Multi-step forecasting: predict multiple future time steps.

Strategy 1: Recursive (iterative)
  - Train model to predict t+1
  - Use prediction as input to predict t+2, t+3, ...
  - Error accumulates over horizon

Strategy 2: Direct
  - Train separate model for each horizon (t+1, t+2, ..., t+H)
  - No error accumulation, but more models to train

Strategy 3: Multi-output
  - Single model predicts [t+1, t+2, ..., t+H] simultaneously
  - Captures correlations between horizons

Strategy 4: Rectified (Recursive + Direct)
  - Use recursive for features, direct model for prediction
"""

def recursive_forecast(model, last_features, n_steps, feature_cols, target_col_idx):
    """Multi-step recursive forecast."""
    predictions = []
    current_features = last_features.copy()

    for _ in range(n_steps):
        pred = model.predict(current_features.reshape(1, -1))[0]
        predictions.append(pred)

        # Shift lag features (simplified: assumes lag_1 is at target_col_idx)
        current_features = np.roll(current_features, 1)
        current_features[0] = pred

    return predictions

print("Multi-step strategies: Recursive adds convenience but accumulates error.")
print("For critical applications, prefer Direct or Multi-output strategies.")
```

---

## 5. Prophet and NeuralProphet

### 5.1 Facebook Prophet

```python
"""
Prophet is designed for business time series forecasting.
Key features:
  - Automatic trend detection (linear or logistic growth)
  - Multiple seasonality (yearly, weekly, daily)
  - Holiday effects
  - Robust to missing data and outliers
  - Intuitive parameters

# pip install prophet

from prophet import Prophet

# Prophet requires columns: 'ds' (datetime) and 'y' (value)
prophet_df = df[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})

# Train
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.05,  # Flexibility of trend changes
    seasonality_prior_scale=10,     # Flexibility of seasonality
)
model.fit(prophet_df.iloc[:split_point])

# Predict
future = model.make_future_dataframe(periods=60)
forecast = model.predict(future)

# Plot
fig = model.plot(forecast)
plt.title('Prophet Forecast')
plt.show()

# Decomposition
fig = model.plot_components(forecast)
plt.show()

# Custom seasonality
model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

# Add holiday effects
holidays = pd.DataFrame({
    'holiday': 'christmas',
    'ds': pd.to_datetime(['2022-12-25', '2023-12-25']),
    'lower_window': -2,  # 2 days before
    'upper_window': 1,   # 1 day after
})
model = Prophet(holidays=holidays)
"""
print("Prophet: Best for business forecasting with interpretable components.")
print("Handles missing data, outliers, and trend changes automatically.")
```

### 5.2 Prophet vs Tree-Based: When to Use Which

| Aspect | Prophet | Tree-Based (XGBoost/LightGBM) |
|--------|---------|-------------------------------|
| **Strength** | Decomposition, interpretability | Flexibility, many features |
| **Seasonality** | Built-in (Fourier) | Must engineer manually |
| **Exogenous vars** | add_regressor() | Natural (any features) |
| **Multiple series** | One model per series | Global model possible |
| **Missing data** | Handles naturally | Requires imputation |
| **Non-linear** | Limited | Excellent |
| **Speed** | Fast for single series | Fast for batch |
| **Best for** | Business planning, forecasting | Competition, complex patterns |

---

## 6. Time Series Classification

### 6.1 Feature-Based Classification

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Generate synthetic time series classification data
np.random.seed(42)
n_series = 300
series_length = 100

# Class 0: stationary (random walk)
# Class 1: trending (upward drift)
# Class 2: seasonal (sine wave)
X_series = []
y_labels = []

for _ in range(n_series):
    label = np.random.choice([0, 1, 2])
    t = np.arange(series_length)

    if label == 0:  # Stationary
        series = np.cumsum(np.random.normal(0, 1, series_length))
    elif label == 1:  # Trending
        series = 0.5 * t + np.random.normal(0, 3, series_length)
    else:  # Seasonal
        series = 10 * np.sin(2 * np.pi * t / 20) + np.random.normal(0, 1, series_length)

    X_series.append(series)
    y_labels.append(label)

X_series = np.array(X_series)
y_labels = np.array(y_labels)

# Extract features from each time series
def extract_ts_features(series):
    """Extract statistical features from a time series."""
    return {
        'mean': np.mean(series),
        'std': np.std(series),
        'min': np.min(series),
        'max': np.max(series),
        'range': np.ptp(series),
        'skew': pd.Series(series).skew(),
        'kurtosis': pd.Series(series).kurtosis(),
        'trend': np.polyfit(np.arange(len(series)), series, 1)[0],
        'autocorr_1': pd.Series(series).autocorr(lag=1),
        'autocorr_7': pd.Series(series).autocorr(lag=7),
        'crossing_rate': np.mean(np.diff(np.sign(series - np.mean(series))) != 0),
        'energy': np.sum(series**2) / len(series),
        'entropy': -np.sum(np.abs(np.fft.fft(series)[:len(series)//2])**2 *
                          np.log(np.abs(np.fft.fft(series)[:len(series)//2])**2 + 1e-10)) / len(series),
    }

features = pd.DataFrame([extract_ts_features(s) for s in X_series])
print(f"Extracted {features.shape[1]} features from {features.shape[0]} time series")

# Classify
X_train, X_test, y_train, y_test = train_test_split(features, y_labels, test_size=0.3, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

print(f"\nAccuracy: {clf.score(X_test, y_test):.4f}")
print(classification_report(y_test, clf.predict(X_test),
                           target_names=['Stationary', 'Trending', 'Seasonal']))
```

### 6.2 Dynamic Time Warping (DTW)

```python
"""
DTW (Dynamic Time Warping) measures similarity between time series
that may be shifted or scaled differently in time.

Unlike Euclidean distance, DTW allows non-linear alignment:
  - Handles time series of different lengths
  - Accounts for temporal shifts and stretches
  - Used in: speech recognition, gesture recognition, financial analysis

# pip install dtaidistance  or  tslearn

from dtaidistance import dtw

# Two similar series with temporal shift
series_a = np.sin(np.linspace(0, 4*np.pi, 100))
series_b = np.sin(np.linspace(0.5, 4.5*np.pi, 120))  # shifted, different length

distance = dtw.distance(series_a, series_b)
print(f"DTW distance: {distance:.4f}")

# DTW-based nearest neighbor classifier
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
knn_dtw = KNeighborsTimeSeriesClassifier(n_neighbors=3, metric='dtw')
knn_dtw.fit(X_train_3d, y_train)
"""
print("DTW: Essential for time series similarity when alignment varies.")
```

### 6.3 tsfresh for Automated Feature Extraction

```python
"""
tsfresh automatically extracts hundreds of features from time series.

# pip install tsfresh

import tsfresh
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute

# Format data for tsfresh (long format)
long_df = pd.DataFrame({
    'id': np.repeat(range(n_series), series_length),
    'time': np.tile(range(series_length), n_series),
    'value': X_series.flatten(),
})

# Extract features (can take a while for large datasets)
features = extract_features(long_df, column_id='id', column_sort='time')
impute(features)  # Handle NaN/Inf

# Select relevant features
selected = select_features(features, pd.Series(y_labels), fdr_level=0.05)
print(f"Extracted {features.shape[1]} features, selected {selected.shape[1]}")

# Common tsfresh features:
# - Statistical: mean, variance, skewness, kurtosis
# - Temporal: autocorrelation, partial autocorrelation
# - Spectral: FFT coefficients, spectral density
# - Complexity: sample entropy, approximate entropy
# - Trend: linear trend, number of crossings
"""
print("tsfresh: Extracts 700+ features automatically, then selects relevant ones.")
```

---

## 7. Backtesting and Evaluation

### 7.1 Walk-Forward Validation

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

def walk_forward_validation(X, y, train_size, test_size, step, model_fn):
    """
    Walk-forward validation (backtesting).
    Simulates real-time forecasting by training on past data
    and testing on future data, sliding forward.
    """
    results = []
    start = 0

    while start + train_size + test_size <= len(X):
        train_end = start + train_size
        test_end = train_end + test_size

        X_train = X.iloc[start:train_end]
        y_train = y.iloc[start:train_end]
        X_test = X.iloc[train_end:test_end]
        y_test = y.iloc[train_end:test_end]

        model = model_fn()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        results.append({
            'train_start': start,
            'train_end': train_end,
            'test_start': train_end,
            'test_end': test_end,
            'mae': mae,
            'mean_pred': y_pred.mean(),
            'mean_actual': y_test.mean(),
        })

        start += step

    return pd.DataFrame(results)

# Run backtesting
model_fn = lambda: GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
backtest_results = walk_forward_validation(X, y, train_size=365, test_size=30, step=30, model_fn=model_fn)

print("Walk-forward validation results:")
print(backtest_results[['test_start', 'test_end', 'mae']].to_string(index=False))
print(f"\nOverall MAE: {backtest_results['mae'].mean():.2f} (+/- {backtest_results['mae'].std():.2f})")
```

### 7.2 Time Series Metrics

```python
def time_series_metrics(y_true, y_pred):
    """Compute common time series forecasting metrics."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100

    # MASE: Mean Absolute Scaled Error (relative to naive forecast)
    naive_mae = np.mean(np.abs(np.diff(y_true)))
    mase = mae / naive_mae if naive_mae > 0 else np.inf

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'sMAPE': smape,
        'MASE': mase,
    }

metrics = time_series_metrics(y_test.values, y_pred)
print("Time Series Metrics:")
for name, value in metrics.items():
    print(f"  {name:6s}: {value:.4f}")

print("""
Metric Guide:
  MAE:   Easy to interpret (same unit as target)
  RMSE:  Penalizes large errors more
  MAPE:  Percentage error (problematic when y ≈ 0)
  sMAPE: Symmetric MAPE (handles near-zero better)
  MASE:  Scale-free, compares to naive forecast (MASE < 1 = beats naive)
""")
```

---

## 8. Practical Considerations

### 8.1 Handling Multiple Time Series (Global Models)

```python
"""
Global model: Train one model on ALL time series simultaneously.
Each row has a 'series_id' feature alongside temporal features.

Benefits:
  - More training data → better generalization
  - Shared patterns across series
  - New series with no history can still be forecasted (cold start)

Example:
  | date       | store_id | lag_1 | lag_7 | dow | month | sales |
  |------------|----------|-------|-------|-----|-------|-------|
  | 2023-01-02 | A        | 100   | 95    | 0   | 1     | 105   |
  | 2023-01-02 | B        | 200   | 190   | 0   | 1     | 210   |
  | 2023-01-03 | A        | 105   | 98    | 1   | 1     | 110   |

The model learns both time patterns AND cross-series patterns.
LightGBM handles this very well with categorical features for series_id.
"""
```

### 8.2 Feature Leakage Prevention

```python
"""
Time Series Feature Leakage Checklist:

1. LAG FEATURES
   ✗ lag_0 (current value as feature)
   ✓ lag_1, lag_7, lag_28 (past values only)

2. ROLLING STATISTICS
   ✗ df['sales'].rolling(7).mean()  ← includes current row!
   ✓ df['sales'].shift(1).rolling(7).mean()  ← excludes current

3. TRAIN/TEST SPLIT
   ✗ Random split (shuffles time)
   ✓ Temporal split (train before test)

4. CROSS-VALIDATION
   ✗ KFold (future data leaks into training)
   ✓ TimeSeriesSplit or walk-forward validation

5. TARGET ENCODING
   ✗ Use future data for encoding
   ✓ Only use data up to time t for encoding at time t

6. SCALING
   ✗ Fit scaler on entire dataset (includes test period)
   ✓ Fit scaler on training period only
"""
```

---

## 9. Practice Problems

### Exercise 1: Energy Demand Forecasting

```python
"""
1. Generate or load hourly energy demand data (2 years).
2. Create features:
   - Lag: 1h, 24h, 168h (1 week)
   - Rolling: 24h mean/std, 7d mean
   - Calendar: hour, day_of_week, month, is_weekend
   - Cyclical encoding for hour and month
3. Split: last 30 days for testing.
4. Train GradientBoosting and compare with a naive baseline (lag_168).
5. Evaluate: MAE, RMSE, MAPE, MASE.
6. Identify top 10 features and interpret their importance.
"""
```

### Exercise 2: Stock Price Direction Classification

```python
"""
1. Generate daily stock price data with trends and volatility.
2. Create target: 1 if next-day return > 0, else 0.
3. Features:
   - Returns: 1d, 5d, 20d
   - Volatility: 5d, 20d rolling std of returns
   - Momentum: RSI (14-day), MACD
   - Volume features: volume ratio, volume trend
4. Use purged walk-forward CV (gap between train and test).
5. Evaluate: Accuracy, F1, and most importantly — profitability simulation.
"""
```

### Exercise 3: Time Series Classification with tsfresh

```python
"""
1. Generate 3 classes of time series:
   - AR(1) process (mean-reverting)
   - Random walk (non-stationary)
   - Periodic (seasonal)
2. Extract features using statistical methods from Section 6.1.
3. Train a classifier and evaluate with confusion matrix.
4. Bonus: If tsfresh is installed, compare manual features vs tsfresh.
"""
```

---

## 10. Summary

### Key Takeaways

| Concept | Description |
|---------|-------------|
| **Supervised framing** | Transform time series → tabular with lag/rolling features |
| **Temporal CV** | Always use TimeSeriesSplit or walk-forward (never random CV) |
| **Feature types** | Lags, rolling stats, calendar, Fourier, cyclical encoding |
| **Leakage prevention** | shift(1) before rolling; temporal train/test split |
| **Tree-based models** | XGBoost/LightGBM excel at tabular time series |
| **Prophet** | Best for business forecasting with interpretable decomposition |
| **Global models** | One model for many series → more data, better generalization |
| **Metrics** | MASE is scale-free; avoid MAPE near zero values |

### Best Practices

1. **Start with a naive baseline** (lag_1 or seasonal naive) — beat it first
2. **Respect temporal order** in all preprocessing and evaluation
3. **Engineer domain features** (holidays, events, promotions, weather)
4. **Use walk-forward validation** for realistic performance estimates
5. **Consider Prophet** for quick business forecasting, XGBoost for competitions

### Next Steps

- **L19**: AutoML — automate model and hyperparameter selection
- **L20**: Anomaly Detection — detect unusual patterns in data
- **Data_Science L20-21**: Classical time series (ARIMA, SARIMA) for complementary methods
