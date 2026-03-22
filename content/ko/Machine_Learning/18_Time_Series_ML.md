# 시계열 머신러닝(Time Series Machine Learning)

**이전**: [불균형 데이터 처리](./17_Imbalanced_Data.md) | **다음**: [AutoML과 하이퍼파라미터 최적화](./19_AutoML_Hyperparameter_Optimization.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 지연 특징(lag features)을 사용하여 시계열 예측 문제를 지도 학습(supervised learning)으로 재구성하는 방법을 설명할 수 있습니다
2. 지연값, 롤링 통계량(rolling statistics), 달력 성분, 푸리에 항(Fourier terms)을 포함한 시간적 특징을 설계할 수 있습니다
3. TimeSeriesSplit과 워크포워드 검증(walk-forward validation)을 적용하여 교차검증에서 시간적 데이터 누수(temporal data leakage)를 방지할 수 있습니다
4. 시계열 예측을 위해 트리 기반 모델(GradientBoosting, XGBoost, LightGBM)을 훈련할 수 있습니다
5. 다중 스텝 예측 전략(재귀적, 직접, 다중 출력)을 비교하고 각각의 오차 특성을 설명할 수 있습니다
6. 비즈니스 예측에서 Prophet과 트리 기반 접근법을 언제 사용해야 하는지 설명할 수 있습니다
7. MAE, RMSE, MAPE, sMAPE, MASE를 사용하여 예측을 평가하고 각 지표의 강점을 해석할 수 있습니다

---

ARIMA와 같은 고전적 시계열 방법은 선형 자기상관(linear autocorrelation) 구조를 이용하여 단일 시계열을 모델링합니다. 하지만 수십 개의 외생 특징(exogenous features), 비선형 관계, 또는 수백 개의 관련 시계열이 있다면 어떨까요? 머신러닝 접근법은 예측을 테이블형 지도 학습 문제로 재구성합니다 -- 시간적 특징을 설계하고, 그래디언트 부스팅 트리에 입력하여, 이미 알고 있는 동일한 도구를 활용합니다. 이 레슨에서는 특징 엔지니어링부터 적절한 시간적 검증까지, 이러한 전환을 언제 그리고 어떻게 수행하는지 보여줍니다.

---

## 1. ML 대 고전적 시계열

### 1.1 두 가지 패러다임

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

### 1.2 지도 학습으로서의 시계열

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

## 2. 시계열을 위한 특징 공학(Feature Engineering for Time Series)

### 2.1 지연 특징(Lag Features)

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

### 2.2 롤링 및 확장 통계량(Rolling and Expanding Statistics)

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

### 2.3 달력 특징(Calendar Features)

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

### 2.4 푸리에 특징(Fourier Features)

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

## 3. 시계열을 위한 교차 검증(Cross-Validation for Time Series)

### 3.1 무작위 CV가 실패하는 이유

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

### 3.2 TimeSeriesSplit 구현

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

### 3.3 슬라이딩 윈도우 CV 구현

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

## 4. 트리 기반 시계열 예측

### 4.1 XGBoost / LightGBM을 이용한 예측

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

### 4.2 시계열에서의 특징 중요도(Feature Importance)

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

### 4.3 다중 스텝 예측 전략(Multi-Step Forecasting Strategies)

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

## 5. Prophet과 NeuralProphet

### 5.1 페이스북 Prophet

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

### 5.2 Prophet 대 트리 기반 모델: 선택 기준

| 측면 | Prophet | 트리 기반 (XGBoost/LightGBM) |
|--------|---------|-------------------------------|
| **강점** | 분해(Decomposition), 해석 가능성 | 유연성, 다양한 특징 처리 |
| **계절성(Seasonality)** | 내장됨 (푸리에) | 수동으로 설계 필요 |
| **외생 변수(Exogenous vars)** | add_regressor() | 자연스럽게 지원 (모든 특징) |
| **다중 시계열** | 시리즈당 하나의 모델 | 전역 모델(Global model) 가능 |
| **결측 데이터** | 자연스럽게 처리 | 대체(Imputation) 필요 |
| **비선형성** | 제한적 | 탁월함 |
| **속도** | 단일 시계열에 빠름 | 배치 처리에 빠름 |
| **최적 용도** | 비즈니스 계획, 예측 | 경진대회, 복잡한 패턴 |

---

## 6. 시계열 분류(Time Series Classification)

### 6.1 특징 기반 분류(Feature-Based Classification)

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

### 6.2 동적 시간 워핑(DTW, Dynamic Time Warping)

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

### 6.3 tsfresh를 이용한 자동화된 특징 추출

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

## 7. 백테스팅과 평가(Backtesting and Evaluation)

### 7.1 워크포워드 검증(Walk-Forward Validation)

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

### 7.2 시계열 지표(Time Series Metrics)

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

## 8. 실무적 고려사항

### 8.1 다중 시계열 처리 (전역 모델, Global Models)

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

### 8.2 특징 누수 방지(Feature Leakage Prevention)

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

## 9. 연습 문제

### 연습 1: 에너지 수요 예측

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

### 연습 2: 주가 방향 분류

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

### 연습 3: tsfresh를 이용한 시계열 분류

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

## 10. 요약

### 핵심 정리

| 개념 | 설명 |
|---------|-------------|
| **지도 학습으로의 변환(Supervised framing)** | 시계열 → 지연/롤링 특징을 가진 테이블 형식으로 변환 |
| **시간적 CV(Temporal CV)** | 항상 TimeSeriesSplit 또는 워크포워드 사용 (무작위 CV 절대 사용 금지) |
| **특징 유형(Feature types)** | 지연값, 롤링 통계량, 달력, 푸리에, 순환 인코딩(Cyclical Encoding) |
| **누수 방지(Leakage prevention)** | 롤링 전 shift(1) 적용; 시간적 훈련/테스트 분할 |
| **트리 기반 모델(Tree-based models)** | XGBoost/LightGBM은 테이블 형식 시계열에 탁월 |
| **Prophet** | 해석 가능한 분해로 비즈니스 예측에 최적 |
| **전역 모델(Global models)** | 다수의 시계열에 하나의 모델 → 더 많은 데이터, 더 나은 일반화 |
| **지표(Metrics)** | MASE는 스케일 불변; 0 근처 값에서 MAPE 사용 지양 |

### 모범 사례

1. **나이브 기준선(lag_1 또는 계절적 나이브)부터 시작** — 먼저 이를 능가해야 함
2. **모든 전처리와 평가에서 시간적 순서 준수**
3. **도메인 특징 설계** (공휴일, 이벤트, 프로모션, 날씨)
4. **현실적인 성능 추정을 위해 워크포워드 검증 사용**
5. **빠른 비즈니스 예측에는 Prophet, 경진대회에는 XGBoost 고려**

### 다음 단계

- **L19**: AutoML — 모델 및 하이퍼파라미터 선택 자동화
- **L20**: 이상 탐지(Anomaly Detection) — 데이터에서 비정상적인 패턴 탐지
- **Data_Science L20-21**: 고전적 시계열 (ARIMA, SARIMA) — 상호 보완적 방법
