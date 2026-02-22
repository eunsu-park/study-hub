# 피처 엔지니어링(Feature Engineering)

## 개요

피처 엔지니어링(Feature Engineering)은 도메인 지식을 활용하여 머신러닝 모델의 성능을 향상시키는 피처(feature)를 생성, 변환, 선택하는 과정입니다. 이는 ML 파이프라인에서 가장 큰 영향을 미치는 단계로 — 좋은 피처는 단순한 모델이 원시 데이터로 훈련된 복잡한 모델보다 더 나은 성능을 내도록 만들 수 있습니다.

---

## 1. 피처 엔지니어링이 중요한 이유

### 1.1 피처가 모델 성능을 좌우한다

```python
"""
"피처를 만들어내는 것은 어렵고, 시간이 많이 걸리며, 전문 지식이 필요합니다.
응용 머신러닝은 기본적으로 피처 엔지니어링입니다." — Andrew Ng

ML 파이프라인:
  원시 데이터 → 피처 엔지니어링 → 모델 학습 → 예측
                  ↑ 가장 큰 영향

핵심 인사이트: 훌륭한 피처를 가진 단순한 모델이
빈약한 피처를 가진 복잡한 모델을 이기는 경우가 많습니다.
"""
```

### 1.2 피처 엔지니어링 워크플로우

| 단계 | 설명 | 예시 |
|------|------|------|
| **이해(Understand)** | 데이터 타입, 분포, 도메인 파악 | "나이는 오른쪽 치우침(right-skewed)" |
| **생성(Create)** | 기존 피처로부터 새 피처 도출 | `total_spend = price × quantity` |
| **변환(Transform)** | 피처 스케일 또는 분포 변경 | 로그 변환, 구간화(binning) |
| **인코딩(Encode)** | 비수치형을 수치형으로 변환 | 원-핫 인코딩, 타겟 인코딩 |
| **선택(Select)** | 유용한 피처만 유지 | 중복되거나 노이즈가 많은 피처 제거 |
| **검증(Validate)** | 데이터 누수(data leakage) 없음 확인 | 훈련셋 통계만 사용 |

---

## 2. 수치형 피처 변환

### 2.1 스케일링(Scaling)과 정규화(Normalization)

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    MaxAbsScaler, PowerTransformer, QuantileTransformer
)
import matplotlib.pyplot as plt

np.random.seed(42)

# Generate skewed data
data = np.random.exponential(scale=2, size=1000).reshape(-1, 1)

scalers = {
    'Original': None,
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler(),
    'PowerTransformer\n(Yeo-Johnson)': PowerTransformer(method='yeo-johnson'),
    'QuantileTransformer\n(Normal)': QuantileTransformer(output_distribution='normal'),
}

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, (name, scaler) in zip(axes.ravel(), scalers.items()):
    if scaler is None:
        transformed = data
    else:
        transformed = scaler.fit_transform(data)
    ax.hist(transformed, bins=50, edgecolor='black', alpha=0.7)
    ax.set_title(name)
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')

plt.tight_layout()
plt.savefig('scaling_comparison.png', dpi=150)
plt.show()
```

**각 스케일러 사용 시기**:

| 스케일러 | 사용 시기 | 비고 |
|---------|----------|------|
| `StandardScaler` | 피처가 대략 가우시안 분포일 때 | 이상값에 민감 |
| `MinMaxScaler` | [0, 1] 범위로 한정해야 할 때 | 이상값에 민감 |
| `RobustScaler` | 데이터에 이상값이 있을 때 | 중앙값과 IQR 사용 |
| `PowerTransformer` | 치우친 분포일 때 | 데이터를 가우시안에 가깝게 변환 |
| `QuantileTransformer` | 심하게 비가우시안 분포일 때 | 균일 또는 정규 분포로 강제 변환 |

### 2.2 로그(Log) 및 거듭제곱 변환

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# Simulate house prices (right-skewed)
prices = np.random.lognormal(mean=12, sigma=0.5, size=1000)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Original
axes[0].hist(prices, bins=50, edgecolor='black', alpha=0.7)
axes[0].set_title(f'Original (skew={pd.Series(prices).skew():.2f})')

# Log transform
log_prices = np.log1p(prices)  # log1p handles zeros: log(1 + x)
axes[1].hist(log_prices, bins=50, edgecolor='black', alpha=0.7, color='orange')
axes[1].set_title(f'Log Transform (skew={pd.Series(log_prices).skew():.2f})')

# Square root transform
sqrt_prices = np.sqrt(prices)
axes[2].hist(sqrt_prices, bins=50, edgecolor='black', alpha=0.7, color='green')
axes[2].set_title(f'Square Root (skew={pd.Series(sqrt_prices).skew():.2f})')

plt.tight_layout()
plt.savefig('transform_comparison.png', dpi=150)
plt.show()
```

### 2.3 구간화(Binning, Discretization)

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

np.random.seed(42)
ages = np.random.normal(loc=35, scale=12, size=500).clip(18, 70)
df = pd.DataFrame({'age': ages})

# Method 1: Equal-width bins
df['age_equal_width'] = pd.cut(df['age'], bins=5, labels=['Very Young', 'Young', 'Middle', 'Senior', 'Elder'])

# Method 2: Quantile bins (equal frequency)
df['age_quantile'] = pd.qcut(df['age'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

# Method 3: Domain-knowledge bins
bins = [0, 25, 35, 45, 55, 100]
labels = ['18-25', '26-35', '36-45', '46-55', '55+']
df['age_custom'] = pd.cut(df['age'], bins=bins, labels=labels)

# Method 4: KBinsDiscretizer with different strategies
kbd = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='kmeans')
df['age_kmeans'] = kbd.fit_transform(df[['age']])

print(df[['age', 'age_equal_width', 'age_quantile', 'age_custom']].head(10))
print("\nCustom bin distribution:")
print(df['age_custom'].value_counts().sort_index())
```

### 2.4 상호작용(Interaction) 및 다항식(Polynomial) 피처

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

# Example: house features
data = {
    'length': [10, 15, 20, 12, 18],
    'width': [8, 10, 12, 9, 11],
    'floors': [1, 2, 1, 2, 3],
}
df = pd.DataFrame(data)

# Manual interaction features
df['area'] = df['length'] * df['width']             # Interaction
df['volume'] = df['length'] * df['width'] * df['floors']  # 3-way interaction
df['length_per_floor'] = df['length'] / df['floors']  # Ratio

# Polynomial features (degree=2)
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(df[['length', 'width']])
poly_names = poly.get_feature_names_out(['length', 'width'])

print("Polynomial feature names:", poly_names)
print("\nPolynomial features (first 3 rows):")
print(pd.DataFrame(X_poly, columns=poly_names).head(3))

# Interaction-only features (no squared terms)
poly_interact = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_interact = poly_interact.fit_transform(df[['length', 'width', 'floors']])
interact_names = poly_interact.get_feature_names_out(['length', 'width', 'floors'])
print("\nInteraction-only features:", interact_names)
```

---

## 3. 범주형 피처 인코딩(Categorical Feature Encoding)

### 3.1 기본 인코딩 방법 복습

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder

data = {
    'color': ['red', 'blue', 'green', 'red', 'blue', 'green', 'red'],
    'size': ['S', 'M', 'L', 'XL', 'M', 'S', 'L'],
    'target': [1, 0, 1, 1, 0, 0, 1]
}
df = pd.DataFrame(data)

# Ordinal encoding (for ordered categories)
size_order = [['S', 'M', 'L', 'XL']]
oe = OrdinalEncoder(categories=size_order)
df['size_ordinal'] = oe.fit_transform(df[['size']])

# One-hot encoding (for nominal categories)
ohe = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' avoids multicollinearity
color_encoded = ohe.fit_transform(df[['color']])
color_cols = ohe.get_feature_names_out(['color'])
df[color_cols] = color_encoded

print(df)
```

### 3.2 타겟 인코딩(Target Encoding)

```python
import numpy as np
import pandas as pd

class TargetEncoder:
    """
    Target encoding with smoothing to prevent overfitting.
    Formula: encoding = (n * category_mean + m * global_mean) / (n + m)
    where m is the smoothing factor.
    """
    def __init__(self, smoothing=10):
        self.smoothing = smoothing
        self.global_mean = None
        self.encoding_map = {}

    def fit(self, X, y):
        self.global_mean = y.mean()
        for col in X.columns:
            stats = pd.DataFrame({'value': X[col], 'target': y})
            agg = stats.groupby('value')['target'].agg(['mean', 'count'])

            # Apply smoothing
            smooth = (agg['count'] * agg['mean'] + self.smoothing * self.global_mean) / (agg['count'] + self.smoothing)
            self.encoding_map[col] = smooth.to_dict()
        return self

    def transform(self, X):
        X_encoded = X.copy()
        for col in X.columns:
            X_encoded[col] = X[col].map(self.encoding_map[col])
            X_encoded[col].fillna(self.global_mean, inplace=True)
        return X_encoded

# Example
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix'], n),
    'target': np.random.binomial(1, 0.3, n)
})

# Manually bias some cities
df.loc[df['city'] == 'NYC', 'target'] = np.random.binomial(1, 0.7, (df['city'] == 'NYC').sum())
df.loc[df['city'] == 'LA', 'target'] = np.random.binomial(1, 0.2, (df['city'] == 'LA').sum())

encoder = TargetEncoder(smoothing=10)
encoder.fit(df[['city']], df['target'])
df['city_encoded'] = encoder.transform(df[['city']])['city']

print("Target encoding results:")
print(df.groupby('city').agg(
    target_mean=('target', 'mean'),
    encoded_value=('city_encoded', 'first'),
    count=('target', 'count')
))
```

**중요**: 타겟 인코딩(target encoding)은 데이터 누수(data leakage)를 방지하기 위해 훈련 세트에서만 계산해야 합니다. 훈련 세트에는 교차 검증 기반 인코딩(KFold)을 사용하세요.

### 3.3 빈도(Frequency) 및 횟수(Count) 인코딩

```python
import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    'browser': np.random.choice(
        ['Chrome', 'Safari', 'Firefox', 'Edge', 'Opera'],
        size=1000,
        p=[0.45, 0.25, 0.15, 0.10, 0.05]
    )
})

# Frequency encoding (proportion)
freq_map = df['browser'].value_counts(normalize=True).to_dict()
df['browser_freq'] = df['browser'].map(freq_map)

# Count encoding (raw count)
count_map = df['browser'].value_counts().to_dict()
df['browser_count'] = df['browser'].map(count_map)

print("Frequency encoding:")
print(df.drop_duplicates('browser').sort_values('browser_freq', ascending=False))
```

### 3.4 해시 인코딩(Hash Encoding)

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction import FeatureHasher

# High-cardinality feature (e.g., user IDs, zip codes)
data = pd.DataFrame({
    'zip_code': [f'{code:05d}' for code in np.random.randint(10000, 99999, 100)]
})

# Hash encoding: maps to fixed-size feature space
hasher = FeatureHasher(n_features=16, input_type='string')
hash_features = hasher.transform(data['zip_code'].apply(lambda x: [x]))

print(f"Original unique values: {data['zip_code'].nunique()}")
print(f"Hash feature dimensions: {hash_features.shape}")
print(f"\nFirst 3 rows of hash features:")
print(pd.DataFrame(hash_features.toarray()[:3], columns=[f'hash_{i}' for i in range(16)]))
```

**인코딩 방법 선택 가이드**:

| 방법 | 카디널리티(Cardinality) | 순서 보존 | 데이터 누수 위험 | 사용 사례 |
|------|------------------------|----------|-----------------|----------|
| 원-핫(One-Hot) | 낮음 (<15) | 아니오 | 없음 | 명목형 범주 |
| 순서형(Ordinal) | 임의 | 예 | 없음 | 순서 있는 범주 |
| 타겟(Target) | 중간-높음 | 아니오 | 높음 (CV 필요) | 분류/회귀 |
| 빈도(Frequency) | 임의 | 아니오 | 낮음 | 빈도가 정보적일 때 |
| 해시(Hash) | 매우 높음 | 아니오 | 없음 | 고카디널리티 피처 |

---

## 4. 날짜/시간 피처 엔지니어링

### 4.1 기본 시간적 피처

```python
import pandas as pd
import numpy as np

# Generate sample timestamp data
np.random.seed(42)
dates = pd.date_range('2023-01-01', '2023-12-31', freq='h')
df = pd.DataFrame({
    'timestamp': np.random.choice(dates, 1000),
    'sales': np.random.exponential(100, 1000)
})

# Extract temporal components
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
df['day'] = df['timestamp'].dt.day
df['day_of_week'] = df['timestamp'].dt.dayofweek      # 0=Monday
df['day_of_year'] = df['timestamp'].dt.dayofyear
df['hour'] = df['timestamp'].dt.hour
df['is_weekend'] = df['timestamp'].dt.dayofweek.isin([5, 6]).astype(int)
df['quarter'] = df['timestamp'].dt.quarter
df['is_month_start'] = df['timestamp'].dt.is_month_start.astype(int)
df['is_month_end'] = df['timestamp'].dt.is_month_end.astype(int)

print("Temporal features:")
print(df[['timestamp', 'month', 'day_of_week', 'hour', 'is_weekend', 'quarter']].head(10))
```

### 4.2 순환 인코딩(Cyclical Encoding)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Problem: Month 12 and Month 1 are far apart in ordinal encoding
# Solution: Encode cyclical features using sin/cos

df = pd.DataFrame({'hour': range(24)})

# Cyclical encoding
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Linear encoding: distance(23, 0) = 23
axes[0].plot(df['hour'], marker='o')
axes[0].set_title('Linear Encoding: hour 23 and 0 are far apart')
axes[0].set_xlabel('Index')
axes[0].set_ylabel('Hour Value')

# Cyclical encoding: distance(23, 0) ≈ 0.26
axes[1].scatter(df['hour_sin'], df['hour_cos'], c=df['hour'], cmap='twilight')
for i in range(0, 24, 3):
    axes[1].annotate(f'{i}h', (df['hour_sin'][i], df['hour_cos'][i]),
                     fontsize=9, ha='center', va='bottom')
axes[1].set_title('Cyclical Encoding: hour 23 and 0 are adjacent')
axes[1].set_xlabel('sin(hour)')
axes[1].set_ylabel('cos(hour)')
axes[1].set_aspect('equal')

plt.tight_layout()
plt.savefig('cyclical_encoding.png', dpi=150)
plt.show()

# General function for cyclical encoding
def cyclical_encode(series, period):
    """Encode a cyclical feature using sin/cos."""
    return pd.DataFrame({
        f'{series.name}_sin': np.sin(2 * np.pi * series / period),
        f'{series.name}_cos': np.cos(2 * np.pi * series / period),
    })

# Apply to month and day_of_week
months = pd.Series(range(1, 13), name='month')
print(cyclical_encode(months, period=12))
```

### 4.3 래그(Lag) 및 롤링(Rolling) 피처

```python
import pandas as pd
import numpy as np

# Simulate daily sales
np.random.seed(42)
dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
sales = 100 + 20 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365) + np.random.normal(0, 10, len(dates))
df = pd.DataFrame({'date': dates, 'sales': sales})

# Lag features (past values as features)
for lag in [1, 7, 14, 28]:
    df[f'sales_lag_{lag}'] = df['sales'].shift(lag)

# Rolling statistics
for window in [7, 14, 30]:
    df[f'sales_rolling_mean_{window}'] = df['sales'].shift(1).rolling(window).mean()
    df[f'sales_rolling_std_{window}'] = df['sales'].shift(1).rolling(window).std()

# Expanding statistics
df['sales_expanding_mean'] = df['sales'].shift(1).expanding().mean()

# Percentage change
df['sales_pct_change'] = df['sales'].pct_change()

# Difference
df['sales_diff_1'] = df['sales'].diff(1)
df['sales_diff_7'] = df['sales'].diff(7)

print("Lag and rolling features (last 5 rows):")
print(df[['date', 'sales', 'sales_lag_1', 'sales_lag_7',
          'sales_rolling_mean_7', 'sales_rolling_std_7']].tail())
```

**경고**: 데이터 누수를 방지하기 위해 `.rolling()` 전에 항상 `.shift(1)`을 사용하세요 — 롤링 통계는 현재 행의 값을 포함해서는 안 됩니다.

---

## 5. 텍스트 피처 엔지니어링

### 5.1 기본 텍스트 피처

```python
import pandas as pd
import numpy as np

texts = pd.DataFrame({
    'review': [
        "This product is absolutely amazing! I love it.",
        "Terrible quality. Waste of money.",
        "Good value for the price. Would recommend.",
        "Not great, not terrible. Average product.",
        "BEST PURCHASE EVER!!! 5 stars!!!",
    ]
})

# Basic text statistics
texts['char_count'] = texts['review'].str.len()
texts['word_count'] = texts['review'].str.split().str.len()
texts['avg_word_length'] = texts['review'].apply(
    lambda x: np.mean([len(w) for w in x.split()])
)
texts['uppercase_count'] = texts['review'].str.count(r'[A-Z]')
texts['exclamation_count'] = texts['review'].str.count('!')
texts['question_count'] = texts['review'].str.count(r'\?')
texts['has_uppercase_word'] = texts['review'].apply(
    lambda x: int(any(w.isupper() and len(w) > 1 for w in x.split()))
)

print("Text statistics:")
print(texts[['review', 'word_count', 'avg_word_length', 'exclamation_count']])
```

### 5.2 TF-IDF 피처

```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd

documents = [
    "machine learning is a subset of artificial intelligence",
    "deep learning uses neural networks for learning",
    "natural language processing deals with text data",
    "computer vision processes image and video data",
    "reinforcement learning trains agents with rewards",
]

# TF-IDF
tfidf = TfidfVectorizer(max_features=20, stop_words='english')
tfidf_matrix = tfidf.fit_transform(documents)

tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf.get_feature_names_out(),
    index=[f'doc_{i}' for i in range(len(documents))]
)

print("TF-IDF features:")
print(tfidf_df.round(3))

# N-gram features
bigram_vec = CountVectorizer(ngram_range=(2, 2), max_features=10)
bigram_matrix = bigram_vec.fit_transform(documents)
print("\nBigram features:", bigram_vec.get_feature_names_out())
```

---

## 6. 도메인 특화 피처 엔지니어링

### 6.1 전자상거래(E-Commerce) 피처

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n_users = 100

# Simulate user transaction data
transactions = pd.DataFrame({
    'user_id': np.random.randint(1, n_users + 1, 1000),
    'amount': np.random.exponential(50, 1000),
    'timestamp': pd.date_range('2023-01-01', periods=1000, freq='8h'),
    'category': np.random.choice(['electronics', 'clothing', 'food', 'books'], 1000),
})

# RFM Features (Recency, Frequency, Monetary)
reference_date = transactions['timestamp'].max()
rfm = transactions.groupby('user_id').agg(
    recency=('timestamp', lambda x: (reference_date - x.max()).days),
    frequency=('user_id', 'count'),
    monetary=('amount', 'sum'),
    avg_order_value=('amount', 'mean'),
    std_order_value=('amount', 'std'),
    max_order=('amount', 'max'),
    min_order=('amount', 'min'),
    unique_categories=('category', 'nunique'),
).reset_index()

# Time-based features
time_features = transactions.groupby('user_id').agg(
    days_since_first=('timestamp', lambda x: (reference_date - x.min()).days),
    days_between_orders=('timestamp', lambda x: x.diff().mean().days if len(x) > 1 else 0),
).reset_index()

user_features = rfm.merge(time_features, on='user_id')

# Derived ratios
user_features['orders_per_day'] = user_features['frequency'] / user_features['days_since_first'].clip(1)
user_features['monetary_per_category'] = user_features['monetary'] / user_features['unique_categories']

print("User features (first 5):")
print(user_features.head())
```

### 6.2 지리공간(Geospatial) 피처

```python
import numpy as np
import pandas as pd

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points (km)."""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# Example: distance from key landmarks
stores = pd.DataFrame({
    'store_id': range(1, 6),
    'lat': [40.7128, 34.0522, 41.8781, 29.7604, 33.4484],
    'lon': [-74.0060, -118.2437, -87.6298, -95.3698, -112.0740],
    'city': ['NYC', 'LA', 'Chicago', 'Houston', 'Phoenix'],
})

# Distance to NYC (central point)
center_lat, center_lon = 40.7128, -74.0060
stores['distance_to_nyc_km'] = haversine_distance(
    stores['lat'], stores['lon'], center_lat, center_lon
)

# Cluster distance features
stores['lat_bin'] = pd.cut(stores['lat'], bins=3, labels=['South', 'Central', 'North'])

print("Geospatial features:")
print(stores)
```

---

## 7. 자동화된 피처 엔지니어링

### 7.1 feature_engine 사용

```python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load data
housing = fetch_california_housing(as_frame=True)
X, y = housing.data, housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# feature_engine provides sklearn-compatible transformers
# pip install feature_engine

# Example transformers (conceptual — install feature_engine to run):
"""
from feature_engine.creation import MathFeatures, CyclicalFeatures
from feature_engine.encoding import MeanEncoder, WoEEncoder
from feature_engine.selection import (
    DropCorrelatedFeatures,
    SmartCorrelatedSelection,
    SelectByShuffling,
)

# Create math features (combinations of existing features)
math_tf = MathFeatures(
    variables=['MedInc', 'AveRooms', 'AveOccup'],
    func=['sum', 'mean', 'std', 'min', 'max'],
)
X_math = math_tf.fit_transform(X_train)
print(f"Math features added: {math_tf.get_feature_names_out()}")

# Drop highly correlated features
corr_sel = DropCorrelatedFeatures(threshold=0.9)
X_reduced = corr_sel.fit_transform(X_train)
print(f"Dropped: {corr_sel.features_to_drop_}")
"""
```

### 7.2 자동 피처 생성을 위한 Featuretools

```python
"""
Featuretools automates feature engineering using Deep Feature Synthesis (DFS).
It generates features from relational datasets using aggregation and transformation primitives.

# pip install featuretools

import featuretools as ft

# Define entities (tables)
customers = pd.DataFrame({
    'customer_id': [1, 2, 3],
    'signup_date': pd.to_datetime(['2022-01-01', '2022-06-15', '2023-01-01']),
    'age': [25, 35, 45],
})

transactions = pd.DataFrame({
    'transaction_id': range(1, 11),
    'customer_id': [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
    'amount': [50, 100, 75, 200, 150, 50, 300, 25, 75, 50],
    'timestamp': pd.date_range('2023-01-01', periods=10, freq='W'),
})

# Create an EntitySet
es = ft.EntitySet(id='ecommerce')
es = es.add_dataframe(dataframe=customers, dataframe_name='customers',
                      index='customer_id', time_index='signup_date')
es = es.add_dataframe(dataframe=transactions, dataframe_name='transactions',
                      index='transaction_id', time_index='timestamp')
es = es.add_relationship('customers', 'customer_id', 'transactions', 'customer_id')

# Run Deep Feature Synthesis
feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_dataframe_name='customers',
    max_depth=2,  # How many levels of primitives to stack
    agg_primitives=['mean', 'sum', 'count', 'std', 'max', 'min'],
    trans_primitives=['month', 'year', 'day'],
)

print(f"Generated {len(feature_defs)} features:")
for fd in feature_defs[:10]:
    print(f"  {fd}")
"""
```

---

## 8. 피처 선택(Feature Selection)

### 8.1 필터 방법(Filter Methods)

```python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.feature_selection import (
    mutual_info_regression, f_regression, SelectKBest
)
from sklearn.model_selection import train_test_split

# Load data
housing = fetch_california_housing(as_frame=True)
X, y = housing.data, housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Method 1: Correlation with target
correlations = X_train.corrwith(pd.Series(y_train, index=X_train.index)).abs().sort_values(ascending=False)
print("Correlation with target:")
print(correlations)

# Method 2: Mutual Information
mi_scores = mutual_info_regression(X_train, y_train, random_state=42)
mi_df = pd.Series(mi_scores, index=X_train.columns).sort_values(ascending=False)
print("\nMutual Information scores:")
print(mi_df)

# Method 3: F-test (linear relationship)
f_scores, p_values = f_regression(X_train, y_train)
f_df = pd.DataFrame({
    'F-score': f_scores,
    'p-value': p_values
}, index=X_train.columns).sort_values('F-score', ascending=False)
print("\nF-test scores:")
print(f_df)

# Select top K features
selector = SelectKBest(score_func=mutual_info_regression, k=5)
X_selected = selector.fit_transform(X_train, y_train)
selected_features = X_train.columns[selector.get_support()].tolist()
print(f"\nTop 5 features (MI): {selected_features}")
```

### 8.2 래퍼 방법(Wrapper Methods, RFE)

```python
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Recursive Feature Elimination with Cross-Validation
estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
rfecv = RFECV(
    estimator=estimator,
    step=1,
    cv=5,
    scoring='r2',
    min_features_to_select=2,
    n_jobs=-1,
)
rfecv.fit(X_train, y_train)

print(f"Optimal number of features: {rfecv.n_features_}")
print(f"Selected features: {X_train.columns[rfecv.support_].tolist()}")
print(f"Feature ranking: {dict(zip(X_train.columns, rfecv.ranking_))}")

# Plot CV scores vs number of features
plt.figure(figsize=(8, 5))
n_features_range = range(rfecv.min_features_to_select, len(rfecv.cv_results_['mean_test_score']) + rfecv.min_features_to_select)
plt.plot(n_features_range, rfecv.cv_results_['mean_test_score'], marker='o')
plt.fill_between(
    n_features_range,
    rfecv.cv_results_['mean_test_score'] - rfecv.cv_results_['std_test_score'],
    rfecv.cv_results_['mean_test_score'] + rfecv.cv_results_['std_test_score'],
    alpha=0.2
)
plt.axvline(rfecv.n_features_, color='r', linestyle='--', label=f'Optimal: {rfecv.n_features_}')
plt.xlabel('Number of Features')
plt.ylabel('Cross-Validation R² Score')
plt.title('RFECV: Optimal Feature Count')
plt.legend()
plt.tight_layout()
plt.savefig('rfecv_scores.png', dpi=150)
plt.show()
```

### 8.3 임베디드 방법(Embedded Methods)

```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt
import numpy as np

# Method 1: L1 Regularization (Lasso)
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_train, y_train)
lasso_importance = pd.Series(np.abs(lasso.coef_), index=X_train.columns).sort_values(ascending=True)

# Method 2: Tree-based importance
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_importance = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=True)

# Comparison plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

lasso_importance.plot(kind='barh', ax=axes[0], color='steelblue')
axes[0].set_title('Lasso (L1) Feature Importance')
axes[0].set_xlabel('|Coefficient|')

rf_importance.plot(kind='barh', ax=axes[1], color='forestgreen')
axes[1].set_title('Random Forest Feature Importance')
axes[1].set_xlabel('Impurity-based Importance')

plt.tight_layout()
plt.savefig('feature_importance_comparison.png', dpi=150)
plt.show()

# Identify zero-importance features (Lasso)
zero_features = lasso_importance[lasso_importance == 0].index.tolist()
print(f"Features with zero Lasso coefficient (can be dropped): {zero_features}")
```

---

## 9. 피처 엔지니어링에서 데이터 누수 방지

### 9.1 일반적인 누수 패턴

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

np.random.seed(42)
X = np.random.randn(500, 5)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# WRONG: Fit scaler on entire dataset before split
scaler_wrong = StandardScaler()
X_scaled_wrong = scaler_wrong.fit_transform(X)  # Uses test set statistics!
X_train_wrong = X_scaled_wrong[:400]
X_test_wrong = X_scaled_wrong[400:]

# CORRECT: Fit scaler on training set only
scaler_correct = StandardScaler()
X_train_correct = scaler_correct.fit_transform(X_train)
X_test_correct = scaler_correct.transform(X_test)  # Only transform, don't fit!

# BEST: Use Pipeline (automatically handles train/test separation)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression()),
])
scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
print(f"Pipeline CV accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### 9.2 누수 체크리스트

```python
"""
Feature Engineering Leakage Checklist:

1. SCALING / ENCODING
   ✗ Fit scaler on entire dataset before splitting
   ✓ Fit scaler on training set, transform both train and test

2. TARGET ENCODING
   ✗ Compute target statistics using all data
   ✓ Use KFold target encoding on training set, apply to test set

3. MISSING VALUE IMPUTATION
   ✗ Impute using global mean/median (includes test data)
   ✓ Impute using training set mean/median only

4. FEATURE SELECTION
   ✗ Select features using entire dataset
   ✓ Select features using training set, apply selection to test set

5. TIME SERIES
   ✗ Include future data in lag/rolling features
   ✓ Use only past data (shift before rolling, respect temporal order)

6. DOMAIN FEATURES
   ✗ Use features that contain target information (e.g., "was_returned" for predicting returns)
   ✓ Only use features available at prediction time

Rule of thumb: At prediction time, would this feature be available?
If not, it's leakage.
"""
```

---

## 10. 완전한 피처 엔지니어링 파이프라인

### 10.1 엔드-투-엔드 예제

```python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, PowerTransformer, OneHotEncoder,
    FunctionTransformer
)
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load data
housing = fetch_california_housing(as_frame=True)
X, y = housing.data, housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Custom feature engineering function
def create_features(X):
    """Create domain-specific features for housing data."""
    X = X.copy()
    X['rooms_per_household'] = X['AveRooms'] / X['AveOccup'].clip(lower=0.1)
    X['bedrooms_ratio'] = X['AveBedrms'] / X['AveRooms'].clip(lower=0.1)
    X['population_density'] = X['Population'] / X['AveOccup'].clip(lower=0.1)
    X['income_per_room'] = X['MedInc'] / X['AveRooms'].clip(lower=0.1)
    return X

# Feature engineering transformer (works in pipeline)
feature_creator = FunctionTransformer(create_features, validate=False)

# Define column groups (after feature creation)
original_numeric = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                    'Population', 'AveOccup', 'Latitude', 'Longitude']
created_numeric = ['rooms_per_household', 'bedrooms_ratio',
                   'population_density', 'income_per_room']
all_numeric = original_numeric + created_numeric

# Build pipeline
preprocessor = ColumnTransformer([
    ('power', PowerTransformer(method='yeo-johnson'), ['MedInc', 'Population', 'AveOccup']),
    ('standard', StandardScaler(), ['HouseAge', 'AveRooms', 'AveBedrms',
                                     'Latitude', 'Longitude',
                                     'rooms_per_household', 'bedrooms_ratio',
                                     'population_density', 'income_per_room']),
], remainder='drop')

pipeline = Pipeline([
    ('features', feature_creator),
    ('preprocessor', preprocessor),
    ('model', GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
    )),
])

# Cross-validation
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
print(f"CV R² scores: {cv_scores}")
print(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Fit and evaluate
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(f"\nTest R²: {r2_score(y_test, y_pred):.4f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

# Compare with baseline (no feature engineering)
baseline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
    )),
])
baseline_scores = cross_val_score(baseline, X_train, y_train, cv=5, scoring='r2')
print(f"\nBaseline CV R²: {baseline_scores.mean():.4f} (+/- {baseline_scores.std():.4f})")
print(f"Improvement: {(cv_scores.mean() - baseline_scores.mean()):.4f}")
```

---

## 11. 연습 문제

### 연습 1: 고객 이탈(Customer Churn) 피처 엔지니어링

```python
"""
주어진 통신사 데이터셋 열:
- customer_id, tenure (months), monthly_charges, total_charges
- contract_type (Month-to-month, One year, Two year)
- internet_service (DSL, Fiber optic, No)
- payment_method (Electronic check, Mailed check, Bank transfer, Credit card)
- churn (Yes/No) — 타겟

최소 10개의 의미 있는 피처를 생성하세요:
1. 비율 피처 (예: avg_monthly = total_charges / tenure)
2. 구간화 피처 (예: tenure_group)
3. 인코딩 피처 (contract_type에 대한 타겟 인코딩)
4. 상호작용 피처
5. 도메인 특화 피처

파이프라인을 구축하고 베이스라인 대비 엔지니어링된 피처를 비교하세요.
"""
```

### 연습 2: 시계열 피처 추출

```python
"""
주어진 일별 웹사이트 트래픽 데이터:
- date, page_views, unique_visitors, bounce_rate, avg_session_duration

시간적 및 롤링 피처를 생성하세요:
1. 요일, 월, 분기 (순환 인코딩 포함)
2. 래그 피처 (1, 7, 14, 28일)
3. 롤링 평균과 표준편차 (7, 14, 30일)
4. 전년 동기 대비 변화율
5. 공휴일 플래그

엔지니어링된 피처를 사용해 다음날 page_views를 예측하세요.
"""
```

### 연습 3: 텍스트 + 수치형 결합 피처

```python
"""
주어진 제품 데이터셋:
- product_name (text), description (text), price, category, rating, num_reviews

결합 피처를 생성하세요:
1. 텍스트 피처: name_length, description_length, has_discount_word, tfidf_top_10
2. 수치형: price_per_rating, log_num_reviews, price_rank_in_category
3. 범주형: 빈도 인코딩, rating을 위한 타겟 인코딩
4. 복합: description_richness = description_length / price

rating 예측 모델을 구축하세요. 원시 피처와 비교하세요.
"""
```

---

## 12. 요약

### 핵심 정리

| 개념 | 설명 |
|------|------|
| **수치형 변환(Numerical transforms)** | 로그, 거듭제곱, 구간화, 다항식 — 모델 가정에 맞게 분포 조정 |
| **범주형 인코딩(Categorical encoding)** | 낮은 카디널리티에는 원-핫, 높은 카디널리티에는 타겟/빈도/해시 |
| **시간적 피처(Temporal features)** | 주기적 피처에 순환 인코딩, 시계열에 래그/롤링 |
| **텍스트 피처(Text features)** | 통계량 (횟수, 길이), TF-IDF, n-gram |
| **도메인 피처(Domain features)** | RFM, 비율, 집계 — 도메인 지식은 대체 불가 |
| **자동화 FE(Automated FE)** | feature_engine, Featuretools — 자동화하되 검증 필수 |
| **피처 선택(Feature selection)** | 필터 (MI, F-test), 래퍼 (RFE), 임베디드 (Lasso, 트리 중요도) |
| **데이터 누수(Data leakage)** | 항상 훈련 세트에서만 적합(fit); 안전을 위해 Pipeline 사용 |

### 모범 사례

1. **단순하게 시작**: 기본 피처부터, 그 다음 복잡도 추가
2. **데이터 이해**: 피처 엔지니어링 전에 탐색적 데이터 분석(EDA) 수행
3. **교차 검증으로 검증**: 피처 영향을 확인하기 위해 항상 교차 검증
4. **누수 방지**: 모든 변환에 sklearn `Pipeline` 사용
5. **피처 문서화**: 각 피처의 의미와 계산 방법 추적
6. **반복적으로 개선**: 피처 엔지니어링은 반복적인 과정 — 실험하고 평가

### 다음 단계

- **L16**: 모델 설명 가능성(Model Explainability) — 어떤 피처가 예측을 이끄는지 이해 (SHAP, LIME)
- **L17**: 불균형 데이터(Imbalanced Data) — 샘플링과 비용 민감 방법으로 클래스 불균형 처리
