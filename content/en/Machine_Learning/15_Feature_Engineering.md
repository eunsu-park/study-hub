# Feature Engineering

**Previous**: [Practical Projects](./14_Practical_Projects.md) | **Next**: [Model Explainability](./16_Model_Explainability.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why feature engineering is often more impactful than model selection
2. Apply numerical transformations (log, power, binning, polynomial) to match model assumptions
3. Implement categorical encoding strategies (one-hot, target, frequency, hash) and choose the right one based on cardinality
4. Engineer temporal features using cyclical encoding, lag values, and rolling statistics
5. Extract text features using TF-IDF, n-grams, and basic text statistics
6. Identify and prevent data leakage during feature engineering using sklearn Pipeline
7. Build a complete feature engineering pipeline that improves model performance over a raw-feature baseline

---

The difference between a mediocre model and a competition-winning one is rarely the algorithm -- it is the features. Feature engineering is the art and science of transforming raw data into representations that expose the underlying patterns a model needs to learn. A date becomes "day of week" and "is_holiday"; an address becomes distance-to-downtown; raw text becomes TF-IDF vectors. This lesson equips you with a comprehensive toolkit of transformations so you can make any model work harder for you.

> **Cooking preparation before the recipe.** A chef doesn't just throw raw ingredients into a pot -- they wash, peel, chop, marinate, and measure first. Feature engineering is the "mise en place" of machine learning: transforming raw data into ingredients the model can actually use. A date becomes "day of week" and "is_holiday"; an address becomes distance-to-downtown; raw text becomes TF-IDF vectors. Often, clever feature engineering matters more than choosing a fancier algorithm -- the best model can't learn patterns that aren't represented in the features.

---

## 1. Why Feature Engineering Matters

### 1.1 Features Drive Model Performance

```python
"""
"Coming up with features is difficult, time-consuming, requires expert knowledge.
Applied machine learning is basically feature engineering." — Andrew Ng

The ML pipeline:
  Raw Data → Feature Engineering → Model Training → Predictions
               ↑ Most impact here

Key insight: A simple model with great features often beats
a complex model with poor features.
"""
```

### 1.2 Feature Engineering Workflow

| Step | Description | Example |
|------|-------------|---------|
| **Understand** | Know data types, distributions, domain | "Age is right-skewed" |
| **Create** | Derive new features from existing ones | `total_spend = price × quantity` |
| **Transform** | Change feature scale or distribution | Log transform, binning |
| **Encode** | Convert non-numeric to numeric | One-hot, target encoding |
| **Select** | Keep only useful features | Remove redundant or noisy features |
| **Validate** | Ensure no data leakage | Use only training set statistics |

---

## 2. Numerical Feature Transformations

### 2.1 Scaling and Normalization

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

**When to use each scaler**:

| Scaler | Use When | Note |
|--------|----------|------|
| `StandardScaler` | Features are roughly Gaussian | Sensitive to outliers |
| `MinMaxScaler` | Need bounded [0, 1] range | Sensitive to outliers |
| `RobustScaler` | Data has outliers | Uses median and IQR |
| `PowerTransformer` | Skewed distributions | Makes data more Gaussian |
| `QuantileTransformer` | Heavily non-Gaussian | Forces uniform or normal distribution |

### 2.2 Log and Power Transforms

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

### 2.3 Binning (Discretization)

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

### 2.4 Interaction and Polynomial Features

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

## 3. Categorical Feature Encoding

### 3.1 Review of Basic Encodings

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

### 3.2 Target Encoding

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

**Important**: Target encoding must be computed on the training set only to avoid data leakage. Use cross-validation-based encoding (KFold) for the training set.

### 3.3 Frequency and Count Encoding

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

### 3.4 Hash Encoding

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

**Encoding method selection guide**:

| Method | Cardinality | Preserves Order | Data Leakage Risk | Use Case |
|--------|------------|-----------------|-------------------|----------|
| One-Hot | Low (<15) | No | None | Nominal categories |
| Ordinal | Any | Yes | None | Ordered categories |
| Target | Medium-High | No | High (needs CV) | Classification/regression |
| Frequency | Any | No | Low | When frequency is informative |
| Hash | Very High | No | None | High-cardinality features |

---

## 4. Date/Time Feature Engineering

### 4.1 Basic Temporal Features

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

### 4.2 Cyclical Encoding

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

### 4.3 Lag and Rolling Features

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

**Warning**: Always use `.shift(1)` before `.rolling()` to prevent data leakage — rolling statistics should not include the current row's value.

---

## 5. Text Feature Engineering

### 5.1 Basic Text Features

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

### 5.2 TF-IDF Features

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

## 6. Domain-Specific Feature Engineering

### 6.1 E-Commerce Features

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

### 6.2 Geospatial Features

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

## 7. Automated Feature Engineering

### 7.1 Using feature_engine

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

### 7.2 Featuretools for Automated Feature Generation

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

## 8. Feature Selection

### 8.1 Filter Methods

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

### 8.2 Wrapper Methods (RFE)

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

### 8.3 Embedded Methods

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

## 9. Preventing Data Leakage in Feature Engineering

### 9.1 Common Leakage Patterns

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

### 9.2 Leakage Checklist

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

## 10. Complete Feature Engineering Pipeline

### 10.1 End-to-End Example

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

## 11. Practice Problems

### Exercise 1: Customer Churn Feature Engineering

```python
"""
Given a telecom dataset with the following columns:
- customer_id, tenure (months), monthly_charges, total_charges
- contract_type (Month-to-month, One year, Two year)
- internet_service (DSL, Fiber optic, No)
- payment_method (Electronic check, Mailed check, Bank transfer, Credit card)
- churn (Yes/No) — target

Create at least 10 meaningful features:
1. Ratio features (e.g., avg_monthly = total_charges / tenure)
2. Binned features (e.g., tenure_group)
3. Encoded features (target encoding for contract_type)
4. Interaction features
5. Domain-specific features

Build a pipeline and compare baseline vs. engineered features.
"""
```

### Exercise 2: Time Series Feature Extraction

```python
"""
Given daily website traffic data:
- date, page_views, unique_visitors, bounce_rate, avg_session_duration

Create temporal and rolling features:
1. Day of week, month, quarter (with cyclical encoding)
2. Lag features (1, 7, 14, 28 days)
3. Rolling mean and std (7, 14, 30 days)
4. Year-over-year change
5. Holiday flags

Predict next-day page_views using the engineered features.
"""
```

### Exercise 3: Text + Numeric Combined Features

```python
"""
Given a product dataset:
- product_name (text), description (text), price, category, rating, num_reviews

Create combined features:
1. Text features: name_length, description_length, has_discount_word, tfidf_top_10
2. Numeric: price_per_rating, log_num_reviews, price_rank_in_category
3. Category: frequency encoding, target encoding for rating
4. Combined: description_richness = description_length / price

Build a model to predict rating. Compare with raw features.
"""
```

---

## 12. Summary

### Key Takeaways

| Concept | Description |
|---------|-------------|
| **Numerical transforms** | Log, power, binning, polynomial — match distribution to model assumptions |
| **Categorical encoding** | One-hot for low cardinality, target/frequency/hash for high cardinality |
| **Temporal features** | Cyclical encoding for periodic features, lag/rolling for time series |
| **Text features** | Statistics (counts, lengths), TF-IDF, n-grams |
| **Domain features** | RFM, ratios, aggregations — domain knowledge is irreplaceable |
| **Automated FE** | feature_engine, Featuretools — automate but verify |
| **Feature selection** | Filter (MI, F-test), Wrapper (RFE), Embedded (Lasso, tree importance) |
| **Data leakage** | Always fit on train only; use Pipeline for safety |

### Best Practices

1. **Start simple**: Basic features first, then add complexity
2. **Understand your data**: EDA before feature engineering
3. **Validate with CV**: Always cross-validate to check feature impact
4. **Prevent leakage**: Use sklearn `Pipeline` for all transformations
5. **Document features**: Track what each feature means and how it's computed
6. **Iterate**: Feature engineering is an iterative process — experiment and evaluate

### Next Steps

- **L16**: Model Explainability — understand which features drive predictions (SHAP, LIME)
- **L17**: Imbalanced Data — handle class imbalance with sampling and cost-sensitive approaches
