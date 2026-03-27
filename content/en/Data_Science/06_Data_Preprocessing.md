# 6. Data Preprocessing

[Previous: Pandas Advanced](./05_Pandas_Advanced.md) | [Next: Descriptive Stats & EDA](./07_Descriptive_Stats_EDA.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Identify missing values using `isna` and visualize missing data patterns
2. Apply appropriate missing value strategies including deletion, imputation, and interpolation
3. Implement outlier detection using IQR, Z-score, and modified Z-score methods
4. Compare normalization techniques including Min-Max, Z-score, Robust, and Quantile scaling
5. Apply label encoding, one-hot encoding, ordinal encoding, and target encoding for categorical variables
6. Implement numerical transformations including log, Box-Cox, and Yeo-Johnson for skewed data
7. Extract date features and apply cyclic encoding for temporal variables
8. Build end-to-end preprocessing pipelines using scikit-learn's `Pipeline` and `ColumnTransformer`

---

Raw data is messy -- it contains missing entries, outliers, inconsistent formats, and features on wildly different scales. Feeding such data directly into an analysis or model produces unreliable results at best and complete failures at worst. Data preprocessing is the essential bridge between raw data collection and trustworthy analysis, and practitioners routinely spend the majority of their project time on this step.

---

## 1. Missing Value Handling

### 1.1 Identifying Missing Values

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'A': [1, 2, None, 4, 5],
    'B': [None, 2, 3, None, 5],
    'C': ['a', 'b', None, 'd', 'e'],
    'D': [1.0, 2.0, 3.0, 4.0, 5.0]
})

# Check for missing values
print(df.isna())          # boolean mask
print(df.isna().sum())    # missing count per column
print(df.isna().sum().sum())  # total missing count

# Missing value ratio
print(df.isna().mean() * 100)

# Rows/columns with missing values
print(df[df.isna().any(axis=1)])  # rows with missing values
print(df.columns[df.isna().any()])  # columns with missing values

# Visualize missing values (missingno library)
# import missingno as msno
# msno.matrix(df)
```

### 1.2 Dropping Missing Values

```python
df = pd.DataFrame({
    'A': [1, 2, None, 4],
    'B': [None, 2, 3, 4],
    'C': [1, None, 3, None]
})

# Drop rows with any missing value
print(df.dropna())

# Drop rows where all values are missing
print(df.dropna(how='all'))

# Based on specific columns
print(df.dropna(subset=['A']))
print(df.dropna(subset=['A', 'B']))

# Set threshold (minimum number of non-missing values)
print(df.dropna(thresh=2))  # at least 2 non-missing values
```

### 1.3 Imputing Missing Values

```python
df = pd.DataFrame({
    'numeric': [1, 2, None, 4, 5, None],
    'category': ['A', 'B', None, 'A', 'B', 'A']
})

# Fill with a specific value
df_filled = df.fillna(0)
df_filled = df.fillna({'numeric': 0, 'category': 'Unknown'})

# Fill with statistical values
df['numeric'] = df['numeric'].fillna(df['numeric'].mean())     # mean
df['numeric'] = df['numeric'].fillna(df['numeric'].median())   # median
df['category'] = df['category'].fillna(df['category'].mode()[0])  # mode

# Forward/backward fill
df_ffill = df.fillna(method='ffill')  # forward fill
df_bfill = df.fillna(method='bfill')  # backward fill

# Interpolation
df['numeric'] = df['numeric'].interpolate(method='linear')
df['numeric'] = df['numeric'].interpolate(method='polynomial', order=2)
```

### 1.4 Group-wise Missing Value Handling

```python
df = pd.DataFrame({
    'group': ['A', 'A', 'B', 'B', 'A', 'B'],
    'value': [1, None, 3, None, 5, 6]
})

# Fill with group mean
df['value'] = df.groupby('group')['value'].transform(
    lambda x: x.fillna(x.mean())
)
print(df)
```

---

## 2. Outlier Detection

### 2.1 Statistical Methods

```python
df = pd.DataFrame({
    'value': [10, 12, 11, 13, 100, 11, 12, 10, 9, 11]
})

# IQR method
Q1 = df['value'].quantile(0.25)
Q3 = df['value'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['value'] < lower_bound) | (df['value'] > upper_bound)]
print("Outliers:", outliers)

# Z-score method
from scipy import stats

z_scores = np.abs(stats.zscore(df['value']))
outliers = df[z_scores > 3]  # cases where |z| > 3
print("Outliers:", outliers)

# Modified Z-score (MAD-based)
median = df['value'].median()
mad = np.median(np.abs(df['value'] - median))
modified_z = 0.6745 * (df['value'] - median) / mad
outliers = df[np.abs(modified_z) > 3.5]
```

### 2.2 Visual Methods

```python
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'value': np.concatenate([np.random.randn(100), [10, -10]])
})

# Box plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].boxplot(df['value'])
axes[0].set_title('Box Plot')

# Histogram
axes[1].hist(df['value'], bins=30, edgecolor='black')
axes[1].set_title('Histogram')

plt.tight_layout()
plt.show()
```

### 2.3 Handling Outliers

```python
df = pd.DataFrame({
    'value': [10, 12, 11, 13, 100, 11, 12, 10, 9, -50]
})

# 1. Removal
Q1, Q3 = df['value'].quantile([0.25, 0.75])
IQR = Q3 - Q1
df_clean = df[(df['value'] >= Q1 - 1.5 * IQR) &
              (df['value'] <= Q3 + 1.5 * IQR)]

# 2. Replacement (clipping)
lower = df['value'].quantile(0.05)
upper = df['value'].quantile(0.95)
df['value_clipped'] = df['value'].clip(lower, upper)

# 3. Winsorizing
from scipy.stats import mstats
df['value_winsorized'] = mstats.winsorize(df['value'], limits=[0.05, 0.05])

# 4. Log transformation (for highly skewed data)
df['value_log'] = np.log1p(df['value'] - df['value'].min() + 1)
```

---

## 3. Data Normalization and Standardization

### 3.1 Min-Max Normalization

```python
df = pd.DataFrame({
    'A': [10, 20, 30, 40, 50],
    'B': [100, 200, 300, 400, 500]
})

# Manual implementation
df_normalized = (df - df.min()) / (df.max() - df.min())
print(df_normalized)

# Using sklearn
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_normalized = pd.DataFrame(
    scaler.fit_transform(df),
    columns=df.columns
)
print(df_normalized)
```

### 3.2 Standardization (Z-score)

```python
df = pd.DataFrame({
    'A': [10, 20, 30, 40, 50],
    'B': [100, 200, 300, 400, 500]
})

# Manual implementation
df_standardized = (df - df.mean()) / df.std()
print(df_standardized)

# Using sklearn
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_standardized = pd.DataFrame(
    scaler.fit_transform(df),
    columns=df.columns
)
print(df_standardized)
```

### 3.3 Various Scaling Methods

```python
from sklearn.preprocessing import (
    MinMaxScaler, StandardScaler, RobustScaler,
    MaxAbsScaler, QuantileTransformer, PowerTransformer
)

df = pd.DataFrame({
    'value': [1, 2, 3, 4, 5, 100]  # includes outlier
})

# RobustScaler (robust to outliers)
scaler = RobustScaler()  # uses median and IQR
robust_scaled = scaler.fit_transform(df)

# MaxAbsScaler (scales by absolute maximum)
scaler = MaxAbsScaler()
maxabs_scaled = scaler.fit_transform(df)

# QuantileTransformer (quantile-based)
scaler = QuantileTransformer(output_distribution='normal')
quantile_scaled = scaler.fit_transform(df)

# PowerTransformer (approximate normal distribution)
scaler = PowerTransformer(method='yeo-johnson')
power_scaled = scaler.fit_transform(df)
```

---

## 4. Categorical Variable Encoding

### 4.1 Label Encoding

```python
from sklearn.preprocessing import LabelEncoder

df = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'red', 'blue']
})

# sklearn LabelEncoder
le = LabelEncoder()
df['color_encoded'] = le.fit_transform(df['color'])
print(df)
print("Classes:", le.classes_)

# Inverse transform
original = le.inverse_transform(df['color_encoded'])
print("Original:", original)

# pandas factorize
codes, uniques = pd.factorize(df['color'])
df['color_factorized'] = codes
print(df)
```

### 4.2 One-Hot Encoding

```python
df = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'red'],
    'size': ['S', 'M', 'L', 'M']
})

# pandas get_dummies
df_encoded = pd.get_dummies(df, columns=['color', 'size'])
print(df_encoded)

# drop_first option (prevents multicollinearity)
df_encoded = pd.get_dummies(df, columns=['color'], drop_first=True)
print(df_encoded)

# sklearn OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded = encoder.fit_transform(df[['color', 'size']])
print(encoded)
print("Feature names:", encoder.get_feature_names_out())
```

### 4.3 Ordinal Encoding

```python
from sklearn.preprocessing import OrdinalEncoder

df = pd.DataFrame({
    'education': ['high school', 'bachelor', 'master', 'phd', 'bachelor']
})

# Define order
order = ['high school', 'bachelor', 'master', 'phd']

# sklearn OrdinalEncoder
encoder = OrdinalEncoder(categories=[order])
df['education_encoded'] = encoder.fit_transform(df[['education']])
print(df)

# pandas Categorical
df['education_cat'] = pd.Categorical(
    df['education'],
    categories=order,
    ordered=True
)
df['education_codes'] = df['education_cat'].cat.codes
print(df)
```

### 4.4 Frequency Encoding

```python
df = pd.DataFrame({
    'category': ['A', 'B', 'A', 'C', 'B', 'A', 'A', 'C']
})

# Compute frequency
freq_map = df['category'].value_counts() / len(df)
df['category_freq'] = df['category'].map(freq_map)
print(df)
```

### 4.5 Target Encoding

```python
df = pd.DataFrame({
    'category': ['A', 'B', 'A', 'C', 'B', 'A'],
    'target': [1, 0, 1, 0, 1, 1]
})

# Target mean per category
target_mean = df.groupby('category')['target'].mean()
df['category_target_encoded'] = df['category'].map(target_mean)
print(df)

# Smoothing to prevent overfitting
def target_encode_smoothed(df, col, target, weight=10):
    global_mean = df[target].mean()
    agg = df.groupby(col)[target].agg(['mean', 'count'])
    smoothed = (agg['count'] * agg['mean'] + weight * global_mean) / (agg['count'] + weight)
    return df[col].map(smoothed)

df['category_smoothed'] = target_encode_smoothed(df, 'category', 'target')
```

---

## 5. Numerical Transformations

### 5.1 Log Transformation

```python
df = pd.DataFrame({
    'value': [1, 10, 100, 1000, 10000]
})

# Log transformation
df['log'] = np.log(df['value'])
df['log10'] = np.log10(df['value'])
df['log1p'] = np.log1p(df['value'])  # log(1 + x), handles zero values

print(df)
```

### 5.2 Box-Cox / Yeo-Johnson Transformation

```python
from scipy import stats
from sklearn.preprocessing import PowerTransformer

df = pd.DataFrame({
    'value': [1, 2, 5, 10, 50, 100, 500]
})

# Box-Cox (positive values only)
df['boxcox'], lambda_param = stats.boxcox(df['value'])
print(f"Optimal lambda: {lambda_param}")

# Yeo-Johnson (supports negative values)
pt = PowerTransformer(method='yeo-johnson')
df['yeojohnson'] = pt.fit_transform(df[['value']])

print(df)
```

### 5.3 Binning

```python
df = pd.DataFrame({
    'age': [15, 22, 35, 45, 55, 65, 75, 85]
})

# Equal-width binning
df['age_bin_equal'] = pd.cut(df['age'], bins=4)

# Custom bins
bins = [0, 20, 40, 60, 100]
labels = ['youth', 'adult', 'middle', 'senior']
df['age_bin_custom'] = pd.cut(df['age'], bins=bins, labels=labels)

# Equal-frequency binning
df['age_qcut'] = pd.qcut(df['age'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

print(df)
```

---

## 6. Date and Time Handling

### 6.1 Date Parsing

```python
df = pd.DataFrame({
    'date_str': ['2023-01-15', '2023/02/20', '15-03-2023', '04.25.2023']
})

# Automatic parsing
df['date1'] = pd.to_datetime(df['date_str'].iloc[0:2])

# Specify format
df['date'] = pd.to_datetime(df['date_str'], format='mixed', dayfirst=True)

# Error handling
df['date'] = pd.to_datetime(df['date_str'], errors='coerce')  # NaT on error
```

### 6.2 Date Feature Extraction

```python
df = pd.DataFrame({
    'timestamp': pd.date_range('2023-01-01', periods=100, freq='D')
})

# Basic features
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
df['day'] = df['timestamp'].dt.day
df['dayofweek'] = df['timestamp'].dt.dayofweek  # 0=Monday
df['dayofyear'] = df['timestamp'].dt.dayofyear
df['weekofyear'] = df['timestamp'].dt.isocalendar().week
df['quarter'] = df['timestamp'].dt.quarter

# Boolean features
df['is_weekend'] = df['timestamp'].dt.dayofweek >= 5
df['is_month_start'] = df['timestamp'].dt.is_month_start
df['is_month_end'] = df['timestamp'].dt.is_month_end

# Cyclic features (trigonometric encoding)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

print(df.head())
```

---

## 7. Text Preprocessing

### 7.1 Basic Cleaning

```python
df = pd.DataFrame({
    'text': ['  Hello, World!  ', 'PYTHON 3.9', 'data-science', None]
})

# Lowercase conversion
df['lower'] = df['text'].str.lower()

# Strip whitespace
df['stripped'] = df['text'].str.strip()

# Remove special characters
df['cleaned'] = df['text'].str.replace(r'[^\w\s]', '', regex=True)

# Remove digits
df['no_numbers'] = df['text'].str.replace(r'\d+', '', regex=True)

print(df)
```

### 7.2 Tokenization and Stop Word Removal

```python
import re

# Simple tokenization
df = pd.DataFrame({
    'text': ['This is a sample text.', 'Another example here.']
})

df['tokens'] = df['text'].str.lower().str.split()

# Remove stop words
stopwords = {'a', 'an', 'the', 'is', 'this', 'here'}
df['filtered'] = df['tokens'].apply(
    lambda x: [word for word in x if word not in stopwords] if x else []
)

print(df)
```

---

## 8. Preprocessing Pipeline

### 8.1 sklearn Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Example data
df = pd.DataFrame({
    'age': [25, None, 35, 45, None],
    'salary': [50000, 60000, None, 80000, 70000],
    'department': ['IT', 'HR', 'IT', None, 'Sales']
})

# Numeric pipeline
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Full preprocessor
preprocessor = ColumnTransformer([
    ('numeric', numeric_pipeline, ['age', 'salary']),
    ('categorical', categorical_pipeline, ['department'])
])

# Run transformation
X_transformed = preprocessor.fit_transform(df)
print(X_transformed)
```

### 8.2 Custom Transformer

```python
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        self.lower_ = Q1 - self.factor * IQR
        self.upper_ = Q3 + self.factor * IQR
        return self

    def transform(self, X):
        X_clipped = np.clip(X, self.lower_, self.upper_)
        return X_clipped

# Usage example
remover = OutlierRemover(factor=1.5)
data = np.array([[1], [2], [3], [100], [4], [5]])
transformed = remover.fit_transform(data)
print(transformed)
```

---

## Practice Problems

### Problem 1: Missing Value Handling
Handle the missing values in the following data appropriately.

```python
df = pd.DataFrame({
    'A': [1, 2, None, 4, 5],
    'B': [None, 'X', 'Y', 'X', None]
})

# Solution
df['A'] = df['A'].fillna(df['A'].median())
df['B'] = df['B'].fillna(df['B'].mode()[0])
print(df)
```

### Problem 2: Outlier Detection
Find and remove outliers using the IQR method.

```python
df = pd.DataFrame({
    'value': [10, 12, 11, 13, 100, 11, 12, 10]
})

# Solution
Q1 = df['value'].quantile(0.25)
Q3 = df['value'].quantile(0.75)
IQR = Q3 - Q1
df_clean = df[(df['value'] >= Q1 - 1.5 * IQR) &
              (df['value'] <= Q3 + 1.5 * IQR)]
print(df_clean)
```

### Problem 3: Encoding
Apply one-hot encoding to the categorical variable.

```python
df = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'red']
})

# Solution
df_encoded = pd.get_dummies(df, columns=['color'], prefix='color')
print(df_encoded)
```

---

## Summary

| Feature | Method |
|---------|--------|
| Identify missing values | `isna()`, `isnull()` |
| Handle missing values | `dropna()`, `fillna()`, `interpolate()` |
| Detect outliers | IQR, Z-score, box plot |
| Normalization / Standardization | `MinMaxScaler`, `StandardScaler`, `RobustScaler` |
| Categorical encoding | `LabelEncoder`, `OneHotEncoder`, `get_dummies()` |
| Numerical transformation | Log transformation, Box-Cox, binning |
| Date handling | `to_datetime()`, `dt` accessor |
