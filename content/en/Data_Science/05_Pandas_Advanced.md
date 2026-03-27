# 5. Pandas Advanced

[Previous: Pandas Data Manipulation](./04_Pandas_Data_Manipulation.md) | [Next: Data Preprocessing](./06_Data_Preprocessing.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Create and manipulate MultiIndex DataFrames using `from_tuples`, `from_arrays`, and `from_product`
2. Apply cross-section selection (`xs`), level swapping, and level-based aggregation on hierarchical indices
3. Implement time series operations including date creation, datetime indexing, and date arithmetic
4. Apply resampling, rolling windows, and exponentially weighted moving averages to time series data
5. Describe Categorical data types and demonstrate their memory savings over string columns
6. Apply regex-based string extraction, splitting, and formatting using advanced `.str` accessor methods
7. Implement performance optimization using vectorized operations, `eval`, `query`, and chunk processing
8. Build data transformation pipelines using `pipe` and method chaining

---

As datasets grow in complexity, you need tools that go beyond basic selection and aggregation. MultiIndex structures let you model hierarchical data naturally, time series operations handle temporal patterns with ease, and performance optimization techniques keep your analyses fast even on millions of rows. These advanced Pandas features are what separate a beginner from a proficient data practitioner.

---

## 1. MultiIndex

### 1.1 Creating a MultiIndex

```python
import pandas as pd
import numpy as np

# Create from a list of tuples
arrays = [
    ['A', 'A', 'B', 'B'],
    [1, 2, 1, 2]
]
tuples = list(zip(*arrays))
index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])

s = pd.Series([10, 20, 30, 40], index=index)
print(s)
# first  second
# A      1         10
#        2         20
# B      1         30
#        2         40

# from_arrays
index = pd.MultiIndex.from_arrays(arrays, names=['first', 'second'])

# from_product (Cartesian product)
index = pd.MultiIndex.from_product(
    [['A', 'B'], [1, 2, 3]],
    names=['letter', 'number']
)
print(index)

# Apply to a DataFrame
df = pd.DataFrame({
    'value': [10, 20, 30, 40, 50, 60]
}, index=index)
print(df)
```

### 1.2 MultiIndex DataFrame

```python
# MultiIndex on columns as well
col_index = pd.MultiIndex.from_product(
    [['2023', '2024'], ['Q1', 'Q2']],
    names=['year', 'quarter']
)
row_index = pd.MultiIndex.from_product(
    [['Sales', 'IT'], ['Seoul', 'Busan']],
    names=['dept', 'city']
)

data = np.random.randint(100, 1000, (4, 4))
df = pd.DataFrame(data, index=row_index, columns=col_index)
print(df)
```

### 1.3 MultiIndex Selection

```python
df = pd.DataFrame({
    'year': [2022, 2022, 2023, 2023, 2022, 2023],
    'quarter': ['Q1', 'Q2', 'Q1', 'Q2', 'Q1', 'Q1'],
    'department': ['Sales', 'Sales', 'IT', 'IT', 'IT', 'Sales'],
    'revenue': [100, 150, 200, 250, 180, 160]
})
df = df.set_index(['year', 'quarter', 'department'])
print(df)

# Single-level selection
print(df.loc[2022])
print(df.loc[(2022, 'Q1')])
print(df.loc[(2022, 'Q1', 'Sales')])

# xs method (cross-section)
print(df.xs('Q1', level='quarter'))
print(df.xs('Sales', level='department'))
print(df.xs((2022, 'Sales'), level=['year', 'department']))

# Slicing
print(df.loc[2022:2023])
print(df.loc[(2022, 'Q1'):(2023, 'Q1')])
```

### 1.4 MultiIndex Manipulation

```python
df = pd.DataFrame({
    'year': [2022, 2022, 2023, 2023],
    'quarter': ['Q1', 'Q2', 'Q1', 'Q2'],
    'revenue': [100, 150, 200, 250]
}).set_index(['year', 'quarter'])

# Swap levels
print(df.swaplevel())

# Sort levels
df_unsorted = df.iloc[[2, 0, 3, 1]]
print(df_unsorted.sort_index())
print(df_unsorted.sort_index(level=1))

# Reset index
print(df.reset_index())
print(df.reset_index(level='quarter'))

# Rename levels
df.index = df.index.set_names(['year', 'quarter'])
print(df)

# Change level values
df.index = df.index.set_levels([['2022', '2023'], ['Q1', 'Q2']])
```

### 1.5 MultiIndex Aggregation

```python
df = pd.DataFrame({
    'year': [2022, 2022, 2022, 2023, 2023, 2023],
    'quarter': ['Q1', 'Q1', 'Q2', 'Q1', 'Q1', 'Q2'],
    'department': ['Sales', 'IT', 'Sales', 'Sales', 'IT', 'Sales'],
    'revenue': [100, 150, 120, 200, 180, 220]
}).set_index(['year', 'quarter', 'department'])

# Sum by level
print(df.groupby(level='year').sum())
print(df.groupby(level=['year', 'quarter']).sum())

# Pivot with unstack
print(df.unstack(level='department'))
print(df.unstack(level=['quarter', 'department']))

# Reverse pivot with stack
df_wide = df.unstack(level='department')
print(df_wide.stack())
```

---

## 2. Time Series Data

### 2.1 Creating Dates and Times

```python
# Timestamp
ts = pd.Timestamp('2023-01-15')
ts = pd.Timestamp('2023-01-15 10:30:00')
ts = pd.Timestamp(year=2023, month=1, day=15, hour=10)

# to_datetime
dates = pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01'])
dates = pd.to_datetime(['01/15/2023', '02/15/2023'], format='%m/%d/%Y')

# date_range
dates = pd.date_range('2023-01-01', periods=10, freq='D')
dates = pd.date_range('2023-01-01', '2023-12-31', freq='M')  # month end
dates = pd.date_range('2023-01-01', '2023-12-31', freq='MS')  # month start
dates = pd.date_range('2023-01-01', periods=5, freq='W-MON')  # every Monday

# Common freq options
# 'D': day, 'W': week, 'M': month end, 'MS': month start
# 'Q': quarter end, 'QS': quarter start, 'Y': year end, 'YS': year start
# 'H': hour, 'T' or 'min': minute, 'S': second
# 'B': business day, 'BM': business month end

# period_range (periods)
periods = pd.period_range('2023-01', periods=12, freq='M')
print(periods)
```

### 2.2 Time Series Indexing

```python
# Series with DatetimeIndex
dates = pd.date_range('2023-01-01', periods=365, freq='D')
ts = pd.Series(np.random.randn(365), index=dates)

# Select by string
print(ts['2023-03-15'])
print(ts['2023-03'])  # entire March
print(ts['2023'])     # entire 2023

# Range selection
print(ts['2023-03-01':'2023-03-10'])
print(ts['2023-03':'2023-06'])

# Using loc
print(ts.loc['2023-03-15'])
print(ts.loc['2023-03'])
```

### 2.3 Date and Time Attributes

```python
df = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=10, freq='D')
})

# dt accessor
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek  # 0=Monday
df['day_name'] = df['date'].dt.day_name()
df['month_name'] = df['date'].dt.month_name()
df['quarter'] = df['date'].dt.quarter
df['is_month_end'] = df['date'].dt.is_month_end
df['is_month_start'] = df['date'].dt.is_month_start

print(df)
```

### 2.4 Date Arithmetic

```python
# Timedelta
td = pd.Timedelta('1 days')
td = pd.Timedelta(days=1, hours=2, minutes=30)

# Date arithmetic
dates = pd.date_range('2023-01-01', periods=5, freq='D')
print(dates + pd.Timedelta('1 days'))
print(dates + pd.DateOffset(months=1))

# Date difference
df = pd.DataFrame({
    'start': pd.to_datetime(['2023-01-01', '2023-02-15']),
    'end': pd.to_datetime(['2023-01-10', '2023-03-20'])
})
df['duration'] = df['end'] - df['start']
df['days'] = df['duration'].dt.days

# DateOffset
from pandas.tseries.offsets import MonthEnd, BDay

date = pd.Timestamp('2023-01-15')
print(date + MonthEnd())  # month end
print(date + BDay(5))     # 5 business days later
```

### 2.5 Resampling

```python
dates = pd.date_range('2023-01-01', periods=100, freq='D')
ts = pd.Series(np.random.randn(100), index=dates)

# Downsampling (to a larger interval)
print(ts.resample('W').mean())    # weekly average
print(ts.resample('M').sum())     # monthly sum
print(ts.resample('M').agg(['mean', 'std', 'min', 'max']))

# Upsampling (to a smaller interval)
monthly = pd.Series([100, 110, 120],
                    index=pd.date_range('2023-01-01', periods=3, freq='M'))
print(monthly.resample('D').ffill())   # forward fill
print(monthly.resample('D').bfill())   # backward fill
print(monthly.resample('D').interpolate())  # interpolate

# OHLC aggregation
print(ts.resample('W').ohlc())  # Open, High, Low, Close
```

### 2.6 Rolling Windows

```python
dates = pd.date_range('2023-01-01', periods=30, freq='D')
ts = pd.Series(np.random.randn(30).cumsum(), index=dates)

# Moving average
print(ts.rolling(window=7).mean())

# Various aggregations
print(ts.rolling(window=7).std())
print(ts.rolling(window=7).min())
print(ts.rolling(window=7).max())
print(ts.rolling(window=7).sum())

# Centered moving average
print(ts.rolling(window=7, center=True).mean())

# Exponentially weighted moving average (EWMA)
print(ts.ewm(span=7).mean())
print(ts.ewm(alpha=0.3).mean())

# Expanding window (from start to current row)
print(ts.expanding().mean())  # cumulative mean
print(ts.expanding().sum())   # cumulative sum
```

### 2.7 Timezone Handling

```python
# Assign timezone
ts = pd.Timestamp('2023-01-15 10:00', tz='Asia/Seoul')
print(ts)

dates = pd.date_range('2023-01-01', periods=5, freq='D', tz='UTC')
print(dates)

# Convert timezone
ts_utc = pd.Timestamp('2023-01-15 10:00', tz='UTC')
ts_seoul = ts_utc.tz_convert('Asia/Seoul')
print(ts_seoul)

# Localize timezone
ts_naive = pd.Timestamp('2023-01-15 10:00')
ts_localized = ts_naive.tz_localize('Asia/Seoul')
print(ts_localized)

# Series/DataFrame
s = pd.Series([1, 2, 3], index=pd.date_range('2023-01-01', periods=3, freq='D'))
s_utc = s.tz_localize('UTC')
s_seoul = s_utc.tz_convert('Asia/Seoul')
```

---

## 3. Categorical Data

### 3.1 Categorical Type

```python
# Create Categorical
cat = pd.Categorical(['a', 'b', 'c', 'a', 'b'])
print(cat)
print(cat.categories)
print(cat.codes)

# Specify ordering
cat = pd.Categorical(['low', 'medium', 'high', 'low'],
                     categories=['low', 'medium', 'high'],
                     ordered=True)
print(cat)
print(cat.min())  # low
print(cat.max())  # high

# Use in a DataFrame
df = pd.DataFrame({
    'grade': pd.Categorical(['A', 'B', 'A', 'C', 'B'],
                            categories=['C', 'B', 'A'],
                            ordered=True)
})
print(df.sort_values('grade'))
```

### 3.2 Type Conversion

```python
df = pd.DataFrame({
    'category': ['apple', 'banana', 'apple', 'cherry', 'banana']
})

# Convert to category type
df['category'] = df['category'].astype('category')
print(df['category'].dtype)
print(df['category'].cat.categories)

# Verify memory savings
df_str = pd.DataFrame({'col': ['A'] * 1000000})
df_cat = pd.DataFrame({'col': pd.Categorical(['A'] * 1000000)})
print(f"String: {df_str.memory_usage(deep=True).sum():,} bytes")
print(f"Category: {df_cat.memory_usage(deep=True).sum():,} bytes")
```

### 3.3 Category Manipulation

```python
s = pd.Series(['a', 'b', 'c', 'a', 'b']).astype('category')

# Add categories
s = s.cat.add_categories(['d', 'e'])
print(s.cat.categories)

# Remove categories
s = s.cat.remove_categories(['e'])

# Rename categories
s = s.cat.rename_categories({'a': 'A', 'b': 'B', 'c': 'C'})
print(s)

# Remove unused categories
s = s.cat.remove_unused_categories()

# Reorder categories
s = s.cat.reorder_categories(['C', 'B', 'A'])
```

---

## 4. Advanced String Operations

### 4.1 Regular Expressions

```python
df = pd.DataFrame({
    'text': ['apple 123', 'banana 456', 'cherry 789', 'date'],
    'email': ['test@example.com', 'user@domain.org', 'invalid', 'admin@site.net']
})

# Pattern matching
print(df['text'].str.contains(r'\d+', regex=True))

# Pattern extraction
print(df['text'].str.extract(r'(\w+)\s(\d+)'))

# Extract all matches
print(df['text'].str.findall(r'\d'))

# Extract email domain
print(df['email'].str.extract(r'@(.+)$'))

# Replace
print(df['text'].str.replace(r'\d+', 'NUM', regex=True))
```

### 4.2 String Splitting and Joining

```python
df = pd.DataFrame({
    'full_name': ['John Smith', 'Jane Doe', 'Bob Johnson']
})

# Split
names = df['full_name'].str.split(' ', expand=True)
names.columns = ['first', 'last']
print(names)

# Join
df['formatted'] = df['full_name'].str.replace(' ', ', ')

# Concatenate strings (Series)
s = pd.Series(['a', 'b', 'c'])
print(s.str.cat(sep='-'))  # 'a-b-c'

# Join two Series
s1 = pd.Series(['a', 'b', 'c'])
s2 = pd.Series(['1', '2', '3'])
print(s1.str.cat(s2, sep='-'))  # ['a-1', 'b-2', 'c-3']
```

### 4.3 String Formatting

```python
df = pd.DataFrame({
    'name': ['Alice', 'Bob'],
    'score': [95.5, 87.3]
})

# Formatting
df['formatted'] = df['name'] + ': ' + df['score'].astype(str)
df['formatted2'] = df.apply(lambda x: f"{x['name']}: {x['score']:.1f}", axis=1)
print(df)

# Padding
s = pd.Series(['1', '22', '333'])
print(s.str.pad(5, side='left', fillchar='0'))  # ['00001', '00022', '00333']
print(s.str.zfill(5))  # ['00001', '00022', '00333']
print(s.str.center(7, '*'))  # ['***1***', '**22***', '*333**']
```

---

## 5. Performance Optimization

### 5.1 Data Type Optimization

```python
def reduce_mem_usage(df):
    """Optimize DataFrame memory usage"""
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage: {start_mem:.2f} MB -> {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')

    return df
```

### 5.2 Vectorized Operations

```python
import time

df = pd.DataFrame({
    'A': np.random.randn(100000),
    'B': np.random.randn(100000)
})

# Bad example: iterrows
start = time.time()
result = []
for idx, row in df.iterrows():
    result.append(row['A'] + row['B'])
print(f"iterrows: {time.time() - start:.4f}s")

# Good example: vectorized
start = time.time()
result = df['A'] + df['B']
print(f"vectorized: {time.time() - start:.4f}s")

# apply vs vectorized
start = time.time()
result = df.apply(lambda x: x['A'] + x['B'], axis=1)
print(f"apply: {time.time() - start:.4f}s")
```

### 5.3 eval and query

```python
df = pd.DataFrame({
    'A': np.random.randn(100000),
    'B': np.random.randn(100000),
    'C': np.random.randn(100000)
})

# Use eval (complex expressions)
df['D'] = pd.eval('df.A + df.B * df.C')

# More complex calculation
result = df.eval('(A + B) / (C + 1)')

# Combined with query
result = df.query('A > 0 and B < 0')

# Using local variables
threshold = 0.5
result = df.query('A > @threshold')
```

### 5.4 Processing Large Data

```python
# Process in chunks
def process_large_file(filename, chunksize=10000):
    results = []
    for chunk in pd.read_csv(filename, chunksize=chunksize):
        # Process each chunk
        processed = chunk[chunk['value'] > 0].groupby('category')['value'].sum()
        results.append(processed)

    return pd.concat(results).groupby(level=0).sum()

# Read only specific columns
df = pd.read_csv('large_file.csv', usecols=['col1', 'col2', 'col3'])

# Read with specified data types
dtypes = {'col1': 'int32', 'col2': 'float32', 'col3': 'category'}
df = pd.read_csv('large_file.csv', dtype=dtypes)
```

---

## 6. Pipelines

### 6.1 The pipe Method

```python
def remove_outliers(df, column, n_std=3):
    """Remove outliers"""
    mean = df[column].mean()
    std = df[column].std()
    return df[(df[column] - mean).abs() <= n_std * std]

def add_features(df):
    """Add features"""
    df = df.copy()
    df['log_value'] = np.log1p(df['value'])
    df['squared'] = df['value'] ** 2
    return df

def normalize(df, columns):
    """Normalize"""
    df = df.copy()
    for col in columns:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df

# Run the pipeline
df = pd.DataFrame({
    'value': np.random.randn(1000) * 10 + 50,
    'category': np.random.choice(['A', 'B', 'C'], 1000)
})

result = (df
    .pipe(remove_outliers, 'value')
    .pipe(add_features)
    .pipe(normalize, ['value', 'log_value'])
)
print(result.head())
```

### 6.2 Method Chaining

```python
df = pd.DataFrame({
    'name': ['  Alice  ', 'BOB', 'charlie', None],
    'age': [25, 30, 35, 28],
    'salary': [50000, 60000, 70000, 55000]
})

result = (df
    .dropna()
    .assign(name=lambda x: x['name'].str.strip().str.title())
    .query('age >= 25')
    .sort_values('salary', ascending=False)
    .reset_index(drop=True)
)
print(result)
```

---

## Practice Problems

### Problem 1: Working with MultiIndex
Select only the 2023 data from yearly and quarterly sales data.

```python
df = pd.DataFrame({
    'year': [2022, 2022, 2023, 2023, 2022, 2023],
    'quarter': ['Q1', 'Q2', 'Q1', 'Q2', 'Q3', 'Q3'],
    'sales': [100, 120, 150, 180, 110, 200]
}).set_index(['year', 'quarter'])

# Solution
print(df.loc[2023])
```

### Problem 2: Time Series Resampling
Resample daily data to weekly averages.

```python
dates = pd.date_range('2023-01-01', periods=30, freq='D')
ts = pd.Series(np.random.randn(30), index=dates)

# Solution
weekly_avg = ts.resample('W').mean()
print(weekly_avg)
```

### Problem 3: Moving Average
Calculate a 7-day moving average and display it alongside the original data.

```python
dates = pd.date_range('2023-01-01', periods=30, freq='D')
ts = pd.Series(np.random.randn(30).cumsum(), index=dates)

# Solution
df = pd.DataFrame({
    'original': ts,
    'ma_7': ts.rolling(window=7).mean()
})
print(df)
```

---

## Summary

| Feature | Functions/Methods |
|------|------------|
| MultiIndex | `MultiIndex.from_*()`, `xs()`, `swaplevel()`, `stack()`, `unstack()` |
| Time Series | `to_datetime()`, `date_range()`, `resample()`, `rolling()`, `ewm()` |
| Categorical | `Categorical()`, `astype('category')`, `cat` accessor |
| Strings | `str` accessor, regular expressions, `extract()`, `split()` |
| Performance | vectorized operations, `eval()`, `query()`, chunk processing |
| Pipelines | `pipe()`, method chaining |
