# 3. Pandas Basics

[Previous: NumPy Advanced](./02_NumPy_Advanced.md) | [Next: Pandas Data Manipulation](./04_Pandas_Data_Manipulation.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Distinguish between Series and DataFrame data structures and create each from various sources
2. Describe key DataFrame attributes including shape, columns, dtypes, and memory usage
3. Apply `read_csv`, `read_excel`, `read_json`, and `read_sql` to load data from multiple formats
4. Implement label-based (`loc`), integer-based (`iloc`), and single-value (`at`/`iat`) selection
5. Apply conditional filtering, unique value counting, and missing value detection for data exploration
6. Demonstrate arithmetic operations, aggregation functions, and sorting on DataFrames
7. Implement column and row additions, modifications, and deletions
8. Apply string accessor methods (`.str`) for text cleaning and extraction

---

Real-world data almost always arrives in tabular form -- spreadsheets, database exports, CSV logs. Pandas gives you a powerful, expressive interface for loading, exploring, and transforming this kind of data. Learning Pandas is arguably the single most impactful skill for any aspiring data scientist, because every analysis pipeline begins with data wrangling.

---

## 1. Pandas Data Structures

### 1.1 Series

Series is a one-dimensional labeled array.

```python
import pandas as pd
import numpy as np

# Create Series from list
s = pd.Series([10, 20, 30, 40, 50])
print(s)
# 0    10
# 1    20
# 2    30
# 3    40
# 4    50
# dtype: int64

# Specify index
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
print(s)
# a    10
# b    20
# c    30

# Create from dictionary
d = {'apple': 100, 'banana': 200, 'cherry': 150}
s = pd.Series(d)
print(s)

# Series attributes
print(s.values)  # value array
print(s.index)   # index
print(s.dtype)   # data type
print(s.name)    # Series name
```

### 1.2 DataFrame

DataFrame is a two-dimensional tabular data structure.

```python
# Create DataFrame from dictionary
data = {
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['Seoul', 'Busan', 'Incheon']
}
df = pd.DataFrame(data)
print(df)
#       name  age     city
# 0    Alice   25    Seoul
# 1      Bob   30    Busan
# 2  Charlie   35  Incheon

# Create from list of lists
data = [
    ['Alice', 25, 'Seoul'],
    ['Bob', 30, 'Busan'],
    ['Charlie', 35, 'Incheon']
]
df = pd.DataFrame(data, columns=['name', 'age', 'city'])

# Create from NumPy array
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
df = pd.DataFrame(arr, columns=['A', 'B', 'C'])

# Specify index
df = pd.DataFrame(data,
                  columns=['name', 'age', 'city'],
                  index=['p1', 'p2', 'p3'])
```

### 1.3 DataFrame Attributes

```python
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'salary': [50000, 60000, 70000]
})

# Basic attributes
print(df.shape)      # (3, 3)
print(df.columns)    # Index(['name', 'age', 'salary'], dtype='object')
print(df.index)      # RangeIndex(start=0, stop=3, step=1)
print(df.dtypes)     # data type of each column
print(df.values)     # NumPy array
print(df.size)       # 9 (total elements)
print(len(df))       # 3 (number of rows)

# Memory usage
print(df.memory_usage())

# Data summary
print(df.info())
print(df.describe())  # statistical summary of numeric columns
```

---

## 2. Loading Data

### 2.1 CSV Files

```python
# Read CSV
df = pd.read_csv('data.csv')

# Specify options
df = pd.read_csv('data.csv',
                 sep=',',           # delimiter
                 header=0,          # header row (None if no header)
                 index_col=0,       # column to use as index
                 usecols=['A', 'B'], # columns to read
                 dtype={'A': int},   # specify data types
                 na_values=['NA', 'N/A'],  # values to treat as missing
                 encoding='utf-8',   # encoding
                 nrows=100)          # number of rows to read

# Read large files in chunks
chunks = pd.read_csv('large_data.csv', chunksize=10000)
for chunk in chunks:
    process(chunk)

# Save CSV
df.to_csv('output.csv', index=False)
```

### 2.2 Excel Files

```python
# Read Excel (requires openpyxl or xlrd)
df = pd.read_excel('data.xlsx')

# Specify sheet
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# Read multiple sheets
sheets = pd.read_excel('data.xlsx', sheet_name=None)  # returns dictionary

# Save Excel
df.to_excel('output.xlsx', index=False, sheet_name='Data')

# Save multiple sheets
with pd.ExcelWriter('output.xlsx') as writer:
    df1.to_excel(writer, sheet_name='Sheet1')
    df2.to_excel(writer, sheet_name='Sheet2')
```

### 2.3 JSON Files

```python
# Read JSON
df = pd.read_json('data.json')

# Specify format
df = pd.read_json('data.json', orient='records')
# orient: 'split', 'records', 'index', 'columns', 'values'

# Save JSON
df.to_json('output.json', orient='records')

# JSON Lines (newline-delimited)
df = pd.read_json('data.jsonl', lines=True)
df.to_json('output.jsonl', orient='records', lines=True)
```

### 2.4 SQL Databases

```python
import sqlite3
from sqlalchemy import create_engine

# SQLite connection
conn = sqlite3.connect('database.db')
df = pd.read_sql('SELECT * FROM users', conn)
conn.close()

# Using SQLAlchemy engine
engine = create_engine('postgresql://user:pass@host:5432/db')
df = pd.read_sql('SELECT * FROM users', engine)

# Read table
df = pd.read_sql_table('users', engine)

# Execute query
df = pd.read_sql_query('SELECT * FROM users WHERE age > 30', engine)

# Save DataFrame to SQL
df.to_sql('users', engine, if_exists='replace', index=False)
# if_exists: 'fail', 'replace', 'append'
```

### 2.5 Other Formats

```python
# HTML tables
dfs = pd.read_html('https://example.com/table.html')
df = dfs[0]  # first table

# Clipboard
df = pd.read_clipboard()

# Parquet (requires pyarrow)
df = pd.read_parquet('data.parquet')
df.to_parquet('output.parquet')

# Pickle
df = pd.read_pickle('data.pkl')
df.to_pickle('output.pkl')

# HDF5 (requires tables)
df = pd.read_hdf('data.h5', key='df')
df.to_hdf('output.h5', key='df')
```

---

## 3. Data Selection and Access

### 3.1 Column Selection

```python
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['Seoul', 'Busan', 'Incheon']
})

# Select single column (returns Series)
print(df['name'])
print(df.name)  # attribute access (if column name is valid identifier)

# Select multiple columns (returns DataFrame)
print(df[['name', 'age']])
```

### 3.2 Row Selection

```python
# Slicing
print(df[0:2])  # first 2 rows

# Conditional filtering
print(df[df['age'] > 25])
print(df[df['city'].isin(['Seoul', 'Busan'])])
```

### 3.3 loc - Label-based Selection

```python
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['Seoul', 'Busan', 'Incheon']
}, index=['a', 'b', 'c'])

# Single row
print(df.loc['a'])

# Multiple rows
print(df.loc[['a', 'c']])

# Rows and columns
print(df.loc['a', 'name'])        # single value
print(df.loc['a':'b', 'name':'age'])  # range slicing

# With conditions
print(df.loc[df['age'] > 25, ['name', 'city']])
```

### 3.4 iloc - Integer-based Selection

```python
# Single row
print(df.iloc[0])

# Multiple rows
print(df.iloc[[0, 2]])

# Rows and columns
print(df.iloc[0, 1])        # single value
print(df.iloc[0:2, 0:2])    # range slicing
print(df.iloc[[0, 2], [0, 2]])  # specific positions

# Negative indices
print(df.iloc[-1])  # last row
```

### 3.5 at and iat - Single Value Access

```python
# at: label-based single value
print(df.at['a', 'name'])

# iat: integer-based single value
print(df.iat[0, 0])

# Modify values
df.at['a', 'age'] = 26
df.iat[0, 1] = 27
```

---

## 4. Data Exploration

### 4.1 Preview Data

```python
df = pd.DataFrame({
    'A': range(100),
    'B': range(100, 200),
    'C': range(200, 300)
})

# View first/last rows
print(df.head())     # first 5 rows
print(df.head(10))   # first 10 rows
print(df.tail())     # last 5 rows
print(df.tail(3))    # last 3 rows

# Random sample
print(df.sample(5))  # random 5 rows
print(df.sample(frac=0.1))  # 10% sample
```

### 4.2 Data Information

```python
df = pd.DataFrame({
    'name': ['Alice', 'Bob', None, 'Diana'],
    'age': [25, 30, 35, None],
    'salary': [50000.0, 60000.0, 70000.0, 80000.0]
})

# Basic information
print(df.info())

# Example output:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 4 entries, 0 to 3
# Data columns (total 3 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   name    3 non-null      object
#  1   age     3 non-null      float64
#  2   salary  4 non-null      float64
# dtypes: float64(2), object(1)
# memory usage: 224.0+ bytes

# Statistical summary
print(df.describe())
print(df.describe(include='all'))  # include all columns
```

### 4.3 Unique Values and Frequencies

```python
df = pd.DataFrame({
    'category': ['A', 'B', 'A', 'C', 'B', 'A', 'A', 'C'],
    'value': [10, 20, 30, 40, 50, 60, 70, 80]
})

# Unique values
print(df['category'].unique())    # ['A' 'B' 'C']
print(df['category'].nunique())   # 3

# Frequencies
print(df['category'].value_counts())
# A    4
# B    2
# C    2

# Normalized frequencies
print(df['category'].value_counts(normalize=True))
```

### 4.4 Check Missing Values

```python
df = pd.DataFrame({
    'A': [1, 2, None, 4],
    'B': [None, 2, 3, 4],
    'C': [1, 2, 3, None]
})

# Check missing values
print(df.isna())      # boolean DataFrame
print(df.isnull())    # same as isna

# Count missing values
print(df.isna().sum())        # missing per column
print(df.isna().sum().sum())  # total missing

# Rows/columns with missing values
print(df[df.isna().any(axis=1)])  # rows with any missing
```

---

## 5. Basic Operations

### 5.1 Arithmetic Operations

```python
df = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [10, 20, 30, 40]
})

# Scalar operations
print(df + 10)
print(df * 2)
print(df ** 2)

# Column operations
df['C'] = df['A'] + df['B']
df['D'] = df['B'] / df['A']

# DataFrame operations (index alignment)
df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}, index=[0, 1])
df2 = pd.DataFrame({'A': [10, 20], 'B': [30, 40]}, index=[1, 2])
print(df1 + df2)  # only matching indices

# Handle missing values in operations
print(df1.add(df2, fill_value=0))
```

### 5.2 Aggregation Functions

```python
df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50]
})

# Basic aggregations
print(df.sum())      # column sum
print(df.mean())     # column mean
print(df.median())   # median
print(df.std())      # standard deviation
print(df.var())      # variance
print(df.min())      # minimum
print(df.max())      # maximum
print(df.count())    # count non-null values

# Specify axis
print(df.sum(axis=0))  # by column (default)
print(df.sum(axis=1))  # by row

# Cumulative functions
print(df.cumsum())   # cumulative sum
print(df.cumprod())  # cumulative product
print(df.cummax())   # cumulative max
print(df.cummin())   # cumulative min
```

### 5.3 Sorting

```python
df = pd.DataFrame({
    'name': ['Charlie', 'Alice', 'Bob'],
    'age': [35, 25, 30],
    'score': [85, 95, 75]
})

# Sort by values
print(df.sort_values('age'))
print(df.sort_values('age', ascending=False))

# Multiple columns
print(df.sort_values(['age', 'score']))
print(df.sort_values(['age', 'score'], ascending=[True, False]))

# Sort by index
df = df.set_index('name')
print(df.sort_index())
print(df.sort_index(ascending=False))

# Ranking
print(df.rank())  # ranks
```

---

## 6. Data Modification

### 6.1 Add/Modify Columns

```python
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

# Add new columns
df['C'] = [7, 8, 9]
df['D'] = df['A'] + df['B']
df['E'] = 10  # scalar value

# assign method (keeps original)
df2 = df.assign(F=lambda x: x['A'] * 2,
                G=[10, 20, 30])

# insert (at specific position)
df.insert(1, 'new_col', [100, 200, 300])
```

### 6.2 Delete Columns

```python
# drop method
df = df.drop('C', axis=1)
df = df.drop(['D', 'E'], axis=1)

# del keyword
del df['B']

# pop method (delete and return)
col = df.pop('A')
```

### 6.3 Add/Modify Rows

```python
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

# Add row (using concat)
new_row = pd.DataFrame({'A': [4], 'B': [7]})
df = pd.concat([df, new_row], ignore_index=True)

# Add using loc
df.loc[len(df)] = [5, 8]

# Delete rows
df = df.drop(0)  # drop index 0
df = df.drop([1, 2])  # drop multiple rows
```

### 6.4 Modify Values

```python
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

# Conditional modification
df.loc[df['A'] > 1, 'B'] = 0

# replace
df['A'] = df['A'].replace(1, 100)
df = df.replace({2: 200, 3: 300})

# where (modify where condition is False)
df['A'] = df['A'].where(df['A'] > 100, 0)

# mask (modify where condition is True)
df['B'] = df['B'].mask(df['B'] < 5, -1)
```

---

## 7. String Operations

Pandas provides string methods through the `.str` accessor.

```python
df = pd.DataFrame({
    'name': ['  Alice  ', 'BOB', 'charlie'],
    'email': ['alice@test.com', 'bob@example.com', 'charlie@test.com']
})

# Case conversion
print(df['name'].str.lower())
print(df['name'].str.upper())
print(df['name'].str.title())
print(df['name'].str.capitalize())

# Remove whitespace
print(df['name'].str.strip())
print(df['name'].str.lstrip())
print(df['name'].str.rstrip())

# String length
print(df['name'].str.len())

# Contains
print(df['email'].str.contains('test'))
print(df['name'].str.startswith('A'))
print(df['name'].str.endswith('e'))

# Split strings
print(df['email'].str.split('@'))
print(df['email'].str.split('@').str[0])  # first element

# Replace
print(df['email'].str.replace('test', 'example'))

# Regular expressions
print(df['email'].str.extract(r'@(.+)\.com'))
print(df['email'].str.findall(r'\w+'))
```

---

## Practice Problems

### Problem 1: Data Loading and Exploration
Create a DataFrame from the following data and check basic information.

```python
data = {
    'product': ['Apple', 'Banana', 'Cherry', 'Date'],
    'price': [1000, 500, 2000, 1500],
    'quantity': [50, 100, 30, 45]
}

# Solution
df = pd.DataFrame(data)
print(df.info())
print(df.describe())
print(df['price'].mean())  # average price
```

### Problem 2: Data Selection
Select product names and quantities where price is 1000 or more.

```python
# Solution
result = df.loc[df['price'] >= 1000, ['product', 'quantity']]
print(result)
```

### Problem 3: Add Column
Add a total amount column (price * quantity).

```python
# Solution
df['total'] = df['price'] * df['quantity']
print(df)
```

---

## Summary

| Feature | Functions/Methods |
|------|------------|
| Data Loading | `pd.read_csv()`, `pd.read_excel()`, `pd.read_json()`, `pd.read_sql()` |
| Data Saving | `to_csv()`, `to_excel()`, `to_json()`, `to_sql()` |
| Column Selection | `df['col']`, `df[['col1', 'col2']]` |
| Row Selection | `df.loc[]`, `df.iloc[]`, `df[condition]` |
| Data Exploration | `head()`, `tail()`, `info()`, `describe()` |
| Aggregation | `sum()`, `mean()`, `count()`, `min()`, `max()` |
| Sorting | `sort_values()`, `sort_index()` |
| String Operations | `df['col'].str.method()` |
