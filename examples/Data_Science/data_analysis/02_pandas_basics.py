"""
Pandas Basics — Fundamental DataFrame Operations

Demonstrates:
- Series and DataFrame creation from dicts, lists, and arrays
- Data inspection, selection (loc/iloc/query), and modification
- Missing value strategies (drop, fill, interpolate)
- GroupBy split-apply-combine pattern
- Merge/join operations (inner, left, outer)
- Pivot tables and time-series resampling

Theory:
- Pandas is built on top of NumPy but adds labeled axes (index + columns),
  heterogeneous column types, and missing-value handling (NaN propagation).
- The GroupBy pattern splits a DataFrame by key, applies an aggregation
  to each group independently, then combines results — conceptually
  identical to SQL's GROUP BY.
- Merge implements relational joins; the `how` parameter maps to SQL's
  INNER/LEFT/RIGHT/FULL OUTER JOIN.

Adapted from Data_Science Lesson 02.
"""

import pandas as pd
import numpy as np


# =============================================================================
# 1. DataFrame and Series Creation
# =============================================================================
def create_dataframe():
    """DataFrame and Series creation."""
    print("\n[1] DataFrame and Series Creation")
    print("=" * 50)

    # Why: A Series is a single labeled column. DataFrames are dict-of-Series
    # under the hood, which is why creating a DataFrame from a dict is the
    # most natural and common pattern.
    s = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
    print(f"Series:\n{s}\n")

    # Create DataFrame from dict
    data = {
        'name': ['Alice', 'Bob', 'Carol', 'Dave', 'Eve'],
        'age': [25, 30, 35, 28, 32],
        'city': ['Seoul', 'Busan', 'Daegu', 'Seoul', 'Incheon'],
        'score': [85, 92, 78, 95, 88]
    }
    df = pd.DataFrame(data)
    print(f"DataFrame:\n{df}\n")

    # Create from list of lists
    df2 = pd.DataFrame(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        columns=['A', 'B', 'C'],
        index=['row1', 'row2', 'row3']
    )
    print(f"From list of lists:\n{df2}")

    return df


# =============================================================================
# 2. Data Inspection
# =============================================================================
def inspect_data(df):
    """Data inspection methods."""
    print("\n[2] Data Inspection")
    print("=" * 50)

    print(f"First 2 rows:\n{df.head(2)}\n")
    print(f"Last 2 rows:\n{df.tail(2)}\n")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Index: {df.index.tolist()}")
    print(f"\nData types:\n{df.dtypes}\n")
    # Why: describe() computes count, mean, std, min, quartiles, max in one
    # call — the fastest way to sanity-check numeric columns for NaNs,
    # unreasonable ranges, or skewed distributions.
    print(f"Descriptive statistics:\n{df.describe()}\n")
    print(f"Info:")
    df.info()


# =============================================================================
# 3. Indexing and Selection
# =============================================================================
def indexing_selection(df):
    """Indexing and selection."""
    print("\n[3] Indexing and Selection")
    print("=" * 50)

    print(f"Original DataFrame:\n{df}\n")

    # Column selection
    print(f"df['name']:\n{df['name']}\n")
    print(f"df[['name', 'age']]:\n{df[['name', 'age']]}\n")

    # Why: loc uses labels, iloc uses integer positions. This distinction
    # matters when the index is non-default (e.g., after filtering or
    # setting a string index) — loc[0] finds label 0, iloc[0] always
    # returns the first physical row.
    print(f"df.loc[0]:\n{df.loc[0]}\n")  # first row (by label)
    print(f"df.iloc[0:2]:\n{df.iloc[0:2]}\n")  # first 2 rows (by position)
    print(f"df.loc[0, 'name'] = {df.loc[0, 'name']}")  # specific value
    print(f"df.iloc[0, 1] = {df.iloc[0, 1]}")  # by position

    # Conditional selection
    print(f"\ndf[df['age'] > 28]:\n{df[df['age'] > 28]}")
    print(f"\ndf[(df['age'] > 25) & (df['city'] == 'Seoul')]:\n{df[(df['age'] > 25) & (df['city'] == 'Seoul')]}")

    # Why: query() is more readable than chained boolean indexing for
    # complex conditions, and it avoids the need for parentheses around
    # each sub-expression.
    result = df.query("age > 28 and score >= 90")
    print(f"\ndf.query(\"age > 28 and score >= 90\"):\n{result}")


# =============================================================================
# 4. Data Modification
# =============================================================================
def modify_data():
    """Data modification."""
    print("\n[4] Data Modification")
    print("=" * 50)

    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]
    })
    print(f"Original:\n{df}\n")

    # Add column
    df['D'] = df['A'] + df['B']
    print(f"Add column (D = A + B):\n{df}\n")

    # Drop column
    df_dropped = df.drop('D', axis=1)
    print(f"Drop column:\n{df_dropped}\n")

    # Modify value
    df.loc[0, 'A'] = 100
    print(f"Modify value (df.loc[0, 'A'] = 100):\n{df}\n")

    # Conditional modification
    df.loc[df['B'] > 4, 'C'] = 0
    print(f"Conditional modification:\n{df}\n")

    # Rename columns
    df_renamed = df.rename(columns={'A': 'Alpha', 'B': 'Beta'})
    print(f"Rename columns:\n{df_renamed}")


# =============================================================================
# 5. Missing Value Handling
# =============================================================================
def handle_missing():
    """Missing value handling."""
    print("\n[5] Missing Value Handling")
    print("=" * 50)

    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4],
        'B': [5, np.nan, np.nan, 8],
        'C': [9, 10, 11, 12]
    })
    print(f"Original (with NaN):\n{df}\n")

    # Detect missing values
    print(f"Missing value mask:\n{df.isnull()}\n")
    print(f"Missing count per column:\n{df.isnull().sum()}\n")

    # Drop rows with any missing value
    df_dropna = df.dropna()
    print(f"dropna() - drop rows:\n{df_dropna}\n")

    # Fill missing values
    df_filled = df.fillna(0)
    print(f"fillna(0):\n{df_filled}\n")

    # Why: Forward fill propagates the last valid observation. This is
    # appropriate for time-series where values persist until changed
    # (e.g., stock prices, sensor readings), but not for cross-sectional data.
    df_ffill = df.fillna(method='ffill')
    print(f"fillna(method='ffill') - forward fill:\n{df_ffill}\n")

    df_mean = df.fillna(df.mean())
    print(f"fillna(df.mean()) - fill with column mean:\n{df_mean}")


# =============================================================================
# 6. GroupBy and Aggregation
# =============================================================================
def groupby_aggregation():
    """GroupBy split-apply-combine pattern."""
    print("\n[6] GroupBy and Aggregation")
    print("=" * 50)

    df = pd.DataFrame({
        'dept': ['Sales', 'Dev', 'Sales', 'Dev', 'Sales', 'Dev'],
        'name': ['Alice', 'Bob', 'Carol', 'Dave', 'Eve', 'Frank'],
        'revenue': [100, 80, 120, 90, 110, 85],
        'experience': [3, 5, 7, 4, 6, 2]
    })
    print(f"Original:\n{df}\n")

    # Why: groupby() implements the split-apply-combine pattern — the most
    # fundamental aggregation paradigm in data analysis. It is conceptually
    # identical to SQL's GROUP BY and avoids manual loop-based grouping.
    grouped = df.groupby('dept')
    print(f"Revenue sum by dept:\n{grouped['revenue'].sum()}\n")
    print(f"Revenue mean by dept:\n{grouped['revenue'].mean()}\n")

    # Multiple aggregation functions via agg()
    agg_result = grouped.agg({
        'revenue': ['sum', 'mean', 'max'],
        'experience': ['mean', 'min', 'max']
    })
    print(f"Multiple aggregations:\n{agg_result}\n")

    # Multi-column groupby
    df['year'] = [2023, 2023, 2024, 2024, 2023, 2024]
    multi_group = df.groupby(['dept', 'year'])['revenue'].sum()
    print(f"Revenue by dept and year:\n{multi_group}")


# =============================================================================
# 7. Sorting and Ranking
# =============================================================================
def sorting_ranking():
    """Sorting and ranking."""
    print("\n[7] Sorting and Ranking")
    print("=" * 50)

    df = pd.DataFrame({
        'name': ['A', 'B', 'C', 'D', 'E'],
        'score': [85, 92, 78, 95, 88],
        'age': [25, 30, 25, 35, 28]
    })
    print(f"Original:\n{df}\n")

    # Single column sort
    sorted_df = df.sort_values('score', ascending=False)
    print(f"Sort by score (descending):\n{sorted_df}\n")

    # Multi-column sort
    sorted_df2 = df.sort_values(['age', 'score'], ascending=[True, False])
    print(f"Sort by age (asc), then score (desc):\n{sorted_df2}\n")

    # Index sort
    df_shuffled = df.sample(frac=1)
    print(f"Shuffled:\n{df_shuffled}")
    print(f"Sort by index:\n{df_shuffled.sort_index()}\n")

    # Ranking
    df['rank'] = df['score'].rank(ascending=False)
    print(f"With rank column:\n{df}")


# =============================================================================
# 8. Merge and Join
# =============================================================================
def merge_data():
    """Data merging (relational joins)."""
    print("\n[8] Merge and Join")
    print("=" * 50)

    # Two DataFrames to join
    df1 = pd.DataFrame({
        'emp_id': [1, 2, 3, 4],
        'name': ['Alice', 'Bob', 'Carol', 'Dave']
    })

    df2 = pd.DataFrame({
        'emp_id': [2, 3, 4, 5],
        'dept': ['Dev', 'Sales', 'Marketing', 'HR']
    })

    print(f"df1:\n{df1}\n")
    print(f"df2:\n{df2}\n")

    # Why: pd.merge() implements SQL-style joins. Inner keeps only matching
    # keys; left keeps all left rows (NaN where no match); outer keeps all
    # rows from both sides. Choosing the wrong join type is a frequent
    # source of accidental data loss or duplication.
    inner = pd.merge(df1, df2, on='emp_id', how='inner')
    print(f"Inner Join:\n{inner}\n")

    left = pd.merge(df1, df2, on='emp_id', how='left')
    print(f"Left Join:\n{left}\n")

    outer = pd.merge(df1, df2, on='emp_id', how='outer')
    print(f"Outer Join:\n{outer}\n")

    # Concat (stacking, not joining)
    df_a = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    df_b = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})

    concat_rows = pd.concat([df_a, df_b], ignore_index=True)
    print(f"Vertical concat:\n{concat_rows}\n")

    concat_cols = pd.concat([df_a, df_b], axis=1)
    print(f"Horizontal concat (axis=1):\n{concat_cols}")


# =============================================================================
# 9. Pivot Tables
# =============================================================================
def pivot_tables():
    """Pivot tables."""
    print("\n[9] Pivot Tables")
    print("=" * 50)

    df = pd.DataFrame({
        'date': ['2024-01', '2024-01', '2024-02', '2024-02'] * 2,
        'region': ['Seoul', 'Busan'] * 4,
        'product': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
        'revenue': [100, 80, 120, 90, 60, 70, 80, 50]
    })
    print(f"Original:\n{df}\n")

    # Why: pivot_table reshapes long-format data into a cross-tabulation,
    # combining groupby + reshape in one step. This is the Pandas equivalent
    # of Excel's PivotTable or SQL's PIVOT clause.
    pivot = df.pivot_table(
        values='revenue',
        index='region',
        columns='product',
        aggfunc='sum'
    )
    print(f"Pivot table (region x product):\n{pivot}\n")

    # Multi-aggregation pivot
    pivot2 = df.pivot_table(
        values='revenue',
        index=['date', 'region'],
        columns='product',
        aggfunc=['sum', 'mean']
    )
    print(f"Multi-aggregation pivot:\n{pivot2}")


# =============================================================================
# 10. Time Series
# =============================================================================
def time_series():
    """Time series operations."""
    print("\n[10] Time Series")
    print("=" * 50)

    # Date range generation
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    print(f"Date range:\n{dates}\n")

    # Time series DataFrame
    df = pd.DataFrame({
        'date': dates,
        'value': np.random.randn(10).cumsum()
    })
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    print(f"Time series data:\n{df}\n")

    # Why: resample() is the time-series equivalent of groupby() — it
    # groups by time periods (day, month, quarter) and aggregates.
    # This avoids manual date-extraction + groupby logic.
    df_monthly = pd.DataFrame({
        'value': np.random.randn(100)
    }, index=pd.date_range('2024-01-01', periods=100, freq='D'))

    monthly_mean = df_monthly.resample('M').mean()
    print(f"Monthly mean:\n{monthly_mean}\n")

    # Rolling window (moving average)
    df['moving_avg'] = df['value'].rolling(window=3).mean()
    print(f"Moving average (window=3):\n{df}")


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 60)
    print("Pandas Basics Examples")
    print("=" * 60)

    df = create_dataframe()
    inspect_data(df)
    indexing_selection(df)
    modify_data()
    handle_missing()
    groupby_aggregation()
    sorting_ranking()
    merge_data()
    pivot_tables()
    time_series()

    print("\n" + "=" * 60)
    print("Pandas Summary")
    print("=" * 60)
    print("""
    Core Data Structures:
    - Series: 1D labeled array
    - DataFrame: 2D labeled table (dict of Series)

    Commonly Used Methods:
    - Inspect: head, tail, info, describe, shape
    - Select:  loc (label), iloc (position), query
    - Modify:  drop, rename, fillna
    - Aggregate: groupby, agg, pivot_table
    - Merge:   merge, concat, join

    Tips:
    - Method chaining: df.dropna().groupby('col').mean()
    - Copy vs view: df.copy() vs slicing
    - Memory: use category dtype for repeated strings
    - Performance: prefer vectorized ops over apply()
    """)


if __name__ == "__main__":
    main()
