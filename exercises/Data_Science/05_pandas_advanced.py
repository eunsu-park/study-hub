"""
Exercises for Lesson 05: Pandas Advanced
Topic: Data_Science

Solutions to practice problems from the lesson.
"""
import numpy as np
import pandas as pd


# === Exercise 1: Working with MultiIndex ===
# Problem: Select only the 2023 data from yearly and quarterly sales data.
def exercise_1():
    """Solution demonstrating MultiIndex creation and slicing."""
    df = pd.DataFrame({
        'year': [2022, 2022, 2023, 2023, 2022, 2023],
        'quarter': ['Q1', 'Q2', 'Q1', 'Q2', 'Q3', 'Q3'],
        'sales': [100, 120, 150, 180, 110, 200]
    }).set_index(['year', 'quarter'])

    print("Full MultiIndex DataFrame:")
    print(df)

    # .loc[2023] selects all rows where the first index level equals 2023
    # The result drops that level, leaving only 'quarter' as the index
    print("\n2023 data only:")
    print(df.loc[2023])

    # To keep the MultiIndex structure, use a slice or list
    print("\n2023 data (preserving MultiIndex):")
    print(df.loc[[2023]])

    # Cross-section selection: get Q1 data for all years
    print("\nQ1 data across all years (xs method):")
    print(df.xs('Q1', level='quarter'))

    # Sort the MultiIndex for efficient slicing
    df_sorted = df.sort_index()
    print("\nSorted MultiIndex (enables range slicing):")
    print(df_sorted)


# === Exercise 2: Time Series Resampling ===
# Problem: Resample daily data to weekly averages.
def exercise_2():
    """Solution demonstrating time series resampling for temporal aggregation."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=30, freq='D')
    ts = pd.Series(np.random.randn(30), index=dates)

    print("Daily data (first 10 days):")
    print(ts.head(10))

    # .resample('W') groups by week (ending Sunday by default)
    # .mean() computes the average within each week
    # This is analogous to GROUP BY on a temporal dimension
    weekly_avg = ts.resample('W').mean()
    print("\nWeekly averages:")
    print(weekly_avg)

    # Additional resampling frequencies for comparison
    print("\nBi-weekly sum:")
    print(ts.resample('2W').sum())

    # Monthly statistics
    monthly = ts.resample('ME').agg(['mean', 'std', 'min', 'max'])
    print("\nMonthly statistics:")
    print(monthly)


# === Exercise 3: Moving Average ===
# Problem: Calculate a 7-day moving average and display alongside original data.
def exercise_3():
    """Solution using rolling windows for smoothing time series."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=30, freq='D')
    # cumsum creates a random walk, making the moving average more meaningful
    ts = pd.Series(np.random.randn(30).cumsum(), index=dates)

    # .rolling(window=7) creates a sliding window of 7 observations
    # The first 6 values will be NaN because there aren't enough prior points
    df = pd.DataFrame({
        'original': ts,
        'ma_7': ts.rolling(window=7).mean()
    })
    print("Original vs 7-day Moving Average:")
    print(df)

    # Add exponential moving average for comparison
    # EMA gives more weight to recent observations (controlled by span parameter)
    df['ema_7'] = ts.ewm(span=7).mean()
    print("\nWith Exponential Moving Average:")
    print(df.tail(10))

    # Rolling statistics beyond just the mean
    print("\n7-day rolling statistics:")
    rolling_stats = pd.DataFrame({
        'rolling_mean': ts.rolling(7).mean(),
        'rolling_std': ts.rolling(7).std(),
        'rolling_min': ts.rolling(7).min(),
        'rolling_max': ts.rolling(7).max()
    })
    print(rolling_stats.tail(10))


if __name__ == "__main__":
    print("=== Exercise 1: Working with MultiIndex ===")
    exercise_1()
    print("\n=== Exercise 2: Time Series Resampling ===")
    exercise_2()
    print("\n=== Exercise 3: Moving Average ===")
    exercise_3()
    print("\nAll exercises completed!")
