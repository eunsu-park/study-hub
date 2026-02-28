"""
Exercises for Lesson 06: Data Preprocessing
Topic: Data_Science

Solutions to practice problems from the lesson.
"""
import numpy as np
import pandas as pd


# === Exercise 1: Missing Value Handling ===
# Problem: Handle the missing values in the given data appropriately.
def exercise_1():
    """Solution demonstrating imputation strategies for different data types."""
    df = pd.DataFrame({
        'A': [1, 2, None, 4, 5],
        'B': [None, 'X', 'Y', 'X', None]
    })
    print("Original data:")
    print(df)
    print(f"\nMissing values:\n{df.isnull().sum()}")

    # Strategy for numerical column A: fill with median
    # Median is preferred over mean when data may have outliers
    # because median is robust to extreme values
    df['A'] = df['A'].fillna(df['A'].median())

    # Strategy for categorical column B: fill with mode (most frequent value)
    # Mode is the standard imputation for categorical data
    df['B'] = df['B'].fillna(df['B'].mode()[0])

    print("\nAfter imputation:")
    print(df)
    print(f"\nMissing values after: {df.isnull().sum().sum()}")

    # Demonstrate other strategies for comparison
    df2 = pd.DataFrame({'A': [1.0, 2.0, np.nan, 4.0, 5.0]})
    print("\nAlternative strategies for column A:")
    print(f"  Mean fill:          {df2['A'].fillna(df2['A'].mean()).tolist()}")
    print(f"  Forward fill:       {df2['A'].ffill().tolist()}")
    print(f"  Linear interpolate: {df2['A'].interpolate().tolist()}")


# === Exercise 2: Outlier Detection ===
# Problem: Find and remove outliers using the IQR method.
def exercise_2():
    """Solution using IQR-based outlier detection.

    The IQR method defines outliers as points falling below Q1 - 1.5*IQR
    or above Q3 + 1.5*IQR. This threshold corresponds to ~0.7% of data
    under a normal distribution (roughly 2.7 sigma).
    """
    df = pd.DataFrame({
        'value': [10, 12, 11, 13, 100, 11, 12, 10]
    })
    print("Original data:")
    print(df['value'].tolist())

    Q1 = df['value'].quantile(0.25)
    Q3 = df['value'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    print(f"\nQ1: {Q1}, Q3: {Q3}, IQR: {IQR}")
    print(f"Lower bound: {lower_bound:.2f}")
    print(f"Upper bound: {upper_bound:.2f}")

    # Identify outliers
    outlier_mask = (df['value'] < lower_bound) | (df['value'] > upper_bound)
    print(f"\nOutliers detected: {df.loc[outlier_mask, 'value'].tolist()}")

    # Remove outliers by keeping only values within bounds
    df_clean = df[(df['value'] >= lower_bound) & (df['value'] <= upper_bound)]
    print(f"Cleaned data: {df_clean['value'].tolist()}")
    print(f"Removed {len(df) - len(df_clean)} outlier(s)")

    # Also show Z-score method for comparison
    from scipy import stats
    z_scores = np.abs(stats.zscore(df['value']))
    z_outliers = df[z_scores > 2]
    print(f"\nZ-score method (|z| > 2) outliers: {z_outliers['value'].tolist()}")


# === Exercise 3: Encoding ===
# Problem: Apply one-hot encoding to the categorical variable.
def exercise_3():
    """Solution demonstrating one-hot encoding for categorical features."""
    df = pd.DataFrame({
        'color': ['red', 'blue', 'green', 'red']
    })
    print("Original data:")
    print(df)

    # pd.get_dummies creates binary columns for each unique category
    # prefix parameter controls the column name prefix
    # This is essential for ML models that require numerical input
    df_encoded = pd.get_dummies(df, columns=['color'], prefix='color')
    print("\nOne-hot encoded:")
    print(df_encoded)

    # Note: for k categories, we get k binary columns
    # To avoid multicollinearity in linear models, drop one column (drop_first=True)
    df_drop_first = pd.get_dummies(df, columns=['color'], prefix='color',
                                   drop_first=True)
    print("\nWith drop_first=True (avoids dummy variable trap):")
    print(df_drop_first)

    # Label encoding alternative (ordinal assignment)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df_label = df.copy()
    df_label['color_encoded'] = le.fit_transform(df['color'])
    print("\nLabel encoded (use only for ordinal categories):")
    print(df_label)
    print(f"Mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")


if __name__ == "__main__":
    print("=== Exercise 1: Missing Value Handling ===")
    exercise_1()
    print("\n=== Exercise 2: Outlier Detection ===")
    exercise_2()
    print("\n=== Exercise 3: Encoding ===")
    exercise_3()
    print("\nAll exercises completed!")
