"""
Data Cleaning and Preprocessing Techniques

Covers the most important preprocessing techniques used in real-world data analysis.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple


# =============================================================================
# 1. Handling Missing Values
# =============================================================================
def handle_missing_values():
    """Detect and handle missing values"""
    print("\n[1] Handling Missing Values")
    print("=" * 50)

    # Create data with missing values
    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, np.nan, 5],
        'C': [1, 2, 3, 4, 5],
        'D': ['a', None, 'c', 'd', np.nan]
    })

    print("Original data:")
    print(df)
    print()

    # Detect missing values
    print("Missing value counts:")
    print(df.isnull().sum())
    print(f"\nMissing value ratios:\n{df.isnull().mean() * 100}")

    # Treatment methods
    print("\n--- Missing Value Treatment Methods ---")

    # 1. Drop rows
    df_dropna = df.dropna()
    print(f"\n1. Drop rows (dropna):\n{df_dropna}")

    # 2. Drop based on specific columns only
    df_drop_subset = df.dropna(subset=['A', 'C'])
    print(f"\n2. Drop based on columns A, C:\n{df_drop_subset}")

    # 3. Fill with values
    df_fillna = df.copy()
    df_fillna['A'] = df_fillna['A'].fillna(df_fillna['A'].mean())
    df_fillna['B'] = df_fillna['B'].fillna(df_fillna['B'].median())
    print(f"\n3. Fill with mean/median:\n{df_fillna}")

    # 4. Forward/backward fill
    df_ffill = df.fillna(method='ffill')
    print(f"\n4. Forward fill (ffill):\n{df_ffill}")

    # 5. Interpolation
    df_interpolate = df.copy()
    df_interpolate['A'] = df_interpolate['A'].interpolate()
    df_interpolate['B'] = df_interpolate['B'].interpolate()
    print(f"\n5. Interpolation:\n{df_interpolate}")


# =============================================================================
# 2. Outlier Detection and Treatment
# =============================================================================
def handle_outliers():
    """Detect and handle outliers"""
    print("\n[2] Outlier Detection and Treatment")
    print("=" * 50)

    np.random.seed(42)

    # Data with outliers
    normal_data = np.random.normal(100, 10, 100)
    outliers = np.array([200, -50, 250])
    data = np.concatenate([normal_data, outliers])
    np.random.shuffle(data)

    df = pd.DataFrame({'value': data})

    print(f"Data size: {len(df)}")
    print(f"Mean: {df['value'].mean():.2f}")
    print(f"Std dev: {df['value'].std():.2f}")

    # Method 1: IQR method
    print("\n--- IQR Method ---")
    Q1 = df['value'].quantile(0.25)
    Q3 = df['value'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    print(f"Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
    print(f"Normal range: [{lower_bound:.2f}, {upper_bound:.2f}]")

    outliers_iqr = df[(df['value'] < lower_bound) | (df['value'] > upper_bound)]
    print(f"Number of outliers: {len(outliers_iqr)}")
    print(f"Outlier values: {outliers_iqr['value'].values}")

    # Method 2: Z-score method
    print("\n--- Z-score Method ---")
    z_scores = np.abs((df['value'] - df['value'].mean()) / df['value'].std())
    outliers_z = df[z_scores > 3]
    print(f"Number of outliers (|z| > 3): {len(outliers_z)}")

    # Outlier treatment
    print("\n--- Outlier Treatment ---")

    # 1. Removal
    df_no_outliers = df[(df['value'] >= lower_bound) & (df['value'] <= upper_bound)]
    print(f"1. Size after removal: {len(df_no_outliers)}")

    # 2. Replace with boundary values (Winsorizing)
    df_winsorized = df.copy()
    df_winsorized['value'] = df_winsorized['value'].clip(lower_bound, upper_bound)
    print(f"2. Max value after winsorizing: {df_winsorized['value'].max():.2f}")

    # 3. Replace with median
    df_median = df.copy()
    median_val = df['value'].median()
    df_median.loc[(df['value'] < lower_bound) | (df['value'] > upper_bound), 'value'] = median_val
    print(f"3. Mean after median replacement: {df_median['value'].mean():.2f}")


# =============================================================================
# 3. Data Type Conversion
# =============================================================================
def data_type_conversion():
    """Data type conversion"""
    print("\n[3] Data Type Conversion")
    print("=" * 50)

    df = pd.DataFrame({
        'int_col': ['1', '2', '3', '4', '5'],
        'float_col': ['1.1', '2.2', '3.3', '4.4', '5.5'],
        'date_col': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
        'bool_col': ['True', 'False', 'True', 'False', 'True'],
        'cat_col': ['A', 'B', 'A', 'C', 'B']
    })

    print("Original data types:")
    print(df.dtypes)
    print()

    # Type conversion
    df['int_col'] = df['int_col'].astype(int)
    df['float_col'] = df['float_col'].astype(float)
    df['date_col'] = pd.to_datetime(df['date_col'])
    df['bool_col'] = df['bool_col'].map({'True': True, 'False': False})
    df['cat_col'] = df['cat_col'].astype('category')

    print("Data types after conversion:")
    print(df.dtypes)
    print()

    print("Converted data:")
    print(df)

    # Memory usage comparison
    print(f"\nCategory type memory savings:")
    print(f"  object type: {df['cat_col'].astype('object').memory_usage()} bytes")
    print(f"  category type: {df['cat_col'].memory_usage()} bytes")


# =============================================================================
# 4. Handling Duplicate Data
# =============================================================================
def handle_duplicates():
    """Handle duplicate data"""
    print("\n[4] Handling Duplicate Data")
    print("=" * 50)

    df = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Alice', 'Charlie', 'Bob', 'David'],
        'age': [25, 30, 25, 35, 30, 40],
        'city': ['Seoul', 'Busan', 'Seoul', 'Daegu', 'Busan', 'Seoul']
    })

    print("Original data:")
    print(df)

    # Check duplicates
    print(f"\nNumber of duplicate rows: {df.duplicated().sum()}")
    print("Duplicated rows:")
    print(df[df.duplicated()])

    # Duplicates based on specific column
    print(f"\nDuplicates based on 'name': {df.duplicated(subset=['name']).sum()}")

    # Remove duplicates
    df_unique = df.drop_duplicates()
    print(f"\nAfter removing duplicates:\n{df_unique}")

    df_unique_name = df.drop_duplicates(subset=['name'], keep='first')
    print(f"\nRemove duplicates by 'name' (keep first):\n{df_unique_name}")


# =============================================================================
# 5. Normalization and Standardization
# =============================================================================
def normalization_standardization():
    """Normalization and standardization"""
    print("\n[5] Normalization and Standardization")
    print("=" * 50)

    np.random.seed(42)

    df = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 10),
        'feature2': np.random.normal(50, 5, 10),
        'feature3': np.random.exponential(10, 10)
    })

    print("Original data statistics:")
    print(df.describe().round(2))

    # 1. Min-Max normalization (0-1 scaling)
    df_minmax = df.copy()
    for col in df.columns:
        min_val = df[col].min()
        max_val = df[col].max()
        df_minmax[col] = (df[col] - min_val) / (max_val - min_val)

    print("\n1. Min-Max normalization (0-1):")
    print(df_minmax.describe().round(4))

    # 2. Z-score standardization
    df_zscore = df.copy()
    for col in df.columns:
        mean_val = df[col].mean()
        std_val = df[col].std()
        df_zscore[col] = (df[col] - mean_val) / std_val

    print("\n2. Z-score standardization:")
    print(df_zscore.describe().round(4))

    # 3. Robust scaling (robust to outliers)
    df_robust = df.copy()
    for col in df.columns:
        median_val = df[col].median()
        iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
        df_robust[col] = (df[col] - median_val) / iqr

    print("\n3. Robust scaling (IQR-based):")
    print(df_robust.describe().round(4))


# =============================================================================
# 6. Categorical Variable Encoding
# =============================================================================
def categorical_encoding():
    """Categorical variable encoding"""
    print("\n[6] Categorical Variable Encoding")
    print("=" * 50)

    df = pd.DataFrame({
        'color': ['red', 'blue', 'green', 'blue', 'red'],
        'size': ['S', 'M', 'L', 'M', 'S'],
        'price': [100, 150, 200, 150, 100]
    })

    print("Original data:")
    print(df)

    # 1. Label encoding
    print("\n1. Label encoding:")
    df_label = df.copy()
    df_label['color_encoded'] = df_label['color'].astype('category').cat.codes
    df_label['size_encoded'] = df_label['size'].map({'S': 0, 'M': 1, 'L': 2})
    print(df_label)

    # 2. One-hot encoding
    print("\n2. One-hot encoding:")
    df_onehot = pd.get_dummies(df, columns=['color', 'size'])
    print(df_onehot)

    # 3. Frequency encoding
    print("\n3. Frequency encoding:")
    df_freq = df.copy()
    freq_map = df['color'].value_counts() / len(df)
    df_freq['color_freq'] = df_freq['color'].map(freq_map)
    print(df_freq)


# =============================================================================
# 7. String Processing
# =============================================================================
def string_processing():
    """String processing"""
    print("\n[7] String Processing")
    print("=" * 50)

    df = pd.DataFrame({
        'name': ['  John Doe  ', 'jane smith', 'BOB JONES', 'Alice Brown'],
        'email': ['john@example.com', 'jane@EXAMPLE.COM', 'bob@Example.com', 'alice@example.com'],
        'phone': ['010-1234-5678', '01098765432', '010 1111 2222', '010.3333.4444']
    })

    print("Original data:")
    print(df)

    # String processing
    df_clean = df.copy()

    # Strip whitespace and normalize case
    df_clean['name'] = df_clean['name'].str.strip().str.title()

    # Convert to lowercase
    df_clean['email'] = df_clean['email'].str.lower()

    # Normalize phone numbers
    df_clean['phone'] = df_clean['phone'].str.replace(r'[^0-9]', '', regex=True)

    print("\nCleaned data:")
    print(df_clean)

    # String extraction
    print("\nString splitting:")
    df_clean[['first_name', 'last_name']] = df_clean['name'].str.split(' ', n=1, expand=True)
    print(df_clean[['name', 'first_name', 'last_name']])


# =============================================================================
# 8. Date/Time Processing
# =============================================================================
def datetime_processing():
    """Date/time processing"""
    print("\n[8] Date/Time Processing")
    print("=" * 50)

    df = pd.DataFrame({
        'date_str': ['2024-01-15', '2024/02/20', '15-Mar-2024', '2024.04.10'],
        'timestamp': pd.date_range('2024-01-01', periods=4, freq='ME'),
        'value': [100, 150, 120, 180]
    })

    print("Original data:")
    print(df)

    # Date parsing
    df['date_parsed'] = pd.to_datetime(df['date_str'])

    # Extract date components
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['weekday'] = df['timestamp'].dt.day_name()
    df['quarter'] = df['timestamp'].dt.quarter

    print("\nExtracted date components:")
    print(df[['timestamp', 'year', 'month', 'day', 'weekday', 'quarter']])

    # Date arithmetic
    df['days_since'] = (pd.Timestamp('2024-12-31') - df['timestamp']).dt.days

    print("\nDate arithmetic (days until 2024-12-31):")
    print(df[['timestamp', 'days_since']])


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 60)
    print("Data Preprocessing Example")
    print("=" * 60)

    handle_missing_values()
    handle_outliers()
    data_type_conversion()
    handle_duplicates()
    normalization_standardization()
    categorical_encoding()
    string_processing()
    datetime_processing()

    print("\n" + "=" * 60)
    print("Data Preprocessing Checklist")
    print("=" * 60)
    print("""
    1. Load and Inspect Data
       - head(), info(), describe()
       - shape, dtypes

    2. Handle Missing Values
       - Check with isnull().sum()
       - Drop or impute (mean, median, mode, interpolation)

    3. Handle Outliers
       - Detect with IQR or Z-score
       - Remove, clip to boundaries, or transform

    4. Data Type Conversion
       - Convert to numeric, datetime, categorical as appropriate
       - Use category type for memory savings

    5. Remove Duplicates
       - Check with duplicated()
       - drop_duplicates()

    6. Scaling/Normalization
       - Min-Max: when range matters
       - Z-score: when distribution matters
       - Robust: when outliers are present

    7. Categorical Encoding
       - Label encoding: ordinal variables
       - One-hot encoding: nominal variables

    8. String/Date Cleaning
       - Strip whitespace, normalize case
       - Parse dates and extract components
    """)


if __name__ == "__main__":
    main()
