"""
Exercises for Lesson 25: Practical Projects
Topic: Data_Science

Solutions to practice problems from the lesson.
"""
import numpy as np


# === Exercise 1: EDA Report Generator ===
# Problem: Build a reusable EDA report function that covers
#          data types, missing values, statistics, and outliers.
def exercise_1():
    """Solution for automated EDA report generator.

    A good EDA report covers:
    1. Basic data overview (shape, types, memory)
    2. Missing value analysis
    3. Descriptive statistics for numeric columns
    4. Categorical variable summaries
    5. Outlier detection via IQR method
    """
    np.random.seed(42)

    # Generate a synthetic dataset
    n = 500
    data = {
        'age': np.random.normal(40, 12, n).clip(18, 80).astype(int),
        'income': np.random.lognormal(10.5, 0.8, n).astype(int),
        'score': np.random.normal(75, 15, n).clip(0, 100),
        'department': np.random.choice(
            ['Sales', 'Engineering', 'Marketing', 'HR'], n,
            p=[0.3, 0.35, 0.2, 0.15]
        ),
        'satisfaction': np.random.choice([1, 2, 3, 4, 5], n,
                                         p=[0.05, 0.1, 0.25, 0.35, 0.25]),
    }

    # Introduce some missing values
    missing_idx_age = np.random.choice(n, 15, replace=False)
    missing_idx_income = np.random.choice(n, 30, replace=False)
    ages = data['age'].astype(float)
    ages[missing_idx_age] = np.nan
    incomes = data['income'].astype(float)
    incomes[missing_idx_income] = np.nan
    data['age'] = ages
    data['income'] = incomes

    # Column names and values
    columns = list(data.keys())
    values = data

    def generate_report(col_names, col_data, n_rows):
        """Generate EDA report from column data."""
        print("=" * 60)
        print("        Exploratory Data Analysis (EDA) Report")
        print("=" * 60)

        # 1. Basic info
        print("\n1. Basic Data Information")
        print("-" * 40)
        print(f"   Rows: {n_rows:,}")
        print(f"   Columns: {len(col_names)}")

        # 2. Missing values
        print("\n2. Missing Value Overview")
        print("-" * 40)
        has_missing = False
        for col in col_names:
            arr = np.array(col_data[col], dtype=float) if \
                np.issubdtype(np.array(col_data[col]).dtype, np.floating) else \
                col_data[col]
            if isinstance(arr[0], (int, float, np.integer, np.floating)):
                n_miss = int(np.sum(np.isnan(np.array(arr, dtype=float))))
                if n_miss > 0:
                    has_missing = True
                    pct = n_miss / n_rows * 100
                    print(f"   {col}: {n_miss} ({pct:.1f}%)")
        if not has_missing:
            print("   No missing values")

        # 3. Numeric statistics
        print("\n3. Numeric Variable Statistics")
        print("-" * 40)
        for col in col_names:
            arr = np.array(col_data[col], dtype=float)
            if not np.all(np.isnan(arr)):
                try:
                    clean = arr[~np.isnan(arr)]
                    if len(np.unique(clean)) > 10:  # likely continuous
                        print(f"\n   [{col}]")
                        print(f"   Mean: {np.mean(clean):.2f}, "
                              f"Median: {np.median(clean):.2f}")
                        print(f"   Std: {np.std(clean, ddof=1):.2f}")
                        print(f"   Range: [{np.min(clean):.2f}, "
                              f"{np.max(clean):.2f}]")
                        q1 = np.percentile(clean, 25)
                        q3 = np.percentile(clean, 75)
                        print(f"   IQR: [{q1:.2f}, {q3:.2f}]")
                except (TypeError, ValueError):
                    pass

        # 4. Outlier detection (IQR method)
        print("\n4. Outlier Detection (IQR Method)")
        print("-" * 40)
        for col in col_names:
            try:
                arr = np.array(col_data[col], dtype=float)
                clean = arr[~np.isnan(arr)]
                if len(np.unique(clean)) > 10:
                    q1 = np.percentile(clean, 25)
                    q3 = np.percentile(clean, 75)
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    n_outliers = int(np.sum((clean < lower) | (clean > upper)))
                    pct = n_outliers / len(clean) * 100
                    flag = " *** CHECK" if pct > 5 else ""
                    print(f"   {col}: {n_outliers} outliers ({pct:.1f}%){flag}")
            except (TypeError, ValueError):
                pass

        # 5. Skewness
        print("\n5. Skewness Assessment")
        print("-" * 40)
        for col in col_names:
            try:
                arr = np.array(col_data[col], dtype=float)
                clean = arr[~np.isnan(arr)]
                if len(np.unique(clean)) > 10:
                    mean_val = np.mean(clean)
                    std_val = np.std(clean, ddof=1)
                    n_c = len(clean)
                    skew = (n_c / ((n_c - 1) * (n_c - 2))) * \
                           np.sum(((clean - mean_val) / std_val) ** 3)
                    label = "highly skewed" if abs(skew) > 1 else \
                            "moderately skewed" if abs(skew) > 0.5 else "symmetric"
                    suggest = " (consider log transform)" if skew > 1 else ""
                    print(f"   {col}: skewness={skew:.3f} ({label}){suggest}")
            except (TypeError, ValueError):
                pass

        print("\n" + "=" * 60)
        print("                     End of Report")
        print("=" * 60)

    generate_report(columns, values, n)


# === Exercise 2: Missing Value Strategy Comparison ===
# Problem: Compare four imputation strategies and measure their
#          impact on downstream statistics.
def exercise_2():
    """Solution for comparing missing value imputation strategies.

    Strategies:
    1. Listwise deletion (drop rows)
    2. Mean imputation
    3. Median imputation
    4. Group median imputation

    Mean imputation artificially reduces variance and can distort
    correlations. Median imputation is more robust to outliers.
    Group-based imputation preserves between-group differences.
    """
    np.random.seed(42)

    n = 500
    # Simulate data with class-dependent fares
    pclass = np.random.choice([1, 2, 3], n, p=[0.25, 0.35, 0.40])
    fare = np.where(pclass == 1, np.random.lognormal(4.5, 0.5, n),
            np.where(pclass == 2, np.random.lognormal(3.5, 0.4, n),
                     np.random.lognormal(2.8, 0.5, n)))
    survived = np.random.binomial(1, 0.3 + 0.1 * (fare / np.max(fare)), n)

    # Original statistics
    orig_mean = np.mean(fare)
    orig_std = np.std(fare, ddof=1)
    orig_corr = np.corrcoef(fare, survived)[0, 1]

    print("Missing Value Imputation Strategy Comparison:")
    print(f"  Original: mean={orig_mean:.2f}, std={orig_std:.2f}, "
          f"corr(fare,survived)={orig_corr:.4f}")
    print(f"  N={n}, introducing 20% missing in fare column")

    # Introduce 20% missing at random
    missing_mask = np.random.choice([True, False], n, p=[0.2, 0.8])
    fare_missing = fare.copy()
    fare_missing[missing_mask] = np.nan

    n_missing = int(np.sum(missing_mask))
    print(f"  Missing values: {n_missing} ({n_missing/n*100:.0f}%)")

    strategies = {}

    # Strategy 1: Drop rows
    keep = ~missing_mask
    fare_drop = fare_missing[keep]
    surv_drop = survived[keep]
    strategies['Drop rows'] = (
        np.mean(fare_drop), np.std(fare_drop, ddof=1),
        np.corrcoef(fare_drop, surv_drop)[0, 1], len(fare_drop)
    )

    # Strategy 2: Mean imputation
    fare_mean_imp = fare_missing.copy()
    global_mean = np.nanmean(fare_mean_imp)
    fare_mean_imp[missing_mask] = global_mean
    strategies['Mean fill'] = (
        np.mean(fare_mean_imp), np.std(fare_mean_imp, ddof=1),
        np.corrcoef(fare_mean_imp, survived)[0, 1], n
    )

    # Strategy 3: Median imputation
    fare_median_imp = fare_missing.copy()
    global_median = np.nanmedian(fare_median_imp)
    fare_median_imp[missing_mask] = global_median
    strategies['Median fill'] = (
        np.mean(fare_median_imp), np.std(fare_median_imp, ddof=1),
        np.corrcoef(fare_median_imp, survived)[0, 1], n
    )

    # Strategy 4: Group median (by pclass)
    fare_group_imp = fare_missing.copy()
    for cls in [1, 2, 3]:
        cls_mask = pclass == cls
        cls_median = np.nanmedian(fare_missing[cls_mask])
        fill_mask = missing_mask & cls_mask
        fare_group_imp[fill_mask] = cls_median
    strategies['Group median'] = (
        np.mean(fare_group_imp), np.std(fare_group_imp, ddof=1),
        np.corrcoef(fare_group_imp, survived)[0, 1], n
    )

    # Comparison table
    print(f"\n  {'Strategy':<18} {'Mean':>10} {'Std':>10} {'Corr':>10} {'N':>6}")
    print(f"  {'-' * 58}")
    print(f"  {'Original':<18} {orig_mean:>10.2f} {orig_std:>10.2f} "
          f"{orig_corr:>10.4f} {n:>6}")
    print(f"  {'-' * 58}")

    best_corr_err = float('inf')
    best_strategy = ''
    for name, (m, s, c, nn) in strategies.items():
        corr_err = abs(c - orig_corr)
        if corr_err < best_corr_err:
            best_corr_err = corr_err
            best_strategy = name
        print(f"  {name:<18} {m:>10.2f} {s:>10.2f} {c:>10.4f} {nn:>6}")

    print(f"\n  Best correlation preservation: {best_strategy}")
    print(f"  Note: Mean imputation reduces variance because all imputed")
    print(f"  values are identical. Group median better preserves the")
    print(f"  relationship between fare and class.")


# === Exercise 3: Correlation Analysis and Heatmap Data ===
# Problem: Compute the full correlation matrix and identify
#          the strongest positive and negative correlations.
def exercise_3():
    """Solution for correlation matrix analysis.

    Pearson correlation measures linear association (-1 to 1).
    For a matrix of p variables, there are p*(p-1)/2 unique pairs.
    Identifying the strongest correlations helps detect:
    - Multicollinearity (strong positive)
    - Inverse relationships (strong negative)
    - Candidate features for dimensionality reduction
    """
    np.random.seed(42)

    n = 300
    # Generate correlated features
    x1 = np.random.normal(0, 1, n)
    x2 = 0.8 * x1 + np.random.normal(0, 0.6, n)        # strong positive
    x3 = -0.6 * x1 + np.random.normal(0, 0.7, n)       # moderate negative
    x4 = np.random.normal(0, 1, n)                       # independent
    x5 = 0.4 * x2 + 0.3 * x4 + np.random.normal(0, 0.5, n)
    x6 = -0.9 * x2 + np.random.normal(0, 0.3, n)       # strong negative with x2

    features = {'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5, 'x6': x6}
    names = list(features.keys())
    p = len(names)

    # Compute correlation matrix
    data_matrix = np.column_stack([features[n] for n in names])
    corr_matrix = np.corrcoef(data_matrix, rowvar=False)

    print("Correlation Matrix Analysis:")
    print(f"  Variables: {names}")
    print(f"  Unique pairs: {p * (p - 1) // 2}")

    # Print correlation matrix
    header = "       " + "  ".join(f"{n:>6}" for n in names)
    print(f"\n{header}")
    for i, name in enumerate(names):
        row = f"  {name:>4} " + "  ".join(
            f"{corr_matrix[i, j]:>6.3f}" for j in range(p)
        )
        print(row)

    # Collect all unique pairs with correlations
    pairs = []
    for i in range(p):
        for j in range(i + 1, p):
            pairs.append((names[i], names[j], corr_matrix[i, j]))

    # Sort by absolute correlation
    pairs_sorted = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)

    # Top 5 strongest
    print(f"\n  Top 5 Strongest Correlations:")
    print(f"    {'Pair':<12} {'r':>8} {'Strength':>12}")
    print(f"    {'-' * 35}")
    for v1, v2, r in pairs_sorted[:5]:
        strength = "strong" if abs(r) > 0.7 else \
                   "moderate" if abs(r) > 0.4 else "weak"
        direction = "positive" if r > 0 else "negative"
        print(f"    {v1}-{v2:<7} {r:>8.4f} {direction + ' ' + strength:>12}")

    # Positive vs negative
    positive = [(v1, v2, r) for v1, v2, r in pairs_sorted if r > 0]
    negative = [(v1, v2, r) for v1, v2, r in pairs_sorted if r < 0]

    print(f"\n  Strongest positive: {positive[0][0]}-{positive[0][1]} "
          f"(r={positive[0][2]:.4f})")
    print(f"  Strongest negative: {negative[0][0]}-{negative[0][1]} "
          f"(r={negative[0][2]:.4f})")

    # Flag multicollinearity
    print(f"\n  Multicollinearity warnings (|r| > 0.7):")
    flagged = [p for p in pairs_sorted if abs(p[2]) > 0.7]
    if flagged:
        for v1, v2, r in flagged:
            print(f"    {v1}-{v2}: r={r:.4f} *** Consider dropping one")
    else:
        print(f"    None detected")


if __name__ == "__main__":
    print("=== Exercise 1: EDA Report Generator ===")
    exercise_1()
    print("\n=== Exercise 2: Missing Value Strategy Comparison ===")
    exercise_2()
    print("\n=== Exercise 3: Correlation Analysis ===")
    exercise_3()
    print("\nAll exercises completed!")
