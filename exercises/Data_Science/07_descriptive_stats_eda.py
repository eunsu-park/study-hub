"""
Exercises for Lesson 07: Descriptive Statistics & EDA
Topic: Data_Science

Solutions to practice problems from the lesson.
"""
import numpy as np
import pandas as pd


# === Exercise 1: Descriptive Statistics ===
# Problem: Compute the five-number summary for the given data.
def exercise_1():
    """Solution computing the five-number summary (min, Q1, median, Q3, max).

    The five-number summary provides a concise picture of a distribution's
    location and spread. It forms the basis of box plots and is robust
    to outliers (unlike mean and standard deviation).
    """
    data = [12, 15, 18, 22, 25, 28, 30, 35, 40, 100]
    s = pd.Series(data)

    print("Data:", data)
    print("\nFive-Number Summary:")
    print(f"  Min:    {s.min()}")
    print(f"  Q1:     {s.quantile(0.25)}")
    print(f"  Median: {s.median()}")
    print(f"  Q3:     {s.quantile(0.75)}")
    print(f"  Max:    {s.max()}")

    # Additional context: the large gap between Q3 and Max suggests
    # the value 100 may be an outlier
    IQR = s.quantile(0.75) - s.quantile(0.25)
    upper_fence = s.quantile(0.75) + 1.5 * IQR
    print(f"\n  IQR:           {IQR}")
    print(f"  Upper fence:   {upper_fence}")
    print(f"  100 > {upper_fence}? {'Yes -> outlier' if 100 > upper_fence else 'No'}")

    # Compare mean vs median to detect skewness
    print(f"\n  Mean:   {s.mean():.2f}")
    print(f"  Median: {s.median():.2f}")
    print(f"  Skewness: {s.skew():.4f}")
    print("  (Mean > Median indicates right-skewed distribution)")


# === Exercise 2: Correlation Analysis ===
# Problem: Compute and interpret the correlation coefficient between two variables.
def exercise_2():
    """Solution computing Pearson correlation and interpreting its strength.

    Pearson's r measures the strength and direction of a linear relationship
    between two continuous variables. Values range from -1 to +1.
    """
    df = pd.DataFrame({
        'study_hours': [1, 2, 3, 4, 5, 6, 7, 8],
        'score': [50, 55, 60, 65, 70, 80, 85, 95]
    })
    print("Data:")
    print(df)

    # Pearson correlation coefficient
    corr = df['study_hours'].corr(df['score'])
    print(f"\nPearson correlation: {corr:.4f}")

    # Interpretation guidelines (Cohen's conventions):
    # |r| < 0.3:  weak
    # 0.3 <= |r| < 0.7: moderate
    # |r| >= 0.7: strong
    abs_corr = abs(corr)
    if abs_corr >= 0.7:
        strength = "strong"
    elif abs_corr >= 0.3:
        strength = "moderate"
    else:
        strength = "weak"
    direction = "positive" if corr > 0 else "negative"
    print(f"Interpretation: {strength} {direction} correlation")
    print(f"  As study hours increase, scores tend to increase.")

    # R-squared: proportion of variance in score explained by study_hours
    r_squared = corr ** 2
    print(f"\nR-squared: {r_squared:.4f}")
    print(f"  {r_squared*100:.1f}% of score variance is explained by study hours.")

    # Full correlation matrix
    print("\nCorrelation matrix:")
    print(df.corr().round(4))


# === Exercise 3: Group Comparison ===
# Problem: Compute mean and standard deviation by group.
def exercise_3():
    """Solution demonstrating group-level descriptive statistics."""
    df = pd.DataFrame({
        'group': ['A', 'A', 'B', 'B', 'A', 'B'],
        'value': [10, 12, 20, 22, 11, 21]
    })
    print("Data:")
    print(df)

    # groupby + agg computes statistics separately for each group
    result = df.groupby('group')['value'].agg(['mean', 'std', 'count'])
    print("\nGroup statistics:")
    print(result)

    # The groups are clearly separated: A ~ 11 vs B ~ 21
    # A formal test (like a t-test) would confirm this difference
    from scipy import stats
    group_a = df.loc[df['group'] == 'A', 'value']
    group_b = df.loc[df['group'] == 'B', 'value']

    t_stat, p_value = stats.ttest_ind(group_a, group_b)
    print(f"\nIndependent t-test:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Significant at alpha=0.05? {'Yes' if p_value < 0.05 else 'No'}")

    # Effect size (Cohen's d)
    pooled_std = np.sqrt(
        ((len(group_a) - 1) * group_a.std(ddof=1)**2 +
         (len(group_b) - 1) * group_b.std(ddof=1)**2) /
        (len(group_a) + len(group_b) - 2)
    )
    cohens_d = (group_b.mean() - group_a.mean()) / pooled_std
    print(f"\n  Cohen's d: {cohens_d:.4f}")
    print(f"  Effect size: {'large' if abs(cohens_d) >= 0.8 else 'medium' if abs(cohens_d) >= 0.5 else 'small'}")


if __name__ == "__main__":
    print("=== Exercise 1: Descriptive Statistics ===")
    exercise_1()
    print("\n=== Exercise 2: Correlation Analysis ===")
    exercise_2()
    print("\n=== Exercise 3: Group Comparison ===")
    exercise_3()
    print("\nAll exercises completed!")
