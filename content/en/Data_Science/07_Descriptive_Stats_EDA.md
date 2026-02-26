# 7. Descriptive Stats & EDA

[Previous: Data Preprocessing](./06_Data_Preprocessing.md) | [Next: Data Visualization Basics](./08_Data_Visualization_Basics.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Calculate measures of central tendency including mean, median, mode, and trimmed mean
2. Calculate measures of dispersion including variance, standard deviation, IQR, and coefficient of variation
3. Describe distribution shape using skewness and kurtosis and interpret their values
4. Apply the `describe` method and compute correlation and covariance matrices on DataFrames
5. Implement a systematic EDA workflow covering univariate, bivariate, and multivariate analysis
6. Apply normality tests (Shapiro-Wilk, Kolmogorov-Smirnov, Anderson-Darling) and interpret Q-Q plots
7. Compare groups using t-tests, ANOVA, and chi-square tests within an exploratory context
8. Identify when to use EDA automation libraries such as ydata-profiling and sweetviz

---

Before you can model data or draw conclusions, you need to understand what you are working with. Descriptive statistics condense thousands of data points into a handful of interpretable numbers, while EDA uses visualization and systematic exploration to reveal hidden patterns, anomalies, and relationships. Together they form the indispensable first stage of every data science project, guiding every decision that follows.

---

## 1. Descriptive Statistics

### 1.1 Measures of Central Tendency

```python
import pandas as pd
import numpy as np
from scipy import stats

data = [10, 15, 20, 25, 30, 35, 40, 100]  # includes outlier
s = pd.Series(data)

# Mean
print(f"Mean: {s.mean():.2f}")  # 34.38

# Median
print(f"Median: {s.median():.2f}")  # 27.50

# Mode
data_with_mode = [1, 2, 2, 3, 3, 3, 4, 4]
print(f"Mode: {pd.Series(data_with_mode).mode().values}")  # [3]

# Trimmed Mean - reduces outlier influence
# Choose trimmed mean over regular mean when you suspect outliers but don't want to
# remove them entirely: the regular mean is pulled toward the outlier (here: 100),
# while the trimmed mean discards the top and bottom 10% of values before averaging,
# giving a more stable estimate of the "typical" value without permanently deleting data.
print(f"Trimmed Mean (10%): {stats.trim_mean(data, 0.1):.2f}")

# Weighted Mean
values = [10, 20, 30]
weights = [0.2, 0.3, 0.5]
weighted_mean = np.average(values, weights=weights)
print(f"Weighted Mean: {weighted_mean}")
```

### 1.2 Measures of Dispersion

```python
s = pd.Series([10, 15, 20, 25, 30, 35, 40])

# Range
print(f"Range: {s.max() - s.min()}")

# Variance
print(f"Variance (sample): {s.var():.2f}")
print(f"Variance (population): {s.var(ddof=0):.2f}")

# Standard Deviation
print(f"Std Dev (sample): {s.std():.2f}")
print(f"Std Dev (population): {s.std(ddof=0):.2f}")

# Interquartile Range (IQR)
Q1 = s.quantile(0.25)
Q3 = s.quantile(0.75)
IQR = Q3 - Q1
print(f"IQR: {IQR}")

# Coefficient of Variation (CV) - relative dispersion
# CV tells you what std dev alone cannot: how large the spread is *relative to the
# mean*. This matters when comparing datasets with different units or scales —
# e.g., income in dollars (std ≈ 20,000) vs height in cm (std ≈ 8) are incomparable
# by std dev alone, but CV (std/mean) expresses both as a unitless fraction,
# making cross-dataset comparisons meaningful.
cv = s.std() / s.mean()
print(f"CV: {cv:.4f}")

# Mean Absolute Deviation (MAD)
mad = (s - s.mean()).abs().mean()
print(f"MAD: {mad:.2f}")
```

### 1.3 Measures of Distribution Shape

```python
s = pd.Series([1, 2, 2, 3, 3, 3, 4, 4, 5, 10])

# Skewness - degree of asymmetry
print(f"Skewness: {s.skew():.4f}")
# > 0: right tail (positive skew)
# < 0: left tail (negative skew)
# = 0: symmetric

# Kurtosis - tail thickness
print(f"Kurtosis: {s.kurtosis():.4f}")
# > 0: more peaked than normal (heavy tails)
# < 0: flatter than normal (thin tails)
# = 0: similar to normal distribution

# Using scipy (Fisher definition)
print(f"Skewness (scipy): {stats.skew(s):.4f}")
print(f"Kurtosis (scipy): {stats.kurtosis(s):.4f}")
```

### 1.4 Percentiles and Quantiles

```python
s = pd.Series(range(1, 101))

# Percentiles
print(f"25th percentile: {s.quantile(0.25)}")
print(f"50th percentile: {s.quantile(0.50)}")
print(f"75th percentile: {s.quantile(0.75)}")
print(f"90th percentile: {s.quantile(0.90)}")

# Multiple quantiles at once
print(s.quantile([0.1, 0.25, 0.5, 0.75, 0.9]))

# Five-number summary
print("Five-number summary:")
print(f"Min: {s.min()}")
print(f"Q1: {s.quantile(0.25)}")
print(f"Median: {s.median()}")
print(f"Q3: {s.quantile(0.75)}")
print(f"Max: {s.max()}")
```

---

## 2. DataFrame Descriptive Statistics

### 2.1 The describe Method

```python
df = pd.DataFrame({
    'age': [25, 30, 35, 40, 45, 50, 55],
    'salary': [40000, 45000, 50000, 60000, 70000, 80000, 100000],
    'department': ['IT', 'HR', 'IT', 'Sales', 'IT', 'HR', 'Sales']
})

# Numeric columns only
print(df.describe())

# All columns (including categorical)
print(df.describe(include='all'))

# Custom percentiles
print(df.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]))
```

### 2.2 Correlation Analysis

```python
df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [2, 4, 5, 4, 5],
    'C': [5, 4, 3, 2, 1]
})

# Pearson correlation coefficient (linear relationship)
print("Pearson Correlation:")
print(df.corr())

# Spearman correlation coefficient (rank-based, non-linear relationship)
print("\nSpearman Correlation:")
print(df.corr(method='spearman'))

# Kendall correlation coefficient (rank-based)
print("\nKendall Correlation:")
print(df.corr(method='kendall'))

# Correlation between specific columns
print(f"\nCorrelation between A and B: {df['A'].corr(df['B']):.4f}")

# With p-value
from scipy.stats import pearsonr, spearmanr
corr, p_value = pearsonr(df['A'], df['B'])
print(f"Correlation: {corr:.4f}, p-value: {p_value:.4f}")
```

### 2.3 Covariance

```python
df = pd.DataFrame({
    'X': [1, 2, 3, 4, 5],
    'Y': [2, 4, 5, 4, 5]
})

# Covariance matrix
print(df.cov())

# Covariance between two variables
print(f"Covariance of X and Y: {df['X'].cov(df['Y']):.4f}")
```

---

## 3. Exploratory Data Analysis (EDA)

### 3.1 Data Overview

```python
import pandas as pd
import numpy as np

# Generate sample data
np.random.seed(42)
df = pd.DataFrame({
    'id': range(1, 1001),
    'age': np.random.randint(18, 70, 1000),
    'income': np.random.lognormal(10, 1, 1000),
    'gender': np.random.choice(['M', 'F'], 1000),
    'category': np.random.choice(['A', 'B', 'C', 'D'], 1000),
    'score': np.random.normal(75, 15, 1000)
})
df.loc[np.random.choice(1000, 50), 'income'] = np.nan  # add missing values

# 1. Basic information
print("="*50)
print("1. Basic Data Information")
print("="*50)
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"\nColumn list: {df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")

# 2. Missing value summary
print("\n" + "="*50)
print("2. Missing Value Summary")
print("="*50)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Missing Rate (%)': missing_pct
})
print(missing_df[missing_df['Missing Count'] > 0])

# 3. Descriptive statistics
print("\n" + "="*50)
print("3. Descriptive Statistics for Numeric Variables")
print("="*50)
print(df.describe())

# 4. Categorical variable frequencies
print("\n" + "="*50)
print("4. Categorical Variable Frequencies")
print("="*50)
for col in df.select_dtypes(include='object').columns:
    print(f"\n[{col}]")
    print(df[col].value_counts())
```

### 3.2 Univariate Analysis

```python
import matplotlib.pyplot as plt

# Numeric variables
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove('id')

fig, axes = plt.subplots(2, len(numeric_cols), figsize=(15, 8))

for i, col in enumerate(numeric_cols):
    # Histogram
    axes[0, i].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
    axes[0, i].set_title(f'{col} - Histogram')
    axes[0, i].set_xlabel(col)
    axes[0, i].set_ylabel('Frequency')

    # Boxplot
    axes[1, i].boxplot(df[col].dropna())
    axes[1, i].set_title(f'{col} - Boxplot')

plt.tight_layout()
plt.show()

# Categorical variables
categorical_cols = df.select_dtypes(include='object').columns.tolist()

fig, axes = plt.subplots(1, len(categorical_cols), figsize=(12, 4))

for i, col in enumerate(categorical_cols):
    df[col].value_counts().plot(kind='bar', ax=axes[i], edgecolor='black')
    axes[i].set_title(f'{col} - Bar Chart')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Count')
    axes[i].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.show()
```

### 3.3 Bivariate Analysis

```python
import seaborn as sns

# Numeric vs Numeric: Scatter plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(df['age'], df['income'], alpha=0.5)
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Income')
axes[0].set_title('Age vs Income')

axes[1].scatter(df['age'], df['score'], alpha=0.5)
axes[1].set_xlabel('Age')
axes[1].set_ylabel('Score')
axes[1].set_title('Age vs Score')

plt.tight_layout()
plt.show()

# Categorical vs Numeric: Boxplot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

df.boxplot(column='income', by='gender', ax=axes[0])
axes[0].set_title('Income by Gender')

df.boxplot(column='score', by='category', ax=axes[1])
axes[1].set_title('Score by Category')

plt.suptitle('')  # remove auto-generated title
plt.tight_layout()
plt.show()

# Categorical vs Categorical: Cross-tabulation
print("Cross-tabulation of Gender and Category:")
print(pd.crosstab(df['gender'], df['category']))
print(pd.crosstab(df['gender'], df['category'], normalize='index'))  # row-wise proportions
```

### 3.4 Multivariate Analysis

```python
# Correlation matrix heatmap
numeric_df = df[['age', 'income', 'score']].dropna()
correlation_matrix = numeric_df.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm',
            center=0, vmin=-1, vmax=1, fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# Scatter matrix (Pair Plot)
sns.pairplot(df[['age', 'income', 'score', 'gender']].dropna(),
             hue='gender', diag_kind='hist')
plt.suptitle('Pair Plot', y=1.02)
plt.tight_layout()
plt.show()
```

---

## 4. Distribution Analysis

### 4.1 Checking Distributions

```python
from scipy import stats

data = df['income'].dropna()

# Normality test - Shapiro-Wilk (n < 5000)
if len(data) < 5000:
    stat, p_value = stats.shapiro(data[:5000])
    print(f"Shapiro-Wilk test: statistic={stat:.4f}, p-value={p_value:.4f}")

# Normality test - Kolmogorov-Smirnov
stat, p_value = stats.kstest(data, 'norm',
                             args=(data.mean(), data.std()))
print(f"K-S test: statistic={stat:.4f}, p-value={p_value:.4f}")

# Normality test - Anderson-Darling
result = stats.anderson(data, dist='norm')
print(f"Anderson-Darling test: statistic={result.statistic:.4f}")

# Visual check: Q-Q plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Histogram with normal distribution curve
axes[0].hist(data, bins=50, density=True, alpha=0.7, edgecolor='black')
x = np.linspace(data.min(), data.max(), 100)
axes[0].plot(x, stats.norm.pdf(x, data.mean(), data.std()),
             'r-', linewidth=2, label='Normal')
axes[0].set_title('Histogram with Normal Curve')
axes[0].legend()

# Q-Q plot
stats.probplot(data, dist="norm", plot=axes[1])
axes[1].set_title('Q-Q Plot')

plt.tight_layout()
plt.show()
```

### 4.2 Distribution Transformation

```python
# Log-normally distributed data
data = df['income'].dropna()

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Original
axes[0, 0].hist(data, bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].set_title(f'Original (Skew: {stats.skew(data):.2f})')

# Log transformation
log_data = np.log1p(data)
axes[0, 1].hist(log_data, bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].set_title(f'Log Transform (Skew: {stats.skew(log_data):.2f})')

# Square root transformation
sqrt_data = np.sqrt(data)
axes[1, 0].hist(sqrt_data, bins=50, edgecolor='black', alpha=0.7)
axes[1, 0].set_title(f'Square Root Transform (Skew: {stats.skew(sqrt_data):.2f})')

# Box-Cox transformation
boxcox_data, lambda_param = stats.boxcox(data)
axes[1, 1].hist(boxcox_data, bins=50, edgecolor='black', alpha=0.7)
axes[1, 1].set_title(f'Box-Cox Transform (λ={lambda_param:.2f}, Skew: {stats.skew(boxcox_data):.2f})')

plt.tight_layout()
plt.show()
```

---

## 5. Group-wise Analysis

### 5.1 Group-wise Descriptive Statistics

```python
# Group-wise summary
print("Statistics by gender group:")
print(df.groupby('gender')[['age', 'income', 'score']].agg(['mean', 'std', 'count']))

print("\nStatistics by category:")
print(df.groupby('category')[['income', 'score']].describe())

# Multiple groups
print("\nMean income by gender and category:")
print(df.groupby(['gender', 'category'])['income'].mean().unstack())
```

### 5.2 Between-Group Comparisons

```python
# t-test (two groups)
male_income = df[df['gender'] == 'M']['income'].dropna()
female_income = df[df['gender'] == 'F']['income'].dropna()

stat, p_value = stats.ttest_ind(male_income, female_income)
print(f"Independent samples t-test: t={stat:.4f}, p-value={p_value:.4f}")

# ANOVA (three or more groups)
groups = [df[df['category'] == cat]['score'].dropna() for cat in df['category'].unique()]
stat, p_value = stats.f_oneway(*groups)
print(f"ANOVA: F={stat:.4f}, p-value={p_value:.4f}")

# Chi-square test (categorical variables)
contingency_table = pd.crosstab(df['gender'], df['category'])
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
print(f"Chi-square test: χ²={chi2:.4f}, p-value={p_value:.4f}")
```

---

## 6. EDA Automation Libraries

### 6.1 pandas-profiling (ydata-profiling)

Automated profiling is most appropriate when you are starting fresh on an unfamiliar dataset and need a broad overview quickly — it catches missing-value patterns, distributions, correlations, and duplicates in one shot. For production pipelines or when you need targeted domain-specific checks, manual EDA gives you more control and is more reproducible.

```python
# pip install ydata-profiling

# from ydata_profiling import ProfileReport

# report = ProfileReport(df, title="EDA Report", explorative=True)
# report.to_file("eda_report.html")

# Minimal version (for large datasets)
# report = ProfileReport(df, minimal=True)
```

### 6.2 sweetviz

```python
# pip install sweetviz

# import sweetviz as sv

# report = sv.analyze(df)
# report.show_html("sweetviz_report.html")

# Compare two datasets — particularly useful for train/test split validation:
# if the two distributions diverge strongly, your model may face distribution shift.
# report = sv.compare(df1, df2)
```

---

## 7. EDA Checklist

```markdown
## EDA Checklist

### 1. Data Overview
- [ ] Check number of rows and columns
- [ ] Check data types
- [ ] Check memory usage

### 2. Missing Values
- [ ] Check for missing values
- [ ] Check missing value rate
- [ ] Check missing value patterns (MCAR, MAR, MNAR)

### 3. Numeric Variables
- [ ] Descriptive statistics (mean, median, std, etc.)
- [ ] Distribution shape (skewness, kurtosis)
- [ ] Check for outliers
- [ ] Histogram / Boxplot

### 4. Categorical Variables
- [ ] Number of categories
- [ ] Frequency distribution
- [ ] Check for rare categories

### 5. Relationships Between Variables
- [ ] Correlation analysis (numeric)
- [ ] Cross-tabulation (categorical)
- [ ] Group-wise comparisons

### 6. Target Variable
- [ ] Target distribution (class imbalance)
- [ ] Relationship between target and features
```

---

## Practice Problems

### Problem 1: Descriptive Statistics
Compute the five-number summary for the following data.

```python
data = [12, 15, 18, 22, 25, 28, 30, 35, 40, 100]
s = pd.Series(data)

# Solution
print(f"Min: {s.min()}")
print(f"Q1: {s.quantile(0.25)}")
print(f"Median: {s.median()}")
print(f"Q3: {s.quantile(0.75)}")
print(f"Max: {s.max()}")
```

### Problem 2: Correlation Analysis
Compute and interpret the correlation coefficient between two variables.

```python
df = pd.DataFrame({
    'study_hours': [1, 2, 3, 4, 5, 6, 7, 8],
    'score': [50, 55, 60, 65, 70, 80, 85, 95]
})

# Solution
corr = df['study_hours'].corr(df['score'])
print(f"Correlation: {corr:.4f}")
# Strong positive correlation (score increases with study hours)
```

### Problem 3: Group Comparison
Compute mean and standard deviation by group.

```python
df = pd.DataFrame({
    'group': ['A', 'A', 'B', 'B', 'A', 'B'],
    'value': [10, 12, 20, 22, 11, 21]
})

# Solution
print(df.groupby('group')['value'].agg(['mean', 'std']))
```

---

## Summary

| Measure Type | Measure | Function |
|----------|------|------|
| Central Tendency | Mean, Median, Mode | `mean()`, `median()`, `mode()` |
| Dispersion | Variance, Std Dev, IQR | `var()`, `std()`, `quantile()` |
| Distribution Shape | Skewness, Kurtosis | `skew()`, `kurtosis()` |
| Relationship | Correlation, Covariance | `corr()`, `cov()` |
| Summary | Descriptive Statistics | `describe()` |
