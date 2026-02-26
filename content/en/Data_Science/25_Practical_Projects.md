# 25. Practical Projects - Comprehensive Data Analysis Practice

[Previous: Experimental Design](./24_Experimental_Design.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Apply a structured EDA workflow from data loading through insight generation on a real dataset
2. Identify and visualize missing data patterns and make informed decisions about handling them
3. Perform univariate, bivariate, and multivariate analyses using appropriate visualization techniques
4. Conduct statistical tests (chi-square, t-test) to validate observed patterns in the data
5. Synthesize analysis findings into clear, actionable insights
6. Build a reusable EDA report template that automates common exploratory tasks

---

Learning individual statistical techniques is only the first step. The real challenge is knowing how to combine them into a coherent end-to-end analysis that starts with raw data and ends with actionable insights. These practical projects walk you through complete analyses on real datasets, building the judgment and workflow habits that separate a textbook reader from a practicing data scientist.

---

## Project 1: Titanic Survival Analysis

### 1.1 Data Loading and Overview

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Seaborn built-in dataset
titanic = sns.load_dataset('titanic')

# Data overview
print("="*50)
print("Basic Data Information")
print("="*50)
print(f"Rows: {len(titanic)}")
print(f"Columns: {len(titanic.columns)}")
print(f"\nColumn list:\n{titanic.columns.tolist()}")
print(f"\nData types:\n{titanic.dtypes}")

# First 5 rows
print("\n" + "="*50)
print("Data Preview")
print("="*50)
print(titanic.head())
```

### 1.2 Missing Value Analysis

```python
print("="*50)
print("Missing Value Analysis")
print("="*50)

# Missing value summary
missing = titanic.isnull().sum()
missing_pct = (missing / len(titanic) * 100).round(2)
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Missing Rate (%)': missing_pct
}).sort_values('Missing Rate (%)', ascending=False)

print(missing_df[missing_df['Missing Count'] > 0])

# Visualization
fig, ax = plt.subplots(figsize=(10, 6))
missing_cols = missing_df[missing_df['Missing Count'] > 0].index
missing_vals = missing_df.loc[missing_cols, 'Missing Rate (%)']
ax.barh(missing_cols, missing_vals, color='coral')
ax.set_xlabel('Missing Rate (%)')
ax.set_title('Missing Value Overview')
for i, v in enumerate(missing_vals):
    ax.text(v + 0.5, i, f'{v}%', va='center')
plt.tight_layout()
plt.show()
```

### 1.3 Target Variable Analysis

```python
print("="*50)
print("Target Variable (Survival) Analysis")
print("="*50)

# Survival rate
survival_rate = titanic['survived'].value_counts(normalize=True)
print(f"Survival rate: {survival_rate[1]:.1%}")
print(f"Death rate: {survival_rate[0]:.1%}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Frequency
titanic['survived'].value_counts().plot(kind='bar', ax=axes[0],
                                         color=['coral', 'steelblue'])
axes[0].set_title('Survival Count')
axes[0].set_xticklabels(['Died', 'Survived'], rotation=0)
axes[0].set_ylabel('Count')

# Proportion
titanic['survived'].value_counts().plot(kind='pie', ax=axes[1],
                                         autopct='%1.1f%%',
                                         colors=['coral', 'steelblue'],
                                         labels=['Died', 'Survived'])
axes[1].set_title('Survival Proportion')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()
```

### 1.4 Survival Rate by Categorical Variable

```python
print("="*50)
print("Survival Rate by Categorical Variable")
print("="*50)

categorical_vars = ['sex', 'pclass', 'embarked', 'alone']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, var in zip(axes.flat, categorical_vars):
    survival_by_var = titanic.groupby(var)['survived'].mean().sort_values(ascending=False)
    survival_by_var.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
    ax.set_title(f'Survival Rate by {var}')
    ax.set_ylabel('Survival Rate')
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', rotation=45)

    # Show values
    for i, v in enumerate(survival_by_var):
        ax.text(i, v + 0.02, f'{v:.1%}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Statistical summary
print("\nSurvival rate by sex:")
print(titanic.groupby('sex')['survived'].agg(['mean', 'count']))

print("\nSurvival rate by passenger class:")
print(titanic.groupby('pclass')['survived'].agg(['mean', 'count']))
```

### 1.5 Numeric Variable Analysis

```python
print("="*50)
print("Numeric Variable Analysis")
print("="*50)

numeric_vars = ['age', 'fare']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for i, var in enumerate(numeric_vars):
    # Histogram (by survival status)
    for survived, label, color in [(0, 'Died', 'coral'), (1, 'Survived', 'steelblue')]:
        data = titanic[titanic['survived'] == survived][var].dropna()
        axes[i, 0].hist(data, bins=30, alpha=0.6, label=label, color=color)
    axes[i, 0].set_title(f'{var} Distribution (by Survival)')
    axes[i, 0].set_xlabel(var)
    axes[i, 0].legend()

    # Box plot
    titanic.boxplot(column=var, by='survived', ax=axes[i, 1])
    axes[i, 1].set_title(f'{var} (by Survival)')
    axes[i, 1].set_xlabel('Survived')

plt.suptitle('')
plt.tight_layout()
plt.show()

# Statistical summary
print("\nAge statistics by survival:")
print(titanic.groupby('survived')['age'].describe())
```

### 1.6 Multivariate Analysis

```python
print("="*50)
print("Multivariate Analysis")
print("="*50)

# Survival rate by sex & passenger class
pivot = pd.pivot_table(titanic, values='survived',
                       index='pclass', columns='sex', aggfunc='mean')
print("Survival Rate by Sex & Passenger Class:")
print(pivot)

# Heatmap
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(pivot, annot=True, cmap='RdYlGn', fmt='.1%',
            vmin=0, vmax=1, ax=ax)
ax.set_title('Survival Rate by Sex & Passenger Class')
plt.show()

# Create age groups
titanic['age_group'] = pd.cut(titanic['age'],
                              bins=[0, 12, 18, 35, 60, 100],
                              labels=['Child', 'Teen', 'Young Adult', 'Middle-aged', 'Senior'])

# Survival rate by age group
age_survival = titanic.groupby('age_group')['survived'].mean()
print("\nSurvival Rate by Age Group:")
print(age_survival)
```

### 1.7 Statistical Tests

```python
from scipy import stats

print("="*50)
print("Statistical Tests")
print("="*50)

# Difference in survival rate by sex (chi-square test)
contingency = pd.crosstab(titanic['sex'], titanic['survived'])
chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
print(f"\nSex-Survival Chi-Square Test:")
print(f"χ² = {chi2:.4f}, p-value = {p_value:.4f}")

# Age difference by survival status (t-test)
survived_age = titanic[titanic['survived'] == 1]['age'].dropna()
died_age = titanic[titanic['survived'] == 0]['age'].dropna()

stat, p_value = stats.ttest_ind(survived_age, died_age)
print(f"\nAge-Survival t-Test:")
print(f"t = {stat:.4f}, p-value = {p_value:.4f}")
print(f"Mean age of survivors: {survived_age.mean():.1f}")
print(f"Mean age of non-survivors: {died_age.mean():.1f}")
```

### 1.8 Key Insights

```python
print("="*50)
print("Key Insights")
print("="*50)

insights = """
1. Overall survival rate: ~38%

2. Sex:
   - Female survival rate (74%) is much higher than male (19%)
   - Reflects the "women and children first" principle

3. Passenger class:
   - 1st class (63%) > 2nd class (47%) > 3rd class (24%)
   - Higher class correlates with higher survival rate

4. Age:
   - Children have the highest survival rate
   - Survivors are slightly younger on average than non-survivors

5. Travel companions:
   - Passengers traveling alone have lower survival rates

6. Fare:
   - Passengers who paid higher fares have higher survival rates
   - (Correlated with passenger class)
"""
print(insights)
```

---

## Project 2: Tips Data Analysis

### 2.1 Data Exploration

```python
tips = sns.load_dataset('tips')

print("="*50)
print("Tips Dataset Overview")
print("="*50)
print(tips.info())
print("\nDescriptive statistics:")
print(tips.describe())
```

### 2.2 Tip Amount Analysis

```python
# Calculate tip percentage
tips['tip_pct'] = tips['tip'] / tips['total_bill'] * 100

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Tip amount distribution
axes[0, 0].hist(tips['tip'], bins=30, edgecolor='black', alpha=0.7)
axes[0, 0].axvline(tips['tip'].mean(), color='red', linestyle='--',
                   label=f'Mean: ${tips["tip"].mean():.2f}')
axes[0, 0].set_title('Tip Amount Distribution')
axes[0, 0].set_xlabel('Tip ($)')
axes[0, 0].legend()

# Tip percentage distribution
axes[0, 1].hist(tips['tip_pct'], bins=30, edgecolor='black', alpha=0.7)
axes[0, 1].axvline(tips['tip_pct'].mean(), color='red', linestyle='--',
                   label=f'Mean: {tips["tip_pct"].mean():.1f}%')
axes[0, 1].set_title('Tip Percentage Distribution')
axes[0, 1].set_xlabel('Tip Percentage (%)')
axes[0, 1].legend()

# Average tip by day
tips.groupby('day')['tip'].mean().plot(kind='bar', ax=axes[1, 0],
                                        color='steelblue', edgecolor='black')
axes[1, 0].set_title('Average Tip by Day')
axes[1, 0].set_ylabel('Average Tip ($)')
axes[1, 0].tick_params(axis='x', rotation=45)

# Average tip by time of day
tips.groupby('time')['tip'].mean().plot(kind='bar', ax=axes[1, 1],
                                         color='coral', edgecolor='black')
axes[1, 1].set_title('Average Tip by Time of Day')
axes[1, 1].set_ylabel('Average Tip ($)')
axes[1, 1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.show()

print(f"Average tip: ${tips['tip'].mean():.2f}")
print(f"Average tip percentage: {tips['tip_pct'].mean():.1f}%")
```

### 2.3 Relationship Between Bill and Tip

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter plot
sns.scatterplot(data=tips, x='total_bill', y='tip', hue='time',
                size='size', ax=axes[0])
axes[0].set_title('Total Bill vs Tip')

# Regression line
sns.regplot(data=tips, x='total_bill', y='tip', ax=axes[1],
            scatter_kws={'alpha': 0.5})
axes[1].set_title('Total Bill vs Tip (with Regression Line)')

plt.tight_layout()
plt.show()

# Correlation coefficient
corr, p_value = stats.pearsonr(tips['total_bill'], tips['tip'])
print(f"Correlation coefficient: {corr:.4f} (p-value: {p_value:.4f})")
```

### 2.4 Group Comparison

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# By sex
sns.boxplot(data=tips, x='sex', y='tip_pct', ax=axes[0, 0])
axes[0, 0].set_title('Tip Percentage by Sex')

# By smoking status
sns.boxplot(data=tips, x='smoker', y='tip_pct', ax=axes[0, 1])
axes[0, 1].set_title('Tip Percentage by Smoking Status')

# By day
sns.boxplot(data=tips, x='day', y='tip_pct', ax=axes[1, 0])
axes[1, 0].set_title('Tip Percentage by Day')

# By party size
sns.boxplot(data=tips, x='size', y='tip_pct', ax=axes[1, 1])
axes[1, 1].set_title('Tip Percentage by Party Size')

plt.tight_layout()
plt.show()

# Statistical test: sex difference
male_tip = tips[tips['sex'] == 'Male']['tip_pct']
female_tip = tips[tips['sex'] == 'Female']['tip_pct']
stat, p_value = stats.ttest_ind(male_tip, female_tip)
print(f"\nTip Percentage t-Test by Sex: t={stat:.4f}, p={p_value:.4f}")
```

---

## Project 3: Analysis Report Template

```python
def generate_eda_report(df, target=None):
    """
    Auto-generate an EDA report.

    Parameters:
    -----------
    df : DataFrame
        DataFrame to analyze
    target : str, optional
        Target variable name
    """
    print("="*60)
    print("        Exploratory Data Analysis (EDA) Report")
    print("="*60)

    # 1. Basic information
    print("\n1. Basic Data Information")
    print("-"*40)
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # 2. Data types
    print("\n2. Data Type Summary")
    print("-"*40)
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"   {dtype}: {count} column(s)")

    # 3. Missing values
    print("\n3. Missing Value Overview")
    print("-"*40)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100)
    for col, (cnt, pct) in zip(df.columns, zip(missing, missing_pct)):
        if cnt > 0:
            print(f"   {col}: {cnt} ({pct:.1f}%)")
    if missing.sum() == 0:
        print("   No missing values")

    # 4. Numeric variables
    print("\n4. Numeric Variable Statistics")
    print("-"*40)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        for col in numeric_cols[:5]:  # Top 5 only
            print(f"\n   [{col}]")
            print(f"   Mean: {df[col].mean():.2f}, Median: {df[col].median():.2f}")
            print(f"   Std dev: {df[col].std():.2f}")
            print(f"   Range: [{df[col].min():.2f}, {df[col].max():.2f}]")

    # 5. Categorical variables
    print("\n5. Categorical Variable Summary")
    print("-"*40)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols[:5]:  # Top 5 only
        print(f"\n   [{col}]")
        print(f"   Unique values: {df[col].nunique()}")
        print(f"   Mode: {df[col].mode().values[0]}")

    # 6. Target variable (if provided)
    if target and target in df.columns:
        print(f"\n6. Target Variable ({target}) Analysis")
        print("-"*40)
        if df[target].dtype in ['int64', 'float64']:
            print(f"   Mean: {df[target].mean():.2f}")
            print(f"   Distribution: continuous")
        else:
            print(f"   Class distribution:")
            for val, cnt in df[target].value_counts().items():
                print(f"   - {val}: {cnt} ({cnt/len(df)*100:.1f}%)")

    print("\n" + "="*60)
    print("                     End of Report")
    print("="*60)

# Usage example
# generate_eda_report(titanic, target='survived')
```

---

## Analysis Checklist

```markdown
## Data Analysis Checklist

### Step 1: Data Understanding
- [ ] Verify data source and collection method
- [ ] Understand the meaning of each variable
- [ ] Understand the business context

### Step 2: Data Quality Check
- [ ] Identify missing values and plan how to handle them
- [ ] Detect outliers
- [ ] Verify data types
- [ ] Check for duplicate records

### Step 3: Univariate Analysis
- [ ] Examine distribution of numeric variables
- [ ] Examine frequency of categorical variables
- [ ] Calculate descriptive statistics

### Step 4: Bivariate/Multivariate Analysis
- [ ] Analyze correlations between variables
- [ ] Compare across groups
- [ ] Analyze relationship with the target variable

### Step 5: Statistical Testing
- [ ] Select appropriate test method
- [ ] Verify assumptions
- [ ] Interpret results

### Step 6: Insight Generation
- [ ] Summarize key findings
- [ ] Interpret business implications
- [ ] Suggest further analyses
```

---

## Summary

| Step | Key Tasks | Tools/Functions |
|------|----------|----------|
| Data Loading | Load CSV/Excel/DB | `pd.read_*()` |
| Overview | Check shape and types | `info()`, `describe()` |
| Missing Values | Identify and handle | `isna()`, `fillna()` |
| Univariate | Distribution and frequency analysis | `histplot()`, `countplot()` |
| Bivariate | Relationship analysis | `scatterplot()`, `boxplot()` |
| Multivariate | Pattern discovery | `heatmap()`, `pairplot()` |
| Statistical Testing | Significance testing | `scipy.stats` |
| Insights | Summarize findings | Markdown report |

---

## Exercises

### Exercise 1: Iris Dataset Full EDA

Perform a complete exploratory data analysis on the Iris dataset.

1. Load the dataset:
   ```python
   from sklearn.datasets import load_iris
   import pandas as pd
   iris = pd.DataFrame(load_iris().data, columns=load_iris().feature_names)
   iris['species'] = load_iris().target_names[load_iris().target]
   ```
2. Run `generate_eda_report(iris, target='species')` and interpret each section of the output.
3. Create a pair plot (`sns.pairplot`) with `hue='species'` and `diag_kind='kde'`. Identify: which two features have the cleanest linear separation between species?
4. For the two features you identified, compute the one-way ANOVA F-statistic (`scipy.stats.f_oneway`) across the three species groups. Report the p-value and state your conclusion.
5. Summarize your findings in 3-5 bullet points following the format of Section 1.8 (Key Insights).

### Exercise 2: Missing Value Strategy Comparison

Investigate how different missing value imputation strategies affect downstream statistics.

1. Take the Titanic dataset and artificially introduce 20% missing values into the `fare` column at random.
2. Apply four imputation strategies:
   - Drop rows with missing `fare`
   - Fill with the column mean
   - Fill with the column median
   - Fill with the group median (grouped by `pclass`)
3. For each strategy, compute: (a) mean fare, (b) standard deviation of fare, (c) correlation between `fare` and `survived`.
4. Display the four sets of statistics in a comparison table.
5. Which strategy best preserves the original correlation? Explain why median imputation is generally preferred over mean imputation in the presence of outliers.

### Exercise 3: Multi-Dataset Comparative Analysis

Apply the analysis workflow to a new dataset and compare it with the Titanic findings.

1. Load the `penguins` dataset from Seaborn: `penguins = sns.load_dataset('penguins')`.
2. Follow the full checklist from the lesson (Steps 1-6):
   - Document missing values and drop rows with any missing data.
   - Perform univariate analysis on all four numeric columns.
   - Perform bivariate analysis: create a scatter plot of `bill_length_mm` vs. `flipper_length_mm`, colored by `species`.
   - Perform a chi-square test: is there a significant association between `island` and `species`?
3. Create a 2×2 subplot showing histograms of all four numeric features, color-coded by species.
4. Write a 5-bullet insight section analogous to the Titanic insights in Section 1.8.
5. Compare: how does the analysis workflow change when the target variable is categorical (Titanic `survived`) vs. when there is no designated target (Penguins)?

### Exercise 4: Extend the EDA Report Generator

Add three new analysis sections to the `generate_eda_report()` function.

1. Add **Section 7: Outlier Detection** — for each numeric column, compute the IQR and flag values outside [Q1 − 1.5·IQR, Q3 + 1.5·IQR] as outliers. Print the outlier count and percentage per column.
2. Add **Section 8: Correlation Summary** — compute the full correlation matrix and print the five strongest positive and five strongest negative correlations (skip self-correlations).
3. Add **Section 9: Skewness and Kurtosis** — for each numeric column, print skewness and kurtosis values. Flag columns with |skewness| > 1 as "highly skewed" and suggest log transformation.
4. Test your extended function on both the Titanic and tips datasets.
5. Refactor the function so each section can be toggled on or off via keyword arguments (e.g., `outliers=True, correlation=True, skewness=True`).

### Exercise 5: End-to-End Analysis on a Real Dataset

Apply the full workflow to a dataset of your choosing (or use the `diamonds` dataset from Seaborn).

1. Load the dataset: `diamonds = sns.load_dataset('diamonds')`. It has 53,940 rows and 10 columns including price, carat, cut, color, and clarity.
2. Run the EDA report generator. Note the data types, missing values, and basic statistics.
3. Formulate three hypotheses about what drives diamond price (e.g., "higher carat predicts higher price" or "cut quality is associated with price").
4. For each hypothesis:
   - Choose the appropriate visualization (scatter plot, box plot, or bar plot with error bars).
   - Choose the appropriate statistical test (Pearson correlation, t-test, or ANOVA).
   - Report the test statistic, p-value, and effect size (Cohen's d or η²).
5. Write a concise analysis report (≤ 300 words) summarizing your three findings and their practical implications (e.g., which factors should a buyer prioritize?).

[Previous: Experimental Design](./24_Experimental_Design.md)
