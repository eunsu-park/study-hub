# 10. From EDA to Statistical Inference

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the limitations of descriptive statistics and EDA for drawing general conclusions
2. Distinguish between populations and samples and describe why sampling variability necessitates inference
3. Identify different types of statistical questions: estimation, hypothesis testing, prediction, and association
4. Apply a decision guide to select the appropriate statistical method based on data type and research question
5. Demonstrate the end-to-end workflow from exploratory analysis to confirmatory hypothesis testing
6. Identify common pitfalls including p-hacking, confusing correlation with causation, and ignoring assumptions
7. Distinguish between statistical significance and practical significance using effect sizes

---

You have spent the previous lessons mastering data manipulation, visualization, and descriptive statistics -- powerful tools for exploring what is happening in your dataset. But exploration alone cannot tell you whether an observed pattern is real or merely a product of random chance, nor whether your findings generalize beyond the data at hand. This lesson bridges the gap from exploratory analysis to formal statistical inference, equipping you with the thinking framework needed to draw rigorous, defensible conclusions from data.

---

## 1. Introduction: The Limits of "Just Looking"

In the previous lessons, we've learned powerful tools for **Exploratory Data Analysis (EDA)**:
- Data manipulation with Pandas
- Visualization with Matplotlib and Seaborn
- Descriptive statistics (mean, median, standard deviation)
- Pattern detection and outlier identification

But EDA alone cannot answer critical questions:
- "Is this difference **real** or just random noise?"
- "Can we **generalize** these findings beyond our dataset?"
- "How **confident** are we in our conclusions?"
- "What can we **predict** about future observations?"

This is where **statistical inference** comes in.

### The Detective Analogy

Think of data science as detective work:
- **EDA** = gathering clues, examining the crime scene, forming hypotheses
- **Statistical Inference** = testing those hypotheses rigorously, building a case for court

EDA tells you *what happened in your data*. Inference tells you *what it means for the world beyond your data*.

---

## 2. Population vs Sample: Why We Need Inference

### Key Concepts

- **Population**: The complete set of all individuals/items we want to study
  - Example: *All* customers of an e-commerce platform
  - Example: *All* possible measurements of a physical constant

- **Sample**: A subset of the population we actually observe
  - Example: 10,000 customers from our database
  - Example: 100 measurements in our experiment

- **Sampling Variability**: Different samples from the same population will give different results

### Why Can't We Just Use the Sample Statistics?

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Simulate a population: 1 million customers with average purchase $50, std $15
np.random.seed(42)
population = np.random.normal(loc=50, scale=15, size=1_000_000)
true_mean = population.mean()
print(f"True population mean: ${true_mean:.2f}")

# Take 5 different samples of size 100
sample_means = []
for i in range(5):
    sample = np.random.choice(population, size=100, replace=False)
    sample_mean = sample.mean()
    sample_means.append(sample_mean)
    print(f"Sample {i+1} mean: ${sample_mean:.2f}")

print(f"\nRange of sample means: ${min(sample_means):.2f} to ${max(sample_means):.2f}")
```

**Output:**
```
True population mean: $50.00
Sample 1 mean: $48.93
Sample 2 mean: $51.24
Sample 3 mean: $49.67
Sample 4 mean: $50.82
Sample 5 mean: $49.15

Range of sample means: $48.93 to $51.24
```

**Key insight**: Each sample gives a *different* estimate! Statistical inference helps us:
1. Quantify this uncertainty
2. Make probabilistic statements about the population
3. Test hypotheses with controlled error rates

---

## 3. The Statistical Thinking Shift

### From Descriptive to Inferential

| **Descriptive Statistics (EDA)** | **Inferential Statistics** |
|----------------------------------|----------------------------|
| "The sample mean is 50.2" | "The population mean is likely between 48.5 and 51.9 (95% CI)" |
| "Group A has a higher average" | "Group A's mean is significantly higher (p < 0.01)" |
| "Variables X and Y correlate at 0.7" | "The population correlation is positive (p < 0.001)" |
| "This pattern appears in our data" | "This pattern generalizes beyond our sample (AIC comparison)" |

### The Inference Mindset

When moving from EDA to inference, ask:

1. **What is my population of interest?**
   - Not just "my dataset" but the broader context

2. **How was my sample obtained?**
   - Random sampling? Convenience sample? This affects validity

3. **What assumptions am I making?**
   - Normality? Independence? Homogeneity of variance?

4. **What is my uncertainty?**
   - Confidence intervals, p-values, credible intervals

5. **What are the practical consequences of being wrong?**
   - Type I vs Type II errors, effect sizes

---

## 4. Types of Statistical Questions

### 4.1 Estimation Questions

**Question**: "What is the value of a population parameter?"

**Examples**:
- What is the average customer lifetime value?
- What proportion of users click the button?

**Tools**: Confidence intervals, point estimates

```python
# Example: Estimate mean customer spend with confidence interval
sample = np.random.choice(population, size=100, replace=False)
sample_mean = sample.mean()
sample_se = stats.sem(sample)  # Standard error of the mean
ci_95 = stats.t.interval(0.95, len(sample)-1, loc=sample_mean, scale=sample_se)

print(f"Sample mean: ${sample_mean:.2f}")
print(f"95% Confidence Interval: ${ci_95[0]:.2f} to ${ci_95[1]:.2f}")
print(f"Interpretation: We are 95% confident the true population mean is in this range")
```

### 4.2 Hypothesis Testing Questions

**Question**: "Is there a significant difference/effect?"

**Examples**:
- Does treatment A work better than treatment B?
- Did the website redesign increase conversion rates?

**Tools**: t-tests, chi-square tests, ANOVA, permutation tests

```python
# Example: A/B test - did the new design increase conversion?
# Control group (old design)
control_conversions = np.random.binomial(1, 0.10, size=1000)  # 10% conversion
# Treatment group (new design)
treatment_conversions = np.random.binomial(1, 0.12, size=1000)  # 12% conversion

# Hypothesis test
from statsmodels.stats.proportion import proportions_ztest

count = np.array([treatment_conversions.sum(), control_conversions.sum()])
nobs = np.array([len(treatment_conversions), len(control_conversions)])

z_stat, p_value = proportions_ztest(count, nobs)
print(f"Treatment conversion: {treatment_conversions.mean():.3f}")
print(f"Control conversion: {control_conversions.mean():.3f}")
print(f"p-value: {p_value:.4f}")
print(f"Result: {'Significant' if p_value < 0.05 else 'Not significant'} at α=0.05")
```

### 4.3 Prediction Questions

**Question**: "What will happen for new observations?"

**Examples**:
- What will this customer spend next month?
- How many units will we sell?

**Tools**: Regression, time series models, machine learning

### 4.4 Association Questions

**Question**: "How are variables related?"

**Examples**:
- Does education level correlate with income?
- Are variables independent or dependent?

**Tools**: Correlation, regression, contingency tables

---

## 5. When to Use Which Method: A Decision Guide

### 5.1 Based on Data Type

```
┌─── What type of outcome variable? ───┐
│                                       │
│  Continuous (numeric)                 │
│  ├─ One group → One-sample t-test    │
│  ├─ Two groups → Two-sample t-test   │
│  ├─ 3+ groups → ANOVA                │
│  └─ Predictor variables → Regression │
│                                       │
│  Categorical (binary/count)           │
│  ├─ One proportion → Proportion test │
│  ├─ Two proportions → Chi-square     │
│  └─ Predictor variables → Logistic   │
│                                       │
│  Time series                          │
│  └─ Temporal patterns → ARIMA, etc   │
└───────────────────────────────────────┘
```

### 5.2 Based on Research Question

```python
def suggest_test(data_type, num_groups, paired=False, question_type="difference"):
    """
    Simple decision tree for choosing statistical test

    Parameters:
    -----------
    data_type : str
        'continuous' or 'categorical'
    num_groups : int
        Number of groups to compare
    paired : bool
        Are observations paired/matched?
    question_type : str
        'difference', 'association', 'prediction'
    """

    if question_type == "association":
        if data_type == "continuous":
            return "Pearson/Spearman correlation, Linear regression"
        else:
            return "Chi-square test of independence, Odds ratio"

    if question_type == "prediction":
        return "Regression (linear/logistic), Machine learning"

    # For difference questions
    if data_type == "continuous":
        if num_groups == 1:
            return "One-sample t-test"
        elif num_groups == 2:
            if paired:
                return "Paired t-test"
            else:
                return "Independent two-sample t-test (or Mann-Whitney if not normal)"
        else:
            return "One-way ANOVA (or Kruskal-Wallis if not normal)"
    else:  # categorical
        if num_groups == 1:
            return "One-proportion z-test, Binomial test"
        elif num_groups == 2:
            return "Two-proportion z-test, Chi-square test"
        else:
            return "Chi-square test for multiple groups"

# Examples
print(suggest_test('continuous', 2, paired=False))
# → Independent two-sample t-test (or Mann-Whitney if not normal)

print(suggest_test('categorical', 2, question_type='difference'))
# → Two-proportion z-test, Chi-square test

print(suggest_test('continuous', 1, question_type='association'))
# → Pearson/Spearman correlation, Linear regression
```

---

## 6. Connecting EDA to Inference

### The Workflow: From Exploration to Confirmation

```python
import pandas as pd
import seaborn as sns

# Step 1: EXPLORATORY - Load and visualize data
np.random.seed(42)
data = pd.DataFrame({
    'group': ['A']*50 + ['B']*50,
    'score': np.concatenate([
        np.random.normal(75, 10, 50),  # Group A
        np.random.normal(80, 10, 50)   # Group B
    ])
})

# EDA: Visualize the difference
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.boxplot(data=data, x='group', y='score')
plt.title('EDA: Boxplot suggests Group B scores higher')

plt.subplot(1, 2, 2)
sns.histplot(data=data, x='score', hue='group', kde=True)
plt.title('EDA: Distributions appear roughly normal')

plt.tight_layout()
plt.savefig('/tmp/eda_to_inference.png', dpi=100, bbox_inches='tight')
plt.close()

# EDA: Descriptive statistics
print("=== EXPLORATORY PHASE ===")
print(data.groupby('group')['score'].describe())
print("\nObservation: Group B has a higher mean (80.5 vs 74.9)")
print("Question: Is this difference statistically significant?\n")

# Step 2: CONFIRMATORY - Hypothesis test
print("=== INFERENTIAL PHASE ===")

# Check assumptions
group_a = data[data['group'] == 'A']['score']
group_b = data[data['group'] == 'B']['score']

# Normality test
_, p_norm_a = stats.shapiro(group_a)
_, p_norm_b = stats.shapiro(group_b)
print(f"Normality test (Shapiro-Wilk):")
print(f"  Group A: p={p_norm_a:.3f} → {'Normal' if p_norm_a > 0.05 else 'Not normal'}")
print(f"  Group B: p={p_norm_b:.3f} → {'Normal' if p_norm_b > 0.05 else 'Not normal'}")

# Variance equality test
_, p_var = stats.levene(group_a, group_b)
print(f"\nEqual variance test (Levene):")
print(f"  p={p_var:.3f} → {'Equal variances' if p_var > 0.05 else 'Unequal variances'}")

# Two-sample t-test
t_stat, p_value = stats.ttest_ind(group_a, group_b)
print(f"\nTwo-sample t-test:")
print(f"  t-statistic: {t_stat:.3f}")
print(f"  p-value: {p_value:.4f}")
print(f"  Result: {'Reject H0' if p_value < 0.05 else 'Fail to reject H0'} at α=0.05")

# Effect size (Cohen's d)
pooled_std = np.sqrt((group_a.std()**2 + group_b.std()**2) / 2)
cohens_d = (group_b.mean() - group_a.mean()) / pooled_std
print(f"  Cohen's d: {cohens_d:.3f} ({'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'} effect)")

# Confidence interval for difference
diff_mean = group_b.mean() - group_a.mean()
se_diff = np.sqrt(group_a.var()/len(group_a) + group_b.var()/len(group_b))
ci_diff = stats.t.interval(0.95, len(group_a)+len(group_b)-2, loc=diff_mean, scale=se_diff)
print(f"  95% CI for difference: ({ci_diff[0]:.2f}, {ci_diff[1]:.2f})")

print("\n=== CONCLUSION ===")
print(f"Group B scores are significantly higher than Group A (p={p_value:.4f}).")
print(f"The mean difference is {diff_mean:.2f} points (95% CI: {ci_diff[0]:.2f} to {ci_diff[1]:.2f}).")
print(f"This represents a {('small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large')} effect size.")
```

**Key Takeaway**: EDA guides your inference strategy:
- Histograms → Check normality assumption
- Boxplots → Identify appropriate test (parametric vs non-parametric)
- Scatter plots → Inform regression choices
- Missing data patterns → Handle before inference

---

## 7. Common Pitfalls in Statistical Inference

### 7.1 p-Hacking (Data Dredging)

**Problem**: Testing many hypotheses until you find p < 0.05

```python
# BAD PRACTICE: Testing many variables without correction
np.random.seed(123)
num_tests = 20
p_values = []

for i in range(num_tests):
    # Generate random data (no real effect)
    group1 = np.random.normal(0, 1, 30)
    group2 = np.random.normal(0, 1, 30)
    _, p = stats.ttest_ind(group1, group2)
    p_values.append(p)
    if p < 0.05:
        print(f"Test {i+1}: p={p:.4f} 🎉 Significant!")

print(f"\nFound {sum(p < 0.05 for p in p_values)} 'significant' results out of {num_tests} tests")
print("But all data was random! This is Type I error (false positive)")
```

**Solution**:
- Use multiple testing correction (Bonferroni, Benjamini-Hochberg)
- Pre-register hypotheses
- Report all tests performed

```python
from statsmodels.stats.multitest import multipletests

# Correct for multiple comparisons
rejected, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')
print(f"\nAfter Bonferroni correction: {sum(rejected)} significant results")
```

### 7.2 Confusing Correlation with Causation

**Problem**: "X and Y correlate, therefore X causes Y"

```python
# Spurious correlation example
np.random.seed(42)
years = np.arange(2000, 2020)
ice_cream_sales = 100 + 2*years + np.random.normal(0, 50, len(years)) - 4000
drowning_deaths = 50 + 1*years + np.random.normal(0, 20, len(years)) - 2000

corr, p_corr = stats.pearsonr(ice_cream_sales, drowning_deaths)
print(f"Correlation between ice cream sales and drowning deaths: r={corr:.3f}, p={p_corr:.4f}")
print("Conclusion: Ice cream causes drowning? NO!")
print("Explanation: Both are caused by a confounding variable (summer/temperature)")
```

**Remember**:
- Correlation ≠ Causation
- Need experimental design (randomization, control) for causal claims
- Consider confounding variables, reverse causation, third variables

### 7.3 Ignoring Assumptions

**Problem**: Using tests without checking their assumptions

```python
# Example: t-test on heavily skewed data
np.random.seed(42)
skewed_data1 = np.random.exponential(scale=2, size=30)
skewed_data2 = np.random.exponential(scale=2.5, size=30)

# Wrong: Using t-test without checking normality
t_stat, p_ttest = stats.ttest_ind(skewed_data1, skewed_data2)
print(f"t-test p-value: {p_ttest:.4f}")

# Right: Check assumption first
_, p_norm = stats.shapiro(skewed_data1)
print(f"Shapiro-Wilk test for normality: p={p_norm:.4f}")
if p_norm < 0.05:
    print("Data is not normal! Use Mann-Whitney U test instead")
    u_stat, p_mann = stats.mannwhitneyu(skewed_data1, skewed_data2)
    print(f"Mann-Whitney U test p-value: {p_mann:.4f}")
```

### 7.4 Confusing Statistical and Practical Significance

**Problem**: "p < 0.05, therefore it's important!"

```python
# Large sample can make tiny effects "significant"
np.random.seed(42)
large_group1 = np.random.normal(100, 15, 10000)
large_group2 = np.random.normal(100.5, 15, 10000)  # Tiny difference

t_stat, p_value = stats.ttest_ind(large_group1, large_group2)
cohens_d = (large_group2.mean() - large_group1.mean()) / large_group1.std()

print(f"Mean difference: {large_group2.mean() - large_group1.mean():.3f}")
print(f"p-value: {p_value:.4f} → Statistically significant!")
print(f"Cohen's d: {cohens_d:.3f} → Practically negligible (tiny effect)")
print("\nAlways report effect sizes, not just p-values!")
```

---

## 8. Practical Example: From EDA to Full Inference

### Scenario: E-commerce A/B Test

We want to know if a new checkout flow increases purchase amounts.

```python
# Generate realistic data
np.random.seed(42)
n = 200

data_ab = pd.DataFrame({
    'user_id': range(n),
    'variant': ['control']*100 + ['treatment']*100,
    'purchase_amount': np.concatenate([
        np.random.gamma(shape=2, scale=25, size=100),  # Control
        np.random.gamma(shape=2.3, scale=25, size=100)  # Treatment (slightly higher)
    ])
})

# Add some confounding variable: user tenure
data_ab['tenure_months'] = np.random.poisson(lam=12, size=n)

print("=== STAGE 1: EXPLORATORY DATA ANALYSIS ===\n")

# 1. Summary statistics
print(data_ab.groupby('variant')['purchase_amount'].describe())

# 2. Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Histogram
data_ab.hist(column='purchase_amount', by='variant', bins=20, ax=axes[0:2], alpha=0.7)
axes[0].set_title('Control Group')
axes[1].set_title('Treatment Group')

# Boxplot
axes[2].boxplot([
    data_ab[data_ab['variant']=='control']['purchase_amount'],
    data_ab[data_ab['variant']=='treatment']['purchase_amount']
], labels=['Control', 'Treatment'])
axes[2].set_ylabel('Purchase Amount')
axes[2].set_title('Distribution Comparison')

plt.tight_layout()
plt.savefig('/tmp/ab_test_eda.png', dpi=100, bbox_inches='tight')
plt.close()

print("\n=== STAGE 2: FORMULATE STATISTICAL QUESTION ===\n")
print("Research Question: Does the new checkout flow (treatment) increase purchase amounts?")
print("Null Hypothesis (H0): μ_treatment = μ_control")
print("Alternative Hypothesis (H1): μ_treatment > μ_control (one-tailed)")
print("Significance level: α = 0.05")

print("\n=== STAGE 3: CHECK ASSUMPTIONS ===\n")

control = data_ab[data_ab['variant']=='control']['purchase_amount']
treatment = data_ab[data_ab['variant']=='treatment']['purchase_amount']

# Normality (note: with n=100, CLT applies, but let's check)
_, p_norm_c = stats.shapiro(control)
_, p_norm_t = stats.shapiro(treatment)
print(f"Normality (Shapiro-Wilk):")
print(f"  Control: p={p_norm_c:.4f}")
print(f"  Treatment: p={p_norm_t:.4f}")
print(f"  → Data is {'normal' if min(p_norm_c, p_norm_t) > 0.05 else 'not normal, but n=100 so CLT applies'}")

# Equal variance
_, p_var = stats.levene(control, treatment)
print(f"\nEqual variance (Levene): p={p_var:.4f}")
print(f"  → Variances are {'equal' if p_var > 0.05 else 'unequal'}")

print("\n=== STAGE 4: CONDUCT INFERENCE ===\n")

# Two-sample t-test (one-tailed)
t_stat, p_twotail = stats.ttest_ind(treatment, control, equal_var=(p_var>0.05))
p_onetail = p_twotail / 2 if t_stat > 0 else 1 - p_twotail / 2

print(f"Two-sample t-test (one-tailed):")
print(f"  t-statistic: {t_stat:.3f}")
print(f"  p-value: {p_onetail:.4f}")
print(f"  Decision: {'Reject H0' if p_onetail < 0.05 else 'Fail to reject H0'}")

# Effect size
cohens_d = (treatment.mean() - control.mean()) / np.sqrt((control.var() + treatment.var())/2)
print(f"\nEffect size (Cohen's d): {cohens_d:.3f}")

# Confidence interval
diff_mean = treatment.mean() - control.mean()
se_diff = np.sqrt(control.var()/len(control) + treatment.var()/len(treatment))
ci_95 = stats.t.interval(0.95, len(control)+len(treatment)-2, loc=diff_mean, scale=se_diff)
print(f"95% CI for difference: (${ci_95[0]:.2f}, ${ci_95[1]:.2f})")

# Practical significance
revenue_increase = (diff_mean / control.mean()) * 100
print(f"\nRevenue increase: {revenue_increase:.1f}%")

print("\n=== STAGE 5: REPORT RESULTS ===\n")
print(f"The treatment group had significantly higher purchase amounts than the control group")
print(f"(M_treatment = ${treatment.mean():.2f}, M_control = ${control.mean():.2f}, t({len(control)+len(treatment)-2}) = {t_stat:.2f}, p = {p_onetail:.4f}).")
print(f"The mean difference was ${diff_mean:.2f} (95% CI: ${ci_95[0]:.2f} to ${ci_95[1]:.2f}),")
print(f"representing a {revenue_increase:.1f}% increase in average purchase amount.")
print(f"The effect size was {('small' if abs(cohens_d)<0.5 else 'medium' if abs(cohens_d)<0.8 else 'large')} (Cohen's d = {cohens_d:.2f}).")
```

---

## 9. Exercises

### Exercise 1: Choose the Right Test
For each scenario, identify the appropriate statistical test:

a) A company wants to know if customer satisfaction scores (1-10) differ between three service centers.

b) A researcher wants to test if the proportion of left-handed people differs between men and women.

c) A data scientist wants to predict house prices based on square footage, number of bedrooms, and location.

d) An analyst wants to know if there's a relationship between hours studied and exam scores.

**Answers**:
- a) One-way ANOVA (continuous outcome, 3 groups)
- b) Two-proportion z-test or Chi-square test (categorical outcome, 2 groups)
- c) Multiple linear regression (continuous outcome, multiple predictors)
- d) Pearson correlation / Simple linear regression (two continuous variables)

### Exercise 2: From EDA to Hypothesis

You perform EDA on employee data and notice:
- The median salary for Department A is $75,000
- The median salary for Department B is $82,000
- Boxplots show some overlap but Department B appears higher

**Questions**:
1. What is the population of interest?
2. Formulate a null and alternative hypothesis
3. What test would you use? What assumptions would you check?
4. If p = 0.03, what would you conclude?
5. What additional information would help assess practical significance?

### Exercise 3: Identify the Pitfall

Identify the statistical pitfall in each scenario:

a) A researcher tests 50 different variables and reports only the 3 that had p < 0.05.

b) A study finds that coffee consumption correlates with heart disease and concludes coffee causes heart disease.

c) A company reports "statistically significant improvement" (p=0.04) but the actual increase in conversion rate was 0.1%.

d) An analyst uses a t-test on heavily right-skewed income data without checking assumptions.

**Answers**:
- a) p-hacking / multiple comparisons problem
- b) Confusing correlation with causation
- c) Confusing statistical and practical significance
- d) Violating test assumptions

### Exercise 4: Complete Analysis Pipeline

Using the tips dataset (available in seaborn), perform a complete analysis:

```python
import seaborn as sns
tips = sns.load_dataset('tips')

# Your task:
# 1. EDA: Explore if smokers tip differently than non-smokers
# 2. Formulate hypothesis
# 3. Check assumptions
# 4. Conduct appropriate test
# 5. Report results with effect size and confidence interval
```

---

## 10. Summary

### Key Takeaways

1. **EDA is exploration; Inference is confirmation**
   - EDA generates hypotheses; inference tests them rigorously

2. **Samples are imperfect windows into populations**
   - Sampling variability means we need probabilistic statements
   - Confidence intervals and p-values quantify uncertainty

3. **Different questions require different methods**
   - Estimation → Confidence intervals
   - Testing → Hypothesis tests (t-test, ANOVA, chi-square, etc.)
   - Prediction → Regression, ML models
   - Association → Correlation, contingency tables

4. **Always check assumptions**
   - Normality, independence, equal variance
   - Use robust alternatives when assumptions fail

5. **Report effect sizes, not just p-values**
   - Statistical significance ≠ Practical significance
   - Context matters for interpretation

6. **Beware common pitfalls**
   - Multiple testing without correction
   - Correlation ≠ Causation
   - Violating assumptions
   - Overinterpreting p-values

### The Bridge You've Crossed

You now understand:
- ✅ Why we can't just "trust the data" without inference
- ✅ How to move from descriptive patterns to formal statistical questions
- ✅ When to use which statistical method
- ✅ How to connect EDA findings to rigorous tests
- ✅ Common mistakes to avoid

### What's Next?

The remaining lessons will dive deeper into:
- **L11-L13**: Probability foundations and distributions
- **L14-L16**: Hypothesis testing frameworks and power analysis
- **L17-L18**: Regression and model evaluation
- **L19-L21**: Bayesian inference
- **L22-L24**: Time series and advanced topics

You're now ready to move beyond "what the data shows" to "what we can conclude with confidence."

---

## 11. Additional Resources

### Books
- **"The Art of Statistics"** by David Spiegelhalter - accessible introduction to statistical thinking
- **"Statistical Rethinking"** by Richard McElreath - Bayesian approach with intuitive examples
- **"Naked Statistics"** by Charles Wheelan - conceptual understanding without heavy math

### Online Resources
- [Seeing Theory](https://seeing-theory.brown.edu/) - Visual introduction to probability and statistics
- [StatQuest](https://statquest.org/) - Video explanations of statistical concepts
- [Cross Validated](https://stats.stackexchange.com/) - Q&A for statistics

### Python Libraries
- **scipy.stats**: Statistical tests and distributions
- **statsmodels**: Regression, hypothesis testing, time series
- **pingouin**: User-friendly statistical tests with effect sizes
- **scikit-learn**: Machine learning and predictive modeling

---

## Navigation
- **Previous**: [09_Data_Visualization_Advanced](./09_Data_Visualization_Advanced.md)
- **Next**: [11_Probability_Review](./11_Probability_Review.md)
- **Overview**: [00_Overview](./00_Overview.md)
