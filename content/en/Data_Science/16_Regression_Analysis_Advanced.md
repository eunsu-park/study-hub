# 16. Advanced Regression Analysis

[Previous: ANOVA](./15_ANOVA.md) | [Next: Generalized Linear Models](./17_Generalized_Linear_Models.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Fit and interpret multiple regression models using `statsmodels`, including standardized coefficients
2. Evaluate all four regression assumptions (linearity, independence, homoscedasticity, normality) through diagnostic plots and formal tests
3. Identify influential observations using leverage, Cook's distance, and DFFITS
4. Diagnose multicollinearity with VIF and apply remedies (variable removal, PCA, Ridge regression)
5. Implement forward selection, backward elimination, and stepwise variable selection procedures
6. Compare candidate models using adjusted R-squared, AIC, BIC, and cross-validation
7. Apply partial regression and CCPR plots for visual assessment of individual predictor contributions

---

Simple regression tells you whether one variable predicts another; multiple regression reveals how several predictors work together -- and where they can mislead you. Collinear predictors can flip coefficient signs, violated assumptions can invalidate your p-values, and including irrelevant variables inflates variance. This lesson teaches you to diagnose these pitfalls and systematically build reliable regression models.

---

## 1. Multiple Regression Basics

### 1.1 Model and Estimation

```python
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Generate example data
np.random.seed(42)
n = 200

# Independent variables
X1 = np.random.normal(50, 10, n)  # Experience (months)
X2 = np.random.normal(70, 15, n)  # Education score
X3 = np.random.normal(30, 5, n)   # Age

# Dependent variable (salary): linear relationship + noise
Y = 30000 + 500*X1 + 200*X2 + 100*X3 + np.random.normal(0, 5000, n)

# Create DataFrame
df = pd.DataFrame({
    'salary': Y,
    'experience': X1,
    'education': X2,
    'age': X3
})

# Descriptive statistics
print("Data descriptive statistics:")
print(df.describe())

# Correlation matrix
print("\nCorrelation matrix:")
print(df.corr().round(3))
```

### 1.2 statsmodels OLS

```python
# OLS regression

# Method 1: formula API (R style)
model_formula = smf.ols('salary ~ experience + education + age', data=df).fit()

# Method 2: matrix API
X = df[['experience', 'education', 'age']]
X = sm.add_constant(X)  # Add intercept
y = df['salary']
model_matrix = sm.OLS(y, X).fit()

# Print results
print("Regression results:")
print(model_formula.summary())

# Extract key statistics
print("\nKey statistics:")
print(f"R-squared: {model_formula.rsquared:.4f}")
print(f"Adjusted R-squared: {model_formula.rsquared_adj:.4f}")
print(f"F-statistic: {model_formula.fvalue:.2f}")
print(f"F p-value: {model_formula.f_pvalue:.4e}")
print(f"AIC: {model_formula.aic:.2f}")
print(f"BIC: {model_formula.bic:.2f}")
```

### 1.3 Interpreting Coefficients

```python
# Detailed coefficient analysis
print("Coefficient analysis:")
print("-" * 70)
print(f"{'Variable':<15} {'Coef':<12} {'Std Err':<12} {'t-value':<10} {'p-value':<12} {'95% CI'}")
print("-" * 70)

coef_table = pd.DataFrame({
    'coef': model_formula.params,
    'std_err': model_formula.bse,
    't_value': model_formula.tvalues,
    'p_value': model_formula.pvalues,
    'ci_lower': model_formula.conf_int()[0],
    'ci_upper': model_formula.conf_int()[1]
})

for idx, row in coef_table.iterrows():
    ci_str = f"[{row['ci_lower']:.1f}, {row['ci_upper']:.1f}]"
    sig = "***" if row['p_value'] < 0.001 else ("**" if row['p_value'] < 0.01 else ("*" if row['p_value'] < 0.05 else ""))
    print(f"{idx:<15} {row['coef']:<12.2f} {row['std_err']:<12.2f} {row['t_value']:<10.2f} {row['p_value']:<10.4f}{sig:<2} {ci_str}")

# Standardized coefficients (Beta)
print("\nStandardized coefficients (Beta):")
df_standardized = (df - df.mean()) / df.std()
model_std = smf.ols('salary ~ experience + education + age', data=df_standardized).fit()

for var in ['experience', 'education', 'age']:
    print(f"  {var}: {model_std.params[var]:.4f}")
```

---

## 2. Checking Regression Assumptions

### 2.1 Assumption Overview

```python
# Regression assumptions:
# 1. Linearity: Linear relationship between Y and X
# 2. Independence: Residuals are independent
# 3. Homoscedasticity: Constant variance of residuals
# 4. Normality: Residuals follow normal distribution

# Values for diagnostics
fitted = model_formula.fittedvalues
residuals = model_formula.resid
standardized_residuals = model_formula.get_influence().resid_studentized_internal
```

### 2.2 Residual Analysis

```python
# Comprehensive diagnostic plots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Residuals vs Fitted (linearity, homoscedasticity)
axes[0, 0].scatter(fitted, residuals, alpha=0.5)
axes[0, 0].axhline(0, color='red', linestyle='--')
axes[0, 0].set_xlabel('Fitted values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Residuals vs Fitted (linearity, homoscedasticity)')

# Add Lowess curve
from statsmodels.nonparametric.smoothers_lowess import lowess
z = lowess(residuals, fitted, frac=0.3)
axes[0, 0].plot(z[:, 0], z[:, 1], 'g-', linewidth=2, label='Lowess')
axes[0, 0].legend()

# 2. Q-Q Plot (normality)
stats.probplot(residuals, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Q-Q Plot (normality)')

# 3. Scale-Location Plot (homoscedasticity)
sqrt_std_residuals = np.sqrt(np.abs(standardized_residuals))
axes[1, 0].scatter(fitted, sqrt_std_residuals, alpha=0.5)
axes[1, 0].set_xlabel('Fitted values')
axes[1, 0].set_ylabel('√|Standardized residuals|')
axes[1, 0].set_title('Scale-Location (homoscedasticity)')

z = lowess(sqrt_std_residuals, fitted, frac=0.3)
axes[1, 0].plot(z[:, 0], z[:, 1], 'g-', linewidth=2)

# 4. Residual histogram (normality)
axes[1, 1].hist(residuals, bins=30, density=True, alpha=0.7, edgecolor='black')
xmin, xmax = axes[1, 1].get_xlim()
x = np.linspace(xmin, xmax, 100)
axes[1, 1].plot(x, stats.norm.pdf(x, residuals.mean(), residuals.std()),
                'r-', linewidth=2, label='Normal distribution')
axes[1, 1].set_xlabel('Residuals')
axes[1, 1].set_ylabel('Density')
axes[1, 1].set_title('Residual histogram (normality)')
axes[1, 1].legend()

plt.tight_layout()
plt.show()
```

### 2.3 Normality Tests

```python
# Residual normality tests

# Shapiro-Wilk test
stat_shapiro, p_shapiro = stats.shapiro(residuals)
print(f"Shapiro-Wilk test: W = {stat_shapiro:.4f}, p = {p_shapiro:.4f}")

# Kolmogorov-Smirnov test
stat_ks, p_ks = stats.kstest(residuals, 'norm', args=(residuals.mean(), residuals.std()))
print(f"Kolmogorov-Smirnov test: D = {stat_ks:.4f}, p = {p_ks:.4f}")

# Jarque-Bera test
from scipy.stats import jarque_bera
stat_jb, p_jb = jarque_bera(residuals)
print(f"Jarque-Bera test: JB = {stat_jb:.4f}, p = {p_jb:.4f}")

# Skewness and kurtosis
print(f"\nSkewness: {stats.skew(residuals):.4f} (close to 0 is symmetric)")
print(f"Kurtosis: {stats.kurtosis(residuals):.4f} (close to 0 is normal)")
```

### 2.4 Homoscedasticity Tests

```python
# Breusch-Pagan test
from statsmodels.stats.diagnostic import het_breuschpagan

bp_stat, bp_pvalue, bp_fstat, bp_f_pvalue = het_breuschpagan(residuals, X)
print(f"Breusch-Pagan test:")
print(f"  LM statistic: {bp_stat:.4f}")
print(f"  LM p-value: {bp_pvalue:.4f}")
print(f"  F statistic: {bp_fstat:.4f}")
print(f"  F p-value: {bp_f_pvalue:.4f}")

# White test
from statsmodels.stats.diagnostic import het_white

white_stat, white_pvalue, white_fstat, white_f_pvalue = het_white(residuals, X)
print(f"\nWhite test:")
print(f"  LM statistic: {white_stat:.4f}")
print(f"  LM p-value: {white_pvalue:.4f}")

# Goldfeld-Quandt test
from statsmodels.stats.diagnostic import het_goldfeldquandt

gq_stat, gq_pvalue, gq_alt = het_goldfeldquandt(y, X, alternative='two-sided')
print(f"\nGoldfeld-Quandt test:")
print(f"  F statistic: {gq_stat:.4f}")
print(f"  p-value: {gq_pvalue:.4f}")
```

### 2.5 Independence Test (Autocorrelation)

```python
# Durbin-Watson test
from statsmodels.stats.stattools import durbin_watson

dw_stat = durbin_watson(residuals)
print(f"Durbin-Watson test: DW = {dw_stat:.4f}")
print("  DW ≈ 2: no autocorrelation")
print("  DW < 2: positive autocorrelation")
print("  DW > 2: negative autocorrelation")

# Ljung-Box test
from statsmodels.stats.diagnostic import acorr_ljungbox

lb_result = acorr_ljungbox(residuals, lags=[1, 5, 10], return_df=True)
print(f"\nLjung-Box test:")
print(lb_result)
```

---

## 3. Advanced Diagnostic Plots

### 3.1 Leverage and Influence

```python
# Influence analysis
influence = model_formula.get_influence()

# Leverage (Hat values)
leverage = influence.hat_matrix_diag

# Cook's Distance
cooks_d = influence.cooks_distance[0]

# DFFITS
dffits = influence.dffits[0]

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Residuals vs Leverage
axes[0, 0].scatter(leverage, standardized_residuals, alpha=0.5)
axes[0, 0].axhline(0, color='red', linestyle='--')
axes[0, 0].set_xlabel('Leverage')
axes[0, 0].set_ylabel('Standardized residuals')
axes[0, 0].set_title('Residuals vs Leverage')

# Cook's D contours
n = len(y)
p = len(model_formula.params)
leverage_range = np.linspace(0.001, max(leverage), 100)
for cook_threshold in [0.5, 1]:
    cook_line = np.sqrt(cook_threshold * p * (1 - leverage_range) / leverage_range)
    axes[0, 0].plot(leverage_range, cook_line, 'g--', alpha=0.5)
    axes[0, 0].plot(leverage_range, -cook_line, 'g--', alpha=0.5)

# 2. Cook's Distance
axes[0, 1].stem(range(len(cooks_d)), cooks_d, markerfmt='o', basefmt=' ')
axes[0, 1].axhline(4/n, color='red', linestyle='--', label=f"4/n = {4/n:.4f}")
axes[0, 1].set_xlabel('Observation index')
axes[0, 1].set_ylabel("Cook's Distance")
axes[0, 1].set_title("Cook's Distance")
axes[0, 1].legend()

# Influential observations
influential = np.where(cooks_d > 4/n)[0]
print(f"Influential observations (Cook's D > 4/n): {influential}")

# 3. Leverage distribution
axes[1, 0].hist(leverage, bins=30, density=True, alpha=0.7, edgecolor='black')
axes[1, 0].axvline(2*p/n, color='red', linestyle='--', label=f'2p/n = {2*p/n:.4f}')
axes[1, 0].set_xlabel('Leverage')
axes[1, 0].set_ylabel('Density')
axes[1, 0].set_title('Leverage distribution')
axes[1, 0].legend()

high_leverage = np.where(leverage > 2*p/n)[0]
print(f"High leverage observations: {len(high_leverage)} cases")

# 4. DFFITS
axes[1, 1].stem(range(len(dffits)), dffits, markerfmt='o', basefmt=' ')
threshold = 2 * np.sqrt(p/n)
axes[1, 1].axhline(threshold, color='red', linestyle='--')
axes[1, 1].axhline(-threshold, color='red', linestyle='--', label=f'±2√(p/n) = ±{threshold:.4f}')
axes[1, 1].set_xlabel('Observation index')
axes[1, 1].set_ylabel('DFFITS')
axes[1, 1].set_title('DFFITS')
axes[1, 1].legend()

plt.tight_layout()
plt.show()
```

### 3.2 Partial Regression Plots

```python
# Partial Regression Plots (Added Variable Plots)
fig = sm.graphics.plot_partregress_grid(model_formula)
fig.suptitle('Partial Regression Plots', y=1.02)
plt.tight_layout()
plt.show()

# Individual partial regression plots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, var in enumerate(['experience', 'education', 'age']):
    sm.graphics.plot_partregress(
        'salary', var, ['experience', 'education', 'age'],
        data=df, obs_labels=False, ax=axes[i]
    )
    axes[i].set_title(f'salary ~ {var} | other variables')

plt.tight_layout()
plt.show()
```

### 3.3 Component-Component plus Residual (CCPR) Plots

```python
# Component-Component plus Residual Plot
fig = sm.graphics.plot_ccpr_grid(model_formula)
fig.suptitle('Component-Component plus Residual Plots (CCPR)', y=1.02)
plt.tight_layout()
plt.show()
```

---

## 4. Multicollinearity

### 4.1 Diagnosing Multicollinearity

```python
# Generate data with multicollinearity
np.random.seed(42)
n = 200

X1 = np.random.normal(50, 10, n)
X2 = X1 + np.random.normal(0, 3, n)  # High correlation with X1
X3 = np.random.normal(30, 5, n)
Y = 100 + 10*X1 + 5*X2 + 20*X3 + np.random.normal(0, 50, n)

df_multi = pd.DataFrame({
    'Y': Y, 'X1': X1, 'X2': X2, 'X3': X3
})

# Correlation matrix
print("Correlation matrix:")
print(df_multi.corr().round(3))

# Regression
model_multi = smf.ols('Y ~ X1 + X2 + X3', data=df_multi).fit()
print("\nModel with multicollinearity:")
print(model_multi.summary().tables[1])
```

### 4.2 VIF (Variance Inflation Factor)

```python
# Calculate VIF
def calculate_vif(df, features):
    """Calculate VIF"""
    X = df[features]
    X = sm.add_constant(X)

    vif_data = pd.DataFrame()
    vif_data['feature'] = features
    vif_data['VIF'] = [variance_inflation_factor(X.values, i+1)
                       for i in range(len(features))]
    return vif_data

features = ['X1', 'X2', 'X3']
vif_result = calculate_vif(df_multi, features)

print("VIF (Variance Inflation Factor):")
print(vif_result)
print("\nInterpretation:")
print("  VIF = 1: no multicollinearity")
print("  VIF 1-5: low multicollinearity")
print("  VIF 5-10: moderate multicollinearity")
print("  VIF > 10: high multicollinearity (problem)")

# VIF of original data
features_orig = ['experience', 'education', 'age']
vif_orig = calculate_vif(df, features_orig)
print("\nOriginal data VIF:")
print(vif_orig)
```

### 4.3 Addressing Multicollinearity

```python
# Solution 1: Remove variable
model_reduced = smf.ols('Y ~ X1 + X3', data=df_multi).fit()  # Remove X2

print("Model after variable removal (X2 removed):")
print(model_reduced.summary().tables[1])

# Solution 2: Principal Component Regression (PCR)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_multi[['X1', 'X2', 'X3']])

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\nPCA results:")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Cumulative explained variance: {pca.explained_variance_ratio_.cumsum()}")

# PCR model
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['Y'] = df_multi['Y']

model_pcr = smf.ols('Y ~ PC1 + PC2', data=df_pca).fit()
print("\nPCR model:")
print(model_pcr.summary().tables[1])

# Solution 3: Ridge regression
from sklearn.linear_model import Ridge

X = df_multi[['X1', 'X2', 'X3']]
y = df_multi['Y']

ridge = Ridge(alpha=1.0)
ridge.fit(X, y)

print("\nRidge regression coefficients:")
for name, coef in zip(['X1', 'X2', 'X3'], ridge.coef_):
    print(f"  {name}: {coef:.4f}")
```

---

## 5. Variable Selection

### 5.1 Forward Selection, Backward Elimination, Stepwise Selection

```python
# Generate data with more variables
np.random.seed(42)
n = 200

# Relevant variables
X1 = np.random.normal(50, 10, n)  # Important
X2 = np.random.normal(30, 5, n)   # Important
X3 = np.random.normal(100, 20, n) # Important

# Irrelevant variables
X4 = np.random.normal(0, 1, n)
X5 = np.random.normal(0, 1, n)
X6 = np.random.normal(0, 1, n)

Y = 10 + 5*X1 + 3*X2 + 2*X3 + np.random.normal(0, 30, n)

df_select = pd.DataFrame({
    'Y': Y, 'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'X5': X5, 'X6': X6
})

# Full model
model_full = smf.ols('Y ~ X1 + X2 + X3 + X4 + X5 + X6', data=df_select).fit()
print("Full model:")
print(model_full.summary().tables[1])
```

```python
def forward_selection(data, response, features, significance_level=0.05):
    """Forward selection"""
    selected = []
    remaining = features.copy()

    while remaining:
        best_pvalue = 1
        best_feature = None

        for feature in remaining:
            formula = f'{response} ~ ' + ' + '.join(selected + [feature])
            model = smf.ols(formula, data=data).fit()
            pvalue = model.pvalues[feature]

            if pvalue < best_pvalue:
                best_pvalue = pvalue
                best_feature = feature

        if best_pvalue < significance_level:
            selected.append(best_feature)
            remaining.remove(best_feature)
            print(f"Added: {best_feature} (p = {best_pvalue:.4f})")
        else:
            break

    return selected

def backward_elimination(data, response, features, significance_level=0.05):
    """Backward elimination"""
    selected = features.copy()

    while selected:
        formula = f'{response} ~ ' + ' + '.join(selected)
        model = smf.ols(formula, data=data).fit()

        # Find least significant variable
        max_pvalue = 0
        worst_feature = None

        for feature in selected:
            pvalue = model.pvalues[feature]
            if pvalue > max_pvalue:
                max_pvalue = pvalue
                worst_feature = feature

        if max_pvalue > significance_level:
            selected.remove(worst_feature)
            print(f"Removed: {worst_feature} (p = {max_pvalue:.4f})")
        else:
            break

    return selected

# Forward selection
print("Forward selection:")
print("-" * 40)
features = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6']
selected_forward = forward_selection(df_select, 'Y', features)
print(f"Selected variables: {selected_forward}")

# Backward elimination
print("\nBackward elimination:")
print("-" * 40)
selected_backward = backward_elimination(df_select, 'Y', features)
print(f"Selected variables: {selected_backward}")
```

### 5.2 Information Criteria (AIC, BIC)

```python
# Compare all possible models (for small variable sets)
from itertools import combinations

def all_possible_models(data, response, features):
    """Compare all possible models"""
    results = []

    for k in range(1, len(features) + 1):
        for subset in combinations(features, k):
            formula = f'{response} ~ ' + ' + '.join(subset)
            model = smf.ols(formula, data=data).fit()

            results.append({
                'features': subset,
                'n_features': k,
                'R2': model.rsquared,
                'R2_adj': model.rsquared_adj,
                'AIC': model.aic,
                'BIC': model.bic
            })

    return pd.DataFrame(results)

all_models = all_possible_models(df_select, 'Y', features)

# Best model by each criterion
print("Best model by each criterion:")
print("-" * 60)

best_r2_adj = all_models.loc[all_models['R2_adj'].idxmax()]
best_aic = all_models.loc[all_models['AIC'].idxmin()]
best_bic = all_models.loc[all_models['BIC'].idxmin()]

print(f"Best Adjusted R²: {best_r2_adj['features']}")
print(f"  R² = {best_r2_adj['R2']:.4f}, Adj R² = {best_r2_adj['R2_adj']:.4f}")

print(f"\nMinimum AIC: {best_aic['features']}")
print(f"  AIC = {best_aic['AIC']:.2f}")

print(f"\nMinimum BIC: {best_bic['features']}")
print(f"  BIC = {best_bic['BIC']:.2f}")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Best model by number of variables
for metric, ax, title in zip(['R2_adj', 'AIC', 'BIC'], axes, ['Adjusted R²', 'AIC', 'BIC']):
    for k in range(1, 7):
        subset_models = all_models[all_models['n_features'] == k]
        if metric == 'R2_adj':
            best_val = subset_models[metric].max()
        else:
            best_val = subset_models[metric].min()
        ax.scatter([k], [best_val], s=100)

    ax.set_xlabel('Number of variables')
    ax.set_ylabel(title)
    ax.set_title(f'Number of variables vs {title}')
    ax.set_xticks(range(1, 7))
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

### 5.3 Variable Selection Using Cross-Validation

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# Find optimal model using cross-validation
X_all = df_select[features]
y = df_select['Y']

print("Cross-validation results (5-fold):")
print("-" * 50)

cv_results = []
for k in range(1, len(features) + 1):
    best_cv_score = -np.inf
    best_subset = None

    for subset in combinations(features, k):
        X_subset = df_select[list(subset)]
        model = LinearRegression()
        scores = cross_val_score(model, X_subset, y, cv=5, scoring='r2')
        mean_score = scores.mean()

        if mean_score > best_cv_score:
            best_cv_score = mean_score
            best_subset = subset

    cv_results.append({
        'n_features': k,
        'best_features': best_subset,
        'cv_r2': best_cv_score
    })
    print(f"k={k}: {best_subset}, CV R² = {best_cv_score:.4f}")

# Best model
best_cv = max(cv_results, key=lambda x: x['cv_r2'])
print(f"\nBest model: {best_cv['best_features']}")
print(f"CV R² = {best_cv['cv_r2']:.4f}")
```

---

## 6. Practical Example: House Price Prediction

### 6.1 Data Preparation

```python
# Simulate house price data
np.random.seed(42)
n = 500

# Generate features
size = np.random.normal(1500, 400, n)  # Square feet
bedrooms = np.random.poisson(3, n) + 1
bathrooms = bedrooms * 0.5 + np.random.normal(0, 0.5, n)
bathrooms = np.clip(bathrooms, 1, 5)
age = np.random.exponential(20, n)
distance = np.random.uniform(1, 30, n)  # Distance to downtown

# Price (including nonlinear relationships)
price = (100000 + 100*size + 20000*bedrooms + 15000*bathrooms
         - 2000*age - 1000*distance
         + 0.02*size*bedrooms  # Interaction
         + np.random.normal(0, 30000, n))
price = np.maximum(price, 50000)  # Minimum price

df_house = pd.DataFrame({
    'price': price,
    'size': size,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'age': age,
    'distance': distance
})

print("House data descriptive statistics:")
print(df_house.describe())
```

### 6.2 Exploratory Analysis

```python
# Correlation analysis
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for i, var in enumerate(['size', 'bedrooms', 'bathrooms', 'age', 'distance']):
    ax = axes[i // 3, i % 3]
    ax.scatter(df_house[var], df_house['price'], alpha=0.3)
    ax.set_xlabel(var)
    ax.set_ylabel('price')

    # Regression line
    z = np.polyfit(df_house[var], df_house['price'], 1)
    p = np.poly1d(z)
    ax.plot(sorted(df_house[var]), p(sorted(df_house[var])), 'r-')

    # Correlation coefficient
    corr = df_house['price'].corr(df_house[var])
    ax.set_title(f'r = {corr:.3f}')

# Correlation matrix heatmap
axes[1, 2].set_visible(False)
fig.delaxes(axes[1, 2])

plt.tight_layout()
plt.show()

# Correlation matrix heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df_house.corr(), annot=True, cmap='RdBu_r', center=0,
            fmt='.2f', square=True)
plt.title('Correlation matrix')
plt.tight_layout()
plt.show()
```

### 6.3 Model Building and Diagnostics

```python
# Build model
model_house = smf.ols('price ~ size + bedrooms + bathrooms + age + distance',
                       data=df_house).fit()

print("House price regression model:")
print(model_house.summary())

# Diagnostics
print("\nModel diagnostics:")
print("-" * 40)

# VIF
features_house = ['size', 'bedrooms', 'bathrooms', 'age', 'distance']
vif_house = calculate_vif(df_house, features_house)
print("\nVIF:")
print(vif_house)

# Normality test
stat, p = stats.shapiro(model_house.resid[:5000])  # Shapiro limited to 5000
print(f"\nResidual normality (Shapiro): p = {p:.4f}")

# Homoscedasticity test
X_house = sm.add_constant(df_house[features_house])
bp_stat, bp_p, _, _ = het_breuschpagan(model_house.resid, X_house)
print(f"Homoscedasticity (Breusch-Pagan): p = {bp_p:.4f}")
```

### 6.4 Model Improvement

```python
# Add interaction term
model_improved = smf.ols('price ~ size * bedrooms + bathrooms + age + distance',
                          data=df_house).fit()

print("Improved model (with interaction):")
print(model_improved.summary().tables[1])

# Compare AIC/BIC
print(f"\nModel comparison:")
print(f"Basic model: AIC = {model_house.aic:.2f}, BIC = {model_house.bic:.2f}")
print(f"Improved model: AIC = {model_improved.aic:.2f}, BIC = {model_improved.bic:.2f}")

# ANOVA model comparison
print("\nANOVA (model comparison):")
print(sm.stats.anova_lm(model_house, model_improved))
```

---

## Practice Problems

### Problem 1: Regression Diagnostics
Perform regression analysis with the following data and check assumptions.
```python
np.random.seed(42)
X = np.random.normal(0, 1, 100)
Y = 2 + 3*X + np.random.normal(0, 1, 100) * np.abs(X)  # Heteroscedasticity
```

### Problem 2: Multicollinearity
Diagnose and address multicollinearity in the data below.
```python
np.random.seed(42)
X1 = np.random.normal(0, 1, 100)
X2 = 0.9*X1 + np.random.normal(0, 0.1, 100)
X3 = np.random.normal(0, 1, 100)
Y = 1 + 2*X1 + 3*X3 + np.random.normal(0, 1, 100)
```

### Problem 3: Variable Selection
Select optimal variable combination from 5 independent variables.
- Apply forward selection, backward elimination, and AIC criteria
- Compare results

---

## Summary

| Diagnostic Item | Test/Method | Python Function |
|-----------------|-------------|-----------------|
| Normality | Shapiro-Wilk | `stats.shapiro()` |
| Homoscedasticity | Breusch-Pagan | `het_breuschpagan()` |
| Independence | Durbin-Watson | `durbin_watson()` |
| Influence | Cook's D | `get_influence().cooks_distance` |
| Multicollinearity | VIF | `variance_inflation_factor()` |
| Variable selection | AIC/BIC | `model.aic`, `model.bic` |
