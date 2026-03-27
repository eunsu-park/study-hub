# 17. Generalized Linear Models (GLM)

[Previous: Advanced Regression Analysis](./16_Regression_Analysis_Advanced.md) | [Next: Introduction to Bayesian Statistics](./18_Bayesian_Statistics_Basics.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Describe the three components of a GLM (random component, systematic component, link function) and identify the appropriate GLM for a given data type
2. Fit and interpret binary logistic regression models, including odds ratios and confidence intervals
3. Evaluate classification models using confusion matrices, ROC curves, and AUC
4. Implement Poisson regression for count data and use offsets to model rates
5. Detect overdispersion and apply negative binomial regression as a remedy
6. Perform GLM diagnostics using Pearson residuals, deviance residuals, and influence measures
7. Compare nested and non-nested GLMs using deviance tests, AIC, and BIC

---

Standard linear regression assumes a normally distributed response, but real-world outcomes are often binary (purchased or not), counts (number of incidents), or strictly positive (insurance claims). GLMs unify these cases under a single framework by pairing each response type with an appropriate distribution and link function. This lesson gives you the practical tools to model the diverse data types you will encounter beyond the Gaussian world.

---

## 1. GLM Framework

### 1.1 Components of GLM

```python
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import links

# Three components of GLM:
# 1. Random Component: Probability distribution of Y (exponential family)
# 2. Systematic Component: η = β₀ + β₁X₁ + ... + βₚXₚ
# 3. Link Function: g(μ) = η, μ = E[Y]

# Major GLM types
glm_types = pd.DataFrame({
    'Model': ['Linear', 'Logistic', 'Poisson', 'Gamma', 'Negative Binomial'],
    'Distribution': ['Normal', 'Binomial', 'Poisson', 'Gamma', 'Negative Binomial'],
    'Link Function': ['Identity', 'Logit', 'Log', 'Inverse', 'Log'],
    'Y Type': ['Continuous', 'Binary/Proportion', 'Count', 'Positive Continuous', 'Over-dispersed Count']
})

print("GLM Types:")
print(glm_types.to_string(index=False))
```

### 1.2 Link Functions

```python
# Major link functions

def identity_link(mu):
    """Identity link: η = μ"""
    return mu

def logit_link(mu):
    """Logit link: η = log(μ/(1-μ))"""
    return np.log(mu / (1 - mu))

def log_link(mu):
    """Log link: η = log(μ)"""
    return np.log(mu)

def inverse_link(mu):
    """Inverse link: η = 1/μ"""
    return 1 / mu

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Identity
mu = np.linspace(-2, 2, 100)
axes[0, 0].plot(mu, identity_link(mu), 'b-', linewidth=2)
axes[0, 0].set_xlabel('μ')
axes[0, 0].set_ylabel('η = g(μ)')
axes[0, 0].set_title('Identity Link: η = μ')
axes[0, 0].grid(alpha=0.3)

# Logit
mu = np.linspace(0.01, 0.99, 100)
axes[0, 1].plot(mu, logit_link(mu), 'r-', linewidth=2)
axes[0, 1].set_xlabel('μ (probability)')
axes[0, 1].set_ylabel('η = logit(μ)')
axes[0, 1].set_title('Logit Link: η = log(μ/(1-μ))')
axes[0, 1].grid(alpha=0.3)

# Log
mu = np.linspace(0.1, 10, 100)
axes[1, 0].plot(mu, log_link(mu), 'g-', linewidth=2)
axes[1, 0].set_xlabel('μ')
axes[1, 0].set_ylabel('η = log(μ)')
axes[1, 0].set_title('Log Link: η = log(μ)')
axes[1, 0].grid(alpha=0.3)

# Inverse logit (sigmoid) - inverse function
eta = np.linspace(-6, 6, 100)
sigmoid = 1 / (1 + np.exp(-eta))
axes[1, 1].plot(eta, sigmoid, 'purple', linewidth=2)
axes[1, 1].set_xlabel('η = Xβ')
axes[1, 1].set_ylabel('μ = P(Y=1)')
axes[1, 1].set_title('Inverse Logit (Sigmoid): μ = 1/(1+e^(-η))')
axes[1, 1].grid(alpha=0.3)
axes[1, 1].axhline(0.5, color='gray', linestyle='--', alpha=0.5)
axes[1, 1].axvline(0, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
```

### 1.3 Exponential Family Distributions

```python
# General form of exponential family:
# f(y|θ,φ) = exp{(yθ - b(θ))/a(φ) + c(y,φ)}
# θ: canonical parameter
# φ: dispersion parameter

# Mean-variance relationship of major distributions
print("Mean-variance relationship of exponential family distributions:")
print("-" * 50)
print(f"{'Distribution':<15} {'Mean (μ)':<20} {'Variance (V(μ))'}")
print("-" * 50)
print(f"{'Normal':<15} {'μ':<20} {'σ²'}")
print(f"{'Binomial':<15} {'nπ':<20} {'nπ(1-π)'}")
print(f"{'Poisson':<15} {'λ':<20} {'λ'}")
print(f"{'Gamma':<15} {'μ':<20} {'μ²/ν'}")
print(f"{'Inverse Gaussian':<15} {'μ':<20} {'μ³/λ'}")
```

---

## 2. Logistic Regression

### 2.1 Binary Logistic Regression

```python
# Generate data: customer churn prediction
np.random.seed(42)
n = 500

# Independent variables
age = np.random.normal(40, 10, n)
tenure = np.random.exponential(24, n)  # Subscription duration (months)
monthly_charge = np.random.normal(65, 20, n)

# Logit model
eta = -5 + 0.05*age - 0.03*tenure + 0.04*monthly_charge
prob = 1 / (1 + np.exp(-eta))
churn = np.random.binomial(1, prob, n)

df_churn = pd.DataFrame({
    'churn': churn,
    'age': age,
    'tenure': tenure,
    'monthly_charge': monthly_charge
})

print("Churn rate:", df_churn['churn'].mean())
print("\nData sample:")
print(df_churn.head())
```

### 2.2 Model Fitting

```python
# Logistic regression with statsmodels
model_logit = smf.glm('churn ~ age + tenure + monthly_charge',
                       data=df_churn,
                       family=sm.families.Binomial()).fit()

print("Logistic regression results:")
print(model_logit.summary())

# Or using logit() function
model_logit2 = smf.logit('churn ~ age + tenure + monthly_charge', data=df_churn).fit()
```

### 2.3 Interpreting Coefficients

```python
# Calculate Odds Ratios
odds_ratios = np.exp(model_logit.params)
conf_int = np.exp(model_logit.conf_int())

print("Odds Ratios:")
print("-" * 60)
print(f"{'Variable':<20} {'OR':<10} {'95% CI':<25} {'p-value'}")
print("-" * 60)

for var in model_logit.params.index:
    or_val = odds_ratios[var]
    ci_low, ci_high = conf_int.loc[var]
    pval = model_logit.pvalues[var]
    print(f"{var:<20} {or_val:<10.4f} [{ci_low:.4f}, {ci_high:.4f}] {pval:<.4f}")

print("\nInterpretation example:")
print(f"- 1-month increase in tenure changes churn odds by {(odds_ratios['tenure']-1)*100:.2f}%")
print(f"  (OR = {odds_ratios['tenure']:.4f})")
```

### 2.4 Prediction and Evaluation

```python
# Predicted probabilities
df_churn['pred_prob'] = model_logit.predict(df_churn)
df_churn['pred_class'] = (df_churn['pred_prob'] >= 0.5).astype(int)

# Confusion matrix
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve

cm = confusion_matrix(df_churn['churn'], df_churn['pred_class'])
print("Confusion matrix:")
print(cm)

print("\nClassification report:")
print(classification_report(df_churn['churn'], df_churn['pred_class']))

# ROC curve
fpr, tpr, thresholds = roc_curve(df_churn['churn'], df_churn['pred_prob'])
auc = roc_auc_score(df_churn['churn'], df_churn['pred_prob'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC curve
axes[0].plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc:.3f})')
axes[0].plot([0, 1], [0, 1], 'r--', alpha=0.5)
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Predicted probability distribution
axes[1].hist(df_churn[df_churn['churn']==0]['pred_prob'], bins=30, alpha=0.5,
             label='Retained (y=0)', density=True)
axes[1].hist(df_churn[df_churn['churn']==1]['pred_prob'], bins=30, alpha=0.5,
             label='Churned (y=1)', density=True)
axes[1].set_xlabel('Predicted probability')
axes[1].set_ylabel('Density')
axes[1].set_title('Predicted probability distribution')
axes[1].legend()
axes[1].axvline(0.5, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

print(f"AUC: {auc:.4f}")
```

### 2.5 Multinomial Logistic Regression

```python
# Dependent variable with 3+ categories
np.random.seed(42)
n = 600

# 3 choices: A, B, C
X1 = np.random.normal(0, 1, n)
X2 = np.random.normal(0, 1, n)

# Multinomial logit model (reference category: C)
eta_A = 0.5 + 1.5*X1 - 0.5*X2
eta_B = -0.5 - 0.5*X1 + 1.5*X2
eta_C = np.zeros(n)  # Reference category

# Softmax probabilities
exp_eta = np.exp(np.column_stack([eta_A, eta_B, eta_C]))
probs = exp_eta / exp_eta.sum(axis=1, keepdims=True)

# Category selection
choice = np.array(['A', 'B', 'C'])[np.array([np.random.choice(3, p=p) for p in probs])]

df_multi = pd.DataFrame({
    'choice': choice,
    'X1': X1,
    'X2': X2
})

print("Frequency by category:")
print(df_multi['choice'].value_counts())

# statsmodels MNLogit
from statsmodels.discrete.discrete_model import MNLogit

# Convert categories to numbers
df_multi['choice_code'] = df_multi['choice'].map({'A': 0, 'B': 1, 'C': 2})

X_multi = sm.add_constant(df_multi[['X1', 'X2']])
model_mnlogit = MNLogit(df_multi['choice_code'], X_multi).fit()

print("\nMultinomial logistic regression results:")
print(model_mnlogit.summary())
```

---

## 3. Poisson Regression

### 3.1 Basic Poisson Regression

```python
# Generate data: website visit counts
np.random.seed(42)
n = 400

# Independent variables
ad_spend = np.random.exponential(100, n)  # Ad spending
day_of_week = np.random.randint(0, 7, n)  # Day of week
is_weekend = (day_of_week >= 5).astype(int)

# Poisson model
log_lambda = 2 + 0.01*ad_spend - 0.5*is_weekend
lambda_param = np.exp(log_lambda)
visits = np.random.poisson(lambda_param)

df_poisson = pd.DataFrame({
    'visits': visits,
    'ad_spend': ad_spend,
    'is_weekend': is_weekend
})

print("Visit count statistics:")
print(df_poisson['visits'].describe())

# Check Poisson distribution
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df_poisson['visits'], bins=30, density=True, alpha=0.7, edgecolor='black')
ax.set_xlabel('Visit count')
ax.set_ylabel('Density')
ax.set_title('Visit count distribution')
plt.show()
```

### 3.2 Model Fitting

```python
# Poisson regression
model_poisson = smf.glm('visits ~ ad_spend + is_weekend',
                         data=df_poisson,
                         family=sm.families.Poisson()).fit()

print("Poisson regression results:")
print(model_poisson.summary())

# Incidence Rate Ratio (IRR)
irr = np.exp(model_poisson.params)
irr_ci = np.exp(model_poisson.conf_int())

print("\nIncidence Rate Ratio (IRR):")
print("-" * 60)
for var in model_poisson.params.index:
    irr_val = irr[var]
    ci_low, ci_high = irr_ci.loc[var]
    print(f"{var}: IRR = {irr_val:.4f} [{ci_low:.4f}, {ci_high:.4f}]")

print("\nInterpretation:")
print(f"- Weekend visits decrease by average {(1-irr['is_weekend'])*100:.1f}%")
print(f"- 1-unit increase in ad spending increases visits by {(irr['ad_spend']-1)*100:.2f}%")
```

### 3.3 Using Offset

```python
# Offset: used when exposure time/space differs
# log(λ/t) = β₀ + β₁X  →  log(λ) = log(t) + β₀ + β₁X

np.random.seed(42)
n = 300

# Crime counts by area (different population sizes)
population = np.random.uniform(10000, 100000, n)
poverty_rate = np.random.uniform(5, 30, n)
unemployment = np.random.uniform(3, 15, n)

# Crime rate = population × exp(β)
log_crime_rate = -6 + 0.05*poverty_rate + 0.03*unemployment
crime_rate = np.exp(log_crime_rate)
crimes = np.random.poisson(population * crime_rate)

df_crime = pd.DataFrame({
    'crimes': crimes,
    'population': population,
    'poverty_rate': poverty_rate,
    'unemployment': unemployment,
    'log_population': np.log(population)
})

# Model with offset
model_offset = smf.glm('crimes ~ poverty_rate + unemployment',
                        data=df_crime,
                        family=sm.families.Poisson(),
                        offset=df_crime['log_population']).fit()

print("Poisson regression with offset:")
print(model_offset.summary())

# Model without offset (incorrect)
model_no_offset = smf.glm('crimes ~ poverty_rate + unemployment',
                           data=df_crime,
                           family=sm.families.Poisson()).fit()

print("\nModel without offset (for comparison):")
print(f"poverty_rate coefficient: {model_no_offset.params['poverty_rate']:.4f}")
print(f"With offset coefficient: {model_offset.params['poverty_rate']:.4f} (more accurate)")
```

### 3.4 Overdispersion

```python
# Poisson assumption: E[Y] = Var(Y) = λ
# Overdispersion: Var(Y) > E[Y]

# Check overdispersion
def check_overdispersion(model):
    """Overdispersion test"""
    pearson_chi2 = model.pearson_chi2
    df_resid = model.df_resid
    dispersion = pearson_chi2 / df_resid

    print(f"Pearson χ²: {pearson_chi2:.2f}")
    print(f"Degrees of freedom: {df_resid}")
    print(f"Dispersion parameter: {dispersion:.4f}")
    print(f"→ If {dispersion:.2f} > 1, suspect overdispersion")

    return dispersion

print("Overdispersion test:")
print("-" * 40)
dispersion = check_overdispersion(model_poisson)

# Generate overdispersed data
np.random.seed(42)
n = 500

X = np.random.normal(0, 1, n)
# Negative binomial for overdispersed data
log_mu = 2 + 0.5*X
mu = np.exp(log_mu)
# Negative binomial with r=5 (variance > mean)
Y_overdispersed = np.random.negative_binomial(5, 5/(5+mu), n)

df_over = pd.DataFrame({'Y': Y_overdispersed, 'X': X})

print(f"\nOverdispersed data:")
print(f"Mean: {df_over['Y'].mean():.2f}")
print(f"Variance: {df_over['Y'].var():.2f}")
print(f"Variance/Mean ratio: {df_over['Y'].var()/df_over['Y'].mean():.2f}")

# Poisson model (incorrect)
model_pois_over = smf.glm('Y ~ X', data=df_over, family=sm.families.Poisson()).fit()
print("\nPoisson model overdispersion:")
check_overdispersion(model_pois_over)
```

### 3.5 Negative Binomial Regression

```python
# Address overdispersion: negative binomial regression
from statsmodels.genmod.families import NegativeBinomial

model_nb = smf.glm('Y ~ X', data=df_over,
                    family=NegativeBinomial(alpha=1)).fit()

print("Negative binomial regression results:")
print(model_nb.summary())

# Poisson vs negative binomial comparison
print("\nModel comparison:")
print("-" * 40)
print(f"{'Model':<15} {'AIC':<12} {'Deviance':<12}")
print("-" * 40)
print(f"{'Poisson':<15} {model_pois_over.aic:<12.2f} {model_pois_over.deviance:<12.2f}")
print(f"{'Neg Binomial':<15} {model_nb.aic:<12.2f} {model_nb.deviance:<12.2f}")

# Quasi-Poisson regression
# Model variance as φμ (φ is estimated)
model_quasi = smf.glm('Y ~ X', data=df_over,
                       family=sm.families.Poisson()).fit(scale='X2')  # Based on Pearson χ²

print(f"\nQuasi-Poisson dispersion parameter: {model_quasi.scale:.4f}")
```

---

## 4. Model Diagnostics

### 4.1 Residual Analysis

```python
# GLM residual types
# 1. Pearson residuals: (Y - μ̂) / √V(μ̂)
# 2. Deviance residuals: sign(Y-μ̂) × √d_i
# 3. Standardized residuals

# Poisson model diagnostics
model = model_poisson

# Calculate residuals
pearson_resid = model.resid_pearson
deviance_resid = model.resid_deviance
fitted = model.fittedvalues

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Pearson residuals vs fitted
axes[0, 0].scatter(fitted, pearson_resid, alpha=0.5)
axes[0, 0].axhline(0, color='red', linestyle='--')
axes[0, 0].set_xlabel('Fitted values')
axes[0, 0].set_ylabel('Pearson residuals')
axes[0, 0].set_title('Pearson residuals vs Fitted')

# 2. Deviance residuals vs fitted
axes[0, 1].scatter(fitted, deviance_resid, alpha=0.5)
axes[0, 1].axhline(0, color='red', linestyle='--')
axes[0, 1].set_xlabel('Fitted values')
axes[0, 1].set_ylabel('Deviance residuals')
axes[0, 1].set_title('Deviance residuals vs Fitted')

# 3. Residual Q-Q plot
stats.probplot(pearson_resid, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Pearson residuals Q-Q Plot')

# 4. Residual histogram
axes[1, 1].hist(pearson_resid, bins=30, density=True, alpha=0.7, edgecolor='black')
axes[1, 1].set_xlabel('Pearson residuals')
axes[1, 1].set_ylabel('Density')
axes[1, 1].set_title('Pearson residuals distribution')

plt.tight_layout()
plt.show()
```

### 4.2 Goodness-of-Fit Test

```python
# Deviance test
deviance = model_poisson.deviance
df = model_poisson.df_resid
p_deviance = 1 - stats.chi2.cdf(deviance, df)

print("Goodness-of-fit test:")
print("-" * 40)
print(f"Deviance: {deviance:.2f}")
print(f"Degrees of freedom: {df}")
print(f"p-value: {p_deviance:.4f}")
print(f"→ p > 0.05 indicates model fits well")

# Pearson test
pearson = model_poisson.pearson_chi2
p_pearson = 1 - stats.chi2.cdf(pearson, df)

print(f"\nPearson χ²: {pearson:.2f}")
print(f"p-value: {p_pearson:.4f}")
```

### 4.3 Influence Diagnostics

```python
# GLM influence analysis
from statsmodels.stats.outliers_influence import GLMInfluence

influence = GLMInfluence(model_poisson)

# Cook's Distance
cooks_d = influence.cooks_distance[0]

# Leverage (Hat values)
leverage = influence.hat_matrix_diag

# DFFITS
dffits = influence.dffits[0]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Cook's D
n = len(cooks_d)
axes[0].stem(range(n), cooks_d, markerfmt='o', basefmt=' ')
axes[0].axhline(4/n, color='red', linestyle='--', label=f'4/n = {4/n:.4f}')
axes[0].set_xlabel('Observation index')
axes[0].set_ylabel("Cook's Distance")
axes[0].set_title("Cook's Distance")
axes[0].legend()

# Leverage
axes[1].hist(leverage, bins=30, alpha=0.7, edgecolor='black')
p = len(model_poisson.params)
axes[1].axvline(2*p/n, color='red', linestyle='--', label=f'2p/n = {2*p/n:.4f}')
axes[1].set_xlabel('Leverage')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Leverage distribution')
axes[1].legend()

# DFFITS
axes[2].stem(range(n), dffits, markerfmt='o', basefmt=' ')
threshold = 2 * np.sqrt(p/n)
axes[2].axhline(threshold, color='red', linestyle='--')
axes[2].axhline(-threshold, color='red', linestyle='--', label=f'±2√(p/n)')
axes[2].set_xlabel('Observation index')
axes[2].set_ylabel('DFFITS')
axes[2].set_title('DFFITS')
axes[2].legend()

plt.tight_layout()
plt.show()

# Influential observations
influential_cooks = np.where(cooks_d > 4/n)[0]
print(f"Observations with Cook's D > 4/n: {len(influential_cooks)} cases")
```

---

## 5. Model Comparison

### 5.1 Deviance Test

```python
# Nested model comparison: Deviance difference test

# Reduced model
model_reduced = smf.glm('visits ~ ad_spend', data=df_poisson,
                         family=sm.families.Poisson()).fit()

# Full model
model_full = model_poisson

# Deviance difference test
dev_diff = model_reduced.deviance - model_full.deviance
df_diff = model_reduced.df_resid - model_full.df_resid
p_value = 1 - stats.chi2.cdf(dev_diff, df_diff)

print("Nested model comparison (Deviance test):")
print("-" * 50)
print(f"Reduced model Deviance: {model_reduced.deviance:.2f}")
print(f"Full model Deviance: {model_full.deviance:.2f}")
print(f"Deviance difference: {dev_diff:.2f}")
print(f"Degrees of freedom difference: {df_diff}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("→ Full model fits significantly better")
else:
    print("→ Reduced model is sufficient")
```

### 5.2 Information Criteria Comparison

```python
# AIC, BIC comparison

# Define multiple models
models = {
    'Model 1: ad_spend': smf.glm('visits ~ ad_spend', data=df_poisson,
                                  family=sm.families.Poisson()).fit(),
    'Model 2: is_weekend': smf.glm('visits ~ is_weekend', data=df_poisson,
                                    family=sm.families.Poisson()).fit(),
    'Model 3: both': smf.glm('visits ~ ad_spend + is_weekend', data=df_poisson,
                              family=sm.families.Poisson()).fit(),
}

print("Model comparison (information criteria):")
print("-" * 60)
print(f"{'Model':<25} {'AIC':<12} {'BIC':<12} {'Deviance':<12}")
print("-" * 60)

for name, model in models.items():
    print(f"{name:<25} {model.aic:<12.2f} {model.bic:<12.2f} {model.deviance:<12.2f}")

# Best model
best_aic = min(models.items(), key=lambda x: x[1].aic)
best_bic = min(models.items(), key=lambda x: x[1].bic)

print(f"\nBest model (AIC): {best_aic[0]}")
print(f"Best model (BIC): {best_bic[0]}")
```

---

## 6. Practical Example

### 6.1 Insurance Claims Count Prediction

```python
# Simulate insurance claims data
np.random.seed(42)
n = 1000

# Customer characteristics
age = np.random.normal(40, 12, n)
age = np.clip(age, 18, 80)
years_driving = np.clip(age - 18 - np.random.exponential(2, n), 0, None)
previous_claims = np.random.poisson(0.5, n)
vehicle_age = np.random.exponential(5, n)

# Claims count (negative binomial for overdispersion simulation)
log_mu = -1 + 0.02*age - 0.01*years_driving + 0.3*previous_claims + 0.05*vehicle_age
mu = np.exp(log_mu)
claims = np.random.negative_binomial(2, 2/(2+mu), n)

df_insurance = pd.DataFrame({
    'claims': claims,
    'age': age,
    'years_driving': years_driving,
    'previous_claims': previous_claims,
    'vehicle_age': vehicle_age
})

print("Insurance claims data:")
print(df_insurance.describe())
print(f"\nClaims count mean: {claims.mean():.2f}, variance: {claims.var():.2f}")
print(f"Variance/Mean ratio: {claims.var()/claims.mean():.2f} (overdispersion)")
```

### 6.2 Model Fitting and Comparison

```python
# Poisson model
model_pois = smf.glm('claims ~ age + years_driving + previous_claims + vehicle_age',
                      data=df_insurance,
                      family=sm.families.Poisson()).fit()

# Negative binomial model
model_nb = smf.glm('claims ~ age + years_driving + previous_claims + vehicle_age',
                    data=df_insurance,
                    family=NegativeBinomial(alpha=1)).fit()

print("Model comparison:")
print("=" * 70)

# Check overdispersion
print("\nPoisson model overdispersion:")
check_overdispersion(model_pois)

# Coefficient comparison
print("\nCoefficient comparison:")
print("-" * 70)
coef_comparison = pd.DataFrame({
    'Poisson': model_pois.params,
    'Neg Binomial': model_nb.params,
    'Poisson SE': model_pois.bse,
    'NB SE': model_nb.bse
})
print(coef_comparison)

# Information criteria
print("\nInformation criteria:")
print(f"Poisson AIC: {model_pois.aic:.2f}")
print(f"Neg Binomial AIC: {model_nb.aic:.2f}")
print(f"→ Negative binomial model fits better (lower AIC)")
```

### 6.3 Final Model Interpretation

```python
# Interpret negative binomial model
print("Negative binomial regression results:")
print(model_nb.summary())

# Incidence Rate Ratio (IRR)
irr = np.exp(model_nb.params)
irr_ci = np.exp(model_nb.conf_int())

print("\nIncidence Rate Ratio (IRR):")
print("-" * 70)
print(f"{'Variable':<20} {'IRR':<10} {'95% CI':<25} {'Interpretation'}")
print("-" * 70)

interpretations = {
    'age': '1-year increase in age',
    'years_driving': '1-year increase in driving experience',
    'previous_claims': '1 additional previous claim',
    'vehicle_age': '1-year increase in vehicle age'
}

for var in ['age', 'years_driving', 'previous_claims', 'vehicle_age']:
    irr_val = irr[var]
    ci_low, ci_high = irr_ci.loc[var]
    pct_change = (irr_val - 1) * 100
    direction = "increase" if pct_change > 0 else "decrease"
    print(f"{var:<20} {irr_val:<10.4f} [{ci_low:.4f}, {ci_high:.4f}] "
          f"{interpretations[var]}: claims {abs(pct_change):.1f}% {direction}")
```

---

## Practice Problems

### Problem 1: Logistic Regression
Build a logistic regression model to predict pass/fail with the following data.
```python
np.random.seed(42)
study_hours = np.random.uniform(1, 10, 100)
prep_course = np.random.binomial(1, 0.5, 100)
pass_exam = (0.3*study_hours + 1.5*prep_course + np.random.normal(0, 1, 100) > 3).astype(int)
```

### Problem 2: Poisson Regression
Apply Poisson regression to website click count data and check for overdispersion.

### Problem 3: Model Selection
Select optimal variable combination in logistic regression using AIC.

---

## Summary

| GLM Type | Distribution | Link Function | Python |
|----------|--------------|---------------|--------|
| Linear regression | Normal | Identity | `sm.OLS()` |
| Logistic | Binomial | Logit | `smf.logit()` |
| Poisson | Poisson | Log | `family=Poisson()` |
| Negative binomial | Neg Binomial | Log | `family=NegativeBinomial()` |
| Gamma | Gamma | Inverse | `family=Gamma()` |

| Diagnostic Item | Method | Criterion |
|-----------------|--------|-----------|
| Overdispersion | Pearson χ²/df | > 1 |
| Goodness-of-fit | Deviance test | p > 0.05 |
| Influence | Cook's D | > 4/n |
| Model comparison | AIC, BIC | Lower is better |
