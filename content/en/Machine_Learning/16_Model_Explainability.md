# Model Explainability

**Previous**: [Feature Engineering](./15_Feature_Engineering.md) | **Next**: [Imbalanced Data](./17_Imbalanced_Data.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Distinguish between intrinsic interpretability and post-hoc explainability methods
2. Compare impurity-based and permutation-based feature importance and identify their limitations
3. Explain Shapley values theory and apply SHAP to generate both global and local explanations
4. Implement LIME to explain individual predictions and compare its results with SHAP
5. Interpret Partial Dependence Plots, ICE plots, and ALE plots for feature effect visualization
6. Apply explainability techniques to detect fairness and bias issues in ML models
7. Design an explainability wrapper for production models that returns predictions with human-readable reasons

---

A model that makes accurate predictions but cannot explain them is a liability in any high-stakes domain. Regulators demand explanations (GDPR Article 22, EU AI Act), clinicians refuse to trust black boxes, and data scientists need to debug models that fail silently. This lesson gives you the tools -- SHAP, LIME, PDP, ALE -- to open any black-box model and understand what it has actually learned, turning opaque predictions into actionable insights.

---

## 1. Interpretability vs. Explainability

### 1.1 Key Concepts

```python
"""
Interpretability: How easily a human can understand the model's mechanism.
  - Linear regression: coefficient = direct effect → highly interpretable
  - Deep neural network: millions of parameters → low interpretability

Explainability: The ability to explain a model's predictions after the fact.
  - SHAP, LIME: post-hoc explanation methods
  - Can be applied to any black-box model

Taxonomy:
┌──────────────────────┬───────────────────────┐
│     Intrinsic        │      Post-hoc         │
│  (built-in)          │  (after training)     │
├──────────────────────┼───────────────────────┤
│ Linear Regression    │ SHAP                  │
│ Decision Trees       │ LIME                  │
│ Rule-based Models    │ PDP / ICE / ALE       │
│ GAMs                 │ Permutation Importance│
│ (small) Logistic Reg │ Counterfactuals       │
└──────────────────────┴───────────────────────┘

Scope:
- Global: Understand the overall model behavior
- Local: Understand a single prediction
"""
```

### 1.2 The Accuracy-Interpretability Trade-off

| Model | Interpretability | Typical Accuracy |
|-------|-----------------|------------------|
| Linear / Logistic Regression | Very High | Low-Medium |
| Decision Tree (shallow) | High | Low-Medium |
| Random Forest | Medium | High |
| Gradient Boosting (XGBoost) | Low | Very High |
| Neural Networks | Very Low | Very High |

**Important**: This trade-off is a tendency, not a rule. With proper feature engineering, simple models can rival complex ones. And with SHAP/LIME, complex models can be explained.

---

## 2. Feature Importance Methods

### 2.1 Impurity-Based Importance (Trees)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# Load data
housing = fetch_california_housing(as_frame=True)
X, y = housing.data, housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

gb = GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
gb.fit(X_train, y_train)

# Impurity-based importance
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

rf_imp = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values()
rf_imp.plot(kind='barh', ax=axes[0], color='steelblue')
axes[0].set_title(f'Random Forest (R²={rf.score(X_test, y_test):.3f})')
axes[0].set_xlabel('Impurity-based Importance')

gb_imp = pd.Series(gb.feature_importances_, index=X_train.columns).sort_values()
gb_imp.plot(kind='barh', ax=axes[1], color='forestgreen')
axes[1].set_title(f'Gradient Boosting (R²={gb.score(X_test, y_test):.3f})')
axes[1].set_xlabel('Impurity-based Importance')

plt.tight_layout()
plt.savefig('impurity_importance.png', dpi=150)
plt.show()
```

**Limitation**: Impurity-based importance is biased toward high-cardinality features and doesn't account for feature interactions properly.

### 2.2 Permutation Importance

```python
from sklearn.inspection import permutation_importance

# Permutation importance: shuffle one feature, measure performance drop
perm_result = permutation_importance(
    gb, X_test, y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1,
    scoring='r2',
)

perm_imp = pd.DataFrame({
    'mean': perm_result.importances_mean,
    'std': perm_result.importances_std,
}, index=X_train.columns).sort_values('mean')

fig, ax = plt.subplots(figsize=(8, 5))
perm_imp['mean'].plot(kind='barh', xerr=perm_imp['std'], ax=ax, color='coral')
ax.set_title('Permutation Importance (Gradient Boosting)')
ax.set_xlabel('Decrease in R² when feature is shuffled')
plt.tight_layout()
plt.savefig('permutation_importance.png', dpi=150)
plt.show()

print("Permutation importance ranking:")
print(perm_imp.sort_values('mean', ascending=False).round(4))
```

**Advantages over impurity-based**:
- Works with any model (not just trees)
- Computed on test set (reflects generalization)
- Not biased by feature cardinality

**Limitation**: Correlated features can share importance — shuffling one doesn't reduce performance much if its correlated partner remains.

---

## 3. SHAP (SHapley Additive exPlanations)

### 3.1 Shapley Values Theory

```python
"""
Shapley values come from cooperative game theory (Lloyd Shapley, 1953).

Idea: How much does each player (feature) contribute to the payout (prediction)?

For prediction f(x) with features {x1, x2, ..., xp}:
  φᵢ = Σ_{S ⊆ N\{i}} |S|!(p-|S|-1)! / p! × [f(S ∪ {i}) - f(S)]

where:
  - S is a subset of features (coalition)
  - f(S) is the model prediction using only features in S
  - φᵢ is the Shapley value for feature i

Properties (axioms):
1. Efficiency:   Σ φᵢ = f(x) - E[f(x)]   (values sum to prediction - average)
2. Symmetry:     Equal contributors get equal values
3. Dummy:        Non-contributing features get zero
4. Additivity:   Values for combined models = sum of individual values

SHAP connects Shapley values to ML:
  f(x) = base_value + Σ φᵢ(x)
  where base_value = E[f(x)] = average prediction
"""
```

### 3.2 SHAP in Practice

```python
# pip install shap
import shap

# Use the Gradient Boosting model from above
explainer = shap.TreeExplainer(gb)
shap_values = explainer.shap_values(X_test)

print(f"SHAP values shape: {shap_values.shape}")
print(f"Base value (average prediction): {explainer.expected_value:.4f}")
print(f"Actual average of y_test: {y_test.mean():.4f}")

# Verify additivity: base_value + sum(shap_values) ≈ prediction
sample_idx = 0
prediction = gb.predict(X_test.iloc[[sample_idx]])[0]
shap_sum = explainer.expected_value + shap_values[sample_idx].sum()
print(f"\nSample {sample_idx}:")
print(f"  Model prediction:           {prediction:.4f}")
print(f"  base_value + Σ(SHAP values): {shap_sum:.4f}")
print(f"  Difference:                  {abs(prediction - shap_sum):.6f}")
```

### 3.3 SHAP Visualization

```python
import shap

# 1. Summary Plot (Global): Feature importance + direction of effect
shap.summary_plot(shap_values, X_test, show=False)
plt.title('SHAP Summary Plot')
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
plt.show()
```

```python
# 2. Bar Plot (Global): Mean absolute SHAP values
shap.summary_plot(shap_values, X_test, plot_type='bar', show=False)
plt.title('SHAP Feature Importance (mean |SHAP|)')
plt.tight_layout()
plt.savefig('shap_bar.png', dpi=150, bbox_inches='tight')
plt.show()
```

```python
# 3. Waterfall Plot (Local): Explain a single prediction
shap.plots.waterfall(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_test.iloc[0],
    feature_names=X_test.columns.tolist(),
), show=False)
plt.title('Waterfall Plot: Single Prediction Explanation')
plt.tight_layout()
plt.savefig('shap_waterfall.png', dpi=150, bbox_inches='tight')
plt.show()
```

```python
# 4. Dependence Plot: How one feature affects predictions
shap.dependence_plot('MedInc', shap_values, X_test, interaction_index='AveOccup', show=False)
plt.title('SHAP Dependence: MedInc (colored by AveOccup)')
plt.tight_layout()
plt.savefig('shap_dependence.png', dpi=150, bbox_inches='tight')
plt.show()
```

```python
# 5. Force Plot (Local): Single prediction breakdown
shap.initjs()
shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    X_test.iloc[0],
    matplotlib=True,
    show=False,
)
plt.title('Force Plot: Single Prediction')
plt.tight_layout()
plt.savefig('shap_force.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 3.4 SHAP Variants

```python
"""
SHAP has optimized algorithms for different model types:

┌─────────────────────┬───────────────────┬────────────────────┐
│ Explainer           │ Models            │ Speed              │
├─────────────────────┼───────────────────┼────────────────────┤
│ TreeExplainer       │ XGBoost, LightGBM │ Very fast (exact)  │
│                     │ Random Forest     │                    │
├─────────────────────┼───────────────────┼────────────────────┤
│ LinearExplainer     │ Linear models     │ Very fast (exact)  │
├─────────────────────┼───────────────────┼────────────────────┤
│ KernelExplainer     │ Any model         │ Slow (approximate) │
├─────────────────────┼───────────────────┼────────────────────┤
│ DeepExplainer       │ Neural networks   │ Fast (approximate) │
├─────────────────────┼───────────────────┼────────────────────┤
│ GradientExplainer   │ Neural networks   │ Fast (approximate) │
└─────────────────────┴───────────────────┴────────────────────┘

Rule of thumb:
- Tree models → TreeExplainer (always)
- Linear models → LinearExplainer
- Any other model → KernelExplainer (slower but universal)
- Deep learning → DeepExplainer or GradientExplainer
"""

# KernelExplainer example (model-agnostic)
from sklearn.svm import SVR

# Train an SVM (non-tree model)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

svm = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=10))
svm.fit(X_train, y_train)
print(f"SVM R²: {svm.score(X_test, y_test):.4f}")

# Use KernelExplainer (sample background for speed)
background = shap.sample(X_train, 100)
kernel_explainer = shap.KernelExplainer(svm.predict, background)

# Explain a small subset (KernelExplainer is slow)
X_explain = X_test.iloc[:20]
kernel_shap_values = kernel_explainer.shap_values(X_explain)

shap.summary_plot(kernel_shap_values, X_explain, show=False)
plt.title('SHAP Summary (SVM via KernelExplainer)')
plt.tight_layout()
plt.show()
```

### 3.5 SHAP Interaction Values

```python
# TreeExplainer can compute interaction effects
shap_interaction = explainer.shap_values(X_test.iloc[:200], check_additivity=False)

# For interaction values (pairwise), TreeExplainer provides:
interaction_values = explainer.shap_interaction_values(X_test.iloc[:200])
print(f"Interaction values shape: {interaction_values.shape}")
# Shape: (n_samples, n_features, n_features)

# Mean absolute interaction strengths
mean_interactions = np.abs(interaction_values).mean(axis=0)
interaction_df = pd.DataFrame(
    mean_interactions,
    index=X_test.columns,
    columns=X_test.columns,
)

plt.figure(figsize=(10, 8))
import matplotlib
mask = np.triu(np.ones_like(interaction_df, dtype=bool), k=1)
sns_available = True
try:
    import seaborn as sns
    sns.heatmap(interaction_df, mask=mask, annot=True, fmt='.3f',
                cmap='YlOrRd', square=True)
except ImportError:
    sns_available = False
    plt.imshow(interaction_df.values, cmap='YlOrRd')
    plt.colorbar()
    plt.xticks(range(len(interaction_df.columns)), interaction_df.columns, rotation=45)
    plt.yticks(range(len(interaction_df.index)), interaction_df.index)
plt.title('Mean |SHAP Interaction Values|')
plt.tight_layout()
plt.savefig('shap_interactions.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 4. LIME (Local Interpretable Model-agnostic Explanations)

### 4.1 How LIME Works

```python
"""
LIME explains individual predictions by fitting a simple, interpretable model
(usually linear regression) around the neighborhood of the prediction.

Algorithm:
1. Pick a sample to explain
2. Generate perturbed samples around it (add noise)
3. Get black-box model predictions for perturbed samples
4. Weight perturbed samples by proximity to original
5. Fit a simple model (linear regression) on weighted samples
6. Use simple model coefficients as the explanation

Key differences from SHAP:
- LIME: local linear approximation (approximate)
- SHAP: exact Shapley values (theoretically grounded)
- LIME: faster but less consistent
- SHAP: slower but satisfies uniqueness axioms
"""
```

### 4.2 LIME for Tabular Data

```python
# pip install lime
import lime
import lime.lime_tabular

# Create LIME explainer
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    mode='regression',
    verbose=False,
)

# Explain a single prediction
sample_idx = 0
explanation = lime_explainer.explain_instance(
    X_test.iloc[sample_idx].values,
    gb.predict,
    num_features=8,
    num_samples=5000,
)

# Display explanation
print(f"Prediction: {gb.predict(X_test.iloc[[sample_idx]])[0]:.4f}")
print(f"LIME intercept: {explanation.intercept[0]:.4f}")
print("\nFeature contributions:")
for feature, weight in explanation.as_list():
    print(f"  {feature}: {weight:+.4f}")

# Visualize
fig = explanation.as_pyplot_figure()
plt.title('LIME Explanation for Single Prediction')
plt.tight_layout()
plt.savefig('lime_explanation.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 4.3 Comparing SHAP vs LIME

```python
# Compare explanations for the same prediction
sample_idx = 0
sample = X_test.iloc[[sample_idx]]

# SHAP explanation
shap_exp = pd.Series(
    shap_values[sample_idx],
    index=X_test.columns,
    name='SHAP'
).sort_values(key=abs, ascending=False)

# LIME explanation
lime_exp = lime_explainer.explain_instance(
    X_test.iloc[sample_idx].values,
    gb.predict,
    num_features=8,
)
lime_dict = {
    feat.split(' ')[0]: weight
    for feat, weight in lime_exp.as_list()
}
lime_series = pd.Series(lime_dict, name='LIME')

# Compare
comparison = pd.DataFrame({
    'SHAP': shap_exp,
    'LIME': lime_series,
}).fillna(0)

fig, ax = plt.subplots(figsize=(10, 6))
comparison.plot(kind='barh', ax=ax)
ax.set_title('SHAP vs LIME: Feature Contributions')
ax.set_xlabel('Contribution to Prediction')
ax.axvline(0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig('shap_vs_lime.png', dpi=150, bbox_inches='tight')
plt.show()

print("Key differences:")
print("- SHAP: theoretically consistent, satisfies Shapley axioms")
print("- LIME: local linear approximation, may vary between runs")
print("- Both agree on most important features for most predictions")
```

---

## 5. Partial Dependence Plots (PDP) and ICE

### 5.1 Partial Dependence Plots

```python
from sklearn.inspection import PartialDependenceDisplay

# PDP: Average effect of a feature on prediction
fig, axes = plt.subplots(2, 4, figsize=(20, 8))

features_to_plot = X_train.columns.tolist()
PartialDependenceDisplay.from_estimator(
    gb, X_test, features_to_plot,
    kind='average',  # PDP (average effect)
    ax=axes.ravel()[:len(features_to_plot)],
    n_jobs=-1,
)

plt.suptitle('Partial Dependence Plots', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('pdp_all_features.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 5.2 Individual Conditional Expectation (ICE)

```python
# ICE: Individual lines for each sample (PDP is the average of ICE lines)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

key_features = ['MedInc', 'AveOccup', 'Latitude']
PartialDependenceDisplay.from_estimator(
    gb, X_test.iloc[:200], key_features,
    kind='both',  # Show both ICE lines and PDP (average)
    ax=axes,
    ice_lines_kw={'alpha': 0.1, 'color': 'steelblue'},
    pd_line_kw={'color': 'red', 'linewidth': 2},
    n_jobs=-1,
)

for ax, feat in zip(axes, key_features):
    ax.set_title(f'ICE + PDP: {feat}')

plt.tight_layout()
plt.savefig('ice_plots.png', dpi=150, bbox_inches='tight')
plt.show()

print("Interpretation:")
print("- Blue lines: Individual predictions (ICE)")
print("- Red line: Average effect (PDP)")
print("- Crossing ICE lines suggest feature interactions")
```

### 5.3 2D Partial Dependence

```python
# 2D PDP: Interaction between two features
fig, ax = plt.subplots(figsize=(10, 8))

PartialDependenceDisplay.from_estimator(
    gb, X_test,
    features=[('MedInc', 'AveOccup')],
    kind='average',
    ax=ax,
    n_jobs=-1,
)
ax.set_title('2D PDP: MedInc × AveOccup Interaction')
plt.tight_layout()
plt.savefig('pdp_2d.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 6. Accumulated Local Effects (ALE)

### 6.1 ALE vs PDP

```python
"""
ALE (Accumulated Local Effects) plots address a key limitation of PDPs:
PDPs assume features are independent, which is often wrong.

Example problem:
- MedInc and HouseAge are correlated
- PDP for MedInc evaluates model at (MedInc=2, HouseAge=50) even if
  that combination never occurs in real data
- ALE only uses realistic feature combinations

How ALE works:
1. Divide feature range into bins
2. For each bin, compute the average difference in prediction
   when the feature moves from the left bin edge to the right
3. Accumulate these differences → ALE curve

Advantages:
- Unbiased with correlated features
- Faster to compute than PDP
- Centered around zero (easy to interpret)

Disadvantage:
- Less intuitive than PDP for non-technical audiences
"""

# ALE implementation concept (simplified)
def compute_ale_1d(model, X, feature_idx, n_bins=20):
    """Simplified ALE computation for one feature."""
    feature_values = X.iloc[:, feature_idx]
    bins = np.percentile(feature_values, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)

    ale_values = []
    bin_centers = []

    for i in range(len(bins) - 1):
        mask = (feature_values >= bins[i]) & (feature_values < bins[i + 1])
        if i == len(bins) - 2:
            mask = mask | (feature_values == bins[i + 1])

        if mask.sum() == 0:
            continue

        X_lower = X[mask].copy()
        X_upper = X[mask].copy()
        X_lower.iloc[:, feature_idx] = bins[i]
        X_upper.iloc[:, feature_idx] = bins[i + 1]

        effect = (model.predict(X_upper) - model.predict(X_lower)).mean()
        ale_values.append(effect)
        bin_centers.append((bins[i] + bins[i + 1]) / 2)

    # Accumulate
    ale_accumulated = np.cumsum(ale_values)
    # Center
    ale_accumulated -= ale_accumulated.mean()

    return np.array(bin_centers), ale_accumulated

# Compute and plot ALE for key features
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
key_features_idx = [X_test.columns.get_loc(f) for f in ['MedInc', 'AveOccup', 'HouseAge']]

for ax, feat_idx, feat_name in zip(axes, key_features_idx, ['MedInc', 'AveOccup', 'HouseAge']):
    centers, ale_vals = compute_ale_1d(gb, X_test, feat_idx)
    ax.plot(centers, ale_vals, 'b-', linewidth=2)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(f'ALE: {feat_name}')
    ax.set_xlabel(feat_name)
    ax.set_ylabel('ALE (centered)')
    ax.fill_between(centers, ale_vals, alpha=0.2)

plt.tight_layout()
plt.savefig('ale_plots.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 7. Global Surrogate Models

### 7.1 Approximating Black-Box with Interpretable Model

```python
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.metrics import r2_score

# Train a complex model
gb.fit(X_train, y_train)
gb_predictions = gb.predict(X_train)

# Train a simple surrogate on the complex model's predictions
surrogate = DecisionTreeRegressor(max_depth=4, random_state=42)
surrogate.fit(X_train, gb_predictions)

# Measure how well the surrogate approximates the complex model
surrogate_preds = surrogate.predict(X_train)
fidelity = r2_score(gb_predictions, surrogate_preds)
print(f"Surrogate fidelity (R² on GB predictions): {fidelity:.4f}")

# The surrogate is interpretable
print("\nSurrogate Decision Rules:")
print(export_text(surrogate, feature_names=X_train.columns.tolist(), max_depth=3))

# Visualize surrogate tree
from sklearn.tree import plot_tree
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(surrogate, feature_names=X_train.columns.tolist(),
          filled=True, rounded=True, fontsize=8, ax=ax, max_depth=3)
plt.title(f'Global Surrogate Tree (Fidelity R²={fidelity:.3f})')
plt.tight_layout()
plt.savefig('surrogate_tree.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 8. Fairness and Bias Detection

### 8.1 Detecting Bias with SHAP

```python
import numpy as np
import pandas as pd

# Simulated loan approval model with potential bias
np.random.seed(42)
n = 2000

# Generate features
data = pd.DataFrame({
    'income': np.random.lognormal(10.5, 0.5, n),
    'credit_score': np.random.normal(700, 50, n).clip(300, 850),
    'debt_ratio': np.random.uniform(0.1, 0.8, n),
    'age': np.random.normal(40, 12, n).clip(18, 75),
    'gender': np.random.choice([0, 1], n),  # 0=female, 1=male
})

# Simulated (biased) approval probability
log_odds = (
    0.00002 * data['income']
    + 0.01 * data['credit_score']
    - 3 * data['debt_ratio']
    + 0.005 * data['age']
    + 0.3 * data['gender']  # Gender bias in historical data
    - 10
)
data['approved'] = (1 / (1 + np.exp(-log_odds)) > 0.5).astype(int)

from sklearn.ensemble import GradientBoostingClassifier

X = data.drop('approved', axis=1)
y = data['approved']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# SHAP analysis to detect bias
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Check gender contribution
gender_shap = pd.DataFrame({
    'gender': X_test['gender'],
    'gender_shap': shap_values[:, X_test.columns.get_loc('gender')],
})

print("Gender SHAP values by group:")
print(gender_shap.groupby('gender')['gender_shap'].describe())

# Demographic parity check
approval_by_gender = pd.DataFrame({
    'actual_gender': X_test['gender'],
    'predicted': model.predict(X_test),
})
print("\nApproval rates by gender:")
print(approval_by_gender.groupby('actual_gender')['predicted'].mean())
print("\nIf rates differ significantly, the model may perpetuate historical bias.")
```

### 8.2 Fairness Metrics

```python
"""
Common fairness metrics:

1. Demographic Parity:
   P(Ŷ=1 | A=0) = P(Ŷ=1 | A=1)
   → Equal approval rates across groups

2. Equalized Odds:
   P(Ŷ=1 | Y=1, A=0) = P(Ŷ=1 | Y=1, A=1)  (equal TPR)
   P(Ŷ=1 | Y=0, A=0) = P(Ŷ=1 | Y=0, A=1)  (equal FPR)
   → Equal error rates across groups

3. Calibration:
   P(Y=1 | Ŷ=p, A=0) = P(Y=1 | Ŷ=p, A=1)
   → Same predicted probability means same actual probability

Note: It is mathematically impossible to satisfy all fairness criteria
simultaneously (except in trivial cases). Choose based on context.
"""

def fairness_report(y_true, y_pred, sensitive_feature, group_names=None):
    """Compute basic fairness metrics."""
    groups = sorted(sensitive_feature.unique())
    if group_names is None:
        group_names = {g: f'Group {g}' for g in groups}

    results = {}
    for g in groups:
        mask = sensitive_feature == g
        y_t = y_true[mask]
        y_p = y_pred[mask]

        results[group_names[g]] = {
            'count': mask.sum(),
            'approval_rate': y_p.mean(),
            'TPR': y_p[y_t == 1].mean() if (y_t == 1).sum() > 0 else 0,
            'FPR': y_p[y_t == 0].mean() if (y_t == 0).sum() > 0 else 0,
        }

    df = pd.DataFrame(results).T
    print("Fairness Report:")
    print(df.round(4))

    # Disparate impact ratio
    rates = [r['approval_rate'] for r in results.values()]
    di_ratio = min(rates) / max(rates) if max(rates) > 0 else 0
    print(f"\nDisparate Impact Ratio: {di_ratio:.4f}")
    print(f"(Rule of thumb: ratio < 0.8 indicates potential discrimination)")
    return df

y_pred = model.predict(X_test)
fairness_report(y_test, y_pred, X_test['gender'], {0: 'Female', 1: 'Male'})
```

---

## 9. Explainability in Production

### 9.1 Generating Explanations at Inference Time

```python
class ExplainableModel:
    """Wrapper that provides predictions with explanations."""

    def __init__(self, model, X_train):
        self.model = model
        self.explainer = shap.TreeExplainer(model)
        self.feature_names = X_train.columns.tolist()
        self.base_value = self.explainer.expected_value

    def predict_with_explanation(self, X, top_k=3):
        """Return prediction and top contributing features."""
        predictions = self.model.predict(X)
        shap_values = self.explainer.shap_values(X)

        explanations = []
        for i in range(len(X)):
            # Get top features by absolute SHAP value
            feature_contributions = pd.Series(
                shap_values[i], index=self.feature_names
            ).sort_values(key=abs, ascending=False)

            top_features = []
            for feat, shap_val in feature_contributions.head(top_k).items():
                direction = "increases" if shap_val > 0 else "decreases"
                top_features.append({
                    'feature': feat,
                    'value': X.iloc[i][feat],
                    'shap_value': shap_val,
                    'direction': direction,
                })

            explanations.append({
                'prediction': predictions[i],
                'base_value': self.base_value,
                'top_features': top_features,
            })

        return explanations

# Usage
explainable = ExplainableModel(gb, X_train)
results = explainable.predict_with_explanation(X_test.iloc[:3])

for i, result in enumerate(results):
    print(f"\nSample {i}: Predicted value = {result['prediction']:.4f}")
    print(f"  Base value: {result['base_value']:.4f}")
    for feat in result['top_features']:
        print(f"  {feat['feature']} = {feat['value']:.2f} → "
              f"{feat['direction']} prediction by {abs(feat['shap_value']):.4f}")
```

### 9.2 Explanation Dashboard

```python
"""
Tools for building explanation dashboards:

1. SHAP built-in plots:
   shap.plots.waterfall() → Single prediction breakdown
   shap.plots.force() → Interactive HTML explanation
   shap.plots.bar() → Global feature importance

2. ExplainerDashboard (pip install explainerdashboard):
   from explainerdashboard import ClassifierExplainer, ExplainerDashboard
   explainer = ClassifierExplainer(model, X_test, y_test)
   dashboard = ExplainerDashboard(explainer)
   dashboard.run()

3. Custom Flask/Streamlit app:
   - Accept input features via form
   - Generate prediction + SHAP explanation
   - Display waterfall plot and feature table

4. Regulatory requirements:
   - GDPR Article 22: Right to explanation for automated decisions
   - US Equal Credit Opportunity Act: Adverse action notices
   - EU AI Act: Transparency requirements for high-risk AI systems
"""
```

---

## 10. Practice Problems

### Exercise 1: Explain a Classification Model

```python
"""
1. Load the breast cancer dataset (sklearn.datasets.load_breast_cancer)
2. Train a GradientBoostingClassifier
3. Generate SHAP values using TreeExplainer
4. Create:
   a) Summary plot (global feature importance)
   b) Waterfall plot for a correctly classified sample
   c) Waterfall plot for a misclassified sample
   d) Dependence plot for the top 2 features
5. Compare with permutation importance
"""
```

### Exercise 2: LIME vs SHAP Comparison

```python
"""
1. Train a Random Forest on California Housing
2. For 5 different test samples:
   a) Generate SHAP explanations
   b) Generate LIME explanations
   c) Compare the top 3 contributing features — do they agree?
3. Run LIME multiple times on the same sample (different random seeds)
   and measure how stable the explanations are.
4. Discuss: When would you prefer LIME over SHAP?
"""
```

### Exercise 3: Fairness Audit

```python
"""
1. Use the Adult Income dataset (sklearn.datasets.fetch_openml('adult'))
2. Train a classifier to predict income >$50K
3. Compute:
   a) SHAP values — how much does each demographic feature contribute?
   b) Demographic parity for 'sex' and 'race'
   c) Equalized odds for 'sex'
4. If bias is detected, propose and implement a mitigation strategy:
   - Remove sensitive features?
   - Resampling?
   - Threshold adjustment per group?
5. Report: fairness metrics before and after mitigation.
"""
```

---

## 11. Summary

### Key Takeaways

| Method | Type | Scope | Speed | Theory |
|--------|------|-------|-------|--------|
| Impurity Importance | Model-specific | Global | Fast | Biased by cardinality |
| Permutation Importance | Model-agnostic | Global | Medium | Affected by correlation |
| **SHAP** | Model-agnostic | Local + Global | Varies | Shapley axioms (exact) |
| **LIME** | Model-agnostic | Local | Fast | Approximate (may vary) |
| PDP | Model-agnostic | Global | Medium | Assumes independence |
| ICE | Model-agnostic | Local | Medium | Individual-level PDP |
| ALE | Model-agnostic | Global | Fast | Handles correlation |
| Surrogate | Model-agnostic | Global | Fast | Approximation |

### When to Use What

1. **Quick global overview**: Permutation importance or SHAP bar plot
2. **Explain single prediction**: SHAP waterfall or LIME
3. **Feature effect visualization**: PDP/ICE (uncorrelated features) or ALE (correlated)
4. **Fairness audit**: SHAP + demographic group analysis
5. **Production explanations**: SHAP TreeExplainer (for tree models)

### Best Practices

1. **Use multiple methods**: No single method tells the complete story
2. **Check SHAP additivity**: `base_value + Σ(SHAP) = prediction`
3. **Be cautious with correlated features**: Use ALE over PDP
4. **Test explanation stability**: LIME can vary between runs
5. **Consider the audience**: Technical stakeholders → SHAP values; Business → surrogate rules

### Next Steps

- **L17**: Imbalanced Data — handle class imbalance with sampling and cost-sensitive approaches
- **L18**: Time Series ML — apply ML techniques to temporal forecasting problems
