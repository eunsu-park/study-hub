# Handling Imbalanced Data

## Overview

Imbalanced datasets occur when one class significantly outnumbers the other(s). Standard classifiers tend to predict the majority class, resulting in misleadingly high accuracy. This lesson covers evaluation metrics, sampling techniques, cost-sensitive learning, and threshold optimization for imbalanced classification problems.

---

## 1. The Imbalance Problem

### 1.1 Why Accuracy Fails

```python
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# Simulated fraud detection: 99% legitimate, 1% fraud
np.random.seed(42)
n = 10000
y_true = np.zeros(n, dtype=int)
y_true[:100] = 1  # 1% fraud

# "Model" that always predicts legitimate (majority class)
y_pred_naive = np.zeros(n, dtype=int)

print(f"Accuracy of always predicting majority: {accuracy_score(y_true, y_pred_naive):.2%}")
print(f"But we caught 0 out of {y_true.sum()} frauds!\n")
print(classification_report(y_true, y_pred_naive, target_names=['Legitimate', 'Fraud']))
```

### 1.2 Real-World Imbalanced Problems

| Domain | Problem | Typical Ratio |
|--------|---------|--------------|
| Finance | Fraud detection | 1:1000 |
| Healthcare | Rare disease diagnosis | 1:10000 |
| Manufacturing | Defect detection | 1:100 |
| Cybersecurity | Intrusion detection | 1:1000 |
| E-commerce | Churn prediction | 1:5 to 1:20 |

---

## 2. Evaluation Metrics for Imbalanced Data

### 2.1 Beyond Accuracy

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    fbeta_score, matthews_corrcoef, cohen_kappa_score,
    precision_recall_curve, roc_curve, auc, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)
import matplotlib.pyplot as plt

# Create imbalanced dataset (5% positive)
X, y = make_classification(
    n_samples=5000, n_features=20, n_informative=10,
    weights=[0.95, 0.05], random_state=42, flip_y=0.01
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Training set class distribution: {np.bincount(y_train)}")
print(f"Test set class distribution: {np.bincount(y_test)}")

# Train a basic model
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

# Comprehensive metrics
print(f"\nAccuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1:        {f1_score(y_test, y_pred):.4f}")
print(f"F2 (recall-weighted): {fbeta_score(y_test, y_pred, beta=2):.4f}")
print(f"MCC:       {matthews_corrcoef(y_test, y_pred):.4f}")
print(f"Cohen's κ: {cohen_kappa_score(y_test, y_pred):.4f}")
print(f"AP (avg precision): {average_precision_score(y_test, y_proba):.4f}")
```

### 2.2 Metrics Comparison

```python
"""
Metric Guide for Imbalanced Data:

┌──────────────────────┬──────────────────────────────────────────┐
│ Metric               │ When to Use                              │
├──────────────────────┼──────────────────────────────────────────┤
│ Precision            │ Cost of false positive is high           │
│                      │ (e.g., spam filter blocking legit email) │
├──────────────────────┼──────────────────────────────────────────┤
│ Recall (Sensitivity) │ Cost of false negative is high           │
│                      │ (e.g., missing cancer diagnosis)         │
├──────────────────────┼──────────────────────────────────────────┤
│ F1 Score             │ Balance between precision and recall     │
├──────────────────────┼──────────────────────────────────────────┤
│ F2 Score             │ Recall is more important than precision  │
│ (β=2)               │ (e.g., fraud detection)                  │
├──────────────────────┼──────────────────────────────────────────┤
│ PR-AUC               │ Threshold-independent, imbalance-aware   │
│                      │ Better than ROC-AUC for skewed datasets  │
├──────────────────────┼──────────────────────────────────────────┤
│ MCC                  │ Single metric for binary classification  │
│                      │ Balanced even with imbalance (-1 to +1)  │
├──────────────────────┼──────────────────────────────────────────┤
│ Cohen's κ            │ Agreement beyond chance                  │
└──────────────────────┴──────────────────────────────────────────┘
"""
```

### 2.3 PR Curve vs ROC Curve

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC Curve
fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
axes[0].plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC AUC = {roc_auc:.3f}')
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve')
axes[0].legend()

# Precision-Recall Curve
precision, recall, pr_thresholds = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recall, precision)
baseline = y_test.mean()
axes[1].plot(recall, precision, 'r-', linewidth=2, label=f'PR AUC = {pr_auc:.3f}')
axes[1].axhline(baseline, color='k', linestyle='--', alpha=0.5, label=f'Random = {baseline:.3f}')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve')
axes[1].legend()

plt.tight_layout()
plt.savefig('roc_vs_pr.png', dpi=150)
plt.show()

print("Key insight: With imbalanced data, PR curves are more informative.")
print(f"ROC AUC ({roc_auc:.3f}) can look good even when PR AUC ({pr_auc:.3f}) reveals poor minority-class performance.")
```

---

## 3. Sampling Techniques

### 3.1 Random Oversampling and Undersampling

```python
from collections import Counter

# pip install imbalanced-learn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

print(f"Original: {Counter(y_train)}")

# Oversampling: duplicate minority samples
ros = RandomOverSampler(random_state=42)
X_over, y_over = ros.fit_resample(X_train, y_train)
print(f"After oversampling: {Counter(y_over)}")

# Undersampling: remove majority samples
rus = RandomUnderSampler(random_state=42)
X_under, y_under = rus.fit_resample(X_train, y_train)
print(f"After undersampling: {Counter(y_under)}")
```

### 3.2 SMOTE (Synthetic Minority Over-sampling Technique)

```python
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE

# SMOTE: Create synthetic samples by interpolating between minority neighbors
smote = SMOTE(random_state=42, k_neighbors=5)
X_smote, y_smote = smote.fit_resample(X_train, y_train)
print(f"After SMOTE: {Counter(y_smote)}")

# Borderline-SMOTE: Focus on samples near the decision boundary
bl_smote = BorderlineSMOTE(random_state=42, kind='borderline-1')
X_bl, y_bl = bl_smote.fit_resample(X_train, y_train)
print(f"After Borderline-SMOTE: {Counter(y_bl)}")

# ADASYN: Adaptive — generate more samples where density is lower
adasyn = ADASYN(random_state=42)
X_adasyn, y_adasyn = adasyn.fit_resample(X_train, y_train)
print(f"After ADASYN: {Counter(y_adasyn)}")
```

```python
"""
How SMOTE works:
1. Pick a random minority sample x
2. Find its k nearest minority neighbors
3. Pick one neighbor x_nn
4. Create synthetic sample: x_new = x + λ * (x_nn - x)
   where λ ~ Uniform(0, 1)

Variants:
- SMOTE: Basic interpolation between random pairs
- Borderline-SMOTE: Only oversample near decision boundary
- ADASYN: More synthetic samples where minority density is lower
- SMOTE-ENN: SMOTE + Edited Nearest Neighbors (clean noisy samples)
- SMOTE-Tomek: SMOTE + remove Tomek links (borderline samples)
"""
```

### 3.3 Combined Sampling (SMOTE + Cleaning)

```python
from imblearn.combine import SMOTEENN, SMOTETomek

# SMOTE-ENN: Oversample then clean with Edited Nearest Neighbors
smote_enn = SMOTEENN(random_state=42)
X_se, y_se = smote_enn.fit_resample(X_train, y_train)
print(f"SMOTE-ENN: {Counter(y_se)}")

# SMOTE-Tomek: Oversample then remove Tomek links
smote_tomek = SMOTETomek(random_state=42)
X_st, y_st = smote_tomek.fit_resample(X_train, y_train)
print(f"SMOTE-Tomek: {Counter(y_st)}")
```

### 3.4 Comparing Sampling Methods

```python
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from imblearn.pipeline import Pipeline as ImbPipeline

# Use imbalanced-learn's Pipeline (supports samplers)
strategies = {
    'No Sampling': None,
    'Random Over': RandomOverSampler(random_state=42),
    'Random Under': RandomUnderSampler(random_state=42),
    'SMOTE': SMOTE(random_state=42),
    'Borderline-SMOTE': BorderlineSMOTE(random_state=42),
    'SMOTE-ENN': SMOTEENN(random_state=42),
}

results = {}
f1_scorer = make_scorer(f1_score)

for name, sampler in strategies.items():
    if sampler is None:
        pipeline = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        pipeline = ImbPipeline([
            ('sampler', sampler),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ])

    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring=f1_scorer)
    results[name] = {'mean_f1': scores.mean(), 'std_f1': scores.std()}

results_df = pd.DataFrame(results).T.sort_values('mean_f1', ascending=False)
print("Sampling Strategy Comparison (CV F1 Score):")
print(results_df.round(4))

fig, ax = plt.subplots(figsize=(10, 5))
results_df['mean_f1'].plot(kind='barh', xerr=results_df['std_f1'], ax=ax, color='steelblue')
ax.set_xlabel('F1 Score')
ax.set_title('Sampling Strategy Comparison')
plt.tight_layout()
plt.savefig('sampling_comparison.png', dpi=150)
plt.show()
```

---

## 4. Cost-Sensitive Learning

### 4.1 Class Weights

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Method 1: class_weight parameter
rf_balanced = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',  # Automatically adjusts weights inversely proportional to class frequency
    random_state=42,
    n_jobs=-1,
)
rf_balanced.fit(X_train, y_train)
y_pred_balanced = rf_balanced.predict(X_test)

print("Without class_weight:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
print("\nWith class_weight='balanced':")
print(classification_report(y_test, y_pred_balanced, target_names=['Negative', 'Positive']))
```

```python
# Method 2: Custom class weights
# Higher weight for minority class = higher penalty for misclassifying it
custom_weights = {0: 1, 1: 20}  # 20x penalty for misclassifying positive

rf_custom = RandomForestClassifier(
    n_estimators=100,
    class_weight=custom_weights,
    random_state=42,
    n_jobs=-1,
)
rf_custom.fit(X_train, y_train)
y_pred_custom = rf_custom.predict(X_test)

print(f"Custom weights {custom_weights}:")
print(classification_report(y_test, y_pred_custom, target_names=['Negative', 'Positive']))
```

### 4.2 Sample Weights

```python
# sample_weight: assign different importance to each training sample
sample_weights = np.ones(len(y_train))
minority_mask = y_train == 1
majority_mask = y_train == 0
imbalance_ratio = majority_mask.sum() / minority_mask.sum()
sample_weights[minority_mask] = imbalance_ratio

print(f"Imbalance ratio: {imbalance_ratio:.1f}")
print(f"Majority sample weight: 1.0")
print(f"Minority sample weight: {imbalance_ratio:.1f}")

# GradientBoosting supports sample_weight in fit()
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, y_train, sample_weight=sample_weights)
y_pred_weighted = gb.predict(X_test)

print(f"\nWith sample weights:")
print(classification_report(y_test, y_pred_weighted, target_names=['Negative', 'Positive']))
```

### 4.3 XGBoost scale_pos_weight

```python
"""
XGBoost has a dedicated parameter for imbalanced data:
  scale_pos_weight = count(negative) / count(positive)

import xgboost as xgb

ratio = (y_train == 0).sum() / (y_train == 1).sum()
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    scale_pos_weight=ratio,
    eval_metric='aucpr',  # Use PR-AUC as metric (not accuracy)
    random_state=42,
)
xgb_model.fit(X_train, y_train)

# LightGBM equivalent:
import lightgbm as lgb
lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    is_unbalance=True,  # or set scale_pos_weight
    random_state=42,
)
"""
```

---

## 5. Threshold Optimization

### 5.1 Moving the Decision Threshold

```python
# Default threshold is 0.5, but for imbalanced data this is often suboptimal
y_proba = rf.predict_proba(X_test)[:, 1]

# Evaluate at different thresholds
thresholds = np.arange(0.05, 0.95, 0.05)
results = []

for threshold in thresholds:
    y_pred_t = (y_proba >= threshold).astype(int)
    results.append({
        'threshold': threshold,
        'precision': precision_score(y_test, y_pred_t, zero_division=0),
        'recall': recall_score(y_test, y_pred_t),
        'f1': f1_score(y_test, y_pred_t, zero_division=0),
        'f2': fbeta_score(y_test, y_pred_t, beta=2, zero_division=0),
    })

threshold_df = pd.DataFrame(results)

fig, ax = plt.subplots(figsize=(10, 6))
for metric in ['precision', 'recall', 'f1', 'f2']:
    ax.plot(threshold_df['threshold'], threshold_df[metric], label=metric, linewidth=2)

optimal_f1 = threshold_df.loc[threshold_df['f1'].idxmax()]
ax.axvline(optimal_f1['threshold'], color='red', linestyle='--',
           label=f"Optimal F1 threshold = {optimal_f1['threshold']:.2f}")
ax.axvline(0.5, color='gray', linestyle=':', alpha=0.5, label='Default threshold = 0.5')

ax.set_xlabel('Classification Threshold')
ax.set_ylabel('Score')
ax.set_title('Metrics vs. Classification Threshold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('threshold_optimization.png', dpi=150)
plt.show()

print(f"Default threshold (0.5): F1 = {f1_score(y_test, rf.predict(X_test)):.4f}")
print(f"Optimal threshold ({optimal_f1['threshold']:.2f}): F1 = {optimal_f1['f1']:.4f}")
```

### 5.2 Cost-Based Threshold Selection

```python
def find_optimal_threshold(y_true, y_proba, fp_cost=1, fn_cost=10):
    """
    Find the threshold that minimizes total cost.
    fp_cost: Cost of false positive (e.g., wasted investigation)
    fn_cost: Cost of false negative (e.g., missed fraud)
    """
    thresholds = np.arange(0.01, 1.0, 0.01)
    costs = []

    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred_t)
        tn, fp, fn, tp = cm.ravel()
        total_cost = fp * fp_cost + fn * fn_cost
        costs.append(total_cost)

    optimal_idx = np.argmin(costs)
    optimal_threshold = thresholds[optimal_idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, costs, 'b-', linewidth=2)
    ax.axvline(optimal_threshold, color='red', linestyle='--',
               label=f'Optimal = {optimal_threshold:.2f} (cost={costs[optimal_idx]:.0f})')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Total Cost')
    ax.set_title(f'Cost-Based Threshold (FP cost={fp_cost}, FN cost={fn_cost})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return optimal_threshold

# Fraud scenario: missing fraud (FN) costs 10x more than false alarm (FP)
optimal_t = find_optimal_threshold(y_test, y_proba, fp_cost=1, fn_cost=10)
print(f"Cost-optimal threshold: {optimal_t:.2f}")

y_pred_optimal = (y_proba >= optimal_t).astype(int)
print(classification_report(y_test, y_pred_optimal, target_names=['Negative', 'Positive']))
```

---

## 6. Algorithm-Level Approaches

### 6.1 Balanced Ensemble Methods

```python
from imblearn.ensemble import (
    BalancedRandomForestClassifier,
    BalancedBaggingClassifier,
    EasyEnsembleClassifier,
    RUSBoostClassifier,
)

ensemble_methods = {
    'RandomForest (default)': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'RandomForest (balanced)': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
    'BalancedRandomForest': BalancedRandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'BalancedBagging': BalancedBaggingClassifier(n_estimators=50, random_state=42, n_jobs=-1),
    'EasyEnsemble': EasyEnsembleClassifier(n_estimators=10, random_state=42, n_jobs=-1),
}

results = {}
for name, model in ensemble_methods.items():
    model.fit(X_train, y_train)
    y_pred_m = model.predict(X_test)
    y_proba_m = model.predict_proba(X_test)[:, 1]

    results[name] = {
        'F1': f1_score(y_test, y_pred_m),
        'Precision': precision_score(y_test, y_pred_m),
        'Recall': recall_score(y_test, y_pred_m),
        'PR-AUC': average_precision_score(y_test, y_proba_m),
        'MCC': matthews_corrcoef(y_test, y_pred_m),
    }

results_df = pd.DataFrame(results).T
print("Ensemble Method Comparison:")
print(results_df.round(4))

fig, ax = plt.subplots(figsize=(10, 5))
results_df[['F1', 'Precision', 'Recall']].plot(kind='barh', ax=ax)
ax.set_title('Balanced Ensemble Methods Comparison')
ax.set_xlabel('Score')
plt.tight_layout()
plt.savefig('ensemble_comparison.png', dpi=150)
plt.show()
```

### 6.2 When to Reframe as Anomaly Detection

```python
"""
When class imbalance is extreme (>1:100), consider anomaly detection instead:

Classification approach:
  - Needs labeled data for both classes
  - Requires balanced evaluation
  - Models: SMOTE + RF, Cost-sensitive GBM

Anomaly detection approach:
  - Only needs normal class data for training
  - Treats minority as "anomalous"
  - Models: Isolation Forest, One-Class SVM, Autoencoders

Rule of thumb:
  - Imbalance 1:2 to 1:20  → Sampling + cost-sensitive classification
  - Imbalance 1:20 to 1:100 → Both approaches viable, compare
  - Imbalance > 1:100        → Consider anomaly detection
  - No labeled minority data  → Must use anomaly detection

See L20 (Anomaly Detection) for details.
"""
```

---

## 7. Complete Pipeline

### 7.1 End-to-End Imbalanced Classification

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

# Build a complete pipeline
pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('sampler', SMOTE(random_state=42)),
    ('classifier', GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
    )),
])

# Stratified K-Fold (maintains class ratio in each fold)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Multi-metric evaluation
scoring = {
    'f1': 'f1',
    'precision': 'precision',
    'recall': 'recall',
    'pr_auc': 'average_precision',
    'roc_auc': 'roc_auc',
}

cv_results = cross_validate(
    pipeline, X_train, y_train,
    cv=cv, scoring=scoring,
    return_train_score=False,
    n_jobs=-1,
)

print("Cross-Validation Results:")
for metric in scoring:
    scores = cv_results[f'test_{metric}']
    print(f"  {metric:12s}: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### 7.2 Hyperparameter Tuning for Imbalanced Data

```python
from sklearn.model_selection import GridSearchCV

# Tune both the sampler and classifier
param_grid = {
    'sampler__k_neighbors': [3, 5, 7],
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [3, 5],
    'classifier__learning_rate': [0.05, 0.1],
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring='f1',  # Optimize for F1, not accuracy!
    n_jobs=-1,
    verbose=0,
)
grid_search.fit(X_train, y_train)

print(f"Best F1: {grid_search.best_score_:.4f}")
print(f"Best params: {grid_search.best_params_}")

# Evaluate on test set
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
y_proba_best = best_model.predict_proba(X_test)[:, 1]

print(f"\nTest set performance:")
print(classification_report(y_test, y_pred_best, target_names=['Negative', 'Positive']))

# Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_best,
                                         display_labels=['Negative', 'Positive'],
                                         cmap='Blues', ax=ax)
ax.set_title('Confusion Matrix (Best Model)')
plt.tight_layout()
plt.savefig('confusion_matrix_best.png', dpi=150)
plt.show()
```

---

## 8. Strategy Selection Guide

### 8.1 Decision Framework

```python
"""
Choosing an imbalanced data strategy:

                    ┌── Small dataset (<1000)?
                    │   └── Yes → Random Oversampling or SMOTE
                    │   └── No ─┐
                    │            │
                    │   ┌── Extreme imbalance (>1:100)?
                    │   │   └── Yes → Anomaly Detection (L20)
                    │   │   └── No ─┐
                    │   │            │
                    │   │   ┌── Need probability outputs?
                    │   │   │   └── Yes → Cost-sensitive + Calibration
                    │   │   │   └── No ─┐
                    │   │   │            │
                    │   │   │   ┌── Tree-based model?
                    │   │   │   │   └── Yes → class_weight='balanced'
                    │   │   │   │              or BalancedRandomForest
                    │   │   │   │   └── No → SMOTE + Pipeline
                    │   │   │   │
"""
```

### 8.2 Quick Reference

| Strategy | Pros | Cons | Best When |
|----------|------|------|-----------|
| **Random Oversampling** | Simple, no info loss | Overfitting risk | Small datasets |
| **Random Undersampling** | Fast, reduces training time | Loses majority info | Large datasets |
| **SMOTE** | Creates diverse samples | May generate noise | Moderate imbalance |
| **Borderline-SMOTE** | Focuses on boundary | Sensitive to k | Complex boundaries |
| **SMOTE-ENN** | Clean + oversample | Slow | Noisy data |
| **Class Weights** | No data change needed | Model-dependent | Quick baseline |
| **Threshold Tuning** | Post-hoc adjustment | Needs probabilities | Production tuning |
| **Balanced Ensembles** | Built-in balancing | Slower training | General purpose |

---

## 9. Practice Problems

### Exercise 1: Credit Card Fraud Detection

```python
"""
1. Load the credit card fraud dataset:
   from sklearn.datasets import fetch_openml
   cc = fetch_openml('creditcard', version=1, as_frame=True, parser='auto')

2. Explore the class distribution.
3. Implement and compare at least 4 strategies:
   a) Baseline Random Forest (no handling)
   b) SMOTE + Random Forest
   c) Random Forest with class_weight='balanced'
   d) BalancedRandomForestClassifier
4. Evaluate using: F1, PR-AUC, and MCC (not accuracy!)
5. Find the optimal threshold for the best model using cost-based optimization
   (assume FN costs 100x FP).
6. Report final confusion matrix.
"""
```

### Exercise 2: Multi-Class Imbalance

```python
"""
1. Create a multi-class imbalanced dataset:
   X, y = make_classification(n_samples=5000, n_classes=4,
                              n_informative=10, weights=[0.60, 0.25, 0.10, 0.05])
2. Implement SMOTE for multi-class (imblearn handles this automatically).
3. Use macro-averaged F1 and per-class classification report.
4. Compare: no sampling vs SMOTE vs class_weight='balanced'.
5. Which class benefits most from balancing techniques?
"""
```

### Exercise 3: Pipeline Design Challenge

```python
"""
Design a complete pipeline for medical diagnosis (rare disease):
1. Data: 10000 patients, 0.5% positive (disease present)
2. Requirements:
   - Recall >= 0.95 (cannot miss sick patients)
   - Precision >= 0.10 (acceptable false alarm rate)
3. Steps:
   a) Choose appropriate sampling strategy
   b) Choose classifier with cost-sensitive support
   c) Optimize threshold for the recall constraint
   d) Report all metrics including confusion matrix
   e) Estimate the real-world impact:
      - How many patients correctly identified?
      - How many false alarms?
      - What is the cost saving vs. no screening?
"""
```

---

## 10. Summary

### Key Takeaways

| Concept | Description |
|---------|-------------|
| **Accuracy is misleading** | Always use F1, PR-AUC, or MCC for imbalanced data |
| **PR-AUC > ROC-AUC** | PR curves are more informative with class imbalance |
| **SMOTE** | Generates synthetic minority samples via interpolation |
| **Class weights** | Penalize minority misclassification without changing data |
| **Threshold tuning** | Adjust decision boundary based on cost analysis |
| **Balanced ensembles** | Built-in sampling within ensemble learning |
| **imblearn Pipeline** | Ensures sampling only affects training data (no leakage) |

### Best Practices

1. **Never evaluate with accuracy alone** on imbalanced datasets
2. **Use stratified splits** (StratifiedKFold) to maintain class ratio in each fold
3. **Apply sampling inside cross-validation** using imblearn Pipeline (not before split!)
4. **Tune threshold** after model training — it's cheap and effective
5. **Try class_weight first** — it requires no additional library and often works well
6. **Combine strategies**: SMOTE + cost-sensitive + threshold tuning

### Next Steps

- **L18**: Time Series ML — apply ML techniques to temporal forecasting problems
- **L19**: AutoML — let algorithms search for the best model and hyperparameters
