# 불균형 데이터 처리(Handling Imbalanced Data)

**이전**: [모델 설명 가능성](./16_Model_Explainability.md) | **다음**: [시계열 머신러닝](./18_Time_Series_ML.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 불균형 데이터셋에서 정확도(Accuracy)가 오해를 유발하는 이유를 설명하고 적절한 지표(F1, PR-AUC, MCC)를 선택할 수 있습니다
2. 정밀도-재현율 곡선(Precision-Recall Curve)과 ROC 곡선을 비교하고 PR-AUC가 더 유익한 정보를 제공하는 경우를 파악할 수 있습니다
3. SMOTE(Synthetic Minority Over-sampling Technique)와 그 변형(Borderline-SMOTE, SMOTE-ENN)을 구현하여 합성 과샘플링을 수행할 수 있습니다
4. 트리 기반 모델에서 클래스 가중치(Class Weights)와 샘플 가중치(Sample Weights)를 사용한 비용 민감 학습(Cost-Sensitive Learning)을 적용할 수 있습니다
5. 비용 기반 분석을 이용한 분류 임계값(Classification Threshold) 최적화를 수행할 수 있습니다
6. 데이터 누수(Data Leakage)를 방지하는 imblearn 파이프라인으로 완전한 불균형 분류 파이프라인을 구축할 수 있습니다
7. 극단적인 불균형 문제를 이상 탐지(Anomaly Detection)로 재구성해야 하는 시점을 결정할 수 있습니다

---

99.9% 정확도를 달성하는 사기 탐지 모델은 인상적으로 들립니다 -- 모든 거래를 합법적인 것으로 분류하고 사기를 단 하나도 잡아내지 못한다는 사실을 깨닫기 전까지는 말이죠. 불균형 데이터는 실제 머신러닝에서 일반적인 현상입니다: 사기, 희귀 질환, 제조 결함, 침입 탐지는 모두 소수 클래스가 가장 중요한 '건초 더미 속 바늘 찾기' 문제입니다. 이 레슨에서는 실제로 중요한 희귀 사건을 포착하는 모델을 만들기 위해 평가, 샘플링, 재가중치 부여, 임계값 튜닝을 다루는 방법을 가르칩니다.

---

## 1. 불균형 문제(The Imbalance Problem)

### 1.1 정확도가 실패하는 이유

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

### 1.2 실제 불균형 문제 사례

| 도메인 | 문제 | 일반적인 비율 |
|--------|---------|--------------|
| 금융 | 사기 탐지(Fraud detection) | 1:1000 |
| 의료 | 희귀 질환 진단 | 1:10000 |
| 제조 | 불량 탐지(Defect detection) | 1:100 |
| 사이버 보안 | 침입 탐지(Intrusion detection) | 1:1000 |
| 전자상거래 | 이탈 예측(Churn prediction) | 1:5 ~ 1:20 |

---

## 2. 불균형 데이터를 위한 평가 지표

### 2.1 정확도를 넘어서

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

### 2.2 지표 비교

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

### 2.3 PR 곡선 대 ROC 곡선

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

## 3. 샘플링 기법(Sampling Techniques)

### 3.1 무작위 과샘플링과 과소샘플링

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

### 3.2 SMOTE (합성 소수 과샘플링 기법, Synthetic Minority Over-sampling Technique)

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

### 3.3 결합 샘플링 (SMOTE + 정제)

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

### 3.4 샘플링 방법 비교

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

## 4. 비용 민감 학습(Cost-Sensitive Learning)

### 4.1 클래스 가중치(Class Weights)

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

### 4.2 샘플 가중치(Sample Weights)

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

## 5. 임계값 최적화(Threshold Optimization)

### 5.1 결정 임계값 조정

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

### 5.2 비용 기반 임계값 선택

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

## 6. 알고리즘 수준의 접근법

### 6.1 균형 앙상블 방법(Balanced Ensemble Methods)

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

### 6.2 이상 탐지(Anomaly Detection)로 재구성할 시점

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

## 7. 완전한 파이프라인(Complete Pipeline)

### 7.1 불균형 분류 종단간 처리

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

### 7.2 불균형 데이터를 위한 하이퍼파라미터 튜닝

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

## 8. 전략 선택 가이드

### 8.1 의사결정 프레임워크

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

### 8.2 빠른 참조표

| 전략 | 장점 | 단점 | 최적 사용 시점 |
|----------|------|------|-----------|
| **무작위 과샘플링(Random Oversampling)** | 단순, 정보 손실 없음 | 과적합(Overfitting) 위험 | 소규모 데이터셋 |
| **무작위 과소샘플링(Random Undersampling)** | 빠름, 학습 시간 단축 | 다수 클래스 정보 손실 | 대규모 데이터셋 |
| **SMOTE** | 다양한 샘플 생성 | 노이즈 생성 가능 | 중간 수준의 불균형 |
| **Borderline-SMOTE** | 결정 경계에 집중 | k에 민감 | 복잡한 결정 경계 |
| **SMOTE-ENN** | 정제 + 과샘플링 | 느림 | 노이즈가 많은 데이터 |
| **클래스 가중치(Class Weights)** | 데이터 변경 불필요 | 모델 의존적 | 빠른 기준선 설정 |
| **임계값 튜닝(Threshold Tuning)** | 사후 조정 가능 | 확률값 필요 | 프로덕션 튜닝 |
| **균형 앙상블(Balanced Ensembles)** | 내장된 균형 처리 | 학습 속도 저하 | 범용 목적 |

---

## 9. 연습 문제

### 연습 1: 신용카드 사기 탐지

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

### 연습 2: 다중 클래스 불균형

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

### 연습 3: 파이프라인 설계 도전

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

## 10. 요약

### 핵심 정리

| 개념 | 설명 |
|---------|-------------|
| **정확도는 오해를 유발함** | 불균형 데이터에서는 항상 F1, PR-AUC, 또는 MCC를 사용 |
| **PR-AUC > ROC-AUC** | 클래스 불균형 시 PR 곡선이 더 유익한 정보 제공 |
| **SMOTE** | 보간(Interpolation)을 통해 소수 클래스의 합성 샘플 생성 |
| **클래스 가중치(Class weights)** | 데이터를 변경하지 않고 소수 클래스 오분류에 패널티 부여 |
| **임계값 튜닝(Threshold tuning)** | 비용 분석에 기반하여 결정 경계 조정 |
| **균형 앙상블(Balanced ensembles)** | 앙상블 학습 내에 내장된 샘플링 |
| **imblearn Pipeline** | 샘플링이 훈련 데이터에만 적용되도록 보장(데이터 누수 방지) |

### 모범 사례

1. **불균형 데이터셋에서는 정확도만으로 절대 평가하지 않기**
2. **계층화 분할(Stratified splits) 사용** (StratifiedKFold)으로 각 폴드의 클래스 비율 유지
3. **교차 검증 내부에서 샘플링 적용** — imblearn Pipeline 사용 (분할 전에 적용하지 말 것!)
4. **모델 학습 후 임계값 튜닝** — 저비용이며 효과적
5. **먼저 클래스 가중치 시도** — 추가 라이브러리 없이도 잘 작동하는 경우가 많음
6. **전략 결합**: SMOTE + 비용 민감 + 임계값 튜닝

### 다음 단계

- **L18**: 시계열 머신러닝(Time Series ML) — 시간적 예측 문제에 ML 기법 적용
- **L19**: AutoML — 알고리즘이 최적 모델과 하이퍼파라미터를 자동으로 탐색
