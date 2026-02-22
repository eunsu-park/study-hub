# 모델 설명 가능성(Model Explainability)

## 개요

모델 설명 가능성(Model Explainability)은 모델이 특정 예측을 내리는 이유를 이해할 수 있도록 도와줍니다. ML 모델이 고위험 분야(의료, 금융, 형사 사법)에 배포됨에 따라, 예측을 설명할 수 있는 능력은 단순히 있으면 좋은 것이 아니라 — 종종 법적, 윤리적 요건이 됩니다.

---

## 1. 해석 가능성(Interpretability) vs. 설명 가능성(Explainability)

### 1.1 핵심 개념

```python
"""
해석 가능성(Interpretability): 인간이 모델의 메커니즘을 얼마나 쉽게 이해할 수 있는가.
  - 선형 회귀: 계수 = 직접적 효과 → 높은 해석 가능성
  - 심층 신경망: 수백만 개의 파라미터 → 낮은 해석 가능성

설명 가능성(Explainability): 사후에 모델의 예측을 설명할 수 있는 능력.
  - SHAP, LIME: 사후 설명(post-hoc explanation) 방법
  - 어떤 블랙박스 모델에도 적용 가능

분류 체계:
┌──────────────────────┬───────────────────────┐
│     내재적(Intrinsic) │      사후(Post-hoc)    │
│  (내장된)             │  (학습 후)             │
├──────────────────────┼───────────────────────┤
│ 선형 회귀             │ SHAP                  │
│ 결정 트리             │ LIME                  │
│ 규칙 기반 모델        │ PDP / ICE / ALE       │
│ GAMs                 │ 순열 중요도            │
│ (소규모) 로지스틱 회귀 │ 반사실적 설명          │
└──────────────────────┴───────────────────────┘

범위:
- 전역(Global): 전체 모델 동작 이해
- 지역(Local): 단일 예측 이해
"""
```

### 1.2 정확도-해석 가능성 상충 관계(Accuracy-Interpretability Trade-off)

| 모델 | 해석 가능성 | 일반적인 정확도 |
|------|------------|----------------|
| 선형/로지스틱 회귀(Linear / Logistic Regression) | 매우 높음 | 낮음-중간 |
| 결정 트리(Decision Tree, 얕은) | 높음 | 낮음-중간 |
| 랜덤 포레스트(Random Forest) | 중간 | 높음 |
| 그래디언트 부스팅(Gradient Boosting, XGBoost) | 낮음 | 매우 높음 |
| 신경망(Neural Networks) | 매우 낮음 | 매우 높음 |

**중요**: 이 상충 관계는 경향이지, 규칙이 아닙니다. 적절한 피처 엔지니어링으로 단순 모델이 복잡한 모델에 필적할 수 있습니다. 또한 SHAP/LIME으로 복잡한 모델을 설명할 수 있습니다.

---

## 2. 피처 중요도(Feature Importance) 방법

### 2.1 불순도 기반 중요도(Impurity-Based Importance, 트리)

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

**한계**: 불순도 기반 중요도는 고카디널리티 피처에 편향되며, 피처 상호작용을 제대로 반영하지 못합니다.

### 2.2 순열 중요도(Permutation Importance)

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

**불순도 기반 대비 장점**:
- 모든 모델에 적용 가능 (트리 전용이 아님)
- 테스트 세트에서 계산 (일반화 반영)
- 피처 카디널리티에 편향되지 않음

**한계**: 상관된 피처끼리 중요도를 나눌 수 있음 — 상관된 파트너가 남아 있으면 하나를 섞어도 성능이 크게 하락하지 않음.

---

## 3. SHAP (SHapley Additive exPlanations)

### 3.1 샤플리 값(Shapley Values) 이론

```python
"""
샤플리 값(Shapley values)은 협력 게임 이론(cooperative game theory)에서 유래합니다 (Lloyd Shapley, 1953).

아이디어: 각 플레이어(피처)가 보상(예측)에 얼마나 기여하는가?

특징값 {x1, x2, ..., xp}를 가진 예측 f(x)에 대해:
  φᵢ = Σ_{S ⊆ N\{i}} |S|!(p-|S|-1)! / p! × [f(S ∪ {i}) - f(S)]

여기서:
  - S는 피처의 부분집합(연합, coalition)
  - f(S)는 S에 있는 피처만 사용한 모델 예측
  - φᵢ는 피처 i의 샤플리 값

속성 (공리):
1. 효율성(Efficiency):   Σ φᵢ = f(x) - E[f(x)]   (값의 합 = 예측 - 평균)
2. 대칭성(Symmetry):     동등한 기여자는 동등한 값을 받음
3. 더미(Dummy):          기여하지 않는 피처는 0을 받음
4. 가산성(Additivity):   결합 모델의 값 = 개별 값의 합

SHAP는 샤플리 값을 ML에 연결:
  f(x) = base_value + Σ φᵢ(x)
  where base_value = E[f(x)] = 평균 예측값
"""
```

### 3.2 SHAP 실전 활용

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

### 3.3 SHAP 시각화

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

### 3.4 SHAP 변형(Variants)

```python
"""
SHAP는 모델 유형별로 최적화된 알고리즘을 제공합니다:

┌─────────────────────┬───────────────────┬────────────────────┐
│ 설명자(Explainer)    │ 모델              │ 속도               │
├─────────────────────┼───────────────────┼────────────────────┤
│ TreeExplainer       │ XGBoost, LightGBM │ 매우 빠름 (정확)   │
│                     │ Random Forest     │                    │
├─────────────────────┼───────────────────┼────────────────────┤
│ LinearExplainer     │ 선형 모델         │ 매우 빠름 (정확)   │
├─────────────────────┼───────────────────┼────────────────────┤
│ KernelExplainer     │ 임의 모델         │ 느림 (근사)        │
├─────────────────────┼───────────────────┼────────────────────┤
│ DeepExplainer       │ 신경망            │ 빠름 (근사)        │
├─────────────────────┼───────────────────┼────────────────────┤
│ GradientExplainer   │ 신경망            │ 빠름 (근사)        │
└─────────────────────┴───────────────────┴────────────────────┘

선택 기준:
- 트리 모델 → TreeExplainer (항상)
- 선형 모델 → LinearExplainer
- 다른 모든 모델 → KernelExplainer (느리지만 범용)
- 딥러닝 → DeepExplainer 또는 GradientExplainer
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

### 3.5 SHAP 상호작용 값(Interaction Values)

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

### 4.1 LIME의 작동 원리

```python
"""
LIME은 예측 주변 이웃 영역에 단순하고 해석 가능한 모델
(일반적으로 선형 회귀)을 적합시켜 개별 예측을 설명합니다.

알고리즘:
1. 설명할 샘플 선택
2. 주변에 교란된(perturbed) 샘플 생성 (노이즈 추가)
3. 교란된 샘플에 대한 블랙박스 모델 예측 획득
4. 원본과의 근접도로 교란된 샘플에 가중치 부여
5. 가중된 샘플에 단순 모델(선형 회귀) 적합
6. 단순 모델 계수를 설명으로 사용

SHAP와의 주요 차이점:
- LIME: 지역 선형 근사 (근사)
- SHAP: 정확한 샤플리 값 (이론적 근거 있음)
- LIME: 빠르지만 일관성이 낮음
- SHAP: 느리지만 고유성 공리 만족
"""
```

### 4.2 표 형식 데이터에 대한 LIME

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

### 4.3 SHAP vs LIME 비교

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

## 5. 부분 의존도 플롯(PDP)과 ICE

### 5.1 부분 의존도 플롯(Partial Dependence Plots)

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

### 5.2 개별 조건부 기댓값(Individual Conditional Expectation, ICE)

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

### 5.3 2D 부분 의존도

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

## 6. 누적 지역 효과(ALE, Accumulated Local Effects)

### 6.1 ALE vs PDP

```python
"""
ALE(Accumulated Local Effects) 플롯은 PDP의 핵심 한계를 해결합니다:
PDP는 피처가 독립적이라고 가정하는데, 이는 종종 틀린 가정입니다.

문제 예시:
- MedInc와 HouseAge는 상관 관계가 있음
- MedInc에 대한 PDP는 실제 데이터에 존재하지 않는
  (MedInc=2, HouseAge=50) 조합에서도 모델을 평가함
- ALE는 현실적인 피처 조합만 사용

ALE 작동 방식:
1. 피처 범위를 구간으로 나눔
2. 각 구간에서, 피처가 왼쪽 구간 경계에서 오른쪽으로
   이동할 때의 평균 예측 차이를 계산
3. 이 차이들을 누적 → ALE 곡선

장점:
- 상관된 피처에 대해 편향 없음
- PDP보다 빠르게 계산
- 0을 중심으로 정렬 (해석 용이)

단점:
- 비기술적 청중에게 PDP보다 덜 직관적
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

## 7. 전역 대리 모델(Global Surrogate Models)

### 7.1 블랙박스를 해석 가능한 모델로 근사

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

## 8. 공정성(Fairness) 및 편향(Bias) 탐지

### 8.1 SHAP으로 편향 탐지

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

### 8.2 공정성 지표(Fairness Metrics)

```python
"""
일반적인 공정성 지표:

1. 인구 통계적 동등성(Demographic Parity):
   P(Ŷ=1 | A=0) = P(Ŷ=1 | A=1)
   → 집단 간 동등한 승인율

2. 균등화된 오즈(Equalized Odds):
   P(Ŷ=1 | Y=1, A=0) = P(Ŷ=1 | Y=1, A=1)  (동등한 TPR)
   P(Ŷ=1 | Y=0, A=0) = P(Ŷ=1 | Y=0, A=1)  (동등한 FPR)
   → 집단 간 동등한 오류율

3. 보정(Calibration):
   P(Y=1 | Ŷ=p, A=0) = P(Y=1 | Ŷ=p, A=1)
   → 동일한 예측 확률은 동일한 실제 확률을 의미

참고: 모든 공정성 기준을 동시에 만족하는 것은 수학적으로 불가능합니다
(사소한 경우 제외). 맥락에 따라 선택하세요.
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

## 9. 프로덕션에서의 설명 가능성

### 9.1 추론 시 설명 생성

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

### 9.2 설명 대시보드

```python
"""
설명 대시보드 구축 도구:

1. SHAP 내장 플롯:
   shap.plots.waterfall() → 단일 예측 분해
   shap.plots.force() → 대화형 HTML 설명
   shap.plots.bar() → 전역 피처 중요도

2. ExplainerDashboard (pip install explainerdashboard):
   from explainerdashboard import ClassifierExplainer, ExplainerDashboard
   explainer = ClassifierExplainer(model, X_test, y_test)
   dashboard = ExplainerDashboard(explainer)
   dashboard.run()

3. 커스텀 Flask/Streamlit 앱:
   - 폼을 통해 입력 피처 수신
   - 예측 + SHAP 설명 생성
   - 폭포수 플롯과 피처 테이블 표시

4. 규제 요건:
   - GDPR 22조: 자동화된 결정에 대한 설명 요청 권리
   - 미국 평등 신용 기회법(Equal Credit Opportunity Act): 부정적 조치 통지
   - EU AI법(EU AI Act): 고위험 AI 시스템에 대한 투명성 요건
"""
```

---

## 10. 연습 문제

### 연습 1: 분류 모델 설명

```python
"""
1. 유방암 데이터셋 로드 (sklearn.datasets.load_breast_cancer)
2. GradientBoostingClassifier 훈련
3. TreeExplainer로 SHAP 값 생성
4. 다음을 생성:
   a) 요약 플롯 (전역 피처 중요도)
   b) 올바르게 분류된 샘플에 대한 폭포수 플롯
   c) 잘못 분류된 샘플에 대한 폭포수 플롯
   d) 상위 2개 피처에 대한 의존도 플롯
5. 순열 중요도와 비교
"""
```

### 연습 2: LIME vs SHAP 비교

```python
"""
1. California Housing에 랜덤 포레스트 훈련
2. 5개의 서로 다른 테스트 샘플에 대해:
   a) SHAP 설명 생성
   b) LIME 설명 생성
   c) 상위 3개 기여 피처 비교 — 일치하는가?
3. 동일한 샘플에 LIME을 여러 번 실행 (다른 랜덤 시드)하여
   설명의 안정성 측정
4. 토론: 언제 SHAP보다 LIME을 선호하는가?
"""
```

### 연습 3: 공정성 감사(Fairness Audit)

```python
"""
1. Adult Income 데이터셋 사용 (sklearn.datasets.fetch_openml('adult'))
2. 소득 >$50K 예측 분류기 훈련
3. 다음을 계산:
   a) SHAP 값 — 각 인구 통계적 피처가 얼마나 기여하는가?
   b) 'sex'와 'race'에 대한 인구 통계적 동등성
   c) 'sex'에 대한 균등화된 오즈
4. 편향이 발견되면, 완화 전략을 제안하고 구현:
   - 민감한 피처 제거?
   - 재샘플링?
   - 집단별 임계값 조정?
5. 보고: 완화 전후 공정성 지표
"""
```

---

## 11. 요약

### 핵심 정리

| 방법 | 유형 | 범위 | 속도 | 이론 |
|------|------|------|------|------|
| 불순도 중요도(Impurity Importance) | 모델 특화 | 전역 | 빠름 | 카디널리티에 편향 |
| 순열 중요도(Permutation Importance) | 모델 비의존 | 전역 | 중간 | 상관관계에 영향 받음 |
| **SHAP** | 모델 비의존 | 지역 + 전역 | 다양 | 샤플리 공리 (정확) |
| **LIME** | 모델 비의존 | 지역 | 빠름 | 근사 (실행마다 다를 수 있음) |
| PDP | 모델 비의존 | 전역 | 중간 | 독립성 가정 |
| ICE | 모델 비의존 | 지역 | 중간 | 개인 수준 PDP |
| ALE | 모델 비의존 | 전역 | 빠름 | 상관관계 처리 |
| 대리 모델(Surrogate) | 모델 비의존 | 전역 | 빠름 | 근사 |

### 언제 무엇을 사용할지

1. **빠른 전역 개요**: 순열 중요도 또는 SHAP 막대 플롯
2. **단일 예측 설명**: SHAP 폭포수 플롯 또는 LIME
3. **피처 효과 시각화**: PDP/ICE (비상관 피처) 또는 ALE (상관 피처)
4. **공정성 감사**: SHAP + 인구 통계 집단 분석
5. **프로덕션 설명**: SHAP TreeExplainer (트리 모델용)

### 모범 사례

1. **여러 방법 사용**: 단일 방법만으로는 전체 그림을 볼 수 없음
2. **SHAP 가산성 확인**: `base_value + Σ(SHAP) = prediction`
3. **상관된 피처 주의**: PDP 대신 ALE 사용
4. **설명 안정성 테스트**: LIME은 실행 간에 달라질 수 있음
5. **청중 고려**: 기술적 이해관계자 → SHAP 값; 비즈니스 → 대리 모델 규칙

### 다음 단계

- **L17**: 불균형 데이터(Imbalanced Data) — 샘플링과 비용 민감 방법으로 클래스 불균형 처리
- **L18**: 시계열 ML(Time Series ML) — 시간적 예측 문제에 ML 기법 적용
