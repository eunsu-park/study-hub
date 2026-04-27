# 앙상블 학습 - 배깅 (Bagging)

**이전**: [결정 트리](./06_Decision_Trees.md) | **다음**: [앙상블 학습 - 부스팅](./08_Ensemble_Boosting.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 앙상블 학습 원리와 여러 약한 학습기(weak learner)를 결합하면 강한 학습기(strong learner)가 만들어지는 이유를 설명할 수 있습니다
2. 부트스트랩(bootstrap) 샘플링의 동작 방식을 설명하고, 기대 OOB(Out-of-Bag) 비율을 계산할 수 있습니다
3. 부트스트랩 샘플링과 다수결 투표를 사용하여 배깅 분류기를 처음부터 구현할 수 있습니다
4. Random Forest와 일반 배깅을 비교하고, 무작위 특성 선택이 트리 간 상관관계를 줄이는 방법을 설명할 수 있습니다
5. 불순도 기반 특성 중요도와 순열(permutation) 기반 특성 중요도 점수를 해석할 수 있습니다
6. 별도의 홀드아웃 세트 없이 내장 검증 추정치로 OOB 오류를 활용할 수 있습니다
7. Random Forest, Extra Trees, Voting 분류기의 차이를 구별하고 적절한 사용 사례를 파악할 수 있습니다

---

단일 결정 트리는 빠르고 해석 가능하지만 취약합니다. 데이터의 작은 변화만으로도 완전히 다른 트리가 만들어질 수 있습니다. 배깅(Bagging)은 약간씩 다른 무작위 샘플로 여러 트리를 학습시키고 그 예측을 평균화함으로써 이 불안정성을 해결합니다. 분산을 대폭 줄이면서도 복잡한 패턴을 포착하는 능력은 그대로 유지됩니다.

---

## 1. 앙상블 학습의 기본 개념

### 이론: 분산 감소 정리

`Z_1, Z_2, ..., Z_M`을 평균 `μ`, 분산 `σ²`인 `M`개 확률 변수라 합시다. 평균은 `Z̄ = (1/M) · Σ Z_m`. 그러면:

```
E[Z̄]   = μ                                     ← 편향 변하지 않음
Var[Z̄] = σ²/M                                  ← Z_m이 독립일 때
       = ρ · σ² + (1 - ρ) · σ²/M               ← Z_m이 쌍별 상관 ρ를 가질 때
```

독립 공식은 교과서적인 평균의 분산입니다. 상관 버전(우리가 실제 가진 것 — `M`개 트리가 같은 데이터셋의 겹치는 부트스트랩 표본에서 학습되므로)은 함정을 보여줍니다: `M → ∞`일 때 두 번째 항은 사라지지만 `ρ · σ²`는 사라지지 않습니다. 분산 감소는 *상관 바닥(correlation floor)* `ρ · σ²`에서 점근(asymptote)합니다.

이것이 배깅의 모든 수학적 엔진입니다: 많은 추정량을 평균화하여 분산을 줄이는 것. 편향 항은 건드리지 않습니다 — 과소적합 모델을 배깅하면 또 다른 과소적합 모델이 나옵니다. 배깅이 도움이 되려면 *기저 학습기가 이미 저편향*이어야 합니다(예: 깊고 완전히 자란 트리). 배깅이 할 수 있는 모든 것은 분산을 줄이는 것뿐이기 때문입니다.

### 1.1 앙상블이란?

```python
"""
앙상블 학습 (Ensemble Learning):
- 여러 개의 약한 학습기(weak learner)를 결합하여 강한 학습기 생성
- "군중의 지혜" (Wisdom of Crowds)

앙상블의 주요 유형:
1. 배깅 (Bagging): 병렬 학습, 분산 감소
   - Random Forest
   - Bagging Classifier/Regressor

2. 부스팅 (Boosting): 순차 학습, 편향 감소
   - AdaBoost
   - Gradient Boosting
   - XGBoost, LightGBM

3. 스태킹 (Stacking): 메타 모델 학습
   - 다양한 모델의 예측을 입력으로 사용

4. 보팅 (Voting): 단순 투표
   - Hard Voting, Soft Voting
"""
```

### 1.2 배깅의 원리

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# 부트스트랩 샘플링 시각화
np.random.seed(42)
original_data = np.arange(10)

print("부트스트랩 샘플링 예시:")
print(f"원본 데이터: {original_data}")

for i in range(3):
    bootstrap_sample = np.random.choice(original_data, size=len(original_data), replace=True)
    oob = set(original_data) - set(bootstrap_sample)
    print(f"샘플 {i+1}: {bootstrap_sample} (OOB: {oob})")

# 부트스트랩 샘플에서 OOB 비율
"""
기대되는 OOB 비율:
- 각 샘플이 선택되지 않을 확률 = (1 - 1/n)^n
- n이 커지면 → e^(-1) ≈ 0.368 (약 37%)
- 즉, 각 모델은 원본 데이터의 약 63%만 사용
"""

n = 1000
selected = np.zeros(n)
for _ in range(n):
    idx = np.random.randint(0, n)
    selected[idx] = 1
oob_ratio = 1 - np.mean(selected)
print(f"\n실험적 OOB 비율: {oob_ratio:.4f}")
print(f"이론적 OOB 비율: {1/np.e:.4f}")
```

---

## 2. 직접 구현하는 배깅

### 이론: 부트스트랩 표본추출 — 다양성을 만드는 법

배깅은 각 기저 추정기를 **부트스트랩 표본(bootstrap sample)**에서 학습합니다: `N`행 학습셋에서 *복원 추출(with replacement)*로 `N`개를 뽑습니다. 어떤 행은 여러 번 나오고 어떤 행은 전혀 나오지 않습니다. 특정 행이 `N`번의 독립 추출에서 *뽑히지 않을* 확률은

```
P(행 i가 전혀 뽑히지 않음) = (1 - 1/N)^N  →  1/e ≈ 0.368  N → ∞일 때
```

따라서 각 부트스트랩 표본은 약 `1 - 1/e ≈ 63.2%`의 고유 행을 포함합니다; 다른 `36.8%`는 그 트리에 대해 **OOB(out-of-bag)**입니다. 각 트리는 자신의 부트스트랩 표본에서 학습되며, 다른 표본과 다릅니다 — 거기서 다양성이 옵니다. 다양성이 없으면 모든 `M` 트리가 동일하고 `ρ = 1`이 되어 분산 감소를 죽입니다.

OOB 행은 즐거운 부수 효과가 있습니다: 각 행은 ~37%의 트리에 대해 OOB입니다. 행 `i`에 대한 예측을 `i`를 보지 못한 트리들에서만 집계하면 **OOB 오차**를 얻습니다 — 별도의 검증셋 없이 계산되는 무료의, 거의 비편향인 일반화 오차 추정.

```python
from sklearn.base import clone

class SimpleBagging:
    """간단한 배깅 구현"""

    def __init__(self, base_estimator, n_estimators=10, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimators_ = []
        self.oob_indices_ = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        n_samples = len(X)
        self.estimators_ = []
        self.oob_indices_ = []

        for _ in range(self.n_estimators):
            # 부트스트랩 샘플링
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            oob_indices = list(set(range(n_samples)) - set(indices))

            X_bootstrap = X[indices]
            y_bootstrap = y[indices]

            # 모델 학습
            estimator = clone(self.base_estimator)
            estimator.fit(X_bootstrap, y_bootstrap)

            self.estimators_.append(estimator)
            self.oob_indices_.append(oob_indices)

        return self

    def predict(self, X):
        # 각 모델의 예측 수집
        predictions = np.array([est.predict(X) for est in self.estimators_])
        # 다수결 투표
        return np.apply_along_axis(
            lambda x: np.bincount(x.astype(int)).argmax(),
            axis=0,
            arr=predictions
        )

    def predict_proba(self, X):
        # 확률 평균
        probas = np.array([est.predict_proba(X) for est in self.estimators_])
        return np.mean(probas, axis=0)

# 테스트
X, y = make_classification(n_samples=500, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 단일 트리 vs 배깅
single_tree = DecisionTreeClassifier(random_state=42)
single_tree.fit(X_train, y_train)

bagging = SimpleBagging(DecisionTreeClassifier(), n_estimators=10, random_state=42)
bagging.fit(X_train, y_train)

print("배깅 효과 비교:")
print(f"  단일 결정 트리: {single_tree.score(X_test, y_test):.4f}")
print(f"  배깅 (10 trees): {np.mean(bagging.predict(X_test) == y_test):.4f}")
```

---

## 3. sklearn의 BaggingClassifier

```python
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# BaggingClassifier 사용
bagging_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=1.0,        # 각 부트스트랩 샘플 크기 (비율)
    max_features=1.0,       # 각 모델에서 사용할 특성 비율
    bootstrap=True,         # 부트스트랩 샘플링 사용
    bootstrap_features=False,  # 특성 부트스트랩
    oob_score=True,         # OOB 점수 계산
    n_jobs=-1,              # 병렬 처리
    random_state=42
)

bagging_clf.fit(X_train, y_train)
y_pred = bagging_clf.predict(X_test)

print("BaggingClassifier 결과:")
print(f"  훈련 정확도: {bagging_clf.score(X_train, y_train):.4f}")
print(f"  테스트 정확도: {accuracy_score(y_test, y_pred):.4f}")
print(f"  OOB 점수: {bagging_clf.oob_score_:.4f}")
```

### 3.1 모델 수에 따른 성능 변화

```python
# 모델 수 증가에 따른 성능 변화
n_estimators_range = [1, 5, 10, 20, 50, 100, 200]
train_scores = []
test_scores = []
oob_scores = []

for n_est in n_estimators_range:
    clf = BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=n_est,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    train_scores.append(clf.score(X_train, y_train))
    test_scores.append(clf.score(X_test, y_test))
    oob_scores.append(clf.oob_score_)

# 시각화
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, train_scores, 'o-', label='Train')
plt.plot(n_estimators_range, test_scores, 's-', label='Test')
plt.plot(n_estimators_range, oob_scores, '^-', label='OOB')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.title('Bagging: Performance vs Number of Estimators')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 4. Random Forest

### 이론: 랜덤 포레스트 — 두 번째 비상관화의 원천

배깅된 트리들은 한두 특성이 지배할 때 여전히 강하게 상관됩니다 — 모든 트리가 같은 루트 분할을 고릅니다. 랜덤 포레스트(Breiman, 2001)는 이 상관을 깨기 위해 두 번째 무작위화를 추가합니다:

> **각 분할에서**, 모든 `p`개가 아니라 `m_try`개 특성의 무작위 부분집합만 고려합니다(분류는 보통 `√p`, 회귀는 `p/3`).

이것이 "배깅된 트리"를 "랜덤 포레스트"로 바꾸는 변경입니다. 각 트리가 가끔 더 나쁜 분할을 고르도록 강제하면 개별 트리 정확도가 약간 줄지만 쌍별 상관 `ρ`가 상당히 줄어듭니다. 분산 공식 `ρ · σ² + (1 - ρ) · σ²/M`이 `ρ`에 대해 감소하므로 거래는 유리합니다: 작은 편향 증가를 큰 분산 감소와 맞바꿉니다.

경험적으로 랜덤 포레스트는 정확히 이 `m_try` 메커니즘 때문에 평범한 배깅을 압도합니다. scikit-learn의 `max_features` 하이퍼파라미터가 정확히 `m_try`입니다.

### 4.1 기본 사용법

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_iris

# 데이터 로드
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Random Forest 분류기
rf_clf = RandomForestClassifier(
    n_estimators=100,       # 트리 수
    max_depth=None,         # 최대 깊이
    min_samples_split=2,    # 분할 최소 샘플
    min_samples_leaf=1,     # 리프 최소 샘플
    max_features='sqrt',    # 분할 시 고려할 특성 수
    bootstrap=True,         # 부트스트랩 샘플링
    oob_score=True,         # OOB 점수
    n_jobs=-1,              # 병렬 처리
    random_state=42
)

rf_clf.fit(X_train, y_train)

print("Random Forest 결과:")
print(f"  훈련 정확도: {rf_clf.score(X_train, y_train):.4f}")
print(f"  테스트 정확도: {rf_clf.score(X_test, y_test):.4f}")
print(f"  OOB 점수: {rf_clf.oob_score_:.4f}")
```

### 4.2 Random Forest vs 일반 Bagging

```python
"""
Random Forest와 Bagging의 차이:

1. 특성 무작위 선택:
   - Bagging: 모든 특성 사용 (max_features=1.0)
   - Random Forest: sqrt(n_features) 또는 log2(n_features) 사용

2. 트리 상관관계:
   - Bagging: 트리 간 상관관계 높음
   - Random Forest: 트리 간 상관관계 낮음 (다양성 증가)

3. 분산 감소:
   - Var(average) = Var(single) / n + (n-1)/n * Cov
   - 상관관계(Cov)가 낮을수록 분산 더 감소
"""

# 비교 실험
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_features=1.0,  # 모든 특성 사용
    random_state=42,
    n_jobs=-1
)

rf = RandomForestClassifier(
    n_estimators=100,
    max_features='sqrt',  # sqrt(n_features) 사용
    random_state=42,
    n_jobs=-1
)

bagging.fit(X_train, y_train)
rf.fit(X_train, y_train)

print("Bagging vs Random Forest:")
print(f"  Bagging 정확도: {bagging.score(X_test, y_test):.4f}")
print(f"  Random Forest 정확도: {rf.score(X_test, y_test):.4f}")
```

### 4.3 max_features 파라미터

```python
# max_features에 따른 성능 변화
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42
)

n_features = X_train.shape[1]
max_features_options = [1, 'sqrt', 'log2', 0.5, n_features]

print("max_features에 따른 성능:")
for max_feat in max_features_options:
    rf = RandomForestClassifier(
        n_estimators=100,
        max_features=max_feat,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    print(f"  max_features={max_feat}: {rf.score(X_test, y_test):.4f}")
```

---

## 5. 특성 중요도 (Feature Importance)

### 5.1 기본 특성 중요도

```python
# Random Forest 학습
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 특성 중요도
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# 시각화
plt.figure(figsize=(12, 6))
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)),
           [cancer.feature_names[i] for i in indices],
           rotation=90)
plt.ylabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.show()

# 상위 10개 특성
print("\n상위 10개 특성:")
for i in range(10):
    print(f"  {i+1}. {cancer.feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
```

### 5.2 특성 중요도 해석 방법

```python
"""
특성 중요도 계산 방법:

1. 불순도 기반 중요도 (Mean Decrease in Impurity, MDI):
   - 각 특성이 분할에 사용될 때 불순도 감소량의 평균
   - feature_importances_ 기본값
   - 단점: 고카디널리티 특성에 편향

2. 순열 중요도 (Permutation Importance):
   - 특성 값을 무작위로 섞었을 때 성능 감소 측정
   - 더 신뢰성 있는 중요도
"""

from sklearn.inspection import permutation_importance

# 순열 중요도 계산
perm_importance = permutation_importance(
    rf, X_test, y_test,
    n_repeats=30,
    random_state=42,
    n_jobs=-1
)

# 비교 시각화
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# MDI (불순도 기반)
sorted_idx_mdi = rf.feature_importances_.argsort()[-10:]
axes[0].barh(range(10), rf.feature_importances_[sorted_idx_mdi])
axes[0].set_yticks(range(10))
axes[0].set_yticklabels([cancer.feature_names[i] for i in sorted_idx_mdi])
axes[0].set_title('MDI (Impurity-based) Feature Importance')

# 순열 중요도
sorted_idx_perm = perm_importance.importances_mean.argsort()[-10:]
axes[1].barh(range(10), perm_importance.importances_mean[sorted_idx_perm])
axes[1].set_yticks(range(10))
axes[1].set_yticklabels([cancer.feature_names[i] for i in sorted_idx_perm])
axes[1].set_title('Permutation Feature Importance')

plt.tight_layout()
plt.show()
```

### 5.3 특성 선택에 활용

```python
from sklearn.feature_selection import SelectFromModel

# 중요도 기반 특성 선택
selector = SelectFromModel(
    RandomForestClassifier(n_estimators=100, random_state=42),
    threshold='median'  # 중요도 중간값 이상인 특성만 선택
)
selector.fit(X_train, y_train)

# 선택된 특성
selected_features = cancer.feature_names[selector.get_support()]
print(f"선택된 특성 수: {len(selected_features)}")
print(f"선택된 특성: {list(selected_features)}")

# 선택된 특성으로 학습
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

rf_selected = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selected.fit(X_train_selected, y_train)

print(f"\n전체 특성 정확도: {rf.score(X_test, y_test):.4f}")
print(f"선택된 특성 정확도: {rf_selected.score(X_test_selected, y_test):.4f}")
```

---

## 6. OOB (Out-of-Bag) 에러

### 이론: 배깅으로 해결되지 않는 것

분산 감소에는 천장이 있습니다. 기저 학습기가 편향되어 있다면 배깅은 그것을 구할 수 없습니다 — 앙상블이 편향을 상속받습니다. 특성이 너무 적거나 너무 약하면 비상관화가 작동할 여지가 없습니다. 그리고 환원 불가능한 잡음 `σ²`(Lesson 1의 편향-분산 분해)이 크면 앙상블 오차는 `M`과 무관하게 그 잡음 바닥에서 정체됩니다.

배깅이 빛나는 고전적 영역: 표 형식 데이터에 보통의 정보적 특성이 많을 때, 유연하고 저편향, 고분산인 기저 학습기(깊은 결정 트리). 그 영역이 실세계 ML 문제의 막대한 부분을 차지하며, 그래서 랜덤 포레스트가 실전에서 가장 널리 배포되는 알고리즘 중 하나입니다.

### 6.1 OOB 점수 이해

```python
"""
OOB (Out-of-Bag) 에러:
- 각 트리는 부트스트랩 샘플로 학습
- 각 샘플은 평균 37%의 트리에서 OOB (학습에 사용되지 않음)
- OOB 샘플로 검증 → 별도 검증 세트 불필요

장점:
1. 추가 데이터 분할 불필요
2. 교차검증과 유사한 효과
3. 학습과 동시에 검증 가능
"""

# OOB 점수 활용
rf = RandomForestClassifier(
    n_estimators=100,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

print("OOB 점수 분석:")
print(f"  OOB 점수: {rf.oob_score_:.4f}")
print(f"  테스트 점수: {rf.score(X_test, y_test):.4f}")

# OOB 예측 확률
print(f"\nOOB 예측 확률 (처음 5개 샘플):")
print(rf.oob_decision_function_[:5])
```

### 6.2 OOB vs 교차검증 비교

```python
from sklearn.model_selection import cross_val_score

# 교차검증
cv_scores = cross_val_score(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X_train, y_train, cv=5
)

# OOB
rf_oob = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
rf_oob.fit(X_train, y_train)

print("OOB vs 교차검증 비교:")
print(f"  OOB 점수: {rf_oob.oob_score_:.4f}")
print(f"  CV 평균 점수: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
```

---

## 7. 하이퍼파라미터 튜닝

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint, uniform

# Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.5]
}

# 더 효율적인 Randomized Search
param_distributions = {
    'n_estimators': randint(50, 300),
    'max_depth': [None] + list(range(5, 31)),
    'min_samples_split': randint(2, 21),
    'min_samples_leaf': randint(1, 11),
    'max_features': uniform(0.1, 0.9)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print("하이퍼파라미터 튜닝 결과:")
print(f"  최적 파라미터: {random_search.best_params_}")
print(f"  최적 CV 점수: {random_search.best_score_:.4f}")
print(f"  테스트 점수: {random_search.score(X_test, y_test):.4f}")
```

---

## 8. Random Forest 회귀

```python
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 데이터 로드
diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, test_size=0.2, random_state=42
)

# Random Forest 회귀
rf_reg = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)
rf_reg.fit(X_train, y_train)
y_pred = rf_reg.predict(X_test)

print("Random Forest 회귀 결과:")
print(f"  MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"  R²: {r2_score(y_test, y_pred):.4f}")

# 실제값 vs 예측값
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title(f'Random Forest Regression (R² = {r2_score(y_test, y_pred):.4f})')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 9. Extra Trees (Extremely Randomized Trees)

```python
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

"""
Extra Trees vs Random Forest:

1. 분할점 선택:
   - Random Forest: 각 특성의 최적 분할점 선택
   - Extra Trees: 각 특성에서 무작위 분할점 선택

2. 부트스트랩:
   - Random Forest: 기본적으로 부트스트랩 사용
   - Extra Trees: 기본적으로 전체 데이터 사용

3. 특성:
   - Extra Trees: 더 빠름, 더 많은 무작위성
   - Random Forest: 일반적으로 더 좋은 성능
"""

# 비교
rf = RandomForestClassifier(n_estimators=100, random_state=42)
et = ExtraTreesClassifier(n_estimators=100, random_state=42)

rf.fit(X_train, y_train)
et.fit(X_train, y_train)

print("Random Forest vs Extra Trees:")
print(f"  Random Forest: {rf.score(X_test, y_test):.4f}")
print(f"  Extra Trees: {et.score(X_test, y_test):.4f}")
```

---

## 10. Voting Classifier

### 이론: 집계 — 평균 vs 투표

회귀의 경우 앙상블 예측은 단순 평균:

```
ŷ_ensemble(x) = (1/M) · Σ_m  T_m(x)
```

분류의 경우 두 가지 자연스러운 옵션:

- **하드 투표(Hard voting)**: 각 트리가 클래스를 예측; 앙상블은 다수결로 고름. 원-핫 인코딩된 투표를 평균한 뒤 arg-max한 것과 동등.
- **소프트 투표(Soft voting)**: 각 트리가 클래스 확률을 출력; 앙상블이 확률을 평균한 뒤 argmax. 실전에서 엄격히 우월 — 각 트리에서 더 많은 정보를 사용하며, 특히 확률이 잘 보정될 때.

scikit-learn의 `RandomForestClassifier`는 기본이 소프트 투표(predict_proba 출력을 평균한 뒤 argmax).

```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# 다양한 모델 정의
clf1 = LogisticRegression(random_state=42, max_iter=1000)
clf2 = RandomForestClassifier(n_estimators=50, random_state=42)
clf3 = SVC(probability=True, random_state=42)

# Hard Voting (다수결)
hard_voting = VotingClassifier(
    estimators=[
        ('lr', clf1),
        ('rf', clf2),
        ('svc', clf3)
    ],
    voting='hard'
)

# Soft Voting (확률 평균)
soft_voting = VotingClassifier(
    estimators=[
        ('lr', clf1),
        ('rf', clf2),
        ('svc', clf3)
    ],
    voting='soft'
)

# 학습 및 비교
print("Voting Classifier 비교:")
for clf, label in [(clf1, 'Logistic'), (clf2, 'RF'), (clf3, 'SVC'),
                   (hard_voting, 'Hard Voting'), (soft_voting, 'Soft Voting')]:
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(f"  {label}: {score:.4f}")
```

---

## 연습 문제

### 문제 1: Random Forest 분류
유방암 데이터로 Random Forest를 학습하고 특성 중요도를 분석하세요.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42
)

# 풀이
rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
rf.fit(X_train, y_train)

print(f"테스트 정확도: {rf.score(X_test, y_test):.4f}")
print(f"OOB 점수: {rf.oob_score_:.4f}")

print("\n상위 5개 특성:")
indices = np.argsort(rf.feature_importances_)[::-1][:5]
for i, idx in enumerate(indices):
    print(f"  {i+1}. {cancer.feature_names[idx]}: {rf.feature_importances_[idx]:.4f}")
```

### 문제 2: 하이퍼파라미터 튜닝
Grid Search로 최적의 Random Forest 파라미터를 찾으세요.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, None],
    'min_samples_leaf': [1, 2, 5]
}

# 풀이
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"최적 파라미터: {grid_search.best_params_}")
print(f"최적 CV 점수: {grid_search.best_score_:.4f}")
print(f"테스트 점수: {grid_search.score(X_test, y_test):.4f}")
```

### 문제 3: Voting Ensemble
여러 모델을 결합한 Voting Classifier를 만드세요.

```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# 풀이
voting_clf = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=1000)),
        ('rf', RandomForestClassifier(n_estimators=50)),
        ('dt', DecisionTreeClassifier(max_depth=5))
    ],
    voting='soft'
)
voting_clf.fit(X_train, y_train)
print(f"Voting 정확도: {voting_clf.score(X_test, y_test):.4f}")
```

---

## 요약

| 모델 | 특징 | 장점 | 단점 |
|------|------|------|------|
| Bagging | 부트스트랩 + 평균 | 분산 감소, 과적합 방지 | 해석 어려움 |
| Random Forest | 배깅 + 특성 랜덤 | 높은 성능, 특성 중요도 | 많은 계산량 |
| Extra Trees | 완전 랜덤 분할 | 빠른 학습 | RF보다 낮은 성능 가능 |
| Voting | 다양한 모델 결합 | 다양성 활용 | 개별 모델 튜닝 필요 |

### Random Forest 하이퍼파라미터 가이드

| 파라미터 | 기본값 | 권장 범위 | 효과 |
|----------|--------|----------|------|
| n_estimators | 100 | 100-500 | 많을수록 안정적 |
| max_depth | None | 10-30 | 과적합 제어 |
| min_samples_split | 2 | 2-20 | 과적합 제어 |
| min_samples_leaf | 1 | 1-10 | 과적합 제어 |
| max_features | 'sqrt' | 'sqrt', 'log2', 0.3-0.7 | 트리 다양성 |
