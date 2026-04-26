# 교차검증과 하이퍼파라미터 튜닝

**이전**: [모델 평가](./04_Model_Evaluation.md) | **다음**: [결정 트리](./06_Decision_Trees.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 단일 학습/테스트 분할이 신뢰할 수 있는 모델 평가에 왜 불충분한지 설명한다
2. K-Fold, Stratified K-Fold, 시계열 교차검증(Time Series cross-validation) 전략을 구현한다
3. 다중 지표 평가를 위해 cross_val_score와 cross_validate를 비교한다
4. 그리드 서치(Grid Search)와 랜덤 서치(Randomized Search)를 적용하여 하이퍼파라미터를 체계적으로 튜닝한다
5. 중첩 교차검증(nested cross-validation)이 모델 선택과 모델 평가를 어떻게 분리하는지 설명한다
6. scikit-learn 파이프라인(Pipeline) 내에서 하이퍼파라미터 튜닝을 시연한다
7. 그리드 서치, 랜덤 서치, 베이즈 최적화(Bayesian optimization) 간의 트레이드오프를 평가한다

---

단일 학습/테스트 분할로 평가한 모델은 오해의 소지가 있는 결과를 낼 수 있습니다 — 데이터가 분할된 방식에서 운이 좋거나 나빴을 수도 있습니다. 교차검증(cross-validation)은 일반화 성능을 더 견고하게 추정하며, 하이퍼파라미터 튜닝(hyperparameter tuning)은 기본 설정에 만족하지 않고 선택한 알고리즘에서 최대한의 성능을 끌어내도록 보장합니다.

---

## 이론과 원리

교차검증과 하이퍼파라미터 탐색은 같은 통계적 문제를 공격합니다: 테스트셋을 소비하지 않고 어떻게 모델 사이에서 선택할 것인가? 답은 *일반화 오차 추정량 자체*의 편향-분산 성질을, 그리고 탐색 절차가 그 추정량과 어떻게 상호작용하는지를 신중히 생각하는 것을 요구합니다.

### A. 단일 홀드아웃이 충분하지 않은 이유

Lesson 1의 학습/테스트 분할을 분할의 무작위 선택에 대한 확률 변수로 보고 테스트 오차를 봅시다. 기댓값은 대략 진짜 일반화 오차이지만(낮은 편향) 분산이 클 수 있습니다 — 특히 테스트셋이 작을 때. 단일 분할에서 나온 단일 숫자는 그 숫자가 얼마나 신뢰할 만한지에 대해 아무것도 말해주지 않습니다.

K-폴드 교차검증은 같은 적합-평가 절차를 `K`개의 겹치지 않는 홀드아웃 조각에서 `K`번 실행하고 평균을 냅니다. 이 평균화는 *추정량*(모델 자체가 아님)의 분산을 거의 `1/K` 인수로 줄이는 반면, 편향은 거의 같게 유지됩니다. 트레이드오프:

```
테스트 분할:    분산 큼, 편향 작음, 저렴(1번 적합)
K-폴드 CV:      분산 작음, ~같은 편향, K배 비쌈
```

### B. K-폴드 추정량의 편향과 분산

`K` 선택 자체가 편향-분산 트레이드오프이며, 이번엔 CV 추정량에 대한 것입니다:

| K | 폴드당 학습 크기 | 추정량 편향 | 추정량 분산 | 비용 |
|---|------------------|-------------|-------------|------|
| 2 | N/2 | 큼(작은 학습셋 ⟹ 낙관적 테스트) | 작음 | 저렴 |
| 5–10 | 0.8N–0.9N | 작음 | 보통 | 보통 |
| N (LOO) | N-1 | ~0(각 폴드가 거의 모든 데이터로 학습) | 큼(폴드들이 강하게 상관) | 비쌈 |

**Leave-one-out (LOO) CV**는 `N-1`개로 학습하고 `1`개로 테스트하는 것을 `N`번 반복합니다. `N-1`점에서 학습된 모델의 일반화 오차에 대해 *비편향*이지만, `N`개 학습셋이 거의 동일하므로 분산이 큽니다 — 폴드 수준 오차들이 강하게 상관되어 평균화가 `N`개 독립 표본에서 기대하는 만큼 분산을 줄이지 못합니다.

경험적 균형점은 `K = 5` 또는 `K = 10`입니다. 작은 편향(`~0.9N`이 `N`에 가까움)과 다룰 만한 분산을, 대부분의 워크플로우가 흡수할 수 있는 비용으로 줍니다.

### C. Stratified, Grouped, 시계열 CV: 구조 존중하기

기본 `KFold`는 예제를 균일하게 섞습니다. 평가가 누설(leak)될 수 있는 구조가 데이터에 있을 때마다 이는 잘못입니다.

- **Stratified K-폴드**는 각 폴드의 클래스 비율을 보존합니다. 분류, 특히 희귀 클래스에 필수 — 그렇지 않으면 운 나쁜 분할이 양성 예제 0개인 폴드를 만들어 무의미한 점수를 줍니다.
- **Group K-폴드**는 같은 "그룹"(환자, 사용자, 문서)의 모든 예제를 한 폴드에 유지합니다. 그룹이 데이터셋에 여러 번 등장할 때 필수 — 그렇지 않으면 모델이 그룹 수준 특성을 외울 수 있고 CV 점수가 새 그룹에 대해 낙관적이 됩니다.
- **시계열 CV**는 *과거*만으로 *미래*를 예측합니다. 표준 확장 윈도우 또는 롤링 윈도우 분할은 `[1..t]`로 학습하고 `[t+1..t+h]`에서 테스트하며, 결코 거꾸로 가지 않습니다. 시계열에서 무작위 셔플링은 미래 정보를 누설하여 CV 점수를 무용지물로 만듭니다.

일반 원칙: CV 분할은 모델이 배포될 방식을 모방해야 합니다. 배포가 "새 환자"라면 환자별로 분할. "내일 데이터"라면 시간별로 분할.

### D. Nested CV: 탐색 자체에 비용이 있을 때

하이퍼파라미터 탐색은 두 번째 루프를 추가합니다. 순진한 워크플로우는 잘못입니다:

```
잘못: CV로 하이퍼파라미터 튜닝 → 최고 CV 점수를 테스트 오차로 보고
```

이는 *최고* CV 점수가 많은 후보들에 대한 최댓값이기 때문에 잘못입니다 — 잡음 있는 추정량 집합의 최댓값은 위쪽으로 편향됩니다. CV 폴드에서 운이 좋았던 구성을 골랐고, 그 구성의 진짜 일반화 오차는 더 나쁩니다.

해법은 **중첩 교차검증(nested cross-validation)**입니다: 외부 CV 루프가 일반화 오차를 추정하고, *각* 외부 폴드 안에서 내부 CV 루프가 하이퍼파라미터를 선택합니다. 내부 루프의 최적화 잡음이 외부 루프에 의해 평균화됩니다:

```
for outer_fold in 1..K_outer:
    for hp in hp_grid:
        inner_score = inner_CV(hp, outer_train)        ← 내부 CV
    best_hp = argmax inner_score
    outer_train에서 best_hp로 재적합
    outer_test_score 기록
mean(outer_test_scores) 보고                            ← 비편향 추정량
```

비용은 `K_outer × K_inner × |hp_grid|` 적합. 비싸지만 원칙적입니다.

### E. 탐색 알고리즘: Grid, Random, Bayesian

예산이 빠듯해지면 하이퍼파라미터를 어떻게 제안하는지가 중요합니다.

- **그리드 탐색(Grid search)**은 카르테시안 곱의 모든 점을 평가합니다. 완전 탐색, 자명한 병렬화, 단 차원의 저주에 시달림: 하이퍼파라미터 수를 두 배로 하면 비용이 제곱.
- **무작위 탐색(Random search)**은 각 하이퍼파라미터를 사전분포에서 독립적으로 샘플링합니다. Bergstra & Bengio (2012)는 소수의 하이퍼파라미터만 중요할 때 그리드보다 *엄격히 우월*함을 보였습니다 — 무작위 탐색은 무관한 축에 더 적은 시도를 낭비합니다. 같은 예산에 각 중요 축을 더 조밀하게 탐색합니다.
- **베이지안 최적화(Bayesian optimization)**는 지금까지 본 평가에 대리 모델(가우시안 과정 또는 트리 기반)을 적합한 다음, 획득 함수(Expected Improvement, Upper Confidence Bound)를 사용해 미지 영역의 탐험과 유망 영역의 활용을 균형 잡는 다음 점을 고릅니다. 표본 효율적이지만 단계당 자체 최적화 오버헤드가 추가됩니다. 각 평가가 비쌀 때 최선(딥 모델, 긴 학습 실행).

공짜 점심은 없습니다: 그리드는 매우 적은 하이퍼파라미터와 무한 컴퓨팅이 있을 때, 무작위는 중간 예산에, 베이지안은 각 적합이 시간 단위 비용일 때 이깁니다.

### From Theory to the Code Below

- 섹션 1.1의 `cross_val_score(..., cv=5)`는 (A)/(B)의 K=5 평균화입니다.
- 섹션 1.2의 `StratifiedKFold`는 (C)의 구조 인식 분할기입니다.
- 섹션 2의 `GridSearchCV`는 (E)의 완전 탐색이며; `RandomizedSearchCV`는 무작위 표본 변형입니다.
- 섹션 3의 `cross_val_score(GridSearchCV(...), ...)`는 (D)의 중첩 CV입니다 — 외부 `cross_val_score`가 비편향 추정을 제공하고 내부 `GridSearchCV`가 하이퍼파라미터를 선택합니다.
- `optuna`나 `scikit-optimize` 같은 도구(Lesson 19에서 도입될 때)는 (E)의 베이지안 최적화 분기를 구현합니다.

---

## 1. 교차검증 (Cross-Validation)

### 1.1 K-Fold 교차검증

```python
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# 데이터 로드
iris = load_iris()
X, y = iris.data, iris.target

# 모델 생성
model = LogisticRegression(max_iter=1000)

# K-Fold 교차검증
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print("K-Fold 교차검증 (K=5)")
print(f"각 폴드 점수: {scores}")
print(f"평균 정확도: {scores.mean():.4f}")
print(f"표준편차: {scores.std():.4f}")
print(f"95% 신뢰구간: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
```

### 1.2 Stratified K-Fold

```python
from sklearn.model_selection import StratifiedKFold

# 클래스 비율 유지
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

print("\nStratified K-Fold")
print(f"평균 정확도: {scores.mean():.4f}")

# 각 폴드의 클래스 분포 확인
print("\n각 폴드의 클래스 분포:")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    train_classes = np.bincount(y[train_idx])
    val_classes = np.bincount(y[val_idx])
    print(f"  Fold {fold}: Train={train_classes}, Val={val_classes}")
```

### 1.3 다양한 교차검증 방법

```python
from sklearn.model_selection import (
    LeaveOneOut,
    LeavePOut,
    ShuffleSplit,
    RepeatedKFold,
    RepeatedStratifiedKFold
)

# Leave-One-Out (LOO)
loo = LeaveOneOut()
print(f"LOO 분할 수: {loo.get_n_splits(X)}")  # 데이터 수와 동일

# Shuffle Split (랜덤 분할)
ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
scores = cross_val_score(model, X, y, cv=ss)
print(f"\nShuffle Split 평균: {scores.mean():.4f}")

# Repeated K-Fold (반복)
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
scores = cross_val_score(model, X, y, cv=rkf)
print(f"Repeated K-Fold 평균: {scores.mean():.4f}")
print(f"Repeated K-Fold 총 분할 수: {len(scores)}")  # 5 * 10 = 50
```

### 1.4 시계열 교차검증

```python
from sklearn.model_selection import TimeSeriesSplit

# 시계열 데이터용 (과거 → 미래 예측)
tscv = TimeSeriesSplit(n_splits=5)

print("Time Series Split:")
for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
    print(f"  Fold {fold}: Train=[{train_idx[0]}:{train_idx[-1]}], Test=[{test_idx[0]}:{test_idx[-1]}]")
```

---

## 2. cross_val_score vs cross_validate

```python
from sklearn.model_selection import cross_validate

# 여러 지표 동시 평가
scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']

cv_results = cross_validate(
    model, X, y,
    cv=5,
    scoring=scoring,
    return_train_score=True
)

print("cross_validate 결과:")
for metric in scoring:
    train_key = f'train_{metric}'
    test_key = f'test_{metric}'
    print(f"\n{metric}:")
    print(f"  Train: {cv_results[train_key].mean():.4f} (+/- {cv_results[train_key].std():.4f})")
    print(f"  Test:  {cv_results[test_key].mean():.4f} (+/- {cv_results[test_key].std():.4f})")

# 학습 시간 정보
print(f"\n평균 학습 시간: {cv_results['fit_time'].mean():.4f}초")
print(f"평균 예측 시간: {cv_results['score_time'].mean():.4f}초")
```

---

## 3. 하이퍼파라미터 튜닝

### 3.1 Grid Search

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# 데이터 준비
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 하이퍼파라미터 그리드
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear']
}

# Grid Search
grid_search = GridSearchCV(
    SVC(),
    param_grid,
    cv=5,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1  # 모든 CPU 사용
)

grid_search.fit(X_scaled, y)

print("\nGrid Search 결과:")
print(f"최적 파라미터: {grid_search.best_params_}")
print(f"최적 점수: {grid_search.best_score_:.4f}")

# 모든 결과 확인
import pandas as pd
results = pd.DataFrame(grid_search.cv_results_)
print(f"\n상위 5개 조합:")
print(results.nsmallest(5, 'rank_test_score')[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']])
```

### 3.2 Randomized Search

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

# 하이퍼파라미터 분포
param_distributions = {
    'C': uniform(0.1, 100),  # 0.1 ~ 100.1 균등 분포
    'gamma': uniform(0.001, 1),
    'kernel': ['rbf', 'linear', 'poly']
}

# Randomized Search
random_search = RandomizedSearchCV(
    SVC(),
    param_distributions,
    n_iter=50,  # 50개 조합 시도
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_scaled, y)

print("Randomized Search 결과:")
print(f"최적 파라미터: {random_search.best_params_}")
print(f"최적 점수: {random_search.best_score_:.4f}")
```

### 3.3 Grid Search vs Randomized Search

```python
"""
Grid Search:
- 장점: 모든 조합 탐색, 최적해 보장 (그리드 내에서)
- 단점: 조합 수가 기하급수적으로 증가

Randomized Search:
- 장점: 계산 효율적, 연속 분포 탐색 가능
- 단점: 최적해 보장 없음

선택 기준:
- 파라미터 수 적고 범위 명확 → Grid Search
- 파라미터 수 많거나 범위 불확실 → Randomized Search
"""
```

---

## 4. 고급 튜닝 기법

### 4.1 Halving Search (반감 탐색)

```python
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

# 자원을 점진적으로 할당하며 탐색
halving_search = HalvingGridSearchCV(
    SVC(),
    param_grid,
    cv=5,
    factor=3,  # 각 라운드에서 후보 1/3로 축소
    resource='n_samples',
    random_state=42
)

halving_search.fit(X_scaled, y)

print("Halving Grid Search 결과:")
print(f"최적 파라미터: {halving_search.best_params_}")
print(f"최적 점수: {halving_search.best_score_:.4f}")
```

### 4.2 Bayesian Optimization (Optuna)

```python
# pip install optuna

import optuna
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

def objective(trial):
    # 하이퍼파라미터 제안
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 2, 32)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )

    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    return scores.mean()

# 최적화 실행
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=100)

# print(f"최적 파라미터: {study.best_params}")
# print(f"최적 점수: {study.best_value:.4f}")
```

---

## 5. 중첩 교차검증 (Nested CV)

```python
from sklearn.model_selection import cross_val_score, GridSearchCV

# 외부 루프: 모델 평가
# 내부 루프: 하이퍼파라미터 튜닝

# 내부 CV (하이퍼파라미터 튜닝)
param_grid = {'C': [0.1, 1, 10], 'gamma': [0.1, 0.01]}
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid_search = GridSearchCV(SVC(), param_grid, cv=inner_cv, scoring='accuracy')

# 외부 CV (모델 평가)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
nested_scores = cross_val_score(grid_search, X_scaled, y, cv=outer_cv, scoring='accuracy')

print("중첩 교차검증 결과:")
print(f"각 외부 폴드 점수: {nested_scores}")
print(f"평균 점수: {nested_scores.mean():.4f} (+/- {nested_scores.std():.4f})")

# 비교: 일반 CV vs 중첩 CV
grid_search.fit(X_scaled, y)
print(f"\n일반 CV 최적 점수: {grid_search.best_score_:.4f}")
print(f"중첩 CV 평균 점수: {nested_scores.mean():.4f}")
# 중첩 CV가 더 현실적인 일반화 성능 추정
```

---

## 6. 파이프라인과 함께 사용

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 파이프라인 정의
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC())
])

# 파라미터 이름: step__parameter
param_grid = {
    'svm__C': [0.1, 1, 10],
    'svm__gamma': [0.1, 0.01, 0.001],
    'svm__kernel': ['rbf', 'linear']
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)

print("파이프라인 Grid Search 결과:")
print(f"최적 파라미터: {grid_search.best_params_}")
print(f"최적 점수: {grid_search.best_score_:.4f}")
```

---

## 7. 실전 팁

### 7.1 스코어링 함수

```python
from sklearn.metrics import make_scorer, f1_score, mean_squared_error

# 내장 스코어링
# 분류: 'accuracy', 'precision', 'recall', 'f1', 'roc_auc'
# 회귀: 'r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'

# 커스텀 스코어링 함수
def custom_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')

custom_scorer = make_scorer(custom_score)

scores = cross_val_score(model, X, y, cv=5, scoring=custom_scorer)
print(f"커스텀 스코어: {scores.mean():.4f}")
```

### 7.2 조기 종료 콜백

```python
# Optuna에서 조기 종료
# import optuna

# def objective(trial):
#     # ...
#     for epoch in range(100):
#         accuracy = train_epoch()
#         trial.report(accuracy, epoch)
#         if trial.should_prune():
#             raise optuna.TrialPruned()
#     return accuracy

# study = optuna.create_study(direction='maximize',
#                            pruner=optuna.pruners.MedianPruner())
```

### 7.3 결과 저장

```python
import joblib
import json

# 최적 모델 저장
best_model = grid_search.best_estimator_
joblib.dump(best_model, 'best_model.pkl')

# 결과 저장
results = {
    'best_params': grid_search.best_params_,
    'best_score': grid_search.best_score_,
    'cv_results': {k: v.tolist() if isinstance(v, np.ndarray) else v
                   for k, v in grid_search.cv_results_.items()}
}

with open('tuning_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

---

## 연습 문제

### 문제 1: K-Fold 교차검증
Iris 데이터로 10-Fold 교차검증을 수행하세요.

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

iris = load_iris()
model = LogisticRegression(max_iter=1000)

# 풀이
scores = cross_val_score(model, iris.data, iris.target, cv=10)
print(f"평균 정확도: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### 문제 2: Grid Search
로지스틱 회귀의 C 파라미터를 튜닝하세요.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.01, 0.1, 1, 10, 100]}

# 풀이
grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5)
grid.fit(iris.data, iris.target)
print(f"최적 C: {grid.best_params_['C']}")
print(f"최적 점수: {grid.best_score_:.4f}")
```

---

## 요약

| 기법 | 용도 | 특징 |
|------|------|------|
| K-Fold | 모델 평가 | 데이터를 K개로 분할 |
| Stratified K-Fold | 불균형 데이터 | 클래스 비율 유지 |
| Time Series Split | 시계열 | 시간 순서 유지 |
| Grid Search | 파라미터 튜닝 | 모든 조합 탐색 |
| Randomized Search | 파라미터 튜닝 | 랜덤 샘플링 |
| Nested CV | 신뢰성 높은 평가 | 튜닝과 평가 분리 |
