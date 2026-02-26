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
