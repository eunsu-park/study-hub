# 앙상블 학습 - 부스팅 (Boosting)

**이전**: [앙상블 학습 - 배깅](./07_Ensemble_Bagging.md) | **다음**: [서포트 벡터 머신](./09_SVM.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 부스팅이 배깅과 학습 전략 및 오류 감소 측면에서 어떻게 다른지 설명할 수 있습니다
2. 샘플 가중치 업데이트와 모델 가중치 부여를 포함한 AdaBoost 알고리즘을 설명할 수 있습니다
3. Gradient Boosting을 구현하고, 함수 공간에서의 경사 하강법(gradient descent)이 어떻게 수행되는지 설명할 수 있습니다
4. XGBoost, LightGBM, CatBoost를 트리 성장 전략, 속도, 범주형 특성 처리 측면에서 비교할 수 있습니다
5. 부스팅 모델의 과적합 방지를 위한 조기 종료(early stopping)와 정규화(regularization) 기법을 적용할 수 있습니다
6. 부스팅 알고리즘에 대한 체계적인 하이퍼파라미터 튜닝 순서를 실행할 수 있습니다
7. 데이터셋 크기, 특성 유형, 속도 요구사항에 따라 적합한 부스팅 알고리즘을 선택할 수 있습니다

---

배깅이 독립적인 모델들을 평균화하여 분산을 줄인다면, 부스팅은 근본적으로 다른 접근 방식을 취합니다. 모델을 순차적으로 구축하여 편향(bias)을 줄이는데, 각 새 모델은 이전 모델들의 실수를 집중적으로 보완합니다. 이 전략은 XGBoost와 LightGBM처럼 표 형식 데이터 경진대회를 석권하는 머신러닝 역사상 가장 정확한 즉시 사용 가능한 알고리즘들을 탄생시켰습니다.

---

> **병렬 투표가 아닌 직렬 교정.** 배깅이 모두가 독립적으로 투표하는 위원회라면, 부스팅은 각 주자가 이전 주자의 약점을 보완하는 릴레이 팀에 가깝습니다. 첫 번째 모델이 예측을 내놓고, 두 번째 모델은 첫 번째 모델이 틀린 사례에 집중하며, 세 번째 모델은 두 번의 시도 후에도 여전히 틀린 부분에 집중합니다. 각 새 모델은 이전 모델들의 실패에 특화된 "전문가"이며, 최종 예측은 모든 전문가의 의견을 적절한 가중치로 결합합니다.

## 1. 부스팅의 기본 개념

### 1.1 배깅 vs 부스팅

```python
"""
배깅 (Bagging):
- 병렬 학습: 각 모델 독립적으로 학습
- 분산 감소: 과적합 방지
- 결합 방법: 평균 또는 다수결

부스팅 (Boosting):
- 순차 학습: 이전 모델의 오류 보완
- 편향 감소: 과소적합 해결
- 결합 방법: 가중 투표

비유:
- 배깅: 여러 전문가가 독립적으로 의견 제시 후 종합
- 부스팅: 한 전문가가 실수한 부분을 다음 전문가가 집중 보완
"""
```

### 1.2 부스팅 알고리즘 종류

```python
"""
주요 부스팅 알고리즘:

1. AdaBoost (Adaptive Boosting):
   - 잘못 분류된 샘플에 가중치 증가
   - 분류 문제에 주로 사용

2. Gradient Boosting:
   - 잔차(residual)를 예측하는 방식
   - 분류와 회귀 모두 가능

3. XGBoost (eXtreme Gradient Boosting):
   - Gradient Boosting 최적화 버전
   - 정규화, 병렬처리 지원

4. LightGBM:
   - 리프 중심 분할 방식
   - 대용량 데이터에 효율적

5. CatBoost:
   - 범주형 특성 자동 처리
   - Ordered Boosting
"""
```

---

## 2. AdaBoost

### 2.1 AdaBoost 원리

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

"""
AdaBoost 알고리즘:

1. 초기화: 모든 샘플에 동일한 가중치 (1/n)

2. 반복 (t = 1, 2, ..., T):
   a. 가중치 기반으로 약한 학습기 학습
   b. 가중 오류율 계산: ε_t = Σ w_i * I(y_i ≠ h_t(x_i))
   c. 학습기 가중치 계산: α_t = 0.5 * log((1-ε_t)/ε_t)
   d. 샘플 가중치 업데이트:
      - 틀린 샘플: w_i *= exp(α_t)
      - 맞은 샘플: w_i *= exp(-α_t)
   e. 가중치 정규화

3. 최종 예측: sign(Σ α_t * h_t(x))
"""
```

### 2.2 AdaBoost 기본 사용법

```python
# 데이터 생성
X, y = make_classification(
    n_samples=1000, n_features=20,
    n_informative=15, n_redundant=5,
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# AdaBoost 분류기
ada_clf = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # 약한 학습기 (stump)
    n_estimators=50,
    learning_rate=1.0,
    algorithm='SAMME',  # 'SAMME' or 'SAMME.R'
    random_state=42
)

ada_clf.fit(X_train, y_train)
y_pred = ada_clf.predict(X_test)

print("AdaBoost 결과:")
print(f"  훈련 정확도: {ada_clf.score(X_train, y_train):.4f}")
print(f"  테스트 정확도: {accuracy_score(y_test, y_pred):.4f}")
```

### 2.3 학습기 수에 따른 성능

```python
# 학습기 수 증가에 따른 성능 변화
n_estimators_range = [1, 5, 10, 20, 50, 100, 200]
train_scores = []
test_scores = []

for n_est in n_estimators_range:
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=n_est,
        random_state=42
    )
    ada.fit(X_train, y_train)
    train_scores.append(ada.score(X_train, y_train))
    test_scores.append(ada.score(X_test, y_test))

# 시각화
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, train_scores, 'o-', label='Train')
plt.plot(n_estimators_range, test_scores, 's-', label='Test')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.title('AdaBoost: Performance vs Number of Estimators')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 2.4 스테이지별 에러 분석

```python
# 스테이지별 에러
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    random_state=42
)
ada.fit(X_train, y_train)

# 스테이지별 예측
staged_train_scores = list(ada.staged_score(X_train, y_train))
staged_test_scores = list(ada.staged_score(X_test, y_test))

# 시각화
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(staged_train_scores)+1), staged_train_scores, label='Train')
plt.plot(range(1, len(staged_test_scores)+1), staged_test_scores, label='Test')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.title('AdaBoost: Staged Performance')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 3. Gradient Boosting

### 3.1 Gradient Boosting 원리

```python
"""
Gradient Boosting 알고리즘:

목표: 손실 함수 L(y, F(x))를 최소화하는 F(x) 찾기

1. 초기화: F_0(x) = argmin_γ Σ L(y_i, γ)

2. 반복 (m = 1, 2, ..., M):
   a. 의사 잔차(pseudo-residual) 계산:
      r_im = -[∂L(y_i, F(x_i))/∂F(x_i)]_{F=F_{m-1}}

   b. 잔차에 대해 약한 학습기 h_m(x) 학습

   c. 최적 스텝 크기 계산:
      γ_m = argmin_γ Σ L(y_i, F_{m-1}(x_i) + γ * h_m(x_i))

   d. 모델 업데이트:
      F_m(x) = F_{m-1}(x) + learning_rate * γ_m * h_m(x)

손실 함수 예:
- 회귀: MSE → 잔차 = y - F(x)
- 분류: Logloss → 잔차 = y - sigmoid(F(x))
"""
```

### 3.2 sklearn Gradient Boosting

```python
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

# Gradient Boosting 분류기
gb_clf = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=1,
    subsample=1.0,          # 각 트리에 사용할 샘플 비율
    max_features=None,      # 분할에 사용할 특성 수
    random_state=42
)

gb_clf.fit(X_train, y_train)

print("Gradient Boosting 결과:")
print(f"  훈련 정확도: {gb_clf.score(X_train, y_train):.4f}")
print(f"  테스트 정확도: {gb_clf.score(X_test, y_test):.4f}")

# 특성 중요도
print("\n상위 5개 특성 중요도:")
indices = np.argsort(gb_clf.feature_importances_)[::-1][:5]
for i, idx in enumerate(indices):
    print(f"  {i+1}. Feature {idx}: {gb_clf.feature_importances_[idx]:.4f}")
```

### 3.3 학습률과 학습기 수의 균형

```python
# learning_rate vs n_estimators 트레이드오프
learning_rates = [0.01, 0.1, 0.5, 1.0]
n_estimators_list = [200, 100, 50, 20]

plt.figure(figsize=(12, 4))

for i, (lr, n_est) in enumerate(zip(learning_rates, n_estimators_list)):
    gb = GradientBoostingClassifier(
        n_estimators=n_est,
        learning_rate=lr,
        max_depth=3,
        random_state=42
    )
    gb.fit(X_train, y_train)

    staged_scores = list(gb.staged_score(X_test, y_test))

    plt.subplot(1, 4, i+1)
    plt.plot(range(1, len(staged_scores)+1), staged_scores)
    plt.xlabel('Estimators')
    plt.ylabel('Accuracy')
    plt.title(f'LR={lr}, n={n_est}\nFinal={staged_scores[-1]:.4f}')
    plt.ylim(0.7, 1.0)
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 3.4 Gradient Boosting 회귀

```python
from sklearn.datasets import load_diabetes

# 데이터 로드
diabetes = load_diabetes()
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    diabetes.data, diabetes.target, test_size=0.2, random_state=42
)

# Gradient Boosting 회귀
gb_reg = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    loss='squared_error',  # 'squared_error', 'absolute_error', 'huber'
    random_state=42
)
gb_reg.fit(X_train_r, y_train_r)

from sklearn.metrics import mean_squared_error, r2_score

y_pred_r = gb_reg.predict(X_test_r)

print("Gradient Boosting 회귀 결과:")
print(f"  MSE: {mean_squared_error(y_test_r, y_pred_r):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test_r, y_pred_r)):.4f}")
print(f"  R²: {r2_score(y_test_r, y_pred_r):.4f}")
```

---

## 4. XGBoost

### 4.1 XGBoost 소개

```python
"""
XGBoost 특징:

1. 정규화:
   - L1, L2 정규화로 과적합 방지
   - 목표 함수: Σ L(y_i, ŷ_i) + Σ Ω(f_k)
   - Ω(f) = γT + 0.5λ||w||²

2. 효율적인 계산:
   - 2차 테일러 전개 사용
   - 히스토그램 기반 분할
   - 캐시 최적화

3. 결측치 처리:
   - 자동으로 최적 방향 학습

4. 병렬 처리:
   - 특성별 병렬 분할점 탐색
"""

# pip install xgboost
import xgboost as xgb
```

### 4.2 XGBoost 기본 사용법

```python
from xgboost import XGBClassifier, XGBRegressor

# XGBoost 분류기
xgb_clf = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    min_child_weight=1,     # 리프 노드 최소 가중치
    gamma=0,                # 분할에 필요한 최소 손실 감소
    subsample=1.0,          # 행 샘플링 비율
    colsample_bytree=1.0,   # 트리별 열 샘플링 비율
    reg_alpha=0,            # L1 정규화
    reg_lambda=1,           # L2 정규화
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

xgb_clf.fit(X_train, y_train)

print("XGBoost 결과:")
print(f"  훈련 정확도: {xgb_clf.score(X_train, y_train):.4f}")
print(f"  테스트 정확도: {xgb_clf.score(X_test, y_test):.4f}")
```

### 4.3 조기 종료 (Early Stopping)

```python
# 조기 종료 사용
xgb_clf_early = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    early_stopping_rounds=10,  # 10 라운드 동안 개선 없으면 중지
    eval_metric='logloss'
)

# 검증 데이터 분리
X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

xgb_clf_early.fit(
    X_train_sub, y_train_sub,
    eval_set=[(X_val, y_val)],
    verbose=False
)

print("조기 종료 결과:")
print(f"  최적 반복 횟수: {xgb_clf_early.best_iteration}")
print(f"  최적 점수: {xgb_clf_early.best_score:.4f}")
print(f"  테스트 정확도: {xgb_clf_early.score(X_test, y_test):.4f}")
```

### 4.4 XGBoost 특성 중요도

```python
# 특성 중요도 타입
importance_types = ['weight', 'gain', 'cover']

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, imp_type in zip(axes, importance_types):
    importance = xgb_clf.get_booster().get_score(importance_type=imp_type)

    if importance:
        features = list(importance.keys())[:10]
        values = [importance[f] for f in features]

        ax.barh(range(len(features)), values)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_title(f'Feature Importance ({imp_type})')

plt.tight_layout()
plt.show()

"""
중요도 타입:
- weight: 특성이 분할에 사용된 횟수
- gain: 특성 사용 시 평균 이득
- cover: 특성이 커버하는 평균 샘플 수
"""
```

---

## 5. LightGBM

### 5.1 LightGBM 소개

```python
"""
LightGBM 특징:

1. Leaf-wise 성장:
   - 기존: Level-wise (수평 분할)
   - LightGBM: Leaf-wise (손실 최대 감소 리프 분할)
   - 더 빠르고 정확하지만 과적합 위험

2. 히스토그램 기반 분할:
   - 연속형 값을 이산화
   - 메모리 효율적, 빠른 학습

3. GOSS (Gradient-based One-Side Sampling):
   - 그래디언트가 큰 샘플 위주로 샘플링

4. EFB (Exclusive Feature Bundling):
   - 상호 배타적 특성들을 묶음
   - 희소 특성에 효과적
"""

# pip install lightgbm
import lightgbm as lgb
```

### 5.2 LightGBM 기본 사용법

```python
from lightgbm import LGBMClassifier, LGBMRegressor

# LightGBM 분류기
lgb_clf = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=-1,           # -1: 제한 없음
    num_leaves=31,          # 리프 노드 최대 수
    min_child_samples=20,   # 리프 노드 최소 샘플 수
    subsample=1.0,          # 행 샘플링 (bagging_fraction)
    colsample_bytree=1.0,   # 열 샘플링
    reg_alpha=0,            # L1 정규화
    reg_lambda=0,           # L2 정규화
    random_state=42,
    verbose=-1
)

lgb_clf.fit(X_train, y_train)

print("LightGBM 결과:")
print(f"  훈련 정확도: {lgb_clf.score(X_train, y_train):.4f}")
print(f"  테스트 정확도: {lgb_clf.score(X_test, y_test):.4f}")
```

### 5.3 num_leaves vs max_depth

```python
"""
num_leaves와 max_depth의 관계:
- max_depth = d일 때, 최대 리프 수 = 2^d
- num_leaves = 31이면 대략 max_depth = 5 수준
- 과적합 방지: num_leaves < 2^max_depth

권장 설정:
- 대용량 데이터: num_leaves = 2^max_depth - 1 이하
- 소규모 데이터: num_leaves를 작게 (15~31)
"""

# num_leaves에 따른 성능
num_leaves_range = [15, 31, 63, 127, 255]
train_scores = []
test_scores = []

for num_leaves in num_leaves_range:
    lgb_temp = LGBMClassifier(
        n_estimators=100,
        num_leaves=num_leaves,
        random_state=42,
        verbose=-1
    )
    lgb_temp.fit(X_train, y_train)
    train_scores.append(lgb_temp.score(X_train, y_train))
    test_scores.append(lgb_temp.score(X_test, y_test))

plt.figure(figsize=(10, 6))
plt.plot(num_leaves_range, train_scores, 'o-', label='Train')
plt.plot(num_leaves_range, test_scores, 's-', label='Test')
plt.xlabel('num_leaves')
plt.ylabel('Accuracy')
plt.title('LightGBM: num_leaves Effect')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 5.4 범주형 특성 처리

```python
# LightGBM은 범주형 특성을 직접 처리 가능
import pandas as pd

# 예시 데이터
df = pd.DataFrame({
    'num_feature': np.random.randn(1000),
    'cat_feature': np.random.choice(['A', 'B', 'C', 'D'], 1000),
    'target': np.random.randint(0, 2, 1000)
})

# 범주형으로 변환
df['cat_feature'] = df['cat_feature'].astype('category')

X_cat = df[['num_feature', 'cat_feature']]
y_cat = df['target']

X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(
    X_cat, y_cat, test_size=0.2, random_state=42
)

# LightGBM은 자동으로 범주형 처리
lgb_cat = LGBMClassifier(random_state=42, verbose=-1)
lgb_cat.fit(
    X_train_cat, y_train_cat,
    categorical_feature=['cat_feature']
)

print("범주형 특성 처리 결과:")
print(f"  테스트 정확도: {lgb_cat.score(X_test_cat, y_test_cat):.4f}")
```

---

## 6. CatBoost

```python
"""
CatBoost 특징:

1. 범주형 특성 자동 처리:
   - Target Encoding 자동 적용
   - Ordered Target Statistics로 데이터 누수 방지

2. Ordered Boosting:
   - 학습 순서를 랜덤화하여 편향 감소
   - 과적합 방지

3. 대칭 트리:
   - 같은 수준의 모든 노드가 동일한 분할 조건 사용
   - 예측 속도 향상
"""

# pip install catboost
from catboost import CatBoostClassifier, CatBoostRegressor

# CatBoost 분류기
cat_clf = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    l2_leaf_reg=3,           # L2 정규화
    random_state=42,
    verbose=False
)

cat_clf.fit(X_train, y_train)

print("CatBoost 결과:")
print(f"  훈련 정확도: {cat_clf.score(X_train, y_train):.4f}")
print(f"  테스트 정확도: {cat_clf.score(X_test, y_test):.4f}")
```

---

## 7. 부스팅 알고리즘 비교

```python
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import time

# 모델 정의
models = {
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
    'LightGBM': LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
    'CatBoost': CatBoostClassifier(iterations=100, random_state=42, verbose=False)
}

# 비교
print("부스팅 알고리즘 비교:")
print("-" * 60)
print(f"{'모델':<20} {'정확도':>10} {'학습시간(초)':>15}")
print("-" * 60)

results = {}
for name, model in models.items():
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    accuracy = model.score(X_test, y_test)
    results[name] = {'accuracy': accuracy, 'time': train_time}

    print(f"{name:<20} {accuracy:>10.4f} {train_time:>15.4f}")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 정확도 비교
names = list(results.keys())
accuracies = [results[n]['accuracy'] for n in names]
axes[0].barh(names, accuracies)
axes[0].set_xlabel('Accuracy')
axes[0].set_title('Accuracy Comparison')

# 학습 시간 비교
times = [results[n]['time'] for n in names]
axes[1].barh(names, times)
axes[1].set_xlabel('Training Time (seconds)')
axes[1].set_title('Training Time Comparison')

plt.tight_layout()
plt.show()
```

---

## 8. 하이퍼파라미터 튜닝

### 8.1 XGBoost 튜닝

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform, randint

# XGBoost 파라미터 그리드
xgb_param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Grid Search
xgb_grid = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric='logloss'),
    xgb_param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

xgb_grid.fit(X_train, y_train)

print("\nXGBoost Grid Search 결과:")
print(f"  최적 파라미터: {xgb_grid.best_params_}")
print(f"  최적 CV 점수: {xgb_grid.best_score_:.4f}")
print(f"  테스트 점수: {xgb_grid.score(X_test, y_test):.4f}")
```

### 8.2 LightGBM 튜닝

```python
# LightGBM 파라미터 분포 (Randomized Search)
lgb_param_dist = {
    'num_leaves': randint(20, 100),
    'learning_rate': uniform(0.01, 0.3),
    'n_estimators': randint(100, 500),
    'min_child_samples': randint(10, 50),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(0, 1)
}

lgb_random = RandomizedSearchCV(
    LGBMClassifier(random_state=42, verbose=-1),
    lgb_param_dist,
    n_iter=30,
    cv=3,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

lgb_random.fit(X_train, y_train)

print("\nLightGBM Randomized Search 결과:")
print(f"  최적 파라미터: {lgb_random.best_params_}")
print(f"  최적 CV 점수: {lgb_random.best_score_:.4f}")
print(f"  테스트 점수: {lgb_random.score(X_test, y_test):.4f}")
```

### 8.3 Optuna를 이용한 튜닝

```python
# pip install optuna

import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
        'random_state': 42,
        'verbose': -1
    }

    model = LGBMClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
    return scores.mean()

# 최적화 실행
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=50, show_progress_bar=True)

# print(f"최적 파라미터: {study.best_params}")
# print(f"최적 점수: {study.best_value:.4f}")
```

---

## 9. 과적합 방지 전략

```python
"""
부스팅 과적합 방지 전략:

1. 조기 종료:
   - early_stopping_rounds 사용
   - 검증 손실이 개선되지 않으면 중지

2. 정규화:
   - L1 (reg_alpha, lambda_l1)
   - L2 (reg_lambda, lambda_l2)

3. 샘플링:
   - subsample (행 샘플링)
   - colsample_bytree (열 샘플링)

4. 트리 제한:
   - max_depth (깊이 제한)
   - min_samples_leaf / min_child_weight

5. 학습률 조절:
   - learning_rate 낮추기
   - n_estimators 늘리기
"""

# 정규화 효과 비교
reg_params = [
    {'reg_alpha': 0, 'reg_lambda': 0},
    {'reg_alpha': 0.1, 'reg_lambda': 0},
    {'reg_alpha': 0, 'reg_lambda': 1},
    {'reg_alpha': 0.1, 'reg_lambda': 1}
]

print("정규화 효과:")
for params in reg_params:
    xgb_temp = XGBClassifier(
        n_estimators=100,
        max_depth=10,  # 깊은 트리
        random_state=42,
        eval_metric='logloss',
        **params
    )
    xgb_temp.fit(X_train, y_train)
    train_acc = xgb_temp.score(X_train, y_train)
    test_acc = xgb_temp.score(X_test, y_test)
    print(f"  alpha={params['reg_alpha']}, lambda={params['reg_lambda']}: "
          f"Train={train_acc:.4f}, Test={test_acc:.4f}")
```

---

## 10. HistGradientBoosting (sklearn)

```python
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

"""
sklearn의 HistGradientBoosting:
- sklearn 1.0부터 정식 지원
- LightGBM과 유사한 히스토그램 기반 알고리즘
- 대용량 데이터에 효율적
- 결측치 자동 처리
"""

hgb_clf = HistGradientBoostingClassifier(
    max_iter=100,
    learning_rate=0.1,
    max_depth=None,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=0,
    early_stopping='auto',  # 자동 조기 종료
    random_state=42
)

hgb_clf.fit(X_train, y_train)

print("HistGradientBoosting 결과:")
print(f"  훈련 정확도: {hgb_clf.score(X_train, y_train):.4f}")
print(f"  테스트 정확도: {hgb_clf.score(X_test, y_test):.4f}")
```

---

## 연습 문제

### 문제 1: XGBoost 분류
유방암 데이터로 XGBoost를 학습하고 조기 종료를 적용하세요.

```python
from sklearn.datasets import load_breast_cancer
from xgboost import XGBClassifier

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42
)

# 풀이
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

xgb = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    early_stopping_rounds=20,
    eval_metric='logloss',
    random_state=42
)

xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

print(f"최적 반복 횟수: {xgb.best_iteration}")
print(f"테스트 정확도: {xgb.score(X_test, y_test):.4f}")
```

### 문제 2: LightGBM 하이퍼파라미터 튜닝
Grid Search로 LightGBM 최적 파라미터를 찾으세요.

```python
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {
    'num_leaves': [15, 31, 63],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [100, 200]
}

# 풀이
grid = GridSearchCV(
    LGBMClassifier(random_state=42, verbose=-1),
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)
grid.fit(X_train, y_train)

print(f"최적 파라미터: {grid.best_params_}")
print(f"최적 점수: {grid.best_score_:.4f}")
print(f"테스트 점수: {grid.score(X_test, y_test):.4f}")
```

### 문제 3: 앙상블 비교
여러 부스팅 알고리즘을 비교하세요.

```python
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

models = {
    'GB': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGB': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
    'LGB': LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
}

# 풀이
for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name}: {model.score(X_test, y_test):.4f}")
```

---

## 요약

| 알고리즘 | 특징 | 장점 | 단점 |
|----------|------|------|------|
| AdaBoost | 가중치 기반 | 간단, 해석 용이 | 노이즈에 민감 |
| Gradient Boosting | 잔차 학습 | 높은 정확도 | 느린 학습 |
| XGBoost | 정규화 + 병렬화 | 빠름, 정확함 | 메모리 사용 |
| LightGBM | Leaf-wise | 매우 빠름, 대용량 | 과적합 위험 |
| CatBoost | 범주형 처리 | 튜닝 적게 필요 | 느린 시작 |

### 하이퍼파라미터 가이드

| 파라미터 | XGBoost | LightGBM | 효과 |
|----------|---------|----------|------|
| 학습률 | learning_rate | learning_rate | 낮으면 안정적 |
| 트리 수 | n_estimators | n_estimators | 많으면 정확 |
| 깊이 | max_depth | max_depth | 깊으면 복잡 |
| 리프 수 | - | num_leaves | 많으면 복잡 |
| L1 정규화 | reg_alpha | reg_alpha | 과적합 방지 |
| L2 정규화 | reg_lambda | reg_lambda | 과적합 방지 |
| 행 샘플링 | subsample | subsample | 분산 감소 |
| 열 샘플링 | colsample_bytree | colsample_bytree | 다양성 증가 |
