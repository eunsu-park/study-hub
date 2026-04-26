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

## 이론과 원리

부스팅은 "약한 학습기들을 차례로 적합한다"고 읽힙니다 — 하지만 이 패턴을 따르는 알고리즘들(AdaBoost, Gradient Boosting, XGBoost, LightGBM)은 훨씬 더 날카로운 아이디어로 통합됩니다: 각각이 약한 학습기들을 단계로 사용하여 특정 손실에 대해 **함수 공간 경사 하강(functional gradient descent)**을 수행합니다. 이를 보고 나면, 그들 사이의 차이 — 지수 손실 vs 임의 손실, 1차 vs 2차, GOSS vs EFB — 가 같은 수학적 골격 위의 공학적 선택이 됩니다.

### A. AdaBoost: 지수 손실 최소화

AdaBoost(Freund & Schapire, 1995)는 약한 학습기 `h_m(x) ∈ {-1, +1}`를 순차적으로 적합하고 다음과 같이 결합합니다:

```
F_M(x) = Σ_{m=1..M}  α_m · h_m(x)        예측 = sign(F_M(x))
```

학습 알고리즘:

```
표본 가중치 w_i = 1/N로 초기화
for m = 1..M:
    가중 학습 데이터에 h_m을 적합        ← 가중치 w_i
    err_m = Σ w_i · 1{h_m(x_i) ≠ y_i} / Σ w_i
    α_m   = ½ · log((1 - err_m) / err_m)      ← 단계 가중치
    w_i  ← w_i · exp(-α_m · y_i · h_m(x_i))   ← 재가중
```

재가중 규칙이 결정적입니다. 잘못 분류된 예제는 `y · h(x) = -1`이므로 가중치가 `exp(α_m) > 1`로 곱해집니다. 올바르게 분류된 예제는 `exp(-α_m) < 1`로 곱해집니다. 다음 학습기는 여전히 틀린 예제에 집중하도록 강제됩니다.

더 깊은 사실은 전체 절차가 forward stagewise additive modeling을 통한 **지수 손실** `L(F) = Σ exp(-y_i · F(x_i))`의 탐욕적 최소화와 *동등*하다는 것입니다. `α_m` 공식은 지수 손실에 대한 닫힌 형태의 라인 탐색이고; 가중치 갱신은 그 손실의 그래디언트에서 떨어집니다. AdaBoost는 이 관점 이전에 발견되었지만, 지수 손실 최소화로 보는 것이 임의 손실로의 일반화를 가능하게 합니다.

### B. Gradient Boosting: 임의 손실에 대한 함수 공간 경사 하강

Friedman(2001)은 AdaBoost를 일반화했습니다: 지수 손실을 *임의의* 미분 가능한 손실 `L(y, F(x))`로 대체하고 반복:

```
F_0(x) = argmin_c  Σ L(y_i, c)              ← 상수 초기 예측
for m = 1..M:
    r_im = -[ ∂L(y_i, F(x_i)) / ∂F(x_i) ]_{F=F_{m-1}}    ← 음의 그래디언트
    잔차 r_im을 예측하도록 h_m을 적합        ← 회귀 트리
    γ_m = argmin_γ  Σ L(y_i, F_{m-1}(x_i) + γ · h_m(x_i))   ← 라인 탐색
    F_m(x) = F_{m-1}(x) + ν · γ_m · h_m(x)   ← 수축(학습률 ν)
```

이것이 경사 하강입니다 — 단 매개변수 공간이 아니라 **함수 공간**에서. 반복 `m`의 "단계"는 함수 `h_m`이고, 음의 그래디언트는 `h_m`이 예측하도록 적합되는 **유사 잔차(pseudo-residual)** `r_im`입니다. 제곱 손실에 대해 음의 그래디언트는 그냥 보통 잔차 `y_i - F(x_i)`. 로그 손실에 대해서는 `y_i - p_i`. 같은 알고리즘이 회귀, 분류, 순위, 그리고 미분 가능한 어떤 손실이든 처리합니다.

**수축(shrinkage)** 매개변수 `ν`(학습률, 보통 0.05–0.1)는 각 단계의 크기를 통제합니다. `ν`가 작으면 더 많은 트리 `M`이 필요하지만 더 잘 일반화됩니다 — 어떤 경사 하강에서나 학습률과 같은 트레이드오프.

### C. XGBoost: 2차 테일러 + 명시적 정칙화

XGBoost(Chen & Guestrin, 2016)는 그래디언트 부스팅을 경진대회 지배 알고리즘으로 만든 두 가지 변경을 했습니다:

**1. 2차 테일러 근사.** `L(y_i, F_{m-1} + h_m)`을 2차로 전개:

```
L(y_i, F_{m-1} + h_m(x_i))  ≈  L(y_i, F_{m-1}(x_i)) + g_i · h_m(x_i) + ½ · h_i · h_m(x_i)²
```

여기서 `g_i = ∂L/∂F`, `h_i = ∂²L/∂F²`(표본 `i`의 그래디언트와 헤시안). 제곱 손실이면 `h_i = 1`이고 보통 GBM이 회복됩니다. 로그 손실이면 `h_i = p(1-p)` — 자연스럽게 `p = 0.5` 근처에서 크고 확신 있는 예측에서 작아, 알고리즘이 불확실한 예제에 용량을 씁니다.

**2. 명시적 정칙화.** 트리당 복잡도 페널티 추가:

```
Ω(h) = γ · T + ½ · λ · ‖w‖²
```

`T`는 리프 수, `w`는 리프 점수, `γ`와 `λ`는 페널티. 트리 `m`의 목적은:

```
Obj^{(m)} = Σ_i [ g_i · h_m(x_i) + ½ · h_i · h_m(x_i)² ] + Ω(h_m)
```

리프 점수 최적화는 닫힌 형태 `w_j* = -G_j / (H_j + λ)`(여기서 `G_j`, `H_j`는 리프 `j`의 합산 그래디언트/헤시안)를 가지며, 후보 분할의 이득이 분석적으로 계산 가능합니다 — 트리를 학습시켜 테스트할 필요가 없음. 이것이 XGBoost를 빠르고 정확하게 만드는 것입니다.

### D. LightGBM: 대규모를 위한 GOSS와 EFB

LightGBM(Ke et al., 2017)은 XGBoost의 수학적 핵심을 보존하면서 매우 큰 데이터셋을 위한 두 가지 공학 트릭을 추가합니다:

**GOSS (Gradient-based One-Side Sampling).** 작은 그래디언트를 갖는 예제는 이미 잘 적합되어 있고; 큰 그래디언트를 갖는 예제가 다음 단계를 지배합니다. GOSS는 큰 그래디언트 예제의 `top-a%`를 모두 유지하고, 나머지의 `b%`를 무작위로 표본하며, 그래디언트 추정을 비편향으로 유지하도록 재가중합니다. 결과: 반복당 데이터의 `1 - (1-a-b)`로 비슷한 정확도.

**EFB (Exclusive Feature Bundling).** 희소 고차원 데이터(원-핫 인코딩된 범주형, 텍스트 등)에서 많은 특성이 상호 배타적입니다 — 같은 행에서 절대 동시에 0이 아닌 적이 없습니다. EFB는 그런 특성들을 단일 "번들" 특성으로 패킹하여 정보 손실 없이 유효 차원을 줄입니다. 분할 비용이 `O(#특성)`에서 `O(#번들)`로 떨어집니다.

LightGBM은 또한 XGBoost의 기본 레벨별 성장 대신 **리프별(leaf-wise)** 트리 성장(항상 최대 손실 감소를 갖는 리프를 분할)을 기본으로 합니다. 리프별은 같은 크기의 더 정확한 트리를 만들지만 작은 데이터셋에서 더 과적합되기 쉬움 — `max_depth` 또는 `num_leaves`로 가드.

### E. 통합 그림

이 레슨의 모든 부스팅 알고리즘은 함수 공간 경사 하강이며, 세 축을 따라 맞춤화됩니다:

| 축 | AdaBoost | GBM | XGBoost | LightGBM |
|----|----------|-----|---------|----------|
| 손실 | 지수(고정) | 미분 가능한 모든 것 | 미분 가능한 모든 것 | 미분 가능한 모든 것 |
| 차수 | 1차(≡ 1차 GD) | 1차 | 2차 테일러 | 2차 테일러 |
| 정칙화 | 암묵적(조기 종료) | 암묵적 | 명시적 `γ T + λ‖w‖²` | 명시적 |
| 표본추출 | 재가중 | 부분표본(선택) | 부분표본(선택) | GOSS |
| 공학 | 없음 | 없음 | 캐시 인식 분할 탐색 | GOSS + EFB + 리프별 |

부스팅은 각 새 트리가 잔차 오차를 명시적으로 겨냥하므로 편향을 줄입니다. `M`이 너무 크거나 `ν`가 너무 작으면 분산이 자랄 수 있음 — 검증셋에서 조기 종료가 표준 가드.

### From Theory to the Code Below

- 섹션 1.2의 AdaBoost 예제는 (A)의 루프를 구현합니다; `learning_rate` 매개변수는 `α_m`의 곱셈자입니다.
- 섹션 2의 `GradientBoostingClassifier`는 (B)의 알고리즘입니다; `learning_rate`는 `ν`, `loss`는 음의 그래디언트가 잔차를 정의하는 미분 가능한 손실을 선택합니다.
- 섹션 3의 `XGBClassifier`는 (C)의 2차 목적을 사용합니다; `reg_alpha`는 L1, `reg_lambda`는 리프 점수에 대한 L2 정칙화; `gamma`는 리프당 복잡도 페널티.
- 섹션 4의 `LGBMClassifier`는 같은 XGBoost 스타일 목적 위에 (D)의 GOSS/EFB를 추가합니다; `num_leaves`는 리프별 성장 예산을 통제.
- 그들 모두에 걸친 `early_stopping_rounds` 인자는 (E)에서 언급된 "너무 많은 반복 ⟹ 과적합" 실패 모드에 대한 실용적 방어입니다.

---

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
