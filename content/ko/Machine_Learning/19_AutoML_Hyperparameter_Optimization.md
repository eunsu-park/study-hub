# AutoML과 하이퍼파라미터 최적화(AutoML and Hyperparameter Optimization)

**이전**: [시계열 머신러닝](./18_Time_Series_ML.md) | **다음**: [이상 탐지](./20_Anomaly_Detection.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 그리드 탐색(grid search), 랜덤 탐색(random search), 베이지안 최적화(Bayesian optimization, TPE), 하이퍼밴드(Hyperband) HPO 방법을 구별할 수 있습니다
2. 목적 함수(objective function), 가지치기(pruning), 다중 목적 최적화(multi-objective optimization)를 갖춘 Optuna 스터디를 구현할 수 있습니다
3. Optuna 시각화(최적화 이력, 파라미터 중요도, 등고선 플롯)를 해석하여 탐색을 안내할 수 있습니다
4. AutoML 프레임워크(Auto-sklearn, FLAML, H2O AutoML)를 비교하고 사용 사례에 맞는 것을 선택할 수 있습니다
5. 테스트 데이터를 보류하고 CV-테스트 차이를 모니터링하여 HPO 과적합(overfitting)을 식별하고 완화할 수 있습니다
6. 로그 스케일(log scale), 조건부 파라미터(conditional parameter), 스텝 크기를 사용하여 효율적인 탐색 공간을 설계할 수 있습니다
7. 다중 목적 최적화(multi-objective optimization)를 적용하여 정확도, 추론 속도, 모델 복잡도 간의 균형을 맞출 수 있습니다

---

하이퍼파라미터를 수동으로 튜닝하는 작업은 번거롭고 인간의 직관에 의한 편향이 개입됩니다. AutoML과 베이지안 최적화(Bayesian optimization)는 이를 체계적인 탐색으로 전환합니다. Optuna는 유망하지 않은 시도를 조기에 가지치기(pruning)하며 파라미터 공간을 지능적으로 탐색하고, FLAML과 H2O 같은 완전 AutoML 프레임워크는 여러 알고리즘을 테스트하고 앙상블(ensemble)을 구성합니다. 이를 통해 전문가 수준의 성능을 훨씬 짧은 시간 안에 달성하는 경우가 많습니다. 이 레슨에서는 기계가 기계를 최적화하도록 만드는 방법을 배웁니다.

---

## 이론과 원리

AutoML과 베이지안 하이퍼파라미터 최적화는 마법이 아닙니다 — Lesson 5의 더 단순한 그리드/무작위 대안에 대한 잘 정의된 트레이드오프를 가진 구체적 알고리즘입니다. 각 방법이 탐색 예산에 실제로 무엇을 하는지 이해하는 것이 주어진 컴퓨팅 봉투에 올바른 도구를 고르게 합니다.

### A. 탐색 알고리즘 위계

Lesson 5가 그리드와 무작위 탐색을 도입했습니다. 베이지안 최적화와 successive halving이 다음 두 단계:

| 방법 | 표본 효율성 | 단계당 오버헤드 | 최선일 때 |
|------|-------------|------------------|-----------|
| 그리드 | 낮음(차원의 저주) | 0 | 적은 하이퍼파라미터, 무한 컴퓨팅 |
| 무작위 | 중간 | 0 | 많은 것 중 적은 중요 하이퍼파라미터 |
| 베이지안 | 높음 | 보통(대리 모델 적합) | 각 평가가 비쌀 때 |
| Successive halving | 높음(나쁜 시도 조기 중지) | 낮음 | 저렴한 부분 평가의 많은 시도 |

비교할 올바른 축은 *표본 효율성* — 거의 최적인 하이퍼파라미터 구성을 찾는 데 몇 시도가 필요한가. 그리드가 가장 나쁘게 스케일(차원에 지수적), 베이지안과 successive halving이 가장 잘 스케일.

### B. 베이지안 최적화: 대리 + 획득

베이지안 최적화(BO)는 하이퍼파라미터에서 검증 점수로의 (미지의) 함수 `f(hp)`를 최대화할 객체로 다룹니다. 절차:

```
몇 개의 무작위 평가로 초기화
루프:
    1. 지금까지의 모든 데이터에 대리 모델 p(f | observed) 적합
    2. hp_next = argmax_{hp} A(hp; surrogate) 선택        ← 획득 함수
    3. f(hp_next) 평가, 관찰에 추가
```

두 성분이 알고리즘을 정의:

**대리 모델**: `hp`에서 예측 `f(hp)`로의 확률적 회귀자. 표준 선택:
- **가우시안 과정(GP)**: 매끄러움, 보정된 사후 `(μ(hp), σ(hp))` 제공. 적합당 `O(N³)` 비용, ~1000 평가를 넘으면 나쁘게 스케일.
- **Tree-structured Parzen Estimator (TPE)**: 조건부 `p(hp | f > threshold)`와 `p(hp | f ≤ threshold)`를 모델링하고, 그 비를 최대화하는 `hp` 선택. Optuna 기본 — 빠르고, 범주형 하이퍼파라미터를 자연스럽게 처리.
- **랜덤 포레스트** 또는 **베이지안 신경망**: 고차원 또는 혼합 탐색 공간의 대안.

**획득 함수**: *탐험*(불확실 영역)과 *활용*(좋다고 예측된 영역) 사이를 균형 잡음. 표준 선택:

```
Expected Improvement (EI):  A(hp) = E[max(0, f(hp) - f*)]    f* = 지금까지 최선
Upper Confidence Bound:     A(hp) = μ(hp) + κ · σ(hp)        κ가 탐험 통제
Probability of Improvement: A(hp) = P(f(hp) > f*)
```

EI가 가장 흔한 기본; "현재 최선보다의 개선, 확률로 가중"을 명시적으로 보상. UCB는 단순하고 원칙적인 대안, 튜닝 손잡이 `κ`가 탐험 가중치로 해석 가능.

함정은 단계당 오버헤드: 대리 모델 적합과 획득 함수 최적화에 시간이 듬. 초 단위로 걸리는 평가(작은 모델, 빠른 추론)에서 오버헤드가 절약을 지배할 수 있음. BO는 각 평가가 분-에서-시간 비용일 때 빛납니다.

### C. Successive Halving과 Hyperband

다른 각도: 시도를 더 영리하게 고르는 대신, *더 많은* 시도를 실행하고 나쁜 것을 일찍 끊음. **Successive halving**(Karnin et al., 2013):

```
1. N개 무작위 구성 표본
2. 각각을 r개 자원(에포크, 표본, 시간, ...)으로 학습
3. 상위 N/η 유지, 남은 시도당 자원 예산 두 배로
4. 1개 구성이 남을 때까지 반복
```

총 예산은 `O(N · r · log_η(N))` — 한 구성을 끝까지 실행하는 것보다 단지 `log` 인수 더 많지만, 일찍 `N`개 구성을 탐색. 나쁜 구성은 많은 자원을 소비하기 전에 죽고; 좋은 구성은 살아남아 더 많이 받음.

위험: 천천히 시작하지만 끝이 좋은 구성이 조기에 죽음. **Hyperband**(Li et al., 2017)는 다른 `(N, r)` 트레이드오프로 여러 successive-halving 브래킷을 실행하고 브래킷 전반의 최선을 취함으로써 이를 헤지. 이 실패 모드에 더 강건, 적당한 추가 비용으로.

scikit-learn의 `HalvingRandomSearchCV`와 `HalvingGridSearchCV`가 successive halving을 직접 구현; Optuna는 TPE와 결합 가능한 *pruner*로 포함.

### D. CASH 문제와 완전 AutoML

**CASH**(Combined Algorithm Selection and Hyperparameter optimization)는 결합 문제: 알고리즘과 그 하이퍼파라미터 둘 다 선택. AutoML 프레임워크는 알고리즘 선택을 최상위 범주형 하이퍼파라미터로 다루고 결합 공간에 대해 BO/Hyperband를 실행하여 CASH를 풀음.

Auto-sklearn, FLAML, H2O AutoML, AutoGluon, TPOT 모두 이 변형을 구현. 차이:

- **Auto-sklearn**: 전체 sklearn 공간에 대한 BO + 메타 학습 웜 스타트(유사한 과거 데이터셋 사용해 유망 구성 시드).
- **FLAML**: 컴퓨팅 예산 vs 정확도 이득을 명시적으로 모델링하는 비용 인식 BO — 저렴하고 좋은 구성을 먼저 선택.
- **AutoGluon**: 공격적 HPO보다 스태킹과 앙상블에 집중; 단순 기본값으로 강한 모델 종종 생성.
- **TPOT**: scikit-learn 파이프라인(전처리 포함)에 대한 유전자 프로그래밍.

트레이드오프는 재현성과 해석성. AutoML이 찾은 파이프라인은 설명하기 어려울 수 있음("탐색이 골랐기 때문에 이 7개 모델의 스택을 이 특정 하이퍼파라미터로 사용"). 규제 또는 이해관계자 직면 작업의 경우, 정확도 비용에도 불구하고 더 작고 정당화된 모델의 수동 선택이 종종 이김.

### E. 다중 충실도 아이디어

Hyperband와 현대 BO 변형 모두 **다중 충실도(multi-fidelity)** 통찰을 활용: `f(hp)`의 저렴한 근사(부분집합 학습, 더 적은 에포크 학습)가 완전 충실도 점수와 상관됨. 저렴한 평가로 필터, 비싼 평가로 확인.

BOHB(Bayesian Optimization + Hyperband)가 둘을 결합: Hyperband가 브래킷에 걸쳐 예산 할당을 결정, BO가 브래킷 안에서 구성을 선택. 비싼 평가의 HPO에 현재 알려진 가장 강한 알고리즘 중 하나.

### F. AutoML을 사용하지 말아야 할 때

AutoML은 다음일 때 가장 유용:
- 데이터가 잘 이해되고 깨끗.
- 모델 클래스가 대체로 정해짐(여기서 부스팅이 선형을 이긴다는 것을 앎).
- 탐색 예산을 감당할 수 있음(한 모델 평가 × 많은 시도).

다음일 때 *덜* 유용:
- 병목이 모델 선택이 아니라 데이터 품질.
- 규제 기관에 완전히 설명할 수 있는 모델이 필요.
- 표본 효율적 알고리즘에서 이익을 보기에 탐색 예산이 너무 작음(그냥 무작위 탐색 사용).
- 도메인 지식이 아키텍처 선택에 강한 사전을 줌.

정직한 요약: AutoML은 모델링 단계에 강력한 도구이지만, 모델링 상류의 문제는 고칠 수 없음.

### From Theory to the Code Below

- 섹션 1의 CASH 정의와 "AutoML 수준" 위계는 (D)의 표현.
- 섹션 2의 Optuna 예제는 (B)의 TPE를 사용; `study.optimize(objective, n_trials=N)` 루프는 베이지안 최적화 점화식.
- 섹션 2의 `optuna.pruners.SuccessiveHalvingPruner` 또는 `MedianPruner`는 (C)의 successive halving 필터.
- 섹션 3의 scikit-learn에서 `HalvingRandomSearchCV`는 (C)의 독립형 successive halving 알고리즘.
- 섹션 4의 FLAML / Auto-sklearn / AutoGluon 예제 각각이 (D)의 CASH 솔버 한 변형을 구현.
- Optuna의 획득 함수 선택(`acquisition_func='EI' | 'UCB' | ...`, 구성 가능할 때)이 (B)에 매핑.

---

## 1. AutoML 생태계

### 1.1 AutoML이 자동화하는 것들

```python
"""
CASH 문제: 알고리즘 선택과 하이퍼파라미터 최적화의 결합(Combined Algorithm Selection and Hyperparameter optimization)

수동 ML 워크플로우:
  데이터 → 전처리 → 피처 엔지니어링 → 모델 선택 → 하이퍼파라미터 튜닝 → 평가
           ↑ AutoML이 이 단계 중 일부 또는 전체를 자동화

AutoML 수준:
1. 하이퍼파라미터 최적화(HPO, Hyperparameter Optimization): 하나의 모델 파라미터 튜닝
   → Optuna, Hyperopt, Ray Tune
2. 모델 선택 + HPO: 튜닝을 포함한 여러 알고리즘 시도
   → Auto-sklearn, FLAML, H2O AutoML
3. 완전 파이프라인 AutoML: 전처리와 피처 엔지니어링 포함
   → Auto-sklearn (전처리 포함), TPOT, AutoGluon

트레이드오프(Trade-off):
  + 시간 절약, 좋은 모델을 빠르게 발견
  + 모델 선택 시 인간의 편향 감소
  - 계산 비용이 높을 수 있음
  - "블랙박스(Black box)" 파이프라인은 디버깅이 어려움
  - 많은 시도로 인해 검증 세트에 과적합될 수 있음
"""
```

### 1.2 HPO 방법론 개요

| 방법 | 설명 | 장점 | 단점 |
|------|------|------|------|
| **그리드 탐색(Grid Search)** | 모든 조합 시도 | 완전 탐색 | 지수적 비용 |
| **랜덤 탐색(Random Search)** | 무작위 파라미터 샘플링 | 고차원에서 그리드보다 우수 | 학습 없음 |
| **베이지안 최적화(Bayesian/TPE)** | 과거 시도에서 학습 | 효율적, 좋은 파라미터 발견 | 시도당 오버헤드 |
| **연속적 반감(Successive Halving)** | 나쁜 설정 조기 중단 | 빠름 | 많은 설정 필요 |
| **하이퍼밴드(Hyperband)** | SH + 적응적 할당 | 매우 빠름 | 복잡함 |
| **집단 기반(Population-Based)** | 진화적 접근 | 넓은 탐색 공간에 적합 | 높은 병렬성 필요 |

---

## 2. Optuna 심화

### 2.1 기본 Optuna 사용법

```python
# pip install optuna
import optuna
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import GradientBoostingRegressor

# 데이터 로드
housing = fetch_california_housing(as_frame=True)
X, y = housing.data, housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def objective(trial):
    """Optuna 목적 함수: 최소화할 값을 반환합니다."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
    }

    model = GradientBoostingRegressor(**params, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    return -scores.mean()  # Optuna는 기본적으로 최소화

# 스터디(study) 생성 및 실행
study = optuna.create_study(direction='minimize', study_name='gb_tuning')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"\n최적 MSE: {study.best_value:.4f}")
print(f"최적 파라미터: {study.best_params}")
```

### 2.2 Optuna 가지치기(Pruning, 조기 중단)

```python
from sklearn.model_selection import StratifiedKFold

def objective_with_pruning(trial):
    """중간 값을 사용하여 가망 없는 시도를 조기에 중단합니다."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
    }

    model = GradientBoostingRegressor(**params, random_state=42)

    # 각 CV 폴드에 대한 중간 값 보고
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) if False else None
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    fold_scores = []
    for step, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]

        model.fit(X_fold_train, y_fold_train)
        score = model.score(X_fold_val, y_fold_val)
        fold_scores.append(score)

        # 중간 값 보고 (R²의 누적 평균)
        trial.report(np.mean(fold_scores), step)

        # 이 시도가 가망 없으면 가지치기
        if trial.should_prune():
            raise optuna.TrialPruned()

    return -np.mean(fold_scores)

# 가지치기 스터디 (MedianPruner는 중앙값보다 나쁜 시도를 중단)
pruned_study = optuna.create_study(
    direction='minimize',
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
)
pruned_study.optimize(objective_with_pruning, n_trials=50, show_progress_bar=True)

# 가지치기된 시도 수 확인
pruned = len([t for t in pruned_study.trials if t.state == optuna.trial.TrialState.PRUNED])
print(f"\n전체 {len(pruned_study.trials)}번 시도 중 {pruned}번 가지치기됨")
print(f"최적 R²: {-pruned_study.best_value:.4f}")
```

### 2.3 Optuna 시각화

```python
import optuna.visualization as vis

# 1. 최적화 이력
fig = vis.plot_optimization_history(study)
fig.show()

# 2. 파라미터 중요도
fig = vis.plot_param_importances(study)
fig.show()

# 3. 병렬 좌표 플롯
fig = vis.plot_parallel_coordinate(study, params=['n_estimators', 'max_depth', 'learning_rate'])
fig.show()

# 4. 등고선 플롯(2D 파라미터 상호작용)
fig = vis.plot_contour(study, params=['learning_rate', 'max_depth'])
fig.show()

# 5. 슬라이스 플롯
fig = vis.plot_slice(study, params=['n_estimators', 'learning_rate', 'max_depth'])
fig.show()

print("Optuna는 Plotly를 통한 풍부한 인터랙티브 시각화를 제공합니다.")
print("이를 활용하여 어떤 파라미터가 가장 중요한지 파악하세요.")
```

### 2.4 다목적 최적화(Multi-Objective Optimization)

```python
def multi_objective(trial):
    """정확도와 모델 크기(추론 속도) 모두를 최적화합니다."""
    n_estimators = trial.suggest_int('n_estimators', 10, 500)
    max_depth = trial.suggest_int('max_depth', 2, 10)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)

    model = GradientBoostingRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, random_state=42
    )
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error')

    mse = -scores.mean()
    complexity = n_estimators * (2 ** max_depth)  # 모델 크기의 대리 지표

    return mse, complexity  # 둘 다 최소화

# 다목적 스터디
mo_study = optuna.create_study(
    directions=['minimize', 'minimize'],
    study_name='multi_objective',
)
mo_study.optimize(multi_objective, n_trials=50, show_progress_bar=True)

# 파레토 프론트(Pareto front)
pareto_trials = mo_study.best_trials
print(f"\n파레토 최적해 수: {len(pareto_trials)}")
for t in pareto_trials[:5]:
    print(f"  MSE={t.values[0]:.4f}, 복잡도={t.values[1]:.0f}, "
          f"n_estimators={t.params['n_estimators']}, max_depth={t.params['max_depth']}")
```

---

## 3. Auto-sklearn

### 3.1 자동화된 모델 선택

```python
"""
Auto-sklearn은 다음을 자동으로 탐색합니다:
  - 15개 분류기 / 14개 회귀기
  - 18개 피처 전처리 방법
  - 각각에 대한 하이퍼파라미터

# pip install auto-sklearn  (Linux 전용, 베이지안 최적화에 SMAC 사용)

import autosklearn.regression
import autosklearn.classification

# 회귀
automl = autosklearn.regression.AutoSklearnRegressor(
    time_left_for_this_task=300,     # 총 시간 예산 (초)
    per_run_time_limit=30,           # 모델당 최대 시간
    n_jobs=-1,
    memory_limit=4096,               # MB
    ensemble_size=5,                 # 상위 모델의 최종 앙상블
    seed=42,
)
automl.fit(X_train, y_train)

# 결과
print(automl.leaderboard())
print(f"테스트 R²: {automl.score(X_test, y_test):.4f}")

# 최종 앙상블 확인
print(automl.show_models())

# 최종 파이프라인 가져오기
for weight, pipeline in automl.get_models_with_weights():
    print(f"가중치: {weight:.3f}")
    print(f"파이프라인: {pipeline}")
"""
print("Auto-sklearn: Linux 환경에 최적, 메타 학습(meta-learning)으로 탐색 워밍업.")
print("주요 기능: 자동 앙상블, 메타 학습, SMAC 베이지안 최적화.")
```

---

## 4. FLAML (빠르고 가벼운 AutoML)

### 4.1 빠른 시작

```python
"""
FLAML은 속도와 낮은 계산 비용을 위해 설계되었습니다.
주요 기능:
  - 비용 효율적 탐색 (저렴한 모델부터 시도)
  - 최적 샘플 크기 학습
  - Auto-sklearn보다 훨씬 빠름

# pip install flaml

from flaml import AutoML

automl = AutoML()
automl.fit(
    X_train, y_train,
    task='regression',
    time_budget=60,           # 총 시간 (초)
    metric='rmse',
    estimator_list=['lgbm', 'xgboost', 'rf', 'extra_tree', 'lrl1'],
    eval_method='cv',
    n_splits=5,
    seed=42,
    verbose=1,
)

# 결과
print(f"최적 모델: {automl.best_estimator}")
print(f"최적 설정: {automl.best_config}")
print(f"최적 RMSE: {automl.best_loss:.4f}")
print(f"테스트 R²: {automl.score(X_test, y_test, metric='r2'):.4f}")

# 피처 중요도 (트리 기반인 경우)
if hasattr(automl.model, 'feature_importances_'):
    import pandas as pd
    importances = pd.Series(
        automl.model.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)
    print(importances.head(10))
"""
print("FLAML: 가장 빠른 AutoML 라이브러리, 시간 제약 실험에 적합.")
print("LightGBM 설정을 빠르게 찾는 데 탁월합니다.")
```

---

## 5. H2O AutoML

### 5.1 H2O AutoML 프레임워크

```python
"""
H2O AutoML은 다양한 모델을 훈련하고 스택 앙상블(stacked ensemble)을 생성합니다.

# pip install h2o

import h2o
from h2o.automl import H2OAutoML

h2o.init()

# H2O Frame으로 변환
train = h2o.H2OFrame(pd.concat([X_train, y_train], axis=1))
test = h2o.H2OFrame(pd.concat([X_test, y_test], axis=1))

# AutoML 실행
aml = H2OAutoML(
    max_runtime_secs=300,
    max_models=20,
    seed=42,
    sort_metric='RMSE',
    exclude_algos=['DeepLearning'],  # 느린 알고리즘 제외
)
aml.train(x=X_train.columns.tolist(), y='MedHouseVal', training_frame=train)

# 리더보드
lb = aml.leaderboard
print(lb.head(10))

# 최적 모델
print(f"최적 모델: {aml.leader.model_id}")
print(f"테스트 성능:")
perf = aml.leader.model_performance(test)
print(perf)

h2o.shutdown()
"""
print("H2O AutoML: 엔터프라이즈 수준, 우수한 스택 앙상블.")
print("대규모 데이터셋을 위한 분산 컴퓨팅 지원.")
```

---

## 6. AutoML 프레임워크 비교

### 6.1 프레임워크 비교표

| 기능 | Optuna | Auto-sklearn | FLAML | H2O AutoML |
|------|--------|-------------|-------|------------|
| **유형** | HPO 프레임워크 | 완전 AutoML | 완전 AutoML | 완전 AutoML |
| **속도** | 목적 함수에 의존 | 느림 | 매우 빠름 | 보통 |
| **모델 선택** | 수동 (직접 선택) | 자동 | 자동 | 자동 |
| **앙상블** | 수동 | 내장 | 수동 | 스태킹 |
| **플랫폼** | 모두 | Linux 전용 | 모두 | 모두 (JVM) |
| **학습 곡선** | 보통 | 쉬움 | 쉬움 | 쉬움 |
| **최적 용도** | 커스텀 HPO, 세밀한 제어 | Linux 연구 | 빠른 실험 | 프로덕션, 대용량 데이터 |
| **GPU 지원** | 사용자 코드 통해 | 없음 | LightGBM/XGBoost | 있음 |

### 6.2 상황별 선택 가이드

```python
"""
결정 가이드:

1. 완전한 제어권과 커스텀 목적 함수가 필요한 경우?
   → Optuna (가장 유연한 HPO 프레임워크)

2. 어떤 플랫폼에서든 빠른 베이스라인이 필요한 경우?
   → FLAML (가장 빠르고 어디서나 작동)

3. Linux에서 연구, 메타 학습 원하는 경우?
   → Auto-sklearn (최고의 메타 학습, 강건한 앙상블)

4. 대용량 데이터셋, 프로덕션 배포가 필요한 경우?
   → H2O AutoML (분산, 엔터프라이즈 기능)

5. 딥러닝(Deep Learning) HPO?
   → Optuna + PyTorch/TensorFlow
   → Ray Tune (분산 HPO)

6. 캐글(Kaggle) 대회?
   → Optuna로 XGBoost/LightGBM 튜닝 (우승자들이 가장 많이 사용)
"""
```

---

## 7. 모범 사례

### 7.1 HPO에서의 과적합 방지

```python
"""
HPO 과적합(HPO overfitting): 시도 횟수가 많을수록, 특정 분할에서
운이 좋은 설정을 찾아 검증 세트에 "과적합"될 가능성이 높아집니다.

완화 방법:
1. 견고한 교차 검증(CV) 사용 (5-폴드 이상)
2. HPO가 절대 보지 못하는 최종 테스트 세트 보류
3. 시도 횟수 제한 (100-200번 이후 수익 감소)
4. 조기 중단 사용 (Optuna 가지치기)
5. 최고 점수만이 아닌 신뢰 구간 보고
"""

# 예시: HPO 과적합 확인
import optuna

# 스터디 실행
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100, show_progress_bar=True)

# 전체 훈련 세트로 최적 모델 훈련
best_params = study.best_params
best_model = GradientBoostingRegressor(**best_params, random_state=42)
best_model.fit(X_train, y_train)

# 검증 점수 vs 테스트 점수 비교
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

cv_mse = -cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error').mean()
test_mse = mean_squared_error(y_test, best_model.predict(X_test))

print(f"CV MSE (HPO 중 확인):     {cv_mse:.4f}")
print(f"테스트 MSE (미확인):       {test_mse:.4f}")
print(f"격차 (잠재적 과적합):      {abs(cv_mse - test_mse):.4f}")
```

### 7.2 효율적인 탐색 공간

```python
"""
좋은 탐색 공간 정의를 위한 팁:

1. 학습률과 정규화에는 로그 스케일 사용:
   trial.suggest_float('lr', 1e-5, 1e-1, log=True)

2. 이산 선택에는 정수 사용:
   trial.suggest_int('n_estimators', 50, 500, step=50)

3. 알고리즘 선택에는 범주형(categorical) 사용:
   trial.suggest_categorical('model', ['rf', 'xgb', 'lgbm'])

4. 조건부 파라미터:
   model_name = trial.suggest_categorical('model', ['rf', 'svm'])
   if model_name == 'rf':
       n_estimators = trial.suggest_int('rf_n_estimators', 50, 500)
   elif model_name == 'svm':
       C = trial.suggest_float('svm_C', 0.01, 100, log=True)

5. 넓은 범위로 시작한 후 결과를 바탕으로 좁혀 나가세요.

6. Optuna의 파라미터 중요도를 활용해 중요하지 않은 파라미터를 제거하세요.
"""
```

---

## 8. 연습 문제

### 연습 1: 분류를 위한 Optuna

```python
"""
1. 유방암(breast cancer) 데이터셋을 로드합니다.
2. 다음을 탐색하는 Optuna 목적 함수를 정의합니다:
   - 알고리즘: RandomForest vs GradientBoosting vs SVM
   - 각각에 대한 조건부 하이퍼파라미터
3. F1 점수를 지표로 사용합니다.
4. 가지치기를 사용하여 100번 시도합니다.
5. 시각화: 최적화 이력, 파라미터 중요도, 등고선 플롯.
6. 최적 Optuna 모델과 기본 RandomForest를 비교합니다.
"""
```

### 연습 2: AutoML 비교

```python
"""
1. California Housing 데이터셋을 로드합니다.
2. 60초 예산으로 FLAML을 실행합니다.
3. 동일한 시간 예산으로 Optuna (GradientBoosting만)를 실행합니다.
4. 다음을 비교합니다:
   a) 최적 모델 성능 (R², RMSE)
   b) 평가된 모델 수
   c) 최종 모델 복잡도
5. 어느 방법이 더 나은 모델을 찾았나요? 이유는?
"""
```

### 연습 3: 다목적 HPO

```python
"""
1. Optuna 다목적으로 다음을 최적화합니다:
   - 목적 1: 예측 오차 최소화 (RMSE)
   - 목적 2: 훈련 시간 최소화
   - 목적 3: 모델 크기 최소화 (파라미터 수)
2. 파레토 프론트(Pareto front)를 플롯합니다.
3. 정확도와 속도의 균형을 맞추는 모델을 파레토 프론트에서 선택합니다.
4. 실제 배포 시나리오에 대한 선택을 정당화합니다.
"""
```

---

## 9. 요약

### 핵심 정리

| 개념 | 설명 |
|------|------|
| **Optuna** | TPE, 가지치기, 다목적 최적화를 갖춘 유연한 HPO |
| **Auto-sklearn** | 메타 학습을 갖춘 완전 파이프라인 AutoML (Linux) |
| **FLAML** | 가장 빠른 AutoML, 비용 효율적 탐색 |
| **H2O AutoML** | 스택 앙상블을 갖춘 프로덕션급 AutoML |
| **가지치기(Pruning)** | 나쁜 시도를 조기에 중단하여 계산 절약 |
| **다목적 최적화** | 정확도 vs 속도 vs 복잡도 최적화 |
| **HPO 과적합** | 더 많은 시도 ≠ 더 나은 일반화 |

### 모범 사례

1. **항상 테스트 세트를 보류**하여 HPO가 절대 보지 못하게 하세요
2. **빠른 베이스라인을 위해 FLAML로 시작**하세요 (수 분 내)
3. **세밀한 제어가 필요할 때 Optuna를 사용**하세요
4. **시도 횟수를 제한**하세요 — 100-200번 이후 수익이 감소합니다
5. **학습률과 정규화 파라미터에는 로그 스케일** 사용
6. **가지치기를 사용**하여 나쁜 설정을 조기에 제거하세요
7. **CV와 테스트 성능을 비교**하여 과적합을 확인하세요

### 다음 단계

- **L20**: 이상 탐지(Anomaly Detection) — 이상치와 비정상 패턴 탐지
- **MLOps L03-04**: 실험 추적을 위한 MLflow — Optuna 스터디를 체계적으로 기록
