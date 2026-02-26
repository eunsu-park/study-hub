# 고급 앙상블 기법 — 스태킹(Stacking)과 블렌딩(Blending)

[← 이전: 20. 이상 탐지](20_Anomaly_Detection.md) | [다음: 개요 →](00_Overview.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 편향-분산 트레이드오프(Bias-Variance Tradeoff)와 다양한 학습기 조합 관점에서 스태킹이 작동하는 이유 설명
2. 레벨-0 기저 학습기(Base Learner)와 레벨-1 메타 학습기(Meta-Learner)로 구성된 스태킹 아키텍처를 설계하고 각 층의 역할 이해
3. 교차 검증 기반 학습을 적용한 scikit-learn의 `StackingClassifier`와 `StackingRegressor`로 스태킹 구현
4. 스태킹(CV 기반)과 블렌딩(홀드아웃 기반)의 차이를 구분하고 데이터 크기에 따라 적절한 방법 선택
5. 상관 행렬을 사용하여 기저 학습기 간 예측 다양성을 측정하고 이를 통해 상호 보완적인 모델 선정
6. 스태킹 앙상블에 특화된 하이퍼파라미터 최적화 전략 적용
7. 스태킹의 효과가 감소하는 시점과 단순한 앙상블이 더 적합한 경우 인식

---

배깅(Bagging, L07)과 부스팅(Boosting, L08)이라는 두 가지 주요 앙상블 전략을 이미 학습했습니다. 배깅은 부트스트랩 샘플로 학습된 독립 모델의 평균을 내어 분산(Variance)을 줄이고, 부스팅은 오류를 순차적으로 수정하여 편향(Bias)을 줄입니다. 스태킹(Stacking)은 근본적으로 다른 접근 방식을 취합니다 — 다양한 모델을 *어떻게 결합할지* 학습합니다. 평균이나 가중 투표 같은 고정 규칙 대신, 스태킹은 두 번째 수준의 모델(메타 학습기)을 학습시켜 기저 모델 예측의 최적 조합을 발견합니다. 이 레슨에서는 밀접하게 관련된 블렌딩 기법을 포함하여 스태킹 앙상블을 구축하고 튜닝하며 비판적으로 평가하는 방법을 배웁니다. 레슨이 끝나면 모델을 스태킹하는 방법뿐만 아니라 — 동등하게 중요한 — 스태킹을 사용하지 말아야 할 때도 이해하게 될 것입니다.

---

> **비유**: 스태킹은 전문가 심사위원단과 같습니다 — 각 심사위원(기저 모델)은 독립적으로 점수를 매기고, 수석 심사위원(메타 학습기)이 모든 의견을 종합하여 최종 결정을 내립니다. 유능한 수석 심사위원은 심사위원 A가 기술적 메리트를 발견하는 데는 탁월하지만 예술적 인상에는 지나치게 관대하고, 심사위원 B는 그 반대라는 것을 압니다. 수석 심사위원은 단순히 점수를 평균하는 것이 아니라, 각 심사위원의 강점과 약점을 학습하여 어떤 단일 심사위원보다 더 정확한 최종 점수를 도출합니다.

---

## 1. 스태킹이 작동하는 이유

### 1.1 편향-분산 관점

서로 다른 모델 계열은 서로 다른 편향-분산 프로파일을 가집니다:

```python
"""
모델별 편향-분산 프로파일:

┌─────────────────────┬──────────┬──────────┐
│ 모델                 │ 편향     │ 분산     │
├─────────────────────┼──────────┼──────────┤
│ Linear Regression   │ High     │ Low      │
│ k-NN (small k)      │ Low      │ High     │
│ Random Forest       │ Low      │ Medium   │
│ SVM (RBF kernel)    │ Low      │ Medium   │
│ Gradient Boosting   │ Low      │ Med-High │
└─────────────────────┴──────────┴──────────┘

배깅: 분산 감소 (독립 예측의 평균화)
부스팅: 편향 감소 (순차적 오류 수정)
스태킹: 둘 다 감소 — 서로 다른 편향-분산 프로파일의 모델을 결합하고,
        메타 학습기가 최적 가중치를 학습합니다.
"""
```

### 1.2 다양성이 핵심

스태킹이 작동하는 이유는 서로 다른 모델이 *서로 다른 오류*를 만들기 때문입니다. 모든 기저 모델이 동일한 예측을 한다면, 이를 결합해도 아무런 가치가 없습니다. 핵심은 **다양성(Diversity)** — 각 모델이 데이터에서 서로 다른 패턴을 포착하는 것입니다.

다양성의 세 가지 원천:

1. **알고리즘 다양성(Algorithm Diversity)**: 서로 다른 모델 계열 (선형, 트리 기반, 인스턴스 기반)
2. **데이터 다양성(Data Diversity)**: 서로 다른 특성 부분집합 또는 서로 다른 데이터 뷰
3. **하이퍼파라미터 다양성(Hyperparameter Diversity)**: 동일한 알고리즘에 서로 다른 설정

```python
"""
세 모델이 10개 샘플의 이진 결과를 예측한다고 가정합니다.

모델 A: [1, 1, 0, 0, 1, 1, 0, 0, 1, 1]  (샘플 3, 7에서 오류)
모델 B: [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  (샘플 2, 6에서 오류)
모델 C: [1, 1, 1, 0, 0, 1, 0, 0, 1, 1]  (샘플 5, 8에서 오류)
정답:   [1, 1, 1, 0, 1, 1, 0, 0, 1, 1]

다수결 투표: [1, 1, 1, 0, 1, 1, 0, 0, 1, 1] → 완벽!

각 모델의 오류율은 20%이지만, 오류가 겹치지 않습니다.
메타 학습기는 각 모델이 잘하는 부분을 신뢰하도록 학습할 수 있습니다.
"""
```

### 1.3 모호성 분해(Ambiguity Decomposition)

결합된 예측기의 기대 오류 (Krogh & Vedelsby, 1995):

```
E[error_combined] ≤ (1/N) × Σ E[error_i] - diversity_term

여기서:
- N = 기저 모델의 수
- diversity_term = 모델 간 평균 쌍별 불일치

핵심 통찰: 다양성이 높을수록 → 빼는 항이 커짐 → 결합 오류가 낮아짐
```

---

## 2. 스태킹 아키텍처

### 2.1 2단계 구조

```
레벨 0 (기저 학습기):
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│ Model 1  │  │ Model 2  │  │ Model 3  │  │ Model N  │
│ (e.g.LR) │  │ (e.g.RF) │  │ (e.g.SVM)│  │ (e.g.KNN)│
└────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
     │              │              │              │
     ▼              ▼              ▼              ▼
   pred_1         pred_2         pred_3         pred_N
     │              │              │              │
     └──────────────┴──────┬───────┴──────────────┘
                           │
                    ┌──────▼──────┐
레벨 1             │ Meta-Learner │
(메타 학습기):     │ (e.g. LR)   │
                    └──────┬──────┘
                           │
                     최종 예측
```

**레벨 0 (기저 학습기, Base Learners)**: 각각 데이터의 서로 다른 측면을 학습하는 다양한 모델들.

**레벨 1 (메타 학습기, Meta-Learner)**: 기저 학습기의 예측을 입력 특성으로 받아 최적의 결합 방법을 학습하는 모델.

### 2.2 왜 단순한 메타 학습기를 사용하는가?

메타 학습기는 매우 작은 특성 공간(기저 모델당 특성 하나)에서 작동합니다. 정규화된 선형 모델이 표준 선택인 이유:

```python
"""
1. 작은 특성 공간: 기저 모델이 4개이면 메타 학습기의 입력은 4개뿐입니다.
   계수가 4개인 선형 모델이 이 차원에 적합합니다.

2. 자연스러운 정규화: 로지스틱/릿지 회귀는 자동으로
   결합 가중치를 정규화하여 과적합을 방지합니다.

3. 해석 가능성: 메타 학습기의 계수가 각 기저 모델의
   기여도를 직접 알려줍니다:
   메타-LR 가중치: [0.35, 0.30, 0.25, 0.10]
     → RF 35%, XGBoost 30%, SVM 25%, KNN 10%

4. 예외: 기저 모델이 많은 경우(50+개) 경진대회에서는
   비선형 메타 학습기가 예측 간 상호작용을 포착할 수 있습니다.
"""
```

---

## 3. 교차 검증 스태킹 (데이터 누수 방지)

### 3.1 데이터 누수 문제

단순한 접근 방식은 전체 훈련 세트에서 기저 모델을 학습시키고, 동일한 데이터에 대한 예측을 메타 특성으로 사용합니다. 이는 **데이터 누수(Data Leakage)**를 유발합니다: 모델이 이미 해당 데이터를 보았기 때문에 예측이 지나치게 낙관적이 됩니다.

### 3.2 교차 검증을 통한 해결

표준 접근법은 K-폴드 CV를 사용하여 "폴드 외(Out-of-Fold, OOF)" 예측을 생성합니다:

```
훈련 데이터: [폴드 1 | 폴드 2 | 폴드 3 | 폴드 4 | 폴드 5]

각 기저 모델에 대해:
  반복 1: [2,3,4,5]로 학습 → 폴드 1 예측
  반복 2: [1,3,4,5]로 학습 → 폴드 2 예측
  ...
  반복 5: [1,2,3,4]로 학습 → 폴드 5 예측

OOF 예측 연결 → 모든 훈련 샘플의 메타 특성
각 예측은 해당 훈련 샘플을 보지 않고 생성되었습니다!
```

### 3.3 sklearn을 활용한 구현

```python
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

X, y = make_classification(
    n_samples=2000, n_features=20, n_informative=15,
    n_redundant=3, random_state=42
)

# 이유: 근본적으로 서로 다른 학습 전략을 가진 모델을 선택합니다.
# LR은 선형 경계를, RF는 트리로 특성 공간을 분할하고,
# SVM은 고차원 공간으로 매핑하며, KNN은 로컬 이웃을 사용합니다.
estimators = [
    ('lr', LogisticRegression(max_iter=1000, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(probability=True, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=10))
]

# 이유: 메타 학습기로 LogisticRegression — 메타 특성이 4개뿐입니다.
# cv=5는 폴드 외 예측으로 데이터 누수를 방지합니다.
stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    stack_method='auto',     # predict_proba가 있으면 사용
    passthrough=False        # 원본 특성 포함하지 않음
)

scores = cross_val_score(stacking_clf, X, y, cv=5, scoring='accuracy')
print(f"Stacking CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### 3.4 StackingRegressor

```python
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import make_regression

X_reg, y_reg = make_regression(
    n_samples=2000, n_features=20, n_informative=15, noise=20, random_state=42
)

# 이유: 메타 학습기로 Ridge는 L2 정규화를 추가하여,
# 기저 모델들이 상관된 경우 어느 단일 기저 모델이 지배하는 것을 방지합니다.
stacking_reg = StackingRegressor(
    estimators=[
        ('ridge', Ridge(alpha=1.0)),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('svr', SVR(kernel='rbf')),
        ('knn', KNeighborsRegressor(n_neighbors=10))
    ],
    final_estimator=Ridge(alpha=1.0),
    cv=5
)

reg_scores = cross_val_score(stacking_reg, X_reg, y_reg, cv=5, scoring='r2')
print(f"Stacking CV R²: {reg_scores.mean():.4f} (+/- {reg_scores.std():.4f})")
```

### 3.5 `passthrough` 옵션

```python
"""
passthrough=False (기본값):
  메타 특성 = [pred_model1, pred_model2, ..., pred_modelN]

passthrough=True:
  메타 특성 = [pred_model1, ..., pred_modelN, feature_1, ..., feature_20]

passthrough=True를 사용하는 경우:
  ✓ 기저 모델이 약하여 중요한 특성을 놓칠 때
  ✓ 더 큰 특성 공간을 지원하기에 충분한 데이터가 있을 때

passthrough=False를 유지하는 경우:
  ✓ 기저 모델이 특성 정보를 이미 잘 포착할 때
  ✓ 데이터셋이 작을 때 (더 많은 메타 특성으로 과적합 위험)
"""
```

---

## 4. 블렌딩(Blending) vs. 스태킹(Stacking)

### 4.1 블렌딩이란?

블렌딩은 교차 검증 대신 단일 홀드아웃 세트를 사용합니다:

```
훈련 데이터 → 분할 → 훈련 (80%) + 블렌드 홀드아웃 (20%)
                          │                    │
                   기저 모델 학습         홀드아웃에서 예측
                          │                    │
                          │           홀드아웃 예측으로
                          │           메타 학습기 학습
                          │                    │
                   테스트 데이터: 기저 모델이 예측 →
                   메타 학습기가 결합 → 최종 예측
```

### 4.2 비교

| 측면 | 스태킹(CV 기반) | 블렌딩(홀드아웃 기반) |
|------|----------------|----------------------|
| **데이터 효율성** | 높음 — 전체 데이터 사용 | 낮음 — 20-30% 손실 |
| **연산량** | K배 더 비쌈 | 빠름 (단일 학습) |
| **분산** | 낮음 (폴드에 걸쳐 평균화) | 높음 (단일 분할) |
| **적합한 경우** | 소-중형 데이터셋 | 대형 데이터셋 |

### 4.3 블렌딩 직접 구현

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def blend_models(X_train, y_train, X_test, y_test, base_models, meta_model,
                 blend_ratio=0.2, random_state=42):
    """수동 블렌딩: 스태킹보다 단순하며, 속도를 위해 데이터를 일부 희생합니다."""
    # 단계 1: 훈련 세트와 블렌드 세트로 분할
    X_tr, X_blend, y_tr, y_blend = train_test_split(
        X_train, y_train, test_size=blend_ratio, random_state=random_state,
        stratify=y_train  # 이유: 두 분할 모두에서 클래스 균형을 유지합니다
    )

    # 단계 2: 기저 모델 학습, 블렌드 + 테스트에서 예측
    blend_meta = np.zeros((len(X_blend), len(base_models)))
    test_meta = np.zeros((len(X_test), len(base_models)))

    for i, (name, model) in enumerate(base_models):
        model.fit(X_tr, y_tr)
        # 이유: predict_proba[:, 1]은 하드 레이블보다 더 많은 정보를 담습니다.
        # 메타 학습기는 "SVM이 0.51을 예측하면 불확실하다"는 것을 학습할 수 있습니다.
        if hasattr(model, 'predict_proba'):
            blend_meta[:, i] = model.predict_proba(X_blend)[:, 1]
            test_meta[:, i] = model.predict_proba(X_test)[:, 1]
        else:
            blend_meta[:, i] = model.predict(X_blend)
            test_meta[:, i] = model.predict(X_test)

    # 단계 3: 블렌드 예측으로 메타 학습기 학습
    meta_model.fit(blend_meta, y_blend)

    # 단계 4: 최종 예측
    return meta_model.predict(test_meta), accuracy_score(y_test, meta_model.predict(test_meta))
```

---

## 5. 다단계 스태킹과 특성 가중 앙상블

### 5.1 2단계를 넘어서

다단계 스태킹은 추가적인 메타 학습기 레이어를 추가합니다:

```
레벨 0: [LR, RF, SVM, KNN, XGBoost]   ← 다양한 기저 모델
레벨 1: [Ridge, RF_small]              ← 첫 번째 메타 레이어
레벨 2: [LogisticRegression]           ← 최종 메타 학습기
```

**실용적 지침**:
- 2단계가 최적 (단일 모델 대비 1-3% 향상)
- 3단계: 소폭 추가 향상 (0.1-0.5%), 경진대회용으로만 의미 있음
- 4단계 이상: 복잡성 대비 거의 가치 없음 — 과적합 위험 증가

### 5.2 메타 특성 엔지니어링

원시 예측 이외에도 더 풍부한 메타 특성을 생성할 수 있습니다:

```python
"""
기저 모델별 고급 메타 특성:
  - predict_proba[:, 1]          → 양성 클래스의 확률
  - max(predict_proba)           → 모델 신뢰도
  - -Σ p_i log(p_i)             → 예측 엔트로피 (불확실성)
  - pred_model_A - pred_model_B  → 쌍별 불일치 신호

이것이 도움이 되는 이유: 메타 학습기는 "모델 A는 확신하지만
모델 B가 반대로 예측할 때, 모델 B를 신뢰하라"는 것을 학습할 수 있습니다 —
단순한 확률 특성만으로는 표현할 수 없는 정보입니다.
"""
```

---

## 6. 다양한 기저 학습기 선택

### 6.1 예측 다양성 측정

가장 실용적인 다양성 척도는 **OOF 예측의 상관 행렬(Correlation Matrix)**입니다:

```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict

def plot_prediction_correlation(models, X, y, cv=5):
    """
    이것이 중요한 이유: 높은 상관관계는 모델들이 유사한 오류를 만든다는 의미입니다.
    높은 상관관계의 모델을 추가하면 다양성이 아닌 중복성이 증가합니다.
    쌍별 상관관계 < 0.7을 목표로 합니다.
    """
    predictions = {}
    for name, model in models:
        predictions[name] = cross_val_predict(model, X, y, cv=cv)

    corr = np.corrcoef(np.array(list(predictions.values())))
    sns.heatmap(corr, xticklabels=list(predictions.keys()),
                yticklabels=list(predictions.keys()),
                annot=True, fmt='.2f', cmap='RdYlGn_r', vmin=-1, vmax=1)
    plt.title('Base Model Prediction Correlation')
    plt.tight_layout()
    plt.show()
    return corr
```

### 6.2 모델 선택 전략

```python
"""
단계 1: 표준 다양한 모델들로 시작
  ┌────────────────────┬────────────────────────────────┐
  │ 모델 계열           │ 권장 모델                       │
  ├────────────────────┼────────────────────────────────┤
  │ 선형(Linear)        │ LogisticRegression, Ridge       │
  │ 트리 기반(Tree)     │ RandomForest, ExtraTrees        │
  │ 부스팅(Boosting)    │ XGBoost, LightGBM, CatBoost     │
  │ 인스턴스 기반       │ KNN (k=5, 10, 20)              │
  │ 커널 기반(Kernel)   │ SVM (RBF, polynomial)           │
  │ 확률적(Probabilistic)│ GaussianNB                     │
  └────────────────────┴────────────────────────────────┘

단계 2: 예측 상관관계 확인
  - 상관관계 > 0.9 → 하나 제거
  - 상관관계 0.7-0.9 → 유지하되 모니터링
  - 상관관계 < 0.7 → 훌륭한 다양성

단계 3: 점진적으로 테스트
  - 3개 모델로 시작, 하나씩 추가
  - 스태킹 성능이 정체될 때 추가 중단

안티패턴:
  ✗ 서로 다른 시드의 랜덤 포레스트 5개 → 높은 상관관계
  ✗ 모두 트리 기반 모델 → 유사한 결정 경계
  ✗ 너무 많은 모델 (15+개) → 수익 체감
"""
```

### 6.3 Q-통계량(Q-Statistic)

```python
"""
Q_ij = (N11 × N00 - N01 × N10) / (N11 × N00 + N01 × N10)

N11 = 둘 다 정답, N00 = 둘 다 오답, N01 = i 정답/j 오답, N10 = i 오답/j 정답

Q = 1 → 항상 동의 (다양성 없음)
Q = 0 → 독립 (좋은 다양성)
Q < 0 → 음의 상관관계 (탁월함, 실제로는 드묾)

스태킹에서는 모든 쌍 간 Q < 0.5를 목표로 합니다.
"""
```

---

## 7. 스태킹 앙상블의 하이퍼파라미터 최적화

### 7.1 단계적 최적화 전략

모든 것을 함께 최적화하는 것은 계산 비용이 너무 큽니다. 단계적 접근 방식을 사용하세요:

```python
"""
단계 1: 기저 모델 개별 튜닝 (높은 영향, HPO 예산의 60-70%)
  - Optuna 또는 GridSearchCV를 사용하여 각 모델을 독립적으로 튜닝
  - 스태킹 CV와 동일한 CV 분할 사용

단계 2: 다양성 기반 기저 모델 선택 (중간 비용)
  - 예측 상관 행렬 계산
  - 중복 모델 제거 (상관관계 > 0.9)

단계 3: 메타 학습기 튜닝 (저비용, 1-2개 하이퍼파라미터)
  - LogisticRegression: C 튜닝
  - Ridge: alpha 튜닝

단계 4: 구조적 선택 튜닝 (작은 영향)
  - passthrough: True vs False
  - stack_method: 'predict_proba' vs 'predict'

왜 단계적으로 하는가? 50+개 차원에 대한 결합 HPO는 불가능합니다.
단계적: Π(개별 공간) 대신 Σ(개별 공간).
"""
```

### 7.2 Optuna 예시

```python
"""
import optuna

def objective(trial):
    rf_n = trial.suggest_int('rf_n_estimators', 50, 300, step=50)
    rf_depth = trial.suggest_int('rf_max_depth', 3, 15)
    svm_C = trial.suggest_float('svm_C', 0.01, 100, log=True)
    knn_k = trial.suggest_int('knn_k', 3, 30)
    meta_C = trial.suggest_float('meta_C', 0.01, 10, log=True)
    passthrough = trial.suggest_categorical('passthrough', [True, False])

    stack = StackingClassifier(
        estimators=[
            ('lr', LogisticRegression(max_iter=1000)),
            ('rf', RandomForestClassifier(n_estimators=rf_n, max_depth=rf_depth)),
            ('svm', SVC(probability=True, C=svm_C)),
            ('knn', KNeighborsClassifier(n_neighbors=knn_k)),
        ],
        final_estimator=LogisticRegression(C=meta_C, max_iter=1000),
        cv=5, passthrough=passthrough
    )
    return cross_val_score(stack, X, y, cv=5, scoring='accuracy').mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
"""
```

---

## 8. 스태킹을 사용하지 말아야 할 때

### 8.1 효과가 있는 경우 vs. 없는 경우

```python
"""
✓ 스태킹이 도움이 되는 경우:
  1. 기저 모델들이 비슷한 정확도 (서로 2-3% 이내)
  2. 기저 모델들이 서로 다른 유형의 오류 생성 (상관관계 < 0.7)
  3. 데이터셋이 충분히 큼 (1000개 이상 샘플)
  4. 마지막 0.5-1%의 성능이 필요할 때

✗ 스태킹이 도움이 되지 않는 경우:
  1. 하나의 모델이 나머지보다 3% 이상 우수할 때
  2. 모든 모델이 동일한 계열 (낮은 다양성)
  3. 데이터셋이 매우 작을 때 (500개 미만 샘플)
  4. 해석 가능성이 필요할 때
  5. 추론 지연 시간이 중요할 때 (추론 시 모든 기저 모델이 실행됨)
  6. 문제가 쉬울 때 (단일 모델 > 98%)
"""
```

### 8.2 계산 비용

```python
"""
4개 기저 모델, 5-폴드 CV를 사용한 스태킹:
  학습: 4 × 5 = 20회 적합 (OOF) + 4회 최종 적합 + 1회 메타 학습기 = 25회
  추론: 4 + 1 = 5회 예측

비용 배수: 단일 모델 대비 학습 ~25배, 추론 ~5배

프로덕션 대안:
  - 모델 증류(Model Distillation): 스택을 모방하도록 단일 모델 학습
  - 전체 5개 대신 상위 2-3개 기저 모델만 사용
  - 가중 평균으로 전환 (메타 학습기 학습 없음)
"""
```

### 8.3 의사결정 프레임워크

```
스태킹을 사용해야 할까?
│
├── 하나의 모델이 다른 모든 모델보다 3% 이상 우수한가? → 그 모델 단독 사용
├── 훈련 샘플이 1000개 미만인가? → 단순 평균 또는 최선의 단일 모델 사용
├── 기저 모델 예측이 다양한가 (상관관계 < 0.7)? (아니라면 → 다른 계열 시도)
├── 추론 지연 시간이 중요한가? → 증류 또는 가중 평균 고려
└── 그 외 경우 → 3-5개 다양한 모델로 StackingClassifier 사용
```

---

## 9. 경진대회 우승 전략

### 9.1 Kaggle 스태킹 패턴

```python
"""
패턴 1: 표준 2단계 스택 (가장 일반적)
  레벨 0: 5-10개 모델 (GBM 변형 + NN + 선형 + KNN)
  레벨 1: 선형 모델 (LogisticRegression 또는 Ridge)

패턴 2: 스택의 블렌딩
  스택 A: [XGBoost, LightGBM, CatBoost] → Ridge
  스택 B: [NN1, NN2, NN3] → LR
  최종: 스택 A, B 예측의 가중 평균

패턴 3: 특성 다양성 스태킹
  서로 다른 특성 부분집합 (원본, PCA, 타겟 인코딩)으로 학습된 모델
  메타 학습기가 모든 예측 결합
"""
```

### 9.2 실용적 팁

```python
"""
1. 항상 OOF 예측 사용 — 인샘플 예측은 누수 유발
2. OOF 및 테스트 예측을 .npy 파일로 저장하여 빠른 실험 가능
3. 각 기저 모델에 대해 K 폴드 전체의 테스트 예측 평균
4. 동일한 CV 분할 (StratifiedKFold, 고정 시드)을 모든 곳에 사용
5. 탐욕적 앙상블 선택이 모든 모델 사용보다 나은 경우가 많음:
   - 최선 모델로 시작, CV 점수를 향상시키는 모델을 탐욕적으로 추가
   - 잘 선택된 3-5개 모델이 무작위 20+개 모델을 능가하는 경우 많음
"""
```

---

## 10. 완전한 스태킹 워크플로우

```python
"""
1. 데이터 분할 → train_test_split + StratifiedKFold 정의
2. 기저 모델 선택 → 알고리즘 다양성 극대화
3. 개별 평가 → cross_val_score, 정확도 55% 미만 모델 제거
4. 다양성 측정 → 상관 행렬, 0.9 초과 쌍 제거
5. 스택 구축 → StackingClassifier(estimators, final_estimator, cv)
6. 평가 → cross_val_score, 최선 단일 모델과 비교
7. 튜닝 → 기저 모델 먼저, 그 다음 메타 학습기, 그 다음 passthrough
8. 예측 → stack.fit(X_train, y_train); stack.predict(X_test)
9. 배포 → 전체 스택 또는 단일 모델로 증류
"""
```

---

## 연습 문제

### 연습 1: 앙상블 전략 구축 및 비교

```python
"""
Breast Cancer 데이터셋 (sklearn.datasets) 사용:

1. 5개의 개별 모델 학습: LR, RF, GradientBoosting, SVM, KNN
2. 5-폴드 교차 검증으로 각각 평가
3. 세 가지 앙상블 전략 구축:
   a. 예측 확률의 단순 평균
   b. 20% 홀드아웃을 사용한 블렌딩
   c. StackingClassifier(cv=5)를 사용한 스태킹
4. 모든 8가지 접근법을 막대 차트로 비교
5. 어떤 전략이 가장 잘 작동하는가? 왜인가?
보너스: passthrough=True가 도움이 되는가?
"""
```

### 연습 2: 다양성 분석

```python
"""
make_classification (n_samples=3000, n_features=30) 사용:

1. 6개 기저 모델 학습, OOF 예측 생성
2. 예측 상관 히트맵 시각화
3. 두 개의 스택 구축:
   a. 스택 A: 가장 다양한 3개 모델 (가장 낮은 상관관계)
   b. 스택 B: 가장 덜 다양한 3개 모델 (가장 높은 상관관계)
4. 성능 비교 — 다양성이 향상을 예측하는가?
"""
```

### 연습 3: 스태킹 vs. 최선 단일 모델

```python
"""
실험: 스태킹이 최선의 단일 모델을 언제 이기는가?

1. 크기별 데이터셋 생성: 200, 500, 1000, 5000, 10000
2. 각 크기에서 최선 단일 모델 vs. StackingClassifier 비교
3. 성능 격차 vs. 데이터셋 크기 시각화
4. 학습 시간 비율 시각화 (스태킹 / 단일 모델)
5. 결론: 어떤 크기에서 스태킹이 일관되게 이기는가?
"""
```

### 연습 4: 경진대회 스타일 앙상블

```python
"""
1. sklearn.datasets.load_digits 로드
2. 완전한 스태킹 파이프라인 구현:
   a. 고정 StratifiedKFold, 5개 이상 모델의 OOF 예측
   b. 예측을 .npy 배열로 저장
   c. 메타 학습기 학습, 최종 예측 생성
3. 탐욕적 앙상블 선택 구현
4. 비교: 탐욕적 선택 vs. 모든 모델 사용
"""
```

---

## 11. 요약

### 핵심 요점

| 개념 | 설명 |
|------|------|
| **스태킹(Stacking)** | 기저 모델 예측을 결합하는 메타 학습기를 학습시킴 |
| **블렌딩(Blending)** | CV 대신 홀드아웃 세트를 사용하는 더 단순한 변형 |
| **다양성(Diversity)** | 서로 다른 모델 오류 = 더 나은 앙상블; 상관 행렬로 측정 |
| **교차 검증 OOF** | 메타 특성 생성 시 데이터 누수를 방지 |
| **메타 학습기(Meta-Learner)** | 3-5개 기저 모델에는 단순한 (선형 모델) 형태로 유지 |
| **다단계(Multi-Level)** | 2단계가 최적; 3단계 이상은 거의 충분히 향상되지 않음 |
| **건너뛸 때** | 소규모 데이터, 하나의 지배적 모델, 지연 시간 중요, 해석 가능성 필요 시 |

### 모범 사례

1. **단순하게 시작**: 전체 스택 구축 전에 가중 평균 시도
2. **다양성 극대화**: 서로 다른 알고리즘 계열에서 기저 모델 선택
3. **상관관계 확인**: 중복 모델 제거 (상관관계 > 0.9)
4. **OOF 예측 사용**: 인샘플 예측을 메타 특성으로 절대 사용하지 말 것
5. **메타 학습기를 단순하게 유지**: 로지스틱/릿지 회귀가 거의 항상 충분
6. **단계적으로 튜닝**: 기저 모델 먼저, 그 다음 메타 학습기, 그 다음 구조적 선택
7. **향상 측정**: 스태킹으로 0.2% 미만 향상 시 복잡성이 가치 없을 수 있음

### 다른 레슨과의 연결

- **L07 (배깅)**: 스태킹은 기저 학습기로 랜덤 포레스트를 자주 사용
- **L08 (부스팅)**: XGBoost/LightGBM은 스태킹에서 가장 강력한 기저 모델 중 하나
- **L05 (교차 검증)**: OOF 예측은 동일한 CV 원칙에 의존
- **L19 (AutoML)**: 일부 AutoML 도구 (Auto-sklearn, H2O)는 스태킹 앙상블을 자동으로 구축
