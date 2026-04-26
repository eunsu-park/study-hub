# 이상 탐지(Anomaly Detection)

**이전**: [AutoML과 하이퍼파라미터 최적화](./19_AutoML_Hyperparameter_Optimization.md) | **다음**: [고급 앙상블](./21_Advanced_Ensemble.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 점 이상(point anomaly), 맥락적 이상(contextual anomaly), 집합적 이상(collective anomaly)을 구별하고 각각의 실제 사례를 식별할 수 있습니다
2. 단변량 및 다변량 이상치 탐지를 위한 통계적 방법(Z-점수(Z-score), IQR, 마할라노비스 거리(Mahalanobis distance))을 적용할 수 있습니다
3. 격리 포레스트(Isolation Forest)를 구현하고 이상치가 평균 경로 길이가 짧은 이유를 설명할 수 있습니다
4. 서로 다른 데이터 특성에 따라 지역 이상치 인수(LOF, Local Outlier Factor), 단일 클래스 SVM(One-Class SVM), 격리 포레스트(Isolation Forest)를 비교할 수 있습니다
5. 여러 방법의 정규화된 점수를 결합하여 앙상블 이상치 탐지기(ensemble anomaly detector)를 구축할 수 있습니다
6. 롤링 Z-점수(rolling Z-score)와 EWMA 관리도(control chart)를 사용하여 시계열 데이터에서 이상을 탐지할 수 있습니다
7. 정답 레이블이 없을 때 안정성 분석(stability analysis)을 사용하여 이상 탐지 모델을 평가할 수 있습니다

---

대부분의 머신러닝은 찾고자 하는 것을 알고 있다고 가정합니다. 즉, 모든 클래스에 대한 레이블 예시를 보유하고 있다고 가정하는 것입니다. 이상 탐지(anomaly detection)는 이 가정을 뒤집습니다. 예상치 못한 것, 드문 것, 한 번도 보지 못한 것을 찾아내야 하며, 이 때 이상 클래스에 대한 레이블 데이터가 거의 없거나 전혀 없는 경우가 많습니다. 사기 거래 탐지, 제조 결함 탐지부터 서버 상태 모니터링에 이르기까지, 이상 탐지는 ML에서 가장 실용적으로 중요하면서도 방법론적으로 독특한 분야 중 하나입니다. 이 레슨에서는 간단한 통계 검정부터 앙상블 방법까지, 데이터의 건초 더미에서 바늘을 찾기 위한 완전한 도구 모음을 제공합니다.

---

## 이론과 원리

이상 탐지는 클래스가 빠진 분류처럼 보이지만 — 실제로는 직접 감독이 없는 다른 수학적 문제입니다. 이 레슨의 각 알고리즘은 "정상"의 다른 *정의*를 인코딩하며, 그 선택이 알고리즘이 어떤 종류의 이상을 찾을 수 있고 없는지를 결정. 각 방법 뒤의 가정을 아는 것이 데이터에 올바른 것을 고르게 합니다.

### A. "정상"의 세 가지 정의

표준 이상 탐지 알고리즘 네 가지가 세 가지 정의 가족으로 분리됩니다:

1. **통계적**: 정상 점이 가정된 분포 하에서 높은 밀도. 이상은 낮은 밀도 꼬리. 예: 가우시안 + 마할라노비스 거리, 커널 밀도 추정.
2. **거리 기반**: 정상 점이 가까운 이웃을 많이 가짐. 이상은 고립. 예: Isolation Forest, Local Outlier Factor.
3. **재구성 기반**: 정상 점은 정상 데이터에서 학습된 저용량 모델로 재구성 가능. 이상은 그렇지 않음. 예: 오토인코더, PCA 재구성 오차.

각 정의가 다른 점수 함수를 줌. 알고리즘 선택은 도메인에서 무엇이 점을 "이상"하게 만드는지에 맞아야 함.

### B. 마할라노비스 거리: 통계적 베이스라인

평균 `μ`와 공분산 `Σ`를 가진 다변량 가우시안 데이터에 대해, `x`에서 분포까지의 마할라노비스 거리는:

```
D_M(x) = √[ (x - μ)ᵀ · Σ⁻¹ · (x - μ) ]
```

데이터가 `Σ⁻¹/²`로 "백색화(whitened)"된 후의 유클리드 거리. 모든 방향을 표준화하여 클러스터가 구형으로 보이게 한 다음 거리를 측정하는 것과 동등.

평범한 유클리드보다의 이점: 마할라노비스가 특성 상관과 특성별 분산을 올바르게 고려. 저분산 방향에서 3 표준편차 떨어진 점이 고분산 방향에서 3 SD 떨어진 점보다 더 이상적 — 그리고 마할라노비스가 자동으로 이를 반영.

가우시안 가정 하에서 `D_M(x)²`는 `p` 자유도의 카이제곱 분포를 따름. 이는 원칙적 임계값을 줍니다: `D_M²`이 카이제곱 `1 - α` 분위수를 초과하는 점을 표시(예: 상위 0.1%를 이상으로 하려면 `α = 0.001`).

함정: 가우시안 가정이 제한적. 비가우시안 데이터에 대해 마할라노비스가 잘못된 점을 표시.

### C. Isolation Forest: 이상은 고립하기 더 쉬움

Isolation Forest(Liu et al., 2008)는 보통 논리를 뒤집습니다. 정상 데이터를 모델링하고 편차를 표시하는 대신, 각 점을 *고립*시키려는 무작위 트리를 만들고, 이상이 더 적은 분할로 고립 가능해야 한다고 추론.

절차:

```
T개 트리 만들기, 각각 데이터의 무작위 부분표본에서
각 트리가 각 노드에서 무작위 분할 선택(무작위 특성, 무작위 임계값)
모든 리프가 ≤ 1 점을 가질 때까지(또는 높이 한계에 도달) 각 트리 자라기

점 x에 대해, 고립 깊이 h(x) = 각 트리에서 x가 떨어지는 평균 깊이
```

점수가 정규화됨:

```
s(x) = 2^( -h(x) / c(N) )
```

여기서 `c(N) ≈ 2 ln(N-1) - 2(N-1)/N`은 무작위 이진 트리에서의 기대 고립 깊이. `s(x) → 1`은 `h`가 작을 때(고립하기 쉬움 ⟹ 이상); `s(x) → 0.5`는 `h`가 정상 점과 일치할 때; `s(x) → 0`은 `h`가 매우 클 때.

왜 작동하나? 이상은 정의상 대부분의 점과 다르므로, 무작위 축 정렬 절단이 이를 더 쉽게 고립. 정상 점은 함께 군집되어 있어, 하나를 고립시키려면 둘러싼 인구를 벗기기 위해 많은 분할이 필요.

강점:
- 분포 가정 없음.
- 고차원을 합리적으로 처리(특성의 무작위 부분표본화가 도움).
- `N`과 `T`에 선형; 매우 빠름.

약점:
- 축 정렬 분할 — 결정 트리처럼 대각선/곡선 이상 영역과 어려움.

### D. One-Class SVM: ν가 이상치 비율을 통제

One-class SVM(Schölkopf et al., 2001)은 데이터 본체 주변에 단단한 경계를 찾고 바깥의 점을 이상으로 표시. 형식:

```
minimize  ½ · ‖w‖² + (1/(ν · N)) · Σ ξ_i  -  ρ
subject to  wᵀ φ(x_i) ≥ ρ - ξ_i,  ξ_i ≥ 0
```

`φ`는 커널 특성 맵(RBF 기본); `ρ`는 오프셋; `ξ_i`는 슬랙 변수. 결정 함수는 `f(x) = sign(wᵀ φ(x) - ρ)`: "정상"이면 양수, "이상"이면 음수.

하이퍼파라미터 `ν ∈ (0, 1]`이 *깔끔한 쌍대 해석*을 가짐:

- `ν`는 이상으로 분류된 학습 점 비율의 **상한**(마진 오류).
- `ν`는 서포트 벡터 비율의 **하한**.

따라서 `ν = 0.05` 설정은 "최대 5%의 학습 점이 이상치; 그에 따라 학습"이라 말함. 이는 One-class SVM을 ML 알고리즘 중에서 특이하게 만듭니다 — 주요 하이퍼파라미터가 직접적인 실용적 의미를 가짐.

커널 선택이 중요: RBF가 비선형 정상 영역 주변을 경계가 감싸게 함. 선형은 실전에서 거의 유용하지 않음.

비용: 커널 계산에 `O(N²)`; ~10K 표본 이상에서 비현실적.

### E. 오토인코더: 이상 점수로서의 재구성 오차

오토인코더는 저차원 병목을 통해 입력을 재구성하도록 학습된 신경망:

```
x → 인코더 → z (저차원 잠재) → 디코더 → x̂
loss = ‖x - x̂‖²
```

정상 데이터에서만 학습된 오토인코더는 정상 표본의 *구조*를 포착하는 압축 표현을 학습. 이상 입력은 학습된 매니폴드 위에 있지 않고 잘못 재구성. 재구성 오차 `‖x - x̂‖²`이 이상 점수가 됩니다.

이는 본질적으로 **비선형 매니폴드로 일반화된 PCA**. PCA 재구성 오차 자체가 선형 데이터의 가능한 이상 점수; 오토인코더가 같은 아이디어를 비선형 데이터로 확장.

선택점:
- **병목 크기**: 너무 작으면 ⟹ 정상 데이터조차 잘못 재구성; 너무 크면 ⟹ 네트워크가 항등 함수를 학습하고 이상도 잘 재구성. 보류된 정상 집합에 대한 교차검증으로 크기 선택.
- **변분 오토인코더(Variational autoencoder)**가 가능도 기반 점수를 계산할 수 있는 확률적 잠재를 추가하여 재구성 오차 접근을 일반화.
- **잡음 제거 오토인코더(Denoising autoencoder)**는 잡음 입력에서 재구성하도록 학습 — 결과 모델이 더 강건하고 종종 더 나은 이상 점수를 줌.

### F. 이상 점수의 임계값 선택

(B)–(E)의 모든 알고리즘이 연속 점수를 출력; 임계값을 골라야 함. 세 전략:

1. **오염 가정**: 데이터의 ~`α%`가 이상이라 믿으면, 임계값을 학습 점수의 `(1-α)`-번째 백분위수로 설정. scikit-learn의 `contamination` 매개변수가 이를 직접 함.
2. **통계적**: 모수 가정 하에서(마할라노비스 ⟹ 카이제곱), 카이제곱 `1 - α` 분위수 선택.
3. **운영적**: 다운스트림 프로세스가 흡수할 수 있는 수준으로 알람 비율을 유지하도록 임계값 선택(예: 사기 팀이 하루 100건 조사 가능; 100건이 트리거되도록 점수의 임계값).

운영적 선택이 보통 프로덕션에서 이김 — 이상 탐지는 레이블된 검증 데이터가 거의 없어서 F1 같은 지표를 직접 최적화할 수 없음. 운영적 제약에 보정하고 기저 분포가 표류할 때 재방문.

### G. 레이블 없이 평가

이상 탐지의 가장 어려운 부분은 *평가*. 레이블 없이 ROC-AUC와 PR-AUC를 사용할 수 없음. 세 가지 실용적 우회:

- **센티넬 주입**: 알려진 합성 이상을 테스트셋에 주입하고 표시되는지 확인.
- **적대적 검증**: 두 의심되는 인구(예: "이번 주" vs "2주 전") 사이의 이진 분류기 학습. 그것들을 구별할 수 있다면, 인구가 표류 — 가능하게 이상.
- **운영적 피드백 루프**: 시스템이 표시한 항목 로그, 인간 조사자가 레이블 부여하게 함, 레이블을 사용해 임계값 튜닝(그리고 결국 레이블된 부분집합에 대한 XGBoost 같은 준지도 방법으로 이동).

이상 탐지는 일회성 학습 문제보다 반복적, 피드백 주도 규율.

### From Theory to the Code Below

- 섹션 2의 `mahalanobis_distances` 계산과 카이제곱 임계값은 (B)를 구현; `EllipticEnvelope` 클래스가 둘 다 래핑.
- 섹션 3의 `IsolationForest(n_estimators=..., contamination=...)`는 (C)의 알고리즘; `contamination`은 (F.1)의 임계값 선택 전략.
- 섹션 4의 `OneClassSVM(nu=..., kernel='rbf')`는 (D)의 알고리즘; `nu` 매개변수는 (D)의 경계.
- 섹션 5의 `LocalOutlierFactor`는 (A)의 가족 (1)의 거리 기반 변형 — `x`의 국소 밀도 대 `x`의 `k`-최근접 이웃의 밀도 비율로 점수.
- 섹션 6의 오토인코더 재구성 오차 예제는 (E)의 알고리즘; 같은 섹션의 PCA 재구성 오차와의 비교는 (E)의 선형-vs-비선형 구분.
- 끝의 "제한된 레이블로 평가" 섹션이 (G)의 우회에 매핑.

---

## 1. 이상(Anomaly)의 유형

### 1.1 이상 분류 체계

```python
"""
이상의 유형:

1. 점 이상(Point Anomaly): 단일 데이터 포인트가 이상인 경우
   예시: 일반적으로 $50-$500인 신용카드 거래에서 $50,000 거래

2. 맥락적 이상(Contextual Anomaly): 특정 맥락에서는 이상이지만 그 외에는 정상
   예시: 35°C의 기온은 여름에는 정상이지만 겨울에는 이상

3. 집합적 이상(Collective Anomaly): 데이터 포인트 그룹이 함께 이상인 경우
   예시: 연속적인 소액 고속 거래 (카드 테스트 패턴)

접근 방식:

┌──────────────────────┬──────────────────────────────────────┐
│ 지도 학습(Supervised)│ 레이블된 정상 + 이상 데이터          │
│                      │ → 분류 (L17 참조)                    │
├──────────────────────┼──────────────────────────────────────┤
│ 준지도 학습          │ 레이블된 정상 데이터만               │
│ (Semi-supervised)    │ → 단일 클래스 SVM, 오토인코더        │
├──────────────────────┼──────────────────────────────────────┤
│ 비지도 학습          │ 레이블 없음                          │
│ (Unsupervised)       │ → Isolation Forest, LOF, DBSCAN      │
└──────────────────────┴──────────────────────────────────────┘
"""
```

---

## 2. 통계적 방법

### 2.1 Z-점수와 IQR

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# 이상치가 포함된 데이터 생성
normal_data = np.random.normal(50, 10, 1000)
outliers = np.array([120, 130, -20, -30, 150])
data = np.concatenate([normal_data, outliers])

df = pd.DataFrame({'value': data})

# 방법 1: Z-점수(Z-Score)
df['z_score'] = (df['value'] - df['value'].mean()) / df['value'].std()
df['is_anomaly_zscore'] = df['z_score'].abs() > 3  # |z| > 3

# 방법 2: IQR (사분위 범위, Interquartile Range)
Q1 = df['value'].quantile(0.25)
Q3 = df['value'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df['is_anomaly_iqr'] = (df['value'] < lower) | (df['value'] > upper)

# 방법 3: 수정 Z-점수(Modified Z-Score, 강건, 중앙값 사용)
median = df['value'].median()
mad = np.median(np.abs(df['value'] - median))  # 중앙절대편차(Median Absolute Deviation)
df['modified_z'] = 0.6745 * (df['value'] - median) / mad
df['is_anomaly_modified_z'] = df['modified_z'].abs() > 3.5

print(f"Z-점수 이상치:    {df['is_anomaly_zscore'].sum()}")
print(f"IQR 이상치:       {df['is_anomaly_iqr'].sum()}")
print(f"수정 Z 이상치:    {df['is_anomaly_modified_z'].sum()}")

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
for ax, method in zip(axes, ['is_anomaly_zscore', 'is_anomaly_iqr', 'is_anomaly_modified_z']):
    colors = df[method].map({True: 'red', False: 'blue'})
    ax.scatter(range(len(df)), df['value'], c=colors, alpha=0.5, s=10)
    ax.set_title(method.replace('is_anomaly_', '').replace('_', ' ').title())
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
plt.tight_layout()
plt.savefig('statistical_anomalies.png', dpi=150)
plt.show()
```

### 2.2 마할라노비스 거리(Mahalanobis Distance, 다변량)

```python
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2

np.random.seed(42)

# 상관관계가 있는 다변량 정규 데이터
mean = [50, 100]
cov = [[100, 80], [80, 150]]
normal_2d = np.random.multivariate_normal(mean, cov, 500)
outliers_2d = np.array([[100, 30], [10, 200], [120, 250]])
X = np.vstack([normal_2d, outliers_2d])

# 마할라노비스 거리
cov_inv = np.linalg.inv(np.cov(X.T))
center = X.mean(axis=0)
mahal_distances = np.array([mahalanobis(x, center, cov_inv) for x in X])

# 임계값: df=2, p=0.001인 카이제곱(chi2) 분포
threshold = np.sqrt(chi2.ppf(0.999, df=2))
is_anomaly = mahal_distances > threshold

print(f"마할라노비스 임계값 (p=0.001): {threshold:.2f}")
print(f"탐지된 이상치: {is_anomaly.sum()}")

# 플롯
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 유클리드(Euclidean, 상관된 이상치를 놓침)
eucl_dist = np.sqrt(((X - center)**2).sum(axis=1))
axes[0].scatter(X[:, 0], X[:, 1], c=eucl_dist, cmap='coolwarm', alpha=0.7, s=20)
axes[0].set_title('유클리드 거리(Euclidean Distance)')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')

# 마할라노비스(상관관계 반영)
colors = ['red' if a else 'blue' for a in is_anomaly]
axes[1].scatter(X[:, 0], X[:, 1], c=colors, alpha=0.7, s=20)
axes[1].set_title(f'마할라노비스(Mahalanobis, 임계값={threshold:.1f})')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')

plt.tight_layout()
plt.savefig('mahalanobis.png', dpi=150)
plt.show()
```

---

## 3. 격리 포레스트(Isolation Forest)

### 3.1 알고리즘과 구현

```python
"""
격리 포레스트(Isolation Forest, Liu et al., 2008):
  - 핵심 통찰: 이상치는 적고(FEW) 다름(DIFFERENT) → 격리하기 쉬움
  - 특징(Feature)과 분할값을 무작위로 선택하여 랜덤 이진 트리 구성
  - 이상치는 격리에 필요한 분할 수가 적음 (짧은 경로 길이)
  - 정상 포인트는 분할이 더 많이 필요 (긴 경로 길이)

동작 방식:
  1. 특징을 무작위로 선택
  2. min과 max 사이에서 분할값을 무작위로 선택
  3. 데이터를 좌/우로 분할
  4. 각 포인트가 격리될 때까지 반복
  5. 이상 점수 = 모든 트리의 평균 경로 길이
  6. 짧은 경로 → 이상, 긴 경로 → 정상
"""

from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs

# 이상치가 포함된 데이터 생성
X_normal, _ = make_blobs(n_samples=1000, centers=2, cluster_std=1.0, random_state=42)
X_anomaly = np.random.uniform(-8, 8, (50, 2))  # 50개 무작위 이상치
X_all = np.vstack([X_normal, X_anomaly])
y_true = np.concatenate([np.ones(1000), -np.ones(50)])  # 1=정상, -1=이상

# 격리 포레스트 훈련
iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.05,  # 예상 이상치 비율
    random_state=42,
    n_jobs=-1,
)
y_pred = iso_forest.fit_predict(X_all)  # 1=정상, -1=이상
scores = iso_forest.decision_function(X_all)  # 낮을수록 더 이상함

print(f"탐지된 이상치: {(y_pred == -1).sum()}")
print(f"점수 범위: [{scores.min():.3f}, {scores.max():.3f}]")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 예측
colors_pred = ['red' if p == -1 else 'blue' for p in y_pred]
axes[0].scatter(X_all[:, 0], X_all[:, 1], c=colors_pred, alpha=0.5, s=15)
axes[0].set_title(f'격리 포레스트 예측 (이상치 {(y_pred==-1).sum()}개)')

# 이상 점수 히트맵
xx, yy = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
Z = iso_forest.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
axes[1].contourf(xx, yy, Z, levels=20, cmap='RdBu')
axes[1].scatter(X_all[:, 0], X_all[:, 1], c=colors_pred, alpha=0.5, s=15, edgecolors='k', linewidths=0.3)
axes[1].set_title('이상 점수 등고선 (빨강=이상)')

plt.tight_layout()
plt.savefig('isolation_forest.png', dpi=150)
plt.show()
```

### 3.2 오염도(Contamination) 파라미터 튜닝

```python
from sklearn.metrics import precision_score, recall_score, f1_score

# 다양한 오염도 값 시도
contaminations = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
results = []

for cont in contaminations:
    iso = IsolationForest(n_estimators=200, contamination=cont, random_state=42)
    y_pred = iso.fit_predict(X_all)

    # 이진 변환: 이상=1, 정상=0
    y_pred_binary = (y_pred == -1).astype(int)
    y_true_binary = (y_true == -1).astype(int)

    results.append({
        'contamination': cont,
        'n_detected': (y_pred == -1).sum(),
        'precision': precision_score(y_true_binary, y_pred_binary),
        'recall': recall_score(y_true_binary, y_pred_binary),
        'f1': f1_score(y_true_binary, y_pred_binary),
    })

results_df = pd.DataFrame(results)
print("오염도 튜닝 결과:")
print(results_df.round(3).to_string(index=False))
```

---

## 4. 지역 이상치 인수(LOF, Local Outlier Factor)

### 4.1 밀도 기반 이상 탐지

```python
"""
LOF (Breunig et al., 2000):
  - 이웃 대비 포인트의 지역 밀도 편차를 측정
  - 이웃 대비 저밀도 영역의 포인트 → 이상치
  - 핵심 장점: 서로 다른 밀도의 클러스터 처리 가능

동작 방식:
  1. 각 포인트에 대해 k개의 최근접 이웃 찾기
  2. 지역 도달 가능 밀도(LRD, Local Reachability Density) 계산
  3. LRD를 이웃의 LRD와 비교 → LOF 점수
  4. LOF ≈ 1: 이웃과 유사한 밀도 (정상)
  5. LOF >> 1: 이웃보다 훨씬 낮은 밀도 (이상)
"""

from sklearn.neighbors import LocalOutlierFactor

# LOF
lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.05,
    novelty=False,  # False: 이상치 탐지 (비지도)
)
y_pred_lof = lof.fit_predict(X_all)
lof_scores = -lof.negative_outlier_factor_  # 직관적 해석을 위해 부호 반전

print(f"LOF 탐지 이상치: {(y_pred_lof == -1).sum()}")
print(f"LOF 점수 범위: [{lof_scores.min():.3f}, {lof_scores.max():.3f}]")

# 격리 포레스트와 비교
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

iso_colors = ['red' if p == -1 else 'blue' for p in iso_forest.fit_predict(X_all)]
axes[0].scatter(X_all[:, 0], X_all[:, 1], c=iso_colors, alpha=0.5, s=15)
axes[0].set_title('격리 포레스트(Isolation Forest)')

lof_colors = ['red' if p == -1 else 'blue' for p in y_pred_lof]
axes[1].scatter(X_all[:, 0], X_all[:, 1], c=lof_colors, alpha=0.5, s=15)
axes[1].set_title('지역 이상치 인수(Local Outlier Factor)')

plt.tight_layout()
plt.savefig('iso_vs_lof.png', dpi=150)
plt.show()
```

### 4.2 신규성 탐지(Novelty Detection)를 위한 LOF

```python
# 신규성 탐지: 정상 데이터로 훈련, 새로운 이상치 탐지
lof_novelty = LocalOutlierFactor(n_neighbors=20, novelty=True)
lof_novelty.fit(X_normal)  # 정상 데이터로만 훈련

# 정상과 이상이 혼합된 데이터로 테스트
X_new_normal = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 20) + X_normal.mean(axis=0)
X_new_anomaly = np.random.uniform(-8, 8, (10, 2))
X_new = np.vstack([X_new_normal, X_new_anomaly])
y_new_true = np.concatenate([np.ones(20), -np.ones(10)])

y_new_pred = lof_novelty.predict(X_new)
print(f"이상으로 오탐된 새 정상 데이터: {(y_new_pred[:20] == -1).sum()} / 20")
print(f"탐지된 새 이상치: {(y_new_pred[20:] == -1).sum()} / 10")
```

---

## 5. 단일 클래스 SVM(One-Class SVM)

### 5.1 서포트 벡터 접근법

```python
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# 특징 스케일링 (SVM에서 중요)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

# 단일 클래스 SVM(One-Class SVM)
ocsvm = OneClassSVM(
    kernel='rbf',
    gamma='scale',
    nu=0.05,  # 이상치 비율의 상한
)
y_pred_svm = ocsvm.fit_predict(X_scaled)

print(f"단일 클래스 SVM 이상치: {(y_pred_svm == -1).sum()}")

# 결정 경계 시각화
xx, yy = np.meshgrid(
    np.linspace(X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1, 100),
    np.linspace(X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1, 100),
)
Z = ocsvm.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(figsize=(8, 6))
ax.contourf(xx, yy, Z, levels=20, cmap='RdBu')
ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
svm_colors = ['red' if p == -1 else 'blue' for p in y_pred_svm]
ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=svm_colors, alpha=0.5, s=15, edgecolors='k', linewidths=0.3)
ax.set_title('단일 클래스 SVM 결정 경계(One-Class SVM Decision Boundary)')
plt.tight_layout()
plt.savefig('ocsvm.png', dpi=150)
plt.show()
```

---

## 6. 앙상블(Ensemble)과 PyOD

### 6.1 여러 탐지기 결합

```python
"""
앙상블 접근법: 견고성을 위해 여러 이상 탐지기를 결합합니다.

# pip install pyod

from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.knn import KNN
from pyod.models.combination import average, maximization

# 탐지기 초기화
detectors = {
    'IForest': IForest(contamination=0.05, random_state=42),
    'LOF': LOF(contamination=0.05),
    'OCSVM': OCSVM(contamination=0.05),
    'KNN': KNN(contamination=0.05),
}

# 피팅 및 점수 수집
all_scores = []
for name, detector in detectors.items():
    detector.fit(X_all)
    scores = detector.decision_scores_
    # 점수를 [0, 1]로 정규화
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min())
    all_scores.append(scores_norm)

all_scores = np.array(all_scores)

# 결합: 정규화된 점수의 평균
ensemble_scores = all_scores.mean(axis=0)
threshold = np.percentile(ensemble_scores, 95)  # 상위 5%가 이상치
ensemble_pred = (ensemble_scores > threshold).astype(int)

print(f"앙상블 탐지: {ensemble_pred.sum()}개 이상치")
"""

# pyod 없이 수동 앙상블
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
iso.fit(X_all)
iso_scores = -iso.decision_function(X_all)  # 높을수록 더 이상함

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
lof.fit_predict(X_all)
lof_scores = -lof.negative_outlier_factor_

# 정규화 및 결합
def normalize(scores):
    return (scores - scores.min()) / (scores.max() - scores.min())

combined = (normalize(iso_scores) + normalize(lof_scores)) / 2
threshold = np.percentile(combined, 95)
ensemble_pred = (combined > threshold).astype(int)
y_true_binary = (y_true == -1).astype(int)

print(f"앙상블 이상치: {ensemble_pred.sum()}")
print(f"정밀도(Precision): {precision_score(y_true_binary, ensemble_pred):.3f}")
print(f"재현율(Recall):    {recall_score(y_true_binary, ensemble_pred):.3f}")
print(f"F1:                {f1_score(y_true_binary, ensemble_pred):.3f}")
```

---

## 7. 시계열 이상 탐지

### 7.1 통계적 공정 제어(Statistical Process Control)

```python
np.random.seed(42)

# 이상치가 포함된 시계열 생성
n = 500
t = np.arange(n)
normal_ts = 50 + 5 * np.sin(2 * np.pi * t / 100) + np.random.normal(0, 2, n)

# 이상치 주입
anomaly_indices = [100, 200, 300, 400]
for idx in anomaly_indices:
    normal_ts[idx] += np.random.choice([-1, 1]) * np.random.uniform(15, 25)

ts_df = pd.DataFrame({'value': normal_ts})

# 방법 1: 롤링 Z-점수(Rolling Z-Score)
window = 30
ts_df['rolling_mean'] = ts_df['value'].rolling(window, center=False).mean()
ts_df['rolling_std'] = ts_df['value'].rolling(window, center=False).std()
ts_df['rolling_z'] = (ts_df['value'] - ts_df['rolling_mean']) / ts_df['rolling_std']
ts_df['anomaly_z'] = ts_df['rolling_z'].abs() > 3

# 방법 2: 지수 가중 이동 평균(EWMA, Exponentially Weighted Moving Average)
span = 20
ts_df['ewma'] = ts_df['value'].ewm(span=span).mean()
ts_df['ewma_std'] = ts_df['value'].ewm(span=span).std()
ts_df['ewma_z'] = (ts_df['value'] - ts_df['ewma']) / ts_df['ewma_std']
ts_df['anomaly_ewma'] = ts_df['ewma_z'].abs() > 3

# 플롯
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axes[0].plot(ts_df['value'], 'b-', alpha=0.7, linewidth=0.8)
axes[0].plot(ts_df['rolling_mean'], 'g-', label='롤링 평균(Rolling Mean)')
upper = ts_df['rolling_mean'] + 3 * ts_df['rolling_std']
lower = ts_df['rolling_mean'] - 3 * ts_df['rolling_std']
axes[0].fill_between(range(n), upper, lower, alpha=0.2, color='green')
anomalies_z = ts_df[ts_df['anomaly_z']]
axes[0].scatter(anomalies_z.index, anomalies_z['value'], c='red', s=50, zorder=5, label='이상치')
axes[0].set_title('롤링 Z-점수 이상 탐지(Rolling Z-Score Anomaly Detection)')
axes[0].legend()

axes[1].plot(ts_df['value'], 'b-', alpha=0.7, linewidth=0.8)
axes[1].plot(ts_df['ewma'], 'orange', label='EWMA')
upper_ewma = ts_df['ewma'] + 3 * ts_df['ewma_std']
lower_ewma = ts_df['ewma'] - 3 * ts_df['ewma_std']
axes[1].fill_between(range(n), upper_ewma, lower_ewma, alpha=0.2, color='orange')
anomalies_ewma = ts_df[ts_df['anomaly_ewma']]
axes[1].scatter(anomalies_ewma.index, anomalies_ewma['value'], c='red', s=50, zorder=5, label='이상치')
axes[1].set_title('EWMA 이상 탐지(EWMA Anomaly Detection)')
axes[1].legend()

plt.tight_layout()
plt.savefig('ts_anomaly.png', dpi=150)
plt.show()
```

### 7.2 이상 탐지를 위한 계절 분해(Seasonal Decomposition)

```python
"""
계절성 시계열에 대해:
1. 추세(Trend) + 계절성(Seasonal) + 잔차(Residual)로 분해 (STL 분해)
2. 잔차에 이상 탐지 적용
3. 큰 잔차 = 이상치

from statsmodels.tsa.seasonal import STL

stl = STL(ts_df['value'], period=100, robust=True)
result = stl.fit()
residuals = result.resid

# 잔차에서 이상치 탐지
z_scores = (residuals - residuals.mean()) / residuals.std()
anomalies = z_scores.abs() > 3
"""
print("STL 분해는 추세와 계절성을 제거하여,")
print("실제 편차에 대한 이상 탐지를 더 민감하게 만듭니다.")
```

---

## 8. 레이블 없는 평가

### 8.1 정답(Ground Truth)이 없을 때

```python
"""
실제로 레이블이 있는 이상치는 드뭅니다. 평가 전략:

1. 시각적 검사(Visual Inspection):
   - 탐지된 이상치를 플롯하고 도메인 전문가와 확인
   - 초기 모델 개발에 가장 실용적

2. 안정성 분석(Stability Analysis):
   - 다른 파라미터로 탐지기 실행
   - 일관되게 플래그가 세워진 포인트 = 실제 이상치일 가능성 높음
   - 극단적인 설정에서만 플래그가 세워진 포인트 = 경계선

3. 내부 지표(Internal Metrics):
   - 실루엣 유사 점수: 탐지된 이상치가 정상과 얼마나 다른가?
   - 이상 점수 분포: 이봉(bimodal) 형태여야 함 (정상 피크 + 이상 꼬리)

4. 대리 지표(Proxy Metrics):
   - 이상치가 알려진 이벤트(시스템 중단, 사기 신고)와 상관관계가 있는 경우
   - 외부 신호와의 시간적 상관관계

5. 프로덕션에서 A/B 테스트:
   - 탐지된 이상치에 대한 조치의 비즈니스 영향 측정
   - 진양성(True positive) → 사기 예방, 결함 발견
   - 위양성(False positive) → 조사 시간 낭비
"""

# 안정성 분석 예시
from sklearn.ensemble import IsolationForest

# 여러 오염도 값으로 실행
stability_matrix = np.zeros((len(X_all), 5))
for i, cont in enumerate([0.01, 0.03, 0.05, 0.08, 0.1]):
    iso = IsolationForest(contamination=cont, random_state=42)
    preds = iso.fit_predict(X_all)
    stability_matrix[:, i] = (preds == -1).astype(int)

# 모든 설정에서 플래그가 세워진 포인트 = 높은 신뢰도 이상치
stability_score = stability_matrix.mean(axis=1)
high_conf_anomalies = stability_score >= 0.8  # 80% 이상의 실행에서 플래그

print(f"높은 신뢰도 이상치 (5번 중 4번 이상 플래그): {high_conf_anomalies.sum()}")
print(f"보통 신뢰도 (5번 중 2-3번): {((stability_score >= 0.4) & (stability_score < 0.8)).sum()}")
print(f"낮은 신뢰도 (5번 중 1번): {((stability_score > 0) & (stability_score < 0.4)).sum()}")
```

---

## 9. 방법 선택 가이드

### 9.1 올바른 방법 선택

| 방법 | 데이터 유형 | 레이블 데이터 | 확장성 | 해석 가능성 |
|------|-------------|-------------|--------|-------------|
| Z-점수 / IQR | 단변량 | 불필요 | 우수 | 매우 높음 |
| 마할라노비스(Mahalanobis) | 다변량 | 불필요 | 좋음 | 높음 |
| **격리 포레스트(Isolation Forest)** | 표 형식 | 불필요 | 우수 | 보통 |
| **LOF** | 표 형식 | 불필요 | 보통 | 보통 |
| 단일 클래스 SVM(One-Class SVM) | 표 형식 | 정상만 | 나쁨 (대용량) | 낮음 |
| DBSCAN | 표 형식 | 불필요 | 좋음 | 보통 |
| 오토인코더(Autoencoder) | 모두 (고차원) | 정상만 | 좋음 | 낮음 |
| 롤링 Z-점수(Rolling Z-Score) | 시계열 | 불필요 | 우수 | 매우 높음 |
| STL + 잔차 | 계절성 시계열 | 불필요 | 좋음 | 높음 |

### 9.2 결정 프레임워크

```python
"""
              ┌── 단변량(Univariate)?
              │   └── 예 → Z-점수 또는 IQR (여기서 시작)
              │   └── 아니오 ─┐
              │                │
              │   ┌── 고차원 (특징 50개 이상)?
              │   │   └── 예 → 오토인코더 또는 격리 포레스트
              │   │   └── 아니오 ─┐
              │   │                │
              │   │   ┌── 실시간 / 스트리밍 필요?
              │   │   │   └── 예 → 롤링 Z-점수 또는 EWMA
              │   │   │   └── 아니오 ─┐
              │   │   │                │
              │   │   │   ┌── 정상 전용 훈련 데이터 있음?
              │   │   │   │   └── 예 → 단일 클래스 SVM 또는 LOF (신규성 모드)
              │   │   │   │   └── 아니오 → 격리 포레스트 (가장 견고한 기본값)
"""
```

---

## 10. 연습 문제

### 연습 1: 네트워크 침입 탐지

```python
"""
1. 다음 특징을 가진 네트워크 트래픽 데이터를 생성합니다:
   - bytes_sent, bytes_received, duration, n_packets, protocol_type
   - 정상 트래픽: 적절한 값, 바이트/패킷 간 상관관계
   - 이상치: DDoS (높은 패킷, 낮은 바이트), 데이터 유출 (높은 bytes_sent)
2. 격리 포레스트, LOF, 단일 클래스 SVM을 적용합니다.
3. 탐지율을 비교합니다.
4. 세 방법 모두의 앙상블을 구축합니다.
5. 안정성 분석을 사용하여 높은 신뢰도의 이상치를 식별합니다.
"""
```

### 연습 2: 제조 품질 관리

```python
"""
1. 1000개 제품에 대한 센서 데이터 (온도, 압력, 진동)를 생성합니다.
2. 정상: 상관된 특징, 좋은 제품.
3. 이상치 유형:
   - 과열: 높은 온도, 정상 압력
   - 압력 급등: 높은 압력, 정상 온도
   - 복합: 둘 다 높음 (기계 고장)
4. 마할라노비스 거리를 사용하여 각 이상치 유형을 탐지합니다.
5. 각 이상치 유형에 대한 정밀도와 재현율을 계산합니다.
6. 격리 포레스트는 혼합 이상치 유형을 얼마나 잘 처리하나요?
"""
```

### 연습 3: 시계열 모니터링

```python
"""
1. 1년간의 일별 서버 응답 시간 데이터를 생성합니다:
   - 정상: 평균=200ms, 주간 패턴, 약간의 상승 추세
   - 이상치: 갑작스러운 급등 5회 (서버 문제), 점진적 성능 저하 2회
2. 롤링 Z-점수를 사용하여 점 이상치를 탐지합니다.
3. CUSUM 또는 변화점 탐지(Change-point Detection)를 사용하여 점진적 성능 저하를 탐지합니다.
4. 비교: 어느 방법이 각 이상치 유형을 포착하나요?
5. 설정 가능한 민감도를 가진 알림 시스템을 설계합니다.
"""
```

---

## 11. 요약

### 핵심 정리

| 개념 | 설명 |
|------|------|
| **점/맥락/집합 이상** | 세 가지 이상 유형은 서로 다른 접근법이 필요 |
| **Z-점수 / IQR** | 단순하고 빠름 — 단변량 데이터의 시작점 |
| **마할라노비스(Mahalanobis)** | 상관관계를 고려한 다변량 거리 |
| **격리 포레스트(Isolation Forest)** | 가장 범용적인 방법, 확장성 우수 |
| **LOF** | 밀도 기반, 다양한 클러스터 밀도 처리 |
| **단일 클래스 SVM(One-Class SVM)** | 준지도 학습 (정상 데이터만으로 훈련) |
| **앙상블(Ensemble)** | 견고성을 위해 방법들을 결합 |
| **시계열** | 계절성 데이터를 위한 롤링 통계, STL 분해 |
| **평가** | 레이블이 없을 때 안정성 분석 활용 |

### 모범 사례

1. **격리 포레스트로 시작** — 대부분의 표 형식 데이터에서 견고한 기본값
2. **오염도(contamination) 파라미터**가 중요 — 도메인 지식으로 추정
3. **앙상블 방법**으로 위양성 감소
4. **안정성 분석**으로 레이블 없이 신뢰 수준 제공
5. **신중한 전처리** — SVM과 거리 기반 방법에 특징 스케일링 필수
6. **도메인 전문 지식이 필수** — 검증된 이상치 > 자동화된 점수

### 다음 단계

- **다음**: [L21 — 고급 앙상블](21_Advanced_Ensemble.md)
- **L17**: 불균형 데이터(Imbalanced Data) — 레이블이 있는 이상치가 있으면 분류로 처리
- **Deep_Learning**: 고차원 이상 탐지를 위한 오토인코더(Autoencoder)
- **Data_Science L23**: 비모수 통계(Nonparametric Statistics) — 이상치에 대한 부트스트랩과 순열 검정
