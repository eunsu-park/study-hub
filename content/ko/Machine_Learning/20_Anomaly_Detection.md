# 이상 탐지(Anomaly Detection)

## 개요

이상 탐지(Anomaly Detection)는 예상 동작에서 크게 벗어난 데이터 포인트, 이벤트, 또는 패턴을 식별합니다. 분류(Classification)와 달리, 이상 탐지는 이상 클래스에 대한 레이블 데이터가 거의 없거나 전혀 없이 작동하는 경우가 많습니다. 이는 사기 탐지(Fraud Detection), 시스템 모니터링(System Monitoring), 제조 품질 관리, 사이버 보안(Cybersecurity)에 필수적입니다.

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

- **L17**: 불균형 데이터(Imbalanced Data) — 레이블이 있는 이상치가 있으면 분류로 처리
- **Deep_Learning**: 고차원 이상 탐지를 위한 오토인코더(Autoencoder)
- **Data_Science L23**: 비모수 통계(Nonparametric Statistics) — 이상치에 대한 부트스트랩과 순열 검정
