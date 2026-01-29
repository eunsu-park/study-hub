# 클러스터링 (Clustering)

## 개요

클러스터링은 비지도 학습의 대표적인 방법으로, 레이블 없는 데이터를 유사한 그룹으로 분류합니다. 데이터 탐색, 고객 세분화, 이상 탐지 등에 활용됩니다.

---

## 1. 클러스터링의 기본 개념

### 1.1 클러스터링이란?

```python
"""
클러스터링 (Clustering):
- 비지도 학습 (Unsupervised Learning)
- 유사한 데이터를 그룹으로 묶음
- 레이블(정답) 없이 패턴 발견

목표:
- 클러스터 내 유사도 최대화 (intra-cluster similarity)
- 클러스터 간 유사도 최소화 (inter-cluster dissimilarity)

주요 알고리즘:
1. K-Means: 중심점 기반, 빠름
2. DBSCAN: 밀도 기반, 노이즈 처리
3. 계층적 군집화: 덴드로그램 생성
4. Gaussian Mixture: 확률 기반

응용:
- 고객 세분화
- 문서 분류
- 이미지 분할
- 이상 탐지
- 데이터 전처리
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler
```

---

## 2. K-Means

### 2.1 K-Means 알고리즘

```python
"""
K-Means 알고리즘:

1. 초기화: k개의 중심점(centroid) 무작위 선택
2. 할당 단계: 각 데이터 포인트를 가장 가까운 중심점에 할당
3. 업데이트 단계: 각 클러스터의 새로운 중심점 계산 (평균)
4. 반복: 수렴할 때까지 2-3 반복

목표 함수 (Inertia):
J = Σ Σ ||x - μ_k||²
- 각 클러스터 내 분산의 합 최소화
"""

# 데이터 생성
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

# K-Means 적용
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X)

# 시각화
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', edgecolors='black')
plt.title('True Labels')

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', edgecolors='black')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=300, c='red', marker='X', edgecolors='black', linewidths=2,
            label='Centroids')
plt.title('K-Means Clustering')
plt.legend()

plt.tight_layout()
plt.show()

# 결과 정보
print(f"클러스터 중심점:\n{kmeans.cluster_centers_}")
print(f"Inertia: {kmeans.inertia_:.2f}")
print(f"반복 횟수: {kmeans.n_iter_}")
```

### 2.2 sklearn K-Means 사용법

```python
from sklearn.cluster import KMeans

# K-Means 모델
kmeans = KMeans(
    n_clusters=4,            # 클러스터 수
    init='k-means++',        # 초기화 방법: 'k-means++', 'random'
    n_init=10,               # 서로 다른 초기화로 실행 횟수
    max_iter=300,            # 최대 반복 횟수
    tol=1e-4,                # 수렴 허용 오차
    random_state=42,
    algorithm='lloyd'        # 'lloyd', 'elkan'
)

kmeans.fit(X)

# 주요 속성
print("K-Means 속성:")
print(f"  cluster_centers_: {kmeans.cluster_centers_.shape}")
print(f"  labels_: {np.unique(kmeans.labels_)}")
print(f"  inertia_: {kmeans.inertia_:.2f}")
print(f"  n_iter_: {kmeans.n_iter_}")
```

### 2.3 최적 k 선택 - 엘보우 방법

```python
# 엘보우 방법 (Elbow Method)
inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# 시각화
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-', markersize=10)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.xticks(K_range)
plt.grid(True, alpha=0.3)
plt.show()

# 기울기 변화로 최적 k 찾기
print("엘보우 지점을 눈으로 확인: 그래프가 급격히 꺾이는 지점")
```

### 2.4 최적 k 선택 - 실루엣 분석

```python
from sklearn.metrics import silhouette_score, silhouette_samples

"""
실루엣 계수:
- 범위: -1 ~ 1
- 1에 가까움: 잘 분리된 클러스터
- 0에 가까움: 클러스터 경계에 있음
- 음수: 잘못된 클러스터에 할당

공식:
s(i) = (b(i) - a(i)) / max(a(i), b(i))
- a(i): 같은 클러스터 내 평균 거리
- b(i): 가장 가까운 다른 클러스터까지 평균 거리
"""

# 실루엣 점수 계산
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

# 시각화
plt.figure(figsize=(10, 6))
plt.plot(K_range, silhouette_scores, 'go-', markersize=10)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis for Optimal k')
plt.xticks(K_range)
plt.grid(True, alpha=0.3)
plt.show()

best_k = K_range[np.argmax(silhouette_scores)]
print(f"최적 k (실루엣): {best_k}")
print(f"최고 실루엣 점수: {max(silhouette_scores):.4f}")
```

### 2.5 실루엣 다이어그램

```python
from sklearn.metrics import silhouette_samples
import matplotlib.cm as cm

# 실루엣 다이어그램
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for idx, k in enumerate([2, 3, 4, 5]):
    ax = axes[idx]

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    silhouette_avg = silhouette_score(X, labels)
    sample_silhouette_values = silhouette_samples(X, labels)

    y_lower = 10
    for i in range(k):
        ith_cluster_values = sample_silhouette_values[labels == i]
        ith_cluster_values.sort()

        size_cluster_i = ith_cluster_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / k)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_values,
                         facecolor=color, edgecolor=color, alpha=0.7)

        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster")
    ax.set_title(f"k={k}, Avg Score={silhouette_avg:.3f}")

plt.tight_layout()
plt.show()
```

---

## 3. K-Means++

```python
"""
K-Means++ 초기화:
- 기본 K-Means의 초기화 문제 해결
- 중심점을 더 멀리 떨어지게 초기화

알고리즘:
1. 첫 중심점을 무작위로 선택
2. 각 데이터 포인트에서 가장 가까운 중심점까지의 거리² 계산
3. 거리²에 비례하는 확률로 다음 중심점 선택
4. k개의 중심점이 선택될 때까지 반복

장점:
- 수렴 속도 향상
- 더 좋은 지역 최적해 찾음
- sklearn의 기본값
"""

# 초기화 방법 비교
init_methods = ['k-means++', 'random']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, init in zip(axes, init_methods):
    # 여러 번 실행하여 inertia 비교
    inertias = []
    for _ in range(10):
        kmeans = KMeans(n_clusters=4, init=init, n_init=1, random_state=None)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    ax.hist(inertias, bins=20, edgecolor='black')
    ax.set_xlabel('Inertia')
    ax.set_ylabel('Count')
    ax.set_title(f'init="{init}"\nMean={np.mean(inertias):.2f}, Std={np.std(inertias):.2f}')

plt.tight_layout()
plt.show()
```

---

## 4. Mini-Batch K-Means

```python
from sklearn.cluster import MiniBatchKMeans

"""
Mini-Batch K-Means:
- 대용량 데이터에 적합
- 미니배치로 중심점 업데이트
- 빠르지만 약간의 정확도 손실
"""

# 큰 데이터셋 생성
X_large, _ = make_blobs(n_samples=10000, centers=10, random_state=42)

# 시간 비교
from time import time

# 일반 K-Means
start = time()
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
kmeans.fit(X_large)
kmeans_time = time() - start

# Mini-Batch K-Means
start = time()
mbkmeans = MiniBatchKMeans(n_clusters=10, random_state=42, batch_size=100)
mbkmeans.fit(X_large)
mbkmeans_time = time() - start

print("K-Means vs Mini-Batch K-Means:")
print(f"  K-Means: 시간={kmeans_time:.4f}s, Inertia={kmeans.inertia_:.2f}")
print(f"  Mini-Batch: 시간={mbkmeans_time:.4f}s, Inertia={mbkmeans.inertia_:.2f}")
print(f"  속도 향상: {kmeans_time / mbkmeans_time:.2f}x")
```

---

## 5. DBSCAN

### 5.1 DBSCAN 알고리즘

```python
"""
DBSCAN (Density-Based Spatial Clustering of Applications with Noise):

핵심 개념:
1. 핵심점 (Core Point): eps 반경 내에 min_samples개 이상의 포인트
2. 경계점 (Border Point): 핵심점의 이웃이지만 자체적으로 핵심점 아님
3. 노이즈 (Noise): 핵심점도 아니고 경계점도 아닌 포인트

알고리즘:
1. 아직 방문하지 않은 포인트 선택
2. eps 반경 내 이웃 찾기
3. 이웃 수 >= min_samples이면 핵심점으로 클러스터 시작
4. 클러스터 확장 (밀도 연결)
5. 모든 포인트 방문할 때까지 반복

장점:
- 클러스터 수 자동 결정
- 임의의 모양 클러스터 발견
- 노이즈 처리
- 클러스터 크기에 강건

단점:
- eps, min_samples 튜닝 필요
- 밀도가 다른 클러스터 처리 어려움
"""

from sklearn.cluster import DBSCAN

# 비선형 데이터
X_moons, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

# DBSCAN
dbscan = DBSCAN(
    eps=0.2,             # 이웃 반경
    min_samples=5,       # 핵심점 최소 이웃 수
    metric='euclidean'   # 거리 메트릭
)
labels = dbscan.fit_predict(X_moons)

# 시각화
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
kmeans_moons = KMeans(n_clusters=2, random_state=42)
plt.scatter(X_moons[:, 0], X_moons[:, 1],
            c=kmeans_moons.fit_predict(X_moons), cmap='viridis', edgecolors='black')
plt.title('K-Means (k=2)')

plt.subplot(1, 2, 2)
plt.scatter(X_moons[:, 0], X_moons[:, 1],
            c=labels, cmap='viridis', edgecolors='black')
plt.title(f'DBSCAN (eps=0.2, min_samples=5)\nClusters: {len(set(labels)) - (1 if -1 in labels else 0)}')

plt.tight_layout()
plt.show()

# 노이즈 포인트 수
n_noise = list(labels).count(-1)
print(f"노이즈 포인트 수: {n_noise}")
print(f"클러스터 수: {len(set(labels)) - (1 if -1 in labels else 0)}")
```

### 5.2 eps 선택 - k-distance 그래프

```python
from sklearn.neighbors import NearestNeighbors

# k-distance 계산
k = 5  # min_samples와 동일하게
neighbors = NearestNeighbors(n_neighbors=k)
neighbors.fit(X_moons)
distances, _ = neighbors.kneighbors(X_moons)

# k번째 이웃까지 거리 (정렬)
k_distances = np.sort(distances[:, k-1])

# 시각화
plt.figure(figsize=(10, 6))
plt.plot(k_distances)
plt.xlabel('Data Points (sorted)')
plt.ylabel(f'{k}-th Nearest Neighbor Distance')
plt.title('k-Distance Graph for eps Selection')
plt.grid(True, alpha=0.3)
plt.axhline(y=0.2, color='r', linestyle='--', label='eps=0.2')
plt.legend()
plt.show()

print("엘보우 지점의 y값을 eps로 사용")
```

### 5.3 eps와 min_samples 효과

```python
# 파라미터 효과 비교
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
eps_values = [0.1, 0.2, 0.3]
min_samples_values = [3, 5, 10]

for i, eps in enumerate(eps_values):
    for j, min_samples in enumerate(min_samples_values):
        ax = axes[i, j]

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_moons)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        ax.scatter(X_moons[:, 0], X_moons[:, 1], c=labels, cmap='viridis', edgecolors='black')
        ax.set_title(f'eps={eps}, min_samples={min_samples}\nClusters={n_clusters}, Noise={n_noise}')

plt.tight_layout()
plt.show()
```

---

## 6. 계층적 군집화 (Hierarchical Clustering)

### 6.1 Agglomerative Clustering

```python
"""
계층적 군집화:

1. Agglomerative (Bottom-up):
   - 각 데이터 포인트를 개별 클러스터로 시작
   - 가장 유사한 클러스터 쌍을 병합
   - 원하는 클러스터 수까지 반복

2. Divisive (Top-down):
   - 모든 데이터를 하나의 클러스터로 시작
   - 반복적으로 분할

연결 기준 (Linkage):
- single: 가장 가까운 점 간 거리
- complete: 가장 먼 점 간 거리
- average: 평균 거리
- ward: 분산 최소화 (기본값)
"""

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Agglomerative Clustering
agg_clf = AgglomerativeClustering(
    n_clusters=4,
    metric='euclidean',
    linkage='ward'
)
labels = agg_clf.fit_predict(X)

# 시각화
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolors='black')
plt.title('Agglomerative Clustering (Ward linkage)')
plt.show()
```

### 6.2 덴드로그램

```python
# 덴드로그램 생성
plt.figure(figsize=(15, 8))

# scipy linkage 사용
Z = linkage(X, method='ward')

# 덴드로그램 그리기
dendrogram(
    Z,
    truncate_mode='lastp',  # 마지막 p개 클러스터만
    p=20,
    leaf_rotation=90,
    leaf_font_size=10,
    show_contracted=True
)

plt.xlabel('Sample Index or (Cluster Size)')
plt.ylabel('Distance')
plt.title('Dendrogram')
plt.show()
```

### 6.3 연결 기준 비교

```python
# 연결 기준 비교
linkage_methods = ['single', 'complete', 'average', 'ward']

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for ax, linkage in zip(axes, linkage_methods):
    if linkage == 'ward':
        agg = AgglomerativeClustering(n_clusters=4, linkage=linkage)
    else:
        agg = AgglomerativeClustering(n_clusters=4, linkage=linkage, metric='euclidean')
    labels = agg.fit_predict(X)

    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolors='black')
    ax.set_title(f'Linkage: {linkage}')

plt.tight_layout()
plt.show()
```

---

## 7. Gaussian Mixture Model (GMM)

```python
"""
GMM (Gaussian Mixture Model):
- 확률 기반 클러스터링
- 각 클러스터가 가우시안 분포를 따른다고 가정
- EM 알고리즘으로 학습
- 소프트 클러스터링 가능 (각 클러스터 소속 확률)

K-Means vs GMM:
- K-Means: 하드 클러스터링, 원형 클러스터만
- GMM: 소프트 클러스터링, 타원형 클러스터 가능
"""

from sklearn.mixture import GaussianMixture

# 데이터 생성 (타원형)
np.random.seed(42)
X_ellipse = np.concatenate([
    np.random.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], 150),
    np.random.multivariate_normal([3, 3], [[1, -0.5], [-0.5, 1]], 150)
])

# K-Means vs GMM 비교
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# K-Means
kmeans = KMeans(n_clusters=2, random_state=42)
labels_kmeans = kmeans.fit_predict(X_ellipse)
axes[0].scatter(X_ellipse[:, 0], X_ellipse[:, 1], c=labels_kmeans, cmap='viridis', edgecolors='black')
axes[0].set_title('K-Means')

# GMM
gmm = GaussianMixture(n_components=2, random_state=42)
labels_gmm = gmm.fit_predict(X_ellipse)
axes[1].scatter(X_ellipse[:, 0], X_ellipse[:, 1], c=labels_gmm, cmap='viridis', edgecolors='black')
axes[1].set_title('GMM')

plt.tight_layout()
plt.show()

# GMM 확률 출력
proba = gmm.predict_proba(X_ellipse[:5])
print("처음 5개 샘플의 클러스터 소속 확률:")
print(proba)
```

### 7.1 GMM 파라미터

```python
# GMM 설정
gmm = GaussianMixture(
    n_components=2,          # 클러스터(성분) 수
    covariance_type='full',  # 'full', 'tied', 'diag', 'spherical'
    max_iter=100,
    n_init=1,
    random_state=42
)
gmm.fit(X_ellipse)

# 학습된 파라미터
print("GMM 파라미터:")
print(f"평균:\n{gmm.means_}")
print(f"\n공분산 (첫 번째 성분):\n{gmm.covariances_[0]}")
print(f"\n가중치: {gmm.weights_}")
```

### 7.2 BIC/AIC로 성분 수 선택

```python
"""
BIC (Bayesian Information Criterion):
- 모델 복잡도 패널티 포함
- 낮을수록 좋음

AIC (Akaike Information Criterion):
- BIC보다 덜 엄격한 패널티
"""

# BIC/AIC 계산
n_components_range = range(1, 7)
bics = []
aics = []

for n in n_components_range:
    gmm = GaussianMixture(n_components=n, random_state=42)
    gmm.fit(X_ellipse)
    bics.append(gmm.bic(X_ellipse))
    aics.append(gmm.aic(X_ellipse))

# 시각화
plt.figure(figsize=(10, 6))
plt.plot(n_components_range, bics, 'o-', label='BIC')
plt.plot(n_components_range, aics, 's-', label='AIC')
plt.xlabel('Number of Components')
plt.ylabel('Score')
plt.title('GMM: Optimal Number of Components')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"BIC 최적 성분 수: {n_components_range[np.argmin(bics)]}")
print(f"AIC 최적 성분 수: {n_components_range[np.argmin(aics)]}")
```

---

## 8. 클러스터링 평가

### 8.1 내부 평가 지표

```python
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

"""
내부 평가 지표 (레이블 불필요):

1. 실루엣 점수: -1 ~ 1, 높을수록 좋음
2. Calinski-Harabasz: 높을수록 좋음
3. Davies-Bouldin: 낮을수록 좋음
"""

# K-Means 적용
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

# 평가
print("내부 평가 지표:")
print(f"  실루엣 점수: {silhouette_score(X, labels):.4f}")
print(f"  Calinski-Harabasz: {calinski_harabasz_score(X, labels):.4f}")
print(f"  Davies-Bouldin: {davies_bouldin_score(X, labels):.4f}")
```

### 8.2 외부 평가 지표

```python
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score

"""
외부 평가 지표 (실제 레이블 필요):

1. Adjusted Rand Index (ARI): -1 ~ 1, 1이 완벽
2. Normalized Mutual Information (NMI): 0 ~ 1, 1이 완벽
3. Homogeneity: 각 클러스터가 단일 클래스 포함
"""

# 실제 레이블 있는 경우
print("\n외부 평가 지표:")
print(f"  ARI: {adjusted_rand_score(y_true, labels):.4f}")
print(f"  NMI: {normalized_mutual_info_score(y_true, labels):.4f}")
print(f"  Homogeneity: {homogeneity_score(y_true, labels):.4f}")
```

---

## 9. 클러스터링 알고리즘 비교

```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# 다양한 데이터셋
datasets = [
    make_blobs(n_samples=300, centers=4, random_state=42),
    make_moons(n_samples=300, noise=0.05, random_state=42),
    make_circles(n_samples=300, factor=0.5, noise=0.05, random_state=42)
]
dataset_names = ['Blobs', 'Moons', 'Circles']

# 클러스터링 알고리즘
algorithms = [
    ('K-Means', KMeans(n_clusters=2, random_state=42, n_init=10)),
    ('DBSCAN', DBSCAN(eps=0.3, min_samples=5)),
    ('Agglomerative', AgglomerativeClustering(n_clusters=2)),
    ('GMM', GaussianMixture(n_components=2, random_state=42)),
    ('Spectral', SpectralClustering(n_clusters=2, random_state=42, affinity='nearest_neighbors'))
]

# 시각화
fig, axes = plt.subplots(len(datasets), len(algorithms), figsize=(20, 12))

for i, (X_data, _) in enumerate(datasets):
    # 스케일링
    X_scaled = StandardScaler().fit_transform(X_data)

    for j, (name, algo) in enumerate(algorithms):
        ax = axes[i, j]

        if name == 'GMM':
            algo.fit(X_scaled)
            labels = algo.predict(X_scaled)
        else:
            labels = algo.fit_predict(X_scaled)

        ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis',
                   edgecolors='black', s=30)

        if i == 0:
            ax.set_title(name)
        if j == 0:
            ax.set_ylabel(dataset_names[i])

plt.tight_layout()
plt.show()
```

---

## 10. 실전 예제: 고객 세분화

```python
# 고객 데이터 시뮬레이션
np.random.seed(42)
n_customers = 500

data = {
    'Age': np.concatenate([
        np.random.normal(25, 5, 150),
        np.random.normal(45, 10, 200),
        np.random.normal(65, 8, 150)
    ]),
    'Income': np.concatenate([
        np.random.normal(30000, 5000, 150),
        np.random.normal(60000, 15000, 200),
        np.random.normal(45000, 10000, 150)
    ]),
    'Spending_Score': np.concatenate([
        np.random.normal(70, 15, 150),
        np.random.normal(50, 20, 200),
        np.random.normal(30, 10, 150)
    ])
}

import pandas as pd
df = pd.DataFrame(data)

# 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 최적 k 찾기
silhouette_scores = []
for k in range(2, 8):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

best_k = range(2, 8)[np.argmax(silhouette_scores)]
print(f"최적 클러스터 수: {best_k}")

# 최종 클러스터링
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# 클러스터별 특성 분석
print("\n클러스터별 평균:")
print(df.groupby('Cluster').mean())

# 시각화
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121)
scatter = ax1.scatter(df['Age'], df['Income'], c=df['Cluster'], cmap='viridis')
ax1.set_xlabel('Age')
ax1.set_ylabel('Income')
ax1.set_title('Age vs Income')
plt.colorbar(scatter, ax=ax1)

ax2 = fig.add_subplot(122)
scatter = ax2.scatter(df['Income'], df['Spending_Score'], c=df['Cluster'], cmap='viridis')
ax2.set_xlabel('Income')
ax2.set_ylabel('Spending Score')
ax2.set_title('Income vs Spending Score')
plt.colorbar(scatter, ax=ax2)

plt.tight_layout()
plt.show()
```

---

## 연습 문제

### 문제 1: 엘보우 방법
엘보우 방법으로 최적의 k를 찾으세요.

```python
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=500, centers=5, random_state=42)

# 풀이
inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.plot(range(1, 11), inertias, 'o-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()
```

### 문제 2: DBSCAN 적용
노이즈가 있는 데이터에 DBSCAN을 적용하세요.

```python
from sklearn.datasets import make_moons

X, _ = make_moons(n_samples=300, noise=0.1, random_state=42)

# 풀이
dbscan = DBSCAN(eps=0.2, min_samples=5)
labels = dbscan.fit_predict(X)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"클러스터 수: {n_clusters}")
print(f"노이즈 포인트: {n_noise}")

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title('DBSCAN Clustering')
plt.show()
```

### 문제 3: 클러스터링 평가
여러 평가 지표로 클러스터링 결과를 평가하세요.

```python
from sklearn.metrics import silhouette_score, calinski_harabasz_score

X, y_true = make_blobs(n_samples=300, centers=4, random_state=42)

# 풀이
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

print(f"실루엣 점수: {silhouette_score(X, labels):.4f}")
print(f"Calinski-Harabasz: {calinski_harabasz_score(X, labels):.4f}")
```

---

## 요약

| 알고리즘 | 특징 | 장점 | 단점 |
|----------|------|------|------|
| K-Means | 중심점 기반 | 빠름, 간단 | k 지정 필요, 구형 클러스터만 |
| DBSCAN | 밀도 기반 | 노이즈 처리, 임의 모양 | eps, min_samples 튜닝 |
| Hierarchical | 병합/분할 | 덴드로그램 | 느림, 대용량에 부적합 |
| GMM | 확률 기반 | 소프트 클러스터링, 타원형 | 많은 파라미터 |

### 클러스터링 선택 가이드

| 상황 | 권장 알고리즘 |
|------|---------------|
| 큰 데이터, 빠른 처리 | K-Means, Mini-Batch K-Means |
| 노이즈가 많은 데이터 | DBSCAN |
| 비구형 클러스터 | DBSCAN, Spectral |
| 클러스터 계층 분석 | Hierarchical |
| 확률적 할당 필요 | GMM |

### 평가 지표 요약

| 지표 | 범위 | 좋은 값 | 레이블 필요 |
|------|------|---------|-------------|
| 실루엣 점수 | -1 ~ 1 | 높음 | X |
| Calinski-Harabasz | 0 ~ ∞ | 높음 | X |
| Davies-Bouldin | 0 ~ ∞ | 낮음 | X |
| ARI | -1 ~ 1 | 높음 | O |
| NMI | 0 ~ 1 | 높음 | O |
