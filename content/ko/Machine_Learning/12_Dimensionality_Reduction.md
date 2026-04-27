# 차원 축소 (Dimensionality Reduction)

**이전**: [클러스터링](./11_Clustering.md) | **다음**: [파이프라인과 실무](./13_Pipelines_and_Practice.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 차원의 저주(curse of dimensionality)를 설명하고 고차원 데이터가 모델 성능을 저하시키는 이유 이해하기
2. 특성 선택(feature selection)과 특성 추출(feature extraction) 접근법 구별하기
3. PCA를 구현하고 주성분(principal components), 설명된 분산 비율(explained variance ratios), 스크리 플롯(scree plots) 해석하기
4. t-SNE와 UMAP을 비선형 시각화에 적용하고 PCA와 동작 방식 비교하기
5. 누적 설명 분산(cumulative explained variance) 임계값을 사용해 PCA 구성 요소 수 결정하기
6. scikit-learn을 이용해 필터(filter), 래퍼(wrapper), 임베디드(embedded) 특성 선택 방법 구현하기
7. 차원 축소가 하위 분류(downstream classification) 성능에 미치는 영향 평가하기

---

실제 데이터셋에는 수십, 수백, 심지어 수천 개의 특성이 포함되는 경우가 많습니다 -- 그러나 유용한 정보의 대부분은 훨씬 적은 수의 방향에 집중되어 있는 경우가 대부분입니다. 차원 축소를 사용하면 그러한 방향을 찾아 노이즈와 중복을 제거하여 모델 학습 속도를 높이고, 시각화를 의미 있게 만들며, 차원의 저주가 알고리즘을 마비시키지 않도록 합니다. 이 기법들을 숙달하는 것은 유전체학부터 컴퓨터 비전(computer vision), NLP 임베딩(NLP embeddings)까지 고차원 데이터를 다루기 위한 전제 조건입니다.

> **벽 위의 그림자.** 복잡한 3D 조각상에 손전등을 비춘다고 상상해 보세요 -- 벽에 비친 그림자는 조각상의 핵심 형태를 담은 2D 표현이지만, 일부 세부 정보는 손실됩니다. PCA도 같은 방식으로 작동합니다. 즉, 고차원 데이터를 더 낮은 차원으로 투영할 때 가장 정보가 풍부한 "그림자"를 만들어내는 "시점(주성분)"을 찾습니다. 첫 번째 주성분(PC1)은 분산을 가장 많이 보존하는 방향 -- 그림자를 가장 잘 알아볼 수 있게 하는 각도 -- 입니다.

---

## 1. 차원 축소의 필요성

### 1.1 차원의 저주 (Curse of Dimensionality)

```python
"""
차원의 저주:
1. 고차원에서 데이터 포인트 간 거리가 비슷해짐
2. 데이터가 희소해짐 (sparse)
3. 모델 학습에 더 많은 데이터 필요
4. 과적합 위험 증가
5. 계산 비용 증가

차원 축소의 목적:
1. 시각화 (2D/3D)
2. 노이즈 제거
3. 계산 효율성
4. 다중공선성 제거
5. 특성 추출
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, load_iris, fetch_olivetti_faces
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 차원의 저주 데모: 고차원에서 거리 분포
np.random.seed(42)

def distance_distribution(n_dims, n_points=1000):
    """고차원에서 거리 분포 확인"""
    points = np.random.rand(n_points, n_dims)
    # 랜덤 포인트 쌍 간 거리
    idx = np.random.choice(n_points, size=(500, 2), replace=False)
    distances = [np.linalg.norm(points[i] - points[j]) for i, j in idx]
    return distances

# 다양한 차원에서 거리 분포
dims = [2, 10, 100, 1000]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, d in zip(axes, dims):
    distances = distance_distribution(d)
    ax.hist(distances, bins=30, edgecolor='black')
    ax.set_title(f'Dim={d}\nMean={np.mean(distances):.2f}, Std={np.std(distances):.2f}')
    ax.set_xlabel('Distance')

plt.tight_layout()
plt.show()

print("차원이 증가할수록 거리 분포가 좁아짐 → 포인트들이 비슷한 거리에 위치")
```

---

## 2. 주성분 분석 (PCA)

### 이론: PCA — 공분산 행렬의 고윳값 분해

PCA는 데이터가 최대 분산을 갖는 직교 방향을 찾습니다. 각 열의 평균이 0이 되도록 데이터를 중심화하고, `X ∈ ℝ^{N×p}`로 쌓습니다. 경험적 공분산 행렬은

```
S = (1/(N-1)) · XᵀX        ∈ ℝ^{p×p}
```

PCA가 푸는 문제: 투영 `Xv`의 분산을 최대화하는 단위 벡터 `v` 찾기. `Xv`의 분산은 `vᵀSv`이고, `‖v‖ = 1` 제약 하에 `vᵀSv`를 최대화하면(라그랑주 곱셈자를 통해) 고유값 문제가 됩니다:

```
S v = λ v
```

주성분 `v_1, v_2, ..., v_p`는 `S`의 고유벡터로, 고유값 크기로 정렬됩니다. 고유값 `λ_k`는 데이터를 `v_k`에 투영했을 때의 분산 *그 자체*입니다. 상위 `r`개 고유벡터를 고르면 어떤 선택된 랭크 `r`에 대해서도 가장 많은 분산을 보존하는 투영이 됩니다.

### 이론: SVD로 보는 PCA — 수치적으로 더 안정적인 경로

`S = XᵀX`를 형성하면 조건수가 제곱됩니다 — 특성이 거의 공선일 때 수치 안정성에 나쁨. `X`의 **특이값 분해(SVD)**는 `S`를 형성하지 않고 같은 답을 줍니다:

```
X = U Σ Vᵀ          (U: N×p, Σ: 대각, V: p×p)
```

그러면 `XᵀX = V Σ² Vᵀ`이므로, `S`의 고유벡터는 `V`의 열(우특이벡터)이고, 고유값은 `σ_k² / (N-1)`. 현대 PCA 구현(scikit-learn 포함)은 SVD를 직접 사용. 비용은 `O(min(N²p, Np²))`.

### 이론: PCA가 보존하는 것과 잃는 것

상위 `r`개 성분이 보존하는 분산 비율은:

```
explained_variance_ratio(r) = Σ_{k=1..r} λ_k  /  Σ_{k=1..p} λ_k
```

이것이 `r`을 선택하는 표준 그래프 — 누적으로 그리고 엘보를 찾음.

PCA는 **흥미로운 구조가 선형이고 최대 분산과 정렬**되어 있다고 가정합니다. 두 가정 모두 실패할 수 있습니다:
- 비선형 매니폴드(예: 스위스 롤)는 PCA에 의해 평평한 덩어리로 펼쳐집니다 — 매니폴드 구조가 파괴.
- 분산과 "흥미로움"이 동의하지 않을 수 있음 — 클래스 레이블이 PCA가 버리는 저분산 방향을 따라 변할 수 있음.

특성이 다른 스케일에 있을 때 PCA 전에 표준화(그렇지 않으면 가장 큰 크기의 특성이 분산을 지배).

### 2.1 PCA의 원리

```python
"""
PCA (Principal Component Analysis):
- 데이터의 분산을 최대화하는 축(주성분)을 찾음
- 고차원 → 저차원 투영
- 선형 변환

수학적 원리:
1. 데이터 중심화 (평균 0)
2. 공분산 행렬 계산
3. 고유값 분해 (eigendecomposition)
4. 고유값이 큰 순서로 고유벡터(주성분) 선택
5. 선택된 주성분으로 데이터 투영

주성분:
- 첫 번째 주성분: 분산이 가장 큰 방향
- 두 번째 주성분: 첫 번째와 직교하면서 분산이 큰 방향
- n번째 주성분: 이전 주성분들과 직교
"""

from sklearn.decomposition import PCA

# 2D 예시로 PCA 시각화
np.random.seed(42)
X_2d = np.dot(np.random.randn(200, 2), [[2, 1], [1, 2]])

# PCA 적용
pca = PCA(n_components=2)
pca.fit(X_2d)

# 시각화
plt.figure(figsize=(10, 8))
plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.5)

# 주성분 방향 (화살표)
mean = pca.mean_
for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
    end = mean + comp * np.sqrt(var) * 3
    plt.arrow(mean[0], mean[1], end[0]-mean[0], end[1]-mean[1],
              head_width=0.3, head_length=0.2, fc=f'C{i}', ec=f'C{i}',
              linewidth=2, label=f'PC{i+1} (Var: {var:.2f})')

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('PCA: Principal Components')
plt.legend()
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.show()

print(f"주성분:\n{pca.components_}")
print(f"설명된 분산: {pca.explained_variance_}")
print(f"설명된 분산 비율: {pca.explained_variance_ratio_}")
```

### 2.2 sklearn PCA 사용법

```python
from sklearn.decomposition import PCA

# Iris 데이터
iris = load_iris()
X = iris.data
y = iris.target

# 스케일링 (PCA 전 필수)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA 적용 (2차원으로 축소)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"원본 형상: {X.shape}")
print(f"PCA 후 형상: {X_pca.shape}")
print(f"설명된 분산 비율: {pca.explained_variance_ratio_}")
print(f"누적 설명 분산: {sum(pca.explained_variance_ratio_):.4f}")

# 시각화
plt.figure(figsize=(10, 8))
for i, target_name in enumerate(iris.target_names):
    mask = y == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=target_name, alpha=0.7)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
plt.title('PCA: Iris Dataset')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 2.3 주성분 수 선택

```python
# 전체 주성분으로 PCA
pca_full = PCA()
pca_full.fit(X_scaled)

# 누적 설명 분산
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 개별 분산
axes[0].bar(range(1, len(pca_full.explained_variance_ratio_)+1),
            pca_full.explained_variance_ratio_, edgecolor='black')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Explained Variance Ratio')
axes[0].set_title('Individual Explained Variance')

# 누적 분산
axes[1].plot(range(1, len(cumulative_variance)+1), cumulative_variance, 'o-')
axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% variance')
axes[1].axhline(y=0.99, color='g', linestyle='--', label='99% variance')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Explained Variance')
axes[1].set_title('Cumulative Explained Variance')
axes[1].legend()

plt.tight_layout()
plt.show()

# 95% 분산을 설명하는 주성분 수
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"95% 분산 설명에 필요한 주성분 수: {n_components_95}")
```

### 2.4 PCA로 분산 비율 지정

```python
# 분산 비율로 주성분 수 자동 결정
pca_95 = PCA(n_components=0.95)  # 95% 분산 설명
X_pca_95 = pca_95.fit_transform(X_scaled)

print(f"95% 분산 → {pca_95.n_components_}개 주성분 선택")
print(f"실제 설명된 분산: {sum(pca_95.explained_variance_ratio_):.4f}")

# 다양한 분산 비율
for var_ratio in [0.8, 0.9, 0.95, 0.99]:
    pca_temp = PCA(n_components=var_ratio)
    pca_temp.fit(X_scaled)
    print(f"{var_ratio*100:.0f}% 분산 → {pca_temp.n_components_}개 주성분")
```

### 2.5 PCA 활용: 노이즈 제거

```python
# 숫자 이미지 데이터
digits = load_digits()
X_digits = digits.data
y_digits = digits.target

# 노이즈 추가
np.random.seed(42)
X_noisy = X_digits + np.random.normal(0, 4, X_digits.shape)

# PCA로 노이즈 제거 (주요 주성분만 유지)
pca_denoise = PCA(n_components=20)
X_reduced = pca_denoise.fit_transform(X_noisy)
X_denoised = pca_denoise.inverse_transform(X_reduced)

# 시각화
fig, axes = plt.subplots(3, 10, figsize=(15, 5))

for i in range(10):
    # 원본
    axes[0, i].imshow(X_digits[i].reshape(8, 8), cmap='gray')
    axes[0, i].axis('off')
    if i == 0:
        axes[0, i].set_title('Original')

    # 노이즈
    axes[1, i].imshow(X_noisy[i].reshape(8, 8), cmap='gray')
    axes[1, i].axis('off')
    if i == 0:
        axes[1, i].set_title('Noisy')

    # 복원
    axes[2, i].imshow(X_denoised[i].reshape(8, 8), cmap='gray')
    axes[2, i].axis('off')
    if i == 0:
        axes[2, i].set_title('Denoised')

plt.suptitle('PCA for Noise Reduction')
plt.tight_layout()
plt.show()
```

---

## 3. t-SNE

### 이론: t-SNE — 확률적 국소 이웃 보존

t-SNE(van der Maaten & Hinton, 2008)는 전역 기하학을 포기하고 대신 국소 이웃을 보존합니다. 레시피:

**1. 고차원에서 이웃에 대한 확률 분포 만들기.** 각 쌍 `(i, j)`에 대해, `j`가 `i`의 이웃일 조건부 확률을 정의:

```
p_{j|i} = exp(-‖x_i - x_j‖² / 2σ_i²)  /  Σ_{k ≠ i} exp(-‖x_i - x_k‖² / 2σ_i²)
p_{ij}  = (p_{j|i} + p_{i|j}) / (2N)        ← 대칭화
```

대역폭 `σ_i`는 `p_{·|i}`의 엔트로피가 `log(perplexity)`가 되도록 점별로 설정 — `perplexity`는 대략 유효 이웃 수(통상 5–50). 따라서 각 점은 국소 밀도에 적응된 유사도 분포를 가집니다.

**2. 무거운 꼬리(Student-t) 커널을 사용하여 저차원에서 이웃에 대한 확률 분포 만들기.** 저차원 임베딩 `y_i`에 대해:

```
q_{ij} = (1 + ‖y_i - y_j‖²)⁻¹  /  Σ_{k ≠ l} (1 + ‖y_k - y_l‖²)⁻¹
```

Student-t(자유도 1)는 가우시안보다 무거운 꼬리를 가져, 저차원에서 먼 점이 페널티 없이 더 멀리 임베딩될 수 있게 합니다 — PCA 스타일 방법이 어려워하는 "혼잡 문제(crowding problem)"를 해결.

**3. 두 분포 사이의 KL 발산 최소화.**

```
L = KL(P ∥ Q) = Σ_{ij}  p_{ij} · log(p_{ij} / q_{ij})
```

`L`에 대한 경사 하강이 임베딩을 만듭니다. KL의 비대칭성 — `p_{ij}`이 클 때 `q_{ij}`이 작은 것을 페널티 — 은 t-SNE가 *가까운* 이웃을 강하게 보존하면서 *먼* 이웃을 임의로 이동하는 것을 허용함을 의미. 그래서 t-SNE 플롯이 깔끔한 국소 클러스터를 보이지만 클러스터 간 거리는 해석 불가능합니다.

### 이론: t-SNE 사용 시 주의점

- t-SNE 플롯의 클러스터 *간* 거리는 무의미; *내부* 클러스터 구조만 충실.
- 클러스터 *크기*는 실제 클러스터 부피가 아니라 국소 밀도에 의존 — 작고 조밀한 클러스터가 크게 보임.
- `perplexity` 하이퍼파라미터가 결과를 실질적으로 변화시킴. 항상 여러 값(예: 5, 30, 50)으로 검사.
- 느림: 순진하게 `O(N²)`, Barnes-Hut 근사로 `O(N log N)`. `N ≈ 50K` 이상에서 고통스러움.

### 3.1 t-SNE 원리

```python
"""
t-SNE (t-distributed Stochastic Neighbor Embedding):
- 비선형 차원 축소
- 시각화에 주로 사용 (2D/3D)
- 지역 구조 보존에 뛰어남

원리:
1. 고차원에서 점들 간 유사도를 조건부 확률로 계산
2. 저차원에서 t-분포 기반 유사도 정의
3. KL-divergence 최소화로 저차원 좌표 학습

특징:
- 비선형 관계 포착
- 클러스터 분리에 효과적
- 계산 비용 높음
- 새 데이터 변환 불가 (transform 없음)
- 결과 재현성 문제 (random_state 중요)
"""

from sklearn.manifold import TSNE

# t-SNE 적용
tsne = TSNE(
    n_components=2,
    perplexity=30,          # 지역 이웃 크기 (5-50)
    learning_rate='auto',   # 학습률
    n_iter=1000,            # 반복 횟수
    random_state=42
)

# 시간이 오래 걸리므로 일부만 사용
X_sample = X_digits[:500]
y_sample = y_digits[:500]

X_tsne = tsne.fit_transform(X_sample)

# 시각화
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sample, cmap='tab10', alpha=0.7)
plt.colorbar(scatter)
plt.title('t-SNE: Digits Dataset')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.show()
```

### 3.2 perplexity 파라미터

```python
# perplexity 효과
perplexities = [5, 30, 50, 100]

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for ax, perp in zip(axes, perplexities):
    tsne_temp = TSNE(n_components=2, perplexity=perp, random_state=42)
    X_temp = tsne_temp.fit_transform(X_sample)

    scatter = ax.scatter(X_temp[:, 0], X_temp[:, 1], c=y_sample, cmap='tab10', alpha=0.7)
    ax.set_title(f'perplexity={perp}')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')

plt.tight_layout()
plt.show()

print("perplexity 가이드:")
print("  - 작은 값 (5-10): 지역 구조에 집중")
print("  - 큰 값 (30-50): 전역 구조 고려")
print("  - 데이터 크기에 따라 조절 필요")
```

### 3.3 PCA vs t-SNE 비교

```python
# 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sample)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# 비교 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_sample, cmap='tab10', alpha=0.7)
axes[0].set_title('PCA')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')

scatter2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sample, cmap='tab10', alpha=0.7)
axes[1].set_title('t-SNE')
axes[1].set_xlabel('t-SNE 1')
axes[1].set_ylabel('t-SNE 2')

plt.tight_layout()
plt.show()

print("PCA: 분산 최대화, 선형, 빠름, 전역 구조")
print("t-SNE: 이웃 보존, 비선형, 느림, 지역 구조")
```

---

## 4. UMAP

### 이론: UMAP — 퍼지 단체 복합체

UMAP(McInnes et al., 2018)은 t-SNE의 "국소 이웃 보존" 철학을 공유하지만 다른 수학적 토대 위에 구축: 대수 위상수학의 **퍼지 단순 복합체(fuzzy simplicial complexes)**. 메커니즘:

**1. 고차원에서 퍼지 그래프 구축.** 각 점에 대해 거리의 지수 감쇠 함수로 주어진 가중치를 가진 `k`개 최근접 이웃에 연결, 각 점이 같은 "양"의 연결성을 갖도록 정규화.

**2. 조정 가능한 매개변수 `a, b`(임베딩 거리 함수의 매끄러움)를 사용해 다른 커널로 저차원에서 퍼지 그래프 구축.**

**3. 두 퍼지 그래프 사이의 교차 엔트로피 최소화**(KL 발산을 닮은 목적이지만 양면 — 비슷한 점 간의 인력과 다른 점 간의 척력 모두 페널티).

t-SNE와의 실용적 차이:
- **더 빠름**: 그래프에 대한 확률적 경사 하강으로 에포크당 `O(N · log N)`.
- **더 나은 전역 구조**: 대칭 교차 엔트로피가 t-SNE의 비대칭 KL보다 더 많은 장거리 거리 정보를 보존.
- **재현 가능**: 초기화에 t-SNE보다 덜 민감.
- **임베드 가능**: 변환된 테스트 점을 기존 UMAP 임베딩에 추가 가능(t-SNE는 그런 연산 없음).

하이퍼파라미터 `n_neighbors`는 `perplexity`와 비슷한 역할 — 작은 값은 국소 구조를 강조, 큰 값은 전역을 강조.

```python
"""
UMAP (Uniform Manifold Approximation and Projection):
- t-SNE보다 빠름
- 전역 구조 더 잘 보존
- 새 데이터 변환 가능

# pip install umap-learn
"""

# import umap

# umap_reducer = umap.UMAP(
#     n_neighbors=15,      # 지역 이웃 수
#     min_dist=0.1,        # 포인트 간 최소 거리
#     n_components=2,
#     random_state=42
# )
# X_umap = umap_reducer.fit_transform(X_scaled)

# 설치 없이 설명
print("UMAP 특징:")
print("  - t-SNE보다 빠름")
print("  - 전역 구조 더 잘 보존")
print("  - transform() 지원 (새 데이터 변환)")
print("  - 주요 파라미터: n_neighbors, min_dist")
```

---

## 5. 특성 선택 (Feature Selection)

### 5.1 필터 방법 (Filter Methods)

```python
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile,
    f_classif, mutual_info_classif, chi2
)

"""
필터 방법:
- 모델과 독립적으로 특성 평가
- 빠름, 간단
- 통계적 검정 기반

방법:
1. 분산 기반: VarianceThreshold
2. 상관관계 기반: 타겟과의 상관계수
3. 통계 검정: ANOVA F-value, 카이제곱
4. 정보 이론: 상호 정보량
"""

# 데이터
X, y = load_iris(return_X_y=True)

# ANOVA F-value 기반 특성 선택
selector = SelectKBest(score_func=f_classif, k=2)
X_selected = selector.fit_transform(X, y)

print("ANOVA F-value 특성 선택:")
print(f"원본 특성 수: {X.shape[1]}")
print(f"선택된 특성 수: {X_selected.shape[1]}")
print(f"각 특성 점수: {selector.scores_}")
print(f"선택된 특성 인덱스: {selector.get_support(indices=True)}")

# 상호 정보량 기반
selector_mi = SelectKBest(score_func=mutual_info_classif, k=2)
selector_mi.fit(X, y)
print(f"\n상호 정보량 점수: {selector_mi.scores_}")
```

### 5.2 래퍼 방법 (Wrapper Methods)

```python
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression

"""
래퍼 방법:
- 모델 성능 기반 특성 선택
- 정확하지만 느림
- 과적합 위험

방법:
1. RFE (Recursive Feature Elimination)
2. 전진 선택 (Forward Selection)
3. 후진 제거 (Backward Elimination)
"""

# RFE (재귀적 특성 제거)
model = LogisticRegression(max_iter=1000)
rfe = RFE(estimator=model, n_features_to_select=2, step=1)
rfe.fit(X, y)

print("RFE 특성 선택:")
print(f"선택된 특성: {rfe.get_support()}")
print(f"특성 순위: {rfe.ranking_}")

# RFECV (교차 검증 포함)
rfecv = RFECV(estimator=model, cv=5, scoring='accuracy')
rfecv.fit(X, y)

print(f"\nRFECV 최적 특성 수: {rfecv.n_features_}")
print(f"선택된 특성: {rfecv.get_support()}")

# CV 점수 시각화
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(rfecv.cv_results_['mean_test_score'])+1),
         rfecv.cv_results_['mean_test_score'], 'o-')
plt.xlabel('Number of Features')
plt.ylabel('Cross-Validation Score')
plt.title('RFECV: Optimal Number of Features')
plt.grid(True, alpha=0.3)
plt.show()
```

### 5.3 임베디드 방법 (Embedded Methods)

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso

"""
임베디드 방법:
- 모델 학습 과정에서 특성 선택
- 필터와 래퍼의 중간
- L1 정규화, 트리 기반 모델

방법:
1. L1 정규화 (Lasso)
2. 트리 기반 중요도
"""

# Random Forest 중요도 기반
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# 특성 중요도
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# 시각화
plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), [f'Feature {i}' for i in indices])
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Random Forest Feature Importance')
plt.show()

# SelectFromModel
selector = SelectFromModel(rf, threshold='median')
selector.fit(X, y)
X_selected = selector.transform(X)

print(f"Random Forest 기반 선택된 특성 수: {X_selected.shape[1]}")
print(f"선택된 특성: {selector.get_support()}")
```

---

## 6. 분산 기반 특성 선택

```python
from sklearn.feature_selection import VarianceThreshold

# 샘플 데이터 (분산이 다른 특성)
X_var = np.array([
    [0, 0, 1, 100],
    [0, 0, 0, 101],
    [0, 0, 1, 99],
    [0, 0, 0, 100],
    [0, 0, 1, 102]
])

# 분산이 낮은 특성 제거
selector = VarianceThreshold(threshold=0.5)
X_high_var = selector.fit_transform(X_var)

print("분산 기반 특성 선택:")
print(f"각 특성 분산: {selector.variances_}")
print(f"선택된 특성: {selector.get_support()}")
print(f"원본 형상: {X_var.shape}")
print(f"선택 후 형상: {X_high_var.shape}")
```

---

## 7. 상관관계 기반 특성 제거

```python
import pandas as pd

# 샘플 데이터 (상관된 특성 포함)
np.random.seed(42)
n_samples = 100

X_corr = np.column_stack([
    np.random.randn(n_samples),  # 특성 0
    np.random.randn(n_samples),  # 특성 1
    np.random.randn(n_samples),  # 특성 2
])
# 높은 상관관계 특성 추가
X_corr = np.column_stack([X_corr, X_corr[:, 0] + np.random.randn(n_samples) * 0.1])

df = pd.DataFrame(X_corr, columns=['F0', 'F1', 'F2', 'F3'])

# 상관행렬
corr_matrix = df.corr().abs()

# 상관관계 히트맵
plt.figure(figsize=(8, 6))
plt.imshow(corr_matrix, cmap='coolwarm', vmin=0, vmax=1)
plt.colorbar(label='Correlation')
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.title('Feature Correlation Matrix')

for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                 ha='center', va='center')
plt.show()

# 높은 상관관계 특성 제거 함수
def remove_highly_correlated(df, threshold=0.9):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop), to_drop

df_cleaned, dropped = remove_highly_correlated(df, threshold=0.9)
print(f"제거된 특성: {dropped}")
print(f"남은 특성: {list(df_cleaned.columns)}")
```

---

## 8. 차원 축소 파이프라인

```python
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# 데이터
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# PCA + SVM 파이프라인
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=30)),
    ('svm', SVC(kernel='rbf', random_state=42))
])

# 교차 검증
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
print(f"PCA (30) + SVM CV 점수: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# 전체 특성 vs PCA
pipeline_full = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', random_state=42))
])

cv_scores_full = cross_val_score(pipeline_full, X_train, y_train, cv=5)
print(f"전체 특성 + SVM CV 점수: {cv_scores_full.mean():.4f} (+/- {cv_scores_full.std():.4f})")

print(f"\nPCA로 {X.shape[1]} → 30 차원 축소")
```

---

## 9. Incremental PCA (대용량 데이터)

```python
from sklearn.decomposition import IncrementalPCA

"""
Incremental PCA:
- 대용량 데이터에 적합
- 미니배치로 처리
- 메모리 효율적
"""

# 대용량 데이터 시뮬레이션
X_large = np.random.randn(10000, 100)

# 일반 PCA
pca_regular = PCA(n_components=10)
pca_regular.fit(X_large)

# Incremental PCA
ipca = IncrementalPCA(n_components=10, batch_size=500)
ipca.fit(X_large)

print("일반 PCA vs Incremental PCA:")
print(f"설명된 분산 비율 (일반): {sum(pca_regular.explained_variance_ratio_):.4f}")
print(f"설명된 분산 비율 (증분): {sum(ipca.explained_variance_ratio_):.4f}")

# 배치로 처리 (메모리 효율)
ipca_batch = IncrementalPCA(n_components=10)
for batch_start in range(0, len(X_large), 1000):
    batch = X_large[batch_start:batch_start+1000]
    ipca_batch.partial_fit(batch)

print(f"배치 처리 설명된 분산: {sum(ipca_batch.explained_variance_ratio_):.4f}")
```

---

## 10. 차원 축소 알고리즘 비교

### 이론: PCA vs t-SNE vs UMAP — 언제 무엇을 쓸지

| 필요 | 사용 | 이유 |
|------|------|------|
| 다운스트림 모델용 선형 전처리 | PCA | 빠름, 정확, 분산 보존 |
| 2D/3D에서 클러스터 구조 시각화 | t-SNE 또는 UMAP | 국소 이웃 보존 |
| 전역 거리가 의미 있는 클러스터 시각화 | UMAP | t-SNE보다 나은 전역 구조 |
| 새 테스트 점 임베드 | PCA 또는 UMAP | 둘 다 변환 메서드 유사물 있음; t-SNE는 없음 |
| 큰 `N`에 매우 빠른 임베딩 필요 | UMAP | 최선의 스케일링 |
| 설명된 분산 비율 필요 | PCA | 고유값이 직접 줌 |

흔한 파이프라인: 먼저 PCA로 ~50 차원으로(잡음 제거, 후속 알고리즘 가속), 그다음 2D 시각화에 t-SNE 또는 UMAP.

```python
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.manifold import TSNE, MDS, Isomap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

"""
차원 축소 알고리즘 비교:

1. PCA: 선형, 분산 최대화, 빠름
2. Kernel PCA: 비선형 PCA
3. LDA: 클래스 분리 최대화 (지도 학습)
4. t-SNE: 시각화, 지역 구조
5. UMAP: 시각화, 전역+지역 구조
6. MDS: 거리 보존
7. Isomap: 측지선 거리 보존
"""

# 알고리즘 비교 (작은 데이터셋)
algorithms = {
    'PCA': PCA(n_components=2),
    'Kernel PCA': KernelPCA(n_components=2, kernel='rbf'),
    'LDA': LDA(n_components=2),
    't-SNE': TSNE(n_components=2, random_state=42)
}

# 데이터
X, y = load_iris(return_X_y=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 비교 시각화
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for ax, (name, algo) in zip(axes, algorithms.items()):
    if name == 'LDA':
        X_reduced = algo.fit_transform(X_scaled, y)
    else:
        X_reduced = algo.fit_transform(X_scaled)

    scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis', alpha=0.7)
    ax.set_title(name)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')

plt.tight_layout()
plt.show()
```

---

## 연습 문제

### 문제 1: PCA 적용
Digits 데이터에 PCA를 적용하고 95% 분산을 설명하는 주성분 수를 찾으세요.

```python
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

digits = load_digits()
X = digits.data

# 풀이
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
pca.fit(X_scaled)

cumsum = np.cumsum(pca.explained_variance_ratio_)
n_95 = np.argmax(cumsum >= 0.95) + 1

print(f"95% 분산에 필요한 주성분 수: {n_95}")
print(f"원본 차원: {X.shape[1]}")
```

### 문제 2: t-SNE 시각화
Digits 데이터를 t-SNE로 시각화하세요.

```python
from sklearn.manifold import TSNE

# 풀이 (시간 단축을 위해 일부만)
X_sample = X[:500]
y_sample = digits.target[:500]

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_sample)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sample, cmap='tab10')
plt.colorbar(scatter)
plt.title('t-SNE: Digits')
plt.show()
```

### 문제 3: 특성 선택
Random Forest 중요도 기반으로 상위 20개 특성을 선택하세요.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# 풀이
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, digits.target)

# 상위 20개
selector = SelectFromModel(rf, max_features=20, threshold=-np.inf)
selector.fit(X, digits.target)
X_selected = selector.transform(X)

print(f"선택된 특성 수: {X_selected.shape[1]}")
print(f"선택된 특성 인덱스: {np.where(selector.get_support())[0]}")
```

---

## 요약

| 방법 | 유형 | 특징 | 용도 |
|------|------|------|------|
| PCA | 선형 | 분산 최대화 | 일반적인 차원 축소 |
| Kernel PCA | 비선형 | 커널 트릭 | 비선형 패턴 |
| LDA | 지도 학습 | 클래스 분리 | 분류 전처리 |
| t-SNE | 비선형 | 지역 구조 보존 | 시각화 |
| UMAP | 비선형 | 빠름, 전역 구조 | 시각화 |

### 특성 선택 방법 비교

| 방법 | 유형 | 장점 | 단점 |
|------|------|------|------|
| Filter | 통계 기반 | 빠름 | 특성 간 관계 무시 |
| Wrapper | 모델 기반 | 정확 | 느림, 과적합 |
| Embedded | 학습 중 선택 | 효율적 | 모델 의존적 |

### 차원 축소 선택 가이드

| 상황 | 권장 방법 |
|------|-----------|
| 노이즈 제거, 압축 | PCA |
| 시각화 (2D/3D) | t-SNE, UMAP |
| 분류 전처리 | LDA |
| 비선형 패턴 | Kernel PCA, UMAP |
| 대용량 데이터 | Incremental PCA, TruncatedSVD |
| 특성 해석 필요 | 특성 선택 (Filter/Embedded) |
