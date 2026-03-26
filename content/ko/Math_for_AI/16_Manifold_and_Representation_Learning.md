# 16. 다양체 학습과 표현 학습

## 학습 목표

- 다양체 가설과 고차원 데이터의 저차원 구조를 이해할 수 있다
- 다양체의 수학적 정의와 기하학적 개념(측지 거리, 접선 공간)을 설명할 수 있다
- 선형/비선형 차원 축소 기법(PCA, Isomap, LLE)의 수학적 원리를 이해할 수 있다
- t-SNE의 KL 발산 최소화와 크라우딩 문제 해결 원리를 이해할 수 있다
- UMAP의 퍼지 위상 수학과 리만 기하학적 기초를 이해할 수 있다
- 신경망 기반 표현 학습(오토인코더, 대조 학습)과 다양체의 관계를 설명할 수 있다

---

## 1. 다양체 가설 (Manifold Hypothesis)

### 1.1 고차원 데이터의 저차원 구조

**다양체 가설**: 자연계의 고차원 데이터는 실제로 **저차원 다양체**(low-dimensional manifold) 위에 또는 그 근처에 존재한다.

**예시:**
- **이미지**: $256 \times 256$ RGB 이미지는 $196{,}608$차원 공간의 점이지만, 실제 자연 이미지는 훨씬 낮은 차원의 다양체 위에 존재
- **음성**: 파형 데이터는 고차원이지만, 발음 기관의 자유도는 제한적
- **분자 구조**: 3D 좌표는 많지만, 화학 결합의 제약으로 저차원 다양체 형성

### 1.2 내재 차원 (Intrinsic Dimensionality)

데이터가 실제로 존재하는 최소 차원 $d_{\text{intrinsic}} \ll D$ (주변 공간 차원).

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 스위스 롤 (Swiss Roll) 생성
def generate_swiss_roll(n_samples=1000, noise=0.0):
    """
    3D 공간에 임베딩된 2D 다양체

    Parameters:
    -----------
    n_samples : int
        샘플 수
    noise : float
        노이즈 수준

    Returns:
    --------
    X : ndarray, shape (n_samples, 3)
        3D 공간의 점
    t : ndarray, shape (n_samples,)
        내재 좌표 (회전 각도)
    """
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(n_samples))
    x = t * np.cos(t)
    y = 21 * np.random.rand(n_samples)
    z = t * np.sin(t)

    X = np.column_stack([x, y, z])
    X += noise * np.random.randn(n_samples, 3)

    return X, t

# 스위스 롤 생성
X_swiss, t_swiss = generate_swiss_roll(n_samples=1000, noise=0.5)

# 시각화
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X_swiss[:, 0], X_swiss[:, 1], X_swiss[:, 2],
            c=t_swiss, cmap='viridis', s=10)
ax1.set_title('Swiss Roll (3D embedding)')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

ax2 = fig.add_subplot(122)
ax2.scatter(t_swiss, X_swiss[:, 1], c=t_swiss, cmap='viridis', s=10)
ax2.set_title('Intrinsic 2D coordinates')
ax2.set_xlabel('t (angle)')
ax2.set_ylabel('y')

plt.tight_layout()
plt.savefig('swiss_roll.png', dpi=150, bbox_inches='tight')
print("스위스 롤 시각화 저장 완료")
```

## 2. 다양체의 수학적 기초

### 2.1 위상 다양체 (Topological Manifold)

$d$-차원 위상 다양체 $\mathcal{M}$는 각 점 $p \in \mathcal{M}$ 주변이 **국소적으로 유클리드 공간** $\mathbb{R}^d$와 동형(homeomorphic)인 위상 공간입니다.

**예:**
- **원** $S^1$: 1차원 다양체 (국소적으로 선분처럼 보임)
- **구** $S^2$: 2차원 다양체 (지구 표면처럼 국소적으로 평면)
- **토러스** $T^2$: 2차원 다양체

### 2.2 미분 다양체 (Differentiable Manifold)

좌표 변환이 미분 가능한 다양체. 접선 공간, 곡률 등의 기하학적 개념을 정의할 수 있습니다.

### 2.3 측지 거리 (Geodesic Distance)

다양체 위에서 두 점을 연결하는 **가장 짧은 경로의 길이**.

- **유클리드 거리**: 주변 공간에서의 직선 거리
- **측지 거리**: 다양체를 따라 이동하는 최단 거리

스위스 롤에서: 유클리드 거리가 가까워도 측지 거리는 멀 수 있습니다.

```python
def compute_geodesic_distance_approximation(X, k=10):
    """
    k-최근접 이웃 그래프를 이용한 측지 거리 근사

    Parameters:
    -----------
    X : ndarray, shape (n, d)
        데이터 포인트
    k : int
        최근접 이웃 수

    Returns:
    --------
    D_geodesic : ndarray, shape (n, n)
        측지 거리 행렬 (근사)
    """
    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse.csgraph import shortest_path

    n = X.shape[0]

    # k-최근접 이웃 그래프 생성
    knn_graph = kneighbors_graph(X, n_neighbors=k, mode='distance')

    # 다익스트라 알고리즘으로 최단 경로 계산 (측지 거리 근사)
    D_geodesic = shortest_path(knn_graph, directed=False)

    return D_geodesic

# 측지 거리 계산
D_geodesic = compute_geodesic_distance_approximation(X_swiss, k=10)

# 유클리드 거리
from scipy.spatial.distance import cdist
D_euclidean = cdist(X_swiss, X_swiss)

# 비교
sample_idx = 0
print(f"샘플 {sample_idx}와의 거리 (상위 5개):")
print("\n유클리드 거리:")
euclidean_sorted = np.argsort(D_euclidean[sample_idx])[:6]
print(euclidean_sorted, D_euclidean[sample_idx, euclidean_sorted])

print("\n측지 거리 (근사):")
geodesic_sorted = np.argsort(D_geodesic[sample_idx])[:6]
print(geodesic_sorted, D_geodesic[sample_idx, geodesic_sorted])
```

### 2.4 접선 공간 (Tangent Space)

점 $p \in \mathcal{M}$에서의 접선 공간 $T_p\mathcal{M}$는 다양체가 국소적으로 선형화된 공간입니다.

- **차원**: $\dim(T_p\mathcal{M}) = \dim(\mathcal{M})$
- **기저**: 접선 벡터들
- **응용**: 다양체 위에서의 미분, 최적화

## 3. 차원 축소 기법의 수학

### 3.1 주성분 분석 (PCA)

PCA는 **선형 부분공간**을 찾습니다 (02_Linear_Algebra_Essentials 연결).

데이터의 공분산 행렬 $\Sigma = \frac{1}{n} X^T X$의 고유벡터가 주성분입니다.

**목적**: 재구성 오차 최소화

$$\min_{\mathbf{W}} \| X - X \mathbf{W} \mathbf{W}^T \|_F^2$$

**한계**: 선형 방법이므로 비선형 다양체를 펼칠 수 없습니다.

### 3.2 다차원 척도법 (MDS)

**목표**: 고차원 거리를 저차원에서 보존

$$\min_{\mathbf{Y}} \sum_{i,j} (d_{ij} - \|\mathbf{y}_i - \mathbf{y}_j\|)^2$$

**Classical MDS**: 거리 행렬의 이중 중심화 + 고유값 분해

### 3.3 Isomap (Isometric Mapping)

**아이디어**: 측지 거리를 보존하는 임베딩

1. k-최근접 이웃 그래프 구성
2. 그래프 최단 경로로 측지 거리 근사
3. Classical MDS 적용

```python
from sklearn.manifold import Isomap

def apply_isomap(X, n_components=2, n_neighbors=10):
    """
    Isomap 차원 축소

    Parameters:
    -----------
    X : ndarray, shape (n, d)
        고차원 데이터
    n_components : int
        목표 차원
    n_neighbors : int
        최근접 이웃 수

    Returns:
    --------
    Y : ndarray, shape (n, n_components)
        저차원 임베딩
    """
    isomap = Isomap(n_components=n_components, n_neighbors=n_neighbors)
    Y = isomap.fit_transform(X)
    return Y

Y_isomap = apply_isomap(X_swiss, n_components=2, n_neighbors=10)

plt.figure(figsize=(6, 5))
plt.scatter(Y_isomap[:, 0], Y_isomap[:, 1], c=t_swiss, cmap='viridis', s=10)
plt.title('Isomap embedding (2D)')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.colorbar(label='Original angle t')
plt.tight_layout()
plt.savefig('isomap_result.png', dpi=150, bbox_inches='tight')
print("Isomap 결과 저장 완료")
```

### 3.4 국소 선형 임베딩 (LLE)

**아이디어**: 각 점을 이웃의 선형 결합으로 표현하고, 저차원에서도 동일한 가중치 유지

**단계 1**: 가중치 $W_{ij}$ 계산

$$\min_{\mathbf{W}} \sum_i \left\| \mathbf{x}_i - \sum_{j \in \mathcal{N}(i)} W_{ij} \mathbf{x}_j \right\|^2$$

제약: $\sum_j W_{ij} = 1$

**단계 2**: 저차원 임베딩 $\mathbf{y}_i$ 계산

$$\min_{\mathbf{Y}} \sum_i \left\| \mathbf{y}_i - \sum_{j \in \mathcal{N}(i)} W_{ij} \mathbf{y}_j \right\|^2$$

이는 희소 고유값 문제로 귀결됩니다: $(I - W)^T(I - W) \mathbf{y} = \lambda \mathbf{y}$

```python
from sklearn.manifold import LocallyLinearEmbedding

def apply_lle(X, n_components=2, n_neighbors=10):
    """LLE 차원 축소"""
    lle = LocallyLinearEmbedding(n_components=n_components,
                                  n_neighbors=n_neighbors)
    Y = lle.fit_transform(X)
    return Y

Y_lle = apply_lle(X_swiss, n_components=2, n_neighbors=10)

plt.figure(figsize=(6, 5))
plt.scatter(Y_lle[:, 0], Y_lle[:, 1], c=t_swiss, cmap='viridis', s=10)
plt.title('LLE embedding (2D)')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.colorbar(label='Original angle t')
plt.tight_layout()
plt.savefig('lle_result.png', dpi=150, bbox_inches='tight')
print("LLE 결과 저장 완료")
```

## 4. t-SNE의 수학

### 4.1 확률적 접근

t-SNE (t-distributed Stochastic Neighbor Embedding)는 **확률 분포의 유사성**을 보존합니다.

**고차원**: 가우시안 커널로 조건부 확률 정의

$$p_{j|i} = \frac{\exp(-\|\mathbf{x}_i - \mathbf{x}_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|\mathbf{x}_i - \mathbf{x}_k\|^2 / 2\sigma_i^2)}$$

대칭화: $p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}$

**저차원**: Student-t 분포 (자유도 1)

$$q_{ij} = \frac{(1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|\mathbf{y}_k - \mathbf{y}_l\|^2)^{-1}}$$

### 4.2 KL 발산 최소화

**목적 함수**:

$$C = \text{KL}(P \| Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

**그래디언트**:

$$\frac{\partial C}{\partial \mathbf{y}_i} = 4 \sum_j (p_{ij} - q_{ij}) (\mathbf{y}_i - \mathbf{y}_j) (1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1}$$

### 4.3 Perplexity와 $\sigma_i$ 선택

Perplexity는 "유효 이웃 수"를 나타냅니다:

$$\text{Perplexity}(P_i) = 2^{H(P_i)}$$

여기서 $H(P_i) = -\sum_j p_{j|i} \log_2 p_{j|i}$는 섀넌 엔트로피입니다.

각 $i$에 대해 이진 탐색으로 원하는 perplexity를 달성하는 $\sigma_i$를 찾습니다.

### 4.4 크라우딩 문제와 해결

**크라우딩 문제**: 고차원의 거리 분포를 저차원(특히 2D)에 맞추면 중간 거리의 점들이 "뭉치는" 현상

**해결**: Student-t 분포의 **긴 꼬리**(heavy tail)
- 가우시안보다 느리게 감소
- 저차원에서 중간 거리의 점들을 더 멀리 배치 가능

```python
from sklearn.manifold import TSNE

def apply_tsne(X, n_components=2, perplexity=30, learning_rate=200,
               n_iter=1000, random_state=42):
    """
    t-SNE 차원 축소

    Parameters:
    -----------
    X : ndarray
        고차원 데이터
    n_components : int
        목표 차원
    perplexity : float
        Perplexity 값 (유효 이웃 수)
    learning_rate : float
        학습률
    n_iter : int
        최적화 반복 횟수

    Returns:
    --------
    Y : ndarray
        저차원 임베딩
    """
    tsne = TSNE(n_components=n_components,
                perplexity=perplexity,
                learning_rate=learning_rate,
                n_iter=n_iter,
                random_state=random_state)
    Y = tsne.fit_transform(X)
    return Y

Y_tsne = apply_tsne(X_swiss, perplexity=30, n_iter=1000)

plt.figure(figsize=(6, 5))
plt.scatter(Y_tsne[:, 0], Y_tsne[:, 1], c=t_swiss, cmap='viridis', s=10)
plt.title('t-SNE embedding (2D)')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.colorbar(label='Original angle t')
plt.tight_layout()
plt.savefig('tsne_result.png', dpi=150, bbox_inches='tight')
print("t-SNE 결과 저장 완료")
```

### 4.5 t-SNE의 한계

- **계산 복잡도**: $O(n^2)$ (Barnes-Hut 근사로 $O(n \log n)$)
- **전역 구조 비보존**: 국소 구조에만 집중, 클러스터 간 거리는 의미 없음
- **비결정론적**: 초기화와 하이퍼파라미터에 민감

## 5. UMAP의 수학

### 5.1 리만 기하학적 기초

UMAP (Uniform Manifold Approximation and Projection)는 다양체가 **국소적으로 리만 다양체**라고 가정합니다.

**리만 메트릭**: 각 점에서 거리를 측정하는 방식
- 데이터 밀도가 높은 곳: 메트릭 확장 (거리가 짧아짐)
- 데이터 밀도가 낮은 곳: 메트릭 축소 (거리가 길어짐)

### 5.2 퍼지 위상 구조

**고차원**: 각 점 $x_i$를 중심으로 퍼지 집합 생성

$$\mu_i(x_j) = \exp\left( -\frac{\max(0, d(x_i, x_j) - \rho_i)}{\sigma_i} \right)$$

여기서:
- $\rho_i$: 첫 번째 최근접 이웃까지의 거리
- $\sigma_i$: perplexity와 유사한 역할 (k-최근접 이웃 기반)

**퍼지 합집합**: $\mu(x_i, x_j) = \mu_i(x_j) + \mu_j(x_i) - \mu_i(x_j) \cdot \mu_j(x_i)$

### 5.3 교차 엔트로피 최소화

저차원에서도 유사한 퍼지 집합 $\nu_{ij}$ 생성:

$$\nu_{ij} = \left(1 + a \|\mathbf{y}_i - \mathbf{y}_j\|^{2b}\right)^{-1}$$

**목적 함수**: 교차 엔트로피

$$C = \sum_{ij} \left[ \mu_{ij} \log \frac{\mu_{ij}}{\nu_{ij}} + (1 - \mu_{ij}) \log \frac{1 - \mu_{ij}}{1 - \nu_{ij}} \right]$$

이는 **attractive force**와 **repulsive force**의 균형을 맞춥니다.

```python
import umap

def apply_umap(X, n_components=2, n_neighbors=15, min_dist=0.1,
               random_state=42):
    """
    UMAP 차원 축소

    Parameters:
    -----------
    X : ndarray
        고차원 데이터
    n_components : int
        목표 차원
    n_neighbors : int
        최근접 이웃 수 (perplexity와 유사)
    min_dist : float
        저차원 공간에서 최소 거리

    Returns:
    --------
    Y : ndarray
        저차원 임베딩
    """
    reducer = umap.UMAP(n_components=n_components,
                        n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        random_state=random_state)
    Y = reducer.fit_transform(X)
    return Y

Y_umap = apply_umap(X_swiss, n_neighbors=15, min_dist=0.1)

plt.figure(figsize=(6, 5))
plt.scatter(Y_umap[:, 0], Y_umap[:, 1], c=t_swiss, cmap='viridis', s=10)
plt.title('UMAP embedding (2D)')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.colorbar(label='Original angle t')
plt.tight_layout()
plt.savefig('umap_result.png', dpi=150, bbox_inches='tight')
print("UMAP 결과 저장 완료")
```

### 5.4 UMAP vs t-SNE

| 특성 | t-SNE | UMAP |
|------|-------|------|
| **전역 구조** | 약함 | 강함 |
| **속도** | 느림 ($O(n \log n)$) | 빠름 ($O(n \log n)$) |
| **확장성** | 제한적 | 우수 (수백만 샘플) |
| **이론적 기초** | 확률적 | 위상수학적 + 리만 기하학 |
| **새 데이터 투영** | 불가 | 가능 (transform 메서드) |

## 6. ML 응용: 신경망 기반 표현 학습

### 6.1 오토인코더: 다양체 학습의 신경망 접근

**인코더**: $\mathbf{z} = f_{\text{enc}}(\mathbf{x}; \theta_{\text{enc}})$
**디코더**: $\hat{\mathbf{x}} = f_{\text{dec}}(\mathbf{z}; \theta_{\text{dec}})$

**목적**: 재구성 오차 최소화

$$\mathcal{L} = \mathbb{E}_{\mathbf{x}} \left[ \|\mathbf{x} - \hat{\mathbf{x}}\|^2 \right]$$

**해석**: 인코더는 데이터를 저차원 다양체로 매핑, 디코더는 다양체에서 주변 공간으로 복원

```python
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        # 인코더
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

        # 디코더
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

# 예제: 스위스 롤에 적용
input_dim = 3
latent_dim = 2

model = Autoencoder(input_dim, latent_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# 데이터 준비
X_torch = torch.FloatTensor(X_swiss)

# 학습
n_epochs = 500
for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()

    x_recon, z = model(X_torch)
    loss = criterion(x_recon, X_torch)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")

# 잠재 표현 추출
model.eval()
with torch.no_grad():
    _, Z_ae = model(X_torch)
    Z_ae = Z_ae.numpy()

plt.figure(figsize=(6, 5))
plt.scatter(Z_ae[:, 0], Z_ae[:, 1], c=t_swiss, cmap='viridis', s=10)
plt.title('Autoencoder latent space (2D)')
plt.xlabel('Latent dim 1')
plt.ylabel('Latent dim 2')
plt.colorbar(label='Original angle t')
plt.tight_layout()
plt.savefig('autoencoder_latent.png', dpi=150, bbox_inches='tight')
print("오토인코더 잠재 공간 시각화 저장 완료")
```

### 6.2 잠재 공간의 기하학적 특성

**연속성**: 비슷한 입력은 가까운 잠재 표현
**보간 가능성**: 잠재 공간에서 선형 보간 → 의미 있는 출력

```python
# 잠재 공간에서 보간
def interpolate_in_latent_space(model, x1, x2, n_steps=10):
    """잠재 공간에서 두 점 사이 보간"""
    model.eval()
    with torch.no_grad():
        z1 = model.encoder(torch.FloatTensor(x1).unsqueeze(0))
        z2 = model.encoder(torch.FloatTensor(x2).unsqueeze(0))

        # 선형 보간
        alphas = np.linspace(0, 1, n_steps)
        interpolations = []

        for alpha in alphas:
            z_interp = (1 - alpha) * z1 + alpha * z2
            x_interp = model.decoder(z_interp)
            interpolations.append(x_interp.squeeze().numpy())

    return np.array(interpolations)

# 두 샘플 선택
idx1, idx2 = 0, 500
x1, x2 = X_swiss[idx1], X_swiss[idx2]

interpolated = interpolate_in_latent_space(model, x1, x2, n_steps=10)

fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(111, projection='3d')

# 원본 데이터
ax.scatter(X_swiss[:, 0], X_swiss[:, 1], X_swiss[:, 2],
           c='lightgray', s=5, alpha=0.3)

# 보간 경로
ax.plot(interpolated[:, 0], interpolated[:, 1], interpolated[:, 2],
        'r-o', linewidth=2, markersize=6)
ax.scatter([x1[0], x2[0]], [x1[1], x2[1]], [x1[2], x2[2]],
           c='blue', s=100, marker='*')

ax.set_title('Latent space interpolation')
plt.tight_layout()
plt.savefig('latent_interpolation.png', dpi=150, bbox_inches='tight')
print("잠재 공간 보간 시각화 저장 완료")
```

### 6.3 쌍곡 임베딩 (Hyperbolic Embeddings)

**아이디어**: 계층 구조를 가진 데이터는 **쌍곡 공간**(hyperbolic space)에 더 잘 맞습니다.

**푸앵카레 디스크** (Poincaré disk): 2D 쌍곡 공간 모델
- 중심에서 멀어질수록 공간이 "팽창"
- 트리 구조를 기하급수적 증가 없이 임베딩 가능

**쌍곡 거리**:

$$d_{\mathcal{H}}(\mathbf{x}, \mathbf{y}) = \text{arcosh}\left(1 + 2\frac{\|\mathbf{x} - \mathbf{y}\|^2}{(1 - \|\mathbf{x}\|^2)(1 - \|\mathbf{y}\|^2)}\right)$$

응용: WordNet 계층, 소셜 네트워크, 지식 그래프

### 6.4 대조 학습 (Contrastive Learning)과 다양체

**SimCLR, MoCo** 등의 대조 학습은 암묵적으로 다양체 구조를 학습합니다.

**대조 손실** (InfoNCE):

$$\mathcal{L} = -\log \frac{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_i^+) / \tau)}{\sum_{k=1}^{K} \exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_k) / \tau)}$$

**효과**:
- 같은 데이터의 증강 버전을 잠재 공간에서 가깝게
- 다른 데이터는 멀게 → 다양체 구조 형성

## 연습 문제

### 문제 1: 측지 거리의 중요성
1. 스위스 롤 데이터에서 유클리드 거리와 측지 거리(그래프 최단 경로 근사)를 계산하시오.
2. 두 거리 행렬의 상관관계를 분석하시오.
3. k-최근접 이웃 수 $k$를 변화시키며 측지 거리 추정의 정확도가 어떻게 변하는지 실험하시오.

### 문제 2: t-SNE 하이퍼파라미터 연구
t-SNE를 스위스 롤에 적용하면서 다음을 실험하시오:
1. Perplexity를 [5, 10, 30, 50, 100]으로 변화시키며 결과 비교
2. 학습률(learning rate)의 영향 분석
3. 반복 횟수(n_iter)에 따른 수렴 과정 시각화
4. 각 설정에서의 KL 발산 값 추적

### 문제 3: UMAP과 t-SNE 비교
동일한 데이터(예: MNIST 또는 스위스 롤)에 대해:
1. UMAP과 t-SNE의 임베딩 결과 시각적 비교
2. 실행 시간 측정 (샘플 수를 증가시키며)
3. 전역 구조 보존 평가: trustworthiness와 continuity 지표 계산
4. 클러스터 간 거리의 의미 비교

### 문제 4: 오토인코더의 잠재 공간 분석
스위스 롤 데이터에 오토인코더를 학습시킨 후:
1. 잠재 차원을 [1, 2, 3, 5, 10]으로 변화시키며 재구성 오차 분석
2. 잠재 공간의 부드러움(smoothness) 측정: 인접 샘플 간 잠재 표현 차이
3. 잠재 공간에서 그리드 샘플링 후 디코더로 복원하여 다양체 구조 시각화
4. PCA로 초기화한 오토인코더와 랜덤 초기화의 수렴 속도 비교

### 문제 5: 간단한 t-SNE 구현
NumPy만 사용하여 t-SNE의 핵심 부분을 구현하시오:
1. 주어진 perplexity에 대해 각 점의 $\sigma_i$ 계산 (이진 탐색)
2. 대칭 확률 행렬 $P$ 계산
3. 저차원 확률 행렬 $Q$ 계산 (Student-t 커널)
4. KL 발산의 그래디언트 계산
5. 경사 하강법으로 100-200회 반복하여 임베딩 최적화
6. sklearn의 t-SNE 결과와 비교

## 참고 자료

### 논문
- van der Maaten, L., & Hinton, G. (2008). "Visualizing Data using t-SNE." *JMLR*, 9, 2579-2605.
- McInnes, L., Healy, J., & Melville, J. (2018). "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction." *arXiv:1802.03426*.
- Tenenbaum, J. B., et al. (2000). "A global geometric framework for nonlinear dimensionality reduction." *Science*, 290(5500), 2319-2323. [Isomap]
- Roweis, S. T., & Saul, L. K. (2000). "Nonlinear dimensionality reduction by locally linear embedding." *Science*, 290(5500), 2323-2326. [LLE]
- Bengio, Y., et al. (2013). "Representation Learning: A Review and New Perspectives." *TPAMI*, 35(8), 1798-1828.

### 온라인 자료
- [How to Use t-SNE Effectively (Distill)](https://distill.pub/2016/misread-tsne/)
- [Understanding UMAP (Pair et al.)](https://pair-code.github.io/understanding-umap/)
- [The Illustrated Word2Vec (Jay Alammar)](https://jalammar.github.io/illustrated-word2vec/)
- [Visualizing Representations (colah's blog)](https://colah.github.io/posts/2015-01-Visualizing-Representations/)

### 라이브러리
- `scikit-learn`: PCA, Isomap, LLE, t-SNE
- `umap-learn`: UMAP 구현
- `openTSNE`: 빠른 t-SNE 구현
- `matplotlib`: 시각화
