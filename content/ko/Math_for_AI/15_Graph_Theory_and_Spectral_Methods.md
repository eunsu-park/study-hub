# 15. 그래프 이론과 스펙트럼 방법

## 학습 목표

- 그래프의 수학적 표현(인접 행렬, 차수 행렬, 라플라시안)을 이해하고 구현할 수 있다
- 그래프 라플라시안의 고유값 분해와 스펙트럼 특성을 설명할 수 있다
- 스펙트럼 군집화 알고리즘의 수학적 원리를 이해하고 구현할 수 있다
- 랜덤 워크와 PageRank 알고리즘의 수학적 기초를 이해할 수 있다
- 그래프 신호 처리와 그래프 푸리에 변환의 개념을 이해할 수 있다
- GNN(Graph Neural Networks)의 수학적 기초와 메시지 패싱 메커니즘을 이해할 수 있다

---

## 1. 그래프의 수학적 표현

### 1.1 그래프의 기초

그래프 $G = (V, E)$는 정점(vertex) 집합 $V$와 간선(edge) 집합 $E \subseteq V \times V$로 구성됩니다.

**그래프의 유형:**
- **무방향 그래프**: $(i,j) \in E \Rightarrow (j,i) \in E$
- **방향 그래프**: 간선에 방향이 있음
- **가중 그래프**: 각 간선에 가중치 $w_{ij}$가 할당됨

### 1.2 인접 행렬 (Adjacency Matrix)

$n$개의 정점을 가진 그래프의 인접 행렬 $A \in \mathbb{R}^{n \times n}$:

$$A_{ij} = \begin{cases}
w_{ij} & \text{if } (i,j) \in E \\
0 & \text{otherwise}
\end{cases}$$

**성질:**
- 무방향 그래프: $A = A^T$ (대칭)
- 이진 그래프: $A_{ij} \in \{0, 1\}$
- $A^k$의 $(i,j)$ 원소: $i$에서 $j$로 가는 길이 $k$인 경로의 수

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# 간단한 그래프 생성
def create_sample_graph():
    """
    5개 정점으로 구성된 무방향 그래프
    """
    n = 5
    A = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0],
        [1, 1, 0, 1, 1],
        [0, 1, 1, 0, 1],
        [0, 0, 1, 1, 0]
    ])
    return A

A = create_sample_graph()
print("인접 행렬 A:")
print(A)
print(f"\n대칭성 확인: {np.allclose(A, A.T)}")

# 경로 수 계산
A2 = np.linalg.matrix_power(A, 2)
print(f"\n정점 0에서 정점 4로 가는 길이 2인 경로의 수: {A2[0, 4]}")
```

### 1.3 차수 행렬 (Degree Matrix)

정점 $i$의 차수 $d_i = \sum_{j} A_{ij}$는 연결된 간선의 수입니다.

차수 행렬 $D \in \mathbb{R}^{n \times n}$은 대각 행렬:

$$D = \text{diag}(d_1, d_2, \ldots, d_n)$$

```python
def compute_degree_matrix(A):
    """차수 행렬 계산"""
    degrees = np.sum(A, axis=1)
    D = np.diag(degrees)
    return D, degrees

D, degrees = compute_degree_matrix(A)
print("차수 벡터:", degrees)
print("\n차수 행렬 D:")
print(D)
```

## 2. 그래프 라플라시안 (Graph Laplacian)

### 2.1 라플라시안의 정의

**비정규화 라플라시안**:

$$L = D - A$$

**성질:**
- 대칭: $L = L^T$
- 양반정치(positive semidefinite): $\mathbf{x}^T L \mathbf{x} \geq 0$
- $L \mathbf{1} = \mathbf{0}$ (모든 원소가 1인 벡터는 고유값 0의 고유벡터)

### 2.2 라플라시안의 이차 형식

$$\mathbf{x}^T L \mathbf{x} = \mathbf{x}^T(D - A)\mathbf{x} = \sum_{i} d_i x_i^2 - \sum_{i,j} A_{ij} x_i x_j$$

무방향 그래프의 경우:

$$\mathbf{x}^T L \mathbf{x} = \frac{1}{2} \sum_{i,j} A_{ij}(x_i - x_j)^2$$

이는 **인접한 정점 간의 차이를 측정**하는 smoothness 척도입니다.

```python
def compute_laplacian(A):
    """그래프 라플라시안 계산"""
    D, _ = compute_degree_matrix(A)
    L = D - A
    return L

L = compute_laplacian(A)
print("라플라시안 행렬 L:")
print(L)

# 양반정치 확인 (모든 고유값 >= 0)
eigenvalues = np.linalg.eigvalsh(L)
print(f"\n라플라시안 고유값: {eigenvalues}")
print(f"최소 고유값: {eigenvalues[0]:.10f}")
```

### 2.3 정규화 라플라시안

**대칭 정규화 라플라시안**:

$$L_{\text{sym}} = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}$$

**랜덤 워크 정규화 라플라시안**:

$$L_{\text{rw}} = D^{-1} L = I - D^{-1} A$$

정규화 라플라시안의 고유값은 $[0, 2]$ 범위에 있습니다.

```python
def compute_normalized_laplacian(A):
    """정규화 라플라시안 계산"""
    D, degrees = compute_degree_matrix(A)

    # D^{-1/2} 계산
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))

    # L_sym = D^{-1/2} L D^{-1/2}
    L = compute_laplacian(A)
    L_sym = D_inv_sqrt @ L @ D_inv_sqrt

    # L_rw = D^{-1} L
    D_inv = np.diag(1.0 / degrees)
    L_rw = D_inv @ L

    return L_sym, L_rw

L_sym, L_rw = compute_normalized_laplacian(A)
print("정규화 라플라시안 L_sym:")
print(L_sym)

eig_sym = np.linalg.eigvalsh(L_sym)
print(f"\nL_sym 고유값: {eig_sym}")
```

### 2.4 연결 성분과 고유값

**정리**: 그래프가 $k$개의 연결 성분을 가지면, 라플라시안의 고유값 0의 중복도는 $k$입니다.

```python
def create_disconnected_graph():
    """두 개의 연결 성분을 가진 그래프"""
    # 성분 1: 정점 0, 1, 2
    # 성분 2: 정점 3, 4
    A = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 1, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0]
    ])
    return A

A_disconnected = create_disconnected_graph()
L_disconnected = compute_laplacian(A_disconnected)
eigenvalues_disc = np.linalg.eigvalsh(L_disconnected)

print("비연결 그래프의 라플라시안 고유값:")
print(eigenvalues_disc)
print(f"고유값 0의 개수 (연결 성분 수): {np.sum(np.abs(eigenvalues_disc) < 1e-10)}")
```

## 3. 스펙트럼 군집화 (Spectral Clustering)

### 3.1 그래프 절단 문제

그래프를 두 부분 $S$와 $\bar{S}$로 분할할 때, **절단 비용**:

$$\text{cut}(S, \bar{S}) = \sum_{i \in S, j \in \bar{S}} A_{ij}$$

**정규화 절단 (Normalized Cut)**:

$$\text{Ncut}(S, \bar{S}) = \frac{\text{cut}(S, \bar{S})}{\text{vol}(S)} + \frac{\text{cut}(S, \bar{S})}{\text{vol}(\bar{S})}$$

여기서 $\text{vol}(S) = \sum_{i \in S} d_i$는 부분집합 $S$의 볼륨입니다.

### 3.2 Fiedler 벡터와 스펙트럼 방법

Ncut 문제는 NP-hard이지만, **라플라시안의 두 번째 최소 고유벡터**(Fiedler 벡터)를 이용한 완화로 근사할 수 있습니다.

**Rayleigh 몫**:

$$\min_{\mathbf{y}} \frac{\mathbf{y}^T L \mathbf{y}}{\mathbf{y}^T D \mathbf{y}} \quad \text{s.t. } \mathbf{y}^T D \mathbf{1} = 0$$

해는 일반화 고유값 문제 $L \mathbf{y} = \lambda D \mathbf{y}$의 두 번째 최소 고유벡터입니다.

```python
def spectral_clustering(A, n_clusters=2):
    """
    스펙트럼 군집화 알고리즘

    Parameters:
    -----------
    A : ndarray
        인접 행렬
    n_clusters : int
        군집 수

    Returns:
    --------
    labels : ndarray
        각 정점의 군집 레이블
    """
    # 정규화 라플라시안 계산
    L_sym, _ = compute_normalized_laplacian(A)

    # 고유값 분해 (최소 n_clusters개의 고유벡터)
    eigenvalues, eigenvectors = eigh(L_sym)

    # 최소 n_clusters개의 고유벡터 선택
    U = eigenvectors[:, :n_clusters]

    # 행 정규화
    U_normalized = U / np.linalg.norm(U, axis=1, keepdims=True)

    # k-means 군집화
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(U_normalized)

    return labels, U

# 스펙트럼 군집화 적용
labels, U = spectral_clustering(A, n_clusters=2)
print("군집 레이블:", labels)
print("\n첫 2개 고유벡터:")
print(U)
```

### 3.3 스펙트럼 군집화의 직관

- **Fiedler 벡터의 부호**: 그래프를 두 부분으로 분할하는 좋은 지표
- **고유값의 크기**: 군집 간 분리도를 나타냄
- **eigengap**: $\lambda_k$와 $\lambda_{k+1}$ 간의 차이가 크면 $k$개 군집이 적절함

```python
def visualize_spectral_clustering():
    """스펙트럼 군집화 시각화"""
    # 더 큰 그래프 생성 (두 개의 밀집된 커뮤니티)
    np.random.seed(42)
    n1, n2 = 15, 15
    n = n1 + n2

    # 블록 행렬 구조
    A_block = np.zeros((n, n))

    # 커뮤니티 1 내부 연결 (밀집)
    for i in range(n1):
        for j in range(i+1, n1):
            if np.random.rand() < 0.6:
                A_block[i, j] = A_block[j, i] = 1

    # 커뮤니티 2 내부 연결 (밀집)
    for i in range(n1, n):
        for j in range(i+1, n):
            if np.random.rand() < 0.6:
                A_block[i, j] = A_block[j, i] = 1

    # 커뮤니티 간 연결 (희소)
    for i in range(n1):
        for j in range(n1, n):
            if np.random.rand() < 0.05:
                A_block[i, j] = A_block[j, i] = 1

    # 스펙트럼 군집화
    labels, U = spectral_clustering(A_block, n_clusters=2)

    # 라플라시안 고유값
    L_sym, _ = compute_normalized_laplacian(A_block)
    eigenvalues = np.linalg.eigvalsh(L_sym)

    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 인접 행렬
    axes[0].imshow(A_block, cmap='binary')
    axes[0].set_title('Adjacency Matrix')
    axes[0].set_xlabel('Node')
    axes[0].set_ylabel('Node')

    # 고유값 스펙트럼
    axes[1].plot(eigenvalues, 'o-')
    axes[1].axvline(x=2, color='r', linestyle='--', label='Eigengap')
    axes[1].set_xlabel('Index')
    axes[1].set_ylabel('Eigenvalue')
    axes[1].set_title('Laplacian Spectrum')
    axes[1].legend()
    axes[1].grid(True)

    # Fiedler 벡터
    axes[2].scatter(range(n), U[:, 1], c=labels, cmap='viridis', s=50)
    axes[2].axhline(y=0, color='r', linestyle='--')
    axes[2].set_xlabel('Node')
    axes[2].set_ylabel('Fiedler Vector Value')
    axes[2].set_title('Fiedler Vector (2nd eigenvector)')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig('spectral_clustering.png', dpi=150, bbox_inches='tight')
    print("스펙트럼 군집화 시각화 저장 완료")

visualize_spectral_clustering()
```

## 4. 랜덤 워크 (Random Walk on Graphs)

### 4.1 전이 확률 행렬

랜덤 워크는 현재 정점에서 인접한 정점으로 균등하게 이동하는 확률 과정입니다.

**전이 확률 행렬**:

$$P = D^{-1} A$$

$P_{ij}$는 정점 $i$에서 정점 $j$로 이동할 확률입니다.

```python
def compute_transition_matrix(A):
    """전이 확률 행렬 계산"""
    D, degrees = compute_degree_matrix(A)
    D_inv = np.diag(1.0 / degrees)
    P = D_inv @ A
    return P

P = compute_transition_matrix(A)
print("전이 확률 행렬 P:")
print(P)
print(f"\n각 행의 합 (확률의 합): {np.sum(P, axis=1)}")
```

### 4.2 정상 분포 (Stationary Distribution)

정상 분포 $\pi$는 다음을 만족합니다:

$$\pi^T P = \pi^T$$

연결된 비방향 그래프의 경우, 정상 분포는:

$$\pi_i = \frac{d_i}{\sum_j d_j}$$

```python
def compute_stationary_distribution(A):
    """정상 분포 계산"""
    _, degrees = compute_degree_matrix(A)
    pi = degrees / np.sum(degrees)
    return pi

pi = compute_stationary_distribution(A)
print("정상 분포 π:")
print(pi)

# 검증: π^T P = π^T
P = compute_transition_matrix(A)
pi_next = pi @ P
print(f"\n정상성 확인: {np.allclose(pi, pi_next)}")
```

### 4.3 PageRank 알고리즘

PageRank는 랜덤 워크에 텔레포트(teleportation)를 추가한 것입니다:

$$\mathbf{r} = (1 - d) \mathbf{e} + d P^T \mathbf{r}$$

여기서 $d \in [0, 1]$은 damping factor (보통 0.85), $\mathbf{e}$는 균등 분포입니다.

```python
def pagerank(A, d=0.85, max_iter=100, tol=1e-6):
    """
    PageRank 알고리즘

    Parameters:
    -----------
    A : ndarray
        인접 행렬
    d : float
        Damping factor
    max_iter : int
        최대 반복 횟수
    tol : float
        수렴 임계값

    Returns:
    --------
    r : ndarray
        PageRank 점수
    """
    n = A.shape[0]
    P = compute_transition_matrix(A)

    # 초기화: 균등 분포
    r = np.ones(n) / n

    for iteration in range(max_iter):
        r_new = (1 - d) / n + d * (P.T @ r)

        # 수렴 확인
        if np.linalg.norm(r_new - r, 1) < tol:
            print(f"수렴 완료: {iteration + 1}번 반복")
            break

        r = r_new

    return r

# 방향 그래프 생성 (웹페이지 링크 구조)
A_directed = np.array([
    [0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0],
    [1, 0, 0, 1, 1],
    [0, 0, 0, 0, 1],
    [0, 0, 1, 0, 0]
])

pagerank_scores = pagerank(A_directed)
print("\nPageRank 점수:")
for i, score in enumerate(pagerank_scores):
    print(f"페이지 {i}: {score:.4f}")
```

## 5. 그래프 신호 처리 (Graph Signal Processing)

### 5.1 그래프 신호

그래프 신호 $\mathbf{f} \in \mathbb{R}^n$는 각 정점에 할당된 값입니다.

예: 소셜 네트워크에서 각 사용자의 활동도, 센서 네트워크에서 각 센서의 측정값

### 5.2 그래프 푸리에 변환 (GFT)

라플라시안의 고유벡터 $\mathbf{u}_\ell$를 그래프의 **주파수 기저**로 사용합니다.

$$L \mathbf{u}_\ell = \lambda_\ell \mathbf{u}_\ell$$

**그래프 푸리에 변환**:

$$\hat{f}(\ell) = \langle \mathbf{f}, \mathbf{u}_\ell \rangle = \mathbf{u}_\ell^T \mathbf{f}$$

**역변환**:

$$\mathbf{f} = \sum_{\ell=0}^{n-1} \hat{f}(\ell) \mathbf{u}_\ell$$

```python
def graph_fourier_transform(A, signal):
    """
    그래프 푸리에 변환

    Parameters:
    -----------
    A : ndarray
        인접 행렬
    signal : ndarray
        그래프 신호

    Returns:
    --------
    f_hat : ndarray
        주파수 영역 신호
    eigenvalues : ndarray
        라플라시안 고유값
    eigenvectors : ndarray
        라플라시안 고유벡터
    """
    L_sym, _ = compute_normalized_laplacian(A)
    eigenvalues, eigenvectors = eigh(L_sym)

    # 그래프 푸리에 변환
    f_hat = eigenvectors.T @ signal

    return f_hat, eigenvalues, eigenvectors

# 예제: 저주파 신호 생성
n = A.shape[0]
signal_smooth = np.array([1.0, 1.1, 0.9, 0.8, 1.0])

f_hat, eigenvalues, eigenvectors = graph_fourier_transform(A, signal_smooth)

print("원 신호:", signal_smooth)
print("주파수 영역 신호:", f_hat)
print("\n라플라시안 고유값 (주파수):", eigenvalues)
```

### 5.3 그래프 필터링

주파수 영역에서 신호를 필터링:

$$\mathbf{f}_{\text{filtered}} = \sum_{\ell=0}^{n-1} h(\lambda_\ell) \hat{f}(\ell) \mathbf{u}_\ell$$

여기서 $h(\lambda)$는 필터 함수입니다.

**저역 통과 필터** (smoothing): 작은 $\lambda$ 성분만 유지
**고역 통과 필터** (edge detection): 큰 $\lambda$ 성분만 유지

```python
def graph_filter(A, signal, filter_func):
    """그래프 필터링"""
    f_hat, eigenvalues, eigenvectors = graph_fourier_transform(A, signal)

    # 주파수 영역에서 필터 적용
    f_hat_filtered = f_hat * filter_func(eigenvalues)

    # 역변환
    signal_filtered = eigenvectors @ f_hat_filtered

    return signal_filtered

# 저역 통과 필터
def lowpass_filter(eigenvalues, cutoff=0.5):
    return (eigenvalues < cutoff).astype(float)

# 고역 통과 필터
def highpass_filter(eigenvalues, cutoff=0.5):
    return (eigenvalues >= cutoff).astype(float)

# 노이즈가 있는 신호 생성
np.random.seed(42)
signal_noisy = signal_smooth + 0.3 * np.random.randn(n)

signal_lowpass = graph_filter(A, signal_noisy, lambda eig: lowpass_filter(eig, cutoff=1.0))
signal_highpass = graph_filter(A, signal_noisy, lambda eig: highpass_filter(eig, cutoff=1.0))

print("원 신호:", signal_smooth)
print("노이즈 신호:", signal_noisy)
print("저역 통과 필터 결과:", signal_lowpass)
print("고역 통과 필터 결과:", signal_highpass)
```

## 6. GNN의 수학적 기초

앞 절에서 전개한 스펙트럼 그래프 이론은 그래프 신경망(Graph Neural Networks, GNN)의 이론적 토대를 제공합니다. 그래프 위의 스펙트럼 합성곱은 라플라시안 고유공간에서의 원소별 곱셈으로 정의됩니다: $g_\theta \star \mathbf{x} = U g_\theta(\Lambda) U^T \mathbf{x}$. 여기서 $U$는 고유벡터 행렬이고 $\Lambda$는 $L$의 고유값으로 이루어진 대각 행렬입니다. GCN(Kipf & Welling, 2017)은 스펙트럼 필터를 1차 체비쇼프 다항식으로 근사하여, 우리에게 친숙한 이웃 집계 규칙을 도출합니다. 따라서 모든 GNN 레이어는 그래프의 스펙트럼 영역에서 학습된 국소 필터로 이해할 수 있습니다.

### 6.1 메시지 패싱 프레임워크

GNN의 핵심은 **메시지 패싱**입니다:

$$\mathbf{h}_v^{(\ell+1)} = \sigma\left( \mathbf{W}^{(\ell)} \sum_{u \in \mathcal{N}(v)} \frac{\mathbf{h}_u^{(\ell)}}{c_{vu}} \right)$$

여기서:
- $\mathbf{h}_v^{(\ell)}$: 레이어 $\ell$에서 정점 $v$의 특징
- $\mathcal{N}(v)$: 정점 $v$의 이웃
- $c_{vu}$: 정규화 상수
- $\sigma$: 활성화 함수

### 6.2 스펙트럼 관점: 그래프 합성곱

**스펙트럼 그래프 합성곱**:

$$\mathbf{g}_\theta \star \mathbf{f} = U \left( \text{diag}(\theta) U^T \mathbf{f} \right)$$

여기서 $U$는 라플라시안의 고유벡터 행렬, $\theta$는 학습 가능한 필터 파라미터입니다.

**문제점**: $O(n^2)$ 계산 복잡도, 고유값 분해 필요

### 6.3 ChebNet: 체비셰프 다항식 근사

체비셰프 다항식을 사용한 근사:

$$\mathbf{g}_\theta \star \mathbf{f} \approx \sum_{k=0}^{K-1} \theta_k T_k(\tilde{L}) \mathbf{f}$$

여기서:
- $\tilde{L} = \frac{2}{\lambda_{\max}} L - I$는 재스케일된 라플라시안
- $T_k$는 $k$차 체비셰프 다항식: $T_0(x) = 1, T_1(x) = x, T_{k}(x) = 2xT_{k-1}(x) - T_{k-2}(x)$

### 6.4 GCN: 1차 근사

Graph Convolutional Network (Kipf & Welling, 2017)는 $K=1$인 경우의 단순화:

$$\mathbf{H}^{(\ell+1)} = \sigma\left( \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} \mathbf{H}^{(\ell)} \mathbf{W}^{(\ell)} \right)$$

여기서 $\tilde{A} = A + I$ (자기 루프 추가), $\tilde{D}$는 $\tilde{A}$의 차수 행렬입니다.

```python
def gcn_layer(A, H, W, activation=lambda x: np.maximum(0, x)):
    """
    GCN 레이어 구현

    Parameters:
    -----------
    A : ndarray, shape (n, n)
        인접 행렬
    H : ndarray, shape (n, d_in)
        입력 특징 행렬
    W : ndarray, shape (d_in, d_out)
        가중치 행렬
    activation : function
        활성화 함수 (기본: ReLU)

    Returns:
    --------
    H_out : ndarray, shape (n, d_out)
        출력 특징 행렬
    """
    n = A.shape[0]

    # 자기 루프 추가
    A_tilde = A + np.eye(n)

    # 차수 행렬
    D_tilde = np.diag(np.sum(A_tilde, axis=1))

    # 정규화: D^{-1/2} A D^{-1/2}
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D_tilde)))
    A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt

    # 메시지 패싱 + 선형 변환 + 활성화
    H_out = activation(A_hat @ H @ W)

    return H_out

# 예제: 간단한 2층 GCN
np.random.seed(42)
n = 5
d_in = 3
d_hidden = 4
d_out = 2

# 초기 특징 행렬
X = np.random.randn(n, d_in)

# 가중치 행렬
W1 = np.random.randn(d_in, d_hidden) * 0.1
W2 = np.random.randn(d_hidden, d_out) * 0.1

# 순전파
H1 = gcn_layer(A, X, W1)
print("레이어 1 출력 shape:", H1.shape)

H2 = gcn_layer(A, H1, W2)
print("레이어 2 출력 shape:", H2.shape)
print("\n최종 임베딩:")
print(H2)
```

### 6.5 그래프 어텐션 네트워크 (GAT)

GAT는 이웃 정점에 **어텐션 가중치**를 학습합니다:

$$\alpha_{vu} = \frac{\exp(\text{LeakyReLU}(\mathbf{a}^T [\mathbf{W}\mathbf{h}_v \| \mathbf{W}\mathbf{h}_u]))}{\sum_{u' \in \mathcal{N}(v)} \exp(\text{LeakyReLU}(\mathbf{a}^T [\mathbf{W}\mathbf{h}_v \| \mathbf{W}\mathbf{h}_{u'}]))}$$

$$\mathbf{h}_v^{(\ell+1)} = \sigma\left( \sum_{u \in \mathcal{N}(v)} \alpha_{vu} \mathbf{W}^{(\ell)} \mathbf{h}_u^{(\ell)} \right)$$

```python
def graph_attention_layer(A, H, W, a, alpha=0.2):
    """
    간단한 그래프 어텐션 레이어

    Parameters:
    -----------
    A : ndarray, shape (n, n)
        인접 행렬
    H : ndarray, shape (n, d_in)
        입력 특징
    W : ndarray, shape (d_in, d_out)
        특징 변환 가중치
    a : ndarray, shape (2 * d_out,)
        어텐션 파라미터
    alpha : float
        LeakyReLU 기울기

    Returns:
    --------
    H_out : ndarray, shape (n, d_out)
        출력 특징
    """
    n = A.shape[0]

    # 특징 변환
    H_transformed = H @ W
    d_out = H_transformed.shape[1]

    # 어텐션 계산
    attention_scores = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if A[i, j] > 0 or i == j:  # 연결되어 있거나 자기 자신
                # 연결된 특징 [h_i || h_j]
                concat = np.concatenate([H_transformed[i], H_transformed[j]])

                # 어텐션 점수
                score = a @ concat
                attention_scores[i, j] = np.maximum(alpha * score, score)  # LeakyReLU
            else:
                attention_scores[i, j] = -np.inf

    # 소프트맥스
    attention_weights = np.exp(attention_scores - np.max(attention_scores, axis=1, keepdims=True))
    attention_weights = attention_weights / np.sum(attention_weights, axis=1, keepdims=True)

    # 어텐션 집계
    H_out = attention_weights @ H_transformed

    return H_out, attention_weights

# 예제
W_att = np.random.randn(d_in, d_hidden) * 0.1
a_att = np.random.randn(2 * d_hidden) * 0.1

H_gat, att_weights = graph_attention_layer(A, X, W_att, a_att)
print("GAT 출력 shape:", H_gat.shape)
print("\n어텐션 가중치:")
print(att_weights)
```

## 연습 문제

### 문제 1: 라플라시안의 이차 형식 증명
무방향 그래프의 라플라시안 $L = D - A$에 대해, 다음을 증명하시오:

$$\mathbf{x}^T L \mathbf{x} = \frac{1}{2} \sum_{i,j} A_{ij}(x_i - x_j)^2$$

이 결과를 이용하여 라플라시안이 양반정치임을 보이시오.

### 문제 2: 스펙트럼 군집화 구현
`sklearn`을 사용하지 않고, NumPy만으로 완전한 스펙트럼 군집화 알고리즘을 구현하시오. k-means 단계도 직접 구현해야 합니다. 다음을 포함하시오:

1. 정규화 라플라시안 계산
2. 고유값 분해
3. k-means 군집화 (Lloyd 알고리즘)
4. 실루엣 점수를 이용한 군집 품질 평가

### 문제 3: PageRank의 고유값 해석
PageRank 방정식 $\mathbf{r} = (1 - d) \mathbf{e} + d P^T \mathbf{r}$를 고유벡터 문제로 변환하시오. 행렬 $M = (1-d)\mathbf{e}\mathbf{1}^T + dP^T$의 지배적 고유벡터(dominant eigenvector)가 PageRank 벡터임을 설명하시오. Power iteration 방법으로 이를 계산하는 코드를 작성하시오.

### 문제 4: 그래프 푸리에 변환 응용
20개 정점으로 구성된 링 그래프(ring graph)를 생성하고, 다음을 수행하시오:

1. 라플라시안의 고유값과 고유벡터를 계산하고 시각화
2. 저주파 신호(예: $f_i = \cos(2\pi i / 20)$)와 고주파 신호(예: $f_i = (-1)^i$)의 그래프 푸리에 변환을 계산
3. 가우시안 저역 통과 필터 $h(\lambda) = \exp(-\lambda^2 / (2\sigma^2))$를 적용하고 결과를 시각화

### 문제 5: GCN vs GAT 비교
작은 그래프(10-20개 정점)에서 GCN 레이어와 GAT 레이어의 동작을 비교하는 실험을 설계하시오:

1. 두 방법의 출력 임베딩을 계산
2. 어텐션 가중치를 시각화하여 GAT가 학습한 이웃 중요도 분석
3. 계산 복잡도를 이론적으로 분석하고 실제 실행 시간 측정

## 참고 자료

### 논문
- Chung, F. R. K. (1997). *Spectral Graph Theory*. American Mathematical Society.
- Von Luxburg, U. (2007). "A tutorial on spectral clustering." *Statistics and Computing*, 17(4), 395-416.
- Kipf, T. N., & Welling, M. (2017). "Semi-supervised classification with graph convolutional networks." *ICLR*.
- Veličković, P., et al. (2018). "Graph Attention Networks." *ICLR*.
- Bronstein, M. M., et al. (2021). "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges." *arXiv:2104.13478*.

### 온라인 자료
- [Spectral Graph Theory (Spielman, Yale)](http://www.cs.yale.edu/homes/spielman/561/)
- [Graph Representation Learning Book (Hamilton)](https://www.cs.mcgill.ca/~wlh/grl_book/)
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [NetworkX Tutorial](https://networkx.org/documentation/stable/tutorial.html)

### 라이브러리
- `networkx`: 그래프 생성 및 분석
- `scipy.sparse`: 희소 행렬 연산
- `torch_geometric`: GNN 구현
- `spektral`: Keras 기반 GNN
