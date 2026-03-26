# 레슨 4: 벡터 노름과 내적 (Vector Norms and Inner Products)

## 학습 목표

- 벡터의 L1, L2, Lp, 무한대 노름과 행렬의 Frobenius 노름을 정의하고 계산할 수 있다
- 내적 공간의 공리를 이해하고 Cauchy-Schwarz 부등식을 검증할 수 있다
- 벡터 간의 직교성을 판별하고 두 벡터 사이의 각도를 계산할 수 있다
- 거리 계산과 정규화 등의 실용적인 작업에 노름과 내적을 적용할 수 있다
- 서로 다른 노름을 구별하고 각각이 적절한 상황을 알 수 있다

---

## 1. 벡터 노름 (Vector Norms)

### 1.1 노름이란 무엇인가?

**노름**은 각 벡터에 음이 아닌 "길이"를 부여하는 함수 $\|\cdot\| : \mathbb{R}^n \to \mathbb{R}$입니다. 유효한 노름은 세 가지 공리를 만족해야 합니다:

1. **비음수성**: $\|\mathbf{v}\| \ge 0$이며, $\mathbf{v} = \mathbf{0}$일 때만 등호가 성립
2. **동차성**: $\|\alpha \mathbf{v}\| = |\alpha| \, \|\mathbf{v}\|$
3. **삼각 부등식**: $\|\mathbf{u} + \mathbf{v}\| \le \|\mathbf{u}\| + \|\mathbf{v}\|$

### 1.2 $L^p$ 노름

일반적인 $L^p$ 노름 ($p \ge 1$)은 다음과 같이 정의됩니다:

$$\|\mathbf{v}\|_p = \left( \sum_{i=1}^n |v_i|^p \right)^{1/p}$$

```python
import numpy as np
import matplotlib.pyplot as plt

v = np.array([3, -4])

# L1 norm (Manhattan / taxicab distance)
l1 = np.linalg.norm(v, ord=1)
print(f"L1 norm: {l1}")   # |3| + |-4| = 7

# L2 norm (Euclidean distance) -- the default
l2 = np.linalg.norm(v)    # or np.linalg.norm(v, ord=2)
print(f"L2 norm: {l2}")   # sqrt(9 + 16) = 5

# L3 norm
l3 = np.linalg.norm(v, ord=3)
print(f"L3 norm: {l3:.4f}")

# L-infinity norm (max absolute value)
linf = np.linalg.norm(v, ord=np.inf)
print(f"L-inf norm: {linf}")  # max(|3|, |-4|) = 4
```

### 1.3 각 노름의 상세 설명

#### L1 노름 (맨해튼 노름)

$$\|\mathbf{v}\|_1 = \sum_{i=1}^n |v_i|$$

L1 노름은 "택시 거리"를 측정합니다 -- 축을 따라 이동한 총 거리입니다. 머신러닝에서 L1 정규화(LASSO)는 **희소성**을 촉진합니다: 일부 성분을 정확히 0으로 만듭니다.

#### L2 노름 (유클리드 노름)

$$\|\mathbf{v}\|_2 = \sqrt{\sum_{i=1}^n v_i^2}$$

L2 노름은 우리에게 익숙한 직선 거리입니다. L2 정규화(릿지 회귀)는 큰 가중치를 억제하지만 희소한 해를 생성하지는 않습니다.

#### L-무한대 노름 (최대값 노름)

$$\|\mathbf{v}\|_\infty = \max_i |v_i|$$

절댓값이 가장 큰 성분을 측정합니다. 최악의 경우 분석과 적대적 강건성에 유용합니다.

### 1.4 단위 공의 시각화

노름의 **단위 공(unit ball)**은 집합 $\{\mathbf{v} : \|\mathbf{v}\|_p \le 1\}$입니다. 그 경계는 노름이 정확히 1인 모든 벡터의 집합입니다.

```python
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

theta = np.linspace(0, 2 * np.pi, 1000)
norms = [0.5, 1, 2, np.inf]
titles = ['p = 0.5 (not a norm)', 'p = 1 (L1)', 'p = 2 (L2)', 'p = inf (L-inf)']

for ax, p, title in zip(axes, norms, titles):
    if p == np.inf:
        # Square: max(|x|, |y|) = 1
        x = np.array([-1, 1, 1, -1, -1])
        y = np.array([-1, -1, 1, 1, -1])
        ax.plot(x, y, 'b-', linewidth=2)
        ax.fill(x, y, alpha=0.2)
    else:
        # Parametric: (cos(t), sin(t)) scaled
        pts = []
        for t in theta:
            x = np.cos(t)
            y = np.sin(t)
            vec = np.array([x, y])
            norm_val = (np.abs(x)**p + np.abs(y)**p)**(1/p)
            pts.append(vec / norm_val)
        pts = np.array(pts)
        ax.plot(pts[:, 0], pts[:, 1], 'b-', linewidth=2)
        ax.fill(pts[:, 0], pts[:, 1], alpha=0.2)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)

plt.tight_layout()
plt.show()
```

---

## 2. 행렬 노름 (Matrix Norms)

### 2.1 Frobenius 노름

**Frobenius 노름**은 행렬을 긴 벡터로 취급하고 L2 노름을 계산합니다:

$$\|A\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n a_{ij}^2} = \sqrt{\mathrm{tr}(A^T A)}$$

```python
A = np.array([[1, 2],
              [3, 4]])

fro = np.linalg.norm(A, 'fro')
print(f"Frobenius norm: {fro:.4f}")
print(f"Manual: {np.sqrt(np.sum(A**2)):.4f}")
print(f"Via trace: {np.sqrt(np.trace(A.T @ A)):.4f}")
```

### 2.2 연산자 노름 (Operator Norms)

**연산자 노름** (또는 유도 노름)은 행렬이 벡터를 늘릴 수 있는 최대 배율을 측정합니다:

$$\|A\|_p = \max_{\mathbf{x} \neq \mathbf{0}} \frac{\|A\mathbf{x}\|_p}{\|\mathbf{x}\|_p} = \max_{\|\mathbf{x}\|_p = 1} \|A\mathbf{x}\|_p$$

주요 경우:
- $\|A\|_1$ = 열별 절대값 합의 최댓값
- $\|A\|_2$ = 최대 특이값 ($\sigma_{\max}$)
- $\|A\|_\infty$ = 행별 절대값 합의 최댓값

```python
A = np.array([[1, 2],
              [3, 4]])

print(f"Operator 1-norm:   {np.linalg.norm(A, 1):.4f}")
print(f"Operator 2-norm:   {np.linalg.norm(A, 2):.4f}")
print(f"Operator inf-norm: {np.linalg.norm(A, np.inf):.4f}")
print(f"Frobenius norm:    {np.linalg.norm(A, 'fro'):.4f}")

# Verify 2-norm = largest singular value
sv = np.linalg.svd(A, compute_uv=False)
print(f"Largest singular value: {sv[0]:.4f}")
```

### 2.3 노름 간의 관계

$A \in \mathbb{R}^{m \times n}$에 대해:

$$\|A\|_2 \le \|A\|_F \le \sqrt{r} \, \|A\|_2$$

여기서 $r = \mathrm{rank}(A)$입니다.

---

## 3. 내적 (Inner Products)

### 3.1 정의

실수 벡터 공간 $V$ 위의 **내적**은 다음을 만족하는 함수 $\langle \cdot, \cdot \rangle : V \times V \to \mathbb{R}$입니다:

1. **대칭성**: $\langle \mathbf{u}, \mathbf{v} \rangle = \langle \mathbf{v}, \mathbf{u} \rangle$
2. **첫 번째 인수에 대한 선형성**: $\langle \alpha\mathbf{u} + \beta\mathbf{w}, \mathbf{v} \rangle = \alpha\langle \mathbf{u}, \mathbf{v} \rangle + \beta\langle \mathbf{w}, \mathbf{v} \rangle$
3. **양의 정부호성**: $\langle \mathbf{v}, \mathbf{v} \rangle \ge 0$이며, $\mathbf{v} = \mathbf{0}$일 때만 등호 성립

$\mathbb{R}^n$의 표준 내적은 내적(dot product)입니다:

$$\langle \mathbf{u}, \mathbf{v} \rangle = \mathbf{u}^T \mathbf{v} = \sum_{i=1}^n u_i v_i$$

### 3.2 가중 내적 (Weighted Inner Product)

**가중 내적**은 대칭 양의 정부호 행렬 $W$를 사용합니다:

$$\langle \mathbf{u}, \mathbf{v} \rangle_W = \mathbf{u}^T W \mathbf{v}$$

```python
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])

# Standard inner product
standard = np.dot(u, v)
print(f"Standard: <u, v> = {standard}")  # 1*4 + 2*5 + 3*6 = 32

# Weighted inner product
W = np.array([[2, 0, 0],
              [0, 1, 0],
              [0, 0, 3]])
weighted = u @ W @ v
print(f"Weighted: <u, v>_W = {weighted}")  # 1*2*4 + 2*1*5 + 3*3*6 = 72

# Verify W is positive definite (all eigenvalues > 0)
eigvals = np.linalg.eigvals(W)
print(f"Eigenvalues of W: {eigvals} (all positive: {np.all(eigvals > 0)})")
```

### 3.3 내적이 유도하는 노름

모든 내적은 노름을 유도합니다:

$$\|\mathbf{v}\| = \sqrt{\langle \mathbf{v}, \mathbf{v} \rangle}$$

표준 내적은 L2 노름을 유도합니다.

---

## 4. Cauchy-Schwarz 부등식

### 4.1 식

임의의 내적에 대해:

$$|\langle \mathbf{u}, \mathbf{v} \rangle| \le \|\mathbf{u}\| \, \|\mathbf{v}\|$$

등호는 $\mathbf{u}$와 $\mathbf{v}$가 선형 종속일 때 (하나가 다른 하나의 스칼라 배일 때)만 성립합니다.

### 4.2 결과

1. **각도가 잘 정의됨**: $|\cos\theta| \le 1$이므로, $\cos\theta = \frac{\langle \mathbf{u}, \mathbf{v} \rangle}{\|\mathbf{u}\| \|\mathbf{v}\|}$ 공식은 항상 유효한 각도를 제공합니다.
2. **삼각 부등식**: Cauchy-Schwarz 부등식은 유도된 노름의 삼각 부등식을 증명하는 데 사용됩니다.
3. **상관 관계 한계**: 통계학에서 상관 계수 $\rho \in [-1, 1]$는 Cauchy-Schwarz의 결과입니다.

```python
# Verify Cauchy-Schwarz for random vectors
np.random.seed(42)
for _ in range(5):
    u = np.random.randn(10)
    v = np.random.randn(10)
    lhs = abs(np.dot(u, v))
    rhs = np.linalg.norm(u) * np.linalg.norm(v)
    print(f"|<u,v>| = {lhs:.4f} <= ||u||*||v|| = {rhs:.4f}: {lhs <= rhs + 1e-10}")

# Equality case: v = c * u
u = np.array([1, 2, 3])
v = 2.5 * u
lhs = abs(np.dot(u, v))
rhs = np.linalg.norm(u) * np.linalg.norm(v)
print(f"\nCollinear case: |<u,v>| = {lhs:.4f}, ||u||*||v|| = {rhs:.4f}")
print(f"Equal? {np.isclose(lhs, rhs)}")
```

---

## 5. 직교성 (Orthogonality)

### 5.1 정의

두 벡터 $\mathbf{u}$와 $\mathbf{v}$의 내적이 0이면 **직교**합니다:

$$\langle \mathbf{u}, \mathbf{v} \rangle = 0 \quad (\mathbf{u} \perp \mathbf{v})$$

벡터 집합의 모든 쌍이 직교이면 그 집합은 **직교 집합**입니다. 추가로 모든 벡터의 노름이 1이면 **정규 직교 집합**이라 합니다.

### 5.2 벡터 사이의 각도

두 비영 벡터 사이의 각도 $\theta$는 다음과 같습니다:

$$\cos\theta = \frac{\langle \mathbf{u}, \mathbf{v} \rangle}{\|\mathbf{u}\| \, \|\mathbf{v}\|}$$

```python
u = np.array([1, 0, 0])
v = np.array([0, 1, 0])
w = np.array([1, 1, 0])

def angle_between(a, b):
    """Return the angle between vectors a and b in degrees."""
    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))

print(f"Angle(u, v) = {angle_between(u, v):.1f} degrees")  # 90
print(f"Angle(u, w) = {angle_between(u, w):.1f} degrees")  # 45
print(f"Angle(v, w) = {angle_between(v, w):.1f} degrees")  # 45
```

### 5.3 정규 직교 집합 (Orthonormal Sets)

**정규 직교** 집합 $\{\mathbf{q}_1, \ldots, \mathbf{q}_k\}$은 다음을 만족합니다:

$$\langle \mathbf{q}_i, \mathbf{q}_j \rangle = \delta_{ij} = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}$$

```python
# Check if a set is orthonormal
Q = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]], dtype=float)

# Gram matrix should equal identity
gram = Q.T @ Q
print(f"Gram matrix:\n{gram}")
print(f"Orthonormal? {np.allclose(gram, np.eye(3))}")

# Create an orthonormal set from arbitrary vectors (preview of Gram-Schmidt)
v1 = np.array([1, 1, 0], dtype=float)
v2 = np.array([1, 0, 1], dtype=float)

# Normalize v1
q1 = v1 / np.linalg.norm(v1)

# Orthogonalize v2 against q1
v2_orth = v2 - np.dot(v2, q1) * q1
q2 = v2_orth / np.linalg.norm(v2_orth)

print(f"\nq1 = {q1}")
print(f"q2 = {q2}")
print(f"<q1, q2> = {np.dot(q1, q2):.10f}")  # ~0
print(f"||q1|| = {np.linalg.norm(q1):.10f}")  # 1
print(f"||q2|| = {np.linalg.norm(q2):.10f}")  # 1
```

---

## 6. 코사인 유사도 (Cosine Similarity)

**코사인 유사도**는 크기를 무시하고 두 벡터 사이 각도의 코사인을 측정합니다:

$$\mathrm{sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \, \|\mathbf{v}\|} \in [-1, 1]$$

이것은 NLP(문서 유사도), 추천 시스템, 정보 검색에서 널리 사용됩니다.

```python
# Document vectors (bag-of-words representation)
doc1 = np.array([3, 1, 0, 2, 1])  # word counts
doc2 = np.array([2, 0, 1, 3, 0])
doc3 = np.array([0, 0, 0, 0, 0])
doc4 = np.array([6, 2, 0, 4, 2])  # proportional to doc1

def cosine_similarity(u, v):
    """Compute cosine similarity between u and v."""
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0.0
    return np.dot(u, v) / (norm_u * norm_v)

print(f"sim(doc1, doc2) = {cosine_similarity(doc1, doc2):.4f}")
print(f"sim(doc1, doc4) = {cosine_similarity(doc1, doc4):.4f}")  # 1.0 (same direction)
print(f"sim(doc1, doc3) = {cosine_similarity(doc1, doc3):.4f}")  # 0.0 (zero vector)

# Pairwise similarity matrix
docs = np.array([doc1, doc2, doc4])
n = len(docs)
sim_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        sim_matrix[i, j] = cosine_similarity(docs[i], docs[j])
print(f"\nPairwise cosine similarity:\n{np.round(sim_matrix, 4)}")
```

---

## 7. 거리 메트릭 (Distance Metrics)

노름은 **거리 함수** (메트릭)를 유도합니다:

$$d_p(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|_p$$

### 7.1 주요 거리

| 거리 | 공식 | 사용 사례 |
|------|------|----------|
| 유클리드 (L2) | $\sqrt{\sum(u_i - v_i)^2}$ | 범용 거리 |
| 맨해튼 (L1) | $\sum \|u_i - v_i\|$ | 격자 기반 거리, 희소 피처 |
| 체비셰프 (L-inf) | $\max_i \|u_i - v_i\|$ | 최악의 편차 |
| 코사인 거리 | $1 - \mathrm{sim}(\mathbf{u}, \mathbf{v})$ | 텍스트/문서 유사도 |

```python
from scipy.spatial.distance import cdist

points = np.array([[0, 0], [3, 4], [1, 1], [6, 8]])

# Compute all pairwise distances
for metric in ['euclidean', 'cityblock', 'chebyshev', 'cosine']:
    D = cdist(points, points, metric=metric)
    print(f"\n{metric} distance matrix:")
    print(np.round(D, 4))
```

---

## 8. 머신러닝에서의 응용

### 8.1 L1과 L2 정규화

머신러닝에서 노름은 모델 파라미터를 **정규화**하는 데 사용됩니다:

- **L1 (LASSO)**: $\mathcal{L} + \lambda \|\mathbf{w}\|_1$ -- 희소성 촉진
- **L2 (Ridge)**: $\mathcal{L} + \lambda \|\mathbf{w}\|_2^2$ -- 큰 가중치 억제

```python
# Demonstrate sparsity effect of L1 vs L2
np.random.seed(42)
n_features = 20
w_true = np.zeros(n_features)
w_true[:5] = np.array([3, -2, 1.5, 0.5, -1])  # only 5 non-zero

# Simulate noisy weights
w_noisy = w_true + np.random.randn(n_features) * 0.3

# L1 proximal operator (soft thresholding)
def soft_threshold(w, lam):
    return np.sign(w) * np.maximum(np.abs(w) - lam, 0)

# L2 shrinkage
def l2_shrink(w, lam):
    return w / (1 + lam)

lam = 0.5
w_l1 = soft_threshold(w_noisy, lam)
w_l2 = l2_shrink(w_noisy, lam)

print(f"True sparsity (zeros):  {np.sum(np.abs(w_true) < 1e-10)}")
print(f"L1 sparsity (zeros):    {np.sum(np.abs(w_l1) < 1e-10)}")
print(f"L2 sparsity (zeros):    {np.sum(np.abs(w_l2) < 1e-10)}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].bar(range(n_features), w_true); axes[0].set_title('True weights')
axes[1].bar(range(n_features), w_l1); axes[1].set_title('After L1 (sparse)')
axes[2].bar(range(n_features), w_l2); axes[2].set_title('After L2 (shrunk)')
for ax in axes:
    ax.set_xlabel('Feature index'); ax.set_ylabel('Weight')
plt.tight_layout()
plt.show()
```

### 8.2 최근접 이웃 (Nearest Neighbors)

$k$-최근접 이웃 알고리즘은 거리 계산에 의존합니다:

```python
from collections import Counter

# Simple KNN implementation
def knn_predict(X_train, y_train, x_query, k=3, metric='euclidean'):
    """Predict the label of x_query using k-nearest neighbors."""
    if metric == 'euclidean':
        distances = np.linalg.norm(X_train - x_query, axis=1)
    elif metric == 'manhattan':
        distances = np.sum(np.abs(X_train - x_query), axis=1)

    nearest_idx = np.argsort(distances)[:k]
    nearest_labels = y_train[nearest_idx]
    return Counter(nearest_labels).most_common(1)[0][0]

# Example
X_train = np.array([[1, 2], [2, 3], [3, 1], [6, 5], [7, 7], [8, 6]])
y_train = np.array([0, 0, 0, 1, 1, 1])
x_query = np.array([5, 5])

label = knn_predict(X_train, y_train, x_query, k=3)
print(f"Predicted label for {x_query}: {label}")
```

---

## 9. 요약

| 개념 | 공식 | NumPy |
|------|------|-------|
| L1 노름 | $\sum \|v_i\|$ | `np.linalg.norm(v, 1)` |
| L2 노름 | $\sqrt{\sum v_i^2}$ | `np.linalg.norm(v)` |
| Lp 노름 | $(\sum \|v_i\|^p)^{1/p}$ | `np.linalg.norm(v, p)` |
| L-inf 노름 | $\max \|v_i\|$ | `np.linalg.norm(v, np.inf)` |
| Frobenius 노름 | $\sqrt{\sum a_{ij}^2}$ | `np.linalg.norm(A, 'fro')` |
| 내적 | $\mathbf{u}^T \mathbf{v}$ | `np.dot(u, v)` |
| 코사인 유사도 | $\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}$ | 수동 계산 |
| 직교성 | $\mathbf{u} \cdot \mathbf{v} = 0$ | `np.dot(u, v) == 0` |

---

## 연습 문제

### 연습 문제 1: 노름 계산

$\mathbf{v} = [1, -2, 3, -4, 5]^T$에 대해 $\|\mathbf{v}\|_1$, $\|\mathbf{v}\|_2$, $\|\mathbf{v}\|_3$, 그리고 $\|\mathbf{v}\|_\infty$를 계산하세요. $\|\mathbf{v}\|_\infty \le \|\mathbf{v}\|_2 \le \|\mathbf{v}\|_1$이 성립함을 확인하세요.

### 연습 문제 2: Cauchy-Schwarz

$\mathbf{u} = [1, 2, 3]^T$와 $\mathbf{v} = [4, 5, 6]^T$에 대해:
1. Cauchy-Schwarz 부등식을 수치적으로 검증하세요.
2. $\mathbf{u}$와 $c\mathbf{u}$에 대해 등호가 성립하는 스칼라 $c$를 찾으세요.

### 연습 문제 3: 직교 분해

$\mathbf{v} = [3, 4]^T$와 $\mathbf{u} = [1, 0]^T$가 주어졌을 때, $\mathbf{v}$를 $\mathbf{u}$에 평행한 성분과 $\mathbf{u}$에 직교하는 성분으로 분해하세요. 두 성분이 직교이고 합이 $\mathbf{v}$임을 검증하세요.

### 연습 문제 4: 거리 비교

점 $A = (1, 2)$, $B = (4, 6)$, $C = (7, 2)$를 고려하세요. 모든 쌍 간의 L1, L2, L-무한대 거리를 계산하세요. 어떤 메트릭에서 $A$가 $B$에 가장 가깝습니까? 어떤 메트릭에서 $A$가 $C$에 가장 가깝습니까?

### 연습 문제 5: 단위 공 경계

$p = 0.5, 1, 2, 4, \infty$에 대해 $\mathbb{R}^2$에서 단위 공의 경계를 생성하고 플롯하는 Python 프로그램을 작성하세요. $p$가 증가함에 따라 모양이 어떻게 변하는지 설명하세요.

---

[<< 이전: 레슨 3 - 연립일차방정식](03_Systems_of_Linear_Equations.md) | [개요](00_Overview.md) | [다음: 레슨 5 - 선형 변환 >>](05_Linear_Transformations.md)

**License**: CC BY-NC 4.0
