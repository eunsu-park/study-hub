# 레슨 12: 희소 행렬

[이전: 레슨 11](./11_Positive_Definite_Matrices.md) | [개요](./00_Overview.md) | [다음: 레슨 13](./13_Numerical_Linear_Algebra.md)

---

## 학습 목표

- 희소 행렬이 무엇이고 왜 대규모 계산에 중요한지 이해할 수 있습니다
- 주요 희소 저장 형식을 학습합니다: COO, CSR, CSC, BSR, DOK
- `scipy.sparse`를 사용하여 희소 형식을 구성, 조작, 변환할 수 있습니다
- 희소 선형대수 연산(행렬-벡터 곱, 시스템 풀기)을 수행할 수 있습니다
- 희소 행렬을 사용하여 그래프와 인접 구조를 표현할 수 있습니다
- 채움-감소 순서(fill-reducing orderings)와 희소 직접 솔버에서의 역할을 이해할 수 있습니다

---

## 1. 희소 행렬이란?

행렬의 대부분의 원소가 0일 때 그 행렬은 **희소**(sparse)합니다. 많은 과학 및 공학 응용에서 수백만 행과 열을 가진 행렬이 행당 소수의 비영 원소만 가질 수 있습니다.

**희소도**(sparsity)는 영 원소의 비율로 정의됩니다:

$$\text{sparsity}(A) = 1 - \frac{\text{nnz}(A)}{m \times n}$$

여기서 $\text{nnz}(A)$는 $m \times n$ 행렬 $A$의 비영 원소 수입니다.

### 1.1 희소 행렬이 중요한 이유

| 측면 | 밀집 ($n \times n$) | 희소 ($n \times n$, $\text{nnz} = O(n)$) |
|------|---------------------|------------------------------------------|
| 저장 | $O(n^2)$ | $O(n)$ |
| 행렬-벡터 곱 | $O(n^2)$ | $O(n)$ |
| 직접 풀기 (일반) | $O(n^3)$ | $O(n)$ ~ $O(n^{3/2})$ (2D) |
| 고유값 (소수) | $O(n^3)$ | $O(n \cdot k)$ (반복법) |

$n = 10^6$ 행을 가진 행렬의 경우, 밀집 저장은 8 TB의 메모리(float64)가 필요하지만, 행당 10개의 비영값을 가진 희소 행렬은 80 MB만 필요합니다.

```python
import numpy as np
from scipy import sparse
import time

# Compare memory: dense vs sparse
n = 10000
density = 0.001  # 0.1% nonzero

# Dense matrix
A_dense = np.random.rand(n, n) * (np.random.rand(n, n) < density)
dense_memory = A_dense.nbytes / 1e6  # MB

# Sparse matrix (CSR)
A_sparse = sparse.random(n, n, density=density, format='csr')
sparse_memory = (A_sparse.data.nbytes + A_sparse.indices.nbytes +
                 A_sparse.indptr.nbytes) / 1e6

print(f"Matrix size: {n} x {n}")
print(f"Density: {density:.1%}")
print(f"Nonzeros: {A_sparse.nnz}")
print(f"Dense memory:  {dense_memory:.1f} MB")
print(f"Sparse memory: {sparse_memory:.3f} MB")
print(f"Ratio: {dense_memory / sparse_memory:.0f}x")
```

---

## 2. 희소 저장 형식

### 2.1 COO (Coordinate) 형식

가장 단순한 형식: 동일 길이의 세 배열 -- 행 인덱스, 열 인덱스, 값 -- 을 저장합니다.

**적합한 용도**: 희소 행렬의 점진적 구축, 형식 간 변환.

```python
# COO format
row = np.array([0, 0, 1, 2, 2, 3])
col = np.array([0, 2, 1, 0, 2, 3])
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

A_coo = sparse.coo_matrix((data, (row, col)), shape=(4, 4))
print("COO matrix:")
print(A_coo)
print(f"\nDense form:\n{A_coo.toarray()}")
print(f"row: {A_coo.row}")
print(f"col: {A_coo.col}")
print(f"data: {A_coo.data}")
```

### 2.2 CSR (Compressed Sparse Row) 형식

희소 계산에서 가장 널리 사용되는 형식입니다. 명시적 행 인덱스를 저장하는 대신, CSR은 **indptr** 배열을 사용합니다: `indptr[i]`에서 `indptr[i+1]`까지가 행 `i`의 열과 값을 인덱싱합니다.

**적합한 용도**: 빠른 행 슬라이싱, 행렬-벡터 곱, 희소 솔버.

```python
# CSR format
A_csr = A_coo.tocsr()
print("CSR representation:")
print(f"data:    {A_csr.data}")
print(f"indices: {A_csr.indices}")     # Column indices
print(f"indptr:  {A_csr.indptr}")      # Row pointer

# How to read: row i has columns indices[indptr[i]:indptr[i+1]]
for i in range(A_csr.shape[0]):
    start, end = A_csr.indptr[i], A_csr.indptr[i + 1]
    cols = A_csr.indices[start:end]
    vals = A_csr.data[start:end]
    print(f"Row {i}: columns={cols}, values={vals}")
```

### 2.3 CSC (Compressed Sparse Column) 형식

CSR의 열 방향 대응물입니다. `indptr` 배열이 행 대신 열을 인덱싱합니다.

**적합한 용도**: 빠른 열 슬라이싱, 전치 연산, 직접 솔버 (예: UMFPACK).

```python
# CSC format
A_csc = A_coo.tocsc()
print("CSC representation:")
print(f"data:    {A_csc.data}")
print(f"indices: {A_csc.indices}")     # Row indices
print(f"indptr:  {A_csc.indptr}")      # Column pointer

for j in range(A_csc.shape[1]):
    start, end = A_csc.indptr[j], A_csc.indptr[j + 1]
    rows = A_csc.indices[start:end]
    vals = A_csc.data[start:end]
    print(f"Col {j}: rows={rows}, values={vals}")
```

### 2.4 BSR (Block Sparse Row) 형식

자연스러운 블록 구조를 가진 행렬(예: 유한요소 문제)에 대해, BSR은 개별 원소 대신 밀집 부분블록을 저장합니다.

```python
# BSR format: 2x2 blocks
indptr = np.array([0, 2, 3])
indices = np.array([0, 2, 1])
data = np.array([
    [[1, 2], [3, 4]],    # Block (0,0)
    [[5, 6], [7, 8]],    # Block (0,2)
    [[9, 10], [11, 12]]  # Block (1,1)
])

A_bsr = sparse.bsr_matrix((data, indices, indptr), shape=(4, 6))
print("BSR matrix (dense view):")
print(A_bsr.toarray())
print(f"Block size: {A_bsr.blocksize}")
print(f"Number of stored blocks: {len(A_bsr.data)}")
```

### 2.5 DOK (Dictionary of Keys) 형식

`(행, 열)` 튜플을 키로 사용하는 딕셔너리를 사용합니다. 점진적 구축에 효율적입니다.

```python
# DOK format: good for building sparse matrices element by element
A_dok = sparse.dok_matrix((4, 4))
A_dok[0, 0] = 1.0
A_dok[0, 2] = 2.0
A_dok[1, 1] = 3.0
A_dok[2, 0] = 4.0

print("DOK matrix:")
print(A_dok.toarray())
print(f"Internal dict: {dict(A_dok)}")

# Convert to CSR for computation
A_csr = A_dok.tocsr()
```

### 2.6 LIL (List of Lists) 형식

행당 열 인덱스 리스트와 값 리스트를 하나씩 저장합니다. 점진적 행 기반 구축에 좋지만 산술 연산은 느립니다.

```python
# LIL format
A_lil = sparse.lil_matrix((4, 4))
A_lil[0, 0] = 1.0
A_lil[0, 2] = 2.0
A_lil[1, 1] = 3.0
A_lil[3, 3] = 6.0

print("LIL matrix:")
print(A_lil.toarray())
print(f"Rows: {A_lil.rows}")
print(f"Data: {A_lil.data}")
```

### 2.7 형식 비교

| 형식 | 구축 | 산술 | 행 슬라이스 | 열 슬라이스 | 메모리 |
|------|------|------|-----------|-----------|--------|
| COO | 빠름 | 느림 | 느림 | 느림 | 3 * nnz |
| CSR | 느림 | 빠름 | 빠름 | 느림 | 2 * nnz + n + 1 |
| CSC | 느림 | 빠름 | 느림 | 빠름 | 2 * nnz + n + 1 |
| BSR | 보통 | 빠름 (블록) | 빠름 | 느림 | 블록 오버헤드 |
| DOK | 빠름 (단일) | 느림 | 느림 | 느림 | dict 오버헤드 |
| LIL | 빠름 (행) | 느림 | 보통 | 느림 | list 오버헤드 |

**권장 워크플로**: COO, DOK 또는 LIL로 구축한 후 계산을 위해 CSR이나 CSC로 변환합니다.

---

## 3. 희소 행렬 구성

### 3.1 일반적인 구성 패턴

```python
# 1. Diagonal matrices
n = 5
D = sparse.diags([1, 2, 3, 4, 5], format='csr')
print("Diagonal matrix:")
print(D.toarray())

# 2. Banded matrices (e.g., tridiagonal)
diags = [np.ones(n-1), -2*np.ones(n), np.ones(n-1)]
offsets = [-1, 0, 1]
T = sparse.diags(diags, offsets, shape=(n, n), format='csr')
print("\nTridiagonal matrix:")
print(T.toarray())

# 3. Identity matrix
I = sparse.eye(5, format='csr')
print("\nSparse identity:")
print(I.toarray())

# 4. Block diagonal
blocks = [np.array([[1, 2], [3, 4]]),
          np.array([[5, 6], [7, 8]]),
          np.array([[9]])]
B = sparse.block_diag(blocks, format='csr')
print("\nBlock diagonal:")
print(B.toarray())
```

### 3.2 2D 라플라시안 구성

2D 격자의 이산 라플라시안은 편미분방정식에서 나타나는 전형적인 희소 행렬입니다. $n \times n$ 격자에서 행당 5개의 비영값만 가진 $n^2 \times n^2$ 행렬을 생성합니다.

```python
def laplacian_2d(n):
    """Build the 2D discrete Laplacian on an n x n grid."""
    # 1D Laplacian
    T = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(n, n), format='csr')

    # 2D Laplacian via Kronecker product
    I = sparse.eye(n, format='csr')
    L = sparse.kron(I, T) + sparse.kron(T, I)
    return L

n = 5
L = laplacian_2d(n)
print(f"2D Laplacian shape: {L.shape}")
print(f"Nonzeros: {L.nnz}")
print(f"Density: {L.nnz / (n**2 * n**2):.4f}")
print(f"\nMatrix (dense):\n{L.toarray()}")

# Visualize sparsity pattern
plt.figure(figsize=(6, 6))
plt.spy(L, markersize=3)
plt.title(f'2D Laplacian sparsity pattern ({n}x{n} grid)')
plt.tight_layout()
plt.show()
```

### 3.3 무작위 희소 행렬

```python
# Generate random sparse matrix
A = sparse.random(1000, 1000, density=0.01, format='csr', random_state=42)
print(f"Shape: {A.shape}")
print(f"Nonzeros: {A.nnz}")
print(f"Density: {A.nnz / (1000 * 1000):.4f}")

# Visualize sparsity pattern
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].spy(A, markersize=0.1)
axes[0].set_title('Random sparse matrix')

# 2D Laplacian for comparison
L = laplacian_2d(30)
axes[1].spy(L, markersize=0.1)
axes[1].set_title('2D Laplacian (structured)')

plt.tight_layout()
plt.show()
```

---

## 4. 희소 행렬 연산

### 4.1 산술 연산

```python
n = 1000
A = sparse.random(n, n, density=0.01, format='csr', random_state=42)
B = sparse.random(n, n, density=0.01, format='csr', random_state=43)

# Addition
C = A + B
print(f"A nnz: {A.nnz}, B nnz: {B.nnz}, A+B nnz: {C.nnz}")

# Scalar multiplication
D = 3.0 * A
print(f"3A nnz: {D.nnz}")

# Matrix-matrix multiplication (sparse @ sparse)
E = A @ B
print(f"A*B nnz: {E.nnz}")
print(f"A*B density: {E.nnz / (n * n):.6f}")

# Element-wise multiplication
F = A.multiply(B)
print(f"A.*B nnz: {F.nnz}")  # Usually much fewer nonzeros
```

### 4.2 행렬-벡터 곱 (SpMV)

희소 행렬-벡터 곱(SpMV)은 가장 중요한 단일 희소 연산입니다. 반복 솔버, 고유값 알고리즘, PageRank의 핵심입니다.

```python
n = 100000
A = sparse.random(n, n, density=0.0001, format='csr', random_state=42)
x = np.random.randn(n)

# Sparse matrix-vector product
start = time.time()
y_sparse = A @ x
t_sparse = time.time() - start

# Compare with dense (smaller matrix for feasibility)
n_small = 5000
A_small_sparse = sparse.random(n_small, n_small, density=0.001, format='csr')
A_small_dense = A_small_sparse.toarray()
x_small = np.random.randn(n_small)

start = time.time()
y_dense = A_small_dense @ x_small
t_dense = time.time() - start

start = time.time()
y_sp = A_small_sparse @ x_small
t_sp = time.time() - start

print(f"Dense SpMV ({n_small}x{n_small}): {t_dense*1000:.2f} ms")
print(f"Sparse SpMV ({n_small}x{n_small}): {t_sp*1000:.2f} ms")
print(f"Speedup: {t_dense/t_sp:.1f}x")
print(f"Results match: {np.allclose(y_dense, y_sp)}")
```

### 4.3 희소 선형 시스템 풀기

```python
from scipy.sparse.linalg import spsolve, cg

# Build a sparse SPD system (2D Laplacian with Dirichlet BC)
n = 50
L = laplacian_2d(n)
# Make it positive definite by negating (standard Laplacian has negative eigenvalues)
A = -L + sparse.eye(n**2) * 0.01  # Shift to ensure PD
b = np.random.randn(n**2)

# Direct solve (uses UMFPACK or SuperLU)
start = time.time()
x_direct = spsolve(A.tocsc(), b)
t_direct = time.time() - start
print(f"Direct solve time: {t_direct*1000:.1f} ms")
print(f"Residual norm: {np.linalg.norm(A @ x_direct - b):.2e}")

# Iterative solve (Conjugate Gradient for SPD matrices)
start = time.time()
x_cg, info = cg(A, b, tol=1e-10)
t_cg = time.time() - start
print(f"\nCG solve time: {t_cg*1000:.1f} ms")
print(f"CG converged: {info == 0}")
print(f"Residual norm: {np.linalg.norm(A @ x_cg - b):.2e}")
```

---

## 5. 그래프 인접 행렬

### 5.1 그래프의 희소 행렬 표현

그래프는 자연스럽게 희소 행렬로 표현됩니다. $n$개의 노드와 $m$개의 간선을 가진 그래프에서 인접 행렬은 $2m$개의 비영값(무방향) 또는 $m$개의 비영값(방향)을 가진 $n \times n$ 행렬입니다.

```python
# Build an adjacency matrix for a small graph
edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4), (3, 5), (4, 5)]
n_nodes = 6
row = [e[0] for e in edges] + [e[1] for e in edges]  # Undirected
col = [e[1] for e in edges] + [e[0] for e in edges]
data = np.ones(len(row))

adj = sparse.csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes))
print("Adjacency matrix:")
print(adj.toarray())

# Degree matrix
degrees = np.array(adj.sum(axis=1)).flatten()
D = sparse.diags(degrees, format='csr')
print(f"\nDegrees: {degrees}")

# Laplacian: L = D - A
L = D - adj
print(f"\nGraph Laplacian:\n{L.toarray()}")
```

### 5.2 희소 행렬을 사용한 PageRank

```python
from scipy.sparse.linalg import eigs

def pagerank(adj, alpha=0.85, tol=1e-8, max_iter=100):
    """Compute PageRank using the power method on a sparse adjacency matrix."""
    n = adj.shape[0]

    # Build transition matrix: M[j,i] = 1/out_degree(i) if edge i->j
    out_degree = np.array(adj.sum(axis=1)).flatten()
    out_degree[out_degree == 0] = 1  # Handle dangling nodes

    # Column-normalize: each column sums to 1
    D_inv = sparse.diags(1.0 / out_degree, format='csr')
    M = (D_inv @ adj).T  # Transition matrix (column stochastic)

    # Power iteration: x = alpha * M @ x + (1 - alpha) / n
    x = np.ones(n) / n
    for i in range(max_iter):
        x_new = alpha * M @ x + (1 - alpha) / n
        if np.linalg.norm(x_new - x, 1) < tol:
            print(f"Converged in {i+1} iterations")
            break
        x = x_new

    return x / x.sum()

# Example: directed graph
edges_directed = [(0, 1), (0, 2), (1, 2), (2, 0), (3, 2), (3, 0)]
n_nodes = 4
row = [e[1] for e in edges_directed]  # Target
col = [e[0] for e in edges_directed]  # Source
data = np.ones(len(edges_directed))
adj_directed = sparse.csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes))

ranks = pagerank(adj_directed)
for i, r in enumerate(ranks):
    print(f"Node {i}: PageRank = {r:.4f}")
```

### 5.3 행렬 거듭제곱을 통한 너비 우선 탐색

인접 행렬에 벡터를 곱하면 그래프에서 한 홉 정보를 전파합니다. $k$번째 거듭제곱 $A^k$는 길이 $k$인 경로의 수를 셉니다.

```python
# BFS via sparse matrix-vector product
adj_bool = adj.astype(bool).astype(float)

# Start from node 0
start = np.zeros(n_nodes)
start[0] = 1

print("BFS traversal via matrix powers:")
reached = start.copy()
frontier = start.copy()
for step in range(1, n_nodes):
    frontier = adj_bool @ frontier
    frontier[reached > 0] = 0  # Remove already-visited
    if frontier.sum() == 0:
        break
    reached += frontier
    newly_reached = np.where(frontier > 0)[0]
    print(f"Step {step}: reached nodes {newly_reached}")

print(f"All reached nodes: {np.where(reached > 0)[0]}")
```

---

## 6. 채움-감소 순서

### 6.1 채움 문제 (Fill-In Problem)

직접 방법(LU 또는 Cholesky)으로 $Ax = b$를 풀 때, **채움**(fill-in)이라 불리는 새로운 비영 원소가 인자에 나타납니다. 채움은 메모리와 계산 시간을 극적으로 증가시킬 수 있습니다.

**채움-감소 순서**는 $A$의 행과 열을 재배열하여 인수분해에서의 채움을 최소화합니다.

```python
from scipy.sparse.linalg import splu
from scipy.sparse import csc_matrix

# Build a sparse matrix with potential for fill
n = 20
L = laplacian_2d(n)
A = -L + 5 * sparse.eye(n**2)
A_csc = A.tocsc()

# Factor without reordering (natural order)
# scipy.sparse.linalg.splu uses SuperLU with built-in reordering
lu = splu(A_csc)
fill_ratio = lu.nnz / A.nnz
print(f"Original nnz: {A.nnz}")
print(f"LU factors nnz: {lu.nnz}")
print(f"Fill ratio: {fill_ratio:.2f}x")
```

### 6.2 일반적인 순서

가장 일반적인 채움-감소 순서는 다음과 같습니다:

1. **역 Cuthill-McKee (RCM)**: BFS와 유사한 순회로 대역폭을 줄입니다. 단순하고 효과적입니다.
2. **근사 최소 차수 (AMD)**: 가장 적은 이웃을 가진 노드를 탐욕적으로 제거합니다. 흔히 최선의 범용 선택입니다.
3. **중첩 분할 (Nested Dissection)**: 그래프를 재귀적으로 이등분합니다. 2D 및 3D 메시에 최적입니다.

```python
from scipy.sparse.csgraph import reverse_cuthill_mckee

# Reverse Cuthill-McKee ordering
n = 15
L = laplacian_2d(n)
A = (-L + 5 * sparse.eye(n**2)).tocsr()

# Compute RCM permutation
perm = reverse_cuthill_mckee(A)
A_rcm = A[perm][:, perm]

# Visualize sparsity patterns
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].spy(A, markersize=0.5)
axes[0].set_title(f'Natural order (bandwidth ~{n**2})')

axes[1].spy(A_rcm, markersize=0.5)
# Estimate bandwidth
bandwidth = max(abs(A_rcm.nonzero()[0] - A_rcm.nonzero()[1]))
axes[1].set_title(f'RCM order (bandwidth ~{bandwidth})')

plt.tight_layout()
plt.show()
```

### 6.3 순서가 Cholesky 채움에 미치는 영향

```python
from sksparse.cholmod import cholesky as cholmod_cholesky  # Optional
from scipy.sparse.linalg import splu

# Compare factorization with different orderings
n = 30
L = laplacian_2d(n)
A = (-L + 5 * sparse.eye(n**2)).tocsc()

# Natural ordering
lu_natural = splu(A, permc_spec='NATURAL')
print(f"Natural ordering: LU nnz = {lu_natural.nnz}")

# Minimum degree (column AMD)
lu_colamd = splu(A, permc_spec='COLAMD')
print(f"COLAMD ordering: LU nnz = {lu_colamd.nnz}")

# MMD on A^T A
lu_mmd = splu(A, permc_spec='MMD_AT_PLUS_A')
print(f"MMD ordering:    LU nnz = {lu_mmd.nnz}")

print(f"\nOriginal matrix nnz: {A.nnz}")
print(f"Fill reduction from natural to COLAMD: "
      f"{lu_natural.nnz / lu_colamd.nnz:.2f}x")
```

---

## 7. 희소 고유값 문제

큰 희소 행렬에서 모든 고유값을 계산하는 것은 불가능합니다. 대신 **ARPACK**(묵시적 재시작 아르놀디)과 같은 반복법을 사용하여 소수의 고유값을 계산합니다.

```python
from scipy.sparse.linalg import eigsh, eigs

# Large sparse symmetric matrix
n = 100
L = laplacian_2d(n)
A = -L  # Negative Laplacian (PSD)

print(f"Matrix size: {A.shape}")
print(f"Nonzeros: {A.nnz}")

# Find the 6 smallest eigenvalues
eigenvalues_small, eigenvectors_small = eigsh(A, k=6, which='SM')
print(f"\n6 smallest eigenvalues: {eigenvalues_small}")

# Find the 6 largest eigenvalues
eigenvalues_large, eigenvectors_large = eigsh(A, k=6, which='LM')
print(f"6 largest eigenvalues: {eigenvalues_large}")

# Visualize eigenvectors (modes of the Laplacian)
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for i, ax in enumerate(axes.flat):
    mode = eigenvectors_small[:, i].reshape(n, n)
    ax.imshow(mode, cmap='RdBu_r')
    ax.set_title(f'Mode {i+1} (lambda={eigenvalues_small[i]:.4f})')
    ax.axis('off')

plt.suptitle('Eigenvectors of the 2D Laplacian')
plt.tight_layout()
plt.show()
```

---

## 8. 희소 행렬 실전 팁

### 8.1 형식 선택 가이드

```python
# Demonstration: format affects performance

n = 5000
density = 0.001
A_coo = sparse.random(n, n, density=density, format='coo')
A_csr = A_coo.tocsr()
A_csc = A_coo.tocsc()
x = np.random.randn(n)

# Matrix-vector product: CSR wins
start = time.time()
for _ in range(100):
    _ = A_csr @ x
t_csr = time.time() - start

start = time.time()
for _ in range(100):
    _ = A_csc @ x
t_csc = time.time() - start

print(f"SpMV (100 repeats):")
print(f"  CSR: {t_csr*1000:.1f} ms")
print(f"  CSC: {t_csc*1000:.1f} ms")

# Row slicing: CSR wins
start = time.time()
for i in range(0, n, 10):
    _ = A_csr[i]
t_csr_row = time.time() - start

start = time.time()
for i in range(0, n, 10):
    _ = A_csc[i]
t_csc_row = time.time() - start

print(f"\nRow slicing:")
print(f"  CSR: {t_csr_row*1000:.1f} ms")
print(f"  CSC: {t_csc_row*1000:.1f} ms")
```

### 8.2 일반적인 함정

```python
# Pitfall 1: Converting sparse to dense
n = 1000
A = sparse.random(n, n, density=0.01, format='csr')
# WRONG: A.toarray() creates a huge dense matrix
# RIGHT: Use sparse operations directly

# Pitfall 2: Inefficient construction
# WRONG: Modifying CSR element by element
A_csr = sparse.csr_matrix((n, n))
# A_csr[0, 0] = 1.0  # SparseEfficiencyWarning

# RIGHT: Build in COO or LIL, then convert
A_lil = sparse.lil_matrix((n, n))
A_lil[0, 0] = 1.0
A_lil[1, 1] = 2.0
A_csr = A_lil.tocsr()

# Pitfall 3: Not checking for zeros
# COO format can have explicit zeros
row = [0, 0, 1]
col = [0, 1, 1]
data = [1.0, 0.0, 2.0]  # Explicit zero!
A = sparse.coo_matrix((data, (row, col)), shape=(2, 2))
print(f"Before eliminate_zeros: nnz = {A.nnz}")
A.eliminate_zeros()
print(f"After eliminate_zeros: nnz = {A.nnz}")
```

---

## 연습 문제

### 연습 문제 1: 희소 행렬 구성

`scipy.sparse`를 사용하여 다음 희소 행렬을 효율적으로 구성하세요:

1. 오프셋 $[-2, -1, 0, 1, 2]$에 값 $[1, -4, 6, -4, 1]$을 가진 $100 \times 100$ 오대각 행렬
2. 행당 정확히 5개의 비영값을 가진 $1000 \times 1000$ 무작위 희소 행렬
3. $10 \times 10$ 격자 그래프의 인접 행렬

각각에 대해 희소도, 바이트 단위 저장량을 보고하고 동등한 밀집 행렬과 비교하세요.

### 연습 문제 2: PageRank 구현

다음 방향 웹 그래프에 대해 PageRank를 구현하고 상위 3개 순위 페이지를 찾으세요:

- 페이지: A, B, C, D, E
- 링크: A->B, A->C, B->C, C->A, D->C, E->A, E->B, E->D

감쇠 인자 $\alpha = 0.85$를 사용하고 수렴을 검증하세요.

### 연습 문제 3: 희소 솔버 벤치마크

격자 크기 $n = 10, 20, 50, 100, 200$에 대해 2D 포아송 시스템(라플라시안)을 생성하세요. 각 크기에 대해:

1. `spsolve` (직접)로 풀기
2. `cg` (반복)로 풀기
3. 벽시계 시간과 잔차 노름 비교
4. 스케일링 거동 (시간 vs. $n^2$) 그래프

### 연습 문제 4: 채움 분석

$20 \times 20$ 격자 라플라시안에 대해, 자연 순서와 RCM 순서를 사용한 Cholesky/LU 인자의 비영 원소 수를 비교하세요. 인자의 희소 패턴을 시각화하세요.

### 연습 문제 5: 희소 고유값

$50 \times 50$ 격자 그래프의 그래프 라플라시안에서 10개의 가장 작은 고유값을 계산하세요. 두 번째로 작은 고유값(Fiedler 값)은 그래프 연결도를 측정합니다. 대응하는 고유벡터(Fiedler 벡터)를 사용하여 그래프를 두 구성 요소로 분할하세요.

---

[이전: 레슨 11](./11_Positive_Definite_Matrices.md) | [개요](./00_Overview.md) | [다음: 레슨 13](./13_Numerical_Linear_Algebra.md)

**License**: CC BY-NC 4.0
