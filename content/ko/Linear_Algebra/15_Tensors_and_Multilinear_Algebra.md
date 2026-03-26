# 15강: 텐서와 다중선형대수 (Tensors and Multilinear Algebra)

[이전: 14강](./14_Iterative_Methods.md) | [개요](./00_Overview.md) | [다음: 16강](./16_Linear_Algebra_in_Machine_Learning.md)

---

## 학습 목표

- 텐서를 다중선형 사상으로 이해하고 다차원 배열과의 관계를 파악할 수 있다
- 텐서의 차수(인덱스 수)와 텐서 랭크(분해 복잡도)를 구분할 수 있다
- Einstein 합산 표기법과 `np.einsum`을 사용하여 텐서 연산을 표현할 수 있다
- 텐서곱, 축약, 외적을 계산할 수 있다
- 텐서 분해 방법인 CP 분해와 Tucker 분해를 적용할 수 있다
- 텐서와 딥러닝 연산(배치 연산, attention)의 관계를 이해할 수 있다

---

## 1. 텐서란 무엇인가?

### 1.1 다차원 배열로서의 텐서

계산적 관점에서 **텐서**는 숫자들의 다차원 배열입니다. 텐서의 **차수**(또는 **모드**)는 인덱스의 수를 의미합니다:

| 차수 | 이름 | 예시 | 형상 |
|---|---|---|---|
| 0 | 스칼라 | 온도 | () |
| 1 | 벡터 | 힘 | (n,) |
| 2 | 행렬 | 회전 | (m, n) |
| 3 | 3차 텐서 | 컬러 이미지 | (H, W, C) |
| 4 | 4차 텐서 | 비디오 / 이미지 배치 | (T, H, W, C) |

```python
import numpy as np

# Scalars, vectors, matrices, and higher-order tensors
scalar = np.array(42.0)       # Order 0
vector = np.array([1, 2, 3])  # Order 1
matrix = np.array([[1, 2],
                   [3, 4]])   # Order 2
tensor3 = np.random.randn(3, 4, 5)   # Order 3
tensor4 = np.random.randn(2, 3, 4, 5)  # Order 4

for name, t in [("Scalar", scalar), ("Vector", vector),
                ("Matrix", matrix), ("3-tensor", tensor3),
                ("4-tensor", tensor4)]:
    print(f"{name:10s}: ndim={t.ndim}, shape={t.shape}")
```

### 1.2 다중선형 사상으로서의 텐서 (수학적 관점)

수학에서 텐서는 **다중선형 사상**입니다. $(p, q)$-형 텐서 $T$는 $p$개의 쌍대벡터와 $q$개의 벡터를 스칼라로 대응시킵니다:

$$T: \underbrace{V^* \times \cdots \times V^*}_{p} \times \underbrace{V \times \cdots \times V}_{q} \to \mathbb{R}$$

예를 들면:
- 벡터는 $(1, 0)$-텐서입니다
- 쌍대벡터(선형 범함수)는 $(0, 1)$-텐서입니다
- 행렬(선형 사상)은 $(1, 1)$-텐서입니다
- 물리학의 계량 텐서는 $(0, 2)$-텐서입니다

핵심 성질은 **다중선형성**입니다: 사상이 각 인수에 대해 개별적으로 선형입니다.

```python
# A matrix as a bilinear map: (u, v) -> u^T A v
A = np.array([[2, 1],
              [1, 3]])

u = np.array([1, 2])
v = np.array([3, 1])

# Bilinear: linear in u and in v
bilinear = u @ A @ v
print(f"u^T A v = {bilinear}")

# Linearity in u: T(alpha*u1 + u2, v) = alpha*T(u1, v) + T(u2, v)
u1, u2 = np.array([1, 0]), np.array([0, 1])
alpha = 2.5
lhs = (alpha * u1 + u2) @ A @ v
rhs = alpha * (u1 @ A @ v) + (u2 @ A @ v)
print(f"Linearity in u: {np.isclose(lhs, rhs)}")
```

---

## 2. 텐서 차수와 텐서 랭크

### 2.1 정의

- **차수** (또는 **모드**): 원소를 지정하기 위해 필요한 인덱스의 수. 행렬은 차수 2입니다.
- **랭크**: 분해에서 랭크-1 항의 최소 개수. 행렬의 경우 일반적인 행렬 랭크와 같습니다. 고차 텐서의 경우 랭크 계산은 일반적으로 NP-hard입니다.

**랭크-1 텐서**는 벡터들의 외적입니다:

$$\mathcal{T} = \mathbf{a} \otimes \mathbf{b} \otimes \mathbf{c} \quad \Leftrightarrow \quad T_{ijk} = a_i b_j c_k$$

```python
# Rank-1 tensor: outer product of three vectors
a = np.array([1, 2, 3])
b = np.array([4, 5])
c = np.array([6, 7, 8, 9])

# Outer product
T_rank1 = np.einsum('i,j,k->ijk', a, b, c)
print(f"Rank-1 tensor shape: {T_rank1.shape}")
print(f"T[0,0,:] = {T_rank1[0, 0, :]}")  # a[0] * b[0] * c
print(f"Expected: {a[0] * b[0] * c}")

# Verify rank-1: every matrix slice has rank 1
for i in range(len(a)):
    slice_mat = T_rank1[i, :, :]
    rank = np.linalg.matrix_rank(slice_mat)
    print(f"T[{i},:,:] rank = {rank}")
```

### 2.2 파이버와 슬라이스

텐서는 **파이버**(1차원 단면)와 **슬라이스**(2차원 단면)를 통해 살펴볼 수 있습니다:

```python
# 3rd-order tensor
T = np.random.randn(3, 4, 5)

# Fibers: fix all indices except one
fiber_mode0 = T[:, 1, 2]    # Column fiber (mode-0)
fiber_mode1 = T[0, :, 2]    # Row fiber (mode-1)
fiber_mode2 = T[0, 1, :]    # Tube fiber (mode-2)

print(f"Mode-0 fiber shape: {fiber_mode0.shape}")  # (3,)
print(f"Mode-1 fiber shape: {fiber_mode1.shape}")  # (4,)
print(f"Mode-2 fiber shape: {fiber_mode2.shape}")  # (5,)

# Slices: fix one index
slice_frontal = T[:, :, 0]  # Frontal slice
slice_lateral = T[:, 0, :]  # Lateral slice
slice_horizontal = T[0, :, :]  # Horizontal slice

print(f"\nFrontal slice shape: {slice_frontal.shape}")    # (3, 4)
print(f"Lateral slice shape: {slice_lateral.shape}")      # (3, 5)
print(f"Horizontal slice shape: {slice_horizontal.shape}")  # (4, 5)
```

### 2.3 행렬화 (Unfolding)

**행렬화**는 텐서를 하나의 모드를 행으로, 나머지 모드를 열로 매핑하여 행렬로 펼치는 것입니다. 이는 텐서 분해에 필수적입니다.

```python
def unfold(T, mode):
    """Unfold tensor T along the given mode."""
    return np.reshape(np.moveaxis(T, mode, 0), (T.shape[mode], -1))

T = np.arange(24).reshape(2, 3, 4)
print(f"Original tensor shape: {T.shape}")

for mode in range(3):
    U = unfold(T, mode)
    print(f"Mode-{mode} unfolding shape: {U.shape}")
    print(f"  {U}")
    print()
```

---

## 3. Einstein 합산과 `np.einsum`

### 3.1 Einstein 합산 규약

Einstein 합산 규약: **반복되는 인덱스에 대해 묵시적으로 합산합니다**. 이를 통해 다중선형 연산을 간결하게 표기할 수 있습니다.

| 연산 | Einstein 표기법 | `np.einsum` |
|---|---|---|
| 벡터 내적 | $c = a_i b_i$ | `'i,i->'` |
| 행렬-벡터 곱 | $c_i = A_{ij} b_j$ | `'ij,j->i'` |
| 행렬 곱셈 | $C_{ij} = A_{ik} B_{kj}$ | `'ik,kj->ij'` |
| 대각합(trace) | $t = A_{ii}$ | `'ii->'` |
| 외적 | $C_{ij} = a_i b_j$ | `'i,j->ij'` |
| Hadamard 곱 | $C_{ij} = A_{ij} B_{ij}$ | `'ij,ij->ij'` |
| 배치 행렬 곱 | $C_{bij} = A_{bik} B_{bkj}$ | `'bik,bkj->bij'` |

```python
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)
a = np.random.randn(4)
b = np.random.randn(4)

# 1. Dot product
dot_manual = np.sum(a * b)
dot_einsum = np.einsum('i,i->', a, b)
print(f"Dot product match: {np.isclose(dot_manual, dot_einsum)}")

# 2. Matrix-vector product
mv_manual = A @ a
mv_einsum = np.einsum('ij,j->i', A, a)
print(f"Matrix-vector match: {np.allclose(mv_manual, mv_einsum)}")

# 3. Matrix multiplication
mm_manual = A @ B
mm_einsum = np.einsum('ik,kj->ij', A, B)
print(f"Matrix mult match: {np.allclose(mm_manual, mm_einsum)}")

# 4. Trace
M = np.random.randn(4, 4)
trace_manual = np.trace(M)
trace_einsum = np.einsum('ii->', M)
print(f"Trace match: {np.isclose(trace_manual, trace_einsum)}")

# 5. Outer product
outer_manual = np.outer(a, b)
outer_einsum = np.einsum('i,j->ij', a, b)
print(f"Outer product match: {np.allclose(outer_manual, outer_einsum)}")
```

### 3.2 고급 Einsum 연산

```python
# Batch operations (crucial for deep learning)

# Batch matrix multiplication
batch_size = 8
A_batch = np.random.randn(batch_size, 3, 4)
B_batch = np.random.randn(batch_size, 4, 5)

# Using einsum
C_batch = np.einsum('bij,bjk->bik', A_batch, B_batch)
print(f"Batch matmul shape: {C_batch.shape}")

# Verify against loop
for i in range(batch_size):
    assert np.allclose(C_batch[i], A_batch[i] @ B_batch[i])
print("Batch matmul verified")

# Bilinear form: x^T A y for batched x, A, y
x = np.random.randn(batch_size, 3)
A = np.random.randn(3, 3)
y = np.random.randn(batch_size, 3)

# Each sample: x[b]^T A y[b]
bilinear = np.einsum('bi,ij,bj->b', x, A, y)
print(f"Batched bilinear shape: {bilinear.shape}")

# Tensor contraction: contract mode 1 of T1 with mode 0 of T2
T1 = np.random.randn(2, 3, 4)
T2 = np.random.randn(3, 5)
result = np.einsum('ijk,jl->ikl', T1, T2)
print(f"Tensor contraction: ({T1.shape}) x ({T2.shape}) -> {result.shape}")
```

### 3.3 Einsum 최적화

```python
import time

# np.einsum with optimize=True can find the best contraction order
A = np.random.randn(100, 50)
B = np.random.randn(50, 80)
C = np.random.randn(80, 60)

# Three-matrix chain: A @ B @ C
# Order matters: (AB)C vs A(BC)
start = time.time()
for _ in range(100):
    result1 = np.einsum('ij,jk,kl->il', A, B, C)
t_unopt = time.time() - start

start = time.time()
for _ in range(100):
    result2 = np.einsum('ij,jk,kl->il', A, B, C, optimize=True)
t_opt = time.time() - start

print(f"Unoptimized: {t_unopt*1000:.1f} ms")
print(f"Optimized:   {t_opt*1000:.1f} ms")
print(f"Speedup: {t_unopt/t_opt:.1f}x")

# Show the optimal contraction path
path, desc = np.einsum_path('ij,jk,kl->il', A, B, C, optimize='optimal')
print(f"\nOptimal path: {path}")
print(desc)
```

---

## 4. 텐서곱과 연산

### 4.1 외적 (텐서곱)

$p$차 텐서와 $q$차 텐서의 **외적**은 $(p+q)$차 텐서를 생성합니다:

$$(\mathcal{A} \otimes \mathcal{B})_{i_1 \ldots i_p j_1 \ldots j_q} = \mathcal{A}_{i_1 \ldots i_p} \cdot \mathcal{B}_{j_1 \ldots j_q}$$

```python
# Outer product of vectors (rank-1 matrix)
a = np.array([1, 2, 3])
b = np.array([4, 5])
outer_ab = np.tensordot(a, b, axes=0)
print(f"a x b shape: {outer_ab.shape}")
print(f"a x b:\n{outer_ab}")

# Outer product of vector and matrix (3rd-order tensor)
M = np.array([[1, 2], [3, 4]])
outer_aM = np.tensordot(a, M, axes=0)
print(f"\na x M shape: {outer_aM.shape}")

# Outer product of two matrices (4th-order tensor)
A = np.random.randn(2, 3)
B = np.random.randn(4, 5)
outer_AB = np.tensordot(A, B, axes=0)
print(f"A x B shape: {outer_AB.shape}")  # (2, 3, 4, 5)
```

### 4.2 텐서 축약

**축약**은 인덱스 쌍에 대해 합산하여 텐서 차수를 2만큼 줄입니다. 행렬 곱셈은 하나의 인덱스 쌍에 대한 축약이며, 대각합(trace)은 동일 텐서의 두 인덱스에 대한 축약입니다.

```python
# Contraction is generalized matrix multiplication
A = np.random.randn(3, 4, 5)
B = np.random.randn(4, 6)

# Contract A's mode-1 with B's mode-0: C_{ikl} = sum_j A_{ijk} B_{jl}
C = np.tensordot(A, B, axes=([1], [0]))
print(f"Contraction: ({A.shape}) . ({B.shape}) -> {C.shape}")

# Double contraction (generalized trace)
# A : B = sum_{ij} A_{ij} B_{ij}
A2 = np.random.randn(3, 4)
B2 = np.random.randn(3, 4)
double_contraction = np.tensordot(A2, B2, axes=([0, 1], [0, 1]))
print(f"Double contraction (A:B): {double_contraction:.4f}")
print(f"Equivalent to trace(A^T B): {np.trace(A2.T @ B2):.4f}")

# Mode-n product: multiply tensor by matrix along mode n
def mode_n_product(T, M, mode):
    """Compute the mode-n product of tensor T with matrix M."""
    return np.tensordot(M, T, axes=([1], [mode]))

T = np.random.randn(3, 4, 5)
U = np.random.randn(6, 3)  # Maps mode-0 from 3 to 6

result = mode_n_product(T, U, mode=0)
print(f"\nMode-0 product: ({T.shape}) x ({U.shape}) -> {result.shape}")
```

---

## 5. 텐서 분해

### 5.1 CP 분해 (CANDECOMP/PARAFAC)

**CP 분해**는 텐서를 랭크-1 텐서들의 합으로 표현합니다:

$$\mathcal{T} \approx \sum_{r=1}^{R} \lambda_r \mathbf{a}_r \otimes \mathbf{b}_r \otimes \mathbf{c}_r$$

```python
# Manual CP decomposition construction
def construct_cp_tensor(factors, weights=None):
    """Construct a tensor from CP factors."""
    rank = factors[0].shape[1]
    if weights is None:
        weights = np.ones(rank)

    # Build tensor using einsum
    T = np.zeros([f.shape[0] for f in factors])
    for r in range(rank):
        term = weights[r]
        for f in factors:
            term = np.tensordot(term, f[:, r], axes=0)
        T += term
    return T

# Create a low-rank tensor
I, J, K = 5, 6, 7
R = 3  # CP rank

np.random.seed(42)
A = np.random.randn(I, R)
B = np.random.randn(J, R)
C = np.random.randn(K, R)
weights = np.array([3.0, 2.0, 1.0])

T = construct_cp_tensor([A, B, C], weights)
print(f"Tensor shape: {T.shape}")
print(f"CP rank: {R}")

# Verify with einsum
T_einsum = np.einsum('r,ir,jr,kr->ijk', weights, A, B, C)
print(f"Reconstruction match: {np.allclose(T, T_einsum)}")
```

### 5.2 ALS(교대 최소자승법)를 이용한 CP 분해

```python
def cp_als(T, rank, max_iter=100, tol=1e-6):
    """CP decomposition using Alternating Least Squares."""
    dims = T.shape
    n_modes = len(dims)

    # Initialize factors randomly
    factors = [np.random.randn(d, rank) for d in dims]

    # Normalize
    for n in range(n_modes):
        norms = np.linalg.norm(factors[n], axis=0)
        factors[n] /= norms

    prev_error = np.inf

    for iteration in range(max_iter):
        for n in range(n_modes):
            # Compute the Khatri-Rao product of all factors except n
            V = np.ones((rank, rank))
            for m in range(n_modes):
                if m != n:
                    V *= factors[m].T @ factors[m]

            # Matricize T along mode n
            T_n = unfold(T, n)

            # Khatri-Rao product
            kr = np.ones((1, rank))
            for m in reversed(range(n_modes)):
                if m != n:
                    kr = np.einsum('ir,jr->ijr', factors[m], kr).reshape(-1, rank)

            # Update factor
            factors[n] = T_n @ kr @ np.linalg.pinv(V)

        # Compute reconstruction error
        T_approx = np.einsum('ir,jr,kr->ijk', factors[0], factors[1], factors[2])
        error = np.linalg.norm(T - T_approx) / np.linalg.norm(T)

        if abs(prev_error - error) < tol:
            print(f"ALS converged in {iteration + 1} iterations, error = {error:.6f}")
            break
        prev_error = error

    return factors, error

# Test CP-ALS on a known low-rank tensor
T_test = construct_cp_tensor([A, B, C], weights)
# Add noise
T_noisy = T_test + 0.01 * np.random.randn(*T_test.shape)

factors_est, error = cp_als(T_noisy, rank=3)
print(f"Final relative error: {error:.6f}")
```

### 5.3 Tucker 분해

**Tucker 분해**는 텐서를 각 모드를 따라 행렬을 곱한 코어 텐서로 표현합니다:

$$\mathcal{T} \approx \mathcal{G} \times_1 U^{(1)} \times_2 U^{(2)} \times_3 U^{(3)}$$

여기서 $\mathcal{G}$는 코어 텐서($\mathcal{T}$보다 작음)이고 $U^{(n)}$은 인자 행렬입니다.

```python
def tucker_decomposition(T, ranks):
    """Tucker decomposition using Higher-Order SVD (HOSVD)."""
    n_modes = len(T.shape)
    factors = []

    for n in range(n_modes):
        T_n = unfold(T, n)
        U, S, Vt = np.linalg.svd(T_n, full_matrices=False)
        factors.append(U[:, :ranks[n]])

    # Compute core tensor
    core = T.copy()
    for n in range(n_modes):
        core = np.tensordot(factors[n].T, core, axes=([1], [0]))

    return core, factors

# Tucker decomposition of a 3rd-order tensor
T = np.random.randn(10, 12, 8)
ranks = (3, 4, 2)  # Reduced dimensions

core, factors = tucker_decomposition(T, ranks)
print(f"Original tensor shape: {T.shape}")
print(f"Core tensor shape: {core.shape}")
for i, f in enumerate(factors):
    print(f"Factor {i} shape: {f.shape}")

# Reconstruction
T_approx = core.copy()
for n in range(len(factors)):
    T_approx = np.tensordot(factors[n], T_approx, axes=([1], [0]))

error = np.linalg.norm(T - T_approx) / np.linalg.norm(T)
print(f"\nReconstruction error: {error:.6f}")

# Compression ratio
original_params = np.prod(T.shape)
compressed_params = np.prod(core.shape) + sum(f.size for f in factors)
print(f"Compression ratio: {original_params / compressed_params:.1f}x")
```

---

## 6. 딥러닝에서의 텐서

### 6.1 배치 연산

딥러닝 프레임워크는 데이터를 배치 단위로 처리하며, 모든 연산에서 추가적인 배치 차원을 다루어야 합니다. 이는 텐서 연산으로 자연스럽게 표현됩니다.

```python
import numpy as np

# Batched linear layer: Y = XW + b
batch_size = 32
in_features = 128
out_features = 64

X = np.random.randn(batch_size, in_features)
W = np.random.randn(in_features, out_features)
b = np.random.randn(out_features)

# This is a batched matrix-vector product
Y = np.einsum('bi,io->bo', X, W) + b
print(f"Input shape: {X.shape}")
print(f"Output shape: {Y.shape}")
# Equivalent to:
Y_matmul = X @ W + b
print(f"Match: {np.allclose(Y, Y_matmul)}")
```

### 6.2 텐서 연산으로서의 Attention

Scaled dot-product attention 메커니즘은 일련의 텐서 연산입니다:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

```python
def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

# Multi-head attention (batched)
batch_size = 4
n_heads = 8
seq_len = 16
d_k = 64  # Key/query dimension
d_v = 64  # Value dimension

Q = np.random.randn(batch_size, n_heads, seq_len, d_k)
K = np.random.randn(batch_size, n_heads, seq_len, d_k)
V = np.random.randn(batch_size, n_heads, seq_len, d_v)

# Step 1: Compute attention scores (Q K^T / sqrt(d_k))
# Shape: (batch, heads, seq, seq)
scores = np.einsum('bhqd,bhkd->bhqk', Q, K) / np.sqrt(d_k)
print(f"Attention scores shape: {scores.shape}")

# Step 2: Softmax
weights = softmax(scores, axis=-1)
print(f"Attention weights shape: {weights.shape}")
print(f"Weights sum to 1: {np.allclose(weights.sum(axis=-1), 1.0)}")

# Step 3: Weighted sum of values
output = np.einsum('bhqk,bhkd->bhqd', weights, V)
print(f"Attention output shape: {output.shape}")
```

### 6.3 텐서 연산으로서의 합성곱

2D 합성곱은 텐서 축약으로 표현할 수 있습니다:

```python
# Simple 2D convolution via einsum
def conv2d_einsum(input_tensor, kernel, stride=1):
    """2D convolution using einsum.

    Args:
        input_tensor: (batch, in_channels, height, width)
        kernel: (out_channels, in_channels, kh, kw)
    """
    batch, in_c, h, w = input_tensor.shape
    out_c, _, kh, kw = kernel.shape
    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1

    # Extract patches
    patches = np.zeros((batch, in_c, out_h, out_w, kh, kw))
    for i in range(out_h):
        for j in range(out_w):
            patches[:, :, i, j] = input_tensor[
                :, :, i*stride:i*stride+kh, j*stride:j*stride+kw]

    # Convolution as tensor contraction
    output = np.einsum('bcihmn,ocihmn->boh', patches, kernel[:, :, np.newaxis, np.newaxis, :, :].repeat(out_h, axis=2).repeat(out_w, axis=3))
    return output.reshape(batch, out_c, out_h, out_w)

# Alternative: im2col approach (more standard)
def conv2d_im2col(input_tensor, kernel, stride=1):
    """2D convolution via im2col (matrix multiplication)."""
    batch, in_c, h, w = input_tensor.shape
    out_c, _, kh, kw = kernel.shape
    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1

    # im2col: unfold input into columns
    col = np.zeros((batch, in_c * kh * kw, out_h * out_w))
    for i in range(out_h):
        for j in range(out_w):
            patch = input_tensor[:, :, i*stride:i*stride+kh, j*stride:j*stride+kw]
            col[:, :, i * out_w + j] = patch.reshape(batch, -1)

    # Reshape kernel to matrix
    kernel_mat = kernel.reshape(out_c, -1)

    # Matrix multiplication
    output = np.einsum('oi,bip->bop', kernel_mat, col)
    return output.reshape(batch, out_c, out_h, out_w)

# Test
input_t = np.random.randn(2, 3, 8, 8)   # 2 images, 3 channels, 8x8
kernel = np.random.randn(16, 3, 3, 3)   # 16 filters, 3x3

output = conv2d_im2col(input_t, kernel)
print(f"Input shape:  {input_t.shape}")
print(f"Kernel shape: {kernel.shape}")
print(f"Output shape: {output.shape}")
```

---

## 7. 텐서 네트워크 (간략 소개)

### 7.1 텐서 트레인 (TT) 분해

텐서 트레인 분해는 고차 텐서를 3차 텐서들의 체인으로 표현합니다:

$$\mathcal{T}_{i_1, i_2, \ldots, i_N} = G_1[i_1] \cdot G_2[i_2] \cdots G_N[i_N]$$

여기서 각 $G_k[i_k]$는 행렬(또는 경계 코어의 경우 벡터)입니다.

```python
def tt_decompose(T, max_rank=None):
    """Tensor Train decomposition via sequential SVD."""
    dims = T.shape
    N = len(dims)
    cores = []
    C = T.copy()
    ranks = [1]

    for n in range(N - 1):
        C = C.reshape(ranks[-1] * dims[n], -1)
        U, S, Vt = np.linalg.svd(C, full_matrices=False)

        # Truncate
        r = len(S) if max_rank is None else min(max_rank, len(S))
        r = min(r, np.sum(S > S[0] * 1e-10))
        r = max(r, 1)

        U = U[:, :r]
        S = S[:r]
        Vt = Vt[:r, :]

        cores.append(U.reshape(ranks[-1], dims[n], r))
        C = np.diag(S) @ Vt
        ranks.append(r)

    cores.append(C.reshape(ranks[-1], dims[-1], 1))
    return cores

def tt_to_full(cores):
    """Reconstruct full tensor from TT cores."""
    T = cores[0]
    for core in cores[1:]:
        T = np.tensordot(T, core, axes=([-1], [0]))
    return T.squeeze()

# Example
T = np.random.randn(4, 5, 6, 3)
cores = tt_decompose(T, max_rank=10)

print("TT cores:")
for i, core in enumerate(cores):
    print(f"  Core {i}: shape {core.shape}")

T_reconstructed = tt_to_full(cores)
error = np.linalg.norm(T - T_reconstructed) / np.linalg.norm(T)
print(f"Reconstruction error: {error:.6e}")

# Compression
original = np.prod(T.shape)
compressed = sum(c.size for c in cores)
print(f"Parameters: {original} -> {compressed} ({original/compressed:.1f}x)")
```

---

## 연습 문제

### 문제 1: Einsum 숙달

다음 연산들을 `np.einsum`으로 표현하고 NumPy/SciPy 동치와 비교하여 검증하십시오:

1. 행렬의 Frobenius 노름: $\|A\|_F = \sqrt{\text{tr}(A^T A)}$
2. 배치 외적: $(B, n)$, $(B, m)$ 형상의 배치 벡터 $u, v$에 대해 모든 외적을 계산하여 $(B, n, m)$ 형상을 얻으시오
3. 텐서 축약: $(3, 4, 5)$ 텐서와 $(5, 6)$ 행렬을 마지막 모드를 따라 축약하시오
4. 이차 형식: $(B, n)$ 형상의 벡터 배치 $x$와 $(n, n)$ 형상의 행렬 $A$에 대해 각 샘플의 $x^T A x$를 계산하시오

### 문제 2: CP 분해

1. $(10, 12, 8)$ 형상의 랭크-5 텐서를 5개의 랭크-1 텐서의 합으로 생성하시오
2. 가우시안 노이즈 추가 (SNR = 20 dB)
3. 랭크 3, 5, 7, 10으로 CP-ALS를 적용하시오
4. 재구성 오차 대 랭크를 그래프로 나타내시오

### 문제 3: Tucker vs CP

$(20, 15, 10)$ 형상의 임의 텐서에 대해:

1. 랭크 $(5, 4, 3)$으로 Tucker 분해를 계산하시오
2. 랭크 15로 CP 분해를 계산하시오
3. 재구성 오차와 압축 비율을 비교하시오

### 문제 4: Attention 메커니즘

`np.einsum`을 사용하여 scaled dot-product attention을 처음부터 구현하시오:

1. $(2, 4, 8, 16)$ (배치, 헤드, 시퀀스, 차원) 형상의 임의 Q, K, V 텐서를 생성하시오
2. Attention 가중치와 출력을 계산하시오
3. Attention 가중치가 키 차원을 따라 합이 1이 되는지 검증하시오

### 문제 5: 텐서 트레인

$(4, 4, 4, 4, 4)$ 형상의 5차 텐서를 다양한 최대 랭크(2, 4, 8, 16)로 TT 분해하시오. 각 랭크에 대한 상대 재구성 오차와 압축 비율을 그래프로 나타내시오.

---

[이전: 14강](./14_Iterative_Methods.md) | [개요](./00_Overview.md) | [다음: 16강](./16_Linear_Algebra_in_Machine_Learning.md)

**License**: CC BY-NC 4.0
