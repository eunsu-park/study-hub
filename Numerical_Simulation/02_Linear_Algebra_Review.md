# 선형대수 복습

## 개요

수치 시뮬레이션에서 선형대수는 핵심적인 역할을 합니다. 행렬 연산, 연립방정식 풀이, 고유값 문제, 행렬 분해 등을 NumPy/SciPy로 구현하는 방법을 학습합니다.

---

## 1. 행렬 기본 연산

### 1.1 행렬 생성과 연산

```python
import numpy as np
from scipy import linalg

# 행렬 생성
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 10]])

B = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])

# 기본 연산
print("행렬 A:")
print(A)
print(f"\n전치: A.T =\n{A.T}")
print(f"\n행렬곱: A @ B =\n{A @ B}")
print(f"\n요소별 곱: A * B =\n{A * B}")
print(f"\n역행렬: A⁻¹ =\n{np.linalg.inv(A)}")
print(f"\n행렬식: det(A) = {np.linalg.det(A):.4f}")
print(f"\n대각합(trace): tr(A) = {np.trace(A)}")
```

### 1.2 특수 행렬 생성

```python
n = 4

# 항등 행렬
I = np.eye(n)

# 영 행렬
Z = np.zeros((n, n))

# 대각 행렬
D = np.diag([1, 2, 3, 4])

# 삼중대각 행렬 (tridiagonal)
diag_main = 2 * np.ones(n)
diag_off = -1 * np.ones(n - 1)
T = np.diag(diag_main) + np.diag(diag_off, k=1) + np.diag(diag_off, k=-1)

print("삼중대각 행렬:")
print(T)

# 랜덤 행렬
np.random.seed(42)
R = np.random.randn(3, 3)
print(f"\n랜덤 행렬:\n{R}")
```

### 1.3 행렬 노름

```python
A = np.array([[1, 2], [3, 4]])

# 다양한 노름
print("행렬 노름:")
print(f"  1-노름 (열 합 최대): {np.linalg.norm(A, 1)}")
print(f"  2-노름 (스펙트럴): {np.linalg.norm(A, 2)}")
print(f"  ∞-노름 (행 합 최대): {np.linalg.norm(A, np.inf)}")
print(f"  프로베니우스 노름: {np.linalg.norm(A, 'fro')}")

# 벡터 노름
v = np.array([3, 4])
print(f"\n벡터 노름:")
print(f"  L1: {np.linalg.norm(v, 1)}")
print(f"  L2: {np.linalg.norm(v, 2)}")
print(f"  L∞: {np.linalg.norm(v, np.inf)}")
```

---

## 2. 연립방정식 풀이

### 2.1 직접 풀이

```python
# Ax = b 풀기
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])

# numpy.linalg.solve (권장)
x = np.linalg.solve(A, b)
print(f"해: x = {x}")
print(f"검증: Ax = {A @ x}")

# 역행렬 사용 (비권장 - 느리고 불안정)
x_inv = np.linalg.inv(A) @ b
print(f"역행렬 방식: x = {x_inv}")
```

### 2.2 과결정/미결정 시스템

```python
# 과결정 시스템 (방정식 > 미지수) - 최소자승해
A_over = np.array([[1, 1], [1, 2], [1, 3]])
b_over = np.array([1, 2, 2])

# 최소자승해
x_lstsq, residuals, rank, s = np.linalg.lstsq(A_over, b_over, rcond=None)
print(f"최소자승해: {x_lstsq}")
print(f"잔차: {residuals}")

# 미결정 시스템 (방정식 < 미지수) - 최소 노름해
A_under = np.array([[1, 2, 3]])
b_under = np.array([6])

# 의사역행렬 사용
x_min_norm = np.linalg.pinv(A_under) @ b_under
print(f"\n최소 노름해: {x_min_norm}")
print(f"노름: {np.linalg.norm(x_min_norm):.4f}")
```

---

## 3. 고유값과 고유벡터

### 3.1 고유값 분해

```python
A = np.array([[4, -2],
              [1,  1]])

# 고유값, 고유벡터 계산
eigenvalues, eigenvectors = np.linalg.eig(A)

print("고유값 분해:")
print(f"고유값: {eigenvalues}")
print(f"고유벡터:\n{eigenvectors}")

# 검증: A @ v = λ @ v
for i in range(len(eigenvalues)):
    lam = eigenvalues[i]
    v = eigenvectors[:, i]
    print(f"\nλ_{i} = {lam:.4f}")
    print(f"  A @ v = {A @ v}")
    print(f"  λ * v = {lam * v}")
```

### 3.2 대칭 행렬의 고유값

```python
# 대칭 행렬 - 실수 고유값 보장
S = np.array([[2, 1, 0],
              [1, 3, 1],
              [0, 1, 2]])

# eigh는 대칭 행렬에 최적화 (더 빠르고 안정적)
eigenvalues, eigenvectors = np.linalg.eigh(S)

print("대칭 행렬 고유값 분해:")
print(f"고유값: {eigenvalues}")
print(f"\n고유벡터 직교성 검증:")
print(f"V^T @ V =\n{eigenvectors.T @ eigenvectors}")  # 단위행렬
```

### 3.3 거듭제곱법 (Power Method)

```python
def power_method(A, max_iter=100, tol=1e-10):
    """최대 고유값과 고유벡터를 거듭제곱법으로 계산"""
    n = A.shape[0]
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)

    for _ in range(max_iter):
        v_new = A @ v
        eigenvalue = np.dot(v, v_new)  # Rayleigh quotient
        v_new = v_new / np.linalg.norm(v_new)

        if np.linalg.norm(v_new - v) < tol:
            break
        v = v_new

    return eigenvalue, v_new

A = np.array([[4, 1], [2, 3]])
lam, v = power_method(A)
print(f"거듭제곱법 결과:")
print(f"  최대 고유값: {lam:.6f}")
print(f"  고유벡터: {v}")

# numpy 결과와 비교
lam_np, v_np = np.linalg.eig(A)
print(f"\nnumpy 결과:")
print(f"  고유값: {lam_np}")
```

---

## 4. 행렬 분해

### 4.1 LU 분해

```python
from scipy.linalg import lu, lu_factor, lu_solve

A = np.array([[2, 1, 1],
              [4, 3, 3],
              [8, 7, 9]], dtype=float)

# LU 분해
P, L, U = lu(A)

print("LU 분해:")
print(f"P (순열 행렬):\n{P}")
print(f"L (하삼각):\n{L}")
print(f"U (상삼각):\n{U}")
print(f"\n검증: P @ L @ U =\n{P @ L @ U}")

# 연립방정식 풀이에 활용
b = np.array([4, 10, 24])
lu_piv = lu_factor(A)
x = lu_solve(lu_piv, b)
print(f"\n해: {x}")
```

### 4.2 Cholesky 분해

```python
# 양정치 대칭 행렬에 대해서만 가능
A_spd = np.array([[4, 2, 2],
                  [2, 5, 1],
                  [2, 1, 6]], dtype=float)

# Cholesky 분해: A = L @ L.T
L = np.linalg.cholesky(A_spd)

print("Cholesky 분해:")
print(f"L:\n{L}")
print(f"\n검증: L @ L.T =\n{L @ L.T}")

# 연립방정식 풀이 (LU보다 2배 빠름)
b = np.array([8, 8, 9])
# L @ y = b 풀고, L.T @ x = y 풀기
y = linalg.solve_triangular(L, b, lower=True)
x = linalg.solve_triangular(L.T, y)
print(f"\n해: {x}")
```

### 4.3 QR 분해

```python
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 10],
              [10, 11, 12]], dtype=float)

# QR 분해
Q, R = np.linalg.qr(A)

print("QR 분해:")
print(f"Q (직교 행렬):\n{Q}")
print(f"\nR (상삼각):\n{R}")
print(f"\n검증: Q @ R =\n{Q @ R}")
print(f"\nQ의 직교성: Q.T @ Q =\n{Q.T @ Q}")

# 최소자승 문제에 활용
b = np.array([1, 2, 3, 4])
x = linalg.solve_triangular(R, Q.T @ b)
print(f"\n최소자승해: {x}")
```

### 4.4 SVD (특이값 분해)

```python
A = np.array([[1, 2],
              [3, 4],
              [5, 6]])

# SVD: A = U @ Σ @ V.T
U, s, Vt = np.linalg.svd(A)

print("SVD 분해:")
print(f"U (m×m 직교):\n{U}")
print(f"\n특이값: {s}")
print(f"\nV.T (n×n 직교):\n{Vt}")

# Σ 행렬 구성
Sigma = np.zeros_like(A, dtype=float)
np.fill_diagonal(Sigma, s)

print(f"\n검증: U @ Σ @ V.T =\n{U @ Sigma @ Vt}")

# 조건수 계산
cond = s[0] / s[-1]
print(f"\n조건수: {cond:.4f}")
```

---

## 5. 희소 행렬

### 5.1 희소 행렬 형식

```python
from scipy import sparse

# COO 형식 (좌표 형식)
row = [0, 1, 2, 2]
col = [0, 1, 0, 2]
data = [1, 2, 3, 4]
A_coo = sparse.coo_matrix((data, (row, col)), shape=(3, 3))

print("COO 형식:")
print(A_coo)
print(f"\n밀집 형태:\n{A_coo.toarray()}")

# CSR 형식 (행 압축, 행렬-벡터 곱에 효율적)
A_csr = A_coo.tocsr()
print(f"\nCSR 형식: {A_csr}")

# CSC 형식 (열 압축, 열 슬라이싱에 효율적)
A_csc = A_coo.tocsc()
```

### 5.2 희소 행렬 연산

```python
# 큰 희소 행렬 생성
n = 1000
diagonals = [np.ones(n-1), -2*np.ones(n), np.ones(n-1)]
offsets = [-1, 0, 1]
A_sparse = sparse.diags(diagonals, offsets, format='csr')

print(f"희소 행렬 크기: {A_sparse.shape}")
print(f"비영 요소 수: {A_sparse.nnz}")
print(f"희소도: {1 - A_sparse.nnz / (n*n):.4%}")

# 희소 행렬 연립방정식
b = np.random.randn(n)
x = sparse.linalg.spsolve(A_sparse, b)
print(f"\n희소 연립방정식 해의 노름: {np.linalg.norm(x):.4f}")
```

### 5.3 반복 솔버

```python
from scipy.sparse.linalg import cg, gmres, bicgstab

# 대칭 양정치 행렬 생성
n = 100
A = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(n, n), format='csr')
A = A.T @ A + 0.1 * sparse.eye(n)  # 양정치로 만들기
b = np.random.randn(n)

# CG (Conjugate Gradient) - 대칭 양정치에 최적
x_cg, info_cg = cg(A, b, tol=1e-10)
print(f"CG: 수렴 상태 = {info_cg}, 잔차 = {np.linalg.norm(A @ x_cg - b):.2e}")

# GMRES - 일반 행렬
x_gmres, info_gmres = gmres(A, b, tol=1e-10)
print(f"GMRES: 수렴 상태 = {info_gmres}, 잔차 = {np.linalg.norm(A @ x_gmres - b):.2e}")
```

---

## 6. 응용 예제

### 6.1 열전도 문제의 이산화

```python
def heat_equation_matrix(n, alpha, dx, dt):
    """1D 열방정식의 암시적 이산화 행렬"""
    r = alpha * dt / dx**2

    # 삼중대각 행렬 생성
    main_diag = (1 + 2*r) * np.ones(n)
    off_diag = -r * np.ones(n - 1)

    A = np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
    return A

n = 10
A = heat_equation_matrix(n, alpha=0.1, dx=0.1, dt=0.01)
print("열방정식 이산화 행렬:")
print(A[:5, :5])  # 일부만 출력
```

### 6.2 이미지 압축 (SVD)

```python
def svd_compression(image, k):
    """SVD로 이미지 압축 (상위 k개 특이값 사용)"""
    U, s, Vt = np.linalg.svd(image, full_matrices=False)

    # 상위 k개 성분만 사용
    compressed = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

    # 압축률 계산
    original_size = image.shape[0] * image.shape[1]
    compressed_size = k * (image.shape[0] + image.shape[1] + 1)
    compression_ratio = compressed_size / original_size

    return compressed, compression_ratio

# 예시 (랜덤 이미지)
image = np.random.randn(100, 100)
for k in [5, 10, 20, 50]:
    comp, ratio = svd_compression(image, k)
    error = np.linalg.norm(image - comp, 'fro') / np.linalg.norm(image, 'fro')
    print(f"k={k:2d}: 압축률={ratio:.2%}, 상대오차={error:.4f}")
```

---

## 연습 문제

### 문제 1
3x3 삼중대각 행렬의 고유값을 구하고, 거듭제곱법 결과와 비교하세요.

```python
# 풀이
T = np.array([[2, -1, 0],
              [-1, 2, -1],
              [0, -1, 2]], dtype=float)

# numpy
eig_np, _ = np.linalg.eig(T)
print(f"numpy 고유값: {sorted(eig_np)}")

# 거듭제곱법
lam, v = power_method(T)
print(f"거듭제곱법 최대 고유값: {lam:.6f}")
```

### 문제 2
100x100 희소 행렬의 연립방정식을 직접법과 반복법으로 풀고 시간을 비교하세요.

```python
import time

n = 1000
A = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(n, n), format='csr')
A = A + 3 * sparse.eye(n)  # 대각 우세하게
b = np.random.randn(n)

# 직접법
start = time.time()
x_direct = sparse.linalg.spsolve(A, b)
time_direct = time.time() - start

# 반복법 (CG)
A_spd = A.T @ A
b_spd = A.T @ b
start = time.time()
x_cg, _ = cg(A_spd, b_spd, tol=1e-10)
time_cg = time.time() - start

print(f"직접법: {time_direct:.4f}초")
print(f"CG: {time_cg:.4f}초")
```

---

## 요약

| 분해 방법 | 용도 | 조건 |
|----------|------|------|
| LU | 연립방정식 풀이 | 정방 행렬 |
| Cholesky | 연립방정식 (빠름) | 대칭 양정치 |
| QR | 최소자승, 고유값 | 모든 행렬 |
| SVD | 압축, 의사역행렬 | 모든 행렬 |

| 솔버 | 행렬 유형 | 특징 |
|------|----------|------|
| spsolve | 희소 | 직접법 |
| CG | 대칭 양정치 | 반복법 |
| GMRES | 일반 | 반복법 |
| BiCGSTAB | 비대칭 | 반복법 |
