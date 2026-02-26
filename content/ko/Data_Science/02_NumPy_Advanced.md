# 2. NumPy 고급

[이전: NumPy 기초](./01_NumPy_Basics.md) | [다음: Pandas 기초](./03_Pandas_Basics.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `np.linalg`를 이용해 행렬 곱셈, 분해(decomposition), 선형 방정식 풀이를 구현하다
2. NumPy로 기술 통계량(descriptive statistics), 상관계수(correlation coefficients), 공분산 행렬(covariance matrix)을 계산하다
3. 새로운 방식의 난수 생성기(`default_rng`)를 적용하여 다양한 확률 분포에서 샘플링하다
4. 이종 표 형식 데이터(heterogeneous tabular data)를 위한 구조화된 배열(structured arrays)과 레코드 배열(record arrays)을 설명하다
5. C-order와 Fortran-order 메모리 레이아웃을 비교하고 성능에 미치는 영향을 설명하다
6. 벡터화(vectorized) 연산을 수행하고, Python 루프보다 빠른 이유를 설명하다
7. `np.where`, 마스크 배열(masked arrays), `np.take`/`np.put` 등 고급 인덱싱 기법을 적용하다
8. 바이너리(`.npy`/`.npz`) 및 텍스트(CSV) 형식으로 배열을 저장·불러오기 하고, 대용량 데이터셋에 메모리 매핑(memory mapping)을 사용하다

---

배열 생성과 조작에 익숙해졌다면, 다음 단계는 NumPy의 강력한 계산 도구를 활용하는 것입니다. 선형 방정식 풀이부터 재현 가능한 난수 샘플 생성까지, 이 고급 기능들은 NumPy를 완전한 수치 컴퓨팅 환경으로 만들어 줍니다. 메모리 레이아웃과 벡터화를 이해하면 단순 Python 루프보다 수십 배 빠른 코드를 작성할 수 있습니다.

---

## 1. 선형대수 (Linear Algebra)

### 1.1 행렬 곱셈

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 행렬 곱셈 (dot product)
C = np.dot(A, B)
print(C)
# [[19 22]
#  [43 50]]

# @ 연산자 (Python 3.5+)
C = A @ B

# matmul 함수
C = np.matmul(A, B)

# 벡터 내적
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
dot_product = np.dot(v1, v2)  # 32
```

### 1.2 행렬 분해

```python
A = np.array([[1, 2], [3, 4]])

# 행렬식 (Determinant)
det = np.linalg.det(A)
print(det)  # -2.0

# 역행렬 (Inverse)
A_inv = np.linalg.inv(A)
print(A_inv)
# [[-2.   1. ]
#  [ 1.5 -0.5]]

# 검증: A @ A_inv = I
print(A @ A_inv)
# [[1. 0.]
#  [0. 1.]]

# 고유값과 고유벡터
eigenvalues, eigenvectors = np.linalg.eig(A)
print("고유값:", eigenvalues)
print("고유벡터:\n", eigenvectors)

# 특이값 분해 (SVD)
U, S, Vt = np.linalg.svd(A)
print("U:\n", U)
print("S:", S)
print("Vt:\n", Vt)

# QR 분해
Q, R = np.linalg.qr(A)

# 촐레스키 분해 (대칭 양정치 행렬)
B = np.array([[4, 2], [2, 5]])
L = np.linalg.cholesky(B)
```

### 1.3 선형 방정식 풀이

```python
# Ax = b 형태의 선형 시스템
# 2x + y = 5
# x + 3y = 6

A = np.array([[2, 1], [1, 3]])
b = np.array([5, 6])

# 해 구하기
x = np.linalg.solve(A, b)
print(x)  # [1.8 1.4]

# 검증
print(A @ x)  # [5. 6.]

# 최소 자승법 (Least Squares)
A = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])
b = np.array([2, 3, 4.5, 5])
x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
print("계수:", x)  # [0.75 1.1]
```

### 1.4 행렬 노름과 조건수

```python
A = np.array([[1, 2], [3, 4]])

# 프로베니우스 노름 (Frobenius norm)
fro_norm = np.linalg.norm(A, 'fro')

# L2 노름 (스펙트럴 노름)
l2_norm = np.linalg.norm(A, 2)

# L1 노름
l1_norm = np.linalg.norm(A, 1)

# 무한대 노름
inf_norm = np.linalg.norm(A, np.inf)

# 벡터 노름
v = np.array([3, 4])
print(np.linalg.norm(v))  # 5.0 (유클리드 거리)

# 조건수 (Condition Number)
cond = np.linalg.cond(A)
print("조건수:", cond)
```

### 1.5 행렬 랭크와 트레이스

```python
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 랭크
rank = np.linalg.matrix_rank(A)
print("랭크:", rank)  # 2

# 트레이스 (대각선 합)
trace = np.trace(A)
print("트레이스:", trace)  # 15
```

---

## 2. 통계 함수

### 2.1 기술 통계

```python
data = np.array([23, 45, 67, 89, 12, 34, 56, 78, 90, 11])

# 기본 통계
print("평균:", np.mean(data))      # 50.5
print("중앙값:", np.median(data))  # 50.5
print("표준편차:", np.std(data))   # 28.07
print("분산:", np.var(data))       # 788.25
print("최소:", np.min(data))       # 11
print("최대:", np.max(data))       # 90
print("범위:", np.ptp(data))       # 79 (peak to peak)

# 백분위수
print("25%:", np.percentile(data, 25))
print("50%:", np.percentile(data, 50))
print("75%:", np.percentile(data, 75))

# 분위수
print("1사분위:", np.quantile(data, 0.25))
print("3사분위:", np.quantile(data, 0.75))
```

### 2.2 상관계수와 공분산

```python
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# 상관계수 행렬
corr_matrix = np.corrcoef(x, y)
print(corr_matrix)
# [[1.   0.77]
#  [0.77 1.  ]]

# 공분산 행렬
cov_matrix = np.cov(x, y)
print(cov_matrix)
# [[2.5 1.5]
#  [1.5 1.3]]

# 다변량 데이터
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(np.corrcoef(data))  # 변수 간 상관계수
```

### 2.3 히스토그램과 빈도

```python
data = np.random.randn(1000)

# 히스토그램 계산
counts, bin_edges = np.histogram(data, bins=10)
print("빈도:", counts)
print("구간:", bin_edges)

# 빈 지정
counts, bin_edges = np.histogram(data, bins=[-3, -2, -1, 0, 1, 2, 3])

# 고유값과 빈도
arr = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
unique, counts = np.unique(arr, return_counts=True)
print("고유값:", unique)   # [1 2 3 4]
print("빈도:", counts)     # [1 2 3 4]
```

---

## 3. 난수 생성

### 3.1 기본 난수 생성

```python
# 새로운 방식 (NumPy 1.17+)
rng = np.random.default_rng(seed=42)

# 균일 분포 [0, 1)
print(rng.random(5))

# 정수 난수
print(rng.integers(1, 100, size=10))

# 균일 분포 [low, high)
print(rng.uniform(0, 10, size=5))

# 레거시 방식
np.random.seed(42)
print(np.random.rand(5))        # [0, 1) 균일 분포
print(np.random.randint(1, 10, 5))  # 정수 난수
```

### 3.2 확률 분포

```python
rng = np.random.default_rng(42)

# 정규 분포 (가우시안)
normal = rng.normal(loc=0, scale=1, size=1000)  # 평균 0, 표준편차 1
print(f"평균: {normal.mean():.3f}, 표준편차: {normal.std():.3f}")

# 표준 정규 분포
standard_normal = rng.standard_normal(1000)

# 이항 분포
binomial = rng.binomial(n=10, p=0.5, size=1000)  # n번 시행, 성공 확률 p

# 포아송 분포
poisson = rng.poisson(lam=5, size=1000)  # 평균 5

# 지수 분포
exponential = rng.exponential(scale=2, size=1000)

# 베타 분포
beta = rng.beta(a=2, b=5, size=1000)

# 감마 분포
gamma = rng.gamma(shape=2, scale=1, size=1000)

# 카이제곱 분포
chisquare = rng.chisquare(df=5, size=1000)

# t 분포
t = rng.standard_t(df=10, size=1000)
```

### 3.3 랜덤 샘플링

```python
rng = np.random.default_rng(42)

arr = np.array([10, 20, 30, 40, 50])

# 랜덤 선택
sample = rng.choice(arr, size=3, replace=False)  # 비복원 추출
print(sample)

# 확률 가중치
weights = [0.1, 0.1, 0.3, 0.3, 0.2]
sample = rng.choice(arr, size=10, p=weights)

# 배열 셔플
arr_copy = arr.copy()
rng.shuffle(arr_copy)
print(arr_copy)

# 순열 (새 배열 반환)
permuted = rng.permutation(arr)
print(permuted)
```

---

## 4. 구조화된 배열

### 4.1 구조화된 dtype

```python
# 구조화된 배열 정의
dt = np.dtype([
    ('name', 'U20'),      # 유니코드 문자열 (최대 20자)
    ('age', 'i4'),        # 32비트 정수
    ('height', 'f8'),     # 64비트 실수
    ('is_student', '?')   # 불리언
])

# 데이터 생성
data = np.array([
    ('Alice', 25, 165.5, True),
    ('Bob', 30, 178.2, False),
    ('Charlie', 22, 172.0, True)
], dtype=dt)

# 필드 접근
print(data['name'])    # ['Alice' 'Bob' 'Charlie']
print(data['age'])     # [25 30 22]
print(data[0])         # ('Alice', 25, 165.5, True)
print(data[0]['name']) # Alice

# 조건 필터링
students = data[data['is_student']]
print(students['name'])
```

### 4.2 레코드 배열

```python
# recarray로 변환 (속성 접근 가능)
rec = data.view(np.recarray)
print(rec.name)    # ['Alice' 'Bob' 'Charlie']
print(rec.age)     # [25 30 22]
print(rec[0].name) # Alice
```

---

## 5. 메모리 레이아웃과 성능

### 5.1 C-order vs Fortran-order

```python
# C-order (행 우선): 기본값
c_arr = np.array([[1, 2, 3], [4, 5, 6]], order='C')
print(c_arr.flags['C_CONTIGUOUS'])  # True

# Fortran-order (열 우선)
f_arr = np.array([[1, 2, 3], [4, 5, 6]], order='F')
print(f_arr.flags['F_CONTIGUOUS'])  # True

# 메모리 레이아웃 확인
print(c_arr.strides)  # (24, 8) - 행 이동 24바이트, 열 이동 8바이트
print(f_arr.strides)  # (8, 16)
```

### 5.2 뷰와 복사 성능

```python
import time

arr = np.arange(10000000)

# 슬라이싱 (뷰) - 빠름
start = time.time()
view = arr[::2]
print(f"뷰 생성: {time.time() - start:.6f}초")

# 복사 - 느림
start = time.time()
copy = arr[::2].copy()
print(f"복사: {time.time() - start:.6f}초")
```

### 5.3 벡터화 vs 루프

```python
import time

n = 1000000
arr = np.random.rand(n)

# 파이썬 루프 (느림)
start = time.time()
result = []
for x in arr:
    result.append(x ** 2)
print(f"파이썬 루프: {time.time() - start:.4f}초")

# NumPy 벡터화 (빠름)
start = time.time()
result = arr ** 2
print(f"NumPy 벡터화: {time.time() - start:.4f}초")
```

### 5.4 Universal Functions 최적화

```python
# where 사용
arr = np.array([1, -2, 3, -4, 5])
result = np.where(arr > 0, arr, 0)  # 양수는 유지, 음수는 0
print(result)  # [1 0 3 0 5]

# select 사용 (다중 조건)
conditions = [arr < 0, arr == 0, arr > 0]
choices = [-1, 0, 1]
result = np.select(conditions, choices)
print(result)  # [ 1 -1  1 -1  1]

# clip 사용
arr = np.array([-5, -2, 0, 3, 7, 10])
result = np.clip(arr, 0, 5)  # 0과 5 사이로 제한
print(result)  # [0 0 0 3 5 5]
```

---

## 6. 고급 인덱싱과 마스킹

### 6.1 np.where 활용

```python
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 조건을 만족하는 인덱스
indices = np.where(arr > 5)
print(indices)  # (array([1, 2, 2, 2]), array([2, 0, 1, 2]))

# 조건부 값 할당
result = np.where(arr % 2 == 0, 'even', 'odd')
print(result)
```

### 6.2 np.take와 np.put

```python
arr = np.array([10, 20, 30, 40, 50])

# take: 인덱스로 요소 가져오기
indices = [0, 2, 4]
print(np.take(arr, indices))  # [10 30 50]

# put: 인덱스 위치에 값 넣기
np.put(arr, [0, 2, 4], [100, 300, 500])
print(arr)  # [100  20 300  40 500]
```

### 6.3 마스크 배열

```python
# 마스크 배열 생성
data = np.array([1, 2, -999, 4, -999, 6])
mask = (data == -999)

masked_arr = np.ma.masked_array(data, mask)
print(masked_arr)  # [1 2 -- 4 -- 6]
print(masked_arr.mean())  # 3.25 (마스크된 값 제외)

# 마스크된 값 채우기
filled = masked_arr.filled(0)
print(filled)  # [1 2 0 4 0 6]
```

---

## 7. 배열 저장과 로딩

### 7.1 바이너리 형식

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

# 단일 배열 저장/로딩
np.save('array.npy', arr)
loaded = np.load('array.npy')

# 여러 배열 저장/로딩
np.savez('arrays.npz', arr1=arr, arr2=arr*2)
data = np.load('arrays.npz')
print(data['arr1'])
print(data['arr2'])

# 압축 저장
np.savez_compressed('arrays_compressed.npz', arr1=arr)
```

### 7.2 텍스트 형식

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

# CSV 저장
np.savetxt('array.csv', arr, delimiter=',', fmt='%d')

# CSV 로딩
loaded = np.loadtxt('array.csv', delimiter=',')

# 헤더와 함께 저장
np.savetxt('array_header.csv', arr, delimiter=',',
           header='col1,col2,col3', comments='')

# genfromtxt (결측값 처리 가능)
data = np.genfromtxt('array.csv', delimiter=',',
                     missing_values='NA', filling_values=0)
```

---

## 8. 메모리 매핑

대용량 파일을 메모리에 전부 로딩하지 않고 처리할 때 유용합니다.

```python
# 메모리 매핑된 배열 생성
shape = (10000, 10000)
dtype = np.float64

# 파일 기반 메모리 매핑
mmap = np.memmap('large_array.dat', dtype=dtype, mode='w+', shape=shape)
mmap[:100, :100] = np.random.rand(100, 100)
mmap.flush()  # 디스크에 쓰기

# 읽기 전용 로딩
mmap_read = np.memmap('large_array.dat', dtype=dtype, mode='r', shape=shape)
print(mmap_read[:10, :10])
```

---

## 연습 문제

### 문제 1: 선형 회귀
다음 데이터에 대해 최소 자승법으로 선형 회귀 계수를 구하세요.

```python
x = np.array([1, 2, 3, 4, 5])
y = np.array([2.2, 2.8, 3.6, 4.5, 5.1])

# 풀이
A = np.vstack([x, np.ones(len(x))]).T
m, c = np.linalg.lstsq(A, y, rcond=None)[0]
print(f"기울기: {m:.3f}, 절편: {c:.3f}")
```

### 문제 2: 공분산 행렬의 고유값 분해
3개의 변수를 가진 데이터의 공분산 행렬을 구하고 고유값 분해하세요.

```python
data = np.random.randn(100, 3)
data[:, 1] = data[:, 0] * 2 + np.random.randn(100) * 0.1  # 상관관계 추가

# 풀이
cov_matrix = np.cov(data.T)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print("고유값:", eigenvalues)
print("고유벡터:\n", eigenvectors)
```

### 문제 3: 몬테카를로 시뮬레이션
난수를 이용해 원의 넓이(π)를 추정하세요.

```python
# 풀이
n = 1000000
rng = np.random.default_rng(42)
x = rng.uniform(-1, 1, n)
y = rng.uniform(-1, 1, n)
inside = (x**2 + y**2) <= 1
pi_estimate = 4 * inside.sum() / n
print(f"π 추정값: {pi_estimate:.6f}")
```

---

## 요약

| 기능 | 함수/메서드 |
|------|------------|
| 행렬 곱셈 | `np.dot()`, `@`, `np.matmul()` |
| 선형대수 | `np.linalg.inv()`, `solve()`, `eig()`, `svd()` |
| 통계 | `np.mean()`, `np.std()`, `np.corrcoef()`, `np.cov()` |
| 난수 | `np.random.default_rng()`, `random()`, `normal()`, `choice()` |
| 저장/로딩 | `np.save()`, `np.load()`, `np.savetxt()`, `np.loadtxt()` |
| 성능 | 벡터화 연산, `np.where()`, 메모리 매핑 |
