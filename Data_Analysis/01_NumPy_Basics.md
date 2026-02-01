# NumPy 기초

## 개요

NumPy(Numerical Python)는 Python에서 수치 계산을 위한 핵심 라이브러리입니다. 다차원 배열 객체와 배열 연산을 위한 다양한 함수를 제공합니다.

---

## 1. NumPy 배열 생성

### 1.1 기본 배열 생성

```python
import numpy as np

# 리스트로부터 배열 생성
arr1 = np.array([1, 2, 3, 4, 5])
print(arr1)  # [1 2 3 4 5]
print(type(arr1))  # <class 'numpy.ndarray'>

# 2차원 배열
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print(arr2)
# [[1 2 3]
#  [4 5 6]]

# 3차원 배열
arr3 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(arr3.shape)  # (2, 2, 2)
```

### 1.2 특수 배열 생성

```python
# 0으로 채워진 배열
zeros = np.zeros((3, 4))
print(zeros)

# 1로 채워진 배열
ones = np.ones((2, 3))
print(ones)

# 특정 값으로 채워진 배열
full = np.full((2, 2), 7)
print(full)  # [[7 7], [7 7]]

# 단위 행렬
eye = np.eye(3)
print(eye)

# 빈 배열 (초기화되지 않은 값)
empty = np.empty((2, 3))
```

### 1.3 순차적 배열 생성

```python
# arange: 범위 지정
arr = np.arange(0, 10, 2)  # 0부터 10 미만, 2씩 증가
print(arr)  # [0 2 4 6 8]

# linspace: 등간격 분할
arr = np.linspace(0, 1, 5)  # 0부터 1까지 5개로 균등 분할
print(arr)  # [0.   0.25 0.5  0.75 1.  ]

# logspace: 로그 스케일 등간격
arr = np.logspace(0, 2, 5)  # 10^0 부터 10^2 까지
print(arr)  # [  1.    3.16  10.   31.62 100. ]
```

---

## 2. 배열 속성

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

# 차원 수
print(arr.ndim)  # 2

# 형태 (shape)
print(arr.shape)  # (2, 3)

# 전체 요소 개수
print(arr.size)  # 6

# 데이터 타입
print(arr.dtype)  # int64

# 요소당 바이트 수
print(arr.itemsize)  # 8

# 전체 바이트 수
print(arr.nbytes)  # 48
```

### 데이터 타입 지정

```python
# 정수
arr_int = np.array([1, 2, 3], dtype=np.int32)

# 실수
arr_float = np.array([1, 2, 3], dtype=np.float64)

# 복소수
arr_complex = np.array([1, 2, 3], dtype=np.complex128)

# 불리언
arr_bool = np.array([0, 1, 0, 1], dtype=np.bool_)

# 타입 변환
arr = np.array([1.5, 2.7, 3.9])
arr_int = arr.astype(np.int32)  # [1, 2, 3]
```

---

## 3. 인덱싱과 슬라이싱

### 3.1 기본 인덱싱

```python
arr = np.array([10, 20, 30, 40, 50])

# 단일 요소 접근
print(arr[0])   # 10
print(arr[-1])  # 50

# 2차원 배열
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(arr2d[0, 0])  # 1
print(arr2d[1, 2])  # 6
print(arr2d[-1, -1])  # 9
```

### 3.2 슬라이싱

```python
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# 기본 슬라이싱 [start:stop:step]
print(arr[2:7])     # [2 3 4 5 6]
print(arr[::2])     # [0 2 4 6 8]
print(arr[::-1])    # [9 8 7 6 5 4 3 2 1 0]

# 2차원 슬라이싱
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(arr2d[0:2, 1:3])
# [[2 3]
#  [5 6]]

print(arr2d[:, 0])  # 첫 번째 열: [1 4 7]
print(arr2d[1, :])  # 두 번째 행: [4 5 6]
```

### 3.3 불리언 인덱싱

```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 조건을 만족하는 요소 선택
mask = arr > 5
print(mask)  # [False False False False False  True  True  True  True  True]
print(arr[mask])  # [ 6  7  8  9 10]

# 직접 조건 사용
print(arr[arr > 5])  # [ 6  7  8  9 10]
print(arr[arr % 2 == 0])  # [ 2  4  6  8 10]

# 복합 조건
print(arr[(arr > 3) & (arr < 8)])  # [4 5 6 7]
print(arr[(arr < 3) | (arr > 8)])  # [ 1  2  9 10]
```

### 3.4 팬시 인덱싱

```python
arr = np.array([10, 20, 30, 40, 50])

# 인덱스 배열로 접근
indices = [0, 2, 4]
print(arr[indices])  # [10 30 50]

# 2차원 배열
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 특정 행 선택
print(arr2d[[0, 2]])
# [[1 2 3]
#  [7 8 9]]

# 특정 요소 선택
rows = [0, 1, 2]
cols = [0, 1, 2]
print(arr2d[rows, cols])  # [1 5 9] (대각선 요소)
```

---

## 4. 배열 형태 변환

### 4.1 reshape

```python
arr = np.arange(12)
print(arr)  # [ 0  1  2  3  4  5  6  7  8  9 10 11]

# 2차원으로 변환
arr2d = arr.reshape(3, 4)
print(arr2d)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

# -1 사용: 자동 계산
arr2d = arr.reshape(4, -1)  # 4행, 열 자동 계산
print(arr2d.shape)  # (4, 3)

arr2d = arr.reshape(-1, 6)  # 행 자동 계산, 6열
print(arr2d.shape)  # (2, 6)
```

### 4.2 flatten과 ravel

```python
arr2d = np.array([[1, 2, 3], [4, 5, 6]])

# flatten: 복사본 생성
flat = arr2d.flatten()
print(flat)  # [1 2 3 4 5 6]

# ravel: 뷰(view) 생성 (원본 공유)
rav = arr2d.ravel()
print(rav)  # [1 2 3 4 5 6]
```

### 4.3 전치 (Transpose)

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.shape)  # (2, 3)

# 전치
transposed = arr.T
print(transposed)
# [[1 4]
#  [2 5]
#  [3 6]]
print(transposed.shape)  # (3, 2)

# 다차원 전치
arr3d = np.arange(24).reshape(2, 3, 4)
print(arr3d.transpose(1, 0, 2).shape)  # (3, 2, 4)
```

### 4.4 차원 추가/제거

```python
arr = np.array([1, 2, 3])
print(arr.shape)  # (3,)

# 차원 추가
arr_2d = arr[np.newaxis, :]  # 또는 arr.reshape(1, -1)
print(arr_2d.shape)  # (1, 3)

arr_col = arr[:, np.newaxis]  # 또는 arr.reshape(-1, 1)
print(arr_col.shape)  # (3, 1)

# expand_dims 사용
arr_exp = np.expand_dims(arr, axis=0)
print(arr_exp.shape)  # (1, 3)

# squeeze: 크기 1인 차원 제거
arr = np.array([[[1, 2, 3]]])
print(arr.shape)  # (1, 1, 3)
print(np.squeeze(arr).shape)  # (3,)
```

---

## 5. 배열 연산

### 5.1 기본 산술 연산

```python
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

# 요소별 연산
print(a + b)   # [11 22 33 44]
print(a - b)   # [ -9 -18 -27 -36]
print(a * b)   # [ 10  40  90 160]
print(a / b)   # [0.1 0.1 0.1 0.1]
print(a ** 2)  # [ 1  4  9 16]
print(a % 2)   # [1 0 1 0]
print(a // 2)  # [0 1 1 2]

# 스칼라 연산
print(a + 10)  # [11 12 13 14]
print(a * 2)   # [2 4 6 8]
```

### 5.2 유니버설 함수 (ufuncs)

```python
arr = np.array([1, 4, 9, 16, 25])

# 수학 함수
print(np.sqrt(arr))   # [1. 2. 3. 4. 5.]
print(np.exp(arr))    # 지수 함수
print(np.log(arr))    # 자연로그
print(np.log10(arr))  # 상용로그

# 삼각 함수
angles = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
print(np.sin(angles))
print(np.cos(angles))
print(np.tan(angles))

# 반올림
arr = np.array([1.2, 2.5, 3.7, 4.4])
print(np.round(arr))   # [1. 2. 4. 4.]
print(np.floor(arr))   # [1. 2. 3. 4.]
print(np.ceil(arr))    # [2. 3. 4. 5.]
print(np.trunc(arr))   # [1. 2. 3. 4.]

# 절댓값
arr = np.array([-1, -2, 3, -4])
print(np.abs(arr))  # [1 2 3 4]
```

### 5.3 집계 함수

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

# 전체 집계
print(np.sum(arr))    # 21
print(np.mean(arr))   # 3.5
print(np.std(arr))    # 1.707...
print(np.var(arr))    # 2.916...
print(np.min(arr))    # 1
print(np.max(arr))    # 6
print(np.prod(arr))   # 720 (모든 요소의 곱)

# 축 기준 집계
print(np.sum(arr, axis=0))  # 열 합계: [5 7 9]
print(np.sum(arr, axis=1))  # 행 합계: [6 15]

print(np.mean(arr, axis=0))  # 열 평균: [2.5 3.5 4.5]
print(np.mean(arr, axis=1))  # 행 평균: [2. 5.]

# 누적 합/곱
print(np.cumsum(arr))  # [ 1  3  6 10 15 21]
print(np.cumprod(arr)) # [  1   2   6  24 120 720]

# 인덱스 반환
print(np.argmin(arr))  # 0 (최솟값의 인덱스)
print(np.argmax(arr))  # 5 (최댓값의 인덱스)
```

---

## 6. 브로드캐스팅

브로드캐스팅은 크기가 다른 배열 간의 연산을 가능하게 하는 NumPy의 핵심 기능입니다.

### 6.1 브로드캐스팅 규칙

1. 두 배열의 차원 수가 다르면, 작은 배열의 shape 앞에 1을 추가
2. 각 차원에서 크기가 1인 배열은 다른 배열의 크기에 맞춰 확장
3. 크기가 1이 아니고 서로 다르면 오류 발생

```python
# 스칼라와 배열
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr + 10)
# [[11 12 13]
#  [14 15 16]]

# 1차원과 2차원
arr = np.array([[1, 2, 3], [4, 5, 6]])
row = np.array([10, 20, 30])
print(arr + row)
# [[11 22 33]
#  [14 25 36]]

# 열 벡터와 2차원
col = np.array([[100], [200]])
print(arr + col)
# [[101 102 103]
#  [204 205 206]]
```

### 6.2 브로드캐스팅 예제

```python
# 표준화 (standardization)
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
mean = np.mean(data, axis=0)  # [4. 5. 6.]
std = np.std(data, axis=0)    # [2.449 2.449 2.449]
standardized = (data - mean) / std

# 거리 계산
point = np.array([1, 2])
points = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
distances = np.sqrt(np.sum((points - point) ** 2, axis=1))
print(distances)  # [2.236 1.    1.414 2.236]
```

---

## 7. 배열 결합과 분할

### 7.1 배열 결합

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# 수직 결합 (행 방향)
v_stack = np.vstack([a, b])
print(v_stack)
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]

# 수평 결합 (열 방향)
h_stack = np.hstack([a, b])
print(h_stack)
# [[1 2 5 6]
#  [3 4 7 8]]

# concatenate 사용
concat_v = np.concatenate([a, b], axis=0)  # vstack과 동일
concat_h = np.concatenate([a, b], axis=1)  # hstack과 동일

# 깊이 방향 결합
d_stack = np.dstack([a, b])
print(d_stack.shape)  # (2, 2, 2)
```

### 7.2 배열 분할

```python
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# 수직 분할
v_split = np.vsplit(arr, 3)  # 3개로 분할
print(len(v_split))  # 3

# 수평 분할
h_split = np.hsplit(arr, 2)  # 2개로 분할
print(h_split[0])
# [[ 1  2]
#  [ 5  6]
#  [ 9 10]]

# split 사용
split_arr = np.split(arr, [1, 2], axis=0)  # 인덱스 1, 2에서 분할
print(len(split_arr))  # 3
```

---

## 8. 복사와 뷰

### 8.1 뷰 (View) - 얕은 복사

```python
arr = np.array([1, 2, 3, 4, 5])

# 슬라이싱은 뷰를 생성
view = arr[1:4]
view[0] = 100

print(arr)   # [  1 100   3   4   5]  # 원본도 변경됨
print(view)  # [100   3   4]
```

### 8.2 복사 (Copy) - 깊은 복사

```python
arr = np.array([1, 2, 3, 4, 5])

# 명시적 복사
copy = arr.copy()
copy[0] = 100

print(arr)   # [1 2 3 4 5]  # 원본 유지
print(copy)  # [100 2 3 4 5]
```

---

## 연습 문제

### 문제 1: 배열 생성
1부터 100까지의 정수 중 3의 배수만 포함하는 배열을 생성하세요.

```python
# 풀이
arr = np.arange(3, 101, 3)
# 또는
arr = np.arange(1, 101)
arr = arr[arr % 3 == 0]
```

### 문제 2: 행렬 연산
3x3 단위 행렬의 대각선 요소의 합을 구하세요.

```python
# 풀이
eye = np.eye(3)
diagonal_sum = np.trace(eye)  # 3.0
# 또는
diagonal_sum = np.sum(np.diag(eye))
```

### 문제 3: 브로드캐스팅
4x4 행렬에서 각 열의 최댓값으로 정규화하세요 (각 요소를 해당 열의 최댓값으로 나누기).

```python
# 풀이
arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16]])

col_max = np.max(arr, axis=0)  # [13, 14, 15, 16]
normalized = arr / col_max
```

---

## 요약

| 기능 | 함수/메서드 |
|------|------------|
| 배열 생성 | `np.array()`, `np.zeros()`, `np.ones()`, `np.arange()`, `np.linspace()` |
| 배열 속성 | `shape`, `dtype`, `ndim`, `size` |
| 인덱싱 | `arr[i]`, `arr[i, j]`, `arr[condition]`, `arr[indices]` |
| 형태 변환 | `reshape()`, `flatten()`, `ravel()`, `T` |
| 연산 | `+`, `-`, `*`, `/`, `np.sum()`, `np.mean()`, `np.std()` |
| 결합/분할 | `np.vstack()`, `np.hstack()`, `np.split()` |
