# 텐서 연산 (Tensor Operations)

**이전**: [텐서](./02_Tensors.md) | **다음**: [자동 미분](./04_Autograd.md)

---

## 학습 목표 (Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 정수, 불리언, 팬시 인덱싱을 사용하여 텐서를 인덱싱하고 슬라이싱할 수 있습니다
2. 서로 다른 shape의 텐서를 결합할 때 브로드캐스팅 규칙을 적용할 수 있습니다
3. `@`, `torch.matmul`, `torch.mm`을 사용하여 행렬 곱셈을 수행할 수 있습니다
4. 요소별 연산(산술, 비교, 논리)을 사용할 수 있습니다
5. 특정 차원을 따라 리덕션 연산(sum, mean, max, argmax)을 적용할 수 있습니다
6. 기존 또는 새로운 차원을 따라 텐서를 연결하고 쌓을 수 있습니다
7. `torch.where`, `torch.clamp` 등의 조건부 연산을 사용할 수 있습니다
8. 인플레이스 연산과 그 명명 규칙을 이해할 수 있습니다

---

텐서 연산은 모든 신경망 계산의 기본 구성 요소입니다. 이 레슨에서는 기본 인덱싱부터 행렬 대수까지, PyTorch 작업에서 매일 사용할 연산을 다룹니다.

---

## 1. 인덱싱과 슬라이싱

### 1.1 기본 인덱싱

```python
import torch

x = torch.tensor([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# 단일 원소
print(x[0, 0])     # tensor(1)
print(x[1, 2])     # tensor(6)

# 행과 열
print(x[0])         # tensor([1, 2, 3])  -- 첫 번째 행
print(x[:, 0])      # tensor([1, 4, 7])  -- 첫 번째 열
```

### 1.2 슬라이싱

```python
x = torch.arange(20).view(4, 5)

# 행 슬라이스
print(x[1:3])       # 1-2번 행

# 행과 열 슬라이스
print(x[1:3, 2:4])  # 1-2번 행, 2-3번 열

# 단계 슬라이싱
print(x[::2])       # 짝수 행마다
```

> **참고**: 슬라이스는 **뷰**를 반환합니다 -- 원본 텐서와 메모리를 공유합니다.

### 1.3 불리언 (마스크) 인덱싱

```python
x = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0])

mask = x > 0
print(mask)        # tensor([ True, False,  True, False,  True])
print(x[mask])     # tensor([1., 3., 5.])

# 마스크를 이용한 인플레이스 수정
x[x < 0] = 0
print(x)           # tensor([1., 0., 3., 0., 5.])
```

### 1.4 팬시 (고급) 인덱싱

```python
x = torch.tensor([10, 20, 30, 40, 50])

# 인덱스 리스트로 인덱싱
idx = torch.tensor([0, 2, 4])
print(x[idx])      # tensor([10, 30, 50])

# gather: 배치별 원소 선택
src = torch.tensor([[1, 2], [3, 4], [5, 6]])
idx = torch.tensor([[0], [1], [0]])
print(torch.gather(src, 1, idx))  # tensor([[1], [4], [5]])
```

---

## 2. 요소별 연산

### 2.1 산술 연산

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(a + b)    # tensor([5., 7., 9.])
print(a * b)    # tensor([ 4., 10., 18.])
print(a ** 2)   # tensor([1., 4., 9.])

# 스칼라 연산
print(a + 10)   # tensor([11., 12., 13.])
```

### 2.2 수학 함수

```python
x = torch.tensor([0.0, 1.0, 2.0, 3.0])

print(torch.exp(x))     # 지수 함수
print(torch.log(x + 1)) # 자연 로그
print(torch.sqrt(x))    # 제곱근

# 값을 범위로 클램핑
y = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(torch.clamp(y, min=-1.0, max=1.0))  # tensor([-1., -1.,  0.,  1.,  1.])
```

### 2.3 비교 및 논리 연산

```python
a = torch.tensor([1, 2, 3, 4])
b = torch.tensor([2, 2, 2, 2])

print(a > b)    # tensor([False, False,  True,  True])
print(a == b)   # tensor([False,  True, False, False])

# torch.where: 조건부 선택
cond = torch.tensor([True, False, True, False])
a = torch.tensor([1.0, 2.0, 3.0, 4.0])
b = torch.tensor([10.0, 20.0, 30.0, 40.0])
print(torch.where(cond, a, b))  # tensor([ 1., 20.,  3., 40.])
```

---

## 3. 리덕션 연산

### 3.1 기본 리덕션

```python
x = torch.tensor([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0]])

print(x.sum())     # tensor(21.)
print(x.mean())    # tensor(3.5000)
print(x.max())     # tensor(6.)
```

### 3.2 차원별 리덕션

```python
x = torch.tensor([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0]])

# dim=0을 따라 (행 축소, 결과 shape [3])
print(x.sum(dim=0))   # tensor([5., 7., 9.])

# dim=1을 따라 (열 축소, 결과 shape [2])
print(x.sum(dim=1))   # tensor([ 6., 15.])

# 차원 유지 (브로드캐스팅에 유용)
print(x.sum(dim=1, keepdim=True))
# tensor([[ 6.],
#         [15.]])
```

### 3.3 argmax와 argmin

```python
x = torch.tensor([[3, 1, 4],
                   [1, 5, 9],
                   [2, 6, 5]])

print(x.argmax(dim=0)) # tensor([0, 2, 1])  -- 열별 최대값 인덱스
print(x.argmax(dim=1)) # tensor([2, 2, 1])  -- 행별 최대값 인덱스

# max는 값과 인덱스 모두 반환
values, indices = x.max(dim=1)
print(values)   # tensor([4, 9, 6])
print(indices)  # tensor([2, 2, 1])
```

---

## 4. 브로드캐스팅

브로드캐스팅은 서로 다른 shape의 텐서 간 연산을 가능하게 합니다.

### 4.1 브로드캐스팅 규칙

두 텐서가 브로드캐스트 가능한 조건 (후행 차원부터):
1. 차원이 같거나,
2. 한쪽 차원이 1이거나,
3. 한 텐서의 차원 수가 적은 경우 (앞에 1이 추가됨)

```python
# 예시: 벡터 브로드캐스트
x = torch.randn(2, 3)            # shape [2, 3]
y = torch.randn(3)               # shape [3] -> [2, 3]으로 브로드캐스트
print((x + y).shape)             # [2, 3]

# 예시: 정규화 패턴
x = torch.randn(4, 5)
mean = x.mean(dim=1, keepdim=True)  # shape [4, 1]
std = x.std(dim=1, keepdim=True)    # shape [4, 1]
x_normalized = (x - mean) / std     # 브로드캐스팅: [4, 5] - [4, 1]
```

### 4.2 브로드캐스팅 주의점

```python
# 이 shape들은 브로드캐스트 불가:
# [2, 3]과 [2]  -- 후행 차원 3과 2가 불일치

a = torch.randn(2, 3)
b = torch.randn(2)
# a + b  # 에러

# 수정: 차원 추가
result = a + b.unsqueeze(1)  # [2, 3] + [2, 1] -> [2, 3]
```

---

## 5. 행렬 연산

### 5.1 행렬 곱셈

```python
A = torch.randn(2, 3)
B = torch.randn(3, 4)

# 세 가지 동등한 방법
C = A @ B                    # 연산자
C = torch.matmul(A, B)      # 함수 (가장 범용)
C = torch.mm(A, B)          # 2D만
print(C.shape)               # [2, 4]

# 배치 행렬 곱셈
batch_A = torch.randn(8, 2, 3)
batch_B = torch.randn(8, 3, 4)
batch_C = torch.bmm(batch_A, batch_B)   # [8, 2, 4]
```

### 5.2 내적과 외적

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(torch.dot(a, b))         # tensor(32.)  (1*4 + 2*5 + 3*6)
print(torch.outer(a, b))       # 3x3 외적 행렬
```

### 5.3 선형 대수

```python
A = torch.randn(3, 3)

print(torch.linalg.det(A))         # 행렬식
A_inv = torch.linalg.inv(A)        # 역행렬
eigenvalues, eigenvectors = torch.linalg.eig(A)  # 고유값 분해
U, S, Vh = torch.linalg.svd(A)     # 특이값 분해
```

---

## 6. 연결과 쌓기

### 6.1 torch.cat (연결)

```python
a = torch.randn(2, 3)
b = torch.randn(2, 3)

cat0 = torch.cat([a, b], dim=0)  # shape: [4, 3]  (세로 연결)
cat1 = torch.cat([a, b], dim=1)  # shape: [2, 6]  (가로 연결)
```

### 6.2 torch.stack (새 차원)

```python
a = torch.randn(3, 4)
b = torch.randn(3, 4)

stacked = torch.stack([a, b], dim=0)  # shape: [2, 3, 4]

# 일반적 사용: 개별 샘플에서 배치 생성
samples = [torch.randn(28, 28) for _ in range(16)]
batch = torch.stack(samples)  # [16, 28, 28]
```

---

## 7. 인플레이스 연산

인플레이스 연산은 새 텐서를 생성하지 않고 직접 수정합니다. 언더스코어 접미사로 표시됩니다:

```python
x = torch.tensor([1.0, 2.0, 3.0])

x.add_(10)       # x = tensor([11., 12., 13.])
x.mul_(2)        # x = tensor([22., 24., 26.])
x.zero_()        # x = tensor([0., 0., 0.])
```

> **주의**: `requires_grad=True`인 텐서에 대한 인플레이스 연산은 역전파에 필요한 계산 그래프를 손상시킬 수 있으므로 일반적으로 허용되지 않습니다.

---

## 8. 실용 패턴

### 8.1 원-핫 인코딩

```python
import torch.nn.functional as F

labels = torch.tensor([0, 2, 1, 3])
one_hot = F.one_hot(labels, num_classes=4).float()
```

### 8.2 마스킹과 패딩

```python
# 가변 길이 시퀀스를 위한 패딩 마스크 생성
lengths = torch.tensor([3, 5, 2])
max_len = 5
mask = torch.arange(max_len).expand(3, -1) < lengths.unsqueeze(1)

# 마스크 적용: 패딩 위치를 -inf로 설정 (어텐션용)
scores = torch.randn(3, 5)
scores = scores.masked_fill(~mask, float('-inf'))
```

### 8.3 아인슈타인 합 (Einsum)

```python
# 행렬 곱셈: C_ij = sum_k A_ik * B_kj
A = torch.randn(2, 3)
B = torch.randn(3, 4)
C = torch.einsum('ik,kj->ij', A, B)  # A @ B와 동일

# 배치 행렬 곱셈
A = torch.randn(8, 2, 3)
B = torch.randn(8, 3, 4)
C = torch.einsum('bij,bjk->bik', A, B)
```

---

## 요약

| 범주 | 주요 연산 |
|------|----------|
| 인덱싱 | `x[0]`, `x[:, 1]`, `x[mask]`, `x[indices]` |
| 요소별 | `+`, `-`, `*`, `/`, `**`, `torch.exp`, `torch.clamp` |
| 리덕션 | `.sum()`, `.mean()`, `.max()`, `.argmax()` + `dim=`, `keepdim=` |
| 브로드캐스팅 | 후행 차원이 일치하거나 1이어야 함; `unsqueeze`로 정렬 |
| 행렬 연산 | `@`, `torch.matmul`, `torch.mm`, `torch.bmm`, `torch.linalg.*` |
| 연결 | `torch.cat` (기존 차원), `torch.stack` (새 차원) |
| 인플레이스 | 언더스코어 접미사 (`add_`, `mul_`, `zero_`); `requires_grad`와 함께 사용 주의 |

---

**다음**: [자동 미분](./04_Autograd.md) -- 자동 미분, 계산 그래프, 그래디언트 계산.
