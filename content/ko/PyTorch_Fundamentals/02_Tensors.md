# 텐서 (Tensors)

**이전**: [PyTorch 소개](./01_Introduction_to_PyTorch.md) | **다음**: [텐서 연산](./03_Tensor_Operations.md)

---

## 학습 목표 (Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 다양한 방법(데이터, NumPy, 팩토리 함수)으로 텐서를 생성할 수 있습니다
2. 텐서 속성인 shape, dtype, device, layout을 설명할 수 있습니다
3. 다양한 사용 사례에 적합한 dtype을 선택할 수 있습니다 (float32 vs float16 vs int64)
4. `.to()`로 텐서를 CPU와 GPU 간에 이동하고 성능 영향을 이해할 수 있습니다
5. 뷰(view)와 복사(copy)를 구분하고, PyTorch가 메모리를 공유하는 시점을 예측할 수 있습니다
6. `view()`, `reshape()`, `unsqueeze()`, `squeeze()`, `permute()`로 텐서를 재구성할 수 있습니다
7. 텐서 메모리 레이아웃(연속 vs 비연속)과 성능에 미치는 영향을 이해할 수 있습니다
8. PyTorch 텐서와 다른 형식(NumPy, Python 리스트, PIL 이미지) 간의 변환을 수행할 수 있습니다

---

텐서는 PyTorch의 기본 데이터 구조입니다. 모든 데이터(입력, 가중치, 그래디언트, 출력)는 텐서로 흐릅니다. 상위 수준의 PyTorch API를 사용하기 전에 텐서 생성, 조작, 메모리 동작을 마스터하는 것이 필수적입니다.

---

## 1. 텐서 생성

### 1.1 데이터에서 생성

```python
import torch

# Python 리스트에서
t1 = torch.tensor([1, 2, 3])
print(t1)        # tensor([1, 2, 3])
print(t1.dtype)  # torch.int64  (정수 데이터에서 추론)

# 중첩 리스트 (2D)
t2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(t2.shape)  # torch.Size([2, 2])
print(t2.dtype)  # torch.float32  (실수 데이터에서 추론)

# 명시적 dtype
t3 = torch.tensor([1, 2, 3], dtype=torch.float32)
print(t3.dtype)  # torch.float32
```

### 1.2 NumPy에서 생성

```python
import numpy as np

np_array = np.array([1.0, 2.0, 3.0])

# 메모리 공유 (제로카피)
t_shared = torch.from_numpy(np_array)
np_array[0] = 99.0
print(t_shared[0])  # tensor(99.)  -- 메모리가 공유됨!

# 독립적 복사
t_copy = torch.tensor(np_array)  # 항상 데이터를 복사
np_array[0] = -1.0
print(t_copy[0])    # tensor(99.)  -- 영향 받지 않음
```

### 1.3 팩토리 함수

```python
# 0과 1로 채우기
z = torch.zeros(3, 4)        # 0으로 채운 3x4 행렬
o = torch.ones(2, 3, 5)      # 1로 채운 2x3x5 텐서

# 랜덤 텐서
u = torch.rand(3, 4)         # 균일 분포 [0, 1)
n = torch.randn(3, 4)        # 표준 정규 분포 N(0, 1)
ri = torch.randint(0, 10, (3, 4))  # [0, 10) 정수 랜덤

# 수열
a = torch.arange(0, 10, 2)   # tensor([0, 2, 4, 6, 8])
l = torch.linspace(0, 1, 5)  # tensor([0.0, 0.25, 0.5, 0.75, 1.0])

# 특수 행렬
e = torch.eye(3)              # 3x3 단위 행렬

# 다른 텐서와 같은 형태 (같은 shape, dtype, device)
x = torch.randn(3, 4, device='cpu', dtype=torch.float32)
y = torch.zeros_like(x)      # x와 같은 shape, dtype, device
```

### 1.4 시드를 이용한 재현성

```python
torch.manual_seed(42)
a = torch.randn(3)

torch.manual_seed(42)
b = torch.randn(3)

print(torch.equal(a, b))  # True -- 같은 시드는 같은 값을 생성
```

---

## 2. 텐서 속성

### 2.1 Shape (크기)

```python
t = torch.randn(2, 3, 4)

print(t.shape)      # torch.Size([2, 3, 4])
print(t.size())     # torch.Size([2, 3, 4])  -- 같은 것
print(t.ndim)       # 3  (차원 수)
print(t.numel())    # 24 (전체 원소 수: 2*3*4)
```

### 2.2 데이터 타입 (dtype)

```python
t_f32 = torch.tensor([1.0])         # torch.float32 (실수 기본값)
t_f64 = torch.tensor([1.0]).double() # torch.float64
t_f16 = torch.tensor([1.0]).half()   # torch.float16
t_i64 = torch.tensor([1])           # torch.int64 (정수 기본값)
t_bool = torch.tensor([True, False]) # torch.bool

# 타입 변환
x = torch.tensor([1, 2, 3])       # int64
x_float = x.float()               # float32
x_half = x.half()                 # float16
x_double = x.to(torch.float64)    # float64
```

**언제 어떤 dtype을 사용할까:**

| dtype | 비트 | 사용 사례 |
|-------|------|----------|
| `float32` | 32 | 모델 파라미터 및 학습의 기본값 |
| `float16` | 16 | 혼합 정밀도 학습 (AMP), 추론 |
| `bfloat16` | 16 | 신형 GPU에서의 혼합 정밀도 (fp16보다 범위가 넓음) |
| `float64` | 64 | 과학 계산, 손실 계산 (거의 필요 없음) |
| `int64` | 64 | 인덱스, 레이블, 토큰 ID |
| `bool` | 8 | 마스크, 조건 |

### 2.3 장치 (Device)

```python
# CPU 텐서 (기본값)
cpu_tensor = torch.tensor([1.0, 2.0])
print(cpu_tensor.device)  # cpu

# GPU 텐서
if torch.cuda.is_available():
    gpu_tensor = torch.tensor([1.0, 2.0], device='cuda')
    print(gpu_tensor.device)  # cuda:0

    # CPU 텐서를 GPU로 이동
    moved = cpu_tensor.to('cuda')

    # GPU 텐서를 CPU로 이동
    back = gpu_tensor.to('cpu')

# 장치 불가지론 코드 패턴
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.randn(3, 4, device=device)
```

> **중요**: 서로 다른 장치에 있는 텐서 간의 연산은 에러를 발생시킵니다. 텐서를 결합하기 전에 항상 같은 장치에 있는지 확인하세요.

---

## 3. 뷰 vs 복사

PyTorch가 메모리를 공유하는 시점을 이해하는 것은 정확성과 성능 모두에 중요합니다.

### 3.1 뷰 (메모리 공유)

**뷰(view)**는 같은 데이터를 다른 방식으로 보는 것입니다 -- 데이터가 복사되지 않습니다:

```python
x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

# view()는 뷰를 생성
y = x.view(2, 3)

# y를 수정하면 x도 수정됨 (메모리 공유!)
y[0, 0] = 99.0
print(x[0])  # tensor(99.)
```

### 3.2 복사 (독립 메모리)

```python
x = torch.tensor([1.0, 2.0, 3.0])

# .clone()은 복사를 생성
y = x.clone()
y[0] = 99.0
print(x[0])  # tensor(1.)  -- x는 영향 받지 않음
```

---

## 4. 재구성 연산

### 4.1 view() vs reshape()

```python
x = torch.arange(12)  # tensor([0, 1, 2, ..., 11])

# view()는 연속 메모리가 필요
y = x.view(3, 4)     # OK: x는 연속

# reshape()는 비연속 텐서에도 작동
t = torch.randn(3, 4).T  # 비연속
r = t.reshape(12)         # OK: 필요 시 복사

# -1로 한 차원을 추론
z = x.view(3, -1)   # shape: [3, 4]  (-1이 12/3=4로 추론)
```

### 4.2 squeeze()와 unsqueeze()

```python
x = torch.randn(1, 3, 1, 4)

# squeeze: 크기 1인 차원을 제거
y = x.squeeze()       # shape: [3, 4]
y = x.squeeze(0)      # shape: [3, 1, 4]  (0번 차원만)

# unsqueeze: 크기 1인 차원을 추가
z = torch.randn(3, 4)
z = z.unsqueeze(0)    # shape: [1, 3, 4]  (배치 차원 추가)
```

### 4.3 permute()와 transpose()

```python
x = torch.randn(2, 3, 4)  # [배치, 높이, 너비]

# transpose: 두 차원을 교환
y = x.transpose(1, 2)     # [2, 4, 3]

# permute: 모든 차원을 재배열
z = x.permute(2, 0, 1)    # [4, 2, 3]

# .T는 2D 텐서에서만
m = torch.randn(3, 4)
print(m.T.shape)           # [4, 3]
```

### 4.4 flatten()과 unflatten()

```python
x = torch.randn(2, 3, 4)

# 모든 차원 펼치기
flat = x.flatten()          # shape: [24]

# 특정 범위만 펼치기
flat_partial = x.flatten(1)  # shape: [2, 12]
```

---

## 5. 메모리 레이아웃과 연속성

### 5.1 스트라이드

PyTorch는 **스트라이드(stride)**를 사용하여 다차원 인덱스를 평탄 메모리에 매핑합니다:

```python
x = torch.tensor([[1, 2, 3],
                   [4, 5, 6]])

print(x.stride())  # (3, 1)
# stride[0]=3: 한 행 이동은 3개 원소 건너뜀
# stride[1]=1: 한 열 이동은 1개 원소 건너뜀

# 전치 후 스트라이드가 변하지만 데이터는 이동하지 않음
y = x.T
print(y.stride())  # (1, 3)
print(y.is_contiguous())  # False
```

### 5.2 연속성이 중요한 이유

```python
x = torch.randn(3, 4)
y = x.T  # 비연속

# 일부 연산은 연속 텐서가 필요
# y.view(12)  # 에러!

# 해결 1: reshape 사용 (필요 시 복사)
flat = y.reshape(12)

# 해결 2: 명시적으로 연속으로 만들기
y_c = y.contiguous()
flat = y_c.view(12)
```

---

## 6. 타입 변환과 형식 변환

### 6.1 dtype 간 변환

```python
x = torch.tensor([1, 2, 3])  # int64

# 방법 1: .to()
x_f32 = x.to(torch.float32)

# 방법 2: 편의 메서드
x_f32 = x.float()
x_f64 = x.double()
x_f16 = x.half()
```

### 6.2 형식 간 변환

```python
import numpy as np

# 텐서 -> NumPy
t = torch.tensor([1.0, 2.0, 3.0])
n = t.numpy()                     # 메모리 공유 (CPU만)
n = t.detach().cpu().numpy()       # 모든 텐서에 안전

# 텐서 -> Python 스칼라
s = torch.tensor(3.14)
print(s.item())      # 3.14 (Python float)

# 텐서 -> Python 리스트
t = torch.tensor([[1, 2], [3, 4]])
print(t.tolist())    # [[1, 2], [3, 4]]
```

---

## 7. 일반적인 함정

### 7.1 인플레이스 연산

```python
x = torch.randn(3, requires_grad=True)

# 인플레이스 연산은 언더스코어 접미사로 표시
# x.add_(1)  # 에러: requires_grad=True일 때 (리프 변수 수정)

# 안전: 아웃-오브-플레이스
y = x + 1  # 새 텐서 생성

# 그래디언트가 필요 없는 텐서에서는 인플레이스 가능
z = torch.randn(3)
z.add_(1)    # OK
```

### 7.2 장치 불일치

```python
if torch.cuda.is_available():
    cpu_t = torch.tensor([1.0])
    gpu_t = torch.tensor([2.0], device='cuda')

    # cpu_t + gpu_t  # 에러: 같은 장치를 기대

    # 수정: 같은 장치로 이동
    result = cpu_t.to('cuda') + gpu_t
```

---

## 요약

| 개념 | 핵심 내용 |
|------|----------|
| 생성 | `torch.tensor()`는 복사; `torch.from_numpy()`는 메모리 공유 |
| Shape | `.shape`, `.ndim`, `.numel()`로 차원 확인 |
| dtype | `float32`가 실수 기본값; 효율을 위해 `float16`/`bfloat16` 사용 |
| Device | 상호작용하는 텐서는 항상 같은 장치에 유지 |
| 뷰 | `view()`, 슬라이싱, `.T`는 메모리 공유 -- 수정이 전파됨 |
| 복사 | `.clone()`, `torch.tensor()`는 독립 복사본 생성 |
| 재구성 | `view()`는 연속 필요; `reshape()`는 둘 다 처리 |
| 연속성 | 전치된 텐서는 비연속; 필요 시 `.contiguous()` 사용 |

---

**다음**: [텐서 연산](./03_Tensor_Operations.md) -- 인덱싱, 슬라이싱, 브로드캐스팅, 행렬 연산.
