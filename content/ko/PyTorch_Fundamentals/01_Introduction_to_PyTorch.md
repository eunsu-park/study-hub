# PyTorch 소개

**다음**: [텐서](./02_Tensors.md)

---

## 학습 목표 (Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. PyTorch의 역사와 딥러닝 생태계에서의 위치를 설명할 수 있습니다
2. PyTorch가 동적 계산 그래프(define-by-run)를 사용하는 이유를 설명할 수 있습니다
3. PyTorch를 CPU와 GPU에 설치하고 설치를 검증할 수 있습니다
4. 첫 텐서를 생성하고 기본 산술 연산을 수행할 수 있습니다
5. PyTorch를 다른 프레임워크(TensorFlow, JAX)와 고수준에서 비교할 수 있습니다
6. PyTorch 문서와 커뮤니티 리소스를 탐색할 수 있습니다
7. PyTorch와 전신인 Torch(Lua) 사이의 관계를 이해할 수 있습니다

---

## 1. PyTorch란?

PyTorch는 Meta AI(이전 Facebook AI Research, FAIR)가 개발한 오픈소스 머신러닝 프레임워크입니다. 두 가지 핵심 기능을 제공합니다:

1. **N차원 텐서 연산**: 강력한 GPU 가속 지원 (NumPy와 유사하지만 GPU에서 동작)
2. **자동 미분**: 신경망 구축 및 학습을 위한 자동 미분 기능

### 1.1 주요 특징

| 특징 | 설명 |
|------|------|
| **동적 그래프** | 실행 중에 계산 그래프가 즉시 구성됨 (eager 모드) |
| **파이썬다운 API** | 네이티브 Python처럼 느껴짐 -- 표준 제어 흐름 (`if`, `for`, `while`) 사용 |
| **강력한 GPU 지원** | `.to(device)`로 CPU/GPU 간 매끄러운 텐서 이동 |
| **연구 중심** | 학술 ML 연구에서 지배적 (NeurIPS, ICML 논문의 80% 이상) |
| **프로덕션 대응** | TorchScript, ONNX 내보내기, TorchServe로 배포 가능 |

### 1.2 간략한 역사

```
2002: Torch (Lua)가 NYU의 Ronan Collobert 등에 의해 생성
2016: PyTorch 0.1이 FAIR에서 출시 -- Torch의 C 백엔드에 Python 프론트엔드
2018: PyTorch 1.0이 Caffe2 (프로덕션)와 PyTorch (연구)를 통합
2019: PyTorch가 연구 논문 채택률에서 TensorFlow를 추월
2022: PyTorch 2.0이 torch.compile()로 그래프 모드 최적화를 도입
2023: PyTorch가 Linux Foundation (PyTorch Foundation)으로 이관
2024: PyTorch 2.x가 FlexAttention, torch.export 등을 계속 발전
```

---

## 2. PyTorch vs 다른 프레임워크

### 2.1 비교표

| 측면 | PyTorch | TensorFlow | JAX |
|------|---------|------------|-----|
| **그래프 모드** | Eager (기본), compile 선택적 | Eager (TF2), graph (tf.function) | 함수형 변환 (jit, vmap) |
| **API 스타일** | 객체 지향 (nn.Module) | Keras 레이어 + tf.function | 순수 함수 + 변환 |
| **디버깅** | 표준 Python 디버거 사용 가능 | 그래프 모드에서 어려움 | 함수형 스타일 필요 |
| **연구 채택률** | 지배적 (~80%) | 연구에서 감소 추세 | 성장 중, 특히 Google에서 |
| **배포** | TorchScript, ONNX, ExecuTorch | TF Lite, TF Serving, TF.js | JAX2TF, Orbax |
| **학습 곡선** | Python 개발자에게 완만 | 보통 (Keras는 쉬움, raw TF는 어려움) | 가파름 (함수형 패러다임) |

### 2.2 PyTorch가 연구에서 우위인 이유

```python
# PyTorch: 모델에서 자연스러운 Python 제어 흐름
class DynamicNet(nn.Module):
    def forward(self, x):
        # 표준 Python if/else -- 완벽하게 작동
        if x.sum() > 0:
            return self.positive_branch(x)
        else:
            return self.negative_branch(x)
```

이것이 가능한 이유는 PyTorch가 **즉시 실행(eager execution)**을 사용하기 때문입니다 -- 연산이 일반 Python처럼 즉시 실행됩니다. 계산 그래프는 연산이 실행될 때 암묵적으로 구축되며, 이것이 "define-by-run"이라 불리는 이유입니다.

---

## 3. 설치

### 3.1 CPU 전용 설치

```bash
# pip 사용
pip install torch torchvision torchaudio

# conda 사용
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### 3.2 GPU 설치 (CUDA)

```bash
# CUDA 12.1 (먼저 NVIDIA 드라이버 버전을 확인하세요)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3.3 설치 확인

```python
import torch

# 버전 정보
print(f"PyTorch 버전: {torch.__version__}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA 버전:     {torch.version.cuda}")
    print(f"GPU 장치:      {torch.cuda.get_device_name(0)}")
    print(f"GPU 개수:      {torch.cuda.device_count()}")

# 간단한 기능 테스트
x = torch.tensor([1.0, 2.0, 3.0])
print(f"\n텐서: {x}")
print(f"합계: {x.sum()}")
print(f"장치: {x.device}")
```

---

## 4. 첫 텐서

**텐서(tensor)**는 PyTorch의 기본 데이터 구조로, 자동 미분과 GPU 가속을 지원하는 다차원 배열입니다.

### 4.1 텐서 생성

```python
import torch

# Python 리스트에서 생성
a = torch.tensor([1, 2, 3])
print(a)         # tensor([1, 2, 3])
print(a.dtype)   # torch.int64

# 2D 리스트(행렬)에서 생성
b = torch.tensor([[1.0, 2.0],
                   [3.0, 4.0]])
print(b.shape)   # torch.Size([2, 2])
print(b.dtype)   # torch.float32

# 자주 사용하는 생성 함수들
zeros = torch.zeros(3, 4)          # 0으로 채운 3x4 행렬
ones = torch.ones(2, 3)            # 1로 채운 2x3 행렬
rand = torch.rand(2, 3)            # 균일 분포 랜덤 [0, 1)
randn = torch.randn(2, 3)          # 표준 정규 분포
arange = torch.arange(0, 10, 2)    # tensor([0, 2, 4, 6, 8])
eye = torch.eye(3)                 # 3x3 단위 행렬
```

### 4.2 기본 산술 연산

```python
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

# 요소별 연산
print(x + y)       # tensor([5., 7., 9.])
print(x * y)       # tensor([ 4., 10., 18.])
print(x ** 2)      # tensor([1., 4., 9.])

# 리덕션 연산
print(x.sum())     # tensor(6.)
print(x.mean())    # tensor(2.)
print(x.max())     # tensor(3.)

# 내적
print(torch.dot(x, y))  # tensor(32.)  (1*4 + 2*5 + 3*6)
```

### 4.3 NumPy 상호 운용성

PyTorch 텐서와 NumPy 배열은 메모리를 공유하여 제로카피(zero-copy) 변환이 가능합니다:

```python
import numpy as np

# NumPy에서 PyTorch로 (메모리 공유)
np_array = np.array([1.0, 2.0, 3.0])
tensor_from_np = torch.from_numpy(np_array)

# 하나를 수정하면 다른 것도 영향 받음 (메모리 공유!)
np_array[0] = 99.0
print(tensor_from_np)  # tensor([99.,  2.,  3.])

# 독립적인 복사 (메모리 공유 안 함)
tensor_copy = torch.tensor(np_array)  # 항상 복사
```

> **주의**: NumPy와 PyTorch 간의 메모리 공유는 CPU 텐서에서만 작동합니다. GPU 텐서는 먼저 `.cpu()`로 CPU로 이동해야 합니다.

---

## 5. PyTorch의 핵심 구성 요소

```
torch
├── torch.Tensor          # 다차원 배열 (핵심 데이터 구조)
├── torch.autograd        # 자동 미분 엔진
├── torch.nn              # 신경망 레이어, 손실 함수
├── torch.optim           # 최적화 알고리즘 (SGD, Adam 등)
├── torch.utils.data      # Dataset, DataLoader 데이터 파이프라인
├── torch.cuda            # GPU 연산 및 메모리 관리
├── torch.jit             # TorchScript 모델 컴파일
├── torch.onnx            # ONNX 모델 내보내기
├── torch.distributed     # 분산 학습 유틸리티
└── torch.compile         # 그래프 모드 컴파일러 (PyTorch 2.0+)
```

---

## 6. Eager 모드 vs 그래프 모드

### 6.1 Eager 모드 (기본)

Eager 모드에서는 연산이 즉시 실행됩니다:

```python
x = torch.tensor([1.0, 2.0, 3.0])
y = x * 2        # 지금 바로 실행, 결과를 즉시 사용 가능
z = y + 1        # 지금 바로 실행
print(z)          # tensor([3., 5., 7.])
```

디버깅이 간단합니다 -- `print()`, `breakpoint()` 또는 Python 디버거를 사용할 수 있습니다.

### 6.2 그래프 모드 (torch.compile)

PyTorch 2.0에서 성능 최적화를 위해 `torch.compile()`이 도입되었습니다:

```python
@torch.compile
def optimized_fn(x):
    y = x * 2
    z = y + 1
    return z

# 첫 번째 호출은 컴파일; 이후 호출은 더 빠름
result = optimized_fn(torch.randn(1000))
```

---

## 7. Hello World: PyTorch로 선형 회귀

모든 것을 합쳐 최소한의 예제를 만들어 봅시다:

```python
import torch

# 1. 합성 데이터 생성: y = 2x + 1 + 노이즈
torch.manual_seed(42)
X = torch.rand(100, 1) * 10
y = 2 * X + 1 + torch.randn(100, 1) * 0.5

# 2. 파라미터 초기화
w = torch.randn(1, requires_grad=True)   # 가중치 (기울기)
b = torch.zeros(1, requires_grad=True)   # 편향 (절편)

# 3. 학습 루프
learning_rate = 0.01
for epoch in range(100):
    # 순전파
    y_pred = X * w + b

    # 손실 계산 (MSE)
    loss = ((y_pred - y) ** 2).mean()

    # 역전파 (그래디언트 계산)
    loss.backward()

    # 파라미터 갱신 (경사 하강법)
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    # 다음 반복을 위해 그래디언트 초기화
    w.grad.zero_()
    b.grad.zero_()

    if (epoch + 1) % 20 == 0:
        print(f"에포크 {epoch+1:3d} | 손실: {loss.item():.4f} | "
              f"w: {w.item():.4f} | b: {b.item():.4f}")

print(f"\n학습 결과: y = {w.item():.2f}x + {b.item():.2f}")
print(f"실제 값:   y = 2.00x + 1.00")
```

이 예제는 마스터할 네 가지 핵심 PyTorch 연산을 보여줍니다:
1. **텐서 생성** -- `torch.rand`, `torch.randn`
2. **자동 미분** -- `requires_grad=True`, `loss.backward()`
3. **그래디언트 기반 최적화** -- 수동 SGD (`w -= lr * w.grad`)
4. **그래디언트 관리** -- `torch.no_grad()`, `grad.zero_()`

---

## 8. PyTorch 커뮤니티와 리소스

### 8.1 공식 리소스

| 리소스 | URL |
|--------|-----|
| 문서 | https://pytorch.org/docs/stable/ |
| 튜토리얼 | https://pytorch.org/tutorials/ |
| GitHub | https://github.com/pytorch/pytorch |
| 토론 포럼 | https://discuss.pytorch.org/ |
| 블로그 | https://pytorch.org/blog/ |

---

## 요약

| 개념 | 핵심 내용 |
|------|----------|
| PyTorch | Meta AI의 오픈소스 ML 프레임워크, 연구에서 지배적 |
| 텐서 | 다차원 배열, 기본 데이터 구조 |
| Eager 모드 | 연산이 즉시 실행 (기본, 디버깅에 유리) |
| 동적 그래프 | 순전파 중 즉석에서 계산 그래프 구축 |
| NumPy 브릿지 | NumPy 배열과 CPU 텐서 간 제로카피 변환 |
| torch.compile | 성능을 위한 선택적 그래프 모드 컴파일 (PyTorch 2.0+) |

---

**다음**: [텐서](./02_Tensors.md) -- 텐서 생성, dtype, device, 메모리 레이아웃 심화 학습.
