# Model Implementations 학습 가이드

## 개요

이 폴더는 딥러닝 모델을 **from-scratch 구현**하며 학습하는 자료입니다. 단순히 라이브러리를 사용하는 것을 넘어, 모델의 내부 동작을 깊이 이해하는 것이 목표입니다.

### 학습 철학
> "구현하지 못하면 이해하지 못한 것이다." - Richard Feynman

### 4단계 구현 레벨
| Level | 이름 | 설명 | 목적 |
|-------|------|------|------|
| L1 | **NumPy Scratch** | 행렬 연산으로 직접 구현 | 수학적 원리 이해 |
| L2 | **PyTorch Low-level** | 기본 ops만 사용 | Framework 이해 |
| L3 | **Paper Implementation** | 논문 재현 | 연구 능력 |
| L4 | **Code Analysis** | 프로덕션 코드 분석 | 실무 적용 |

---

## 학습 순서 (12개 모델)

### Tier 1: 기초 (Week 1-2)
| # | 모델 | 핵심 개념 | L1 | L2 | L3 | L4 |
|---|------|----------|----|----|----|----|
| 01 | [Linear/Logistic](01_Linear_Logistic/) | Gradient Descent, Loss | ✅ | ✅ | ✅ | - |
| 02 | [MLP](02_MLP/) | Backpropagation, Activations | ✅ | ✅ | ✅ | - |
| 03 | [CNN (LeNet)](03_CNN_LeNet/) | Convolution, Pooling | ✅ | ✅ | ✅ | - |

### Tier 2: Classic Deep Learning (Week 3-5)
| # | 모델 | 핵심 개념 | L1 | L2 | L3 | L4 |
|---|------|----------|----|----|----|----|
| 04 | [VGG](04_VGG/) | Deep Stacking, Features | - | ✅ | ✅ | - |
| 05 | [ResNet](05_ResNet/) | Skip Connections, Residual | - | ✅ | ✅ | ✅ |
| 06 | [LSTM/GRU](06_LSTM_GRU/) | Gating, BPTT | ✅ | ✅ | ✅ | - |

### Tier 3: Transformer 기반 (Week 6-8)
| # | 모델 | 핵심 개념 | L1 | L2 | L3 | L4 |
|---|------|----------|----|----|----|----|
| 07 | [Transformer](07_Transformer/) | Self-Attention, PE | - | ✅ | ✅ | - |
| 08 | [BERT](08_BERT/) | Masked LM, Bidirectional | - | ✅ | ✅ | ✅ |
| 09 | [GPT](09_GPT/) | Autoregressive, Causal | - | ✅ | ✅ | ✅ |
| 10 | [ViT](10_ViT/) | Patch Embedding, CLS | - | ✅ | ✅ | ✅ |

### Tier 4: 생성/멀티모달 (Week 9-10)
| # | 모델 | 핵심 개념 | L1 | L2 | L3 | L4 |
|---|------|----------|----|----|----|----|
| 11 | [VAE](11_VAE/) | ELBO, Reparameterization | ✅ | ✅ | ✅ | - |
| 12 | [CLIP](12_CLIP/) | Contrastive, Zero-shot | - | ✅ | ✅ | ✅ |

---

## 학습 의존성 그래프

```
┌─────────────────────────────────────────────────────────────────┐
│                    모델 의존성 및 학습 순서                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  01_Linear ─┬─► 02_MLP ─┬─► 03_CNN ──► 04_VGG ──► 05_ResNet    │
│             │           │                                       │
│             │           └─► 06_LSTM                             │
│             │                                                   │
│             └─► 07_Transformer ─┬─► 08_BERT                     │
│                                 │                               │
│                                 ├─► 09_GPT                      │
│                                 │                               │
│                                 └─► 10_ViT ──► 12_CLIP          │
│                                                                 │
│  02_MLP ──────────────────────────► 11_VAE                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

화살표: 선수 지식 관계
```

---

## 폴더 구조

```
Model_Implementations/
├── 00_Overview.md              # 이 파일
├── docs/
│   ├── 01_Implementation_Philosophy.md    # 구현 학습의 가치
│   ├── 02_NumPy_vs_PyTorch_Comparison.md  # 프레임워크 비교
│   ├── 03_Reading_HuggingFace_Code.md     # HF 코드 읽기 가이드
│   └── 04_Reading_timm_Code.md            # timm 코드 읽기 가이드
│
├── 01_Linear_Logistic/
│   ├── README.md               # 모델 개요 및 수학
│   ├── theory.md               # 이론 상세 설명
│   ├── numpy/
│   │   ├── linear_numpy.py     # NumPy 구현
│   │   ├── logistic_numpy.py
│   │   └── test_numpy.py       # 단위 테스트
│   ├── pytorch_lowlevel/
│   │   └── linear_lowlevel.py  # PyTorch 기본 ops
│   ├── paper/
│   │   └── linear_paper.py     # 클린 구현
│   └── exercises/
│       └── 01_regularization.md
│
├── 02_MLP/
│   ├── ... (동일 구조)
│
└── utils/
    ├── data_loaders.py         # 공통 데이터 로딩
    ├── visualization.py        # 시각화 유틸
    └── training_utils.py       # 학습 헬퍼
```

---

## 각 레벨 설명

### Level 1: NumPy From-Scratch

**목표**: 프레임워크 없이 순수 행렬 연산으로 구현

```python
# 예시: Linear Regression
import numpy as np

class LinearRegression:
    def __init__(self, input_dim):
        # Xavier 초기화
        self.W = np.random.randn(input_dim, 1) * np.sqrt(2/input_dim)
        self.b = np.zeros(1)

    def forward(self, X):
        """y = XW + b"""
        self.X = X  # backward를 위해 캐시
        return X @ self.W + self.b

    def backward(self, y, y_pred, lr=0.01):
        """수동 gradient 계산"""
        m = y.shape[0]
        error = y_pred - y

        # dL/dW = X^T @ error / m
        dW = self.X.T @ error / m
        db = np.mean(error)

        # 가중치 업데이트
        self.W -= lr * dW
        self.b -= lr * db
```

**적용 모델**: 01-03, 06, 11

### Level 2: PyTorch Low-Level

**목표**: PyTorch 기본 연산만 사용, nn.Module 최소화

```python
# 예시: MLP (nn.Linear 없이)
import torch
import torch.nn.functional as F

class MLPLowLevel:
    def __init__(self, input_dim, hidden_dim, output_dim):
        # 수동 파라미터 관리
        self.W1 = torch.randn(input_dim, hidden_dim, requires_grad=True) * 0.01
        self.b1 = torch.zeros(hidden_dim, requires_grad=True)
        self.W2 = torch.randn(hidden_dim, output_dim, requires_grad=True) * 0.01
        self.b2 = torch.zeros(output_dim, requires_grad=True)

    def forward(self, x):
        # F.relu 사용, nn.ReLU 미사용
        h = F.relu(x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def parameters(self):
        return [self.W1, self.b1, self.W2, self.b2]

# 수동 SGD
def sgd_step(params, lr):
    with torch.no_grad():
        for p in params:
            p -= lr * p.grad
            p.grad.zero_()
```

**적용 모델**: 01-12 (전체)

### Level 3: Paper Implementation

**목표**: 논문의 아키텍처를 정확하게 재현

```python
# 예시: ResNet BasicBlock
"""
Paper: "Deep Residual Learning for Image Recognition"
       He et al., 2015
"""
import torch.nn as nn

class BasicBlock(nn.Module):
    """
    논문 Section 3.1의 기본 잔차 블록

    구조: conv-bn-relu-conv-bn + shortcut
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        # 첫 번째 3x3 conv (stride로 다운샘플링)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # 두 번째 3x3 conv (항상 stride=1)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Shortcut: 차원이 다르면 1x1 conv로 맞춤
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        # F(x) = conv-bn-relu-conv-bn
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # H(x) = F(x) + x
        out += self.shortcut(x)
        return F.relu(out)
```

**적용 모델**: 01-12 (전체)

### Level 4: Code Analysis

**목표**: 프로덕션 코드를 읽고, 이해하고, 수정하기

```markdown
# HuggingFace BERT 분석 예시

## 파일 구조
transformers/models/bert/
├── configuration_bert.py    # BertConfig
├── modeling_bert.py         # 메인 모델
└── tokenization_bert.py     # 토크나이저

## 핵심 클래스 분석

### BertModel (modeling_bert.py:800)
```python
class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
```

### 수정 과제
1. 커스텀 pooling 추가
2. Attention 패턴 시각화
3. 새로운 task head 구현
```

**적용 모델**: 05, 08-10, 12

---

## 실습 환경

### 필수 패키지
```bash
pip install numpy torch torchvision matplotlib
pip install transformers timm  # L4용
```

### 권장 하드웨어
| 레벨 | 최소 | 권장 |
|------|------|------|
| L1 NumPy | CPU만 | CPU |
| L2-L3 (작은 모델) | CPU/GPU 4GB | GPU 8GB |
| L3 (대형 모델) | GPU 8GB | GPU 16GB |
| L4 | GPU 8GB | GPU 16GB |

### 데이터셋
- **MNIST**: MLP, CNN 테스트
- **CIFAR-10**: VGG, ResNet
- **IMDB**: LSTM, BERT
- **ImageNet (subset)**: ViT, CLIP

---

## 학습 팁

### 1. 단계적 접근
```
NumPy (원리) → PyTorch Low (Framework) → Paper (완성도) → Analysis (실무)
```

### 2. 테스트 중요성
- 각 레이어/모듈 단위 테스트
- 작은 입력으로 shape 검증
- 기존 구현과 출력 비교

### 3. 디버깅 전략
```python
# Shape 출력
print(f"Input: {x.shape}")
print(f"After conv1: {self.conv1(x).shape}")

# Gradient 확인
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad mean={param.grad.mean():.6f}")
```

### 4. 시각화 활용
- Loss 곡선
- Attention 패턴
- Feature map
- Gradient flow

---

## 선수 지식

### 필수
- Python 프로그래밍
- 선형대수 (행렬 연산, 고유값)
- 미적분 (편미분, 연쇄 법칙)
- 기초 확률/통계

### 권장
- Deep_Learning 폴더 01-10 완료
- NumPy, PyTorch 기본 사용법

---

## 참고 자료

### 도서
- "Deep Learning" (Goodfellow et al.) - 이론
- "Dive into Deep Learning" - 구현 중심
- "Neural Networks from Scratch" - NumPy 구현

### 온라인
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [minGPT](https://github.com/karpathy/minGPT)
- [labml.ai Annotated Papers](https://nn.labml.ai/)

### 논문 (필독)
| 모델 | 논문 |
|------|------|
| ResNet | He et al., 2015 |
| Transformer | Vaswani et al., 2017 |
| BERT | Devlin et al., 2018 |
| GPT | Radford et al., 2018 |
| ViT | Dosovitskiy et al., 2020 |
| CLIP | Radford et al., 2021 |

---

## 다음 단계

이 폴더를 완료한 후:
- **Foundation_Models**: 최신 대형 모델 이해
- **Deep_Learning 예제**: 실전 응용
- **MLOps**: 모델 배포
