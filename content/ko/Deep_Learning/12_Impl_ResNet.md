# 05. ResNet

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 깊은 네트워크에서의 성능 저하 문제(Degradation Problem)를 설명하고, 이것이 과적합(Overfitting)이나 기울기 소실(Vanishing Gradient)만으로 귀인될 수 없는 이유를 서술합니다.
2. 잔차 학습(Residual Learning) 프레임워크 — F(x) + x — 를 공식화하고, 참조되지 않은 매핑보다 잔차 매핑 학습이 왜 더 쉬운지 설명합니다.
3. 기본 잔차 블록(Basic Residual Block)과 병목 잔차 블록(Bottleneck Residual Block)을 구별하고, 각각이 언제 사용되는지 설명합니다.
4. PyTorch에서 잔차 블록과 스킵 연결(Skip Connection)을 올바르게 구성하여 ResNet 변형(예: ResNet-50)을 처음부터 구현합니다.
5. 블록 간에 입력과 출력 차원이 다를 때 차원 일치 스킵 연결(프로젝션 스킵 연결, Projection Shortcut)을 처리합니다.
6. 정확도, 파라미터 수, 연산 비용 측면에서 ResNet 깊이 변형(ResNet-18, 34, 50, 101, 152)의 트레이드오프를 평가합니다.

---

## 개요

ResNet(Residual Network)은 2015년 ILSVRC에서 1위를 차지한 혁명적인 모델입니다. Kaiming He 등이 제안한 **Skip Connection (Residual Connection)**을 통해 수백 개 이상의 레이어를 학습할 수 있게 되었습니다.

> "깊이가 깊어질수록 성능이 떨어지는 degradation 문제를 해결"

---

## 수학적 배경

### 1. Degradation Problem

```
문제: 네트워크가 깊어지면 오히려 성능 저하

관찰:
- 56-layer network < 20-layer network (CIFAR-10)
- 이는 overfitting이 아님 (training error도 높음)
- 최적화의 어려움 (vanishing/exploding gradient)

이상적 상황:
- 더 깊은 네트워크 ≥ 얕은 네트워크
- 최소한 identity mapping을 학습할 수 있어야 함
```

### 2. Residual Learning

```
기존 접근:
  H(x) = desired output
  네트워크가 H(x)를 직접 학습

Residual 접근:
  F(x) = H(x) - x  (잔차)
  H(x) = F(x) + x  (원래 목표)

왜 더 쉬운가?
- Identity mapping 학습: F(x) = 0만 되면 됨
- 작은 변화 학습: 큰 변화보다 쉬움
- Gradient flow: 덧셈 연산으로 직접 전파
```

### 3. Skip Connection의 Gradient

```
Forward:
  y = F(x) + x

Backward:
  ∂L/∂x = ∂L/∂y × (∂F/∂x + 1)
              ↑
            항상 1 이상!

결과:
- Gradient가 최소 1의 경로로 직접 전파
- 수백 레이어에서도 gradient 유지
- Vanishing gradient 해결
```

### 4. 차원 맞추기 (Projection Shortcut)

```
차원이 다를 때 (stride=2 또는 채널 변경):

Option A: Zero Padding
  x_padded = pad(x, extra_channels)

Option B: 1×1 Convolution (논문 채택)
  shortcut = Conv1×1(x)

  x: (N, 64, 56, 56)
  ↓ stride=2, channels 64→128
  y: (N, 128, 28, 28)

  shortcut = Conv1×1(64→128, stride=2)
```

---

## ResNet 아키텍처

### BasicBlock vs Bottleneck

```
BasicBlock (ResNet-18, 34):
┌─────────────────────────┐
│  Conv 3×3, BN, ReLU     │
│  Conv 3×3, BN           │
│         ↓               │
│    + ← shortcut         │
│       ReLU              │
└─────────────────────────┘

Bottleneck (ResNet-50, 101, 152):
┌─────────────────────────┐
│  Conv 1×1, BN, ReLU     │  ← 채널 축소
│  Conv 3×3, BN, ReLU     │  ← 주요 연산
│  Conv 1×1, BN           │  ← 채널 복원
│         ↓               │
│    + ← shortcut         │
│       ReLU              │
└─────────────────────────┘

Bottleneck 장점:
- 3×3 연산 전에 채널 축소 → 계산량 감소
- 같은 계산량으로 더 많은 레이어
```

### ResNet 변형 비교

| 모델 | 레이어 | 블록 | 블록 수 | Params |
|------|--------|------|---------|--------|
| ResNet-18 | 18 | Basic | [2,2,2,2] | 11.7M |
| ResNet-34 | 34 | Basic | [3,4,6,3] | 21.8M |
| ResNet-50 | 50 | Bottleneck | [3,4,6,3] | 25.6M |
| ResNet-101 | 101 | Bottleneck | [3,4,23,3] | 44.5M |
| ResNet-152 | 152 | Bottleneck | [3,8,36,3] | 60.2M |

### ResNet-50 상세 구조

```
입력: 224×224×3

Conv1: 7×7, 64, stride=2, padding=3
  → (112×112×64)
MaxPool: 3×3, stride=2, padding=1
  → (56×56×64)

Layer1: Bottleneck × 3 (64→256)
  → (56×56×256)

Layer2: Bottleneck × 4 (128→512, stride=2)
  → (28×28×512)

Layer3: Bottleneck × 6 (256→1024, stride=2)
  → (14×14×1024)

Layer4: Bottleneck × 3 (512→2048, stride=2)
  → (7×7×2048)

AdaptiveAvgPool: → (1×1×2048)
FC: 2048 → 1000
```

---

## 파일 구조

```
05_ResNet/
├── README.md                      # 이 파일
├── pytorch_lowlevel/
│   └── resnet_lowlevel.py        # F.conv2d, 수동 BN
├── paper/
│   └── resnet_paper.py           # 논문 정확 재현
├── analysis/
│   └── gradient_flow.py          # Skip connection 효과 분석
└── exercises/
    ├── 01_gradient_analysis.md   # Gradient flow 비교
    └── 02_ablation_study.md      # Shortcut 종류 비교
```

---

## 핵심 개념

### 1. Identity Mapping이 중요한 이유

```python
# Pre-activation ResNet (v2)
def forward(self, x):
    identity = x

    out = self.bn1(x)
    out = F.relu(out)
    out = self.conv1(out)

    out = self.bn2(out)
    out = F.relu(out)
    out = self.conv2(out)

    return out + identity  # Clean identity path

# Post-activation (original)
def forward(self, x):
    identity = self.shortcut(x)

    out = self.conv1(x)
    out = self.bn1(out)
    out = F.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    out = F.relu(out + identity)  # ReLU가 identity를 변형
    return out
```

### 2. ResNet의 앙상블 관점

```
ResNet은 다양한 깊이의 경로 앙상블로 볼 수 있음

n개 블록 → 2^n 개의 가능한 경로
- 일부 블록을 "건너뛰는" 경로
- 모든 블록을 거치는 경로

실험: 학습 후 일부 블록 제거해도 성능 유지
→ 다양한 깊이의 경로가 함께 학습됨
```

### 3. Batch Normalization의 역할

```
ResNet에서 BN이 중요한 이유:

1. 내부 공변량 변화 감소
   - 레이어 입력의 분포 안정화

2. 학습률 증가 가능
   - 더 빠른 수렴

3. Regularization 효과
   - 미니배치 통계 사용 → 노이즈

4. Gradient flow 개선
   - 정규화로 gradient 안정화
```

### 4. ResNet 이후 발전

```
ResNeXt (2017):
- Grouped convolution으로 cardinality 도입
- ResNeXt-50: ResNet-101 성능, 더 적은 파라미터

DenseNet (2017):
- 모든 레이어를 모든 후속 레이어에 연결
- Feature reuse 극대화

EfficientNet (2019):
- Width, depth, resolution 동시 스케일링
- Compound scaling

RegNet (2020):
- 최적 네트워크 구조 탐색
- 단순하고 규칙적인 설계
```

---

## 구현 레벨

### Level 2: PyTorch Low-Level (pytorch_lowlevel/)

- F.conv2d, 수동 BatchNorm
- BasicBlock, Bottleneck 수동 구현
- Shortcut projection 구현
- 파라미터 수동 관리

### Level 3: Paper Implementation (paper/)

- ResNet-18/34/50/101/152 전체
- Pre-activation ResNet (v2)
- Zero-padding vs Projection shortcut 비교

### Level 4: Code Analysis (analysis/)

- torchvision ResNet 코드 분석
- Gradient flow 시각화
- 중간 블록 제거 실험

---

## 학습 체크리스트

- [ ] Degradation problem 이해
- [ ] Residual learning 수식 유도
- [ ] Skip connection의 gradient 이점
- [ ] BasicBlock vs Bottleneck 차이
- [ ] ResNet-50 아키텍처 암기
- [ ] Projection shortcut 구현 방법
- [ ] Pre/Post-activation 차이
- [ ] ResNet의 앙상블 관점 이해

---

## 참고 자료

- He et al. (2015). "Deep Residual Learning for Image Recognition"
- He et al. (2016). "Identity Mappings in Deep Residual Networks" (v2)
- [torchvision ResNet](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
- [d2l.ai: ResNet](https://d2l.ai/chapter_convolutional-modern/resnet.html)
- [../04_VGG/README.md](../04_VGG/README.md)
