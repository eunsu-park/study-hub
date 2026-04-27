# 12. ResNet

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 깊은 네트워크에서의 성능 저하 문제(Degradation Problem)를 설명하고, 이것이 과적합(Overfitting)이나 기울기 소실(Vanishing Gradient)만으로 귀인될 수 없는 이유를 서술합니다.
2. 잔차 학습(Residual Learning) 프레임워크 — F(x) + x — 를 공식화하고, 참조되지 않은 매핑보다 잔차 매핑 학습이 왜 더 쉬운지 설명합니다.
3. 기본 잔차 블록(Basic Residual Block)과 병목 잔차 블록(Bottleneck Residual Block)을 구별하고, 각각이 언제 사용되는지 설명합니다.
4. PyTorch에서 잔차 블록과 스킵 연결(Skip Connection)을 올바르게 구성하여 ResNet 변형(예: ResNet-50)을 처음부터 구현합니다.
5. 블록 간에 입력과 출력 차원이 다를 때 차원 일치 스킵 연결(프로젝션 스킵 연결, Projection Shortcut)을 처리합니다.
6. 정확도, 파라미터 수, 연산 비용 측면에서 ResNet 깊이 변형(ResNet-18, 34, 50, 101, 152)의 트레이드오프를 평가합니다.

## 개요

ResNet(Residual Network)은 2015년 ILSVRC에서 1위를 차지한 혁명적인 모델입니다. Kaiming He 등이 제안한 **Skip Connection (Residual Connection)**을 통해 수백 개 이상의 레이어를 학습할 수 있게 되었습니다.

> "깊이가 깊어질수록 성능이 떨어지는 degradation 문제를 해결"

---

## 수학적 배경

### 이론: Degradation 문제

ResNet 이전, 통념은 매우 깊은 네트워크가 최악의 경우 더 얕은 것과 같아야 한다는 것이었습니다 — 추가 층을 항상 항등으로 설정할 수 있으니까요. 실험적으로 이는 *일어나지 않았습니다*. He et al.은 20 vs 56층 평이 네트워크의 학습 오차를 그렸습니다; 56층 네트가 *더 높은 학습 오차*를 가졌습니다. 이는 과적합도 아니고(학습 오차가 더 높았지 테스트 오차만이 아니었음) 그래디언트 소실만의 문제도 아니었습니다(BatchNorm이 그것을 고쳤어야 함). 그들은 이를 **degradation 문제**라 불렀습니다: 더 깊은 네트워크는 *더 최적화하기 어렵다*, 끝.

해결책은 알고리즘적이 아니라 아키텍처적이어야 했습니다.


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

### 이론: 보틀넥 블록

더 깊은 ResNet(50+)의 경우, 기본 블록(`3x3 -> 3x3`)이 너무 비싸집니다. **보틀넥 블록**은 인수분해합니다:

```
1x1 conv (채널 축소: 예, 256 -> 64)
3x3 conv (축소된 채널에서 작동: 64 -> 64)
1x1 conv (채널 복원: 64 -> 256)
+ 잔차
```

`1x1` conv는 채널 혼합을 저렴하게 하고; `3x3`은 작은 채널 공간에서 공간 작업을 합니다. 총 파라미터는 두 `3x3` conv의 `2 * 9 * 256^2 = 1.18M`에서 보틀넥의 `(256 * 64 + 9 * 64^2 + 64 * 256) = 69k`로 감소 — 같은 입력/출력 차원에 대해 거의 17배 적은 파라미터.

ResNet-50, 101, 152는 모두 보틀넥 블록을 사용합니다. 대시 뒤의 숫자는 총 가중치 층 수입니다: 50 = 1 스템 + 16개 보틀넥 블록 * 각 3 층 + 1 FC, 등.


### 이론: 잔차 연결

잔차 블록은 층의 함수를 재매개변수화합니다. `H(x)`를 직접 계산하는 대신, *잔차* `F(x) = H(x) - x`를 계산하고 입력을 다시 더합니다:

```
y = F(x; W) + x                  (기본 잔차 블록)
```

스킵 연결(shortcut 또는 항등 매핑이라고도 함)은 단지 덧셈입니다 — 파라미터 없음, 계산 없음. 핵심 통찰은 *층 근처의 최적 함수가 항등에 가깝다면*, 잔차 `F(x)`가 0에 가깝고, SGD에게는 실제 항등 행렬보다 0이 학습하기 훨씬 쉽다는 것입니다. Conv 가중치를 0으로 설정하면 정확히 `y = x`가 되며, 이는 해를 끼치지 않는 유효한 출발점입니다. 그래서 그러한 블록을 안전하게 많이 쌓을 수 있고, 각각은 항등 위에 필요한 *수정*만 학습하면 됩니다.

스트라이드나 채널 수가 변하면, shortcut이 형상을 맞추기 위해 `1x1` conv가 필요합니다. 이것이 shortcut이 파라미터를 가지는 유일한 경우입니다.


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

### 이론: 잔차가 그래디언트 소실을 고치는 이유

잔차 블록을 통과하는 역전파도 똑같이 시사적입니다. `y = F(x) + x`라 합시다. 그러면:

```
dL/dx = dL/dy * d/dx (F(x) + x) = dL/dy * (I + dF/dx)
```

`I` 항은 `F`가 어떻게 행동하든 *그래디언트가 항상 직접 경로를 가지고 돌아가도록* 보장합니다. `F`가 잘못 조건화되어 `dF/dx`가 작거나 거의 특이행렬이라도, `dL/dx`는 여전히 항등을 통해 `dL/dy`를 직접 받습니다. L개 적층된 잔차 블록의 경우:

```
dL/dx_0 = dL/dx_L * prod_{l=1}^{L} (I + dF_l/dx_{l-1})
        ≈ dL/dx_L * (I + sum_l dF_l/dx_{l-1})       (각 dF_l이 작을 때)
```

곱이 항들의 합으로 전개됩니다 — 임의의 깊이의 그래디언트 정보가 바닥에 도달할 수 있습니다. 평이 네트워크는 야코비안의 곱셈적 곱만 가지며, 이는 깊이에 따라 지수적으로 감쇠합니다. 이것이 ResNet-152가 학습되고 평이-152가 그렇지 않은 이유입니다.


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
