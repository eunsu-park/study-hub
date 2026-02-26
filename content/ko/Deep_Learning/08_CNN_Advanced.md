# 06. CNN 심화 - 유명 아키텍처

## 학습 목표

- VGG, ResNet, EfficientNet 아키텍처 이해
- Skip Connection과 Residual Learning
- 깊은 네트워크의 학습 문제와 해결책
- PyTorch로 구현

---

## 1. VGG (2014)

### 핵심 아이디어

- 작은 필터(3×3)만 사용
- 깊이를 늘려 성능 향상
- 단순하고 일관된 구조

### 구조 (VGG16)

```
Input 224×224×3
  ↓
Conv 3×3, 64 ×2 → MaxPool → 112×112×64
  ↓
Conv 3×3, 128 ×2 → MaxPool → 56×56×128
  ↓
Conv 3×3, 256 ×3 → MaxPool → 28×28×256
  ↓
Conv 3×3, 512 ×3 → MaxPool → 14×14×512
  ↓
Conv 3×3, 512 ×3 → MaxPool → 7×7×512
  ↓
FC 4096 → FC 4096 → FC 1000
```

### PyTorch 구현

```python
def make_vgg_block(in_ch, out_ch, num_convs):
    layers = []
    for i in range(num_convs):
        layers.append(nn.Conv2d(
            in_ch if i == 0 else out_ch,
            out_ch, 3, padding=1
        ))
        layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)

class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            make_vgg_block(3, 64, 2),
            make_vgg_block(64, 128, 2),
            make_vgg_block(128, 256, 3),
            make_vgg_block(256, 512, 3),
            make_vgg_block(512, 512, 3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
```

---

## 2. ResNet (2015)

단순히 레이어를 더 쌓으면 성능이 향상될 것 같지만, 실험 결과 약 20층을 넘어서면 정확도가 오히려 *감소*하는 현상(성능 저하 문제(Degradation Problem))이 나타났습니다. 이는 놀라운 결과였는데, 깊은 네트워크는 최소한 얕은 네트워크만큼은 좋아야 하기 때문입니다 — 추가된 층이 항등 매핑(Identity Mapping)을 학습하기만 해도 되니까요. 핵심 통찰: 완전한 매핑 H(x)를 학습하는 것보다 잔차(Residual) F(x)를 학습하는 것이 더 쉽습니다. 최적 매핑이 항등 함수에 가깝다면, F(x) = 0을 학습하는 것은 자명하지만, H(x) = x를 처음부터 학습하는 것은 그렇지 않습니다.

### 문제: 기울기 소실

- 네트워크가 깊어지면 기울기가 소실됨
- 단순히 층을 쌓으면 성능이 떨어짐

### 해결: Residual Connection

```
        ┌─────────────────┐
        │                 │
x ──────┼───► Conv ──► Conv ──►(+)──► ReLU ──► Output
        │                 ↑
        └────────(identity)┘

Output = F(x) + x   (Residual Learning)
```

### 핵심 인사이트

- 항등 함수 학습이 쉬워짐
- 기울기가 skip connection을 통해 직접 전파
- 1000층 이상도 학습 가능

**잔차 연결을 통한 기울기 흐름(Gradient Flow)**: 출력은 H(x) = F(x) + x입니다. 역전파(Backpropagation) 시: dL/dx = dL/dH * (dF/dx + 1). 핵심은 "+1" 항으로, 이 덕분에 기울기가 합성곱 레이어를 완전히 우회하여 스킵 연결(Skip Connection)을 통해 직접 전달되는 경로가 항상 존재합니다. 매우 깊은 네트워크에서처럼 dF/dx가 소실되더라도 기울기는 최소한 dL/dH * 1을 보장받습니다 — 1000층 네트워크에서도 기울기 소실이 발생하지 않습니다.

### PyTorch 구현

```python
class BasicBlock(nn.Module):
    """ResNet basic block"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        # Normalize activations per channel — stabilizes training by reducing
        # internal covariate shift, allowing higher learning rates
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # downsample: 1×1 conv that matches dimensions when residual and
        # main path differ in channels or spatial size (e.g., stride=2)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            # Project identity to match out's shape so addition is valid
            identity = self.downsample(x)

        out += identity  # Skip connection! Gradient flows through this '+' unchanged
        out = F.relu(out)
        return out
```

### Bottleneck Block (ResNet-50+)

C_out < C_in인 1x1 합성곱은 채널 차원을 축소합니다 — 채널 전반에 걸친 학습된 선형 조합으로, 고차원 특성을 압축된 부분공간으로 투영하는 것으로 생각할 수 있습니다. 이를 통해 이후의 비용이 큰 3x3 합성곱의 연산량을 줄입니다. 예를 들어 256채널을 64로 줄이면 해당 레이어의 FLOPs가 16배 감소합니다.

```python
class Bottleneck(nn.Module):
    """1×1 → 3×3 → 1×1 structure"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        # 1×1 conv: reduce channels (e.g., 256→64) to cut 3×3 conv cost
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 3×3 conv: the only spatially-aware layer in the bottleneck —
        # operates on the reduced channel dimension for efficiency
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 1×1 conv: expand back to 4× channels for the residual addition
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)
        return out
```

---

## 3. ResNet 변형들

### Pre-activation ResNet

```
Original: x → Conv → BN → ReLU → Conv → BN → (+) → ReLU
Pre-act: x → BN → ReLU → Conv → BN → ReLU → Conv → (+)
```

### ResNeXt

```python
# Using grouped convolution
self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                       groups=32, padding=1)
```

### SE-ResNet (Squeeze-and-Excitation)

```python
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y  # Channel recalibration
```

---

## 4. EfficientNet (2019)

### 핵심 아이디어

- 깊이, 너비, 해상도의 균형 있는 스케일링
- Compound Scaling

```
depth: α^φ
width: β^φ
resolution: γ^φ

α × β² × γ² ≈ 2 (computation constraint)
```

### MBConv 블록

```python
class MBConv(nn.Module):
    """Mobile Inverted Bottleneck"""
    def __init__(self, in_ch, out_ch, expand_ratio, stride, se_ratio=0.25):
        super().__init__()
        hidden = in_ch * expand_ratio

        self.expand = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU()
        ) if expand_ratio != 1 else nn.Identity()

        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU()
        )

        self.se = SEBlock(hidden, int(in_ch * se_ratio))

        self.project = nn.Sequential(
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

        self.use_skip = stride == 1 and in_ch == out_ch

    def forward(self, x):
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.se(out)
        out = self.project(out)
        if self.use_skip:
            out = out + x
        return out
```

---

## 5. 아키텍처 비교

| 모델 | 파라미터 | Top-1 Acc | 특징 |
|------|----------|-----------|------|
| VGG16 | 138M | 71.5% | 단순, 메모리 많이 사용 |
| ResNet-50 | 26M | 76.0% | Skip Connection |
| ResNet-152 | 60M | 78.3% | 더 깊은 버전 |
| EfficientNet-B0 | 5.3M | 77.1% | 효율적 |
| EfficientNet-B7 | 66M | 84.3% | 최고 성능 |

---

## 6. torchvision 사전 학습 모델

```python
import torchvision.models as models

# Load pretrained models
resnet50 = models.resnet50(weights='IMAGENET1K_V2')
efficientnet = models.efficientnet_b0(weights='IMAGENET1K_V1')
vgg16 = models.vgg16(weights='IMAGENET1K_V1')

# Feature extraction
resnet50.eval()
for param in resnet50.parameters():
    param.requires_grad = False

# Replace last layer (transfer learning)
resnet50.fc = nn.Linear(2048, 10)  # 10 classes
```

---

## 7. 모델 선택 가이드

### 용도별 추천

| 상황 | 추천 모델 |
|------|----------|
| 빠른 추론 필요 | MobileNet, EfficientNet-B0 |
| 높은 정확도 필요 | EfficientNet-B4~B7 |
| 교육/이해 목적 | VGG, ResNet-18 |
| 메모리 제한 | MobileNet, ShuffleNet |

### 실전 팁

```python
# Check model size
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Calculate FLOPs (thop package)
from thop import profile
flops, params = profile(model, inputs=(torch.randn(1, 3, 224, 224),))
```

---

## 정리

### 핵심 개념

1. **VGG**: 작은 필터 반복, 깊은 네트워크
2. **ResNet**: Skip Connection으로 기울기 소실 해결
3. **EfficientNet**: 효율적인 스케일링

### 발전 흐름

```
LeNet (1998)
  ↓
AlexNet (2012) - GPU usage
  ↓
VGG (2014) - Deeper
  ↓
GoogLeNet (2014) - Inception module
  ↓
ResNet (2015) - Skip Connection
  ↓
EfficientNet (2019) - Compound Scaling
  ↓
Vision Transformer (2020) - Attention
```

---

## 연습 문제

### 연습 1: ResNet 스킵 연결(Skip Connection) — 왜 효과적인가

스킵 연결의 이점을 개념적·경험적으로 검증하세요.

1. CIFAR-10에 대해 스킵 연결이 없는 6층 일반 CNN과 6블록 ResNet을 구성하세요.
2. 동일한 하이퍼파라미터로 두 모델을 20 에포크 학습하세요.
3. 최종 테스트 정확도와 학습 손실 변화를 비교하세요.
4. 두 모델에서 학습 중 첫 번째 층의 기울기 노름(Norm)을 확인하세요.
5. `out += identity`를 추가하면 기울기가 더 자유롭게 흐를 수 있는 이유를 자신의 말로 설명하세요.

### 연습 2: BasicBlock 직접 구현하기

`BasicBlock`을 처음부터 구현하고 PyTorch 내장 ResNet과 비교하여 검증하세요.

1. 채널이나 스트라이드(Stride)가 변경될 때 다운샘플 숏컷(Downsample Shortcut)을 포함하는 `BasicBlock(in_channels, out_channels, stride)`를 수업에서 배운 대로 구현하세요.
2. 이러한 블록 4개를 쌓아 MNIST용 소형 ResNet 유사 모델을 만드세요.
3. 순전파에서 `output.shape == (batch, num_classes)`임을 확인하세요.
4. 비슷한 깊이의 일반 CNN과 총 파라미터 수를 비교하세요.

### 연습 3: 스퀴즈 앤 익사이테이션(Squeeze-and-Excitation) 채널 어텐션

기존 모델에 채널 어텐션을 추가하고 정확도 향상을 측정하세요.

1. 수업에서 배운 대로 `SEBlock(channels, reduction=16)`을 구현하세요.
2. 이전 수업의 `CIFAR10Net` 두 번째 합성곱 레이어를 SE 블록으로 감싸세요.
3. CIFAR-10에서 기준 모델과 SE 적용 모델 모두 20 에포크 학습하세요.
4. 테스트 정확도 차이를 보고하세요. SE 블록이 어떤 채널을 증폭할지 결정하는 메커니즘을 설명하세요.

### 연습 4: 모델 효율성 트레이드오프 분석

`count_parameters`와 `thop.profile`을 사용하여 아키텍처를 나란히 비교하세요.

1. torchvision에서 `vgg16`, `resnet50`, `efficientnet_b0`를 불러오세요 (사전 학습 가중치 불필요).
2. `count_parameters`로 각 모델의 학습 가능한 파라미터를 계산하세요.
3. `thop.profile`로 224×224 입력에 대한 FLOPs를 계산하세요.
4. 파라미터, FLOPs, ImageNet Top-1 정확도(수업의 표 기준)를 포함하는 표를 작성하세요.
5. 파라미터 대비 정확도 비율이 가장 좋은 모델은 무엇인지, 그 이유를 설명하세요.

### 연습 5: 아키텍처 발전 실험

LeNet 스타일, VGG 스타일, ResNet 스타일 네트워크를 CIFAR-10에서 학습하여 역사적 개선 과정을 추적하세요.

1. 2-합성곱 LeNet 스타일 네트워크, 4-합성곱 VGG 스타일 블록, 4-블록 ResNet을 구현하세요.
2. 동일한 옵티마이저(Adam, lr=0.001)로 세 모델을 25 에포크 학습하세요.
3. 세 모델의 학습 곡선(학습 손실 및 테스트 정확도)을 하나의 차트에 표시하세요.
4. 수렴 속도와 최종 성능의 질적 차이를 설명하세요.

---

## 다음 단계

[전이학습 (Transfer Learning)](./09_Transfer_Learning.md)에서 사전 학습된 모델을 활용한 전이 학습을 배웁니다.
