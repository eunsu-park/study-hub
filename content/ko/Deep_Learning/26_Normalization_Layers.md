[이전: 옵티마이저](./25_Optimizers.md) | [다음: TensorBoard 시각화](./27_TensorBoard.md)

---

# 26. 정규화 레이어(Normalization Layers)

## 학습 목표

- 딥러닝에서 정규화의 필요성과 손실 함수 지형을 부드럽게 만드는 원리 이해하기
- 배치 정규화, 레이어 정규화, 그룹 정규화와 각각의 사용 사례 마스터하기
- 현대 대형 언어 모델에서 선호되는 RMSNorm 학습하기
- 정규화 레이어를 처음부터 구현하고 계산상의 의미 이해하기
- 아키텍처와 배치 크기 제약에 따라 적절한 정규화 기법 적용하기

---

## 1. 왜 정규화가 필요한가?

### 1.1 문제: 내부 공변량 변화(Internal Covariate Shift)

**내부 공변량 변화**는 훈련 중 네트워크 활성화(activation) 분포의 변화를 의미합니다. 초기 레이어의 파라미터가 변경되면 후반 레이어의 입력이 변화하여 지속적으로 적응해야 합니다.

**원래 동기** (Ioffe & Szegedy, 2015):
- 레이어 간 활성화 분포 안정화
- 각 레이어가 더 안정적인 입력 분포에서 학습할 수 있도록 함
- 발산 없이 더 높은 학습률 사용 가능

### 1.2 현대적 이해: 손실 함수 지형 평활화(Loss Landscape Smoothing)

최근 연구(Santurkar et al., 2018)는 정규화의 주요 이점이 **손실 함수 지형 평활화**라는 것을 보여줍니다:

```
정규화 없음:                   정규화 있음:

    |\                              /\
    | \        /\                  /  \
    |  \  /\  /  \                /    \
    |___\/  \/____\__           /______\_____

    거칠고 불규칙한               더 부드럽고 예측 가능한
    기울기                        기울기
```

**이점**:
1. **더 빠른 수렴** — 부드러운 기울기로 더 큰 스텝 가능
2. **더 높은 학습률** — 발산 위험 감소
3. **정규화 효과** — 배치 통계의 노이즈가 암묵적 정규화 역할
4. **초기화 민감도 감소** — 신중한 가중치 초기화에 대한 의존도 감소

### 1.3 정규화 축(Normalization Axes)

다양한 정규화 방법은 서로 다른 차원에서 정규화를 수행합니다:

```
입력 텐서 형태: (N, C, H, W)
N = 배치 크기
C = 채널
H, W = 공간 차원

┌─────────────────────────────────────────────────────────────┐
│  배치 정규화:       N에 대해 정규화      각 (C, H, W)에 대해  │
│  레이어 정규화:     C,H,W에 대해 정규화  각 N에 대해          │
│  인스턴스 정규화:   H,W에 대해 정규화    각 (N, C)에 대해     │
│  그룹 정규화:       C/G,H,W에 대해 정규화 각 (N, G)에 대해   │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 배치 정규화(Batch Normalization)

### 2.1 핵심 개념

**배치 정규화**(BatchNorm)는 각 특성에 대해 독립적으로 배치 차원에서 활성화를 정규화합니다.

**알고리즘**:

```
입력: 미니배치 B = {x₁, ..., xₘ}
파라미터: γ (스케일), β (시프트) — 학습 가능

1. 배치 통계 계산:
   μ_B = (1/m) Σ xᵢ                    # 평균
   σ²_B = (1/m) Σ (xᵢ - μ_B)²          # 분산

2. 정규화:
   x̂ᵢ = (xᵢ - μ_B) / √(σ²_B + ε)       # ε는 수치 안정성을 위함

3. 스케일 및 시프트:
   yᵢ = γ x̂ᵢ + β                        # 학습 가능한 변환
```

**왜 스케일과 시프트가 필요한가?** 네트워크가 필요한 경우 정규화를 취소하는 방법을 학습할 수 있습니다 (예: `γ = √σ²`, `β = μ`로 원래 분포 복원).

### 2.2 훈련 모드 vs 추론 모드

**훈련**:
- 배치 통계 사용 (μ_B, σ²_B)
- 추론을 위한 이동 평균 업데이트:
  ```
  running_mean = momentum × running_mean + (1 - momentum) × μ_B
  running_var = momentum × running_var + (1 - momentum) × σ²_B
  ```

**추론**:
- 이동 통계 사용 (고정됨)
- 현재 배치에 의존하지 않음

```python
import torch
import torch.nn as nn

# 훈련 모드
bn = nn.BatchNorm2d(64)
bn.train()
out = bn(x)  # 배치 통계 사용

# 추론 모드
bn.eval()
out = bn(x)  # 이동 통계 사용
```

### 2.3 BatchNorm을 어디에 배치할까?

**옵션 1: 활성화 함수 이후** (원본 논문)
```
Linear/Conv → Activation → BatchNorm
```

**옵션 2: 활성화 함수 이전** (일반적인 관행)
```
Linear/Conv → BatchNorm → Activation
```

**현대적 합의**: 활성화 함수 이전이 실제로 더 잘 작동하며, 특히 ReLU와 함께 사용할 때 그렇습니다.

### 2.4 PyTorch BatchNorm

```python
import torch.nn as nn

# 완전 연결 레이어용 (1D)
bn1d = nn.BatchNorm1d(num_features=128)

# 합성곱 레이어용 (2D)
bn2d = nn.BatchNorm2d(num_features=64)

# 3D 합성곱용 (비디오, 볼륨 데이터)
bn3d = nn.BatchNorm3d(num_features=32)

# 예제 CNN 블록
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# 사용법
model = ConvBlock(3, 64)
x = torch.randn(32, 3, 224, 224)  # (N, C, H, W)
out = model(x)
print(out.shape)  # torch.Size([32, 64, 224, 224])
```

### 2.5 수동 구현

```python
import torch
import torch.nn as nn

class BatchNorm2dManual(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum

        # 학습 가능한 파라미터
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        # 이동 통계 (경사 하강법으로 업데이트되지 않음)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        # x 형태: (N, C, H, W)

        if self.training:
            # 배치 통계 계산
            # 각 채널 C에 대해 (N, H, W)에서 평균과 분산 계산
            mean = x.mean(dim=[0, 2, 3], keepdim=False)  # 형태: (C,)
            var = x.var(dim=[0, 2, 3], unbiased=False, keepdim=False)  # 형태: (C,)

            # 이동 통계 업데이트
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            self.num_batches_tracked += 1
        else:
            # 이동 통계 사용
            mean = self.running_mean
            var = self.running_var

        # 정규화
        # 브로드캐스팅을 위해 재구성: (1, C, 1, 1)
        mean = mean.view(1, -1, 1, 1)
        var = var.view(1, -1, 1, 1)
        gamma = self.gamma.view(1, -1, 1, 1)
        beta = self.beta.view(1, -1, 1, 1)

        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        out = gamma * x_norm + beta

        return out

# 테스트
manual_bn = BatchNorm2dManual(64)
pytorch_bn = nn.BatchNorm2d(64)

x = torch.randn(32, 64, 16, 16)

# 훈련 모드
manual_bn.train()
pytorch_bn.train()
out_manual = manual_bn(x)
out_pytorch = pytorch_bn(x)

print(f"출력 형태: {out_manual.shape}")
print(f"평균이 0에 가까움: {out_manual.mean().item():.6f}")
print(f"표준편차가 1에 가까움: {out_manual.std().item():.6f}")
```

### 2.6 BatchNorm의 한계

**1. 배치 크기 의존성**
- 작은 배치 → 노이즈가 많은 통계 → 성능 저하
- 배치 크기 < 8은 문제가 됨

**2. 시퀀스 모델 (RNN)**
- 배치 내 다른 시퀀스 길이
- 시간 차원에 적용하기 어려움

**3. 분산 훈련**
- 각 GPU가 다른 배치를 가짐
- Sync BatchNorm 필요 (비용이 높음)

**4. 온라인 학습**
- 한 번에 단일 샘플
- 배치 통계를 사용할 수 없음

---

## 3. 레이어 정규화(Layer Normalization)

### 3.1 핵심 개념

**레이어 정규화**(LayerNorm)는 각 샘플에 대해 독립적으로 모든 특성에서 정규화하므로 배치에 독립적입니다.

```
BatchNorm:  각 특성에 대해 샘플 간 정규화
LayerNorm:  각 샘플에 대해 특성 간 정규화
```

**공식**:

```
배치 내 각 샘플 x에 대해:
  μ = (1/D) Σ xᵢ                        # 특성 간 평균
  σ² = (1/D) Σ (xᵢ - μ)²                # 특성 간 분산
  x̂ᵢ = (xᵢ - μ) / √(σ² + ε)             # 정규화
  yᵢ = γ x̂ᵢ + β                         # 스케일 및 시프트
```

### 3.2 왜 트랜스포머에 LayerNorm을 사용하는가?

**장점**:
1. **배치 독립적** — 배치 크기 = 1에서 작동
2. **시퀀스 길이 독립적** — 각 위치가 동일한 방식으로 정규화됨
3. **추론 시 결정론적** — 이동 통계 불필요

**트랜스포머 아키텍처**:

```
┌─────────────────────────────────────────┐
│  Pre-Norm (현대):                        │
│    x → LayerNorm → Attention → Add(x)   │
│    x → LayerNorm → FFN → Add(x)         │
│                                          │
│  Post-Norm (원본):                       │
│    x → Attention → Add(x) → LayerNorm   │
│    x → FFN → Add(x) → LayerNorm         │
└─────────────────────────────────────────┘
```

**Pre-Norm vs Post-Norm**:
- **Pre-Norm**: 더 나은 기울기 흐름, 훈련하기 쉬움, GPT, LLaMA에서 사용
- **Post-Norm**: 원본 트랜스포머 디자인, 신중한 튜닝으로 약간 더 나은 성능

### 3.3 PyTorch LayerNorm

```python
import torch
import torch.nn as nn

# 트랜스포머용 LayerNorm
ln = nn.LayerNorm(512)  # d_model = 512

# 예제: Pre-Norm을 사용한 셀프 어텐션
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        # Pre-Norm
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.ffn(self.ln2(x))
        return x

# 사용법
model = TransformerBlock(d_model=512, num_heads=8)
x = torch.randn(32, 100, 512)  # (batch, seq_len, d_model)
out = model(x)
print(out.shape)  # torch.Size([32, 100, 512])
```

### 3.4 수동 구현

```python
import torch
import torch.nn as nn

class LayerNormManual(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.normalized_shape = normalized_shape

        # 학습 가능한 파라미터
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        # x 형태: (N, ..., normalized_shape)
        # 예: 트랜스포머의 경우 (N, seq_len, d_model)

        # 마지막 차원에서 평균과 분산 계산
        # 브로드캐스팅을 위해 차원 유지
        dims = list(range(-len(self.gamma.shape) if isinstance(self.normalized_shape, tuple)
                          else -1, 0))

        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)

        # 정규화
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # 스케일 및 시프트
        out = self.gamma * x_norm + self.beta

        return out

# 테스트
manual_ln = LayerNormManual(512)
pytorch_ln = nn.LayerNorm(512)

x = torch.randn(32, 100, 512)  # (batch, seq, features)

out_manual = manual_ln(x)
out_pytorch = pytorch_ln(x)

print(f"출력 형태: {out_manual.shape}")
print(f"샘플당 평균이 0에 가까움: {out_manual[0].mean().item():.6f}")
print(f"샘플당 표준편차가 1에 가까움: {out_manual[0].std().item():.6f}")
```

### 3.5 사용 사례

**가장 적합한 경우**:
- 트랜스포머 (BERT, GPT, ViT)
- RNN, LSTM
- 작은 배치 크기
- 가변 시퀀스 길이

---

## 4. 그룹 정규화(Group Normalization)

### 4.1 핵심 개념

**그룹 정규화**(GroupNorm)는 채널을 그룹으로 나누고 각 그룹 내에서 정규화합니다.

```
입력: (N, C, H, W)
그룹: G

C 채널을 각각 (C/G) 채널씩 G개 그룹으로 분할
각 그룹을 독립적으로 정규화

특수 사례:
  G = 1     → 레이어 정규화 (한 그룹 = 모든 채널)
  G = C     → 인스턴스 정규화 (각 채널이 하나의 그룹)
  G = 32    → 일반적인 선택 (Wu & He, 2018)
```

**시각화**:

```
채널: [c0, c1, c2, c3, c4, c5, c6, c7]
그룹 (G=4): [c0,c1] [c2,c3] [c4,c5] [c6,c7]

배치의 각 샘플에 대해:
  각 그룹에 대해:
    (C/G, H, W)에서 평균/분산 계산
    정규화
```

### 4.2 공식

```
각 샘플 n, 그룹 g에 대해:
  μₙ,ₘ = (1/(C/G · H · W)) Σ x_n,g,h,w
  σ²ₙ,ₘ = (1/(C/G · H · W)) Σ (x_n,g,h,w - μₙ,ₘ)²
  x̂_n,g,h,w = (x_n,g,h,w - μₙ,ₘ) / √(σ²ₙ,ₘ + ε)
  y_n,c,h,w = γ_c · x̂_n,c,h,w + β_c
```

### 4.3 PyTorch GroupNorm

```python
import torch
import torch.nn as nn

# 32개 그룹을 사용한 GroupNorm (일반적인 선택)
gn = nn.GroupNorm(num_groups=32, num_channels=64)

# 예제: GroupNorm을 사용한 ResNet 블록
class ResNetBlock(nn.Module):
    def __init__(self, channels, groups=32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn2 = nn.GroupNorm(groups, channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)

        out += identity
        out = self.relu(out)

        return out

# 사용법
model = ResNetBlock(channels=64, groups=32)
x = torch.randn(4, 64, 56, 56)  # 작은 배치 크기!
out = model(x)
print(out.shape)  # torch.Size([4, 64, 56, 56])
```

### 4.4 그룹 수 선택하기

```python
import torch
import torch.nn as nn

# 규칙: num_channels는 num_groups로 나누어떨어져야 함

# 일반적인 구성
configs = [
    (64, 32),   # 64 채널, 32 그룹 → 그룹당 2 채널
    (128, 32),  # 128 채널, 32 그룹 → 그룹당 4 채널
    (256, 32),  # 256 채널, 32 그룹 → 그룹당 8 채널
]

for channels, groups in configs:
    gn = nn.GroupNorm(groups, channels)
    x = torch.randn(2, channels, 16, 16)  # 작은 배치!
    out = gn(x)
    print(f"{channels} 채널, {groups} 그룹 → "
          f"그룹당 {channels // groups} 채널, 형태: {out.shape}")

# 특수 사례
gn_layer = nn.GroupNorm(1, 64)      # G=1 → LayerNorm 동작
gn_instance = nn.GroupNorm(64, 64)  # G=C → InstanceNorm 동작
```

### 4.5 사용 사례

**가장 적합한 경우**:
- 객체 탐지 (Mask R-CNN, Faster R-CNN)
- 이미지 분할
- 작은 배치 크기 (배치 크기 = 1, 2, 4)
- 고정된 BatchNorm을 사용한 전이 학습
- BatchNorm 통계가 신뢰할 수 없는 시나리오

**성능**:
- COCO 객체 탐지: GroupNorm은 큰 배치에서 BatchNorm과 일치
- 작은 배치(1-4)에서: GroupNorm이 BatchNorm을 크게 능가함

---

## 5. 인스턴스 정규화(Instance Normalization)

### 5.1 핵심 개념

**인스턴스 정규화**(InstanceNorm)는 각 샘플의 각 채널을 독립적으로 정규화합니다.

```
각 샘플, 각 채널에 대해:
  공간 차원 (H, W)에서 평균과 분산 계산
  정규화
```

**G = C인 GroupNorm과 동일** (각 채널이 자체 그룹).

### 5.2 공식

```
각 샘플 n, 채널 c에 대해:
  μₙ,c = (1/(H · W)) Σ x_n,c,h,w
  σ²ₙ,c = (1/(H · W)) Σ (x_n,c,h,w - μₙ,c)²
  x̂_n,c,h,w = (x_n,c,h,w - μₙ,c) / √(σ²ₙ,c + ε)
  y_n,c,h,w = γ_c · x̂_n,c,h,w + β_c
```

### 5.3 PyTorch InstanceNorm

```python
import torch
import torch.nn as nn

# 2D 이미지용
in2d = nn.InstanceNorm2d(64)

# 1D 시퀀스용
in1d = nn.InstanceNorm1d(128)

# 예제: 스타일 전이 네트워크
class StyleTransferBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

# 사용법
model = StyleTransferBlock(64)
x = torch.randn(1, 64, 256, 256)  # 배치 크기 = 1도 괜찮음!
out = model(x)
print(out.shape)  # torch.Size([1, 64, 256, 256])
```

### 5.4 왜 Instance Normalization을 사용하는가?

**핵심 통찰**: 스타일 전이의 경우, 인스턴스별 대비 정보를 정규화하고 싶습니다.

**예제**:
```python
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# 콘텐츠와 스타일 이미지 로드
content = Image.open('content.jpg')
style = Image.open('style.jpg')

# InstanceNorm을 사용한 스타일 전이
class FastStyleTransfer(nn.Module):
    def __init__(self):
        super().__init__()
        # 인코더
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 9, padding=4),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
        )

        # InstanceNorm을 사용한 잔차 블록
        self.residual = nn.Sequential(
            *[self._residual_block(64) for _ in range(5)]
        )

        # 디코더
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 9, padding=4),
            nn.Tanh()
        )

    def _residual_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels, affine=True)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.residual(x)
        x = self.decoder(x)
        return x
```

### 5.5 사용 사례

**가장 적합한 경우**:
- 스타일 전이 (neural style, fast style transfer)
- 이미지 간 변환 (pix2pix, CycleGAN)
- 생성 모델 (이미지 합성용 GAN)
- 텍스처 합성

---

## 6. RMSNorm (Root Mean Square Normalization)

### 6.1 핵심 개념

**RMSNorm**은 평균 중심화를 제거하여 LayerNorm을 단순화하고 제곱 평균 제곱근(root mean square)으로만 정규화합니다.

**핵심 차이**:
```
LayerNorm:  x̂ = (x - μ) / σ           # 중심화 후 스케일 조정
RMSNorm:    x̂ = x / RMS(x)            # 스케일 조정만
```

**공식**:
```
RMS(x) = √((1/n) Σ xᵢ²)
x̂ᵢ = xᵢ / RMS(x)
yᵢ = γ · x̂ᵢ                            # 스케일 (바이어스 β 없음)
```

### 6.2 왜 RMSNorm을 사용하는가?

**장점**:
1. **더 간단한 계산** — 평균 계산이나 빼기 없음
2. **더 빠름** — 대형 모델에서 약 10-15% 속도 향상
3. **유사한 성능** — 경험적으로 LayerNorm과 일치
4. **광범위한 채택** — LLaMA, LLaMA 2, LLaMA 3, Gemma, Mistral

**계산 절감**:
```
LayerNorm:  2번의 패스 (평균, 그 다음 분산) + 2번의 연산 (빼기, 나누기)
RMSNorm:    1번의 패스 (RMS) + 1번의 연산 (나누기)
```

### 6.3 수동 구현

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x 형태: (..., dim)

        # RMS 계산
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        # 정규화 및 스케일
        x_norm = x / rms
        out = self.weight * x_norm

        return out

# 테스트
rms_norm = RMSNorm(512)
x = torch.randn(32, 100, 512)  # (batch, seq, features)
out = rms_norm(x)

print(f"출력 형태: {out.shape}")
print(f"RMS: {torch.sqrt(torch.mean(out[0] ** 2)).item():.6f}")  # 약 1.0이어야 함
```

### 6.4 LayerNorm과 비교

```python
import torch
import torch.nn as nn
import time

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)

# 벤치마크
dim = 4096
seq_len = 2048
batch_size = 8

x = torch.randn(batch_size, seq_len, dim, device='cuda')

layer_norm = nn.LayerNorm(dim).cuda()
rms_norm = RMSNorm(dim).cuda()

# 워밍업
for _ in range(10):
    _ = layer_norm(x)
    _ = rms_norm(x)

# LayerNorm 타이밍
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    _ = layer_norm(x)
torch.cuda.synchronize()
ln_time = time.time() - start

# RMSNorm 타이밍
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    _ = rms_norm(x)
torch.cuda.synchronize()
rms_time = time.time() - start

print(f"LayerNorm: {ln_time:.4f}s")
print(f"RMSNorm:   {rms_time:.4f}s")
print(f"속도 향상:   {ln_time / rms_time:.2f}x")
```

### 6.5 LLaMA에서의 RMSNorm

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LLaMATransformerBlock(nn.Module):
    """RMSNorm을 사용한 LLaMA 스타일 트랜스포머 블록."""

    def __init__(self, dim, num_heads, mlp_ratio=4):
        super().__init__()

        # RMSNorm을 사용한 Pre-normalization
        self.attn_norm = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.ffn_norm = RMSNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=False),
            nn.SiLU(),  # LLaMA는 SiLU (Swish) 활성화 함수 사용
            nn.Linear(mlp_ratio * dim, dim, bias=False)
        )

    def forward(self, x):
        # RMSNorm을 사용한 어텐션
        h = self.attn_norm(x)
        h = self.attn(h, h, h)[0]
        x = x + h

        # RMSNorm을 사용한 FFN
        h = self.ffn_norm(x)
        h = self.ffn(h)
        x = x + h

        return x

# 사용 예제
model = LLaMATransformerBlock(dim=4096, num_heads=32)
x = torch.randn(1, 2048, 4096)  # (batch, seq_len, dim)
out = model(x)
print(out.shape)  # torch.Size([1, 2048, 4096])
```

### 6.6 사용 사례

**가장 적합한 경우**:
- 대형 언어 모델 (LLaMA, Mistral, Gemma)
- 속도가 중요한 모든 트랜스포머 기반 모델
- 처음부터 훈련하는 모델 (LayerNorm 모델 미세 조정이 아님)

---

## 7. 기타 정규화 기법

### 7.1 가중치 정규화(Weight Normalization)

**가중치 정규화**는 크기와 방향을 분리하기 위해 가중치 벡터를 재매개변수화합니다.

```
원본:                w
재매개변수화:        w = g · (v / ||v||)

g = 스칼라 크기 (학습 가능)
v = 방향 벡터 (학습 가능)
```

```python
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

# 레이어에 가중치 정규화 적용
linear = nn.Linear(128, 64)
linear = weight_norm(linear, name='weight')

# 가중치는 다음과 같이 재매개변수화됨: weight = g * v / ||v||
print(linear.weight_g)  # 크기 파라미터
print(linear.weight_v)  # 방향 파라미터

# 순전파
x = torch.randn(32, 128)
out = linear(x)

# 가중치 정규화 제거 (g와 v를 다시 weight로 병합)
linear = nn.utils.remove_weight_norm(linear)
```

**사용 사례**: RNN, GAN, 강화 학습 (A3C)

### 7.2 스펙트럼 정규화(Spectral Normalization)

**스펙트럼 정규화**는 가중치 행렬의 스펙트럼 노름(최대 특이값)을 1로 제한하여 GAN 훈련을 안정화합니다.

```
스펙트럼 노름: σ(W) = W의 최대 특이값
정규화된 가중치: W_SN = W / σ(W)
```

```python
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

# 스펙트럼 정규화 적용
conv = nn.Conv2d(3, 64, 3, padding=1)
conv = spectral_norm(conv)

# 스펙트럼 정규화를 사용한 판별자 (GAN용)
class SNDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(3, 64, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 1, 4, stride=1, padding=0)),
        )

    def forward(self, x):
        return self.model(x)

# 사용법
disc = SNDiscriminator()
x = torch.randn(16, 3, 64, 64)
out = disc(x)
print(out.shape)  # torch.Size([16, 1, 5, 5])
```

**사용 사례**: GAN 판별자 (SNGAN, BigGAN, StyleGAN2)

### 7.3 적응적 인스턴스 정규화(Adaptive Instance Normalization, AdaIN)

**AdaIN**은 스타일 입력에 기반하여 InstanceNorm 통계를 적응적으로 조정하여 실시간 스타일 전이를 가능하게 합니다.

```
AdaIN(content, style) = σ(style) · ((content - μ(content)) / σ(content)) + μ(style)

스타일 통계 (평균, 표준편차)를 콘텐츠 특성으로 전송
```

```python
import torch
import torch.nn as nn

class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, content, style):
        # content, style: (N, C, H, W)

        # 통계 계산
        content_mean = content.mean(dim=[2, 3], keepdim=True)
        content_std = content.std(dim=[2, 3], keepdim=True)

        style_mean = style.mean(dim=[2, 3], keepdim=True)
        style_std = style.std(dim=[2, 3], keepdim=True)

        # 콘텐츠 정규화, 그 다음 스타일 통계 적용
        normalized = (content - content_mean) / (content_std + 1e-5)
        stylized = normalized * style_std + style_mean

        return stylized

# AdaIN을 사용한 스타일 전이
class StyleTransferNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.adain = AdaIN()
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1)
        )

    def forward(self, content, style):
        content_feat = self.encoder(content)
        style_feat = self.encoder(style)

        # AdaIN 레이어
        t = self.adain(content_feat, style_feat)

        out = self.decoder(t)
        return out

# 사용법
model = StyleTransferNet()
content = torch.randn(1, 3, 256, 256)
style = torch.randn(1, 3, 256, 256)
stylized = model(content, style)
print(stylized.shape)  # torch.Size([1, 3, 256, 256])
```

**사용 사례**: 실시간 스타일 전이, 이미지 간 변환

### 7.4 비교 표

| 방법 | 정규화 대상 | 학습 가능한 파라미터 | 배치 의존적 | 사용 사례 |
|--------|-------------------|------------------|-----------------|----------|
| **Batch Norm** | 각 (C,H,W)에 대해 (N) | γ, β, 이동 통계 | 예 | CNN, 큰 배치 |
| **Layer Norm** | 각 N에 대해 (C,H,W) | γ, β | 아니오 | 트랜스포머, RNN |
| **Instance Norm** | 각 (N,C)에 대해 (H,W) | γ, β (선택적) | 아니오 | 스타일 전이, GAN |
| **Group Norm** | 각 (N,G)에 대해 (C/G,H,W) | γ, β | 아니오 | 탐지, 작은 배치 |
| **RMSNorm** | 각 N에 대해 (C,H,W) | γ | 아니오 | LLM, 빠른 트랜스포머 |
| **Weight Norm** | 가중치 벡터 | g, v | 아니오 | RNN, GAN |
| **Spectral Norm** | 가중치 행렬 | — | 아니오 | GAN 판별자 |
| **AdaIN** | 스타일에 따라 (H,W) | — | 아니오 | 스타일 전이 |

---

## 8. 종합 비교

### 8.1 시각적 비교

```
입력 텐서: (N, C, H, W)
N = 배치 (4개 샘플)
C = 채널 (3)
H, W = 높이, 너비 (32 × 32)

┌─────────────────────────────────────────────────────────────────────┐
│  배치 정규화                                                         │
│  ┌──────┬──────┬──────┬──────┐                                      │
│  │ N=0  │ N=1  │ N=2  │ N=3  │                                      │
│  ├──────┼──────┼──────┼──────┤  각 (C, H, W) 위치에 대해:           │
│  │ c=0  │ c=0  │ c=0  │ c=0  │  N에서 평균/분산 계산                │
│  │ h,w  │ h,w  │ h,w  │ h,w  │  (4개 값)                            │
│  └──────┴──────┴──────┴──────┘                                      │
│  정규화: (x - μ_batch) / σ_batch                                    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  레이어 정규화                                                       │
│  ┌───────────────────────────┐                                      │
│  │       N=0                 │  각 샘플에 대해:                      │
│  ├───────┬───────┬───────────┤  모든 (C, H, W)에서                  │
│  │ c=0   │ c=1   │ c=2       │  평균/분산 계산                       │
│  │ (all  │ (all  │ (all      │  (3 × 32 × 32 = 3072개 값)           │
│  │ h,w)  │ h,w)  │ h,w)      │                                      │
│  └───────┴───────┴───────────┘                                      │
│  정규화: (x - μ_layer) / σ_layer                                    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  인스턴스 정규화                                                     │
│  ┌────────────────┐                                                 │
│  │  N=0, C=0      │  각 (샘플, 채널)에 대해:                         │
│  ├────────────────┤  (H, W)에서                                     │
│  │   (all h,w)    │  평균/분산 계산                                  │
│  │                │  (32 × 32 = 1024개 값)                          │
│  └────────────────┘                                                 │
│  정규화: (x - μ_instance) / σ_instance                              │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  그룹 정규화 (G=3, 그룹당 1채널)                                     │
│  ┌────────────────┐                                                 │
│  │ N=0, G=0       │  각 (샘플, 그룹)에 대해:                         │
│  │ (c=0만)        │  (C/G, H, W)에서                                │
│  ├────────────────┤  평균/분산 계산                                  │
│  │   (all h,w)    │  (1 × 32 × 32 = 1024개 값)                      │
│  └────────────────┘                                                 │
│  정규화: (x - μ_group) / σ_group                                    │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.2 언제 무엇을 사용할까?

#### 결정 트리

```
트랜스포머를 사용하고 있나요?
  ├─ 예 → LayerNorm 또는 RMSNorm
  │         ├─ 속도가 중요? → RMSNorm (LLaMA, Mistral)
  │         └─ 그렇지 않으면 → LayerNorm (BERT, ViT)
  │
  └─ 아니오 → CNN을 사용하고 있나요?
           ├─ 예 → 배치 크기가 큰가요 (≥16)?
           │         ├─ 예 → BatchNorm
           │         └─ 아니오 → GroupNorm
           │
           └─ 아니오 → GAN이나 스타일 전이인가요?
                     ├─ 예 → InstanceNorm 또는 AdaIN
                     └─ 아니오 → LayerNorm (안전한 기본값)
```

#### 아키텍처별 권장 사항

**합성곱 신경망(CNN)**:
```python
# 표준 분류 (ImageNet)
# 배치 크기: 32-256
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),  # ← 큰 배치에 BatchNorm
            nn.ReLU(inplace=True)
        )
```

**객체 탐지 / 분할**:
```python
# Mask R-CNN, Faster R-CNN
# 배치 크기: 1-4 (큰 이미지에 대해 GPU 메모리 제한)
class DetectionBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.GroupNorm(32, 64),  # ← 작은 배치에 GroupNorm
            nn.ReLU(inplace=True)
        )
```

**트랜스포머 (비전 또는 언어)**:
```python
# BERT, GPT, ViT
class TransformerEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)  # ← LayerNorm 표준
        self.attn = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
```

**대형 언어 모델**:
```python
# LLaMA, Mistral, Gemma
class LLMTransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn_norm = RMSNorm(dim)  # ← 속도를 위한 RMSNorm
        self.ffn_norm = RMSNorm(dim)
```

**생성적 적대 신경망(GAN)**:
```python
# 생성자: 스타일을 위한 InstanceNorm
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.InstanceNorm2d(64),  # ← InstanceNorm
            nn.ReLU(inplace=True)
        )

# 판별자: 스펙트럼 정규화
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            spectral_norm(nn.Conv2d(3, 64, 4, stride=2, padding=1)),  # ← SpectralNorm
            nn.LeakyReLU(0.2, inplace=True)
        )
```

### 8.3 성능 벤치마크

```python
import torch
import torch.nn as nn
import time

def benchmark_normalization(norm_layer, input_shape, num_iterations=1000):
    """정규화 레이어 벤치마크."""
    x = torch.randn(*input_shape, device='cuda')

    # 워밍업
    for _ in range(10):
        _ = norm_layer(x)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        _ = norm_layer(x)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    return elapsed

# 테스트 구성
batch_size = 32
channels = 256
height, width = 56, 56
input_shape = (batch_size, channels, height, width)

# 정규화 레이어
norms = {
    'BatchNorm2d': nn.BatchNorm2d(channels).cuda(),
    'GroupNorm (G=32)': nn.GroupNorm(32, channels).cuda(),
    'InstanceNorm2d': nn.InstanceNorm2d(channels).cuda(),
}

print(f"입력 형태: {input_shape}")
print(f"반복 횟수: 1000\n")

results = {}
for name, norm in norms.items():
    norm.eval()
    elapsed = benchmark_normalization(norm, input_shape)
    results[name] = elapsed
    print(f"{name:20s}: {elapsed:.4f}s")

# 상대 속도
baseline = results['BatchNorm2d']
print("\nBatchNorm2d 대비:")
for name, elapsed in results.items():
    print(f"{name:20s}: {elapsed / baseline:.2f}x")
```

**일반적인 결과** (RTX 3090):
```
BatchNorm2d         : 0.1234s  (1.00x)
GroupNorm (G=32)    : 0.1456s  (1.18x)
InstanceNorm2d      : 0.1389s  (1.13x)
```

**트랜스포머의 경우** (seq_len=2048, d_model=4096):
```
LayerNorm           : 0.2145s  (1.00x)
RMSNorm             : 0.1876s  (0.87x)  ← 약 13% 빠름
```

---

## 9. 실용적인 팁

### 9.1 γ와 β의 초기화

**기본 초기화** (PyTorch):
```python
# γ (스케일)은 1로 초기화
# β (시프트)는 0으로 초기화
# 처음에는 원래 분포를 유지함
```

**특수 사례**:

```python
import torch.nn as nn

# 잔차 블록에 대해 γ를 0으로 초기화 (He et al., 2019)
# "Fixup Initialization" — 매우 깊은 네트워크에 도움
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

        # 각 블록의 마지막 BatchNorm을 0으로 초기화
        nn.init.constant_(self.bn2.weight, 0)  # γ = 0
        nn.init.constant_(self.bn2.bias, 0)    # β = 0

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)  # 처음에는 0을 출력하므로 out = identity
        out += identity
        out = F.relu(out)
        return out
```

### 9.2 가중치 초기화와의 상호작용

**BatchNorm은 네트워크를 가중치 초기화에 덜 민감하게 만들지만**, 여전히 적절한 초기화를 사용해야 합니다:

```python
import torch.nn as nn
import torch.nn.init as init

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # ReLU를 위한 He 초기화
        init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        init.constant_(self.conv.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
```

### 9.3 정규화와 학습률

**핵심 통찰**: 정규화는 더 높은 학습률을 허용합니다.

```python
import torch.optim as optim

# 정규화 없음
model_no_norm = MyModel(use_norm=False)
optimizer = optim.SGD(model_no_norm.parameters(), lr=0.01)  # 보수적인 학습률

# BatchNorm/LayerNorm 사용
model_with_norm = MyModel(use_norm=True)
optimizer = optim.SGD(model_with_norm.parameters(), lr=0.1)  # 10배 높은 학습률!

# LayerNorm을 사용한 현대 트랜스포머
# 워밍업과 함께 더 높은 학습률 사용 가능
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
```

**학습률 스케일링 규칙** (Goyal et al., 2017):
```
배치 크기를 k배 증가시킬 때, 학습률도 k배 증가
(BatchNorm에서만 작동!)

배치 256, 학습률 0.1  →  배치 1024, 학습률 0.4
```

### 9.4 일반적인 버그와 함정

#### 버그 1: model.eval() 잊기

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU()
)

# 훈련 모드
model.train()
x = torch.randn(1, 3, 32, 32)
out_train = model(x)

# 추론 — 잘못됨! 여전히 배치 통계 사용
# out_test = model(x)  # 버그: 여전히 훈련 모드!

# 추론 — 올바름
model.eval()  # 평가 모드로 전환!
out_test = model(x)

print(f"출력이 같은가? {torch.allclose(out_train, out_test)}")  # False
```

#### 버그 2: 잘못된 차원 순서

```python
# 잘못됨: LayerNorm에 대해 (seq_len, batch, features)
x = torch.randn(100, 32, 512)  # (seq, batch, features)
ln = nn.LayerNorm(512)
out = ln(x)  # 작동하지만 마지막 차원만 정규화

# 올바름: 특성에 대해 정규화 (마지막 차원)
# 텐서 레이아웃이 normalized_shape과 일치하는지 확인!

# (batch, seq, features)의 경우 — 현재 표준
x = torch.randn(32, 100, 512)  # (batch, seq, features)
ln = nn.LayerNorm(512)
out = ln(x)  # 올바름: 특성 차원에서 정규화
```

#### 버그 3: 호환되지 않는 채널을 가진 GroupNorm

```python
# 잘못됨: num_channels가 num_groups로 나누어떨어지지 않음
try:
    gn = nn.GroupNorm(num_groups=32, num_channels=50)  # 50 % 32 != 0
except ValueError as e:
    print(f"오류: {e}")

# 올바름: 나누어떨어지도록 확인
gn = nn.GroupNorm(num_groups=32, num_channels=64)  # 64 % 32 == 0 ✓
```

#### 버그 4: batch_size = 1인 BatchNorm

```python
# 문제: 단일 샘플을 가진 BatchNorm
bn = nn.BatchNorm2d(64)
x = torch.randn(1, 64, 32, 32)  # 배치 크기 = 1

bn.train()
out = bn(x)  # 분산 = 0! (단일 샘플)
# NaN 또는 불안정한 훈련 발생

# 해결책 1: GroupNorm 또는 LayerNorm 사용
gn = nn.GroupNorm(32, 64)
out = gn(x)  # batch_size=1에서 잘 작동

# 해결책 2: BatchNorm을 평가 모드로 설정
bn.eval()
out = bn(x)  # 이동 통계 사용
```

#### 버그 5: 고정 및 훈련 가능 BatchNorm 혼합

```python
# 문제: 고정된 BatchNorm 통계로 미세 조정
pretrained_model = torchvision.models.resnet50(pretrained=True)

# 모든 파라미터 고정
for param in pretrained_model.parameters():
    param.requires_grad = False

# 이것으로 충분하지 않음! BatchNorm은 여전히 훈련 모드 통계 사용
pretrained_model.train()  # 버그: BatchNorm이 훈련 모드!

# 해결책: 평가 모드로 설정 또는 BatchNorm 모듈을 평가로 설정
pretrained_model.eval()  # 추론에 안전

# 미세 조정의 경우:
def set_bn_eval(module):
    if isinstance(module, nn.BatchNorm2d):
        module.eval()

pretrained_model.train()
pretrained_model.apply(set_bn_eval)  # 훈련 중 BatchNorm을 평가 모드로 유지
```

### 9.5 모범 사례 체크리스트

✅ **추론 전에 항상 `model.eval()` 호출**

✅ **아키텍처에 맞는 정규화 사용**:
   - CNN (큰 배치) → BatchNorm
   - CNN (작은 배치) → GroupNorm
   - 트랜스포머 → LayerNorm 또는 RMSNorm
   - GAN → InstanceNorm (생성자), SpectralNorm (판별자)

✅ **적절한 초기화 사용** (ReLU에 Kaiming, Tanh에 Xavier)

✅ **정규화 사용 시 학습률 증가**

✅ **전이 학습의 경우**, BatchNorm 통계 고정 또는 GroupNorm으로 대체 고려

✅ **이동 통계 모니터링** — 훈련 중 안정화되는지 확인

✅ **분산 훈련의 경우**, 필요시 SyncBatchNorm 사용:
```python
model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
```

---

## 연습 문제

### 연습 문제 1: 정규화 방법 구현 및 비교

다양한 정규화 방법을 사용하는 CNN을 구현하고 CIFAR-10에서 성능을 비교하세요.

**과제**:
1. 각각 다른 정규화를 사용하는 4개의 동일한 CNN 생성:
   - BatchNorm2d
   - GroupNorm (32개 그룹)
   - LayerNorm
   - 정규화 없음 (기준선)
2. 각각 CIFAR-10에서 20 에포크 훈련
3. 훈련 곡선 (손실 및 정확도) 플롯
4. 각각의 최종 테스트 정확도 보고
5. 배치 크기 [4, 16, 64]로 실험하고 어떤 정규화가 가장 견고한지 관찰

**시작 코드**:
```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

class CIFAR10Net(nn.Module):
    def __init__(self, norm_type='batch'):
        super().__init__()
        self.norm_type = norm_type

        def conv_block(in_c, out_c):
            layers = [nn.Conv2d(in_c, out_c, 3, padding=1)]

            if norm_type == 'batch':
                layers.append(nn.BatchNorm2d(out_c))
            elif norm_type == 'group':
                layers.append(nn.GroupNorm(32, out_c))
            elif norm_type == 'layer':
                # 2D용 LayerNorm: (C, H, W)에 대해 정규화
                layers.append(nn.GroupNorm(1, out_c))  # G=1은 LayerNorm
            # 'none': 정규화 없음

            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            conv_block(3, 64),
            conv_block(64, 64),
            nn.MaxPool2d(2),
            conv_block(64, 128),
            conv_block(128, 128),
            nn.MaxPool2d(2),
            conv_block(128, 256),
            conv_block(256, 256),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# TODO: 훈련 루프 및 비교 구현
```

### 연습 문제 2: 트랜스포머에서 RMSNorm vs LayerNorm

작은 트랜스포머를 구현하고 속도와 성능 측면에서 RMSNorm과 LayerNorm을 비교하세요.

**과제**:
1. 문자 수준 언어 모델 구현 (다음 문자 예측)
2. 두 버전 훈련: LayerNorm 사용, RMSNorm 사용
3. 텍스트 데이터셋 사용 (예: Shakespeare, WikiText-2)
4. 측정:
   - 에포크당 훈련 시간
   - 최종 perplexity
   - 추론 속도
5. 분석: RMSNorm이 더 빠르면서도 LayerNorm 성능과 일치하는가?

**시작 코드**:
```python
import torch
import torch.nn as nn

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8,
                 num_layers=6, norm_type='layer'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, d_model))

        # 정규화 선택
        if norm_type == 'layer':
            norm_cls = lambda: nn.LayerNorm(d_model)
        elif norm_type == 'rms':
            norm_cls = lambda: RMSNorm(d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, norm_cls)
            for _ in range(num_layers)
        ])

        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]

        for layer in self.layers:
            x = layer(x)

        logits = self.output(x)
        return logits

# TODO: 훈련 및 벤치마킹 구현
```

### 연습 문제 3: 스타일 전이를 위한 적응적 인스턴스 정규화

AdaIN을 사용한 간단한 스타일 전이 네트워크를 구현하세요.

**과제**:
1. 중간에 AdaIN을 사용한 인코더-디코더 아키텍처 구현
2. 인코더로 사전 훈련된 VGG 네트워크 사용 (가중치 고정)
3. 이미지를 재구성하는 디코더 훈련
4. 스타일 통계를 전송하는 AdaIN 레이어 구현
5. 콘텐츠 및 스타일 이미지에서 테스트 (torchvision 데이터셋 또는 자체 이미지 사용)
6. 스타일화된 출력 시각화
7. **보너스**: 제어 가능한 스타일 전이 구현 (콘텐츠/스타일 혼합을 위한 α 파라미터)

**시작 코드**:
```python
import torch
import torch.nn as nn
from torchvision import models

class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, content, style):
        # TODO: AdaIN 구현
        # 1. 콘텐츠의 평균과 표준편차 계산
        # 2. 스타일의 평균과 표준편차 계산
        # 3. 콘텐츠 정규화, 스타일 통계 적용
        pass

class StyleTransferNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 인코더: VGG19 (고정됨)
        vgg = models.vgg19(pretrained=True).features
        self.encoder = nn.Sequential(*list(vgg.children())[:21])  # relu4_1까지
        for param in self.encoder.parameters():
            param.requires_grad = False

        # AdaIN 레이어
        self.adain = AdaIN()

        # 디코더: 인코더의 거울
        self.decoder = nn.Sequential(
            # TODO: 디코더 구현 (인코더의 역순)
            # ConvTranspose2d 또는 Upsample + Conv2d 사용
        )

    def forward(self, content, style):
        # TODO:
        # 1. 콘텐츠와 스타일 인코딩
        # 2. AdaIN 적용
        # 3. 디코딩
        pass

# TODO: 지각 손실을 사용한 훈련 루프 구현
```

---

## 참고 자료

1. **Batch Normalization**:
   - Ioffe & Szegedy (2015). "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift." ICML.
   - Santurkar et al. (2018). "How Does Batch Normalization Help Optimization?" NeurIPS.

2. **Layer Normalization**:
   - Ba et al. (2016). "Layer Normalization." arXiv:1607.06450.

3. **Group Normalization**:
   - Wu & He (2018). "Group Normalization." ECCV.

4. **Instance Normalization**:
   - Ulyanov et al. (2016). "Instance Normalization: The Missing Ingredient for Fast Stylization." arXiv:1607.08022.

5. **RMSNorm**:
   - Zhang & Sennrich (2019). "Root Mean Square Layer Normalization." NeurIPS.
   - Touvron et al. (2023). "LLaMA: Open and Efficient Foundation Language Models." arXiv:2302.13971.

6. **Spectral Normalization**:
   - Miyato et al. (2018). "Spectral Normalization for Generative Adversarial Networks." ICLR.

7. **AdaIN**:
   - Huang & Belongie (2017). "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization." ICCV.

8. **종합 분석**:
   - Bjorck et al. (2018). "Understanding Batch Normalization." NeurIPS.
   - Goyal et al. (2017). "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour." arXiv:1706.02677.

9. **PyTorch 문서**:
   - https://pytorch.org/docs/stable/nn.html#normalization-layers
   - https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
   - https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
   - https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html

10. **실용 가이드**:
    - He et al. (2019). "Bag of Tricks for Image Classification with Convolutional Neural Networks." CVPR.
    - Xiong et al. (2020). "On Layer Normalization in the Transformer Architecture." ICML.
