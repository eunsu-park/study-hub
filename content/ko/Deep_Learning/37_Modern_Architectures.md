[이전: Self-Supervised Learning](./36_Self_Supervised_Learning.md) | [다음: 객체 탐지](./38_Object_Detection.md)

---

# 37. 현대 딥러닝 아키텍처

## 학습 목표

- 최근 딥러닝 아키텍처 혁신(2020-2024) 살펴보기
- ConvNeXt와 Transformer 시대의 순수 ConvNet의 진화 이해하기
- EfficientNetV2와 점진적 학습 전략(progressive training strategies)에 대해 배우기
- 자기지도 학습 비전 파운데이션 모델인 DINOv2 탐구하기
- 빠른 확산 샘플링을 위한 잠재 일관성 모델(Latent Consistency Models, LCM) 이해하기
- timm 및 transformers 라이브러리를 사용한 사전학습된 현대 아키텍처 적용하기

---

## 1. 아키텍처 진화 타임라인

딥러닝 아키텍처 환경은 빠르게 진화해왔습니다:

```
2017: ResNet/ResNeXt dominance
      └─ Bottleneck blocks, skip connections

2017: Transformer (NLP)
      └─ Self-attention, positional encoding

2020: Vision Transformer (ViT)
      └─ Pure attention for vision

2021: Swin Transformer
      └─ Hierarchical vision transformer with shifted windows

2022: ConvNeXt
      └─ Modernized ConvNet matching Transformers

2022: EfficientNetV2
      └─ Progressive training + Fused-MBConv

2023: DINOv2
      └─ Self-supervised vision foundation model

2023: Latent Consistency Models
      └─ Fast diffusion sampling (1-4 steps)

2024: ConvNeXt V2, Mamba, Hyena
      └─ Continued innovation in architectures
```

### 주요 트렌드

1. **하이브리드 아키텍처**: 합성곱(convolutions)과 어텐션(attention) 결합
2. **자기지도 사전학습(Self-supervised pretraining)**: DINO, MAE, CLIP
3. **스케일링 법칙(Scaling laws)**: 더 큰 모델, 더 많은 데이터, 더 긴 학습
4. **효율성**: FLOPs, 파라미터, 지연시간 감소
5. **파운데이션 모델(Foundation models)**: 범용 사전학습 모델

---

## 2. ConvNeXt: ConvNet의 현대화

**ConvNeXt** (Liu et al., 2022)는 순수 ConvNet이 최신 설계 선택으로 현대화될 때 Transformer와 경쟁할 수 있음을 보여줍니다.

### 2.1 ResNet에서 ConvNeXt로의 설계 진화

ResNet-50에서 시작하여 단계별로 현대적 개선사항을 적용:

```
Step 1: Training procedure (90 → 300 epochs, AdamW, mixup, cutmix)
        Accuracy: 76.1% → 78.8%

Step 2: Macro design (stage ratio 3:4:6:3 → 3:3:9:3)
        Patchify stem (7×7 stride-2 → 4×4 stride-4)
        Accuracy: 78.8% → 79.4%

Step 3: ResNeXt-ify (grouped convolutions)
        Depthwise convolution (groups = channels)
        Accuracy: 79.4% → 80.5%

Step 4: Inverted bottleneck (narrow → wide → narrow)
        Expansion ratio 4× (similar to Transformers' MLP)
        Accuracy: 80.5% → 80.6%

Step 5: Large kernel sizes (3×3 → 7×7)
        Accuracy: 80.6% → 81.0%

Step 6: Micro design (ReLU → GELU, BN → LN, fewer layers)
        Accuracy: 81.0% → 82.0%

Final ConvNeXt-T: 82.0% (matches Swin-T)
```

### 2.2 ConvNeXt 블록 아키텍처

```
Input (C channels)
    |
    ├──────────────────┐  (Residual connection)
    |                  |
Depthwise Conv 7×7     |
    |                  |
LayerNorm              |
    |                  |
1×1 Conv (4C)          |  (Expansion)
    |                  |
GELU                   |
    |                  |
1×1 Conv (C)           |  (Projection)
    |                  |
    +──────────────────┘
    |
Output (C channels)
```

**ResNet과의 주요 차이점**:
- 3×3 표준 합성곱 대신 **Depthwise convolution** (7×7)
- **Inverted bottleneck**: 4C로 확장한 후 다시 C로 투영
- BatchNorm 대신 **LayerNorm**
- ReLU 대신 **GELU**
- **더 적은 활성화 함수**: 블록당 하나만

### 2.3 PyTorch 구현

```python
import torch
import torch.nn as nn

class ConvNeXtBlock(nn.Module):
    """ConvNeXt block with inverted bottleneck design."""

    def __init__(self, dim, expansion_ratio=4, kernel_size=7, layer_scale_init=1e-6):
        super().__init__()

        # Depthwise convolution
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=dim
        )

        # Normalization and projection
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, expansion_ratio * dim)  # Expansion
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expansion_ratio * dim, dim)  # Projection

        # Layer scale (learned per-channel scaling)
        self.gamma = nn.Parameter(
            layer_scale_init * torch.ones(dim)
        ) if layer_scale_init > 0 else None

    def forward(self, x):
        shortcut = x

        # Depthwise conv
        x = self.dwconv(x)

        # Permute for LayerNorm and pointwise convs
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

        # Inverted bottleneck with LayerNorm
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        # Layer scale
        if self.gamma is not None:
            x = self.gamma * x

        # Permute back
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        # Residual connection
        x = shortcut + x
        return x


class ConvNeXt(nn.Module):
    """ConvNeXt model."""

    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        depths=[3, 3, 9, 3],  # Number of blocks per stage
        dims=[96, 192, 384, 768],  # Channels per stage
        **kwargs
    ):
        super().__init__()

        # Stem: patchify with 4×4 conv, stride 4
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            nn.LayerNorm(dims[0], eps=1e-6, elementwise_affine=True)
        )

        # Build 4 stages
        self.stages = nn.ModuleList()
        for i in range(4):
            # Downsampling layer (except first stage)
            if i > 0:
                downsample = nn.Sequential(
                    nn.LayerNorm(dims[i-1], eps=1e-6),
                    nn.Conv2d(dims[i-1], dims[i], kernel_size=2, stride=2)
                )
            else:
                downsample = nn.Identity()

            # Stack ConvNeXt blocks
            blocks = nn.Sequential(*[
                ConvNeXtBlock(dims[i], **kwargs) for _ in range(depths[i])
            ])

            stage = nn.Sequential(downsample, blocks)
            self.stages.append(stage)

        # Head
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

    def forward_features(self, x):
        # Stem
        x = self.stem(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

        # Stages
        for stage in self.stages:
            x = x.permute(0, 3, 1, 2)  # -> (N, C, H, W)
            x = stage(x)
            x = x.permute(0, 2, 3, 1)  # -> (N, H, W, C)

        return self.norm(x.mean([1, 2]))  # Global average pooling

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


# Example usage
model = ConvNeXt(
    depths=[3, 3, 9, 3],  # ConvNeXt-T
    dims=[96, 192, 384, 768]
)

x = torch.randn(2, 3, 224, 224)
output = model(x)
print(f"Output shape: {output.shape}")  # (2, 1000)
```

### 2.4 ConvNeXt V2 개선사항 (2023)

**ConvNeXt V2**는 다음을 도입했습니다:
1. **전역 응답 정규화(Global Response Normalization, GRN)**: 채널 간 특징 경쟁 강화
2. **완전 합성곱 MAE(Fully convolutional MAE)**: ConvNet을 위한 마스크 오토인코더 사전학습
3. **성능 향상**: ImageNet-1K에서 87.3% (ConvNeXt V2-H)

```python
class GRN(nn.Module):
    """Global Response Normalization layer."""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        # x: (N, H, W, C)
        # Compute global feature map
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        # Normalize
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        # Scale and shift
        return self.gamma * (x * Nx) + self.beta + x
```

---

## 3. EfficientNetV2

**EfficientNetV2** (Tan & Le, 2021)는 다음을 통해 학습 속도와 파라미터 효율성을 개선합니다:
1. **Fused-MBConv 블록**: 확장(expansion)과 depthwise 합성곱 융합
2. **점진적 학습(Progressive training)**: 이미지 크기와 정규화를 점진적으로 증가
3. **신경망 아키텍처 탐색(Neural Architecture Search, NAS)**: 학습 속도에 최적화

### 3.1 Fused-MBConv vs. MBConv

```
MBConv (MobileNetV2):                Fused-MBConv:
  Input                                Input
    |                                    |
  1×1 Conv (expand)                    3×3 Conv (expand)
    |                                    |
  DW 3×3                               [Fused operation]
    |                                    |
  1×1 Conv (project)                   1×1 Conv (project)
    |                                    |
  Output                               Output

  3 separate ops                       2 ops (faster for small FLOPs)
```

**트레이드오프**:
- **MBConv**: 더 큰 모델에 적합 (더 적은 파라미터)
- **Fused-MBConv**: 더 작은 모델에 적합 (더 빠른 학습)

EfficientNetV2는 서로 다른 단계에서 **둘 다** 사용합니다.

### 3.2 점진적 학습

**핵심 아이디어**: 처음에는 더 작은 이미지와 약한 정규화로 학습하고, 점진적으로 증가시킵니다.

```
Stage 1 (epochs 0-50):
  - Image size: 128×128
  - RandAugment magnitude: 5
  - Mixup alpha: 0

Stage 2 (epochs 50-100):
  - Image size: 192×192
  - RandAugment magnitude: 10
  - Mixup alpha: 0.2

Stage 3 (epochs 100-150):
  - Image size: 256×256
  - RandAugment magnitude: 15
  - Mixup alpha: 0.4
```

**이점**:
- **더 빠른 수렴**: 더 작은 이미지로 최적화가 더 쉬움
- **더 나은 정규화**: 더 큰 이미지에 더 강력한 증강
- **향상된 정확도**: ImageNet에서 85.7% (EfficientNetV2-L)

### 3.3 timm으로 EfficientNetV2 사용하기

```python
import timm
import torch

# List available EfficientNetV2 models
models = timm.list_models('*efficientnetv2*', pretrained=True)
print(models)
# ['tf_efficientnetv2_b0', 'tf_efficientnetv2_b1', ..., 'tf_efficientnetv2_l']

# Load pretrained EfficientNetV2-S
model = timm.create_model('tf_efficientnetv2_s', pretrained=True)
model.eval()

# Get model info
print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
print(f"Input size: {model.default_cfg['input_size']}")

# Inference
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

from PIL import Image
img = Image.open('cat.jpg')
x = transforms(img).unsqueeze(0)  # (1, 3, 384, 384)

with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    top5_idx = torch.topk(probs, 5).indices[0]

# Print top-5 predictions
labels = timm.data.ImageNetInfo.label_names()
for idx in top5_idx:
    print(f"{labels[idx]}: {probs[0, idx]:.3f}")
```

---

## 4. DINOv2: 자기지도 학습 비전 파운데이션 모델

**DINOv2** (Oquab et al., 2023)는 레이블 없이 1억 4200만 이미지로 사전학습된 자기지도 학습 Vision Transformer입니다.

### 4.1 주요 혁신

1. **레이블 없는 자기 증류(Self-distillation with no labels)** (DINO 프레임워크)
2. **레지스터 토큰을 가진 ViT 백본**
3. **대규모 사전학습** (1억 4200만 이미지, LVD-142M 데이터셋)
4. **멀티태스크 헤드**: 분류, 분할, 깊이 추정

```
DINO Self-Distillation:

   Student (ViT-S)          Teacher (EMA of Student)
         |                          |
    [CLS] token               [CLS] token
         |                          |
    ┌─────────┐              ┌─────────┐
    │ Predict │              │ Target  │
    └─────────┘              └─────────┘
         |                          |
         └──────── Match ───────────┘
              (no labels!)

Augmentations:
  - Student: strong crops (multi-crop)
  - Teacher: weak crops (global views)
```

2. **레지스터 토큰(Register tokens)** (추가 학습 가능 토큰):
   - 배경 아티팩트를 흡수하여 특징 품질 향상
   - [CLS] 토큰과 유사하지만 분류에는 사용되지 않음

3. **동결된 백본 + 선형 프로브(linear probes)**:
   - 동결된 DINOv2로 특징 추출
   - 다운스트림 태스크를 위한 경량 헤드 학습

### 4.2 모델 변형

| 모델 | 파라미터 | 레이어 | Hidden Dim | Patch Size |
|-------|--------|--------|------------|------------|
| DINOv2-S | 22M | 12 | 384 | 14×14 |
| DINOv2-B | 86M | 12 | 768 | 14×14 |
| DINOv2-L | 304M | 24 | 1024 | 14×14 |
| DINOv2-g | 1.1B | 40 | 1536 | 14×14 |

### 4.3 사전학습된 DINOv2 사용하기

```python
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

# Load pretrained DINOv2-base
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base')
model.eval()

# Load image
img = Image.open('cat.jpg')

# Extract features
inputs = processor(images=img, return_tensors='pt')
with torch.no_grad():
    outputs = model(**inputs)

# Get patch embeddings (excluding [CLS])
patch_embeddings = outputs.last_hidden_state[:, 1:, :]  # (1, num_patches, 768)
print(f"Patch embeddings shape: {patch_embeddings.shape}")

# Get [CLS] token (global image representation)
cls_token = outputs.last_hidden_state[:, 0, :]  # (1, 768)
print(f"CLS token shape: {cls_token.shape}")

# Use as feature extractor for downstream tasks
# Example: k-NN classification
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Assume we have a training set
train_features = []  # Extract from training images
train_labels = []

# Fit k-NN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(np.array(train_features), train_labels)

# Predict
pred = knn.predict(cls_token.numpy())
```

### 4.4 DINOv2를 사용한 다운스트림 태스크

**1. 이미지 분류(Image Classification)**
```python
from transformers import Dinov2ForImageClassification

model = Dinov2ForImageClassification.from_pretrained(
    'facebook/dinov2-base',
    num_labels=10,  # Custom dataset
    ignore_mismatched_sizes=True
)

# Fine-tune on custom dataset
# ... training loop ...
```

**2. 의미론적 분할(Semantic Segmentation)**
```python
# Use patch embeddings for dense prediction
B, N, D = patch_embeddings.shape
H = W = int(N ** 0.5)  # Assume square

# Reshape to spatial grid
spatial_features = patch_embeddings.reshape(B, H, W, D)
spatial_features = spatial_features.permute(0, 3, 1, 2)  # (B, D, H, W)

# Add segmentation head
seg_head = nn.Conv2d(D, num_classes, kernel_size=1)
logits = seg_head(spatial_features)  # (B, num_classes, H, W)
```

**3. 깊이 추정(Depth Estimation)**
```python
# Similar to segmentation, but regress depth
depth_head = nn.Sequential(
    nn.Conv2d(D, 256, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(256, 1, kernel_size=1)
)
depth_map = depth_head(spatial_features)  # (B, 1, H, W)
```

---

## 5. 잠재 일관성 모델(Latent Consistency Models, LCM)

**잠재 일관성 모델(Latent Consistency Models)** (Luo et al., 2023)은 확산 모델에서 1-4 단계로 빠른 샘플링을 가능하게 합니다 (표준 확산의 25-50 단계 vs.).

### 5.1 일관성 증류(Consistency Distillation)

**핵심 아이디어**: 사전학습된 확산 모델을 노이즈가 있는 잠재 변수를 깨끗한 잠재 변수로 직접 매핑하는 일관성 모델로 증류합니다.

```
Standard Diffusion (DDPM):
  x_T (noise) → x_{T-1} → ... → x_1 → x_0 (clean)
  (50 steps, slow)

Latent Consistency Model:
  x_T (noise) ───────────────────────→ x_0 (clean)
  (1-4 steps, fast!)

Consistency property:
  For any t, t' ∈ [0, T]:
    f(x_t, t) ≈ f(x_{t'}, t')
  (all noisy latents map to same clean latent)
```

### 5.2 LCM 학습

1. **사전학습된 확산 모델로 시작** (예: Stable Diffusion)
2. **일관성 손실을 사용하여 LCM으로 증류**:

```
Consistency loss:
  L = E_{x, t, t'} [ || f(x_t, t) - sg(f(x_{t'}, t')) ||^2 ]

  where:
    - x_t, x_{t'} are noisy latents at different timesteps
    - f is the consistency model (student)
    - sg is stop-gradient (teacher is EMA of student)
```

3. **몇 단계 샘플링**: 2-4 단계로 ODE 솔버 사용 (예: DDIM)

### 5.3 빠른 파인튜닝을 위한 LCM-LoRA

**LCM-LoRA**는 일관성 증류에 저랭크 적응(Low-Rank Adaptation)을 적용합니다:
- **더 빠른 학습**: LoRA 가중치만 학습 (~파라미터의 1-5%)
- **조합 가능(Composable)**: 다른 LoRA(스타일, 캐릭터 등)와 결합
- **효율적**: 단일 GPU에서 증류 가능

### 5.4 Diffusers로 LCM 사용하기

```python
from diffusers import DiffusionPipeline, LCMScheduler
import torch

# Load LCM pipeline
pipe = DiffusionPipeline.from_pretrained(
    "SimianLuo/LCM_Dreamshaper_v7",
    torch_dtype=torch.float16
)
pipe.to("cuda")

# LCM uses special scheduler
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

# Generate with 4 steps (vs. 50 for standard diffusion!)
prompt = "A beautiful sunset over mountains, highly detailed, 8k"
image = pipe(
    prompt=prompt,
    num_inference_steps=4,  # Very fast!
    guidance_scale=1.0,  # LCM works best with guidance_scale=1
).images[0]

image.save("sunset_lcm.png")
```

**LCM-LoRA 사용하기**:
```python
from diffusers import StableDiffusionPipeline, LCMScheduler

# Load base model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)

# Load LCM-LoRA weights
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

# Generate with 4-8 steps
image = pipe(
    prompt="Portrait of a cat, oil painting",
    num_inference_steps=8,
    guidance_scale=1.0
).images[0]
```

---

## 6. 아키텍처 비교 표

| 아키텍처 | 파라미터 | FLOPs (G) | ImageNet 정확도 | 학습 데이터 | 사전학습 방법 |
|--------------|--------|-----------|--------------|---------------|-------------------|
| ResNet-50 | 25M | 4.1 | 76.2% | 1.3M | Supervised |
| EfficientNet-B4 | 19M | 4.5 | 82.9% | 1.3M | Supervised + AutoAug |
| EfficientNetV2-S | 24M | 8.4 | 84.9% | 1.3M | Supervised + Progressive |
| ViT-B/16 | 86M | 17.6 | 84.5% | 300M | Supervised (JFT-300M) |
| Swin-B | 88M | 15.4 | 85.2% | 1.3M | Supervised |
| ConvNeXt-B | 89M | 15.4 | 85.8% | 1.3M | Supervised |
| ConvNeXt V2-B | 89M | 15.4 | 86.8% | 1.3M | FCMAE (self-supervised) |
| DINOv2-B | 86M | 17.6 | 84.5% (linear) | 142M | Self-supervised (DINO) |
| DINOv2-g | 1.1B | 280 | 88.5% (linear) | 142M | Self-supervised (DINO) |

**참고사항**:
- **FLOPs**: 224×224 해상도에서 측정
- **ImageNet Acc**: ImageNet-1K 검증 세트에서 Top-1 정확도
- **DINOv2 (linear)**: 선형 프로브 평가 (동결된 특징 + 선형 분류기)

---

## 7. 사전학습된 모델 사용하기: 실용 가이드

### 7.1 timm 라이브러리 (PyTorch Image Models)

**timm**은 통합된 인터페이스로 700개 이상의 사전학습된 모델을 제공합니다.

```python
import timm
import torch

# List all models
all_models = timm.list_models(pretrained=True)
print(f"Total models: {len(all_models)}")

# Search for specific architecture
convnext_models = timm.list_models('convnext*', pretrained=True)
print(convnext_models)

# Create model
model = timm.create_model(
    'convnext_base.fb_in22k_ft_in1k',  # Pretrained on ImageNet-22k, fine-tuned on 1k
    pretrained=True,
    num_classes=1000
)

# Inspect model
print(model.default_cfg)  # Config dict
print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

# Feature extraction mode
model = timm.create_model('convnext_base', pretrained=True, num_classes=0)
# Returns features instead of logits

# Get intermediate features
model = timm.create_model('convnext_base', pretrained=True, features_only=True)
x = torch.randn(1, 3, 224, 224)
features = model(x)
for i, feat in enumerate(features):
    print(f"Stage {i}: {feat.shape}")
# Stage 0: (1, 128, 56, 56)
# Stage 1: (1, 256, 28, 28)
# Stage 2: (1, 512, 14, 14)
# Stage 3: (1, 1024, 7, 7)
```

### 7.2 Hugging Face Transformers

**transformers** 라이브러리는 AutoModel을 통해 비전 모델을 지원합니다.

```python
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

# Load processor and model
processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224")
model = AutoModelForImageClassification.from_pretrained("microsoft/swin-base-patch4-window7-224")

# Prepare input
from PIL import Image
img = Image.open("cat.jpg")
inputs = processor(images=img, return_tensors="pt")

# Inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(-1).item()

print(f"Predicted class: {model.config.id2label[predicted_class]}")
```

### 7.3 전이 학습 모범 사례

**1. 특징 추출(Feature Extraction)**:
```python
# Freeze pretrained weights
model = timm.create_model('convnext_base', pretrained=True)
for param in model.parameters():
    param.requires_grad = False

# Replace classifier head
num_classes = 10
model.head = torch.nn.Linear(model.head.in_features, num_classes)

# Only train the head
optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-3)
```

**2. 파인튜닝(Fine-tuning)**:
```python
# Unfreeze all layers
for param in model.parameters():
    param.requires_grad = True

# Use lower learning rate for pretrained weights
optimizer = torch.optim.AdamW([
    {'params': model.stem.parameters(), 'lr': 1e-5},
    {'params': model.stages.parameters(), 'lr': 5e-5},
    {'params': model.head.parameters(), 'lr': 1e-3}
])
```

**3. 점진적 동결 해제(Progressive unfreezing)** (ULMFiT 전략):
```python
# Epoch 0-5: Train head only
# Epoch 5-10: Unfreeze last stage
# Epoch 10+: Unfreeze all

def unfreeze_layers(model, epoch):
    if epoch < 5:
        # Freeze all except head
        for param in model.stem.parameters():
            param.requires_grad = False
        for param in model.stages.parameters():
            param.requires_grad = False
    elif epoch < 10:
        # Unfreeze last stage
        for param in model.stages[-1].parameters():
            param.requires_grad = True
    else:
        # Unfreeze all
        for param in model.parameters():
            param.requires_grad = True
```

---

## 8. 아키텍처 선택 가이드

### 8.1 의사 결정 트리

```
┌─ Need supervised pretraining?
│  ├─ Yes
│  │  ├─ Priority: Accuracy
│  │  │  └─ ConvNeXt V2, EfficientNetV2-L, Swin-L
│  │  └─ Priority: Speed
│  │     └─ EfficientNetV2-S, MobileNetV3
│  └─ No (self-supervised)
│     ├─ Vision foundation model
│     │  └─ DINOv2-L/g (best features)
│     └─ Custom dataset
│        └─ DINO, MAE, SimCLR

┌─ Need generative model?
│  ├─ Fast sampling (1-4 steps)
│  │  └─ Latent Consistency Models
│  └─ Best quality (25-50 steps)
│     └─ Stable Diffusion, DALL-E 3

┌─ Deployment constraints?
│  ├─ Edge device (mobile, IoT)
│  │  └─ MobileNetV3, EfficientNet-B0
│  ├─ Low latency (< 10ms)
│  │  └─ ConvNeXt-T, EfficientNetV2-S
│  └─ No constraints
│     └─ Any large model
```

### 8.2 실용적 권장사항

**범용 비전 태스크** (분류, 탐지, 분할):
- **DINOv2**: 퓨샷 학습(few-shot learning)을 위한 최고의 동결된 특징
- **ConvNeXt V2**: 최고의 파인튜닝 성능
- **EfficientNetV2**: 최고의 속도-정확도 트레이드오프

**생성 태스크** (이미지 합성):
- **Stable Diffusion XL**: 최고 품질 (50 단계)
- **LCM**: 최고 속도 (4 단계)
- **LCM-LoRA**: 최고 커스터마이징

**자원 제약**:
- **MobileNetV3**: 모바일 배포
- **EfficientNet-B0/B1**: 엣지 디바이스에서 좋은 정확도

---

## 9. 연습 문제

### 문제 1: ConvNeXt 블록 구현
제공된 코드를 사용하지 않고 처음부터 ConvNeXt 블록을 구현하세요. 다음을 포함해야 합니다:
- Depthwise 7×7 합성곱
- LayerNorm
- Inverted bottleneck (4× 확장을 가진 1×1 합성곱)
- GELU 활성화
- Layer scale
- Residual connection

입력 shape `(2, 64, 32, 32)`로 테스트하세요.

<details>
<summary>솔루션</summary>

```python
import torch
import torch.nn as nn

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, expansion_ratio=4, layer_scale_init=1e-6):
        super().__init__()

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, expansion_ratio * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expansion_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return shortcut + x

# Test
block = ConvNeXtBlock(64)
x = torch.randn(2, 64, 32, 32)
out = block(x)
assert out.shape == (2, 64, 32, 32)
print("ConvNeXt block test passed!")
```
</details>

### 문제 2: 점진적 학습 스케줄
다음을 수행하는 EfficientNetV2를 위한 점진적 학습 스케줄러를 구현하세요:
- 이미지 크기를 128 → 192 → 256으로 증가
- RandAugment magnitude를 5 → 10 → 15로 증가
- Mixup alpha를 0 → 0.2 → 0.4로 증가
- 각 단계는 50 에폭 지속

<details>
<summary>솔루션</summary>

```python
class ProgressiveTrainingScheduler:
    def __init__(self, total_epochs=150):
        self.total_epochs = total_epochs
        self.stages = [
            {'epochs': (0, 50), 'img_size': 128, 'rand_aug_mag': 5, 'mixup_alpha': 0.0},
            {'epochs': (50, 100), 'img_size': 192, 'rand_aug_mag': 10, 'mixup_alpha': 0.2},
            {'epochs': (100, 150), 'img_size': 256, 'rand_aug_mag': 15, 'mixup_alpha': 0.4},
        ]

    def get_config(self, epoch):
        for stage in self.stages:
            if stage['epochs'][0] <= epoch < stage['epochs'][1]:
                return {
                    'img_size': stage['img_size'],
                    'rand_aug_mag': stage['rand_aug_mag'],
                    'mixup_alpha': stage['mixup_alpha']
                }
        return self.stages[-1]  # Return last stage config

    def __call__(self, epoch):
        return self.get_config(epoch)

# Usage
scheduler = ProgressiveTrainingScheduler()
for epoch in [0, 25, 50, 75, 100, 125]:
    config = scheduler(epoch)
    print(f"Epoch {epoch}: img_size={config['img_size']}, "
          f"rand_aug={config['rand_aug_mag']}, mixup={config['mixup_alpha']}")
```
</details>

### 문제 3: DINOv2 특징 추출
DINOv2에서 패치 레벨 특징을 추출하고 코사인 유사도를 사용하여 특징 유사도를 시각화하세요.
1. DINOv2-small 로드
2. 이미지의 패치 임베딩 추출
3. 패치 간 쌍별 코사인 유사도 계산
4. 히트맵으로 시각화

<details>
<summary>솔루션</summary>

```python
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Load model
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
model = AutoModel.from_pretrained('facebook/dinov2-small')
model.eval()

# Load image
img = Image.open('cat.jpg')
inputs = processor(images=img, return_tensors='pt')

# Extract features
with torch.no_grad():
    outputs = model(**inputs)
    patch_embeddings = outputs.last_hidden_state[:, 1:, :]  # Exclude [CLS]

# Reshape to spatial grid
B, N, D = patch_embeddings.shape
H = W = int(N ** 0.5)
patches = patch_embeddings.reshape(B, H, W, D)[0]  # (H, W, D)

# Compute cosine similarity
patches_flat = patches.reshape(-1, D)  # (H*W, D)
# Normalize
patches_norm = patches_flat / patches_flat.norm(dim=1, keepdim=True)
# Cosine similarity matrix
sim_matrix = patches_norm @ patches_norm.T  # (H*W, H*W)

# Visualize
plt.figure(figsize=(10, 10))
plt.imshow(sim_matrix.numpy(), cmap='viridis')
plt.colorbar(label='Cosine Similarity')
plt.title('Patch-level Feature Similarity (DINOv2)')
plt.xlabel('Patch index')
plt.ylabel('Patch index')
plt.tight_layout()
plt.savefig('dinov2_similarity.png')
```
</details>

### 문제 4: LCM 빠른 생성
표준 DDIM (50 단계)과 LCM (4 단계) 간의 생성 속도와 품질을 비교하세요:
1. Stable Diffusion 1.5 로드
2. DDIM으로 생성 (50 단계)
3. LCM-LoRA 로드
4. LCM으로 생성 (4 단계)
5. 두 경우의 시간 측정

<details>
<summary>솔루션</summary>

```python
from diffusers import StableDiffusionPipeline, LCMScheduler
import torch
import time

# Load base model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipe.to("cuda")

prompt = "A serene lake with mountains in background, sunset, highly detailed"

# Standard DDIM
print("Generating with DDIM (50 steps)...")
start = time.time()
image_ddim = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
ddim_time = time.time() - start
print(f"DDIM time: {ddim_time:.2f}s")
image_ddim.save("ddim_50steps.png")

# Load LCM-LoRA
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

# LCM generation
print("Generating with LCM (4 steps)...")
start = time.time()
image_lcm = pipe(prompt, num_inference_steps=4, guidance_scale=1.0).images[0]
lcm_time = time.time() - start
print(f"LCM time: {lcm_time:.2f}s")
image_lcm.save("lcm_4steps.png")

# Speed comparison
speedup = ddim_time / lcm_time
print(f"\nSpeedup: {speedup:.1f}x faster with LCM")
```
</details>

### 문제 5: 모델 비교
사용자 정의 데이터셋에서 ConvNeXt-T, EfficientNetV2-S, DINOv2-S를 비교하세요:
1. timm/transformers에서 세 모델 모두 로드
2. 학습 세트에 대한 특징 추출 (동결)
3. 특징에 대해 선형 SVM 학습
4. 정확도 및 추론 시간 보고

<details>
<summary>솔루션</summary>

```python
import timm
import torch
from transformers import AutoImageProcessor, AutoModel
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import numpy as np
import time

# Assume we have a dataset loader
train_loader = ...  # DataLoader for training set
test_loader = ...   # DataLoader for test set

def extract_features(model, loader, is_dinov2=False):
    features, labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.cuda()
            if is_dinov2:
                # DINOv2 uses different interface
                outputs = model(imgs)
                feats = outputs.last_hidden_state[:, 0, :]  # [CLS]
            else:
                feats = model(imgs)  # timm feature extractor
            features.append(feats.cpu().numpy())
            labels.append(lbls.numpy())
    return np.concatenate(features), np.concatenate(labels)

# 1. ConvNeXt-T
print("Loading ConvNeXt-T...")
convnext = timm.create_model('convnext_tiny', pretrained=True, num_classes=0)
convnext = convnext.cuda().eval()

start = time.time()
train_feats_cn, train_labels = extract_features(convnext, train_loader)
test_feats_cn, test_labels = extract_features(convnext, test_loader)
cn_time = time.time() - start

# 2. EfficientNetV2-S
print("Loading EfficientNetV2-S...")
effnet = timm.create_model('tf_efficientnetv2_s', pretrained=True, num_classes=0)
effnet = effnet.cuda().eval()

start = time.time()
train_feats_eff, _ = extract_features(effnet, train_loader)
test_feats_eff, _ = extract_features(effnet, test_loader)
eff_time = time.time() - start

# 3. DINOv2-S
print("Loading DINOv2-S...")
dinov2 = AutoModel.from_pretrained('facebook/dinov2-small')
dinov2 = dinov2.cuda().eval()

start = time.time()
train_feats_dino, _ = extract_features(dinov2, train_loader, is_dinov2=True)
test_feats_dino, _ = extract_features(dinov2, test_loader, is_dinov2=True)
dino_time = time.time() - start

# Train linear SVM on each
for name, train_feats, test_feats, infer_time in [
    ('ConvNeXt-T', train_feats_cn, test_feats_cn, cn_time),
    ('EfficientNetV2-S', train_feats_eff, test_feats_eff, eff_time),
    ('DINOv2-S', train_feats_dino, test_feats_dino, dino_time)
]:
    svm = LinearSVC(max_iter=10000)
    svm.fit(train_feats, train_labels)
    preds = svm.predict(test_feats)
    acc = accuracy_score(test_labels, preds)
    print(f"{name}: Accuracy={acc:.3f}, Inference time={infer_time:.2f}s")
```
</details>

---

## 네비게이션

- **이전**: [26. 정규화 레이어(Normalization Layers)](26_Normalization_Layers.md)
- **다음**: [개요](00_Overview.md)

---

## 추가 자료

- **ConvNeXt**: [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545) (Liu et al., 2022)
- **ConvNeXt V2**: [Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/abs/2301.00808) (Woo et al., 2023)
- **EfficientNetV2**: [Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298) (Tan & Le, 2021)
- **DINOv2**: [Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) (Oquab et al., 2023)
- **Latent Consistency Models**: [Synthesizing High-Resolution Images with Few-Step Inference](https://arxiv.org/abs/2310.04378) (Luo et al., 2023)
- **timm Documentation**: https://timm.fast.ai/
- **Hugging Face Models**: https://huggingface.co/models
