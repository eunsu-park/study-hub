[Previous: Self-Supervised Learning](./36_Self_Supervised_Learning.md) | [Next: Object Detection](./38_Object_Detection.md)

---

# 37. Modern Deep Learning Architectures

## Learning Objectives

- Survey recent architectural innovations in deep learning (2020-2024)
- Understand ConvNeXt and the evolution of pure ConvNets in the Transformer era
- Learn about EfficientNetV2 and progressive training strategies
- Explore DINOv2 as a self-supervised vision foundation model
- Understand Latent Consistency Models (LCM) for fast diffusion sampling
- Apply pretrained modern architectures using timm and transformers libraries

---

## 1. Architecture Evolution Timeline

The landscape of deep learning architectures has evolved rapidly:

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

### Key Trends

1. **Hybrid architectures**: Combining convolutions and attention
2. **Self-supervised pretraining**: DINO, MAE, CLIP
3. **Scaling laws**: Bigger models, more data, longer training
4. **Efficiency**: Reducing FLOPs, parameters, and latency
5. **Foundation models**: General-purpose pretrained models

---

## 2. ConvNeXt: Modernizing ConvNets

**ConvNeXt** (Liu et al., 2022) demonstrates that pure ConvNets can match Transformers when modernized with recent design choices.

### 2.1 Design Evolution from ResNet to ConvNeXt

Starting from ResNet-50, apply modern improvements step-by-step:

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

### 2.2 ConvNeXt Block Architecture

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

**Key differences from ResNet**:
- **Depthwise convolution** (7×7) instead of 3×3 standard conv
- **Inverted bottleneck**: expand to 4C, then project back to C
- **LayerNorm** instead of BatchNorm
- **GELU** instead of ReLU
- **Fewer activation functions**: only one per block

### 2.3 PyTorch Implementation

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

### 2.4 ConvNeXt V2 Improvements (2023)

**ConvNeXt V2** introduced:
1. **Global Response Normalization (GRN)**: enhance inter-channel feature competition
2. **Fully convolutional MAE**: masked autoencoder pretraining for ConvNets
3. **Improved performance**: 87.3% on ImageNet-1K (ConvNeXt V2-H)

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

**EfficientNetV2** (Tan & Le, 2021) improves training speed and parameter efficiency through:
1. **Fused-MBConv blocks**: fused expansion and depthwise convolution
2. **Progressive training**: gradually increase image size and regularization
3. **Neural Architecture Search (NAS)**: optimized for training speed

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

**Trade-off**:
- **MBConv**: Better for larger models (fewer parameters)
- **Fused-MBConv**: Better for smaller models (faster training)

EfficientNetV2 uses **both** in different stages.

### 3.2 Progressive Training

**Key idea**: Train with smaller images and weaker regularization initially, then gradually increase.

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

**Benefits**:
- **Faster convergence**: Easier to optimize with smaller images
- **Better regularization**: Stronger augmentation on larger images
- **Improved accuracy**: 85.7% on ImageNet (EfficientNetV2-L)

### 3.3 Using EfficientNetV2 with timm

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

## 4. DINOv2: Self-Supervised Vision Foundation Model

**DINOv2** (Oquab et al., 2023) is a self-supervised Vision Transformer pretrained on 142M images without labels.

### 4.1 Key Innovations

1. **Self-distillation with no labels** (DINO framework)
2. **ViT backbone** with register tokens
3. **Large-scale pretraining** (142M images, LVD-142M dataset)
4. **Multi-task head**: classification, segmentation, depth estimation

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

2. **Register tokens** (additional learnable tokens):
   - Improve feature quality by absorbing background artifacts
   - Similar to [CLS] token but not used for classification

3. **Frozen backbone + linear probes**:
   - Extract features with frozen DINOv2
   - Train lightweight heads for downstream tasks

### 4.2 Model Variants

| Model | Params | Layers | Hidden Dim | Patch Size |
|-------|--------|--------|------------|------------|
| DINOv2-S | 22M | 12 | 384 | 14×14 |
| DINOv2-B | 86M | 12 | 768 | 14×14 |
| DINOv2-L | 304M | 24 | 1024 | 14×14 |
| DINOv2-g | 1.1B | 40 | 1536 | 14×14 |

### 4.3 Using Pretrained DINOv2

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

### 4.4 Downstream Tasks with DINOv2

**1. Image Classification**
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

**2. Semantic Segmentation**
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

**3. Depth Estimation**
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

## 5. Latent Consistency Models (LCM)

**Latent Consistency Models** (Luo et al., 2023) enable fast sampling from diffusion models in 1-4 steps (vs. 25-50 steps for standard diffusion).

### 5.1 Consistency Distillation

**Key idea**: Distill a pretrained diffusion model into a consistency model that maps any noisy latent directly to the clean latent.

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

### 5.2 LCM Training

1. **Start with pretrained diffusion model** (e.g., Stable Diffusion)
2. **Distill into LCM** using consistency loss:

```
Consistency loss:
  L = E_{x, t, t'} [ || f(x_t, t) - sg(f(x_{t'}, t')) ||^2 ]

  where:
    - x_t, x_{t'} are noisy latents at different timesteps
    - f is the consistency model (student)
    - sg is stop-gradient (teacher is EMA of student)
```

3. **Few-step sampling**: Use ODE solver (e.g., DDIM) with 2-4 steps

### 5.3 LCM-LoRA for Fast Fine-tuning

**LCM-LoRA** applies Low-Rank Adaptation to consistency distillation:
- **Faster training**: Only train LoRA weights (~1-5% of parameters)
- **Composable**: Combine with other LoRAs (style, character, etc.)
- **Efficient**: Can distill on single GPU

### 5.4 Using LCM with Diffusers

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

**Using LCM-LoRA**:
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

## 6. Architecture Comparison Table

| Architecture | Params | FLOPs (G) | ImageNet Acc | Training Data | Pretraining Method |
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

**Notes**:
- **FLOPs**: Measured at 224×224 resolution
- **ImageNet Acc**: Top-1 accuracy on ImageNet-1K validation set
- **DINOv2 (linear)**: Linear probe evaluation (frozen features + linear classifier)

---

## 7. Using Pretrained Models: Practical Guide

### 7.1 timm Library (PyTorch Image Models)

**timm** provides 700+ pretrained models with unified interface.

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

**transformers** library supports vision models via AutoModel.

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

### 7.3 Transfer Learning Best Practices

**1. Feature Extraction**:
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

**2. Fine-tuning**:
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

**3. Progressive unfreezing** (ULMFiT strategy):
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

## 8. Architecture Selection Guide

### 8.1 Decision Tree

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

### 8.2 Practical Recommendations

**General-purpose vision tasks** (classification, detection, segmentation):
- **DINOv2**: Best frozen features for few-shot learning
- **ConvNeXt V2**: Best fine-tuning performance
- **EfficientNetV2**: Best speed-accuracy trade-off

**Generative tasks** (image synthesis):
- **Stable Diffusion XL**: Best quality (50 steps)
- **LCM**: Best speed (4 steps)
- **LCM-LoRA**: Best customization

**Resource-constrained**:
- **MobileNetV3**: Mobile deployment
- **EfficientNet-B0/B1**: Good accuracy on edge devices

---

## 9. Practice Problems

### Problem 1: ConvNeXt Block Implementation
Implement a ConvNeXt block from scratch without using the provided code. Include:
- Depthwise 7×7 convolution
- LayerNorm
- Inverted bottleneck (1×1 conv with 4× expansion)
- GELU activation
- Layer scale
- Residual connection

Test with input shape `(2, 64, 32, 32)`.

<details>
<summary>Solution</summary>

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

### Problem 2: Progressive Training Schedule
Implement a progressive training scheduler for EfficientNetV2 that:
- Increases image size from 128 → 192 → 256
- Increases RandAugment magnitude from 5 → 10 → 15
- Increases Mixup alpha from 0 → 0.2 → 0.4
- Each stage lasts 50 epochs

<details>
<summary>Solution</summary>

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

### Problem 3: DINOv2 Feature Extraction
Extract patch-level features from DINOv2 and visualize feature similarity using cosine similarity.
1. Load DINOv2-small
2. Extract patch embeddings for an image
3. Compute pairwise cosine similarity between patches
4. Visualize as heatmap

<details>
<summary>Solution</summary>

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

### Problem 4: LCM Fast Generation
Compare generation speed and quality between standard DDIM (50 steps) and LCM (4 steps):
1. Load Stable Diffusion 1.5
2. Generate with DDIM (50 steps)
3. Load LCM-LoRA
4. Generate with LCM (4 steps)
5. Measure time for both

<details>
<summary>Solution</summary>

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

### Problem 5: Model Comparison
Compare ConvNeXt-T, EfficientNetV2-S, and DINOv2-S on a custom dataset:
1. Load all three models from timm/transformers
2. Extract features (frozen) for training set
3. Train linear SVM on features
4. Report accuracy and inference time

<details>
<summary>Solution</summary>

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

## Navigation

- **Previous**: [26. Normalization Layers](26_Normalization_Layers.md)
- **Next**: [Overview](00_Overview.md)

---

## Further Reading

- **ConvNeXt**: [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545) (Liu et al., 2022)
- **ConvNeXt V2**: [Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/abs/2301.00808) (Woo et al., 2023)
- **EfficientNetV2**: [Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298) (Tan & Le, 2021)
- **DINOv2**: [Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) (Oquab et al., 2023)
- **Latent Consistency Models**: [Synthesizing High-Resolution Images with Few-Step Inference](https://arxiv.org/abs/2310.04378) (Luo et al., 2023)
- **timm Documentation**: https://timm.fast.ai/
- **Hugging Face Models**: https://huggingface.co/models
