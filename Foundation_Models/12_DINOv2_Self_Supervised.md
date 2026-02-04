# DINOv2 & Self-Supervised Vision

## 학습 목표
- DINO/DINOv2의 Self-distillation 메커니즘 이해
- Teacher-Student 학습 패러다임 파악
- Dense Visual Features 활용법 습득
- Vision Foundation Model로서의 DINOv2 활용

---

## 1. Self-Supervised Learning in Vision 복습

### 1.1 왜 Self-Supervised인가?

```
┌─────────────────────────────────────────────────────────────────┐
│              Vision에서 Self-Supervised Learning                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Supervised Learning의 한계:                                     │
│  • ImageNet: 1.4M 이미지, 1000 클래스                            │
│  • 레이블링 비용 높음                                             │
│  • 클래스 레이블 = 제한된 정보                                     │
│                                                                 │
│  Self-Supervised Learning:                                       │
│  • 레이블 없이 학습 (pretext task 활용)                           │
│  • 수십억 이미지 활용 가능                                         │
│  • 더 풍부한 표현 학습                                            │
│                                                                 │
│  주요 방법론:                                                     │
│  ┌────────────────────────────────────────────────────┐         │
│  │ Contrastive   │ SimCLR, MoCo  │ 유사/비유사 쌍 학습 │         │
│  │ Distillation  │ DINO, BYOL    │ Teacher-Student    │         │
│  │ Masked        │ MAE, BEiT     │ 마스킹 후 복원      │         │
│  └────────────────────────────────────────────────────┘         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Deep_Learning 폴더 복습

> **선수 지식**: [Deep_Learning/21_Self_Supervised_Learning.md](../Deep_Learning/21_Self_Supervised_Learning.md)
> - SimCLR: Contrastive Learning 기초
> - MoCo: Momentum Contrast
> - BYOL: Bootstrap Your Own Latent
> - MAE: Masked Autoencoders

---

## 2. DINO (2021)

### 2.1 핵심 아이디어

**DINO** (Self-**Di**stillation with **No** labels)는 Knowledge Distillation을 Self-supervised로 적용합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    DINO Architecture                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                        Input Image                              │
│                            │                                    │
│              ┌─────────────┴─────────────┐                      │
│              ▼                           ▼                      │
│     ┌─────────────────┐         ┌─────────────────┐            │
│     │  Global Crops   │         │  Local Crops    │            │
│     │   (224×224)     │         │   (96×96)       │            │
│     │    × 2          │         │    × 6+         │            │
│     └────────┬────────┘         └────────┬────────┘            │
│              │                           │                      │
│              ▼                           ▼                      │
│     ┌─────────────────┐         ┌─────────────────┐            │
│     │ Teacher Network │         │ Student Network │            │
│     │   (EMA update)  │         │   (Gradient)    │            │
│     │   [stop-grad]   │         │                 │            │
│     └────────┬────────┘         └────────┬────────┘            │
│              │                           │                      │
│              ▼                           ▼                      │
│     ┌─────────────────┐         ┌─────────────────┐            │
│     │  Teacher Head   │         │  Student Head   │            │
│     │  (Projection)   │         │  (Projection)   │            │
│     └────────┬────────┘         └────────┬────────┘            │
│              │                           │                      │
│              ▼                           ▼                      │
│          P_teacher                   P_student                  │
│              │                           │                      │
│              └───────────┬───────────────┘                      │
│                          ▼                                      │
│                  Cross-Entropy Loss                             │
│                  H(P_t, P_s) = -Σ P_t log(P_s)                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 주요 구성 요소

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOHead(nn.Module):
    """
    DINO Projection Head

    구조: Linear → GELU → Linear → L2 Norm
    출력: K 차원 (예: 65536)
    """
    def __init__(self, in_dim, out_dim=65536, hidden_dim=2048):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        # L2 정규화
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(out_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

class DINOLoss(nn.Module):
    """
    DINO Loss: Cross-entropy between teacher and student

    특징:
    - Teacher: Centering + Sharpening (temperature τ_t < τ_s)
    - Student: 일반 softmax
    - Center: 모든 teacher 출력의 moving average (collapse 방지)
    """
    def __init__(self, out_dim, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output):
        """
        Args:
            student_output: (batch, n_crops, out_dim)
            teacher_output: (batch, n_global_crops, out_dim)
        """
        # Teacher: centering + sharpening
        teacher_out = F.softmax(
            (teacher_output - self.center) / self.teacher_temp, dim=-1
        )
        teacher_out = teacher_out.detach()  # stop gradient

        # Student: softmax with higher temperature
        student_out = F.log_softmax(student_output / self.student_temp, dim=-1)

        # Cross-entropy loss
        loss = torch.sum(-teacher_out * student_out, dim=-1).mean()

        # Update center (EMA)
        self.update_center(teacher_output)

        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
```

### 2.3 Multi-crop 전략

```python
"""
Multi-crop Strategy:

Global crops (2개):
- 크기: 224×224 (원본의 50-100%)
- Teacher와 Student 모두에 입력
- 전체 이미지 맥락 학습

Local crops (여러 개, 보통 6-8개):
- 크기: 96×96 (원본의 5-50%)
- Student에만 입력
- 지역 패턴 학습

목적:
- "Local-to-Global" 대응 학습
- 작은 영역이 전체 이미지의 어떤 부분인지 학습
- Semantic segmentation 능력 자연스럽게 습득
"""

from torchvision import transforms

class DINODataAugmentation:
    def __init__(self, global_crops_scale=(0.4, 1.0), local_crops_scale=(0.05, 0.4),
                 n_local_crops=8):
        # Global crops (224×224)
        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # Local crops (96×96)
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.n_local_crops = n_local_crops

    def __call__(self, image):
        crops = []
        # 2 global crops
        crops.append(self.global_transform(image))
        crops.append(self.global_transform(image))
        # n local crops
        for _ in range(self.n_local_crops):
            crops.append(self.local_transform(image))
        return crops
```

### 2.4 Teacher-Student 업데이트

```python
class DINOTrainer:
    """
    DINO 학습 루프

    핵심:
    - Student: gradient로 업데이트
    - Teacher: Student의 EMA (Exponential Moving Average)
    """
    def __init__(self, student, teacher, optimizer, loss_fn, momentum=0.996):
        self.student = student
        self.teacher = teacher
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.momentum = momentum

        # Teacher는 Student로 초기화
        self.teacher.load_state_dict(self.student.state_dict())
        # Teacher는 gradient 계산 안 함
        for p in self.teacher.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_teacher(self):
        """EMA update: θ_t = m * θ_t + (1-m) * θ_s"""
        for param_s, param_t in zip(self.student.parameters(), self.teacher.parameters()):
            param_t.data.mul_(self.momentum).add_((1 - self.momentum) * param_s.data)

    def train_step(self, images):
        """
        images: list of crops [global1, global2, local1, ..., localN]
        """
        # Global crops만 Teacher에 입력
        teacher_output = self.teacher(torch.cat(images[:2]))

        # 모든 crops를 Student에 입력
        student_output = self.student(torch.cat(images))

        # Loss 계산 (각 student crop vs 각 teacher crop)
        loss = self.loss_fn(student_output, teacher_output)

        # Student 업데이트
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Teacher EMA 업데이트
        self.update_teacher()

        return loss.item()
```

---

## 3. DINOv2 (2023)

### 3.1 DINOv2의 개선점

```
┌─────────────────────────────────────────────────────────────────┐
│                 DINO vs DINOv2 비교                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  항목              │ DINO (2021)      │ DINOv2 (2023)          │
│  ─────────────────│──────────────────│───────────────────────  │
│  데이터            │ ImageNet (1.3M)  │ LVD-142M (142M)        │
│  데이터 큐레이션    │ 없음             │ 자동 큐레이션 파이프라인  │
│  모델 크기         │ ViT-S/B          │ ViT-S/B/L/g            │
│  학습 목표         │ DINO만           │ DINO + iBOT (masked)   │
│  Regularization   │ 기본             │ KoLeo + 정규화 강화     │
│  Resolution       │ 224              │ 518 (고해상도)          │
│  성능 (k-NN)      │ ~74% (IN-1K)    │ ~86% (IN-1K)           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 LVD-142M 데이터셋

```python
"""
LVD-142M (Learning with large Visual Datasets)

자동 큐레이션 파이프라인:
1. 웹에서 이미지 수집 (billions)
2. 중복 제거 (copy detection)
3. 품질 필터링
4. ImageNet과 유사도 기반 샘플링
5. 최종 142M 이미지

핵심 기술:
- Self-supervised copy detection
- Embedding 기반 클러스터링
- Retrieval 기반 데이터 선택

왜 중요한가:
- 데이터 품질이 모델 성능의 핵심
- Scaling은 데이터 큐레이션이 필수
- 자동화된 파이프라인으로 확장 가능
"""
```

### 3.3 iBOT 통합

```
┌─────────────────────────────────────────────────────────────────┐
│                 DINOv2 = DINO + iBOT                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  DINO Loss (이미지 레벨):                                        │
│  • Global/Local crop 간 consistency                             │
│  • CLS token 기반                                               │
│                                                                 │
│  iBOT Loss (패치 레벨):                                          │
│  • Masked patches 예측                                          │
│  • MAE와 유사하지만 Teacher 사용                                  │
│                                                                 │
│                    Input Image                                  │
│                         │                                       │
│          ┌─────────────┴─────────────┐                          │
│          ▼                           ▼                          │
│     ┌─────────┐                ┌─────────┐                      │
│     │ Teacher │                │ Student │                      │
│     │ (full)  │                │ (masked)│ ← 일부 패치 마스킹     │
│     └────┬────┘                └────┬────┘                      │
│          │                          │                           │
│     ┌────┴────┐                ┌────┴────┐                      │
│     │CLS│Patch│                │CLS│Patch│                      │
│     └─┬───┬───┘                └─┬───┬───┘                      │
│       │   │                      │   │                          │
│       │   └──────────────────────│───┤                          │
│       │          iBOT Loss       │   │                          │
│       │     (masked patches)     │   │                          │
│       │                          │   │                          │
│       └──────────────────────────┘   │                          │
│              DINO Loss               │                          │
│           (CLS tokens)               │                          │
│                                                                 │
│  Total Loss = L_DINO + λ × L_iBOT                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.4 모델 구조

```python
"""
DINOv2 모델 사양

Model      │ Layers │ Hidden │ Heads │ Params │ Patch
──────────│────────│────────│───────│────────│───────
ViT-S/14  │ 12     │ 384    │ 6     │ 21M    │ 14×14
ViT-B/14  │ 12     │ 768    │ 12    │ 86M    │ 14×14
ViT-L/14  │ 24     │ 1024   │ 16    │ 300M   │ 14×14
ViT-g/14  │ 40     │ 1536   │ 24    │ 1.1B   │ 14×14

특징:
- Patch size 14 (기존 ViT는 16)
- 더 높은 해상도 지원
- Register tokens (attention artifact 해결)
"""
```

---

## 4. DINOv2 사용하기

### 4.1 HuggingFace로 로드

```python
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests

# 모델 로드
model_name = "facebook/dinov2-base"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 이미지 로드
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# 전처리 및 추론
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# 출력 구조
print(f"Last hidden state: {outputs.last_hidden_state.shape}")
# (1, 257, 768) = (batch, 1 CLS + 256 patches, hidden_dim)

# CLS token (전체 이미지 표현)
cls_token = outputs.last_hidden_state[:, 0]
print(f"CLS token: {cls_token.shape}")  # (1, 768)

# Patch tokens (지역 표현)
patch_tokens = outputs.last_hidden_state[:, 1:]
print(f"Patch tokens: {patch_tokens.shape}")  # (1, 256, 768)
```

### 4.2 특징 추출 및 활용

```python
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
import numpy as np
from sklearn.neighbors import NearestNeighbors

class DINOv2FeatureExtractor:
    """DINOv2를 이용한 이미지 특징 추출기"""

    def __init__(self, model_name="facebook/dinov2-base"):
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def extract_features(self, images, return_patches=False):
        """
        이미지에서 특징 추출

        Args:
            images: PIL Image 또는 리스트
            return_patches: 패치별 특징도 반환할지

        Returns:
            cls_features: (n_images, hidden_dim)
            patch_features: (n_images, n_patches, hidden_dim) - optional
        """
        if not isinstance(images, list):
            images = [images]

        inputs = self.processor(images=images, return_tensors="pt")
        outputs = self.model(**inputs)

        cls_features = outputs.last_hidden_state[:, 0]

        if return_patches:
            patch_features = outputs.last_hidden_state[:, 1:]
            return cls_features, patch_features

        return cls_features

    def compute_similarity(self, image1, image2):
        """두 이미지 간 유사도 (코사인)"""
        feat1 = self.extract_features(image1)
        feat2 = self.extract_features(image2)
        similarity = F.cosine_similarity(feat1, feat2)
        return similarity.item()

# 사용 예시
extractor = DINOv2FeatureExtractor()

# 이미지 검색
def build_image_index(images):
    """이미지 인덱스 구축"""
    features = []
    for img in images:
        feat = extractor.extract_features(img)
        features.append(feat.numpy())
    features = np.vstack(features)

    # k-NN 인덱스
    index = NearestNeighbors(n_neighbors=5, metric='cosine')
    index.fit(features)
    return index, features

def search_similar(query_image, index, features, k=5):
    """유사 이미지 검색"""
    query_feat = extractor.extract_features(query_image).numpy()
    distances, indices = index.kneighbors(query_feat, n_neighbors=k)
    return indices[0], distances[0]
```

### 4.3 Dense Prediction (Semantic Segmentation)

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_attention_maps(model, processor, image):
    """DINOv2의 attention map 시각화"""

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # 마지막 레이어의 attention
    attentions = outputs.attentions[-1]  # (1, n_heads, n_tokens, n_tokens)

    # CLS token이 각 패치에 주는 attention
    cls_attn = attentions[0, :, 0, 1:]  # (n_heads, n_patches)

    # 평균
    cls_attn_mean = cls_attn.mean(dim=0)  # (n_patches,)

    # Reshape to 2D
    n_patches = int(np.sqrt(cls_attn_mean.shape[0]))
    attn_map = cls_attn_mean.reshape(n_patches, n_patches)

    return attn_map.numpy()

def visualize_patch_pca(model, processor, image, n_components=3):
    """패치 특징의 PCA 시각화 (의미론적 영역 확인)"""

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # 패치 토큰
    patch_tokens = outputs.last_hidden_state[0, 1:].numpy()  # (n_patches, hidden)

    # PCA
    pca = PCA(n_components=n_components)
    patch_pca = pca.fit_transform(patch_tokens)

    # Normalize to [0, 1] for visualization
    patch_pca = (patch_pca - patch_pca.min()) / (patch_pca.max() - patch_pca.min())

    # Reshape
    n_patches = int(np.sqrt(patch_tokens.shape[0]))
    pca_image = patch_pca.reshape(n_patches, n_patches, n_components)

    return pca_image

# 시각화
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# axes[0].imshow(image)
# axes[0].set_title('Original')
# axes[1].imshow(visualize_attention_maps(model, processor, image), cmap='hot')
# axes[1].set_title('Attention Map')
# axes[2].imshow(visualize_patch_pca(model, processor, image))
# axes[2].set_title('PCA of Patches')
```

---

## 5. DINOv2 응용

### 5.1 Zero-shot Semantic Segmentation

```python
"""
DINOv2의 패치 특징을 이용한 세그멘테이션

방법:
1. 이미지에서 DINOv2 패치 특징 추출
2. 예시 이미지에서 관심 영역의 특징 추출
3. 코사인 유사도로 해당 영역 찾기

장점:
- 학습 없이 세그멘테이션 가능
- 새로운 객체 클래스도 처리 가능
"""

def segment_with_reference(model, processor, target_image, reference_image, reference_mask):
    """
    참조 이미지의 마스크를 이용해 타겟 이미지 세그멘테이션

    Args:
        target_image: 세그멘테이션할 이미지
        reference_image: 참조 이미지
        reference_mask: 참조 이미지의 관심 영역 마스크 (binary)
    """
    # 특징 추출
    with torch.no_grad():
        target_inputs = processor(images=target_image, return_tensors="pt")
        target_outputs = model(**target_inputs)
        target_patches = target_outputs.last_hidden_state[0, 1:]  # (n_patches, hidden)

        ref_inputs = processor(images=reference_image, return_tensors="pt")
        ref_outputs = model(**ref_inputs)
        ref_patches = ref_outputs.last_hidden_state[0, 1:]  # (n_patches, hidden)

    # 참조 마스크에서 관심 영역의 특징 평균
    n_patches = int(np.sqrt(ref_patches.shape[0]))
    mask_resized = F.interpolate(
        reference_mask.unsqueeze(0).unsqueeze(0).float(),
        size=(n_patches, n_patches),
        mode='nearest'
    ).squeeze().bool()

    foreground_features = ref_patches[mask_resized.flatten()].mean(dim=0)

    # 타겟 이미지의 각 패치와 유사도 계산
    similarities = F.cosine_similarity(
        target_patches,
        foreground_features.unsqueeze(0),
        dim=1
    )

    # Reshape to 2D
    similarity_map = similarities.reshape(n_patches, n_patches)

    return similarity_map.numpy()
```

### 5.2 Depth Estimation

```python
"""
DINOv2 + Linear Probe로 Depth Estimation

방법:
1. DINOv2로 패치 특징 추출
2. 간단한 Linear layer로 depth 예측
3. 적은 데이터로도 좋은 성능

이유:
- DINOv2가 이미 3D 구조 정보를 학습
- 패치 특징에 depth cue가 인코딩됨
"""

class DepthEstimator(nn.Module):
    def __init__(self, dinov2_model, hidden_dim=768):
        super().__init__()
        self.backbone = dinov2_model
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x).last_hidden_state[:, 1:]  # patch tokens

        depth = self.head(features)  # (batch, n_patches, 1)

        # Reshape to image
        batch, n_patches, _ = depth.shape
        h = w = int(np.sqrt(n_patches))
        depth = depth.reshape(batch, h, w)

        return depth
```

---

## 정리

### DINO/DINOv2 핵심
| 개념 | 설명 |
|------|------|
| **Self-distillation** | Teacher-Student 구조, 레이블 없이 학습 |
| **Multi-crop** | Global + Local crops로 다양한 스케일 학습 |
| **Centering** | Teacher 출력 centering으로 collapse 방지 |
| **EMA Teacher** | Momentum으로 안정적인 타겟 제공 |
| **iBOT** | Masked patch prediction 추가 (DINOv2) |

### 활용
- **Image Retrieval**: CLS token으로 유사 이미지 검색
- **Semantic Segmentation**: 패치 특징으로 zero-shot 세그멘테이션
- **Depth Estimation**: Linear probe로 depth 예측
- **Fine-tuning**: 다운스트림 태스크 학습

### 다음 단계
- [13_Segment_Anything.md](13_Segment_Anything.md): SAM의 promptable segmentation
- [14_Unified_Vision_Models.md](14_Unified_Vision_Models.md): 통합 Vision Foundation Models

---

## 참고 자료

### 논문
- Caron et al. (2021). "Emerging Properties in Self-Supervised Vision Transformers" (DINO)
- Oquab et al. (2023). "DINOv2: Learning Robust Visual Features without Supervision"
- Zhou et al. (2021). "iBOT: Image BERT Pre-Training with Online Tokenizer"

### 코드
- [DINO GitHub](https://github.com/facebookresearch/dino)
- [DINOv2 GitHub](https://github.com/facebookresearch/dinov2)
- [HuggingFace DINOv2](https://huggingface.co/facebook/dinov2-base)
