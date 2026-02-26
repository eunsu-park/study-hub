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

---

## 연습 문제

### 연습 문제 1: 센터링(Centering)과 모드 붕괴(Mode Collapse) 방지
DINO 손실에서 교사 출력은 소프트맥스(softmax) 적용 전 누적 평균을 빼는 "센터링(centering)" 처리를 받습니다. 센터링이 없을 때 자기 증류(self-distillation) 설정에서 모드 붕괴(mode collapse)가 어떤 형태로 나타나는지 설명하고, 센터 벡터를 빼는 것이 왜 이를 방지하는지 설명하세요. 또한, 센터를 단순 배치 평균이 아닌 교사 출력의 지수 이동 평균(EMA, Exponential Moving Average)으로 계산하는 이유는 무엇인가요?

<details>
<summary>정답 보기</summary>

**센터링 없이 발생하는 모드 붕괴**: 센터링 없이는 교사가 상수 분포를 출력하는 방향으로 수렴할 수 있습니다 — 특정 차원(예: 항상 클래스 0)이 지속적으로 지배하게 됩니다. 학생은 이 상수 출력을 단순히 복사함으로써 교차 엔트로피(cross-entropy)를 최소화하게 되고, 두 네트워크 모두 입력을 무시하는 퇴화된 해법으로 붕괴합니다.

**센터링이 붕괴를 방지하는 이유**: 소프트맥스 전에 교사 로짓에서 누적 평균 `c`를 빼면, 로짓이 평균 0을 갖도록 강제됩니다. 어떤 단일 차원도 지속적으로 지배할 수 없으며, 소프트맥스는 더 균등한 분포를 향해 밀려납니다. 이는 학생이 실제로 입력에 의존하는 패턴을 학습하도록 강제합니다.

**EMA를 사용하는 이유**: 단일 배치 평균은 노이즈가 많고 불안정성을 유발할 수 있습니다. EMA 센터(`c ← m*c + (1-m)*배치평균`)는 수많은 배치에 걸쳐 전역 교사 출력 분포의 부드럽고 안정적인 추정치를 제공합니다. 또한, 진정한 전역 배치 평균과 달리 GPU 간 동기화가 필요 없어 분산 학습에서 효율적입니다.

</details>

### 연습 문제 2: 다중 크롭(Multi-crop) 로컬-글로벌 대응
DINO는 2개의 글로벌 크롭(224×224)을 교사와 학생 모두에게 입력하고, 6~8개의 로컬 크롭(96×96)은 학생에게만 입력합니다. 이 설계가 인코딩하는 핵심 통찰을 설명하고, 로컬 크롭을 교사에게 입력하지 않는 이유를 설명하세요.

<details>
<summary>정답 보기</summary>

**핵심 통찰**: 다중 크롭 전략은 "로컬-글로벌 대응(local-to-global correspondence)"을 강제합니다 — 학생은 작은 패치(로컬 크롭)만 보고도 교사가 전체 이미지(글로벌 맥락)에서 보는 것을 예측해야 합니다. 이는 학생이 의미론적으로 의미 있는 표현을 학습하도록 강제합니다: 강아지 귀의 작은 패치가 전체 강아지 이미지와 같은 객체에 속한다는 것을 알기 위해서는 의미론적 이해가 필요합니다.

**로컬 크롭을 교사에게 입력하지 않는 이유**:
1. 교사가 작고 맥락이 부족한 패치로부터 노이즈가 많은 타겟을 생성하게 됩니다.
2. 고품질의 안정적인 타겟 신호는 글로벌 크롭에서 나옵니다 — 교사는 전체 이미지 맥락에 접근합니다.
3. 로컬 크롭을 교사에도 입력하면 계산 비용이 크게 증가합니다(교사가 배치당 2번 대신 N+2번 실행).

비대칭성은 의도적입니다: 교사 = 안정적인 글로벌 신호, 학생 = 제한된 로컬 뷰에서 학습.

</details>

### 연습 문제 3: DINOv2 iBOT 손실 분석
DINOv2는 DINO 손실(CLS 토큰 수준)과 iBOT 손실(패치 토큰 수준)을 결합합니다. 다음 분석을 완성하세요:

```python
# DINOv2 전체 손실
# L_total = L_DINO + lambda * L_iBOT
# L_DINO: 교사와 학생 CLS 토큰 간 교차 엔트로피
# L_iBOT: 교사와 학생의 마스킹된 패치 토큰 간 교차 엔트로피

# 질문 A: 각 손실 컴포넌트가 포착하는 것은?
# L_DINO가 포착하는 것: ???
# L_iBOT가 포착하는 것: ???

# 질문 B: lambda = 0 (iBOT 비활성화)이면 어떤 능력이 손실되나?
# 답변: ???

# 질문 C: 학생은 일부 패치가 마스킹되어 있고(토큰이 [MASK]로 대체),
# 교사는 전체 이미지를 봅니다. 이 비대칭성이 iBOT에 왜 중요한가?
# 답변: ???
```

<details>
<summary>정답 보기</summary>

**질문 A**:
- `L_DINO`는 CLS 토큰을 통해 **전역 이미지 수준 의미론**을 포착합니다 — 다른 크롭/뷰 간에 일관된 전역 표현을 생성하도록 모델을 훈련합니다.
- `L_iBOT`는 **지역 패치 수준 의미론**을 포착합니다 — 각 마스킹된 패치가 맥락 속에서 어떤 모습이어야 하는지 예측하도록 훈련하여 밀집/공간적 이해를 가능하게 합니다.

**질문 B**: iBOT 없이(λ=0) 모델은 **밀집 시각 특징 품질**을 잃습니다. 패치 토큰이 공간적으로 의미 있는 지역 의미론 정보를 인코딩하도록 훈련받지 못합니다. CLS 토큰만 강한 훈련 신호를 받으므로, 패치 수준 특징에 의존하는 세그멘테이션(segmentation) 같은 태스크가 크게 저하됩니다.

**질문 C**: 비대칭성은 다음과 같은 이유로 필수적입니다:
- **교사**는 완전하고 마스킹되지 않은 이미지를 봄 → **실제 패치 표현**을 학습 타겟으로 제공.
- **학생**은 문맥으로부터 마스킹된 패치를 **예측**해야 함 → 가시적 패치와 마스킹된 이웃 간의 관계를 모델링해야 함.
- 교사도 마스킹된다면 신뢰할 수 있는 타겟을 생성하지 못합니다(교사도 추측해야 함). 교사의 마스킹되지 않은 뷰가 각 마스킹 위치에 대한 안정적인 감독 신호를 제공합니다.

</details>

### 연습 문제 4: DINOv2 특징 평가
DINOv2-Base(86M 파라미터)를 동결(frozen) 백본으로 두 가지 다운스트림 태스크에 활용하려고 합니다: (A) 클래스당 500개의 레이블된 예시가 있는 10개 클래스 소규모 데이터셋의 이미지 분류, (B) 의료 이미지 데이터셋의 의미론적 세그멘테이션(semantic segmentation).

각 태스크에 대해 어떤 DINOv2 특징 유형(CLS 토큰, 패치 토큰, 또는 둘 다)을 사용할지 설명하고, 간단한 헤드 아키텍처를 제안하세요. 선택 이유를 정당화하세요.

<details>
<summary>정답 보기</summary>

**태스크 A: 이미지 분류 (500개 레이블 예시)**

- **특징**: CLS 토큰 (형태: `[batch, 768]`)
- **헤드**: 선형 분류기 또는 얕은 MLP (예: Linear(768, 10))
- **정당화**: 단 500개의 예시로는 최대한의 정규화가 필요합니다. CLS 토큰에 대한 선형 프로브(linear probe)는 과적합을 방지하면서 DINOv2의 풍부한 전역 의미 표현을 활용합니다. k-NN 분류(헤드 없음)도 강한 기준선으로 잘 작동합니다. CLS 토큰은 전역 정보를 집약하므로 전체 이미지 분류에 이상적입니다.

**태스크 B: 의미론적 세그멘테이션 (의료 이미지)**

- **특징**: 패치 토큰 (형태: `[batch, n_patches, 768]`) → 공간 격자로 재형성
- **헤드**: 경량 디코더, 예:
  ```python
  # n_patches = (518/14)^2 = 37^2 = 1369 (DINOv2 ViT-L/14 @ 518px 기준)
  # (batch, 768, 37, 37)로 재형성 후 업샘플링
  nn.Sequential(
      nn.Conv2d(768, 256, 1),
      nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
      nn.ConvTranspose2d(128, num_classes, 4, stride=2, padding=1),
  )
  ```
- **정당화**: 세그멘테이션은 공간적으로 기반한 픽셀 단위 예측이 필요합니다. 패치 토큰은 CLS 토큰이 버리는 공간적/지역적 의미 정보를 담고 있습니다. DINOv2의 패치 토큰은 iBOT 훈련이 패치 수준 표현을 명시적으로 최적화하기 때문에 특히 고품질입니다. 동결 백본과 경량 합성곱 디코더의 조합은 데이터 효율적이며, 레이블 데이터가 부족한 의료 데이터셋에 매우 중요합니다.

</details>
