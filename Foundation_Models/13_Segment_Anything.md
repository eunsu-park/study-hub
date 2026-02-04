# Segment Anything Model (SAM)

## 학습 목표
- SAM의 "Promptable Segmentation" 패러다임 이해
- Image Encoder, Prompt Encoder, Mask Decoder 구조 파악
- SAM의 학습 데이터와 방법론 이해
- 실무에서 SAM 활용법 습득

---

## 1. SAM 개요

### 1.1 Foundation Model for Segmentation

**SAM** (Segment Anything Model)은 Meta AI가 2023년 발표한 Vision Foundation Model로, **어떤 이미지에서든 어떤 객체든** 세그멘테이션할 수 있습니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    SAM의 혁신                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  기존 세그멘테이션:                                               │
│  • 특정 클래스만 (사람, 자동차 등)                                 │
│  • 학습 데이터에 있는 객체만                                       │
│  • 클래스별 모델 또는 고정된 클래스 수                              │
│                                                                 │
│  SAM:                                                           │
│  • 어떤 객체든 세그멘테이션 가능                                   │
│  • 프롬프트로 원하는 객체 지정                                     │
│  • Zero-shot: 새로운 객체도 바로 처리                             │
│                                                                 │
│  프롬프트 종류:                                                   │
│  ┌────────────────────────────────────────────────────┐         │
│  │ Point   │ 클릭 위치 (foreground/background)        │         │
│  │ Box     │ 바운딩 박스                              │         │
│  │ Mask    │ 대략적인 마스크 (refinement)             │         │
│  │ Text    │ 텍스트 설명 (SAM 2, Grounding SAM)      │         │
│  └────────────────────────────────────────────────────┘         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 SA-1B 데이터셋

```
┌─────────────────────────────────────────────────────────────────┐
│                    SA-1B Dataset                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  규모:                                                           │
│  • 11M 이미지                                                    │
│  • 1.1B (11억) 마스크                                            │
│  • 이미지당 평균 ~100 마스크                                      │
│                                                                 │
│  수집 방법 (Data Engine):                                        │
│                                                                 │
│  Phase 1: Assisted Manual (4.3M masks)                          │
│  ───────────────────────────────────                            │
│  • 전문 annotator가 SAM 도움받아 레이블링                          │
│  • SAM이 제안 → 사람이 수정                                       │
│                                                                 │
│  Phase 2: Semi-Automatic (5.9M masks)                           │
│  ───────────────────────────────────                            │
│  • SAM이 confident한 마스크 자동 생성                              │
│  • 사람은 나머지만 레이블링                                        │
│                                                                 │
│  Phase 3: Fully Automatic (1.1B masks)                          │
│  ───────────────────────────────────                            │
│  • 32×32 grid points로 자동 생성                                 │
│  • 필터링 후 최종 마스크 선별                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. SAM 아키텍처

### 2.1 전체 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                    SAM Architecture                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                         Input Image                             │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   Image Encoder                          │    │
│  │           (MAE pre-trained ViT-H/16)                    │    │
│  │                                                          │    │
│  │  • 1024×1024 입력 → 64×64 feature map                   │    │
│  │  • 632M parameters                                       │    │
│  │  • 한 번만 실행 (비용 큼)                                 │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                  │
│                              ▼                                  │
│                     Image Embeddings                            │
│                       (64×64×256)                               │
│                              │                                  │
│              ┌───────────────┴───────────────┐                  │
│              │                               │                  │
│              ▼                               ▼                  │
│  ┌───────────────────┐           ┌───────────────────┐         │
│  │  Prompt Encoder   │           │  Prompt Encoder   │         │
│  │  (Points/Boxes)   │           │  (Dense: Mask)    │         │
│  │                   │           │                   │         │
│  │  Sparse Embed     │           │  Conv downscale   │         │
│  │  (N×256)          │           │  (256×64×64)      │         │
│  └─────────┬─────────┘           └─────────┬─────────┘         │
│            │                               │                    │
│            └───────────────┬───────────────┘                    │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   Mask Decoder                           │    │
│  │           (Lightweight Transformer)                      │    │
│  │                                                          │    │
│  │  • 2-layer Transformer decoder                          │    │
│  │  • Cross-attention: prompt ↔ image                      │    │
│  │  • Self-attention: prompt tokens                        │    │
│  │  • 4M parameters (매우 가벼움)                           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                            │                                    │
│              ┌─────────────┴─────────────┐                      │
│              ▼                           ▼                      │
│         3 Mask Outputs             IoU Scores                   │
│     (256×256, upscaled)          (confidence)                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Image Encoder

```python
"""
SAM Image Encoder: MAE pre-trained ViT-H

특징:
- ViT-H/16: 632M parameters
- 입력: 1024×1024 (고해상도)
- 출력: 64×64×256 feature map
- Positional Embedding: Windowed + Global attention

왜 MAE pre-training?
- 마스킹 기반 학습으로 dense prediction에 적합
- 자기 지도 학습으로 대규모 데이터 활용
- Patch-level 표현 학습에 효과적
"""

import torch
import torch.nn as nn

class SAMImageEncoder(nn.Module):
    """
    SAM의 Image Encoder (간소화 버전)

    실제로는 ViT-H를 사용하지만,
    여기서는 구조 이해를 위한 간소화
    """
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        embed_dim: int = 1280,  # ViT-H
        depth: int = 32,
        num_heads: int = 16,
        out_chans: int = 256,
    ):
        super().__init__()

        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, (img_size // patch_size) ** 2, embed_dim)
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(depth)
        ])

        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1),
            nn.LayerNorm(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.LayerNorm(out_chans),
        )

    def forward(self, x):
        # x: (B, 3, 1024, 1024)
        x = self.patch_embed(x)  # (B, embed_dim, 64, 64)
        x = x.flatten(2).transpose(1, 2)  # (B, 4096, embed_dim)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        # Reshape back to 2D
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.transpose(1, 2).reshape(B, C, H, W)

        x = self.neck(x)  # (B, 256, 64, 64)
        return x
```

### 2.3 Prompt Encoder

```python
class SAMPromptEncoder(nn.Module):
    """
    SAM Prompt Encoder

    프롬프트 종류:
    1. Points: (x, y) + label (foreground/background)
    2. Boxes: (x1, y1, x2, y2)
    3. Masks: 이전 마스크 (refinement용)
    """
    def __init__(self, embed_dim: int = 256, image_size: int = 1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size

        # Point embeddings
        self.point_embeddings = nn.ModuleList([
            nn.Embedding(1, embed_dim),  # foreground
            nn.Embedding(1, embed_dim),  # background
        ])

        # Positional encoding for points
        self.pe_layer = PositionalEncoding(embed_dim, image_size)

        # Box corner embeddings
        self.box_embeddings = nn.Embedding(2, embed_dim)  # top-left, bottom-right

        # Mask encoder (for dense prompts)
        self.mask_downscaler = nn.Sequential(
            nn.Conv2d(1, embed_dim // 4, kernel_size=2, stride=2),
            nn.LayerNorm(embed_dim // 4),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, embed_dim, kernel_size=2, stride=2),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
        )

        # No-mask embedding
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def forward(self, points=None, boxes=None, masks=None):
        """
        Args:
            points: (B, N, 2) 좌표 + (B, N) 레이블
            boxes: (B, 4) 바운딩 박스
            masks: (B, 1, H, W) 이전 마스크

        Returns:
            sparse_embeddings: (B, N_prompts, embed_dim)
            dense_embeddings: (B, embed_dim, H, W)
        """
        sparse_embeddings = []

        # Point prompts
        if points is not None:
            coords, labels = points
            point_embed = self.pe_layer(coords)  # positional encoding

            for i in range(coords.shape[1]):
                label = labels[:, i]
                type_embed = self.point_embeddings[label](label)
                sparse_embeddings.append(point_embed[:, i] + type_embed)

        # Box prompts
        if boxes is not None:
            # Box = 2 corner points
            corners = boxes.reshape(-1, 2, 2)  # (B, 2, 2)
            corner_embed = self.pe_layer(corners)
            corner_embed += self.box_embeddings.weight
            sparse_embeddings.extend([corner_embed[:, 0], corner_embed[:, 1]])

        sparse_embeddings = torch.stack(sparse_embeddings, dim=1) if sparse_embeddings else None

        # Dense prompt (mask)
        if masks is not None:
            dense_embeddings = self.mask_downscaler(masks)
        else:
            # No mask: learnable embedding
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1)
            dense_embeddings = dense_embeddings.expand(-1, -1, 64, 64)

        return sparse_embeddings, dense_embeddings
```

### 2.4 Mask Decoder

```python
class SAMMaskDecoder(nn.Module):
    """
    SAM Mask Decoder

    구조:
    - 2-layer Transformer decoder
    - Cross-attention: tokens ↔ image
    - Self-attention: tokens
    - 3개의 마스크 출력 (multi-scale)
    - IoU prediction head
    """
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_mask_tokens: int = 4,  # 3 masks + 1 IoU
    ):
        super().__init__()

        # Mask tokens (learnable)
        self.mask_tokens = nn.Embedding(num_mask_tokens, embed_dim)

        # Transformer layers
        self.transformer = TwoWayTransformer(
            depth=2,
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

        # Output heads
        self.iou_prediction_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_mask_tokens - 1),  # 3 IoU scores
        )

        self.mask_prediction_head = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 4, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=2, stride=2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 8, num_mask_tokens - 1, kernel_size=1),
        )

    def forward(self, image_embeddings, sparse_embeddings, dense_embeddings):
        """
        Args:
            image_embeddings: (B, 256, 64, 64)
            sparse_embeddings: (B, N_prompts, 256)
            dense_embeddings: (B, 256, 64, 64)

        Returns:
            masks: (B, 3, 256, 256)
            iou_predictions: (B, 3)
        """
        # Combine sparse and mask tokens
        mask_tokens = self.mask_tokens.weight.unsqueeze(0).expand(
            sparse_embeddings.shape[0], -1, -1
        )
        tokens = torch.cat([mask_tokens, sparse_embeddings], dim=1)

        # Add dense embeddings to image
        image_pe = dense_embeddings
        src = image_embeddings + dense_embeddings

        # Transformer decoder
        # Cross-attention between tokens and image
        tokens, src = self.transformer(tokens, src, image_pe)

        # Extract mask tokens
        mask_tokens_out = tokens[:, :self.mask_tokens.num_embeddings - 1]

        # IoU prediction
        iou_predictions = self.iou_prediction_head(mask_tokens_out[:, 0])

        # Mask prediction
        # Upscale and predict
        src = src.reshape(-1, 256, 64, 64)
        masks = self.mask_prediction_head(src)  # (B, 3, 256, 256)

        return masks, iou_predictions


class TwoWayTransformer(nn.Module):
    """
    Two-way Transformer for SAM

    특징:
    - Token → Image cross-attention
    - Image → Token cross-attention
    - Token self-attention
    """
    def __init__(self, depth, embed_dim, num_heads):
        super().__init__()
        self.layers = nn.ModuleList([
            TwoWayAttentionBlock(embed_dim, num_heads)
            for _ in range(depth)
        ])

    def forward(self, tokens, image, image_pe):
        for layer in self.layers:
            tokens, image = layer(tokens, image, image_pe)
        return tokens, image
```

---

## 3. SAM 사용하기

### 3.1 기본 사용법

```python
from segment_anything import SamPredictor, sam_model_registry
import cv2
import numpy as np

# 모델 로드
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam.to(device="cuda")
predictor = SamPredictor(sam)

# 이미지 설정
image = cv2.imread("image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image)

# Point prompt로 세그멘테이션
input_point = np.array([[500, 375]])  # 클릭 위치
input_label = np.array([1])  # 1: foreground, 0: background

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,  # 3개 마스크 출력
)

# 가장 높은 score의 마스크 선택
best_mask = masks[np.argmax(scores)]
```

### 3.2 다양한 프롬프트

```python
# 1. Multiple points
input_points = np.array([[500, 375], [600, 400], [450, 350]])
input_labels = np.array([1, 1, 0])  # 2 foreground, 1 background

masks, scores, _ = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
    multimask_output=False,  # 단일 마스크
)

# 2. Box prompt
input_box = np.array([100, 100, 500, 400])  # x1, y1, x2, y2

masks, scores, _ = predictor.predict(
    box=input_box,
    multimask_output=False,
)

# 3. Point + Box combined
masks, scores, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    box=input_box,
    multimask_output=False,
)

# 4. Iterative refinement (이전 마스크 사용)
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    mask_input=logits[np.argmax(scores)][None, :, :],  # 이전 logits
    multimask_output=False,
)
```

### 3.3 Automatic Mask Generation

```python
from segment_anything import SamAutomaticMaskGenerator

# 자동 마스크 생성기
mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=32,           # 32×32 grid
    pred_iou_thresh=0.88,         # IoU 임계값
    stability_score_thresh=0.95,  # 안정성 임계값
    min_mask_region_area=100,     # 최소 마스크 크기
)

# 이미지의 모든 마스크 생성
masks = mask_generator.generate(image)

# 결과: list of dicts
# {
#     'segmentation': binary mask,
#     'area': mask area,
#     'bbox': bounding box,
#     'predicted_iou': IoU score,
#     'stability_score': stability score,
#     'crop_box': crop used for generation,
# }

print(f"Found {len(masks)} masks")

# 시각화
import matplotlib.pyplot as plt

def show_masks(image, masks):
    plt.figure(figsize=(15, 10))
    plt.imshow(image)
    for mask in masks:
        m = mask['segmentation']
        color = np.random.random(3)
        colored_mask = np.zeros((*m.shape, 4))
        colored_mask[m] = [*color, 0.5]
        plt.imshow(colored_mask)
    plt.axis('off')
    plt.show()

show_masks(image, masks)
```

### 3.4 HuggingFace Transformers 사용

```python
from transformers import SamModel, SamProcessor
import torch
from PIL import Image

# 모델 로드
model = SamModel.from_pretrained("facebook/sam-vit-huge")
processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

# 이미지 로드
image = Image.open("image.jpg")

# Point prompt
input_points = [[[500, 375]]]  # batch of points

inputs = processor(image, input_points=input_points, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# Post-process
masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(),
    inputs["original_sizes"].cpu(),
    inputs["reshaped_input_sizes"].cpu()
)

scores = outputs.iou_scores
```

---

## 4. SAM 2 (2024)

### 4.1 SAM 2의 발전

```
┌─────────────────────────────────────────────────────────────────┐
│                    SAM vs SAM 2                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  SAM (2023):                                                    │
│  • 이미지 전용                                                   │
│  • 프레임별 독립 처리                                            │
│  • 비디오: 프레임마다 프롬프트 필요                               │
│                                                                 │
│  SAM 2 (2024):                                                  │
│  • 이미지 + 비디오 통합                                          │
│  • Memory attention으로 시간 일관성                              │
│  • 한 번 프롬프트 → 전체 비디오 추적                              │
│                                                                 │
│  새로운 구성요소:                                                 │
│  ┌────────────────────────────────────────────────────┐         │
│  │ Memory Encoder   │ 과거 프레임 정보 인코딩          │         │
│  │ Memory Bank      │ 과거 마스크와 특징 저장          │         │
│  │ Memory Attention │ 현재 프레임 ↔ 과거 정보 attention│         │
│  └────────────────────────────────────────────────────┘         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 SAM 2 비디오 사용

```python
from sam2.build_sam import build_sam2_video_predictor

predictor = build_sam2_video_predictor(
    "sam2_hiera_large.pt",
    device="cuda"
)

# 비디오 프레임들 로드
video_path = "video.mp4"

with predictor.init_state(video_path) as state:
    # 첫 프레임에서 프롬프트
    _, _, masks = predictor.add_new_points_or_box(
        state,
        frame_idx=0,
        obj_id=1,
        points=[[500, 375]],
        labels=[1],
    )

    # 나머지 프레임 자동 전파
    for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
        # masks: 각 프레임의 세그멘테이션 결과
        print(f"Frame {frame_idx}: {len(object_ids)} objects")
```

---

## 5. SAM 응용

### 5.1 Grounding SAM (Text → Segment)

```python
"""
Grounding SAM = Grounding DINO + SAM

1. Grounding DINO: 텍스트 → 바운딩 박스
2. SAM: 바운딩 박스 → 세그멘테이션

결과: 텍스트 프롬프트로 세그멘테이션
"""

from groundingdino.util.inference import load_model, predict
from segment_anything import SamPredictor, sam_model_registry

# Grounding DINO로 박스 검출
grounding_dino = load_model("groundingdino_swinb.pth")
boxes, logits, phrases = predict(
    grounding_dino,
    image,
    text_prompt="a cat",
    box_threshold=0.3,
    text_threshold=0.25,
)

# SAM으로 세그멘테이션
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
predictor = SamPredictor(sam)
predictor.set_image(image)

masks = []
for box in boxes:
    mask, _, _ = predictor.predict(box=box.numpy(), multimask_output=False)
    masks.append(mask)
```

### 5.2 Interactive Annotation Tool

```python
"""
SAM 기반 인터랙티브 레이블링 도구

1. 이미지 로드
2. 사용자가 포인트/박스 클릭
3. SAM이 실시간 마스크 생성
4. 사용자가 수정 (positive/negative points)
5. 최종 마스크 저장
"""

import cv2
import numpy as np
from segment_anything import SamPredictor

class SAMAnnotator:
    def __init__(self, sam_checkpoint):
        self.sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        self.predictor = SamPredictor(self.sam)
        self.points = []
        self.labels = []

    def set_image(self, image):
        self.image = image.copy()
        self.predictor.set_image(image)
        self.points = []
        self.labels = []

    def add_point(self, x, y, is_foreground=True):
        self.points.append([x, y])
        self.labels.append(1 if is_foreground else 0)
        return self.predict()

    def predict(self):
        if not self.points:
            return None

        masks, scores, _ = self.predictor.predict(
            point_coords=np.array(self.points),
            point_labels=np.array(self.labels),
            multimask_output=False,
        )
        return masks[0]

    def reset(self):
        self.points = []
        self.labels = []

# 사용 예시 (OpenCV 마우스 콜백과 함께)
# annotator = SAMAnnotator("sam_vit_h.pth")
# annotator.set_image(image)
# mask = annotator.add_point(500, 375, is_foreground=True)
```

### 5.3 Medical Imaging

```python
"""
의료 영상 세그멘테이션

SAM의 강점:
- Zero-shot으로 새로운 장기/병변 세그멘테이션
- 전문가의 포인트 클릭만으로 정밀 마스크

MedSAM: 의료 영상에 fine-tuned SAM
"""

# MedSAM 사용 예시
from medsam import MedSAMPredictor

predictor = MedSAMPredictor("medsam_checkpoint.pth")

# CT/MRI 이미지 로드
medical_image = load_medical_image("ct_scan.nii")

# 슬라이스별 세그멘테이션
for slice_idx in range(medical_image.shape[0]):
    slice_img = medical_image[slice_idx]
    predictor.set_image(slice_img)

    # 전문가가 병변 위치 클릭
    mask, _, _ = predictor.predict(
        point_coords=np.array([[tumor_x, tumor_y]]),
        point_labels=np.array([1]),
    )
```

---

## 정리

### SAM 핵심 구성
| 구성요소 | 역할 | 특징 |
|---------|------|------|
| **Image Encoder** | 이미지 특징 추출 | MAE ViT-H, 632M params |
| **Prompt Encoder** | 프롬프트 인코딩 | Point/Box/Mask 지원 |
| **Mask Decoder** | 마스크 생성 | 2-layer Transformer, 4M params |

### 프롬프트 종류
- **Point**: 클릭 위치 (foreground/background)
- **Box**: 바운딩 박스
- **Mask**: 이전 마스크 (refinement)
- **Text**: Grounding SAM 통해 지원

### 활용
| 용도 | 방법 |
|------|------|
| Interactive Annotation | 클릭으로 빠른 레이블링 |
| Automatic Segmentation | Grid points로 전체 객체 |
| Video Tracking | SAM 2로 객체 추적 |
| Medical Imaging | MedSAM으로 특화 |

### 다음 단계
- [14_Unified_Vision_Models.md](14_Unified_Vision_Models.md): 통합 Vision Models
- [16_Vision_Language_Deep.md](16_Vision_Language_Deep.md): Multimodal (LLaVA)

---

## 참고 자료

### 논문
- Kirillov et al. (2023). "Segment Anything"
- Ravi et al. (2024). "SAM 2: Segment Anything in Images and Videos"
- Liu et al. (2023). "Grounding DINO"
- Ma et al. (2023). "Segment Anything in Medical Images" (MedSAM)

### 코드
- [SAM GitHub](https://github.com/facebookresearch/segment-anything)
- [SAM 2 GitHub](https://github.com/facebookresearch/segment-anything-2)
- [Grounding SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything)
- [HuggingFace SAM](https://huggingface.co/facebook/sam-vit-huge)
