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

---

## 연습 문제

### 연습 문제 1: SAM 아키텍처 비대칭성
SAM의 이미지 인코더(Image Encoder)(ViT-H, 6억 3200만 파라미터)는 마스크 디코더(Mask Decoder)(2-레이어 트랜스포머, 약 400만 파라미터)보다 훨씬 큽니다. 이 극단적인 크기 비대칭의 설계 근거를 설명하세요. 대화형 어노테이션(interactive annotation) 워크플로에서 어떤 실용적 이점을 제공하나요?

<details>
<summary>정답 보기</summary>

**설계 근거**: 이미지 인코더는 사용자가 제공하는 프롬프트 수와 관계없이 **이미지당 단 한 번**만 실행됩니다. 풍부한 밀집 이미지 특징 추출이라는 무거운 계산이 이후 모든 상호작용에 걸쳐 분산됩니다. 반면 마스크 디코더는 **프롬프트당 한 번** 실행되며(사용자가 선택을 정제하면서 수십 번 재실행될 수 있음), 따라서 매우 빠르게 실행되어야 합니다.

**대화형 어노테이션에서의 실용적 이점**:
1. **사전 계산**: 이미지 임베딩(64×64×256)이 한 번 계산되어 캐시됩니다.
2. **즉각적 응답**: 각 새 프롬프트(포인트 클릭, 박스 그리기)는 400만 파라미터 디코더만 실행하면 되므로 밀리초 단위입니다.
3. **실시간 상호작용**: 사용자가 전경/배경 포인트를 반복적으로 추가하면서 눈에 띄는 지연 없이 즉각적인 마스크 정제를 볼 수 있습니다.
4. **분리된 연산**: 웹 서비스에서 이미지 임베딩은 GPU 서버에서 백그라운드로 계산되고, 경량 디코더는 클라이언트 측에서도 실행 가능합니다.

이 비대칭성이 SAM을 배치 처리가 아닌 "대화형"으로 느끼게 만드는 핵심입니다.

</details>

### 연습 문제 2: SA-1B 데이터 엔진 분석
SAM의 훈련 데이터(SA-1B: 1,100만 이미지에 11억 개 마스크)는 3단계 데이터 엔진으로 수집되었습니다. 각 단계의 진행을 분석하고, 이 부트스트랩(bootstrapped) 접근법이 왜 필요했는지 설명하세요.

| 단계 | 방법 | 수집된 마스크 수 |
|------|------|------------------|
| 1단계 | 보조 수동(Assisted Manual) | 430만 |
| 2단계 | 반자동(Semi-Automatic) | 590만 |
| 3단계 | 완전 자동(Fully Automatic) | 11억 |

답변: 3단계(완전 자동)를 처음부터 사용할 수 없는 이유는 무엇인가요? 각 단계는 다음 단계를 가능하게 하기 위해 무엇을 기여하나요?

<details>
<summary>정답 보기</summary>

**3단계를 처음부터 사용할 수 없는 이유**: 완전 자동 생성은 다양한 이미지에서 고품질의 신뢰할 수 있는 세그멘테이션이 가능한 모델이 필요합니다. 이 모델은 처음에는 존재하지 않으며 먼저 훈련되어야 합니다. 부트스트랩 문제: 좋은 모델을 훈련하려면 좋은 데이터가 필요하고, 좋은 데이터를 생성하려면 좋은 모델이 필요합니다.

**1단계 (보조 수동 → 430만 마스크)**:
- 전문 어노테이터들이 초기 SAM 프로토타입을 사용하여 이미지에 레이블을 붙입니다.
- SAM이 마스크를 제안하고, 사람이 수정하고 정제합니다.
- 첫 번째 실제 SAM 모델을 훈련하기 위한 초기 고품질 훈련 세트를 생성합니다.
- **기여**: 실측 데이터(ground truth) 품질 기준을 확립하고 첫 번째 유능한 모델을 훈련합니다.

**2단계 (반자동 → 590만 마스크)**:
- 더 강력한 SAM(1단계 데이터로 훈련)이 자신 있는 마스크를 자동으로 생성합니다.
- 사람 어노테이터는 모델이 불확실한 객체만 레이블링합니다.
- 마스크 다양성을 증가시킵니다 — 모델이 일반적인 객체를 자동으로 처리하고, 사람은 희귀하거나 어려운 경우에 집중합니다.
- **기여**: 품질을 유지하면서 데이터를 확장하고, 희귀 객체 클래스에 대한 모델을 개선합니다.

**3단계 (완전 자동 → 11억 마스크)**:
- 성숙한 SAM 모델이 각 이미지에서 32×32 그리드 포인트를 사용하여 마스크를 생성합니다.
- 사람 어노테이션 불필요 — 마스크는 품질(IoU, 안정성)에 따라 자동으로 필터링됩니다.
- **기여**: 진정한 파운데이션 모델(foundation model)에 필요한 규모를 제공합니다.

부트스트랩 접근법은 데이터 수집 문제를 O(사람 × 이미지)에서 O(모델 품질 × 이미지)로 변환하여, 10억 규모의 어노테이션을 경제적으로 실현 가능하게 만듭니다.

</details>

### 연습 문제 3: 프롬프트 종류와 다중 마스크 출력
SAM은 `multimask_output=True`일 때 3개의 마스크 후보를 생성합니다. 여러 마스크를 출력하는 것이 왜 유용한지 설명하고, `multimask_output=False`와 `True`를 각각 언제 사용해야 하는지 설명하세요. 또한 각 마스크 출력의 IoU 점수가 무엇을 나타내는지 설명하세요.

```python
masks, scores, logits = predictor.predict(
    point_coords=np.array([[500, 375]]),
    point_labels=np.array([1]),
    multimask_output=True,  # 3개의 마스크 반환
)
# masks.shape: (3, H, W)
# scores.shape: (3,)
# logits.shape: (3, 256, 256)

# 다음 호출에서 logits를 언제 사용하나요?
# 답변: ???
```

<details>
<summary>정답 보기</summary>

**3개의 마스크를 출력하는 이유**: 단일 포인트 클릭은 본질적으로 모호합니다. 사람의 얼굴을 클릭하면 (1) 얼굴만, (2) 머리, 또는 (3) 전체 사람을 의미할 수 있습니다. SAM의 3개 출력은 일반적으로 다른 세밀도(fine → medium → coarse) 수준에 해당합니다. 이를 통해 모델이 여러 스케일에서 동시에 정확할 수 있으며, 사용자나 다운스트림 시스템이 적절한 수준을 선택할 수 있습니다.

**`multimask_output=False`** (단일 마스크): 모호성을 해소하는 추가 컨텍스트가 있을 때 사용합니다 — 예를 들어 포인트 + 박스 프롬프트를 결합할 때(박스가 이미 범위를 제한함), 또는 `mask_input` 파라미터를 통해 원하는 마스크 수준을 이미 확립한 반복적 정제 중에 사용합니다.

**`multimask_output=True`** (3개 마스크): 단일 포인트만 입력으로 주어지고 모호성이 높을 때 사용합니다. 다운스트림 로직(가장 높은 IoU 점수, 또는 대화형 UI에서 사용자 선택)이 최적 마스크를 선택합니다.

**IoU 점수**: 각 점수는 생성된 마스크가 의도한 객체를 정확하게 커버하는지에 대한 SAM의 예측 신뢰도입니다 — 구체적으로는 생성된 마스크와 가상의 실측 마스크(ground-truth mask) 간의 예측 교집합/합집합(IoU) 비율입니다. 높은 점수는 이 마스크가 대상을 올바르게 세그멘테이션한다는 모델의 더 높은 신뢰도를 나타냅니다.

**다음 호출에서 `logits` 사용**: 이전 예측의 원시 로짓(시그모이드 이전, 256×256 형태)을 다음 호출의 `mask_input`으로 전달할 수 있습니다. 이는 SAM에게 "이것이 내 이전 추정값입니다"를 알려줍니다 — SAM은 추가 포인트 프롬프트를 받아 처음부터 시작하지 않고 마스크를 정제할 수 있습니다. 이것이 각 새 포인트 클릭이 이전 마스크를 사전 정보(prior)로 사용하는 반복적 정제 워크플로입니다.

</details>

### 연습 문제 4: SAM vs SAM 2 아키텍처 확장
SAM 2는 비디오 처리를 위해 메모리 인코더(Memory Encoder), 메모리 뱅크(Memory Bank), 메모리 어텐션(Memory Attention) 모듈을 추가합니다. 이 컴포넌트들 없이 비디오에 프레임별 SAM 추론을 단순 적용하면 객체 추적에서 왜 좋지 않은 결과가 나오는지 설명하고, 각 새 컴포넌트의 역할을 설명하세요.

<details>
<summary>정답 보기</summary>

**프레임별 SAM이 비디오 추적에 실패하는 이유**:
- SAM은 각 프레임을 독립적으로 처리합니다 — 프레임 간에 정보가 전달되지 않습니다.
- 객체 외형이 프레임 간에 변합니다(조명, 포즈, 폐색, 모션 블러). 각 프레임마다 새 프롬프트가 필요한데, 이는 비실용적입니다.
- 시간적 일관성이 보장되지 않습니다: 마스크가 다른 객체로 이동하거나 객체가 부분적으로 폐색된 프레임에서 실패할 수 있습니다(폐색 전 객체가 어떻게 생겼는지 기억이 없으므로).
- 전파 없음: 0번 프레임의 단일 프롬프트가 50번 프레임의 SAM에게 어떤 안내도 제공하지 않습니다.

**SAM 2의 각 컴포넌트 역할**:

1. **메모리 인코더(Memory Encoder)**: 과거 프레임의 세그멘테이션 마스크와 이미지 특징을 컴팩트한 메모리 표현으로 인코딩합니다. "시간 t에 객체가 어떻게 생겼는지"를 고정 크기 메모리 항목으로 압축합니다.

2. **메모리 뱅크(Memory Bank)**: 이전 프레임(최근 프레임과 프롬프트된 프레임)의 메모리 항목을 고정 크기 큐에 저장합니다. 시간적 컨텍스트 버퍼 역할을 합니다 — 모델이 이전 프레임에서 객체가 어떻게 나타났는지를 "되돌아볼" 수 있습니다.

3. **메모리 어텐션(Memory Attention)**: 현재 프레임의 이미지 특징을 메모리 뱅크에 조건화하는 교차 어텐션(cross-attention) 모듈입니다. 현재 프레임의 각 패치에 대해 과거 객체 표현에 주의를 기울여 "이 영역이 내가 이전에 본 객체처럼 보이는가?"를 묻습니다. 이를 통해 시간적 컨텍스트를 활용하여 폐색과 외형 변화 사이에서도 강건한 추적이 가능합니다.

</details>
