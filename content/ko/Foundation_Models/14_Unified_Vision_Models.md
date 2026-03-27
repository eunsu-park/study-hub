# 14. Unified Vision Models

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 태스크별 모델에서 통합 비전 모델(Unified Vision Model)로의 패러다임 전환을 설명하고 통합 접근법의 핵심 장점을 명확히 서술할 수 있다
2. Florence, PaLI, Unified-IO 등 주요 통합 비전 모델의 아키텍처(architecture)와 학습 전략을 설명할 수 있다
3. 프롬프트 기반 태스크 조건화(prompt-based task conditioning)를 구현하여 단일 모델이 여러 비전 태스크를 처리하도록 할 수 있다
4. 시퀀스-투-시퀀스(sequence-to-sequence), 대조 학습(contrastive), 생성(generative) 등 다양한 통합 전략과 그 트레이드오프를 비교할 수 있다
5. 다양한 벤치마크에서 통합 비전 모델의 제로샷(zero-shot) 및 퓨샷(few-shot) 전이 성능을 평가할 수 있다

---

## 개요

Unified Vision Models는 다양한 비전 태스크(분류, 검출, 세그멘테이션 등)를 **단일 모델로 처리**하는 패러다임입니다. 태스크별 모델 대신 범용 비전 모델을 목표로 합니다.

---

## 1. 패러다임 전환

### 1.1 전통적 접근 vs 통합 접근

```
┌──────────────────────────────────────────────────────────────────┐
│                    비전 모델 패러다임                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  전통적 (Task-Specific):                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ ResNet       │  │ Faster R-CNN │  │ DeepLab      │           │
│  │ (분류)       │  │ (검출)       │  │ (세그멘테이션)│           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│                                                                  │
│  통합 (Task-Agnostic):                                          │
│  ┌───────────────────────────────────────────────┐              │
│  │              Unified Vision Model              │              │
│  │  "분류해줘" → 분류 결과                        │              │
│  │  "객체 찾아줘" → 바운딩 박스                   │              │
│  │  "세그멘테이션해줘" → 마스크                   │              │
│  └───────────────────────────────────────────────┘              │
│                                                                  │
│  장점: 지식 공유, 유지보수 용이, Zero-shot 전이                  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 주요 모델 비교

| 모델 | 개발 | 특징 | 지원 태스크 |
|------|------|------|-------------|
| **Florence** | Microsoft | 대규모 Image-Text | 분류, 검출, 캡셔닝, VQA |
| **PaLI** | Google | 다국어 VLM | 캡셔닝, VQA, OCR |
| **Unified-IO** | Allen AI | 모든 모달리티 | 이미지, 오디오, 텍스트 |
| **OFA** | Alibaba | Seq2Seq 통합 | 다양한 비전-언어 |
| **GPT-4V** | OpenAI | 상용 멀티모달 | 범용 비전 이해 |

---

## 2. Florence: Foundation Model for Vision

### 2.1 아키텍처

```
Florence 아키텍처:

이미지 인코더: CoSwin Transformer (Hierarchical)
텍스트 인코더: UniCL (Unified Contrastive Learning)

학습:
1. Image-Text Contrastive (CLIP 스타일)
2. Image-Text Matching
3. Masked Language Modeling

특징:
- 9억 Image-Text 쌍으로 학습
- 다양한 granularity (이미지 → 영역 → 픽셀)
- Dynamic Head로 태스크 적응
```

### 2.2 구현 예시

```python
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel

class FlorenceStyleModel(nn.Module):
    """
    Florence 스타일 통합 비전 모델 (간소화)

    핵심: CLIP 백본 + Task-specific Heads
    """

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-large-patch14",
        num_classes: int = 1000,
        num_detection_classes: int = 80
    ):
        super().__init__()

        # CLIP 백본 (Image + Text 인코더)
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)

        hidden_size = self.clip.config.vision_config.hidden_size

        # Task Heads
        self.classification_head = nn.Linear(hidden_size, num_classes)
        self.detection_head = DetectionHead(hidden_size, num_detection_classes)
        self.segmentation_head = SegmentationHead(hidden_size)
        self.caption_head = CaptionHead(hidden_size, self.clip.config.text_config)

    def forward(
        self,
        images: torch.Tensor,
        task: str = "classification",
        text_prompts: list = None
    ):
        """
        Args:
            images: (B, 3, H, W)
            task: "classification", "detection", "segmentation", "caption"
            text_prompts: 텍스트 프롬프트 (zero-shot용)
        """
        # Image features
        vision_outputs = self.clip.vision_model(images)
        image_features = vision_outputs.last_hidden_state  # (B, num_patches+1, hidden)
        pooled_features = vision_outputs.pooler_output  # (B, hidden)

        if task == "classification":
            if text_prompts:
                # Zero-shot classification (CLIP 스타일)
                return self._zero_shot_classify(pooled_features, text_prompts)
            else:
                return self.classification_head(pooled_features)

        elif task == "detection":
            return self.detection_head(image_features)

        elif task == "segmentation":
            return self.segmentation_head(image_features)

        elif task == "caption":
            return self.caption_head(pooled_features)

    def _zero_shot_classify(
        self,
        image_features: torch.Tensor,
        text_prompts: list
    ) -> torch.Tensor:
        """Zero-shot classification with text prompts"""
        # Text encoding
        text_inputs = self.processor(
            text=text_prompts,
            return_tensors="pt",
            padding=True
        ).to(image_features.device)

        text_features = self.clip.get_text_features(**text_inputs)

        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Similarity
        similarity = image_features @ text_features.T
        return similarity


class DetectionHead(nn.Module):
    """Object Detection Head (DETR 스타일)"""

    def __init__(self, hidden_size: int, num_classes: int, num_queries: int = 100):
        super().__init__()
        self.num_queries = num_queries

        # Object queries
        self.query_embed = nn.Embedding(num_queries, hidden_size)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(hidden_size, 8, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        # Prediction heads
        self.class_head = nn.Linear(hidden_size, num_classes + 1)  # +1 for no-object
        self.bbox_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 4)  # (cx, cy, w, h)
        )

    def forward(self, image_features: torch.Tensor):
        B = image_features.size(0)

        # Query embedding
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)

        # Decoder
        hs = self.decoder(queries, image_features)

        # Predictions
        class_logits = self.class_head(hs)
        bbox_pred = self.bbox_head(hs).sigmoid()

        return {
            'class_logits': class_logits,
            'bbox_pred': bbox_pred
        }


class SegmentationHead(nn.Module):
    """Semantic Segmentation Head"""

    def __init__(self, hidden_size: int, num_classes: int = 150):
        super().__init__()

        # FPN-style decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_size, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, 1)
        )

    def forward(self, image_features: torch.Tensor):
        # Reshape patches to spatial
        B, N, C = image_features.shape
        H = W = int((N - 1) ** 0.5)  # -1 for CLS token
        features = image_features[:, 1:, :].transpose(1, 2).view(B, C, H, W)

        return self.decoder(features)


class CaptionHead(nn.Module):
    """Image Captioning Head"""

    def __init__(self, hidden_size: int, text_config):
        super().__init__()
        self.vocab_size = text_config.vocab_size

        # Cross-attention decoder
        decoder_layer = nn.TransformerDecoderLayer(hidden_size, 8, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        self.lm_head = nn.Linear(hidden_size, self.vocab_size)

    def forward(
        self,
        image_features: torch.Tensor,
        target_ids: torch.Tensor = None
    ):
        # 생성 시에는 autoregressive
        # 학습 시에는 teacher forcing
        pass  # 구현 생략
```

---

## 3. PaLI (Pathways Language and Image model)

### 3.1 아키텍처

```
PaLI 구조:

┌────────────────────────────────────────────────────────┐
│                      PaLI                              │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Image Encoder: ViT-e (4B params, 22B 이미지로 학습)   │
│       ↓                                                │
│  Visual Tokens: [IMG1] [IMG2] ... [IMGn]              │
│       ↓                                                │
│  Text Encoder-Decoder: mT5 (다국어)                   │
│       ↓                                                │
│  Output: 텍스트 (다국어 지원)                          │
│                                                        │
│  입력 형식:                                            │
│  "<image> 이 이미지를 설명해주세요" → "고양이가..."    │
│  "<image> What is in the image?" → "A cat..."         │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### 3.2 태스크 통합

```python
class PaLITaskFormats:
    """PaLI 태스크별 입력 형식"""

    TASK_FORMATS = {
        # 분류
        "classification": "What is in this image?",
        "fine_grained": "What species of bird is this?",

        # 캡셔닝
        "caption_en": "Generate a caption for this image.",
        "caption_ko": "이 이미지에 대한 설명을 작성하세요.",

        # VQA
        "vqa": "Question: {question} Answer:",

        # OCR
        "ocr": "What text is in this image?",

        # 검출 (텍스트로 표현)
        "detection": "Detect all objects in this image.",
        # 출력: "cat [100, 200, 300, 400]; dog [50, 60, 150, 200]"

        # 세그멘테이션 참조
        "referring": "Segment the {object}.",
    }

    @staticmethod
    def format_input(task: str, **kwargs) -> str:
        template = PaLITaskFormats.TASK_FORMATS.get(task, "")
        return template.format(**kwargs)


# 사용 예시
def process_with_pali(model, image, task, **kwargs):
    """PaLI 스타일 처리"""

    # 태스크별 프롬프트
    prompt = PaLITaskFormats.format_input(task, **kwargs)

    # Visual tokens + Text tokens
    inputs = model.prepare_inputs(image, prompt)

    # Generate
    outputs = model.generate(**inputs)

    # Parse output based on task
    if task == "detection":
        return parse_detection_output(outputs)
    elif task == "caption_en":
        return outputs
    else:
        return outputs
```

---

## 4. Unified-IO

### 4.1 진정한 통합: 모든 모달리티

```
Unified-IO: 단일 모델로 모든 I/O 처리

입력/출력 형식:
- 이미지 → VQ-VAE 토큰
- 텍스트 → 서브워드 토큰
- 바운딩 박스 → 좌표 토큰 (이산화)
- 마스크 → VQ-VAE 토큰
- 오디오 → 스펙트로그램 VQ-VAE

모든 것을 토큰 시퀀스로 변환 → Seq2Seq Transformer
```

### 4.2 구현 개념

```python
class UnifiedIOTokenizer:
    """Unified-IO 스타일 토큰화"""

    def __init__(self, vocab_size: int = 50000, image_vocab_size: int = 16384):
        self.vocab_size = vocab_size
        self.image_vocab_size = image_vocab_size

        # 특수 토큰
        self.SPECIAL_TOKENS = {
            '<image>': vocab_size,
            '</image>': vocab_size + 1,
            '<box>': vocab_size + 2,
            '</box>': vocab_size + 3,
            '<mask>': vocab_size + 4,
            '</mask>': vocab_size + 5,
            '<audio>': vocab_size + 6,
            '</audio>': vocab_size + 7,
        }

        # 좌표 이산화 bins
        self.num_bins = 1000

    def tokenize_image(self, image: torch.Tensor) -> torch.Tensor:
        """VQ-VAE로 이미지 토큰화"""
        # VQ-VAE 인코더로 discrete codes 추출
        # codes shape: (H', W')
        codes = self.vqvae.encode(image)

        # Flatten + offset
        tokens = codes.flatten() + self.vocab_size + len(self.SPECIAL_TOKENS)

        return tokens

    def tokenize_bbox(self, bbox: torch.Tensor) -> torch.Tensor:
        """
        바운딩 박스를 이산 토큰으로

        bbox: (x1, y1, x2, y2) normalized [0, 1]
        """
        # 각 좌표를 bin으로 이산화
        bins = (bbox * self.num_bins).long()

        # 특수 토큰 + bins
        tokens = torch.tensor([
            self.SPECIAL_TOKENS['<box>'],
            bins[0], bins[1], bins[2], bins[3],
            self.SPECIAL_TOKENS['</box>']
        ])

        return tokens

    def decode_bbox(self, tokens: torch.Tensor) -> torch.Tensor:
        """토큰에서 바운딩 박스 복원"""
        # <box> 토큰 위치 찾기
        # 4개의 숫자 토큰 추출
        # 정규화 해제
        pass


class UnifiedIOModel(nn.Module):
    """Unified-IO 스타일 모델"""

    def __init__(self, config):
        super().__init__()

        # Unified Embedding
        self.embeddings = nn.ModuleDict({
            'text': nn.Embedding(config.text_vocab_size, config.hidden_size),
            'image': nn.Embedding(config.image_vocab_size, config.hidden_size),
            'coord': nn.Embedding(config.num_bins, config.hidden_size),
        })

        # Encoder-Decoder Transformer
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)

        # Unified LM Head
        self.lm_head = nn.Linear(config.hidden_size, config.total_vocab_size)

    def forward(self, input_tokens, output_tokens=None):
        """
        Seq2Seq forward

        input_tokens: 혼합 모달리티 토큰
        output_tokens: 목표 출력 토큰
        """
        # 토큰 타입별 임베딩
        embeddings = self._get_embeddings(input_tokens)

        # Encoder
        encoder_output = self.encoder(embeddings)

        # Decoder
        if output_tokens is not None:
            decoder_input = self._get_embeddings(output_tokens)
            decoder_output = self.decoder(decoder_input, encoder_output)
            logits = self.lm_head(decoder_output)
            return logits

        return encoder_output

    def _get_embeddings(self, tokens):
        """토큰 타입에 따라 적절한 임베딩 선택"""
        # 토큰 범위에 따라 text/image/coord 구분
        pass


# 다양한 태스크 예시
def unified_io_examples():
    """Unified-IO 태스크 예시"""

    examples = {
        # Image Captioning
        "caption": {
            "input": "<image> {image_tokens} </image> Describe this image.",
            "output": "A cat sitting on a windowsill."
        },

        # Object Detection
        "detection": {
            "input": "<image> {image_tokens} </image> Detect all objects.",
            "output": "cat <box> 100 200 300 400 </box> dog <box> 50 60 150 200 </box>"
        },

        # Segmentation
        "segmentation": {
            "input": "<image> {image_tokens} </image> Segment the cat.",
            "output": "<mask> {mask_tokens} </mask>"
        },

        # Image Generation (역방향)
        "generation": {
            "input": "Generate an image of a sunset over mountains.",
            "output": "<image> {image_tokens} </image>"
        },

        # VQA
        "vqa": {
            "input": "<image> {image_tokens} </image> How many cats are there?",
            "output": "2"
        }
    }

    return examples
```

---

## 5. 실전 활용

### 5.1 Florence-2 사용 (HuggingFace)

```python
from transformers import AutoProcessor, AutoModelForCausalLM

def use_florence2():
    """Florence-2 실전 사용"""

    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-large",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-large",
        trust_remote_code=True
    )

    from PIL import Image
    import requests

    url = "https://example.com/image.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    # 다양한 태스크
    tasks = {
        "<CAPTION>": "짧은 캡션",
        "<DETAILED_CAPTION>": "상세 캡션",
        "<MORE_DETAILED_CAPTION>": "매우 상세한 캡션",
        "<OD>": "객체 검출",
        "<DENSE_REGION_CAPTION>": "영역별 캡션",
        "<REGION_PROPOSAL>": "영역 제안",
        "<CAPTION_TO_PHRASE_GROUNDING>": "텍스트→영역 그라운딩",
        "<REFERRING_EXPRESSION_SEGMENTATION>": "참조 표현 세그멘테이션",
        "<OCR>": "OCR",
        "<OCR_WITH_REGION>": "영역별 OCR",
    }

    for task_prompt, description in tasks.items():
        inputs = processor(text=task_prompt, images=image, return_tensors="pt")

        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3
        )

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(generated_text, task=task_prompt, image_size=image.size)

        print(f"\n{description} ({task_prompt}):")
        print(parsed)


# 실행
use_florence2()
```

### 5.2 커스텀 태스크 학습

```python
from transformers import Trainer, TrainingArguments
from datasets import Dataset

def finetune_unified_vision():
    """통합 비전 모델 fine-tuning"""

    # 멀티태스크 데이터셋 준비
    def create_multitask_dataset():
        """여러 태스크를 하나의 데이터셋으로"""
        samples = []

        # 분류 샘플
        for img_path, label in classification_data:
            samples.append({
                'image': img_path,
                'task': '<CLASSIFICATION>',
                'input_text': '<CLASSIFICATION>',
                'output_text': label
            })

        # 캡션 샘플
        for img_path, caption in caption_data:
            samples.append({
                'image': img_path,
                'task': '<CAPTION>',
                'input_text': '<CAPTION>',
                'output_text': caption
            })

        # VQA 샘플
        for img_path, question, answer in vqa_data:
            samples.append({
                'image': img_path,
                'task': '<VQA>',
                'input_text': f'<VQA> {question}',
                'output_text': answer
            })

        return Dataset.from_list(samples)

    dataset = create_multitask_dataset()

    # 학습
    training_args = TrainingArguments(
        output_dir="./unified-vision-finetuned",
        per_device_train_batch_size=8,
        num_train_epochs=3,
        learning_rate=1e-5,
        # 태스크 샘플링 전략
        dataloader_drop_last=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
```

---

## 6. 미래 방향

### 6.1 World Models

```
다음 단계: World Models

비전 모델 + 물리 이해 + 행동 예측

예시:
- 이미지에서 물리 법칙 이해
- "공을 던지면 어디로 갈까?"
- 비디오의 다음 프레임 예측
- 로봇 조작 계획
```

### 6.2 통합의 한계와 트레이드오프

```
장점:
✓ 태스크 간 지식 공유
✓ 단일 모델 유지보수
✓ Zero-shot 전이
✓ 새로운 태스크 적응 용이

단점:
✗ 개별 태스크 최고 성능 미달
✗ 학습 복잡성
✗ 태스크 간 간섭
✗ 큰 모델 크기

트레이드오프:
- 범용성 vs 전문성
- 편의성 vs 최적 성능
```

---

## 참고 자료

### 논문
- Yuan et al. (2021). "Florence: A New Foundation Model for Computer Vision"
- Chen et al. (2022). "PaLI: A Jointly-Scaled Multilingual Language-Image Model"
- Lu et al. (2022). "Unified-IO: A Unified Model for Vision, Language, and Multi-Modal Tasks"

### 모델
- [Florence-2](https://huggingface.co/microsoft/Florence-2-large)
- [PaLI](https://github.com/google-research/pali)

### 관련 레슨
- [12_DINOv2_Self_Supervised.md](12_DINOv2_Self_Supervised.md)
- [13_Segment_Anything.md](13_Segment_Anything.md)

---

## 연습 문제

### 연습 문제 1: 태스크 통합 전략 비교
주요 통합 비전 모델에서 사용되는 세 가지 통합 전략을 비교하세요. 표를 완성하고, 최소한의 재훈련으로 기존 프로덕션 시스템에 새 태스크(3D 깊이 추정)를 추가해야 하는 팀에 어떤 전략을 추천할지 설명하세요.

| 전략 | 예시 모델 | 태스크 통합 방법 | 트레이드오프 |
|------|----------|----------------|------------|
| 특수 토큰을 활용한 시퀀스-투-시퀀스(Seq2Seq) | ??? | ??? | ??? |
| 대조 학습(Contrastive) + 태스크 헤드 | ??? | ??? | ??? |
| 프롬프트 조건부 생성(Prompt-conditioned Generation) | ??? | ??? | ??? |

<details>
<summary>정답 보기</summary>

| 전략 | 예시 모델 | 태스크 통합 방법 | 트레이드오프 |
|------|----------|----------------|------------|
| Seq2Seq + 특수 토큰 | Unified-IO, OFA | 모든 입출력이 공유 시퀀스로 토크나이즈됨; 특수 토큰으로 모달리티 구분(`<image>`, `<box>`) | 가장 유연; 임의의 입출력 처리; 하지만 복잡한 토크나이제이션 구조와 느린 생성 속도 |
| 대조 학습 + 태스크 헤드 | Florence, CLIP 기반 모델 | 공유 인코더가 특징 생성; 태스크별 헤드가 각 태스크 디코딩 | 빠른 추론; 태스크별 컴포넌트의 독립적 확장; 하지만 헤드 간 지식 공유 제한 |
| 프롬프트 조건부 생성 | PaLI, Florence-2 | 자연어 프롬프트로 태스크 동작 선택; 단일 생성 디코더 | 가장 사용자 친화적; 제로샷(zero-shot) 일반화; 하지만 구조화된 태스크(예: 바운딩 박스)의 출력 형식 파싱이 불안정할 수 있음 |

**3D 깊이 추정 추가를 위한 추천**: **대조 학습 + 태스크 헤드** 전략을 추천합니다. 근거:
- 공유 인코더(이미 사전 훈련됨)가 풍부한 시각 특징을 제공 — 인코더 재훈련 불필요.
- 새 깊이 추정 헤드만 추가하고 훈련하면 됩니다(DINOv2 예시처럼 경량 디코더).
- 기존 프로덕션 태스크에 영향 없이 새 헤드를 독립적으로 배포 가능.
- Seq2Seq는 깊이 토큰 생성 학습을 위해 전체 모델 재훈련이 필요; 프롬프트 기반은 모델이 깊이를 텍스트로 출력하도록 학습해야 하는데 이는 다루기 어렵습니다.

</details>

### 연습 문제 2: PaLI 태스크 형식 설계
PaLI는 자연어 프롬프트를 통해 태스크를 통합합니다. 레슨에 나온 패턴을 따라 다음 세 가지 태스크에 대한 적절한 입출력 형식을 설계하세요:

A) **밀집 객체 계수(Dense Object Counting)**: 지정된 객체 클래스의 모든 가시적 인스턴스를 세고 숫자를 반환합니다.

B) **시간적 추론(비디오 프레임 쌍)**: 비디오의 두 프레임이 주어졌을 때, 발생한 동작을 설명합니다.

C) **문서 레이아웃 분석**: 페이지 이미지가 주어지면, 각 구조적 요소(제목, 단락, 표, 그림)를 식별하고 위치를 지정합니다.

<details>
<summary>정답 보기</summary>

**A) 밀집 객체 계수**:
- 입력: `"<image> 이미지에서 [객체]의 수를 세어보세요."` (예: "Count the number of cars in this image.")
- 출력: `"3"` (순수 숫자)
- 설명이 있는 대안: `"There are 3 cars: 2 in the foreground and 1 partially visible on the right."` (훈련에 더 강건함)
- 설계 참고: 정확도를 위해 사고의 연쇄(chain-of-thought) 출력도 고려: `"왼쪽부터 오른쪽으로 스캔: (100,200)에 자동차, (350,300)에 자동차, (580,250)에 자동차. 합계: 3."`

**B) 시간적 추론 (비디오 프레임 쌍)**:
- 입력: `"<image1> <image2> 이 두 프레임 사이에 발생한 동작이나 변화를 설명하세요."`
- 출력: `"빨간 자동차가 프레임 왼쪽에서 중앙으로 이동하며, 프레임 너비의 약 50%를 이동했습니다."`
- 정량적 동작을 위한 대안: `"방향, 속도 추정치, 나타나거나 사라진 객체를 포함하여 동작을 설명하세요."`

**C) 문서 레이아웃 분석**:
- 입력: `"<image> 이 문서 페이지의 모든 구조적 요소와 위치를 식별하세요."`
- 출력을 텍스트로 구조화:
  ```
  title: "Introduction to Neural Networks" [0.05, 0.02, 0.95, 0.08]
  paragraph: "This paper presents..." [0.05, 0.10, 0.95, 0.35]
  figure: [0.10, 0.38, 0.90, 0.65]
  table: [0.05, 0.68, 0.95, 0.92]
  ```
  여기서 `[x1, y1, x2, y2]`는 정규화된 좌표(0-1 범위)입니다. 이 형식은 공간 정보를 이산화된 텍스트 토큰으로 인코딩하는 Unified-IO 관례를 따릅니다.

</details>

### 연습 문제 3: Unified-IO 토크나이제이션
Unified-IO는 모든 입출력을 토큰 시퀀스로 변환합니다. 다음 객체 탐지 출력의 토크나이제이션을 단계별로 추적하세요:

- 이미지: 640×480
- 탐지된 객체: "bicycle", 바운딩 박스 (x1=128, y1=96, x2=384, y2=384)
- num_bins = 1000, 텍스트 어휘 크기 = 50000, 특수 토큰 = 8개

```python
class UnifiedIOTokenizer:
    def __init__(self, vocab_size=50000, num_bins=1000):
        self.vocab_size = vocab_size
        self.num_bins = num_bins
        self.SPECIAL_TOKENS = {
            '<box>': vocab_size,      # = 50000
            '</box>': vocab_size + 1, # = 50001
            # ... 6개 더 있는 특수 토큰
        }

    def tokenize_bbox(self, bbox, image_size):
        # bbox = (x1, y1, x2, y2) 픽셀 좌표
        # image_size = (W, H)
        # 단계 1: [0, 1]로 정규화
        # 단계 2: num_bins 빈으로 이산화
        # 단계 3: <box> 특수 토큰으로 감싸기
        pass

# 위 bicycle 탐지의 토큰 시퀀스는 무엇인가요?
# 풀이 과정을 보여주세요.
```

<details>
<summary>정답 보기</summary>

```python
# 단계 1: [0, 1]로 정규화
W, H = 640, 480
x1_norm = 128 / 640 = 0.200
y1_norm = 96 / 480  = 0.200
x2_norm = 384 / 640 = 0.600
y2_norm = 384 / 480 = 0.800

# 단계 2: 1000 빈으로 이산화 (num_bins 곱하고 내림)
x1_bin = int(0.200 * 1000) = 200
y1_bin = int(0.200 * 1000) = 200
x2_bin = int(0.600 * 1000) = 600
y2_bin = int(0.800 * 1000) = 800

# 참고: 빈 토큰은 텍스트 토큰과의 충돌을 피하기 위해
# vocab_size + num_special_tokens만큼 오프셋됨
# 오프셋 = 50000 + 8 = 50008
x1_token = 200 + 50008 = 50208
y1_token = 200 + 50008 = 50208
x2_token = 600 + 50008 = 50608
y2_token = 800 + 50008 = 50808

# 단계 3: "bicycle at (128,96,384,384)"의 전체 토큰 시퀀스
# "bicycle"의 텍스트 토큰 = [0, 49999] 범위의 어떤 id
# bicycle_token = 12345라고 가정

token_sequence = [
    12345,   # "bicycle" 텍스트 토큰
    50000,   # <box> 특수 토큰
    50208,   # x1 빈 토큰
    50208,   # y1 빈 토큰
    50608,   # x2 빈 토큰
    50808,   # y2 빈 토큰
    50001,   # </box> 특수 토큰
]
# 전체 탐지 출력: "bicycle <box> 200 200 600 800 </box>"
```

핵심 통찰: 좌표 빈 값(200, 200, 600, 800)은 텍스트 토큰과 특수 토큰 모두와 완전히 분리된 전용 좌표 어휘(vocabulary) 범위로 오프셋된 토큰 ID로 저장됩니다. 이를 통해 단일 트랜스포머가 동일한 시퀀스에서 텍스트와 좌표를 모호함 없이 생성할 수 있습니다.

</details>

### 연습 문제 4: 범용성 vs. 전문성 트레이드오프
한 스타트업이 5가지 태스크가 필요한 컴퓨터 비전 제품을 보유하고 있습니다: (1) 품질 관리 결함 탐지, (2) 제품 분류, (3) 제품 라벨의 OCR(광학 문자 인식), (4) 전자상거래용 제품 설명 생성, (5) 시각적 유사도 검색.

아래 두 가지 아키텍처 전략을 비교하고 하나를 추천하세요:

- **전략 A**: 태스크별 프롬프트 토큰으로 5가지 태스크를 모두 처리하는 단일 통합 모델(Florence-2 스타일) 훈련.
- **전략 B**: 각 태스크에 대해 하나씩, 5개의 전문화된 모델을 사용.

<details>
<summary>정답 보기</summary>

**전략 A (통합 모델) — 장점**:
- 유지 관리, 배포, 업데이트할 모델 하나.
- 공유 표현이 데이터 효율성을 향상시킬 수 있음(예: OCR 특징이 설명 생성에 도움).
- 새 태스크에 대한 제로샷(zero-shot) 또는 퓨샷(few-shot) 적응.
- 낮은 인프라 비용(GPU 서버 하나).

**전략 A — 단점**:
- 태스크 간섭: 출력 구조가 매우 다른 태스크들(결함 탐지 바운딩 박스 vs. OCR 텍스트 vs. 유사도 임베딩)이 훈련 중 경쟁할 수 있음.
- 개별 태스크 정확도가 전문화된 모델보다 낮을 수 있음.
- 다른 태스크에 잠재적으로 영향을 주지 않고 하나의 태스크를 개선하기 더 어려움.

**전략 B (전문화된 모델) — 장점**:
- 각 모델이 특정 태스크에 대해 최대 정확도로 최적화 가능.
- 장애가 격리됨 — OCR 모델의 버그가 결함 탐지에 영향 없음.
- 각 태스크의 독립적인 업데이트 사이클.

**전략 B — 단점**:
- 유지 관리, 배포, 모니터링할 모델 5개.
- 태스크 간 지식 공유 없음.
- 높은 인프라 비용(배포 5개).
- 중복 특징 추출(5개 모델이 모두 이미지를 별도로 인코딩).

**추천**: **하이브리드 접근법** — 강력한 통합 백본(예: DINOv2 또는 Florence 인코더)을 공유 특징 추출기로 사용하되, 각 태스크에 대해 경량 전문화 헤드를 훈련합니다. 이는 두 전략의 장점을 모두 포착합니다:
- 단일 이미지 인코더 = 특징 추출에 단일 추론 비용.
- 전문화 헤드 = 태스크당 최대 정확도.
- 태스크 1-4는 생성 또는 분류 헤드 사용; 태스크 5(유사도 검색)는 벡터 데이터베이스와 함께 CLS 임베딩을 직접 사용.

둘 중 하나를 선택해야 한다면: 전략 A가 초기 단계 스타트업에 더 적합합니다(더 간단한 인프라, 더 쉬운 출시). 하지만 출력 형식이 매우 다른 태스크들(유사도 임베딩 vs. 바운딩 박스 탐지)은 어차피 분리해야 할 수도 있습니다.

</details>
