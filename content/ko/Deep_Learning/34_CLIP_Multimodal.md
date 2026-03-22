[이전: 확산 모델(DDPM)](./33_Impl_Diffusion.md) | [다음: CLIP (Contrastive Language-Image Pre-training)](./35_Impl_CLIP.md)

---

# 34. CLIP과 멀티모달 학습

## 학습 목표

- CLIP 아키텍처와 원리 이해
- Contrastive Learning 기반 Image-Text 매칭
- Zero-shot Classification 구현
- BLIP, ALIGN 등 후속 모델 소개
- PyTorch 활용 및 실습

---

## 1. 멀티모달 학습 개요

### 멀티모달이란?

```
여러 종류의 데이터 (modality)를 함께 학습

Vision + Language: CLIP, BLIP, Flamingo
Vision + Audio: AudioCLIP
Text + Audio: CLAP
Vision + Text + Audio: ImageBind
```

### 왜 멀티모달인가?

```
1. 풍부한 표현 학습
   - 텍스트: 추상적, 의미적 정보
   - 이미지: 시각적, 공간적 정보
   - 상호보완적 학습 가능

2. Zero-shot 능력
   - 새로운 클래스도 텍스트로 정의 가능
   - 레이블 없이 분류 가능

3. 다양한 다운스트림 태스크
   - Image-Text Retrieval
   - Visual Question Answering
   - Image Captioning
```

---

## 2. CLIP 아키텍처

### Contrastive Language-Image Pre-training

```
┌─────────────────────────────────────────────────────────────┐
│                       CLIP Architecture                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Image                          Text                        │
│     │                              │                         │
│     ▼                              ▼                         │
│ ┌─────────┐                  ┌─────────┐                    │
│ │  Image  │                  │  Text   │                    │
│ │ Encoder │                  │ Encoder │                    │
│ │  (ViT)  │                  │(Transf.)│                    │
│ └────┬────┘                  └────┬────┘                    │
│      │                            │                          │
│      ▼                            ▼                          │
│  Image                        Text                           │
│  Embedding                    Embedding                      │
│  (I_1...I_n)                  (T_1...T_n)                   │
│      │                            │                          │
│      └──────────┬─────────────────┘                         │
│                 ▼                                            │
│         Contrastive Loss                                     │
│         (maximize I_i · T_i)                                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 학습 목표

```
N개의 (이미지, 텍스트) 쌍이 있을 때:

올바른 쌍 (diagonal): 유사도 최대화
잘못된 쌍 (off-diagonal): 유사도 최소화

손실 함수: InfoNCE (Contrastive Loss)
```

---

## 3. CLIP 손실 함수

### InfoNCE Loss

```python
import torch
import torch.nn.functional as F

def clip_loss(image_features, text_features, temperature=0.07):
    """CLIP Contrastive Loss (⭐⭐⭐)

    Args:
        image_features: (N, D) 정규화된 이미지 임베딩
        text_features: (N, D) 정규화된 텍스트 임베딩
        temperature: 온도 파라미터 (낮을수록 sharp)

    Returns:
        loss: 이미지→텍스트 + 텍스트→이미지 손실
    """
    # 유사도 행렬 (N x N)
    logits = (image_features @ text_features.T) / temperature

    # Ground truth: 대각선이 정답
    labels = torch.arange(len(logits), device=logits.device)

    # 양방향 CrossEntropy
    loss_i2t = F.cross_entropy(logits, labels)      # 이미지 → 텍스트
    loss_t2i = F.cross_entropy(logits.T, labels)    # 텍스트 → 이미지

    return (loss_i2t + loss_t2i) / 2
```

### 온도 파라미터

```python
# temperature가 낮을수록:
# - 분포가 더 sharp
# - 정답에 더 집중
# - 학습 초기에는 높게, 점차 낮게

# CLIP 기본값: 0.07 (학습 가능한 파라미터)
log_temperature = nn.Parameter(torch.log(torch.tensor(1/0.07)))
temperature = log_temperature.exp()
```

---

## 4. CLIP 모델 구현

### 이미지 인코더

```python
import torch
import torch.nn as nn

class ImageEncoder(nn.Module):
    """CLIP Image Encoder (ViT-based) (⭐⭐⭐)"""
    def __init__(self, embed_dim=512, vision_width=768, vision_layers=12,
                 vision_heads=12, image_size=224, patch_size=16):
        super().__init__()

        self.conv1 = nn.Conv2d(3, vision_width, patch_size, patch_size, bias=False)

        num_patches = (image_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, vision_width))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, vision_width))

        self.ln_pre = nn.LayerNorm(vision_width)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=vision_width,
                nhead=vision_heads,
                dim_feedforward=vision_width * 4,
                activation='gelu',
                batch_first=True
            ),
            num_layers=vision_layers
        )

        self.ln_post = nn.LayerNorm(vision_width)
        self.projection = nn.Linear(vision_width, embed_dim, bias=False)

    def forward(self, x):
        # Patch Embedding
        x = self.conv1(x)  # (B, vision_width, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, vision_width)

        # CLS Token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Position Embedding
        x = x + self.pos_embed
        x = self.ln_pre(x)

        # Transformer
        x = self.transformer(x)

        # CLS Token 출력
        x = self.ln_post(x[:, 0])

        # Projection
        x = self.projection(x)

        return x
```

### 텍스트 인코더

```python
class TextEncoder(nn.Module):
    """CLIP Text Encoder (Transformer-based) (⭐⭐⭐)"""
    def __init__(self, embed_dim=512, vocab_size=49408, context_length=77,
                 transformer_width=512, transformer_layers=12, transformer_heads=8):
        super().__init__()

        self.context_length = context_length

        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(
            torch.randn(context_length, transformer_width)
        )

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=transformer_width,
                nhead=transformer_heads,
                dim_feedforward=transformer_width * 4,
                activation='gelu',
                batch_first=True
            ),
            num_layers=transformer_layers
        )

        self.ln_final = nn.LayerNorm(transformer_width)
        self.projection = nn.Linear(transformer_width, embed_dim, bias=False)

    def forward(self, text):
        # text: (B, context_length) - 토큰 인덱스

        x = self.token_embedding(text)  # (B, L, transformer_width)
        x = x + self.positional_embedding

        # Causal Mask
        mask = torch.triu(torch.ones(self.context_length, self.context_length), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf')).to(x.device)

        x = self.transformer(x, mask=mask)
        x = self.ln_final(x)

        # EOT (End of Text) 토큰 위치의 출력 사용
        # 실제로는 argmax로 EOT 위치 찾음
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]

        x = self.projection(x)

        return x
```

### CLIP 전체 모델

```python
class CLIP(nn.Module):
    """CLIP Model (⭐⭐⭐⭐)"""
    def __init__(self, embed_dim=512):
        super().__init__()

        self.image_encoder = ImageEncoder(embed_dim=embed_dim)
        self.text_encoder = TextEncoder(embed_dim=embed_dim)

        # 학습 가능한 온도 파라미터
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))

    def encode_image(self, image):
        features = self.image_encoder(image)
        return F.normalize(features, dim=-1)

    def encode_text(self, text):
        features = self.text_encoder(text)
        return F.normalize(features, dim=-1)

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # 유사도 계산
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * (image_features @ text_features.T)
        logits_per_text = logits_per_image.T

        return logits_per_image, logits_per_text
```

---

## 5. Zero-shot Classification

### 개념

```
CLIP의 핵심 능력: 학습 시 본 적 없는 클래스도 분류 가능

방법:
1. 각 클래스를 텍스트로 설명 ("a photo of a {class}")
2. 텍스트 임베딩 계산
3. 이미지 임베딩과 유사도 계산
4. 가장 유사한 클래스 선택
```

### 구현

```python
def zero_shot_classify(model, image, class_names, templates=None):
    """CLIP Zero-shot Classification (⭐⭐⭐)"""
    if templates is None:
        templates = [
            "a photo of a {}",
            "a blurry photo of a {}",
            "a photo of the {}",
            "a drawing of a {}",
            "a photo of my {}",
        ]

    # 텍스트 임베딩 계산 (클래스별 템플릿 평균)
    text_features_list = []
    for class_name in class_names:
        class_texts = [template.format(class_name) for template in templates]
        # 토큰화 (실제로는 tokenizer 사용)
        # text_tokens = tokenizer(class_texts)
        # text_features = model.encode_text(text_tokens)
        # text_features = text_features.mean(dim=0)  # 템플릿 평균
        # text_features_list.append(text_features)
        pass

    text_features = torch.stack(text_features_list)
    text_features = F.normalize(text_features, dim=-1)

    # 이미지 임베딩
    image_features = model.encode_image(image)

    # 유사도 계산
    similarity = (image_features @ text_features.T)

    # Top-1 예측
    probs = similarity.softmax(dim=-1)
    pred = probs.argmax(dim=-1)

    return pred, probs
```

### 프롬프트 엔지니어링

```python
# 더 나은 결과를 위한 프롬프트 템플릿

# ImageNet용
imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    # ... 더 많은 템플릿
]

# CIFAR-10용
cifar10_templates = [
    'a photo of a {}.',
    'a blurry photo of a {}.',
    'a black and white photo of a {}.',
    'a low contrast photo of a {}.',
    'a high contrast photo of a {}.',
    'a bad photo of a {}.',
    'a good photo of a {}.',
    'a photo of a small {}.',
    'a photo of a big {}.',
    'a photo of the {}.',
]
```

---

## 6. OpenAI CLIP 사용

### 설치 및 기본 사용

```python
import torch
import clip
from PIL import Image

# 모델 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 이미지 전처리 및 인코딩
image = preprocess(Image.open("cat.jpg")).unsqueeze(0).to(device)

# 텍스트 토큰화
text = clip.tokenize(["a cat", "a dog", "a bird"]).to(device)

# 추론
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    # 유사도 계산
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Similarity:", similarity)
# 예: tensor([[0.95, 0.03, 0.02]])
```

### 사용 가능한 모델

```python
# 모델 목록
print(clip.available_models())
# ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64',
#  'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']

# 모델 특성
models_info = {
    'ViT-B/32': {'params': '151M', 'image_size': 224, 'context_length': 77},
    'ViT-B/16': {'params': '149M', 'image_size': 224, 'context_length': 77},
    'ViT-L/14': {'params': '428M', 'image_size': 224, 'context_length': 77},
    'ViT-L/14@336px': {'params': '428M', 'image_size': 336, 'context_length': 77},
}
```

---

## 7. Image-Text Retrieval

### Text-to-Image Retrieval

```python
def text_to_image_retrieval(model, images, text_query, top_k=5):
    """텍스트로 이미지 검색 (⭐⭐⭐)"""
    with torch.no_grad():
        # 이미지 임베딩 (미리 계산 가능)
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # 텍스트 임베딩
        text_tokens = clip.tokenize([text_query]).to(images.device)
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # 유사도 계산
        similarity = (text_features @ image_features.T).squeeze(0)

        # Top-K 검색
        values, indices = similarity.topk(top_k)

    return indices, values
```

### Image-to-Text Retrieval

```python
def image_to_text_retrieval(model, image, text_candidates, top_k=5):
    """이미지로 텍스트 검색 (⭐⭐⭐)"""
    with torch.no_grad():
        # 이미지 임베딩
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # 텍스트 임베딩
        text_tokens = clip.tokenize(text_candidates).to(image.device)
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # 유사도 계산
        similarity = (image_features @ text_features.T).squeeze(0)

        # Top-K 검색
        values, indices = similarity.topk(top_k)

    return indices, values
```

---

## 8. BLIP (Bootstrapping Language-Image Pre-training)

### CLIP의 한계와 BLIP의 개선

```
CLIP의 한계:
1. 노이즈가 많은 웹 데이터
2. Image Captioning 불가 (matching만)
3. 단방향 텍스트 인코더

BLIP의 개선:
1. CapFilt: 캡션 필터링으로 데이터 정제
2. 생성과 이해 모두 가능
3. 양방향 + 자동회귀 텍스트 인코더
```

### BLIP 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                     BLIP Architecture                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  [Image Encoder (ViT)]                                       │
│           │                                                  │
│           ▼                                                  │
│  Image Representation                                        │
│           │                                                  │
│     ┌─────┼─────────────────┐                               │
│     │     │                 │                                │
│     ▼     ▼                 ▼                                │
│  ┌─────┐ ┌────────┐  ┌──────────┐                          │
│  │ ITC │ │ ITM    │  │  LM      │                          │
│  │     │ │        │  │ (생성)   │                          │
│  └─────┘ └────────┘  └──────────┘                          │
│  Contrastive  Matching    Captioning                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘

ITC: Image-Text Contrastive (CLIP과 유사)
ITM: Image-Text Matching (binary classification)
LM: Language Modeling (caption generation)
```

### BLIP 사용

```python
from transformers import BlipProcessor, BlipForConditionalGeneration

# 모델 로드
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Image Captioning
image = Image.open("cat.jpg")
inputs = processor(image, return_tensors="pt")
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)
print(f"Caption: {caption}")
# 예: "a cat sitting on a couch"

# Conditional Captioning (with prompt)
inputs = processor(image, text="a photo of", return_tensors="pt")
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)
```

---

## 9. 기타 멀티모달 모델

### ALIGN (Google)

```
특징:
- CLIP과 유사하지만 더 큰 스케일
- 18억 개의 노이즈 많은 이미지-텍스트 쌍
- EfficientNet + BERT 기반

장점:
- 노이즈에 강건
- 대규모 데이터 활용
```

### Flamingo (DeepMind)

```
특징:
- Few-shot Learning 능력
- 이미지/비디오 + 텍스트 입력
- Visual Question Answering 강점

구조:
- Perceiver Resampler로 시각 정보 압축
- 언어 모델에 시각 정보 주입
```

### LLaVA (Large Language and Vision Assistant)

```
특징:
- 시각적 instruction tuning
- 대화형 비전-언어 모델
- GPT-4 수준의 멀티모달 이해

구조:
- CLIP 이미지 인코더
- Vicuna/LLaMA 언어 모델
- 프로젝션 레이어로 연결
```

---

## 10. CLIP Fine-tuning

### Linear Probe

```python
class CLIPLinearProbe(nn.Module):
    """CLIP Linear Probe for Classification (⭐⭐)"""
    def __init__(self, clip_model, num_classes, freeze_clip=True):
        super().__init__()
        self.clip = clip_model

        if freeze_clip:
            for param in self.clip.parameters():
                param.requires_grad = False

        # 선형 분류기만 학습
        self.classifier = nn.Linear(512, num_classes)  # CLIP 임베딩 차원

    def forward(self, images):
        with torch.no_grad() if self.training else torch.inference_mode():
            features = self.clip.encode_image(images)
            features = features.float()

        return self.classifier(features)
```

### Full Fine-tuning

```python
def finetune_clip(model, train_loader, epochs=10, lr=1e-5):
    """CLIP Full Fine-tuning (⭐⭐⭐)"""
    # CLIP 파라미터는 낮은 학습률
    optimizer = torch.optim.AdamW([
        {'params': model.visual.parameters(), 'lr': lr},
        {'params': model.transformer.parameters(), 'lr': lr},
        {'params': model.logit_scale, 'lr': lr * 10}  # 온도는 더 빠르게
    ])

    for epoch in range(epochs):
        for images, texts in train_loader:
            logits_per_image, logits_per_text = model(images, texts)

            labels = torch.arange(len(images), device=images.device)
            loss_i = F.cross_entropy(logits_per_image, labels)
            loss_t = F.cross_entropy(logits_per_text, labels)
            loss = (loss_i + loss_t) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

---

## 정리

### 핵심 개념

1. **Contrastive Learning**: 이미지-텍스트 쌍의 유사도 학습
2. **Zero-shot**: 학습 시 본 적 없는 클래스 분류
3. **Temperature**: 유사도 분포의 sharpness 조절
4. **Prompt Engineering**: 텍스트 템플릿으로 성능 향상
5. **멀티모달 표현**: 공통 임베딩 공간에서 검색/비교

### 모델 비교

| 모델 | 특징 | 장점 |
|------|------|------|
| CLIP | Contrastive | Zero-shot, 검색 |
| BLIP | 생성+이해 | Captioning, VQA |
| Flamingo | Few-shot | 대화형, 유연성 |
| LLaVA | Instruction | 복잡한 질의 처리 |

### 실전 팁

```python
# 1. 프롬프트 템플릿 다양하게
templates = ["a photo of {}", "an image of {}", ...]

# 2. 앙상블 사용
features = average([encode(template.format(class_name)) for template in templates])

# 3. 온도 조절 실험
# 낮은 온도: 더 확신 있는 예측
# 높은 온도: 더 부드러운 분포

# 4. 큰 모델 사용 (성능 순)
# ViT-L/14@336px > ViT-L/14 > ViT-B/16 > ViT-B/32
```

---

## 연습 문제

### 연습 1: InfoNCE 손실 이해하기

N=4개의 이미지-텍스트 쌍 배치에 대해 `clip_loss` 함수를 수동으로 추적하세요:
1. 4×4 유사도(similarity) 행렬을 스케치하세요. 어느 항목이 "양성(positive)" 쌍(정답 매칭)인가요?
2. `loss_i2t` (이미지→텍스트)와 `loss_t2i` (텍스트→이미지)가 모두 필요한 이유를 설명하세요. 한 방향만 사용하면 어떤 비대칭이 발생하나요?
3. 온도(temperature)가 손실에 미치는 영향은? 온도가 매우 작으면 (예: 0.01) 소프트맥스 분포에 어떤 일이 일어나나요? 매우 크면 (예: 10.0)?

### 연습 2: OpenAI CLIP으로 제로샷 분류(Zero-shot Classification)

`clip` 라이브러리를 사용하여:
1. `ViT-B/32` 모델을 불러오고 샘플 이미지(예: 고양이 사진)를 전처리하세요.
2. 템플릿 `"a photo of a {}"`를 사용하여 10개의 CIFAR-10 클래스명에 대해 이미지를 분류하세요.
3. 상위 3개의 예측 클래스와 해당 확률을 출력하세요.
4. 5개의 다른 프롬프트 템플릿(예: `"a blurry photo of a {}"`, `"a drawing of a {}"`)으로 반복하고 텍스트 임베딩을 평균화(앙상블)하세요. 앙상블 프롬프팅이 올바른 클래스에 대한 확신도를 향상시키나요?

### 연습 3: 이미지 검색 시스템 구축

사전학습된 CLIP 모델을 사용하여 간단한 텍스트-이미지 검색 엔진을 구현하세요:
1. 5개의 서로 다른 카테고리에서 50개의 이미지를 다운로드하세요 (카테고리당 10개).
2. `model.encode_image`를 사용하여 50개의 이미지 모두에 대한 이미지 임베딩을 사전 계산하고 캐시하세요.
3. `"a red sports car"`와 같은 텍스트 쿼리를 입력하면 코사인 유사도를 사용하여 가장 유사한 상위 5개 이미지를 검색하세요.
4. 상위 5개 결과를 시각화하고 검색 정밀도(검색된 이미지 중 실제로 쿼리 카테고리에 맞는 이미지 수)를 평가하세요.

### 연습 4: 선형 프로브(Linear Probe)로 CLIP 미세조정

CLIP을 고정된 특성 추출기로 사용하여:
1. 제공된 코드로 `CLIPLinearProbe`를 구현하고 모든 CLIP 파라미터를 고정하세요.
2. CIFAR-10 학습 세트에 대한 CLIP 이미지 특성을 추출하고 그 위에 선형 분류기를 학습하세요.
3. CIFAR-10 테스트 세트에서 평가하고 다음과 비교하세요: (a) 제로샷 CLIP, (b) 처음부터 학습한 CNN.
4. 이제 CLIP 모델의 고정을 해제하고 작은 학습률(`lr=1e-5`)로 전체 미세조정을 수행하세요. 전체 미세조정이 선형 프로브보다 향상되나요? 다운스트림 태스크에서 전체 미세조정의 위험은 무엇인가요?

---

## 참고 자료

- CLIP: https://arxiv.org/abs/2103.00020
- BLIP: https://arxiv.org/abs/2201.12086
- ALIGN: https://arxiv.org/abs/2102.05918
- OpenAI CLIP: https://github.com/openai/CLIP
