[이전: CLIP과 멀티모달 학습](./34_CLIP_Multimodal.md) | [다음: Self-Supervised Learning](./36_Self_Supervised_Learning.md)

---

# 35. CLIP (Contrastive Language-Image Pre-training)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. CLIP 학습 목적함수(이미지-텍스트 쌍에 대한 대조 학습)를 설명하고, InfoNCE 손실이 시각과 언어 표현을 어떻게 정렬하는지 서술합니다.
2. CLIP의 이중 인코더 아키텍처(비전 인코더 + 텍스트 인코더)를 설명하고, 공유 임베딩 공간이 어떻게 크로스 모달 유사도 계산을 가능하게 하는지 설명합니다.
3. 유사도 행렬에 대한 대칭 크로스 엔트로피 손실(Symmetric Cross-Entropy Loss)을 포함하여 PyTorch에서 CLIP 대조 학습 루프를 처음부터 구현합니다.
4. 텍스트 프롬프트를 구성하고 이미지-텍스트 유사도를 계산하여 사전 훈련된 CLIP 모델로 제로샷(Zero-shot) 이미지 분류를 수행합니다.
5. 이미지 검색, 시맨틱 이미지 검색, 퓨샷(Few-shot) 분류를 위한 동결 특성(Frozen Features) 등 다운스트림 작업에 CLIP 임베딩을 적용합니다.
6. InfoNCE 손실에서 온도 스케일링(Temperature Scaling)의 역할을 분석하고, 데이터 규모와 프롬프트 엔지니어링이 CLIP의 제로샷 성능에 어떤 영향을 미치는지 설명합니다.

## 개요

CLIP은 이미지와 텍스트를 같은 임베딩 공간에 매핑하여 zero-shot 이미지 분류를 가능하게 합니다. "Learning Transferable Visual Models From Natural Language Supervision" (Radford et al., 2021)

---

## 수학적 배경

### 이론: InfoNCE = 유사도 행렬의 Cross-Entropy

`N`개 일치 쌍의 배치에 대한 contrastive 손실:

```
S_{ij} = (img_i . text_j) / tau                    # (N, N) 유사도 행렬
L_image_to_text = -log( exp(S_{ii}) / sum_j exp(S_{ij}) )
                = cross_entropy(S, labels=arange(N))   # 행 방향
L_text_to_image = cross_entropy(S.T, labels=arange(N)) # 열 방향
L = (L_image_to_text + L_text_to_image) / 2
```

개념적으로: `S`의 각 행이 `N`개 "클래스"(어느 캡션이 이 이미지와 일치?)에 대한 softmax로 다뤄짐; 올바른 라벨은 대각 인덱스. 그러면 cross-entropy가 정확히 배치메이트 음성과 InfoNCE.

이것이 전체 CLIP 손실입니다. PyTorch 세 줄.


### 1. Contrastive Learning

```
목표: 이미지-텍스트 쌍의 유사도 학습

Batch 내 N개의 (image, text) 쌍:
- 대각선 (i, i): 일치하는 쌍 (positive)
- 비대각선 (i, j): 불일치 쌍 (negative)

유사도 행렬 (N × N):
S[i, j] = <image_i, text_j> / τ

여기서 τ는 temperature parameter
```

### 2. InfoNCE Loss

```
Image-to-Text Loss:
L_i2t = -1/N Σᵢ log(exp(S[i,i]) / Σⱼ exp(S[i,j]))

Text-to-Image Loss:
L_t2i = -1/N Σᵢ log(exp(S[i,i]) / Σⱼ exp(S[j,i]))

Total Loss:
L = (L_i2t + L_t2i) / 2

직관:
- 분자: 일치하는 쌍의 유사도 ↑
- 분모: 다른 쌍과의 유사도 ↓
```

### 3. Zero-shot Classification

```
새로운 이미지 분류:

1. 클래스별 텍스트 프롬프트 생성:
   "A photo of a {class_name}"

2. 텍스트 임베딩 계산:
   T = [text_enc("A photo of a cat"),
        text_enc("A photo of a dog"),
        ...]

3. 이미지 임베딩 계산:
   I = image_enc(image)

4. 유사도로 분류:
   probs = softmax(I @ T.T / τ)
   prediction = argmax(probs)

학습 없이 새로운 클래스 분류 가능!
```

---

## CLIP 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                         CLIP                                 │
│                                                              │
│  ┌───────────────────┐         ┌───────────────────┐        │
│  │   Image Encoder   │         │   Text Encoder    │        │
│  │                   │         │                   │        │
│  │  ViT-B/32         │         │  Transformer      │        │
│  │  or               │         │  (12 layers)      │        │
│  │  ResNet-50        │         │                   │        │
│  └─────────┬─────────┘         └─────────┬─────────┘        │
│            │                             │                   │
│            ▼                             ▼                   │
│     Image Embedding              Text Embedding              │
│        (B, D)                       (B, D)                   │
│            │                             │                   │
│            │      L2 Normalize           │                   │
│            ▼                             ▼                   │
│     ┌──────────────────────────────────────────┐            │
│     │         Contrastive Loss                 │            │
│     │   maximize similarity of matching pairs   │            │
│     │   minimize similarity of non-matching    │            │
│     └──────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────┘

모델 변형:
- CLIP ViT-B/32: 512 dim, 86M image + 63M text params
- CLIP ViT-B/16: 512 dim, 86M image + 63M text params
- CLIP ViT-L/14: 768 dim, 304M image + 123M text params
- CLIP RN50: ResNet-50 image encoder
```

---

## 파일 구조

```
12_CLIP/
├── README.md
├── numpy/
│   └── clip_forward.py       # NumPy forward pass
├── pytorch_lowlevel/
│   └── clip_lowlevel.py      # PyTorch Low-Level CLIP
├── paper/
│   └── clip_paper.py         # 논문 재현
└── exercises/
    ├── 01_zero_shot.md       # Zero-shot 분류
    └── 02_retrieval.md       # 이미지-텍스트 검색
```

---

## 핵심 개념

### 이론: 동결된 특징 추출기로서 CLIP

다운스트림 작업(이미지 분류, 검색)의 경우, 종종 CLIP을 완전히 동결하고 그 이미지 임베딩을 특징으로 사용하는 것이 가장 좋습니다:

```
features = clip_image_encoder(images)                  # (B, d), grad 없음
classifier = nn.Linear(d, num_classes)                 # 이것만 학습
```

두 이점:

- **계산 효율성**: 작은 분류기 헤드만 학습.
- **더 나은 few-shot 성능**: 동결된 CLIP 특징의 작은 분류기가 종종 완전 파인튜닝된 더 작은 모델을 이김, CLIP 표현이 이미 매우 일반적이기 때문.

"linear probe" 벤치마킹(SSL 문헌의 흔한 평가)의 경우, 이것이 정확히 측정되는 것: 헤드를 사소하게 단순하게 유지할 때, 표현이 특징 공간으로 얼마나 좋은가?


### 이론: 프롬프트 엔지니어링

CLIP의 "프롬프트"는 클래스 이름 주변의 텍스트 래퍼입니다: `"a photo of a {}"`, `"a sketch of a {}"`, `"a small {}"` 등. 실험적 발견 (Radford et al. 2021):

- 빈 클래스 이름(`"dog"`, `"cat"`)이 저성능.
- `"a photo of a {}"`이 ImageNet zero-shot에 ~1-2% 정확도 추가.
- 많은 프롬프트 앙상블링(각각으로 인코딩, 임베딩 평균)이 추가 ~1% 추가.
- 도메인 특이적 프롬프트(`"a satellite photo of a {}"`)가 도메인 시프트된 데이터에 도움.

왜 이것이 중요한가: CLIP은 웹 이미지-캡션 쌍에서 학습되었고, 거기서 "a photo of a dog"이 그냥 "dog"보다 훨씬 빈번. "a photo of a dog"의 텍스트 임베딩이 텍스트 임베딩 공간의 더 풍부하고 더 잘 지원되는 영역에 위치 — 개의 실제 이미지의 텍스트 캡션이 보일 가능성이 있는 것과 일치.

이는 *프롬프트 엔지니어링*이 진짜 기법이라는 첫 힌트: 모델의 동작이 입력 정식화에 깊고 내용 관련된 방식으로 의존.


### 이론: 배치 크기가 Contrastive 성능을 지배

예제당 음성 수가 `batch_size - 1`과 같음. 더 많은 음성 = 더 어려운 식별 작업 = 더 나은 표현. 실험적으로:

- 배치 256: 약한 contrastive 신호, 빠르게 평탄화.
- 배치 4096: 상당한 개선.
- 배치 32k (CLIP 설정): 대략 수익이 감소하는 영역.

이것이 CLIP이 거대한 계산을 필요로 한 이유: 배치 크기가 모델 크기보다 더 중요하고, 다중 GPU 설정의 큰 배치는 그래디언트 동기화(특징의 all-gather)가 필요하며, 이는 자체 엔지니어링 도전이 있음. MoCo(특징 메모리 뱅크)와 SimCLR(큰 배치 내 음성 + projection head) 같은 프레임워크가 배치 크기 요구사항의 부분적 해결책.


### 1. 대규모 데이터셋

```
WebImageText (WIT) 데이터셋:
- 4억 개의 (image, text) 쌍
- 인터넷에서 수집
- 자연어 supervision

데이터 수집:
1. 이미지와 alt-text 쌍 수집
2. 필터링 (품질, 중복 제거)
3. 클래스 균형 맞추기
```

### 2. Prompt Engineering

```
단순 프롬프트:
"cat"  →  "A photo of a cat"

프롬프트 앙상블:
templates = [
    "A photo of a {}",
    "A picture of a {}",
    "An image showing a {}",
    "A {} in the scene"
]

# 여러 템플릿의 평균
text_embeddings = []
for template in templates:
    prompt = template.format(class_name)
    embedding = text_encoder(prompt)
    text_embeddings.append(embedding)
final_embedding = mean(text_embeddings)
```

### 3. 응용 분야

```
1. Zero-shot Classification
   - 새로운 도메인에 바로 적용
   - 프롬프트로 클래스 정의

2. Image-Text Retrieval
   - 텍스트로 이미지 검색
   - 이미지로 텍스트 검색

3. Image Generation Guidance
   - DALL-E, Stable Diffusion의 guidance
   - CLIP score로 생성 품질 측정

4. Multimodal Embedding
   - 이미지와 텍스트의 공통 표현
   - 다운스트림 태스크 기반
```

---

## 구현 레벨

### Level 2: PyTorch Low-Level (pytorch_lowlevel/)
- Image encoder (ViT) 직접 구현
- Text encoder (Transformer) 직접 구현
- Contrastive loss 구현

### Level 3: Paper Implementation (paper/)
- 전체 학습 파이프라인
- Zero-shot 평가
- Prompt engineering

### Level 4: Code Analysis (별도)
- OpenAI CLIP 코드 분석
- open_clip 라이브러리 분석

---

## 학습 체크리스트

- [ ] Contrastive learning 이해
- [ ] InfoNCE loss 수식 이해
- [ ] Zero-shot classification 구현
- [ ] Temperature의 역할 이해
- [ ] Prompt engineering 실습
- [ ] Image-text retrieval 구현

---

## 참고 자료

- Radford et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision"
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [CLIP과 멀티모달 학습](./34_CLIP_Multimodal.md)
