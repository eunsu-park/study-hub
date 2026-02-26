# 16. Vision-Language 심화

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. CLIP에서 LLaVA, Qwen-VL까지 비전-언어 모델(Vision-Language Model)의 발전 과정을 추적하고 각 단계의 핵심 아키텍처 혁신을 설명할 수 있다
2. 선형 프로젝션(linear projection)과 크로스 어텐션(cross-attention) 방법을 사용하여 시각적 토큰(visual token)이 대형 언어 모델(LLM) 어텐션 레이어에 어떻게 투영되고 통합되는지 설명할 수 있다
3. Visual Instruction Tuning을 구현하여 이미지 캡셔닝(image captioning)과 시각적 질의응답(Visual Question Answering) 등 특정 멀티모달 태스크에 VLM을 파인튜닝할 수 있다
4. 다양한 VLM이 사용하는 연결 방식(선형 프로젝션, 크로스 어텐션, MLP)을 비교하고 성능에 미치는 영향을 분석할 수 있다
5. 다중 이미지 및 고해상도 시나리오에서 VLM의 역량을 평가하고 현재 한계와 새로운 해결책을 파악할 수 있다

---

## 개요

Vision-Language Models (VLMs)는 이미지와 텍스트를 함께 이해하는 모델입니다. 이 레슨에서는 LLaVA, Qwen-VL 등 최신 VLM 아키텍처와 Visual Instruction Tuning 기법을 다룹니다.

---

## 1. VLM 패러다임

### 1.1 발전 과정

```
┌──────────────────────────────────────────────────────────────────┐
│                    VLM 발전 과정                                 │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  2021: CLIP                                                      │
│  - Image-Text contrastive learning                              │
│  - Zero-shot 분류 가능                                          │
│                                                                  │
│  2022: Flamingo                                                  │
│  - LLM에 visual tokens 주입                                     │
│  - Few-shot 비전-언어 학습                                      │
│                                                                  │
│  2023: LLaVA                                                     │
│  - Visual Instruction Tuning                                    │
│  - 오픈소스 GPT-4V 대안                                         │
│                                                                  │
│  2024: LLaVA-NeXT, Qwen-VL, Phi-3-Vision                        │
│  - 고해상도, 다중 이미지, 비디오                                 │
│  - 상용 수준 성능                                                │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 아키텍처 비교

| 모델 | Vision Encoder | LLM | 연결 방식 |
|------|---------------|-----|----------|
| **LLaVA** | CLIP ViT-L | Vicuna/LLaMA | Linear Projection |
| **Qwen-VL** | ViT-G | Qwen | Cross-Attention |
| **InternVL** | InternViT | InternLM | MLP |
| **Phi-3-Vision** | CLIP ViT | Phi-3 | Linear |
| **GPT-4V** | Unknown | GPT-4 | Unknown |

---

## 2. LLaVA (Large Language and Vision Assistant)

### 2.1 아키텍처

```
LLaVA 구조:

이미지 → CLIP ViT-L/14 → Visual Features (576 tokens)
                ↓
         Linear Projection
                ↓
         Visual Tokens
                ↓
[System] [Visual Tokens] [User Query] → LLaMA/Vicuna → Response

학습 단계:
1. Pre-training: Image-Text alignment (CC3M)
2. Fine-tuning: Visual Instruction Tuning (158K)
```

### 2.2 구현

```python
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, LlamaForCausalLM, LlamaTokenizer

class LLaVAModel(nn.Module):
    """LLaVA 스타일 Vision-Language Model"""

    def __init__(
        self,
        vision_encoder: str = "openai/clip-vit-large-patch14",
        llm: str = "lmsys/vicuna-7b-v1.5",
        freeze_vision: bool = True,
        freeze_llm: bool = False
    ):
        super().__init__()

        # Vision Encoder
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_encoder)
        self.vision_hidden_size = self.vision_encoder.config.hidden_size

        # Language Model
        self.llm = LlamaForCausalLM.from_pretrained(llm)
        self.llm_hidden_size = self.llm.config.hidden_size

        # Vision-Language Projection
        self.vision_projection = nn.Linear(
            self.vision_hidden_size,
            self.llm_hidden_size
        )

        # Freeze encoders
        if freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        이미지 인코딩

        Args:
            images: (B, C, H, W)

        Returns:
            visual_tokens: (B, num_patches, llm_hidden_size)
        """
        # CLIP encoding
        vision_outputs = self.vision_encoder(images)
        image_features = vision_outputs.last_hidden_state  # (B, 257, 1024)

        # [CLS] 토큰 제외
        image_features = image_features[:, 1:, :]  # (B, 256, 1024)

        # Project to LLM space
        visual_tokens = self.vision_projection(image_features)

        return visual_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: torch.Tensor = None,
        image_positions: torch.Tensor = None,
        labels: torch.Tensor = None
    ):
        """
        Forward pass

        Args:
            input_ids: (B, seq_len) 텍스트 토큰
            attention_mask: (B, seq_len)
            images: (B, C, H, W) 이미지
            image_positions: 이미지 토큰이 들어갈 위치
            labels: (B, seq_len) for training
        """
        B, seq_len = input_ids.shape

        # Text embeddings
        text_embeds = self.llm.model.embed_tokens(input_ids)

        # Image embeddings
        if images is not None:
            visual_tokens = self.encode_images(images)  # (B, num_patches, hidden)

            # Interleave visual tokens with text
            # 간소화: 이미지를 텍스트 앞에 추가
            combined_embeds = torch.cat([visual_tokens, text_embeds], dim=1)

            # Attention mask 조정
            visual_mask = torch.ones(B, visual_tokens.shape[1], device=attention_mask.device)
            combined_mask = torch.cat([visual_mask, attention_mask], dim=1)
        else:
            combined_embeds = text_embeds
            combined_mask = attention_mask

        # LLM forward
        outputs = self.llm(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            labels=labels,
            return_dict=True
        )

        return outputs


class VisualInstructionDataset:
    """Visual Instruction Tuning 데이터셋"""

    INSTRUCTION_TEMPLATES = [
        "Describe this image in detail.",
        "What can you see in this image?",
        "Explain what is happening in this picture.",
        "<question>",  # VQA
    ]

    def __init__(self, data_path: str):
        """
        데이터 형식:
        {
            "image": "path/to/image.jpg",
            "conversations": [
                {"from": "human", "value": "<image>\nDescribe this image."},
                {"from": "gpt", "value": "This image shows..."}
            ]
        }
        """
        import json
        with open(data_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 이미지 로드
        from PIL import Image
        image = Image.open(item['image']).convert('RGB')

        # 대화 구성
        conversations = item['conversations']
        human_input = conversations[0]['value']
        assistant_output = conversations[1]['value']

        return {
            'image': image,
            'human': human_input,
            'assistant': assistant_output
        }
```

### 2.3 LLaVA-NeXT 개선점

```python
class LLaVANeXTConfig:
    """
    LLaVA-NeXT 개선 사항

    1. 고해상도 지원 (AnyRes)
    2. 더 나은 Vision Encoder (SigLIP)
    3. 더 큰 LLM (Llama 3, Qwen 2)
    """

    # AnyRes: 다양한 해상도 처리
    SUPPORTED_RESOLUTIONS = [
        (336, 336),
        (672, 336),
        (336, 672),
        (672, 672),
        (1008, 336),
        (336, 1008),
    ]

    @staticmethod
    def select_best_resolution(image_size: tuple, resolutions: list):
        """이미지에 가장 적합한 해상도 선택"""
        img_h, img_w = image_size
        img_ratio = img_w / img_h

        best_res = None
        best_ratio_diff = float('inf')

        for res in resolutions:
            res_ratio = res[1] / res[0]
            ratio_diff = abs(img_ratio - res_ratio)

            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_res = res

        return best_res


def anyres_processing(image, base_resolution=336):
    """
    AnyRes 이미지 처리

    고해상도 이미지를 기본 해상도 타일로 분할
    + 전체 이미지 축소본
    """
    from PIL import Image
    import torch

    # 1. 전체 이미지 리사이즈 (전역 컨텍스트)
    global_image = image.resize((base_resolution, base_resolution))

    # 2. 타일 분할 (지역 디테일)
    W, H = image.size
    num_tiles_w = (W + base_resolution - 1) // base_resolution
    num_tiles_h = (H + base_resolution - 1) // base_resolution

    tiles = []
    for i in range(num_tiles_h):
        for j in range(num_tiles_w):
            left = j * base_resolution
            top = i * base_resolution
            right = min(left + base_resolution, W)
            bottom = min(top + base_resolution, H)

            tile = image.crop((left, top, right, bottom))
            # 패딩
            padded_tile = Image.new('RGB', (base_resolution, base_resolution))
            padded_tile.paste(tile, (0, 0))
            tiles.append(padded_tile)

    # [global_image] + [tile1, tile2, ...]
    all_images = [global_image] + tiles

    return all_images
```

---

## 3. Qwen-VL

### 3.1 아키텍처

```
Qwen-VL 특징:

1. Vision Encoder: ViT-bigG (1.9B params)
2. 고해상도: 448×448 (가변)
3. Grounding 지원: 바운딩 박스 출력
4. OCR 강점: 텍스트 인식 우수

입력 형식:
<img>image_path</img> User question
<ref>object name</ref><box>(x1,y1),(x2,y2)</box>
```

### 3.2 사용 예시

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

def use_qwen_vl():
    """Qwen-VL 사용"""

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen-VL-Chat",
        trust_remote_code=True,
        torch_dtype=torch.float16
    ).to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen-VL-Chat",
        trust_remote_code=True
    )

    # 기본 VQA
    query = tokenizer.from_list_format([
        {'image': 'path/to/image.jpg'},
        {'text': 'What is in this image?'},
    ])

    response, history = model.chat(tokenizer, query=query, history=None)
    print(response)

    # Grounding (객체 위치 찾기)
    query = tokenizer.from_list_format([
        {'image': 'path/to/image.jpg'},
        {'text': 'Find all the cats in this image and output their bounding boxes.'},
    ])

    response, history = model.chat(tokenizer, query=query, history=None)
    # 출력: <ref>cat</ref><box>(100,200),(300,400)</box>

    # 다중 이미지
    query = tokenizer.from_list_format([
        {'image': 'image1.jpg'},
        {'image': 'image2.jpg'},
        {'text': 'What is the difference between these two images?'},
    ])

    response, history = model.chat(tokenizer, query=query, history=None)

    return response
```

---

## 4. Visual Instruction Tuning

### 4.1 데이터 생성

```python
class VisualInstructionGenerator:
    """Visual Instruction 데이터 생성"""

    def __init__(self, teacher_model="gpt-4-vision-preview"):
        from openai import OpenAI
        self.client = OpenAI()
        self.teacher_model = teacher_model

    def generate_conversation(
        self,
        image_path: str,
        task_type: str = "detailed_description"
    ):
        """GPT-4V로 학습 데이터 생성"""
        import base64

        # 이미지 인코딩
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()

        task_prompts = {
            "detailed_description": "Describe this image in detail.",
            "reasoning": "What conclusions can you draw from this image? Explain your reasoning.",
            "conversation": "Generate a multi-turn conversation about this image.",
            "creative": "Write a creative story inspired by this image."
        }

        prompt = task_prompts.get(task_type, task_prompts["detailed_description"])

        response = self.client.chat.completions.create(
            model=self.teacher_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                    ]
                }
            ],
            max_tokens=1024
        )

        return {
            "image": image_path,
            "task": task_type,
            "question": prompt,
            "answer": response.choices[0].message.content
        }

    def generate_dataset(
        self,
        image_paths: list,
        output_path: str,
        tasks: list = None
    ):
        """대규모 데이터셋 생성"""
        import json
        from tqdm import tqdm

        if tasks is None:
            tasks = ["detailed_description", "reasoning", "conversation"]

        dataset = []

        for image_path in tqdm(image_paths):
            for task in tasks:
                try:
                    data = self.generate_conversation(image_path, task)
                    dataset.append(data)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)

        return dataset
```

### 4.2 학습 전략

```python
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

def finetune_vlm():
    """VLM Fine-tuning"""

    # 모델 로드
    model = LLaVAModel(
        freeze_vision=True,  # Vision encoder 고정
        freeze_llm=False     # LLM fine-tune
    )

    # LoRA 적용 (효율적 학습)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
    )

    model.llm = get_peft_model(model.llm, lora_config)

    # 학습 설정
    training_args = TrainingArguments(
        output_dir="./llava-finetuned",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=2e-5,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        save_steps=500,
        dataloader_num_workers=4,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=vlm_data_collator,
    )

    trainer.train()


def vlm_data_collator(features):
    """VLM 데이터 콜레이터"""
    batch = {
        'input_ids': torch.stack([f['input_ids'] for f in features]),
        'attention_mask': torch.stack([f['attention_mask'] for f in features]),
        'images': torch.stack([f['image'] for f in features]),
        'labels': torch.stack([f['labels'] for f in features]),
    }
    return batch
```

---

## 5. 평가 벤치마크

### 5.1 주요 벤치마크

```
VLM 평가 벤치마크:

1. VQA-v2: 일반 Visual QA
2. GQA: 구조적 추론 QA
3. TextVQA: 이미지 내 텍스트 이해
4. POPE: 환각(hallucination) 평가
5. MME: 14개 하위 태스크 종합
6. MMBench: 20개 능력 평가
7. SEED-Bench: 19K 다지선다 문제
```

### 5.2 평가 코드

```python
def evaluate_vlm(model, dataset_name: str = "vqav2"):
    """VLM 평가"""

    if dataset_name == "vqav2":
        return evaluate_vqa_v2(model)
    elif dataset_name == "textvqa":
        return evaluate_textvqa(model)
    elif dataset_name == "pope":
        return evaluate_pope(model)


def evaluate_pope(model):
    """
    POPE: Polling-based Object Probing Evaluation

    환각 평가: "Is there a [object] in the image?"
    """
    from datasets import load_dataset

    dataset = load_dataset("lmms-lab/POPE")

    correct = 0
    total = 0

    for item in dataset['test']:
        image = item['image']
        question = item['question']  # "Is there a dog in the image?"
        answer = item['answer']      # "yes" or "no"

        # 모델 예측
        prediction = model.generate(image, question)
        pred_answer = "yes" if "yes" in prediction.lower() else "no"

        if pred_answer == answer:
            correct += 1
        total += 1

    accuracy = correct / total
    print(f"POPE Accuracy: {accuracy:.4f}")

    return accuracy
```

---

## 6. 실전 응용

### 6.1 문서 이해

```python
def document_understanding():
    """문서 이해 응용"""

    model = load_qwen_vl()  # OCR 강점

    # PDF 페이지 분석
    def analyze_document_page(image_path: str, questions: list):
        results = []

        for question in questions:
            query = f"<img>{image_path}</img>{question}"
            answer = model.generate(query)
            results.append({
                'question': question,
                'answer': answer
            })

        return results

    # 예시 질문
    questions = [
        "What is the title of this document?",
        "Summarize the main points.",
        "Extract all dates mentioned.",
        "What tables are present? Describe their contents.",
    ]

    results = analyze_document_page("document_page.png", questions)


def chart_understanding():
    """차트/그래프 이해"""

    prompts = [
        "What type of chart is this?",
        "What is the trend shown in this chart?",
        "What are the maximum and minimum values?",
        "Describe the relationship between X and Y.",
    ]

    # VLM으로 차트 분석
    for prompt in prompts:
        response = model.generate(chart_image, prompt)
        print(f"Q: {prompt}")
        print(f"A: {response}\n")
```

---

## 참고 자료

### 논문
- Liu et al. (2023). "Visual Instruction Tuning" (LLaVA)
- Liu et al. (2024). "LLaVA-NeXT: Improved reasoning, OCR, and world knowledge"
- Bai et al. (2023). "Qwen-VL: A Versatile Vision-Language Model"

### 모델
- [LLaVA](https://llava-vl.github.io/)
- [Qwen-VL](https://huggingface.co/Qwen/Qwen-VL-Chat)
- [InternVL](https://github.com/OpenGVLab/InternVL)

### 관련 레슨
- [../Deep_Learning/20_CLIP_Multimodal.md](../Deep_Learning/20_CLIP_Multimodal.md)
- [12_DINOv2_Self_Supervised.md](12_DINOv2_Self_Supervised.md)

---

## 연습 문제

### 연습 문제 1: VLM 연결 방법 비교
레슨의 표는 비전 인코더를 LLM에 연결하는 세 가지 방법을 보여줍니다: 선형 투영(Linear Projection)(LLaVA), 교차 어텐션(Cross-Attention)(Qwen-VL), MLP. 각 방법의 트레이드오프를 분석하고 각각이 가장 적합한 경우를 설명하세요.

| 연결 방법 | 훈련 파라미터 | 지연 시간 영향 | 최적 사용 사례 |
|----------|------------|-------------|------------|
| 선형 투영(Linear Projection) | ??? | ??? | ??? |
| MLP (2-3 레이어) | ??? | ??? | ??? |
| 교차 어텐션(Cross-Attention) | ??? | ??? | ??? |

<details>
<summary>정답 보기</summary>

| 연결 방법 | 훈련 파라미터 | 지연 시간 영향 | 최적 사용 사례 |
|----------|------------|-------------|------------|
| 선형 투영 | 최소 (vision_dim × llm_dim) | 최소 — 단일 행렬 곱 | 빠른 훈련/추론; 리소스 제한 환경; 시각 특징이 이미 고품질인 경우 (CLIP-L) |
| MLP (2-3 레이어) | 보통 | 낮음 — 순차적 소형 레이어 | 선형보다 나은 모달리티 정렬; 가장 일반적인 균형 선택 (LLaVA-1.5는 2-레이어 MLP 사용) |
| 교차 어텐션 | 대형 (완전한 교차 어텐션 블록) | 높음 — 각 LLM 레이어에서 모든 시각 토큰에 대한 어텐션 | 시각-언어 상호작용이 깊고 맥락적이어야 할 때; 공간적 디테일이 중요한 고해상도 이미지 |

**상세 분석**:

**선형 투영** (LLaVA 원본): 단일 가중치 행렬로 시각 특징을 LLM 임베딩 공간에 직접 매핑합니다. 장점: 훈련이 매우 빠름(1단계에서 투영 레이어만 훈련), 추론 오버헤드 거의 없음. 단점: 모달리티 정렬 용량 제한 — 시각과 언어 공간 간의 복잡한 비선형 매핑을 학습할 수 없음.

**MLP** (LLaVA-1.5): 활성화 함수를 가진 1-2개의 은닉층 추가. 추가 용량이 시각 이해 벤치마크를 크게 향상시킵니다(예: 2-레이어 MLP를 사용한 LLaVA-1.5는 선형 투영을 사용한 LLaVA-1.0을 크게 능가). 용량과 효율성의 좋은 균형.

**교차 어텐션** (Flamingo, Qwen-VL): LLM의 전용 교차 어텐션 레이어를 통해 모든 텍스트 토큰이 각 레이어에서 시각 토큰에 어텐션할 수 있습니다. 더 풍부한 시각-언어 그라운딩(grounding) 가능. OCR, 그라운딩(grounding), 계수와 같이 세밀한 시각적 읽기가 필요한 태스크나 고해상도 이미지 타일 처리에 최적. 계산 비용이 크게 증가.

</details>

### 연습 문제 2: 시각적 지시 튜닝(Visual Instruction Tuning) 데이터 품질
LLaVA는 GPT-4V를 "교사" 모델로 사용하여 훈련 데이터를 생성합니다. (A) 이 접근법이 잘 포착할 가능성이 높은 시각적 이해 유형, (B) 놓치거나 왜곡할 가능성이 높은 것, (C) 데이터 생성 파이프라인에 대한 구체적인 개선 방법을 파악하여 이 접근법의 품질과 커버리지를 분석하세요.

<details>
<summary>정답 보기</summary>

**A) GPT-4V 생성 데이터가 잘 포착하는 것**:
- **서술적 및 내러티브 능력**: 이미지 캡셔닝(captioning), 상세한 장면 설명 — GPT-4V가 뛰어납니다.
- **일반적인 시각적 추론**: "왼쪽 사람이 웃고 있는 것 같은 이유는..." — 이미지에 대한 일반적인 상식 추론.
- **다중 턴 대화 형식**: GPT-4V는 이미지에 대한 현실적인 주고받는 대화를 생성할 수 있습니다.
- **고자원 도메인**: 일상 용품, 실내/실외 장면, 일반적인 활동 — GPT-4V의 훈련 데이터에 잘 표현됨.

**B) 놓치거나 왜곡하는 것**:
- **도메인별 정확도**: 의료 영상, 특수 기술 다이어그램, 과학 데이터 시각화 — GPT-4V가 그럴듯하게 들리지만 부정확한 설명을 생성할 수 있음.
- **정밀한 공간적 추론**: 정확한 객체 수, 정밀한 측정, 세밀한 공간 관계("세 번째 선반의 왼쪽에서 두 번째 객체") — GPT-4V는 오류를 범하고 학생 모델이 이를 복제하도록 학습.
- **보정된 불확실성**: GPT-4V는 불확실할 때도 자신감 있는 경향이 있어, 학생이 불확실성을 표현하는 대신 환각(hallucination)을 하도록 훈련.
- **데이터셋 편향**: GPT-4V의 편향(문화적, 인구통계학적, 언어적)을 반영하는 설명 생성, 학생에서 증폭됨.

**C) 구체적인 개선**:
사실적 태스크에 GPT-4V 생성 대신 **검증된 실측 데이터 소스**를 사용합니다. 예시: 객체 계수의 경우, COCO 어노테이션(여러 사람이 검증)을 사용하여 "이 이미지에 고양이가 몇 마리 있나요? 3마리입니다."와 같은 QA 쌍을 생성합니다. 이렇게 하면 학생이 GPT-4V의 근사 답변이 아닌 정확한 계수를 학습합니다. 하이브리드 파이프라인: 설명적/대화적 데이터는 GPT-4V로, 정량적/사실적 태스크는 실측 어노테이션으로.

</details>

### 연습 문제 3: AnyRes 고해상도 처리
LLaVA-NeXT는 "AnyRes"를 사용하여 고해상도 이미지를 처리합니다. 1680×672 픽셀의 입력 이미지와 기본 해상도 336×336이 주어졌을 때, AnyRes 처리 파이프라인을 단계별로 추적하세요:

```python
def anyres_processing(image, base_resolution=336):
    """
    AnyRes 처리:
    1. 전체 이미지를 336×336으로 리사이즈 (글로벌 컨텍스트)
    2. 336×336 타일로 분할 (로컬 디테일)
    3. 반환: [global_image] + [tile1, tile2, ..., tileN]
    """
    W, H = image.size  # (1680, 672)
    # 타일 수는? 총 토큰 수는?
    pass

# 주어진 조건:
# - 각 336×336 이미지는 576개의 시각 토큰 생성 (14px 패치로 24×24 패치)
# - 전체 이미지는 336×336으로 리사이즈 → 576 토큰
# - 각 타일은 336×336 → 576 토큰
# 계산: 1680×672 이미지의 총 시각 토큰 수
```

<details>
<summary>정답 보기</summary>

```python
# 단계 1: 타일 수 계산
W, H = 1680, 672
base_resolution = 336

num_tiles_w = (W + base_resolution - 1) // base_resolution
            = (1680 + 336 - 1) // 336
            = 2015 // 336 = 5 타일 (가로 방향)

num_tiles_h = (H + base_resolution - 1) // base_resolution
            = (672 + 336 - 1) // 336
            = 1007 // 336 = 2 타일 (세로 방향, 올림 나눗셈)

# 실제로: 672 / 336 = 2.0 정확히 → 세로 방향 2 타일
# 1680 / 336 = 5.0 정확히 → 가로 방향 5 타일
total_tiles = 5 × 2 = 10 타일

# 단계 2: 총 시각 토큰 계산
tokens_per_image = 576  # 336×336 이미지당 24×24 패치

global_tokens = 576          # 1개 리사이즈된 전체 이미지
tile_tokens = 10 × 576       # 10개 로컬 디테일 타일
              = 5760

total_visual_tokens = 576 + 5760 = 6336 토큰

# 비교:
# - LLaVA 1.0 (단일 224×224): 256 토큰
# - LLaVA-NeXT (이 이미지): 6336 토큰 (24.75배 더 많음)

# 메모리 영향:
# 4096-차원 LLM에서 6336 시각 + ~100 텍스트 토큰에 대한 어텐션
# 어텐션 행렬: (6436)^2 = ~4100만 항목 (레이어당)
# vs. LLaVA 1.0: (356)^2 = ~12만 7천 항목 (레이어당)
# → 레이어당 약 330배 더 많은 어텐션 계산
```

이것이 고해상도 VLM 추론이 왜 비용이 많이 들고, 시각 토큰 압축(예: 시각 토큰 병합)이 왜 활발한 연구 분야인지를 설명합니다.

</details>

### 연습 문제 4: POPE 환각(Hallucination) 평가
POPE 벤치마크는 객체에 대한 이진 예/아니오 질문("이 이미지에 개가 있나요?")으로 VLM 환각(hallucination)을 평가합니다. 높은 POPE 정확도를 가진 VLM이 실제로 여전히 해로운 환각을 보일 수 있는 시나리오를 설계하고, 더 포괄적인 평가 방법을 제안하세요.

<details>
<summary>정답 보기</summary>

**높은 POPE 정확도에서도 해로운 환각이 발생하는 시나리오**:

POPE는 단순한 이진 질문을 사용하여 일반적인 COCO 객체의 존재/부재만 테스트합니다. 다음을 고려하세요:

**시나리오: 의료 문서 분석**
- 이미지: 우상엽 폐에 작은 결절이 있는 흉부 X선을 보여주는 방사선 보고서.
- POPE 스타일 질문: "이 이미지에 결절이 있나요?" — 모델이 정확하게 "예"라고 답합니다. (POPE 점수: 환각 없음)
- POPE가 포착하지 못한 해로운 환각: 모델이 "결절은 좌하엽에 위치하며, 약 2cm이고, 악성을 시사하는 불규칙적인 경계를 보입니다."라고 설명 — 위치 오류, 크기 날조, 악성 평가 환각.
- POPE는 이 모델에 우수한 점수를 줄 것이지만, 실제로는 오진으로 이어질 수 있습니다.

**POPE가 놓치는 다른 실패 모드**:
- 속성 환각: 트럭이 파란색인데 "빨간 소방차"라고 함.
- 관계 환각: 고양이가 탁자 아래에 있는데 "고양이가 탁자 위에 있다"고 함.
- 계수 환각: 5명이 있는데 3명이 있다고 함.
- 날조된 텍스트: 표지판/라벨의 텍스트를 없는 것을 만들거나 잘못 읽음.

**더 포괄적인 평가**:

1. **속성 환각 벤치마크**: "자동차 색깔이 무엇인가요?"와 같은 질문 — 설명된 속성이 실측과 일치하는지 테스트.

2. **공간 관계 벤치마크**: "컵이 접시의 왼쪽인가요 오른쪽인가요?" — 공간적 이해 정확도 테스트.

3. **계수 벤치마크**: "이 이미지에 사람이 몇 명 있나요?" — 단순 존재가 아닌 개수 정확도 평가.

4. **세밀한 OCR 검증**: 텍스트가 있는 이미지의 경우, 모델의 전사를 검증된 실측과 비교.

5. **신뢰도 보정 테스트**: 모델의 표현된 불확실성(예: "아마도 있을 것 같습니다...")이 실제 정확도와 상관관계가 있는지 측정 — 잘 보정된 모델은 틀릴 때 불확실해야 함.

</details>
