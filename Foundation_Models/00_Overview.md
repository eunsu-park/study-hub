# Foundation Models 학습 가이드

## 개요

Foundation Models(기반 모델)은 대규모 데이터로 사전 학습되어 다양한 하위 작업에 적용 가능한 모델을 의미합니다. 이 폴더는 Foundation Model의 **패러다임**, **Scaling Laws**, **최신 아키텍처**, 그리고 **실무 적용**을 다룹니다.

### 선수 지식
- **Deep_Learning 폴더**: ViT, CLIP, Self-Supervised Learning, Transformer
- **LLM_and_NLP 폴더**: BERT, GPT, HuggingFace, Fine-tuning, RAG

### 학습 목표
1. Foundation Model 패러다임과 Scaling Laws 이해
2. LLaMA, Mistral, DINOv2, SAM 등 최신 모델 아키텍처 파악
3. 효율적인 적응(PEFT) 및 배포 전략 습득
4. Multimodal Foundation Models의 동작 원리 이해

---

## 파일 목록

### Section 1: Foundation Model 패러다임 (01-03)
| 파일 | 주제 | 핵심 내용 | 난이도 |
|------|------|----------|--------|
| [01_Foundation_Model_Paradigm.md](01_Foundation_Model_Paradigm.md) | FM 패러다임 | 정의, 역사, In-context Learning, Emergent Capabilities | ⭐⭐ |
| [02_Scaling_Laws.md](02_Scaling_Laws.md) | Scaling Laws | Chinchilla, Compute-optimal, Power Laws | ⭐⭐⭐ |
| [03_Emergent_Abilities.md](03_Emergent_Abilities.md) | 창발적 능력 | CoT 출현, Phase Transitions, Capability Elicitation | ⭐⭐⭐ |

### Section 2: Pre-training Deep Dive (04-07)
| 파일 | 주제 | 핵심 내용 | 난이도 |
|------|------|----------|--------|
| [04_Pretraining_Objectives.md](04_Pretraining_Objectives.md) | 목적함수 | Causal LM, Masked LM, Prefix LM, UL2 | ⭐⭐⭐ |
| [05_Data_Curation.md](05_Data_Curation.md) | 데이터 큐레이션 | The Pile, RedPajama, 중복제거, 품질 필터링 | ⭐⭐⭐ |
| [06_Pretraining_Infrastructure.md](06_Pretraining_Infrastructure.md) | 학습 인프라 | FSDP, DeepSpeed ZeRO, 분산학습 | ⭐⭐⭐⭐ |
| [07_Tokenization_Advanced.md](07_Tokenization_Advanced.md) | Tokenization | BPE, Unigram, 다국어, Tokenizer-free | ⭐⭐⭐ |

### Section 3: 최신 LLM 아키텍처 (08-11)
| 파일 | 주제 | 핵심 내용 | 난이도 |
|------|------|----------|--------|
| [08_LLaMA_Family.md](08_LLaMA_Family.md) | LLaMA | LLaMA 1/2/3, RoPE, RMSNorm, SwiGLU, GQA | ⭐⭐⭐ |
| [09_Mistral_MoE.md](09_Mistral_MoE.md) | Mistral & MoE | Mixtral, Sparse MoE, Router 설계, 효율성 | ⭐⭐⭐⭐ |
| [10_Long_Context_Models.md](10_Long_Context_Models.md) | Long Context | Longformer, Ring Attention, YaRN, PI | ⭐⭐⭐ |
| [11_Small_Language_Models.md](11_Small_Language_Models.md) | 소형 LM | Phi, Gemma, Qwen, TinyLlama, 지식 증류 | ⭐⭐⭐ |

### Section 4: Vision Foundation Models (12-15)
| 파일 | 주제 | 핵심 내용 | 난이도 |
|------|------|----------|--------|
| [12_DINOv2_Self_Supervised.md](12_DINOv2_Self_Supervised.md) | DINOv2 | DINO, DINOv2, Self-distillation, Dense Features | ⭐⭐⭐ |
| [13_Segment_Anything.md](13_Segment_Anything.md) | SAM | Promptable Segmentation, Image/Prompt/Mask Encoder | ⭐⭐⭐⭐ |
| [14_Unified_Vision_Models.md](14_Unified_Vision_Models.md) | 통합 Vision | Florence, PaLI, Unified-IO | ⭐⭐⭐⭐ |
| [15_Image_Generation_Advanced.md](15_Image_Generation_Advanced.md) | 이미지 생성 심화 | SDXL, ControlNet, IP-Adapter, LCM | ⭐⭐⭐⭐ |

### Section 5: Multimodal Foundation Models (16-18)
| 파일 | 주제 | 핵심 내용 | 난이도 |
|------|------|----------|--------|
| [16_Vision_Language_Deep.md](16_Vision_Language_Deep.md) | Vision-Language | LLaVA, Qwen-VL, Visual Instruction Tuning | ⭐⭐⭐⭐ |
| [17_GPT4V_Gemini.md](17_GPT4V_Gemini.md) | GPT-4V & Gemini | Multimodal Input, Interleaved, API 활용 | ⭐⭐⭐ |
| [18_Audio_Video_Foundation.md](18_Audio_Video_Foundation.md) | Audio/Video | Whisper, AudioLM, MusicGen, VideoLLaMA | ⭐⭐⭐⭐ |

### Section 6: Efficient Adaptation (19-21)
| 파일 | 주제 | 핵심 내용 | 난이도 |
|------|------|----------|--------|
| [19_PEFT_Unified.md](19_PEFT_Unified.md) | PEFT 통합 | LoRA, QLoRA, DoRA, Adapters, IA3 | ⭐⭐⭐ |
| [20_Instruction_Tuning.md](20_Instruction_Tuning.md) | Instruction Tuning | FLAN, Self-Instruct, Evol-Instruct | ⭐⭐⭐ |
| [21_Continued_Pretraining.md](21_Continued_Pretraining.md) | Continued Pre-training | 도메인 적응, Catastrophic Forgetting 방지 | ⭐⭐⭐⭐ |

### Section 7: 배포와 프로덕션 (22-24)
| 파일 | 주제 | 핵심 내용 | 난이도 |
|------|------|----------|--------|
| [22_Inference_Optimization.md](22_Inference_Optimization.md) | Inference 최적화 | vLLM, TGI, Speculative Decoding, PagedAttention | ⭐⭐⭐ |
| [23_Advanced_RAG.md](23_Advanced_RAG.md) | Advanced RAG | Agentic RAG, HyDE, RAPTOR, ColBERT | ⭐⭐⭐⭐ |
| [24_API_and_Evaluation.md](24_API_and_Evaluation.md) | API & 평가 | OpenAI/Anthropic/Google API, 벤치마크 | ⭐⭐⭐ |

### Section 8: 미래 방향 (25)
| 파일 | 주제 | 핵심 내용 | 난이도 |
|------|------|----------|--------|
| [25_Research_Frontiers.md](25_Research_Frontiers.md) | 연구 최전선 | World Models, o1 Reasoning, Synthetic Data | ⭐⭐⭐⭐ |

---

## 학습 로드맵

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Foundation Models 학습 경로                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  [선수 학습]                                                             │
│  Deep_Learning (ViT, CLIP, Transformer) + LLM_and_NLP (BERT, GPT, RAG)  │
│                              │                                          │
│                              ▼                                          │
│  ┌──────────────────────────────────────────────┐                       │
│  │      Phase 1: 패러다임 (Week 1)               │                       │
│  │      01 → 02 → 03                            │                       │
│  │      (FM 정의 → Scaling Laws → Emergence)     │                       │
│  └──────────────────────────────────────────────┘                       │
│                              │                                          │
│              ┌───────────────┴───────────────┐                          │
│              ▼                               ▼                          │
│  ┌─────────────────────┐        ┌─────────────────────┐                 │
│  │  Path A: LLM 중심    │        │  Path B: Vision 중심 │                 │
│  │  04-11 (Pre-train   │        │  12-15 (DINOv2,     │                 │
│  │  + LLM 아키텍처)     │        │  SAM, 이미지 생성)   │                 │
│  └─────────────────────┘        └─────────────────────┘                 │
│              │                               │                          │
│              └───────────────┬───────────────┘                          │
│                              ▼                                          │
│  ┌──────────────────────────────────────────────┐                       │
│  │      Phase 3: Multimodal (Week 3-4)          │                       │
│  │      16 → 17 → 18                            │                       │
│  │      (LLaVA → GPT-4V → Audio/Video)          │                       │
│  └──────────────────────────────────────────────┘                       │
│                              │                                          │
│                              ▼                                          │
│  ┌──────────────────────────────────────────────┐                       │
│  │      Phase 4: 실무 적용 (Week 5-6)            │                       │
│  │      19 → 20 → 21 → 22 → 23 → 24             │                       │
│  │      (PEFT → Instruction → Deploy → RAG)     │                       │
│  └──────────────────────────────────────────────┘                       │
│                              │                                          │
│                              ▼                                          │
│  ┌──────────────────────────────────────────────┐                       │
│  │      Phase 5: 미래 방향 (Week 7)              │                       │
│  │      25 (Research Frontiers)                 │                       │
│  └──────────────────────────────────────────────┘                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 기존 폴더와의 관계

### Deep_Learning 폴더와의 연계
| Deep_Learning 레슨 | Foundation_Models 확장 |
|-------------------|----------------------|
| 19_ViT | 10_Long_Context (ViT 기반 확장) |
| 20_CLIP | 16_Vision_Language_Deep (LLaVA 등) |
| 21_Self_Supervised | 12_DINOv2 (최신 SSL) |
| 17_Diffusion | 15_Image_Generation_Advanced (SDXL, ControlNet) |

### LLM_and_NLP 폴더와의 연계
| LLM_and_NLP 레슨 | Foundation_Models 확장 |
|-----------------|----------------------|
| 04-05_BERT_GPT | 08-09_LLaMA_Mistral (최신 오픈소스) |
| 07_Fine_Tuning | 19_PEFT_Unified (LoRA 변형 통합) |
| 09_RAG | 23_Advanced_RAG (Agentic RAG 등) |
| 13_Quantization | 22_Inference_Optimization (vLLM, Speculative) |

---

## 권장 학습 순서

### 빠른 실무 적용 (2주)
```
01 → 02 → 08 → 09 → 19 → 22
(패러다임 → Scaling → LLaMA → Mistral → PEFT → Inference)
```

### Vision Foundation 집중 (2주)
```
01 → 03 → 12 → 13 → 14 → 15
(패러다임 → Emergence → DINOv2 → SAM → Unified → 이미지 생성)
```

### Multimodal 전문가 (3주)
```
01 → 02 → 03 → 12 → 16 → 17 → 18 → 23
(기초 → Vision → VLM → GPT-4V → Audio/Video → RAG)
```

### 완전 학습 (6-7주)
```
모든 레슨 순차 학습 (01 → 25)
```

---

## 실습 환경 설정

### 최소 요구사항
```bash
# Python 환경
python >= 3.10

# 핵심 라이브러리
pip install torch>=2.0 transformers>=4.36 accelerate
pip install bitsandbytes peft  # PEFT 학습용
pip install vllm               # Inference 최적화
```

### 추가 라이브러리 (레슨별)
```bash
# Vision Foundation Models
pip install timm segment-anything

# Multimodal
pip install open-clip-torch

# RAG
pip install langchain chromadb sentence-transformers
```

### 권장 GPU 메모리
| 학습 내용 | 최소 VRAM | 권장 VRAM |
|----------|----------|----------|
| Inference (7B 모델, 4bit) | 6GB | 8GB |
| Inference (7B 모델, FP16) | 14GB | 16GB |
| Fine-tuning (LoRA) | 8GB | 16GB |
| SAM 실행 | 8GB | 12GB |

---

## 참고 자료

### 핵심 논문
- **Scaling Laws**: Kaplan et al. (2020), Hoffmann et al. (2022, Chinchilla)
- **LLaMA**: Touvron et al. (2023)
- **Mistral/Mixtral**: Jiang et al. (2023, 2024)
- **DINOv2**: Oquab et al. (2023)
- **SAM**: Kirillov et al. (2023)
- **LLaVA**: Liu et al. (2023)

### 온라인 자료
- [HuggingFace Model Hub](https://huggingface.co/models)
- [Papers With Code - Foundation Models](https://paperswithcode.com/methods/category/foundation-models)
- [LLaMA GitHub](https://github.com/facebookresearch/llama)
- [Segment Anything GitHub](https://github.com/facebookresearch/segment-anything)

---

## 다음 단계

이 폴더를 완료한 후:
- **Model_Implementations**: 주요 모델 from-scratch 구현으로 깊은 이해
- **MLOps**: 모델 배포 및 운영 파이프라인 구축
- **Reinforcement_Learning**: RLHF 심화 학습
