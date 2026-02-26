# Foundation Models Learning Guide

## Overview

Foundation Models refer to models that are pre-trained on large-scale data and can be adapted to various downstream tasks. This folder covers the **paradigm**, **Scaling Laws**, **state-of-the-art architectures**, and **practical applications** of Foundation Models.

### Prerequisites
- **Deep_Learning folder**: ViT, CLIP, Self-Supervised Learning, Transformer
- **LLM_and_NLP folder**: BERT, GPT, HuggingFace, Fine-tuning, RAG

### Learning Objectives
1. Understand the Foundation Model paradigm and Scaling Laws
2. Learn state-of-the-art model architectures like LLaMA, Mistral, DINOv2, SAM
3. Master efficient adaptation (PEFT) and deployment strategies
4. Understand the working principles of Multimodal Foundation Models

---

## File List

### Section 1: Foundation Model Paradigm (01-03)
| File | Topic | Key Content | Difficulty |
|------|------|----------|--------|
| [01_Foundation_Model_Paradigm.md](01_Foundation_Model_Paradigm.md) | FM Paradigm | Definition, History, In-context Learning, Emergent Capabilities | ⭐⭐ |
| [02_Scaling_Laws.md](02_Scaling_Laws.md) | Scaling Laws | Chinchilla, Compute-optimal, Power Laws | ⭐⭐⭐ |
| [03_Emergent_Abilities.md](03_Emergent_Abilities.md) | Emergent Abilities | CoT Emergence, Phase Transitions, Capability Elicitation | ⭐⭐⭐ |

### Section 2: Pre-training Deep Dive (04-07)
| File | Topic | Key Content | Difficulty |
|------|------|----------|--------|
| [04_Pretraining_Objectives.md](04_Pretraining_Objectives.md) | Objectives | Causal LM, Masked LM, Prefix LM, UL2 | ⭐⭐⭐ |
| [05_Data_Curation.md](05_Data_Curation.md) | Data Curation | The Pile, RedPajama, Deduplication, Quality Filtering | ⭐⭐⭐ |
| [06_Pretraining_Infrastructure.md](06_Pretraining_Infrastructure.md) | Training Infrastructure | FSDP, DeepSpeed ZeRO, Distributed Training | ⭐⭐⭐⭐ |
| [07_Tokenization_Deep_Dive.md](07_Tokenization_Deep_Dive.md) | Tokenization | BPE, Unigram, Multilingual, Tokenizer-free | ⭐⭐⭐ |

### Section 3: State-of-the-art LLM Architectures (08-11)
| File | Topic | Key Content | Difficulty |
|------|------|----------|--------|
| [08_LLaMA_Family.md](08_LLaMA_Family.md) | LLaMA | LLaMA 1/2/3, RoPE, RMSNorm, SwiGLU, GQA | ⭐⭐⭐ |
| [09_Mistral_MoE.md](09_Mistral_MoE.md) | Mistral & MoE | Mixtral, Sparse MoE, Router Design, Efficiency | ⭐⭐⭐⭐ |
| [10_Long_Context_Models.md](10_Long_Context_Models.md) | Long Context | Longformer, Ring Attention, YaRN, PI | ⭐⭐⭐ |
| [11_Small_Language_Models.md](11_Small_Language_Models.md) | Small LMs | Phi, Gemma, Qwen, TinyLlama, Knowledge Distillation | ⭐⭐⭐ |

### Section 4: Vision Foundation Models (12-15)
| File | Topic | Key Content | Difficulty |
|------|------|----------|--------|
| [12_DINOv2_Self_Supervised.md](12_DINOv2_Self_Supervised.md) | DINOv2 | DINO, DINOv2, Self-distillation, Dense Features | ⭐⭐⭐ |
| [13_Segment_Anything.md](13_Segment_Anything.md) | SAM | Promptable Segmentation, Image/Prompt/Mask Encoder | ⭐⭐⭐⭐ |
| [14_Unified_Vision_Models.md](14_Unified_Vision_Models.md) | Unified Vision | Florence, PaLI, Unified-IO | ⭐⭐⭐⭐ |
| [15_Image_Generation_Advanced.md](15_Image_Generation_Advanced.md) | Advanced Image Generation | SDXL, ControlNet, IP-Adapter, LCM | ⭐⭐⭐⭐ |

### Section 5: Multimodal Foundation Models (16-18)
| File | Topic | Key Content | Difficulty |
|------|------|----------|--------|
| [16_Vision_Language_Advanced.md](16_Vision_Language_Advanced.md) | Vision-Language | LLaVA, Qwen-VL, Visual Instruction Tuning | ⭐⭐⭐⭐ |
| [17_GPT4V_Gemini.md](17_GPT4V_Gemini.md) | GPT-4V & Gemini | Multimodal Input, Interleaved, API Usage | ⭐⭐⭐ |
| [18_Audio_Video_Foundation.md](18_Audio_Video_Foundation.md) | Audio/Video | Whisper, AudioLM, MusicGen, VideoLLaMA | ⭐⭐⭐⭐ |

### Section 6: Efficient Adaptation (19-21)
| File | Topic | Key Content | Difficulty |
|------|------|----------|--------|
| [19_PEFT_Unified.md](19_PEFT_Unified.md) | Unified PEFT | LoRA, QLoRA, DoRA, Adapters, IA3 | ⭐⭐⭐ |
| [20_Instruction_Tuning.md](20_Instruction_Tuning.md) | Instruction Tuning | FLAN, Self-Instruct, Evol-Instruct | ⭐⭐⭐ |
| [21_Continued_Pretraining.md](21_Continued_Pretraining.md) | Continued Pre-training | Domain Adaptation, Preventing Catastrophic Forgetting | ⭐⭐⭐⭐ |

### Section 7: Deployment and Production (22-24)
| File | Topic | Key Content | Difficulty |
|------|------|----------|--------|
| [22_Inference_Optimization.md](22_Inference_Optimization.md) | Inference Optimization | vLLM, TGI, Speculative Decoding, PagedAttention | ⭐⭐⭐ |
| [23_Advanced_RAG.md](23_Advanced_RAG.md) | Advanced RAG | Agentic RAG, HyDE, RAPTOR, ColBERT | ⭐⭐⭐⭐ |
| [24_API_and_Evaluation.md](24_API_and_Evaluation.md) | API & Evaluation | OpenAI/Anthropic/Google API, Benchmarks | ⭐⭐⭐ |

### Section 8: Future Directions (25)
| File | Topic | Key Content | Difficulty |
|------|------|----------|--------|
| [25_Research_Frontiers.md](25_Research_Frontiers.md) | Research Frontiers | World Models, o1 Reasoning, Synthetic Data | ⭐⭐⭐⭐ |

---

## Learning Roadmap

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Foundation Models Learning Path                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  [Prerequisites]                                                        │
│  Deep_Learning (ViT, CLIP, Transformer) + LLM_and_NLP (BERT, GPT, RAG)  │
│                              │                                          │
│                              ▼                                          │
│  ┌──────────────────────────────────────────────┐                       │
│  │      Phase 1: Paradigm (Week 1)              │                       │
│  │      01 → 02 → 03                            │                       │
│  │      (FM Definition → Scaling Laws → Emergence)│                     │
│  └──────────────────────────────────────────────┘                       │
│                              │                                          │
│              ┌───────────────┴───────────────┐                          │
│              ▼                               ▼                          │
│  ┌─────────────────────┐        ┌─────────────────────┐                 │
│  │  Path A: LLM Focus  │        │  Path B: Vision Focus│                │
│  │  04-11 (Pre-train   │        │  12-15 (DINOv2,     │                 │
│  │  + LLM Architecture)│        │  SAM, Image Gen)    │                 │
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
│  │      Phase 4: Practical Application (Week 5-6)│                      │
│  │      19 → 20 → 21 → 22 → 23 → 24             │                       │
│  │      (PEFT → Instruction → Deploy → RAG)     │                       │
│  └──────────────────────────────────────────────┘                       │
│                              │                                          │
│                              ▼                                          │
│  ┌──────────────────────────────────────────────┐                       │
│  │      Phase 5: Future Directions (Week 7)     │                       │
│  │      25 (Research Frontiers)                 │                       │
│  └──────────────────────────────────────────────┘                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Relationship with Existing Folders

### Connection with Deep_Learning Folder
| Deep_Learning Lesson | Foundation_Models Extension |
|-------------------|----------------------|
| 19_ViT | 10_Long_Context (ViT-based extensions) |
| 20_CLIP | 16_Vision_Language_Advanced (LLaVA, etc.) |
| 21_Self_Supervised | 12_DINOv2 (Latest SSL) |
| 17_Diffusion | 15_Image_Generation_Advanced (SDXL, ControlNet) |

### Connection with LLM_and_NLP Folder
| LLM_and_NLP Lesson | Foundation_Models Extension |
|-----------------|----------------------|
| 04-05_BERT_GPT | 08-09_LLaMA_Mistral (Latest open-source) |
| 07_Fine_Tuning | 19_PEFT_Unified (LoRA variants unified) |
| 09_RAG | 23_Advanced_RAG (Agentic RAG, etc.) |
| 13_Quantization | 22_Inference_Optimization (vLLM, Speculative) |

---

## Recommended Learning Paths

### Quick Practical Application (2 weeks)
```
01 → 02 → 08 → 09 → 19 → 22
(Paradigm → Scaling → LLaMA → Mistral → PEFT → Inference)
```

### Vision Foundation Focus (2 weeks)
```
01 → 03 → 12 → 13 → 14 → 15
(Paradigm → Emergence → DINOv2 → SAM → Unified → Image Gen)
```

### Multimodal Specialist (3 weeks)
```
01 → 02 → 03 → 12 → 16 → 17 → 18 → 23
(Basics → Vision → VLM → GPT-4V → Audio/Video → RAG)
```

### Complete Learning (6-7 weeks)
```
Sequential learning of all lessons (01 → 25)
```

---

## Environment Setup

### Minimum Requirements
```bash
# Python environment
python >= 3.10

# Core libraries
pip install torch>=2.0 transformers>=4.36 accelerate
pip install bitsandbytes peft  # For PEFT training
pip install vllm               # For inference optimization
```

### Additional Libraries (per lesson)
```bash
# Vision Foundation Models
pip install timm segment-anything

# Multimodal
pip install open-clip-torch

# RAG
pip install langchain chromadb sentence-transformers
```

### Recommended GPU Memory
| Learning Content | Minimum VRAM | Recommended VRAM |
|----------|----------|----------|
| Inference (7B model, 4bit) | 6GB | 8GB |
| Inference (7B model, FP16) | 14GB | 16GB |
| Fine-tuning (LoRA) | 8GB | 16GB |
| SAM execution | 8GB | 12GB |

---

## References

### Key Papers
- **Scaling Laws**: Kaplan et al. (2020), Hoffmann et al. (2022, Chinchilla)
- **LLaMA**: Touvron et al. (2023)
- **Mistral/Mixtral**: Jiang et al. (2023, 2024)
- **DINOv2**: Oquab et al. (2023)
- **SAM**: Kirillov et al. (2023)
- **LLaVA**: Liu et al. (2023)

### Online Resources
- [HuggingFace Model Hub](https://huggingface.co/models)
- [Papers With Code - Foundation Models](https://paperswithcode.com/methods/category/foundation-models)
- [LLaMA GitHub](https://github.com/facebookresearch/llama)
- [Segment Anything GitHub](https://github.com/facebookresearch/segment-anything)

---

## Next Steps

After completing this folder:
- **Model_Implementations**: Implement major models from scratch for deeper understanding
- **MLOps**: Build model deployment and operational pipelines
- **Reinforcement_Learning**: Advanced learning of RLHF
