[Previous: CLIP and Multimodal Learning](./34_CLIP_Multimodal.md) | [Next: Self-Supervised Learning](./36_Self_Supervised_Learning.md)

---

# 35. CLIP (Contrastive Language-Image Pre-training)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the CLIP training objective — contrastive learning over image-text pairs — and describe how the InfoNCE loss aligns vision and language representations.
2. Describe the dual-encoder architecture of CLIP (vision encoder + text encoder) and explain how the shared embedding space enables cross-modal similarity computation.
3. Implement the CLIP contrastive training loop from scratch in PyTorch, including the symmetric cross-entropy loss over the similarity matrix.
4. Perform zero-shot image classification using a pretrained CLIP model by constructing text prompts and computing image-text similarities.
5. Apply CLIP embeddings for downstream tasks such as image retrieval, semantic image search, and as frozen features for few-shot classification.
6. Analyze the role of temperature scaling in the InfoNCE loss and explain how data scale and prompt engineering affect CLIP's zero-shot performance.

---

## Overview

CLIP maps images and text to the same embedding space, enabling zero-shot image classification. "Learning Transferable Visual Models From Natural Language Supervision" (Radford et al., 2021)

---

## Mathematical Background

### 1. Contrastive Learning

```
Goal: learn similarity of image-text pairs

N (image, text) pairs in a batch:
- Diagonal (i, i): matching pairs (positive)
- Off-diagonal (i, j): non-matching pairs (negative)

Similarity matrix (N × N):
S[i, j] = <image_i, text_j> / τ

where τ is temperature parameter
```

### 2. InfoNCE Loss

```
Image-to-Text Loss:
L_i2t = -1/N Σᵢ log(exp(S[i,i]) / Σⱼ exp(S[i,j]))

Text-to-Image Loss:
L_t2i = -1/N Σᵢ log(exp(S[i,i]) / Σⱼ exp(S[j,i]))

Total Loss:
L = (L_i2t + L_t2i) / 2

Intuition:
- Numerator: similarity of matching pairs ↑
- Denominator: similarity with other pairs ↓
```

### 3. Zero-shot Classification

```
Classify new image:

1. Generate text prompts per class:
   "A photo of a {class_name}"

2. Compute text embeddings:
   T = [text_enc("A photo of a cat"),
        text_enc("A photo of a dog"),
        ...]

3. Compute image embedding:
   I = image_enc(image)

4. Classify by similarity:
   probs = softmax(I @ T.T / τ)
   prediction = argmax(probs)

Can classify new classes without training!
```

---

## CLIP Architecture

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

Model variants:
- CLIP ViT-B/32: 512 dim, 86M image + 63M text params
- CLIP ViT-B/16: 512 dim, 86M image + 63M text params
- CLIP ViT-L/14: 768 dim, 304M image + 123M text params
- CLIP RN50: ResNet-50 image encoder
```

---

## File Structure

```
12_CLIP/
├── README.md
├── numpy/
│   └── clip_forward.py       # NumPy forward pass
├── pytorch_lowlevel/
│   └── clip_lowlevel.py      # PyTorch Low-Level CLIP
├── paper/
│   └── clip_paper.py         # Paper reproduction
└── exercises/
    ├── 01_zero_shot.md       # Zero-shot classification
    └── 02_retrieval.md       # Image-text retrieval
```

---

## Core Concepts

### 1. Large-scale Dataset

```
WebImageText (WIT) dataset:
- 400 million (image, text) pairs
- Collected from internet
- Natural language supervision

Data collection:
1. Collect image and alt-text pairs
2. Filtering (quality, deduplication)
3. Class balancing
```

### 2. Prompt Engineering

```
Simple prompt:
"cat"  →  "A photo of a cat"

Prompt ensemble:
templates = [
    "A photo of a {}",
    "A picture of a {}",
    "An image showing a {}",
    "A {} in the scene"
]

# Average of multiple templates
text_embeddings = []
for template in templates:
    prompt = template.format(class_name)
    embedding = text_encoder(prompt)
    text_embeddings.append(embedding)
final_embedding = mean(text_embeddings)
```

### 3. Applications

```
1. Zero-shot Classification
   - Directly apply to new domains
   - Define classes with prompts

2. Image-Text Retrieval
   - Search images with text
   - Search text with images

3. Image Generation Guidance
   - Guidance for DALL-E, Stable Diffusion
   - Measure generation quality with CLIP score

4. Multimodal Embedding
   - Common representation for images and text
   - Foundation for downstream tasks
```

---

## Implementation Levels

### Level 2: PyTorch Low-Level (pytorch_lowlevel/)
- Direct image encoder (ViT) implementation
- Direct text encoder (Transformer) implementation
- Implement contrastive loss

### Level 3: Paper Implementation (paper/)
- Complete training pipeline
- Zero-shot evaluation
- Prompt engineering

### Level 4: Code Analysis (separate)
- Analyze OpenAI CLIP code
- Analyze open_clip library

---

## Learning Checklist

- [ ] Understand contrastive learning
- [ ] Understand InfoNCE loss formula
- [ ] Implement zero-shot classification
- [ ] Understand role of temperature
- [ ] Practice prompt engineering
- [ ] Implement image-text retrieval

---

## References

- Radford et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision"
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [34_CLIP_Multimodal.md](./34_CLIP_Multimodal.md)
