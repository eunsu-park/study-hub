[Previous: Diffusion Models (DDPM)](./33_Impl_Diffusion.md) | [Next: CLIP (Contrastive Language-Image Pre-training)](./35_Impl_CLIP.md)

---

# 34. CLIP and Multimodal Learning

## Learning Objectives

- Understand CLIP architecture and principles
- Contrastive Learning-based Image-Text matching
- Implement Zero-shot Classification
- Introduction to follow-up models (BLIP, ALIGN, etc.)
- PyTorch implementation and practice

---

## 1. Multimodal Learning Overview

### What is Multimodal?

```
Learning multiple types of data (modalities) together

Vision + Language: CLIP, BLIP, Flamingo
Vision + Audio: AudioCLIP
Text + Audio: CLAP
Vision + Text + Audio: ImageBind
```

### Why Multimodal?

```
1. Rich Representation Learning
   - Text: Abstract, semantic information
   - Image: Visual, spatial information
   - Complementary learning possible

2. Zero-shot Capability
   - New classes can be defined with text
   - Classification without labels

3. Diverse Downstream Tasks
   - Image-Text Retrieval
   - Visual Question Answering
   - Image Captioning
```

---

## 2. CLIP Architecture

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

### Training Objective

```
Given N (image, text) pairs:

Correct pairs (diagonal): maximize similarity
Incorrect pairs (off-diagonal): minimize similarity

Loss function: InfoNCE (Contrastive Loss)
```

---

## 3. CLIP Loss Function

### InfoNCE Loss

```python
import torch
import torch.nn.functional as F

def clip_loss(image_features, text_features, temperature=0.07):
    """CLIP Contrastive Loss (⭐⭐⭐)

    Args:
        image_features: (N, D) normalized image embeddings
        text_features: (N, D) normalized text embeddings
        temperature: temperature parameter (lower = sharper)

    Returns:
        loss: image→text + text→image loss
    """
    # Similarity matrix (N x N)
    logits = (image_features @ text_features.T) / temperature

    # Ground truth: diagonal is correct
    labels = torch.arange(len(logits), device=logits.device)

    # Bidirectional CrossEntropy
    loss_i2t = F.cross_entropy(logits, labels)      # image → text
    loss_t2i = F.cross_entropy(logits.T, labels)    # text → image

    return (loss_i2t + loss_t2i) / 2
```

### Temperature Parameter

```python
# Lower temperature:
# - Sharper distribution
# - More focus on correct pairs
# - Start high, gradually decrease

# CLIP default: 0.07 (learnable parameter)
log_temperature = nn.Parameter(torch.log(torch.tensor(1/0.07)))
temperature = log_temperature.exp()
```

---

## 4. CLIP Model Implementation

### Image Encoder

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

        # CLS Token output
        x = self.ln_post(x[:, 0])

        # Projection
        x = self.projection(x)

        return x
```

### Text Encoder

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
        # text: (B, context_length) - token indices

        x = self.token_embedding(text)  # (B, L, transformer_width)
        x = x + self.positional_embedding

        # Causal Mask
        mask = torch.triu(torch.ones(self.context_length, self.context_length), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf')).to(x.device)

        x = self.transformer(x, mask=mask)
        x = self.ln_final(x)

        # Use output at EOT (End of Text) token position
        # In practice, find EOT position with argmax
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]

        x = self.projection(x)

        return x
```

### Complete CLIP Model

```python
class CLIP(nn.Module):
    """CLIP Model (⭐⭐⭐⭐)"""
    def __init__(self, embed_dim=512):
        super().__init__()

        self.image_encoder = ImageEncoder(embed_dim=embed_dim)
        self.text_encoder = TextEncoder(embed_dim=embed_dim)

        # Learnable temperature parameter
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

        # Compute similarity
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * (image_features @ text_features.T)
        logits_per_text = logits_per_image.T

        return logits_per_image, logits_per_text
```

---

## 5. Zero-shot Classification

### Concept

```
CLIP's core capability: classify classes never seen during training

Method:
1. Describe each class with text ("a photo of a {class}")
2. Compute text embeddings
3. Calculate similarity with image embedding
4. Select most similar class
```

### Implementation

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

    # Compute text embeddings (average over templates per class)
    text_features_list = []
    for class_name in class_names:
        class_texts = [template.format(class_name) for template in templates]
        # Tokenize (use tokenizer in practice)
        # text_tokens = tokenizer(class_texts)
        # text_features = model.encode_text(text_tokens)
        # text_features = text_features.mean(dim=0)  # average templates
        # text_features_list.append(text_features)
        pass

    text_features = torch.stack(text_features_list)
    text_features = F.normalize(text_features, dim=-1)

    # Image embedding
    image_features = model.encode_image(image)

    # Compute similarity
    similarity = (image_features @ text_features.T)

    # Top-1 prediction
    probs = similarity.softmax(dim=-1)
    pred = probs.argmax(dim=-1)

    return pred, probs
```

### Prompt Engineering

```python
# Better prompt templates for results

# For ImageNet
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
    # ... more templates
]

# For CIFAR-10
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

## 6. Using OpenAI CLIP

### Installation and Basic Usage

```python
import torch
import clip
from PIL import Image

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Preprocess and encode image
image = preprocess(Image.open("cat.jpg")).unsqueeze(0).to(device)

# Tokenize text
text = clip.tokenize(["a cat", "a dog", "a bird"]).to(device)

# Inference
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    # Compute similarity
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Similarity:", similarity)
# e.g.: tensor([[0.95, 0.03, 0.02]])
```

### Available Models

```python
# Model list
print(clip.available_models())
# ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64',
#  'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']

# Model characteristics
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
    """Search images by text (⭐⭐⭐)"""
    with torch.no_grad():
        # Image embeddings (can be precomputed)
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Text embedding
        text_tokens = clip.tokenize([text_query]).to(images.device)
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Compute similarity
        similarity = (text_features @ image_features.T).squeeze(0)

        # Top-K retrieval
        values, indices = similarity.topk(top_k)

    return indices, values
```

### Image-to-Text Retrieval

```python
def image_to_text_retrieval(model, image, text_candidates, top_k=5):
    """Search text by image (⭐⭐⭐)"""
    with torch.no_grad():
        # Image embedding
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Text embeddings
        text_tokens = clip.tokenize(text_candidates).to(image.device)
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Compute similarity
        similarity = (image_features @ text_features.T).squeeze(0)

        # Top-K retrieval
        values, indices = similarity.topk(top_k)

    return indices, values
```

---

## 8. BLIP (Bootstrapping Language-Image Pre-training)

### CLIP's Limitations and BLIP's Improvements

```
CLIP Limitations:
1. Noisy web data
2. Cannot do Image Captioning (matching only)
3. Unidirectional text encoder

BLIP Improvements:
1. CapFilt: Data refinement with caption filtering
2. Both generation and understanding
3. Bidirectional + autoregressive text encoder
```

### BLIP Architecture

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
│  │     │ │        │  │ (gen)    │                          │
│  └─────┘ └────────┘  └──────────┘                          │
│  Contrastive  Matching    Captioning                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘

ITC: Image-Text Contrastive (similar to CLIP)
ITM: Image-Text Matching (binary classification)
LM: Language Modeling (caption generation)
```

### Using BLIP

```python
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Image Captioning
image = Image.open("cat.jpg")
inputs = processor(image, return_tensors="pt")
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)
print(f"Caption: {caption}")
# e.g.: "a cat sitting on a couch"

# Conditional Captioning (with prompt)
inputs = processor(image, text="a photo of", return_tensors="pt")
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)
```

---

## 9. Other Multimodal Models

### ALIGN (Google)

```
Features:
- Similar to CLIP but larger scale
- 1.8 billion noisy image-text pairs
- EfficientNet + BERT based

Advantages:
- Robust to noise
- Large-scale data utilization
```

### Flamingo (DeepMind)

```
Features:
- Few-shot Learning capability
- Image/video + text input
- Strong in Visual Question Answering

Architecture:
- Perceiver Resampler to compress visual information
- Inject visual information into language model
```

### LLaVA (Large Language and Vision Assistant)

```
Features:
- Visual instruction tuning
- Conversational vision-language model
- GPT-4 level multimodal understanding

Architecture:
- CLIP image encoder
- Vicuna/LLaMA language model
- Connected with projection layer
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

        # Train only linear classifier
        self.classifier = nn.Linear(512, num_classes)  # CLIP embedding dimension

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
    # Lower learning rate for CLIP parameters
    optimizer = torch.optim.AdamW([
        {'params': model.visual.parameters(), 'lr': lr},
        {'params': model.transformer.parameters(), 'lr': lr},
        {'params': model.logit_scale, 'lr': lr * 10}  # Temperature faster
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

## Summary

### Key Concepts

1. **Contrastive Learning**: Learning similarity of image-text pairs
2. **Zero-shot**: Classify classes never seen during training
3. **Temperature**: Control sharpness of similarity distribution
4. **Prompt Engineering**: Improve performance with text templates
5. **Multimodal Representation**: Retrieval/comparison in common embedding space

### Model Comparison

| Model | Features | Advantages |
|-------|----------|------------|
| CLIP | Contrastive | Zero-shot, retrieval |
| BLIP | Generation+Understanding | Captioning, VQA |
| Flamingo | Few-shot | Conversational, flexible |
| LLaVA | Instruction | Complex query handling |

### Practical Tips

```python
# 1. Diverse prompt templates
templates = ["a photo of {}", "an image of {}", ...]

# 2. Use ensemble
features = average([encode(template.format(class_name)) for template in templates])

# 3. Experiment with temperature
# Low temperature: More confident predictions
# High temperature: Softer distribution

# 4. Use larger models (performance order)
# ViT-L/14@336px > ViT-L/14 > ViT-B/16 > ViT-B/32
```

---

## Exercises

### Exercise 1: Understand the InfoNCE Loss

For a batch of N=4 image-text pairs, manually trace through the `clip_loss` function:
1. Sketch the 4×4 similarity matrix. Which entries are the "positive" pairs (ground-truth matches)?
2. Explain why both `loss_i2t` (image→text) and `loss_t2i` (text→image) are needed. What asymmetry would result from using only one direction?
3. What is the effect of temperature on the loss? If temperature is very small (e.g., 0.01), what happens to the softmax distribution? If very large (e.g., 10.0)?

### Exercise 2: Zero-shot Classification with OpenAI CLIP

Using the `clip` library:
1. Load the `ViT-B/32` model and preprocess a sample image (e.g., a photo of a cat).
2. Classify the image against 10 CIFAR-10 class names using the template `"a photo of a {}"`.
3. Print the top-3 predicted classes and their probabilities.
4. Repeat with 5 different prompt templates (e.g., `"a blurry photo of a {}"`, `"a drawing of a {}"`) and average the text embeddings (ensemble). Does ensemble prompting improve confidence for the correct class?

### Exercise 3: Build an Image Retrieval System

Using a pre-trained CLIP model, implement a simple text-to-image search engine:
1. Download 50 images from 5 different categories (10 per category).
2. Precompute and cache image embeddings for all 50 images using `model.encode_image`.
3. Given a text query like `"a red sports car"`, retrieve the top-5 most similar images using cosine similarity.
4. Visualize the top-5 results and evaluate retrieval precision (how many retrieved images actually match the query category?).

### Exercise 4: Fine-tune CLIP with a Linear Probe

Using CLIP as a frozen feature extractor:
1. Implement `CLIPLinearProbe` with the provided code, freezing all CLIP parameters.
2. Extract CLIP image features for the CIFAR-10 training set and train a linear classifier on top.
3. Evaluate on the CIFAR-10 test set and compare accuracy to: (a) zero-shot CLIP, (b) a CNN trained from scratch.
4. Now unfreeze the CLIP model and fine-tune end-to-end with a small learning rate (`lr=1e-5`). Does full fine-tuning improve over the linear probe? What is the risk of full fine-tuning for downstream tasks?

---

## References

- CLIP: https://arxiv.org/abs/2103.00020
- BLIP: https://arxiv.org/abs/2201.12086
- ALIGN: https://arxiv.org/abs/2102.05918
- OpenAI CLIP: https://github.com/openai/CLIP
