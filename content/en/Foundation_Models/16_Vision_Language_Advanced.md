# 16. Advanced Vision-Language

## Learning Objectives

After completing this lesson, you will be able to:

1. Trace the evolution of Vision-Language Models from CLIP to LLaVA and Qwen-VL, explaining the key architectural innovations at each stage
2. Describe how visual tokens are projected and integrated into large language model (LLM) attention layers using linear projection and cross-attention methods
3. Implement Visual Instruction Tuning to fine-tune a VLM for specific multimodal tasks such as image captioning and visual question answering
4. Compare the connection methods (linear projection, cross-attention, MLP) used by different VLMs and analyze their impact on performance
5. Evaluate VLM capabilities on multi-image and high-resolution scenarios, identifying current limitations and emerging solutions

---

## Overview

Vision-Language Models (VLMs) are models that understand both images and text together. This lesson covers state-of-the-art VLM architectures like LLaVA and Qwen-VL, as well as Visual Instruction Tuning techniques.

---

## 1. VLM Paradigm

### 1.1 Evolution

```
┌──────────────────────────────────────────────────────────────────┐
│                    VLM Evolution                                 │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  2021: CLIP                                                      │
│  - Image-Text contrastive learning                              │
│  - Zero-shot classification                                     │
│                                                                  │
│  2022: Flamingo                                                  │
│  - Inject visual tokens into LLM                                │
│  - Few-shot vision-language learning                            │
│                                                                  │
│  2023: LLaVA                                                     │
│  - Visual Instruction Tuning                                    │
│  - Open-source GPT-4V alternative                               │
│                                                                  │
│  2024: LLaVA-NeXT, Qwen-VL, Phi-3-Vision                        │
│  - High resolution, multi-image, video                          │
│  - Commercial-grade performance                                 │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 Architecture Comparison

| Model | Vision Encoder | LLM | Connection Method |
|------|---------------|-----|----------|
| **LLaVA** | CLIP ViT-L | Vicuna/LLaMA | Linear Projection |
| **Qwen-VL** | ViT-G | Qwen | Cross-Attention |
| **InternVL** | InternViT | InternLM | MLP |
| **Phi-3-Vision** | CLIP ViT | Phi-3 | Linear |
| **GPT-4V** | Unknown | GPT-4 | Unknown |

---

## 2. LLaVA (Large Language and Vision Assistant)

### 2.1 Architecture

```
LLaVA Structure:

Image → CLIP ViT-L/14 → Visual Features (576 tokens)
                ↓
         Linear Projection
                ↓
         Visual Tokens
                ↓
[System] [Visual Tokens] [User Query] → LLaMA/Vicuna → Response

Training Stages:
1. Pre-training: Image-Text alignment (CC3M)
2. Fine-tuning: Visual Instruction Tuning (158K)
```

### 2.2 Implementation

```python
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, LlamaForCausalLM, LlamaTokenizer

class LLaVAModel(nn.Module):
    """LLaVA-style Vision-Language Model"""

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
        Image encoding

        Args:
            images: (B, C, H, W)

        Returns:
            visual_tokens: (B, num_patches, llm_hidden_size)
        """
        # CLIP encoding
        vision_outputs = self.vision_encoder(images)
        image_features = vision_outputs.last_hidden_state  # (B, 257, 1024)

        # Exclude [CLS] token
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
            input_ids: (B, seq_len) text tokens
            attention_mask: (B, seq_len)
            images: (B, C, H, W) images
            image_positions: positions where image tokens should be inserted
            labels: (B, seq_len) for training
        """
        B, seq_len = input_ids.shape

        # Text embeddings
        text_embeds = self.llm.model.embed_tokens(input_ids)

        # Image embeddings
        if images is not None:
            visual_tokens = self.encode_images(images)  # (B, num_patches, hidden)

            # Interleave visual tokens with text
            # Simplified: add images before text
            combined_embeds = torch.cat([visual_tokens, text_embeds], dim=1)

            # Adjust attention mask
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
    """Visual Instruction Tuning Dataset"""

    INSTRUCTION_TEMPLATES = [
        "Describe this image in detail.",
        "What can you see in this image?",
        "Explain what is happening in this picture.",
        "<question>",  # VQA
    ]

    def __init__(self, data_path: str):
        """
        Data format:
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

        # Load image
        from PIL import Image
        image = Image.open(item['image']).convert('RGB')

        # Construct conversation
        conversations = item['conversations']
        human_input = conversations[0]['value']
        assistant_output = conversations[1]['value']

        return {
            'image': image,
            'human': human_input,
            'assistant': assistant_output
        }
```

### 2.3 LLaVA-NeXT Improvements

```python
class LLaVANeXTConfig:
    """
    LLaVA-NeXT Improvements

    1. High-resolution support (AnyRes)
    2. Better Vision Encoder (SigLIP)
    3. Larger LLM (Llama 3, Qwen 2)
    """

    # AnyRes: handle various resolutions
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
        """Select best resolution for image"""
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
    AnyRes image processing

    Split high-resolution image into base resolution tiles
    + downscaled full image
    """
    from PIL import Image
    import torch

    # 1. Resize full image (global context)
    global_image = image.resize((base_resolution, base_resolution))

    # 2. Split into tiles (local details)
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
            # Padding
            padded_tile = Image.new('RGB', (base_resolution, base_resolution))
            padded_tile.paste(tile, (0, 0))
            tiles.append(padded_tile)

    # [global_image] + [tile1, tile2, ...]
    all_images = [global_image] + tiles

    return all_images
```

---

## 3. Qwen-VL

### 3.1 Architecture

```
Qwen-VL Features:

1. Vision Encoder: ViT-bigG (1.9B params)
2. High resolution: 448×448 (variable)
3. Grounding support: bounding box output
4. OCR strength: excellent text recognition

Input format:
<img>image_path</img> User question
<ref>object name</ref><box>(x1,y1),(x2,y2)</box>
```

### 3.2 Usage Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

def use_qwen_vl():
    """Using Qwen-VL"""

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen-VL-Chat",
        trust_remote_code=True,
        torch_dtype=torch.float16
    ).to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen-VL-Chat",
        trust_remote_code=True
    )

    # Basic VQA
    query = tokenizer.from_list_format([
        {'image': 'path/to/image.jpg'},
        {'text': 'What is in this image?'},
    ])

    response, history = model.chat(tokenizer, query=query, history=None)
    print(response)

    # Grounding (find object locations)
    query = tokenizer.from_list_format([
        {'image': 'path/to/image.jpg'},
        {'text': 'Find all the cats in this image and output their bounding boxes.'},
    ])

    response, history = model.chat(tokenizer, query=query, history=None)
    # Output: <ref>cat</ref><box>(100,200),(300,400)</box>

    # Multiple images
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

### 4.1 Data Generation

```python
class VisualInstructionGenerator:
    """Visual Instruction data generator"""

    def __init__(self, teacher_model="gpt-4-vision-preview"):
        from openai import OpenAI
        self.client = OpenAI()
        self.teacher_model = teacher_model

    def generate_conversation(
        self,
        image_path: str,
        task_type: str = "detailed_description"
    ):
        """Generate training data with GPT-4V"""
        import base64

        # Encode image
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
        """Generate large-scale dataset"""
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

### 4.2 Training Strategy

```python
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

def finetune_vlm():
    """VLM Fine-tuning"""

    # Load model
    model = LLaVAModel(
        freeze_vision=True,  # Freeze vision encoder
        freeze_llm=False     # Fine-tune LLM
    )

    # Apply LoRA (efficient training)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
    )

    model.llm = get_peft_model(model.llm, lora_config)

    # Training setup
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
    """VLM data collator"""
    batch = {
        'input_ids': torch.stack([f['input_ids'] for f in features]),
        'attention_mask': torch.stack([f['attention_mask'] for f in features]),
        'images': torch.stack([f['image'] for f in features]),
        'labels': torch.stack([f['labels'] for f in features]),
    }
    return batch
```

---

## 5. Evaluation Benchmarks

### 5.1 Major Benchmarks

```
VLM Evaluation Benchmarks:

1. VQA-v2: General Visual QA
2. GQA: Structural reasoning QA
3. TextVQA: Understanding text in images
4. POPE: Hallucination evaluation
5. MME: 14 subtask suite
6. MMBench: 20 capability evaluation
7. SEED-Bench: 19K multiple choice problems
```

### 5.2 Evaluation Code

```python
def evaluate_vlm(model, dataset_name: str = "vqav2"):
    """VLM evaluation"""

    if dataset_name == "vqav2":
        return evaluate_vqa_v2(model)
    elif dataset_name == "textvqa":
        return evaluate_textvqa(model)
    elif dataset_name == "pope":
        return evaluate_pope(model)


def evaluate_pope(model):
    """
    POPE: Polling-based Object Probing Evaluation

    Hallucination evaluation: "Is there a [object] in the image?"
    """
    from datasets import load_dataset

    dataset = load_dataset("lmms-lab/POPE")

    correct = 0
    total = 0

    for item in dataset['test']:
        image = item['image']
        question = item['question']  # "Is there a dog in the image?"
        answer = item['answer']      # "yes" or "no"

        # Model prediction
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

## 6. Practical Applications

### 6.1 Document Understanding

```python
def document_understanding():
    """Document understanding application"""

    model = load_qwen_vl()  # OCR strength

    # PDF page analysis
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

    # Example questions
    questions = [
        "What is the title of this document?",
        "Summarize the main points.",
        "Extract all dates mentioned.",
        "What tables are present? Describe their contents.",
    ]

    results = analyze_document_page("document_page.png", questions)


def chart_understanding():
    """Chart/graph understanding"""

    prompts = [
        "What type of chart is this?",
        "What is the trend shown in this chart?",
        "What are the maximum and minimum values?",
        "Describe the relationship between X and Y.",
    ]

    # Chart analysis with VLM
    for prompt in prompts:
        response = model.generate(chart_image, prompt)
        print(f"Q: {prompt}")
        print(f"A: {response}\n")
```

---

## References

### Papers
- Liu et al. (2023). "Visual Instruction Tuning" (LLaVA)
- Liu et al. (2024). "LLaVA-NeXT: Improved reasoning, OCR, and world knowledge"
- Bai et al. (2023). "Qwen-VL: A Versatile Vision-Language Model"

### Models
- [LLaVA](https://llava-vl.github.io/)
- [Qwen-VL](https://huggingface.co/Qwen/Qwen-VL-Chat)
- [InternVL](https://github.com/OpenGVLab/InternVL)

### Related Lessons
- [../Deep_Learning/20_CLIP_Multimodal.md](../Deep_Learning/20_CLIP_Multimodal.md)
- [12_DINOv2_Self_Supervised.md](12_DINOv2_Self_Supervised.md)

---

## Exercises

### Exercise 1: VLM Connection Methods Comparison
The table in the lesson shows three different methods for connecting a vision encoder to an LLM: Linear Projection (LLaVA), Cross-Attention (Qwen-VL), and MLP. Analyze the trade-offs of each method and explain when each is most appropriate.

| Connection Method | Trainable Params | Latency Impact | Best For |
|------------------|-----------------|----------------|----------|
| Linear Projection | ??? | ??? | ??? |
| MLP (2-3 layers) | ??? | ??? | ??? |
| Cross-Attention | ??? | ??? | ??? |

<details>
<summary>Show Answer</summary>

| Connection Method | Trainable Params | Latency Impact | Best For |
|------------------|-----------------|----------------|----------|
| Linear Projection | Minimal (vision_dim × llm_dim) | Minimal — single matrix multiply | Fast training/inference; resource-constrained settings; when visual features are already high quality (CLIP-L) |
| MLP (2-3 layers) | Moderate | Low — sequential small layers | Better modality alignment than linear; most common balanced choice (LLaVA-1.5 uses 2-layer MLP) |
| Cross-Attention | Large (full cross-attention blocks) | Higher — attention over all visual tokens at each LLM layer | When visual-language interaction needs to be deep and contextual; high-resolution images where spatial detail matters |

**Detailed analysis**:

**Linear Projection** (LLaVA original): Maps visual features directly to LLM embedding space with a single weight matrix. Pros: extremely fast to train (only the projection layer is trained in stage 1), near-zero inference overhead. Cons: limited capacity for modality alignment — it cannot learn complex non-linear mappings between vision and language spaces.

**MLP** (LLaVA-1.5): Adds 1-2 hidden layers with activation functions. The extra capacity significantly improves visual understanding benchmarks (e.g., LLaVA-1.5 with 2-layer MLP greatly outperforms LLaVA-1.0 with linear projection). Good balance of capacity and efficiency.

**Cross-Attention** (Flamingo, Qwen-VL): Dedicated cross-attention layers in the LLM let every text token attend to visual tokens at each layer. Enables richer visual-language grounding. Best for tasks requiring detailed visual reading (OCR, grounding, counting) or when processing high-resolution image tiles. Significant computational cost increase.

</details>

### Exercise 2: Visual Instruction Tuning Data Quality
LLaVA generates training data using GPT-4V as a "teacher" model. Analyze the quality and coverage of this approach by identifying: (A) what types of visual understanding it likely captures well, (B) what it likely misses or distorts, and (C) one concrete improvement to the data generation pipeline.

<details>
<summary>Show Answer</summary>

**A) What GPT-4V-generated data captures well**:
- **Descriptive and narrative capabilities**: Image captioning, detailed scene descriptions — GPT-4V excels at these.
- **Common visual reasoning**: "The person on the left appears to be laughing because..." — general commonsense reasoning about images.
- **Multi-turn conversation format**: GPT-4V can generate realistic back-and-forth dialogues about images.
- **High-resource domains**: Everyday objects, indoor/outdoor scenes, common activities — well-represented in GPT-4V's training data.

**B) What it misses or distorts**:
- **Domain-specific accuracy**: Medical imaging, specialized technical diagrams, scientific data visualization — GPT-4V may generate plausible-sounding but inaccurate descriptions.
- **Precise spatial reasoning**: Exact object counts, precise measurements, fine-grained spatial relationships ("the second object from the left on the third shelf") — GPT-4V makes errors the student model will learn to replicate.
- **Calibrated uncertainty**: GPT-4V tends to be confident even when uncertain, training the student to hallucinate rather than express uncertainty.
- **Dataset biases**: Generates descriptions reflecting GPT-4V's biases (cultural, demographic, linguistic) which get amplified in the student.

**C) Concrete improvement**:
Use **verified ground-truth data sources** for factual tasks rather than GPT-4V generation. Example: for object counting, use COCO annotations (verified by multiple human annotators) to generate QA pairs like "How many cats are in this image? 3." This ensures the student learns accurate counting rather than GPT-4V's approximate answers. A hybrid pipeline: GPT-4V for descriptive/conversational data, ground-truth annotations for quantitative/factual tasks.

</details>

### Exercise 3: AnyRes High-Resolution Processing
LLaVA-NeXT uses "AnyRes" to handle high-resolution images. Given an input image of 1680×672 pixels and a base resolution of 336×336, trace through the AnyRes processing pipeline:

```python
def anyres_processing(image, base_resolution=336):
    """
    AnyRes processing:
    1. Resize full image to 336×336 (global context)
    2. Split into 336×336 tiles (local details)
    3. Return: [global_image] + [tile1, tile2, ..., tileN]
    """
    W, H = image.size  # (1680, 672)
    # How many tiles? What is the total token count?
    pass

# Given:
# - Each 336×336 image produces 576 visual tokens (24×24 patches with 14px patches)
# - The full image is resized to 336×336 → 576 tokens
# - Each tile is 336×336 → 576 tokens
# Calculate: total visual tokens for a 1680×672 image
```

<details>
<summary>Show Answer</summary>

```python
# Step 1: Calculate number of tiles
W, H = 1680, 672
base_resolution = 336

num_tiles_w = (W + base_resolution - 1) // base_resolution
            = (1680 + 336 - 1) // 336
            = 2015 // 336 = 5 tiles horizontally

num_tiles_h = (H + base_resolution - 1) // base_resolution
            = (672 + 336 - 1) // 336
            = 1007 // 336 = 2 tiles vertically (ceiling division)

# Actually: 672 / 336 = 2.0 exactly → 2 tiles vertically
# 1680 / 336 = 5.0 exactly → 5 tiles horizontally
total_tiles = 5 × 2 = 10 tiles

# Step 2: Calculate total visual tokens
tokens_per_image = 576  # 24×24 patches per 336×336 image

global_tokens = 576          # 1 resized full image
tile_tokens = 10 × 576       # 10 local detail tiles
              = 5760

total_visual_tokens = 576 + 5760 = 6336 tokens

# For comparison:
# - LLaVA 1.0 (single 224×224): 256 tokens
# - LLaVA-NeXT for this image: 6336 tokens (24.75x more)

# Memory implications:
# With a 4096-dim LLM, attention over 6336 visual + ~100 text tokens
# Attention matrix: (6436)^2 = ~41M entries per layer
# vs. LLaVA 1.0: (356)^2 = ~127K entries per layer
# → ~330x more attention computation per layer
```

This explains why high-resolution VLM inference is expensive and why techniques like token compression (e.g., visual token merging) are an active research area.

</details>

### Exercise 4: POPE Hallucination Evaluation
The POPE benchmark evaluates VLM hallucination by asking binary yes/no questions about objects ("Is there a dog in this image?"). Design a scenario where a VLM with high POPE accuracy could still exhibit harmful hallucination in practice, and propose a more comprehensive evaluation.

<details>
<summary>Show Answer</summary>

**Scenario where high POPE accuracy fails to detect harmful hallucination**:

POPE only tests object presence/absence for common COCO objects using simple binary questions. Consider:

**Scenario: Medical document analysis**
- Image: A radiology report showing a chest X-ray with a small nodule in the upper-right lung.
- POPE-style question: "Is there a nodule in this image?" — Model correctly answers "Yes." (POPE says: no hallucination)
- Harmful hallucination not caught by POPE: The model describes "The nodule is located in the lower-left lung, measures approximately 2cm, and shows irregular edges consistent with malignancy." — location is wrong, size is fabricated, and the malignancy assessment is hallucinated.
- POPE would score this model as excellent; in practice, it could lead to misdiagnosis.

**Other failure modes POPE misses**:
- Attribute hallucination: "A red fire truck" when the truck is blue.
- Relational hallucination: "The cat is on the table" when the cat is under the table.
- Counting hallucination: Saying there are 3 people when there are 5.
- Fabricated text: Making up text on signs/labels not present or misreading it.

**More comprehensive evaluation**:

1. **Attribute hallucination benchmark**: Questions like "What color is the car?" — test whether described attributes match ground truth.

2. **Spatial relation benchmark**: "Is the cup to the left or right of the plate?" — test spatial understanding accuracy.

3. **Counting benchmark**: "How many people are in this image?" — evaluate count accuracy, not just presence.

4. **Fine-grained OCR verification**: For images with text, compare model's transcription to verified ground truth.

5. **Confidence calibration test**: Measure whether the model's expressed uncertainty (e.g., "I think there might be...") correlates with its actual accuracy — a well-calibrated model should be uncertain when it's wrong.

</details>
