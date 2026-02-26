# Foundation Model Paradigm

## Learning Objectives
- Understand the definition and characteristics of Foundation Models
- Grasp the paradigm shift from traditional ML to Foundation Models
- Learn the concepts of In-context Learning and Emergent Capabilities
- Identify the major Foundation Model lineage

---

## 1. What are Foundation Models?

### 1.1 Definition

**Foundation Model** is a term proposed by Stanford HAI in 2021, referring to models with the following characteristics:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Foundation Model Definition                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Pre-trained on broad data                                   │
│     - Billions to trillions of text tokens                      │
│     - Hundreds of millions to billions of images               │
│                                                                 │
│  2. Adaptable to many tasks                                     │
│     - Single model performs classification, generation, QA,     │
│       translation, etc.                                         │
│     - Adapted through fine-tuning or prompting                  │
│                                                                 │
│  3. General-purpose representations                             │
│     - Task-agnostic knowledge encoding                          │
│     - Maximizes transfer learning                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Traditional ML vs Foundation Model

```
┌─────────────────────────────────────────────────────────────────┐
│            Traditional Machine Learning Pipeline                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Task A ───► Data A ───► Model A ───► Deploy A                  │
│  Task B ───► Data B ───► Model B ───► Deploy B                  │
│  Task C ───► Data C ───► Model C ───► Deploy C                  │
│                                                                 │
│  • Separate data collection for each task                       │
│  • Separate model training for each task                        │
│  • Limited knowledge sharing between tasks                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│              Foundation Model Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                    ┌─────────────────┐                          │
│  Massive Data ───► │ Foundation Model │                         │
│  (Web-scale)       └────────┬────────┘                          │
│                             │                                   │
│              ┌──────────────┼──────────────┐                    │
│              ▼              ▼              ▼                    │
│         Adapt A        Adapt B        Adapt C                   │
│         (Fine-tune)    (Prompt)       (LoRA)                    │
│              │              │              │                    │
│              ▼              ▼              ▼                    │
│         Task A         Task B         Task C                    │
│                                                                 │
│  • Single large-scale pre-training                              │
│  • Lightweight adaptation for various tasks                     │
│  • Maximizes knowledge transfer between tasks                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 Types of Foundation Models

| Category | Representative Models | Input/Output |
|------|----------|----------|
| **Language Models** | GPT-4, LLaMA, Claude | Text → Text |
| **Vision Models** | ViT, DINOv2, SAM | Image → Features/Segmentation |
| **Multimodal** | CLIP, LLaVA, GPT-4V | Text+Image → Text |
| **Generative** | Stable Diffusion, DALL-E | Text → Image |
| **Audio** | Whisper, AudioLM | Audio ↔ Text |
| **Code** | Codex, CodeLlama | Text → Code |

---

## 2. History of the Paradigm Shift

### 2.1 Timeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Foundation Model History                          │
├──────┬──────────────────────────────────────────────────────────────┤
│ 2017 │ Transformer (Vaswani et al.) - Introduced self-attention     │
├──────┼──────────────────────────────────────────────────────────────┤
│ 2018 │ BERT (Google) - Bidirectional context via Masked LM          │
│      │ GPT-1 (OpenAI) - First large-scale autoregressive LM         │
├──────┼──────────────────────────────────────────────────────────────┤
│ 2019 │ GPT-2 (1.5B params) - "Too dangerous to release"             │
│      │ T5 - Text-to-Text Transfer Transformer                       │
├──────┼──────────────────────────────────────────────────────────────┤
│ 2020 │ GPT-3 (175B) - Discovered In-context Learning                │
│      │ Scaling Laws paper (Kaplan et al.)                           │
│      │ ViT - Applied Transformer to Vision                          │
├──────┼──────────────────────────────────────────────────────────────┤
│ 2021 │ CLIP - Connected Vision and Language                         │
│      │ DALL-E - Text-to-Image generation                            │
│      │ "Foundation Models" term coined (Stanford HAI)               │
├──────┼──────────────────────────────────────────────────────────────┤
│ 2022 │ ChatGPT - Popularization of LLMs                             │
│      │ Chinchilla - Compute-optimal Scaling                         │
│      │ Stable Diffusion - Open-source image generation              │
├──────┼──────────────────────────────────────────────────────────────┤
│ 2023 │ GPT-4 - Multimodal Foundation Model                          │
│      │ LLaMA - Open-source LLM revolution                           │
│      │ SAM - Promptable Vision Foundation Model                     │
├──────┼──────────────────────────────────────────────────────────────┤
│ 2024 │ GPT-4o, Claude 3, Gemini 1.5 - Performance competition       │
│      │ LLaMA 3, Mistral - Open-source advancement                   │
│      │ Sora - Video Foundation Model                                │
└──────┴──────────────────────────────────────────────────────────────┘
```

### 2.2 Major Turning Points

#### (1) GPT-3's In-context Learning (2020)

GPT-3 demonstrated the potential of few-shot learning, becoming a catalyst for the paradigm shift:

```python
# Traditional Approach: Fine-tuning required for each task
model = load_pretrained("bert-base")
model = fine_tune(model, sentiment_dataset, epochs=3)
result = model.predict("This movie was great!")

# GPT-3 In-context Learning: Learning only through prompts
prompt = """
Classify the sentiment:
Text: "I love this product!" → Positive
Text: "Terrible experience." → Negative
Text: "This movie was great!" →
"""
result = gpt3.generate(prompt)  # "Positive"
```

#### (2) CLIP's Vision-Language Connection (2021)

CLIP enabled zero-shot classification by mapping images and text to the same space:

```python
# Zero-shot Image Classification with CLIP
import clip

model, preprocess = clip.load("ViT-B/32")

# Embed images and text in the same space
image_features = model.encode_image(preprocess(image))
text_features = model.encode_text(clip.tokenize(["a dog", "a cat", "a bird"]))

# Classify by similarity (without training!)
similarity = (image_features @ text_features.T).softmax(dim=-1)
# [0.95, 0.03, 0.02] → "a dog"
```

#### (3) ChatGPT's RLHF (2022)

ChatGPT generates human-aligned responses using RLHF (Reinforcement Learning from Human Feedback):

```
┌─────────────────────────────────────────────────────────────────┐
│                    ChatGPT Training Process                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 1: Pre-training (GPT-3.5 base)                            │
│          Learn to predict next token from web text              │
│                         │                                       │
│                         ▼                                       │
│  Step 2: Supervised Fine-tuning (SFT)                           │
│          Train on high-quality human-written responses          │
│                         │                                       │
│                         ▼                                       │
│  Step 3: Reward Model Training                                  │
│          Train model to predict preference between response pairs│
│                         │                                       │
│                         ▼                                       │
│  Step 4: RLHF with PPO                                          │
│          Optimize policy using Reward Model as reward           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. In-context Learning (ICL)

### 3.1 Concept

In-context Learning is the ability to perform tasks using only examples in the prompt, without updating model weights.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Types of In-context Learning                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Zero-shot:  "Translate to French: Hello"                       │
│              → "Bonjour"                                        │
│                                                                 │
│  One-shot:   "English: Hello → French: Bonjour                  │
│               English: Goodbye →"                               │
│              → "Au revoir"                                      │
│                                                                 │
│  Few-shot:   "English: Hello → French: Bonjour                  │
│               English: Goodbye → French: Au revoir              │
│               English: Thank you → French: Merci                │
│               English: Good morning →"                          │
│              → "Bonjour" (or "Bon matin")                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Why ICL Works (Hypotheses)

```python
"""
Hypothesis 1: Bayesian Inference
- Infer task distribution from prompt examples
- P(output | input, examples) ∝ P(examples | task) × P(task)

Hypothesis 2: Implicit Gradient Descent
- Transformer's attention implicitly performs gradient steps
- Mechanism similar to meta-learning

Hypothesis 3: Task Vector Retrieval
- Retrieve task vectors learned during pre-training
- Prompt activates appropriate task vectors
"""
```

### 3.3 Few-shot Prompt Example

```python
# Sentiment Analysis Few-shot
sentiment_prompt = """
Analyze the sentiment of the following reviews:

Review: "The food was delicious and the service was excellent!"
Sentiment: Positive

Review: "I waited for an hour and the waiter was rude."
Sentiment: Negative

Review: "It was okay, nothing special but not bad either."
Sentiment: Neutral

Review: "Best experience ever! Will definitely come back!"
Sentiment:"""

# API call
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": sentiment_prompt}]
)
print(response.choices[0].message.content)  # "Positive"
```

---

## 4. Emergent Capabilities

### 4.1 Definition

**Emergent Capabilities** are abilities that are absent in smaller models but suddenly appear beyond a certain scale.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Characteristics of Emergent Abilities         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Performance                                                    │
│       │                                                         │
│   100%├─────────────────────────────────────────○───── Large    │
│       │                                      ╱         models   │
│       │                                    ╱                    │
│       │                              ╱                          │
│    50%├─ · · · · · · · · · · · ·╱· · · · · · · · · · · · · · ·  │
│       │                      ╱    ↑ Phase Transition            │
│       │                    ╱      (Sudden performance jump)     │
│       │──────────────────○─────────────────────────── Small     │
│     0%├───────┬───────┬───────┬───────┬───────┬──────▶  models │
│       │      10B    50B    100B   200B   500B     Parameters   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Representative Emergent Capabilities

| Capability | Description | Emergence Scale (approx.) |
|------|------|-----------------|
| **Arithmetic** | Multi-digit addition/multiplication | ~10B params |
| **Chain-of-Thought** | Step-by-step reasoning | ~60B params |
| **Word Unscrambling** | Restoring scrambled words | ~60B params |
| **Multi-step Math** | Complex math problems | ~100B params |
| **Code Generation** | Complex code writing | ~100B params |

### 4.3 Chain-of-Thought (CoT) Prompting

```python
# Without CoT - often fails
prompt_direct = """
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
   Each can has 3 tennis balls. How many tennis balls does he have now?
A:"""
# GPT-3 (small): "8" (incorrect)

# With CoT - improved accuracy through step-by-step reasoning
prompt_cot = """
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
   Each can has 3 tennis balls. How many tennis balls does he have now?
A: Let's think step by step.
   Roger started with 5 tennis balls.
   He bought 2 cans, each with 3 balls, so 2 × 3 = 6 balls.
   Total: 5 + 6 = 11 tennis balls.
   The answer is 11.

Q: The cafeteria had 23 apples. If they used 20 to make lunch and
   bought 6 more, how many apples do they have?
A: Let's think step by step."""
# GPT-3: "They started with 23, used 20, so 23-20=3.
#         Then bought 6 more: 3+6=9. The answer is 9." (correct)
```

---

## 5. Core Components of Foundation Models

### 5.1 Architecture Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│                    Major Architecture Patterns                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Encoder-only (BERT, DINOv2)                                    │
│  ┌─────────────────────────────────────────┐                    │
│  │ [CLS] Token1 Token2 ... TokenN [SEP]    │                    │
│  │       ↓      ↓      ↓    ↓              │                    │
│  │    ┌──┴──────┴──────┴────┴──┐           │                    │
│  │    │   Bidirectional Attn   │           │                    │
│  │    └──┬──────┬──────┬────┬──┘           │                    │
│  │       ↓      ↓      ↓    ↓              │                    │
│  │    Pooled / Token Representations       │                    │
│  └─────────────────────────────────────────┘                    │
│  • Utilizes bidirectional context                               │
│  • Suitable for classification, embeddings                      │
│                                                                 │
│  Decoder-only (GPT, LLaMA)                                      │
│  ┌─────────────────────────────────────────┐                    │
│  │ Token1 → Token2 → Token3 → ...          │                    │
│  │   ↓        ↓        ↓                   │                    │
│  │ ┌─┴────────┴────────┴─┐                 │                    │
│  │ │  Causal (Masked) Attn│                │                    │
│  │ └─┬────────┬────────┬─┘                 │                    │
│  │   ↓        ↓        ↓                   │                    │
│  │ Next     Next     Next                  │                    │
│  │ Token    Token    Token                 │                    │
│  └─────────────────────────────────────────┘                    │
│  • Autoregressive generation                                    │
│  • Optimized for text generation                                │
│                                                                 │
│  Encoder-Decoder (T5, BART)                                     │
│  ┌─────────────────────────────────────────┐                    │
│  │ ┌──────────┐    ┌──────────┐            │                    │
│  │ │ Encoder  │───▶│ Decoder  │            │                    │
│  │ │(Bi-dir)  │    │(Causal)  │            │                    │
│  │ └──────────┘    └──────────┘            │                    │
│  └─────────────────────────────────────────┘                    │
│  • Separate input understanding and output generation           │
│  • Suitable for translation, summarization                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Key Components

```python
"""
Core Components of Foundation Models:

1. Self-Attention
   - Query, Key, Value operations
   - Learn relationships between all positions

2. Feed-Forward Network (FFN)
   - Acts as knowledge storage
   - Accounts for most parameters

3. Positional Encoding
   - Inject sequence information
   - Sinusoidal, Learnable, RoPE, etc.

4. Normalization
   - LayerNorm (BERT, GPT)
   - RMSNorm (LLaMA) - more efficient

5. Activation Function
   - GELU (BERT, GPT)
   - SwiGLU (LLaMA) - better performance
"""
```

---

## 6. Using Foundation Models

### 6.1 Getting Started with HuggingFace

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model (e.g., LLaMA-2-7B)
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Save memory
    device_map="auto"           # Automatic GPU allocation
)

# Generate text
prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,
    do_sample=True,
    top_p=0.9
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### 6.2 Using Vision Foundation Model

```python
# DINOv2 - Universal image feature extraction
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base")

# Extract image embeddings
image = Image.open("image.jpg")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    features = outputs.last_hidden_state  # (1, num_patches+1, 768)
    cls_embedding = features[:, 0]        # CLS token (whole image representation)

# Use this embedding for classification, retrieval, segmentation, etc.
```

### 6.3 Using API (OpenAI)

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain foundation models in simple terms."}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)
```

---

## 7. Limitations and Challenges of Foundation Models

### 7.1 Current Limitations

| Limitation | Description | Solution Attempts |
|------|------|----------|
| **Hallucination** | Generate false information | RAG, Grounding |
| **Outdated Knowledge** | Unaware of post-training information | RAG, Fine-tuning |
| **Reasoning Limits** | Difficulty with complex logical reasoning | CoT, Self-consistency |
| **High Compute Cost** | Enormous training/inference costs | Quantization, Distillation |
| **Safety/Alignment** | Can generate harmful content | RLHF, Constitutional AI |

### 7.2 Research Directions

```
┌─────────────────────────────────────────────────────────────────┐
│                    Future Research Directions                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Efficient Models                                            │
│     └─ Mixture of Experts, Sparse Attention, Quantization       │
│                                                                 │
│  2. Multimodal Integration                                      │
│     └─ Unified Vision + Language + Audio + Code                 │
│                                                                 │
│  3. Reasoning Enhancement                                       │
│     └─ Test-time Compute (o1), Tree of Thoughts                 │
│                                                                 │
│  4. Continual Learning                                          │
│     └─ Continuous learning, Solving Catastrophic Forgetting     │
│                                                                 │
│  5. Safety & Alignment                                          │
│     └─ Constitutional AI, Red-teaming, Interpretability         │
│                                                                 │
│  6. Agentic Systems                                             │
│     └─ Tool Use, Multi-Agent, Autonomous Planning               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Summary

### Key Concepts
- **Foundation Model**: General-purpose models trained on large-scale data and applicable to various tasks
- **Paradigm Shift**: Task-specific → Pre-train & Adapt
- **In-context Learning**: Learning through prompts without weight updates
- **Emergent Capabilities**: Abilities that suddenly appear at scale

### Next Steps
- [02_Scaling_Laws.md](02_Scaling_Laws.md): Relationship between model size and performance
- [03_Emergent_Abilities.md](03_Emergent_Abilities.md): In-depth analysis of emergent abilities

---

## References

### Key Papers
- Bommasani et al. (2021). "On the Opportunities and Risks of Foundation Models"
- Brown et al. (2020). "Language Models are Few-Shot Learners" (GPT-3)
- Radford et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision" (CLIP)
- Wei et al. (2022). "Emergent Abilities of Large Language Models"

### Additional Resources
- [Stanford HAI Foundation Models Report](https://crfm.stanford.edu/report.html)
- [HuggingFace Model Hub](https://huggingface.co/models)
- [Papers With Code - Foundation Models](https://paperswithcode.com/methods/category/foundation-models)

---

## Exercises

### Exercise 1: Foundation Model Classification

For each of the following models, identify (a) its type (Language, Vision, Multimodal, Generative, Audio, Code) and (b) whether it uses an encoder-only, decoder-only, or encoder-decoder architecture. Explain your reasoning.

1. GPT-4
2. BERT
3. T5
4. DALL-E 3
5. Whisper

<details>
<summary>Show Answer</summary>

1. **GPT-4** — Language Model (and Multimodal in its vision variant), **decoder-only**. It generates text autoregressively using a causal mask, attending only to past tokens.
2. **BERT** — Language Model (embedding/understanding), **encoder-only**. It uses bidirectional self-attention (masked LM) to produce context-rich token representations.
3. **T5** — Language Model (text-to-text), **encoder-decoder**. The encoder processes the input with bidirectional attention; the decoder generates the output autoregressively.
4. **DALL-E 3** — Generative Model (Text → Image), architecture varies by version (VQVAE + transformer / diffusion). It maps text descriptions into image representations.
5. **Whisper** — Audio Model (Audio ↔ Text), **encoder-decoder**. An audio encoder processes mel-spectrogram features; a decoder generates transcription tokens.

</details>

---

### Exercise 2: In-context Learning Prompt Design

Write a few-shot prompt for a **named entity recognition (NER)** task that identifies PERSON and LOCATION entities. Include three examples in the prompt, then add an unlabeled test sentence.

<details>
<summary>Show Answer</summary>

```python
ner_prompt = """
Extract PERSON and LOCATION entities from the following sentences.

Sentence: "Albert Einstein was born in Ulm, Germany."
Entities: PERSON: Albert Einstein | LOCATION: Ulm, Germany

Sentence: "Marie Curie conducted research in Paris."
Entities: PERSON: Marie Curie | LOCATION: Paris

Sentence: "Elon Musk founded SpaceX in Hawthorne, California."
Entities: PERSON: Elon Musk | LOCATION: Hawthorne, California

Sentence: "Alan Turing worked at Bletchley Park during World War II."
Entities:"""

# Expected output:
# PERSON: Alan Turing | LOCATION: Bletchley Park
```

Key design principles:
- Consistent format across all examples (makes pattern extraction easier)
- Same entity types demonstrated in each example
- The final item ends at the same structural point to guide completion

</details>

---

### Exercise 3: Paradigm Comparison Analysis

Compare the **Traditional ML Pipeline** and the **Foundation Model Pipeline** across the following dimensions. Provide a concrete example for each dimension.

| Dimension | Traditional ML | Foundation Model |
|-----------|---------------|-----------------|
| Data per task | ? | ? |
| Training per task | ? | ? |
| Knowledge transfer | ? | ? |
| Adaptation method | ? | ? |
| Failure mode | ? | ? |

<details>
<summary>Show Answer</summary>

| Dimension | Traditional ML | Foundation Model |
|-----------|---------------|-----------------|
| Data per task | Each task needs its own labeled dataset (e.g., 10K labeled images for each classifier) | One massive pre-training corpus shared across all tasks (e.g., 1T tokens) |
| Training per task | Full model trained from scratch or fine-tuned independently for each task | Single large pre-training run; lightweight adaptation per task |
| Knowledge transfer | Limited — a sentiment classifier doesn't help a translation model | Strong — linguistic, factual, and reasoning knowledge transfers across tasks |
| Adaptation method | Retrain / fine-tune with task-specific labeled data | Prompt engineering, few-shot examples, or parameter-efficient fine-tuning (LoRA) |
| Failure mode | Data drift — new tasks require expensive data collection | Hallucination — model may generate plausible but incorrect outputs |

</details>

---

### Exercise 4: Emergent Capabilities Threshold Analysis

The following capabilities have been observed to emerge at different scale thresholds. Order them from **earliest to latest emergence** (in terms of training compute) and briefly explain why each requires more scale than the previous one.

- Chain-of-Thought reasoning
- Simple 3-digit addition
- Theory of Mind
- Word unscrambling

<details>
<summary>Show Answer</summary>

**Emergence order (earliest → latest):**

1. **Simple 3-digit addition** (~10^22 FLOPs) — Requires learning basic digit patterns and carry operations. Relatively few sub-skills to compose.

2. **Word unscrambling** (~10^22 FLOPs) — Requires letter pattern recognition and vocabulary knowledge, but is a self-contained lexical task.

3. **Chain-of-Thought reasoning** (~10^23 FLOPs) — Requires not just factual knowledge, but the ability to decompose multi-step problems and maintain intermediate state in generated text. This demands a much richer internal representation.

4. **Theory of Mind** (~10^24 FLOPs) — Requires modeling other agents' beliefs and intentions, which involves nested reasoning ("Alice believes that Bob thinks..."). This is the most compositionally complex of the four.

The pattern: later-emerging capabilities require composing more sub-skills and maintaining more abstract internal state, both of which benefit from the additional representational capacity that comes with scale.

</details>

---

### Exercise 5: RLHF Pipeline Design

The ChatGPT training process involves four stages. For each stage below, identify what data is required and what is learned.

1. Pre-training (GPT-3.5 base)
2. Supervised Fine-tuning (SFT)
3. Reward Model Training
4. RLHF with PPO

<details>
<summary>Show Answer</summary>

| Stage | Data Required | What is Learned |
|-------|--------------|-----------------|
| **1. Pre-training** | Massive web-scale text corpus (hundreds of billions of tokens) | Next-token prediction; broad linguistic, factual, and reasoning knowledge |
| **2. SFT** | Human-written high-quality instruction-response pairs (thousands to tens of thousands of examples) | To follow instructions and produce well-structured, helpful responses |
| **3. Reward Model Training** | Pairs of model responses ranked by human annotators (one response preferred over another) | To predict human preference scores — assigns a scalar reward to any model output |
| **4. RLHF with PPO** | The reward model (as an environment) + the SFT model (as starting policy) | To generate responses that maximize reward (human preference) while not drifting too far from the SFT distribution (via KL penalty) |

The key insight: each stage refines a different aspect — (1) raw knowledge, (2) instruction format, (3) preference model, (4) alignment optimization.

</details>
