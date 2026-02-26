# 04. Pre-training Objectives

## Learning Objectives

After completing this lesson, you will be able to:

1. Compare the three main pre-training paradigms (Causal LM, Masked LM, Prefix LM) and explain how each shapes a model's context access, training signal density, and suitability for generation vs. understanding tasks.
2. Derive the mathematical objective functions for Causal Language Modeling (CLM) and Masked Language Modeling (MLM), and explain the role of the causal mask and masking rate.
3. Explain Span Corruption (T5-style) and other advanced objectives such as CLM with prefix, and identify the representative models that use each objective.
4. Analyze how pre-training objective choice affects emergent capabilities such as few-shot prompting, instruction following, and zero-shot generalization.
5. Evaluate the trade-offs between training efficiency and representational quality when choosing between CLM, MLM, and encoder-decoder objectives.
6. Identify how continued pre-training, domain-adaptive pre-training, and multi-task objectives extend standard pre-training to specialized domains.

---

## Overview

Pre-training objectives determine **what patterns** a Foundation Model learns from large-scale data. The choice of objective directly impacts the model's capabilities and downstream task performance.

---

## 1. Language Modeling Paradigms

### 1.1 Three Main Approaches

```
┌─────────────────────────────────────────────────────────────────┐
│                    Language Modeling Paradigms                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Causal LM (Autoregressive)        Masked LM (Bidirectional)   │
│  ┌───┬───┬───┬───┬───┐            ┌───┬───┬───┬───┬───┐        │
│  │ A │ B │ C │ D │ ? │            │ A │[M]│ C │[M]│ E │        │
│  └───┴───┴───┴───┴───┘            └───┴───┴───┴───┴───┘        │
│       ↓                                 ↓                       │
│  P(x_t | x_<t)                     P(x_mask | x_context)        │
│  "Predict next token"              "Restore masked token"       │
│                                                                 │
│  Prefix LM (Encoder-Decoder)                                    │
│  ┌───┬───┬───┐ → ┌───┬───┬───┐                                 │
│  │ A │ B │ C │   │ X │ Y │ Z │                                 │
│  └───┴───┴───┘   └───┴───┴───┘                                 │
│  Bidirectional    Autoregressive                                │
│  "Encode input"   "Generate output"                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Paradigm Comparison

| Feature | Causal LM | Masked LM | Prefix LM |
|---------|-----------|-----------|-----------|
| Representative Models | GPT, LLaMA | BERT, RoBERTa | T5, BART |
| Context | Left-only | Bidirectional | Encoder: bidirectional, Decoder: left |
| Training Signal | All tokens | Masked tokens only (15%) | Span/sequence |
| Generation | Natural generation | Requires additional training | Natural generation |
| Understanding | Zero-shot capable | Strong representation learning | Balanced |

---

## 2. Causal Language Modeling (CLM)

### 2.1 Mathematical Definition

```
Objective Function:
L_CLM = -Σ log P(x_t | x_1, x_2, ..., x_{t-1})

Characteristics:
- Uses all tokens in sequence as training signal
- Autoregressive: left→right sequential generation
- Causal Mask blocks access to future tokens
```

### 2.2 PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalLMHead(nn.Module):
    """Causal Language Model output layer"""

    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        return self.lm_head(hidden_states)


def causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100
) -> torch.Tensor:
    """
    Calculate Causal LM Loss

    Args:
        logits: (batch, seq_len, vocab_size)
        labels: (batch, seq_len) - next token as label
    """
    # Shift: logits[:-1] predicts labels[1:]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index
    )
    return loss


# Create Causal Mask
def create_causal_mask(seq_len: int) -> torch.Tensor:
    """
    Generate upper triangular mask (block future tokens)

    Returns:
        mask: (seq_len, seq_len) - True = masked
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask


# Usage example
batch_size, seq_len, hidden_dim, vocab_size = 4, 128, 768, 50257
hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
labels = torch.randint(0, vocab_size, (batch_size, seq_len))

lm_head = CausalLMHead(hidden_dim, vocab_size)
logits = lm_head(hidden_states)
loss = causal_lm_loss(logits, labels)
print(f"CLM Loss: {loss.item():.4f}")
```

### 2.3 GPT-style Training

```python
class GPTPretraining:
    """GPT-style Pre-training"""

    def __init__(self, model, tokenizer, max_length=1024):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def prepare_data(self, texts: list[str]) -> dict:
        """
        Split continuous text into fixed-length chunks

        Document 1: "The cat sat on..."
        Document 2: "Machine learning is..."

        → [BOS] The cat sat on... [EOS] [BOS] Machine learning is... [EOS]
        → Split into fixed-length chunks (max_length units)
        """
        # Concatenate all texts
        all_tokens = []
        for text in texts:
            tokens = self.tokenizer.encode(text)
            all_tokens.extend(tokens)
            all_tokens.append(self.tokenizer.eos_token_id)

        # Split into fixed-length chunks
        chunks = []
        for i in range(0, len(all_tokens) - self.max_length, self.max_length):
            chunk = all_tokens[i:i + self.max_length]
            chunks.append(chunk)

        return {
            'input_ids': torch.tensor(chunks),
            'labels': torch.tensor(chunks)  # Same (shift happens in loss)
        }

    def train_step(self, batch):
        """Single training step"""
        input_ids = batch['input_ids']
        labels = batch['labels']

        # Forward
        outputs = self.model(input_ids)
        logits = outputs.logits

        # Loss
        loss = causal_lm_loss(logits, labels)

        return loss
```

---

## 3. Masked Language Modeling (MLM)

### 3.1 BERT-style MLM

```
Original: "The quick brown fox jumps over the lazy dog"

Masking Strategy (15% of tokens):
- 80%: Replace with [MASK] token
- 10%: Replace with random token
- 10%: Keep original

Result: "The [MASK] brown fox jumps over the [MASK] dog"
              ↓                          ↓
Target:    "quick"                    "lazy"
```

### 3.2 Implementation

```python
import random

class MLMDataCollator:
    """Masked Language Modeling data preprocessing"""

    def __init__(
        self,
        tokenizer,
        mlm_probability: float = 0.15,
        mask_token_ratio: float = 0.8,
        random_token_ratio: float = 0.1
    ):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.mask_token_ratio = mask_token_ratio
        self.random_token_ratio = random_token_ratio

        # Special token IDs
        self.mask_token_id = tokenizer.mask_token_id
        self.vocab_size = tokenizer.vocab_size
        self.special_tokens = set([
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.pad_token_id
        ])

    def __call__(self, batch: list[dict]) -> dict:
        """Process batch"""
        input_ids = torch.stack([item['input_ids'] for item in batch])

        # Masking
        input_ids, labels = self.mask_tokens(input_ids)

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': torch.stack([item['attention_mask'] for item in batch])
        }

    def mask_tokens(
        self,
        input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform token masking

        Returns:
            masked_input_ids: Masked input
            labels: Original tokens (unmasked positions are -100)
        """
        labels = input_ids.clone()

        # Masking probability matrix
        probability_matrix = torch.full(input_ids.shape, self.mlm_probability)

        # Don't mask special tokens
        special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in self.special_tokens:
            special_tokens_mask |= (input_ids == token_id)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # Select positions to mask
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Unmasked positions are -100 (ignore in loss)
        labels[~masked_indices] = -100

        # 80%: Replace with [MASK]
        indices_replaced = torch.bernoulli(
            torch.full(input_ids.shape, self.mask_token_ratio)
        ).bool() & masked_indices
        input_ids[indices_replaced] = self.mask_token_id

        # 10%: Random token
        indices_random = torch.bernoulli(
            torch.full(input_ids.shape, self.random_token_ratio / (1 - self.mask_token_ratio))
        ).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.vocab_size, input_ids.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        # Remaining 10%: Keep original (handled implicitly)

        return input_ids, labels


def mlm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """MLM Loss (compute only on masked positions)"""
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )
```

### 3.3 RoBERTa Improvements

```python
class RoBERTaMLM:
    """
    RoBERTa: Improved MLM version

    Changes from BERT:
    1. Dynamic Masking: Different masking per epoch
    2. Longer sequences (512 → longer)
    3. Larger batches (256 → 8K)
    4. NSP removal
    5. More data, longer training
    """

    def __init__(self, tokenizer):
        self.collator = MLMDataCollator(tokenizer)

    def create_epoch_data(self, texts: list[str], epoch: int):
        """
        Dynamic Masking: New masking pattern each epoch
        """
        # Change seed based on epoch
        random.seed(epoch)
        torch.manual_seed(epoch)

        # Data preprocessing (apply new masking)
        # ...
```

---

## 4. Span Corruption (T5)

### 4.1 Concept

```
Original: "The quick brown fox jumps over the lazy dog"

Span Corruption:
- Replace consecutive token spans with a single sentinel token
- Decoder restores original span

Input: "The <X> fox <Y> over the lazy dog"
Output: "<X> quick brown <Y> jumps"

Characteristics:
- Average span length: 3 tokens
- Masking ratio: 15%
- Sentinels: <extra_id_0>, <extra_id_1>, ...
```

### 4.2 Implementation

```python
class SpanCorruptionCollator:
    """T5-style Span Corruption"""

    def __init__(
        self,
        tokenizer,
        noise_density: float = 0.15,
        mean_span_length: float = 3.0
    ):
        self.tokenizer = tokenizer
        self.noise_density = noise_density
        self.mean_span_length = mean_span_length

        # Sentinel tokens (<extra_id_0>, <extra_id_1>, ...)
        self.sentinel_start_id = tokenizer.convert_tokens_to_ids("<extra_id_0>")

    def __call__(self, examples: list[dict]) -> dict:
        """Process batch"""
        batch_inputs = []
        batch_targets = []

        for example in examples:
            input_ids = example['input_ids']
            inputs, targets = self.corrupt_span(input_ids)
            batch_inputs.append(inputs)
            batch_targets.append(targets)

        # Padding
        inputs_padded = self._pad_sequences(batch_inputs)
        targets_padded = self._pad_sequences(batch_targets)

        return {
            'input_ids': inputs_padded,
            'labels': targets_padded
        }

    def corrupt_span(
        self,
        input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Span Corruption
        """
        length = len(input_ids)
        num_noise_tokens = int(length * self.noise_density)
        num_spans = max(1, int(num_noise_tokens / self.mean_span_length))

        # Sample span start positions
        span_starts = sorted(random.sample(range(length - 1), num_spans))

        # Span lengths (exponential distribution)
        span_lengths = torch.poisson(
            torch.full((num_spans,), self.mean_span_length - 1)
        ).long() + 1

        # Create span mask
        noise_mask = torch.zeros(length, dtype=torch.bool)
        for start, span_len in zip(span_starts, span_lengths):
            end = min(start + span_len, length)
            noise_mask[start:end] = True

        # Construct input: replace noise spans with sentinels
        input_tokens = []
        target_tokens = []
        sentinel_id = self.sentinel_start_id

        i = 0
        while i < length:
            if noise_mask[i]:
                # Span start: add sentinel
                input_tokens.append(sentinel_id)
                target_tokens.append(sentinel_id)

                # Add span content to target
                while i < length and noise_mask[i]:
                    target_tokens.append(input_ids[i].item())
                    i += 1

                sentinel_id += 1
            else:
                input_tokens.append(input_ids[i].item())
                i += 1

        return torch.tensor(input_tokens), torch.tensor(target_tokens)

    def _pad_sequences(self, sequences: list[torch.Tensor]) -> torch.Tensor:
        """Pad sequences"""
        max_len = max(len(seq) for seq in sequences)
        padded = torch.full((len(sequences), max_len), self.tokenizer.pad_token_id)
        for i, seq in enumerate(sequences):
            padded[i, :len(seq)] = seq
        return padded
```

---

## 5. UL2: Unified Language Learner

### 5.1 Mixture of Denoisers (MoD)

```
UL2: Training with mixture of objectives

┌────────────────────────────────────────────────────────────────┐
│                    Mixture of Denoisers                        │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  R-Denoiser (Regular)      S-Denoiser (Short)                 │
│  - Short spans (3-8 tokens) - Very short spans (≤3 tokens)    │
│  - 15% masking              - 15% masking                      │
│  - Good for NLU tasks       - Good for fine-grained understanding │
│                                                                │
│  X-Denoiser (Extreme)                                          │
│  - Long spans (12-64 tokens)                                   │
│  - 50% masking                                                 │
│  - Good for generation tasks                                   │
│                                                                │
│  Mode Switching: Add [R], [S], [X] prefix to input            │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 5.2 Implementation

```python
class UL2Collator:
    """UL2 Mixture of Denoisers"""

    DENOISERS = {
        'R': {  # Regular
            'span_length': (3, 8),
            'noise_density': 0.15,
            'prefix': '[R]'
        },
        'S': {  # Short
            'span_length': (1, 3),
            'noise_density': 0.15,
            'prefix': '[S]'
        },
        'X': {  # Extreme
            'span_length': (12, 64),
            'noise_density': 0.5,
            'prefix': '[X]'
        }
    }

    def __init__(self, tokenizer, denoiser_weights: dict = None):
        self.tokenizer = tokenizer
        # Default weights: R=50%, S=25%, X=25%
        self.weights = denoiser_weights or {'R': 0.5, 'S': 0.25, 'X': 0.25}

    def __call__(self, examples: list[dict]) -> dict:
        """Process batch: Apply random denoiser to each example"""
        batch_inputs = []
        batch_targets = []

        for example in examples:
            # Select denoiser
            denoiser = random.choices(
                list(self.DENOISERS.keys()),
                weights=list(self.weights.values())
            )[0]

            config = self.DENOISERS[denoiser]

            # Apply span corruption
            inputs, targets = self.apply_denoiser(
                example['input_ids'],
                config
            )

            batch_inputs.append(inputs)
            batch_targets.append(targets)

        return self._collate(batch_inputs, batch_targets)

    def apply_denoiser(
        self,
        input_ids: torch.Tensor,
        config: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply corruption with specific denoiser config"""
        # Add prefix
        prefix_ids = self.tokenizer.encode(
            config['prefix'],
            add_special_tokens=False
        )

        # Span corruption (according to config)
        span_len = random.randint(*config['span_length'])
        # ... corruption logic

        # Prefix + input
        inputs = torch.cat([
            torch.tensor(prefix_ids),
            input_ids  # corrupted
        ])

        return inputs, targets
```

---

## 6. Next Sentence Prediction (NSP) vs Sentence Order Prediction (SOP)

### 6.1 NSP (BERT)

```python
class NSPDataCollator:
    """
    Next Sentence Prediction

    50%: Actual next sentence (IsNext)
    50%: Random sentence (NotNext)

    Problem: Too easy → Removed in RoBERTa
    """

    def create_nsp_pair(
        self,
        sentence_a: str,
        sentence_b: str,
        all_sentences: list[str]
    ) -> tuple[str, str, int]:
        """Create NSP data"""
        if random.random() < 0.5:
            # Actual next sentence
            return sentence_a, sentence_b, 1  # IsNext
        else:
            # Random sentence
            random_sentence = random.choice(all_sentences)
            return sentence_a, random_sentence, 0  # NotNext
```

### 6.2 SOP (ALBERT)

```python
class SOPDataCollator:
    """
    Sentence Order Prediction (harder task)

    50%: Normal order (A → B)
    50%: Reversed (B → A)

    Order prediction not topic prediction → More useful training signal
    """

    def create_sop_pair(
        self,
        sentence_a: str,
        sentence_b: str
    ) -> tuple[str, str, int]:
        """Create SOP data"""
        if random.random() < 0.5:
            return sentence_a, sentence_b, 1  # Normal order
        else:
            return sentence_b, sentence_a, 0  # Reversed
```

---

## 7. Pre-training Objective Selection Guide

### 7.1 Recommended Objectives by Task

```
┌──────────────────┬─────────────────────────────────────────┐
│ Downstream Task  │ Recommended Pre-training Objective      │
├──────────────────┼─────────────────────────────────────────┤
│ Text Generation  │ Causal LM (GPT-style)                   │
│ Text Classification │ MLM (BERT) or Causal LM + Fine-tuning │
│ Question Answering │ Span Corruption (T5) or MLM            │
│ Translation/Summarization │ Encoder-Decoder (T5, BART)     │
│ General (Few-shot) │ Large-scale Causal LM (GPT-3 style)   │
│ General (Diverse) │ UL2 (Mixture of Denoisers)             │
└──────────────────┴─────────────────────────────────────────┘
```

### 7.2 Strategy by Model Size

| Model Size | Recommended Approach | Reason |
|------------|----------------------|--------|
| < 1B | MLM + Fine-tuning | Excellent task-specific performance |
| 1B - 10B | Causal LM | Balance of versatility and efficiency |
| > 10B | Causal LM | Emergence of in-context learning |

---

## 8. Practice: Comparing Objectives

```python
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    T5ForConditionalGeneration,
    AutoTokenizer
)

def compare_objectives():
    """Compare three objectives"""

    # 1. Causal LM (GPT-2)
    causal_tokenizer = AutoTokenizer.from_pretrained('gpt2')
    causal_model = AutoModelForCausalLM.from_pretrained('gpt2')

    text = "The capital of France is"
    inputs = causal_tokenizer(text, return_tensors='pt')

    # Generate
    outputs = causal_model.generate(
        inputs['input_ids'],
        max_new_tokens=5,
        do_sample=False
    )
    print("Causal LM:", causal_tokenizer.decode(outputs[0]))
    # → "The capital of France is Paris."

    # 2. Masked LM (BERT)
    mlm_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    mlm_model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')

    text = "The capital of France is [MASK]."
    inputs = mlm_tokenizer(text, return_tensors='pt')

    outputs = mlm_model(**inputs)
    mask_idx = (inputs['input_ids'] == mlm_tokenizer.mask_token_id).nonzero()[0, 1]
    predicted_id = outputs.logits[0, mask_idx].argmax()
    print("Masked LM:", mlm_tokenizer.decode(predicted_id))
    # → "paris"

    # 3. Span Corruption (T5)
    t5_tokenizer = AutoTokenizer.from_pretrained('t5-small')
    t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')

    text = "translate English to French: The house is wonderful."
    inputs = t5_tokenizer(text, return_tensors='pt')

    outputs = t5_model.generate(inputs['input_ids'], max_new_tokens=20)
    print("T5:", t5_tokenizer.decode(outputs[0], skip_special_tokens=True))
    # → "La maison est merveilleuse."

if __name__ == "__main__":
    compare_objectives()
```

---

## References

### Papers
- Devlin et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers"
- Radford et al. (2019). "Language Models are Unsupervised Multitask Learners" (GPT-2)
- Raffel et al. (2019). "Exploring the Limits of Transfer Learning with T5"
- Tay et al. (2022). "UL2: Unifying Language Learning Paradigms"

### Related Lessons
- [../LLM_and_NLP/03_BERT_GPT_Architecture.md](../LLM_and_NLP/03_BERT_GPT_Architecture.md)
- [../Deep_Learning/12_Transformer_Architecture.md](../Deep_Learning/12_Transformer_Architecture.md)

---

## Exercises

### Exercise 1: Objective Function Comparison

For each scenario below, identify the most appropriate pre-training objective (CLM, MLM, or Span Corruption) and justify your choice.

1. You are building a chatbot that needs to generate fluent multi-turn conversations.
2. You want to fine-tune a model for sentiment classification on movie reviews.
3. You are building a question-answering system that takes a document and a question and produces a short answer.
4. You want a single model that performs well at both classification and generation with no task-specific fine-tuning.

<details>
<summary>Show Answer</summary>

1. **Chatbot (Causal LM / CLM)** — Generation requires predicting the next token autoregressively. CLM is the natural objective for this since the model learns to generate fluent continuations given prior context. GPT-style models excel here.

2. **Sentiment Classification (MLM)** — For understanding-focused tasks with fine-tuning, bidirectional context (MLM) produces richer token representations than CLM. BERT-style models typically outperform GPT models on classification with the same parameter count.

3. **Question Answering (Span Corruption / Encoder-Decoder)** — T5-style span corruption teaches the model to reconstruct missing spans from context, which directly mimics the QA task structure (given a document with a masked span, recover the answer). The encoder-decoder architecture also allows flexible length output.

4. **Single model for both (UL2 / Mixture of Denoisers)** — UL2 trains with multiple denoising objectives (R, S, X denoisers) covering both short-span MLM-like tasks and long-span generation tasks. The mode prefix ([R], [S], [X]) allows task-type specification at inference time without fine-tuning.

</details>

---

### Exercise 2: BERT Masking Strategy

In BERT's MLM, 15% of tokens are selected for masking. Of those, 80% are replaced with `[MASK]`, 10% with a random token, and 10% kept unchanged.

1. Why are only 10% of selected tokens kept unchanged (rather than 0%)?
2. Why are 10% of selected tokens replaced with random tokens (rather than always using `[MASK]`)?
3. What problem would arise if you masked 50% of tokens instead of 15%?

<details>
<summary>Show Answer</summary>

**1. Why keep 10% unchanged:**
Without keeping some tokens unchanged, the model never learns to produce good representations for tokens that appear exactly as written. During fine-tuning, `[MASK]` tokens don't appear — the model must rely on representations of actual tokens. The 10% unchanged forces the model to maintain a useful representation for every token, not just `[MASK]` positions.

**2. Why use 10% random tokens:**
Without random token substitution, the model could "cheat" by learning to recognize that `[MASK]` tokens always need to be predicted and treating other tokens as passthrough. Random substitution forces the model to consider every token as potentially "wrong" and produce contextually grounded predictions for each position — improving the quality of all token representations, not just masked ones.

**3. Problem with 50% masking:**
- Too much context is destroyed: the model no longer has enough surrounding tokens to correctly infer the masked positions, making the task too hard or even unsolvable.
- The training signal becomes noisy because many predictions are guesses rather than context-informed.
- The resulting representations may be degraded because the model sees sequences that bear little resemblance to natural text.
- The 15% rate was empirically found to balance learning signal quality and context availability.

</details>

---

### Exercise 3: Causal Mask Implementation

The `create_causal_mask` function in the lesson creates a boolean upper-triangular matrix. Trace through what happens when this mask is applied in scaled dot-product attention.

Given the query-key dot product matrix for a 4-token sequence:
```
scores = [[0.9, 0.3, 0.7, 0.5],
          [0.2, 0.8, 0.1, 0.4],
          [0.6, 0.5, 0.9, 0.3],
          [0.4, 0.7, 0.8, 0.6]]
```

Show the masked scores matrix and the final attention weights after softmax (conceptually, not numerically).

<details>
<summary>Show Answer</summary>

**Step 1: The causal mask (True = block)**

```python
mask = torch.triu(torch.ones(4, 4), diagonal=1).bool()
# [[False, True,  True,  True ],
#  [False, False, True,  True ],
#  [False, False, False, True ],
#  [False, False, False, False]]
```

**Step 2: Apply mask (set masked positions to -infinity)**

```
masked_scores = [[ 0.9, -inf, -inf, -inf],
                 [ 0.2,  0.8, -inf, -inf],
                 [ 0.6,  0.5,  0.9, -inf],
                 [ 0.4,  0.7,  0.8,  0.6]]
```

**Step 3: Softmax row-wise**

After softmax, `-inf` positions become 0 (they receive zero attention weight):
- Token 1 (row 0): attends only to itself (position 0)
- Token 2 (row 1): distributes attention between positions 0 and 1
- Token 3 (row 2): distributes attention between positions 0, 1, 2
- Token 4 (row 3): distributes attention across all four positions

This ensures position t can only attend to positions ≤ t, preventing information "leakage" from future tokens during training. The model learns to predict the next token using only past context.

</details>

---

### Exercise 4: Span Corruption vs MLM Signal Density

T5's span corruption masks ~15% of tokens but replaces each span (average 3 tokens) with a single sentinel token, whereas BERT masks individual 15% tokens.

Calculate the **training signal density** (fraction of output tokens that are predicted) for each objective, given a sequence of 512 tokens. Which provides denser supervision signal?

<details>
<summary>Show Answer</summary>

**BERT MLM:**
- Input tokens: 512
- Masked tokens: 512 × 0.15 = 76.8 ≈ 77 tokens predicted
- Fraction of output predicted: 77 / 512 = **15%**

**T5 Span Corruption:**
- Input tokens: 512
- Noise tokens: 512 × 0.15 = 76.8 ≈ 77 tokens corrupted
- Number of spans (avg 3 tokens): 77 / 3 ≈ 26 spans
- Output sequence: 26 sentinels + 77 original tokens = 103 tokens
- The decoder must predict all 103 output tokens
- Fraction of output predicted: 103 / 512 = **~20%**

However, the more important difference is the **decoder's role**: in T5, the decoder predicts the entire target sequence autoregressively, which means it also learns to generate the sentinel structure. In BERT, the encoder predicts all masked positions in parallel (non-autoregressive).

**Key insight:** Span corruption requires the model to predict longer contiguous sequences (full spans) rather than isolated tokens, which better prepares it for generative tasks where coherent multi-token outputs are needed.

</details>
