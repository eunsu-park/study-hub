# 19. BERT

[Previous: Transformer](./18_Impl_Transformer.md) | [Next: GPT](./20_Impl_GPT.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the key innovations in BERT — bidirectional context, masked language modeling (MLM), and next sentence prediction (NSP) — and contrast them with unidirectional language models like GPT.
2. Describe the mathematical formulation of the MLM objective and explain the token masking strategy (80/10/10 split).
3. Implement the BERT encoder architecture from scratch in PyTorch, including multi-head self-attention, feed-forward layers, and positional embeddings.
4. Fine-tune a pretrained BERT model for downstream tasks such as text classification, named entity recognition, and question answering.
5. Interpret BERT's special tokens ([CLS], [SEP], [MASK]) and explain how they are used during pretraining and fine-tuning.
6. Compare the BERT-base and BERT-large configurations and evaluate the trade-off between model size and task performance.

## Overview

BERT (Bidirectional Encoder Representations from Transformers) is a model released by Google in 2018 that revolutionized NLP. It uses **bidirectional context** to understand word meanings.

---

## Mathematical Background

### Theory: Masked Language Modeling

BERT randomly replaces 15% of input tokens and asks the model to reconstruct the original:

- 80% of selected tokens: replaced with `[MASK]`
- 10%: replaced with a random token
- 10%: left unchanged

The model sees the corrupted sequence and produces predictions at every position; loss is computed only at the masked positions. This is a **denoising autoencoder** in disguise: the network learns to use bidirectional context to fill in missing words.

Why the random token / unchanged splits? At fine-tuning time, no token will ever be `[MASK]` (that token is unique to pretraining). If 100% of corrupted tokens were `[MASK]`, the model would learn to ignore non-`[MASK]` positions. The 10/10 splits force the model to use *all* token representations, not just the masked ones.

The MLM objective is much harder than next-token prediction (you only get gradients on 15% of positions per example), but it produces representations that transfer remarkably well.


### 1. Masked Language Modeling (MLM)

```
Objective function:
L_MLM = -Σ log P(x_mask | x_context)

Masking strategy (15% of tokens):
- 80%: replace with [MASK] token
- 10%: replace with random token
- 10%: keep original

Example:
Input: "The [MASK] sat on the mat"
Goal: predict "cat"
```

### 2. Next Sentence Prediction (NSP)

```
50% IsNext:    Sentence A → Sentence B (actual continuation)
50% NotNext:   Sentence A → Random B

Input: [CLS] Sentence A [SEP] Sentence B [SEP]
Output: IsNext / NotNext classification
```

### 3. BERT Embedding

```
Token Embedding:     word meaning
Segment Embedding:   distinguish sentence A/B
Position Embedding:  position information

Input = Token_Emb + Segment_Emb + Position_Emb
```

---

## BERT Architecture

```
BERT-Base:
- Hidden size: 768
- Layers: 12
- Attention heads: 12
- Parameters: 110M

BERT-Large:
- Hidden size: 1024
- Layers: 24
- Attention heads: 16
- Parameters: 340M

Structure:
[CLS] Token1 Token2 ... [SEP] Token1 ... [SEP]
  ↓
Embedding Layer (Token + Segment + Position)
  ↓
Transformer Encoder × L layers
  ↓
[CLS]: classification / Token: token prediction
```

### Theory: Next Sentence Prediction and Its Decline

BERT was also pretrained on **NSP**: given two sentences A and B, predict whether B actually follows A in the source corpus, or is a random sentence. The CLS token's final hidden state was used as the input to a binary classifier.

Subsequent work (RoBERTa, ALBERT) found NSP added little value and sometimes hurt: the random-sentence negative is too easy because random sentences usually come from a different topic, so the model learns a topic classifier rather than a sentence-relationship classifier. Modern BERT-derivatives drop NSP and rely on MLM alone, often with longer training and bigger batches.


### Theory: Bidirectional Self-Attention

In a standard left-to-right LM (GPT), token `t` attends only to `<= t`. In BERT's encoder, each token attends to *all* tokens in the input — past and future. This bidirectionality is essential for many tasks: classifying sentiment of a sentence requires seeing the whole sentence; named entity recognition often requires later context.

The challenge: how do you train a bidirectional model with a language-modeling-like objective? You cannot just predict the next token, because the model can already see it. The answer is masking.


---

## File Structure

```
08_BERT/
├── README.md
├── pytorch_lowlevel/
│   └── bert_lowlevel.py        # Direct BERT Encoder implementation
├── paper/
│   └── bert_paper.py           # Paper reproduction
└── exercises/
    ├── 01_mlm_training.md      # MLM training practice
    └── 02_finetuning.md        # Classification fine-tuning
```

---

## Core Concepts

### Theory: Pretrain-Then-Fine-Tune

The recipe BERT established:

1. **Pretrain** on a huge unlabeled corpus (BooksCorpus + Wikipedia, 3.3B words) with MLM (+ NSP). Cost: ~64 TPUs for 4 days.
2. **Fine-tune** on a small labeled task dataset by adding a task-specific head and training the whole model end-to-end with a small learning rate (~5e-5). Cost: minutes to hours.

This decoupled the expensive part (representation learning) from the cheap part (task adaptation), and made it economic for individual labs to push the state of the art on many tasks. Every modern Foundation Model — BERT, GPT, T5, LLaMA — descends from this recipe, with only the size and the pretraining objective varying.

The math has not changed; the leverage is entirely from pretraining scale.


### 1. Bidirectional Context

```
GPT (Left-to-Right):
"The cat sat" → reference only left to predict next

BERT (Bidirectional):
"The [MASK] sat on the mat" → reference both sides to predict [MASK]

Advantage: richer contextual understanding
Disadvantage: unsuitable for text generation
```

### 2. Pre-training & Fine-tuning

```
Phase 1: Pre-training (large corpus)
- MLM + NSP tasks
- Wikipedia + BookCorpus (3.3B tokens)

Phase 2: Fine-tuning (downstream task)
- Classify with [CLS] token
- Or sequence labeling with all token outputs
```

### 3. Input Format

```
Single sentence: [CLS] tokens [SEP]
Sentence pair:   [CLS] tokens_A [SEP] tokens_B [SEP]

Segment IDs:
[CLS] A A A [SEP] B B B [SEP]
  0   0 0 0   0   1 1 1   1
```

---

## Implementation Levels

### Level 2: PyTorch Low-Level (pytorch_lowlevel/)
- Use F.linear, F.layer_norm
- Don't use nn.TransformerEncoder
- Manual embedding implementation

### Level 3: Paper Implementation (paper/)
- Reproduce exact paper specifications
- MLM + NSP pre-training
- Classification fine-tuning

### Level 4: Code Analysis (separate document)
- Analyze HuggingFace transformers code
- BertModel, BertForSequenceClassification

---

## Learning Checklist

- [ ] Understand MLM masking strategy
- [ ] Understand NSP task
- [ ] Understand Token/Segment/Position Embedding
- [ ] Role of [CLS] token
- [ ] Fine-tuning methods (classification, NER, QA)
- [ ] Differences between BERT vs GPT

---

## References

- Devlin et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
- [HuggingFace BERT](https://huggingface.co/docs/transformers/model_doc/bert)
- [../LLM_and_NLP/03_BERT_GPT_Architecture.md](../LLM_and_NLP/03_BERT_GPT_Architecture.md)
