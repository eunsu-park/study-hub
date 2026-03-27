"""
Exercises for Lesson 04: BERT Understanding
Topic: LLM_and_NLP

Solutions to practice problems from the lesson.
"""

import random
import numpy as np


# === Exercise 1: MLM Data Preparation ===
# Problem: Implement the create_mlm_data function more robustly, and trace
# through what happens to "The cat sat on the mat" when processed with 15%
# masking probability.

def exercise_1():
    """MLM data preparation with BERT's masking strategy."""

    def create_mlm_data(tokens, vocab, mask_prob=0.15, mask_token='[MASK]'):
        """
        Generate MLM training data following BERT's masking strategy.
        For 15% of selected tokens:
        - 80%: Replace with [MASK]
        - 10%: Replace with random token from vocabulary
        - 10%: Keep unchanged (but still predict)
        """
        tokens = tokens.copy()
        labels = [-100] * len(tokens)  # -100 = ignore in cross-entropy loss

        for i, token in enumerate(tokens):
            # Skip special tokens
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue

            if random.random() < mask_prob:
                labels[i] = vocab.get(token, vocab.get('[UNK]', 0))

                rand = random.random()
                if rand < 0.8:
                    tokens[i] = mask_token                           # 80%: [MASK]
                elif rand < 0.9:
                    tokens[i] = random.choice(list(vocab.keys()))    # 10%: random
                # else: keep original token                           # 10%: unchanged

        return tokens, labels

    # Set up vocabulary and tokens
    vocab = {
        '[CLS]': 101, '[SEP]': 102, '[PAD]': 0, '[MASK]': 103,
        '[UNK]': 100, 'the': 1, 'cat': 2, 'sat': 3, 'on': 4, 'mat': 5,
        'dog': 6, 'a': 7, 'is': 8
    }

    tokens = ['[CLS]', 'the', 'cat', 'sat', 'on', 'the', 'mat', '[SEP]']
    print(f"Original tokens: {tokens}")
    print(f"Vocabulary: {vocab}")

    # Run multiple times to show different outcomes
    random.seed(42)
    print("\n--- Multiple MLM data generation runs ---")
    for run in range(3):
        masked_tokens, labels = create_mlm_data(tokens.copy(), vocab, mask_prob=0.3)
        print(f"\nRun {run + 1}:")
        print(f"  Input tokens: {masked_tokens}")
        print(f"  Labels:       {labels}")

        # Show which positions are being predicted
        for i, (tok, label) in enumerate(zip(masked_tokens, labels)):
            if label != -100:
                orig_word = [k for k, v in vocab.items() if v == label][0]
                print(f"  Position {i}: '{tok}' -> predict '{orig_word}' (label={label})")

    # Explain the 10% unchanged strategy
    print("\n--- Why 10% of selected tokens are kept unchanged ---")
    print("If all selected tokens were always replaced with [MASK], the model")
    print("would only learn to predict at [MASK] positions -- it would never")
    print("need to represent the actual token during fine-tuning (where there")
    print("are no [MASK] tokens). By keeping 10% as-is and still predicting")
    print("them, the model must maintain useful representations for all tokens.")


# === Exercise 2: BERT Input Formatting ===
# Problem: For an NLI task with premise and hypothesis, construct the full
# BERT input format showing input_ids, segment_ids, and attention_mask.

def exercise_2():
    """BERT input formatting for NLI tasks."""
    # Simulated tokenizer (mapping words to IDs)
    word2id = {
        '[CLS]': 101, '[SEP]': 102, '[PAD]': 0,
        'the': 1, 'cat': 2, 'is': 3, 'on': 4,
        'mat': 5, 'sleeping': 6
    }

    premise = "the cat is on the mat"
    hypothesis = "the cat is sleeping"

    # Manual construction (what HuggingFace does internally)
    premise_tokens = ['[CLS]'] + premise.split() + ['[SEP]']
    hypothesis_tokens = hypothesis.split() + ['[SEP]']
    all_tokens = premise_tokens + hypothesis_tokens

    # Convert to IDs
    input_ids = [word2id.get(t, 100) for t in all_tokens]

    # Segment IDs: 0 for premise (including CLS/SEP), 1 for hypothesis
    segment_ids = [0] * len(premise_tokens) + [1] * len(hypothesis_tokens)

    # Attention mask: 1 for all real tokens
    attention_mask = [1] * len(all_tokens)

    # Pad to max_length
    max_length = 20
    pad_length = max_length - len(all_tokens)
    input_ids_padded = input_ids + [0] * pad_length
    segment_ids_padded = segment_ids + [0] * pad_length
    attention_mask_padded = attention_mask + [0] * pad_length

    print("BERT Input Formatting for NLI")
    print("=" * 60)
    print(f"Premise:    '{premise}'")
    print(f"Hypothesis: '{hypothesis}'")

    print(f"\nTokens: {all_tokens}")
    print(f"Length: {len(all_tokens)} tokens (+ {pad_length} padding = {max_length})")

    print(f"\nInput IDs (padded to {max_length}):")
    print(f"  {input_ids_padded}")

    print(f"\nSegment IDs (token_type_ids):")
    print(f"  {segment_ids_padded}")
    print(f"  Segment 0 = premise, Segment 1 = hypothesis")

    print(f"\nAttention mask:")
    print(f"  {attention_mask_padded}")
    print(f"  1 = real token, 0 = padding")

    # Visual alignment
    print(f"\nVisual alignment:")
    print(f"  {'Token':<12} {'ID':>4} {'Seg':>4} {'Mask':>5}")
    print(f"  {'-'*12} {'-'*4} {'-'*4} {'-'*5}")
    for i in range(max_length):
        tok = all_tokens[i] if i < len(all_tokens) else '[PAD]'
        print(f"  {tok:<12} {input_ids_padded[i]:>4d} {segment_ids_padded[i]:>4d} "
              f"{attention_mask_padded[i]:>5d}")

    print("\nKey insight: The [CLS] token at position 0 serves as the aggregate")
    print("sequence representation. For NLI, a linear classifier on top of the")
    print("[CLS] output is trained to predict entailment, contradiction, or neutral.")


# === Exercise 3: BERT vs RoBERTa Differences ===
# Problem: For each RoBERTa change, explain why it was made and what
# improvement it provides.

def exercise_3():
    """BERT vs RoBERTa training procedure differences."""

    changes = [
        {
            "change": "1. Removing NSP (Next Sentence Prediction)",
            "why": (
                "The original NSP was problematic because 'NotNext' examples came "
                "from different documents, so the model could distinguish them based "
                "on topic rather than coherent sentence reasoning. NSP also forced "
                "shorter sequences (two half-length sentences), reducing the benefit "
                "of long-range context modeling."
            ),
            "improvement": (
                "MLM alone on full-length single documents provides stronger "
                "bidirectional representations. RoBERTa showed consistent "
                "improvements on several benchmarks after removing NSP."
            ),
        },
        {
            "change": "2. Dynamic masking instead of static masking",
            "why": (
                "In original BERT, masking was applied once during preprocessing -- "
                "the same tokens were always masked for a given example across all epochs. "
                "This limits the diversity of training signal."
            ),
            "improvement": (
                "Dynamic masking applies different masks each epoch, providing more "
                "diverse training signal. It acts as additional data augmentation -- "
                "the model learns to predict any token given any context, not just "
                "specific positions."
            ),
        },
        {
            "change": "3. Using a larger batch size with more data",
            "why": (
                "Larger batches provide better gradient estimation per update and "
                "faster convergence in wall-clock time. More data provides more "
                "diverse language patterns."
            ),
            "improvement": (
                "BERT: batch_size=256, 1M steps, 16GB data. "
                "RoBERTa: batch_size=8192, 500K steps, 160GB data. "
                "Combined result: 2-4% improvement on GLUE benchmark without "
                "any architectural changes."
            ),
        },
    ]

    print("BERT vs RoBERTa: Training Procedure Differences")
    print("=" * 60)

    for item in changes:
        print(f"\n{item['change']}")
        print(f"  Why: {item['why']}")
        print(f"  Improvement: {item['improvement']}")

    # Demonstrate dynamic masking concept
    print("\n\n--- Dynamic Masking Illustration ---")
    tokens = ['the', 'cat', 'sat', 'on', 'the', 'mat']
    random.seed(0)

    for epoch in range(3):
        masked = tokens.copy()
        for i in range(len(masked)):
            if random.random() < 0.15:
                masked[i] = '[MASK]'
        print(f"  Epoch {epoch + 1}: {masked}")
    print("  Each epoch sees different masked positions!")

    print("\nKey takeaway: RoBERTa demonstrated that the training procedure")
    print("is as important as the architecture -- simple changes to data,")
    print("masking, and optimization led to consistent improvements.")


# === Exercise 4: Fine-tuning BERT for Sentiment Analysis ===
# Problem: Write code structure for fine-tuning bert-base-uncased for binary
# sentiment classification, including discriminative learning rates.

def exercise_4():
    """Fine-tuning BERT for sentiment analysis with discriminative LR."""

    print("Fine-tuning BERT for Sentiment Analysis")
    print("=" * 60)

    # Demonstrate discriminative learning rate concept
    print("\nDiscriminative Learning Rates:")
    print("  BERT layers (pre-trained): lr = 2e-5  (preserve existing knowledge)")
    print("  Classifier head (new):     lr = 1e-3  (train from scratch)")

    # Simulate the parameter groups
    bert_params = 110_000_000  # ~110M
    classifier_params = 768 * 2 + 2  # d_model * num_classes + bias
    total_params = bert_params + classifier_params

    print(f"\nParameter breakdown:")
    print(f"  BERT layers:      {bert_params:>12,} ({bert_params / total_params * 100:.2f}%)")
    print(f"  Classifier head:  {classifier_params:>12,} ({classifier_params / total_params * 100:.4f}%)")
    print(f"  Total:            {total_params:>12,}")

    # Simulate training progress
    print("\nSimulated training progress (3 epochs on IMDB):")
    # These are typical values for BERT fine-tuning
    epochs = [
        {"epoch": 1, "train_loss": 0.42, "val_acc": 0.87},
        {"epoch": 2, "train_loss": 0.22, "val_acc": 0.91},
        {"epoch": 3, "train_loss": 0.15, "val_acc": 0.93},
    ]

    print(f"  {'Epoch':>5} {'Train Loss':>12} {'Val Accuracy':>14}")
    print(f"  {'-'*5} {'-'*12} {'-'*14}")
    for e in epochs:
        print(f"  {e['epoch']:>5d} {e['train_loss']:>12.4f} {e['val_acc']:>14.2%}")

    print("\nWhy discriminative learning rates work:")
    print("  BERT's lower layers encode general linguistic knowledge (syntax,")
    print("  morphology) -- already well-trained on billions of tokens.")
    print("  Using a small LR (2e-5) makes tiny adjustments to refine this.")
    print("  The classifier head is randomly initialized and needs larger LR")
    print("  (1e-3) to converge quickly.")
    print("\n  Catastrophic forgetting would occur with a large uniform LR --")
    print("  pre-trained weights would shift dramatically and lose general")
    print("  language understanding.")


if __name__ == "__main__":
    print("=== Exercise 1: MLM Data Preparation ===")
    exercise_1()
    print("\n=== Exercise 2: BERT Input Formatting ===")
    exercise_2()
    print("\n=== Exercise 3: BERT vs RoBERTa Differences ===")
    exercise_3()
    print("\n=== Exercise 4: Fine-tuning BERT ===")
    exercise_4()
    print("\nAll exercises completed!")
