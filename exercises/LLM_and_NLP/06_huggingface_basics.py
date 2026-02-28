"""
Exercises for Lesson 06: HuggingFace Basics
Topic: LLM_and_NLP

Solutions to practice problems from the lesson.
"""

import numpy as np


# === Exercise 1: Pipeline Task Exploration ===
# Problem: Describe the outputs and score interpretations for three different
# NLP tasks: NER, zero-shot classification, and question answering.

def exercise_1():
    """Pipeline task exploration: NER, zero-shot, and QA output analysis."""

    # Simulated NER results (as HuggingFace pipeline would return)
    ner_results = [
        {"word": "Elon Musk", "entity_group": "PER", "score": 0.998},
        {"word": "SpaceX", "entity_group": "ORG", "score": 0.995},
        {"word": "California", "entity_group": "LOC", "score": 0.992},
        {"word": "Tesla", "entity_group": "ORG", "score": 0.989},
        {"word": "Texas", "entity_group": "LOC", "score": 0.991},
    ]

    print("Task 1: Named Entity Recognition (NER)")
    print(f"  Input: 'Elon Musk founded SpaceX in California and Tesla in Texas.'")
    print(f"  Results:")
    for entity in ner_results:
        print(f"    '{entity['word']}' -> {entity['entity_group']} "
              f"(score: {entity['score']:.3f})")
    print(f"  Score interpretation: Per-entity confidence of label assignment.")
    print(f"  Values near 1.0 indicate high confidence.")

    # Simulated zero-shot classification results
    print(f"\nTask 2: Zero-shot Classification")
    print(f"  Input: 'Scientists discover new exoplanet with potential for liquid water'")
    print(f"  Labels: ['astronomy', 'biology', 'technology', 'sports', 'politics']")
    zero_shot_results = [
        ("astronomy", 0.812),
        ("biology", 0.124),
        ("technology", 0.047),
        ("sports", 0.012),
        ("politics", 0.005),
    ]
    print(f"  Results:")
    for label, score in zero_shot_results:
        bar = "#" * int(score * 40)
        print(f"    {label:<12}: {score:.3f} {bar}")
    print(f"  Score interpretation: Probability distribution over candidate labels.")
    print(f"  Sums to ~1.0. No fine-tuning needed -- uses NLI entailment internally.")

    # Simulated QA results
    print(f"\nTask 3: Question Answering")
    context = ("The HuggingFace Transformers library was created in 2018 by "
               "Thomas Wolf and Lysandre Debut. It supports PyTorch, TensorFlow, "
               "and JAX frameworks.")
    qa_results = [
        {"question": "Who created the Transformers library?",
         "answer": "Thomas Wolf and Lysandre Debut", "score": 0.97,
         "start": 58, "end": 88},
        {"question": "What frameworks does it support?",
         "answer": "PyTorch, TensorFlow, and JAX", "score": 0.92,
         "start": 102, "end": 130},
    ]
    print(f"  Context: '{context[:60]}...'")
    for qa in qa_results:
        print(f"\n  Q: {qa['question']}")
        print(f"  A: '{qa['answer']}' (score: {qa['score']:.3f})")
        print(f"     Character span: [{qa['start']}, {qa['end']}]")
    print(f"\n  Score interpretation: Probability that the extracted span is")
    print(f"  the correct answer. Low scores (<0.5) suggest the answer may")
    print(f"  not be in the context.")


# === Exercise 2: Custom Dataset with Trainer API ===
# Problem: Write a function to convert (text, label) tuples into a format
# suitable for the Trainer API, and show how to add F1 score.

def exercise_2():
    """Custom dataset creation for Trainer API with F1 metric."""

    # Sample data
    train_data = [
        ("I love this product!", 1),
        ("Terrible quality, don't buy.", 0),
        ("Amazing experience, highly recommend!", 1),
        ("Worst purchase I've ever made.", 0),
        ("Five stars, very satisfied!", 1),
        ("Broke after one week, total waste.", 0),
    ]

    test_data = [
        ("Great value for money.", 1),
        ("Not what I expected, disappointed.", 0),
    ]

    def create_dataset_dict(data, max_length=64):
        """
        Convert (text, label) tuples to a dictionary format suitable for
        HuggingFace Dataset.from_dict().
        """
        texts, labels = zip(*data)

        # Simulated tokenization (in practice, use AutoTokenizer)
        # Each text gets a list of token IDs and an attention mask
        input_ids_list = []
        attention_mask_list = []

        for text in texts:
            # Simple word-to-index mapping (simulated)
            words = text.lower().split()
            ids = [hash(w) % 30000 for w in words]  # Simulated token IDs

            # Pad/truncate to max_length
            if len(ids) > max_length:
                ids = ids[:max_length]
                mask = [1] * max_length
            else:
                mask = [1] * len(ids) + [0] * (max_length - len(ids))
                ids = ids + [0] * (max_length - len(ids))

            input_ids_list.append(ids)
            attention_mask_list.append(mask)

        return {
            'input_ids': input_ids_list,
            'attention_mask': attention_mask_list,
            'labels': list(labels),  # Trainer expects 'labels' key (not 'label')
        }

    train_dict = create_dataset_dict(train_data)
    test_dict = create_dataset_dict(test_data)

    print("Custom Dataset for Trainer API")
    print("=" * 60)
    print(f"\nTraining samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"\nDataset dictionary keys: {list(train_dict.keys())}")
    print(f"input_ids shape: ({len(train_dict['input_ids'])}, "
          f"{len(train_dict['input_ids'][0])})")
    print(f"Labels: {train_dict['labels']}")

    # Custom metrics: accuracy + F1 score
    def compute_metrics_demo(predictions, labels):
        """Simulated compute_metrics function."""
        preds = np.argmax(predictions, axis=-1)

        # Accuracy
        accuracy = (preds == labels).mean()

        # F1 score (binary)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    # Simulated predictions
    predictions = np.array([[0.3, 0.7], [0.8, 0.2]])  # 2 test samples
    labels = np.array([1, 0])  # Ground truth

    metrics = compute_metrics_demo(predictions, labels)

    print(f"\nCustom Metrics (simulated):")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    print(f"\nKey format requirements for Trainer:")
    print(f"  - Dataset must contain 'labels' key (not 'label')")
    print(f"  - Input tensors must be PyTorch-compatible")
    print(f"  - compute_metrics receives (logits, labels) as EvalPrediction")


# === Exercise 3: Model Selection for Tasks ===
# Problem: For each NLP task, specify the correct AutoModel class and explain
# why that specific class is appropriate.

def exercise_3():
    """AutoModel class selection for different NLP tasks."""

    tasks = [
        {
            "task": "1. Sentence embeddings for semantic similarity",
            "model_class": "AutoModel",
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "reason": "We need raw hidden states to compute embeddings via mean "
                      "pooling. No task-specific head needed.",
            "output": "last_hidden_state -> mean pooling -> (batch, 384)",
        },
        {
            "task": "2. Named Entity Recognition",
            "model_class": "AutoModelForTokenClassification",
            "model_name": "dbmdz/bert-large-cased-finetuned-conll03-english",
            "reason": "NER requires a prediction FOR EACH TOKEN (not just [CLS]). "
                      "TokenClassification adds a linear head per token.",
            "output": "(batch, seq_len, num_labels)",
        },
        {
            "task": "3. Machine translation (English to French)",
            "model_class": "AutoModelForSeq2SeqLM",
            "model_name": "Helsinki-NLP/opus-mt-en-fr",
            "reason": "Translation requires an encoder (understand source) + decoder "
                      "(generate target). Seq2SeqLM handles cross-attention.",
            "output": "Generated target sequence",
        },
        {
            "task": "4. Fill-mask (masked token prediction)",
            "model_class": "AutoModelForMaskedLM",
            "model_name": "bert-base-uncased",
            "reason": "MaskedLM adds a prediction head over the entire vocabulary "
                      "at [MASK] positions.",
            "output": "(batch, seq_len, vocab_size) at mask positions",
        },
        {
            "task": "5. Open-ended text generation",
            "model_class": "AutoModelForCausalLM",
            "model_name": "gpt2",
            "reason": "CausalLM (decoder-only) with causal mask, trained on "
                      "next-token prediction. generate() supports various strategies.",
            "output": "Autoregressively generated token sequence",
        },
    ]

    print("AutoModel Class Selection for NLP Tasks")
    print("=" * 70)

    for task_info in tasks:
        print(f"\n{task_info['task']}")
        print(f"  Model class: {task_info['model_class']}")
        print(f"  Example:     {task_info['model_name']}")
        print(f"  Why:         {task_info['reason']}")
        print(f"  Output:      {task_info['output']}")

    # Summary table
    print(f"\n{'='*70}")
    print(f"Summary: Why the right model class matters")
    print(f"{'='*70}")
    print(f"{'Class':<40} {'Output head':<20} {'Objective'}")
    print(f"{'-'*40} {'-'*20} {'-'*25}")
    summary = [
        ("AutoModel", "None (raw states)", "--"),
        ("AutoModelForTokenClassification", "Linear per token", "CE per token"),
        ("AutoModelForSeq2SeqLM", "Decoder + cross-attn", "Seq2seq CE"),
        ("AutoModelForMaskedLM", "Linear over vocab@mask", "MLM CE"),
        ("AutoModelForCausalLM", "Linear over vocab@all", "Next-token CE"),
    ]
    for cls, head, obj in summary:
        print(f"  {cls:<40} {head:<20} {obj}")

    print(f"\nUsing the wrong class would either output the wrong shape or")
    print(f"fail to load the correct pre-trained weights for the task head.")


if __name__ == "__main__":
    print("=== Exercise 1: Pipeline Task Exploration ===")
    exercise_1()
    print("\n=== Exercise 2: Custom Dataset with Trainer API ===")
    exercise_2()
    print("\n=== Exercise 3: Model Selection for Tasks ===")
    exercise_3()
    print("\nAll exercises completed!")
