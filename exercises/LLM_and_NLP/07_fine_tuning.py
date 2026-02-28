"""
Exercises for Lesson 07: Fine-Tuning
Topic: LLM_and_NLP

Solutions to practice problems from the lesson.
"""


# === Exercise 1: LoRA Parameter Count Analysis ===
# Problem: Calculate the number of trainable parameters with LoRA (rank r=8)
# applied to query and value projection matrices of BERT-base.

def exercise_1():
    """LoRA parameter count analysis for BERT-base."""
    # BERT-base architecture
    num_layers = 12
    d_model = 768
    num_heads = 12
    d_k = d_model // num_heads  # 64 per head
    d_ff = 3072
    vocab_size = 30522

    # Total BERT-base parameters (approximate)
    embeddings = vocab_size * d_model + 512 * d_model + 2 * d_model  # token + pos + segment
    per_layer = (4 * d_model ** 2) + (2 * d_model * d_ff) + (4 * d_model)
    pooler = d_model * d_model + d_model
    total_bert = embeddings + (num_layers * per_layer) + pooler

    # LoRA parameters for rank r=8
    r = 8

    # For each LoRA layer: A matrix (d_model, r) + B matrix (r, d_model)
    lora_params_per_matrix = d_model * r + r * d_model  # A + B
    lora_targets = 2  # query AND value

    total_lora_params = num_layers * lora_targets * lora_params_per_matrix

    percentage = total_lora_params / total_bert * 100
    reduction = total_bert / total_lora_params

    print("LoRA Parameter Count Analysis")
    print("=" * 60)
    print(f"\nBERT-base architecture:")
    print(f"  Layers: {num_layers}")
    print(f"  d_model: {d_model}")
    print(f"  Heads: {num_heads}")
    print(f"  d_k: {d_k}")
    print(f"  d_ff: {d_ff}")
    print(f"  Total parameters: {total_bert:,} (~{total_bert / 1e6:.0f}M)")

    print(f"\nLoRA configuration:")
    print(f"  Rank (r): {r}")
    print(f"  Target modules: query, value")
    print(f"  Layers affected: all {num_layers}")

    print(f"\nLoRA parameter calculation:")
    print(f"  Per matrix: A ({d_model}x{r}) + B ({r}x{d_model}) = "
          f"{lora_params_per_matrix:,} params")
    print(f"  Per layer: {lora_targets} targets x {lora_params_per_matrix:,} = "
          f"{lora_targets * lora_params_per_matrix:,}")
    print(f"  Total: {num_layers} layers x {lora_targets * lora_params_per_matrix:,} = "
          f"{total_lora_params:,}")

    print(f"\nComparison:")
    print(f"  Full fine-tuning:  {total_bert:>12,} parameters (100%)")
    print(f"  LoRA fine-tuning:  {total_lora_params:>12,} parameters ({percentage:.2f}%)")
    print(f"  Reduction:         {reduction:.0f}x fewer trainable parameters")

    print(f"\nWhy LoRA works:")
    print(f"  LoRA adds low-rank matrices A and B such that the weight update")
    print(f"  dW = BA. During forward pass: h = W0*x + BA*x = (W0 + BA)*x.")
    print(f"  Only A and B are updated -- W0 is frozen.")
    print(f"  The hypothesis is that task adaptation has much lower intrinsic")
    print(f"  dimensionality than d_model, so rank-{r} captures most of the")
    print(f"  necessary adaptation.")


# === Exercise 2: Token Alignment for NER ===
# Problem: Write a function that properly aligns NER labels to subword tokens,
# and trace through a concrete example.

def exercise_2():
    """Token alignment for NER with subword tokenization."""

    # Simulated WordPiece tokenization of "Barack Obama was born in Hawaii"
    words = ['Barack', 'Obama', 'was', 'born', 'in', 'Hawaii']
    ner_labels = [1, 2, 0, 0, 0, 5]  # B-PER, I-PER, O, O, O, B-LOC
    label_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']

    # Simulated subword tokenization result
    # "Barack" -> ["bar", "##ack"], "Obama" -> ["obama"], etc.
    subword_tokens = ['[CLS]', 'bar', '##ack', 'obama', 'was', 'born', 'in',
                      'hawaii', '[SEP]']
    # word_ids maps each token position to the original word index
    word_ids = [None, 0, 0, 1, 2, 3, 4, 5, None]

    def align_labels_with_tokens(word_ids, labels):
        """
        Align word-level NER labels to subword tokens.
        Rules:
        - Special tokens (None word_id) get label -100 (ignored in loss)
        - First subword of a word gets the word's label
        - Subsequent subwords of the same word get label -100 (ignored)
        """
        aligned_labels = []
        previous_word_id = None

        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)  # Special token
            elif word_id != previous_word_id:
                aligned_labels.append(labels[word_id])  # First subword
            else:
                aligned_labels.append(-100)  # Continuation subword
            previous_word_id = word_id

        return aligned_labels

    aligned_labels = align_labels_with_tokens(word_ids, ner_labels)

    print("Token Alignment for NER")
    print("=" * 60)
    print(f"\nOriginal words:  {words}")
    print(f"NER labels:      {ner_labels}")
    print(f"Label names:     {[label_names[l] for l in ner_labels]}")

    print(f"\nSubword tokens:  {subword_tokens}")
    print(f"Word IDs:        {word_ids}")
    print(f"Aligned labels:  {aligned_labels}")

    print(f"\n{'Token':<12} {'Word ID':<10} {'Label':<8} {'Label Name'}")
    print("-" * 45)
    for token, word_id, label in zip(subword_tokens, word_ids, aligned_labels):
        label_str = label_names[label] if label != -100 else "IGNORE"
        word_id_str = str(word_id) if word_id is not None else "special"
        marker = " <-- subword" if (word_id is not None and
                 word_ids[subword_tokens.index(token) - 1] == word_id
                 if subword_tokens.index(token) > 0 else False) else ""
        print(f"{token:<12} {word_id_str:<10} {str(label):<8} {label_str}{marker}")

    print(f"\nWhy this matters:")
    print(f"  If we assigned B-PER to all subwords of 'Barack', the model would")
    print(f"  try to predict B-PER for '##ack' even though entities never start")
    print(f"  mid-word. Using -100 for continuation subwords correctly focuses")
    print(f"  learning on first-subword predictions only.")


# === Exercise 3: Fine-Tuning Strategy Selection ===
# Problem: For each scenario, choose the most appropriate fine-tuning strategy.

def exercise_3():
    """Fine-tuning strategy selection for different scenarios."""

    scenarios = [
        {
            "description": "100,000 labeled movie reviews, 4 A100 GPUs",
            "strategy": "Full Fine-tuning",
            "justification": (
                "100k samples is sufficient to update all parameters without "
                "overfitting. With 4 A100 GPUs (40GB VRAM each), you can fit "
                "the full model in memory. Full fine-tuning gives maximum "
                "flexibility and typically best performance when data is abundant."
            ),
            "config": "batch_size=32, epochs=3, lr=2e-5, fp16=True",
        },
        {
            "description": "7B parameter LLM, laptop with 16GB RAM",
            "strategy": "QLoRA (Quantized LoRA)",
            "justification": (
                "7B model in fp16 needs ~14GB just for weights -- barely fits. "
                "4-bit quantization reduces to ~3.5GB, leaving room for "
                "activations and LoRA adapters (~0.3% additional params)."
            ),
            "config": "load_in_4bit=True, r=16, target_modules=['q_proj', 'v_proj']",
        },
        {
            "description": "50 labeled examples, specialized medical classification",
            "strategy": "Prompt Tuning or Few-Shot",
            "justification": (
                "50 examples is too few for reliable LoRA fine-tuning (overfitting "
                "risk). Prompt tuning trains only soft prompt tokens (~1% of params). "
                "Alternative: Use few-shot in-context learning with no training."
            ),
            "config": "num_virtual_tokens=20, prompt_tuning_init='TEXT'",
        },
        {
            "description": "Instruction following on preference data (chosen/rejected)",
            "strategy": "SFT + DPO",
            "justification": (
                "Step 1: SFT on chosen responses to learn target behavior. "
                "Step 2: DPO uses (chosen, rejected) pairs to optimize preference "
                "alignment without a separate reward model."
            ),
            "config": "SFT: epochs=3, then DPO: beta=0.1",
        },
    ]

    print("Fine-Tuning Strategy Selection Guide")
    print("=" * 70)

    for i, scenario in enumerate(scenarios, 1):
        print(f"\nScenario {i}: {scenario['description']}")
        print(f"  Strategy:      {scenario['strategy']}")
        print(f"  Justification: {scenario['justification']}")
        print(f"  Config:        {scenario['config']}")

    # Summary table
    print(f"\n{'='*70}")
    print(f"Quick Reference:")
    print(f"{'='*70}")
    print(f"  {'Situation':<40} {'Method'}")
    print(f"  {'-'*40} {'-'*25}")
    situations = [
        ("Sufficient data + GPU", "Full Fine-tuning"),
        ("Limited GPU memory", "LoRA / QLoRA"),
        ("Very limited data (<100 samples)", "Prompt Tuning / Few-shot"),
        ("LLM alignment", "SFT + DPO/RLHF"),
        ("Need to preserve base capabilities", "LoRA (small r)"),
        ("Domain adaptation with limited labels", "Pre-trained + freeze=True"),
    ]
    for situation, method in situations:
        print(f"  {situation:<40} {method}")


if __name__ == "__main__":
    print("=== Exercise 1: LoRA Parameter Count Analysis ===")
    exercise_1()
    print("\n=== Exercise 2: Token Alignment for NER ===")
    exercise_2()
    print("\n=== Exercise 3: Fine-Tuning Strategy Selection ===")
    exercise_3()
    print("\nAll exercises completed!")
