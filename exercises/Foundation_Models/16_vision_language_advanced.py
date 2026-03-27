"""
Exercises for Lesson 16: Vision-Language Models Advanced
Topic: Foundation_Models

Solutions to practice problems from the lesson.
"""


# === Exercise 1: VLM Connection Methods Comparison ===
# Problem: Analyze trade-offs of Linear Projection, MLP, and Cross-Attention.

def exercise_1():
    """Solution: VLM connection methods comparison"""
    methods = [
        {
            "name": "Linear Projection",
            "params": "Minimal (vision_dim x llm_dim)",
            "latency": "Minimal -- single matrix multiply",
            "best_for": (
                "Fast training/inference; resource-constrained settings; "
                "when visual features already high quality (CLIP-L)"
            ),
        },
        {
            "name": "MLP (2-3 layers)",
            "params": "Moderate",
            "latency": "Low -- sequential small layers",
            "best_for": (
                "Better modality alignment than linear; most common "
                "balanced choice (LLaVA-1.5 uses 2-layer MLP)"
            ),
        },
        {
            "name": "Cross-Attention",
            "params": "Large (full cross-attention blocks)",
            "latency": "Higher -- attention over all visual tokens at each layer",
            "best_for": (
                "Deep visual-language interaction; high-resolution images "
                "where spatial detail matters"
            ),
        },
    ]

    for m in methods:
        print(f"  {m['name']}:")
        print(f"    Trainable params: {m['params']}")
        print(f"    Latency impact: {m['latency']}")
        print(f"    Best for: {m['best_for']}")
        print()


# === Exercise 2: Visual Instruction Tuning Data Quality ===
# Problem: Analyze GPT-4V as teacher for instruction tuning data.

def exercise_2():
    """Solution: Visual instruction tuning data quality"""
    print("  A) What GPT-4V-generated data captures well:")
    print("    - Descriptive/narrative capabilities (captioning, descriptions)")
    print("    - Common visual reasoning ('person appears to be laughing because...')")
    print("    - Multi-turn conversation format")
    print("    - High-resource domains (everyday objects, indoor/outdoor)")
    print()

    print("  B) What it misses or distorts:")
    print("    - Domain-specific accuracy (medical imaging, technical diagrams)")
    print("    - Precise spatial reasoning (exact counts, measurements)")
    print("    - Calibrated uncertainty (tends to be confident even when wrong)")
    print("    - Dataset biases (cultural, demographic, linguistic)")
    print()

    print("  C) Concrete improvement:")
    print("    Use VERIFIED ground-truth data sources for factual tasks.")
    print("    Example: for object counting, use COCO annotations (verified by")
    print("    multiple annotators) rather than GPT-4V generation.")
    print("    Hybrid pipeline: GPT-4V for descriptive data,")
    print("    ground-truth annotations for quantitative/factual tasks.")


# === Exercise 3: AnyRes High-Resolution Processing ===
# Problem: Trace AnyRes processing for a 1680x672 image.

def exercise_3():
    """Solution: AnyRes processing trace"""
    W, H = 1680, 672
    base_res = 336
    tokens_per_image = 576  # 24x24 patches with 14px patches

    # Calculate tiles
    import math
    num_tiles_w = math.ceil(W / base_res)
    num_tiles_h = math.ceil(H / base_res)
    total_tiles = num_tiles_w * num_tiles_h

    print(f"  Input image: {W}x{H}, base resolution: {base_res}x{base_res}")
    print()
    print(f"  Step 1: Calculate number of tiles")
    print(f"    Horizontal: ceil({W}/{base_res}) = {num_tiles_w} tiles")
    print(f"    Vertical: ceil({H}/{base_res}) = {num_tiles_h} tiles")
    print(f"    Total tiles: {num_tiles_w} x {num_tiles_h} = {total_tiles} tiles")
    print()

    global_tokens = tokens_per_image
    tile_tokens = total_tiles * tokens_per_image
    total_tokens = global_tokens + tile_tokens

    print(f"  Step 2: Calculate total visual tokens")
    print(f"    Each {base_res}x{base_res} image produces {tokens_per_image} tokens")
    print(f"    Global image (resized to {base_res}x{base_res}): {global_tokens} tokens")
    print(f"    Local tiles ({total_tiles} tiles x {tokens_per_image}): {tile_tokens} tokens")
    print(f"    Total visual tokens: {global_tokens} + {tile_tokens} = {total_tokens} tokens")
    print()

    print(f"  For comparison:")
    print(f"    LLaVA 1.0 (single 224x224): 256 tokens")
    print(f"    LLaVA-NeXT for this image: {total_tokens} tokens ({total_tokens/256:.1f}x more)")
    print()

    # Attention computation comparison
    attn_anyres = (total_tokens + 100) ** 2
    attn_llava1 = (256 + 100) ** 2
    print(f"  Attention: ({total_tokens}+100_text)^2 = ~{attn_anyres/1e6:.0f}M entries/layer")
    print(f"  vs LLaVA 1.0: (256+100)^2 = ~{attn_llava1/1e3:.0f}K entries/layer")
    print(f"  ~{attn_anyres/attn_llava1:.0f}x more attention computation per layer")


# === Exercise 4: POPE Hallucination Evaluation ===
# Problem: Design scenario where high POPE accuracy is insufficient.

def exercise_4():
    """Solution: POPE hallucination evaluation limitations"""
    print("  Scenario where high POPE accuracy fails:")
    print()
    print("  Medical document analysis:")
    print("    Image: Chest X-ray with small nodule in upper-right lung")
    print("    POPE question: 'Is there a nodule?' -> Model: 'Yes' (CORRECT)")
    print("    Harmful hallucination NOT caught by POPE:")
    print("      'Nodule located in lower-left lung, measures ~2cm, shows")
    print("       irregular edges consistent with malignancy.'")
    print("      Location WRONG, size FABRICATED, malignancy HALLUCINATED.")
    print()

    print("  Other failure modes POPE misses:")
    print("    - Attribute hallucination ('red truck' when it's blue)")
    print("    - Relational hallucination ('cat ON table' when it's under)")
    print("    - Counting hallucination (says 3 people when there are 5)")
    print("    - Fabricated text (making up sign/label text)")
    print()

    print("  More comprehensive evaluation:")
    print("    1. Attribute benchmark: 'What color is the car?'")
    print("    2. Spatial relation benchmark: 'Is cup left or right of plate?'")
    print("    3. Counting benchmark: 'How many people in this image?'")
    print("    4. Fine-grained OCR verification against ground truth")
    print("    5. Confidence calibration: does uncertainty correlate with accuracy?")


if __name__ == "__main__":
    print("=== Exercise 1: VLM Connection Methods ===")
    exercise_1()
    print("\n=== Exercise 2: Visual Instruction Tuning Quality ===")
    exercise_2()
    print("\n=== Exercise 3: AnyRes Processing ===")
    exercise_3()
    print("\n=== Exercise 4: POPE Hallucination Evaluation ===")
    exercise_4()
    print("\nAll exercises completed!")
