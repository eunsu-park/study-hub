"""
Exercises for Lesson 19: Parameter-Efficient Fine-Tuning (PEFT)
Topic: Foundation_Models

Solutions to practice problems from the lesson.
"""


# === Exercise 1: LoRA Trainable Parameter Calculation ===
# Problem: Calculate LoRA params at r=8 vs r=64 vs full fine-tuning.

def exercise_1():
    """Solution: LoRA parameter calculation"""
    d_model = 4096
    num_layers = 32

    # Full fine-tuning: q_proj + v_proj per layer
    full_ft_params = 2 * d_model * d_model * num_layers
    print(f"  Full fine-tuning (q_proj + v_proj, all {num_layers} layers):")
    print(f"    2 * {d_model} * {d_model} * {num_layers} = {full_ft_params:,} = ~{full_ft_params/1e9:.2f}B params")
    print()

    for r in [8, 64]:
        # Each LoRA: W_A (d x r) + W_B (r x d)
        lora_per_matrix = d_model * r + r * d_model
        lora_total = 2 * lora_per_matrix * num_layers  # 2 matrices, all layers

        reduction = full_ft_params / lora_total
        pct = lora_total / full_ft_params * 100

        print(f"  LoRA at r={r}:")
        print(f"    Per matrix: W_A({d_model}x{r}) + W_B({r}x{d_model}) = {lora_per_matrix:,}")
        print(f"    Total: 2 * {lora_per_matrix:,} * {num_layers} = {lora_total:,} = ~{lora_total/1e6:.1f}M params")
        print(f"    Reduction: {reduction:.0f}x fewer ({pct:.2f}% of full FT)")
        print()

    print("  Note: At r=8, LoRA adds only ~4.2M parameters -- comparable to")
    print("  a tiny 2-layer MLP, yet enables significant task adaptation.")


# === Exercise 2: QLoRA Memory Analysis ===
# Problem: Calculate memory for full FT, LoRA, and QLoRA on 13B model.

def exercise_2():
    """Solution: QLoRA memory analysis"""
    params = 13e9
    lora_params = 8e6  # r=8 on q+v proj, 32 layers

    # Full FT (FP16)
    model_fp16 = params * 2 / (1024**3)
    optimizer_fp32 = params * 8 / (1024**3)  # 2x fp32 for Adam
    gradients_fp16 = params * 2 / (1024**3)
    full_total = model_fp16 + optimizer_fp32 + gradients_fp16

    # LoRA (BF16)
    lora_adapters = lora_params * 2 / (1024**3)
    lora_optimizer = lora_params * 8 / (1024**3)
    lora_total = model_fp16 + lora_adapters + lora_optimizer

    # QLoRA (NF4 + BF16)
    model_nf4 = params * 0.5 / (1024**3)
    qlora_total = model_nf4 + lora_adapters + lora_optimizer

    print(f"  {'Component':<35} | {'Full FT (FP16)':<15} | {'LoRA (BF16)':<15} | {'QLoRA (NF4)':<15}")
    print("  " + "-" * 90)
    print(f"  {'Model weights':<35} | {model_fp16:>10.0f} GB   | {model_fp16:>10.0f} GB   | {model_nf4:>10.1f} GB")
    print(f"  {'Optimizer (Adam, fp32)':<35} | {optimizer_fp32:>10.0f} GB   | {'N/A':>10}      | {'N/A':>10}")
    print(f"  {'Gradients':<35} | {gradients_fp16:>10.0f} GB   | {'N/A':>10}      | {'N/A':>10}")
    print(f"  {'LoRA adapters (BF16)':<35} | {'N/A':>10}      | {lora_adapters*1024:>7.0f} MB   | {lora_adapters*1024:>7.0f} MB")
    print(f"  {'LoRA optimizer (Adam, FP32)':<35} | {'N/A':>10}      | {lora_optimizer*1024:>7.0f} MB   | {lora_optimizer*1024:>7.0f} MB")
    print(f"  {'TOTAL':<35} | {full_total:>10.0f} GB   | {lora_total:>10.0f} GB   | {qlora_total:>10.1f} GB")
    print()

    print("  Analysis:")
    print(f"    Full FT: ~{full_total:.0f}GB -> needs 2x A100 80GB")
    print(f"    LoRA BF16: ~{lora_total:.0f}GB -> fits on single A100 80GB")
    print(f"    QLoRA NF4: ~{qlora_total:.1f}GB -> fits on consumer GPU (RTX 3090/4090 24GB)")
    print()
    print("  QLoRA was revolutionary: enabled fine-tuning 13B+ models on consumer hardware.")


# === Exercise 3: LoRA Rank Selection ===
# Problem: Select appropriate LoRA rank for 3 different tasks.

def exercise_3():
    """Solution: LoRA rank selection"""
    tasks = [
        {
            "task": "A) Translate tech docs EN->FR (50K sentence pairs)",
            "rank": "r=4 or r=8",
            "justification": (
                "Translation is well-defined; base model already partially knows French. "
                "Low rank captures alignment without overfitting corpus-specific phrases."
            ),
        },
        {
            "task": "B) Learn proprietary API style (200 examples)",
            "rank": "r=16 or r=32",
            "justification": (
                "Completely new API style requires capturing specific syntactic "
                "patterns. With only 200 examples, need sufficient rank for novel "
                "patterns but also higher lora_dropout (0.1-0.2) to prevent overfitting."
            ),
        },
        {
            "task": "C) General instruction following (100K diverse examples)",
            "rank": "r=64 or r=128",
            "justification": (
                "Broad behavioral changes across many task types. "
                "Following formats, CoT reasoning, refusing harmful requests. "
                "Higher rank for diverse patterns. 100K examples supports "
                "higher rank without overfitting."
            ),
        },
    ]

    for t in tasks:
        print(f"  {t['task']}")
        print(f"    Recommended rank: {t['rank']}")
        print(f"    Justification: {t['justification']}")
        print()

    print("  General heuristic:")
    print("    r=1-4:     Very targeted style/format adjustment")
    print("    r=8-16:    Single task domain adaptation (most common default)")
    print("    r=32-64:   Multi-task or complex behavioral changes")
    print("    r=128-256: Near-full-fine-tuning capability needed")


# === Exercise 4: Adapter vs LoRA Inference Speed ===
# Problem: Explain why LoRA can be merged but Adapters cannot.

def exercise_4():
    """Solution: LoRA mergeability analysis"""
    print("  Why LoRA can be merged:")
    print("    LoRA: y = W*x + (B*A)*x = (W + B*A)*x")
    print("    Both paths are LINEAR operations on the same input.")
    print("    Matrix addition combines them:")
    print()
    print("    # Before merging: 2 operations")
    print("    y = W @ x + (lora_B @ lora_A) * (alpha/rank) @ x")
    print()
    print("    # After merging: 1 operation (same numerical result)")
    print("    W_merged = W + (lora_B @ lora_A) * (alpha/rank)  # done once")
    print("    y = W_merged @ x  # at inference")
    print()

    print("  Why Adapters CANNOT be merged:")
    print("    Adapters: x -> Down-project -> GELU -> Up-project -> + residual")
    print("    The non-linear activation (GELU/ReLU) between projections")
    print("    prevents collapsing into a single matrix multiplication.")
    print("    Up(GELU(Down(x))) + x cannot become W_merged * x")
    print()

    print("  Inference overhead comparison:")
    print("    LoRA (merged):    ZERO extra computation -- identical to base model")
    print("    LoRA (unmerged):  +2 matrix multiplications per LoRA layer")
    print("    Adapters:         Sequential bottleneck (down + activation + up)")
    print("    Prefix Tuning:    Extends K/V sequence length, increasing attention")
    print()
    print("  LoRA's mergeability = key practical advantage:")
    print("  Train efficiently (small adapter), deploy at full base model speed.")


if __name__ == "__main__":
    print("=== Exercise 1: LoRA Parameter Calculation ===")
    exercise_1()
    print("\n=== Exercise 2: QLoRA Memory Analysis ===")
    exercise_2()
    print("\n=== Exercise 3: LoRA Rank Selection ===")
    exercise_3()
    print("\n=== Exercise 4: LoRA vs Adapter Inference ===")
    exercise_4()
    print("\nAll exercises completed!")
