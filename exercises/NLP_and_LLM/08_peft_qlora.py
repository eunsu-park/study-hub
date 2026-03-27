"""
Exercises for Lesson 08: PEFT and QLoRA
Topic: NLP_and_LLM

Practice problems for parameter-efficient fine-tuning techniques.
"""

from typing import Dict, List, Tuple
import math


# === Exercise 1: LoRA Parameter Count ===
# Problem: Calculate the number of trainable parameters when applying LoRA
# to a weight matrix of shape (d_in, d_out) with rank r.
# Compare this to full fine-tuning.

def exercise_1():
    """Calculate LoRA trainable parameters."""
    print("=" * 60)
    print("Exercise 1: LoRA Parameter Count")
    print("=" * 60)

    d_in = 4096   # Input dimension
    d_out = 4096  # Output dimension
    rank = 16     # LoRA rank

    # TODO: Calculate full fine-tuning parameters
    # full_params = ???
    full_params = d_in * d_out

    # TODO: Calculate LoRA parameters (A matrix: d_in x r, B matrix: r x d_out)
    # lora_params = ???
    lora_params = d_in * rank + rank * d_out

    # TODO: Calculate the parameter reduction ratio
    # reduction = ???
    reduction = lora_params / full_params

    print(f"Full fine-tuning params: {full_params:,}")
    print(f"LoRA params (rank={rank}): {lora_params:,}")
    print(f"Reduction: {reduction:.4%} of original")
    print(f"Savings: {(1 - reduction):.2%}")


# === Exercise 2: LoRA Forward Pass ===
# Problem: Implement a simplified LoRA forward pass.
# Given original weight W, LoRA matrices A and B, and scaling factor alpha/r,
# compute the output: y = Wx + (alpha/r) * BAx

def exercise_2():
    """Implement LoRA forward pass (simulated with lists)."""
    print("\n" + "=" * 60)
    print("Exercise 2: LoRA Forward Pass")
    print("=" * 60)

    # Simulated matrices as nested lists (2D)
    # W: 3x3, A: 3x2 (rank=2), B: 2x3
    W = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # Identity
    A = [[0.1, 0.2], [0.3, 0.1], [0.2, 0.3]]  # d_in x r
    B = [[0.1, 0.2, 0.1], [0.3, 0.1, 0.2]]    # r x d_out
    x = [1.0, 2.0, 3.0]
    alpha = 1.0
    rank = 2

    def matmul_vec(mat: list[list[float]], vec: list[float]) -> list[float]:
        """Multiply matrix by vector."""
        return [sum(row[j] * vec[j] for j in range(len(vec))) for row in mat]

    def matmul_mat(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
        """Multiply two matrices."""
        rows_a, cols_b = len(a), len(b[0])
        cols_a = len(a[0])
        result = [[0.0] * cols_b for _ in range(rows_a)]
        for i in range(rows_a):
            for j in range(cols_b):
                for k in range(cols_a):
                    result[i][j] += a[i][k] * b[k][j]
        return result

    # TODO: Compute Wx (original forward pass)
    # wx = ???
    wx = matmul_vec(W, x)

    # TODO: Compute BA (low-rank update matrix)
    # ba = ???
    ba = matmul_mat(A, B)

    # TODO: Compute BAx
    # bax = ???
    bax = matmul_vec(ba, x)

    # TODO: Compute final output: y = Wx + (alpha/r) * BAx
    # y = ???
    scale = alpha / rank
    y = [wx[i] + scale * bax[i] for i in range(len(wx))]

    print(f"Input x: {x}")
    print(f"Wx (original): {[round(v, 4) for v in wx]}")
    print(f"BAx (LoRA delta): {[round(v, 4) for v in bax]}")
    print(f"Scale (alpha/r): {scale}")
    print(f"Output y: {[round(v, 4) for v in y]}")


# === Exercise 3: Quantization Basics ===
# Problem: Implement simple quantization (FP32 -> INT8) and dequantization.
# Quantize: q = round(x / scale), where scale = max(|x|) / 127
# Dequantize: x_approx = q * scale

def exercise_3():
    """Implement basic quantization and dequantization."""
    print("\n" + "=" * 60)
    print("Exercise 3: INT8 Quantization")
    print("=" * 60)

    values = [0.5, -1.2, 3.7, -0.8, 2.1, -3.5, 1.0, 0.0]

    # TODO: Calculate the scale factor
    # scale = max(|values|) / 127
    # scale = ???
    abs_max = max(abs(v) for v in values)
    scale = abs_max / 127

    # TODO: Quantize values to INT8
    # quantized = [round(v / scale) for v in values]
    # quantized = ???
    quantized = [round(v / scale) for v in values]

    # TODO: Dequantize back to float
    # dequantized = [q * scale for q in quantized]
    # dequantized = ???
    dequantized = [q * scale for q in quantized]

    # TODO: Calculate quantization error
    # errors = [abs(orig - deq) for orig, deq in zip(values, dequantized)]
    # errors = ???
    errors = [abs(orig - deq) for orig, deq in zip(values, dequantized)]

    print(f"Original:     {values}")
    print(f"Scale factor: {scale:.6f}")
    print(f"Quantized:    {quantized}")
    print(f"Dequantized:  {[round(v, 4) for v in dequantized]}")
    print(f"Errors:       {[round(e, 6) for e in errors]}")
    print(f"Max error:    {max(errors):.6f}")
    print(f"Avg error:    {sum(errors)/len(errors):.6f}")


# === Exercise 4: Rank Selection Analysis ===
# Problem: Analyze the trade-off between LoRA rank and parameter count
# for different model sizes. Generate a table showing rank vs params.

def exercise_4():
    """Analyze rank vs parameter count trade-off."""
    print("\n" + "=" * 60)
    print("Exercise 4: Rank Selection Analysis")
    print("=" * 60)

    model_dims = {
        "Small (768)": 768,
        "Base (1024)": 1024,
        "Large (4096)": 4096,
        "XL (8192)": 8192,
    }
    ranks = [1, 4, 8, 16, 32, 64]

    # TODO: For each model size and rank, calculate:
    # 1. LoRA parameters per layer (A: d x r + B: r x d)
    # 2. Percentage of full parameters
    # Display as a table

    print(f"{'Model':<15} {'Rank':<6} {'LoRA Params':<15} {'% of Full':<10}")
    print("-" * 50)

    for model_name, d in model_dims.items():
        full = d * d
        for r in ranks:
            lora = d * r + r * d  # A + B
            pct = lora / full * 100
            print(f"{model_name:<15} {r:<6} {lora:>12,}   {pct:>6.2f}%")
        print()


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
