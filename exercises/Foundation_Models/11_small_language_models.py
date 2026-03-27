"""
Exercises for Lesson 11: Small Language Models
Topic: Foundation_Models

Solutions to practice problems from the lesson.
"""

import re
import math


# === Exercise 1: SLM Use Case Analysis ===
# Problem: For each scenario, determine SLM (<=7B) or LLM (70B+).

def exercise_1():
    """Solution: SLM use case analysis"""
    cases = [
        {
            "scenario": "Mobile grammar correction while typing, no internet",
            "choice": "SLM",
            "reasons": [
                "Privacy: grammar correction of personal messages should stay on-device",
                "Latency: real-time (<100ms) requires fast inference on mobile CPU/NPU",
                "Hardware: smartphones have 6-16GB RAM; 7B quantized to 4-bit ~= 4GB",
            ],
        },
        {
            "scenario": "Legal document review (200-page contracts, 50+ jurisdictions)",
            "choice": "LLM",
            "reasons": [
                "Long context: 200-page contracts = 150K+ tokens, need large context window",
                "Knowledge breadth: legal reasoning across 50+ jurisdictions needs deep knowledge",
                "Accuracy stakes: legal errors have serious consequences",
            ],
        },
        {
            "scenario": "E-commerce customer service chatbot (refunds, tracking, FAQs)",
            "choice": "SLM",
            "reasons": [
                "Narrow domain: limited, well-defined task scope a fine-tuned SLM handles well",
                "Cost/throughput: thousands of queries/day, SLM inference is dramatically cheaper",
                "Fine-tuning advantage: 7B fine-tuned on company data often beats general 70B",
            ],
        },
        {
            "scenario": "Scientific research synthesis across 500+ papers",
            "choice": "LLM",
            "reasons": [
                "Context length: synthesizing 500 papers needs very large context or complex RAG",
                "Deep reasoning: understanding causality, methodology, statistics scales with size",
                "Nuanced evaluation: identifying conflicting findings needs strong reasoning",
            ],
        },
    ]

    for c in cases:
        print(f"  Scenario: {c['scenario']}")
        print(f"    Choice: {c['choice']}")
        for r in c["reasons"]:
            print(f"      - {r}")
        print()


# === Exercise 2: Knowledge Distillation Loss Design ===
# Problem: Explain temperature T and alpha in distillation loss.

def exercise_2():
    """Solution: Knowledge distillation loss design"""
    import numpy as np

    print("  Distillation loss: L = alpha * L_hard + (1-alpha) * L_soft")
    print()

    # Demonstrate temperature effect
    logits = np.array([5.0, 2.5, 0.1])
    labels = ["class A", "class B", "class C"]

    print("  1. Temperature T controls sharpness of probability distribution:")
    print(f"     Teacher logits: {logits.tolist()}")
    print()

    for T in [1.0, 2.0, 4.0, 10.0]:
        scaled = logits / T
        probs = np.exp(scaled) / np.exp(scaled).sum()
        probs_str = ", ".join(f"{p:.3f}" for p in probs)
        print(f"     T={T:<5}: [{probs_str}]")

    print()
    print("  2. Why high temperature (T > 1) during distillation:")
    print("     At T=1, dominant class drowns out 'dark knowledge' in non-zero")
    print("     probabilities for wrong classes. These encode similarity structure")
    print("     (e.g., 'this looks slightly like class B'). High T amplifies these")
    print("     small probabilities, giving richer gradient signal to the student.")
    print()

    print("  3. Alpha value to prioritize teacher knowledge:")
    print("     Choose alpha close to 0 (e.g., alpha = 0.1).")
    print("     alpha=0: pure soft label loss -> maximum teacher knowledge transfer")
    print("     alpha=1: standard supervised learning, no distillation benefit")
    print("     alpha=0.1-0.3: dominant soft labels + small hard-label anchor")


# === Exercise 3: Quantization Format Comparison ===
# Problem: Compare GPTQ, AWQ, and GGUF.

def exercise_3():
    """Solution: Quantization format comparison"""
    formats = {
        "GPTQ": {
            "approach": (
                "Post-training quantization using Hessian info to minimize "
                "layer-wise reconstruction error"
            ),
            "calibration": "Yes -- ~128 calibration samples for Hessian computation",
            "best_use": "GPU inference on NVIDIA hardware (vLLM, HuggingFace)",
            "mixed_precision": "Limited -- typically uniform bit-width per layer",
        },
        "AWQ": {
            "approach": (
                "Activation-aware: protects 1% of 'salient' weights from "
                "quantization while aggressively quantizing the rest"
            ),
            "calibration": "Yes -- ~128 samples for activation statistics",
            "best_use": "GPU inference; often slightly better quality than GPTQ",
            "mixed_precision": "Yes -- naturally supports via saliency-aware protection",
        },
        "GGUF": {
            "approach": (
                "Format-agnostic container (llama.cpp) supporting multiple "
                "quant types (Q4_0, Q4_K_M, Q8_0), block-wise quantization"
            ),
            "calibration": "No -- static quantization without calibration",
            "best_use": "CPU and edge inference (Apple Silicon, x86, Ollama)",
            "mixed_precision": "Yes -- per-tensor mixed precision supported",
        },
    }

    dims = ["approach", "calibration", "best_use", "mixed_precision"]
    dim_labels = {
        "approach": "Quantization approach",
        "calibration": "Calibration data required",
        "best_use": "Best use case",
        "mixed_precision": "Mixed precision support",
    }

    for dim in dims:
        print(f"  {dim_labels[dim]}:")
        for fmt, vals in formats.items():
            print(f"    {fmt:<6}: {vals[dim]}")
        print()

    print("  Key insight: GPU -> AWQ preferred. CPU/edge -> GGUF (llama.cpp).")


# === Exercise 4: "Textbooks Are All You Need" Data Strategy ===
# Problem: Analyze synthetic textbook data for small models.

def is_textbook_like(text: str) -> bool:
    """Classify text as textbook-like based on structural signals."""
    signals = []

    has_definitions = bool(re.search(r'\b(is defined as|refers to|means that)\b', text))
    has_examples = bool(re.search(r'\b(for example|for instance|such as|e\.g\.)\b', text))
    has_steps = bool(re.search(r'\b(step [0-9]+|first,|second,|finally,)\b', text, re.I))

    words = text.split()
    sentences = max(text.count('.'), 1)
    avg_sentence_length = len(words) / sentences
    good_length = 15 < avg_sentence_length < 40

    unique_words = len(set(w.lower() for w in words))
    high_vocabulary = (unique_words / max(len(words), 1)) > 0.5

    signals = [has_definitions, has_examples, has_steps, good_length, high_vocabulary]
    return sum(signals) >= 3


def exercise_4():
    """Solution: Textbook data strategy analysis"""
    print("  1. Why synthetic textbooks are more efficient:")
    print("     - Information density: no ads, boilerplate, SEO filler")
    print("     - Reasoning patterns: explicit problem-solving steps")
    print("     - Conceptual coverage: systematic domain coverage")
    print("     - Small models need efficiency: limited capacity")
    print()

    print("  2. Risks and limitations:")
    print("     - Synthetic distribution mismatch (too formal, repetitive)")
    print("     - Generator bias amplification")
    print("     - Limited diversity (underrepresents informal reasoning)")
    print("     - Evaluation contamination")
    print()

    # Demonstrate textbook filter
    test_texts = [
        (
            "A neural network is defined as a computational model that consists "
            "of interconnected nodes. For example, a simple perceptron computes "
            "a weighted sum. First, multiply each input by its weight. Second, "
            "sum all weighted inputs. Finally, apply an activation function.",
            True,
        ),
        (
            "BUY NOW! Best deals on electronics! Click here for amazing "
            "discounts on laptops and phones! Limited time offer!",
            False,
        ),
        (
            "Gradient descent refers to an optimization algorithm that "
            "iteratively updates parameters. For instance, in linear regression "
            "we minimize the mean squared error. Step 1: compute the gradient. "
            "Step 2: update weights in the negative gradient direction.",
            True,
        ),
    ]

    print("  3. Textbook quality filter demonstration:")
    for text, expected in test_texts:
        result = is_textbook_like(text)
        status = "PASS" if result == expected else "FAIL"
        snippet = text[:60] + "..."
        print(f"    [{status}] '{snippet}' -> textbook_like={result}")


if __name__ == "__main__":
    print("=== Exercise 1: SLM Use Case Analysis ===")
    exercise_1()
    print("\n=== Exercise 2: Knowledge Distillation Loss Design ===")
    exercise_2()
    print("\n=== Exercise 3: Quantization Format Comparison ===")
    exercise_3()
    print("\n=== Exercise 4: Textbooks Are All You Need ===")
    exercise_4()
    print("\nAll exercises completed!")
