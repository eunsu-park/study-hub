"""
Exercises for Lesson 14: Unified Vision Models
Topic: Foundation_Models

Solutions to practice problems from the lesson.
"""


# === Exercise 1: Task Unification Strategies ===
# Problem: Compare three unification strategies and recommend one for adding
# 3D depth estimation to an existing system.

def exercise_1():
    """Solution: Task unification strategies"""
    strategies = [
        {
            "name": "Seq2Seq with special tokens",
            "models": "Unified-IO, OFA",
            "how": (
                "All inputs/outputs tokenized into shared sequence; "
                "special tokens delimit modalities (<image>, <box>)"
            ),
            "tradeoff": "Most flexible; handles arbitrary I/O; but complex and slower",
        },
        {
            "name": "Contrastive + task heads",
            "models": "Florence, CLIP-based",
            "how": (
                "Shared encoder produces features; "
                "task-specific heads decode for each task"
            ),
            "tradeoff": (
                "Fast inference; heads scale independently; "
                "but limited knowledge sharing between tasks"
            ),
        },
        {
            "name": "Prompt-conditioned generation",
            "models": "PaLI, Florence-2",
            "how": (
                "Natural language prompts select task behavior; "
                "single generative decoder"
            ),
            "tradeoff": (
                "Most user-friendly; zero-shot generalization; "
                "but output parsing fragile for structured tasks"
            ),
        },
    ]

    for s in strategies:
        print(f"  {s['name']} ({s['models']}):")
        print(f"    How: {s['how']}")
        print(f"    Trade-off: {s['tradeoff']}")
        print()

    print("  Recommendation for adding 3D depth estimation:")
    print("    Use Contrastive + task heads strategy.")
    print("    - Shared encoder already pre-trained, no re-training needed")
    print("    - Only a new depth head needs training (lightweight)")
    print("    - Existing tasks unaffected; new head deployed independently")


# === Exercise 2: PaLI Task Format Design ===
# Problem: Design input/output formats for 3 tasks.

def exercise_2():
    """Solution: PaLI task format design"""
    tasks = [
        {
            "task": "A) Dense object counting",
            "input": '<image> Count the number of [object] in this image.',
            "output": '3',
            "note": (
                "Alternative with CoT: 'Scanning left to right: "
                "car at (100,200), car at (350,300), car at (580,250). Total: 3.'"
            ),
        },
        {
            "task": "B) Temporal reasoning (video frame pair)",
            "input": '<image1> <image2> Describe the motion between these two frames.',
            "output": (
                "A red car moved from left of frame to center, "
                "traveling approximately 50% of the frame width."
            ),
            "note": (
                "Alternative for quantitative: "
                "'Describe motion with direction, speed estimate, and changes.'"
            ),
        },
        {
            "task": "C) Document layout analysis",
            "input": '<image> Identify all structural elements and their locations.',
            "output": (
                'title: "Introduction to Neural Networks" [0.05, 0.02, 0.95, 0.08]\n'
                '    paragraph: "This paper presents..." [0.05, 0.10, 0.95, 0.35]\n'
                '    figure: [0.10, 0.38, 0.90, 0.65]\n'
                '    table: [0.05, 0.68, 0.95, 0.92]'
            ),
            "note": "[x1, y1, x2, y2] are normalized coordinates (0-1 range).",
        },
    ]

    for t in tasks:
        print(f"  {t['task']}")
        print(f"    Input: {t['input']}")
        print(f"    Output: {t['output']}")
        print(f"    Note: {t['note']}")
        print()


# === Exercise 3: Unified-IO Tokenization ===
# Problem: Trace tokenization of object detection output.

def exercise_3():
    """Solution: Unified-IO tokenization trace"""
    W, H = 640, 480
    x1, y1, x2, y2 = 128, 96, 384, 384
    num_bins = 1000
    vocab_size = 50000
    num_special = 8

    # Step 1: Normalize to [0, 1]
    x1_norm = x1 / W
    y1_norm = y1 / H
    x2_norm = x2 / W
    y2_norm = y2 / H

    print("  Step 1: Normalize to [0, 1]")
    print(f"    x1_norm = {x1}/{W} = {x1_norm:.3f}")
    print(f"    y1_norm = {y1}/{H} = {y1_norm:.3f}")
    print(f"    x2_norm = {x2}/{W} = {x2_norm:.3f}")
    print(f"    y2_norm = {y2}/{H} = {y2_norm:.3f}")
    print()

    # Step 2: Discretize to bins
    x1_bin = int(x1_norm * num_bins)
    y1_bin = int(y1_norm * num_bins)
    x2_bin = int(x2_norm * num_bins)
    y2_bin = int(y2_norm * num_bins)

    print(f"  Step 2: Discretize to {num_bins} bins")
    print(f"    x1_bin = int({x1_norm:.3f} * {num_bins}) = {x1_bin}")
    print(f"    y1_bin = int({y1_norm:.3f} * {num_bins}) = {y1_bin}")
    print(f"    x2_bin = int({x2_norm:.3f} * {num_bins}) = {x2_bin}")
    print(f"    y2_bin = int({y2_norm:.3f} * {num_bins}) = {y2_bin}")
    print()

    # Offset to avoid collision with text tokens
    offset = vocab_size + num_special
    x1_tok = x1_bin + offset
    y1_tok = y1_bin + offset
    x2_tok = x2_bin + offset
    y2_tok = y2_bin + offset

    print(f"  Step 3: Token IDs (offset = {vocab_size} + {num_special} = {offset})")
    bicycle_token = 12345  # placeholder
    box_open = vocab_size  # 50000
    box_close = vocab_size + 1  # 50001

    token_sequence = [bicycle_token, box_open, x1_tok, y1_tok, x2_tok, y2_tok, box_close]
    labels = ["'bicycle'", "<box>", f"x1={x1_bin}", f"y1={y1_bin}",
              f"x2={x2_bin}", f"y2={y2_bin}", "</box>"]

    print(f"    Full token sequence for 'bicycle at ({x1},{y1},{x2},{y2})':")
    for tok, label in zip(token_sequence, labels):
        print(f"      {tok:>6}  # {label}")

    print()
    print(f"  Output as text: 'bicycle <box> {x1_bin} {y1_bin} {x2_bin} {y2_bin} </box>'")
    print("  Coordinate bins stored in dedicated vocabulary range, separate from text tokens.")


# === Exercise 4: Versatility vs Specialization Trade-off ===
# Problem: Compare unified model vs 5 specialized models for a startup.

def exercise_4():
    """Solution: Versatility vs specialization trade-off"""
    print("  Strategy A (Unified model) -- Pros:")
    print("    - Single model to maintain, deploy, update")
    print("    - Shared representations may improve data efficiency")
    print("    - Zero-shot or few-shot adaptation to new tasks")
    print("    - Lower infrastructure cost")
    print()
    print("  Strategy A -- Cons:")
    print("    - Task interference (different output structures compete)")
    print("    - Individual task accuracy may be suboptimal")
    print("    - Harder to iterate on one task without impacting others")
    print()
    print("  Strategy B (5 Specialized) -- Pros:")
    print("    - Each optimized to maximum accuracy for its task")
    print("    - Failures isolated; independent update cycles")
    print()
    print("  Strategy B -- Cons:")
    print("    - 5 models to maintain/deploy/monitor")
    print("    - No knowledge sharing; redundant feature extraction")
    print("    - Higher infrastructure cost")
    print()
    print("  Recommendation: HYBRID approach")
    print("    Strong unified backbone (DINOv2/Florence) as shared encoder")
    print("    + lightweight specialized heads per task.")
    print("    Single encoder = single inference cost for features.")
    print("    Specialized heads = maximum accuracy per task.")
    print("    Similarity search (task 5) uses CLS embedding directly.")


if __name__ == "__main__":
    print("=== Exercise 1: Task Unification Strategies ===")
    exercise_1()
    print("\n=== Exercise 2: PaLI Task Format Design ===")
    exercise_2()
    print("\n=== Exercise 3: Unified-IO Tokenization ===")
    exercise_3()
    print("\n=== Exercise 4: Versatility vs Specialization ===")
    exercise_4()
    print("\nAll exercises completed!")
