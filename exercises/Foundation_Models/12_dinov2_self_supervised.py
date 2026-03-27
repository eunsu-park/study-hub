"""
Exercises for Lesson 12: DINOv2 Self-Supervised Vision
Topic: Foundation_Models

Solutions to practice problems from the lesson.
"""


# === Exercise 1: Centering and Collapse Prevention ===
# Problem: Explain mode collapse in self-distillation and why centering prevents it.

def exercise_1():
    """Solution: Centering and collapse prevention"""
    print("  Mode collapse without centering:")
    print("    Teacher converges to constant output (one dimension dominates).")
    print("    Student trivially minimizes cross-entropy by copying the constant.")
    print("    Both networks ignore input entirely -- degenerate solution.")
    print()
    print("  Why centering prevents this:")
    print("    Subtracting running mean c from teacher logits before softmax")
    print("    forces zero-mean logits. No single dimension can persistently")
    print("    dominate. Softmax pushed toward more uniform distribution,")
    print("    forcing genuine input-dependent patterns.")
    print()
    print("  Why EMA over batch mean:")
    print("    Single-batch mean is noisy and introduces instability.")
    print("    EMA center c <- m*c + (1-m)*batch_mean provides smooth,")
    print("    stable estimate of global teacher output distribution.")
    print("    Also requires no synchronization across GPUs.")


# === Exercise 2: Multi-crop Local-to-Global Correspondence ===
# Problem: Explain multi-crop strategy and why local crops only go to student.

def exercise_2():
    """Solution: Multi-crop local-to-global correspondence"""
    print("  Core insight: 'local-to-global correspondence'")
    print("    Student must predict what teacher sees in full image (global)")
    print("    given only a small patch (local crop). This forces semantic")
    print("    understanding: recognizing a dog's ear belongs to the same")
    print("    object as the full dog image.")
    print()
    print("  Why local crops go to student only:")
    print("    1. Teacher would produce noisy targets from small, context-poor patches")
    print("    2. High-quality, stable target signal comes from global crops")
    print("    3. Feeding local crops to teacher significantly increases compute cost")
    print()
    print("  The asymmetry is intentional:")
    print("    Teacher = stable global signal")
    print("    Student = learns from limited local views")


# === Exercise 3: DINOv2 iBOT Loss Analysis ===
# Problem: Analyze L_DINO and L_iBOT components.

def exercise_3():
    """Solution: DINOv2 iBOT loss analysis"""
    print("  DINOv2 total loss: L_total = L_DINO + lambda * L_iBOT")
    print()

    print("  Question A: What does each loss capture?")
    print("    L_DINO: Global image-level semantics via CLS tokens.")
    print("      Trains consistent global representations across views.")
    print("    L_iBOT: Local patch-level semantics.")
    print("      Trains model to predict masked patches from context,")
    print("      enabling dense/spatial understanding.")
    print()

    print("  Question B: If lambda = 0 (iBOT disabled)?")
    print("    Model loses dense visual feature quality.")
    print("    Patch tokens not trained for spatially meaningful info.")
    print("    Tasks like segmentation (relying on patch features) degrade.")
    print("    Only CLS token gets strong training signal.")
    print()

    print("  Question C: Why student-masked, teacher-unmasked asymmetry?")
    print("    Teacher sees complete image -> produces ground-truth patch targets.")
    print("    Student must PREDICT masked patches from context ->")
    print("    must model relationships between visible and masked patches.")
    print("    If teacher were also masked, it couldn't produce reliable targets.")


# === Exercise 4: DINOv2 Feature Evaluation ===
# Problem: Choose feature types and head architectures for 2 downstream tasks.

def exercise_4():
    """Solution: DINOv2 feature evaluation"""
    print("  Task A: Image classification (500 labeled examples, 10 classes)")
    print("    Feature: CLS token (shape: [batch, 768])")
    print("    Head: Linear classifier Linear(768, 10)")
    print("    Justification: With 500 examples, max regularization needed.")
    print("      Linear probe on CLS avoids overfitting while leveraging")
    print("      DINOv2's rich global semantic representation.")
    print("      k-NN classification (no head) is also a strong baseline.")
    print()

    print("  Task B: Semantic segmentation (medical images)")
    print("    Feature: Patch tokens (shape: [batch, n_patches, 768])")
    print("      reshaped to spatial grid")
    print("    Head: Lightweight decoder:")
    print("      Conv2d(768, 256, 1)")
    print("      ConvTranspose2d(256, 128, 4, stride=2, padding=1)")
    print("      ConvTranspose2d(128, num_classes, 4, stride=2, padding=1)")
    print("    Justification: Segmentation needs per-pixel predictions.")
    print("      Patch tokens carry spatial/local semantic info.")
    print("      DINOv2's iBOT training makes patch tokens high-quality.")
    print("      Frozen backbone + lightweight decoder is data-efficient.")


if __name__ == "__main__":
    print("=== Exercise 1: Centering and Collapse Prevention ===")
    exercise_1()
    print("\n=== Exercise 2: Multi-crop Local-to-Global Correspondence ===")
    exercise_2()
    print("\n=== Exercise 3: DINOv2 iBOT Loss Analysis ===")
    exercise_3()
    print("\n=== Exercise 4: DINOv2 Feature Evaluation ===")
    exercise_4()
    print("\nAll exercises completed!")
