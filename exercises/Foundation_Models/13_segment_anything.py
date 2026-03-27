"""
Exercises for Lesson 13: Segment Anything (SAM)
Topic: Foundation_Models

Solutions to practice problems from the lesson.
"""


# === Exercise 1: SAM Architecture Asymmetry ===
# Problem: Explain design rationale for huge encoder vs tiny decoder.

def exercise_1():
    """Solution: SAM architecture asymmetry"""
    print("  Image Encoder: ViT-H, 632M params (runs ONCE per image)")
    print("  Mask Decoder: 2-layer Transformer, ~4M params (runs PER PROMPT)")
    print()
    print("  Design rationale:")
    print("    All heavy computation (extracting rich features) is amortized")
    print("    over all subsequent interactions. The decoder runs per prompt")
    print("    and can be re-run dozens of times during interactive annotation.")
    print()
    print("  Practical advantages:")
    print("    1. Pre-computation: image embedding (64x64x256) computed and cached")
    print("    2. Instant response: each new prompt only runs 4M-param decoder (ms)")
    print("    3. Real-time interactivity: iterative point additions -> instant masks")
    print("    4. Decoupled compute: encoder on GPU server, decoder can run client-side")
    print()
    print("  This asymmetry makes SAM feel 'interactive' rather than batch-processing.")


# === Exercise 2: SA-1B Data Engine Analysis ===
# Problem: Analyze three-phase bootstrap data engine.

def exercise_2():
    """Solution: SA-1B data engine analysis"""
    phases = [
        {
            "phase": "1: Assisted Manual",
            "masks": "4.3M",
            "method": (
                "Annotators use primitive SAM prototype. SAM proposes masks, "
                "humans correct/refine."
            ),
            "contribution": (
                "Establishes ground truth quality. Trains the first capable model."
            ),
        },
        {
            "phase": "2: Semi-Automatic",
            "masks": "5.9M",
            "method": (
                "Stronger SAM auto-generates confident masks. "
                "Humans label only uncertain objects."
            ),
            "contribution": (
                "Scales data while maintaining quality. "
                "Improves coverage of rare/difficult object classes."
            ),
        },
        {
            "phase": "3: Fully Automatic",
            "masks": "1.1B",
            "method": (
                "Mature SAM generates masks using 32x32 grid of points. "
                "No human annotation. Filtered by quality (IoU, stability)."
            ),
            "contribution": (
                "Provides the scale needed for a true foundation model."
            ),
        },
    ]

    print("  Why Phase 3 can't start from scratch:")
    print("    Fully automatic generation requires a capable model.")
    print("    Bootstrap problem: need good data for good model,")
    print("    but need good model to generate good data.")
    print()

    for p in phases:
        print(f"  Phase {p['phase']} ({p['masks']} masks):")
        print(f"    Method: {p['method']}")
        print(f"    Contribution: {p['contribution']}")
        print()

    print("  The bootstrapped approach transforms data collection from")
    print("  O(humans x images) to O(model_quality x images), making")
    print("  billion-scale annotation economically feasible.")


# === Exercise 3: Prompt Types and Multi-mask Output ===
# Problem: Explain multi-mask output and IoU scores.

def exercise_3():
    """Solution: Multi-mask output and prompt types"""
    print("  Why 3 masks with multimask_output=True:")
    print("    A single point click is inherently ambiguous.")
    print("    Clicking on a person's face could mean:")
    print("      (1) just the face, (2) the head, (3) the whole person")
    print("    3 outputs correspond to different granularity levels.")
    print()

    print("  When to use multimask_output=False (single mask):")
    print("    When additional context resolves ambiguity:")
    print("    - Point + box prompts (box constrains extent)")
    print("    - Iterative refinement (already established which level)")
    print()

    print("  When to use multimask_output=True (3 masks):")
    print("    When single point is the only input and ambiguity is high.")
    print("    Let downstream logic (highest IoU or user selection) pick best.")
    print()

    print("  IoU scores:")
    print("    Each score is SAM's predicted confidence that this mask")
    print("    accurately covers the intended object. Predicted Intersection")
    print("    over Union between generated mask and hypothetical ground truth.")
    print()

    print("  Using logits for subsequent calls:")
    print("    Raw logits (pre-sigmoid, 256x256) from a previous prediction")
    print("    can be passed as mask_input to the next call. Tells SAM")
    print("    'here is my previous estimate' for iterative refinement.")


# === Exercise 4: SAM vs SAM 2 Architecture Extension ===
# Problem: Explain why frame-by-frame SAM fails for video tracking.

def exercise_4():
    """Solution: SAM vs SAM 2 for video"""
    print("  Why frame-by-frame SAM fails for video tracking:")
    print("    - No information passes between frames")
    print("    - Object appearance changes (lighting, pose, occlusion)")
    print("    - A new prompt needed for each frame (impractical)")
    print("    - No temporal consistency guaranteed")
    print("    - User's single prompt on frame 0 gives no guidance to frame 50")
    print()

    components = [
        {
            "name": "Memory Encoder",
            "role": (
                "Encodes segmented mask + image features from past frames "
                "into compact memory representations. Compresses 'what the "
                "object looked like at time t' into fixed-size entry."
            ),
        },
        {
            "name": "Memory Bank",
            "role": (
                "Stores fixed-size queue of memory entries from previous "
                "frames (recent + prompted frame). Acts as temporal context "
                "buffer -- model can 'look back' at earlier appearances."
            ),
        },
        {
            "name": "Memory Attention",
            "role": (
                "Cross-attention module conditioning current frame features "
                "on the Memory Bank. For each patch, attends to past object "
                "representations: 'does this region look like the object "
                "I've seen before?' Enables robust tracking through "
                "occlusions and appearance changes."
            ),
        },
    ]

    print("  SAM 2 new components:")
    for c in components:
        print(f"    {c['name']}: {c['role']}")
        print()


if __name__ == "__main__":
    print("=== Exercise 1: SAM Architecture Asymmetry ===")
    exercise_1()
    print("\n=== Exercise 2: SA-1B Data Engine Analysis ===")
    exercise_2()
    print("\n=== Exercise 3: Prompt Types and Multi-mask Output ===")
    exercise_3()
    print("\n=== Exercise 4: SAM vs SAM 2 ===")
    exercise_4()
    print("\nAll exercises completed!")
