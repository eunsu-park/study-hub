"""
Exercises for Lesson 01: Foundation Model Paradigm
Topic: Foundation_Models

Solutions to practice problems from the lesson.
"""

import re
from collections import Counter


# === Exercise 1: Foundation Model Classification ===
# Problem: For each model, identify (a) its type and (b) its architecture
# (encoder-only, decoder-only, or encoder-decoder). Explain reasoning.

def exercise_1():
    """Solution: Foundation Model Classification"""
    models = {
        "GPT-4": {
            "type": "Language Model (+ Multimodal in vision variant)",
            "architecture": "decoder-only",
            "reasoning": (
                "Generates text autoregressively using a causal mask, "
                "attending only to past tokens."
            ),
        },
        "BERT": {
            "type": "Language Model (embedding/understanding)",
            "architecture": "encoder-only",
            "reasoning": (
                "Uses bidirectional self-attention (masked LM) to produce "
                "context-rich token representations."
            ),
        },
        "T5": {
            "type": "Language Model (text-to-text)",
            "architecture": "encoder-decoder",
            "reasoning": (
                "The encoder processes input with bidirectional attention; "
                "the decoder generates output autoregressively."
            ),
        },
        "DALL-E 3": {
            "type": "Generative Model (Text -> Image)",
            "architecture": "diffusion-based (varies by version)",
            "reasoning": (
                "Maps text descriptions into image representations. "
                "Architecture evolved from VQVAE+transformer to diffusion."
            ),
        },
        "Whisper": {
            "type": "Audio Model (Audio <-> Text)",
            "architecture": "encoder-decoder",
            "reasoning": (
                "An audio encoder processes mel-spectrogram features; "
                "a decoder generates transcription tokens."
            ),
        },
    }

    for model_name, info in models.items():
        print(f"  {model_name}:")
        print(f"    Type: {info['type']}")
        print(f"    Architecture: {info['architecture']}")
        print(f"    Reasoning: {info['reasoning']}")
        print()


# === Exercise 2: In-context Learning Prompt Design ===
# Problem: Write a few-shot prompt for NER that identifies PERSON and LOCATION
# entities. Include 3 examples then an unlabeled test sentence.

def exercise_2():
    """Solution: NER few-shot prompt design"""
    ner_prompt = """Extract PERSON and LOCATION entities from the following sentences.

Sentence: "Albert Einstein was born in Ulm, Germany."
Entities: PERSON: Albert Einstein | LOCATION: Ulm, Germany

Sentence: "Marie Curie conducted research in Paris."
Entities: PERSON: Marie Curie | LOCATION: Paris

Sentence: "Elon Musk founded SpaceX in Hawthorne, California."
Entities: PERSON: Elon Musk | LOCATION: Hawthorne, California

Sentence: "Alan Turing worked at Bletchley Park during World War II."
Entities:"""

    # Expected output:
    # PERSON: Alan Turing | LOCATION: Bletchley Park

    print("  Few-shot NER prompt:")
    print(ner_prompt)
    print()
    print("  Expected output: PERSON: Alan Turing | LOCATION: Bletchley Park")
    print()
    print("  Key design principles:")
    print("  - Consistent format across all examples")
    print("  - Same entity types demonstrated in each example")
    print("  - Final item ends at the same structural point to guide completion")


# === Exercise 3: Paradigm Comparison Analysis ===
# Problem: Compare Traditional ML Pipeline and Foundation Model Pipeline
# across 5 dimensions with concrete examples.

def exercise_3():
    """Solution: Paradigm comparison table"""
    comparison = {
        "Data per task": {
            "Traditional ML": (
                "Each task needs its own labeled dataset "
                "(e.g., 10K labeled images for each classifier)"
            ),
            "Foundation Model": (
                "One massive pre-training corpus shared across all tasks "
                "(e.g., 1T tokens)"
            ),
        },
        "Training per task": {
            "Traditional ML": (
                "Full model trained from scratch or fine-tuned "
                "independently for each task"
            ),
            "Foundation Model": (
                "Single large pre-training run; lightweight adaptation per task"
            ),
        },
        "Knowledge transfer": {
            "Traditional ML": (
                "Limited -- a sentiment classifier doesn't help a "
                "translation model"
            ),
            "Foundation Model": (
                "Strong -- linguistic, factual, and reasoning knowledge "
                "transfers across tasks"
            ),
        },
        "Adaptation method": {
            "Traditional ML": (
                "Retrain / fine-tune with task-specific labeled data"
            ),
            "Foundation Model": (
                "Prompt engineering, few-shot examples, or "
                "parameter-efficient fine-tuning (LoRA)"
            ),
        },
        "Failure mode": {
            "Traditional ML": (
                "Data drift -- new tasks require expensive data collection"
            ),
            "Foundation Model": (
                "Hallucination -- model may generate plausible "
                "but incorrect outputs"
            ),
        },
    }

    header = f"  {'Dimension':<22} | {'Traditional ML':<45} | {'Foundation Model'}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for dim, vals in comparison.items():
        trad = vals["Traditional ML"][:45]
        fm = vals["Foundation Model"][:60]
        print(f"  {dim:<22} | {trad:<45} | {fm}")


# === Exercise 4: Emergent Capabilities Threshold Analysis ===
# Problem: Order these capabilities from earliest to latest emergence
# and explain why each requires more scale.

def exercise_4():
    """Solution: Order capabilities by emergence threshold"""
    capabilities = [
        {
            "name": "Simple 3-digit addition",
            "compute_threshold": "~10^22 FLOPs",
            "reason": (
                "Requires learning basic digit patterns and carry operations. "
                "Relatively few sub-skills to compose."
            ),
        },
        {
            "name": "Word unscrambling",
            "compute_threshold": "~10^22 FLOPs",
            "reason": (
                "Requires letter pattern recognition and vocabulary knowledge, "
                "but is a self-contained lexical task."
            ),
        },
        {
            "name": "Chain-of-Thought reasoning",
            "compute_threshold": "~10^23 FLOPs",
            "reason": (
                "Requires not just factual knowledge, but ability to decompose "
                "multi-step problems and maintain intermediate state."
            ),
        },
        {
            "name": "Theory of Mind",
            "compute_threshold": "~10^24 FLOPs",
            "reason": (
                "Requires modeling other agents' beliefs and intentions, "
                "involving nested reasoning ('Alice believes that Bob thinks...')."
            ),
        },
    ]

    print("  Emergence order (earliest -> latest):\n")
    for i, cap in enumerate(capabilities, 1):
        print(f"  {i}. {cap['name']} ({cap['compute_threshold']})")
        print(f"     {cap['reason']}")
        print()

    print(
        "  Pattern: later-emerging capabilities require composing more "
        "sub-skills\n  and maintaining more abstract internal state."
    )


# === Exercise 5: RLHF Pipeline Design ===
# Problem: For each stage of ChatGPT training, identify data required
# and what is learned.

def exercise_5():
    """Solution: RLHF pipeline stages"""
    stages = [
        {
            "stage": "1. Pre-training (GPT-3.5 base)",
            "data": "Massive web-scale text corpus (hundreds of billions of tokens)",
            "learned": (
                "Next-token prediction; broad linguistic, factual, "
                "and reasoning knowledge"
            ),
        },
        {
            "stage": "2. Supervised Fine-tuning (SFT)",
            "data": (
                "Human-written high-quality instruction-response pairs "
                "(thousands to tens of thousands)"
            ),
            "learned": (
                "To follow instructions and produce well-structured, "
                "helpful responses"
            ),
        },
        {
            "stage": "3. Reward Model Training",
            "data": (
                "Pairs of model responses ranked by human annotators "
                "(one preferred over another)"
            ),
            "learned": (
                "To predict human preference scores -- assigns a scalar "
                "reward to any model output"
            ),
        },
        {
            "stage": "4. RLHF with PPO",
            "data": (
                "The reward model (as environment) + SFT model "
                "(as starting policy)"
            ),
            "learned": (
                "To generate responses that maximize reward (human preference) "
                "while not drifting too far from SFT distribution (KL penalty)"
            ),
        },
    ]

    for s in stages:
        print(f"  {s['stage']}")
        print(f"    Data: {s['data']}")
        print(f"    Learned: {s['learned']}")
        print()

    print(
        "  Key insight: each stage refines a different aspect -- "
        "(1) raw knowledge,\n"
        "  (2) instruction format, (3) preference model, "
        "(4) alignment optimization."
    )


if __name__ == "__main__":
    print("=== Exercise 1: Foundation Model Classification ===")
    exercise_1()
    print("\n=== Exercise 2: In-context Learning Prompt Design ===")
    exercise_2()
    print("\n=== Exercise 3: Paradigm Comparison Analysis ===")
    exercise_3()
    print("\n=== Exercise 4: Emergent Capabilities Threshold Analysis ===")
    exercise_4()
    print("\n=== Exercise 5: RLHF Pipeline Design ===")
    exercise_5()
    print("\nAll exercises completed!")
