"""
Exercises for Lesson 21: Continued Pre-training
Topic: Foundation_Models

Solutions to practice problems from the lesson.
"""


# === Exercise 1: Continued Pre-training vs Instruction Tuning ===
# Problem: Explain why a biomedical AI company needs BOTH continued pre-training
# and instruction tuning, and describe the ideal pipeline order.

def exercise_1():
    """Solution: CPT vs instruction tuning pipeline"""
    print("  Why instruction tuning alone is insufficient:")
    print("    Base model (LLaMA-7B) knows little about drug interactions.")
    print("    5K QA pairs teach format, but cannot inject vast medical knowledge.")
    print("    Model learns pattern-matching while hallucinating actual content.")
    print()

    print("  Why continued pre-training alone is insufficient:")
    print("    After pre-training on PubMed, model understands bio language")
    print("    but behaves as text completion engine. Cannot follow instructions,")
    print("    answer concisely, or format responses for clinical use.")
    print()

    print("  Ideal pipeline:")
    print("    Base LLM (LLaMA-7B)")
    print("      |")
    print("    [Phase 1] Continued Pre-training")
    print("      Data: 50GB PubMed + clinical trials")
    print("      Objective: Causal LM, LR: 5e-6 (1/5 of base)")
    print("      Goal: Inject pharmacological knowledge")
    print("      |")
    print("    Domain-adapted LLM")
    print("      |")
    print("    [Phase 2] Supervised Fine-tuning (SFT)")
    print("      Data: 5,000 clinical QA pairs")
    print("      Objective: Instruction following, LR: 1e-5")
    print("      Goal: Format responses for clinical use")
    print("      |")
    print("    Medical Instruction-Following LLM")
    print("      |")
    print("    [Optional Phase 3] RLHF/DPO")
    print("      Data: Clinical expert preferences")
    print("      Goal: Align with clinical best practices")


# === Exercise 2: Catastrophic Forgetting Data Mixing ===
# Problem: Design data mixing strategy to recover lost capabilities.

def exercise_2():
    """Solution: Catastrophic forgetting data mixing"""
    print("  Observed degradation after medical pre-training:")
    print("    Medical QA: +25%  (improved)")
    print("    BBH reasoning: -18% (degraded)")
    print("    Coding (HumanEval): -31% (severely degraded)")
    print()

    print("  Root cause: 50GB medical dataset contained nearly zero code")
    print("  or mathematical reasoning. Representations overwritten.")
    print()

    mixing = {
        "medical": (0.55, "Primary target (reduced from 100%)"),
        "code": (0.20, "Recovery: matches degradation severity (-31%)"),
        "general_text": (0.15, "General reasoning recovery (-18%)"),
        "math": (0.10, "Supports reasoning capabilities"),
    }

    print("  Data mixing strategy:")
    for source, (pct, reason) in mixing.items():
        bar = "#" * int(pct * 40)
        print(f"    {source:<15} {pct:>4.0%} {bar} ({reason})")

    print()
    print("  Key principles:")
    print("    1. Proportional recovery: code 20% > reasoning 15% (more degraded)")
    print("    2. Medical still dominant (55%): maintains primary goal")
    print("    3. Diverse mixing: prevents over-specializing in any replacement domain")
    print("    4. Monitor all 3 benchmarks during training")
    print()
    print("  Expected outcome:")
    print("    Medical QA: +20% (slight reduction but still substantial)")
    print("    Coding: -5% (vs -31% without mixing)")
    print("    Reasoning: -3% (vs -18% without mixing)")


# === Exercise 3: Learning Rate and Data Order Strategy ===
# Problem: Explain low LR rationale and curriculum learning for financial data.

def exercise_3():
    """Solution: Learning rate and curriculum strategy"""
    print("  Lower learning rate rationale:")
    print("    Original pre-training: random init, high LR needed")
    print("    Continued pre-training: well-calibrated weights already exist")
    print("    High LR -> large updates -> catastrophic forgetting")
    print("    Low LR (5e-6 vs 3e-4) ensures:")
    print("      - New knowledge integrated ON TOP of existing representations")
    print("      - Small, conservative adjustments (not overwrites)")
    print("      - Loss surface explored cautiously around existing good minimum")
    print("    Use 3-5% warm-up even with low LR (initial domain gradients noisy)")
    print()

    print("  Curriculum learning for financial news:")
    phases = [
        ("Phase 1 (10%)", "High-quality, clean text",
         "Well-edited 10-K/10-Q filings. Establishes domain vocabulary."),
        ("Phase 2 (40%)", "Structured domain data",
         "Analyst reports, earnings call transcripts. Builds financial reasoning."),
        ("Phase 3 (40%)", "Diverse domain data",
         "News articles, press releases, commentary. Broad coverage."),
        ("Phase 4 (10%)", "Noisy/social data",
         "Reddit/Twitter financial discussions. Slang, abbreviations."),
    ]

    for name, desc, detail in phases:
        print(f"    {name}: {desc}")
        print(f"      {detail}")

    print()
    print("  Rationale: Starting with formal text establishes clean foundations.")
    print("  Adding noisy data later means stable base for informal mappings.")
    print("  Use cosine LR decay with optional warm restarts at phase boundaries.")


# === Exercise 4: Continued Pre-training Evaluation ===
# Problem: Design evaluation suite for legal domain adaptation.

def exercise_4():
    """Solution: CPT evaluation suite design"""
    print("  A) Metrics to measure:")
    print()
    print("  Domain-specific:")
    print("    1. Legal NER F1: entity recognition for legal entities")
    print("    2. Legal QA accuracy: CUAD dataset (contract clauses)")
    print("    3. Legal document perplexity: held-out legal text")
    print("    4. Citation format accuracy: Bluebook-style citations")
    print()
    print("  General capability (regression tests):")
    print("    5. BIG-Bench Hard (BBH): 23 reasoning tasks")
    print("    6. MMLU: 57-subject knowledge benchmark")
    print("    7. HellaSwag: commonsense reasoning")
    print()

    print("  B) Datasets:")
    datasets = {
        "legal_ner": "LexNER or MultiLegalPile NER annotations",
        "legal_qa": "CUAD (Contract Understanding Atticus Dataset)",
        "legal_perplexity": "Held-out 10% of legal pretraining data",
        "bbh": "BIG-Bench Hard - 6,511 examples",
        "mmlu": "MMLU - 14,000+ examples, 57 subjects",
        "hellaswag": "HellaSwag - 10,000 validation examples",
    }
    for key, desc in datasets.items():
        print(f"    {key}: {desc}")
    print()

    print("  C) Success thresholds:")
    thresholds = [
        ("Legal QA accuracy", "~25%", ">=60%", "Domain goal"),
        ("Legal perplexity", "~40", "<=15", "Language quality"),
        ("BBH accuracy", "baseline", ">= baseline - 5%", "Regression tolerance"),
        ("MMLU accuracy", "baseline", ">= baseline - 5%", "Regression tolerance"),
        ("HellaSwag", "baseline", ">= baseline - 3%", "Regression tolerance"),
    ]
    print(f"    {'Metric':<20} {'Pre-CPT':<12} {'Target':<20} {'Category'}")
    print("    " + "-" * 65)
    for metric, pre, target, cat in thresholds:
        print(f"    {metric:<20} {pre:<12} {target:<20} {cat}")
    print()
    print("    Success: ALL domain goals met AND all regression tests pass.")
    print("    If any general capability drops >8%, rerun with more replay data.")


if __name__ == "__main__":
    print("=== Exercise 1: CPT vs Instruction Tuning ===")
    exercise_1()
    print("\n=== Exercise 2: Catastrophic Forgetting Data Mixing ===")
    exercise_2()
    print("\n=== Exercise 3: Learning Rate and Curriculum ===")
    exercise_3()
    print("\n=== Exercise 4: CPT Evaluation Suite ===")
    exercise_4()
    print("\nAll exercises completed!")
