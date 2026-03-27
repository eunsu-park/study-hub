"""
Exercises for Lesson 05: Data Curation
Topic: Foundation_Models

Solutions to practice problems from the lesson.
"""

import re
import hashlib
from collections import Counter


# === Exercise 1: Quality Filter Design ===
# Problem: For each document, explain which heuristic filter(s) would flag it.

def caps_ratio(text: str) -> float:
    """Calculate ratio of uppercase characters to all alphabetic characters."""
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return 0.0
    return sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)


def unique_lines_ratio(text: str) -> float:
    """Fraction of unique lines in text."""
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    if not lines:
        return 0.0
    return len(set(lines)) / len(lines)


def punctuation_ratio(text: str) -> float:
    """Ratio of punctuation characters to total characters."""
    if not text:
        return 0.0
    punct_count = sum(1 for c in text if c in "!@#$%^&*()_+-=[]{}|;':\",./<>?")
    return punct_count / len(text)


def quality_filter(text: str) -> dict:
    """Apply quality heuristics and return results."""
    return {
        "caps_ratio": caps_ratio(text),
        "unique_lines_ratio": unique_lines_ratio(text),
        "punctuation_ratio": punctuation_ratio(text),
        "avg_words_per_line": (
            sum(len(line.split()) for line in text.strip().split("\n"))
            / max(len(text.strip().split("\n")), 1)
        ),
    }


def exercise_1():
    """Solution: Quality filter design"""
    doc_a = """BUY NOW!!! CLICK HERE!!! LIMITED TIME OFFER!!!
BEST PRICES GUARANTEED!!! ACT FAST!!!
BUY BUY BUY!!! DISCOUNT DISCOUNT!!!"""

    doc_b = """a a a a a a a a a a a a a a a a a a a a a a
b b b b b b b b b b b b b b b b b b b b b b"""

    doc_c = """def sort_array(arr): return sorted(arr)
x=1;y=2;z=x+y;print(z);a=[];for i in range(10):a.append(i)"""

    documents = [("A (Spam)", doc_a), ("B (Repetitive)", doc_b), ("C (Code)", doc_c)]

    for name, doc in documents:
        scores = quality_filter(doc)
        print(f"  Document {name}:")
        print(f"    Text: {doc[:60]}...")
        print(f"    Caps ratio: {scores['caps_ratio']:.2f}", end="")
        print(" [FLAGGED]" if scores["caps_ratio"] > 0.3 else "")
        print(f"    Unique lines ratio: {scores['unique_lines_ratio']:.2f}", end="")
        print(" [FLAGGED]" if scores["unique_lines_ratio"] < 0.5 else "")
        print(f"    Punctuation ratio: {scores['punctuation_ratio']:.2f}", end="")
        print(" [FLAGGED]" if scores["punctuation_ratio"] > 0.15 else "")
        print(f"    Avg words/line: {scores['avg_words_per_line']:.1f}")

        # Analysis
        flags = []
        if scores["caps_ratio"] > 0.3:
            flags.append("high_caps")
        if scores["unique_lines_ratio"] < 0.5:
            flags.append("low_uniqueness")
        if scores["punctuation_ratio"] > 0.15:
            flags.append("high_punctuation")
        print(f"    Triggered filters: {flags if flags else 'none (context-dependent)'}")
        print()


# === Exercise 2: MinHash Deduplication ===
# Problem: Explain MinHash LSH algorithm and answer sub-questions.

def compute_shingles(text: str, k: int = 3) -> set:
    """Convert text to set of k-character shingles."""
    words = text.lower().split()
    shingles = set()
    for i in range(len(words) - k + 1):
        shingle = " ".join(words[i:i + k])
        shingles.add(shingle)
    return shingles


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute exact Jaccard similarity."""
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def minhash_signature(shingles: set, num_hashes: int = 100) -> list:
    """Compute MinHash signature using simulated hash functions."""
    signature = []
    for i in range(num_hashes):
        min_hash = float("inf")
        for shingle in shingles:
            h = int(hashlib.md5(f"{i}_{shingle}".encode()).hexdigest(), 16)
            min_hash = min(min_hash, h)
        signature.append(min_hash)
    return signature


def minhash_similarity(sig_a: list, sig_b: list) -> float:
    """Estimate Jaccard similarity from MinHash signatures."""
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def exercise_2():
    """Solution: MinHash deduplication"""
    print("  1. Exact vs Near-duplicate deduplication:")
    print("     Exact: Remove identical documents (same hash). Fast, misses variants.")
    print("     Near-dup: Remove highly similar but not identical docs. More comprehensive.")
    print()

    print("  2. Why exact hashing is insufficient:")
    print("     Web data has many near-duplicates (same article with minor edits,")
    print("     different timestamps, added bylines) that get different hashes.")
    print()

    # Demonstration
    doc1 = "The quick brown fox jumps over the lazy dog in the park"
    doc2 = "The quick brown fox jumps over the lazy dog in the garden"
    doc3 = "Machine learning is a subset of artificial intelligence"

    shingles1 = compute_shingles(doc1, k=3)
    shingles2 = compute_shingles(doc2, k=3)
    shingles3 = compute_shingles(doc3, k=3)

    exact_jac_12 = jaccard_similarity(shingles1, shingles2)
    exact_jac_13 = jaccard_similarity(shingles1, shingles3)

    sig1 = minhash_signature(shingles1, num_hashes=100)
    sig2 = minhash_signature(shingles2, num_hashes=100)
    sig3 = minhash_signature(shingles3, num_hashes=100)

    est_12 = minhash_similarity(sig1, sig2)
    est_13 = minhash_similarity(sig1, sig3)

    print("  3. Jaccard similarity and MinHash demonstration:")
    print(f"     Doc1: '{doc1}'")
    print(f"     Doc2: '{doc2}'")
    print(f"     Doc3: '{doc3}'")
    print()
    print(f"     Exact Jaccard(doc1, doc2) = {exact_jac_12:.3f}")
    print(f"     MinHash estimate(doc1, doc2) = {est_12:.3f}")
    print(f"     Exact Jaccard(doc1, doc3) = {exact_jac_13:.3f}")
    print(f"     MinHash estimate(doc1, doc3) = {est_13:.3f}")
    print()
    print("     MinHash approximates Jaccard well, enabling near-linear dedup at scale.")


# === Exercise 3: Data Mixing Strategy ===
# Problem: LLaMA 2 data mixture analysis.

def exercise_3():
    """Solution: Data mixing strategy analysis"""
    mixture = {
        "English web (CommonCrawl)": 67,
        "Code (GitHub)": 8,
        "Wikipedia": 4,
        "Books": 4,
        "ArXiv": 2,
        "StackExchange": 2,
        "Other": 13,
    }

    print("  LLaMA 2 data mixture:")
    for source, pct in mixture.items():
        bar = "#" * (pct // 2)
        print(f"    {source:<30} {pct:>3}% {bar}")
    print()

    print("  Q1: Why include code data?")
    print("    - Code improves logical and structured reasoning")
    print("    - Programming requires precise step-by-step thinking")
    print("    - Docstrings link natural language to formal specs")
    print("    - Empirically: models without code perform worse on reasoning")
    print()

    print("  Q2: Why upsample Wikipedia?")
    print("    - High-quality, factually verified, encyclopedic")
    print("    - Knowledge density orders of magnitude higher than raw web")
    print("    - Domain upsampling: standard curation technique")
    print()

    print("  Q3: Sources to upsample for math reasoning:")
    improved = {
        "ArXiv": (2, 8, "Contains proofs, derivations, mathematical exposition"),
        "StackExchange": (2, 5, "Step-by-step problem-solving with explanations"),
        "Code (math libs)": (8, 12, "Mathematical code forces precise reasoning"),
        "Synthetic math": (0, 5, "Generated math problems and solutions"),
    }
    print(f"    {'Source':<25} {'Current':<10} {'Proposed':<10} {'Reason'}")
    print("    " + "-" * 75)
    for source, (curr, prop, reason) in improved.items():
        print(f"    {source:<25} {curr:>3}%{'':<6} {prop:>3}%{'':<6} {reason}")


# === Exercise 4: Ethical Considerations ===
# Problem: Identify ethical concerns and mitigations for 3 data curation decisions.

def exercise_4():
    """Solution: Ethical considerations in data curation"""
    decisions = [
        {
            "decision": "Using all publicly available web text including social media",
            "concern": (
                "Personal information exposure (PII), privacy violation, "
                "amplification of harmful speech, data from users who did "
                "not consent to AI training use."
            ),
            "mitigation": (
                "Apply PII removal (regex + NER for names, emails, phone numbers), "
                "content safety classification, respect robots.txt and ToS, "
                "consider opt-out mechanisms."
            ),
        },
        {
            "decision": "Training on GitHub code without license filtering",
            "concern": (
                "Copyright infringement. Code under GPL, AGPL, or "
                "non-commercial licenses may prohibit commercial AI use. "
                "The GitHub Copilot lawsuit raised exactly this issue."
            ),
            "mitigation": (
                "Filter to permissive licenses only (MIT, Apache 2.0, BSD), "
                "document license composition, separate legal analysis for "
                "ambiguous cases."
            ),
        },
        {
            "decision": "Perplexity filtering with English-trained reference model",
            "concern": (
                "Systematic disadvantage to non-English languages. "
                "Non-English text gets artificially high perplexity, "
                "creating linguistically biased dataset."
            ),
            "mitigation": (
                "Use per-language perplexity thresholds calibrated to "
                "language-specific reference models. Language-ID-first "
                "filtering followed by per-language quality filtering."
            ),
        },
    ]

    for i, d in enumerate(decisions, 1):
        print(f"  Decision {i}: {d['decision']}")
        print(f"    Concern: {d['concern']}")
        print(f"    Mitigation: {d['mitigation']}")
        print()


if __name__ == "__main__":
    print("=== Exercise 1: Quality Filter Design ===")
    exercise_1()
    print("\n=== Exercise 2: MinHash Deduplication ===")
    exercise_2()
    print("\n=== Exercise 3: Data Mixing Strategy ===")
    exercise_3()
    print("\n=== Exercise 4: Ethical Considerations ===")
    exercise_4()
    print("\nAll exercises completed!")
