"""
20. LLM Evaluation Metrics Example

BLEU, ROUGE, BERTScore, and custom LLM-as-judge evaluation
"""

import re
import math
from collections import Counter
from typing import Callable

print("=" * 60)
print("LLM Evaluation Metrics")
print("=" * 60)


# ============================================
# 1. BLEU Score (simplified)
# ============================================
print("\n[1] BLEU Score")
print("-" * 40)


def tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"\w+", text.lower())


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Extract n-grams from token list."""
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def bleu_score(reference: str, candidate: str, max_n: int = 4) -> dict:
    """Calculate BLEU score (simplified, single reference)."""
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)

    if not cand_tokens:
        return {"bleu": 0.0, "brevity_penalty": 0.0, "precisions": []}

    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(ref_tokens) / len(cand_tokens)))

    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(ngrams(ref_tokens, n))
        cand_ngrams = Counter(ngrams(cand_tokens, n))

        # Clipped counts
        clipped = sum(min(cand_ngrams[ng], ref_ngrams.get(ng, 0))
                       for ng in cand_ngrams)
        total = sum(cand_ngrams.values())

        precision = clipped / total if total > 0 else 0.0
        precisions.append(precision)

    # Geometric mean of precisions (with smoothing)
    log_avg = sum(math.log(max(p, 1e-10)) for p in precisions) / max_n
    bleu = bp * math.exp(log_avg)

    return {"bleu": round(bleu, 4), "brevity_penalty": round(bp, 4),
            "precisions": [round(p, 4) for p in precisions]}


reference = "The cat sat on the mat and looked at the window"
candidate1 = "The cat sat on the mat and gazed at the window"
candidate2 = "A dog stood near the door"

print(f"Reference: '{reference}'")
print(f"Candidate 1: '{candidate1}'")
result1 = bleu_score(reference, candidate1)
print(f"  BLEU: {result1['bleu']}, BP: {result1['brevity_penalty']}")

print(f"Candidate 2: '{candidate2}'")
result2 = bleu_score(reference, candidate2)
print(f"  BLEU: {result2['bleu']}, BP: {result2['brevity_penalty']}")


# ============================================
# 2. ROUGE Score (simplified)
# ============================================
print("\n[2] ROUGE Score")
print("-" * 40)


def rouge_n(reference: str, candidate: str, n: int = 1) -> dict:
    """Calculate ROUGE-N (recall-oriented)."""
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)

    ref_ngrams_list = Counter(ngrams(ref_tokens, n))
    cand_ngrams_list = Counter(ngrams(cand_tokens, n))

    overlap = sum(min(ref_ngrams_list[ng], cand_ngrams_list.get(ng, 0))
                  for ng in ref_ngrams_list)

    ref_total = sum(ref_ngrams_list.values())
    cand_total = sum(cand_ngrams_list.values())

    recall = overlap / ref_total if ref_total > 0 else 0.0
    precision = overlap / cand_total if cand_total > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4)}


def rouge_l(reference: str, candidate: str) -> dict:
    """Calculate ROUGE-L using longest common subsequence."""
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)

    m, n = len(ref_tokens), len(cand_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == cand_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_length = dp[m][n]
    recall = lcs_length / m if m > 0 else 0.0
    precision = lcs_length / n if n > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "lcs_length": lcs_length}


print(f"ROUGE-1: {rouge_n(reference, candidate1, n=1)}")
print(f"ROUGE-2: {rouge_n(reference, candidate1, n=2)}")
print(f"ROUGE-L: {rouge_l(reference, candidate1)}")
print()
print(f"ROUGE-1 (poor): {rouge_n(reference, candidate2, n=1)}")
print(f"ROUGE-L (poor): {rouge_l(reference, candidate2)}")


# ============================================
# 3. Semantic Similarity (cosine)
# ============================================
print("\n[3] Cosine Similarity (bag-of-words)")
print("-" * 40)


def cosine_similarity_bow(text1: str, text2: str) -> float:
    """Cosine similarity using bag-of-words vectors."""
    tokens1 = Counter(tokenize(text1))
    tokens2 = Counter(tokenize(text2))

    all_words = set(tokens1.keys()) | set(tokens2.keys())

    dot_product = sum(tokens1.get(w, 0) * tokens2.get(w, 0) for w in all_words)
    norm1 = math.sqrt(sum(v ** 2 for v in tokens1.values()))
    norm2 = math.sqrt(sum(v ** 2 for v in tokens2.values()))

    if norm1 == 0 or norm2 == 0:
        return 0.0
    return round(dot_product / (norm1 * norm2), 4)


print(f"Similarity (close): {cosine_similarity_bow(reference, candidate1)}")
print(f"Similarity (far):   {cosine_similarity_bow(reference, candidate2)}")


# ============================================
# 4. LLM-as-Judge (simulated)
# ============================================
print("\n[4] LLM-as-Judge Framework")
print("-" * 40)


def simulated_llm_judge(question: str, answer: str, reference: str) -> dict:
    """Simulated LLM judge (in production, call actual LLM)."""
    # Simulate scoring based on token overlap
    answer_tokens = set(tokenize(answer))
    ref_tokens = set(tokenize(reference))
    overlap = len(answer_tokens & ref_tokens) / max(len(ref_tokens), 1)

    relevance = min(5, max(1, int(overlap * 5) + 1))
    accuracy = min(5, max(1, int(overlap * 4) + 1))
    completeness = min(5, max(1, int(overlap * 3) + 2))

    return {
        "relevance": relevance,
        "accuracy": accuracy,
        "completeness": completeness,
        "overall": round((relevance + accuracy + completeness) / 3, 1),
        "feedback": f"Answer covers {overlap:.0%} of reference content.",
    }


qa_pairs = [
    {
        "question": "What is a transformer model?",
        "answer": "A transformer is a neural network architecture using self-attention mechanisms for sequence processing.",
        "reference": "A transformer is a deep learning architecture that uses self-attention to process sequential data in parallel.",
    },
    {
        "question": "What is a transformer model?",
        "answer": "It is a type of car.",
        "reference": "A transformer is a deep learning architecture that uses self-attention to process sequential data in parallel.",
    },
]

for pair in qa_pairs:
    result = simulated_llm_judge(pair["question"], pair["answer"], pair["reference"])
    print(f"Q: {pair['question']}")
    print(f"A: {pair['answer'][:60]}...")
    print(f"  Scores: rel={result['relevance']}, acc={result['accuracy']}, "
          f"comp={result['completeness']}, overall={result['overall']}")
    print(f"  Feedback: {result['feedback']}")
    print()


# ============================================
# 5. Evaluation Pipeline
# ============================================
print("[5] Evaluation Pipeline")
print("-" * 40)


class EvaluationPipeline:
    """Run multiple metrics on a set of predictions."""

    def __init__(self):
        self.metrics: dict[str, Callable] = {}

    def add_metric(self, name: str, func: Callable):
        self.metrics[name] = func

    def evaluate(self, references: list[str], predictions: list[str]) -> dict:
        results = {name: [] for name in self.metrics}

        for ref, pred in zip(references, predictions):
            for name, func in self.metrics.items():
                score = func(ref, pred)
                results[name].append(score)

        # Aggregate
        summary = {}
        for name, scores in results.items():
            if isinstance(scores[0], dict):
                # Average each key
                keys = scores[0].keys()
                summary[name] = {
                    k: round(sum(s[k] for s in scores) / len(scores), 4)
                    for k in keys if isinstance(scores[0][k], (int, float))
                }
            else:
                summary[name] = round(sum(scores) / len(scores), 4)

        return summary


pipeline = EvaluationPipeline()
pipeline.add_metric("bleu", lambda ref, pred: bleu_score(ref, pred)["bleu"])
pipeline.add_metric("rouge_1", lambda ref, pred: rouge_n(ref, pred, n=1))
pipeline.add_metric("rouge_l", rouge_l)
pipeline.add_metric("cosine", cosine_similarity_bow)

refs = [
    "Machine learning is a subset of artificial intelligence",
    "Python is a popular programming language for data science",
]
preds = [
    "Machine learning is part of AI and artificial intelligence",
    "Python is widely used in data science and programming",
]

summary = pipeline.evaluate(refs, preds)
for metric, scores in summary.items():
    print(f"  {metric}: {scores}")

print("\n" + "=" * 60)
print("Evaluation Metrics example complete!")
print("=" * 60)
