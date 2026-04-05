# 11_prompt_optimization.py — Manual optimization loop, A/B comparison, prompt compression
#
# Run: python 11_prompt_optimization.py

import anthropic
import time
import textwrap

# ---------------------------------------------------------------------------
# 1. Define two prompt variants for A/B comparison
# ---------------------------------------------------------------------------
VARIANT_A = {
    "name": "Verbose (baseline)",
    "system": (
        "You are a helpful assistant. When the user asks a question, "
        "think carefully, provide a thorough and detailed answer, and "
        "make sure your response is accurate and well-structured."
    ),
    "template": "Please answer the following question:\n\n{question}",
}

VARIANT_B = {
    "name": "Compressed (optimized)",
    "system": "Answer accurately and concisely. Use bullet points when listing.",
    "template": "{question}",
}

# ---------------------------------------------------------------------------
# 2. Evaluation dataset — question + expected keywords
# ---------------------------------------------------------------------------
TEST_CASES = [
    {"question": "What are the three states of matter?",
     "required_keywords": ["solid", "liquid", "gas"]},
    {"question": "Name two benefits of version control.",
     "required_keywords": ["history", "collaborat"]},
    {"question": "What does HTTP stand for?",
     "required_keywords": ["hypertext", "transfer", "protocol"]},
]


# ---------------------------------------------------------------------------
# 3. Scoring helpers
# ---------------------------------------------------------------------------
def keyword_score(text: str, keywords: list[str]) -> float:
    lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in lower)
    return hits / len(keywords) if keywords else 1.0


def brevity_score(text: str, max_chars: int = 400) -> float:
    length = len(text)
    return 1.0 if length <= max_chars else max(0.0, 1.0 - (length - max_chars) / max_chars)


def composite_score(text: str, keywords: list[str]) -> dict:
    kw = keyword_score(text, keywords)
    br = brevity_score(text)
    return {"keyword": round(kw, 2), "brevity": round(br, 2), "combined": round(0.7 * kw + 0.3 * br, 2)}


# ---------------------------------------------------------------------------
# 4. Run a variant against all test cases
# ---------------------------------------------------------------------------
def evaluate_variant(client: anthropic.Anthropic, variant: dict, test_cases: list[dict]) -> dict:
    scores, total_tokens = [], 0
    start = time.time()
    for tc in test_cases:
        resp = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=300,
            system=variant["system"],
            messages=[{"role": "user", "content": variant["template"].format(question=tc["question"])}],
        )
        total_tokens += resp.usage.input_tokens + resp.usage.output_tokens
        scores.append(composite_score(resp.content[0].text, tc["required_keywords"]))

    avg = lambda key: round(sum(s[key] for s in scores) / len(scores), 3)
    return {"variant": variant["name"], "avg_keyword": avg("keyword"), "avg_brevity": avg("brevity"),
            "avg_combined": avg("combined"), "total_tokens": total_tokens,
            "elapsed_sec": round(time.time() - start, 2)}


# ---------------------------------------------------------------------------
# 5. Prompt compression utility
# ---------------------------------------------------------------------------
def compress_prompt(client: anthropic.Anthropic, original: str) -> str:
    """Use the LLM to compress a verbose prompt while preserving intent."""
    meta_prompt = textwrap.dedent("""\
        Rewrite the following system prompt to be as short as possible \
        while preserving all instructions. Remove filler. Use imperative voice. \
        Output ONLY the compressed prompt.

        Original:
        ---
        {original}
        ---""").format(original=original)
    resp = client.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=256,
        messages=[{"role": "user", "content": meta_prompt}],
    )
    return resp.content[0].text.strip()


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------
def main() -> None:
    client = anthropic.Anthropic()

    print("=" * 60)
    print("PROMPT A/B COMPARISON")
    print("=" * 60)

    results = []
    for variant in [VARIANT_A, VARIANT_B]:
        try:
            res = evaluate_variant(client, variant, TEST_CASES)
            results.append(res)
            print(f"\n--- {res['variant']} ---")
            print(f"  Keyword accuracy : {res['avg_keyword']}")
            print(f"  Brevity          : {res['avg_brevity']}")
            print(f"  Combined score   : {res['avg_combined']}")
            print(f"  Total tokens     : {res['total_tokens']}")
            print(f"  Latency          : {res['elapsed_sec']}s")
        except anthropic.APIError as exc:
            print(f"[API Error for {variant['name']}] {exc}")

    if len(results) == 2:
        winner = max(results, key=lambda r: r["avg_combined"])
        print(f"\nWinner: {winner['variant']} (combined={winner['avg_combined']})")

    # Prompt compression demo
    print("\n" + "=" * 60)
    print("PROMPT COMPRESSION DEMO")
    print("=" * 60)
    try:
        compressed = compress_prompt(client, VARIANT_A["system"])
        print(f"\nOriginal ({len(VARIANT_A['system'])} chars):\n  {VARIANT_A['system']}")
        print(f"\nCompressed ({len(compressed)} chars):\n  {compressed}")
        print(f"\nReduction: {1 - len(compressed) / len(VARIANT_A['system']):.0%}")
    except anthropic.APIError as exc:
        print(f"[API Error] {exc}")


if __name__ == "__main__":
    main()
