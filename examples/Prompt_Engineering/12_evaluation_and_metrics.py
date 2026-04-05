# 12_evaluation_and_metrics.py — Evaluation dataset creation, LLM-as-judge, scoring
#
# Run: python 12_evaluation_and_metrics.py

import anthropic
import json
import statistics

# ---------------------------------------------------------------------------
# 1. Evaluation dataset — each case has input, expected output, and rubric
# ---------------------------------------------------------------------------
EVAL_DATASET = [
    {
        "id": "sum-1",
        "input": (
            "Quantum computing leverages quantum-mechanical phenomena such as "
            "superposition and entanglement to process information. Unlike classical "
            "bits that are 0 or 1, qubits can exist in both states simultaneously. "
            "This enables quantum computers to solve certain problems exponentially "
            "faster, though they require extremely low temperatures to operate."
        ),
        "reference": (
            "Quantum computers use qubits that exploit superposition and "
            "entanglement, potentially solving some problems exponentially faster "
            "than classical computers, but need very low temperatures."
        ),
        "rubric": [
            "Mentions superposition or entanglement",
            "Mentions speed advantage over classical computing",
            "Mentions temperature or cooling requirement",
            "Is concise (under 60 words)",
        ],
    },
    {
        "id": "sum-2",
        "input": (
            "The Mediterranean diet emphasizes fruits, vegetables, whole grains, "
            "legumes, nuts, and olive oil. It limits red meat and processed foods. "
            "Studies show it reduces cardiovascular risk by up to 30% and may "
            "improve cognitive function in older adults."
        ),
        "reference": (
            "The Mediterranean diet focuses on plant-based foods and olive oil "
            "while limiting red meat. Research links it to 30% lower heart "
            "disease risk and potential cognitive benefits."
        ),
        "rubric": [
            "Mentions key food groups (plants, olive oil)",
            "Mentions cardiovascular / heart disease benefit",
            "Includes the 30% statistic",
            "Is concise (under 60 words)",
        ],
    },
]

# ---------------------------------------------------------------------------
# 2. Prompt under test
# ---------------------------------------------------------------------------
SUMMARIZE_SYSTEM = "Summarize the following text in 1-2 sentences. Be accurate and concise."


def generate_summary(client: anthropic.Anthropic, text: str) -> str:
    """Generate a summary using the prompt under test."""
    resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=200,
        system=SUMMARIZE_SYSTEM,
        messages=[{"role": "user", "content": text}],
    )
    return resp.content[0].text


# ---------------------------------------------------------------------------
# 3. LLM-as-Judge — scores the generated output against a rubric
# ---------------------------------------------------------------------------
JUDGE_PROMPT = """\
You are an impartial evaluator. Score the CANDIDATE summary against each \
rubric criterion on a 0-1 scale (0 = not met, 0.5 = partially met, 1 = fully met).

<source>{source}</source>
<reference>{reference}</reference>
<candidate>{candidate}</candidate>

Rubric criteria:
{rubric_list}

Respond ONLY with a JSON object:
{{"scores": [<score for criterion 1>, ...], "reasoning": "<brief explanation>"}}"""


def judge_summary(client: anthropic.Anthropic, case: dict, candidate: str) -> dict:
    """Use a second LLM call to evaluate the candidate summary."""
    rubric_list = "\n".join(f"{i+1}. {r}" for i, r in enumerate(case["rubric"]))
    prompt = JUDGE_PROMPT.format(
        source=case["input"], reference=case["reference"],
        candidate=candidate, rubric_list=rubric_list,
    )
    resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text
    try:
        return json.loads(text[text.index("{"):text.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return {"scores": [], "reasoning": text}


# ---------------------------------------------------------------------------
# 4. Aggregate metrics
# ---------------------------------------------------------------------------
def compute_metrics(all_scores: list[list[float]]) -> dict:
    """Compute mean, min, and per-criterion averages."""
    flat = [s for row in all_scores for s in row]
    n_criteria = len(all_scores[0]) if all_scores else 0
    per_criterion = []
    for i in range(n_criteria):
        col = [row[i] for row in all_scores if i < len(row)]
        per_criterion.append(round(statistics.mean(col), 2) if col else 0)
    return {
        "overall_mean": round(statistics.mean(flat), 3) if flat else 0,
        "overall_min": min(flat) if flat else 0,
        "per_criterion_avg": per_criterion,
        "n_cases": len(all_scores),
    }


# ---------------------------------------------------------------------------
# 5. Main — run evaluation pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    client = anthropic.Anthropic()

    print("=" * 60)
    print("PROMPT EVALUATION PIPELINE")
    print(f"Prompt: \"{SUMMARIZE_SYSTEM}\"")
    print(f"Dataset: {len(EVAL_DATASET)} cases")
    print("=" * 60)

    all_scores = []
    for case in EVAL_DATASET:
        print(f"\n--- Case {case['id']} ---")
        try:
            candidate = generate_summary(client, case["input"])
            print(f"  Candidate: {candidate[:120]}...")
            verdict = judge_summary(client, case, candidate)
            scores = verdict.get("scores", [])
            all_scores.append(scores)
            print(f"  Scores   : {scores}")
            print(f"  Reasoning: {verdict.get('reasoning', '')[:120]}")
        except anthropic.APIError as exc:
            print(f"  [API Error] {exc}")
            all_scores.append([0] * len(case["rubric"]))

    # Aggregate report
    metrics = compute_metrics(all_scores)
    print("\n" + "=" * 60)
    print("AGGREGATE METRICS")
    print("=" * 60)
    print(f"  Cases evaluated    : {metrics['n_cases']}")
    print(f"  Overall mean score : {metrics['overall_mean']}")
    print(f"  Overall min score  : {metrics['overall_min']}")
    print(f"  Per-criterion avg  : {metrics['per_criterion_avg']}")

    threshold = 0.75
    passed = metrics["overall_mean"] >= threshold
    print(f"\n  Quality gate ({threshold}): [{'PASS' if passed else 'FAIL'}]")


if __name__ == "__main__":
    main()
