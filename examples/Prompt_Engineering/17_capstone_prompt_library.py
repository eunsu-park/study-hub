# 17_capstone_prompt_library.py — Mini prompt library with template loading, evaluation, and CLI
#
# Run: python 17_capstone_prompt_library.py
# Run: python 17_capstone_prompt_library.py list
# Run: python 17_capstone_prompt_library.py run summarizer --text "Your text here"
# Run: python 17_capstone_prompt_library.py eval summarizer

import anthropic
import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Prompt template data model
# ---------------------------------------------------------------------------
@dataclass
class PromptTemplate:
    name: str
    description: str
    system: str
    user_template: str
    variables: list[str]
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 512
    tags: list[str] = field(default_factory=list)

    def render_user(self, **kwargs) -> str:
        return self.user_template.format(**kwargs)


# ---------------------------------------------------------------------------
# 2. Built-in prompt library
# ---------------------------------------------------------------------------
LIBRARY: dict[str, PromptTemplate] = {}


def register(t: PromptTemplate) -> None:
    LIBRARY[t.name] = t


register(PromptTemplate(
    name="summarizer", description="Summarize text concisely",
    system="Summarize the following text in 2-3 sentences. Be accurate and concise.",
    user_template="{text}", variables=["text"], tags=["summarization"],
))
register(PromptTemplate(
    name="code_reviewer", description="Review code for bugs and improvements",
    system="You are a senior engineer. Review the code. List: 1) Bugs, 2) Security issues, 3) Improvements.",
    user_template="Language: {language}\n\n```\n{code}\n```",
    variables=["language", "code"], tags=["code", "review"],
))
register(PromptTemplate(
    name="email_drafter", description="Draft a professional email",
    system="Draft a professional email. Tone: {tone}. Keep it concise and actionable.",
    user_template="Subject: {subject}\nKey points: {points}",
    variables=["tone", "subject", "points"], tags=["writing"],
))
register(PromptTemplate(
    name="data_extractor", description="Extract structured JSON from text",
    system="Extract structured data. Output ONLY valid JSON matching the schema. Use null for missing.",
    user_template="Schema: {schema}\n\nText: {text}",
    variables=["schema", "text"], tags=["extraction", "json"],
))
register(PromptTemplate(
    name="explainer", description="Explain a concept at a specified level",
    system="Explain the concept as if speaking to a {audience}. Use analogies when helpful.",
    user_template="Concept: {concept}", variables=["audience", "concept"], tags=["education"],
))

# ---------------------------------------------------------------------------
# 3. Evaluation dataset (for the summarizer)
# ---------------------------------------------------------------------------
EVAL_CASES = [
    {"vars": {"text": (
        "Python 3.12 introduces performance improvements including a new optimizer "
        "that reduces memory by 10%. Type hints now support TypeVar defaults. "
        "The deprecated distutils module has been fully removed."
    )}, "keywords": ["performance", "type", "distutils"]},
    {"vars": {"text": (
        "The Mars Perseverance rover has collected 20 rock samples. NASA plans a "
        "sample-return mission in the 2030s. Initial analysis suggests some rocks "
        "may contain biosignatures."
    )}, "keywords": ["sample", "mars", "biosignature"]},
]

# ---------------------------------------------------------------------------
# 4. Core functions: run, evaluate, list, export
# ---------------------------------------------------------------------------
def run_prompt(client: anthropic.Anthropic, tpl: PromptTemplate, variables: dict) -> tuple[str, dict]:
    """Execute a prompt template and return (text, usage)."""
    sys_text = tpl.system.format(**{k: variables.get(k, f"{{{k}}}") for k in tpl.variables if f"{{{k}}}" in tpl.system})
    start = time.time()
    resp = client.messages.create(
        model=tpl.model, max_tokens=tpl.max_tokens, system=sys_text,
        messages=[{"role": "user", "content": tpl.render_user(**variables)}],
    )
    usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens,
             "latency_sec": round(time.time() - start, 2)}
    return resp.content[0].text, usage


def evaluate_prompt(client: anthropic.Anthropic, tpl: PromptTemplate, cases: list[dict]) -> dict:
    """Run eval cases and compute keyword-hit scores."""
    results = []
    for case in cases:
        text, usage = run_prompt(client, tpl, case["vars"])
        hits = sum(1 for kw in case["keywords"] if kw.lower() in text.lower())
        score = hits / len(case["keywords"]) if case["keywords"] else 1.0
        results.append({"score": round(score, 2), "tokens": usage["input_tokens"] + usage["output_tokens"]})
    avg_score = sum(r["score"] for r in results) / len(results)
    return {"template": tpl.name, "cases": len(results), "avg_score": round(avg_score, 3),
            "total_tokens": sum(r["tokens"] for r in results), "details": results}


def list_templates() -> None:
    print(f"{'Name':<18} {'Tags':<24} {'Description'}")
    print("-" * 66)
    for t in LIBRARY.values():
        print(f"{t.name:<18} {', '.join(t.tags):<24} {t.description}")


def export_library(path: str) -> None:
    data = {name: asdict(t) for name, t in LIBRARY.items()}
    Path(path).write_text(json.dumps(data, indent=2))
    print(f"Exported {len(data)} templates to {path}")


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prompt Library CLI")
    sub = p.add_subparsers(dest="command")
    sub.add_parser("list", help="List all templates")
    rp = sub.add_parser("run", help="Run a template")
    rp.add_argument("template"); rp.add_argument("--text"); rp.add_argument("--vars")
    ep = sub.add_parser("eval", help="Evaluate a template")
    ep.add_argument("template")
    xp = sub.add_parser("export", help="Export library to JSON")
    xp.add_argument("--output", default="prompt_library.json")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if not args.command:
        # Interactive demo
        print("=" * 60)
        print("CAPSTONE PROMPT LIBRARY")
        print("=" * 60)
        list_templates()
        client = anthropic.Anthropic()
        print("\n" + "-" * 60)
        print("DEMO: Running 'summarizer'")
        try:
            text, usage = run_prompt(client, LIBRARY["summarizer"], EVAL_CASES[0]["vars"])
            print(f"Output: {text}\nUsage : {usage}")
        except anthropic.APIError as exc:
            print(f"[API Error] {exc}")
        print("\n" + "-" * 60)
        print("DEMO: Evaluating 'summarizer'")
        try:
            print(json.dumps(evaluate_prompt(client, LIBRARY["summarizer"], EVAL_CASES), indent=2))
        except anthropic.APIError as exc:
            print(f"[API Error] {exc}")
        return

    if args.command == "list":
        list_templates()
    elif args.command == "run":
        if args.template not in LIBRARY:
            print(f"Unknown: {args.template}. Available: {', '.join(LIBRARY)}"); sys.exit(1)
        t = LIBRARY[args.template]
        variables = json.loads(args.vars) if args.vars else {"text": args.text} if args.text else None
        if not variables:
            print(f"Provide --text or --vars. Required: {t.variables}"); sys.exit(1)
        try:
            output, usage = run_prompt(anthropic.Anthropic(), t, variables)
            print(f"\n{output}\n\n--- Usage: {usage} ---")
        except anthropic.APIError as exc:
            print(f"[API Error] {exc}")
    elif args.command == "eval":
        if args.template != "summarizer":
            print("Eval cases only defined for 'summarizer' in this demo."); sys.exit(1)
        try:
            print(json.dumps(evaluate_prompt(anthropic.Anthropic(), LIBRARY["summarizer"], EVAL_CASES), indent=2))
        except anthropic.APIError as exc:
            print(f"[API Error] {exc}")
    elif args.command == "export":
        export_library(args.output)


if __name__ == "__main__":
    main()
