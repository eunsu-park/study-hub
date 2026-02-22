"""
LLMOps — Prompt Registry and Testing
=====================================
Demonstrates:
- Prompt version management (register, get, rollback)
- Prompt A/B testing framework
- Prompt test suite with assertions
- Cost-aware model routing

Run: python prompt_registry.py <example>
Available: registry, test, router, all
"""

import json
import hashlib
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path


# ── 1. Prompt Registry ─────────────────────────────────────────────

class PromptRegistry:
    """Version-controlled prompt template registry.

    Stores prompt templates with version history, metadata, and rollback.
    Uses file-based storage (production: use a database or API).
    """

    def __init__(self, registry_dir="prompts"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.registry_dir / "index.json"
        self._index = self._load_index()

    def _load_index(self):
        if self._index_path.exists():
            return json.loads(self._index_path.read_text())
        return {}

    def _save_index(self):
        self._index_path.write_text(json.dumps(self._index, indent=2))

    def register(self, name, template, metadata=None):
        """Register a new prompt version."""
        content_hash = hashlib.sha256(template.encode()).hexdigest()[:12]
        version = len(self._index.get(name, {}).get("versions", [])) + 1

        entry = {
            "version": version,
            "template": template,
            "hash": content_hash,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        # Save template file
        prompt_dir = self.registry_dir / name
        prompt_dir.mkdir(exist_ok=True)
        (prompt_dir / f"v{version}.txt").write_text(template)

        # Update index
        if name not in self._index:
            self._index[name] = {"active_version": 1, "versions": []}
        self._index[name]["versions"].append(entry)
        self._index[name]["active_version"] = version
        self._save_index()

        print(f"  Registered '{name}' v{version} (hash={content_hash})")
        return version

    def get(self, name, version=None):
        """Get a prompt template (default: active version)."""
        if name not in self._index:
            raise KeyError(f"Prompt '{name}' not found")
        if version is None:
            version = self._index[name]["active_version"]
        return self._index[name]["versions"][version - 1]["template"]

    def get_active_version(self, name):
        """Get the active version number."""
        return self._index[name]["active_version"]

    def rollback(self, name, version):
        """Set active version to a previous version."""
        if name not in self._index:
            raise KeyError(f"Prompt '{name}' not found")
        versions = self._index[name]["versions"]
        if version < 1 or version > len(versions):
            raise ValueError(f"Invalid version: {version} (max: {len(versions)})")
        self._index[name]["active_version"] = version
        self._save_index()
        print(f"  Rolled back '{name}' to v{version}")

    def list_versions(self, name):
        """List all versions of a prompt."""
        if name not in self._index:
            print(f"  Prompt '{name}' not found")
            return []
        active = self._index[name]["active_version"]
        versions = self._index[name]["versions"]
        for v in versions:
            marker = " ← active" if v["version"] == active else ""
            meta = v.get("metadata", {})
            note = f" ({meta.get('note', '')})" if meta.get("note") else ""
            print(f"  v{v['version']} [{v['created_at'][:10]}] "
                  f"hash={v['hash']}{note}{marker}")
        return versions

    def diff(self, name, v1, v2):
        """Show diff between two versions."""
        t1 = self.get(name, v1)
        t2 = self.get(name, v2)
        print(f"  --- v{v1}")
        print(f"  +++ v{v2}")
        lines1 = t1.splitlines()
        lines2 = t2.splitlines()
        for i, (l1, l2) in enumerate(zip(lines1, lines2)):
            if l1 != l2:
                print(f"  Line {i+1}:")
                print(f"  - {l1}")
                print(f"  + {l2}")


def demo_registry():
    """Demonstrate prompt registry usage."""
    import tempfile
    print("=" * 60)
    print("PROMPT REGISTRY DEMO")
    print("=" * 60)

    registry = PromptRegistry(registry_dir=tempfile.mkdtemp())

    # Register versions
    print("\nRegistering prompt versions:")
    registry.register(
        "summarizer",
        template="Summarize the following text in {num_sentences} sentences:\n\n{text}",
        metadata={"model": "claude-sonnet-4-20250514", "author": "team-a"},
    )
    registry.register(
        "summarizer",
        template=(
            "You are a concise summarizer. Given the text below, "
            "produce exactly {num_sentences} sentences that capture the key points.\n\n"
            "Text: {text}\n\n"
            "Summary:"
        ),
        metadata={"model": "claude-sonnet-4-20250514", "author": "team-a", "note": "added role + format"},
    )
    registry.register(
        "summarizer",
        template=(
            "You are an expert summarizer. Read the text carefully and produce "
            "a summary of exactly {num_sentences} sentences.\n\n"
            "Rules:\n"
            "- Each sentence must convey a distinct key point\n"
            "- Use clear, simple language\n"
            "- Do not add information not in the original text\n\n"
            "Text: {text}\n\n"
            "Summary:"
        ),
        metadata={"model": "claude-sonnet-4-20250514", "author": "team-a", "note": "added rules"},
    )

    # List versions
    print("\nAll versions:")
    registry.list_versions("summarizer")

    # Get active prompt
    print(f"\nActive prompt (v{registry.get_active_version('summarizer')}):")
    prompt = registry.get("summarizer")
    print(f"  {prompt[:80]}...")

    # Rollback
    print("\nRolling back to v1:")
    registry.rollback("summarizer", 1)
    registry.list_versions("summarizer")

    # Diff
    print("\nDiff v1 vs v2:")
    registry.diff("summarizer", 1, 2)


# ── 2. Prompt Testing ─────────────────────────────────────────────

class PromptTestSuite:
    """Run test cases against a prompt template."""

    def __init__(self, template, model_fn):
        """
        Args:
            template: Prompt template string with {placeholders}
            model_fn: Function(prompt_str) → response_str
        """
        self.template = template
        self.model_fn = model_fn

    def run_tests(self, test_cases):
        """Run test cases and report results.

        test_cases: list of {
            "name": str,
            "inputs": dict (template variables),
            "assertions": [
                {"type": "contains", "value": "..."},
                {"type": "not_contains", "value": "..."},
                {"type": "max_length", "value": int},
                {"type": "min_length", "value": int},
                {"type": "json_valid"},
            ]
        }
        """
        results = []
        for tc in test_cases:
            prompt = self.template.format(**tc["inputs"])
            response = self.model_fn(prompt)

            passed = True
            details = []
            for assertion in tc["assertions"]:
                ok = self._check_assertion(assertion, response)
                passed = passed and ok
                status = "PASS" if ok else "FAIL"
                details.append(f"{status}: {assertion['type']} "
                               f"{assertion.get('value', '')}")

            results.append({
                "name": tc["name"],
                "passed": passed,
                "details": details,
                "response_preview": response[:100],
            })

        # Report
        total = len(results)
        passed = sum(1 for r in results if r["passed"])
        print(f"\nPrompt Test Results: {passed}/{total} passed")
        for r in results:
            status = "PASS" if r["passed"] else "FAIL"
            print(f"  [{status}] {r['name']}")
            for d in r["details"]:
                print(f"         {d}")
        return results

    def _check_assertion(self, assertion, response):
        atype = assertion["type"]
        value = assertion.get("value")

        if atype == "contains":
            return value.lower() in response.lower()
        elif atype == "not_contains":
            return value.lower() not in response.lower()
        elif atype == "max_length":
            return len(response) <= value
        elif atype == "min_length":
            return len(response) >= value
        elif atype == "json_valid":
            try:
                json.loads(response)
                return True
            except json.JSONDecodeError:
                return False
        return True


def demo_test():
    """Demonstrate prompt testing."""
    print("=" * 60)
    print("PROMPT TESTING DEMO")
    print("=" * 60)

    # Mock model function (in production: actual LLM call)
    def mock_model(prompt):
        if "summarize" in prompt.lower():
            return "Python is a versatile programming language widely used in data science."
        return "I can help you with that."

    template = "Summarize the following text in {num_sentences} sentences:\n\n{text}"

    suite = PromptTestSuite(template, mock_model)
    suite.run_tests([
        {
            "name": "Basic summarization",
            "inputs": {
                "num_sentences": 1,
                "text": "Python is a programming language. It is used in data science. It was created by Guido.",
            },
            "assertions": [
                {"type": "contains", "value": "python"},
                {"type": "max_length", "value": 500},
                {"type": "min_length", "value": 10},
            ],
        },
        {
            "name": "No hallucination",
            "inputs": {
                "num_sentences": 1,
                "text": "The cat sat on the mat.",
            },
            "assertions": [
                {"type": "not_contains", "value": "dog"},
                {"type": "not_contains", "value": "elephant"},
            ],
        },
    ])


# ── 3. Model Router ───────────────────────────────────────────────

class ModelRouter:
    """Route queries to cost-appropriate models.

    Simple queries → cheap model (Haiku)
    Medium queries → balanced model (Sonnet)
    Complex queries → capable model (Opus)
    """

    PRICING = {
        "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0},
        "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
        "claude-opus-4-20250514": {"input": 15.0, "output": 75.0},
    }

    def __init__(self):
        self.model_map = {
            "simple": "claude-haiku-4-5-20251001",
            "medium": "claude-sonnet-4-20250514",
            "complex": "claude-opus-4-20250514",
        }
        self.stats = defaultdict(int)

    def classify(self, query):
        """Classify query complexity."""
        words = len(query.split())
        has_code = any(kw in query.lower() for kw in ["code", "function", "class", "implement"])
        has_analysis = any(kw in query.lower() for kw in ["analyze", "compare", "explain why", "evaluate"])

        if has_code or has_analysis:
            return "complex" if words > 50 else "medium"
        elif words < 20:
            return "simple"
        else:
            return "medium"

    def route(self, query):
        """Select model for query and return (model_name, complexity)."""
        complexity = self.classify(query)
        model = self.model_map[complexity]
        self.stats[complexity] += 1
        return model, complexity

    def estimate_cost(self, query, estimated_output_tokens=200):
        """Estimate cost for a query."""
        model, _ = self.route(query)
        input_tokens = len(query.split()) * 1.3  # rough estimate
        pricing = self.PRICING[model]
        cost = (input_tokens * pricing["input"] +
                estimated_output_tokens * pricing["output"]) / 1_000_000
        return cost, model

    def report(self):
        """Print routing statistics."""
        total = sum(self.stats.values())
        print(f"\nModel Router Stats ({total} queries):")
        for complexity, count in sorted(self.stats.items()):
            model = self.model_map[complexity]
            pct = count / total * 100 if total > 0 else 0
            print(f"  {complexity:8s} → {model:40s} {count:4d} ({pct:.1f}%)")


def demo_router():
    """Demonstrate model routing."""
    print("=" * 60)
    print("MODEL ROUTER DEMO")
    print("=" * 60)

    router = ModelRouter()

    queries = [
        "What is the capital of France?",
        "Explain the difference between supervised and unsupervised learning.",
        "Write a Python function that implements binary search with error handling "
        "and comprehensive docstrings.",
        "Hi",
        "Analyze the trade-offs between microservices and monolithic architecture "
        "for a high-traffic e-commerce platform, considering scalability, "
        "development velocity, and operational complexity.",
    ]

    print("\nRouting queries:")
    for q in queries:
        model, complexity = router.route(q)
        cost, _ = router.estimate_cost(q)
        print(f"\n  Query: \"{q[:60]}...\"" if len(q) > 60 else f"\n  Query: \"{q}\"")
        print(f"  → {complexity} → {model} (est. ${cost:.6f})")

    router.report()


# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    demos = {
        "registry": demo_registry,
        "test": demo_test,
        "router": demo_router,
    }

    if len(sys.argv) < 2 or sys.argv[1] not in demos and sys.argv[1] != "all":
        print("Usage: python prompt_registry.py <example>")
        print(f"Available: {', '.join(demos.keys())}, all")
    elif sys.argv[1] == "all":
        for fn in demos.values():
            fn()
            print()
    else:
        demos[sys.argv[1]]()
