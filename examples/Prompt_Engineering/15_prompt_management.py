# 15_prompt_management.py — Prompt versioning, Jinja2 templating, registry pattern
#
# Run: python 15_prompt_management.py
# Requires: pip install jinja2

import anthropic
import json
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict

try:
    from jinja2 import Environment, BaseLoader
except ImportError:
    raise SystemExit("Install jinja2: pip install jinja2")

# ---------------------------------------------------------------------------
# 1. PromptVersion — immutable snapshot of a prompt
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PromptVersion:
    name: str
    version: str
    system_template: str
    user_template: str
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 512
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def fingerprint(self) -> str:
        """SHA-256 hash of prompt content for change detection."""
        content = f"{self.system_template}|{self.user_template}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# 2. PromptRegistry — stores and retrieves versioned prompts
# ---------------------------------------------------------------------------
class PromptRegistry:
    def __init__(self):
        self._store: dict[str, list[PromptVersion]] = {}

    def register(self, prompt: PromptVersion) -> None:
        self._store.setdefault(prompt.name, []).append(prompt)

    def get(self, name: str, version: str | None = None) -> PromptVersion:
        versions = self._store.get(name)
        if not versions:
            raise KeyError(f"Prompt '{name}' not found")
        if version:
            for v in versions:
                if v.version == version:
                    return v
            raise KeyError(f"Version '{version}' not found for '{name}'")
        return versions[-1]

    def list_all(self) -> dict[str, list[str]]:
        return {k: [v.version for v in vs] for k, vs in self._store.items()}


# ---------------------------------------------------------------------------
# 3. Jinja2-based template renderer
# ---------------------------------------------------------------------------
_JINJA_ENV = Environment(loader=BaseLoader(), keep_trailing_newline=True)


def render_prompt(prompt: PromptVersion, variables: dict) -> tuple[str, str]:
    """Render system and user templates with Jinja2."""
    sys_tpl = _JINJA_ENV.from_string(prompt.system_template)
    usr_tpl = _JINJA_ENV.from_string(prompt.user_template)
    return sys_tpl.render(**variables), usr_tpl.render(**variables)


def run_prompt(client: anthropic.Anthropic, prompt: PromptVersion, variables: dict) -> str:
    """Execute a rendered prompt against the API."""
    system_text, user_text = render_prompt(prompt, variables)
    resp = client.messages.create(
        model=prompt.model, max_tokens=prompt.max_tokens,
        system=system_text,
        messages=[{"role": "user", "content": user_text}],
    )
    return resp.content[0].text


# ---------------------------------------------------------------------------
# 4. Define prompt versions (demonstrating evolution)
# ---------------------------------------------------------------------------
SUMMARIZER_V1 = PromptVersion(
    name="summarizer", version="1.0",
    system_template="Summarize the text provided by the user.",
    user_template="{{ text }}",
)

SUMMARIZER_V2 = PromptVersion(
    name="summarizer", version="2.0",
    system_template=(
        "You are a {{ audience }} summarizer. "
        "Summarize in {{ style }} style. Max {{ max_words }} words."
    ),
    user_template="Text to summarize:\n\n{{ text }}",
)

QA_V1 = PromptVersion(
    name="qa_bot", version="1.0",
    system_template="You are a {{ domain }} expert. Answer using only the provided context.",
    user_template="<context>\n{{ context }}\n</context>\n\nQuestion: {{ question }}",
)

SAMPLE_TEXT = (
    "The James Webb Space Telescope, launched in December 2021, has captured "
    "unprecedented images of deep space. Its infrared sensors can peer through "
    "cosmic dust to reveal forming stars and ancient galaxies."
)


# ---------------------------------------------------------------------------
# 5. Main — demonstrate the prompt management workflow
# ---------------------------------------------------------------------------
def main() -> None:
    client = anthropic.Anthropic()

    # Build registry
    print("=" * 60)
    print("PROMPT REGISTRY")
    print("=" * 60)
    registry = PromptRegistry()
    for p in [SUMMARIZER_V1, SUMMARIZER_V2, QA_V1]:
        registry.register(p)
        print(f"  Registered: {p.name} v{p.version} [{p.fingerprint}]")
    print(f"\n  All prompts: {json.dumps(registry.list_all())}")

    # Run v1
    print("\n" + "=" * 60)
    print("RUN: summarizer v1.0")
    print("=" * 60)
    try:
        out = run_prompt(client, registry.get("summarizer", "1.0"), {"text": SAMPLE_TEXT})
        print(f"  Output: {out[:200]}")
    except anthropic.APIError as exc:
        print(f"  [API Error] {exc}")

    # Run v2 with parameters
    print("\n" + "=" * 60)
    print("RUN: summarizer v2.0 (parameterized)")
    print("=" * 60)
    try:
        out = run_prompt(client, registry.get("summarizer", "2.0"), {
            "text": SAMPLE_TEXT, "audience": "technical",
            "style": "bullet-point", "max_words": 50,
        })
        print(f"  Output: {out[:200]}")
    except anthropic.APIError as exc:
        print(f"  [API Error] {exc}")

    # Fingerprint comparison
    print("\n" + "=" * 60)
    print("VERSION COMPARISON")
    print("=" * 60)
    v1, v2 = registry.get("summarizer", "1.0"), registry.get("summarizer", "2.0")
    print(f"  v1.0 fingerprint: {v1.fingerprint}")
    print(f"  v2.0 fingerprint: {v2.fingerprint}")
    print(f"  Same content?   : {v1.fingerprint == v2.fingerprint}")

    # Export as JSON
    print("\n" + "=" * 60)
    print("EXPORT (JSON-serializable)")
    print("=" * 60)
    print(json.dumps(asdict(v2), indent=2))


if __name__ == "__main__":
    main()
