# Exercise: Lesson 15 — Prompt Management
# Complete the TODO items below.
#
# Run: python 15_prompt_management.py

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


# === Exercise 1: Prompt Template System ===
# Build a template engine that supports variable substitution.

@dataclass
class PromptTemplate:
    name: str
    template: str
    required_vars: list[str]
    defaults: dict[str, str] = field(default_factory=dict)

    def render(self, **kwargs: str) -> str:
        """Render the template with the given variables.

        Hint: Merge defaults with kwargs (kwargs take priority).
        Check that all required_vars are present.
        Use str.format_map() for substitution.
        Raise ValueError if a required variable is missing.
        """
        # TODO: Merge defaults with kwargs
        # TODO: Validate all required_vars are present
        # TODO: Render and return the template string
        pass

    def get_variables(self) -> list[str]:
        """Return all variable names found in the template.

        Hint: Use a regex or str.Formatter().parse() to find
        all {variable_name} placeholders.
        """
        # TODO: Extract and return variable names from the template
        pass


def exercise_1():
    """Verify the template system works."""
    tmpl = PromptTemplate(
        name="summarizer",
        template="Summarize the following {doc_type} in {style} style:\n\n{text}",
        required_vars=["text"],
        defaults={"doc_type": "document", "style": "concise"},
    )

    rendered = tmpl.render(text="Hello world content here.")
    assert rendered is not None, "Must return a string"
    assert "Hello world content here." in rendered
    assert "document" in rendered, "Default should be applied"
    assert "concise" in rendered, "Default should be applied"

    rendered2 = tmpl.render(text="Test", style="verbose")
    assert "verbose" in rendered2, "Explicit value should override default"

    variables = tmpl.get_variables()
    assert "text" in variables, "Must find 'text' variable"
    assert len(variables) >= 3, "Must find all 3 variables"

    try:
        tmpl2 = PromptTemplate("strict", "Hello {name}", ["name"])
        tmpl2.render()
        assert False, "Should raise ValueError for missing required var"
    except ValueError:
        pass

    print(f"  Template variables: {variables}")
    print(f"  Rendered length: {len(rendered)} chars")


# === Exercise 2: Prompt Version Control ===
# Track prompt versions with content hashing and change history.

@dataclass
class PromptVersion:
    version: int
    content: str
    content_hash: str
    timestamp: str
    changelog: str


class PromptVersionStore:
    """A simple version control system for prompts."""

    def __init__(self, name: str):
        self.name = name
        self.versions: list[PromptVersion] = []

    def _compute_hash(self, content: str) -> str:
        """Return the SHA-256 hex digest of content (first 12 chars).

        Hint: Use hashlib.sha256.
        """
        # TODO: Compute and return truncated hash
        pass

    def add_version(self, content: str, changelog: str) -> PromptVersion:
        """Add a new version if content has changed.

        Hint: Compare content_hash with the latest version.
        If identical, raise ValueError('No changes detected').
        Version numbers start at 1 and auto-increment.
        """
        # TODO: Compute hash
        # TODO: Check for duplicate content
        # TODO: Create and store new PromptVersion
        pass

    def get_version(self, version: int) -> PromptVersion:
        """Retrieve a specific version by number.

        Hint: Raise KeyError if version not found.
        """
        # TODO: Look up and return the requested version
        pass

    def get_latest(self) -> PromptVersion | None:
        """Return the latest version, or None if empty."""
        # TODO: Return last version or None
        pass

    def diff(self, v1: int, v2: int) -> dict:
        """Compare two versions and return a diff summary.

        Return: {"v1": v1, "v2": v2, "v1_length": int, "v2_length": int,
                 "length_delta": int, "same_content": bool}
        """
        # TODO: Retrieve both versions and compute diff summary
        pass


def exercise_2():
    """Verify version control operations."""
    store = PromptVersionStore("my_prompt")

    v1 = store.add_version("Summarize: {text}", "Initial version")
    assert v1.version == 1
    assert len(v1.content_hash) == 12

    v2 = store.add_version("Summarize concisely: {text}", "Added concisely")
    assert v2.version == 2

    try:
        store.add_version("Summarize concisely: {text}", "Duplicate")
        assert False, "Should reject duplicate content"
    except ValueError:
        pass

    latest = store.get_latest()
    assert latest.version == 2

    d = store.diff(1, 2)
    assert d["same_content"] is False
    assert d["length_delta"] != 0

    print(f"  Version 1 hash: {v1.content_hash}")
    print(f"  Version 2 hash: {v2.content_hash}")
    print(f"  Diff: length delta = {d['length_delta']:+d} chars")


# === Exercise 3: Prompt Registry ===
# A registry that organizes templates by name and category.

class PromptRegistry:
    """Central registry for managing prompt templates."""

    def __init__(self):
        self._templates: dict[str, dict[str, Any]] = {}

    def register(
        self, name: str, template: PromptTemplate, category: str = "general",
    ) -> None:
        """Register a template under a name and category.

        Hint: Store both the template and category. Raise ValueError
        if name is already registered.
        """
        # TODO: Check for duplicate name
        # TODO: Store the template with its category
        pass

    def get(self, name: str) -> PromptTemplate:
        """Retrieve a template by name. Raise KeyError if not found."""
        # TODO: Look up and return the template
        pass

    def list_by_category(self, category: str) -> list[str]:
        """Return template names filtered by category."""
        # TODO: Filter and return matching template names
        pass

    def search(self, keyword: str) -> list[str]:
        """Search templates by keyword in name or template text.

        Hint: Case-insensitive substring search.
        """
        # TODO: Search names and template content for keyword
        pass


def exercise_3():
    """Verify registry operations."""
    reg = PromptRegistry()

    reg.register("summarize", PromptTemplate(
        "summarize", "Summarize: {text}", ["text"]
    ), category="generation")

    reg.register("classify", PromptTemplate(
        "classify", "Classify this text: {text}", ["text"]
    ), category="analysis")

    reg.register("extract", PromptTemplate(
        "extract", "Extract entities from: {text}", ["text"]
    ), category="analysis")

    tmpl = reg.get("summarize")
    assert tmpl.name == "summarize"

    analysis = reg.list_by_category("analysis")
    assert len(analysis) == 2

    results = reg.search("text")
    assert len(results) >= 3, "All templates contain 'text'"

    results2 = reg.search("entities")
    assert "extract" in results2

    print(f"  Registered templates: {len(reg._templates)}")
    print(f"  Analysis category: {analysis}")
    print(f"  Search 'entities': {results2}")


if __name__ == "__main__":
    print("=== Exercise 1: Prompt Templates ===")
    exercise_1()

    print("=== Exercise 2: Version Control ===")
    exercise_2()

    print("=== Exercise 3: Prompt Registry ===")
    exercise_3()

    print("\nAll exercises completed!")
