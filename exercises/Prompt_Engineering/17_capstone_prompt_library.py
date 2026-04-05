# Exercise: Lesson 17 — Capstone Prompt Library
# Complete the TODO items below.
#
# Run: python 17_capstone_prompt_library.py

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


# === Exercise 1: Prompt Entry Model ===
# Define the core data model for a prompt library entry.

@dataclass
class PromptEntry:
    """A single entry in the prompt library.

    Fields:
        name: Unique identifier (e.g., 'rag_with_citations')
        category: Category tag (e.g., 'rag', 'agent', 'extraction')
        system_prompt: The system prompt text
        user_template: User message template with {placeholders}
        variables: List of variable names in the template
        version: Integer version number
        tags: List of searchable tags
        created_at: ISO timestamp
    """

    name: str
    category: str
    system_prompt: str
    user_template: str
    variables: list[str] = field(default_factory=list)
    version: int = 1
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def render(self, **kwargs: str) -> dict[str, str]:
        """Render the system and user prompts with variable substitution.

        Returns: {"system": rendered_system, "user": rendered_user}

        Hint: Use str.format_map(). Raise ValueError if required
        variables are missing from kwargs.
        """
        # TODO: Check that all self.variables are present in kwargs
        # TODO: Render user_template with kwargs
        # TODO: Return dict with system and user prompts
        pass

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entry to a JSON-compatible dict.

        Hint: Include all fields. This should be invertible.
        """
        # TODO: Return a dict of all fields
        pass


def exercise_1():
    """Verify the PromptEntry model works."""
    entry = PromptEntry(
        name="summarizer_v1",
        category="generation",
        system_prompt="You are a concise summarizer.",
        user_template="Summarize this {doc_type}:\n\n{text}",
        variables=["doc_type", "text"],
        tags=["summarization", "concise"],
    )

    rendered = entry.render(doc_type="article", text="Hello world.")
    assert rendered is not None, "Must return a dict"
    assert "article" in rendered["user"], "Must substitute doc_type"
    assert "Hello world." in rendered["user"], "Must substitute text"

    d = entry.to_dict()
    assert d["name"] == "summarizer_v1"
    assert "tags" in d

    try:
        entry.render(doc_type="report")
        assert False, "Should raise ValueError for missing 'text'"
    except ValueError:
        pass

    print(f"  Entry: {entry.name} v{entry.version}")
    print(f"  Variables: {entry.variables}")
    print(f"  Rendered user prompt: {len(rendered['user'])} chars")


# === Exercise 2: Prompt Library with CRUD ===
# Build a library that stores, retrieves, and searches entries.

class PromptLibrary:
    """In-memory prompt library with CRUD and search."""

    def __init__(self):
        self._entries: dict[str, PromptEntry] = {}

    def add(self, entry: PromptEntry) -> None:
        """Add an entry. Raise ValueError if name already exists.

        Hint: Use entry.name as the key.
        """
        # TODO: Check for duplicate
        # TODO: Store the entry
        pass

    def get(self, name: str) -> PromptEntry:
        """Retrieve an entry by name. Raise KeyError if not found."""
        # TODO: Look up and return
        pass

    def update(self, name: str, **updates: Any) -> PromptEntry:
        """Update fields of an existing entry and bump version.

        Hint: Only update fields that are provided in **updates.
        Auto-increment the version number.
        Raise KeyError if name not found.
        """
        # TODO: Retrieve existing entry
        # TODO: Apply updates to matching fields
        # TODO: Increment version
        # TODO: Return the updated entry
        pass

    def delete(self, name: str) -> None:
        """Delete an entry by name. Raise KeyError if not found."""
        # TODO: Remove the entry
        pass

    def search(self, query: str) -> list[PromptEntry]:
        """Search entries by keyword in name, category, or tags.

        Hint: Case-insensitive substring matching.
        """
        # TODO: Search across name, category, and tags
        pass

    def list_by_category(self, category: str) -> list[PromptEntry]:
        """Return all entries in a given category."""
        # TODO: Filter by category
        pass

    def export_json(self) -> str:
        """Export the entire library as a JSON string.

        Hint: Use each entry's to_dict() method.
        """
        # TODO: Serialize all entries to JSON
        pass


def exercise_2():
    """Verify CRUD operations work."""
    lib = PromptLibrary()

    lib.add(PromptEntry("summarize", "gen", "System.", "Summarize: {text}", ["text"]))
    lib.add(PromptEntry("classify", "analysis", "System.", "Classify: {text}", ["text"]))
    lib.add(PromptEntry("extract", "analysis", "System.", "Extract: {text}", ["text"]))

    entry = lib.get("summarize")
    assert entry.name == "summarize"

    updated = lib.update("summarize", system_prompt="Better system.", tags=["v2"])
    assert updated.version == 2
    assert updated.system_prompt == "Better system."

    results = lib.search("analysis")
    assert len(results) == 2

    by_cat = lib.list_by_category("analysis")
    assert len(by_cat) == 2

    lib.delete("extract")
    assert len(lib.search("extract")) == 0

    exported = lib.export_json()
    data = json.loads(exported)
    assert len(data) == 2

    print(f"  Library size: {len(lib._entries)}")
    print(f"  Search 'analysis': {len(results)} results")
    print(f"  Export: {len(exported)} chars of JSON")


# === Exercise 3: Prompt Evaluation Harness ===
# Add evaluation capabilities to the library.

@dataclass
class EvalCase:
    input_vars: dict[str, str]
    expected_keywords: list[str]  # Keywords that should appear in output
    category: str = "general"


def build_eval_suite() -> dict[str, list[EvalCase]]:
    """Build evaluation test cases for library prompts.

    Return a dict mapping prompt name to a list of EvalCase instances.
    Create at least 2 prompts with 2 test cases each.

    Hint: The expected_keywords are words you expect to find in a
    good model response (used for keyword-based evaluation).
    """
    # TODO: Create eval cases for at least 2 prompt names
    # TODO: Each prompt should have at least 2 test cases
    pass


def evaluate_prompt(
    entry: PromptEntry,
    cases: list[EvalCase],
    mock_respond=None,
) -> dict:
    """Evaluate a prompt against its test cases.

    Args:
        entry: The prompt to evaluate.
        cases: Test cases with input variables and expected keywords.
        mock_respond: Optional callable(system, user) -> str. If None,
                      return the rendered user prompt as the 'response'.

    Returns:
        {"prompt": name, "total": int, "passed": int, "pass_rate": float,
         "details": [{"case_idx": int, "passed": bool, "missing_keywords": list}]}

    Hint: For each case, render the prompt with input_vars, get response,
    check that all expected_keywords appear in the response (case-insensitive).
    """
    # TODO: Define default mock_respond if None
    # TODO: Run each case through render + respond
    # TODO: Check expected keywords in the response
    # TODO: Return evaluation summary
    pass


def exercise_3():
    """Verify evaluation harness works."""
    suite = build_eval_suite()
    assert suite is not None and len(suite) >= 2, "Need at least 2 prompts"
    for name, cases in suite.items():
        assert len(cases) >= 2, f"Need at least 2 cases for '{name}'"

    entry = PromptEntry(
        "test_prompt", "test", "System.", "Tell me about {topic}", ["topic"],
    )
    cases = [
        EvalCase({"topic": "Python"}, ["Python"]),
        EvalCase({"topic": "AI"}, ["AI"]),
    ]
    report = evaluate_prompt(entry, cases)
    assert report is not None
    assert report["total"] == 2
    assert "pass_rate" in report
    print(f"  Eval suite: {len(suite)} prompts")
    print(f"  Test prompt pass rate: {report['pass_rate']:.0%}")


# === Exercise 4: Library Statistics Dashboard ===
# Compute aggregate statistics about the prompt library.

def library_stats(lib: PromptLibrary) -> dict[str, Any]:
    """Compute statistics about the prompt library.

    Return:
        {
            "total_prompts": int,
            "categories": dict[str, int],  # category -> count
            "avg_system_length": float,     # average system prompt char length
            "avg_template_length": float,   # average user template char length
            "avg_variables": float,         # average number of variables per prompt
            "all_tags": list[str],          # unique tags across all prompts, sorted
            "version_distribution": dict[int, int],  # version -> count
        }

    Hint: Iterate over all entries and accumulate counts.
    """
    # TODO: Count prompts per category
    # TODO: Compute average lengths
    # TODO: Collect all unique tags
    # TODO: Build version distribution
    pass


def exercise_4():
    """Verify statistics computation."""
    lib = PromptLibrary()
    lib.add(PromptEntry("a", "gen", "Short.", "Do {x}", ["x"], tags=["v1"]))
    lib.add(PromptEntry("b", "gen", "Longer system prompt here.", "Do {x} {y}",
                         ["x", "y"], version=2, tags=["v2", "tested"]))
    lib.add(PromptEntry("c", "analysis", "Sys.", "Analyze {text}",
                         ["text"], tags=["v1"]))

    stats = library_stats(lib)
    assert stats is not None
    assert stats["total_prompts"] == 3
    assert stats["categories"]["gen"] == 2
    assert stats["categories"]["analysis"] == 1
    assert stats["avg_variables"] > 1.0
    assert "tested" in stats["all_tags"]
    print(f"  Total prompts: {stats['total_prompts']}")
    print(f"  Categories: {stats['categories']}")
    print(f"  Avg variables: {stats['avg_variables']:.1f}")
    print(f"  Tags: {stats['all_tags']}")


if __name__ == "__main__":
    print("=== Exercise 1: Prompt Entry Model ===")
    exercise_1()

    print("=== Exercise 2: Prompt Library CRUD ===")
    exercise_2()

    print("=== Exercise 3: Evaluation Harness ===")
    exercise_3()

    print("=== Exercise 4: Library Statistics ===")
    exercise_4()

    print("\nAll exercises completed!")
