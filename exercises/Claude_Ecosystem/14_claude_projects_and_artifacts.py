"""
Exercises for Lesson 14: Claude Projects and Artifacts
Topic: Claude_Ecosystem

Solutions to practice problems from the lesson.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# === Exercise 1: Project Organization Manager ===
# Problem: Model a Claude Project with custom instructions,
#   knowledge documents, and conversations.

@dataclass
class KnowledgeDoc:
    """A knowledge document in a Claude project."""
    name: str
    content: str
    source: str  # "uploaded", "pasted", "url"

    @property
    def token_estimate(self) -> int:
        return max(1, len(self.content) // 4)


@dataclass
class Conversation:
    """A conversation within a Claude project."""
    title: str
    message_count: int
    starred: bool = False


@dataclass
class ClaudeProject:
    """A Claude.ai project with knowledge grounding."""
    name: str
    instructions: str
    knowledge: list[KnowledgeDoc] = field(default_factory=list)
    conversations: list[Conversation] = field(default_factory=list)

    def total_knowledge_tokens(self) -> int:
        return sum(d.token_estimate for d in self.knowledge)

    def add_knowledge(self, doc: KnowledgeDoc,
                      max_tokens: int = 200_000) -> bool:
        if self.total_knowledge_tokens() + doc.token_estimate > max_tokens:
            return False
        self.knowledge.append(doc)
        return True

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "knowledge_docs": len(self.knowledge),
            "knowledge_tokens": self.total_knowledge_tokens(),
            "conversations": len(self.conversations),
            "starred": sum(1 for c in self.conversations if c.starred),
        }


def exercise_1():
    """Demonstrate project organization."""
    project = ClaudeProject(
        name="API Documentation Assistant",
        instructions="Generate OpenAPI specs from code. Use concise descriptions. "
                     "Include request/response examples.",
    )
    docs = [
        KnowledgeDoc("OpenAPI 3.1 Spec", "paths: ..." * 200, "uploaded"),
        KnowledgeDoc("Style Guide", "Use active voice..." * 50, "pasted"),
        KnowledgeDoc("Existing API", "GET /users ..." * 100, "url"),
    ]
    for doc in docs:
        added = project.add_knowledge(doc)
        print(f"  {doc.name}: {'added' if added else 'rejected'} "
              f"(~{doc.token_estimate:,} tokens)")

    project.conversations.append(Conversation("Generate /users endpoint", 12))
    project.conversations.append(Conversation("Review auth flows", 8, starred=True))

    print(f"\n  Project summary: {project.summary()}")


# === Exercise 2: Artifact Version Tracker ===
# Problem: Track artifact versions as they are created and updated
#   across conversation turns.

@dataclass
class ArtifactVersion:
    """A version of an artifact."""
    version: int
    content: str
    change_description: str


@dataclass
class TrackedArtifact:
    """An artifact with version history."""
    identifier: str
    title: str
    artifact_type: str
    versions: list[ArtifactVersion] = field(default_factory=list)

    @property
    def current(self) -> ArtifactVersion | None:
        return self.versions[-1] if self.versions else None

    def create(self, content: str, description: str = "Initial version") -> None:
        self.versions.append(ArtifactVersion(1, content, description))

    def update(self, content: str, description: str) -> int:
        version = len(self.versions) + 1
        self.versions.append(ArtifactVersion(version, content, description))
        return version

    def diff_summary(self) -> list[dict[str, Any]]:
        return [
            {"version": v.version,
             "size": len(v.content),
             "change": v.change_description}
            for v in self.versions
        ]


class ArtifactManager:
    """Manages artifacts across a project."""

    def __init__(self) -> None:
        self._artifacts: dict[str, TrackedArtifact] = {}

    def create(self, identifier: str, title: str,
               art_type: str, content: str) -> TrackedArtifact:
        artifact = TrackedArtifact(identifier, title, art_type)
        artifact.create(content)
        self._artifacts[identifier] = artifact
        return artifact

    def update(self, identifier: str, content: str,
               description: str) -> int | None:
        artifact = self._artifacts.get(identifier)
        if not artifact:
            return None
        return artifact.update(content, description)

    def list_artifacts(self) -> list[dict[str, Any]]:
        return [
            {"id": a.identifier, "title": a.title, "type": a.artifact_type,
             "versions": len(a.versions),
             "current_size": len(a.current.content) if a.current else 0}
            for a in self._artifacts.values()
        ]


def exercise_2():
    """Demonstrate artifact version tracking."""
    mgr = ArtifactManager()
    mgr.create("csv-parser", "CSV Parser", "code",
               "import csv\ndef parse(f): ...")
    mgr.update("csv-parser",
               "import csv\ndef parse(f, delimiter=','): ...",
               "Added delimiter parameter")
    mgr.update("csv-parser",
               "import csv\nfrom pathlib import Path\ndef parse(f, delimiter=','): ...",
               "Added type hints and Path support")

    mgr.create("arch-diagram", "Architecture Diagram", "mermaid",
               "graph TD\n  A-->B")

    print("  Artifacts:")
    for a in mgr.list_artifacts():
        print(f"    {a['id']} ({a['type']}): v{a['versions']}, "
              f"{a['current_size']} chars")

    artifact = mgr._artifacts["csv-parser"]
    print(f"\n  Version history for '{artifact.title}':")
    for d in artifact.diff_summary():
        print(f"    v{d['version']}: {d['change']} ({d['size']} chars)")


# === Exercise 3: Knowledge Grounding Optimizer ===
# Problem: Given a set of knowledge documents and a token budget,
#   select the optimal subset to include in the project.

def optimize_knowledge(
    documents: list[KnowledgeDoc],
    max_tokens: int,
    priorities: dict[str, int] | None = None,
) -> list[KnowledgeDoc]:
    """Select documents that fit within token budget, prioritized by relevance.

    Uses a greedy approach: sort by priority (descending), then by
    token efficiency (tokens per priority point).
    """
    if priorities is None:
        priorities = {d.name: 1 for d in documents}

    scored = [
        (d, priorities.get(d.name, 0))
        for d in documents
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    selected: list[KnowledgeDoc] = []
    remaining = max_tokens

    for doc, _ in scored:
        tokens = doc.token_estimate
        if tokens <= remaining:
            selected.append(doc)
            remaining -= tokens

    return selected


def exercise_3():
    """Demonstrate knowledge optimization."""
    documents = [
        KnowledgeDoc("API Specification", "spec..." * 5000, "uploaded"),
        KnowledgeDoc("Style Guide", "guide..." * 500, "pasted"),
        KnowledgeDoc("Code Examples", "examples..." * 2000, "uploaded"),
        KnowledgeDoc("FAQ", "faq..." * 300, "pasted"),
        KnowledgeDoc("Changelog", "changes..." * 3000, "url"),
    ]
    priorities = {
        "API Specification": 10,
        "Style Guide": 8,
        "Code Examples": 7,
        "FAQ": 5,
        "Changelog": 2,
    }
    budget = 10_000
    selected = optimize_knowledge(documents, budget, priorities)

    print(f"  Budget: {budget:,} tokens")
    print(f"  Available: {len(documents)} docs, "
          f"{sum(d.token_estimate for d in documents):,} tokens total")
    print(f"\n  Selected ({len(selected)} docs):")
    total_used = 0
    for doc in selected:
        total_used += doc.token_estimate
        print(f"    {doc.name}: ~{doc.token_estimate:,} tokens "
              f"(priority: {priorities[doc.name]})")
    print(f"  Total used: {total_used:,}/{budget:,} tokens")


if __name__ == "__main__":
    print("=== Exercise 1: Project Organization ===")
    exercise_1()

    print("\n=== Exercise 2: Artifact Versioning ===")
    exercise_2()

    print("\n=== Exercise 3: Knowledge Optimization ===")
    exercise_3()

    print("\nAll exercises completed!")
