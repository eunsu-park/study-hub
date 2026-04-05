"""
Exercises for Lesson 10: Claude Desktop Application
Topic: Claude_Ecosystem

Solutions to practice problems from the lesson.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# === Exercise 1: Artifact Type Classifier ===
# Problem: Classify Claude Desktop artifacts by type and determine
#   rendering behavior.

class ArtifactType(Enum):
    CODE = "code"
    DOCUMENT = "document"
    SVG = "svg"
    HTML = "html"
    MERMAID = "mermaid"
    REACT = "react"


@dataclass
class Artifact:
    """A Claude Desktop artifact."""
    title: str
    artifact_type: ArtifactType
    content: str
    language: str = ""
    version: int = 1


ARTIFACT_SIGNATURES: dict[str, ArtifactType] = {
    ".py": ArtifactType.CODE,
    ".js": ArtifactType.CODE,
    ".ts": ArtifactType.CODE,
    ".html": ArtifactType.HTML,
    ".svg": ArtifactType.SVG,
    ".md": ArtifactType.DOCUMENT,
    ".txt": ArtifactType.DOCUMENT,
}

CONTENT_SIGNATURES: list[tuple[str, ArtifactType]] = [
    ("```mermaid", ArtifactType.MERMAID),
    ("<svg", ArtifactType.SVG),
    ("<!DOCTYPE html>", ArtifactType.HTML),
    ("<html", ArtifactType.HTML),
    ("import React", ArtifactType.REACT),
    ("export default function", ArtifactType.REACT),
]


def classify_artifact(title: str, content: str) -> ArtifactType:
    """Classify an artifact based on its title extension and content."""
    for ext, art_type in ARTIFACT_SIGNATURES.items():
        if title.endswith(ext):
            return art_type

    for signature, art_type in CONTENT_SIGNATURES:
        if signature in content[:200]:
            return art_type

    return ArtifactType.DOCUMENT


def exercise_1():
    """Demonstrate artifact classification."""
    test_cases = [
        ("utils.py", "def add(a, b): return a + b"),
        ("chart.svg", "<svg viewBox='0 0 100 100'>...</svg>"),
        ("report.md", "# Quarterly Report\n\n## Summary"),
        ("app.html", "<!DOCTYPE html><html>...</html>"),
        ("diagram", "```mermaid\ngraph TD\n  A-->B\n```"),
        ("Dashboard", "import React from 'react';\nexport default function App(){}"),
    ]
    for title, content in test_cases:
        art_type = classify_artifact(title, content)
        print(f"  '{title}' → {art_type.value}")


# === Exercise 2: Project Knowledge Base ===
# Problem: Model a Claude Desktop project with knowledge documents
#   and custom instructions.

@dataclass
class ProjectKnowledge:
    """A knowledge document attached to a Claude project."""
    name: str
    content: str
    token_count: int


@dataclass
class DesktopProject:
    """A Claude Desktop project with instructions and knowledge."""
    name: str
    custom_instructions: str
    knowledge: list[ProjectKnowledge] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    max_knowledge_tokens: int = 200_000

    def add_knowledge(self, name: str, content: str) -> bool:
        """Add a knowledge document if within token budget."""
        tokens = max(1, len(content) // 4)
        current_total = sum(k.token_count for k in self.knowledge)
        if current_total + tokens > self.max_knowledge_tokens:
            return False
        self.knowledge.append(ProjectKnowledge(name, content, tokens))
        return True

    def knowledge_budget(self) -> dict[str, int]:
        used = sum(k.token_count for k in self.knowledge)
        return {
            "used": used,
            "available": self.max_knowledge_tokens - used,
            "total": self.max_knowledge_tokens,
            "documents": len(self.knowledge),
        }


def exercise_2():
    """Demonstrate project knowledge management."""
    project = DesktopProject(
        name="API Documentation Writer",
        custom_instructions="You are a technical writer. Generate clear, "
                           "concise API documentation with examples.",
    )
    docs = [
        ("API Spec", "OpenAPI 3.0 specification..." * 100),
        ("Style Guide", "Use active voice. Keep sentences short." * 50),
        ("Examples", "GET /users\nResponse: {\"users\": [...]}" * 30),
    ]
    for name, content in docs:
        success = project.add_knowledge(name, content)
        status = "added" if success else "rejected (over budget)"
        print(f"  {name}: {status}")

    budget = project.knowledge_budget()
    print(f"\n  Budget: {budget['used']:,}/{budget['total']:,} tokens "
          f"({budget['documents']} docs)")


# === Exercise 3: Conversation State Tracker ===
# Problem: Track conversation state including messages, artifacts,
#   and model usage within a Claude Desktop session.

@dataclass
class ConversationTurn:
    """A single turn in a Desktop conversation."""
    role: str  # "user" or "assistant"
    text_length: int
    artifacts_created: int = 0
    model_used: str = "sonnet"


@dataclass
class ConversationTracker:
    """Track statistics for a Claude Desktop conversation."""
    turns: list[ConversationTurn] = field(default_factory=list)
    total_artifacts: int = 0

    def add_turn(self, role: str, text: str,
                 artifacts: int = 0, model: str = "sonnet") -> None:
        self.turns.append(ConversationTurn(
            role=role, text_length=len(text),
            artifacts_created=artifacts, model_used=model,
        ))
        self.total_artifacts += artifacts

    def stats(self) -> dict[str, Any]:
        user_turns = [t for t in self.turns if t.role == "user"]
        assistant_turns = [t for t in self.turns if t.role == "assistant"]
        return {
            "total_turns": len(self.turns),
            "user_messages": len(user_turns),
            "assistant_messages": len(assistant_turns),
            "total_artifacts": self.total_artifacts,
            "avg_user_length": (
                sum(t.text_length for t in user_turns) // max(len(user_turns), 1)
            ),
            "avg_assistant_length": (
                sum(t.text_length for t in assistant_turns)
                // max(len(assistant_turns), 1)
            ),
            "models_used": list({t.model_used for t in self.turns}),
        }


def exercise_3():
    """Simulate a Desktop conversation and track stats."""
    tracker = ConversationTracker()
    tracker.add_turn("user", "Create a Python script to parse CSV files")
    tracker.add_turn("assistant", "Here's a CSV parser..." * 20,
                     artifacts=1, model="sonnet")
    tracker.add_turn("user", "Add error handling and type hints")
    tracker.add_turn("assistant", "Updated version with error handling..." * 30,
                     artifacts=1, model="sonnet")
    tracker.add_turn("user", "Now create a flowchart of the parsing logic")
    tracker.add_turn("assistant", "Here's the flowchart..." * 10,
                     artifacts=1, model="sonnet")

    stats = tracker.stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    print("=== Exercise 1: Artifact Classifier ===")
    exercise_1()

    print("\n=== Exercise 2: Project Knowledge ===")
    exercise_2()

    print("\n=== Exercise 3: Conversation Tracker ===")
    exercise_3()

    print("\nAll exercises completed!")
