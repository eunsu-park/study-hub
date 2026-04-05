# 04_constitutional_ai.py — Self-critique and revision pipeline
#
# Run: python 04_constitutional_ai.py

"""
Demonstrates the Constitutional AI (CAI) approach: a model critiques
its own outputs against a set of principles (constitution), then
revises them. Includes both rule-based and LLM-simulated critique.
"""

from dataclasses import dataclass, field
from enum import Enum


class Principle(Enum):
    HARMLESSNESS = "harmlessness"
    HELPFULNESS = "helpfulness"
    HONESTY = "honesty"
    FAIRNESS = "fairness"
    PRIVACY = "privacy"


@dataclass
class ConstitutionRule:
    principle: Principle
    description: str
    violation_keywords: list[str]
    revision_guidance: str


@dataclass
class CritiqueResult:
    rule: ConstitutionRule
    violated: bool
    explanation: str
    severity: float  # 0.0 to 1.0


@dataclass
class RevisionResult:
    original: str
    revised: str
    critiques: list[CritiqueResult]
    revision_count: int


# Default constitution for demonstration
DEFAULT_CONSTITUTION = [
    ConstitutionRule(
        Principle.HARMLESSNESS,
        "Response must not contain instructions for harmful activities.",
        ["how to hack", "how to steal", "weapon", "exploit", "attack",
         "harm", "illegal", "destroy"],
        "Remove harmful instructions. Explain why the request is unsafe."
    ),
    ConstitutionRule(
        Principle.HELPFULNESS,
        "Response should provide useful, actionable information.",
        ["i don't know", "figure it out", "not my problem",
         "can't help"],
        "Provide constructive guidance or redirect to helpful resources."
    ),
    ConstitutionRule(
        Principle.HONESTY,
        "Response must not make false claims or present speculation as fact.",
        ["definitely", "guaranteed", "100%", "always works",
         "proven fact", "everyone knows"],
        "Add appropriate hedging. Distinguish facts from opinions."
    ),
    ConstitutionRule(
        Principle.FAIRNESS,
        "Response must not contain stereotypes or biased generalizations.",
        ["all men", "all women", "those people", "they always",
         "typical of", "naturally better"],
        "Remove generalizations. Use inclusive, nuanced language."
    ),
    ConstitutionRule(
        Principle.PRIVACY,
        "Response must not request or reveal personal information.",
        ["social security", "credit card", "password", "home address",
         "phone number", "send me your"],
        "Remove PII requests. Explain why sharing PII is risky."
    ),
]


class ConstitutionalCritic:
    """Critiques model outputs against a constitution."""

    def __init__(self, constitution: list[ConstitutionRule] = None):
        self.constitution = constitution or DEFAULT_CONSTITUTION

    def critique(self, response: str) -> list[CritiqueResult]:
        results = []
        response_lower = response.lower()

        for rule in self.constitution:
            violations = [kw for kw in rule.violation_keywords
                          if kw in response_lower]
            violated = len(violations) > 0
            severity = min(len(violations) / 3.0, 1.0) if violated else 0.0

            explanation = (
                f"Violated {rule.principle.value}: found [{', '.join(violations)}]"
                if violated
                else f"No {rule.principle.value} violations detected."
            )
            results.append(CritiqueResult(rule, violated, explanation,
                                          severity))
        return results


class ConstitutionalReviser:
    """Revises responses based on critique feedback."""

    def __init__(self):
        self.revision_rules = {
            Principle.HARMLESSNESS: self._revise_harmful,
            Principle.HELPFULNESS: self._revise_unhelpful,
            Principle.HONESTY: self._revise_dishonest,
            Principle.FAIRNESS: self._revise_unfair,
            Principle.PRIVACY: self._revise_privacy,
        }

    def _revise_harmful(self, text: str, critique: CritiqueResult) -> str:
        for kw in critique.rule.violation_keywords:
            if kw in text.lower():
                idx = text.lower().find(kw)
                text = (text[:idx] + "[REDACTED: harmful content removed]"
                        + text[idx + len(kw):])
        return text + " Note: I cannot assist with harmful activities."

    def _revise_unhelpful(self, text: str, critique: CritiqueResult) -> str:
        return ("I'd be happy to help with that. " + text +
                " Let me provide more specific guidance.")

    def _revise_dishonest(self, text: str, critique: CritiqueResult) -> str:
        for kw in critique.rule.violation_keywords:
            text = text.replace(kw, f"likely {kw}")
        return text + " (Note: this is based on available evidence.)"

    def _revise_unfair(self, text: str, critique: CritiqueResult) -> str:
        for kw in critique.rule.violation_keywords:
            if kw in text.lower():
                text = text.replace(kw, "some individuals")
                text = text.replace(kw.title(), "Some individuals")
        return text

    def _revise_privacy(self, text: str, critique: CritiqueResult) -> str:
        for kw in critique.rule.violation_keywords:
            if kw in text.lower():
                idx = text.lower().find(kw)
                text = (text[:idx] + "[PII request removed]"
                        + text[idx + len(kw):])
        return text + " Please never share personal information online."

    def revise(self, text: str,
               critiques: list[CritiqueResult]) -> RevisionResult:
        revised = text
        revision_count = 0

        for critique in critiques:
            if critique.violated:
                revise_fn = self.revision_rules.get(
                    critique.rule.principle)
                if revise_fn:
                    revised = revise_fn(revised, critique)
                    revision_count += 1

        return RevisionResult(text, revised, critiques, revision_count)


class CAIPipeline:
    """Full Constitutional AI pipeline: generate -> critique -> revise."""

    def __init__(self, max_rounds: int = 3):
        self.critic = ConstitutionalCritic()
        self.reviser = ConstitutionalReviser()
        self.max_rounds = max_rounds

    def process(self, response: str) -> RevisionResult:
        current = response
        all_critiques = []

        for round_num in range(self.max_rounds):
            critiques = self.critic.critique(current)
            violations = [c for c in critiques if c.violated]
            all_critiques.extend(violations)

            if not violations:
                break

            result = self.reviser.revise(current, critiques)
            current = result.revised

        return RevisionResult(response, current, all_critiques,
                              len(all_critiques))


if __name__ == "__main__":
    print("=== Constitutional AI Pipeline ===\n")

    pipeline = CAIPipeline(max_rounds=3)

    test_responses = [
        "Here's how to hack into a WiFi network and exploit the system.",
        "I don't know, figure it out yourself. Not my problem.",
        "This method is guaranteed to work 100% of the time, always works.",
        "All women are naturally better at this. Typical of men to fail.",
        "Send me your credit card number and home address to verify.",
        "Python is a great programming language for beginners.",
    ]

    for response in test_responses:
        result = pipeline.process(response)
        print(f"Original:  {result.original}")
        print(f"Revised:   {result.revised[:100]}...")
        print(f"Revisions: {result.revision_count}")

        if result.critiques:
            for c in result.critiques:
                print(f"  - {c.rule.principle.value}: "
                      f"severity={c.severity:.1f}")
        else:
            print("  No violations found.")
        print()
