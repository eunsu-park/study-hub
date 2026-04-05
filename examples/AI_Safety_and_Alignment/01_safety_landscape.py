# 01_safety_landscape.py — Risk taxonomy classifier and incident database
#
# Run: python 01_safety_landscape.py

"""
Demonstrates an AI safety risk taxonomy with a simple incident database.
Classifies incidents by risk category and severity, and provides
basic analytics over the incident corpus.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import json
from datetime import datetime


class RiskCategory(Enum):
    MISUSE = "misuse"
    ALIGNMENT_FAILURE = "alignment_failure"
    ROBUSTNESS = "robustness"
    PRIVACY = "privacy"
    FAIRNESS = "fairness"
    TRANSPARENCY = "transparency"
    SECURITY = "security"
    SOCIETAL = "societal"


class Severity(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Incident:
    id: int
    title: str
    description: str
    category: RiskCategory
    severity: Severity
    date: str
    mitigated: bool = False
    tags: list = field(default_factory=list)


class IncidentDatabase:
    """Simple in-memory incident database with search and analytics."""

    def __init__(self):
        self.incidents: list[Incident] = []
        self._next_id = 1

    def add_incident(self, title: str, description: str,
                     category: RiskCategory, severity: Severity,
                     date: str, tags: Optional[list] = None) -> Incident:
        incident = Incident(
            id=self._next_id, title=title, description=description,
            category=category, severity=severity, date=date,
            tags=tags or []
        )
        self.incidents.append(incident)
        self._next_id += 1
        return incident

    def search(self, category: Optional[RiskCategory] = None,
               min_severity: Optional[Severity] = None) -> list[Incident]:
        results = self.incidents
        if category:
            results = [i for i in results if i.category == category]
        if min_severity:
            results = [i for i in results
                       if i.severity.value >= min_severity.value]
        return results

    def severity_distribution(self) -> dict[str, int]:
        dist = {s.name: 0 for s in Severity}
        for inc in self.incidents:
            dist[inc.severity.name] += 1
        return dist

    def category_distribution(self) -> dict[str, int]:
        dist = {c.value: 0 for c in RiskCategory}
        for inc in self.incidents:
            dist[inc.category.value] += 1
        return dist

    def mitigation_rate(self) -> float:
        if not self.incidents:
            return 0.0
        return sum(1 for i in self.incidents if i.mitigated) / len(self.incidents)


class RiskTaxonomyClassifier:
    """Rule-based classifier that maps keywords to risk categories."""

    KEYWORD_MAP = {
        RiskCategory.MISUSE: ["deepfake", "spam", "scam", "weapon", "fraud"],
        RiskCategory.ALIGNMENT_FAILURE: ["reward hacking", "misaligned",
                                         "unintended", "goal"],
        RiskCategory.ROBUSTNESS: ["adversarial", "perturbation", "jailbreak",
                                  "prompt injection"],
        RiskCategory.PRIVACY: ["data leak", "personal information", "pii",
                               "surveillance"],
        RiskCategory.FAIRNESS: ["bias", "discrimination", "demographic",
                                "disparate"],
        RiskCategory.TRANSPARENCY: ["black box", "unexplainable",
                                    "opaque", "interpretability"],
        RiskCategory.SECURITY: ["exploit", "vulnerability", "attack",
                                "backdoor"],
        RiskCategory.SOCIETAL: ["job displacement", "misinformation",
                                "inequality", "democracy"],
    }

    def classify(self, text: str) -> list[tuple[RiskCategory, float]]:
        text_lower = text.lower()
        scores = []
        for category, keywords in self.KEYWORD_MAP.items():
            hits = sum(1 for kw in keywords if kw in text_lower)
            if hits > 0:
                score = hits / len(keywords)
                scores.append((category, round(score, 3)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores if scores else [(RiskCategory.SOCIETAL, 0.1)]


def build_sample_database() -> IncidentDatabase:
    db = IncidentDatabase()
    db.add_incident(
        "Chatbot jailbreak via prompt injection",
        "Users bypassed safety filters using adversarial prompt injection.",
        RiskCategory.ROBUSTNESS, Severity.HIGH, "2024-03-15",
        tags=["jailbreak", "prompt injection"]
    )
    db.add_incident(
        "Biased hiring model",
        "Resume screening showed demographic bias and discrimination.",
        RiskCategory.FAIRNESS, Severity.CRITICAL, "2024-01-10",
        tags=["bias", "hiring"]
    )
    db.add_incident(
        "Deepfake political ad",
        "AI-generated deepfake used in election fraud campaign.",
        RiskCategory.MISUSE, Severity.CRITICAL, "2024-06-01",
        tags=["deepfake", "politics"]
    )
    db.add_incident(
        "Reward hacking in RL agent",
        "Agent found unintended shortcut, misaligned with true goal.",
        RiskCategory.ALIGNMENT_FAILURE, Severity.MEDIUM, "2024-02-20",
        tags=["reward hacking", "RL"]
    )
    db.incidents[0].mitigated = True
    return db


if __name__ == "__main__":
    # Build and query the incident database
    db = build_sample_database()

    print("=== AI Safety Incident Database ===\n")
    print(f"Total incidents: {len(db.incidents)}")
    print(f"Mitigation rate: {db.mitigation_rate():.0%}\n")

    print("Severity distribution:", db.severity_distribution())
    print("Category distribution:", db.category_distribution())

    print("\n--- High+ severity incidents ---")
    for inc in db.search(min_severity=Severity.HIGH):
        print(f"  [{inc.severity.name}] {inc.title} ({inc.category.value})")

    # Classify a new report
    print("\n=== Risk Taxonomy Classifier ===\n")
    classifier = RiskTaxonomyClassifier()
    test_texts = [
        "Users found a jailbreak via adversarial prompt injection attack.",
        "Model exhibits bias against certain demographic groups.",
        "AI-generated deepfake used for fraud and scam operations.",
    ]
    for text in test_texts:
        results = classifier.classify(text)
        top = results[0]
        print(f"Text: {text[:60]}...")
        print(f"  -> {top[0].value} (confidence: {top[1]:.2f})\n")
