# 11_guardrails.py — Input/output filtering pipeline
#
# Run: python 11_guardrails.py

"""
Implements a multi-layer guardrails system for AI applications.
Includes input validation, topic classification, output filtering,
and PII detection in a composable pipeline architecture.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable


class FilterAction(Enum):
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"
    MODIFY = "modify"


@dataclass
class FilterResult:
    action: FilterAction
    original: str
    modified: Optional[str] = None
    reason: str = ""
    filter_name: str = ""


@dataclass
class PipelineResult:
    allowed: bool
    final_text: str
    filters_applied: list[FilterResult] = field(default_factory=list)
    blocked_by: Optional[str] = None


class InputLengthFilter:
    """Enforces input length constraints."""

    def __init__(self, min_len: int = 1, max_len: int = 5000):
        self.min_len = min_len
        self.max_len = max_len

    def check(self, text: str) -> FilterResult:
        if len(text) < self.min_len:
            return FilterResult(FilterAction.BLOCK, text,
                                reason="Input too short",
                                filter_name="length")
        if len(text) > self.max_len:
            truncated = text[:self.max_len] + "... [truncated]"
            return FilterResult(FilterAction.MODIFY, text,
                                modified=truncated,
                                reason="Input truncated to max length",
                                filter_name="length")
        return FilterResult(FilterAction.ALLOW, text,
                            filter_name="length")


class PIIDetector:
    """Detects and optionally redacts personally identifiable info."""

    PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    }

    def __init__(self, redact: bool = True):
        self.redact = redact

    def check(self, text: str) -> FilterResult:
        found = []
        modified = text
        for pii_type, pattern in self.PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                found.append(pii_type)
                if self.redact:
                    modified = re.sub(pattern,
                                      f"[REDACTED-{pii_type.upper()}]",
                                      modified)

        if found:
            action = FilterAction.MODIFY if self.redact \
                else FilterAction.WARN
            return FilterResult(
                action, text, modified=modified,
                reason=f"PII detected: {', '.join(found)}",
                filter_name="pii")
        return FilterResult(FilterAction.ALLOW, text, filter_name="pii")


class TopicFilter:
    """Blocks or warns on specific topic categories."""

    BLOCKED_TOPICS = {
        "weapons": ["bomb", "weapon", "explosive", "firearm",
                     "ammunition"],
        "illegal": ["drug synthesis", "counterfeit", "money laundering",
                     "trafficking"],
        "malware": ["ransomware", "trojan", "keylogger", "exploit code",
                     "zero-day"],
    }

    WARNED_TOPICS = {
        "medical": ["diagnosis", "prescription", "treatment plan",
                     "dosage"],
        "legal": ["legal advice", "sue", "lawsuit", "court order"],
        "financial": ["invest", "stock pick", "guaranteed returns"],
    }

    def check(self, text: str) -> FilterResult:
        text_lower = text.lower()

        for topic, keywords in self.BLOCKED_TOPICS.items():
            if any(kw in text_lower for kw in keywords):
                return FilterResult(
                    FilterAction.BLOCK, text,
                    reason=f"Blocked topic: {topic}",
                    filter_name="topic")

        for topic, keywords in self.WARNED_TOPICS.items():
            if any(kw in text_lower for kw in keywords):
                return FilterResult(
                    FilterAction.WARN, text,
                    reason=f"Sensitive topic: {topic}",
                    filter_name="topic")

        return FilterResult(FilterAction.ALLOW, text, filter_name="topic")


class ToxicityFilter:
    """Simple toxicity detection based on keyword patterns."""

    TOXIC_PATTERNS = [
        r"\b(idiot|stupid|dumb|moron)\b",
        r"\b(hate|kill|destroy|attack)\s+(you|them|everyone)\b",
        r"\b(shut\s+up|go\s+away|get\s+lost)\b",
    ]

    def __init__(self, threshold: int = 1):
        self.threshold = threshold

    def check(self, text: str) -> FilterResult:
        matches = []
        for pattern in self.TOXIC_PATTERNS:
            found = re.findall(pattern, text, re.IGNORECASE)
            if found:
                matches.extend(found)

        if len(matches) >= self.threshold:
            return FilterResult(
                FilterAction.BLOCK, text,
                reason=f"Toxicity detected ({len(matches)} matches)",
                filter_name="toxicity")
        return FilterResult(FilterAction.ALLOW, text,
                            filter_name="toxicity")


class OutputDisclaimer:
    """Adds disclaimers to outputs on sensitive topics."""

    DISCLAIMERS = {
        "medical": "\n\nDisclaimer: This is not medical advice. "
                   "Please consult a healthcare professional.",
        "legal": "\n\nDisclaimer: This is not legal advice. "
                 "Please consult a licensed attorney.",
        "financial": "\n\nDisclaimer: This is not financial advice. "
                     "Past performance does not guarantee future results.",
    }

    def check(self, text: str) -> FilterResult:
        text_lower = text.lower()
        for topic, disclaimer in self.DISCLAIMERS.items():
            triggers = {
                "medical": ["symptom", "medicine", "health"],
                "legal": ["court", "law", "regulation"],
                "financial": ["invest", "stock", "portfolio"],
            }
            if any(t in text_lower for t in triggers.get(topic, [])):
                return FilterResult(
                    FilterAction.MODIFY, text,
                    modified=text + disclaimer,
                    reason=f"Added {topic} disclaimer",
                    filter_name="disclaimer")
        return FilterResult(FilterAction.ALLOW, text,
                            filter_name="disclaimer")


class GuardrailsPipeline:
    """Composable pipeline of input/output filters."""

    def __init__(self):
        self.input_filters = []
        self.output_filters = []

    def add_input_filter(self, filter_obj):
        self.input_filters.append(filter_obj)
        return self

    def add_output_filter(self, filter_obj):
        self.output_filters.append(filter_obj)
        return self

    def process_input(self, text: str) -> PipelineResult:
        return self._run_filters(text, self.input_filters)

    def process_output(self, text: str) -> PipelineResult:
        return self._run_filters(text, self.output_filters)

    def _run_filters(self, text: str, filters: list) -> PipelineResult:
        current_text = text
        results = []

        for f in filters:
            result = f.check(current_text)
            results.append(result)

            if result.action == FilterAction.BLOCK:
                return PipelineResult(
                    allowed=False, final_text="",
                    filters_applied=results,
                    blocked_by=result.filter_name)
            elif result.action == FilterAction.MODIFY and result.modified:
                current_text = result.modified

        return PipelineResult(
            allowed=True, final_text=current_text,
            filters_applied=results)


if __name__ == "__main__":
    print("=== Guardrails Pipeline ===\n")

    # Build pipeline
    pipeline = GuardrailsPipeline()
    pipeline.add_input_filter(InputLengthFilter())
    pipeline.add_input_filter(PIIDetector(redact=True))
    pipeline.add_input_filter(TopicFilter())
    pipeline.add_input_filter(ToxicityFilter())
    pipeline.add_output_filter(PIIDetector(redact=True))
    pipeline.add_output_filter(OutputDisclaimer())

    # Test inputs
    test_inputs = [
        "What is machine learning?",
        "My email is john@example.com and SSN is 123-45-6789",
        "How do I make a bomb?",
        "You idiot, shut up and go away",
        "What dosage of aspirin should I take for my headache?",
        "",
        "Explain how stock portfolio diversification works",
    ]

    print("--- Input Processing ---\n")
    for text in test_inputs:
        result = pipeline.process_input(text)
        status = "ALLOWED" if result.allowed else "BLOCKED"
        display = text[:55] if text else "(empty)"
        print(f"  [{status:>7}] {display}")

        if result.blocked_by:
            print(f"           Blocked by: {result.blocked_by}")
        for fr in result.filters_applied:
            if fr.action not in (FilterAction.ALLOW,):
                print(f"           {fr.filter_name}: {fr.reason}")
        if result.allowed and result.final_text != text:
            print(f"           Modified: {result.final_text[:55]}...")
        print()

    # Test output filtering
    print("--- Output Processing ---\n")
    test_outputs = [
        "The recommended medicine for headaches varies by symptom.",
        "Contact support at admin@company.com or call 555-123-4567.",
        "Consider diversifying your invest portfolio across sectors.",
    ]

    for text in test_outputs:
        result = pipeline.process_output(text)
        print(f"  Original: {text[:60]}")
        print(f"  Final:    {result.final_text[:70]}")
        for fr in result.filters_applied:
            if fr.action != FilterAction.ALLOW:
                print(f"  Filter:   {fr.filter_name} -> {fr.reason}")
        print()
