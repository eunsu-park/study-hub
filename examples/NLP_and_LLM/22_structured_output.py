"""
22. Structured Output from LLMs Example

JSON parsing, Pydantic models, validation, and extraction pipelines
"""

import json
import re
from dataclasses import dataclass
from typing import Literal
from enum import Enum

print("=" * 60)
print("Structured Output from LLMs")
print("=" * 60)


# ============================================
# 1. JSON Extraction (simulated)
# ============================================
print("\n[1] JSON Extraction from Text")
print("-" * 40)


def extract_entities_regex(text: str) -> dict:
    """Extract structured data using regex (baseline approach)."""
    # Simple entity extraction
    orgs = re.findall(r"(?:Apple|Google|Microsoft|Goldman Sachs|Morgan Stanley)", text)
    people = re.findall(r"(?:Tim Cook|Sundar Pichai|Satya Nadella)", text)
    percentages = re.findall(r"(\d+(?:\.\d+)?%)", text)
    numbers = re.findall(r"\$?([\d,]+(?:\.\d+)?)\s*(?:billion|million|%)", text)

    return {
        "organizations": list(set(orgs)),
        "people": list(set(people)),
        "metrics": percentages,
        "monetary_values": numbers,
    }


sample_text = """
Apple announced the new M4 chip today at their Cupertino headquarters.
CEO Tim Cook demonstrated 50% faster CPU and 2x GPU performance.
The stock rose 3% in after-hours. Goldman Sachs and Morgan Stanley
issued positive ratings.
"""

result = extract_entities_regex(sample_text)
print(f"Extracted (regex): {json.dumps(result, indent=2)}")


# ============================================
# 2. Schema Validation
# ============================================
print("\n[2] Schema Validation")
print("-" * 40)


def validate_json_schema(data: dict, schema: dict) -> tuple[bool, list[str]]:
    """Validate a JSON object against a simple schema."""
    errors = []

    # Check required fields
    for field in schema.get("required", []):
        if field not in data:
            errors.append(f"Missing required field: {field}")

    # Check types
    for field, expected_type in schema.get("types", {}).items():
        if field in data:
            if expected_type == "string" and not isinstance(data[field], str):
                errors.append(f"Field '{field}' should be string, got {type(data[field]).__name__}")
            elif expected_type == "number" and not isinstance(data[field], (int, float)):
                errors.append(f"Field '{field}' should be number, got {type(data[field]).__name__}")
            elif expected_type == "array" and not isinstance(data[field], list):
                errors.append(f"Field '{field}' should be array, got {type(data[field]).__name__}")

    # Check enum values
    for field, allowed in schema.get("enums", {}).items():
        if field in data and data[field] not in allowed:
            errors.append(f"Field '{field}' must be one of {allowed}, got '{data[field]}'")

    return len(errors) == 0, errors


schema = {
    "required": ["title", "entities", "sentiment"],
    "types": {"title": "string", "entities": "array", "sentiment": "string"},
    "enums": {"sentiment": ["positive", "negative", "neutral"]},
}

# Good data
good_data = {"title": "M4 Chip Launch", "entities": ["Apple"], "sentiment": "positive"}
valid, errs = validate_json_schema(good_data, schema)
print(f"Good data valid: {valid}")

# Bad data
bad_data = {"title": 123, "sentiment": "great"}
valid, errs = validate_json_schema(bad_data, schema)
print(f"Bad data valid: {valid}, errors: {errs}")


# ============================================
# 3. Pydantic-style Models (no dependency)
# ============================================
print("\n[3] Pydantic-style Data Models")
print("-" * 40)


@dataclass
class Entity:
    name: str
    entity_type: str  # person, organization, product
    relevance: float = 0.0

    def __post_init__(self):
        if self.entity_type not in ("person", "organization", "location", "product"):
            raise ValueError(f"Invalid entity_type: {self.entity_type}")
        if not 0.0 <= self.relevance <= 1.0:
            raise ValueError(f"Relevance must be 0-1, got {self.relevance}")


@dataclass
class DocumentExtraction:
    title: str
    summary: str
    entities: list[Entity]
    topics: list[str]
    sentiment: str
    confidence: float

    def __post_init__(self):
        if self.sentiment not in ("positive", "negative", "neutral"):
            raise ValueError(f"Invalid sentiment: {self.sentiment}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0-1, got {self.confidence}")
        if not self.entities:
            raise ValueError("Must have at least one entity")
        self.topics = [t.lower().strip() for t in self.topics]


# Create a valid extraction
extraction = DocumentExtraction(
    title="Apple M4 Chip Announcement",
    summary="Apple unveiled M4 with major performance gains.",
    entities=[
        Entity("Apple", "organization", 0.95),
        Entity("Tim Cook", "person", 0.85),
        Entity("M4", "product", 0.90),
    ],
    topics=["Technology", "Semiconductors", "Stock Market"],
    sentiment="positive",
    confidence=0.92,
)

print(f"Title: {extraction.title}")
print(f"Entities: {[(e.name, e.entity_type) for e in extraction.entities]}")
print(f"Topics: {extraction.topics}")
print(f"Sentiment: {extraction.sentiment} (confidence: {extraction.confidence})")

# Test validation
try:
    bad = DocumentExtraction("Test", "Test", [], ["t"], "bad_sentiment", 0.5)
except ValueError as e:
    print(f"\nValidation error caught: {e}")


# ============================================
# 4. Type Coercion
# ============================================
print("\n[4] Type Coercion")
print("-" * 40)


def coerce_types(data: dict, type_schema: dict[str, type]) -> dict:
    """Coerce JSON values to expected types."""
    coerced = {}
    for key, expected_type in type_schema.items():
        if key in data:
            try:
                coerced[key] = expected_type(data[key])
            except (ValueError, TypeError):
                coerced[key] = data[key]
    return coerced


raw_llm_output = {
    "price": "29.99",
    "quantity": "5",
    "name": "Widget Pro",
    "in_stock": "true",
    "rating": "4.5",
}

schema_types = {"price": float, "quantity": int, "name": str, "rating": float}
clean = coerce_types(raw_llm_output, schema_types)
print(f"Raw: {raw_llm_output}")
print(f"Coerced: {clean}")
print(f"Types: { {k: type(v).__name__ for k, v in clean.items()} }")


# ============================================
# 5. Extraction Pipeline
# ============================================
print("\n[5] Extraction Pipeline")
print("-" * 40)


@dataclass
class ExtractionResult:
    success: bool
    data: dict | None
    errors: list[str]
    retries: int


class ExtractionPipeline:
    """Multi-step extraction with validation and retry."""

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    def extract(self, text: str, required_fields: list[str],
                extractor_fn=None) -> ExtractionResult:
        """Extract with validation and retry."""
        if extractor_fn is None:
            extractor_fn = extract_entities_regex

        errors = []

        for attempt in range(self.max_retries):
            result = extractor_fn(text)

            # Validate required fields
            missing = [f for f in required_fields if f not in result or not result[f]]
            if not missing:
                return ExtractionResult(True, result, errors, attempt)

            error_msg = f"Attempt {attempt + 1}: Missing fields: {missing}"
            errors.append(error_msg)

        return ExtractionResult(False, None, errors, self.max_retries)


pipeline = ExtractionPipeline(max_retries=3)

# Successful extraction
result = pipeline.extract(
    sample_text,
    required_fields=["organizations", "people"],
)
print(f"Success: {result.success}, Retries: {result.retries}")
if result.data:
    print(f"Data: {json.dumps(result.data, indent=2)}")

# Failed extraction (looking for fields that don't exist)
result2 = pipeline.extract(
    "Just a simple sentence with no entities.",
    required_fields=["organizations", "people"],
)
print(f"\nSuccess: {result2.success}")
print(f"Errors: {result2.errors}")


# ============================================
# 6. Invoice Extraction (simulated)
# ============================================
print("\n[6] Invoice Data Extraction")
print("-" * 40)


@dataclass
class InvoiceItem:
    description: str
    quantity: int
    unit_price: float
    total: float

    def __post_init__(self):
        expected = round(self.quantity * self.unit_price, 2)
        if abs(self.total - expected) > 0.01:
            raise ValueError(
                f"Total {self.total} != quantity * unit_price = {expected}"
            )


@dataclass
class Invoice:
    invoice_number: str
    vendor: str
    items: list[InvoiceItem]
    subtotal: float
    tax: float
    total: float

    def __post_init__(self):
        items_sum = round(sum(item.total for item in self.items), 2)
        if abs(self.subtotal - items_sum) > 0.01:
            raise ValueError(f"Subtotal {self.subtotal} != sum of items {items_sum}")


invoice = Invoice(
    invoice_number="INV-2026-0042",
    vendor="Acme Cloud Services",
    items=[
        InvoiceItem("Compute instances", 10, 45.00, 450.00),
        InvoiceItem("Storage 1TB", 2, 12.50, 25.00),
        InvoiceItem("Load balancer", 1, 30.00, 30.00),
    ],
    subtotal=505.00,
    tax=42.93,
    total=547.93,
)

print(f"Invoice: {invoice.invoice_number}")
print(f"Vendor: {invoice.vendor}")
for item in invoice.items:
    print(f"  {item.description}: {item.quantity} x ${item.unit_price:.2f} = ${item.total:.2f}")
print(f"Subtotal: ${invoice.subtotal:.2f}")
print(f"Tax: ${invoice.tax:.2f}")
print(f"Total: ${invoice.total:.2f}")

# Test validation
try:
    bad_item = InvoiceItem("Bad", 2, 10.00, 25.00)  # 2*10 != 25
except ValueError as e:
    print(f"\nValidation caught: {e}")

print("\n" + "=" * 60)
print("Structured Output example complete!")
print("=" * 60)
