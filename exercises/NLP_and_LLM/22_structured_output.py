"""
Exercises for Lesson 22: Structured Output
Topic: NLP_and_LLM

Practice problems for structured data extraction and validation.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any


# === Exercise 1: Schema Validator ===
# Problem: Build a JSON schema validator that checks types, required fields,
# enum constraints, and nested objects.

def exercise_1():
    """Build a recursive JSON schema validator."""
    print("=" * 60)
    print("Exercise 1: JSON Schema Validator")
    print("=" * 60)

    def validate(data: Any, schema: dict, path: str = "") -> list[str]:
        """Validate data against a JSON-schema-like definition."""
        errors = []
        expected_type = schema.get("type")

        # TODO: Type checking
        type_map = {"string": str, "integer": int, "number": (int, float),
                     "boolean": bool, "array": list, "object": dict}
        if expected_type and expected_type in type_map:
            if not isinstance(data, type_map[expected_type]):
                errors.append(f"{path}: expected {expected_type}, got {type(data).__name__}")
                return errors

        # TODO: Enum validation
        if "enum" in schema:
            if data not in schema["enum"]:
                errors.append(f"{path}: must be one of {schema['enum']}, got '{data}'")

        # TODO: String constraints
        if expected_type == "string":
            if "minLength" in schema and len(data) < schema["minLength"]:
                errors.append(f"{path}: length {len(data)} < minLength {schema['minLength']}")
            if "maxLength" in schema and len(data) > schema["maxLength"]:
                errors.append(f"{path}: length {len(data)} > maxLength {schema['maxLength']}")

        # TODO: Number constraints
        if expected_type in ("integer", "number"):
            if "minimum" in schema and data < schema["minimum"]:
                errors.append(f"{path}: {data} < minimum {schema['minimum']}")
            if "maximum" in schema and data > schema["maximum"]:
                errors.append(f"{path}: {data} > maximum {schema['maximum']}")

        # TODO: Array validation
        if expected_type == "array" and "items" in schema:
            for i, item in enumerate(data):
                errors.extend(validate(item, schema["items"], f"{path}[{i}]"))

        # TODO: Object validation (required fields, properties)
        if expected_type == "object":
            for req in schema.get("required", []):
                if req not in data:
                    errors.append(f"{path}.{req}: required field missing")
            for prop, prop_schema in schema.get("properties", {}).items():
                if prop in data:
                    errors.extend(validate(data[prop], prop_schema, f"{path}.{prop}"))

        return errors

    # Test schema
    schema = {
        "type": "object",
        "required": ["title", "entities", "sentiment"],
        "properties": {
            "title": {"type": "string", "minLength": 1, "maxLength": 200},
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["name", "type"],
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string", "enum": ["person", "org", "product"]},
                    },
                },
            },
            "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
    }

    # Valid data
    good = {
        "title": "Apple M4 Launch",
        "entities": [{"name": "Apple", "type": "org"}, {"name": "M4", "type": "product"}],
        "sentiment": "positive",
        "confidence": 0.95,
    }
    errors = validate(good, schema)
    print(f"Valid data errors: {errors}")

    # Invalid data
    bad = {
        "title": "",
        "entities": [{"name": "Apple", "type": "company"}],  # Invalid enum
        "sentiment": "great",  # Invalid enum
        "confidence": 1.5,  # Over max
    }
    errors = validate(bad, schema)
    print(f"Invalid data errors:")
    for e in errors:
        print(f"  - {e}")


# === Exercise 2: Type Coercion Pipeline ===
# Problem: Build a pipeline that coerces LLM output types to match expected schema.

def exercise_2():
    """Build a type coercion pipeline."""
    print("\n" + "=" * 60)
    print("Exercise 2: Type Coercion Pipeline")
    print("=" * 60)

    def coerce(value: Any, target_type: str) -> tuple[Any, bool]:
        """Coerce a value to target type. Returns (value, success)."""
        # TODO: Handle various type coercions
        try:
            if target_type == "string":
                return str(value), True
            elif target_type == "integer":
                if isinstance(value, str):
                    return int(float(value)), True
                return int(value), True
            elif target_type == "number":
                return float(value), True
            elif target_type == "boolean":
                if isinstance(value, str):
                    return value.lower() in ("true", "1", "yes"), True
                return bool(value), True
            elif target_type == "array":
                if isinstance(value, str):
                    try:
                        parsed = json.loads(value)
                        if isinstance(parsed, list):
                            return parsed, True
                    except json.JSONDecodeError:
                        return value.split(","), True
                return list(value), True
        except (ValueError, TypeError):
            pass
        return value, False

    # Simulated LLM output with wrong types
    raw_output = {
        "price": "29.99",
        "quantity": "5",
        "name": 123,
        "in_stock": "true",
        "rating": "4.5",
        "tags": '["electronics", "sale"]',
    }

    expected_types = {
        "price": "number",
        "quantity": "integer",
        "name": "string",
        "in_stock": "boolean",
        "rating": "number",
        "tags": "array",
    }

    print("Coercion results:")
    for key, target in expected_types.items():
        original = raw_output.get(key)
        coerced, success = coerce(original, target)
        status = "OK" if success else "FAIL"
        print(f"  {key}: {repr(original)} -> {repr(coerced)} [{status}] (type: {type(coerced).__name__})")


# === Exercise 3: Extraction with Retry ===
# Problem: Implement an extraction pipeline that retries on validation failure,
# feeding error messages back to improve the next attempt.

def exercise_3():
    """Implement extraction with validation-based retry."""
    print("\n" + "=" * 60)
    print("Exercise 3: Extraction with Retry")
    print("=" * 60)

    @dataclass
    class ExtractionResult:
        success: bool
        data: dict | None
        errors: list[str]
        attempts: int

    # Simulated LLM that improves with error feedback
    attempt_counter = {"count": 0}

    def simulated_llm_extract(text: str, error_context: str = "") -> dict:
        """Simulated LLM that returns better output when given error feedback."""
        attempt_counter["count"] += 1
        attempt = attempt_counter["count"]

        if attempt == 1:
            # First attempt: missing fields, wrong types
            return {"title": "Product Review", "rating": "4.5"}
        elif attempt == 2:
            # Second attempt: fixes types but still has issues
            return {"title": "Product Review", "rating": 4.5, "pros": ["good quality"]}
        else:
            # Third attempt: correct
            return {
                "title": "Product Review",
                "rating": 4.5,
                "pros": ["good quality", "fast shipping"],
                "cons": ["expensive"],
                "recommendation": "recommend",
            }

    def validate_extraction(data: dict) -> list[str]:
        """Validate extracted data against requirements."""
        errors = []
        required = ["title", "rating", "pros", "cons", "recommendation"]
        for field in required:
            if field not in data:
                errors.append(f"Missing field: {field}")
        if "rating" in data and not isinstance(data["rating"], (int, float)):
            errors.append(f"Rating must be a number, got {type(data['rating']).__name__}")
        if "recommendation" in data:
            valid_recs = ["strongly_recommend", "recommend", "neutral", "not_recommend"]
            if data["recommendation"] not in valid_recs:
                errors.append(f"Recommendation must be one of {valid_recs}")
        return errors

    # TODO: Implement extraction pipeline with retry
    def extract_with_retry(text: str, max_retries: int = 3) -> ExtractionResult:
        all_errors = []

        for attempt in range(max_retries):
            error_context = "\n".join(all_errors[-3:]) if all_errors else ""
            data = simulated_llm_extract(text, error_context)
            errors = validate_extraction(data)

            if not errors:
                return ExtractionResult(True, data, all_errors, attempt + 1)

            all_errors.extend([f"Attempt {attempt + 1}: {e}" for e in errors])
            print(f"  Attempt {attempt + 1}: {len(errors)} validation error(s)")
            for e in errors:
                print(f"    - {e}")

        return ExtractionResult(False, None, all_errors, max_retries)

    attempt_counter["count"] = 0
    result = extract_with_retry("Great product, would recommend!", max_retries=3)
    print(f"\nResult: success={result.success}, attempts={result.attempts}")
    if result.data:
        print(f"Data: {json.dumps(result.data, indent=2)}")


# === Exercise 4: Nested Structure Extraction ===
# Problem: Extract a complex nested structure from meeting transcript text.

def exercise_4():
    """Extract nested structured data from text."""
    print("\n" + "=" * 60)
    print("Exercise 4: Nested Structure Extraction")
    print("=" * 60)

    transcript = """
    Meeting: Q1 Review - 2026-03-10
    Attendees: Alice, Bob, Carol, Dave

    Alice: Sprint velocity was 85%. We completed 17 of 20 stories.
    Bob: I propose migrating to Kubernetes in Q2.
    Carol: I'll own the migration plan, due March 25.
    Dave: I'll evaluate Datadog vs Grafana by March 18.

    Decision: Proceed with K8s migration starting Q2.
    Decision: Allocate $50K for infrastructure upgrade.

    Action: Carol - Migration plan by March 25
    Action: Dave - Monitoring tool evaluation by March 18

    Next meeting: March 24, 2026
    """

    # TODO: Extract structured meeting data using regex
    def extract_meeting(text: str) -> dict:
        # Extract title and date
        title_match = re.search(r"Meeting:\s*(.+?)\s*-\s*(\d{4}-\d{2}-\d{2})", text)
        title = title_match.group(1).strip() if title_match else "Unknown"
        date = title_match.group(2) if title_match else "Unknown"

        # Extract attendees
        attendees_match = re.search(r"Attendees:\s*(.+)", text)
        attendees = [a.strip() for a in attendees_match.group(1).split(",")] if attendees_match else []

        # Extract decisions
        decisions = re.findall(r"Decision:\s*(.+)", text)
        decisions = [d.strip() for d in decisions]

        # Extract action items
        actions = []
        for match in re.finditer(r"Action:\s*(\w+)\s*-\s*(.+?)(?:by\s+(.+))?$", text, re.MULTILINE):
            actions.append({
                "assignee": match.group(1).strip(),
                "task": match.group(2).strip(),
                "due_date": match.group(3).strip() if match.group(3) else None,
            })

        # Extract next meeting
        next_match = re.search(r"Next meeting:\s*(.+)", text)
        next_meeting = next_match.group(1).strip() if next_match else None

        return {
            "title": title,
            "date": date,
            "attendees": attendees,
            "decisions": decisions,
            "action_items": actions,
            "next_meeting": next_meeting,
        }

    result = extract_meeting(transcript)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
