# Exercise: Lesson 05 — Structured Output
# Complete the TODO items below.
#
# Run: python 05_structured_output.py

import anthropic
import json

client = anthropic.Anthropic()  # expects ANTHROPIC_API_KEY env var

MODEL = "claude-sonnet-4-20250514"


# === Exercise 1: Extract JSON from Text ===
# Prompt Claude to extract structured data from unstructured text.
# Hint: Specify the exact JSON schema you want in the prompt.

SAMPLE_TEXT = (
    "John Smith is a 34-year-old software engineer living in San Francisco. "
    "He has 8 years of experience and specializes in distributed systems. "
    "His email is john.smith@example.com and he speaks English and Spanish."
)

def extract_person_json(text: str) -> dict:
    """Extract structured person data from free text.
    Expected keys: name, age, occupation, city, experience_years,
                   specialization, email, languages (list).
    """
    # TODO: Build a prompt that asks Claude to extract the fields above
    # TODO: Instruct Claude to reply with ONLY valid JSON (no markdown fences)
    # TODO: Parse the response with json.loads()
    pass


def exercise_1():
    result = extract_person_json(SAMPLE_TEXT)
    assert isinstance(result, dict), "Must return a dict"
    required = {"name", "age", "occupation", "city", "email", "languages"}
    missing = required - set(result.keys())
    assert not missing, f"Missing keys: {missing}"
    assert isinstance(result["languages"], list), "languages must be a list"
    print(f"[Ex1] Extracted: {json.dumps(result, indent=2)}")


# === Exercise 2: JSON Schema Validation ===
# Validate extracted JSON against a predefined schema.
# This is a pure Python exercise (no API call needed).

PERSON_SCHEMA = {
    "name": str,
    "age": int,
    "occupation": str,
    "city": str,
    "experience_years": int,
    "email": str,
    "languages": list,
}

def validate_schema(data: dict, schema: dict) -> list[str]:
    """Validate that `data` conforms to `schema`.
    Return a list of error strings (empty if valid).
    """
    # TODO: Check for missing keys
    # TODO: Check for type mismatches
    # TODO: Return list of error descriptions
    # Example error: "Missing key: email" or "Type mismatch: age expected int, got str"
    pass


def exercise_2():
    good_data = {
        "name": "Jane", "age": 28, "occupation": "designer",
        "city": "NYC", "experience_years": 5, "email": "j@ex.com",
        "languages": ["English"],
    }
    bad_data = {
        "name": "Jane", "age": "twenty-eight", "city": "NYC",
        "languages": "English",
    }
    errors_good = validate_schema(good_data, PERSON_SCHEMA)
    errors_bad = validate_schema(bad_data, PERSON_SCHEMA)
    assert len(errors_good) == 0, f"Unexpected errors: {errors_good}"
    assert len(errors_bad) >= 2, f"Expected errors for bad data"
    print(f"[Ex2] Good data errors: {errors_good}")
    print(f"[Ex2] Bad data errors:  {errors_bad}")


# === Exercise 3: Extract a List of Items ===
# Extract multiple structured items from a block of text.
# Hint: Ask Claude to return a JSON array.

PRODUCT_TEXT = (
    "Our store offers: 1) Wireless Mouse ($29.99, in stock, 4.5 stars), "
    "2) Mechanical Keyboard ($89.50, in stock, 4.8 stars), "
    "3) USB-C Hub ($45.00, out of stock, 4.2 stars), "
    "4) Monitor Stand ($34.99, in stock, 4.0 stars)."
)

def extract_products(text: str) -> list[dict]:
    """Extract product list from text.
    Each product dict: {name, price, in_stock (bool), rating (float)}.
    """
    # TODO: Prompt Claude to extract all products as a JSON array
    # TODO: Parse and return the list of product dicts
    pass


def exercise_3():
    products = extract_products(PRODUCT_TEXT)
    assert isinstance(products, list), "Must return a list"
    assert len(products) == 4, f"Expected 4 products, got {len(products)}"
    for p in products:
        assert "name" in p and "price" in p and "in_stock" in p
        print(f"[Ex3] {p['name']:>20} | ${p['price']:>6} | "
              f"stock={p['in_stock']} | {p.get('rating', 'N/A')} stars")


# === Exercise 4: Graceful Error Handling ===
# Handle cases where Claude's JSON output is malformed.
# Hint: Use try/except and implement a retry or fixup strategy.

def safe_json_extract(text: str, schema_hint: str,
                      max_retries: int = 2) -> dict | None:
    """Try to extract JSON from text with retries on parse failure.
    Args:
        text: the source text to extract from
        schema_hint: description of expected JSON structure
        max_retries: number of retry attempts if JSON parsing fails
    Returns the parsed dict, or None if all retries fail.
    """
    # TODO: Attempt to extract JSON via the API
    # TODO: If json.loads() fails, retry with a prompt that includes
    #       the malformed output and asks Claude to fix it
    # TODO: Return None after max_retries exhausted
    # Hint: Strip markdown code fences (```json ... ```) before parsing
    pass


def exercise_4():
    result = safe_json_extract(
        "Alice is 30, a teacher in Boston, email alice@school.edu.",
        schema_hint="name (str), age (int), occupation (str), city (str), email (str)",
    )
    if result:
        print(f"[Ex4] Extracted: {result}")
        assert "name" in result
    else:
        print("[Ex4] Extraction failed after retries")


# === Exercise 5: Structured Output with Enum Constraints ===
# Extract data where certain fields must be from a fixed set.
# Hint: List the allowed values explicitly in the prompt.

ALLOWED_CATEGORIES = ["electronics", "clothing", "food", "furniture", "other"]
ALLOWED_URGENCY = ["low", "medium", "high"]

def classify_support_ticket(ticket_text: str) -> dict:
    """Classify a support ticket into structured fields.
    Return: {
        "summary": str (max 20 words),
        "category": str (one of ALLOWED_CATEGORIES),
        "urgency": str (one of ALLOWED_URGENCY),
        "action_items": list[str]
    }
    """
    # TODO: Build a prompt that specifies the allowed enum values
    # TODO: Extract and validate the response
    # TODO: If category or urgency is invalid, default to "other" / "medium"
    pass


def exercise_5():
    tickets = [
        "My laptop screen cracked after one day! I need a replacement ASAP.",
        "The jacket I ordered is the wrong size, would like to exchange it.",
        "Just wondering when the new menu items will be available.",
    ]
    for ticket in tickets:
        result = classify_support_ticket(ticket)
        assert result["category"] in ALLOWED_CATEGORIES
        assert result["urgency"] in ALLOWED_URGENCY
        assert isinstance(result["action_items"], list)
        print(f"[Ex5] {result['urgency']:>6} | {result['category']:>12} | "
              f"{result['summary'][:50]}")


if __name__ == "__main__":
    print("=== Exercise 1: Extract JSON from Text ===")
    exercise_1()

    print("\n=== Exercise 2: JSON Schema Validation (no API) ===")
    exercise_2()

    print("\n=== Exercise 3: Extract a List of Items ===")
    exercise_3()

    print("\n=== Exercise 4: Graceful Error Handling ===")
    exercise_4()

    print("\n=== Exercise 5: Enum-Constrained Classification ===")
    exercise_5()

    print("\nAll exercises completed!")
