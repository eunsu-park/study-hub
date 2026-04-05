# 05_structured_output.py — JSON output and schema validation with Pydantic
#
# Run: python 05_structured_output.py

"""
Demonstrates:
  1. Prompting for raw JSON output
  2. Validating JSON with Pydantic models
  3. Retry-on-failure pattern for malformed responses
  4. Complex nested schema extraction
"""

import json
import os
from typing import Optional

import anthropic
from pydantic import BaseModel, Field, ValidationError

client: anthropic.Anthropic


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str = Field(pattern=r"^\d{5}$")


class ContactInfo(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None
    address: Optional[Address] = None


class BookReview(BaseModel):
    title: str
    author: str
    rating: int = Field(ge=1, le=5)
    summary: str = Field(max_length=300)
    pros: list[str]
    cons: list[str]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def call_claude(prompt: str, system: str = "") -> str:
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        temperature=0.0,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


def extract_json(text: str) -> dict:
    """Extract the first JSON object from a possibly wrapped response."""
    # Try to find JSON between code fences
    import re
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    # Fallback: parse the whole text
    return json.loads(text)


# ---------------------------------------------------------------------------
# 1. Basic JSON Output
# ---------------------------------------------------------------------------

def demo_basic_json():
    """Ask Claude to return structured JSON."""
    prompt = (
        "Extract contact information from this text and return ONLY "
        "valid JSON (no markdown fences, no explanation).\n\n"
        "Text: \"Hi, I'm Alice Chen. You can reach me at "
        "alice@example.com or call 555-0147. I live at "
        "742 Evergreen Terrace, Springfield, IL 62704.\"\n\n"
        "Schema: {name, email, phone, address: {street, city, state, zip_code}}"
    )

    print("=" * 60)
    print("SECTION 1 — Basic JSON Output")
    print("=" * 60)

    raw = call_claude(prompt)
    print(f"\nRaw response:\n  {raw}")

    data = extract_json(raw)
    contact = ContactInfo(**data)
    print(f"\nValidated: {contact.model_dump_json(indent=2)}")


# ---------------------------------------------------------------------------
# 2. Pydantic Schema in the Prompt
# ---------------------------------------------------------------------------

def demo_schema_in_prompt():
    """Embed the Pydantic schema directly so the model knows the contract."""
    schema_str = json.dumps(BookReview.model_json_schema(), indent=2)

    prompt = (
        f"Generate a book review for '1984' by George Orwell.\n\n"
        f"You MUST return ONLY a JSON object matching this schema:\n"
        f"{schema_str}\n\n"
        f"No markdown fences. No extra text."
    )

    print("\n" + "=" * 60)
    print("SECTION 2 — Pydantic Schema in Prompt")
    print("=" * 60)

    raw = call_claude(prompt)
    data = extract_json(raw)
    review = BookReview(**data)
    print(f"\nTitle:  {review.title}")
    print(f"Author: {review.author}")
    print(f"Rating: {'*' * review.rating} ({review.rating}/5)")
    print(f"Summary: {review.summary}")
    print(f"Pros:  {review.pros}")
    print(f"Cons:  {review.cons}")


# ---------------------------------------------------------------------------
# 3. Retry-on-Failure Pattern
# ---------------------------------------------------------------------------

def robust_extract(prompt: str, model_cls: type[BaseModel], retries: int = 2):
    """Try to extract and validate; on failure, send the error back."""
    system = "Always respond with valid JSON only. No explanation."
    raw = call_claude(prompt, system=system)

    for attempt in range(retries + 1):
        try:
            data = extract_json(raw)
            return model_cls(**data)
        except (json.JSONDecodeError, ValidationError) as exc:
            if attempt == retries:
                raise
            # Ask the model to fix its own output
            fix_prompt = (
                f"Your previous response was invalid:\n{raw}\n\n"
                f"Error: {exc}\n\n"
                f"Return corrected JSON only."
            )
            raw = call_claude(fix_prompt, system=system)
    return None  # unreachable


def demo_retry():
    prompt = (
        "Create a fictional contact: name, email, phone, and address "
        "(US format with 5-digit zip). Return JSON only."
    )

    print("\n" + "=" * 60)
    print("SECTION 3 — Retry-on-Failure Pattern")
    print("=" * 60)

    contact = robust_extract(prompt, ContactInfo)
    if contact:
        print(f"\n  Extracted: {contact.model_dump_json(indent=2)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: Set the ANTHROPIC_API_KEY environment variable first.")
        raise SystemExit(1)

    client = anthropic.Anthropic()

    try:
        demo_basic_json()
        demo_schema_in_prompt()
        demo_retry()
    except anthropic.APIError as exc:
        print(f"\nAPI error: {exc}")
