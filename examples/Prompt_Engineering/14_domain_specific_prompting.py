# 14_domain_specific_prompting.py — Data extraction, summarization, translation prompts
#
# Run: python 14_domain_specific_prompting.py

import anthropic
import json

# ---------------------------------------------------------------------------
# 1. Domain: Structured Data Extraction
# ---------------------------------------------------------------------------
EXTRACTION_SYSTEM = """\
You are a structured data extraction engine. Given raw text, extract the \
requested fields into JSON. Rules:
- Output ONLY valid JSON, no commentary.
- Use null for missing fields.
- Normalize dates to YYYY-MM-DD format.
- Normalize currency to numeric values (no symbols)."""

EXTRACTION_CASES = [
    {
        "label": "Invoice extraction",
        "schema": '{"vendor": str, "date": str, "total": float, "currency": str, "items": [{"name": str, "qty": int, "price": float}]}',
        "text": (
            "Invoice #2024-0891 from TechSupply Co.\n"
            "Date: March 15, 2024\n"
            "Items:\n"
            "  - USB-C Hub x3 @ $29.99 each\n"
            "  - Webcam HD Pro x1 @ $89.50\n"
            "Total: $179.47 USD"
        ),
    },
    {
        "label": "Contact extraction",
        "schema": '{"name": str, "email": str, "phone": str, "company": str, "role": str}',
        "text": (
            "Hi, I'm Dr. Sarah Chen, Chief Data Officer at NovaTech Solutions. "
            "You can reach me at s.chen@novatech.io or (415) 555-0192."
        ),
    },
]


# ---------------------------------------------------------------------------
# 2. Domain: Technical Summarization
# ---------------------------------------------------------------------------
SUMMARIZE_SYSTEM = """\
You are a technical writer. Summarize the input for a software engineering audience.

Format:
- **TL;DR**: One sentence.
- **Key Points**: 3-5 bullet points.
- **Impact**: Who is affected and how."""

SUMMARIZE_TEXT = (
    "We are migrating our authentication service from a monolithic Django app to a "
    "dedicated microservice using Go and gRPC. The new service will use JWT tokens "
    "with RS256 signing, replacing the current session-based auth. OAuth2 providers "
    "(Google, GitHub) will be supported via an adapter layer. The migration will "
    "happen in three phases: shadow mode (both systems active), validation (compare "
    "responses), and cutover. Expected timeline: 8 weeks. All client SDKs will need "
    "updates. The legacy session endpoint will remain active for 90 days post-cutover."
)


# ---------------------------------------------------------------------------
# 3. Domain: Translation with style control
# ---------------------------------------------------------------------------
TRANSLATE_SYSTEM = """\
You are a professional translator. Translate the text into {target_language}.

Style guidelines:
- Register: {register}
- Preserve technical terms in parentheses when no standard translation exists.
- Preserve all formatting (bullet points, numbered lists, etc.).
- Output ONLY the translation, no commentary."""

TRANSLATE_CASES = [
    {
        "label": "Technical docs -> Korean (formal)",
        "target_language": "Korean",
        "register": "formal/written (합쇼체)",
        "text": (
            "To deploy the application:\n"
            "1. Run `docker build -t myapp .`\n"
            "2. Push the image to the registry.\n"
            "3. Update the Kubernetes manifest and apply."
        ),
    },
    {
        "label": "Marketing copy -> Spanish (casual)",
        "target_language": "Spanish",
        "register": "casual/friendly (tuteo)",
        "text": (
            "Say goodbye to slow dashboards! Our new analytics engine "
            "processes millions of events in real time. Try it free for 14 days."
        ),
    },
]


# ---------------------------------------------------------------------------
# 4. Helper: call the API and return text
# ---------------------------------------------------------------------------
def call_api(
    client: anthropic.Anthropic,
    system: str,
    user_message: str,
    max_tokens: int = 512,
) -> str:
    resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_message}],
    )
    return resp.content[0].text


# ---------------------------------------------------------------------------
# 5. Main — demonstrate all three domain prompts
# ---------------------------------------------------------------------------
def main() -> None:
    client = anthropic.Anthropic()

    # --- Data Extraction ---
    print("=" * 60)
    print("DOMAIN 1: STRUCTURED DATA EXTRACTION")
    print("=" * 60)
    for case in EXTRACTION_CASES:
        print(f"\n--- {case['label']} ---")
        user_msg = f"Schema: {case['schema']}\n\nText:\n{case['text']}"
        try:
            result = call_api(client, EXTRACTION_SYSTEM, user_msg)
            print(f"  Extracted JSON:\n  {result[:300]}")
            # Validate it is parseable JSON
            parsed = json.loads(result)
            print(f"  [Valid JSON with {len(parsed)} top-level keys]")
        except json.JSONDecodeError:
            print("  [Warning: response was not valid JSON]")
        except anthropic.APIError as exc:
            print(f"  [API Error] {exc}")

    # --- Technical Summarization ---
    print("\n" + "=" * 60)
    print("DOMAIN 2: TECHNICAL SUMMARIZATION")
    print("=" * 60)
    try:
        summary = call_api(client, SUMMARIZE_SYSTEM, SUMMARIZE_TEXT)
        print(f"\n{summary}")
    except anthropic.APIError as exc:
        print(f"  [API Error] {exc}")

    # --- Translation ---
    print("\n" + "=" * 60)
    print("DOMAIN 3: TRANSLATION WITH STYLE CONTROL")
    print("=" * 60)
    for case in TRANSLATE_CASES:
        print(f"\n--- {case['label']} ---")
        system = TRANSLATE_SYSTEM.format(
            target_language=case["target_language"],
            register=case["register"],
        )
        try:
            translation = call_api(client, system, case["text"])
            print(f"  Original : {case['text'][:80]}...")
            print(f"  Translated:\n  {translation[:300]}")
        except anthropic.APIError as exc:
            print(f"  [API Error] {exc}")


if __name__ == "__main__":
    main()
