# 14. Domain-Specific Prompting

**Previous**: [13. Adversarial Prompting](./13_Adversarial_Prompting.md) | **Next**: [15. Prompt Management in Production](./15_Prompt_Management_in_Production.md)

## Learning Objectives

- Design extraction prompts that reliably pull structured data from unstructured text
- Apply summarization strategies with controllable length, style, and faithfulness
- Build domain-adapted prompts for legal, medical, financial, and educational contexts
- Implement translation prompts with terminology control and style preservation
- Create Socratic and scaffolded educational prompts that guide learners effectively

---

General-purpose prompting techniques get you 80% of the way, but domain-specific applications demand specialized patterns. A medical summarizer needs different safeguards than a creative writing assistant. A legal document analyzer requires precision that a casual chatbot does not. This lesson covers prompting patterns tailored to specific professional domains — data extraction, text analysis, summarization, translation, education, and creative writing. Each domain introduces unique challenges around accuracy, tone, terminology, and safety that generic prompts cannot address.

## Table of Contents
1. [Data Extraction Prompting](#1-data-extraction-prompting)
2. [Text Analysis and Classification](#2-text-analysis-and-classification)
3. [Summarization Strategies](#3-summarization-strategies)
4. [Translation Prompting](#4-translation-prompting)
5. [Educational Prompting](#5-educational-prompting)
6. [Legal Domain Prompting](#6-legal-domain-prompting)
7. [Medical and Financial Domain Prompting](#7-medical-and-financial-domain-prompting)
8. [Creative Writing Prompts](#8-creative-writing-prompts)
9. [Research Assistance Prompts](#9-research-assistance-prompts)

---

## 1. Data Extraction Prompting

Data extraction transforms unstructured text into structured, machine-readable formats. The key challenge is reliability — extraction prompts must handle messy, inconsistent real-world text.

### 1.1 Basic Extraction Pattern

```python
import anthropic
import json

client = anthropic.Anthropic()


def extract_contact_info(text: str) -> dict:
    """Extract contact information from unstructured text."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system="""You are a data extraction specialist. Extract structured information
from text with high precision.

RULES:
- Extract ONLY information explicitly stated in the text.
- Use null for any field not found in the text.
- Never infer or guess values.
- If a field is ambiguous, use the most likely interpretation and add a
  "confidence" field with value "low".
""",
        messages=[
            {
                "role": "user",
                "content": f"""Extract contact information from this text and return JSON:

Text: {text}

Return this exact JSON structure:
{{
  "name": "full name or null",
  "email": "email address or null",
  "phone": "phone number or null",
  "company": "company name or null",
  "title": "job title or null",
  "address": "full address or null"
}}""",
            }
        ],
    )
    return json.loads(response.content[0].text)


# Test with messy real-world text
business_card_text = """
Hi there! I'm Sarah Chen, VP of Engineering at DataFlow Inc.
You can reach me at sarah.chen@dataflow.example.com or call
my office at (415) 555-0142. We're located at
450 Market Street, Suite 300, San Francisco CA 94105.
"""

result = extract_contact_info(business_card_text)
print(json.dumps(result, indent=2))
```

### 1.2 Multi-Entity Extraction

When extracting multiple entities from a document, use a schema-driven approach:

```python
import anthropic
import json

client = anthropic.Anthropic()


INVOICE_EXTRACTION_PROMPT = """You are an invoice data extraction system.
Extract ALL line items and metadata from the invoice text.

Return this exact JSON structure:
{
  "invoice_number": "string or null",
  "date": "YYYY-MM-DD format or null",
  "vendor": {
    "name": "string or null",
    "address": "string or null"
  },
  "customer": {
    "name": "string or null",
    "address": "string or null"
  },
  "line_items": [
    {
      "description": "string",
      "quantity": number,
      "unit_price": number,
      "total": number
    }
  ],
  "subtotal": number_or_null,
  "tax": number_or_null,
  "total": number_or_null,
  "currency": "3-letter currency code or USD"
}

RULES:
- Extract numbers as numeric types, not strings.
- Normalize dates to YYYY-MM-DD format.
- If quantity or unit_price is not explicit, calculate from the other fields.
- Preserve exact descriptions without paraphrasing.
"""


def extract_invoice(invoice_text: str) -> dict:
    """Extract structured data from an invoice."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system=INVOICE_EXTRACTION_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Extract data from this invoice:\n\n{invoice_text}",
            }
        ],
    )
    return json.loads(response.content[0].text)


invoice = """
INVOICE #INV-2025-0847
Date: March 15, 2025

FROM: TechSupply Co.
123 Industrial Ave, Austin TX 78701

TO: StartupXYZ
456 Innovation Blvd, San Jose CA 95113

ITEMS:
- Dell Monitor 27" 4K (x3) .............. $449.99 each
- Logitech MX Keys keyboard (x5) ........ $99.99 each
- USB-C Hub 7-port (x5) ................. $39.99 each

Subtotal: $2,049.87
Tax (8.25%): $169.11
TOTAL DUE: $2,218.98
"""

result = extract_invoice(invoice)
print(json.dumps(result, indent=2))
```

### 1.3 Extraction with Validation

```python
import anthropic
import json
from dataclasses import dataclass

client = anthropic.Anthropic()


@dataclass
class ExtractionField:
    name: str
    field_type: str  # "string", "number", "date", "email", "phone"
    required: bool = False
    pattern: str | None = None  # regex pattern for validation


class ValidatedExtractor:
    """Extract and validate data from text using schema-driven prompts."""

    def __init__(self, fields: list[ExtractionField]):
        self.fields = fields

    def _build_schema_description(self) -> str:
        lines = []
        for f in self.fields:
            req = "REQUIRED" if f.required else "optional"
            pattern_hint = f" (format: {f.pattern})" if f.pattern else ""
            lines.append(f'  "{f.name}": {f.field_type} ({req}){pattern_hint}')
        return "{\n" + ",\n".join(lines) + "\n}"

    def extract(self, text: str) -> tuple[dict, list[str]]:
        """Extract data and return (data, validation_errors)."""
        schema = self._build_schema_description()

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=(
                "You are a precise data extractor. Extract ONLY information "
                "explicitly stated in the text. Use null for missing fields. "
                "Return valid JSON only."
            ),
            messages=[
                {
                    "role": "user",
                    "content": f"Extract data matching this schema:\n{schema}\n\nFrom this text:\n{text}",
                }
            ],
        )

        data = json.loads(response.content[0].text)

        # Validate extracted data
        errors = []
        import re
        for field in self.fields:
            value = data.get(field.name)
            if field.required and value is None:
                errors.append(f"Required field '{field.name}' is missing")
            if value is not None and field.pattern:
                if not re.match(field.pattern, str(value)):
                    errors.append(
                        f"Field '{field.name}' value '{value}' "
                        f"does not match pattern '{field.pattern}'"
                    )

        return data, errors


# Define extraction schema for job postings
job_extractor = ValidatedExtractor(fields=[
    ExtractionField("title", "string", required=True),
    ExtractionField("company", "string", required=True),
    ExtractionField("location", "string", required=False),
    ExtractionField("salary_min", "number", required=False),
    ExtractionField("salary_max", "number", required=False),
    ExtractionField("experience_years", "number", required=False),
    ExtractionField("remote", "string", required=False, pattern=r"^(yes|no|hybrid)$"),
])

job_posting = """
Senior Backend Engineer at QuantumLeap Technologies
Location: New York, NY (Hybrid - 3 days in office)
Salary: $180,000 - $240,000/year
Requirements: 5+ years of experience with Python, Go, or Rust.
Strong background in distributed systems.
"""

data, errors = job_extractor.extract(job_posting)
print("Extracted:", json.dumps(data, indent=2))
print("Validation errors:", errors)
```

---

## 2. Text Analysis and Classification

Text analysis prompts classify, categorize, or score text along defined dimensions.

### 2.1 Multi-Label Classification

```python
import anthropic
import json

client = anthropic.Anthropic()


CLASSIFICATION_PROMPT = """You are a text classification system. Classify the given text
into one or more categories.

CATEGORIES:
- technical: Technical content about software, hardware, or engineering
- business: Business strategy, finance, management, operations
- scientific: Scientific research, methodology, findings
- opinion: Subjective views, editorials, reviews
- news: Current events, factual reporting
- tutorial: How-to content, instructional material

RULES:
- Assign ALL applicable categories (multi-label).
- Include a confidence score (0.0-1.0) for each assigned category.
- Only assign categories with confidence >= 0.3.
- Return JSON array of objects.

RESPONSE FORMAT:
[
  {"category": "string", "confidence": 0.0-1.0, "evidence": "brief quote from text"}
]
"""


def classify_text(text: str) -> list[dict]:
    """Classify text into multiple categories with confidence scores."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        system=CLASSIFICATION_PROMPT,
        messages=[
            {"role": "user", "content": f"Classify this text:\n\n{text}"},
        ],
    )
    return json.loads(response.content[0].text)


article = """
A new study published in Nature demonstrates that transformer-based models
can predict protein folding structures with 95% accuracy, potentially
revolutionizing drug discovery. The researchers at DeepMind used a novel
training approach combining self-supervised learning with reinforcement
learning. Industry analysts predict this could save pharmaceutical
companies billions in R&D costs.
"""

classifications = classify_text(article)
for c in classifications:
    print(f"  {c['category']}: {c['confidence']:.1f} — \"{c['evidence']}\"")
```

### 2.2 Sentiment Analysis with Aspect Extraction

```python
import anthropic
import json

client = anthropic.Anthropic()


ASPECT_SENTIMENT_PROMPT = """You are a sentiment analysis system that identifies specific
aspects (features, topics) mentioned in reviews and assigns sentiment to each.

For each aspect found:
1. Identify the aspect (e.g., "battery life", "customer service", "price")
2. Determine sentiment: positive, negative, neutral, mixed
3. Extract the relevant quote
4. Assign intensity: 1 (mild) to 5 (extreme)

Return JSON:
{
  "overall_sentiment": "positive|negative|neutral|mixed",
  "overall_score": -1.0 to 1.0,
  "aspects": [
    {
      "aspect": "string",
      "sentiment": "positive|negative|neutral|mixed",
      "intensity": 1-5,
      "quote": "relevant text"
    }
  ]
}
"""


def analyze_sentiment(review: str) -> dict:
    """Perform aspect-level sentiment analysis on a review."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=ASPECT_SENTIMENT_PROMPT,
        messages=[
            {"role": "user", "content": f"Analyze this review:\n\n{review}"},
        ],
    )
    return json.loads(response.content[0].text)


review = """
I've been using the XPhone Pro for three months now. The camera is absolutely
stunning — night mode produces better photos than my DSLR. Battery life is
decent, lasting about a day with heavy use, though I wish it were better.
The price tag of $1,299 is hard to justify. Customer support was helpful
when I had a screen issue, but the wait time was ridiculous (45 minutes).
The software is buttery smooth, and I love the new gesture navigation.
"""

result = analyze_sentiment(review)
print(f"Overall: {result['overall_sentiment']} ({result['overall_score']})")
for aspect in result["aspects"]:
    icon = {"positive": "+", "negative": "-", "neutral": "~", "mixed": "?"}
    print(
        f"  [{icon[aspect['sentiment']]}] {aspect['aspect']} "
        f"(intensity: {aspect['intensity']}/5): \"{aspect['quote'][:60]}...\""
    )
```

### 2.3 Intent Detection for Chatbots

```python
import anthropic
import json

client = anthropic.Anthropic()


INTENT_DETECTION_PROMPT = """You are an intent classifier for a customer support system.

INTENTS:
- order_status: Checking on an existing order
- return_request: Wanting to return a product
- product_question: Asking about product features or availability
- billing_issue: Questions about charges, payments, refunds
- technical_support: Technical problems with a product
- complaint: Expressing dissatisfaction
- general_inquiry: Other questions
- greeting: Hello, hi, etc.
- farewell: Goodbye, thanks, etc.

RULES:
- Detect the PRIMARY intent and up to 2 SECONDARY intents.
- Extract key entities (order numbers, product names, dates).
- Determine urgency: low, medium, high, critical.

Return JSON:
{
  "primary_intent": "string",
  "secondary_intents": ["string"],
  "entities": {"entity_type": "value"},
  "urgency": "low|medium|high|critical",
  "suggested_routing": "department or queue name"
}
"""


def detect_intent(message: str) -> dict:
    """Detect user intent and extract entities from a support message."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        system=INTENT_DETECTION_PROMPT,
        messages=[
            {"role": "user", "content": f"Classify this message:\n\n{message}"},
        ],
    )
    return json.loads(response.content[0].text)


messages = [
    "Where is my order #ABC-12345? It was supposed to arrive yesterday!",
    "Hi, do you have the new laptop in blue? What's the battery life like?",
    "I was charged twice for order #XYZ-789. I need a refund ASAP.",
]

for msg in messages:
    result = detect_intent(msg)
    print(f"\nMessage: {msg[:60]}...")
    print(f"  Intent: {result['primary_intent']}")
    print(f"  Entities: {result['entities']}")
    print(f"  Urgency: {result['urgency']}")
    print(f"  Route to: {result['suggested_routing']}")
```

---

## 3. Summarization Strategies

Summarization is one of the most common LLM tasks, but getting consistent, high-quality summaries requires careful prompt engineering.

### 3.1 Extractive vs. Abstractive Cues

```python
import anthropic

client = anthropic.Anthropic()


def extractive_summary(text: str, num_sentences: int = 5) -> str:
    """Generate an extractive summary — selecting key sentences from the source."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=f"""You are an extractive summarizer. Your task is to select the
{num_sentences} most important sentences from the text VERBATIM.

RULES:
- Select sentences exactly as they appear in the original text.
- Do NOT paraphrase, combine, or modify sentences.
- Order selected sentences by their importance, most important first.
- Each selected sentence should cover a distinct point.
- Prefix each with its position number from the original text.
""",
        messages=[
            {"role": "user", "content": f"Select the top {num_sentences} sentences:\n\n{text}"},
        ],
    )
    return response.content[0].text


def abstractive_summary(text: str, style: str = "professional", max_words: int = 100) -> str:
    """Generate an abstractive summary — synthesizing a new condensed version."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        system=f"""You are an abstractive summarizer. Synthesize the key points into
a new, concise summary.

STYLE: {style}
MAX LENGTH: {max_words} words

RULES:
- Write a new, coherent paragraph — do not simply list sentences.
- Capture ALL major points but omit minor details.
- Use clear, {style} language.
- Stay within the word limit.
- Never add information not present in the original text.
- Start directly with the content — no preamble like "This text discusses..."
""",
        messages=[
            {"role": "user", "content": f"Summarize this text:\n\n{text}"},
        ],
    )
    return response.content[0].text


article = """
The European Union has approved a landmark artificial intelligence regulation,
making it the first major jurisdiction to create comprehensive AI governance.
The AI Act establishes a risk-based framework that classifies AI systems into
four tiers: unacceptable risk (banned), high risk (heavily regulated),
limited risk (transparency requirements), and minimal risk (no restrictions).

High-risk applications include AI used in law enforcement, healthcare,
education, and employment decisions. These systems must undergo conformity
assessments, maintain human oversight, and provide detailed documentation.
Companies face fines of up to 35 million euros or 7% of global revenue.

The regulation takes effect in phases over 24 months, with bans on
unacceptable-risk systems starting immediately. Industry groups have
expressed concern about compliance costs, while civil society organizations
have praised the protections for fundamental rights.
"""

print("=== EXTRACTIVE SUMMARY ===")
print(extractive_summary(article, num_sentences=3))

print("\n=== ABSTRACTIVE SUMMARY (professional) ===")
print(abstractive_summary(article, style="professional", max_words=80))

print("\n=== ABSTRACTIVE SUMMARY (casual) ===")
print(abstractive_summary(article, style="casual and accessible", max_words=60))
```

### 3.2 Length Control Strategies

```python
import anthropic

client = anthropic.Anthropic()


def controlled_summary(
    text: str,
    target_length: str = "medium",
    format_type: str = "paragraph",
) -> str:
    """Generate a summary with precise length and format control."""
    length_specs = {
        "tweet": "Under 280 characters. Single sentence.",
        "short": "2-3 sentences. Under 75 words.",
        "medium": "4-6 sentences. 100-150 words.",
        "long": "2-3 paragraphs. 200-300 words.",
        "executive": "3-5 bullet points, each 1-2 sentences.",
    }

    format_specs = {
        "paragraph": "Write as flowing prose paragraphs.",
        "bullets": "Use bullet points with brief explanations.",
        "numbered": "Use a numbered list of key points.",
        "tldr": "Start with 'TL;DR:' followed by the core message.",
        "headline": "Write as a news headline plus one-sentence subheadline.",
    }

    spec = length_specs.get(target_length, length_specs["medium"])
    fmt = format_specs.get(format_type, format_specs["paragraph"])

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=f"""Summarize the given text with these constraints:
LENGTH: {spec}
FORMAT: {fmt}

Be precise about length. Count your output mentally before responding.
Never add information not in the source text.
Start directly with the summary — no preamble.""",
        messages=[
            {"role": "user", "content": f"Summarize:\n\n{text}"},
        ],
    )
    return response.content[0].text


# Test with different length/format combinations
text = """
SpaceX successfully launched its Starship rocket on its fifth test flight,
achieving full stage separation and a controlled landing of the Super Heavy
booster at the launch site using the mechanical arm catch system. The upper
stage reached orbital velocity before performing a controlled deorbit burn
over the Indian Ocean. This marks a significant milestone in the development
of the largest and most powerful rocket ever built, which SpaceX plans to
use for Moon and Mars missions. NASA has selected Starship as the human
landing system for the Artemis III mission.
"""

for length in ["tweet", "short", "executive"]:
    print(f"\n=== {length.upper()} ===")
    print(controlled_summary(text, target_length=length))
```

### 3.3 Faithfulness-Preserving Summarization

```python
import anthropic
import json

client = anthropic.Anthropic()


def faithful_summary(text: str) -> dict:
    """Generate a summary with faithfulness verification."""
    # Step 1: Generate the summary
    summary_response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        system="""Summarize the text faithfully. Every claim in your summary must be
directly supported by the source text. Do not add interpretations,
implications, or external knowledge.""",
        messages=[
            {"role": "user", "content": f"Summarize:\n\n{text}"},
        ],
    )
    summary = summary_response.content[0].text

    # Step 2: Verify faithfulness with a separate call
    verify_response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system="""You are a faithfulness checker. Compare a summary against the source
text and identify any claims in the summary that are NOT supported by the source.

Return JSON:
{
  "faithful": true/false,
  "unsupported_claims": ["list of claims not in source"],
  "missing_key_points": ["important points from source not in summary"],
  "score": 0.0-1.0
}""",
        messages=[
            {
                "role": "user",
                "content": f"SOURCE:\n{text}\n\nSUMMARY:\n{summary}",
            }
        ],
    )
    verification = json.loads(verify_response.content[0].text)

    return {
        "summary": summary,
        "verification": verification,
    }


result = faithful_summary("""
Researchers at MIT have developed a new battery technology using aluminum
and sulfur that costs approximately one-sixth the price of comparable
lithium-ion batteries. The batteries can be fully charged in under one
minute and have survived hundreds of charge-discharge cycles without
significant degradation. However, they operate at elevated temperatures
(around 110 degrees Celsius) and are intended for stationary grid storage
rather than portable electronics or electric vehicles.
""")

print("Summary:", result["summary"])
print(f"\nFaithfulness score: {result['verification']['score']}")
print(f"Faithful: {result['verification']['faithful']}")
if result["verification"]["unsupported_claims"]:
    print(f"Unsupported claims: {result['verification']['unsupported_claims']}")
```

---

## 4. Translation Prompting

Translation with LLMs goes beyond word-for-word conversion. Effective translation prompts manage terminology, register, cultural adaptation, and domain-specific vocabulary.

### 4.1 Translation with Terminology Control

```python
import anthropic

client = anthropic.Anthropic()


def translate_with_glossary(
    text: str,
    source_lang: str,
    target_lang: str,
    glossary: dict[str, str],
    style: str = "formal",
) -> str:
    """Translate text using a controlled terminology glossary."""
    glossary_text = "\n".join(f"  - \"{k}\" → \"{v}\"" for k, v in glossary.items())

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system=f"""You are a professional translator ({source_lang} → {target_lang}).

STYLE: {style}

MANDATORY GLOSSARY (always use these exact translations):
{glossary_text}

RULES:
1. Use glossary terms EXACTLY as specified — never deviate.
2. Maintain the original text's formatting (paragraphs, lists, emphasis).
3. Preserve technical accuracy over natural flow when they conflict.
4. Keep proper nouns, brand names, and code examples in their original form.
5. Match the register (formal/informal) of the source unless instructed otherwise.
6. If a glossary term appears in a different grammatical form, adapt the
   translation appropriately while keeping the root term from the glossary.
""",
        messages=[
            {
                "role": "user",
                "content": f"Translate the following {source_lang} text to {target_lang}:\n\n{text}",
            }
        ],
    )
    return response.content[0].text


# Technical translation with glossary
tech_text = """
The machine learning pipeline processes the training data through a
feature engineering stage before feeding it into the neural network.
The model uses gradient descent with backpropagation to optimize the
loss function. After training, the model is deployed to production
using a containerized microservice architecture.
"""

glossary = {
    "machine learning": "기계 학습",
    "pipeline": "파이프라인",
    "training data": "훈련 데이터",
    "feature engineering": "특성 공학",
    "neural network": "신경망",
    "gradient descent": "경사 하강법",
    "backpropagation": "역전파",
    "loss function": "손실 함수",
    "microservice": "마이크로서비스",
}

translation = translate_with_glossary(
    tech_text,
    source_lang="English",
    target_lang="Korean",
    glossary=glossary,
    style="formal technical",
)
print(translation)
```

### 4.2 Style-Preserving Translation

```python
import anthropic

client = anthropic.Anthropic()


def style_preserving_translation(
    text: str,
    source_lang: str,
    target_lang: str,
    preserve: list[str] | None = None,
) -> dict:
    """Translate while preserving specific stylistic elements."""
    preserve = preserve or ["tone", "sentence_structure", "emphasis"]

    preservation_instructions = {
        "tone": "Maintain the same emotional tone (formal, casual, humorous, serious).",
        "sentence_structure": "Mirror the sentence lengths and complexity of the original.",
        "emphasis": "Preserve emphasis markers (bold, italics, exclamations, repetition).",
        "rhythm": "Maintain the cadence and rhythm of the prose.",
        "register": "Keep the same level of formality and social register.",
        "metaphors": "Translate metaphors to culturally equivalent ones, not literally.",
    }

    instructions = "\n".join(
        f"- {preservation_instructions[p]}"
        for p in preserve
        if p in preservation_instructions
    )

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system=f"""You are a literary translator specializing in style preservation.

STYLE PRESERVATION RULES:
{instructions}

Translate from {source_lang} to {target_lang}.
After the translation, provide a brief note on any significant
stylistic choices you made.

Format:
TRANSLATION:
[your translation]

TRANSLATOR'S NOTES:
[brief notes on stylistic choices]
""",
        messages=[
            {"role": "user", "content": f"Translate:\n\n{text}"},
        ],
    )
    raw = response.content[0].text

    # Parse the response
    if "TRANSLATOR'S NOTES:" in raw:
        parts = raw.split("TRANSLATOR'S NOTES:")
        translation = parts[0].replace("TRANSLATION:", "").strip()
        notes = parts[1].strip()
    else:
        translation = raw.replace("TRANSLATION:", "").strip()
        notes = ""

    return {"translation": translation, "notes": notes}


result = style_preserving_translation(
    text=(
        "The old man sat by the window, watching the rain paint "
        "silver streaks on the glass. He remembered — oh, how he "
        "remembered — the summers of his youth, when the world "
        "was green and endless and impossibly beautiful."
    ),
    source_lang="English",
    target_lang="Spanish",
    preserve=["tone", "rhythm", "metaphors"],
)
print("Translation:", result["translation"])
print("Notes:", result["notes"])
```

### 4.3 Back-Translation for Quality Assurance

```python
import anthropic

client = anthropic.Anthropic()


def translate_with_backtranslation(
    text: str,
    source_lang: str,
    target_lang: str,
) -> dict:
    """Translate and verify using back-translation."""
    # Forward translation
    forward = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system=f"Translate from {source_lang} to {target_lang}. Output only the translation.",
        messages=[{"role": "user", "content": text}],
    )
    translation = forward.content[0].text

    # Back-translation
    backward = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system=f"Translate from {target_lang} to {source_lang}. Output only the translation.",
        messages=[{"role": "user", "content": translation}],
    )
    back_translation = backward.content[0].text

    # Compare original and back-translation
    comparison = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system="""Compare the original text with its back-translation.
Identify any semantic differences. Return JSON:
{
  "semantic_match": 0.0-1.0,
  "differences": ["list of meaning differences found"],
  "quality_assessment": "excellent|good|fair|poor"
}""",
        messages=[
            {
                "role": "user",
                "content": f"ORIGINAL:\n{text}\n\nBACK-TRANSLATION:\n{back_translation}",
            }
        ],
    )
    import json
    comparison_result = json.loads(comparison.content[0].text)

    return {
        "original": text,
        "translation": translation,
        "back_translation": back_translation,
        "quality": comparison_result,
    }


result = translate_with_backtranslation(
    text="The early bird catches the worm, but the second mouse gets the cheese.",
    source_lang="English",
    target_lang="German",
)
print(f"Original:         {result['original']}")
print(f"Translation:      {result['translation']}")
print(f"Back-translation: {result['back_translation']}")
print(f"Quality:          {result['quality']}")
```

---

## 5. Educational Prompting

Educational prompts guide learners through material using pedagogical principles like scaffolding, the Socratic method, and progressive disclosure.

### 5.1 Socratic Tutoring

```python
import anthropic

client = anthropic.Anthropic()


SOCRATIC_TUTOR_PROMPT = """You are a Socratic tutor. NEVER give direct answers.
Instead, guide the student to discover the answer through questions.

TECHNIQUE:
1. When a student asks a question, respond with a guiding question.
2. If the student is stuck, offer a hint phrased as a question.
3. If the student gives a wrong answer, ask a question that reveals why
   it is incorrect without directly saying so.
4. When the student arrives at the correct understanding, confirm it
   and extend with a deeper question.
5. Adapt your question difficulty to the student's apparent level.

RULES:
- NEVER state facts directly. Always ask questions.
- Use "What do you think would happen if..." and "Why might that be?" patterns.
- Encourage the student when they make progress.
- If the student is frustrated, simplify your questions.
- After the student demonstrates understanding, pose a related challenge.

SUBJECT: {subject}
LEVEL: {level}
"""


def create_socratic_tutor(subject: str, level: str = "intermediate"):
    """Create a Socratic tutoring session."""
    messages = []
    system = SOCRATIC_TUTOR_PROMPT.format(subject=subject, level=level)

    def ask(student_message: str) -> str:
        messages.append({"role": "user", "content": student_message})
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            system=system,
            messages=messages,
        )
        reply = response.content[0].text
        messages.append({"role": "assistant", "content": reply})
        return reply

    return ask


# Simulate a Socratic tutoring session
tutor = create_socratic_tutor("Python programming", "beginner")

# Simulated student interaction
conversation = [
    "What's the difference between a list and a tuple?",
    "I think tuples are faster?",
    "Because they can't be changed?",
]

for student_msg in conversation:
    print(f"\nStudent: {student_msg}")
    response = tutor(student_msg)
    print(f"Tutor: {response}")
```

### 5.2 Scaffolded Learning

```python
import anthropic
import json

client = anthropic.Anthropic()


def scaffolded_explanation(topic: str, learner_level: str) -> dict:
    """Generate a scaffolded explanation that builds understanding progressively."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system=f"""You are an educational content designer. Create a scaffolded
explanation of the given topic for a {learner_level} learner.

Structure your response as a JSON object with these progressive levels:

{{
  "analogy": "A simple real-world analogy that captures the core concept",
  "level_1_intuition": "2-3 sentences explaining the concept using only everyday language",
  "level_2_mechanics": "A more detailed explanation introducing proper terminology",
  "level_3_details": "Technical details, edge cases, and nuances",
  "common_misconceptions": ["list of 2-3 things learners often get wrong"],
  "check_understanding": ["2-3 questions to test comprehension, ordered by difficulty"],
  "next_topics": ["what to learn next after understanding this concept"]
}}

Each level should BUILD on the previous one, not repeat it.
Use concrete examples at every level.""",
        messages=[
            {"role": "user", "content": f"Explain: {topic}"},
        ],
    )
    return json.loads(response.content[0].text)


result = scaffolded_explanation("recursion in programming", "beginner")
print(f"Analogy: {result['analogy']}")
print(f"\nLevel 1 (intuition): {result['level_1_intuition']}")
print(f"\nLevel 2 (mechanics): {result['level_2_mechanics']}")
print(f"\nLevel 3 (details): {result['level_3_details']}")
print(f"\nMisconceptions: {result['common_misconceptions']}")
```

### 5.3 Adaptive Difficulty

```python
import anthropic
import json

client = anthropic.Anthropic()


class AdaptiveProblemGenerator:
    """Generate problems with adaptive difficulty based on learner performance."""

    def __init__(self, subject: str):
        self.subject = subject
        self.difficulty = 3  # Start at medium (1-10 scale)
        self.history: list[dict] = []

    def generate_problem(self) -> dict:
        """Generate a problem at the current difficulty level."""
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=f"""Generate a {self.subject} practice problem.

DIFFICULTY LEVEL: {self.difficulty}/10
(1 = absolute beginner, 5 = intermediate, 10 = expert)

Return JSON:
{{
  "problem": "the problem statement",
  "hints": ["progressive hints, from vague to specific"],
  "answer": "the correct answer",
  "explanation": "step-by-step solution explanation",
  "difficulty": {self.difficulty},
  "concepts_tested": ["list of concepts this problem tests"]
}}

Ensure the problem matches the difficulty level precisely.""",
            messages=[
                {"role": "user", "content": "Generate a problem."},
            ],
        )
        return json.loads(response.content[0].text)

    def record_result(self, correct: bool):
        """Adjust difficulty based on the learner's result."""
        self.history.append({
            "difficulty": self.difficulty,
            "correct": correct,
        })

        # Adaptive algorithm: increase on correct, decrease on incorrect
        if correct:
            self.difficulty = min(10, self.difficulty + 1)
        else:
            self.difficulty = max(1, self.difficulty - 1)

        # If the student gets 3 in a row correct, jump up faster
        recent = self.history[-3:]
        if len(recent) == 3 and all(r["correct"] for r in recent):
            self.difficulty = min(10, self.difficulty + 1)

    def get_performance_summary(self) -> dict:
        """Summarize learner performance."""
        if not self.history:
            return {"message": "No problems attempted yet."}
        correct = sum(1 for r in self.history if r["correct"])
        total = len(self.history)
        return {
            "total_problems": total,
            "correct": correct,
            "accuracy": round(correct / total * 100, 1),
            "current_difficulty": self.difficulty,
            "progression": [r["difficulty"] for r in self.history],
        }


# Usage
generator = AdaptiveProblemGenerator("Python programming")
problem = generator.generate_problem()
print(f"Problem (difficulty {problem['difficulty']}/10):")
print(f"  {problem['problem']}")
print(f"  Hints: {problem['hints'][0]}")

# Simulate results
generator.record_result(correct=True)
generator.record_result(correct=True)
generator.record_result(correct=False)

print(f"\nPerformance: {generator.get_performance_summary()}")
```

---

## 6. Legal Domain Prompting

Legal domain prompting requires extreme precision, citation awareness, and careful disclaimers.

### 6.1 Legal Document Analysis

```python
import anthropic
import json

client = anthropic.Anthropic()


LEGAL_ANALYSIS_PROMPT = """You are a legal document analysis assistant.

CRITICAL DISCLAIMERS (always include):
- You are an AI assistant, not a lawyer.
- Your analysis is for informational purposes only and does not constitute legal advice.
- Users should consult a qualified attorney for legal decisions.

ANALYSIS RULES:
1. Identify key clauses and their implications.
2. Flag unusual, potentially unfavorable, or ambiguous terms.
3. Compare against common industry standards when possible.
4. Use precise legal terminology but explain it in plain language.
5. NEVER fabricate legal precedents, case citations, or statutes.
6. If unsure about a legal interpretation, say so explicitly.
"""


def analyze_contract_clause(clause: str, context: str = "") -> dict:
    """Analyze a contract clause for key provisions and potential issues."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system=LEGAL_ANALYSIS_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"""Analyze this contract clause:

{clause}

{f"Context: {context}" if context else ""}

Return JSON:
{{
  "clause_type": "type of clause (e.g., indemnification, limitation of liability)",
  "plain_language_summary": "what this means in simple terms",
  "key_provisions": ["list of important provisions"],
  "potential_concerns": ["issues a party should be aware of"],
  "unusual_terms": ["terms that deviate from standard practice"],
  "ambiguities": ["unclear or potentially disputable language"],
  "questions_to_ask": ["questions to discuss with an attorney"],
  "disclaimer": "standard AI disclaimer"
}}""",
            }
        ],
    )
    return json.loads(response.content[0].text)


clause = """
12.3 Limitation of Liability. IN NO EVENT SHALL EITHER PARTY'S AGGREGATE
LIABILITY ARISING OUT OF OR RELATED TO THIS AGREEMENT EXCEED THE AMOUNTS
PAID OR PAYABLE BY CUSTOMER DURING THE TWELVE (12) MONTH PERIOD IMMEDIATELY
PRECEDING THE EVENT GIVING RISE TO THE CLAIM. THIS LIMITATION APPLIES TO
ALL CAUSES OF ACTION IN THE AGGREGATE, INCLUDING BUT NOT LIMITED TO BREACH
OF CONTRACT, BREACH OF WARRANTY, NEGLIGENCE, STRICT LIABILITY, AND OTHER
TORTS. NOTWITHSTANDING THE FOREGOING, THIS LIMITATION SHALL NOT APPLY TO
(A) CUSTOMER'S PAYMENT OBLIGATIONS, OR (B) EITHER PARTY'S INDEMNIFICATION
OBLIGATIONS UNDER SECTION 11.
"""

analysis = analyze_contract_clause(
    clause, context="B2B SaaS subscription agreement"
)
print(json.dumps(analysis, indent=2))
```

### 6.2 Legal Research with Citation Verification

```python
import anthropic

client = anthropic.Anthropic()


LEGAL_RESEARCH_PROMPT = """You are a legal research assistant.

CRITICAL RULES:
1. ONLY cite legal concepts, doctrines, and general legal principles.
2. DO NOT fabricate specific case citations (e.g., "Smith v. Jones, 123 F.3d 456").
3. If you mention a specific case, statute, or regulation, explicitly state:
   "This citation should be verified through an official legal database."
4. Distinguish between:
   - Well-established legal principles (can state with confidence)
   - Jurisdiction-specific rules (note which jurisdiction)
   - Your analysis/interpretation (label as "analysis")
5. Always note when the law may vary by jurisdiction.

Prefix any specific citation with: [VERIFY]
"""


def legal_research(question: str) -> str:
    """Assist with legal research while maintaining citation integrity."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system=LEGAL_RESEARCH_PROMPT,
        messages=[
            {"role": "user", "content": question},
        ],
    )
    return response.content[0].text


result = legal_research(
    "What are the key legal principles around the enforceability "
    "of non-compete clauses in employment contracts?"
)
print(result)
```

---

## 7. Medical and Financial Domain Prompting

Medical and financial domains share the need for extreme caution, disclaimers, and clear boundaries about what AI should and should not do.

### 7.1 Medical Information Assistant

```python
import anthropic

client = anthropic.Anthropic()


MEDICAL_INFO_PROMPT = """You are a medical information assistant.

ABSOLUTE RULES — NEVER VIOLATE:
1. You are NOT a doctor. You provide general health information ONLY.
2. ALWAYS include: "This is not medical advice. Consult a healthcare provider."
3. NEVER diagnose conditions or recommend specific treatments.
4. NEVER advise someone to stop, start, or change medication.
5. For emergency symptoms, ALWAYS direct to emergency services (911).
6. Cite well-established medical knowledge only (major guidelines, WHO, CDC).
7. When uncertain, say "I'm not sure — please consult your doctor."

ACCEPTABLE ACTIONS:
- Explain medical concepts in plain language
- Describe what conditions generally involve
- List questions to ask a healthcare provider
- Explain common medical terminology
- Describe lifestyle factors that generally affect health
"""


def medical_info(question: str) -> str:
    """Provide general medical information with appropriate disclaimers."""
    # Check for emergency keywords
    emergency_keywords = ["chest pain", "can't breathe", "bleeding heavily",
                          "overdose", "suicidal", "stroke", "heart attack"]

    if any(kw in question.lower() for kw in emergency_keywords):
        return (
            "IMPORTANT: Based on your message, this may be a medical emergency. "
            "Please call 911 (or your local emergency number) immediately. "
            "Do not rely on an AI for emergency medical guidance."
        )

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=MEDICAL_INFO_PROMPT,
        messages=[{"role": "user", "content": question}],
    )
    return response.content[0].text


print(medical_info("What is type 2 diabetes?"))
print("---")
print(medical_info("I'm having chest pain and shortness of breath"))
```

### 7.2 Financial Analysis Assistant

```python
import anthropic
import json

client = anthropic.Anthropic()


FINANCIAL_PROMPT = """You are a financial analysis assistant.

CRITICAL DISCLAIMERS:
- This is NOT financial advice. Consult a qualified financial advisor.
- Past performance does not indicate future results.
- All investments carry risk, including potential loss of principal.

RULES:
1. Present factual financial information and general educational content.
2. NEVER recommend specific investments, trades, or financial products.
3. NEVER promise returns or predict market movements.
4. Explain financial concepts clearly with examples.
5. When analyzing data, present multiple perspectives.
6. Always note risks and limitations of any analysis approach.
"""


def financial_analysis(data_description: str, question: str) -> str:
    """Provide financial analysis with appropriate disclaimers."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=FINANCIAL_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Data:\n{data_description}\n\nQuestion: {question}",
            }
        ],
    )
    return response.content[0].text


result = financial_analysis(
    data_description="""
    Company ABC Q3 2025:
    Revenue: $5.2B (up 12% YoY)
    Net Income: $800M (up 18% YoY)
    Debt-to-Equity: 0.45
    P/E Ratio: 28
    Industry Average P/E: 22
    """,
    question="What does the P/E ratio suggest about this company's valuation?",
)
print(result)
```

---

## 8. Creative Writing Prompts

Creative writing prompts balance structure with creative freedom — too much constraint stifles creativity, too little produces unfocused output.

### 8.1 Structured Creative Writing

```python
import anthropic

client = anthropic.Anthropic()


def generate_story(
    premise: str,
    constraints: dict | None = None,
) -> str:
    """Generate a story with controlled creative parameters."""
    constraints = constraints or {}

    constraint_text = ""
    if constraints:
        constraint_text = "\n\nCONSTRAINTS:\n"
        for key, value in constraints.items():
            constraint_text += f"- {key}: {value}\n"

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system=f"""You are a creative writing assistant.

WRITING QUALITY GUIDELINES:
- Show, don't tell. Use concrete sensory details.
- Vary sentence length and structure for rhythm.
- Use dialogue to reveal character, not to info-dump.
- Every scene should advance plot, reveal character, or build atmosphere.
- Avoid cliches. If you catch yourself writing one, find a fresher way.
- End with resonance — the last line should linger.
{constraint_text}""",
        messages=[
            {
                "role": "user",
                "content": f"Write a story based on this premise:\n\n{premise}",
            }
        ],
    )
    return response.content[0].text


story = generate_story(
    premise="A librarian discovers that certain books in the archive can predict the future, but only in metaphors.",
    constraints={
        "length": "500-700 words",
        "tone": "literary fiction, contemplative",
        "pov": "third person limited",
        "must_include": "a scene where the librarian struggles to interpret a metaphor",
        "must_avoid": "happy endings, magic systems explained in detail",
    },
)
print(story)
```

### 8.2 Style Mimicry

```python
import anthropic

client = anthropic.Anthropic()


def write_in_style(
    content_prompt: str,
    style_description: str,
    style_examples: list[str] | None = None,
) -> str:
    """Generate content matching a described writing style."""
    examples_section = ""
    if style_examples:
        examples_section = "\n\nSTYLE EXAMPLES (match this voice and rhythm):\n"
        for i, ex in enumerate(style_examples, 1):
            examples_section += f'\nExample {i}: "{ex}"\n'

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system=f"""Write in the following style:
{style_description}
{examples_section}

RULES:
- Match the sentence structure patterns of the style.
- Use vocabulary consistent with the style.
- Maintain the emotional register throughout.
- The content should feel natural in this style, not forced.
""",
        messages=[
            {"role": "user", "content": content_prompt},
        ],
    )
    return response.content[0].text


result = write_in_style(
    content_prompt="Describe a morning commute on a crowded subway.",
    style_description=(
        "Sparse, minimalist prose. Short declarative sentences. "
        "Concrete nouns, strong verbs. No adverbs. Hemingway-esque. "
        "Understated emotion. The unsaid matters more than the said."
    ),
    style_examples=[
        "The old man sat in the chair. He did not move. The room was quiet and the light was gray.",
        "She walked to the bar and ordered a drink. The bartender poured it without looking up.",
    ],
)
print(result)
```

---

## 9. Research Assistance Prompts

Research assistance prompts help with literature review, hypothesis formation, methodology design, and critical analysis.

### 9.1 Literature Review Assistant

```python
import anthropic
import json

client = anthropic.Anthropic()


def research_synthesis(
    topic: str,
    papers: list[dict],
) -> dict:
    """Synthesize research findings from multiple papers."""
    papers_text = ""
    for i, paper in enumerate(papers, 1):
        papers_text += f"""
Paper {i}:
  Title: {paper['title']}
  Authors: {paper.get('authors', 'Unknown')}
  Year: {paper.get('year', 'Unknown')}
  Key Findings: {paper['findings']}
"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system="""You are a research synthesis assistant. Analyze the provided papers
and produce a structured synthesis.

RULES:
- Identify agreements and disagreements between papers.
- Note methodological differences that may explain conflicting results.
- Highlight gaps in the current research.
- Do NOT add findings not present in the provided papers.
- Use hedging language ("suggests", "indicates") rather than definitive claims.
- Note limitations of the synthesis (e.g., limited sample of papers).

Return JSON:
{
  "synthesis_summary": "2-3 paragraph synthesis of the research landscape",
  "consensus_findings": ["findings that multiple papers agree on"],
  "conflicting_findings": [
    {"topic": "...", "positions": ["paper X says...", "paper Y says..."]}
  ],
  "research_gaps": ["identified gaps in the literature"],
  "methodology_notes": ["important methodological observations"],
  "suggested_next_steps": ["potential research directions"]
}""",
        messages=[
            {
                "role": "user",
                "content": f"Synthesize research on: {topic}\n\n{papers_text}",
            }
        ],
    )
    return json.loads(response.content[0].text)


papers = [
    {
        "title": "Effects of Sleep Duration on Cognitive Performance",
        "authors": "Smith et al.",
        "year": 2024,
        "findings": "7-9 hours of sleep optimized cognitive test scores in adults aged 25-45. Below 6 hours showed 23% decline in working memory tasks.",
    },
    {
        "title": "Sleep Quality vs. Quantity: A Meta-Analysis",
        "authors": "Johnson & Lee",
        "year": 2023,
        "findings": "Sleep quality (measured by sleep efficiency) was a stronger predictor of next-day cognitive performance than total sleep duration. Deep sleep percentage was the single best predictor.",
    },
    {
        "title": "Napping and Cognitive Restoration",
        "authors": "Garcia et al.",
        "year": 2025,
        "findings": "20-minute naps restored cognitive performance by 34% in sleep-deprived subjects. Longer naps (60+ min) showed sleep inertia effects that temporarily worsened performance.",
    },
]

synthesis = research_synthesis("Sleep and Cognitive Performance", papers)
print(json.dumps(synthesis, indent=2))
```

### 9.2 Hypothesis Generator

```python
import anthropic
import json

client = anthropic.Anthropic()


def generate_hypotheses(
    observation: str,
    domain: str,
    constraints: list[str] | None = None,
) -> dict:
    """Generate testable research hypotheses from an observation."""
    constraint_text = ""
    if constraints:
        constraint_text = "\nCONSTRAINTS:\n" + "\n".join(f"- {c}" for c in constraints)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system=f"""You are a research methodology assistant specializing in {domain}.

Generate testable hypotheses from the given observation.
{constraint_text}

Return JSON:
{{
  "observation_analysis": "brief analysis of what the observation implies",
  "hypotheses": [
    {{
      "id": "H1",
      "statement": "formal hypothesis statement",
      "type": "causal|correlational|descriptive",
      "testability": "how this could be tested",
      "independent_variable": "what is manipulated or measured",
      "dependent_variable": "what outcome is measured",
      "potential_confounds": ["list of confounding variables"],
      "novelty": "low|medium|high"
    }}
  ],
  "suggested_study_design": "brief sketch of an appropriate study"
}}""",
        messages=[
            {
                "role": "user",
                "content": f"Observation: {observation}",
            }
        ],
    )
    return json.loads(response.content[0].text)


result = generate_hypotheses(
    observation=(
        "Software teams that use pair programming report higher code quality "
        "but lower individual productivity, yet their project delivery times "
        "are often shorter than solo-programming teams."
    ),
    domain="Software Engineering",
    constraints=[
        "Must be testable in a controlled study",
        "Should account for team size as a variable",
    ],
)

for h in result["hypotheses"]:
    print(f"\n{h['id']}: {h['statement']}")
    print(f"  Type: {h['type']}, Novelty: {h['novelty']}")
    print(f"  Test: {h['testability']}")
```

### 9.3 Critical Analysis Prompt

```python
import anthropic
import json

client = anthropic.Anthropic()


def critical_analysis(
    claim: str,
    evidence: str,
    analysis_type: str = "scientific",
) -> dict:
    """Critically analyze a claim against its supporting evidence."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system=f"""You are a critical analysis assistant performing {analysis_type} analysis.

Evaluate the claim against the provided evidence using these criteria:
1. Logical validity: Does the conclusion follow from the premises?
2. Evidence quality: How strong is the evidence?
3. Alternative explanations: What else could explain the data?
4. Generalizability: How broadly can these findings be applied?
5. Potential biases: What biases might affect the conclusion?

Return JSON:
{{
  "claim_restated": "the claim in precise terms",
  "evidence_assessment": {{
    "strength": "strong|moderate|weak|insufficient",
    "type": "empirical|anecdotal|theoretical|expert_opinion",
    "limitations": ["list of evidence limitations"]
  }},
  "logical_analysis": {{
    "valid": true/false,
    "fallacies": ["any logical fallacies identified"],
    "reasoning_gaps": ["gaps in the reasoning chain"]
  }},
  "alternative_explanations": ["plausible alternative explanations"],
  "overall_assessment": {{
    "credibility": "high|moderate|low",
    "confidence_level": "how confident we should be in this claim",
    "recommendation": "what additional evidence would strengthen/weaken the claim"
  }}
}}

Be rigorous but fair. Note strengths as well as weaknesses.""",
        messages=[
            {
                "role": "user",
                "content": f"CLAIM: {claim}\n\nEVIDENCE: {evidence}",
            }
        ],
    )
    return json.loads(response.content[0].text)


result = critical_analysis(
    claim="Remote workers are more productive than office workers.",
    evidence=(
        "A 2024 survey of 500 tech workers found that self-reported "
        "productivity scores were 15% higher among remote workers. "
        "Remote workers also reported higher job satisfaction."
    ),
    analysis_type="scientific",
)
print(json.dumps(result, indent=2))
```

---

## Exercises

### Exercise 1: Multi-Format Data Extractor

Build a data extraction system that can handle multiple document types (emails, invoices, resumes) using a single configurable prompt pattern.

**Requirements:**
- Define extraction schemas for at least 3 document types
- Auto-detect the document type before extracting
- Validate extracted data against the schema
- Handle partial or malformed documents gracefully

<details><summary>Show Answer</summary>

```python
import anthropic
import json
import re

client = anthropic.Anthropic()


# Document type schemas
SCHEMAS = {
    "email": {
        "fields": {
            "from": {"type": "string", "required": True},
            "to": {"type": "string", "required": True},
            "subject": {"type": "string", "required": True},
            "date": {"type": "date", "required": False},
            "body_summary": {"type": "string", "required": True},
            "action_items": {"type": "list", "required": False},
            "attachments_mentioned": {"type": "list", "required": False},
        },
    },
    "invoice": {
        "fields": {
            "invoice_number": {"type": "string", "required": True},
            "date": {"type": "date", "required": True},
            "vendor_name": {"type": "string", "required": True},
            "customer_name": {"type": "string", "required": True},
            "line_items": {"type": "list", "required": True},
            "total": {"type": "number", "required": True},
            "currency": {"type": "string", "required": False},
        },
    },
    "resume": {
        "fields": {
            "name": {"type": "string", "required": True},
            "email": {"type": "email", "required": False},
            "phone": {"type": "string", "required": False},
            "education": {"type": "list", "required": False},
            "experience": {"type": "list", "required": True},
            "skills": {"type": "list", "required": False},
            "summary": {"type": "string", "required": False},
        },
    },
}


class MultiFormatExtractor:
    """Extract structured data from multiple document types."""

    def __init__(self):
        self.schemas = SCHEMAS

    def detect_type(self, text: str) -> str:
        """Auto-detect the document type."""
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=50,
            system=(
                "Classify the document type. Respond with EXACTLY one word: "
                "email, invoice, or resume. Nothing else."
            ),
            messages=[
                {"role": "user", "content": f"Document:\n{text[:1000]}"},
            ],
        )
        detected = response.content[0].text.strip().lower()
        if detected not in self.schemas:
            return "email"  # Default fallback
        return detected

    def _build_extraction_prompt(self, doc_type: str) -> str:
        schema = self.schemas[doc_type]
        field_descriptions = []
        for name, spec in schema["fields"].items():
            req = "REQUIRED" if spec["required"] else "optional"
            field_descriptions.append(f'  "{name}": {spec["type"]} ({req})')

        fields_text = ",\n".join(field_descriptions)
        return f"""Extract data from this {doc_type} document.
Return a JSON object with these fields:
{{
{fields_text}
}}

Rules:
- Use null for any field not found in the document.
- For list fields, return an empty list [] if none found.
- Extract ONLY what is explicitly in the document.
- For dates, use YYYY-MM-DD format.
- For numbers, return as numeric type."""

    def extract(self, text: str, doc_type: str | None = None) -> dict:
        """Extract data, optionally auto-detecting document type."""
        if doc_type is None:
            doc_type = self.detect_type(text)

        prompt = self._build_extraction_prompt(doc_type)

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=prompt,
            messages=[
                {"role": "user", "content": f"Extract from:\n\n{text}"},
            ],
        )

        try:
            data = json.loads(response.content[0].text)
        except json.JSONDecodeError:
            data = {}

        # Validate
        errors = self._validate(data, doc_type)

        return {
            "document_type": doc_type,
            "extracted_data": data,
            "validation_errors": errors,
            "is_valid": len(errors) == 0,
        }

    def _validate(self, data: dict, doc_type: str) -> list[str]:
        errors = []
        schema = self.schemas[doc_type]
        for field_name, spec in schema["fields"].items():
            value = data.get(field_name)
            if spec["required"] and (value is None or value == "" or value == []):
                errors.append(f"Required field '{field_name}' is missing or empty")
            if value is not None and spec["type"] == "email":
                if not re.match(r"[^@]+@[^@]+\.[^@]+", str(value)):
                    errors.append(f"Field '{field_name}' is not a valid email")
        return errors


# Test with different document types
extractor = MultiFormatExtractor()

email_text = """
From: alice@company.example.com
To: bob@company.example.com
Date: March 15, 2025
Subject: Q3 Budget Review Meeting

Hi Bob,

Please review the attached Q3 budget spreadsheet before our meeting on Thursday.
I've highlighted the areas where we exceeded projections. We need to discuss
the marketing spend increase and decide on Q4 allocations.

Can you also bring the vendor contracts for review?

Thanks,
Alice
"""

resume_text = """
JANE DOE
jane.doe@email.example.com | (555) 123-4567

SUMMARY
Full-stack developer with 8 years of experience in Python and JavaScript.

EXPERIENCE
Senior Developer — TechCorp (2021-present)
- Led team of 5 engineers building microservices platform
- Reduced API latency by 40% through caching optimization

Developer — StartupXYZ (2018-2021)
- Built React frontend for customer-facing dashboard
- Implemented CI/CD pipeline using GitHub Actions

EDUCATION
B.S. Computer Science — MIT (2018)

SKILLS
Python, JavaScript, React, PostgreSQL, Docker, AWS
"""

for doc in [email_text, resume_text]:
    result = extractor.extract(doc)
    print(f"\nType: {result['document_type']}")
    print(f"Valid: {result['is_valid']}")
    print(f"Data: {json.dumps(result['extracted_data'], indent=2)}")
    if result["validation_errors"]:
        print(f"Errors: {result['validation_errors']}")
    print("-" * 60)
```

</details>

### Exercise 2: Configurable Summarizer

Build a summarization system that supports multiple output formats (paragraph, bullets, headline, tweet), length targets, and audience levels (expert, general, child).

**Requirements:**
- Support at least 4 output formats
- Implement length control with word-count targeting
- Adjust vocabulary and complexity for 3 audience levels
- Include a faithfulness check comparing summary to source

<details><summary>Show Answer</summary>

```python
import anthropic
import json

client = anthropic.Anthropic()


class ConfigurableSummarizer:
    """Summarize text with configurable format, length, and audience."""

    FORMAT_INSTRUCTIONS = {
        "paragraph": "Write as flowing prose paragraphs. No bullets or numbers.",
        "bullets": "Use bullet points (- prefix). Each point is one key idea.",
        "numbered": "Use a numbered list (1., 2., ...). Order by importance.",
        "headline": "Write a news headline (under 15 words) plus a 2-sentence subheadline.",
        "tweet": "Write as a tweet (under 280 characters). Use no hashtags.",
        "tldr": "Start with 'TL;DR:' and give the absolute core message in 1-2 sentences.",
        "executive": "Write a 3-part executive summary: Context, Findings, Recommendation.",
    }

    AUDIENCE_INSTRUCTIONS = {
        "expert": (
            "Use domain-specific terminology without explanation. "
            "Assume deep background knowledge. Focus on nuances and implications."
        ),
        "general": (
            "Use clear, accessible language. Briefly explain any technical terms. "
            "Assume educated adult with no domain expertise."
        ),
        "child": (
            "Use simple words and short sentences. Explain concepts with "
            "everyday analogies. Aim for a 10-year-old reading level."
        ),
    }

    LENGTH_TARGETS = {
        "micro": {"words": 25, "tolerance": 10},
        "short": {"words": 50, "tolerance": 15},
        "medium": {"words": 120, "tolerance": 30},
        "long": {"words": 250, "tolerance": 50},
    }

    def summarize(
        self,
        text: str,
        format_type: str = "paragraph",
        length: str = "medium",
        audience: str = "general",
        focus: str | None = None,
    ) -> dict:
        """Generate a summary with the specified configuration."""
        fmt = self.FORMAT_INSTRUCTIONS.get(format_type, self.FORMAT_INSTRUCTIONS["paragraph"])
        aud = self.AUDIENCE_INSTRUCTIONS.get(audience, self.AUDIENCE_INSTRUCTIONS["general"])
        len_spec = self.LENGTH_TARGETS.get(length, self.LENGTH_TARGETS["medium"])
        target_words = len_spec["words"]
        tolerance = len_spec["tolerance"]

        focus_instruction = ""
        if focus:
            focus_instruction = f"\nFOCUS AREA: Emphasize aspects related to: {focus}"

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=f"""Summarize the provided text.

FORMAT: {fmt}
AUDIENCE: {aud}
TARGET LENGTH: approximately {target_words} words (range: {target_words - tolerance}-{target_words + tolerance})
{focus_instruction}

RULES:
- Never add information not in the source text.
- Start directly with content — no preamble.
- Stay within the word count range.
- Match vocabulary to the audience level.""",
            messages=[
                {"role": "user", "content": f"Summarize:\n\n{text}"},
            ],
        )
        summary = response.content[0].text.strip()
        word_count = len(summary.split())

        # Faithfulness check
        faithfulness = self._check_faithfulness(text, summary)

        return {
            "summary": summary,
            "config": {
                "format": format_type,
                "length": length,
                "audience": audience,
                "focus": focus,
            },
            "metrics": {
                "word_count": word_count,
                "target_words": target_words,
                "within_range": abs(word_count - target_words) <= tolerance,
            },
            "faithfulness": faithfulness,
        }

    def _check_faithfulness(self, source: str, summary: str) -> dict:
        """Check if the summary is faithful to the source."""
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            system="""Compare the summary to the source text.
Return JSON:
{
  "score": 0.0-1.0,
  "unsupported_claims": ["claims in summary not in source"],
  "key_omissions": ["important source points missing from summary"],
  "assessment": "faithful|mostly_faithful|unfaithful"
}""",
            messages=[
                {
                    "role": "user",
                    "content": f"SOURCE:\n{source}\n\nSUMMARY:\n{summary}",
                }
            ],
        )
        return json.loads(response.content[0].text)


# Test the summarizer
summarizer = ConfigurableSummarizer()

article = """
CRISPR-Cas9 gene editing technology has reached a new milestone with the
FDA approval of the first CRISPR-based therapy for sickle cell disease.
The treatment, called Casgevy, works by editing patients' stem cells
outside the body to produce a form of hemoglobin that prevents red blood
cells from sickling. In clinical trials, 29 out of 30 patients who
received the treatment were free of severe pain crises for at least
12 months. The treatment costs approximately $2.2 million per patient
and requires a complex process including chemotherapy to prepare the
bone marrow. Critics note the high cost limits accessibility, while
supporters argue it represents a potential one-time cure for a disease
affecting millions worldwide. The approval opens the door for CRISPR-based
treatments for other genetic conditions, with therapies for beta-thalassemia
and certain forms of blindness in advanced clinical trials.
"""

# Generate summaries with different configurations
configs = [
    {"format_type": "tweet", "length": "micro", "audience": "general"},
    {"format_type": "bullets", "length": "medium", "audience": "expert"},
    {"format_type": "paragraph", "length": "short", "audience": "child"},
    {"format_type": "executive", "length": "long", "audience": "general", "focus": "cost and accessibility"},
]

for config in configs:
    result = summarizer.summarize(article, **config)
    print(f"\n{'=' * 60}")
    print(f"Config: {result['config']}")
    print(f"Words: {result['metrics']['word_count']} (target: {result['metrics']['target_words']})")
    print(f"Faithfulness: {result['faithfulness']['assessment']}")
    print(f"\n{result['summary']}")
```

</details>

### Exercise 3: Socratic Tutor with Misconception Detection

Build a Socratic tutoring system that detects common misconceptions in student responses and addresses them through targeted questioning.

**Requirements:**
- Never give direct answers — only ask questions
- Maintain a knowledge model of what the student understands
- Detect misconceptions and address them specifically
- Track progress through the conversation
- Support configurable subject and difficulty level

<details><summary>Show Answer</summary>

```python
import anthropic
import json

client = anthropic.Anthropic()


class SocraticTutor:
    """Socratic tutor with misconception detection and progress tracking."""

    def __init__(self, subject: str, topic: str, difficulty: str = "intermediate"):
        self.subject = subject
        self.topic = topic
        self.difficulty = difficulty
        self.messages: list[dict] = []
        self.progress = {
            "concepts_explored": [],
            "misconceptions_detected": [],
            "misconceptions_resolved": [],
            "understanding_level": "unknown",
            "turn_count": 0,
        }
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        return f"""You are a Socratic tutor teaching {self.subject}, specifically the topic: {self.topic}.
Difficulty level: {self.difficulty}.

CORE RULES:
1. NEVER give direct answers. ONLY ask questions.
2. Guide the student to discover answers through their own reasoning.
3. If the student is wrong, ask a question that reveals WHY their answer is incorrect.
4. If the student is stuck, provide a simpler question or a concrete example to consider.
5. When the student demonstrates understanding, confirm it and go deeper.

MISCONCEPTION DETECTION:
After each student response, analyze it for misconceptions. Format your
internal analysis as follows (the student will not see this):

Before your visible response, output a JSON block wrapped in <analysis> tags:
<analysis>
{{
  "student_claims": ["what the student believes based on their response"],
  "correct_elements": ["parts of their response that are correct"],
  "misconceptions": ["specific misconceptions detected"],
  "understanding_level": "none|emerging|partial|solid|deep",
  "next_concept_to_probe": "what to explore next",
  "question_strategy": "clarifying|challenging|extending|redirecting"
}}
</analysis>

Then provide your Socratic question (no analysis visible to student).

QUESTION TYPES:
- Clarifying: "What do you mean by...?" / "Can you give an example?"
- Challenging: "What would happen if...?" / "How does that explain...?"
- Extending: "How does this connect to...?" / "What about the case where...?"
- Redirecting: "Let's think about it differently. What if...?"
"""

    def interact(self, student_message: str) -> dict:
        """Process a student message and return a Socratic response."""
        self.progress["turn_count"] += 1
        self.messages.append({"role": "user", "content": student_message})

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=self.system_prompt,
            messages=self.messages,
        )
        full_response = response.content[0].text

        # Parse analysis and visible response
        analysis = {}
        visible_response = full_response

        if "<analysis>" in full_response and "</analysis>" in full_response:
            analysis_start = full_response.index("<analysis>") + len("<analysis>")
            analysis_end = full_response.index("</analysis>")
            analysis_json = full_response[analysis_start:analysis_end].strip()
            try:
                analysis = json.loads(analysis_json)
            except json.JSONDecodeError:
                analysis = {}
            visible_response = full_response[analysis_end + len("</analysis>"):].strip()

        # Update progress
        if analysis:
            if analysis.get("misconceptions"):
                for m in analysis["misconceptions"]:
                    if m not in self.progress["misconceptions_detected"]:
                        self.progress["misconceptions_detected"].append(m)
            if analysis.get("understanding_level"):
                self.progress["understanding_level"] = analysis["understanding_level"]
            if analysis.get("next_concept_to_probe"):
                concept = analysis["next_concept_to_probe"]
                if concept not in self.progress["concepts_explored"]:
                    self.progress["concepts_explored"].append(concept)

        self.messages.append({"role": "assistant", "content": full_response})

        return {
            "tutor_response": visible_response,
            "analysis": analysis,
            "progress": self.progress.copy(),
        }

    def get_progress_report(self) -> str:
        """Generate a human-readable progress report."""
        p = self.progress
        report = f"""
Progress Report: {self.subject} — {self.topic}
{'=' * 50}
Turns: {p['turn_count']}
Understanding Level: {p['understanding_level']}
Concepts Explored: {len(p['concepts_explored'])}
  {', '.join(p['concepts_explored'][:5]) or 'None yet'}
Misconceptions Found: {len(p['misconceptions_detected'])}
  {', '.join(p['misconceptions_detected'][:3]) or 'None detected'}
Misconceptions Resolved: {len(p['misconceptions_resolved'])}
"""
        return report


# Simulate a tutoring session
tutor = SocraticTutor(
    subject="Computer Science",
    topic="Hash Tables",
    difficulty="beginner",
)

student_messages = [
    "What is a hash table? Is it like a regular array?",
    "So it's an array but you use a function to find the index?",
    "What happens if two keys give the same index? That seems impossible if the function is good.",
    "Oh, so collisions can happen. Does that mean hash tables are slow?",
]

for msg in student_messages:
    print(f"\nStudent: {msg}")
    result = tutor.interact(msg)
    print(f"Tutor: {result['tutor_response']}")
    if result["analysis"].get("misconceptions"):
        print(f"  [Misconceptions detected: {result['analysis']['misconceptions']}]")
    print(f"  [Understanding: {result['progress']['understanding_level']}]")

print(tutor.get_progress_report())
```

</details>

### Exercise 4: Domain-Safe Medical FAQ Generator

Build a medical FAQ generator that creates patient-friendly explanations of medical conditions while strictly maintaining safety guardrails.

**Requirements:**
- Generate explanations at a configurable reading level
- Always include appropriate disclaimers
- Detect and refuse to answer diagnostic or treatment questions
- Provide "questions to ask your doctor" for each topic
- Validate output for safety compliance before returning

<details><summary>Show Answer</summary>

```python
import anthropic
import json
import re

client = anthropic.Anthropic()


class MedicalFAQGenerator:
    """Generate safe, patient-friendly medical information."""

    # Topics that should always be redirected to a healthcare provider
    REDIRECT_TOPICS = [
        "diagnosis", "treatment plan", "medication dosage",
        "drug interaction", "prognosis", "should I take",
        "should I stop", "is it safe to", "am I at risk",
    ]

    READING_LEVELS = {
        "simple": {
            "instruction": "Use simple words. Short sentences. No medical jargon. A 12-year-old should understand.",
            "max_sentence_words": 15,
        },
        "general": {
            "instruction": "Use clear, accessible language. Define medical terms in parentheses.",
            "max_sentence_words": 25,
        },
        "detailed": {
            "instruction": "Use proper medical terminology with brief explanations. Suitable for health-literate adults.",
            "max_sentence_words": 35,
        },
    }

    MANDATORY_DISCLAIMER = (
        "DISCLAIMER: This information is for educational purposes only and is not "
        "medical advice. Always consult a qualified healthcare provider for medical "
        "decisions. If you are experiencing a medical emergency, call 911 or your "
        "local emergency number immediately."
    )

    def __init__(self):
        pass

    def _is_diagnostic_question(self, question: str) -> bool:
        """Check if the question asks for diagnosis or treatment advice."""
        lower_q = question.lower()
        for topic in self.REDIRECT_TOPICS:
            if topic in lower_q:
                return True
        # Pattern checks
        diagnostic_patterns = [
            r"do I have",
            r"could I have",
            r"what('s| is) wrong with",
            r"should I (take|stop|start|use)",
            r"is it (safe|okay|ok) to",
            r"will I (get|develop|die)",
        ]
        for pattern in diagnostic_patterns:
            if re.search(pattern, lower_q):
                return True
        return False

    def generate_faq(
        self,
        topic: str,
        reading_level: str = "general",
    ) -> dict:
        """Generate a patient-friendly FAQ about a medical topic."""
        # Safety check
        if self._is_diagnostic_question(topic):
            return {
                "status": "redirected",
                "message": (
                    "This question is about a personal medical concern. "
                    "Please consult your healthcare provider for personalized "
                    "medical advice. If this is urgent, contact your doctor's "
                    "office or visit an urgent care facility."
                ),
                "disclaimer": self.MANDATORY_DISCLAIMER,
            }

        level = self.READING_LEVELS.get(reading_level, self.READING_LEVELS["general"])

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=f"""You are a medical information writer for patient education materials.

READING LEVEL: {level['instruction']}

SAFETY RULES:
1. Provide ONLY general educational information.
2. NEVER diagnose or suggest specific treatments.
3. NEVER recommend specific medications or dosages.
4. Use phrases like "some people experience" instead of "you will experience."
5. Always frame information as general — not personalized advice.
6. Include when to see a doctor for each condition discussed.

REQUIRED SECTIONS (return as JSON):
{{
  "topic": "the medical topic",
  "what_is_it": "plain-language explanation of the condition/concept",
  "common_questions": [
    {{"question": "...", "answer": "..."}}
  ],
  "key_facts": ["important facts about this topic"],
  "when_to_see_doctor": ["situations when medical attention is recommended"],
  "questions_for_your_doctor": ["questions patients might want to ask their provider"],
  "reliable_sources": ["reputable organizations for more information (e.g., CDC, WHO, Mayo Clinic)"]
}}
""",
            messages=[
                {"role": "user", "content": f"Create a patient FAQ about: {topic}"},
            ],
        )

        faq_data = json.loads(response.content[0].text)

        # Output safety validation
        safety_issues = self._validate_safety(faq_data)

        if safety_issues:
            # Re-generate with stronger safety constraints
            return {
                "status": "safety_review_needed",
                "issues": safety_issues,
                "faq": faq_data,
                "disclaimer": self.MANDATORY_DISCLAIMER,
            }

        return {
            "status": "success",
            "reading_level": reading_level,
            "faq": faq_data,
            "disclaimer": self.MANDATORY_DISCLAIMER,
        }

    def _validate_safety(self, faq: dict) -> list[str]:
        """Validate FAQ output for safety compliance."""
        issues = []
        full_text = json.dumps(faq).lower()

        # Check for prescriptive language
        prescriptive_patterns = [
            (r"you should take", "prescriptive_medication"),
            (r"take \d+ mg", "specific_dosage"),
            (r"you (have|probably have|likely have)", "diagnosis"),
            (r"stop taking", "medication_change"),
            (r"this will cure", "cure_claim"),
            (r"guaranteed to", "guarantee_claim"),
        ]
        for pattern, label in prescriptive_patterns:
            if re.search(pattern, full_text):
                issues.append(f"Safety issue: {label} detected in output")

        return issues


# Test the generator
generator = MedicalFAQGenerator()

# Test 1: Valid educational topic
result = generator.generate_faq("Type 2 Diabetes", reading_level="simple")
print(f"Status: {result['status']}")
if result["status"] == "success":
    faq = result["faq"]
    print(f"Topic: {faq['topic']}")
    print(f"\nWhat is it: {faq['what_is_it'][:200]}...")
    print(f"\nKey facts: {faq['key_facts'][:3]}")
    print(f"\nWhen to see doctor: {faq['when_to_see_doctor'][:2]}")
    print(f"\nQuestions for your doctor: {faq['questions_for_your_doctor'][:3]}")
print(f"\n{result['disclaimer']}")

# Test 2: Diagnostic question (should be redirected)
print("\n" + "=" * 60)
result2 = generator.generate_faq("Do I have diabetes?")
print(f"Status: {result2['status']}")
print(f"Message: {result2.get('message', 'N/A')}")

# Test 3: Treatment question (should be redirected)
print("\n" + "=" * 60)
result3 = generator.generate_faq("Should I stop taking metformin?")
print(f"Status: {result3['status']}")
print(f"Message: {result3.get('message', 'N/A')}")
```

</details>

### Exercise 5: Translation Quality Pipeline

Build a translation pipeline that translates text, validates quality through back-translation, checks terminology consistency against a glossary, and produces a quality report.

**Requirements:**
- Support glossary-controlled translation
- Implement back-translation quality verification
- Check terminology consistency (every glossary term must use the specified translation)
- Produce a quality score and detailed report
- Handle at least 2 language pairs

<details><summary>Show Answer</summary>

```python
import anthropic
import json

client = anthropic.Anthropic()


class TranslationQualityPipeline:
    """Full translation pipeline with quality verification."""

    def __init__(self):
        pass

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        glossary: dict[str, str] | None = None,
        style: str = "formal",
    ) -> dict:
        """Translate with full quality pipeline."""
        glossary = glossary or {}

        # Step 1: Forward translation
        forward_translation = self._forward_translate(
            text, source_lang, target_lang, glossary, style
        )

        # Step 2: Back-translation
        back_translation = self._back_translate(
            forward_translation, target_lang, source_lang
        )

        # Step 3: Glossary compliance check
        glossary_compliance = self._check_glossary(
            forward_translation, glossary
        )

        # Step 4: Semantic similarity check
        similarity = self._check_similarity(text, back_translation, source_lang)

        # Step 5: Quality report
        quality_report = self._generate_quality_report(
            text, forward_translation, back_translation,
            glossary_compliance, similarity
        )

        return {
            "original": text,
            "translation": forward_translation,
            "back_translation": back_translation,
            "glossary_compliance": glossary_compliance,
            "similarity": similarity,
            "quality_report": quality_report,
        }

    def _forward_translate(
        self, text: str, source: str, target: str,
        glossary: dict, style: str,
    ) -> str:
        glossary_text = ""
        if glossary:
            glossary_text = "\n\nMANDATORY GLOSSARY (use these exact translations):\n"
            glossary_text += "\n".join(f'  "{k}" → "{v}"' for k, v in glossary.items())

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=f"""Translate from {source} to {target}. Style: {style}.
{glossary_text}

Rules:
- Use glossary terms exactly as specified.
- Maintain original formatting.
- Output ONLY the translation, nothing else.""",
            messages=[{"role": "user", "content": text}],
        )
        return response.content[0].text.strip()

    def _back_translate(self, text: str, source: str, target: str) -> str:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=f"Translate from {source} to {target}. Output ONLY the translation.",
            messages=[{"role": "user", "content": text}],
        )
        return response.content[0].text.strip()

    def _check_glossary(
        self, translation: str, glossary: dict[str, str],
    ) -> dict:
        """Check if all glossary terms were used in the translation."""
        results = {}
        for source_term, target_term in glossary.items():
            found = target_term.lower() in translation.lower()
            results[source_term] = {
                "expected": target_term,
                "found": found,
            }
        compliant_count = sum(1 for r in results.values() if r["found"])
        total = len(results) if results else 1
        return {
            "terms": results,
            "compliance_rate": round(compliant_count / total, 2) if results else 1.0,
        }

    def _check_similarity(
        self, original: str, back_translation: str, lang: str,
    ) -> dict:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            system="""Compare two texts and assess semantic similarity.
Return JSON:
{
  "semantic_similarity": 0.0-1.0,
  "preserved_facts": ["facts present in both texts"],
  "lost_facts": ["facts in original but not in back-translation"],
  "added_facts": ["facts in back-translation but not in original"],
  "meaning_shifts": ["subtle meaning changes between the two"]
}""",
            messages=[
                {
                    "role": "user",
                    "content": f"ORIGINAL ({lang}):\n{original}\n\nBACK-TRANSLATION ({lang}):\n{back_translation}",
                }
            ],
        )
        return json.loads(response.content[0].text)

    def _generate_quality_report(
        self, original: str, translation: str, back_translation: str,
        glossary_compliance: dict, similarity: dict,
    ) -> dict:
        """Generate an overall quality report."""
        # Calculate composite score
        sim_score = similarity.get("semantic_similarity", 0)
        gloss_score = glossary_compliance.get("compliance_rate", 1.0)

        # Weight: 60% semantic, 40% glossary
        composite = sim_score * 0.6 + gloss_score * 0.4

        # Quality tier
        if composite >= 0.9:
            tier = "excellent"
        elif composite >= 0.75:
            tier = "good"
        elif composite >= 0.6:
            tier = "acceptable"
        else:
            tier = "needs_revision"

        issues = []
        if sim_score < 0.8:
            issues.append("Semantic fidelity below threshold — key meaning may be lost")
        if gloss_score < 1.0:
            noncompliant = [
                term for term, info in glossary_compliance.get("terms", {}).items()
                if not info["found"]
            ]
            issues.append(f"Glossary terms not used: {noncompliant}")
        if similarity.get("meaning_shifts"):
            issues.append(f"Meaning shifts detected: {similarity['meaning_shifts']}")

        return {
            "composite_score": round(composite, 3),
            "quality_tier": tier,
            "semantic_score": sim_score,
            "glossary_score": gloss_score,
            "issues": issues,
            "recommendation": (
                "Translation approved" if tier in ("excellent", "good")
                else "Manual review recommended"
            ),
        }


# Test the pipeline
pipeline = TranslationQualityPipeline()

# English to Korean with glossary
result = pipeline.translate(
    text=(
        "The machine learning model uses gradient descent to optimize the loss function. "
        "After training on the dataset, the model achieved 95% accuracy on the test set. "
        "We deployed the model using a microservice architecture with Docker containers."
    ),
    source_lang="English",
    target_lang="Korean",
    glossary={
        "machine learning": "기계 학습",
        "gradient descent": "경사 하강법",
        "loss function": "손실 함수",
        "microservice": "마이크로서비스",
    },
    style="formal technical",
)

print("Original:", result["original"])
print("\nTranslation:", result["translation"])
print("\nBack-translation:", result["back_translation"])
print(f"\nGlossary compliance: {result['glossary_compliance']['compliance_rate']:.0%}")
for term, info in result["glossary_compliance"]["terms"].items():
    status = "FOUND" if info["found"] else "MISSING"
    print(f"  [{status}] {term} → {info['expected']}")
print(f"\nSemantic similarity: {result['similarity']['semantic_similarity']}")
print(f"\nQuality Report:")
report = result["quality_report"]
print(f"  Score: {report['composite_score']}")
print(f"  Tier: {report['quality_tier']}")
print(f"  Recommendation: {report['recommendation']}")
if report["issues"]:
    print(f"  Issues:")
    for issue in report["issues"]:
        print(f"    - {issue}")
```

</details>

---

**Previous**: [13. Adversarial Prompting](./13_Adversarial_Prompting.md) | **Next**: [15. Prompt Management in Production](./15_Prompt_Management_in_Production.md)
