# 05. Structured Output Prompting

**Previous**: [Advanced Reasoning Prompts](./04_Advanced_Reasoning_Prompts.md) | **Next**: [System Prompt Design](./06_System_Prompt_Design.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Design prompts that reliably produce structured output in JSON, XML, YAML, and other formats
2. Implement schema-constrained generation using tool/function calling and Pydantic validation
3. Apply grammar-based decoding techniques for guaranteed output conformance
4. Handle nested, recursive, and complex data structures in LLM outputs
5. Build robust error recovery pipelines for malformed structured output

---

Unstructured natural language is easy for humans to read but hard for programs to consume. When LLMs power backend services, their output must conform to precise schemas that downstream code can parse without guessing. A single missing comma in JSON or an unexpected field name can crash a pipeline. Structured output prompting is the discipline of designing prompts and system configurations that guarantee machine-parseable responses -- transforming the LLM from a conversational partner into a reliable data generation engine.

This lesson covers the full spectrum of structured output techniques: from simple prompt-based formatting to schema-constrained decoding that makes malformed output mathematically impossible.

## Table of Contents

1. [Why Structured Output Matters](#1-why-structured-output-matters)
2. [JSON Output Prompting](#2-json-output-prompting)
3. [XML and HTML Output](#3-xml-and-html-output)
4. [YAML Output](#4-yaml-output)
5. [Schema-Constrained Generation](#5-schema-constrained-generation)
6. [Tool and Function Calling as Structured Output](#6-tool-and-function-calling-as-structured-output)
7. [Pydantic Model Validation](#7-pydantic-model-validation)
8. [Grammar-Based Decoding](#8-grammar-based-decoding)
9. [Handling Nested and Recursive Structures](#9-handling-nested-and-recursive-structures)
10. [Error Recovery for Malformed Output](#10-error-recovery-for-malformed-output)

---

## 1. Why Structured Output Matters

### 1.1 The Integration Problem

LLMs generate free-form text by default. When their output feeds into a REST API response, a database insert, or a configuration file, that free-form text must be parsed into a data structure. Without explicit prompting for structure, the model may:

- Wrap JSON in markdown code fences (`\`\`\`json ... \`\`\``)
- Include preamble text ("Here is the JSON output:")
- Use inconsistent field names across responses
- Omit required fields or add unexpected ones
- Produce syntactically invalid output under edge cases

Each of these failure modes requires defensive parsing code that is fragile and hard to maintain.

### 1.2 Reliability Spectrum

Structured output techniques fall on a reliability spectrum:

| Technique | Reliability | Flexibility | Latency Impact |
|-----------|-------------|-------------|----------------|
| Prompt instructions only | Low (~85-95%) | High | None |
| Few-shot examples | Medium (~92-97%) | High | Slight (more tokens) |
| System prompt + prefill | High (~95-99%) | Medium | None |
| Function/tool calling | Very High (~99%+) | Medium | Slight |
| Schema-constrained decoding | Guaranteed (100%) | Low | Moderate |
| Grammar-based decoding | Guaranteed (100%) | Low | Moderate |

The choice depends on your tolerance for parsing failures versus the constraints on latency and flexibility.

### 1.3 When to Use Each Format

- **JSON**: API responses, database records, configuration. Widest ecosystem support.
- **XML**: Document-oriented data, SOAP APIs, structured markup with attributes. Good for hierarchical data with mixed content.
- **YAML**: Configuration files, human-readable data. Beware of parsing ambiguities.
- **CSV/TSV**: Tabular data, spreadsheet export. Simple but limited to flat structures.
- **Custom formats**: Domain-specific needs (e.g., SQL, GraphQL queries).

---

## 2. JSON Output Prompting

### 2.1 Basic JSON Prompting

The simplest approach is to instruct the model to produce JSON:

```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": (
                "Extract the following information from this text and return "
                "it as a JSON object with keys: name, age, occupation, city.\n\n"
                "Text: Maria Chen is a 34-year-old software architect living "
                "in Seattle. She has been working at a major tech company for "
                "the past 8 years."
            )
        }
    ]
)

print(response.content[0].text)
```

This may produce valid JSON, but the model might wrap it in markdown fences or add explanatory text.

### 2.2 Forcing Pure JSON with Prefilling

Claude supports assistant message prefilling, which anchors the model's response to start with specific text:

```python
import anthropic
import json

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": (
                "Extract person info from this text. Return ONLY a JSON object "
                "with keys: name, age, occupation, city. No other text.\n\n"
                "Text: Maria Chen is a 34-year-old software architect living "
                "in Seattle."
            )
        },
        {
            "role": "assistant",
            "content": "{"  # Prefill forces JSON start
        }
    ]
)

# Reconstruct the full JSON (prefill + completion)
raw_json = "{" + response.content[0].text
data = json.loads(raw_json)
print(json.dumps(data, indent=2))
```

The prefill technique is highly effective because it eliminates preamble and forces the model into JSON-generation mode from the first token.

### 2.3 JSON with Explicit Schema

Providing the exact schema in the prompt dramatically improves reliability:

```python
import anthropic
import json

client = anthropic.Anthropic()

SCHEMA_PROMPT = """Extract entities from the text below. Return a JSON object
matching this exact schema:

{
  "people": [
    {
      "name": "string (full name)",
      "role": "string (job title or role)",
      "organization": "string or null",
      "relationships": ["string (relationship descriptions)"]
    }
  ],
  "locations": [
    {
      "name": "string",
      "type": "string (city|country|building|other)"
    }
  ],
  "dates": [
    {
      "value": "string (ISO 8601 format)",
      "context": "string (what the date refers to)"
    }
  ]
}

Rules:
- Return ONLY the JSON object, no other text
- Use null for unknown fields, never omit them
- Dates must be ISO 8601 (YYYY-MM-DD)
- Arrays may be empty but must be present

Text: """

text = (
    "On January 15, 2024, Dr. Sarah Kim, lead researcher at MIT's CSAIL lab, "
    "presented her findings on quantum error correction at the Berlin Conference "
    "Center. Her collaborator, Prof. James Liu from Stanford University, joined "
    "remotely from Palo Alto."
)

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=2048,
    messages=[
        {"role": "user", "content": SCHEMA_PROMPT + text},
        {"role": "assistant", "content": "{"}
    ]
)

result = json.loads("{" + response.content[0].text)
print(json.dumps(result, indent=2))
```

### 2.4 Common JSON Pitfalls

**Problem 1: Trailing commas**

LLMs sometimes produce trailing commas in arrays or objects, which is invalid JSON:

```python
# Invalid JSON the model might produce
bad_json = '{"items": ["apple", "banana", "cherry",]}'

# Fix with regex before parsing
import re

def fix_trailing_commas(json_str: str) -> str:
    """Remove trailing commas before closing brackets."""
    json_str = re.sub(r",\s*}", "}", json_str)
    json_str = re.sub(r",\s*]", "]", json_str)
    return json_str
```

**Problem 2: Unescaped characters in strings**

```python
# The model might produce unescaped newlines or quotes in string values
# Use a lenient JSON parser as fallback
import json

def parse_json_lenient(text: str) -> dict:
    """Try strict parsing first, fall back to repair."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Strip markdown fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
        # Fix common issues
        text = fix_trailing_commas(text)
        return json.loads(text)
```

**Problem 3: Number precision**

```python
# LLMs may produce numbers that lose precision in JSON parsing
# Use decimal for financial data
from decimal import Decimal

raw = '{"price": 19.99, "quantity": 3}'
# json.loads gives float: 19.99 (may have floating point issues)
# Use parse_float to preserve precision
data = json.loads(raw, parse_float=Decimal)
print(data["price"])  # Decimal('19.99')
```

---

## 3. XML and HTML Output

### 3.1 Why XML for LLM Output

XML is naturally suited to LLM output because models have seen enormous amounts of XML/HTML in training data. Claude in particular works well with XML tags for structuring both input and output. Advantages:

- Hierarchical with mixed content (text + structure)
- Attributes provide metadata without extra nesting
- Self-describing tag names
- Robust parsers available in every language

### 3.2 Basic XML Output

```python
import anthropic
from xml.etree import ElementTree as ET

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=2048,
    messages=[
        {
            "role": "user",
            "content": (
                "Analyze the sentiment of each sentence in the text below. "
                "Return the results as XML with this structure:\n\n"
                "<analysis>\n"
                "  <sentence id=\"1\" sentiment=\"positive|negative|neutral\" "
                "confidence=\"0.0-1.0\">\n"
                "    <text>The original sentence</text>\n"
                "    <reasoning>Why this sentiment was assigned</reasoning>\n"
                "  </sentence>\n"
                "</analysis>\n\n"
                "Text: The product arrived quickly and works great. However, "
                "the packaging was damaged. Customer service was helpful when "
                "I reported the issue."
            )
        }
    ]
)

# Parse the XML output
xml_text = response.content[0].text

# Extract just the XML if wrapped in other text
import re
xml_match = re.search(r"<analysis>.*?</analysis>", xml_text, re.DOTALL)
if xml_match:
    xml_text = xml_match.group()

root = ET.fromstring(xml_text)
for sentence in root.findall("sentence"):
    sid = sentence.get("id")
    sentiment = sentence.get("sentiment")
    confidence = sentence.get("confidence")
    text = sentence.find("text").text
    print(f"[{sid}] {sentiment} ({confidence}): {text}")
```

### 3.3 Claude's XML Tag Convention

Anthropic recommends using XML tags to structure both prompts and outputs. This is a distinctive feature of Claude's prompting style:

```python
import anthropic

client = anthropic.Anthropic()

prompt = """Classify the following support tickets. For each ticket, provide:
- Category (billing, technical, account, other)
- Priority (low, medium, high, urgent)
- Summary (one sentence)

Return your analysis inside <tickets> tags:

<tickets>
  <ticket id="1" category="..." priority="...">
    <summary>...</summary>
  </ticket>
</tickets>

<input_tickets>
Ticket 1: "I've been charged twice for my subscription this month. Please refund."
Ticket 2: "The API returns 500 errors intermittently since the last update."
Ticket 3: "How do I change my email address on my account?"
</input_tickets>"""

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": prompt}]
)

print(response.content[0].text)
```

### 3.4 HTML Generation

For generating HTML snippets (e.g., email templates, report fragments):

```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=2048,
    system=(
        "You generate clean, semantic HTML5 snippets. Use only standard HTML "
        "elements. No inline styles -- use class attributes for styling hooks. "
        "Return ONLY the HTML, no explanations."
    ),
    messages=[
        {
            "role": "user",
            "content": (
                "Create an HTML table summarizing this data:\n"
                "Q1 2024: Revenue $1.2M, Costs $800K, Profit $400K\n"
                "Q2 2024: Revenue $1.5M, Costs $900K, Profit $600K\n"
                "Q3 2024: Revenue $1.8M, Costs $950K, Profit $850K\n"
                "Include a <caption>, <thead>, and <tbody>. Add a totals row "
                "in <tfoot>."
            )
        }
    ]
)

html = response.content[0].text
print(html)
```

---

## 4. YAML Output

### 4.1 YAML Prompting Considerations

YAML is whitespace-sensitive, which makes it trickier for LLMs. Models can produce inconsistent indentation, and YAML has parsing ambiguities (e.g., `yes`/`no` as booleans, unquoted strings containing colons). Use YAML output when the result will be read or edited by humans.

```python
import anthropic
import yaml

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=2048,
    messages=[
        {
            "role": "user",
            "content": (
                "Generate a YAML configuration for a web application with "
                "the following requirements:\n"
                "- App name: TaskManager\n"
                "- Port: 8080\n"
                "- Database: PostgreSQL on localhost:5432\n"
                "- Redis cache on localhost:6379\n"
                "- Logging: INFO level, file and console outputs\n"
                "- CORS: allow origins from localhost:3000 and example.com\n\n"
                "Return ONLY valid YAML, no markdown fences or explanations. "
                "Use 2-space indentation consistently."
            )
        }
    ]
)

yaml_text = response.content[0].text

# Strip markdown fences if present
import re
yaml_text = re.sub(r"^```(?:yaml)?\s*\n?", "", yaml_text)
yaml_text = re.sub(r"\n?```\s*$", "", yaml_text)

config = yaml.safe_load(yaml_text)
print(yaml.dump(config, default_flow_style=False))
```

### 4.2 YAML Safety Concerns

Always use `yaml.safe_load()` rather than `yaml.load()` when parsing LLM output. The full `yaml.load()` can execute arbitrary Python code through YAML tags:

```python
import yaml

# DANGEROUS: Never do this with LLM output
# data = yaml.load(llm_output, Loader=yaml.FullLoader)

# SAFE: Always use safe_load
data = yaml.safe_load(llm_output)
```

### 4.3 YAML vs JSON Trade-offs

| Feature | JSON | YAML |
|---------|------|------|
| LLM reliability | Higher (strict syntax) | Lower (whitespace-sensitive) |
| Human readability | Good | Excellent |
| Comment support | No | Yes |
| Multi-line strings | Escaped `\n` | Block scalars `|` / `>` |
| Parsing ambiguity | Low | Higher (booleans, nulls) |
| Ecosystem support | Universal | Broad but not universal |

**Recommendation**: Use JSON for machine-to-machine pipelines and YAML only when human editing is a primary use case.

---

## 5. Schema-Constrained Generation

### 5.1 What is Schema-Constrained Generation?

Schema-constrained generation forces the model's output to conform to a predefined schema at the decoding level, not just through prompt instructions. This means the model literally cannot produce invalid output because tokens that would violate the schema are masked during generation.

### 5.2 OpenAI Structured Outputs

OpenAI provides native structured output support through the `response_format` parameter:

```python
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()


class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]
    location: str | None
    is_recurring: bool


response = client.responses.create(
    model="gpt-4o-2024-08-06",
    input=[
        {
            "role": "user",
            "content": (
                "Extract the event details: 'Team standup every Monday at "
                "9am in Room 301 with Alice, Bob, and Carol.'"
            )
        }
    ],
    text={
        "format": {
            "type": "json_schema",
            "name": "calendar_event",
            "schema": CalendarEvent.model_json_schema(),
            "strict": True
        }
    }
)

import json
event = json.loads(response.output_text)
print(json.dumps(event, indent=2))
```

### 5.3 Anthropic's Approach

As of 2025, Anthropic supports structured output through tool use (covered in Section 6) and prompt-based techniques. Claude excels at following explicit JSON schemas in prompts:

```python
import anthropic
import json

client = anthropic.Anthropic()

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "Event name"},
        "date": {"type": "string", "format": "date"},
        "participants": {
            "type": "array",
            "items": {"type": "string"}
        },
        "location": {"type": ["string", "null"]},
        "is_recurring": {"type": "boolean"}
    },
    "required": ["name", "date", "participants", "is_recurring"]
}

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system=(
        "You extract structured data from text. Always respond with a single "
        "JSON object matching the provided schema. Never include explanatory text."
    ),
    messages=[
        {
            "role": "user",
            "content": (
                f"Schema: {json.dumps(schema, indent=2)}\n\n"
                "Text: Team standup every Monday at 9am in Room 301 with "
                "Alice, Bob, and Carol."
            )
        },
        {"role": "assistant", "content": "{"}
    ]
)

event = json.loads("{" + response.content[0].text)
print(json.dumps(event, indent=2))
```

### 5.4 JSON Schema Tips for LLMs

When providing schemas to LLMs, follow these guidelines:

1. **Include descriptions**: Field descriptions guide the model more than type constraints alone
2. **Use enums for controlled vocabularies**: `"enum": ["low", "medium", "high"]`
3. **Specify formats**: `"format": "date"`, `"format": "email"`, etc.
4. **Set required fields explicitly**: Do not rely on the model inferring which fields matter
5. **Provide examples in descriptions**: `"description": "ISO 8601 date, e.g. 2024-01-15"`

---

## 6. Tool and Function Calling as Structured Output

### 6.1 The Tool Use Pattern

Tool/function calling was designed for LLMs to interact with external tools, but it doubles as an excellent structured output mechanism. When you define a "tool" whose schema matches your desired output, the model fills in the parameters -- and the API validates the structure.

### 6.2 Claude Tool Use for Structured Output

```python
import anthropic
import json

client = anthropic.Anthropic()

# Define a "tool" that is actually an output schema
tools = [
    {
        "name": "extract_product_review",
        "description": (
            "Extract structured information from a product review. "
            "Call this tool with the extracted data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "product_name": {
                    "type": "string",
                    "description": "Name of the product being reviewed"
                },
                "rating": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "description": "Star rating (1-5)"
                },
                "pros": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of positive aspects"
                },
                "cons": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of negative aspects"
                },
                "verdict": {
                    "type": "string",
                    "enum": ["recommended", "neutral", "not_recommended"],
                    "description": "Overall recommendation"
                },
                "key_quote": {
                    "type": "string",
                    "description": "Most representative quote from the review"
                }
            },
            "required": [
                "product_name", "rating", "pros", "cons",
                "verdict", "key_quote"
            ]
        }
    }
]

review_text = (
    "I bought the UltraSound X50 wireless headphones last month. The noise "
    "cancellation is phenomenal -- easily the best I've tried under $200. "
    "Battery life is solid at around 30 hours. However, the ear cushions "
    "started peeling after just two weeks, and the Bluetooth range is "
    "disappointing -- drops out past 15 feet. The companion app is also "
    "buggy. Overall, the sound quality makes up for the build quality "
    "issues, but I'd wait for the next version. 3 out of 5 stars."
)

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=tools,
    tool_choice={"type": "tool", "name": "extract_product_review"},
    messages=[
        {
            "role": "user",
            "content": f"Extract review data:\n\n{review_text}"
        }
    ]
)

# The tool call contains validated structured output
for block in response.content:
    if block.type == "tool_use":
        print(json.dumps(block.input, indent=2))
```

### 6.3 OpenAI Function Calling

```python
from openai import OpenAI
import json

client = OpenAI()

tools = [
    {
        "type": "function",
        "function": {
            "name": "extract_product_review",
            "description": "Extract structured data from a product review",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string"},
                    "rating": {"type": "integer", "minimum": 1, "maximum": 5},
                    "pros": {"type": "array", "items": {"type": "string"}},
                    "cons": {"type": "array", "items": {"type": "string"}},
                    "verdict": {
                        "type": "string",
                        "enum": ["recommended", "neutral", "not_recommended"]
                    }
                },
                "required": [
                    "product_name", "rating", "pros", "cons", "verdict"
                ]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": (
                "Extract review data: The UltraSound X50 headphones have "
                "great noise cancellation but poor build quality. 3/5 stars."
            )
        }
    ],
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "extract_product_review"}}
)

tool_call = response.choices[0].message.tool_calls[0]
data = json.loads(tool_call.function.arguments)
print(json.dumps(data, indent=2))
```

### 6.4 Benefits of Tool Calling for Structured Output

1. **API-level validation**: The API validates the output against the schema
2. **No parsing needed**: Output arrives as a structured object, not a string
3. **No prefill hacks**: Works cleanly within the standard API flow
4. **Enum enforcement**: The model respects enum constraints more reliably
5. **Type coercion**: Numbers, booleans, and nulls are properly typed

---

## 7. Pydantic Model Validation

### 7.1 Pydantic as a Validation Layer

Even with structured output techniques, adding a Pydantic validation layer provides defense in depth:

```python
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum


class Verdict(str, Enum):
    RECOMMENDED = "recommended"
    NEUTRAL = "neutral"
    NOT_RECOMMENDED = "not_recommended"


class ProductReview(BaseModel):
    product_name: str = Field(min_length=1, max_length=200)
    rating: int = Field(ge=1, le=5)
    pros: list[str] = Field(min_length=1)
    cons: list[str] = Field(default_factory=list)
    verdict: Verdict
    key_quote: Optional[str] = None

    @field_validator("pros", "cons")
    @classmethod
    def no_empty_strings(cls, v: list[str]) -> list[str]:
        return [item.strip() for item in v if item.strip()]

    @field_validator("key_quote")
    @classmethod
    def quote_not_too_long(cls, v: Optional[str]) -> Optional[str]:
        if v and len(v) > 500:
            return v[:500] + "..."
        return v
```

### 7.2 Full Pipeline with Pydantic Validation

```python
import anthropic
import json
from pydantic import ValidationError


def extract_review(text: str) -> ProductReview:
    """Extract and validate a product review from text."""
    client = anthropic.Anthropic()

    tools = [
        {
            "name": "extract_review",
            "description": "Extract product review data",
            "input_schema": ProductReview.model_json_schema()
        }
    ]

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=tools,
        tool_choice={"type": "tool", "name": "extract_review"},
        messages=[
            {"role": "user", "content": f"Extract review data:\n\n{text}"}
        ]
    )

    for block in response.content:
        if block.type == "tool_use":
            try:
                return ProductReview.model_validate(block.input)
            except ValidationError as e:
                print(f"Validation failed: {e}")
                raise

    raise ValueError("No tool use block found in response")


# Usage
review = extract_review(
    "The X50 headphones have great sound but poor build quality. "
    "3/5 stars. Recommended for audio enthusiasts on a budget."
)
print(review.model_dump_json(indent=2))
```

### 7.3 Generating Schemas from Pydantic Models

Pydantic models can generate JSON schemas that you pass directly to the LLM:

```python
from pydantic import BaseModel, Field
from typing import Optional


class Address(BaseModel):
    street: str
    city: str
    state: Optional[str] = None
    country: str
    postal_code: str = Field(pattern=r"^\d{5}(-\d{4})?$")


class Person(BaseModel):
    name: str = Field(description="Full legal name")
    email: str = Field(description="Primary email address")
    age: Optional[int] = Field(None, ge=0, le=150)
    address: Address
    tags: list[str] = Field(default_factory=list)


# Generate schema for the prompt
schema = Person.model_json_schema()
print(json.dumps(schema, indent=2))
# Pass this schema into your prompt or tool definition
```

---

## 8. Grammar-Based Decoding

### 8.1 What is Grammar-Based Decoding?

Grammar-based decoding constrains the model's token generation using a formal grammar (typically context-free). At each step, only tokens that are valid continuations under the grammar are allowed. This guarantees 100% syntactic validity.

### 8.2 GBNF Grammars (llama.cpp)

The llama.cpp ecosystem uses GBNF (GGML BNF) notation for grammar-based decoding:

```
# GBNF grammar for a simple JSON object with specific fields
root   ::= "{" ws "\"name\"" ws ":" ws string "," ws "\"age\"" ws ":" ws number "," ws "\"city\"" ws ":" ws string "}" ws
string ::= "\"" [^"\\]* "\""
number ::= [0-9]+
ws     ::= [ \t\n]*
```

This grammar ensures the output is always a JSON object with exactly `name`, `age`, and `city` fields.

### 8.3 Outlines Library

The `outlines` library provides grammar-based decoding for Hugging Face models:

```python
# Note: outlines works with local models, not API-based models
# This example shows the concept

# pip install outlines
import outlines

model = outlines.models.transformers("mistralai/Mistral-7B-v0.1")

# JSON schema-based generation
from pydantic import BaseModel

class Character(BaseModel):
    name: str
    age: int
    weapon: str

generator = outlines.generate.json(model, Character)
character = generator("Create a fantasy RPG character:")
print(character)
# Output is guaranteed to be a valid Character object
```

### 8.4 Regex-Constrained Generation

For simpler patterns, regex constraints are sufficient:

```python
import outlines

model = outlines.models.transformers("mistralai/Mistral-7B-v0.1")

# Generate only valid email addresses
email_generator = outlines.generate.regex(
    model,
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
)
email = email_generator("Generate an email for John Smith at Acme Corp:")
print(email)  # Guaranteed to match email pattern
```

### 8.5 When to Use Grammar-Based Decoding

**Use when:**
- You need 100% format guarantees (safety-critical applications)
- You're running local models and can control the inference engine
- The output schema is well-defined and static

**Avoid when:**
- You're using API-based models (grammar decoding requires inference-level control)
- The output structure is dynamic or context-dependent
- Latency is critical (grammar masking adds overhead)

---

## 9. Handling Nested and Recursive Structures

### 9.1 Nested Object Prompting

Real-world data often has deep nesting. The key is to show the full nested structure in your schema:

```python
import anthropic
import json

client = anthropic.Anthropic()

nested_schema = {
    "type": "object",
    "properties": {
        "company": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "departments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "head": {"type": "string"},
                            "teams": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "size": {"type": "integer"},
                                        "projects": {
                                            "type": "array",
                                            "items": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=4096,
    system=(
        "You extract organizational data into structured JSON. "
        "Match the provided schema exactly."
    ),
    messages=[
        {
            "role": "user",
            "content": (
                f"Schema:\n{json.dumps(nested_schema, indent=2)}\n\n"
                "Text: Acme Corp has two departments. Engineering, led by "
                "Sarah Chen, has the Platform team (12 people, working on "
                "API Gateway and Auth Service) and the ML team (8 people, "
                "working on Recommendation Engine). Sales, led by Mike Ross, "
                "has the Enterprise team (15 people, working on Fortune 500 "
                "Accounts and Government Contracts)."
            )
        },
        {"role": "assistant", "content": "{"}
    ]
)

result = json.loads("{" + response.content[0].text)
print(json.dumps(result, indent=2))
```

### 9.2 Recursive Structures

Some data structures are inherently recursive (e.g., file trees, org charts, comment threads). JSON Schema supports `$ref` for recursion, but LLMs handle it better with explicit depth limits:

```python
import anthropic
import json

client = anthropic.Anthropic()

# Define a tool with a recursive schema
tools = [
    {
        "name": "parse_outline",
        "description": "Parse a document outline into a recursive tree structure",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "sections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "heading": {"type": "string"},
                            "level": {"type": "integer", "minimum": 1, "maximum": 4},
                            "summary": {"type": "string"},
                            "subsections": {
                                "type": "array",
                                "description": "Nested subsections (same structure, max depth 3)",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "heading": {"type": "string"},
                                        "level": {"type": "integer"},
                                        "summary": {"type": "string"},
                                        "subsections": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "heading": {"type": "string"},
                                                    "level": {"type": "integer"},
                                                    "summary": {"type": "string"}
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "required": ["title", "sections"]
        }
    }
]

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=4096,
    tools=tools,
    tool_choice={"type": "tool", "name": "parse_outline"},
    messages=[
        {
            "role": "user",
            "content": (
                "Parse this document outline:\n\n"
                "Machine Learning Handbook\n"
                "  1. Supervised Learning\n"
                "    1.1 Classification\n"
                "      1.1.1 Decision Trees\n"
                "      1.1.2 Neural Networks\n"
                "    1.2 Regression\n"
                "  2. Unsupervised Learning\n"
                "    2.1 Clustering\n"
                "    2.2 Dimensionality Reduction\n"
            )
        }
    ]
)

for block in response.content:
    if block.type == "tool_use":
        print(json.dumps(block.input, indent=2))
```

### 9.3 Strategies for Deep Nesting

1. **Flatten then reconstruct**: Ask the model to produce a flat list with parent references, then reconstruct the tree in code
2. **Level-by-level generation**: Generate each level separately, using the previous level as context
3. **Explicit depth limits**: Tell the model the maximum nesting depth
4. **ID-based references**: Use IDs and parent IDs instead of literal nesting

```python
# Flat structure with references (easier for LLMs)
flat_schema = {
    "type": "object",
    "properties": {
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "parent_id": {"type": ["string", "null"]},
                    "label": {"type": "string"},
                    "data": {"type": "object"}
                },
                "required": ["id", "parent_id", "label"]
            }
        }
    }
}

# Reconstruct tree from flat list
def build_tree(nodes: list[dict]) -> dict:
    """Convert flat node list to nested tree."""
    lookup = {n["id"]: {**n, "children": []} for n in nodes}
    root = None
    for node in nodes:
        parent_id = node["parent_id"]
        if parent_id is None:
            root = lookup[node["id"]]
        else:
            lookup[parent_id]["children"].append(lookup[node["id"]])
    return root
```

---

## 10. Error Recovery for Malformed Output

### 10.1 Defense in Depth Strategy

Even the best prompting techniques occasionally produce malformed output. A production system needs layered error handling:

```
Layer 1: Prompt design (prevents most errors)
Layer 2: Output extraction (strips wrapper text)
Layer 3: Syntax repair (fixes common JSON issues)
Layer 4: Validation (Pydantic / schema check)
Layer 5: Retry with error feedback (LLM self-correction)
Layer 6: Fallback (default values or human escalation)
```

### 10.2 Robust JSON Extraction Pipeline

```python
import anthropic
import json
import re
from typing import Any, Optional
from pydantic import BaseModel, ValidationError


def extract_json_from_text(text: str) -> Optional[str]:
    """Extract JSON from text that may contain markdown fences or preamble."""
    # Try to find JSON in code fences
    fence_match = re.search(
        r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL
    )
    if fence_match:
        return fence_match.group(1).strip()

    # Try to find a JSON object or array
    for pattern in [
        r"(\{[\s\S]*\})",   # JSON object
        r"(\[[\s\S]*\])",   # JSON array
    ]:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()

    return text.strip()


def repair_json(text: str) -> str:
    """Fix common JSON syntax errors from LLM output."""
    # Remove trailing commas
    text = re.sub(r",\s*([}\]])", r"\1", text)
    # Fix single quotes to double quotes (naive -- works for simple cases)
    # Only if no double quotes are present at all
    if '"' not in text and "'" in text:
        text = text.replace("'", '"')
    # Remove comments (// style)
    text = re.sub(r"//[^\n]*", "", text)
    # Fix unquoted keys (simple cases)
    text = re.sub(r"(?<=\{|,)\s*(\w+)\s*:", r' "\1":', text)
    return text


def parse_llm_json(
    text: str,
    model_class: Optional[type[BaseModel]] = None,
    max_retries: int = 1
) -> Any:
    """Parse JSON from LLM output with repair and validation."""
    # Layer 2: Extract JSON
    json_str = extract_json_from_text(text)

    # Layer 3: Try parsing, repair if needed
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        repaired = repair_json(json_str)
        try:
            data = json.loads(repaired)
        except json.JSONDecodeError as e:
            if max_retries > 0:
                return None  # Signal for retry
            raise ValueError(f"Could not parse JSON after repair: {e}")

    # Layer 4: Validate with Pydantic if model provided
    if model_class:
        try:
            return model_class.model_validate(data)
        except ValidationError as e:
            if max_retries > 0:
                return None  # Signal for retry with error context
            raise

    return data
```

### 10.3 Retry with Error Feedback

When initial parsing fails, send the error back to the model for self-correction:

```python
import anthropic
import json
from pydantic import BaseModel, ValidationError
from typing import TypeVar, Type

T = TypeVar("T", bound=BaseModel)


def extract_with_retry(
    prompt: str,
    model_class: Type[T],
    max_retries: int = 2
) -> T:
    """Extract structured data with automatic retry on failure."""
    client = anthropic.Anthropic()

    messages = [{"role": "user", "content": prompt}]

    for attempt in range(max_retries + 1):
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=(
                "You extract structured data as JSON. Return ONLY valid JSON "
                "matching the requested schema. No markdown, no explanations."
            ),
            messages=messages
        )

        raw = response.content[0].text
        json_str = extract_json_from_text(raw)

        try:
            data = json.loads(json_str)
            return model_class.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            if attempt < max_retries:
                # Add the failed response and error feedback
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your response had an error:\n{e}\n\n"
                        f"Please fix the JSON and try again. Return ONLY "
                        f"the corrected JSON object."
                    )
                })
            else:
                raise ValueError(
                    f"Failed after {max_retries + 1} attempts: {e}"
                )


# Usage
class MovieReview(BaseModel):
    title: str
    year: int
    rating: float
    genres: list[str]
    summary: str


review = extract_with_retry(
    "Extract movie info: 'Inception (2010) is a mind-bending sci-fi thriller "
    "by Christopher Nolan. 9.2/10. A thief who steals corporate secrets "
    "through dream-sharing technology is given the task of planting an idea "
    "into a CEO\\'s mind.'",
    MovieReview
)
print(review.model_dump_json(indent=2))
```

### 10.4 Streaming JSON Parsing

For large structured outputs, you may want to parse incrementally as tokens arrive:

```python
import anthropic
import json


def stream_json_objects(prompt: str) -> list[dict]:
    """Stream a response and extract JSON objects incrementally."""
    client = anthropic.Anthropic()

    collected = ""
    objects = []

    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    ) as stream:
        for text in stream.text_stream:
            collected += text

            # Try to parse complete JSON objects as they form
            # This is a simplified approach -- production code would
            # use a streaming JSON parser like ijson
            while True:
                try:
                    # Try to find and parse a complete JSON object
                    start = collected.find("{")
                    if start == -1:
                        break

                    # Try parsing from the first { to find a complete object
                    for end in range(start + 1, len(collected) + 1):
                        try:
                            obj = json.loads(collected[start:end])
                            objects.append(obj)
                            collected = collected[end:]
                            break
                        except json.JSONDecodeError:
                            continue
                    else:
                        break  # No complete object yet
                except Exception:
                    break

    return objects
```

### 10.5 Monitoring and Alerting

In production, track structured output failures:

```python
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StructuredOutputMetrics:
    """Track structured output success rates."""
    total_attempts: int = 0
    first_try_success: int = 0
    retry_success: int = 0
    total_failures: int = 0
    parse_errors: list[str] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)
    avg_retries: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.total_attempts == 0:
            return 0.0
        return (
            (self.first_try_success + self.retry_success)
            / self.total_attempts
        )

    def record_success(self, retries: int = 0) -> None:
        self.total_attempts += 1
        if retries == 0:
            self.first_try_success += 1
        else:
            self.retry_success += 1

    def record_failure(
        self, error_type: str, error_msg: str
    ) -> None:
        self.total_attempts += 1
        self.total_failures += 1
        if error_type == "parse":
            self.parse_errors.append(error_msg)
        else:
            self.validation_errors.append(error_msg)

    def report(self) -> dict:
        return {
            "success_rate": f"{self.success_rate:.1%}",
            "total_attempts": self.total_attempts,
            "first_try_success": self.first_try_success,
            "retry_success": self.retry_success,
            "failures": self.total_failures,
            "unique_parse_errors": len(set(self.parse_errors)),
            "unique_validation_errors": len(set(self.validation_errors))
        }
```

---

## Exercises

### Exercise 1: Multi-Format Extraction Pipeline

Build a function that extracts structured data from an article and returns it in the user's choice of format (JSON, XML, or YAML). The extracted data should include: title, author, publication date, summary (max 100 words), key topics (list), and sentiment (positive/negative/neutral).

**Requirements:**
- Support all three output formats via a `format` parameter
- Use Pydantic for validation regardless of output format
- Handle format conversion in code, not by asking the model for different formats

<details><summary>Show Answer</summary>

```python
import anthropic
import json
import yaml
from xml.etree.ElementTree import Element, SubElement, tostring
from pydantic import BaseModel, Field
from typing import Literal
from enum import Enum


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class ArticleData(BaseModel):
    title: str
    author: str
    publication_date: str = Field(description="ISO 8601 date")
    summary: str = Field(max_length=500)
    key_topics: list[str] = Field(min_length=1)
    sentiment: Sentiment


def extract_article(
    text: str,
    output_format: Literal["json", "xml", "yaml"] = "json"
) -> str:
    """Extract article data and return in specified format."""
    client = anthropic.Anthropic()

    tools = [
        {
            "name": "extract_article",
            "description": "Extract structured data from an article",
            "input_schema": ArticleData.model_json_schema()
        }
    ]

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=tools,
        tool_choice={"type": "tool", "name": "extract_article"},
        messages=[
            {"role": "user", "content": f"Extract article data:\n\n{text}"}
        ]
    )

    # Validate with Pydantic
    raw_data = None
    for block in response.content:
        if block.type == "tool_use":
            raw_data = block.input

    article = ArticleData.model_validate(raw_data)
    data = article.model_dump()

    # Convert to requested format
    if output_format == "json":
        return json.dumps(data, indent=2)
    elif output_format == "yaml":
        return yaml.dump(data, default_flow_style=False, allow_unicode=True)
    elif output_format == "xml":
        root = Element("article")
        for key, value in data.items():
            child = SubElement(root, key)
            if isinstance(value, list):
                for item in value:
                    item_el = SubElement(child, "item")
                    item_el.text = str(item)
            else:
                child.text = str(value)
        return tostring(root, encoding="unicode")

    raise ValueError(f"Unknown format: {output_format}")


# Test
sample = (
    "AI Startup Raises $50M in Series B\n"
    "By Jane Doe, March 10, 2025\n\n"
    "TechAI, a leading artificial intelligence startup, announced today "
    "that it has raised $50 million in Series B funding. The round was "
    "led by Sequoia Capital, with participation from existing investors."
)

for fmt in ["json", "xml", "yaml"]:
    print(f"\n--- {fmt.upper()} ---")
    print(extract_article(sample, fmt))
```

</details>

### Exercise 2: Schema Evolution Handler

Create a system that handles schema versioning. Given an old JSON response conforming to schema v1, transform it to conform to schema v2 using an LLM, while maintaining backward compatibility.

**Requirements:**
- Define v1 and v2 Pydantic models with meaningful differences (renamed fields, new required fields, type changes)
- Use Claude to transform v1 data to v2 format
- Validate both input (v1) and output (v2) with Pydantic

<details><summary>Show Answer</summary>

```python
import anthropic
import json
from pydantic import BaseModel, Field
from typing import Optional


# Schema v1
class UserProfileV1(BaseModel):
    name: str
    email: str
    age: int
    city: str
    interests: str  # Comma-separated


# Schema v2 (evolved)
class Address(BaseModel):
    city: str
    country: str = "Unknown"


class UserProfileV2(BaseModel):
    full_name: str  # Renamed from 'name'
    email: str
    age_group: str  # Changed from exact age to group
    address: Address  # Nested object replacing 'city'
    interests: list[str]  # Changed from comma-separated to list
    profile_version: int = 2  # New required field


def migrate_v1_to_v2(v1_data: UserProfileV1) -> UserProfileV2:
    """Migrate user profile from v1 to v2 using LLM for smart transforms."""
    client = anthropic.Anthropic()

    tools = [
        {
            "name": "create_v2_profile",
            "description": "Create a v2 user profile from v1 data",
            "input_schema": UserProfileV2.model_json_schema()
        }
    ]

    prompt = f"""Transform this v1 user profile to v2 format.

V1 data:
{v1_data.model_dump_json(indent=2)}

Transformation rules:
- 'name' -> 'full_name': keep as-is
- 'age' -> 'age_group': map to "under_18", "18-25", "26-35", "36-50", "51-65", "over_65"
- 'city' -> 'address.city': keep city, infer country if possible, else "Unknown"
- 'interests': split comma-separated string into a list of trimmed strings
- 'profile_version': always set to 2"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=tools,
        tool_choice={"type": "tool", "name": "create_v2_profile"},
        messages=[{"role": "user", "content": prompt}]
    )

    for block in response.content:
        if block.type == "tool_use":
            return UserProfileV2.model_validate(block.input)

    raise ValueError("Migration failed: no tool use in response")


# Test
v1 = UserProfileV1(
    name="Alice Johnson",
    email="alice@example.com",
    age=29,
    city="Tokyo",
    interests="machine learning, hiking, photography"
)

v2 = migrate_v1_to_v2(v1)
print("V1:", v1.model_dump_json(indent=2))
print("\nV2:", v2.model_dump_json(indent=2))
```

</details>

### Exercise 3: Robust JSON Array Streaming

Write a function that asks Claude to generate a JSON array of N items (e.g., fictional product entries), streams the response, and yields each complete object as it becomes available. Handle the case where the stream is interrupted mid-object.

**Requirements:**
- Use the Anthropic streaming API
- Yield each complete JSON object as soon as it closes
- Track partial objects for recovery
- Include timeout handling

<details><summary>Show Answer</summary>

```python
import anthropic
import json
import re
from typing import Generator


def stream_json_array(
    prompt: str,
    n_items: int = 5
) -> Generator[dict, None, None]:
    """Stream a JSON array and yield each object as it completes."""
    client = anthropic.Anthropic()

    full_prompt = (
        f"{prompt}\n\n"
        f"Generate exactly {n_items} items as a JSON array. "
        f"Each item should be a JSON object on its own. "
        f"Return ONLY the JSON array."
    )

    buffer = ""
    brace_depth = 0
    in_string = False
    escape_next = False
    object_start = -1
    objects_yielded = 0

    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[
            {"role": "user", "content": full_prompt},
        ]
    ) as stream:
        for text in stream.text_stream:
            buffer += text

            # Scan newly added characters
            i = len(buffer) - len(text)
            while i < len(buffer):
                char = buffer[i]

                if escape_next:
                    escape_next = False
                    i += 1
                    continue

                if char == "\\" and in_string:
                    escape_next = True
                    i += 1
                    continue

                if char == '"':
                    in_string = not in_string
                    i += 1
                    continue

                if in_string:
                    i += 1
                    continue

                if char == "{":
                    if brace_depth == 0:
                        object_start = i
                    brace_depth += 1
                elif char == "}":
                    brace_depth -= 1
                    if brace_depth == 0 and object_start >= 0:
                        # Complete object found
                        obj_str = buffer[object_start:i + 1]
                        try:
                            obj = json.loads(obj_str)
                            objects_yielded += 1
                            yield obj
                        except json.JSONDecodeError:
                            pass  # Skip malformed objects
                        object_start = -1

                i += 1

    # Handle any remaining partial object
    if object_start >= 0 and brace_depth > 0:
        partial = buffer[object_start:]
        # Try to close the object
        repaired = partial + "}" * brace_depth
        try:
            obj = json.loads(repaired)
            yield obj
        except json.JSONDecodeError:
            print(f"Warning: Could not recover partial object")

    if objects_yielded == 0:
        # Fallback: try to parse the entire buffer
        try:
            data = json.loads(buffer)
            if isinstance(data, list):
                for item in data:
                    yield item
        except json.JSONDecodeError:
            print("Warning: No valid JSON objects found in stream")


# Usage
for i, product in enumerate(stream_json_array(
    "Generate fictional product entries with fields: "
    "name, price (float), category, in_stock (bool)",
    n_items=5
)):
    print(f"Item {i+1}: {json.dumps(product)}")
```

</details>

### Exercise 4: XML to Pydantic Pipeline

Build a system that prompts Claude to produce XML output, parses it, and converts it to validated Pydantic models. The use case: parse a recipe from unstructured text into a structured format.

**Requirements:**
- Prompt Claude to output XML (not JSON)
- Parse the XML using `xml.etree.ElementTree`
- Convert to a Pydantic model with proper types (durations, quantities as numbers)
- Handle missing optional fields gracefully

<details><summary>Show Answer</summary>

```python
import anthropic
import re
from xml.etree import ElementTree as ET
from pydantic import BaseModel, Field
from typing import Optional


class Ingredient(BaseModel):
    name: str
    quantity: float
    unit: str
    notes: Optional[str] = None


class Step(BaseModel):
    number: int
    instruction: str
    duration_minutes: Optional[int] = None


class Recipe(BaseModel):
    name: str
    servings: int
    prep_time_minutes: int
    cook_time_minutes: int
    difficulty: str = Field(pattern=r"^(easy|medium|hard)$")
    ingredients: list[Ingredient]
    steps: list[Step]
    tags: list[str] = Field(default_factory=list)


def extract_recipe(text: str) -> Recipe:
    """Extract a recipe from text using XML as intermediate format."""
    client = anthropic.Anthropic()

    xml_template = """<recipe>
  <name>...</name>
  <servings>4</servings>
  <prep_time_minutes>15</prep_time_minutes>
  <cook_time_minutes>30</cook_time_minutes>
  <difficulty>easy|medium|hard</difficulty>
  <ingredients>
    <ingredient quantity="2.0" unit="cups" notes="optional note">flour</ingredient>
  </ingredients>
  <steps>
    <step number="1" duration_minutes="5">Instruction text</step>
  </steps>
  <tags>
    <tag>vegetarian</tag>
  </tags>
</recipe>"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Extract the recipe from this text and format it as XML "
                    f"matching this template:\n\n{xml_template}\n\n"
                    f"Text: {text}\n\n"
                    f"Return ONLY the XML."
                )
            }
        ]
    )

    raw = response.content[0].text
    xml_match = re.search(r"<recipe>.*?</recipe>", raw, re.DOTALL)
    xml_str = xml_match.group() if xml_match else raw.strip()

    root = ET.fromstring(xml_str)

    # Parse ingredients
    ingredients = []
    for ing_el in root.findall(".//ingredient"):
        ingredients.append(Ingredient(
            name=ing_el.text.strip(),
            quantity=float(ing_el.get("quantity", "1")),
            unit=ing_el.get("unit", "piece"),
            notes=ing_el.get("notes")
        ))

    # Parse steps
    steps = []
    for step_el in root.findall(".//step"):
        dur = step_el.get("duration_minutes")
        steps.append(Step(
            number=int(step_el.get("number", len(steps) + 1)),
            instruction=step_el.text.strip(),
            duration_minutes=int(dur) if dur else None
        ))

    # Parse tags
    tags = [t.text.strip() for t in root.findall(".//tag") if t.text]

    return Recipe(
        name=root.findtext("name", "").strip(),
        servings=int(root.findtext("servings", "4")),
        prep_time_minutes=int(root.findtext("prep_time_minutes", "0")),
        cook_time_minutes=int(root.findtext("cook_time_minutes", "0")),
        difficulty=root.findtext("difficulty", "medium").strip(),
        ingredients=ingredients,
        steps=steps,
        tags=tags
    )


# Test
recipe = extract_recipe(
    "Quick Pasta Aglio e Olio (serves 2): Boil 200g spaghetti for 8 minutes. "
    "While that cooks, slice 4 cloves of garlic thinly. Heat 3 tablespoons "
    "olive oil in a pan, add garlic and a pinch of red pepper flakes, cook "
    "for 2 minutes until golden. Toss the drained pasta with the garlic oil. "
    "Season with salt, add chopped parsley. Total time: 15 minutes. Easy."
)
print(recipe.model_dump_json(indent=2))
```

</details>

### Exercise 5: Comparative Format Benchmark

Write a script that benchmarks the reliability of different structured output techniques. Send the same extraction task using: (a) prompt-only JSON, (b) prefill-based JSON, (c) tool calling. Run each N times and measure: parse success rate, schema compliance rate, and average latency.

**Requirements:**
- Same extraction task across all three methods
- At least 3 runs per method (use 10 for real benchmarks)
- Measure and report parse rate, validation rate, and latency
- Print a comparison table

<details><summary>Show Answer</summary>

```python
import anthropic
import json
import time
from pydantic import BaseModel, ValidationError
from typing import Optional
from dataclasses import dataclass, field


class EventInfo(BaseModel):
    event_name: str
    date: str
    location: str
    organizer: str
    attendee_count: Optional[int] = None
    topics: list[str]


@dataclass
class BenchmarkResult:
    method: str
    runs: int = 0
    parse_successes: int = 0
    validation_successes: int = 0
    latencies: list[float] = field(default_factory=list)

    @property
    def parse_rate(self) -> float:
        return self.parse_successes / self.runs if self.runs else 0

    @property
    def validation_rate(self) -> float:
        return self.validation_successes / self.runs if self.runs else 0

    @property
    def avg_latency(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0


TEST_TEXT = (
    "The annual AI Summit 2025 will be held on June 15, 2025 at the "
    "San Francisco Convention Center. Organized by TechEvents Inc., "
    "the conference expects around 5000 attendees. Topics include "
    "large language models, computer vision, and AI safety."
)


def method_prompt_only(client: anthropic.Anthropic) -> str:
    """Method A: prompt-only JSON."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": (
                "Extract event info as JSON with keys: event_name, date, "
                "location, organizer, attendee_count, topics. "
                f"Return ONLY JSON.\n\nText: {TEST_TEXT}"
            )
        }]
    )
    return response.content[0].text


def method_prefill(client: anthropic.Anthropic) -> str:
    """Method B: prefill-based JSON."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": (
                    "Extract event info as JSON with keys: event_name, date, "
                    "location, organizer, attendee_count, topics. "
                    f"Return ONLY JSON.\n\nText: {TEST_TEXT}"
                )
            },
            {"role": "assistant", "content": "{"}
        ]
    )
    return "{" + response.content[0].text


def method_tool_calling(client: anthropic.Anthropic) -> str:
    """Method C: tool calling."""
    tools = [{
        "name": "extract_event",
        "description": "Extract event information",
        "input_schema": EventInfo.model_json_schema()
    }]
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        tools=tools,
        tool_choice={"type": "tool", "name": "extract_event"},
        messages=[{
            "role": "user",
            "content": f"Extract event info:\n\n{TEST_TEXT}"
        }]
    )
    for block in response.content:
        if block.type == "tool_use":
            return json.dumps(block.input)
    return ""


def run_benchmark(n_runs: int = 3) -> None:
    """Run the comparative benchmark."""
    client = anthropic.Anthropic()

    methods = {
        "prompt_only": method_prompt_only,
        "prefill": method_prefill,
        "tool_calling": method_tool_calling,
    }

    results = {name: BenchmarkResult(method=name) for name in methods}

    for name, method_fn in methods.items():
        result = results[name]
        for i in range(n_runs):
            result.runs += 1
            start = time.time()
            try:
                raw = method_fn(client)
                elapsed = time.time() - start
                result.latencies.append(elapsed)

                # Try parsing JSON
                import re
                clean = re.sub(r"^```(?:json)?\s*\n?", "", raw)
                clean = re.sub(r"\n?```\s*$", "", clean).strip()
                data = json.loads(clean)
                result.parse_successes += 1

                # Try Pydantic validation
                EventInfo.model_validate(data)
                result.validation_successes += 1

            except json.JSONDecodeError:
                elapsed = time.time() - start
                result.latencies.append(elapsed)
            except ValidationError:
                pass  # Already counted parse success
            except Exception as e:
                elapsed = time.time() - start
                result.latencies.append(elapsed)
                print(f"  {name} run {i+1} error: {e}")

    # Print results table
    print(f"\n{'Method':<15} {'Parse Rate':>12} {'Valid Rate':>12} {'Avg Latency':>12}")
    print("-" * 55)
    for name, r in results.items():
        print(
            f"{r.method:<15} "
            f"{r.parse_rate:>11.0%} "
            f"{r.validation_rate:>11.0%} "
            f"{r.avg_latency:>10.2f}s"
        )


if __name__ == "__main__":
    run_benchmark(n_runs=3)
```

</details>

---

**Previous**: [Advanced Reasoning Prompts](./04_Advanced_Reasoning_Prompts.md) | **Next**: [System Prompt Design](./06_System_Prompt_Design.md)
