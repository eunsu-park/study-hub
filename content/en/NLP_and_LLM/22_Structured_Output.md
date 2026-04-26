# 22. Structured Output from LLMs

## Learning Objectives

- Use JSON mode and response format controls for structured generation
- Implement function calling for reliable structured extraction
- Build Pydantic-based output parsing with validation and retry logic
- Leverage the instructor library for type-safe LLM outputs
- Design complex nested schemas and production data extraction pipelines

---

## Theory & Principles

A free-text LLM response works for chat. It fails for any downstream system that wants to **parse** the output: extract entities into a database, populate a form, trigger an action with typed parameters. "Output JSON" works most of the time but occasionally produces invalid JSON, missing fields, or unexpected types — and "occasionally" at scale means thousands of failures per day. Structured output techniques close this gap by either constraining the model's generation at the token level, validating after generation with retry, or both.

This section covers:

- **(A) The constrained generation problem** — what guarantees do you actually need, and at what cost.
- **(B) Prompt-level structuring** — JSON mode, system-prompt instructions, the limits of asking nicely.
- **(C) Function calling as structured output** — how OpenAI/Anthropic's tool-calling APIs guarantee structure.
- **(D) Grammar-based constrained decoding** — Outlines, LMQL, JSON-schema-aware token masking.
- **(E) Pydantic parse-and-retry** — validation as a separate stage, with retry-on-failure.
- **(F) The instructor library** — type-safe Python interface, automatic retry, partial outputs for streaming.
- **(G) Designing schemas** — nesting, optionality, enums, the trade-off between precision and model success rate.

### A. The Constrained Generation Problem

You want the LLM to produce a string `s` such that `parse(s)` succeeds and the parsed result satisfies some schema. Three guarantee levels:

- **Level 0 (prompt only)**: ask the model nicely. ~95% success on simple schemas, drops with complexity.
- **Level 1 (validate + retry)**: parse the output; if it fails, re-prompt with the error and try again. ~99% success after 1-2 retries.
- **Level 2 (constrained decoding)**: restrict the model's token choices at each step to those that maintain a valid prefix of the schema. 100% success guaranteed (when supported).

Each level adds capability and cost. Production systems pick the lowest level that gives acceptable failure rate, weighing latency and cost against retry frequency.

### B. Prompt-Level Structuring

The simplest approach: tell the model what you want.

```
Output JSON with keys "name" (string), "age" (integer), "skills" (array of strings).
Output ONLY the JSON. No explanation.
```

OpenAI's "JSON mode" (`response_format = {"type": "json_object"}`) constrains the output to be parseable JSON but does not enforce a specific schema. Field names, types, and structure are still up to the model — and the model can still hallucinate fields, omit required ones, or swap types.

Use prompt-level for prototyping or low-stakes systems. Pair with validation (E) for production.

### C. Function Calling as Structured Output

Although designed for tool calling (lesson 23), the function-calling API is also the cleanest way to get structured output. Define a "tool" whose parameters are the schema you want; ask the model to call the tool with the extracted data:

```
tool = {
  "name": "save_person",
  "parameters": {
    "type": "object",
    "properties": {
      "name": {"type": "string"},
      "age": {"type": "integer"},
      "skills": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["name", "age"]
  }
}
# Model returns: {"tool_calls": [{"name": "save_person", "arguments": {"name": "...", ...}}]}
```

The arguments are guaranteed to be valid JSON of the declared types (the API enforces). You don't need to actually execute a tool — you're just using the API's schema enforcement.

OpenAI's "Strict Mode" function calling and Anthropic's tool use both implement this with token-level constraint at the API level: invalid tokens are simply not sampled. ~100% success rate.

### D. Grammar-Based Constrained Decoding

For open-source models without provider-side enforcement, libraries like **Outlines** (Willard & Louf, 2023) and **LMQL** implement constrained decoding directly.

**The mechanism.** A JSON schema (or regex) is compiled into a finite-state automaton (FSA). At each generation step, the model produces logits for all `V` tokens; the constraint masks out tokens that would not advance any valid path through the FSA, and the model samples from the remainder. This guarantees the output matches the schema by construction.

**Cost.** A small per-step computation to update FSA state; usually negligible. Some constraints (especially complex JSON schemas) can be slow to compile but the result is reusable across queries.

This is the only approach that gives **categorical** guarantees on local models without API support.

### E. Pydantic Parse-and-Retry

The most common production pattern:

```python
class Person(BaseModel):
    name: str
    age: int
    skills: list[str]

def extract(text: str, max_retries=3):
    for _ in range(max_retries):
        response = llm(prompt + text)
        try:
            return Person.model_validate_json(response)
        except ValidationError as e:
            prompt = f"{prompt}\n\nPrevious attempt failed validation: {e}\nPlease fix and retry."
    raise ValueError("Max retries exceeded")
```

The validator catches both JSON parse errors and type/constraint violations. Re-prompting with the error usually produces a valid output on the next attempt. Combines prompt-level (B) with validation as a backstop.

This gives ~99%+ success at modest cost (one retry per ~20 calls on average for simple schemas).

### F. The Instructor Library

`instructor` (Liu, 2023) wraps the OpenAI/Anthropic SDKs to make Pydantic-based extraction the primary interface:

```python
import instructor
from openai import OpenAI

client = instructor.from_openai(OpenAI())
person = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": text}],
    response_model=Person,
)
# `person` is a typed Person instance — no JSON parsing, no validation in user code
```

Behind the scenes: instructor converts the Pydantic model to a function-calling schema (C), invokes the API, validates the result, and retries on failure (E). It also supports partial validation for streaming (yield typed chunks as they arrive) and `Iterable[Person]` for extracting lists.

This is the production pattern: type safety from the user's perspective, robust generation under the hood.

### G. Designing Schemas

Schema design directly affects the model's success rate.

**G.1 Optional fields.** Mark fields not always present as `Optional[T]`. Forces the model to consider whether the data exists; reduces hallucination of fake values.

**G.2 Enums.** Constrain string fields to a closed set: `Literal["pending", "approved", "rejected"]`. Eliminates typos and invented categories.

**G.3 Examples in field descriptions.** Pydantic's `Field(description="...")` is included in the JSON schema. Examples help the model understand what to extract: `Field(description="The person's full legal name, e.g., 'John Smith'")`.

**G.4 Nesting depth.** Each level of nesting increases failure rate (more places for the model to confuse itself). Flatten when possible: prefer `customer_name` over `customer.name`.

**G.5 Numeric constraints.** Use `Field(ge=0, le=120)` for `age`, `Field(min_length=1, max_length=100)` for strings. The model usually respects these; the validator catches the rest.

The general principle: **encode invariants in the schema, not in the prompt.** Schema constraints are checkable; prompt instructions are advisory.

### From Theory to the Functions Below

- §1 (the challenge) — frames §A's three guarantee levels.
- §2 (JSON mode) — implements §B prompt-level structuring with OpenAI/Anthropic JSON modes.
- §3 (function calling) — implements §C function-calling as structured output.
- §4 (Pydantic) — implements §E parse-and-retry pattern.
- §5 (instructor library) — uses §F's wrapper for type-safe extraction.
- §6 (OpenAI Structured Outputs) — provider-native §C/§D fusion (strict-mode function calling).
- §7 (production pipeline) — combines §A-§F into a realistic data-extraction pipeline with §G schema design.

---

## 1. The Structured Output Challenge

### Why Structured Output Matters

> **Structured Output Use Cases**
>
> - **Data Extraction**: Pull structured records from unstructured documents
> - **API Integration**: Generate valid payloads for downstream services
> - **Database Ingestion**: Transform free text into relational or document records
> - **Workflow Automation**: Parse LLM decisions into executable action objects
> - **Analytics Pipelines**: Convert reports into machine-readable formats

### Approach Comparison

| Approach | Reliability | Flexibility | Complexity | Best For |
|----------|-------------|-------------|------------|----------|
| Regex Parsing | Low | Low | Low | Simple patterns |
| JSON Mode | Medium | Medium | Low | Basic JSON objects |
| Function Calling | High | High | Medium | Tool integration |
| Pydantic + Instructor | Very High | Very High | Medium | Production systems |
| OpenAI Structured Outputs | Very High | High | Low | OpenAI-specific apps |
| Grammar-Constrained | Highest | Medium | High | Local models (llama.cpp) |

---

## 2. JSON Mode

### OpenAI JSON Mode

```python
from openai import OpenAI
import json

client = OpenAI()

def extract_with_json_mode(text: str) -> dict:
    """Extract structured data using JSON mode."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": (
                "Extract information from the text and return a JSON object with:\n"
                '- "entities": list of named entities with "name", "type", "context"\n'
                '- "sentiment": one of "positive", "negative", "neutral"\n'
                '- "topics": list of main topics discussed\n'
                '- "summary": one-sentence summary'
            )},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    return json.loads(response.choices[0].message.content)

# Usage
text = """
Apple announced the new M4 chip today at their Cupertino headquarters.
CEO Tim Cook demonstrated significant performance improvements over the M3,
with 50% faster CPU and 2x GPU performance. The stock rose 3% in after-hours trading.
Analysts from Goldman Sachs and Morgan Stanley issued positive ratings.
"""

result = extract_with_json_mode(text)
print(json.dumps(result, indent=2))
# {
#   "entities": [
#     {"name": "Apple", "type": "organization", "context": "product announcement"},
#     {"name": "M4", "type": "product", "context": "new chip"},
#     {"name": "Tim Cook", "type": "person", "context": "CEO, presented demo"},
#     ...
#   ],
#   "sentiment": "positive",
#   "topics": ["technology", "semiconductors", "stock market"],
#   "summary": "Apple unveiled the M4 chip with major performance gains, boosting stock."
# }
```

### Anthropic JSON Mode

```python
from anthropic import Anthropic

anthropic = Anthropic()

def extract_with_claude(text: str) -> dict:
    """Extract structured data using Claude."""
    response = anthropic.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[
            {"role": "user", "content": (
                f"Extract entities, sentiment, and topics from this text. "
                f"Return ONLY a valid JSON object, no other text.\n\n{text}"
            )},
        ],
    )
    # Claude doesn't have a dedicated JSON mode, so we parse the response
    content = response.content[0].text
    # Strip markdown code fences if present
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(content)
```

### JSON Mode Pitfalls

```python
# Problem 1: JSON mode doesn't guarantee schema conformance
# The model might return valid JSON that doesn't match your expected schema

def safe_json_extract(text: str, required_keys: list[str]) -> dict | None:
    """Extract JSON with schema validation."""
    result = extract_with_json_mode(text)

    # Validate required keys
    missing_keys = [k for k in required_keys if k not in result]
    if missing_keys:
        print(f"Warning: Missing keys: {missing_keys}")
        return None

    return result

# Problem 2: Inconsistent types
# The model might return "3" (string) instead of 3 (int)
# Always validate and coerce types

def coerce_types(data: dict, schema: dict) -> dict:
    """Coerce JSON values to expected types."""
    coerced = {}
    for key, expected_type in schema.items():
        if key in data:
            try:
                coerced[key] = expected_type(data[key])
            except (ValueError, TypeError):
                coerced[key] = data[key]  # Keep original if coercion fails
    return coerced

# Usage
schema = {"price": float, "quantity": int, "name": str}
raw = {"price": "29.99", "quantity": "5", "name": "Widget"}
clean = coerce_types(raw, schema)
# {"price": 29.99, "quantity": 5, "name": "Widget"}
```

---

## 3. Function Calling for Structured Extraction

### Schema-Driven Extraction

```python
def extract_with_function_calling(text: str) -> dict:
    """Use function calling to enforce output structure."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "extract_document_info",
                "description": "Extract structured information from a document",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Document title or headline",
                        },
                        "entities": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "type": {
                                        "type": "string",
                                        "enum": ["person", "organization",
                                                 "location", "product", "event"],
                                    },
                                    "role": {"type": "string"},
                                },
                                "required": ["name", "type"],
                            },
                        },
                        "key_metrics": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "metric": {"type": "string"},
                                    "value": {"type": "string"},
                                    "unit": {"type": "string"},
                                    "change_direction": {
                                        "type": "string",
                                        "enum": ["increase", "decrease", "stable", "unknown"],
                                    },
                                },
                                "required": ["metric", "value"],
                            },
                        },
                        "sentiment": {
                            "type": "string",
                            "enum": ["very_positive", "positive", "neutral",
                                     "negative", "very_negative"],
                        },
                        "date_mentioned": {
                            "type": "string",
                            "description": "ISO 8601 date if mentioned",
                        },
                    },
                    "required": ["title", "entities", "sentiment"],
                },
            },
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Extract structured data from the given text."},
            {"role": "user", "content": text},
        ],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "extract_document_info"}},
        temperature=0.0,
    )

    # Extract the function call arguments
    tool_call = response.choices[0].message.tool_calls[0]
    return json.loads(tool_call.function.arguments)

# Usage
result = extract_with_function_calling(text)
print(json.dumps(result, indent=2))
```

### Multiple Extraction Functions

```python
def multi_schema_extract(text: str) -> dict:
    """Let the model choose the appropriate extraction schema."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "extract_financial_data",
                "description": "Extract financial metrics, stock data, and market info",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "company": {"type": "string"},
                        "stock_price_change": {"type": "number"},
                        "revenue": {"type": "number"},
                        "currency": {"type": "string", "default": "USD"},
                        "analyst_ratings": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "firm": {"type": "string"},
                                    "rating": {"type": "string"},
                                },
                            },
                        },
                    },
                    "required": ["company"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "extract_technical_specs",
                "description": "Extract technical specifications and benchmarks",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product": {"type": "string"},
                        "manufacturer": {"type": "string"},
                        "specs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "value": {"type": "string"},
                                    "improvement": {"type": "string"},
                                },
                            },
                        },
                    },
                    "required": ["product", "specs"],
                },
            },
        },
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": f"Extract relevant data:\n\n{text}"},
        ],
        tools=tools,
        tool_choice="auto",  # Model picks the best function
        temperature=0.0,
    )

    results = {}
    for tool_call in response.choices[0].message.tool_calls:
        func_name = tool_call.function.name
        results[func_name] = json.loads(tool_call.function.arguments)

    return results
```

---

## 4. Pydantic Output Parsing

### Basic Pydantic Models

```python
from pydantic import BaseModel, Field, field_validator
from typing import Literal
from datetime import date

class Entity(BaseModel):
    name: str = Field(description="Entity name")
    entity_type: Literal["person", "organization", "location", "product"] = Field(
        description="Category of the entity"
    )
    relevance: float = Field(ge=0.0, le=1.0, description="Relevance score 0-1")

class DocumentExtraction(BaseModel):
    title: str = Field(description="Main title or headline")
    summary: str = Field(max_length=500, description="Brief summary")
    entities: list[Entity] = Field(min_length=1, description="Extracted entities")
    topics: list[str] = Field(min_length=1, max_length=10)
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    published_date: date | None = Field(default=None)

    @field_validator("topics")
    @classmethod
    def topics_lowercase(cls, v: list[str]) -> list[str]:
        return [t.lower().strip() for t in v]

    @field_validator("summary")
    @classmethod
    def summary_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Summary cannot be empty")
        return v.strip()
```

### LangChain Output Parsers

```python
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Create parser from Pydantic model
parser = PydanticOutputParser(pydantic_object=DocumentExtraction)

# Build prompt with format instructions injected
prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Extract structured information from the text.\n\n"
        "{format_instructions}"
    )),
    ("user", "{text}"),
])

# Chain: prompt -> LLM -> parser
llm = ChatOpenAI(model="gpt-4o", temperature=0)
chain = prompt | llm | parser

# Run extraction
result: DocumentExtraction = chain.invoke({
    "text": text,
    "format_instructions": parser.get_format_instructions(),
})

print(f"Title: {result.title}")
print(f"Entities: {[e.name for e in result.entities]}")
print(f"Sentiment: {result.sentiment}")
```

### Retry with Error Feedback

```python
from langchain.output_parsers import RetryWithErrorOutputParser
from langchain_core.runnables import RunnablePassthrough

# Wrap parser with retry logic
retry_parser = RetryWithErrorOutputParser.from_llm(
    parser=parser,
    llm=llm,
    max_retries=3,
)

def extract_with_retry(text: str) -> DocumentExtraction | None:
    """Extract with automatic retry on parsing failure."""
    prompt_value = prompt.invoke({
        "text": text,
        "format_instructions": parser.get_format_instructions(),
    })

    # First attempt
    response = llm.invoke(prompt_value)

    try:
        return parser.parse(response.content)
    except Exception as e:
        print(f"First parse failed: {e}")
        # Retry with error context fed back to the model
        try:
            return retry_parser.parse_with_prompt(
                response.content,
                prompt_value,
            )
        except Exception as e2:
            print(f"Retry failed: {e2}")
            return None
```

---

## 5. Instructor Library

### Type-Safe LLM Outputs

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Literal

# Patch OpenAI client with instructor
client = instructor.from_openai(OpenAI())

class UserProfile(BaseModel):
    name: str
    age: int = Field(ge=0, le=150)
    email: str
    interests: list[str] = Field(min_length=1)
    experience_level: Literal["beginner", "intermediate", "advanced"]

# instructor handles parsing, validation, and retries automatically
profile = client.chat.completions.create(
    model="gpt-4o",
    response_model=UserProfile,
    messages=[
        {"role": "user", "content": (
            "Extract user info: John is 28, works at john@techcorp.com. "
            "He's into machine learning, Python, and distributed systems. "
            "He's been coding for 7 years."
        )},
    ],
)

print(profile)
# UserProfile(name='John', age=28, email='john@techcorp.com',
#   interests=['machine learning', 'Python', 'distributed systems'],
#   experience_level='advanced')

# Access fields with type safety
print(f"Name: {profile.name}, Age: {profile.age}")
```

### Complex Nested Structures

```python
from pydantic import BaseModel, Field
from typing import Literal
from enum import Enum

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ActionItem(BaseModel):
    description: str
    assignee: str | None = None
    priority: Priority
    due_date: str | None = None

class Decision(BaseModel):
    topic: str
    outcome: str
    rationale: str
    dissenting_views: list[str] = Field(default_factory=list)

class MeetingMinutes(BaseModel):
    """Structured meeting minutes extraction."""
    title: str
    date: str
    attendees: list[str] = Field(min_length=1)
    agenda_items: list[str]
    decisions: list[Decision]
    action_items: list[ActionItem]
    next_meeting: str | None = None
    key_discussion_points: list[str]

# Extract structured meeting minutes
minutes = client.chat.completions.create(
    model="gpt-4o",
    response_model=MeetingMinutes,
    messages=[
        {"role": "user", "content": """
        Meeting: Q1 Engineering Review - March 10, 2026
        Attendees: Sarah Chen, Mike Park, Lisa Wang, Tom Garcia

        Sarah opened by reviewing sprint velocity. Team delivered 85% of planned stories.
        Discussion on migrating to Kubernetes - Mike raised concerns about complexity.
        Decision: Proceed with K8s migration in Q2, starting with staging.
        Lisa will own the migration plan, due March 25.
        Tom to evaluate monitoring tools (Datadog vs Grafana) by March 18.

        Budget discussion: agreed to allocate $50K for cloud infrastructure upgrade.
        Mike dissented, preferring to optimize existing setup first.

        Next meeting: March 24, 2026.
        """},
    ],
)

for item in minutes.action_items:
    print(f"[{item.priority.value}] {item.description} -> {item.assignee}")
```

### Instructor with Anthropic

```python
import instructor
from anthropic import Anthropic

# Patch Anthropic client
anthropic_client = instructor.from_anthropic(Anthropic())

result = anthropic_client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=2048,
    response_model=MeetingMinutes,
    messages=[
        {"role": "user", "content": "Extract meeting minutes from: ..."},
    ],
)
```

### Streaming with Partial Objects

```python
from instructor import Partial

# Stream partial results as they arrive
for partial_profile in client.chat.completions.create_partial(
    model="gpt-4o",
    response_model=UserProfile,
    messages=[
        {"role": "user", "content": "Extract: John, 28, john@tech.com, loves ML and Python, expert coder"},
    ],
    stream=True,
):
    # partial_profile has fields populated incrementally
    print(f"Partial: {partial_profile.model_dump()}")
    # First iteration: {"name": "John", "age": None, ...}
    # Next: {"name": "John", "age": 28, "email": None, ...}
    # Final: all fields populated
```

### Validation and Retry Strategies

```python
from pydantic import BaseModel, Field, model_validator
from tenacity import retry, stop_after_attempt, wait_exponential
import instructor

client = instructor.from_openai(OpenAI())

class InvoiceItem(BaseModel):
    description: str
    quantity: int = Field(ge=1)
    unit_price: float = Field(ge=0)
    total: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_total(self):
        expected = round(self.quantity * self.unit_price, 2)
        if abs(self.total - expected) > 0.01:
            raise ValueError(
                f"Total {self.total} doesn't match quantity * unit_price = {expected}"
            )
        return self

class Invoice(BaseModel):
    invoice_number: str
    vendor: str
    date: str
    items: list[InvoiceItem] = Field(min_length=1)
    subtotal: float = Field(ge=0)
    tax: float = Field(ge=0)
    total: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_totals(self):
        items_sum = round(sum(item.total for item in self.items), 2)
        if abs(self.subtotal - items_sum) > 0.01:
            raise ValueError(
                f"Subtotal {self.subtotal} doesn't match sum of items {items_sum}"
            )
        expected_total = round(self.subtotal + self.tax, 2)
        if abs(self.total - expected_total) > 0.01:
            raise ValueError(
                f"Total {self.total} doesn't match subtotal + tax = {expected_total}"
            )
        return self

# instructor automatically retries when validation fails,
# feeding the error message back to the LLM
invoice = client.chat.completions.create(
    model="gpt-4o",
    response_model=Invoice,
    max_retries=3,  # Retry up to 3 times on validation failure
    messages=[
        {"role": "user", "content": """
        Invoice #INV-2026-0042
        Vendor: Acme Cloud Services
        Date: 2026-03-10

        Items:
        - Compute instances (10 units @ $45.00 each)
        - Storage 1TB (2 units @ $12.50 each)
        - Load balancer (1 unit @ $30.00)

        Tax: 8.5%
        """},
    ],
)

print(f"Invoice: {invoice.invoice_number}")
print(f"Total: ${invoice.total:.2f}")
for item in invoice.items:
    print(f"  {item.description}: {item.quantity} x ${item.unit_price} = ${item.total}")
```

---

## 6. OpenAI Structured Outputs

### Strict Mode

```python
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI()

class Step(BaseModel):
    explanation: str
    output: str

class MathSolution(BaseModel):
    steps: list[Step]
    final_answer: str

# Strict structured outputs — guaranteed schema conformance
response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "Solve the math problem step by step."},
        {"role": "user", "content": "Solve: 2x + 5 = 17"},
    ],
    response_format=MathSolution,
)

solution = response.choices[0].message.parsed
for i, step in enumerate(solution.steps, 1):
    print(f"Step {i}: {step.explanation} -> {step.output}")
print(f"Answer: {solution.final_answer}")
```

### Schema Design Best Practices

```python
from pydantic import BaseModel, Field
from typing import Literal

# GOOD: Specific types with clear descriptions and constraints
class GoodSchema(BaseModel):
    """Well-designed schema for LLM extraction."""
    category: Literal["bug", "feature", "improvement", "docs"] = Field(
        description="Type of the issue"
    )
    severity: Literal["low", "medium", "high", "critical"] = Field(
        description="Impact severity"
    )
    title: str = Field(
        max_length=100,
        description="Short descriptive title"
    )
    affected_component: str = Field(
        description="Which system component is affected"
    )
    steps_to_reproduce: list[str] = Field(
        default_factory=list,
        description="Ordered steps to reproduce (for bugs)"
    )

# BAD: Vague types, no descriptions, no constraints
class BadSchema(BaseModel):
    type: str        # Too vague — model might return anything
    level: int       # What range? 1-5? 1-10?
    info: str        # Ambiguous field name
    data: dict       # Completely unstructured
    tags: list       # List of what?

# TIP: Use Literal for categorical fields to constrain outputs
# TIP: Add Field descriptions — they become part of the prompt
# TIP: Use default_factory for optional lists
# TIP: Keep nesting depth <= 3 levels for reliable extraction
```

---

## 7. Production Data Extraction Pipeline

### End-to-End Pipeline

```python
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar, Type
from pydantic import BaseModel, ValidationError
import instructor
from openai import OpenAI

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

@dataclass
class ExtractionResult:
    success: bool
    data: BaseModel | None
    raw_response: str | None
    errors: list[str]
    retries: int
    model: str
    tokens_used: int

class ExtractionPipeline:
    """Production-grade data extraction pipeline."""

    def __init__(self, primary_model: str = "gpt-4o",
                 fallback_model: str = "gpt-4o-mini",
                 max_retries: int = 3):
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.max_retries = max_retries
        self.client = instructor.from_openai(OpenAI())

    def extract(self, text: str, schema: Type[T],
                instructions: str = "") -> ExtractionResult:
        """Extract structured data with fallback and error handling."""
        errors = []
        total_tokens = 0

        # Attempt 1: Primary model
        for attempt in range(self.max_retries):
            try:
                result = self.client.chat.completions.create(
                    model=self.primary_model,
                    response_model=schema,
                    max_retries=1,  # Let our outer loop handle retries
                    messages=self._build_messages(text, instructions, schema, errors),
                )
                return ExtractionResult(
                    success=True,
                    data=result,
                    raw_response=None,
                    errors=errors,
                    retries=attempt,
                    model=self.primary_model,
                    tokens_used=total_tokens,
                )
            except ValidationError as e:
                error_msg = str(e)
                errors.append(f"Attempt {attempt+1}: {error_msg}")
                logger.warning(f"Validation failed (attempt {attempt+1}): {error_msg}")
            except Exception as e:
                errors.append(f"Attempt {attempt+1}: {str(e)}")
                logger.error(f"Extraction failed (attempt {attempt+1}): {e}")

        # Attempt 2: Fallback model
        logger.info(f"Falling back to {self.fallback_model}")
        try:
            result = self.client.chat.completions.create(
                model=self.fallback_model,
                response_model=schema,
                max_retries=2,
                messages=self._build_messages(text, instructions, schema, errors),
            )
            return ExtractionResult(
                success=True,
                data=result,
                raw_response=None,
                errors=errors,
                retries=self.max_retries + 1,
                model=self.fallback_model,
                tokens_used=total_tokens,
            )
        except Exception as e:
            errors.append(f"Fallback failed: {str(e)}")
            return ExtractionResult(
                success=False,
                data=None,
                raw_response=None,
                errors=errors,
                retries=self.max_retries + 1,
                model=self.fallback_model,
                tokens_used=total_tokens,
            )

    def _build_messages(self, text: str, instructions: str,
                        schema: Type[T], previous_errors: list[str]) -> list[dict]:
        """Build messages with optional error context for retries."""
        system_content = (
            f"Extract structured data from the given text.\n"
            f"{instructions}\n"
            f"Be precise and follow the schema exactly."
        )

        if previous_errors:
            system_content += (
                f"\n\nPrevious attempts had these errors:\n"
                + "\n".join(f"- {e}" for e in previous_errors[-2:])
                + "\nPlease fix these issues in your response."
            )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": text},
        ]

    def batch_extract(self, texts: list[str], schema: Type[T],
                      instructions: str = "") -> list[ExtractionResult]:
        """Extract from multiple texts."""
        results = []
        for i, text in enumerate(texts):
            logger.info(f"Processing {i+1}/{len(texts)}")
            result = self.extract(text, schema, instructions)
            results.append(result)
        return results

# Usage
class ProductReview(BaseModel):
    product_name: str
    rating: float = Field(ge=1.0, le=5.0)
    pros: list[str] = Field(min_length=1)
    cons: list[str] = Field(default_factory=list)
    recommendation: Literal["strongly_recommend", "recommend",
                            "neutral", "not_recommend"]
    reviewer_experience: Literal["beginner", "intermediate", "expert"]

pipeline = ExtractionPipeline()

review_text = """
I've been using the Sony WH-1000XM5 headphones for 3 months now as a professional
audio engineer. The noise cancellation is best-in-class, and the sound quality
is excellent with rich bass and clear highs. Battery life is amazing at 30+ hours.
However, they don't fold flat like the XM4, and the carrying case is bulky.
The touch controls can be finicky in cold weather. Despite these minor issues,
these are the best wireless headphones I've used. Highly recommended.
"""

result = pipeline.extract(
    review_text,
    ProductReview,
    instructions="Extract a detailed product review analysis.",
)

if result.success:
    review = result.data
    print(f"Product: {review.product_name}")
    print(f"Rating: {review.rating}/5")
    print(f"Pros: {review.pros}")
    print(f"Cons: {review.cons}")
    print(f"Recommendation: {review.recommendation}")
    print(f"Retries needed: {result.retries}")
else:
    print(f"Extraction failed: {result.errors}")
```

### Pipeline Monitoring

```python
from collections import defaultdict
import time

class PipelineMetrics:
    """Track extraction pipeline performance."""

    def __init__(self):
        self.total_extractions = 0
        self.successful = 0
        self.failed = 0
        self.retry_counts = defaultdict(int)
        self.model_usage = defaultdict(int)
        self.latencies: list[float] = []
        self.schema_errors: list[str] = []

    def record(self, result: ExtractionResult, latency: float):
        self.total_extractions += 1
        self.latencies.append(latency)
        self.model_usage[result.model] += 1
        self.retry_counts[result.retries] += 1

        if result.success:
            self.successful += 1
        else:
            self.failed += 1
            self.schema_errors.extend(result.errors)

    def summary(self) -> dict:
        return {
            "total": self.total_extractions,
            "success_rate": self.successful / max(self.total_extractions, 1),
            "avg_latency_ms": (
                sum(self.latencies) / len(self.latencies) * 1000
                if self.latencies else 0
            ),
            "p99_latency_ms": (
                sorted(self.latencies)[int(len(self.latencies) * 0.99)] * 1000
                if self.latencies else 0
            ),
            "model_usage": dict(self.model_usage),
            "retry_distribution": dict(self.retry_counts),
            "recent_errors": self.schema_errors[-5:],
        }

# Integration
metrics = PipelineMetrics()

start = time.time()
result = pipeline.extract(review_text, ProductReview)
latency = time.time() - start

metrics.record(result, latency)
print(json.dumps(metrics.summary(), indent=2))
```

---

## Next Steps

In [23_Function_Calling_Tools.md](./23_Function_Calling_Tools.md), we dive deep into function calling and tool use APIs, including the Model Context Protocol (MCP) and advanced tool orchestration patterns.
