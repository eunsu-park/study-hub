# 05. 구조화된 출력 프롬프팅(Structured Output Prompting)

**이전**: [고급 추론 프롬프트](./04_Advanced_Reasoning_Prompts.md) | **다음**: [시스템 프롬프트 설계](./06_System_Prompt_Design.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. JSON, XML, YAML 및 기타 형식으로 구조화된 출력(Structured Output)을 안정적으로 생성하는 프롬프트 설계
2. 도구/함수 호출(Tool/Function Calling)과 Pydantic 검증을 사용한 스키마 제약 생성(Schema-Constrained Generation) 구현
3. 출력 적합성을 보장하는 문법 기반 디코딩(Grammar-Based Decoding) 기법 적용
4. LLM 출력에서 중첩, 재귀, 복잡한 데이터 구조 처리
5. 잘못된 형식의 구조화된 출력을 위한 견고한 오류 복구 파이프라인 구축

---

비구조화된 자연어는 사람이 읽기 쉽지만 프로그램이 소비하기 어렵습니다. LLM이 백엔드 서비스를 구동할 때, 출력은 다운스트림 코드가 추측 없이 파싱할 수 있는 정확한 스키마를 따라야 합니다. JSON에서 쉼표 하나가 빠지거나 예상치 못한 필드명이 있으면 파이프라인이 중단될 수 있습니다. 구조화된 출력 프롬프팅(Structured Output Prompting)은 기계가 파싱 가능한 응답을 보장하는 프롬프트와 시스템 구성을 설계하는 분야로, LLM을 대화 파트너에서 신뢰할 수 있는 데이터 생성 엔진으로 변환합니다.

이 레슨은 구조화된 출력 기법의 전체 스펙트럼을 다룹니다: 간단한 프롬프트 기반 포맷팅부터 잘못된 출력을 수학적으로 불가능하게 만드는 스키마 제약 디코딩까지.

## 목차

1. [구조화된 출력이 중요한 이유](#1-why-structured-output-matters)
2. [JSON 출력 프롬프팅](#2-json-output-prompting)
3. [XML 및 HTML 출력](#3-xml-and-html-output)
4. [YAML 출력](#4-yaml-output)
5. [스키마 제약 생성](#5-schema-constrained-generation)
6. [구조화된 출력으로서의 도구 및 함수 호출](#6-tool-and-function-calling-as-structured-output)
7. [Pydantic 모델 검증](#7-pydantic-model-validation)
8. [문법 기반 디코딩](#8-grammar-based-decoding)
9. [중첩 및 재귀 구조 처리](#9-handling-nested-and-recursive-structures)
10. [잘못된 출력에 대한 오류 복구](#10-error-recovery-for-malformed-output)

---

## 1. 구조화된 출력이 중요한 이유

### 1.1 통합 문제

LLM은 기본적으로 자유 형식 텍스트를 생성합니다. 출력이 REST API 응답, 데이터베이스 삽입, 또는 설정 파일에 입력될 때, 자유 형식 텍스트는 데이터 구조로 파싱되어야 합니다. 구조에 대한 명시적 프롬프팅 없이는, 모델이 다음과 같은 문제를 일으킬 수 있습니다:

- JSON을 마크다운 코드 펜스로 감싸기 (```` ```json ... ``` ````)
- 서문 텍스트 포함 ("Here is the JSON output:")
- 응답 간 일관성 없는 필드명 사용
- 필수 필드 누락 또는 예상치 못한 필드 추가
- 에지 케이스에서 구문적으로 유효하지 않은 출력 생성

이러한 각 실패 모드는 취약하고 유지보수가 어려운 방어적 파싱 코드를 필요로 합니다.

### 1.2 신뢰성 스펙트럼

구조화된 출력 기법은 신뢰성 스펙트럼에 위치합니다:

| 기법 | 신뢰성 | 유연성 | 지연 시간 영향 |
|-----------|-------------|-------------|----------------|
| 프롬프트 지시만 | 낮음 (~85-95%) | 높음 | 없음 |
| 퓨샷(Few-shot) 예제 | 중간 (~92-97%) | 높음 | 약간 (토큰 증가) |
| 시스템 프롬프트 + 프리필(Prefill) | 높음 (~95-99%) | 중간 | 없음 |
| 함수/도구 호출(Function/Tool Calling) | 매우 높음 (~99%+) | 중간 | 약간 |
| 스키마 제약 디코딩(Schema-Constrained Decoding) | 보장 (100%) | 낮음 | 보통 |
| 문법 기반 디코딩(Grammar-Based Decoding) | 보장 (100%) | 낮음 | 보통 |

선택은 파싱 실패에 대한 허용 범위와 지연 시간 및 유연성에 대한 제약에 따라 달라집니다.

### 1.3 각 형식의 사용 시기

- **JSON**: API 응답, 데이터베이스 레코드, 설정. 가장 넓은 생태계 지원.
- **XML**: 문서 지향 데이터, SOAP API, 속성이 있는 구조화된 마크업. 혼합 콘텐츠가 있는 계층적 데이터에 적합.
- **YAML**: 설정 파일, 사람이 읽기 쉬운 데이터. 파싱 모호성에 주의.
- **CSV/TSV**: 표 형식 데이터, 스프레드시트 내보내기. 단순하지만 평면 구조로 제한.
- **사용자 정의 형식**: 도메인별 필요 (예: SQL, GraphQL 쿼리).

---

## 2. JSON 출력 프롬프팅

### 2.1 기본 JSON 프롬프팅

가장 간단한 접근법은 모델에게 JSON을 생성하도록 지시하는 것입니다:

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

이것은 유효한 JSON을 생성할 수 있지만, 모델이 마크다운 펜스로 감싸거나 설명 텍스트를 추가할 수 있습니다.

### 2.2 프리필(Prefilling)로 순수 JSON 강제하기

Claude는 어시스턴트 메시지 프리필(Prefilling)을 지원하여, 모델의 응답이 특정 텍스트로 시작하도록 고정합니다:

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

프리필(Prefill) 기법은 서문을 제거하고 모델을 첫 번째 토큰부터 JSON 생성 모드로 강제하기 때문에 매우 효과적입니다.

### 2.3 명시적 스키마를 사용한 JSON

프롬프트에 정확한 스키마를 제공하면 신뢰성이 크게 향상됩니다:

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

### 2.4 일반적인 JSON 함정

**문제 1: 후행 쉼표(Trailing Commas)**

LLM은 때때로 배열이나 객체에서 후행 쉼표를 생성하는데, 이는 유효하지 않은 JSON입니다:

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

**문제 2: 문자열 내 이스케이프되지 않은 문자**

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

**문제 3: 숫자 정밀도**

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

## 3. XML 및 HTML 출력

### 3.1 LLM 출력에 XML을 사용하는 이유

XML은 모델이 학습 데이터에서 엄청난 양의 XML/HTML을 보았기 때문에 LLM 출력에 자연스럽게 적합합니다. 특히 Claude는 입력과 출력 모두를 구조화하기 위한 XML 태그에서 잘 작동합니다. 장점:

- 혼합 콘텐츠(텍스트 + 구조)가 있는 계층적 구조
- 속성이 추가 중첩 없이 메타데이터 제공
- 자기 설명적 태그명
- 모든 언어에서 사용 가능한 견고한 파서

### 3.2 기본 XML 출력

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

### 3.3 Claude의 XML 태그 규칙

Anthropic은 프롬프트와 출력 모두를 구조화하기 위해 XML 태그를 사용하는 것을 권장합니다. 이것은 Claude 프롬프팅 스타일의 독특한 특징입니다:

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

### 3.4 HTML 생성

HTML 스니펫(예: 이메일 템플릿, 보고서 단편)을 생성하는 경우:

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

## 4. YAML 출력

### 4.1 YAML 프롬프팅 고려사항

YAML은 공백에 민감하여 LLM에게 더 까다롭습니다. 모델이 일관성 없는 들여쓰기를 생성할 수 있으며, YAML에는 파싱 모호성이 있습니다 (예: `yes`/`no`가 불리언으로, 콜론이 포함된 비인용 문자열). YAML 출력은 결과가 사람에 의해 읽히거나 편집될 때 사용하세요.

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

### 4.2 YAML 보안 문제

LLM 출력을 파싱할 때는 항상 `yaml.safe_load()`를 사용하고 `yaml.load()`는 사용하지 마세요. 전체 `yaml.load()`는 YAML 태그를 통해 임의의 Python 코드를 실행할 수 있습니다:

```python
import yaml

# DANGEROUS: Never do this with LLM output
# data = yaml.load(llm_output, Loader=yaml.FullLoader)

# SAFE: Always use safe_load
data = yaml.safe_load(llm_output)
```

### 4.3 YAML 대 JSON 트레이드오프

| 특성 | JSON | YAML |
|---------|------|------|
| LLM 신뢰성 | 높음 (엄격한 구문) | 낮음 (공백에 민감) |
| 사람의 가독성 | 좋음 | 매우 좋음 |
| 주석 지원 | 아니오 | 예 |
| 여러 줄 문자열 | 이스케이프된 `\n` | 블록 스칼라 `|` / `>` |
| 파싱 모호성 | 낮음 | 높음 (불리언, null) |
| 생태계 지원 | 보편적 | 넓지만 보편적이지 않음 |

**권장사항**: 기계 간 파이프라인에는 JSON을 사용하고, 사람의 편집이 주요 사용 사례인 경우에만 YAML을 사용하세요.

---

## 5. 스키마 제약 생성(Schema-Constrained Generation)

### 5.1 스키마 제약 생성이란?

스키마 제약 생성(Schema-Constrained Generation)은 프롬프트 지시가 아닌 디코딩 수준에서 모델의 출력이 사전 정의된 스키마를 따르도록 강제합니다. 이는 스키마를 위반하는 토큰이 생성 중에 마스킹되므로 모델이 문자 그대로 유효하지 않은 출력을 생성할 수 없음을 의미합니다.

### 5.2 OpenAI 구조화된 출력(Structured Outputs)

OpenAI는 `response_format` 매개변수를 통해 네이티브 구조화된 출력 지원을 제공합니다:

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

### 5.3 Anthropic의 접근법

2025년 현재, Anthropic은 도구 사용(Tool Use, 섹션 6에서 다룸)과 프롬프트 기반 기법을 통해 구조화된 출력을 지원합니다. Claude는 프롬프트에서 명시적 JSON 스키마를 따르는 데 탁월합니다:

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

### 5.4 LLM을 위한 JSON 스키마 팁

LLM에 스키마를 제공할 때 다음 가이드라인을 따르세요:

1. **설명 포함**: 필드 설명이 타입 제약보다 모델을 더 잘 안내합니다
2. **제어된 어휘에 enum 사용**: `"enum": ["low", "medium", "high"]`
3. **형식 지정**: `"format": "date"`, `"format": "email"` 등
4. **필수 필드를 명시적으로 설정**: 모델이 어떤 필드가 중요한지 추론하는 것에 의존하지 마세요
5. **설명에 예시 제공**: `"description": "ISO 8601 date, e.g. 2024-01-15"`

---

## 6. 구조화된 출력으로서의 도구 및 함수 호출(Tool and Function Calling)

### 6.1 도구 사용 패턴(Tool Use Pattern)

도구/함수 호출(Tool/Function Calling)은 LLM이 외부 도구와 상호작용하도록 설계되었지만, 뛰어난 구조화된 출력 메커니즘으로도 활용됩니다. 원하는 출력과 스키마가 일치하는 "도구"를 정의하면, 모델이 매개변수를 채우고 API가 구조를 검증합니다.

### 6.2 구조화된 출력을 위한 Claude 도구 사용(Tool Use)

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

### 6.3 OpenAI 함수 호출(Function Calling)

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

### 6.4 구조화된 출력을 위한 도구 호출의 이점

1. **API 수준 검증**: API가 스키마에 대해 출력을 검증합니다
2. **파싱 불필요**: 출력이 문자열이 아닌 구조화된 객체로 도착합니다
3. **프리필 해킹 불필요**: 표준 API 흐름 내에서 깔끔하게 작동합니다
4. **Enum 강제**: 모델이 enum 제약을 더 안정적으로 준수합니다
5. **타입 강제 변환**: 숫자, 불리언, null이 적절하게 타입이 지정됩니다

---

## 7. Pydantic 모델 검증

### 7.1 검증 계층으로서의 Pydantic

구조화된 출력 기법을 사용하더라도, Pydantic 검증 계층을 추가하면 심층 방어(Defense in Depth)를 제공합니다:

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

### 7.2 Pydantic 검증이 포함된 전체 파이프라인

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

### 7.3 Pydantic 모델에서 스키마 생성

Pydantic 모델은 LLM에 직접 전달할 수 있는 JSON 스키마를 생성할 수 있습니다:

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

## 8. 문법 기반 디코딩(Grammar-Based Decoding)

### 8.1 문법 기반 디코딩이란?

문법 기반 디코딩(Grammar-Based Decoding)은 형식 문법(일반적으로 문맥 자유 문법)을 사용하여 모델의 토큰 생성을 제약합니다. 각 단계에서 문법 하에서 유효한 연속인 토큰만 허용됩니다. 이는 100% 구문적 유효성을 보장합니다.

### 8.2 GBNF 문법 (llama.cpp)

llama.cpp 생태계는 문법 기반 디코딩을 위해 GBNF (GGML BNF) 표기법을 사용합니다:

```
# GBNF grammar for a simple JSON object with specific fields
root   ::= "{" ws "\"name\"" ws ":" ws string "," ws "\"age\"" ws ":" ws number "," ws "\"city\"" ws ":" ws string "}" ws
string ::= "\"" [^"\\]* "\""
number ::= [0-9]+
ws     ::= [ \t\n]*
```

이 문법은 출력이 항상 정확히 `name`, `age`, `city` 필드를 가진 JSON 객체임을 보장합니다.

### 8.3 Outlines 라이브러리

`outlines` 라이브러리는 Hugging Face 모델을 위한 문법 기반 디코딩을 제공합니다:

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

### 8.4 정규식 제약 생성

더 간단한 패턴의 경우, 정규식 제약으로 충분합니다:

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

### 8.5 문법 기반 디코딩 사용 시기

**사용하는 경우:**
- 100% 형식 보장이 필요한 경우 (안전 중요 애플리케이션)
- 로컬 모델을 실행하고 추론 엔진을 제어할 수 있는 경우
- 출력 스키마가 잘 정의되어 있고 정적인 경우

**피하는 경우:**
- API 기반 모델을 사용하는 경우 (문법 디코딩은 추론 수준 제어가 필요)
- 출력 구조가 동적이거나 컨텍스트 의존적인 경우
- 지연 시간이 중요한 경우 (문법 마스킹이 오버헤드를 추가)

---

## 9. 중첩 및 재귀 구조 처리

### 9.1 중첩 객체 프롬프팅

실제 데이터는 종종 깊은 중첩을 가집니다. 핵심은 스키마에서 전체 중첩 구조를 보여주는 것입니다:

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

### 9.2 재귀 구조

일부 데이터 구조는 본질적으로 재귀적입니다 (예: 파일 트리, 조직도, 댓글 스레드). JSON Schema는 재귀를 위한 `$ref`를 지원하지만, LLM은 명시적 깊이 제한으로 더 잘 처리합니다:

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

### 9.3 깊은 중첩을 위한 전략

1. **평면화 후 재구성**: 모델에게 부모 참조가 있는 평면 목록을 생성하도록 요청한 다음 코드에서 트리를 재구성
2. **레벨별 생성**: 이전 레벨을 컨텍스트로 사용하여 각 레벨을 별도로 생성
3. **명시적 깊이 제한**: 모델에 최대 중첩 깊이를 알려줌
4. **ID 기반 참조**: 리터럴 중첩 대신 ID와 부모 ID 사용

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

## 10. 잘못된 출력에 대한 오류 복구

### 10.1 심층 방어 전략(Defense in Depth Strategy)

최고의 프롬프팅 기법도 가끔 잘못된 출력을 생성합니다. 프로덕션 시스템에는 계층화된 오류 처리가 필요합니다:

```
Layer 1: Prompt design (prevents most errors)
Layer 2: Output extraction (strips wrapper text)
Layer 3: Syntax repair (fixes common JSON issues)
Layer 4: Validation (Pydantic / schema check)
Layer 5: Retry with error feedback (LLM self-correction)
Layer 6: Fallback (default values or human escalation)
```

### 10.2 견고한 JSON 추출 파이프라인

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

### 10.3 오류 피드백을 통한 재시도

초기 파싱이 실패하면, 오류를 모델에 다시 보내 자체 수정하도록 합니다:

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

### 10.4 스트리밍 JSON 파싱

대규모 구조화된 출력의 경우, 토큰이 도착하면서 점진적으로 파싱할 수 있습니다:

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

### 10.5 모니터링 및 알림

프로덕션에서 구조화된 출력 실패를 추적합니다:

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

## 연습문제

### 연습문제 1: 다중 형식 추출 파이프라인

사용자가 선택한 형식(JSON, XML, 또는 YAML)으로 기사에서 구조화된 데이터를 추출하여 반환하는 함수를 빌드하세요. 추출된 데이터에는 제목, 저자, 발행일, 요약(최대 100단어), 주요 주제(목록), 감성(긍정/부정/중립)이 포함되어야 합니다.

**요구사항:**
- `format` 매개변수를 통해 세 가지 출력 형식 모두 지원
- 출력 형식에 관계없이 Pydantic으로 검증
- 모델에 다른 형식을 요청하는 것이 아니라 코드에서 형식 변환 처리

<details><summary>정답 보기</summary>

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

### 연습문제 2: 스키마 진화 핸들러

스키마 버전 관리를 처리하는 시스템을 만드세요. 스키마 v1을 따르는 이전 JSON 응답이 주어지면, 하위 호환성을 유지하면서 LLM을 사용하여 스키마 v2를 따르도록 변환하세요.

**요구사항:**
- 의미 있는 차이가 있는 v1과 v2 Pydantic 모델 정의 (필드명 변경, 새 필수 필드, 타입 변경)
- Claude를 사용하여 v1 데이터를 v2 형식으로 변환
- Pydantic으로 입력(v1)과 출력(v2) 모두 검증

<details><summary>정답 보기</summary>

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

### 연습문제 3: 견고한 JSON 배열 스트리밍

Claude에게 N개 항목의 JSON 배열(예: 가상 제품 항목)을 생성하도록 요청하고, 응답을 스트리밍하며, 각 완전한 객체가 사용 가능해지면 즉시 yield하는 함수를 작성하세요. 스트림이 객체 중간에 중단되는 경우를 처리하세요.

**요구사항:**
- Anthropic 스트리밍 API 사용
- 각 완전한 JSON 객체가 닫히는 즉시 yield
- 부분 객체 복구를 위한 추적
- 타임아웃 처리 포함

<details><summary>정답 보기</summary>

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

### 연습문제 4: XML에서 Pydantic 파이프라인

Claude에게 XML 출력을 생성하도록 프롬프트하고, 파싱한 다음, 검증된 Pydantic 모델로 변환하는 시스템을 빌드하세요. 사용 사례: 비구조화된 텍스트에서 레시피를 구조화된 형식으로 파싱.

**요구사항:**
- Claude에게 XML 출력을 프롬프트 (JSON이 아님)
- `xml.etree.ElementTree`를 사용하여 XML 파싱
- 적절한 타입(지속 시간, 수량을 숫자로)으로 Pydantic 모델로 변환
- 누락된 선택적 필드를 우아하게 처리

<details><summary>정답 보기</summary>

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

### 연습문제 5: 비교 형식 벤치마크

서로 다른 구조화된 출력 기법의 신뢰성을 벤치마크하는 스크립트를 작성하세요. 동일한 추출 작업을 다음을 사용하여 보내세요: (a) 프롬프트 전용 JSON, (b) 프리필(Prefill) 기반 JSON, (c) 도구 호출(Tool Calling). 각각을 N번 실행하고 다음을 측정하세요: 파싱 성공률, 스키마 준수율, 평균 지연 시간.

**요구사항:**
- 세 가지 방법 모두에 동일한 추출 작업
- 방법당 최소 3회 실행 (실제 벤치마크에는 10회 사용)
- 파싱 비율, 검증 비율, 지연 시간 측정 및 보고
- 비교 표 출력

<details><summary>정답 보기</summary>

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

**이전**: [고급 추론 프롬프트](./04_Advanced_Reasoning_Prompts.md) | **다음**: [시스템 프롬프트 설계](./06_System_Prompt_Design.md)
