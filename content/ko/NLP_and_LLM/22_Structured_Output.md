# 22. LLM의 구조화된 출력

## 학습 목표

- JSON 모드와 응답 형식 제어를 사용한 구조화된 생성
- 신뢰성 있는 구조화 추출을 위한 함수 호출 구현
- Pydantic 기반 출력 파싱과 검증 및 재시도 로직 구축
- instructor 라이브러리를 사용한 타입 안전 LLM 출력
- 복잡한 중첩 스키마 및 프로덕션 데이터 추출 파이프라인 설계

---

## 이론과 원리

자유 텍스트 LLM 응답은 채팅에 작동합니다. 출력을 **파싱**하려는 어떤 다운스트림 시스템에도 실패합니다 — 엔티티를 데이터베이스에 추출, 폼 채우기, 타입화된 파라미터로 행동 트리거. "JSON 출력"은 대부분의 시간 작동하지만 가끔 무효 JSON, 누락 필드, 예상치 못한 타입을 만듭니다 — 그리고 규모에서 "가끔"은 하루에 수천 실패를 의미합니다. 구조화된 출력 기법은 토큰 수준에서 모델 생성을 제약하거나, 생성 후 재시도와 함께 검증하거나, 둘 다를 통해 이 격차를 메웁니다.

이 섹션은 다음을 다룹니다:

- **(A) 제약 생성 문제** — 실제로 어떤 보장이 필요하며 어떤 비용으로.
- **(B) 프롬프트 수준 구조화** — JSON 모드, 시스템 프롬프트 지시, 정중히 부탁의 한계.
- **(C) 구조화된 출력으로서의 function calling** — OpenAI/Anthropic의 도구 호출 API가 어떻게 구조를 보장하는가.
- **(D) 문법 기반 제약 디코딩** — Outlines, LMQL, JSON 스키마 인식 토큰 마스킹.
- **(E) Pydantic 파싱-그리고-재시도** — 별도 단계로서의 검증, 실패 시 재시도와 함께.
- **(F) instructor 라이브러리** — 타입 안전 Python 인터페이스, 자동 재시도, 스트리밍을 위한 부분 출력.
- **(G) 스키마 설계** — 중첩, 선택성, enum, 정밀도와 모델 성공률 사이의 트레이드오프.

### A. 제약 생성 문제

`parse(s)`가 성공하고 파싱된 결과가 어떤 스키마를 만족하는 문자열 `s`를 LLM이 만들기를 원합니다. 세 보장 수준:

- **수준 0 (프롬프트만)** — 모델에 정중히 부탁. 단순 스키마에서 ~95% 성공, 복잡도와 함께 떨어짐.
- **수준 1 (검증 + 재시도)** — 출력을 파싱; 실패하면 오류와 함께 재프롬프트하고 다시 시도. 1-2 재시도 후 ~99% 성공.
- **수준 2 (제약 디코딩)** — 각 단계에서 모델의 토큰 선택을 스키마의 유효한 접두사를 유지하는 것으로 제한. 100% 성공 보장(지원될 때).

각 수준이 능력과 비용을 더합니다. 프로덕션 시스템은 수용 가능한 실패율을 주는 가장 낮은 수준을 선택, 지연과 비용을 재시도 빈도와 비교 측정.

### B. 프롬프트 수준 구조화

가장 단순한 접근 — 모델에 무엇을 원하는지 말하기.

```
"name"(문자열), "age"(정수), "skills"(문자열 배열) 키를 가진 JSON 출력.
JSON만 출력. 설명 없음.
```

OpenAI의 "JSON 모드"(`response_format = {"type": "json_object"}`)는 출력을 파싱 가능한 JSON으로 제약하지만 특정 스키마를 강제하지 않습니다. 필드 이름, 타입, 구조는 여전히 모델에 달려 있고 — 모델이 여전히 필드를 환각, 필수 필드를 생략, 또는 타입을 바꿀 수 있습니다.

프로토타이핑이나 위험이 낮은 시스템에 프롬프트 수준 사용. 프로덕션에는 검증(E)과 짝짓기.

### C. 구조화된 출력으로서의 Function Calling

도구 호출(레슨 23)을 위해 설계되었지만, function calling API는 구조화된 출력을 얻는 가장 깔끔한 방법이기도 합니다. 파라미터가 원하는 스키마인 "도구"를 정의; 모델에게 추출된 데이터로 그 도구를 호출하라고 요청:

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
# 모델 반환: {"tool_calls": [{"name": "save_person", "arguments": {"name": "...", ...}}]}
```

인자는 선언된 타입의 유효한 JSON임이 보장됩니다(API가 강제). 실제로 도구를 실행할 필요는 없습니다 — API의 스키마 강제만 사용합니다.

OpenAI의 "Strict 모드" function calling과 Anthropic의 도구 사용 모두 API 수준의 토큰 수준 제약으로 이를 구현 — 무효 토큰은 그저 표본 추출되지 않습니다. ~100% 성공률.

### D. 문법 기반 제약 디코딩

제공자 측 강제가 없는 오픈소스 모델에 대해, **Outlines**(Willard & Louf, 2023)와 **LMQL** 같은 라이브러리가 제약 디코딩을 직접 구현.

**메커니즘.** JSON 스키마(또는 정규식)가 유한 상태 자동기(FSA)로 컴파일됩니다. 각 생성 단계에서 모델이 모든 `V` 토큰에 대한 로짓을 만듭니다; 제약이 FSA를 통한 어떤 유효 경로도 진행시키지 않을 토큰을 마스킹하고, 모델이 나머지에서 표본 추출합니다. 이는 출력이 구성에 의해 스키마와 일치함을 보장합니다.

**비용.** FSA 상태 갱신을 위한 단계당 작은 계산 — 보통 무시할 만함. 일부 제약(특히 복잡한 JSON 스키마)은 컴파일이 느릴 수 있지만 결과는 쿼리 사이 재사용 가능.

이는 API 지원 없는 로컬 모델에 **범주적** 보장을 주는 유일한 접근입니다.

### E. Pydantic 파싱-그리고-재시도

가장 흔한 프로덕션 패턴:

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
            prompt = f"{prompt}\n\n이전 시도가 검증 실패: {e}\n수정해서 재시도해 주세요."
    raise ValueError("최대 재시도 초과")
```

검증기가 JSON 파싱 오류와 타입/제약 위반 모두를 잡습니다. 오류와 함께 재프롬프트는 보통 다음 시도에서 유효한 출력을 만듭니다. 프롬프트 수준(B)을 검증을 백스톱으로 결합.

이는 적당한 비용으로 ~99%+ 성공을 줍니다(단순 스키마에 평균 ~20 호출당 1 재시도).

### F. instructor 라이브러리

`instructor`(Liu, 2023)는 OpenAI/Anthropic SDK를 감싸 Pydantic 기반 추출을 주 인터페이스로 만듭니다:

```python
import instructor
from openai import OpenAI

client = instructor.from_openai(OpenAI())
person = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": text}],
    response_model=Person,
)
# `person`은 타입화된 Person 인스턴스 — JSON 파싱 없음, 사용자 코드의 검증 없음
```

배후에서 — instructor가 Pydantic 모델을 function-calling 스키마(C)로 변환, API 호출, 결과 검증, 실패 시 재시도(E). 스트리밍을 위한 부분 검증(타입화된 청크가 도착하면 yield)과 목록 추출을 위한 `Iterable[Person]`도 지원.

이것이 프로덕션 패턴 — 사용자 관점에서 타입 안전성, 배후의 견고한 생성.

### G. 스키마 설계

스키마 설계가 모델의 성공률에 직접 영향을 줍니다.

**G.1 선택적 필드.** 항상 존재하지 않는 필드를 `Optional[T]`로 표시. 모델이 데이터가 존재하는지 고려하도록 강제; 가짜 값의 환각을 줄임.

**G.2 Enum.** 문자열 필드를 닫힌 집합으로 제약 — `Literal["pending", "approved", "rejected"]`. 오타와 발명된 범주를 제거.

**G.3 필드 설명의 예시.** Pydantic의 `Field(description="...")`이 JSON 스키마에 포함됩니다. 예시가 모델이 무엇을 추출할지 이해하는 데 도움 — `Field(description="사람의 전체 법적 이름, 예: 'John Smith'")`.

**G.4 중첩 깊이.** 각 중첩 수준이 실패율을 증가(모델이 자신을 혼란시킬 곳이 더 많음). 가능하면 평탄화 — `customer.name`보다 `customer_name` 선호.

**G.5 수치 제약.** `age`에 `Field(ge=0, le=120)`, 문자열에 `Field(min_length=1, max_length=100)` 사용. 모델이 보통 이를 존중; 검증기가 나머지를 잡음.

일반 원리 — **불변량을 프롬프트가 아닌 스키마에 인코딩.** 스키마 제약은 검사 가능; 프롬프트 지시는 권고적.

### 이론에서 아래 함수들로

- §1 (도전 과제) — §A 세 보장 수준을 틀.
- §2 (JSON 모드) — OpenAI/Anthropic JSON 모드로 §B 프롬프트 수준 구조화 구현.
- §3 (function calling) — 구조화된 출력으로서의 §C function calling 구현.
- §4 (Pydantic) — §E 파싱-재시도 패턴 구현.
- §5 (instructor 라이브러리) — 타입 안전 추출을 위해 §F 래퍼 사용.
- §6 (OpenAI Structured Outputs) — 제공자 네이티브 §C/§D 융합(strict 모드 function calling).
- §7 (프로덕션 파이프라인) — §G 스키마 설계와 함께 §A-§F를 현실적 데이터 추출 파이프라인으로 결합.

---

## 1. 구조화된 출력의 과제

### 구조화된 출력이 중요한 이유

> **구조화된 출력 사용 사례**
>
> - **데이터 추출**: 비정형 문서에서 구조화된 레코드 추출
> - **API 통합**: 다운스트림 서비스를 위한 유효한 페이로드 생성
> - **데이터베이스 수집**: 자유 텍스트를 관계형 또는 문서 레코드로 변환
> - **워크플로우 자동화**: LLM 결정을 실행 가능한 액션 객체로 파싱
> - **분석 파이프라인**: 보고서를 머신 리더블 형식으로 변환

### 접근 방식 비교

| 접근 방식 | 신뢰성 | 유연성 | 복잡도 | 최적 사용 |
|-----------|--------|--------|--------|-----------|
| Regex 파싱 | 낮음 | 낮음 | 낮음 | 단순 패턴 |
| JSON 모드 | 중간 | 중간 | 낮음 | 기본 JSON 객체 |
| 함수 호출 | 높음 | 높음 | 중간 | 도구 통합 |
| Pydantic + Instructor | 매우 높음 | 매우 높음 | 중간 | 프로덕션 시스템 |
| OpenAI Structured Outputs | 매우 높음 | 높음 | 낮음 | OpenAI 전용 앱 |
| Grammar-Constrained | 최고 | 중간 | 높음 | 로컬 모델 (llama.cpp) |

---

## 2. JSON 모드

### OpenAI JSON 모드

```python
from openai import OpenAI
import json

client = OpenAI()

def extract_with_json_mode(text: str) -> dict:
    """JSON 모드를 사용한 구조화된 데이터 추출."""
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

# 사용 예시
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

### Anthropic JSON 모드

```python
from anthropic import Anthropic

anthropic = Anthropic()

def extract_with_claude(text: str) -> dict:
    """Claude를 사용한 구조화된 데이터 추출."""
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
    # Claude는 전용 JSON 모드가 없으므로 응답을 파싱
    content = response.content[0].text
    # 마크다운 코드 펜스가 있으면 제거
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(content)
```

### JSON 모드의 함정

```python
# 문제 1: JSON 모드는 스키마 적합성을 보장하지 않음
# 모델이 유효한 JSON을 반환하지만 예상 스키마와 일치하지 않을 수 있음

def safe_json_extract(text: str, required_keys: list[str]) -> dict | None:
    """스키마 검증을 포함한 JSON 추출."""
    result = extract_with_json_mode(text)

    # 필수 키 검증
    missing_keys = [k for k in required_keys if k not in result]
    if missing_keys:
        print(f"Warning: Missing keys: {missing_keys}")
        return None

    return result

# 문제 2: 일관되지 않은 타입
# 모델이 3 (int) 대신 "3" (string)을 반환할 수 있음
# 항상 타입을 검증하고 강제 변환

def coerce_types(data: dict, schema: dict) -> dict:
    """JSON 값을 예상 타입으로 강제 변환."""
    coerced = {}
    for key, expected_type in schema.items():
        if key in data:
            try:
                coerced[key] = expected_type(data[key])
            except (ValueError, TypeError):
                coerced[key] = data[key]  # 변환 실패 시 원본 유지
    return coerced

# 사용 예시
schema = {"price": float, "quantity": int, "name": str}
raw = {"price": "29.99", "quantity": "5", "name": "Widget"}
clean = coerce_types(raw, schema)
# {"price": 29.99, "quantity": 5, "name": "Widget"}
```

---

## 3. 구조화 추출을 위한 함수 호출

### 스키마 기반 추출

```python
def extract_with_function_calling(text: str) -> dict:
    """출력 구조를 강제하기 위해 함수 호출 사용."""
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

    # 함수 호출 인자 추출
    tool_call = response.choices[0].message.tool_calls[0]
    return json.loads(tool_call.function.arguments)

# 사용 예시
result = extract_with_function_calling(text)
print(json.dumps(result, indent=2))
```

### 다중 추출 함수

```python
def multi_schema_extract(text: str) -> dict:
    """모델이 적절한 추출 스키마를 선택하게 함."""
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
        tool_choice="auto",  # 모델이 최적의 함수를 선택
        temperature=0.0,
    )

    results = {}
    for tool_call in response.choices[0].message.tool_calls:
        func_name = tool_call.function.name
        results[func_name] = json.loads(tool_call.function.arguments)

    return results
```

---

## 4. Pydantic 출력 파싱

### 기본 Pydantic 모델

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

### LangChain 출력 파서

```python
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Pydantic 모델에서 파서 생성
parser = PydanticOutputParser(pydantic_object=DocumentExtraction)

# 포맷 지시가 주입된 프롬프트 빌드
prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Extract structured information from the text.\n\n"
        "{format_instructions}"
    )),
    ("user", "{text}"),
])

# 체인: 프롬프트 -> LLM -> 파서
llm = ChatOpenAI(model="gpt-4o", temperature=0)
chain = prompt | llm | parser

# 추출 실행
result: DocumentExtraction = chain.invoke({
    "text": text,
    "format_instructions": parser.get_format_instructions(),
})

print(f"제목: {result.title}")
print(f"엔티티: {[e.name for e in result.entities]}")
print(f"감성: {result.sentiment}")
```

### 에러 피드백을 활용한 재시도

```python
from langchain.output_parsers import RetryWithErrorOutputParser
from langchain_core.runnables import RunnablePassthrough

# 재시도 로직으로 파서를 래핑
retry_parser = RetryWithErrorOutputParser.from_llm(
    parser=parser,
    llm=llm,
    max_retries=3,
)

def extract_with_retry(text: str) -> DocumentExtraction | None:
    """파싱 실패 시 자동 재시도로 추출."""
    prompt_value = prompt.invoke({
        "text": text,
        "format_instructions": parser.get_format_instructions(),
    })

    # 첫 번째 시도
    response = llm.invoke(prompt_value)

    try:
        return parser.parse(response.content)
    except Exception as e:
        print(f"첫 번째 파싱 실패: {e}")
        # 에러 컨텍스트를 모델에 다시 전달하여 재시도
        try:
            return retry_parser.parse_with_prompt(
                response.content,
                prompt_value,
            )
        except Exception as e2:
            print(f"재시도 실패: {e2}")
            return None
```

---

## 5. Instructor 라이브러리

### 타입 안전 LLM 출력

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Literal

# OpenAI 클라이언트를 instructor로 패치
client = instructor.from_openai(OpenAI())

class UserProfile(BaseModel):
    name: str
    age: int = Field(ge=0, le=150)
    email: str
    interests: list[str] = Field(min_length=1)
    experience_level: Literal["beginner", "intermediate", "advanced"]

# instructor가 파싱, 검증, 재시도를 자동으로 처리
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

# 타입 안전하게 필드에 접근
print(f"이름: {profile.name}, 나이: {profile.age}")
```

### 복잡한 중첩 구조

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
    """구조화된 회의록 추출."""
    title: str
    date: str
    attendees: list[str] = Field(min_length=1)
    agenda_items: list[str]
    decisions: list[Decision]
    action_items: list[ActionItem]
    next_meeting: str | None = None
    key_discussion_points: list[str]

# 구조화된 회의록 추출
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

### Instructor와 Anthropic

```python
import instructor
from anthropic import Anthropic

# Anthropic 클라이언트 패치
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

### 부분 객체 스트리밍

```python
from instructor import Partial

# 부분 결과를 도착하는 대로 스트리밍
for partial_profile in client.chat.completions.create_partial(
    model="gpt-4o",
    response_model=UserProfile,
    messages=[
        {"role": "user", "content": "Extract: John, 28, john@tech.com, loves ML and Python, expert coder"},
    ],
    stream=True,
):
    # partial_profile은 필드가 점진적으로 채워짐
    print(f"부분: {partial_profile.model_dump()}")
    # 첫 반복: {"name": "John", "age": None, ...}
    # 다음: {"name": "John", "age": 28, "email": None, ...}
    # 최종: 모든 필드가 채워짐
```

### 검증 및 재시도 전략

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

# instructor는 검증 실패 시 에러 메시지를 LLM에 다시 전달하며 자동 재시도
invoice = client.chat.completions.create(
    model="gpt-4o",
    response_model=Invoice,
    max_retries=3,  # 검증 실패 시 최대 3회 재시도
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

print(f"송장: {invoice.invoice_number}")
print(f"합계: ${invoice.total:.2f}")
for item in invoice.items:
    print(f"  {item.description}: {item.quantity} x ${item.unit_price} = ${item.total}")
```

---

## 6. OpenAI Structured Outputs

### Strict 모드

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

# Strict structured outputs -- 스키마 적합성 보장
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
    print(f"단계 {i}: {step.explanation} -> {step.output}")
print(f"답: {solution.final_answer}")
```

### 스키마 설계 모범 사례

```python
from pydantic import BaseModel, Field
from typing import Literal

# 좋은 예: 명확한 설명과 제약이 있는 구체적 타입
class GoodSchema(BaseModel):
    """LLM 추출을 위한 잘 설계된 스키마."""
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

# 나쁜 예: 모호한 타입, 설명 없음, 제약 없음
class BadSchema(BaseModel):
    type: str        # 너무 모호함 -- 모델이 아무거나 반환할 수 있음
    level: int       # 범위가 뭐지? 1-5? 1-10?
    info: str        # 모호한 필드명
    data: dict       # 완전히 비정형
    tags: list       # 무엇의 리스트?

# 팁: 카테고리 필드에는 Literal을 사용하여 출력을 제한
# 팁: Field 설명을 추가 -- 프롬프트의 일부가 됨
# 팁: 선택적 리스트에는 default_factory를 사용
# 팁: 신뢰성 있는 추출을 위해 중첩 깊이를 3단계 이하로 유지
```

---

## 7. 프로덕션 데이터 추출 파이프라인

### 엔드투엔드 파이프라인

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
    """프로덕션 수준의 데이터 추출 파이프라인."""

    def __init__(self, primary_model: str = "gpt-4o",
                 fallback_model: str = "gpt-4o-mini",
                 max_retries: int = 3):
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.max_retries = max_retries
        self.client = instructor.from_openai(OpenAI())

    def extract(self, text: str, schema: Type[T],
                instructions: str = "") -> ExtractionResult:
        """폴백과 에러 처리가 있는 구조화된 데이터 추출."""
        errors = []
        total_tokens = 0

        # 시도 1: 주 모델
        for attempt in range(self.max_retries):
            try:
                result = self.client.chat.completions.create(
                    model=self.primary_model,
                    response_model=schema,
                    max_retries=1,  # 외부 루프가 재시도 처리
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
                logger.warning(f"검증 실패 (시도 {attempt+1}): {error_msg}")
            except Exception as e:
                errors.append(f"Attempt {attempt+1}: {str(e)}")
                logger.error(f"추출 실패 (시도 {attempt+1}): {e}")

        # 시도 2: 폴백 모델
        logger.info(f"{self.fallback_model}로 폴백")
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
        """재시도를 위한 선택적 에러 컨텍스트가 포함된 메시지 빌드."""
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
        """여러 텍스트에서 추출."""
        results = []
        for i, text in enumerate(texts):
            logger.info(f"처리 중 {i+1}/{len(texts)}")
            result = self.extract(text, schema, instructions)
            results.append(result)
        return results

# 사용 예시
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
    print(f"제품: {review.product_name}")
    print(f"평점: {review.rating}/5")
    print(f"장점: {review.pros}")
    print(f"단점: {review.cons}")
    print(f"추천: {review.recommendation}")
    print(f"재시도 횟수: {result.retries}")
else:
    print(f"추출 실패: {result.errors}")
```

### 파이프라인 모니터링

```python
from collections import defaultdict
import time

class PipelineMetrics:
    """추출 파이프라인 성능 추적."""

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

# 통합
metrics = PipelineMetrics()

start = time.time()
result = pipeline.extract(review_text, ProductReview)
latency = time.time() - start

metrics.record(result, latency)
print(json.dumps(metrics.summary(), indent=2))
```

---

## 다음 단계

[23_Function_Calling_Tools.md](./23_Function_Calling_Tools.md)에서는 함수 호출과 도구 사용 API를 심층적으로 다루며, Model Context Protocol (MCP) 및 고급 도구 오케스트레이션 패턴을 포함한다.
