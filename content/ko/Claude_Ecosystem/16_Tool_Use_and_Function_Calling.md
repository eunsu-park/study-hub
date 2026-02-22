# 도구 사용과 함수 호출

**이전**: [15. Claude API 기초](./15_Claude_API_Fundamentals.md) | **다음**: [17. Claude Agent SDK](./17_Claude_Agent_SDK.md)

---

도구 사용(함수 호출(Function Calling)이라고도 함)은 Claude가 여러분이 정의한 함수를 호출할 수 있게 해주는 메커니즘입니다. Claude는 무언가를 *할 것*이라는 텍스트를 생성하는 대신, 실제로 *할 수* 있습니다 -- 데이터를 조회하고, 데이터베이스를 쿼리하고, API를 호출하고, 계산을 수행하고, 외부 시스템과 상호작용합니다. 이 레슨에서는 도구 사용의 전체 수명 주기를 다룹니다: 도구 정의, 다단계 대화 흐름 처리, 병렬 도구 호출 관리, 그리고 프로덕션 시스템을 위한 모범 사례.

**난이도**: ⭐⭐

**사전 요구 사항**:
- 레슨 15 (Claude API 기초) 완료
- JSON Schema 기초에 대한 이해
- Python 3.9+ 또는 Node.js 18+

**학습 목표**:
- 이름, 설명, JSON Schema 입력 사양으로 도구 정의하기
- 전체 도구 사용 대화 흐름 구현하기 (요청, tool_use, tool_result, 응답)
- Claude가 동시에 여러 도구를 요청할 때 병렬 도구 실행 처리하기
- `tool_choice` 매개변수를 사용하여 Claude가 도구를 사용하는 방법과 시기 제어하기
- 도구에서 오류를 반환하고 엣지 케이스 처리하기
- 고급 패턴 적용하기: 체이닝, 구조화된 출력, 이미지 반환
- 모범 사례를 따르는 효과적인 도구 설계하기

---

## 목차

1. [도구 사용이란?](#1-도구-사용이란)
2. [도구 정의 형식](#2-도구-정의-형식)
3. [도구 사용 대화 흐름](#3-도구-사용-대화-흐름)
4. [Python 예제: 날씨 조회](#4-python-예제-날씨-조회)
5. [TypeScript 예제: 데이터베이스 쿼리](#5-typescript-예제-데이터베이스-쿼리)
6. [다중 도구](#6-다중-도구)
7. [병렬 도구 실행](#7-병렬-도구-실행)
8. [도구 사용 제어: tool_choice](#8-도구-사용-제어-tool_choice)
9. [도구에서의 오류 처리](#9-도구에서의-오류-처리)
10. [고급 패턴](#10-고급-패턴)
11. [모범 사례](#11-모범-사례)
12. [연습 문제](#12-연습-문제)
13. [참고 자료](#13-참고-자료)

---

## 1. 도구 사용이란?

도구 사용 없이는 Claude는 텍스트만 생성할 수 있습니다. 날씨를 확인하는 방법을 설명할 수는 있지만, 실제로 확인할 수는 없습니다. 도구 사용은 이 간격을 메웁니다:

```
┌────────────────────────────────────────────────────────────────────┐
│                도구 사용 없이                                        │
│                                                                     │
│  사용자: "도쿄의 날씨는 어때요?"                                    │
│  Claude: "실시간 데이터에 접근할 수 없습니다. weather.com이나       │
│           날씨 앱을 확인해 보세요."                                 │
│                                                                     │
├────────────────────────────────────────────────────────────────────┤
│                도구 사용 시                                          │
│                                                                     │
│  사용자: "도쿄의 날씨는 어때요?"                                    │
│  Claude: [get_weather(city="Tokyo") 호출]                           │
│  시스템: [{"temp": 22, "condition": "sunny"} 반환]                 │
│  Claude: "도쿄는 현재 22°C이고 맑습니다."                          │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

**도구 사용이 중요한 이유:**
- **실시간 데이터**: 현재 정보 조회 (날씨, 주가, 데이터베이스 레코드)
- **부수 효과(Side Effects)**: 파일 생성, 이메일 전송, 데이터베이스 업데이트, 코드 배포
- **계산**: 정밀한 계산 수행, 코드 실행, 데이터 처리
- **통합**: Claude를 API, 서비스 또는 시스템에 연결
- **구조화된 출력**: 도구 스키마를 통해 Claude가 특정 형식으로 데이터를 반환하도록 강제

---

## 2. 도구 정의 형식

각 도구는 세 가지 구성 요소로 정의됩니다:

```json
{
  "name": "get_weather",
  "description": "Get the current weather for a given city. Returns temperature, conditions, and humidity.",
  "input_schema": {
    "type": "object",
    "properties": {
      "city": {
        "type": "string",
        "description": "The city name, e.g., 'Tokyo' or 'New York'"
      },
      "units": {
        "type": "string",
        "enum": ["celsius", "fahrenheit"],
        "description": "Temperature unit. Defaults to celsius."
      }
    },
    "required": ["city"]
  }
}
```

### 2.1 이름(Name)

- 요청 내에서 고유해야 합니다.
- `snake_case` 사용 (관례).
- 설명적으로 작성: `gsp`가 아닌 `get_stock_price`.

### 2.2 설명(Description)

설명은 **핵심**입니다 -- Claude는 이를 사용하여 언제 도구를 호출할지 결정합니다. 동료에게 도구를 설명하듯이 작성하십시오:

```
좋음:  "회사 지식 베이스에서 쿼리와 일치하는 문서를 검색합니다.
        제목, 스니펫, URL이 포함된 최대 10개의 결과를 반환합니다.
        사용자가 회사 정책, 절차 또는 내부 문서에 대해
        질문할 때 사용하십시오."

나쁨:  "KB 검색"
```

### 2.3 입력 스키마(JSON Schema)

`input_schema`는 [JSON Schema](https://json-schema.org/) 명세를 따릅니다:

```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "Search query string",
      "minLength": 1,
      "maxLength": 500
    },
    "limit": {
      "type": "integer",
      "description": "Maximum number of results to return",
      "minimum": 1,
      "maximum": 50,
      "default": 10
    },
    "filters": {
      "type": "object",
      "description": "Optional filters to narrow results",
      "properties": {
        "category": {
          "type": "string",
          "enum": ["engineering", "hr", "finance", "legal"]
        },
        "date_after": {
          "type": "string",
          "format": "date",
          "description": "Only return articles published after this date (YYYY-MM-DD)"
        }
      }
    }
  },
  "required": ["query"]
}
```

**도구 정의를 위한 주요 JSON Schema 기능:**
- `type`: string, integer, number, boolean, array, object
- `enum`: 특정 값으로 제한
- `description`: 각 필드 설명 (Claude가 읽음)
- `required`: 필수 필드 표시
- `default`: 기본값 지정
- `minimum` / `maximum`: 숫자 제약 조건
- `minLength` / `maxLength`: 문자열 길이 제약 조건
- `items`: 배열 요소의 스키마

---

## 3. 도구 사용 대화 흐름

도구 사용은 애플리케이션과 Claude 사이의 다단계 교환을 포함합니다:

```
┌──────────────────────────────────────────────────────────────────┐
│              도구 사용 대화 흐름                                   │
│                                                                   │
│  1단계: 사용자 메시지 + 도구 정의 전송                             │
│  ┌──────────┐         ┌──────────────┐                           │
│  │   앱     │ ──────▶ │    Claude    │                           │
│  │          │  tools   │    API      │                           │
│  └──────────┘ + msg    └──────┬───────┘                          │
│                               │                                   │
│  2단계: Claude가 tool_use 블록으로 응답                            │
│                               │                                   │
│  ┌──────────┐         ┌──────▼───────┐                           │
│  │   앱     │ ◀────── │   응답:      │                           │
│  │          │  tool_   │   tool_use   │                           │
│  └────┬─────┘  use     └──────────────┘                          │
│       │                                                           │
│  3단계: 도구를 로컬에서 실행                                        │
│       │  앱이 실제 함수를 호출함                                    │
│       ▼                                                           │
│  ┌──────────┐                                                     │
│  │  도구    │  get_weather("Tokyo")                               │
│  │ 함수     │  → {"temp": 22, "condition": "sunny"}              │
│  └────┬─────┘                                                     │
│       │                                                           │
│  4단계: Claude에 tool_result 전송                                  │
│       │                                                           │
│  ┌────▼─────┐         ┌──────────────┐                           │
│  │   앱     │ ──────▶ │    Claude    │                           │
│  │          │  tool_   │    API      │                           │
│  └──────────┘  result  └──────┬───────┘                          │
│                               │                                   │
│  5단계: Claude가 최종 응답 생성                                     │
│                               │                                   │
│  ┌──────────┐         ┌──────▼───────┐                           │
│  │   앱     │ ◀────── │  "22°C이고   │                           │
│  │          │  text    │  도쿄는      │                           │
│  └──────────┘          │  맑습니다."  │                           │
│                        └──────────────┘                           │
└──────────────────────────────────────────────────────────────────┘
```

핵심: **Claude는 도구를 직접 실행하지 않습니다**. *어떤* 도구를 호출하고 *어떤 인자*를 전달할지 알려줍니다. 여러분의 애플리케이션이 도구를 실행하고 결과를 다시 전송합니다.

---

## 4. Python 예제: 날씨 조회

다음은 Python에서 도구 사용의 완전한 동작 예제입니다:

```python
import anthropic
import json

client = anthropic.Anthropic()

# 1단계: 도구 정의
tools = [
    {
        "name": "get_weather",
        "description": (
            "Get the current weather for a city. "
            "Returns temperature (Celsius), condition, and humidity."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g., 'London' or 'San Francisco'",
                },
            },
            "required": ["city"],
        },
    }
]

# 2단계: 도구 구현 정의
def get_weather(city: str) -> dict:
    """날씨 API 호출을 시뮬레이션합니다."""
    # 프로덕션에서는 OpenWeatherMap 등의 실제 API를 호출합니다
    weather_data = {
        "tokyo": {"temp": 22, "condition": "Sunny", "humidity": 55},
        "london": {"temp": 12, "condition": "Cloudy", "humidity": 78},
        "new york": {"temp": 18, "condition": "Partly Cloudy", "humidity": 62},
    }
    data = weather_data.get(city.lower())
    if data is None:
        return {"error": f"No weather data available for '{city}'"}
    return {"city": city, **data}

# 3단계: 도구 사용 루프
def chat_with_tools(user_message: str) -> str:
    """메시지를 전송하고 도구 사용 요청을 처리합니다."""
    messages = [{"role": "user", "content": user_message}]

    # 도구와 함께 초기 API 호출
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=tools,
        messages=messages,
    )

    # Claude가 최종 텍스트 응답을 줄 때까지 루프 실행
    while response.stop_reason == "tool_use":
        # 응답에서 tool_use 블록 추출
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  [Tool call: {block.name}({json.dumps(block.input)})]")

                # 도구 실행
                if block.name == "get_weather":
                    result = get_weather(**block.input)
                else:
                    result = {"error": f"Unknown tool: {block.name}"}

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })

        # Claude의 응답과 도구 결과를 대화에 추가
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

        # 도구 결과와 함께 API를 다시 호출
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=tools,
            messages=messages,
        )

    # 최종 텍스트 응답 추출
    return "".join(
        block.text for block in response.content if block.type == "text"
    )

# 사용 예시
answer = chat_with_tools("What is the weather like in Tokyo right now?")
print(f"\nClaude: {answer}")
```

출력:
```
  [Tool call: get_weather({"city": "Tokyo"})]

Claude: It's currently 22°C and sunny in Tokyo with 55% humidity.
```

---

## 5. TypeScript 예제: 데이터베이스 쿼리

```typescript
// database-tool.ts
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

// 도구 정의
const tools: Anthropic.Tool[] = [
  {
    name: "query_database",
    description:
      "Execute a read-only SQL query against the application database. " +
      "Only SELECT queries are allowed. Returns results as JSON rows.",
    input_schema: {
      type: "object" as const,
      properties: {
        query: {
          type: "string",
          description: "SQL SELECT query to execute",
        },
        database: {
          type: "string",
          enum: ["users", "orders", "products"],
          description: "Which database to query",
        },
      },
      required: ["query", "database"],
    },
  },
  {
    name: "get_table_schema",
    description: "Get the column names and types for a database table.",
    input_schema: {
      type: "object" as const,
      properties: {
        table_name: {
          type: "string",
          description: "Name of the table",
        },
      },
      required: ["table_name"],
    },
  },
];

// 모의(Mock) 도구 구현
function queryDatabase(query: string, database: string): object {
  // 데이터베이스 결과를 시뮬레이션합니다
  if (query.toLowerCase().includes("count")) {
    return { rows: [{ count: 1247 }], rowCount: 1 };
  }
  return {
    rows: [
      { id: 1, name: "Alice", email: "alice@example.com" },
      { id: 2, name: "Bob", email: "bob@example.com" },
    ],
    rowCount: 2,
  };
}

function getTableSchema(tableName: string): object {
  const schemas: Record<string, object> = {
    users: {
      columns: [
        { name: "id", type: "INTEGER", nullable: false },
        { name: "name", type: "TEXT", nullable: false },
        { name: "email", type: "TEXT", nullable: false },
        { name: "created_at", type: "TIMESTAMP", nullable: false },
      ],
    },
  };
  return schemas[tableName] || { error: `Table '${tableName}' not found` };
}

// 도구 호출 실행
function executeTool(name: string, input: Record<string, string>): string {
  switch (name) {
    case "query_database":
      return JSON.stringify(queryDatabase(input.query, input.database));
    case "get_table_schema":
      return JSON.stringify(getTableSchema(input.table_name));
    default:
      return JSON.stringify({ error: `Unknown tool: ${name}` });
  }
}

// 메인 도구 사용 루프
async function chatWithTools(userMessage: string): Promise<string> {
  const messages: Anthropic.MessageParam[] = [
    { role: "user", content: userMessage },
  ];

  let response = await client.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 2048,
    tools,
    messages,
  });

  // 도구 사용 루프 처리
  while (response.stop_reason === "tool_use") {
    const toolResults: Anthropic.ToolResultBlockParam[] = [];

    for (const block of response.content) {
      if (block.type === "tool_use") {
        console.log(`  [Tool: ${block.name}(${JSON.stringify(block.input)})]`);
        const result = executeTool(
          block.name,
          block.input as Record<string, string>
        );
        toolResults.push({
          type: "tool_result",
          tool_use_id: block.id,
          content: result,
        });
      }
    }

    messages.push({ role: "assistant", content: response.content });
    messages.push({ role: "user", content: toolResults });

    response = await client.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 2048,
      tools,
      messages,
    });
  }

  // 최종 텍스트 추출
  return response.content
    .filter((b): b is Anthropic.TextBlock => b.type === "text")
    .map((b) => b.text)
    .join("");
}

// 사용 예시
async function main() {
  const answer = await chatWithTools(
    "How many users do we have? Also, what columns does the users table have?"
  );
  console.log(`\nClaude: ${answer}`);
}

main();
```

---

## 6. 다중 도구

단일 요청에 여러 도구를 제공할 수 있습니다. Claude는 사용자 메시지를 기반으로 어떤 도구를 호출할지 선택합니다:

```python
tools = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"],
        },
    },
    {
        "name": "get_stock_price",
        "description": "Get the current stock price for a ticker symbol.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g., 'AAPL'"
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression. Supports +, -, *, /, **, sqrt, sin, cos, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate, e.g., '(42 * 3.14) / 2'"
                }
            },
            "required": ["expression"],
        },
    },
]
```

Claude는 적절한 도구를 지능적으로 선택합니다:
- "파리의 날씨는 어때요?" -> `get_weather` 호출
- "애플 주가는 얼마예요?" -> `get_stock_price` 호출
- "250의 15%는 얼마예요?" -> `calculate` 호출
- "도쿄와 런던의 날씨를 비교해줘" -> `get_weather` 두 번 호출 (병렬)

---

## 7. 병렬 도구 실행

Claude는 단일 응답에서 여러 도구 호출을 요청할 수 있습니다. 그럴 때 모든 `tool_use` 블록이 동일한 `content` 배열에 나타납니다:

```python
# Claude의 응답은 여러 tool_use 블록을 포함할 수 있습니다:
# response.content = [
#   TextBlock(type="text", text="두 도시를 모두 확인해 보겠습니다..."),
#   ToolUseBlock(type="tool_use", id="toolu_01A", name="get_weather", input={"city": "Tokyo"}),
#   ToolUseBlock(type="tool_use", id="toolu_01B", name="get_weather", input={"city": "London"}),
# ]

def handle_parallel_tools(response, execute_fn):
    """여러 tool_use 블록을 포함할 수 있는 응답을 처리합니다."""
    tool_results = []

    for block in response.content:
        if block.type == "tool_use":
            # 각 도구를 실행합니다
            result = execute_fn(block.name, block.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,       # tool_use id와 일치해야 함
                "content": json.dumps(result),
            })

    return tool_results
```

**중요:** 병렬 도구 호출에 대한 결과를 반환할 때, 모든 `tool_use` 블록에 대해 `tool_use_id`로 일치시킨 `tool_result`를 포함해야 합니다. 하나를 건너뛰면 API가 오류를 반환합니다.

진정한 병렬 실행(예: 여러 API를 동시에 호출)의 경우:

```python
import asyncio
import anthropic

async def execute_tools_parallel(tool_calls: list) -> list:
    """여러 도구 호출을 동시에 실행합니다."""
    async def execute_one(call):
        # 도구를 실행합니다 (비동기 I/O를 시뮬레이션)
        if call.name == "get_weather":
            result = await async_get_weather(**call.input)
        elif call.name == "get_stock_price":
            result = await async_get_stock(**call.input)
        else:
            result = {"error": f"Unknown tool: {call.name}"}

        return {
            "type": "tool_result",
            "tool_use_id": call.id,
            "content": json.dumps(result),
        }

    # 모든 도구 호출을 동시에 실행합니다
    results = await asyncio.gather(
        *[execute_one(call) for call in tool_calls]
    )
    return list(results)
```

---

## 8. 도구 사용 제어: tool_choice

`tool_choice` 매개변수는 Claude가 도구를 사용하는 방식을 제어합니다:

```python
# auto (기본값): Claude가 도구 사용 여부를 결정
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=tools,
    tool_choice={"type": "auto"},     # Claude가 결정
    messages=[{"role": "user", "content": "Hello!"}],
)

# any: Claude가 최소 하나의 도구를 사용하도록 강제
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=tools,
    tool_choice={"type": "any"},      # 도구를 사용해야 함
    messages=[{"role": "user", "content": "What's 2+2?"}],
)

# 특정 도구: Claude가 특정 도구를 사용하도록 강제
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=tools,
    tool_choice={"type": "tool", "name": "calculate"},  # 'calculate'를 사용해야 함
    messages=[{"role": "user", "content": "What's 2+2?"}],
)
```

**각 모드를 언제 사용하는가:**

| 모드 | 사용 사례 |
|------|-----------|
| `auto` | 도구가 선택적인 일반 대화 |
| `any` | 항상 구조화된 출력이나 액션을 원할 때 |
| `tool` (특정) | 특정 도구를 호출해야 할 때 (예: 구조화된 추출) |

### 구조화된 출력을 위한 tool_choice 사용

강력한 패턴: 원하는 출력 형식과 일치하는 스키마의 도구를 정의한 다음, Claude가 이를 호출하도록 강제합니다:

```python
# tool_choice를 사용하여 구조화된 출력 강제
tools = [
    {
        "name": "extract_contact_info",
        "description": "Extract structured contact information from text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string", "format": "email"},
                "phone": {"type": "string"},
                "company": {"type": "string"},
                "role": {"type": "string"},
            },
            "required": ["name"],
        },
    }
]

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=tools,
    tool_choice={"type": "tool", "name": "extract_contact_info"},
    messages=[
        {
            "role": "user",
            "content": "Extract contact info from: Hi, I'm Jane Smith, "
                       "CTO at TechCorp. Reach me at jane@techcorp.io "
                       "or 555-0123.",
        }
    ],
)

# Claude가 도구를 통해 구조화된 데이터를 반환하도록 강제됩니다
for block in response.content:
    if block.type == "tool_use":
        print(json.dumps(block.input, indent=2))
        # {
        #   "name": "Jane Smith",
        #   "email": "jane@techcorp.io",
        #   "phone": "555-0123",
        #   "company": "TechCorp",
        #   "role": "CTO"
        # }
```

---

## 9. 도구에서의 오류 처리

도구 실행이 실패하면 충돌 대신 `tool_result`에 오류를 반환합니다:

```python
def execute_tool(name: str, input_data: dict) -> dict:
    """도구를 실행하고 오류를 정상적으로 처리합니다."""
    try:
        if name == "get_weather":
            return get_weather(**input_data)
        elif name == "query_database":
            return query_database(**input_data)
        else:
            return {"error": f"Unknown tool: {name}"}

    except ValueError as e:
        return {"error": f"Invalid input: {str(e)}"}
    except TimeoutError:
        return {"error": "Operation timed out. Please try again."}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

# tool_results를 구성할 때 실패에 is_error를 설정합니다
def build_tool_result(tool_use_id: str, result: dict) -> dict:
    """오류를 표시하는 tool_result 메시지를 구성합니다."""
    content = json.dumps(result)
    is_error = "error" in result

    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
        "is_error": is_error,   # Claude에게 도구가 실패했음을 알립니다
    }
```

`is_error` 플래그는 중요합니다: 이는 Claude에게 도구 호출이 실패했음을 알려, Claude가 오류 텍스트를 유효한 데이터로 처리하는 대신 사용자에게 알리거나 다른 방법을 시도할 수 있게 합니다.

---

## 10. 고급 패턴

### 10.1 도구 호출 체이닝

때로는 Claude가 한 도구의 출력을 다음 도구의 입력으로 사용하며 여러 도구를 순차적으로 호출해야 할 수 있습니다:

```
사용자: "가장 많은 사용자가 있는 도시의 날씨는 어때요?"

Claude → get_top_city() 호출
     ← 결과: {"city": "Tokyo", "users": 5420}
Claude → get_weather(city="Tokyo") 호출
     ← 결과: {"temp": 22, "condition": "Sunny"}
Claude: "가장 인기 있는 도시는 도쿄입니다 (사용자 5,420명).
         현재 그곳은 22°C이고 맑습니다."
```

이것은 자동으로 발생합니다 -- 도구 사용 루프(`while response.stop_reason == "tool_use"` 패턴)가 임의 길이의 체인을 처리합니다.

### 10.2 이미지를 반환하는 도구

도구는 Claude에게 시각적 분석을 위한 이미지를 반환할 수 있습니다:

```python
import base64

def take_screenshot() -> dict:
    """스크린샷을 캡처하고 base64로 반환합니다."""
    # ... 스크린샷 캡처 ...
    with open("screenshot.png", "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": image_data,
        },
    }

# tool_result에 이미지 콘텐츠 반환
tool_result = {
    "type": "tool_result",
    "tool_use_id": block.id,
    "content": [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": image_data,
            },
        },
        {
            "type": "text",
            "text": "Screenshot captured at 2026-02-22 10:30:00",
        },
    ],
}
```

### 10.3 도구 사용과 스트리밍

도구 사용은 스트리밍과 함께 동작하지만 신중한 이벤트 처리가 필요합니다:

```python
import anthropic

client = anthropic.Anthropic()

def stream_with_tools(user_message: str, tools: list):
    """도구 사용을 처리하면서 응답을 스트리밍합니다."""
    messages = [{"role": "user", "content": user_message}]

    while True:
        # API 호출에 스트림을 사용합니다
        with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            tools=tools,
            messages=messages,
        ) as stream:
            # 텍스트 토큰이 도착하는 대로 출력합니다
            for text in stream.text_stream:
                print(text, end="", flush=True)

            # 도구 사용 확인을 위해 최종 메시지를 가져옵니다
            response = stream.get_final_message()

        # 도구 사용이 없으면 완료입니다
        if response.stop_reason != "tool_use":
            print()  # 최종 줄바꿈
            break

        # 도구 호출 처리
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"\n  [Calling {block.name}...]")
                result = execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })

        # 대화에 추가하고 계속합니다
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})
```

### 10.4 도구 실행 없이 구조화된 출력

실제로 아무것도 실행하지 않고 구조화된 출력 추출만을 위해 도구를 사용할 수 있습니다:

```python
# 구조화된 출력을 위한 스키마인 "도구" 정의
sentiment_tool = {
    "name": "classify_sentiment",
    "description": "Classify the sentiment of a text.",
    "input_schema": {
        "type": "object",
        "properties": {
            "sentiment": {
                "type": "string",
                "enum": ["positive", "negative", "neutral", "mixed"],
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
            "key_phrases": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Key phrases that influenced the classification",
            },
        },
        "required": ["sentiment", "confidence", "key_phrases"],
    },
}

# Claude가 이 도구를 사용하도록 강제합니다
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=[sentiment_tool],
    tool_choice={"type": "tool", "name": "classify_sentiment"},
    messages=[
        {"role": "user", "content": "Analyze: 'The product is great but shipping was terrible.'"}
    ],
)

# 구조화된 데이터 추출 -- 도구를 "실행"할 필요 없음
for block in response.content:
    if block.type == "tool_use":
        result = block.input
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Key phrases: {result['key_phrases']}")
```

---

## 11. 모범 사례

### 11.1 도구 설계 원칙

```
┌─────────────────────────────────────────────────────────────────┐
│                도구 설계 모범 사례                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. 단일 책임                                                    │
│     각 도구는 하나의 일을 잘 수행합니다.                          │
│     나쁨:  "manage_database" (너무 광범위)                       │
│     좋음: "query_users", "insert_order", "get_schema"           │
│                                                                  │
│  2. 명확한 설명                                                  │
│     사람 동료에게 설명하듯이 작성합니다.                          │
│     포함: 무엇을 하는지, 무엇을 반환하는지, 언제 사용하는지.     │
│                                                                  │
│  3. 설명적인 매개변수                                            │
│     모든 매개변수에 설명이 필요합니다.                           │
│     고정된 유효 값 집합에는 enum을 사용합니다.                   │
│     숫자 매개변수에는 제약 조건(min/max)을 사용합니다.           │
│                                                                  │
│  4. 기본적으로 안전하게                                          │
│     읽기 전용 도구에는 쓰기 부수 효과가 없어야 합니다.           │
│     파괴적 도구는 명시적인 확인을 요구해야 합니다.               │
│     실행 전에 입력을 검증합니다.                                 │
│                                                                  │
│  5. 유익한 오류                                                  │
│     실행 가능한 오류 메시지를 반환합니다.                        │
│     Claude가 호출 실패를 알 수 있도록 is_error 플래그를 사용합니다.│
│     나쁨:  {"error": "failed"}                                  │
│     좋음: {"error": "'Atlantis' 도시를 찾을 수 없습니다.        │
│            사용 가능한 도시: London, Tokyo, New York, Seoul."}   │
│                                                                  │
│  6. 간결한 결과                                                  │
│     Claude에게 필요한 것만 반환합니다. 큰 페이로드는 토큰을 낭비합니다.│
│     긴 결과는 페이지네이션합니다.                                │
│     원시 데이터를 덤프하는 대신 요약합니다.                      │
│                                                                  │
│  7. 도구 수 최소화                                               │
│     도구가 많을수록 = 토큰이 많을수록 = 비용과 지연이 높아집니다.│
│     매개변수를 공유하는 관련 도구를 결합합니다.                  │
│     거의 사용되지 않는 도구는 제거합니다.                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 11.2 일반적인 함정

| 함정 | 문제 | 해결책 |
|------|------|--------|
| 모호한 설명 | Claude가 잘못된 도구 또는 잘못된 매개변수 호출 | 상세하고 구체적인 설명 작성 |
| 오류 처리 없음 | 도구 실패 시 애플리케이션 충돌 | 모든 도구 실행을 try/except로 감싸기 |
| tool_result 누락 | 도구 결과가 불완전할 때 API 오류 | 항상 tool_use 블록당 하나의 결과 반환 |
| 너무 많은 도구 | 높은 토큰 사용량, 느린 응답 | 10-20개로 제한; 하위 집합에 tool_choice 사용 |
| 민감한 작업 | Claude가 파괴적 도구를 호출할 수 있음 | 쓰기/삭제 작업에 확인 단계 추가 |
| 무제한 출력 | 도구가 10,000개 행 반환 | 결과를 페이지네이션, 제한, 또는 요약 |

---

## 12. 연습 문제

### 연습 1: 계산기 도구 (초급)

수학 표현식을 안전하게 평가하는 `calculate` 도구를 사용한 도구 사용 애플리케이션을 구축하십시오. 다음으로 테스트하십시오: "340의 15%는 얼마예요?", "144의 제곱근은 얼마예요?", "$1000을 연 5% 이자로 10년 동안 투자하면 최종 금액은 얼마예요?"

### 연습 2: 다중 도구 어시스턴트 (중급)

세 가지 도구를 갖춘 어시스턴트를 만드십시오:
1. `search_web` -- 웹 검색을 시뮬레이션합니다 (모의 결과 반환)
2. `get_current_time` -- 현재 날짜와 시간을 반환합니다
3. `take_notes` -- 목록에 메모를 저장합니다

다음으로 테스트하십시오: "최신 Python 릴리스를 검색하고, 버전 번호를 메모하고, 현재 시간을 알려주세요." Claude가 도구를 올바르게 체이닝하는지 확인하십시오.

### 연습 3: 구조화된 데이터 추출 (중급)

`tool_choice` 강제 도구 패턴을 사용하여 비구조화 텍스트에서 구조화된 데이터를 추출하십시오. 5개의 채용 공고 설명 목록을 처리하고 다음을 추출하십시오: 직함, 회사, 위치, 급여 범위, 필수 기술, 경험 수준. 결과를 JSON 배열로 출력하십시오.

### 연습 4: 대화형 데이터베이스 에이전트 (고급)

SQLite 데이터베이스를 쿼리할 수 있는 대화형 에이전트를 구축하십시오. 세 가지 도구를 제공하십시오: `list_tables`, `describe_table`, `run_query`. 완전한 오류 처리, 입력 유효성 검사(SELECT가 아닌 쿼리 차단), 결과 페이지네이션을 구현하십시오. "지난달 모든 주문을 보여줘"와 "어떤 고객이 가장 많이 지출했나요?"와 같은 자연어 질문으로 테스트하십시오.

### 연습 5: 스트리밍을 사용한 도구 사용 (고급)

연습 4를 스트리밍으로 확장하십시오. Claude의 생각을 실시간으로 표시하고, 도구 실행 중에 스피너를 표시하고, 토큰 사용 통계와 함께 최종 답변을 제시하십시오. 엣지 케이스를 처리하십시오: 도구가 타임아웃되면 어떻게 됩니까? Claude가 존재하지 않는 도구를 요청하면 어떻게 됩니까?

---

## 13. 참고 자료

- 도구 사용 문서 - https://docs.anthropic.com/en/docs/build-with-claude/tool-use
- Anthropic Cookbook: 도구 사용 - https://github.com/anthropics/anthropic-cookbook/tree/main/tool_use
- JSON Schema 참조 - https://json-schema.org/understanding-json-schema/
- Anthropic Python SDK - https://github.com/anthropics/anthropic-sdk-python
- Anthropic TypeScript SDK - https://github.com/anthropics/anthropic-sdk-typescript

---

## 다음 레슨

[17. Claude Agent SDK](./17_Claude_Agent_SDK.md)에서는 Agent SDK를 소개합니다 -- 도구 사용 루프, 컨텍스트 관리, 에이전트 오케스트레이션을 프로그래밍 가능한 인터페이스로 감싸는 고수준 추상화입니다. 다단계 문제를 자율적으로 해결하는 에이전트를 만드는 방법을 배우게 됩니다.
