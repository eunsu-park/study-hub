# 23. 함수 호출과 도구 사용

## 학습 목표

- OpenAI 함수 호출과 Anthropic 도구 사용 API 숙달
- 적절한 설명이 포함된 효과적인 도구 정의 스키마 설계
- 병렬 도구 호출 및 도구 선택 전략 구현
- 커스텀 도구 구축 및 Model Context Protocol (MCP) 통합
- 에러의 우아한 처리 및 복잡한 멀티 도구 워크플로우 오케스트레이션

---

## 이론과 원리

Function calling("도구 사용"으로도 불림)은 LLM이 그저 텍스트를 생성하는 대신 **외부 코드를 호출하기로 결정**하는 프로토콜입니다. 그것 없이 LLM은 학습 데이터에 제한되고 세상과 상호작용할 수 없습니다. 그것과 함께 LLM은 *에이전트*가 됩니다 — 웹을 검색하고, 데이터베이스를 쿼리하고, 계산을 실행하고, 이메일을 보내고, API를 제어할 수 있는 시스템. 프로토콜 자체는 단순합니다 — JSON 스키마 기반 — 그러나 그 주위의 설계 공간(도구 설명 품질, 병렬 호출, 선택 전략, 오류 처리)은 큽니다.

이 섹션은 다음을 다룹니다:

- **(A) 프로토콜** — LLM과 호스트 사이에 흐르는 메시지, 루프 구조.
- **(B) 계약으로서의 JSON Schema** — 함수 설명이 어떻게 제약 생성 문법이 되는가.
- **(C) 도구 선택 전략** — auto, required, none, 명명된 도구; 트레이드오프.
- **(D) 병렬 도구 호출** — 여러 동시 호출, 의존성 분석.
- **(E) 제공자 변형** — OpenAI vs Anthropic vs Gemini, 무엇이 이식 가능하고 무엇이 아닌가.
- **(F) 도구 설명 품질** — 함수 코드를 잘 쓰는 것보다 설명을 잘 쓰는 것이 더 중요한 이유.
- **(G) MCP (Model Context Protocol)** — 도구 서버를 위한 Anthropic의 신흥 표준.
- **(H) 오류 처리** — 모델이 회복할 수 있도록 도구 실패를 모델에 다시 전달하는 방법.

### A. 프로토콜

LLM과 호스트 사이의 대화는 구조화된 메시지를 가집니다:

```
1. User → Host:  "What's the weather in Tokyo?"
2. Host → LLM:   사용자 메시지 + 도구 정의
3. LLM → Host:   tool_calls = [{id: "1", name: "get_weather", args: {city: "Tokyo"}}]
4. Host → tool:  get_weather("Tokyo") 실행
5. tool → Host:  {"temp": 22, "condition": "cloudy"}
6. Host → LLM:   이전 + tool_result 메시지 {id: "1", content: "..."}
7. LLM → Host:   "It's 22°C and cloudy in Tokyo."
8. Host → User:  "It's 22°C and cloudy in Tokyo."
```

3-6단계가 **도구 루프**를 형성합니다. LLM이 단일 대화에서 여러 번 도구 호출을 방출할 수 있습니다; 각 도구 결과가 다음 LLM 호출의 새 맥락으로 다시 공급됩니다. 종료 — LLM이 도구 호출 없이 평범한 텍스트 응답을 방출.

### B. 계약으로서의 JSON Schema

각 사용 가능한 도구는 그 파라미터에 대한 JSON 스키마로 설명됩니다:

```json
{
  "name": "get_weather",
  "description": "Get current weather for a city.",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {"type": "string", "description": "City name, e.g., 'Tokyo'"},
      "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"}
    },
    "required": ["city"]
  }
}
```

두 목적을 수행:
- **LLM에 대해** — 어떤 도구가 존재하고 어떤 인자가 유효한지 알려줍니다. 모델 파인튜닝이 JSON 스키마 관습을 내재화했으므로 모델이 스키마를 어떻게 해석할지 이해합니다.
- **제약 디코딩에 대해** (strict 모드 API에서) — 스키마가 모델이 무효한 도구 호출을 방출하는 것을 방지하는 토큰 마스크로 컴파일됩니다.

Strict 모드 보장은 중요합니다 — strict 모드 켜져 있으면 API가 잘못된 인자를 가진 도구 호출을 *절대* 반환하지 않습니다. 모델이 여전히 잘못된 도구를 고르거나 값을 환각할 수 있지만 구조는 강제됩니다.

### C. 도구 선택 전략

LLM이 도구 사용 여부를 결정하는 방법:

- **`auto`** (기본) — LLM이 메시지별로 도구를 호출할지 직접 응답할지 결정. 챗봇의 표준.
- **`required`** / **`any`** — LLM이 적어도 하나의 도구를 호출해야 함. 응답이 외부 데이터를 요구함을 알 때 유용.
- **`none`** — LLM이 도구를 호출할 수 없음. 최종 텍스트 응답을 강제하고 싶을 때 유용.
- **`{"name": "tool_name"}`** — LLM이 이 특정 도구를 호출해야 함. 강제 추출에 유용(레슨 22, 구조화된 출력으로서의 function-calling API 사용).

전략 선택은 호출별, 대화별이 아닙니다. 전형적 에이전트 흐름은 처음에 `auto`, 그 후 최종 요약을 강제하기 위해 `none` 사용.

### D. 병렬 도구 호출

현대 API(GPT-4 Turbo+, Claude 3.5+)는 단일 응답에서 *여러* 도구 호출을 반환할 수 있습니다:

```json
{
  "tool_calls": [
    {"id": "1", "name": "get_weather", "args": {"city": "Tokyo"}},
    {"id": "2", "name": "get_weather", "args": {"city": "Paris"}}
  ]
}
```

호스트가 둘을 병렬로 실행, 다음 사용자 메시지에서 둘 결과를 반환. 두 이점:
- **지연** — 직렬 대신 병렬 I/O.
- **LLM 효율** — 두 라운드트립 대신 하나.

모델은 호출이 독립적인지(Tokyo와 Paris 날씨는 병렬 가능) 의존적인지(검색 → 첫 결과 클릭은 직렬이어야 함) 식별하도록 학습됩니다. 품질이 다양 — 모델이 대부분 옳지만 특정 도구에서 검증이라고 가정.

### E. 제공자 변형

세 큰 제공자가 function calling을 약간 다르게 구현했습니다:

- **OpenAI** — 함수 정의의 `tools` 배열; 응답의 `tool_calls`. 2024년부터 strict 모드 사용 가능.
- **Anthropic** — OpenAI와 비슷한 `tools` 배열; 응답이 `tool_use` 콘텐츠 블록 사용. 도구 결과가 `tool_result` 콘텐츠 블록으로 다시 공급.
- **Google Gemini** — 비슷한 `tools` 정의; 응답에 `function_call` 사용. 약간 다른 메시지 형식.

LangChain, instructor, `litellm` 같은 래퍼가 이 차이들을 추상화. 네이티브 코드에는 한 제공자를 골라 그 형식 고수 — 교차 제공자는 가능하지만 복잡성을 더함.

### F. 도구 설명 품질

function-calling 시스템에서 가장 중요한 코드는 각 도구의 **문자열 설명**입니다. LLM이 설명을 보고 그것에 기반하여 어떤 도구를 호출할지 결정합니다. 거기의 품질이 모든 곳의 품질을 직접 추진합니다.

모범 사례:
- **도구가 무엇을 하는지 명세하라, 어떻게 작동하는지가 아니라.** "Get current weather"이지 "Call the weatherapi.com /v1/current.json endpoint"가 아님.
- **언제 사용할지 명세.** "Use when the user asks about current weather conditions."
- **언제 사용하지 말아야 할지 명세.** "Do not use for weather forecasts; use `get_forecast` instead."
- **예시 입력/출력 보여주기.** 설명에 한 줄의 예시 사용이 모델 선택을 상당히 향상시킵니다.
- **인자 형식에 구체적이기.** "City name in English, e.g., 'Tokyo' (not 'JP')."

모호한 설명을 가진 도구는 잘못 호출되고, 정확한 설명을 가진 도구는 적절할 때 호출되고 그렇지 않으면 건너뜁니다. 이것이 레버리지 — 도구 설명에 30분 보낸 것이 프롬프트 엔지니어링 시간을 능가할 수 있습니다.

### G. MCP (Model Context Protocol)

Anthropic의 MCP(2024)는 **도구 서버**를 위한 신흥 표준 — 표준 프로토콜을 통해 LLM에 도구를 노출하는 외부 프로세스. 도구 구현을 애플리케이션에 묶는 대신, 그것들을 제공하는 MCP 서버에 연결합니다.

이점:
- **재사용성** — 도구를 한 번 작성("GitHub 저장소 검색"), 어떤 MCP 호환 클라이언트에서든 사용.
- **관심사 분리** — 도구 구현이 애플리케이션 코드가 아닌 서버에 살음.
- **보안** — 서버가 자체 인증과 속도 제한을 강제할 수 있음; LLM 클라이언트가 비밀을 보지 못함.

코드 편집기의 LSP(Language Server Protocol)와 개념적으로 비슷. 초기지만 추진력을 얻고 있음.

### H. 오류 처리

도구 호출이 실패합니다. 옳은 패턴 — **오류를 잡아서 도구 결과로 다시 공급**, LLM이 반응할 수 있도록.

```python
try:
    result = execute_tool(name, args)
except Exception as e:
    result = {"error": str(e)}
# `result`를 tool_result 메시지로 다시 공급
```

LLM이 오류 메시지를 보고 보통 수정된 인자로 재시도하거나, 다른 도구로 전환하거나, 사용자에게 명확화를 요청합니다. 예외를 사용자에게 던지는 것은 루프를 깨고 더 나쁜 경험을 만듭니다.

복구 불가능한 오류(인증 실패, 청구 문제)에는 사용자 대상 메시지로 루프 종료를 원할 수 있습니다. 복구 가능한 오류(타임아웃, 속도 제한, 잘못된 인자)에는 LLM에 다시 공급이 옳은 움직임.

### 이론에서 아래 함수들로

- §1 (개요) — §A 프로토콜과 §F 도구 설명 중요성을 틀.
- §2 (OpenAI) — OpenAI API로 §A-§E 구현.
- §3 (Anthropic) — Anthropic의 도구 사용 블록(§E 변형)으로 동일 구현.
- §4 (병렬 호출) — 의존성 추론과 함께 §D 병렬 도구 호출.
- §5 (커스텀 도구) — §B JSON 스키마 작성 + §F 설명 모범 사례.
- §6 (MCP) — §G 프로토콜과 예시 서버.
- §7 (오케스트레이션 패턴) — §H 오류 처리, 재시도, 레슨 14 에이전트 루프 위에 세워진 멀티 도구 워크플로우.

---

## 1. 함수 호출 개요

### 함수 호출이 필요한 이유

> **함수 호출 / 도구 사용**
>
> - **브릿지**: LLM 추론을 실제 동작(API, 데이터베이스, 코드 실행)에 연결
> - **구조화된 출력**: LLM이 스키마에 맞는 유효한 JSON을 생성하도록 강제
> - **신뢰성**: 자유형 텍스트의 정규식 파싱을 제거
> - **조합 가능성**: 도구 호출을 체이닝하여 복잡한 워크플로우 구축
> - **안전성**: 애플리케이션이 실행을 제어하고 LLM은 호출할 것만 결정

### 프로바이더 비교

| 기능 | OpenAI | Anthropic | Google (Gemini) |
|------|--------|-----------|-----------------|
| API 이름 | Function Calling / Tools | Tool Use | Function Calling |
| 최대 도구 수 | 128 | 64 | 128 |
| 병렬 호출 | 지원 | 지원 | 지원 |
| 스트리밍 | 지원 | 지원 | 지원 |
| 강제 도구 사용 | `tool_choice: required` | `tool_choice: {"type": "tool", ...}` | `tool_config` |
| 중지 사유 | `tool_calls` | `tool_use` | `FUNCTION_CALL` |
| Strict 스키마 | 지원 (structured outputs) | 미지원 | 미지원 |

---

## 2. OpenAI 함수 호출 API

### 기본 함수 호출

```python
from openai import OpenAI
import json

client = OpenAI()

# 도구 정의
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city. Use this when the user asks about weather conditions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, e.g., 'San Francisco' or 'Tokyo'",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                        "default": "celsius",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "Search for available flights between two cities on a given date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "Departure city or airport code"},
                    "destination": {"type": "string", "description": "Arrival city or airport code"},
                    "date": {"type": "string", "description": "Travel date in YYYY-MM-DD format"},
                    "passengers": {"type": "integer", "minimum": 1, "maximum": 9, "default": 1},
                },
                "required": ["origin", "destination", "date"],
            },
        },
    },
]

# 도구 구현
def get_weather(city: str, unit: str = "celsius") -> dict:
    """시뮬레이션된 날씨 API."""
    return {
        "city": city,
        "temperature": 22 if unit == "celsius" else 72,
        "unit": unit,
        "condition": "partly cloudy",
        "humidity": 65,
    }

def search_flights(origin: str, destination: str, date: str,
                   passengers: int = 1) -> dict:
    """시뮬레이션된 항공편 검색 API."""
    return {
        "flights": [
            {"airline": "United", "departure": "08:30", "arrival": "11:45",
             "price": 320 * passengers},
            {"airline": "Delta", "departure": "14:15", "arrival": "17:30",
             "price": 285 * passengers},
        ],
        "origin": origin,
        "destination": destination,
        "date": date,
    }

TOOL_REGISTRY = {
    "get_weather": get_weather,
    "search_flights": search_flights,
}

def chat_with_tools(user_message: str) -> str:
    """완전한 함수 호출 루프."""
    messages = [
        {"role": "system", "content": "You are a helpful travel assistant."},
        {"role": "user", "content": user_message},
    ]

    # 단계 1: 도구와 함께 메시지 전송
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    message = response.choices[0].message

    # 단계 2: 모델이 도구를 호출하려는지 확인
    if message.tool_calls:
        messages.append(message)  # 도구 호출이 포함된 어시스턴트 메시지 추가

        # 단계 3: 각 도구 호출 실행
        for tool_call in message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)

            print(f"호출 중: {func_name}({func_args})")

            # 함수 실행
            if func_name in TOOL_REGISTRY:
                result = TOOL_REGISTRY[func_name](**func_args)
            else:
                result = {"error": f"Unknown function: {func_name}"}

            # 단계 4: 도구 결과를 다시 전송
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })

        # 단계 5: 최종 응답 받기
        final_response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
        )
        return final_response.choices[0].message.content

    return message.content

# 사용 예시
print(chat_with_tools("What's the weather in Tokyo and find me flights from NYC to Tokyo on 2026-04-15?"))
```

### 도구 선택 전략

```python
# AUTO: 모델이 도구 사용 여부를 결정 (기본값)
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    tool_choice="auto",
)

# REQUIRED: 모델이 반드시 최소 하나의 도구를 호출
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    tool_choice="required",
)

# SPECIFIC: 특정 도구를 강제
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "get_weather"}},
)

# NONE: 이 요청에서 도구 사용을 비활성화
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    tool_choice="none",
)
```

---

## 3. Anthropic 도구 사용

### Claude Tool Use API

```python
from anthropic import Anthropic

anthropic = Anthropic()

# Anthropic 도구 정의
tools = [
    {
        "name": "get_stock_price",
        "description": (
            "Get the current stock price for a given ticker symbol. "
            "Returns the latest price, change, and volume."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g., 'AAPL', 'GOOGL'",
                },
                "include_history": {
                    "type": "boolean",
                    "description": "Include 5-day price history",
                    "default": False,
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "calculate_portfolio_value",
        "description": "Calculate the total value of a stock portfolio given holdings.",
        "input_schema": {
            "type": "object",
            "properties": {
                "holdings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string"},
                            "shares": {"type": "number"},
                        },
                        "required": ["ticker", "shares"],
                    },
                },
            },
            "required": ["holdings"],
        },
    },
]

def get_stock_price(ticker: str, include_history: bool = False) -> dict:
    """시뮬레이션된 주식 API."""
    prices = {"AAPL": 198.50, "GOOGL": 175.30, "MSFT": 425.80, "NVDA": 890.10}
    price = prices.get(ticker.upper(), 100.00)
    result = {
        "ticker": ticker.upper(),
        "price": price,
        "change": 2.35,
        "change_percent": 1.2,
        "volume": 45_000_000,
    }
    if include_history:
        result["history"] = [price - i * 1.5 for i in range(5, 0, -1)]
    return result

def calculate_portfolio_value(holdings: list[dict]) -> dict:
    """포트폴리오 가치 계산."""
    total = 0
    details = []
    for h in holdings:
        price_data = get_stock_price(h["ticker"])
        value = price_data["price"] * h["shares"]
        total += value
        details.append({
            "ticker": h["ticker"],
            "shares": h["shares"],
            "price": price_data["price"],
            "value": round(value, 2),
        })
    return {"total_value": round(total, 2), "holdings": details}

TOOL_REGISTRY = {
    "get_stock_price": get_stock_price,
    "calculate_portfolio_value": calculate_portfolio_value,
}

def chat_with_claude_tools(user_message: str) -> str:
    """완전한 Anthropic 도구 사용 루프."""
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = anthropic.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            tools=tools,
            messages=messages,
        )

        # 모델이 도구를 사용하려는지 확인
        if response.stop_reason == "tool_use":
            # 어시스턴트 응답 추가 (tool_use 블록 포함)
            messages.append({"role": "assistant", "content": response.content})

            # 각 도구 호출을 실행하고 결과 수집
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    func_name = block.name
                    func_args = block.input

                    print(f"호출 중: {func_name}({json.dumps(func_args)})")

                    try:
                        result = TOOL_REGISTRY[func_name](**func_args)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                        })
                    except Exception as e:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps({"error": str(e)}),
                            "is_error": True,
                        })

            messages.append({"role": "user", "content": tool_results})

        elif response.stop_reason == "end_turn":
            # 최종 텍스트 응답 추출
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return "No text response generated."
        else:
            return f"Unexpected stop reason: {response.stop_reason}"

# 사용 예시
result = chat_with_claude_tools(
    "What's the value of my portfolio? I have 100 shares of AAPL, "
    "50 shares of GOOGL, and 25 shares of NVDA."
)
print(result)
```

---

## 4. 병렬 도구 호출

### OpenAI 병렬 호출

```python
def handle_parallel_tool_calls(user_message: str) -> str:
    """단일 턴에서 여러 도구 호출을 처리."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use tools when needed."},
        {"role": "user", "content": user_message},
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        parallel_tool_calls=True,  # 병렬 활성화 (대부분의 모델에서 기본값)
    )

    message = response.choices[0].message

    if message.tool_calls:
        messages.append(message)

        # 모델이 한 턴에서 여러 도구 호출을 발행할 수 있음
        print(f"모델이 {len(message.tool_calls)}개의 병렬 도구 호출을 요청")

        # 모든 도구 호출 실행 (프로덕션에서는 병렬화 가능)
        for tool_call in message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)

            result = TOOL_REGISTRY.get(func_name, lambda **k: {"error": "unknown"})(**func_args)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })

        # 모든 도구 완료 후 최종 응답 받기
        final = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
        )
        return final.choices[0].message.content

    return message.content

# 모델이 get_weather를 두 번 병렬로 호출
result = handle_parallel_tool_calls(
    "Compare the weather in Tokyo and New York right now"
)
print(result)
```

### 비동기 병렬 실행

```python
import asyncio
from openai import AsyncOpenAI

async_client = AsyncOpenAI()

async def execute_tools_parallel(tool_calls: list) -> list[dict]:
    """여러 도구 호출을 동시에 실행."""
    async def run_tool(tool_call) -> dict:
        func_name = tool_call.function.name
        func_args = json.loads(tool_call.function.arguments)

        # 프로덕션: 비동기 API 호출, 비동기 DB 쿼리 등
        result = TOOL_REGISTRY.get(
            func_name, lambda **k: {"error": "unknown"}
        )(**func_args)

        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result),
        }

    # 모든 도구 호출을 동시에 실행
    results = await asyncio.gather(*[run_tool(tc) for tc in tool_calls])
    return list(results)

async def async_chat_with_tools(user_message: str) -> str:
    """병렬 도구 실행을 포함한 비동기 함수 호출."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_message},
    ]

    response = await async_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
    )
    message = response.choices[0].message

    if message.tool_calls:
        messages.append(message)

        # 모든 도구를 병렬로 실행
        tool_results = await execute_tools_parallel(message.tool_calls)
        messages.extend(tool_results)

        final = await async_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
        )
        return final.choices[0].message.content

    return message.content
```

---

## 5. 커스텀 도구 구축

### 도구 설계 패턴

```python
from typing import Any, Callable
from dataclasses import dataclass
from functools import wraps
import inspect

@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict
    handler: Callable

class ToolRegistry:
    """도구 관리 및 자동 생성을 위한 레지스트리."""

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def tool(self, description: str):
        """함수를 도구로 등록하는 데코레이터."""
        def decorator(func: Callable) -> Callable:
            # 타입 힌트에서 JSON 스키마를 자동 생성
            hints = func.__annotations__
            sig = inspect.signature(func)

            properties = {}
            required = []

            for param_name, param in sig.parameters.items():
                if param_name == "return":
                    continue

                hint = hints.get(param_name, str)
                prop = self._type_to_schema(hint)

                prop["description"] = f"Parameter: {param_name}"
                properties[param_name] = prop

                if param.default is inspect.Parameter.empty:
                    required.append(param_name)

            tool_def = ToolDefinition(
                name=func.__name__,
                description=description,
                parameters={
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
                handler=func,
            )
            self._tools[func.__name__] = tool_def

            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper

        return decorator

    def _type_to_schema(self, hint) -> dict:
        """Python 타입 힌트를 JSON Schema 타입으로 변환."""
        type_map = {
            str: {"type": "string"},
            int: {"type": "integer"},
            float: {"type": "number"},
            bool: {"type": "boolean"},
            list: {"type": "array", "items": {"type": "string"}},
        }
        return type_map.get(hint, {"type": "string"})

    def get_openai_tools(self) -> list[dict]:
        """OpenAI 형식으로 도구 내보내기."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]

    def get_anthropic_tools(self) -> list[dict]:
        """Anthropic 형식으로 도구 내보내기."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            }
            for t in self._tools.values()
        ]

    def execute(self, name: str, arguments: dict) -> Any:
        """이름으로 도구를 실행."""
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")
        return self._tools[name].handler(**arguments)

# 사용 예시
registry = ToolRegistry()

@registry.tool("Search a database of products by name or category")
def search_products(query: str, category: str, max_results: int) -> dict:
    return {"results": [{"name": f"Product matching '{query}'", "category": category}]}

@registry.tool("Create a new support ticket in the help desk system")
def create_ticket(title: str, description: str, priority: str) -> dict:
    return {"ticket_id": "TKT-12345", "status": "created"}

# 자동 생성된 도구 정의
print(json.dumps(registry.get_openai_tools(), indent=2))
```

### Pydantic으로 검증된 도구

```python
from pydantic import BaseModel, Field
from typing import Literal

class SearchParams(BaseModel):
    query: str = Field(min_length=1, max_length=500, description="Search query text")
    filters: dict[str, str] = Field(
        default_factory=dict,
        description="Key-value filters to narrow results",
    )
    sort_by: Literal["relevance", "date", "price"] = Field(
        default="relevance", description="Sort order for results"
    )
    limit: int = Field(default=10, ge=1, le=100, description="Max results to return")

def validated_tool(params_model: type[BaseModel]):
    """Pydantic 모델로 도구 입력을 검증하는 데코레이터."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(**kwargs):
            # 입력 검증 및 강제 변환
            try:
                validated = params_model(**kwargs)
                return func(**validated.model_dump())
            except Exception as e:
                return {"error": f"Validation failed: {str(e)}"}
        wrapper._params_model = params_model
        return wrapper
    return decorator

@validated_tool(SearchParams)
def search_knowledge_base(query: str, filters: dict, sort_by: str, limit: int) -> dict:
    """내부 지식 베이스 검색."""
    return {
        "results": [{"title": f"Result for: {query}", "score": 0.95}],
        "total": 1,
        "query": query,
        "sort_by": sort_by,
    }

# 검증이 도구에 도달하기 전에 잘못된 입력을 포착
result = search_knowledge_base(query="", limit=200)
print(result)  # {"error": "Validation failed: ..."}

result = search_knowledge_base(query="LLM patterns", limit=5)
print(result)  # {"results": [...], "total": 1, ...}
```

---

## 6. Model Context Protocol (MCP)

### MCP 개요

> **Model Context Protocol (MCP)**
>
> - Anthropic이 만든 LLM을 외부 데이터 및 도구에 연결하는 개방형 표준
> - 클라이언트-서버 아키텍처: LLM 애플리케이션(클라이언트)이 MCP 서버에 연결
> - 서버가 표준화된 프로토콜을 통해 **도구**, **리소스**, **프롬프트**를 노출
> - 전송: stdio (로컬) 또는 HTTP+SSE (원격)
> - 서로 다른 LLM 애플리케이션 간 도구 재사용을 가능하게 함

### MCP 서버 구현

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import json

# MCP 서버 생성
server = Server("weather-service")

@server.list_tools()
async def list_tools() -> list[Tool]:
    """사용 가능한 도구를 노출."""
    return [
        Tool(
            name="get_weather",
            description="Get current weather for a location",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "default": "celsius",
                    },
                },
                "required": ["city"],
            },
        ),
        Tool(
            name="get_forecast",
            description="Get 5-day weather forecast for a location",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 7,
                        "default": 5,
                    },
                },
                "required": ["city"],
            },
        ),
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """도구 호출을 처리."""
    if name == "get_weather":
        city = arguments["city"]
        unit = arguments.get("unit", "celsius")
        # 프로덕션: 실제 날씨 API 호출
        result = {
            "city": city,
            "temperature": 22 if unit == "celsius" else 72,
            "condition": "sunny",
            "humidity": 55,
        }
        return [TextContent(type="text", text=json.dumps(result))]

    elif name == "get_forecast":
        city = arguments["city"]
        days = arguments.get("days", 5)
        result = {
            "city": city,
            "forecast": [
                {"day": i + 1, "high": 22 + i, "low": 15 + i, "condition": "clear"}
                for i in range(days)
            ],
        }
        return [TextContent(type="text", text=json.dumps(result))]

    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

# 서버 실행
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### MCP 클라이언트 통합

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def use_mcp_tools():
    """MCP 서버에 연결하고 도구를 사용."""
    # stdio를 통해 로컬 MCP 서버에 연결
    server_params = StdioServerParameters(
        command="python",
        args=["weather_mcp_server.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 사용 가능한 도구 목록 조회
            tools = await session.list_tools()
            print("사용 가능한 도구:")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")

            # 도구 호출
            result = await session.call_tool(
                "get_weather",
                arguments={"city": "Seoul", "unit": "celsius"},
            )
            print(f"날씨 결과: {result.content[0].text}")

            # Claude와 함께 도구 사용
            anthropic = Anthropic()

            # MCP 도구를 Anthropic 형식으로 변환
            anthropic_tools = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                }
                for tool in tools.tools
            ]

            response = anthropic.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                tools=anthropic_tools,
                messages=[
                    {"role": "user", "content": "What's the weather and 3-day forecast for Tokyo?"}
                ],
            )

            # MCP를 통해 도구 호출 실행
            if response.stop_reason == "tool_use":
                for block in response.content:
                    if block.type == "tool_use":
                        mcp_result = await session.call_tool(
                            block.name, arguments=block.input
                        )
                        print(f"도구 {block.name}: {mcp_result.content[0].text}")
```

### MCP 아키텍처

| 컴포넌트 | 역할 | 예시 |
|----------|------|------|
| **클라이언트** | 도구를 소비하는 LLM 애플리케이션 | Claude Desktop, 커스텀 앱 |
| **서버** | 도구, 리소스, 프롬프트를 제공 | 날씨 서비스, DB 커넥터 |
| **전송** | 통신 레이어 | stdio, HTTP+SSE |
| **리소스** | 서버가 노출하는 읽기 전용 데이터 | 파일, DB 레코드, API 데이터 |
| **도구** | 실행 가능한 동작 | search, create, update, delete |
| **프롬프트** | 재사용 가능한 프롬프트 템플릿 | 분석 템플릿, 추출 프롬프트 |

---

## 7. 도구 오케스트레이션 패턴

### 순차 도구 체인

```python
class ToolChain:
    """도구를 순차적으로 실행하며 결과를 단계 간 전달."""

    def __init__(self):
        self.client = OpenAI()
        self.steps: list[dict] = []

    def add_step(self, tool_name: str, description: str):
        self.steps.append({"tool": tool_name, "description": description})
        return self

    def execute(self, initial_input: str, tools: list[dict],
                tool_registry: dict) -> list[dict]:
        """체인을 단계별로 실행."""
        results = []
        context = initial_input

        for i, step in enumerate(self.steps):
            print(f"단계 {i+1}/{len(self.steps)}: {step['description']}")

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": (
                        f"You are executing step {i+1}: {step['description']}\n"
                        f"Previous context:\n{context}\n\n"
                        f"Use the '{step['tool']}' tool to complete this step."
                    )},
                    {"role": "user", "content": initial_input},
                ],
                tools=tools,
                tool_choice={"type": "function", "function": {"name": step["tool"]}},
            )

            msg = response.choices[0].message
            if msg.tool_calls:
                tc = msg.tool_calls[0]
                args = json.loads(tc.function.arguments)
                result = tool_registry[tc.function.name](**args)
                results.append({"step": step["description"], "result": result})
                context = f"{context}\n\n단계 {i+1} 결과: {json.dumps(result)}"

        return results
```

### 도구 호출의 에러 처리

```python
from typing import NamedTuple

class ToolResult(NamedTuple):
    success: bool
    data: Any
    error: str | None

class RobustToolExecutor:
    """포괄적인 에러 처리가 있는 도구 실행."""

    def __init__(self, timeout: float = 30.0, max_retries: int = 2):
        self.timeout = timeout
        self.max_retries = max_retries

    def execute(self, name: str, arguments: dict,
                registry: dict[str, Callable]) -> ToolResult:
        """타임아웃, 재시도, 에러 처리가 있는 도구 실행."""
        import signal

        if name not in registry:
            return ToolResult(
                success=False,
                data=None,
                error=f"Unknown tool: {name}. Available: {list(registry.keys())}",
            )

        func = registry[name]

        for attempt in range(self.max_retries + 1):
            try:
                result = func(**arguments)
                return ToolResult(success=True, data=result, error=None)

            except TypeError as e:
                # 잘못된 인자 -- 재시도하지 않고 LLM이 수정하도록 에러 반환
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Invalid arguments for {name}: {e}",
                )

            except ConnectionError as e:
                if attempt < self.max_retries:
                    import time
                    time.sleep(2 ** attempt)  # 지수 백오프
                    continue
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Connection failed after {self.max_retries + 1} attempts: {e}",
                )

            except Exception as e:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Tool execution error: {type(e).__name__}: {e}",
                )

    def format_for_llm(self, result: ToolResult) -> str:
        """도구 결과를 LLM용 문자열 메시지로 포맷."""
        if result.success:
            return json.dumps(result.data) if not isinstance(result.data, str) else result.data
        else:
            return json.dumps({
                "error": result.error,
                "suggestion": "Please try a different approach or different parameters.",
            })

# 에이전트 루프와 통합
executor = RobustToolExecutor(timeout=30.0, max_retries=2)

def agentic_loop(user_message: str, tools: list[dict],
                 registry: dict[str, Callable], max_turns: int = 10) -> str:
    """에러 처리가 있는 강건한 에이전트 루프."""
    messages = [
        {"role": "system", "content": (
            "You are a helpful assistant with tools. "
            "If a tool call fails, try a different approach. "
            "Don't repeat the same failed call."
        )},
        {"role": "user", "content": user_message},
    ]

    for turn in range(max_turns):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            return msg.content

        messages.append(msg)

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            result = executor.execute(tc.function.name, args, registry)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": executor.format_for_llm(result),
            })

    return "최대 턴에 도달하여 최종 답변을 생성하지 못했습니다."
```

### 실제 통합 예시

```python
import httpx

# 예시: 데이터베이스 도구
def query_database(sql: str) -> dict:
    """읽기 전용 SQL 쿼리 실행."""
    # 안전: SELECT 문만 허용
    normalized = sql.strip().upper()
    if not normalized.startswith("SELECT"):
        return {"error": "Only SELECT queries are allowed"}

    # 프로덕션: 실제 데이터베이스에 연결
    return {"columns": ["id", "name"], "rows": [[1, "Alice"], [2, "Bob"]], "row_count": 2}

# 예시: HTTP API 도구
def call_api(url: str, method: str = "GET", body: dict | None = None) -> dict:
    """HTTP API 호출."""
    # 안전: 허용된 도메인의 화이트리스트
    ALLOWED_DOMAINS = ["api.github.com", "api.openweathermap.org"]
    from urllib.parse import urlparse
    domain = urlparse(url).netloc
    if domain not in ALLOWED_DOMAINS:
        return {"error": f"Domain not allowed: {domain}"}

    try:
        with httpx.Client(timeout=10.0) as http_client:
            if method.upper() == "GET":
                resp = http_client.get(url)
            elif method.upper() == "POST":
                resp = http_client.post(url, json=body)
            else:
                return {"error": f"Unsupported method: {method}"}
            return {"status": resp.status_code, "body": resp.json()}
    except Exception as e:
        return {"error": str(e)}

# 예시: 파일 시스템 도구 (샌드박스)
def read_file(path: str) -> dict:
    """허용된 디렉토리에서 파일 읽기."""
    from pathlib import Path
    ALLOWED_DIR = Path("/data/shared")
    target = (ALLOWED_DIR / path).resolve()

    # 보안: 경로 탐색 방지
    if not str(target).startswith(str(ALLOWED_DIR)):
        return {"error": "Access denied: path traversal detected"}

    try:
        content = target.read_text()
        return {"path": str(target), "content": content[:5000], "size": len(content)}
    except FileNotFoundError:
        return {"error": f"File not found: {path}"}
```

### 도구 설계 모범 사례

| 사례 | 설명 |
|------|------|
| 명확한 설명 | 도구 설명은 프롬프트의 일부; 구체적으로 작성 |
| 제한된 입력 | enum, min/max, required 필드를 사용하여 모델을 안내 |
| 멱등 읽기 | GET/읽기 연산은 안전하게 재시도 가능해야 함 |
| LLM용 에러 메시지 | 모델이 추론할 수 있는 문자열로 에러 반환 |
| 도메인 화이트리스트 | API 호출을 승인된 엔드포인트로 제한 |
| 입력 검증 | 실행 전 모든 인자를 검증 |
| 타임아웃 경계 | 모든 외부 호출에 합리적인 타임아웃 설정 |
| 감사 로깅 | 인자와 결과가 포함된 모든 도구 호출을 로깅 |

---

## 다음 단계

[24_Production_LLM_Patterns.md](./24_Production_LLM_Patterns.md)에서는 캐싱, 비용 최적화, 관측 가능성, 멀티 모델 라우팅 등 프로덕션 배포 패턴을 다룬다.
