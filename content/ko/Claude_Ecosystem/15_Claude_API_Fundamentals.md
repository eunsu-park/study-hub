# Claude API 기초

**이전**: [14. Claude 프로젝트와 아티팩트](./14_Claude_Projects_and_Artifacts.md) | **다음**: [16. 도구 사용과 함수 호출](./16_Tool_Use_and_Function_Calling.md)

---

Claude API는 Claude 모델에 프로그래밍 방식으로 접근할 수 있게 해 주며, 여러분의 애플리케이션, 워크플로우, 자동화 시스템에 Claude의 기능을 통합할 수 있습니다. 이 레슨에서는 Claude API를 사용하여 개발을 시작하는 데 필요한 모든 것을 다룹니다: 인증, Messages API, 스트리밍, 토큰 관리, 오류 처리, 그리고 Python과 TypeScript로 작성한 완전히 동작하는 예제들입니다.

**난이도**: ⭐⭐

**사전 요구 사항**:
- Python 3.9+ 또는 Node.js 18+ 설치
- Anthropic API 키 (console.anthropic.com에서 발급)
- REST API 및 HTTP에 대한 기본 이해

**학습 목표**:
- Python과 TypeScript에서 인증 및 클라이언트 SDK 설정하기
- 시스템 프롬프트를 포함한 Messages API 요청 구성 및 전송하기
- 콘텐츠 블록과 사용량 메타데이터를 포함한 API 응답 처리하기
- 실시간 응답 전달을 위한 스트리밍 구현하기
- 토큰을 효과적으로 관리하기 (계산, 예산 설정, 최적화)
- 적절한 재시도 로직과 지수 백오프(Exponential Backoff)로 오류 처리하기
- Claude API를 사용한 완전한 애플리케이션 구축하기

---

## 목차

1. [API 개요](#1-api-개요)
2. [시작하기](#2-시작하기)
3. [클라이언트 SDK](#3-클라이언트-sdk)
4. [Messages API](#4-messages-api)
5. [스트리밍 응답](#5-스트리밍-응답)
6. [토큰 계산 및 관리](#6-토큰-계산-및-관리)
7. [오류 처리](#7-오류-처리)
8. [완전한 동작 예제](#8-완전한-동작-예제)
9. [연습 문제](#9-연습-문제)
10. [참고 자료](#10-참고-자료)

---

## 1. API 개요

Claude API는 `https://api.anthropic.com`에 호스팅된 REST API입니다. 주요 엔드포인트는 모든 텍스트 생성 작업을 처리하는 **Messages API**입니다.

```
┌────────────────────────────────────────────────────────────────┐
│                    Claude API Architecture                      │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Your Application                                               │
│  ┌─────────────────┐                                           │
│  │  Python / TS    │                                           │
│  │  Client SDK     │                                           │
│  └────────┬────────┘                                           │
│           │  HTTPS (TLS 1.2+)                                  │
│           │  x-api-key: sk-ant-...                             │
│           ▼                                                    │
│  ┌─────────────────────────────────────────┐                   │
│  │  https://api.anthropic.com              │                   │
│  │                                         │                   │
│  │  POST /v1/messages        ← Primary     │                   │
│  │  POST /v1/messages/count_tokens         │                   │
│  │  POST /v1/messages/batches  ← Batch API │                   │
│  │  GET  /v1/models           ← List       │                   │
│  └─────────────────────────────────────────┘                   │
│                                                                 │
│  Models Available:                                              │
│  ├── claude-opus-4-20250514     (most capable)                 │
│  ├── claude-sonnet-4-20250514   (balanced)                     │
│  └── claude-haiku-3-5-20241022  (fastest)                      │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

**주요 특성:**
- **무상태(Stateless)**: 각 API 호출은 독립적입니다. 서버에는 세션 상태가 없습니다.
- **동기식 또는 스트리밍**: 전체 응답을 기다리거나 토큰 단위로 수신하는 방식 중 선택할 수 있습니다.
- **토큰당 요금(Pay-per-token)**: 입력 토큰(여러분의 메시지)과 출력 토큰(Claude의 응답)에 기반하여 요금이 청구됩니다.

---

## 2. 시작하기

### 2.1 API 키 발급

1. [console.anthropic.com](https://console.anthropic.com)으로 이동합니다.
2. 계정을 만들거나 로그인합니다.
3. 대시보드에서 **API Keys**로 이동합니다.
4. **Create Key**를 클릭하고 설명이 담긴 이름을 지정합니다.
5. 키를 즉시 복사합니다 -- 이후에는 다시 표시되지 않습니다.

API 키는 다음과 같은 형태입니다: `sk-ant-api03-...`

### 2.2 인증

모든 API 요청에는 `x-api-key` 헤더에 API 키가 포함되어야 합니다:

```bash
curl https://api.anthropic.com/v1/messages \
  -H "content-type: application/json" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 1024,
    "messages": [
      {"role": "user", "content": "Hello, Claude!"}
    ]
  }'
```

**보안 모범 사례:**
- API 키를 소스 컨트롤에 커밋하지 마십시오.
- 환경 변수를 사용하십시오: `export ANTHROPIC_API_KEY="sk-ant-..."`.
- 주기적으로 키를 교체하십시오.
- 개발 환경과 운영 환경에 별도의 키를 사용하십시오.
- Anthropic 콘솔에서 지출 한도를 설정하십시오.

### 2.3 API 버전 관리

`anthropic-version` 헤더는 API 버전을 지정합니다. 항상 포함하십시오:

```
anthropic-version: 2023-06-01
```

SDK는 이를 자동으로 처리하지만, 직접 HTTP 요청을 만드는 경우에는 반드시 포함해야 합니다.

---

## 3. 클라이언트 SDK

### 3.1 Python SDK

```bash
pip install anthropic
```

```python
import anthropic

# SDK가 환경 변수에서 ANTHROPIC_API_KEY를 자동으로 읽습니다
client = anthropic.Anthropic()

# 또는 키를 직접 제공합니다
client = anthropic.Anthropic(api_key="sk-ant-...")
```

### 3.2 TypeScript SDK

```bash
npm install @anthropic-ai/sdk
```

```typescript
import Anthropic from "@anthropic-ai/sdk";

// 환경 변수에서 ANTHROPIC_API_KEY를 읽습니다
const client = new Anthropic();

// 또는 직접 제공합니다
const client = new Anthropic({ apiKey: "sk-ant-..." });
```

### 3.3 기타 언어 SDK

공식 및 커뮤니티 SDK가 여러 언어에서 제공됩니다:

```
┌────────────────┬──────────────────────────────────────────────┐
│ Language       │ Package                                      │
├────────────────┼──────────────────────────────────────────────┤
│ Python         │ pip install anthropic         (official)     │
│ TypeScript/JS  │ npm install @anthropic-ai/sdk (official)     │
│ Java / Kotlin  │ com.anthropic:anthropic-java  (official)     │
│ Go             │ github.com/anthropics/anthropic-sdk-go       │
│ Ruby           │ gem install anthropic                        │
│ C# / .NET     │ NuGet: Anthropic                             │
│ PHP            │ composer require anthropic/anthropic         │
│ Rust           │ crates.io: anthropic (community)             │
└────────────────┴──────────────────────────────────────────────┘
```

모든 공식 SDK는 동일한 설계 패턴을 따르므로, 개념이 언어 간에 그대로 적용됩니다.

---

## 4. Messages API

Messages API는 모든 Claude 상호작용의 핵심 엔드포인트입니다.

### 4.1 요청 구조

```python
import anthropic

client = anthropic.Anthropic()

message = client.messages.create(
    model="claude-sonnet-4-20250514",     # 필수: 사용할 모델
    max_tokens=1024,                       # 필수: 최대 출력 토큰 수
    system="You are a helpful assistant.",  # 선택: 시스템 프롬프트
    messages=[                             # 필수: 대화 메시지
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
    temperature=0.7,                       # 선택: 무작위성 (0-1)
    top_p=0.9,                             # 선택: 핵 샘플링(Nucleus Sampling)
    stop_sequences=["END"],                # 선택: 생성 중단 시퀀스
    metadata={"user_id": "user-123"},      # 선택: 추적 메타데이터
)
```

### 4.2 메시지 역할

`messages` 배열은 엄격한 교대 패턴으로 두 가지 역할을 사용합니다:

```python
messages = [
    # 사용자 메시지 (입력)
    {"role": "user", "content": "Explain recursion."},

    # 어시스턴트 메시지 (Claude의 응답 -- 다중 턴 대화용)
    {"role": "assistant", "content": "Recursion is when a function calls itself..."},

    # 사용자 후속 질문
    {"role": "user", "content": "Can you show me a Python example?"},
]
```

규칙:
- 메시지는 `user`와 `assistant`가 교대로 나타나야 합니다.
- 첫 번째 메시지는 `user`여야 합니다.
- 마지막 메시지는 `user`여야 합니다 (Claude에게 응답을 요청하는 것이므로).

### 4.3 시스템 프롬프트

시스템 프롬프트는 전체 대화에 대한 Claude의 동작을 설정합니다. messages 배열과는 별도로 지정됩니다:

```python
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "You are a Python expert. Always provide type-annotated code. "
                    "Use descriptive variable names and include docstrings.",
        }
    ],
    messages=[
        {"role": "user", "content": "Write a function to find the nth Fibonacci number."}
    ],
)
```

시스템 프롬프트는 단순한 문자열로도 지정할 수 있습니다: `system="You are a Python expert."`.

### 4.4 응답 구조

```python
# 응답 객체
print(message.id)              # "msg_01XFDUDYJgAACzvnptvVoYEL"
print(message.type)            # "message"
print(message.role)            # "assistant"
print(message.model)           # "claude-sonnet-4-20250514"
print(message.stop_reason)     # "end_turn" | "max_tokens" | "stop_sequence" | "tool_use"
print(message.stop_sequence)   # 일치한 중단 시퀀스 (있는 경우)

# 콘텐츠 블록 (텍스트와 tool_use 블록을 포함할 수 있음)
for block in message.content:
    if block.type == "text":
        print(block.text)      # 실제 응답 텍스트

# 토큰 사용량
print(message.usage.input_tokens)   # 요청의 토큰 수
print(message.usage.output_tokens)  # Claude 응답의 토큰 수
```

일반적인 응답 예시:

```json
{
  "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
  "type": "message",
  "role": "assistant",
  "model": "claude-sonnet-4-20250514",
  "content": [
    {
      "type": "text",
      "text": "The capital of France is Paris."
    }
  ],
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 25,
    "output_tokens": 12,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0
  }
}
```

### 4.5 다중 턴 대화

대화 기록을 유지하려면 각 요청에 이전 메시지를 모두 포함시킵니다:

```python
import anthropic

client = anthropic.Anthropic()
conversation: list[dict] = []

def chat(user_message: str) -> str:
    """메시지를 전송하고 대화 기록을 유지합니다."""
    conversation.append({"role": "user", "content": user_message})

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system="You are a knowledgeable coding tutor.",
        messages=conversation,
    )

    assistant_message = response.content[0].text
    conversation.append({"role": "assistant", "content": assistant_message})

    return assistant_message

# 다중 턴 대화
print(chat("What is a hash table?"))
print(chat("How does collision resolution work?"))
print(chat("Show me a Python implementation."))
```

### 4.6 멀티모달 입력 (이미지)

Claude는 텍스트와 함께 이미지를 처리할 수 있습니다:

```python
import anthropic
import base64
from pathlib import Path

client = anthropic.Anthropic()

# 방법 1: Base64로 인코딩된 이미지
image_data = base64.standard_b64encode(
    Path("screenshot.png").read_bytes()
).decode("utf-8")

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
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
                    "text": "Describe what you see in this image.",
                },
            ],
        }
    ],
)

# 방법 2: URL로 참조하는 이미지
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": "https://example.com/photo.jpg",
                    },
                },
                {
                    "type": "text",
                    "text": "What is shown in this photo?",
                },
            ],
        }
    ],
)
```

---

## 5. 스트리밍 응답

스트리밍은 완전한 응답을 기다리는 대신, Claude의 응답을 생성되는 대로 토큰 단위로 전달합니다. 이는 긴 응답에 대해 체감 지연 시간을 크게 개선합니다.

### 5.1 스트리밍이 필요한 이유

```
비스트리밍(Non-streaming):
  요청 ──────── [5초 대기] ──────── 전체 응답
  전체 응답이 준비될 때까지 사용자는 아무것도 볼 수 없습니다.

스트리밍(Streaming):
  요청 ── 토큰 ── 토큰 ── 토큰 ── 토큰 ── ... ── 완료
  사용자는 밀리초 내에 첫 번째 토큰을 볼 수 있습니다.
```

### 5.2 Python 스트리밍

```python
import anthropic

client = anthropic.Anthropic()

# 기본 스트리밍
with client.messages.stream(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Write a short story about a robot learning to paint."}
    ],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

print()  # 최종 줄바꿈

# 스트리밍 완료 후 최종 메시지에 접근
final_message = stream.get_final_message()
print(f"\nTokens used: {final_message.usage.input_tokens} in, "
      f"{final_message.usage.output_tokens} out")
```

**이벤트 처리를 포함한 스트리밍:**

```python
import anthropic

client = anthropic.Anthropic()

with client.messages.stream(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
) as stream:
    for event in stream:
        # 각 이벤트에는 유형이 있습니다
        if event.type == "content_block_start":
            print("[START]", end="")
        elif event.type == "content_block_delta":
            if event.delta.type == "text_delta":
                print(event.delta.text, end="", flush=True)
        elif event.type == "content_block_stop":
            print("\n[END]")
        elif event.type == "message_stop":
            print("[MESSAGE COMPLETE]")
```

### 5.3 TypeScript 스트리밍

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

async function streamResponse() {
  // 기본 스트리밍
  const stream = client.messages.stream({
    model: "claude-sonnet-4-20250514",
    max_tokens: 1024,
    messages: [
      { role: "user", content: "Write a haiku about programming." },
    ],
  });

  // 텍스트를 콘솔에 스트리밍
  stream.on("text", (text) => {
    process.stdout.write(text);
  });

  // 완료 시 최종 메시지 가져오기
  const finalMessage = await stream.finalMessage();
  console.log(`\nTokens: ${finalMessage.usage.input_tokens} in, ` +
              `${finalMessage.usage.output_tokens} out`);
}

streamResponse();
```

**비동기 이터레이터 패턴:**

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

async function streamWithAsyncIterator() {
  const stream = await client.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 1024,
    stream: true,  // 요청 수준에서 스트리밍 활성화
    messages: [
      { role: "user", content: "List the planets in our solar system." },
    ],
  });

  for await (const event of stream) {
    if (
      event.type === "content_block_delta" &&
      event.delta.type === "text_delta"
    ) {
      process.stdout.write(event.delta.text);
    }
  }
  console.log();
}

streamWithAsyncIterator();
```

### 5.4 서버-전송 이벤트(Server-Sent Events) 형식

내부적으로 스트리밍은 서버-전송 이벤트(SSE)를 사용합니다. 원시 형식은 다음과 같습니다:

```
event: message_start
data: {"type": "message_start", "message": {"id": "msg_...", "model": "claude-sonnet-4-20250514", ...}}

event: content_block_start
data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}

event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "The"}}

event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " capital"}}

event: content_block_stop
data: {"type": "content_block_stop", "index": 0}

event: message_delta
data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 12}}

event: message_stop
data: {"type": "message_stop"}
```

SDK가 이 형식을 자동으로 파싱하지만, 디버깅하거나 커스텀 통합을 구축할 때 이해하면 도움이 됩니다.

---

## 6. 토큰 계산 및 관리

### 6.1 토큰이란?

토큰은 Claude가 텍스트를 처리하는 데 사용하는 단위입니다. 1 토큰은 대략 영어 문자 3-4개 또는 약 0.75 단어에 해당합니다. 몇 가지 예시:

```
"Hello"             → 1 토큰
"Hello, world!"     → 4 토큰
"antidisestablish-  → 4 토큰
 mentarianism"
"def fibonacci(n):" → 5 토큰
"こんにちは"         → 3 토큰 (비영어권은 더 많은 토큰을 사용할 수 있음)
```

### 6.2 전송 전 토큰 계산

요청을 전송하기 전에 토큰 계산 엔드포인트를 사용하여 비용을 추정합니다:

```python
import anthropic

client = anthropic.Anthropic()

# 생성 요청 없이 토큰 계산
token_count = client.messages.count_tokens(
    model="claude-sonnet-4-20250514",
    messages=[
        {"role": "user", "content": "Write a comprehensive guide to Python decorators."}
    ],
    system="You are a Python expert.",
)

print(f"Input tokens: {token_count.input_tokens}")
# 실제 요청을 보내기 전에 비용을 추정하는 데 사용합니다
```

### 6.3 max_tokens 관리

`max_tokens` 매개변수는 Claude가 생성할 최대 토큰 수를 제어합니다:

```python
# 짧은 응답 (빠른 답변)
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=100,
    messages=[{"role": "user", "content": "What is 2+2?"}],
)

# 긴 응답 (상세 설명, 코드)
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=4096,
    messages=[{"role": "user", "content": "Write a full REST API in FastAPI."}],
)
```

Claude의 응답이 `max_tokens`에 도달하면, `stop_reason`이 `"max_tokens"`가 되고 응답이 잘립니다. 이를 처리합니다:

```python
if message.stop_reason == "max_tokens":
    print("Warning: Response was truncated. Consider increasing max_tokens.")
```

### 6.4 토큰 예산 전략

```python
def estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """API 호출 비용을 USD로 추정합니다."""
    # 2026년 초 기준 가격 (현재 가격은 문서를 확인하십시오)
    pricing = {
        "claude-opus-4-20250514":   {"input": 15.00, "output": 75.00},
        "claude-sonnet-4-20250514": {"input": 3.00,  "output": 15.00},
        "claude-haiku-3-5-20241022": {"input": 0.80,  "output": 4.00},
    }

    rates = pricing.get(model, pricing["claude-sonnet-4-20250514"])
    cost = (
        (input_tokens / 1_000_000) * rates["input"] +
        (output_tokens / 1_000_000) * rates["output"]
    )
    return cost

# 예시
cost = estimate_cost(
    input_tokens=1500,
    output_tokens=800,
    model="claude-sonnet-4-20250514"
)
print(f"Estimated cost: ${cost:.4f}")
```

---

## 7. 오류 처리

### 7.1 일반적인 오류 코드

```
┌──────────┬───────────────────────┬──────────────────────────────┐
│ 코드     │ 오류                  │ 처리 방법                    │
├──────────┼───────────────────────┼──────────────────────────────┤
│ 400      │ 잘못된 요청           │ 요청 형식과 매개변수 확인    │
│ 401      │ 인증 오류             │ API 키 확인                  │
│ 403      │ 권한 거부             │ API 키 권한 확인             │
│ 404      │ 찾을 수 없음          │ 엔드포인트 URL 확인          │
│ 429      │ 속도 제한             │ 백오프로 대기 후 재시도      │
│ 500      │ API 오류 (서버)       │ 지연 후 재시도               │
│ 529      │ API 과부하            │ 백오프로 재시도              │
└──────────┴───────────────────────┴──────────────────────────────┘
```

### 7.2 Python 오류 처리

```python
import anthropic

client = anthropic.Anthropic()

try:
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello!"}],
    )
    print(message.content[0].text)

except anthropic.AuthenticationError:
    print("Invalid API key. Check your ANTHROPIC_API_KEY.")

except anthropic.RateLimitError:
    print("Rate limited. Wait before retrying.")

except anthropic.BadRequestError as e:
    print(f"Invalid request: {e.message}")

except anthropic.InternalServerError:
    print("Anthropic server error. Retry later.")

except anthropic.APIStatusError as e:
    print(f"API error (status {e.status_code}): {e.message}")

except anthropic.APIConnectionError:
    print("Could not connect to the API. Check your network.")
```

### 7.3 TypeScript 오류 처리

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

async function callClaude() {
  try {
    const message = await client.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 1024,
      messages: [{ role: "user", content: "Hello!" }],
    });
    console.log(message.content[0].type === "text" ? message.content[0].text : "");

  } catch (error) {
    if (error instanceof Anthropic.AuthenticationError) {
      console.error("Invalid API key.");
    } else if (error instanceof Anthropic.RateLimitError) {
      console.error("Rate limited. Retry later.");
    } else if (error instanceof Anthropic.BadRequestError) {
      console.error(`Bad request: ${error.message}`);
    } else if (error instanceof Anthropic.InternalServerError) {
      console.error("Server error. Retry later.");
    } else if (error instanceof Anthropic.APIError) {
      console.error(`API error (${error.status}): ${error.message}`);
    } else {
      throw error;  // 예상치 못한 오류는 다시 던집니다
    }
  }
}

callClaude();
```

### 7.4 지수 백오프(Exponential Backoff)를 이용한 재시도

```python
import time
import random
import anthropic

client = anthropic.Anthropic()

def call_with_retry(
    messages: list,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
) -> anthropic.types.Message:
    """재시도 가능한 오류에 대해 지수 백오프로 API를 호출합니다."""
    for attempt in range(max_retries):
        try:
            return client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=messages,
            )
        except (anthropic.RateLimitError, anthropic.InternalServerError,
                anthropic.APIConnectionError) as e:
            if attempt == max_retries - 1:
                raise  # 마지막 시도: 다시 발생시킵니다

            # 지터(Jitter)를 포함한 지수 백오프
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0, delay * 0.1)
            wait_time = delay + jitter

            print(f"Attempt {attempt + 1} failed ({type(e).__name__}). "
                  f"Retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)

        except anthropic.BadRequestError:
            raise  # 클라이언트 오류는 재시도하지 않습니다

    raise RuntimeError("Should not reach here")


# 사용 예시
response = call_with_retry([
    {"role": "user", "content": "What is the meaning of life?"}
])
print(response.content[0].text)
```

참고: 공식 Python SDK에는 재시도 로직이 내장되어 있습니다. 기본적으로 `429`, `500`, 연결 오류에 대해 최대 2회 재시도합니다. 이를 구성할 수 있습니다:

```python
client = anthropic.Anthropic(
    max_retries=5,       # 최대 재시도 횟수 (기본값: 2)
    timeout=60.0,        # 요청 타임아웃 (초) (기본값: 600)
)
```

---

## 8. 완전한 동작 예제

### 8.1 Python: 대화형 채팅 애플리케이션

```python
#!/usr/bin/env python3
"""Claude API를 사용한 대화형 채팅 애플리케이션."""

import anthropic
import sys

def main():
    client = anthropic.Anthropic()
    conversation: list[dict] = []
    system_prompt = (
        "You are a helpful assistant. Be concise but thorough. "
        "When providing code, include brief comments explaining key parts."
    )

    print("Claude Chat (type 'quit' to exit, 'clear' to reset)")
    print("=" * 50)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "clear":
            conversation.clear()
            print("[Conversation cleared]")
            continue

        conversation.append({"role": "user", "content": user_input})

        try:
            print("\nClaude: ", end="", flush=True)
            full_response = ""

            with client.messages.stream(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                system=system_prompt,
                messages=conversation,
            ) as stream:
                for text in stream.text_stream:
                    print(text, end="", flush=True)
                    full_response += text

            final = stream.get_final_message()
            print(f"\n  [{final.usage.input_tokens} in / "
                  f"{final.usage.output_tokens} out tokens]")

            conversation.append({
                "role": "assistant",
                "content": full_response,
            })

        except anthropic.APIError as e:
            print(f"\n[Error: {e.message}]")
            # 대화를 유효하게 유지하기 위해 실패한 사용자 메시지를 제거합니다
            conversation.pop()

if __name__ == "__main__":
    main()
```

### 8.2 TypeScript: 코드 리뷰 어시스턴트

```typescript
// code-review.ts — Claude API를 사용한 자동 코드 리뷰
import Anthropic from "@anthropic-ai/sdk";
import { readFileSync } from "fs";

const client = new Anthropic();

interface ReviewResult {
  summary: string;
  issues: Array<{
    severity: "error" | "warning" | "info";
    line: number | null;
    message: string;
    suggestion: string;
  }>;
  score: number; // 1-10
}

async function reviewCode(
  filePath: string,
  language: string
): Promise<ReviewResult> {
  const code = readFileSync(filePath, "utf-8");

  const message = await client.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 4096,
    system: `You are an expert code reviewer. Analyze the provided ${language} code
and respond with a JSON object containing:
- "summary": A 1-2 sentence summary of the code quality.
- "issues": An array of objects, each with "severity" (error/warning/info),
  "line" (line number or null), "message" (description), and "suggestion" (fix).
- "score": An integer 1-10 rating (10 = excellent).

Respond with ONLY the JSON object, no other text.`,
    messages: [
      {
        role: "user",
        content: `Review this ${language} code:\n\n\`\`\`${language}\n${code}\n\`\`\``,
      },
    ],
  });

  const responseText =
    message.content[0].type === "text" ? message.content[0].text : "";

  // JSON 응답 파싱
  const jsonMatch = responseText.match(/\{[\s\S]*\}/);
  if (!jsonMatch) {
    throw new Error("Failed to parse review response as JSON");
  }

  return JSON.parse(jsonMatch[0]) as ReviewResult;
}

// 메인 실행
async function main() {
  const filePath = process.argv[2];
  const language = process.argv[3] || "python";

  if (!filePath) {
    console.error("Usage: npx tsx code-review.ts <file-path> [language]");
    process.exit(1);
  }

  console.log(`Reviewing ${filePath} (${language})...`);
  const review = await reviewCode(filePath, language);

  console.log(`\nScore: ${review.score}/10`);
  console.log(`Summary: ${review.summary}\n`);

  if (review.issues.length === 0) {
    console.log("No issues found!");
  } else {
    console.log(`Issues (${review.issues.length}):`);
    for (const issue of review.issues) {
      const line = issue.line ? `L${issue.line}` : "—";
      const icon =
        issue.severity === "error" ? "[!]" :
        issue.severity === "warning" ? "[~]" : "[i]";
      console.log(`  ${icon} ${line}: ${issue.message}`);
      console.log(`      Fix: ${issue.suggestion}`);
    }
  }
}

main().catch(console.error);
```

### 8.3 Python: 토큰 관리를 포함한 문서 요약기

```python
#!/usr/bin/env python3
"""토큰 한계에 맞는 자동 청킹으로 긴 문서를 요약합니다."""

import anthropic
from pathlib import Path

client = anthropic.Anthropic()

# 모델 컨텍스트 한계 (대략적인 입력 토큰 예산)
MODEL_LIMITS = {
    "claude-opus-4-20250514": 190_000,
    "claude-sonnet-4-20250514": 190_000,
    "claude-haiku-3-5-20241022": 190_000,
}

def count_tokens(text: str, model: str = "claude-sonnet-4-20250514") -> int:
    """텍스트 문자열의 토큰 수를 계산합니다."""
    result = client.messages.count_tokens(
        model=model,
        messages=[{"role": "user", "content": text}],
    )
    return result.input_tokens

def summarize_text(
    text: str,
    model: str = "claude-sonnet-4-20250514",
    max_output_tokens: int = 2048,
) -> str:
    """필요시 청킹을 사용하여 텍스트를 요약합니다."""
    token_count = count_tokens(text, model)
    max_input = MODEL_LIMITS.get(model, 190_000) - max_output_tokens - 500  # 마진

    print(f"Document tokens: {token_count:,}")
    print(f"Max input tokens: {max_input:,}")

    if token_count <= max_input:
        # 문서가 단일 요청에 맞는 경우
        return _summarize_single(text, model, max_output_tokens)
    else:
        # 문서가 너무 긴 경우: 청크로 분할하고 계층적으로 요약
        return _summarize_chunked(text, model, max_input, max_output_tokens)

def _summarize_single(text: str, model: str, max_tokens: int) -> str:
    """단일 컨텍스트 창에 맞는 텍스트를 요약합니다."""
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system="You are an expert summarizer. Create clear, comprehensive summaries "
               "that capture all key points, data, and conclusions.",
        messages=[
            {
                "role": "user",
                "content": f"Summarize the following document:\n\n{text}",
            }
        ],
    )
    return message.content[0].text

def _summarize_chunked(
    text: str, model: str, max_chunk_tokens: int, max_output_tokens: int
) -> str:
    """긴 텍스트를 청크로 분할하여 요약합니다."""
    # 단락으로 단순 분할 (프로덕션 코드에서는 더 나은 청킹 방법을 사용해야 함)
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para, model)
        if current_tokens + para_tokens > max_chunk_tokens and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_tokens = para_tokens
        else:
            current_chunk.append(para)
            current_tokens += para_tokens

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    print(f"Split into {len(chunks)} chunks")

    # 각 청크 요약
    chunk_summaries: list[str] = []
    for i, chunk in enumerate(chunks):
        print(f"  Summarizing chunk {i + 1}/{len(chunks)}...")
        summary = _summarize_single(chunk, model, max_output_tokens // len(chunks))
        chunk_summaries.append(summary)

    # 청크 요약을 최종 요약으로 합칩니다
    combined = "\n\n---\n\n".join(
        f"Section {i + 1} Summary:\n{s}" for i, s in enumerate(chunk_summaries)
    )

    print("  Generating final summary...")
    return _summarize_single(
        f"Combine these section summaries into a single coherent summary:\n\n{combined}",
        model,
        max_output_tokens,
    )

# 사용 예시
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python summarizer.py <file_path>")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    text = file_path.read_text()
    summary = summarize_text(text)
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print('=' * 60)
    print(summary)
```

---

## 9. 연습 문제

### 연습 1: 첫 번째 API 호출 (초급)

Python 또는 TypeScript SDK를 설정하고 첫 번째 API 호출을 만들어 보십시오. Claude에게 원하는 프로그래밍 개념을 설명해 달라고 요청하십시오. 응답 텍스트와 토큰 사용량을 출력하십시오. 서로 다른 `temperature` 값(0.0, 0.5, 1.0)을 시도하고 응답이 어떻게 변하는지 관찰하십시오.

### 연습 2: 다중 턴 계산기 (초급)

누적 합계를 유지하는 대화형 계산기를 구축하십시오. 다중 턴 대화를 사용하여 사용자가 "add 5", "multiply by 3", "what is the current total?"과 같은 말을 할 수 있도록 하고, Claude가 상태를 추적하도록 하십시오.

### 연습 3: 스트리밍 진행 상황 (중급)

Claude의 응답을 스트리밍하고 진행 상황 표시기를 보여주는 Python 스크립트를 구축하십시오. 지금까지 생성된 토큰 수, 경과 시간, 초당 토큰 수를 표시하십시오. 총 비용 추정과 함께 최종 요약을 출력하십시오.

### 연습 4: 배치 처리 (중급)

텍스트 파일 디렉토리를 읽어 각 파일을 요약을 위해 Claude에 전송하고 Messages API를 사용하여 요약을 출력 파일에 쓰는 스크립트를 작성하십시오. 오류 처리, 재시도 로직, 진행 표시줄을 포함하십시오. 요청 사이에 지연을 추가하여 속도 제한을 준수하십시오.

### 연습 5: API 래퍼 라이브러리 (고급)

Anthropic SDK를 감싸는 소규모 래퍼 라이브러리를 만들어 다음을 제공하십시오:
- 지수 백오프를 포함한 자동 재시도
- 여러 호출에 걸친 토큰 사용량 추적
- 비용 추정 및 예산 적용 (달러 한도를 초과하는 호출 거부)
- 컨텍스트가 너무 길어질 때 자동 프루닝(Pruning)을 포함한 대화 기록 관리
- 모든 API 호출의 구조화된 로깅

---

## 10. 참고 자료

- Anthropic API 참조 - https://docs.anthropic.com/en/api/messages
- Anthropic Python SDK - https://github.com/anthropics/anthropic-sdk-python
- Anthropic TypeScript SDK - https://github.com/anthropics/anthropic-sdk-typescript
- Anthropic Cookbook - https://github.com/anthropics/anthropic-cookbook
- API 가격 - https://www.anthropic.com/pricing
- 속도 제한 - https://docs.anthropic.com/en/api/rate-limits
- 토큰 계산 - https://docs.anthropic.com/en/docs/build-with-claude/token-counting
- 스트리밍 - https://docs.anthropic.com/en/api/messages-streaming

---

## 다음 레슨

[16. 도구 사용과 함수 호출](./16_Tool_Use_and_Function_Calling.md)에서는 도구 사용으로 API 지식을 확장합니다 -- Claude가 여러분이 정의한 함수를 호출하도록 허용합니다. 도구 정의 방법, 도구 사용 대화 흐름 처리, 병렬 도구 호출 관리, 실용적인 통합 구축 방법을 배우게 됩니다.
