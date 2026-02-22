# Claude 소개

**다음**: [02. Claude Code: 시작하기](./02_Claude_Code_Getting_Started.md)

---

Claude는 안전성, 유용성, 정직성을 위해 설계된 Anthropic의 대규모 언어 모델 패밀리입니다. 순수하게 능력만을 최적화하는 모델들과 달리, Claude는 헌법적 AI(Constitutional AI) — 모델을 인간의 가치와 일치시키면서 추론, 코딩, 분석, 창의적 작업 전반에 걸쳐 최전선 수준의 성능을 유지하는 학습 방법론 — 을 기반으로 구축되었습니다. 이 레슨은 Claude 모델 패밀리, 제품 생태계, 그리고 Claude를 효과적으로 사용하기 위해 필요한 기초 개념에 대한 종합적인 개요를 제공합니다.

**난이도**: ⭐

**선행 조건**:
- AI 관련 사전 경험 불필요
- 소프트웨어 개발 개념에 대한 기본 이해가 있으면 도움됨

**학습 목표**:
- Claude 모델 패밀리와 각 모델을 언제 사용할지 이해하기
- Claude의 핵심 기능과 차별점 파악하기
- 제품 생태계 탐색: Claude.ai, Claude Code, Claude Desktop, API
- 컨텍스트 윈도우, 토큰, Claude의 텍스트 처리 방식 이해하기
- 작업 요구사항에 따른 합리적인 모델 선택 결정하기
- 가격 책정 모델과 비용 고려사항 이해하기

---

## 목차

1. [Claude 모델 패밀리](#1-claude-모델-패밀리)
2. [핵심 기능](#2-핵심-기능)
3. [Claude의 차별점: 헌법적 AI](#3-claude의-차별점-헌법적-ai)
4. [컨텍스트 윈도우와 토큰](#4-컨텍스트-윈도우와-토큰)
5. [제품 생태계](#5-제품-생태계)
6. [가격 책정과 비용 고려사항](#6-가격-책정과-비용-고려사항)
7. [어떤 제품을 언제 사용할까](#7-어떤-제품을-언제-사용할까)
8. [연습 문제](#8-연습-문제)
9. [다음 단계](#9-다음-단계)

---

## 1. Claude 모델 패밀리

Anthropic은 능력, 속도, 비용 간의 서로 다른 트레이드오프를 위해 최적화된 세 가지 모델 티어를 제공합니다. 모든 모델은 동일한 안전성 학습과 핵심 아키텍처를 공유하지만 규모와 성능 특성에서 차이가 있습니다.

### 모델 비교표

| 속성 | Claude Opus 4 | Claude Sonnet 4 | Claude Haiku |
|------|--------------|-----------------|--------------|
| **지능** | 최고 | 높음 | 양호 |
| **속도** | 느림 | 빠름 | 가장 빠름 |
| **비용** | 최고 | 중간 | 최저 |
| **컨텍스트 윈도우** | 200K 토큰 | 200K 토큰 | 200K 토큰 |
| **최대 출력** | 32K 토큰 | 16K 토큰 | 8K 토큰 |
| **최적 용도** | 복잡한 추론, 연구, 아키텍처 | 일상적 코딩, 분석, 균형 잡힌 작업 | 빠른 쿼리, 분류, 대용량 처리 |
| **확장 사고** | 지원 | 지원 | 미지원 |

### Claude Opus 4

Opus는 Anthropic의 가장 강력한 모델입니다. 깊은 추론, 미묘한 이해, 다단계 문제 해결이 필요한 작업에서 뛰어납니다. 속도보다 정확성과 깊이가 중요할 때 Opus를 사용하세요.

**강점**:
- 복잡한 코드 아키텍처 및 시스템 설계
- 장문의 분석 및 연구 종합
- 엣지 케이스에 대한 미묘한 추론
- 다단계 문제를 위한 확장 사고(extended thinking)
- 모호하거나 불충분하게 명세된 요구사항 처리

**일반적인 사용 사례**:
- 시스템 아키텍처 설계
- 복잡한 풀 리퀘스트(pull request) 검토
- 프로덕션 핵심 코드 작성
- 연구 논문 분석
- 대규모 코드베이스에 걸친 다중 파일 리팩토링

### Claude Sonnet 4

Sonnet은 균형 잡힌 핵심 모델입니다 — 대화형 사용에 충분히 빠르고, 대부분의 전문적 작업에 충분히 유능합니다. Claude Code의 기본 모델이며 프로덕션 환경에서 가장 널리 사용되는 모델입니다.

**강점**:
- 빠른 코드 생성 및 편집
- 대화형 개발 세션
- 품질과 처리량의 좋은 균형
- 안정적인 지침 따르기
- 지속적인 사용에 비용 효율적

**일반적인 사용 사례**:
- 일상적인 코딩 지원
- 테스트 및 문서 작성
- 번역 및 콘텐츠 생성
- 데이터 분석 및 시각화
- API 통합 작업

### Claude Haiku

Haiku는 속도와 비용을 위해 최적화되어 있습니다. 간단한 작업을 안정적으로 처리하며, 지연 시간과 비용이 주요 관심사인 대용량 애플리케이션에 이상적입니다.

**강점**:
- 1초 미만의 응답 시간
- 최저 토큰당 비용
- 안정적인 분류 및 추출
- 구조화된 출력 생성에 적합

**일반적인 사용 사례**:
- 텍스트 분류 및 레이블링
- 문서에서 데이터 추출
- 간단한 코드 완성
- 비용 제약이 있는 채팅 애플리케이션
- 에이전트 파이프라인의 전처리 및 라우팅

### 모델 선택 결정 트리

```
작업이 복잡하거나, 모호하거나, 안전에 중요한가?
├── 예 → Claude Opus 4
│         (깊은 추론, 아키텍처, 복잡한 분석)
│
├── 보통 → Claude Sonnet 4
│           (일상적 코딩, 분석, 콘텐츠 생성)
│
└── 단순/대용량 → Claude Haiku
                  (분류, 추출, 라우팅)
```

---

## 2. 핵심 기능

Claude는 광범위한 작업을 처리하는 범용 AI입니다. 기능을 이해하면 각 상황에 맞는 올바른 기능을 활용하는 데 도움이 됩니다.

### 추론 및 분석

Claude는 다단계 논리적 추론에 참여하고, 복잡한 문제를 분해하며, 구조화된 분석을 제공할 수 있습니다. 확장 사고(extended thinking)가 활성화되면 Opus와 Sonnet은 응답 전에 사고 과정(chain of thought)을 보여줄 수 있습니다.

```python
# 예제: Claude에게 설계 결정 분석 요청하기
prompt = """
Our API currently returns all user data in a single endpoint.
We're considering splitting it into separate endpoints per resource.

Analyze the trade-offs considering:
1. Client complexity
2. Network overhead
3. Caching strategies
4. API versioning
5. Backend complexity
"""

# Claude는 각 요소를 고려한 구조화된 분석,
# 잠재적 마이그레이션 전략, 권장 사항을 제공합니다.
```

### 코드 생성 및 이해

Claude는 수십 가지 프로그래밍 언어로 코드를 읽고 작성합니다. 프로젝트 구조를 이해하고, 코딩 관례를 따르며, 기존 코드베이스와 함께 작업할 수 있습니다.

```python
# Claude는 자연어 설명으로부터 코드를 생성할 수 있습니다
# 예제: "지수 백오프가 있는 재시도 데코레이터 만들기"

import time
import functools
from typing import Type


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
):
    """Retry a function with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts.
        base_delay: Initial delay in seconds.
        backoff_factor: Multiplier for each subsequent delay.
        exceptions: Tuple of exception types to catch.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        delay = base_delay * (backoff_factor ** attempt)
                        time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator


@retry(max_attempts=3, base_delay=0.5, exceptions=(ConnectionError, TimeoutError))
def fetch_data(url: str) -> dict:
    """Fetch data from an API with automatic retry."""
    import urllib.request
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.read()
```

### 다국어 지원

Claude는 다양한 언어로 유창하게 소통하고 기술적 뉘앙스를 보존하며 언어 간 번역을 할 수 있습니다. 영어, 한국어, 일본어, 중국어, 프랑스어, 독일어, 스페인어 등 수많은 언어로 코드 주석, 문서, 기술 문서 작업을 처리합니다.

### 비전(멀티모달)

Claude는 이미지, 스크린샷, 다이어그램, 문서를 분석할 수 있습니다. 이는 UI 목업 이해, 화이트보드 스케치 읽기, 차트 분석, 스캔된 문서 처리에 유용합니다.

```python
# API를 사용하여 Claude에 이미지 전송하기
import anthropic
import base64

client = anthropic.Anthropic()

# 이미지를 읽어 인코딩
with open("architecture_diagram.png", "rb") as f:
    image_data = base64.standard_b64encode(f.read()).decode("utf-8")

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
                    "text": "Describe this architecture diagram. Identify potential bottlenecks."
                }
            ],
        }
    ],
)

print(message.content[0].text)
```

### 긴 컨텍스트 처리

200K 토큰 컨텍스트 윈도우를 통해 Claude는 전체 코드베이스, 긴 문서, 광범위한 대화 기록을 처리할 수 있습니다. 이는 대략 텍스트 500페이지 또는 중간 규모 코드베이스에 해당합니다.

---

## 3. Claude의 차별점: 헌법적 AI

Claude는 **헌법적 AI(Constitutional AI, CAI)** 라 불리는 독특한 접근 방식으로 구축되어 있으며, 이는 주로 인간 피드백 강화 학습(RLHF, Reinforcement Learning from Human Feedback)으로 학습된 모델들과 구별됩니다.

### 헌법적 AI란 무엇인가?

헌법적 AI는 모델이 무엇이 유용하고 무해한지를 결정하기 위해 인간 레이블러에만 의존하지 않고, 일련의 원칙("헌법")에 의해 안내받는 학습 방법론입니다. 이 과정은 두 가지 핵심 단계로 구성됩니다:

1. **지도 학습 단계**: 모델이 응답을 생성한 후, 헌법적 원칙에 기반하여 자신의 출력을 비판하고 수정합니다
2. **강화 학습 단계**: 인간의 선호도 대신, 헌법에 의해 안내된 모델 자신의 평가가 학습 신호를 제공합니다

```
기존 RLHF:
  모델 → 응답 → 인간 레이블러 → 보상 신호 → 모델 업데이트

헌법적 AI:
  모델 → 응답 → 자기 비판(원칙을 통해) → 수정된 응답
  모델 → 응답 쌍 → AI 피드백(원칙을 통해) → 보상 신호 → 모델 업데이트
```

### 실제로 이것이 중요한 이유

| 측면 | 사용자에 대한 영향 |
|------|-------------------|
| **투명성** | Claude는 자신의 추론과 한계를 설명할 수 있음 |
| **일관성** | 엣지 케이스 전반에 걸쳐 더 예측 가능한 동작 |
| **정직성** | 꾸며내기보다 "모르겠습니다"라고 말할 의향 있음 |
| **뉘앙스** | 일괄 거부 대신 세심하게 민감한 주제를 다룸 |
| **유용성** | 안전 경계 내에서 최대한 유용하려 노력함 |

### 확장 사고(Extended Thinking)

Claude Opus와 Sonnet은 **확장 사고** — 최종 답변을 생성하기 전에 모델이 문제를 단계별로 풀어가는 명시적 추론 단계 — 를 지원합니다. 이는 API와 Claude.ai에서 "thinking" 블록으로 표시됩니다.

```json
{
  "model": "claude-opus-4-20250514",
  "max_tokens": 16000,
  "thinking": {
    "type": "enabled",
    "budget_tokens": 10000
  },
  "messages": [
    {
      "role": "user",
      "content": "Prove that the square root of 2 is irrational."
    }
  ]
}
```

확장 사고는 다음과 같은 경우에 특히 유용합니다:
- 수학적 증명 및 유도
- 복잡한 디버깅 시나리오
- 많은 트레이드오프가 있는 아키텍처 결정
- 다양한 해결 경로 탐색이 필요한 문제

---

## 4. 컨텍스트 윈도우와 토큰

Claude가 텍스트를 처리하는 방식을 이해하는 것은, 특히 대규모 코드베이스나 문서를 다룰 때 효과적인 사용에 필수적입니다.

### 토큰이란 무엇인가?

**토큰(token)** 은 Claude가 텍스트를 처리하는 데 사용하는 기본 단위입니다. 토큰은 단어가 아닙니다 — 모델의 토크나이저(tokenizer)에 의해 결정되는 서브워드 단위입니다. 평균적으로:

- **토큰 1개**는 영어 텍스트 약 3-4자에 해당
- **단어 1개**는 약 1.3 토큰
- **코드 1줄**은 약 10-15 토큰
- **텍스트 1페이지** (~500 단어)는 약 650 토큰

```
토크나이즈 예시:
"Hello, world!" → ["Hello", ",", " world", "!"]  (4 토큰)
"def calculate_total(items):" → ["def", " calculate", "_total", "(", "items", "):"]  (6 토큰)
"안녕하세요" → ["안녕", "하세요"]  (2 토큰, 언어에 따라 다름)
```

### 컨텍스트 윈도우

**컨텍스트 윈도우(context window)** 는 Claude가 한 번에 고려할 수 있는 총 토큰 수로, 입력(프롬프트, 시스템 지침, 파일 내용)과 출력(Claude의 응답) 모두를 포함합니다.

```
┌─────────────────────────────────────────────────────────┐
│                 200K 토큰 컨텍스트 윈도우                │
│                                                         │
│  ┌─────────────────────────┐  ┌──────────────────────┐  │
│  │     입력 토큰           │  │   출력 토큰          │  │
│  │                         │  │                      │  │
│  │  시스템 프롬프트         │  │  Claude의 응답       │  │
│  │  CLAUDE.md 내용         │  │  (max_tokens까지)    │  │
│  │  대화 기록              │  │                      │  │
│  │  파일 내용              │  │                      │  │
│  │  도구 결과              │  │                      │  │
│  │                         │  │                      │  │
│  └─────────────────────────┘  └──────────────────────┘  │
│                                                         │
│  ← 입력은 왼쪽부터         출력은 오른쪽부터 →          │
└─────────────────────────────────────────────────────────┘
```

### 실용적 함의

| 시나리오 | 대략적인 토큰 사용량 |
|----------|---------------------|
| 짧은 질문 | 50-100 토큰 |
| CLAUDE.md 파일 | 500-3,000 토큰 |
| 단일 소스 파일 (~200줄) | 2,000-3,000 토큰 |
| 소스 파일 10개 | 20,000-30,000 토큰 |
| 중간 규모 코드베이스 전체 | 100,000-150,000 토큰 |
| 전체 대화 (1시간 세션) | 50,000-200,000 토큰 |

컨텍스트 윈도우가 가득 차면, Claude Code는 **컨텍스트 압축(context compaction)** 이라는 전략을 사용합니다 — 가장 중요한 정보를 보존하면서 공간을 확보하기 위해 지금까지의 대화를 요약합니다.

---

## 5. 제품 생태계

Claude는 각각 서로 다른 사용 사례와 워크플로우를 위해 설계된 여러 제품을 통해 이용 가능합니다.

### 제품 개요

```
┌─────────────────────────────────────────────────────────────┐
│                     Claude 생태계                            │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  Claude.ai   │  │Claude Desktop│  │ Claude Code  │       │
│  │  (웹 앱)     │  │  (macOS/Win) │  │   (CLI)      │       │
│  │              │  │              │  │              │       │
│  │ 채팅, 파일,  │  │ MCP, 앱      │  │ 코드 편집,   │       │
│  │ 프로젝트,    │  │ 미리보기,    │  │ 터미널,      │       │
│  │ 아티팩트     │  │ 시스템       │  │ Git, 테스트  │       │
│  │              │  │ 통합         │  │              │       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
│         │                 │                 │                │
│         └─────────────────┼─────────────────┘                │
│                           │                                  │
│                    ┌──────┴───────┐                           │
│                    │  Claude API  │                           │
│                    │              │                           │
│                    │ Messages API │                           │
│                    │ Tool Use     │                           │
│                    │ Streaming    │                           │
│                    │ Batch API    │                           │
│                    └──────┬───────┘                           │
│                           │                                  │
│                    ┌──────┴───────┐                           │
│                    │  Agent SDK   │                           │
│                    │              │                           │
│                    │ 커스텀 AI    │                           │
│                    │ 에이전트 구축│                           │
│                    └──────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

### Claude.ai (웹 애플리케이션)

[claude.ai](https://claude.ai)의 웹 인터페이스는 Claude와 상호작용하는 가장 접근하기 쉬운 방법입니다.

**주요 기능**:
- 모델 선택 기능이 있는 대화형 채팅 인터페이스
- 파일 업로드 (PDF, 이미지, 코드 파일, CSV)
- **프로젝트**: 공유 컨텍스트와 지침으로 대화 구성
- **아티팩트**: Claude가 대화형 콘텐츠(코드, 문서, 다이어그램) 생성
- 대화 기록 및 검색
- 공유 결제 및 관리 콘솔이 있는 팀 플랜

**최적 용도**: 일반 질문, 문서 분석, 브레인스토밍, 작문, 빠른 프로토타이핑.

### Claude Desktop

macOS와 Windows용 네이티브 데스크탑 애플리케이션으로, 웹 앱보다 더 깊은 시스템 통합을 제공합니다.

**주요 기능**:
- **앱 미리보기**: Desktop 앱에서 직접 웹 애플리케이션을 생성하고 미리보기
- **모델 컨텍스트 프로토콜(MCP)**: Claude를 외부 도구 및 데이터 소스에 연결
- **GitHub 통합**: 풀 리퀘스트, 이슈, 저장소 분석
- 빠른 접근을 위한 키보드 단축키
- 오프라인 대화 기록

**최적 용도**: MCP 기능과 함께 Claude를 데스크탑 워크플로우에 통합하고자 하는 사용자.

### Claude Code (CLI)

Claude Code는 터미널에서 직접 실행되는 커맨드라인 AI 코딩 어시스턴트입니다. AI 지원 소프트웨어 개발의 주요 도구이며 이 토픽의 핵심입니다.

**주요 기능**:
- 코드베이스를 읽고 이해
- 승인을 받아(또는 자동으로) 파일 편집
- 터미널 명령 실행 (테스트, 빌드, git)
- 기존 개발 도구와 통합
- CLAUDE.md와 설정 파일을 통해 설정 가능
- 훅과 스킬로 확장 가능

```bash
# Claude Code 설치
npm install -g @anthropic-ai/claude-code

# 대화형 세션 시작
claude

# 원샷 명령 실행
claude -p "Explain the authentication flow in this project"

# 특정 모델 사용
claude --model claude-opus-4-20250514
```

**최적 용도**: 소프트웨어 개발, 디버깅, 코드 검토, 리팩토링, 테스트 작성.

### Claude API

API는 애플리케이션 구축, 워크플로우 자동화, 기존 시스템에 AI를 통합하기 위한 프로그래밍 방식의 Claude 접근을 제공합니다.

```python
import anthropic

client = anthropic.Anthropic()  # ANTHROPIC_API_KEY 환경 변수 사용

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Write a haiku about programming."}
    ]
)

print(message.content[0].text)
```

**최적 용도**: AI 기반 애플리케이션 구축, 자동화 파이프라인, 커스텀 통합.

### 기업용 Claude

엔터프라이즈 제공 사항에는 다음이 포함됩니다:
- **Claude for Work** (팀 플랜): 공유 결제, 관리 콘솔, 높은 속도 제한
- **Claude for Enterprise**: SSO, SCIM 프로비저닝, 커스텀 데이터 보존, 전용 지원
- **Amazon Bedrock / Google Vertex AI**: 기존 결제 방식으로 클라우드 제공업체 API를 통해 Claude 접근

---

## 6. 가격 책정과 비용 고려사항

Claude는 **토큰당 지불(pay-per-token)** 가격 책정 모델을 사용합니다. 전송하는 토큰(입력)과 Claude가 생성하는 토큰(출력)에 대해 비용을 지불합니다. 출력 토큰은 더 많은 계산이 필요하므로 더 비쌉니다.

### 가격표 (2025년 초 기준)

| 모델 | 입력 (1M 토큰당) | 출력 (1M 토큰당) |
|------|-----------------|-----------------|
| Claude Opus 4 | $15.00 | $75.00 |
| Claude Sonnet 4 | $3.00 | $15.00 |
| Claude Haiku | $0.25 | $1.25 |

> **참고**: 가격은 변경될 수 있습니다. 현재 요금은 [anthropic.com/pricing](https://anthropic.com/pricing)을 확인하세요.

### 비용 추정 예시

```python
# 비용 추정 도우미
def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate the cost of an API call in USD."""
    pricing = {
        "opus":   {"input": 15.00, "output": 75.00},
        "sonnet": {"input":  3.00, "output": 15.00},
        "haiku":  {"input":  0.25, "output":  1.25},
    }

    rates = pricing[model]
    input_cost = (input_tokens / 1_000_000) * rates["input"]
    output_cost = (output_tokens / 1_000_000) * rates["output"]
    return input_cost + output_cost


# 시나리오 예시
print(f"Quick question (Haiku):    ${estimate_cost('haiku', 500, 200):.4f}")
print(f"Code review (Sonnet):      ${estimate_cost('sonnet', 10000, 2000):.4f}")
print(f"Architecture (Opus):       ${estimate_cost('opus', 50000, 5000):.4f}")
print(f"Full codebase scan (Opus): ${estimate_cost('opus', 150000, 10000):.4f}")

# 출력:
# Quick question (Haiku):    $0.0004
# Code review (Sonnet):      $0.0600
# Architecture (Opus):       $1.1250
# Full codebase scan (Opus): $2.9500
```

### 비용 최적화 전략

1. **적절한 모델 사용**: 대부분의 작업에 Opus는 필요하지 않습니다. Sonnet으로 시작하고 필요한 경우에만 업그레이드하세요.
2. **프롬프트 캐싱**: API는 반복적인 시스템 프롬프트와 대용량 컨텍스트에 대한 캐싱을 지원하여 입력 비용을 최대 90%까지 절감합니다.
3. **배치 API**: 시간에 민감하지 않은 워크로드의 경우 배치 API가 50% 할인을 제공합니다.
4. **컨텍스트 최소화**: 스니펫으로 충분할 때 전체 파일 대신 관련 코드와 정보만 전송하세요.
5. **Claude Code의 `/compact` 명령**: 세션 중간에 컨텍스트 크기를 줄이기 위해 대화를 요약합니다.

---

## 7. 어떤 제품을 언제 사용할까

### 결정 매트릭스

| 시나리오 | 추천 제품 | 이유 |
|----------|----------|------|
| "이 오류 이해 도와줘" | Claude Code | 코드베이스와 터미널에 직접 접근 |
| "이 PDF 보고서 분석해줘" | Claude.ai | 파일 업로드 및 아티팩트 생성 |
| "웹 앱 프로토타입 만들어줘" | Claude Desktop | 실시간 렌더링을 위한 앱 미리보기 |
| "이 모듈 리팩토링해줘" | Claude Code | 파일 편집, 테스트 실행, git 통합 |
| "내 제품에 AI 추가하기" | Claude API | 프로그래밍 방식 접근, 커스텀 통합 |
| "빠른 번역 확인" | Claude.ai / Haiku | 빠르고, 비용 낮으며, 설정 불필요 |
| "내 아키텍처 검토해줘" | Claude Code (Opus) | 코드베이스 컨텍스트와 함께 깊은 추론 |
| "문서 1만 건 분류하기" | Claude API (Batch) | 대용량, 비용 효율적인 배치 처리 |
| "내 데이터베이스에 연결하기" | Claude Desktop (MCP) | MCP 서버가 데이터베이스 접근 제공 |

### 워크플로우 통합

소프트웨어 개발자에게 일반적인 일상 워크플로우는 여러 제품을 결합합니다:

```
오전:
  └─ Claude Code: 밤새 CI 실패 검토, 버그 수정

개발:
  └─ Claude Code: 기능, 테스트, 문서 작성

코드 검토:
  └─ Claude Code: PR 검토, 개선 사항 제안

연구:
  └─ Claude.ai: 논문 분석, 설계 옵션 탐색

커뮤니케이션:
  └─ Claude.ai: 기술 문서, 이메일 초안 작성

프로덕션:
  └─ Claude API: 사용자 대면 AI 기능 구동
```

---

## 8. 연습 문제

### 연습 문제 1: 모델 선택

각 시나리오에 대해 어떤 Claude 모델(Opus, Sonnet, Haiku)을 선택할지 결정하고 이유를 설명하세요:

1. 5만 개의 지원 티켓을 카테고리로 분류하기
2. 전자상거래 플랫폼을 위한 마이크로서비스 아키텍처 설계하기
3. 15개 함수가 있는 Python 모듈의 단위 테스트 작성하기
4. 기술 문서를 영어에서 한국어로 번역하기
5. FAQ 질문에 답하는 챗봇 구축하기

### 연습 문제 2: 토큰 추정

Sonnet을 사용하여 다음 시나리오의 토큰 수와 비용을 추정하세요:

1. 200줄 Python 파일을 전송하고 코드 검토 요청하기
2. 파일 3개(각 100줄)를 제공하고 Claude에게 리팩토링 요청하기
3. 20회 대화 교환이 있는 30분 대화형 코딩 세션

### 연습 문제 3: 제품 선택

한 스타트업이 코드 검토 도구를 구축하고 있습니다. 필요 사항:
- GitHub 웹훅으로 트리거되는 자동화된 PR 검토
- 개발자가 코드에 대해 Claude와 채팅할 수 있는 대시보드
- 팀 리더에게 전송되는 주간 아키텍처 보고서

각 구성 요소에 어떤 Claude 제품을 추천하시겠습니까? 비용, 지연 시간, 통합 복잡성을 고려하여 선택을 정당화하세요.

---

## 9. 다음 단계

이 레슨은 Claude의 모델 패밀리, 기능, 제품 생태계에 대한 기초적인 이해를 확립했습니다. 다음 레슨에서는 **Claude Code**를 실제로 다뤄볼 것입니다 — 설치, 첫 번째 세션 실행, 그리고 Claude Code를 강력한 개발 도구로 만드는 핵심 워크플로우를 이해합니다.

**다음**: [02. Claude Code: 시작하기](./02_Claude_Code_Getting_Started.md)
