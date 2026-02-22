# 모델, 가격 및 최적화

**이전**: [18. 커스텀 에이전트 구축](./18_Building_Custom_Agents.md) | **다음**: [20. 고급 개발 워크플로우](./20_Advanced_Workflows.md)

---

Claude의 모델 계층, 가격 구조, 비용 최적화 전략을 이해하는 것은 비용 효율적인 AI 애플리케이션을 구축하는 데 필수적입니다. 이 레슨에서는 각 작업에 맞는 적절한 모델을 선택하고, 캐싱(caching)과 배치(batching)를 활용하여 상당한 비용을 절감하며, 실제 워크플로우에 대한 비용을 추정하는 방법을 종합적으로 안내합니다.

**난이도**: ⭐⭐

**사전 요구 사항**:
- Claude API의 기본 이해 ([레슨 15](./15_Claude_API_Fundamentals.md))
- 토큰 기반 가격 책정 개념에 대한 친숙함
- Claude에 API 호출을 해본 경험

**학습 목표**:
- 지능, 속도, 비용, 기능 면에서 Claude 모델 계층 비교
- 다양한 모델에서 API 사용 비용 계산
- 프롬프트 캐싱(prompt caching)으로 최대 90% 비용 절감 구현
- 시간에 민감하지 않은 작업에 배치 API(Batch API)로 50% 절감
- 다중 계층 아키텍처를 위한 모델 선택 전략 설계
- 일반적인 개발 및 프로덕션 워크플로우의 비용 추정
- 개인 및 팀 사용에 적합한 구독 계획 선택

---

## 목차

1. [Claude 모델 패밀리 개요](#1-claude-모델-패밀리-개요)
2. [모델 기능 비교](#2-모델-기능-비교)
3. [가격 구조](#3-가격-구조)
4. [프롬프트 캐싱](#4-프롬프트-캐싱)
5. [배치 API](#5-배치-api)
6. [모델 선택 전략](#6-모델-선택-전략)
7. [토큰 효율화 기법](#7-토큰-효율화-기법)
8. [구독 계획](#8-구독-계획)
9. [일반적인 워크플로우 비용 추정](#9-일반적인-워크플로우-비용-추정)
10. [연습 문제](#10-연습-문제)

---

## 1. Claude 모델 패밀리 개요

Claude는 지능-속도-비용 스펙트럼에 따라 서로 다른 사용 사례에 최적화된 세 가지 모델 계층으로 제공됩니다.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Claude 모델 패밀리                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────────────────────────────────────┐                 │
│  │  Claude Opus 4                                 │                 │
│  │  "사고하는 모델"                               │                 │
│  │                                               │                 │
│  │  • Claude 패밀리에서 가장 능력 있는 모델       │                 │
│  │  • 복잡한 추론, 수학, 코딩에 최적             │                 │
│  │  • 다단계 분석에서 탁월                       │                 │
│  │  • 이상적: 리서치, 아키텍처 설계,             │                 │
│  │    어려운 디버깅, 세밀한 글쓰기               │                 │
│  └───────────────────────────────────────────────┘                 │
│                                                                     │
│  ┌───────────────────────────────────────────────┐                 │
│  │  Claude Sonnet 4                               │                 │
│  │  "만능 모델"                                  │                 │
│  │                                               │                 │
│  │  • 균형 잡힌 지능과 속도                      │                 │
│  │  • 대부분의 일상적인 코딩 작업에 적합         │                 │
│  │  • 지시 사항 따르기에 강점                   │                 │
│  │  • 이상적: 코드 생성, 리팩토링,               │                 │
│  │    번역, 요약, 분석                           │                 │
│  └───────────────────────────────────────────────┘                 │
│                                                                     │
│  ┌───────────────────────────────────────────────┐                 │
│  │  Claude Haiku                                  │                 │
│  │  "속도 모델"                                  │                 │
│  │                                               │                 │
│  │  • 가장 빠르고 비용 효율적                    │                 │
│  │  • 간단한 고용량 작업에 적합                  │                 │
│  │  • 거의 즉각적인 응답                        │                 │
│  │  • 이상적: 분류, 추출,                        │                 │
│  │    단순 Q&A, 데이터 처리                      │                 │
│  └───────────────────────────────────────────────┘                 │
│                                                                     │
│  지능 ─────▶  Haiku ─── Sonnet ─────── Opus                       │
│  속도 ────────────▶  Opus ──── Sonnet ─────── Haiku               │
│  비용 ─────────────▶  Haiku ─── Sonnet ─────── Opus               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.1 각 모델을 사용하는 경우

**Claude Opus 4** -- 최고의 추론 능력이 필요한 작업:
- 복잡한 다단계 수학 증명
- 여러 트레이드오프가 있는 아키텍처 결정
- 미묘한 동시성 또는 메모리 문제 디버깅
- 깊은 도메인 전문 지식이 필요한 장문 기술 글쓰기
- 속도보다 정확성이 더 중요한 작업

**Claude Sonnet 4** -- 대부분의 개발 작업에 대한 기본 선택:
- 코드 생성 및 리팩토링
- 코드 리뷰 및 버그 식별
- 문서 요약 및 번역
- API 통합 및 보일러플레이트 생성
- Claude Code의 대화형 코딩 세션

**Claude Haiku** -- 대용량, 지연 시간에 민감한 작업:
- 지원 티켓 또는 사용자 의도 분류
- 텍스트에서 구조화된 데이터 추출
- 알려진 콘텐츠에 대한 간단한 Q&A
- 데이터 검증 및 형식 지정
- 속도가 중요한 실시간 사용자 대면 기능

---

## 2. 모델 기능 비교

```
┌──────────────────────┬──────────────┬──────────────┬──────────────┐
│ 기능                 │ Claude Opus 4│Claude Sonnet4│ Claude Haiku │
├──────────────────────┼──────────────┼──────────────┼──────────────┤
│ 지능 수준            │ 최고         │ 높음         │ 양호         │
│ 추론 깊이            │ ★★★★★       │ ★★★★☆       │ ★★★☆☆       │
│ 코딩 능력            │ ★★★★★       │ ★★★★☆       │ ★★★☆☆       │
│ 속도 (토큰/초)       │ 보통         │ 빠름         │ 가장 빠름    │
│ 컨텍스트 창          │ 200K 토큰    │ 200K 토큰    │ 200K 토큰    │
│ 최대 출력 토큰       │ 32,000       │ 16,000       │ 8,192        │
│ 비전 (이미지)        │ Yes          │ Yes          │ Yes          │
│ 확장 사고            │ Yes          │ Yes          │ No           │
│ 도구 사용            │ Yes          │ Yes          │ Yes          │
│ 스트리밍             │ Yes          │ Yes          │ Yes          │
│ 배치 API             │ Yes          │ Yes          │ Yes          │
│ 프롬프트 캐싱        │ Yes          │ Yes          │ Yes          │
├──────────────────────┼──────────────┼──────────────┼──────────────┤
│ 최적 용도            │ 복잡한       │ 일상적인     │ 고용량       │
│                      │ 추론         │ 코딩         │ 간단한 작업  │
└──────────────────────┴──────────────┴──────────────┴──────────────┘
```

### 2.1 컨텍스트 창(Context Window) 심층 분석

모든 Claude 모델은 200K 토큰 컨텍스트 창(약 150,000 단어 또는 500페이지 분량의 텍스트)을 공유합니다. 이를 효과적으로 사용하는 방법을 이해하는 것은 비용 관리에 매우 중요합니다.

```python
import anthropic

client = anthropic.Anthropic()

# 모델 기능을 프로그래밍 방식으로 확인
# 컨텍스트 창은 입력 + 출력 토큰의 합에 적용됨
# 200K 컨텍스트 창의 경우:
#   - 입력 토큰 + 출력 토큰 <= 200,000 (근사치)
#   - 실제 입력 한도는 원하는 출력 길이에 따라 다름

# 예시: 전송 전 토큰 수 추정
def estimate_tokens(text: str) -> int:
    """대략적인 추정: 영어 텍스트는 ~4자당 1토큰."""
    return len(text) // 4

# 200K 컨텍스트 창에 수용 가능한 내용:
examples = {
    "짧은 프롬프트": 50,
    "일반적인 코드 파일 (500줄)": 2_000,
    "전체 프로젝트 README": 1_500,
    "컨텍스트용 소스 파일 10개": 20_000,
    "전체 소규모 코드베이스": 100_000,
    "최대 실용적 입력": 180_000,  # 출력을 위한 공간 남김
}

print("200K 컨텍스트 창의 토큰 예산:")
print(f"{'콘텐츠':<40} {'토큰':>10} {'예산 비율':>12}")
print("-" * 65)
for desc, tokens in examples.items():
    pct = tokens / 200_000 * 100
    print(f"{desc:<40} {tokens:>10,} {pct:>11.1f}%")
```

### 2.2 확장 사고(Extended Thinking)

확장 사고(Extended Thinking)는 Claude가 응답하기 전에 단계별로 추론할 수 있게 하여 복잡한 작업의 성능을 크게 향상시킵니다. Opus 및 Sonnet에서 사용 가능합니다.

```python
import anthropic

client = anthropic.Anthropic()

# 복잡한 추론 작업에 확장 사고 사용
response = client.messages.create(
    model="claude-opus-4-20250514",
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000  # 사고에 최대 10K 토큰 허용
    },
    messages=[{
        "role": "user",
        "content": (
            "Analyze this distributed system design for potential "
            "consistency issues when handling concurrent writes "
            "across three data centers with eventual consistency."
        )
    }]
)

# 응답은 사고 블록과 텍스트 블록을 모두 포함
for block in response.content:
    if block.type == "thinking":
        print(f"[Thinking] ({len(block.thinking)} chars)")
        print(block.thinking[:200] + "...")
    elif block.type == "text":
        print(f"\n[Response]")
        print(block.text)

# 참고: 사고 토큰은 할인된 요율로 청구되지만
# 컨텍스트 창 사용량에 포함됨
```

---

## 3. 가격 구조

Claude API 가격은 모델이 처리하는 텍스트 단위인 토큰을 기반으로 합니다. 입력 토큰(전송하는 것)과 출력 토큰(Claude가 생성하는 것)은 다르게 가격이 책정됩니다.

### 3.1 토큰당 가격

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Claude API 가격 (백만 토큰당)                     │
├──────────────────┬──────────────┬──────────────┬───────────────────┤
│                  │ Claude Opus 4│Claude Sonnet4│ Claude Haiku      │
├──────────────────┼──────────────┼──────────────┼───────────────────┤
│ 입력 토큰        │   $15.00     │    $3.00     │     $0.80         │
│ 출력 토큰        │   $75.00     │   $15.00     │     $4.00         │
├──────────────────┼──────────────┼──────────────┼───────────────────┤
│ 프롬프트 캐싱:   │              │              │                   │
│  캐시 쓰기       │   $18.75     │    $3.75     │     $1.00         │
│  캐시 읽기       │    $1.50     │    $0.30     │     $0.08         │
├──────────────────┼──────────────┼──────────────┼───────────────────┤
│ 배치 API:        │              │              │                   │
│  입력 토큰       │    $7.50     │    $1.50     │     $0.40         │
│  출력 토큰       │   $37.50     │    $7.50     │     $2.00         │
└──────────────────┴──────────────┴──────────────┴───────────────────┘

참고:
- 백만 토큰 (MTok) ≈ 75만 단어 ≈ 2,500페이지 텍스트
- 출력 토큰은 입력 토큰의 5배 비용 (모델이 더 신중하게 생성)
- 프롬프트 캐싱: 첫 번째 쓰기는 1.25배, 이후 읽기는 0.1배
- 배치 API: 모든 토큰 비용 50% 할인
```

### 3.2 토큰 비용 이해하기

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelPricing:
    """Claude 모델의 백만 토큰당 가격."""
    name: str
    input_per_mtok: float
    output_per_mtok: float
    cache_write_per_mtok: float
    cache_read_per_mtok: float
    batch_input_per_mtok: float
    batch_output_per_mtok: float

# 각 모델의 가격 정의
PRICING = {
    "opus": ModelPricing(
        name="Claude Opus 4",
        input_per_mtok=15.00,
        output_per_mtok=75.00,
        cache_write_per_mtok=18.75,
        cache_read_per_mtok=1.50,
        batch_input_per_mtok=7.50,
        batch_output_per_mtok=37.50,
    ),
    "sonnet": ModelPricing(
        name="Claude Sonnet 4",
        input_per_mtok=3.00,
        output_per_mtok=15.00,
        cache_write_per_mtok=3.75,
        cache_read_per_mtok=0.30,
        batch_input_per_mtok=1.50,
        batch_output_per_mtok=7.50,
    ),
    "haiku": ModelPricing(
        name="Claude Haiku",
        input_per_mtok=0.80,
        output_per_mtok=4.00,
        cache_write_per_mtok=1.00,
        cache_read_per_mtok=0.08,
        batch_input_per_mtok=0.40,
        batch_output_per_mtok=2.00,
    ),
}


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
    cache_write_tokens: int = 0,
    use_batch: bool = False,
) -> dict:
    """Claude API 호출 비용 계산.

    Args:
        model: "opus", "sonnet", "haiku" 중 하나
        input_tokens: 캐시되지 않은 입력 토큰 수
        output_tokens: 출력 토큰 수
        cached_input_tokens: 캐시에서 읽은 토큰 수
        cache_write_tokens: 캐시에 쓴 토큰 수
        use_batch: 배치 API 가격 사용 여부

    Returns:
        비용 분류가 담긴 딕셔너리
    """
    pricing = PRICING[model]

    if use_batch:
        input_cost = (input_tokens / 1_000_000) * pricing.batch_input_per_mtok
        output_cost = (output_tokens / 1_000_000) * pricing.batch_output_per_mtok
        # 캐싱은 일반적으로 배치와 함께 사용하지 않지만, 완전성을 위해:
        cache_read_cost = 0
        cache_write_cost = 0
    else:
        input_cost = (input_tokens / 1_000_000) * pricing.input_per_mtok
        output_cost = (output_tokens / 1_000_000) * pricing.output_per_mtok
        cache_read_cost = (cached_input_tokens / 1_000_000) * pricing.cache_read_per_mtok
        cache_write_cost = (cache_write_tokens / 1_000_000) * pricing.cache_write_per_mtok

    total = input_cost + output_cost + cache_read_cost + cache_write_cost

    return {
        "model": pricing.name,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "cache_read_cost": cache_read_cost,
        "cache_write_cost": cache_write_cost,
        "total_cost": total,
        "batch_discount": use_batch,
    }


# 예시: 일반적인 코딩 작업에 대한 모델별 비용 비교
# ~2000 입력 토큰 (프롬프트 + 코드 컨텍스트), ~1000 출력 토큰
print("비용 비교: 일반적인 코딩 작업 (입력 2K, 출력 1K)")
print("=" * 60)
for model in ["opus", "sonnet", "haiku"]:
    result = calculate_cost(model, input_tokens=2000, output_tokens=1000)
    print(f"\n{result['model']}:")
    print(f"  입력:  ${result['input_cost']:.4f}")
    print(f"  출력: ${result['output_cost']:.4f}")
    print(f"  합계:  ${result['total_cost']:.4f}")

# 예시: 캐싱을 사용한 대용량 컨텍스트
print("\n\n캐싱을 사용한 대용량 컨텍스트 (캐시된 50K, 새 입력 2K, 출력 1K)")
print("=" * 60)
for model in ["opus", "sonnet", "haiku"]:
    # 캐싱 없이: 52K 토큰 모두 입력
    no_cache = calculate_cost(model, input_tokens=52000, output_tokens=1000)
    # 캐싱 사용: 캐시된 50K (읽기), 새 입력 2K
    with_cache = calculate_cost(
        model,
        input_tokens=2000,
        output_tokens=1000,
        cached_input_tokens=50000,
    )
    savings = (1 - with_cache["total_cost"] / no_cache["total_cost"]) * 100

    print(f"\n{PRICING[model].name}:")
    print(f"  캐싱 없음: ${no_cache['total_cost']:.4f}")
    print(f"  캐싱 사용:    ${with_cache['total_cost']:.4f}")
    print(f"  절감:       {savings:.1f}%")
```

### 3.3 모델 간 비용 비율

가격을 관점에서 이해하면:

```
동일한 작업에 대한 상대적 비용:

  Opus    ████████████████████████████████████████  $1.00 (기준)
  Sonnet  ████████                                  $0.20 (5배 저렴)
  Haiku   ██                                        $0.05 (20배 저렴)

Opus 1회 호출 비용으로:
  - Sonnet 5회, 또는
  - Haiku 20회 호출 가능

이를 통해 모델 선택이 가장 큰 비용 절감 수단 중 하나임을 알 수 있습니다.
```

---

## 4. 프롬프트 캐싱

프롬프트 캐싱(Prompt Caching)은 자주 사용되는 컨텍스트(시스템 프롬프트, 대용량 문서, 코드 파일)를 캐시하고 여러 API 호출에 걸쳐 재사용할 수 있게 합니다. 캐시된 토큰은 일반 입력 토큰에 비해 90% 할인된 요율로 읽힙니다.

### 4.1 프롬프트 캐싱 작동 방식

```
┌─────────────────────────────────────────────────────────────────────┐
│                    프롬프트 캐싱 흐름                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  첫 번째 요청 (캐시 미스 → 캐시 쓰기):                              │
│  ┌──────────────────────────────────────────────────┐              │
│  │  시스템 프롬프트 (2K 토큰)     ← cache_control   │              │
│  │  대용량 문서 (50K 토큰)        ← cache_control   │              │
│  │  사용자 메시지 (500 토큰)      ← 캐시 안 됨      │              │
│  └──────────────────────────────────────────────────┘              │
│  비용: 52K × 쓰기_가격 + 500 × 입력_가격 + 출력_가격              │
│                                                                     │
│  이후 요청 (캐시 히트 → 캐시 읽기):                                 │
│  ┌──────────────────────────────────────────────────┐              │
│  │  시스템 프롬프트 (2K 토큰)     ← 캐시 히트 ✓    │              │
│  │  대용량 문서 (50K 토큰)        ← 캐시 히트 ✓    │              │
│  │  사용자 메시지 (800 토큰)      ← 캐시 안 됨      │              │
│  └──────────────────────────────────────────────────┘              │
│  비용: 52K × 읽기_가격 + 800 × 입력_가격 + 출력_가격              │
│                                                                     │
│  캐시 TTL: 5분 (각 캐시 히트 시 갱신)                              │
│  최소 캐시 가능: 1,024 토큰 (더 짧은 콘텐츠는 캐싱 가치 없음)     │
│  캐시 키: 정확한 접두사 일치 (변경 시 무효화)                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Python 구현

```python
import anthropic

client = anthropic.Anthropic()

# 캐싱할 가치가 있는 대용량 시스템 프롬프트 (프로젝트 컨텍스트)
SYSTEM_PROMPT = """You are an expert Python developer working on our e-commerce
platform. The codebase uses FastAPI, SQLAlchemy, and PostgreSQL.

Here are the key architectural decisions:
... (imagine 2000+ tokens of context here) ...

Follow these coding standards:
- PEP 8 style
- Type hints on all functions
- Docstrings for public methods
- 90% test coverage minimum
"""

# 캐싱할 가치가 있는 대용량 참조 문서
API_SPEC = """
OpenAPI 3.0 Specification for our REST API:
... (imagine 10,000+ tokens of API spec here) ...
"""

def query_with_caching(user_message: str) -> str:
    """시스템 프롬프트와 API 스펙에 프롬프트 캐싱을 사용하여 쿼리 전송."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"}  # 이 블록 캐시
            }
        ],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": API_SPEC,
                        "cache_control": {"type": "ephemeral"}  # 이 블록 캐시
                    },
                    {
                        "type": "text",
                        "text": user_message  # 요청마다 달라짐
                    }
                ]
            }
        ],
    )

    # 사용 통계에서 캐시 성능 확인
    usage = response.usage
    print(f"입력 토큰:        {usage.input_tokens}")
    print(f"캐시 생성:      {getattr(usage, 'cache_creation_input_tokens', 0)}")
    print(f"캐시 읽기:          {getattr(usage, 'cache_read_input_tokens', 0)}")
    print(f"출력 토큰:       {usage.output_tokens}")

    return response.content[0].text


# 첫 번째 호출: 캐시 미스 (캐시에 씀)
print("=== 첫 번째 호출 (캐시 쓰기) ===")
result1 = query_with_caching("Add a new endpoint GET /api/v1/products/{id}/reviews")
# 출력: 캐시 생성: ~12000, 캐시 읽기: 0

# 5분 내 두 번째 호출: 캐시 히트 (캐시에서 읽음)
print("\n=== 두 번째 호출 (캐시 히트) ===")
result2 = query_with_caching("Now add pagination to the reviews endpoint")
# 출력: 캐시 생성: 0, 캐시 읽기: ~12000 (90% 더 저렴!)

# 세 번째 호출: 여전히 캐시 히트 (두 번째 호출로 TTL 갱신)
print("\n=== 세 번째 호출 (캐시 히트) ===")
result3 = query_with_caching("Add rate limiting to the reviews endpoint")
# 출력: 캐시 생성: 0, 캐시 읽기: ~12000
```

### 4.3 TypeScript 구현

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

const SYSTEM_CONTEXT = `You are a senior TypeScript developer...
${/* Imagine 2000+ tokens of context */ ""}`;

const CODE_BASE_CONTEXT = `// Current codebase structure:
${/* Imagine 10,000+ tokens of code */ ""}`;

async function queryWithCache(userMessage: string): Promise<string> {
  const response = await client.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 4096,
    system: [
      {
        type: "text",
        text: SYSTEM_CONTEXT,
        cache_control: { type: "ephemeral" },
      },
    ],
    messages: [
      {
        role: "user",
        content: [
          {
            type: "text",
            text: CODE_BASE_CONTEXT,
            cache_control: { type: "ephemeral" },
          },
          {
            type: "text",
            text: userMessage,
          },
        ],
      },
    ],
  });

  // 캐시 성능 로깅
  const usage = response.usage as any;
  console.log(`캐시 쓰기: ${usage.cache_creation_input_tokens ?? 0}`);
  console.log(`캐시 읽기:  ${usage.cache_read_input_tokens ?? 0}`);

  const textBlock = response.content.find((b) => b.type === "text");
  return textBlock?.text ?? "";
}
```

### 4.4 캐싱 모범 사례

```
┌─────────────────────────────────────────────────────────────────────┐
│                    프롬프트 캐싱 모범 사례                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  해야 할 것:                                                         │
│  ✓ 대용량의 안정적인 컨텍스트 캐시 (시스템 프롬프트, 참조 문서)   │
│  ✓ 캐시된 콘텐츠를 변하는 콘텐츠 앞에 배치 (접두사 매칭)          │
│  ✓ 1,024 토큰 이상의 콘텐츠 캐시 (캐싱 이익을 위한 최솟값)        │
│  ✓ 5분 TTL 창 내에서 캐시 재사용                                   │
│  ✓ 프롬프트 구조화: [캐시된 접두사] + [변하는 접미사]              │
│  ✓ 프로덕션에서 캐시 히트율 모니터링                               │
│                                                                     │
│  하지 말아야 할 것:                                                  │
│  ✗ 매 요청마다 변하는 콘텐츠 캐시                                  │
│  ✗ 캐시된 콘텐츠 앞에 변하는 콘텐츠 배치 (접두사 깨짐)            │
│  ✗ 작은 프롬프트 캐시 (< 1,024 토큰) — 오버헤드가 가치 없음       │
│  ✗ 캐시가 영원히 지속된다고 가정 (5분 TTL)                         │
│  ✗ 동일한 캐시로 다른 모델 사용 (캐시 분리됨)                      │
│                                                                     │
│  비용 계산:                                                          │
│  - 쓰기: 일반 입력 가격의 1.25배 (첫 번째 호출만)                  │
│  - 읽기:  일반 입력 가격의 0.10배 (이후 호출)                      │
│  - 손익분기점: 2회 읽기로 쓰기 오버헤드 상쇄                       │
│  - 10회 읽기: 캐시된 부분에서 ~87% 절감                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. 배치 API

배치 API(Batch API)는 50% 할인으로 대량의 요청을 전송할 수 있게 합니다. 배치는 24시간 이내에 비동기적으로 처리되므로 시간에 민감하지 않은 워크로드에 이상적입니다.

### 5.1 배치 작동 방식

```
┌─────────────────────────────────────────────────────────────────────┐
│                    배치 API 워크플로우                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. 여러 요청으로 배치 생성                                          │
│     ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐              │
│     │요청 1   │ │요청 2   │ │요청 3   │ │요청 N   │              │
│     └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘              │
│          └───────────┴───────────┴───────────┘                     │
│                          │                                          │
│  2. 배치 제출 ───────────▼─────────────────────────                │
│     POST /v1/messages/batches                                      │
│     상태: "in_progress"                                            │
│                          │                                          │
│  3. 처리 ────────────────▼─────────────────────────                │
│     (최대 24시간, 보통 훨씬 빠름)                                   │
│                          │                                          │
│  4. 결과 준비 ───────────▼─────────────────────────                │
│     상태: "ended"                                                  │
│     ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐              │
│     │결과 1   │ │결과 2   │ │결과 3   │ │결과 N   │              │
│     └─────────┘ └─────────┘ └─────────┘ └─────────┘              │
│                                                                     │
│  가격: 표준 API 요율의 50%                                          │
│  배치당 최대 요청 수: 10,000                                        │
│  처리 창: 최대 24시간                                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 배치 생성

```python
import anthropic
import json
import time

client = anthropic.Anthropic()

# 배치 요청 준비 — 각각은 표준 Messages API 호출
# 추적을 위한 custom_id로 래핑
batch_requests = []

# 예시: 100개의 제품 설명 번역
product_descriptions = [
    {"id": f"product_{i}", "text": f"Sample product description #{i}"}
    for i in range(100)
]

for product in product_descriptions:
    batch_requests.append({
        "custom_id": product["id"],
        "params": {
            "model": "claude-haiku-3-5-20241022",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Translate this product description to Korean. "
                        f"Return only the translation:\n\n{product['text']}"
                    )
                }
            ]
        }
    })

# 배치 생성
batch = client.messages.batches.create(requests=batch_requests)

print(f"배치 생성됨: {batch.id}")
print(f"상태: {batch.processing_status}")
print(f"요청 수: {batch.request_counts}")
```

### 5.3 결과 폴링

```python
import anthropic
import time

client = anthropic.Anthropic()

def wait_for_batch(batch_id: str, poll_interval: int = 30) -> None:
    """완료될 때까지 배치 상태 폴링."""
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status

        succeeded = batch.request_counts.succeeded
        errored = batch.request_counts.errored
        processing = batch.request_counts.processing
        total = succeeded + errored + processing

        print(
            f"상태: {status} | "
            f"성공: {succeeded}/{total} | "
            f"오류: {errored}"
        )

        if status == "ended":
            print("배치 처리 완료!")
            break

        time.sleep(poll_interval)

    return batch


def retrieve_results(batch_id: str) -> list:
    """완료된 배치에서 모든 결과 검색."""
    results = []
    for result in client.messages.batches.results(batch_id):
        results.append({
            "custom_id": result.custom_id,
            "type": result.result.type,
            "message": (
                result.result.message.content[0].text
                if result.result.type == "succeeded"
                else None
            ),
            "error": (
                result.result.error
                if result.result.type == "errored"
                else None
            ),
        })
    return results


# 사용 예
batch_id = "batch_abc123"  # 생성 단계에서 가져옴
batch = wait_for_batch(batch_id)
results = retrieve_results(batch_id)

# 결과 처리
succeeded = [r for r in results if r["type"] == "succeeded"]
failed = [r for r in results if r["type"] == "errored"]

print(f"\n결과: {len(succeeded)}개 성공, {len(failed)}개 실패")
for result in succeeded[:3]:
    print(f"\n[{result['custom_id']}]: {result['message'][:100]}...")
```

### 5.4 배치 API 사용 사례

```
배치 API에 적합한 사례:                    부적합한 사례:
─────────────────────────────              ─────────────────
✓ 대량 콘텐츠 번역                          ✗ 실시간 채팅
✓ 데이터셋 레이블링 / 분류                  ✗ 대화형 코딩
✓ 테스트 데이터 생성                        ✗ 사용자 대면 기능
✓ 야간 코드 분석                            ✗ CI/CD 블로킹 단계
✓ 문서 요약 파이프라인                      ✗ 시간에 민감한 경보
✓ 월간 보고서 생성                          ✗ 스트리밍 응답
✓ 콘텐츠 모더레이션 백로그
✓ 임베딩 생성
```

---

## 6. 모델 선택 전략

### 6.1 계층화 아키텍처

가장 비용 효율적인 접근 방식은 복잡도에 따라 적절한 모델로 작업을 라우팅하는 다중 모델 계층화 아키텍처를 사용하는 것입니다.

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import anthropic

class TaskComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"

@dataclass
class ModelConfig:
    model_id: str
    max_tokens: int
    temperature: float = 0.0

# 모델 구성
MODEL_TIERS = {
    TaskComplexity.SIMPLE: ModelConfig(
        model_id="claude-haiku-3-5-20241022",
        max_tokens=1024,
    ),
    TaskComplexity.MODERATE: ModelConfig(
        model_id="claude-sonnet-4-20250514",
        max_tokens=4096,
    ),
    TaskComplexity.COMPLEX: ModelConfig(
        model_id="claude-opus-4-20250514",
        max_tokens=8192,
    ),
}


def classify_task_complexity(task_description: str) -> TaskComplexity:
    """사용할 모델 계층을 결정하기 위해 작업 분류.

    이것은 단순화된 휴리스틱입니다. 프로덕션에서는
    라우팅 전에 소형 모델(Haiku)로 작업을 분류할 수 있습니다.
    """
    complex_indicators = [
        "architect", "design system", "debug concurrency",
        "mathematical proof", "security audit", "trade-offs",
        "multi-step reasoning", "analyze entire codebase",
    ]
    moderate_indicators = [
        "refactor", "implement", "write tests", "code review",
        "generate", "convert", "translate", "summarize",
    ]

    task_lower = task_description.lower()

    if any(indicator in task_lower for indicator in complex_indicators):
        return TaskComplexity.COMPLEX
    elif any(indicator in task_lower for indicator in moderate_indicators):
        return TaskComplexity.MODERATE
    else:
        return TaskComplexity.SIMPLE


def route_to_model(task: str, user_message: str) -> str:
    """작업을 적절한 모델 계층으로 라우팅."""
    client = anthropic.Anthropic()
    complexity = classify_task_complexity(task)
    config = MODEL_TIERS[complexity]

    print(f"작업: {task}")
    print(f"복잡도: {complexity.value} → 모델: {config.model_id}")

    response = client.messages.create(
        model=config.model_id,
        max_tokens=config.max_tokens,
        messages=[{"role": "user", "content": user_message}],
    )

    return response.content[0].text


# 라우팅 결정 예시
tasks = [
    ("Extract the email from this text", "simple"),
    ("Refactor this function to use async/await", "moderate"),
    ("Design a distributed cache with consistency guarantees", "complex"),
]

for task, expected in tasks:
    complexity = classify_task_complexity(task)
    model = MODEL_TIERS[complexity]
    print(f"  '{task}'")
    print(f"    → {complexity.value} ({expected}) → {model.model_id}")
    print()
```

### 6.2 캐스케이드(Cascade) 패턴

더 저렴한 모델부터 시작하고, 결과가 만족스럽지 않을 때만 더 능력 있는 모델로 에스컬레이션하세요.

```python
import anthropic
import json

client = anthropic.Anthropic()

def cascade_query(
    user_message: str,
    system_prompt: str = "",
    validation_fn=None,
) -> dict:
    """더 저렴한 모델을 먼저 시도하고 품질이 부족하면 에스컬레이션.

    Args:
        user_message: 사용자의 요청
        system_prompt: 선택적 시스템 프롬프트
        validation_fn: 응답을 검증하는 선택적 함수.
                       응답이 허용 가능하면 True 반환.
    """
    models = [
        ("claude-haiku-3-5-20241022", 1024),    # 가장 저렴한 것부터 시도
        ("claude-sonnet-4-20250514", 4096),      # 중간 계층으로 에스컬레이션
        ("claude-opus-4-20250514", 8192),        # 최종 에스컬레이션
    ]

    for model_id, max_tokens in models:
        print(f"{model_id} 시도 중...")

        kwargs = {
            "model": model_id,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": user_message}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = client.messages.create(**kwargs)
        result = response.content[0].text

        # 검증 함수가 없으면 첫 번째 결과 수락
        if validation_fn is None:
            return {"model": model_id, "response": result, "escalations": 0}

        # 응답 검증
        if validation_fn(result):
            return {"model": model_id, "response": result}
        else:
            print(f"  → {model_id}의 응답이 검증 실패, 에스컬레이션...")

    # 모든 모델 시도 후 마지막 결과 반환
    return {"model": models[-1][0], "response": result, "note": "max escalation"}


# 예시: 코드 생성 응답이 컴파일되는지 검증
def validate_json_output(response: str) -> bool:
    """응답에 유효한 JSON이 포함되어 있는지 확인."""
    try:
        # 응답에서 JSON 추출 시도
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            json.loads(response[start:end])
            return True
    except json.JSONDecodeError:
        pass
    return False


result = cascade_query(
    user_message="Generate a JSON schema for a User object with name, email, age, and roles fields.",
    validation_fn=validate_json_output,
)
print(f"\n최종 사용 모델: {result['model']}")
print(f"응답: {result['response'][:200]}...")
```

---

## 7. 토큰 효율화 기법

토큰 사용량을 줄이면 비용이 직접 줄어듭니다. 다음은 더 효율적인 프롬프트를 작성하기 위한 실용적인 기법들입니다.

### 7.1 간결한 프롬프트

```python
# 나쁜 예: 장황한 프롬프트 (불필요한 단어로 토큰 낭비)
verbose_prompt = """
I would really appreciate it if you could help me with something.
I have a Python function that I need you to look at. What I'm hoping
you can do is review the code and tell me if there are any bugs or
issues with it. The function is supposed to calculate the factorial
of a number. Could you please take a look at it and let me know
what you think? Here is the code:

def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)

Thank you so much for your help with this!
"""
# ~100 토큰

# 좋은 예: 간결한 프롬프트 (동일한 정보, 더 적은 토큰)
concise_prompt = """
Review this factorial function for bugs:

```python
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)
```

List any issues found.
"""
# ~40 토큰 (60% 감소)
```

### 7.2 구조화된 출력 요청

```python
# 구조화된 출력을 요청하면 응답에서 불필요한 산문을 줄임
import anthropic

client = anthropic.Anthropic()

# 나쁜 예: 비구조화된 요청 → 길고 장황한 응답
bad_prompt = "What do you think about this code? Is it good or bad?"

# 좋은 예: 구조화된 요청 → 간결하고 실행 가능한 응답
good_prompt = """Review this code. Respond in this exact JSON format:
{
  "issues": [{"line": N, "severity": "high|medium|low", "description": "..."}],
  "suggestions": ["..."],
  "overall_rating": "good|acceptable|needs_work"
}"""

# 좋은 예: 사전 채워진 어시스턴트 응답을 사용하여 출력 제약
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=2048,
    messages=[
        {"role": "user", "content": good_prompt + "\n\n```python\ndef add(a, b): return a + b\n```"},
        {"role": "assistant", "content": "{"}  # JSON 출력 강제
    ],
)
```

### 7.3 컨텍스트 정리(Context Pruning)

```python
def prune_context_for_task(files: dict, task: str) -> dict:
    """작업에 관련된 파일만 컨텍스트에 포함.

    전체 코드베이스를 전송하는 대신, 현재 작업과
    관련된 파일만 전송하세요.
    """
    # 간단한 키워드 기반 관련성 점수화
    task_keywords = set(task.lower().split())
    scored_files = []

    for filepath, content in files.items():
        # 키워드 겹침 기반 점수화
        file_words = set(filepath.lower().replace("/", " ").replace("_", " ").split())
        content_sample = set(content[:500].lower().split())
        overlap = len(task_keywords & (file_words | content_sample))
        scored_files.append((overlap, filepath, content))

    # 관련성별 정렬 후 상위 파일 선택
    scored_files.sort(reverse=True)
    relevant = {fp: content for score, fp, content in scored_files[:5] if score > 0}

    total_tokens_saved = sum(
        len(content) // 4
        for score, fp, content in scored_files[5:]
        if score == 0
    )
    print(f"포함된 파일: {len(relevant)}/{len(files)}")
    print(f"예상 절감 토큰: ~{total_tokens_saved:,}")

    return relevant
```

---

## 8. 구독 계획

claude.ai 및 Claude 앱(API가 아닌)을 통한 대화형 사용에 대해 Anthropic은 구독 계획을 제공합니다.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Claude 구독 계획                                  │
├─────────────────┬──────────────────────────────────────────────────┤
│                 │                                                    │
│  Claude Free    │  월 $0                                            │
│                 │  • Claude Sonnet 접근 (제한적)                    │
│                 │  • 기본 대화 기능                                  │
│                 │  • 일일 메시지 제한                                │
│                 │  • 트래픽 많을 때 우선 접근 없음                  │
│                 │                                                    │
├─────────────────┼──────────────────────────────────────────────────┤
│                 │                                                    │
│  Claude Pro     │  월 $20                                           │
│                 │  • Opus, Sonnet, Haiku 접근                       │
│                 │  • 무료 계층보다 5배 더 많은 사용량               │
│                 │  • 트래픽 많을 때 우선 접근                       │
│                 │  • Claude Projects 접근                           │
│                 │  • 확장 사고 모드                                 │
│                 │  • 새 기능 조기 접근                              │
│                 │                                                    │
├─────────────────┼──────────────────────────────────────────────────┤
│                 │                                                    │
│  Claude Team    │  사용자당 월 $25 (최소 5명)                       │
│                 │  • Pro의 모든 기능, 추가:                         │
│                 │  • Pro보다 높은 사용량 한도                       │
│                 │  • 공유 Projects가 있는 팀 워크스페이스           │
│                 │  • 사용량 모니터링을 위한 관리자 대시보드         │
│                 │  • 모든 모델에 500K 컨텍스트 창                  │
│                 │  • 학습에 데이터 사용 안 함                       │
│                 │                                                    │
├─────────────────┼──────────────────────────────────────────────────┤
│                 │                                                    │
│  Claude         │  맞춤 가격 (영업팀 문의)                          │
│  Enterprise     │  • Team의 모든 기능, 추가:                        │
│                 │  • SSO (SAML) 및 SCIM 프로비저닝                 │
│                 │  • 도메인 캡처 및 관리자 컨트롤                   │
│                 │  • 감사 로그 및 규정 준수 기능                    │
│                 │  • 맞춤 데이터 보존 정책                          │
│                 │  • 전용 지원 및 SLA                               │
│                 │  • 더 높은 속도 제한                              │
│                 │  • 학습에 데이터 사용 안 함                       │
│                 │                                                    │
└─────────────────┴──────────────────────────────────────────────────┘

참고: API 사용은 토큰당 가격을 기반으로 별도 청구됩니다.
구독 계획은 claude.ai 및 Claude 앱의 대화형 사용을 포함합니다.
Claude Code (CLI)는 구독 할당량이 아닌 API 크레딧을 사용합니다.
```

---

## 9. 일반적인 워크플로우 비용 추정

### 9.1 일일 개발자 워크플로우

```python
def estimate_daily_developer_cost():
    """Claude Code를 사용하는 일반적인 개발자의 일일 API 비용 추정."""

    # 일반적인 일일 사용 패턴
    daily_activities = [
        {
            "activity": "아침 코드 리뷰 (파일 3개)",
            "model": "sonnet",
            "input_tokens": 15000,   # 파일 3개 × 각 ~5K 토큰
            "output_tokens": 3000,    # 리뷰 댓글
            "calls": 3,
        },
        {
            "activity": "기능 구현 (5번 반복)",
            "model": "sonnet",
            "input_tokens": 8000,    # 반복당 컨텍스트 + 프롬프트
            "output_tokens": 4000,   # 생성된 코드
            "calls": 5,
        },
        {
            "activity": "버그 디버깅 세션",
            "model": "opus",          # 복잡한 디버깅 → Opus
            "input_tokens": 20000,   # 디버깅을 위한 대용량 컨텍스트
            "output_tokens": 5000,   # 상세 분석
            "calls": 2,
        },
        {
            "activity": "테스트 생성",
            "model": "sonnet",
            "input_tokens": 5000,
            "output_tokens": 8000,   # 테스트는 보통 더 길음
            "calls": 3,
        },
        {
            "activity": "빠른 질문 / 조회",
            "model": "haiku",         # 간단한 질문 → Haiku
            "input_tokens": 1000,
            "output_tokens": 500,
            "calls": 10,
        },
    ]

    total_cost = 0
    print("일일 개발자 비용 추정")
    print("=" * 70)
    print(f"{'활동':<40} {'모델':<10} {'호출':>5} {'비용':>8}")
    print("-" * 70)

    for activity in daily_activities:
        pricing = PRICING[activity["model"]]
        per_call_cost = (
            (activity["input_tokens"] / 1_000_000) * pricing.input_per_mtok +
            (activity["output_tokens"] / 1_000_000) * pricing.output_per_mtok
        )
        activity_cost = per_call_cost * activity["calls"]
        total_cost += activity_cost

        print(
            f"{activity['activity']:<40} "
            f"{activity['model']:<10} "
            f"{activity['calls']:>5} "
            f"${activity_cost:>7.2f}"
        )

    print("-" * 70)
    print(f"{'일일 합계':<40} {'':>10} {'':>5} ${total_cost:>7.2f}")
    print(f"{'월간 (22 근무일)':<40} {'':>10} {'':>5} ${total_cost * 22:>7.2f}")
    print(f"{'연간 추정':<40} {'':>10} {'':>5} ${total_cost * 260:>7.2f}")

    # 최적화 유무 비교
    print(f"\n프롬프트 캐싱 사용 시 (추정 40% 절감): ${total_cost * 22 * 0.6:.2f}/월")
    print(f"모델 계층화 사용 시 (추정 30% 절감):  ${total_cost * 22 * 0.7:.2f}/월")
    print(f"두 최적화 모두 사용 시:                ${total_cost * 22 * 0.42:.2f}/월")

estimate_daily_developer_cost()
```

### 9.2 프로덕션 파이프라인 비용

```python
def estimate_production_pipeline_cost():
    """프로덕션 AI 파이프라인의 월간 비용 추정."""

    monthly_volumes = {
        "고객 지원 분류": {
            "model": "haiku",
            "monthly_requests": 100_000,
            "avg_input_tokens": 500,
            "avg_output_tokens": 50,
        },
        "콘텐츠 모더레이션": {
            "model": "haiku",
            "monthly_requests": 500_000,
            "avg_input_tokens": 200,
            "avg_output_tokens": 30,
        },
        "문서 요약": {
            "model": "sonnet",
            "monthly_requests": 10_000,
            "avg_input_tokens": 5000,
            "avg_output_tokens": 500,
        },
        "코드 리뷰 자동화": {
            "model": "sonnet",
            "monthly_requests": 5_000,
            "avg_input_tokens": 10000,
            "avg_output_tokens": 2000,
        },
        "아키텍처 분석": {
            "model": "opus",
            "monthly_requests": 500,
            "avg_input_tokens": 30000,
            "avg_output_tokens": 5000,
        },
    }

    total_monthly = 0
    print("프로덕션 파이프라인 월간 비용 추정")
    print("=" * 80)
    print(f"{'파이프라인':<35} {'모델':<8} {'요청 수':>10} {'비용':>10}")
    print("-" * 80)

    for pipeline, config in monthly_volumes.items():
        pricing = PRICING[config["model"]]
        total_input = config["monthly_requests"] * config["avg_input_tokens"]
        total_output = config["monthly_requests"] * config["avg_output_tokens"]

        cost = (
            (total_input / 1_000_000) * pricing.input_per_mtok +
            (total_output / 1_000_000) * pricing.output_per_mtok
        )
        total_monthly += cost

        print(
            f"{pipeline:<35} "
            f"{config['model']:<8} "
            f"{config['monthly_requests']:>10,} "
            f"${cost:>9.2f}"
        )

    print("-" * 80)
    print(f"{'월간 합계':<35} {'':>8} {'':>10} ${total_monthly:>9.2f}")
    print(f"\n비실시간에 배치 API 사용 시 (추정): ${total_monthly * 0.65:>9.2f}")
    print(f"반복 컨텍스트에 캐싱 사용 시:      ${total_monthly * 0.50:>9.2f}")

estimate_production_pipeline_cost()
```

---

## 10. 연습 문제

### 연습 1: 비용 계산 (초급)

가격표를 사용하여 각 시나리오의 비용을 계산하세요:

1. 입력 토큰 10,000개와 출력 토큰 2,000개로 Opus API를 단 1회 호출.
2. 각각 입력 토큰 500개와 출력 토큰 100개로 Haiku API를 1,000회 호출.
3. 캐시된 입력 토큰 50,000개, 새 입력 토큰 2,000개, 출력 토큰 1,000개로 Sonnet 호출.
4. 각각 입력 토큰 3,000개와 출력 토큰 1,000개로 Sonnet 요청 500개를 배치 처리.

### 연습 2: 모델 선택 (중급)

아래의 각 작업에 대해 최적의 모델을 추천하고 이유를 설명하세요:

1. 50,000개의 고객 이메일을 5가지 카테고리로 분류
2. 분산 시스템의 경쟁 조건(race condition) 디버깅
3. CRUD API에 대한 단위 테스트 생성
4. 200페이지 기술 매뉴얼 번역
5. 새로운 애플리케이션을 위한 데이터베이스 스키마 설계

### 연습 3: 캐싱 전략 (중급)

시간당 100번의 API 호출을 수행하는 애플리케이션이 있습니다. 모든 호출이 동일한 30,000 토큰 시스템 프롬프트를 공유하지만 서로 다른 500 토큰 사용자 메시지를 가집니다. 다음을 계산하세요:

1. 캐싱 없는 시간당 비용 (Sonnet 사용)
2. 프롬프트 캐싱 사용 시 시간당 비용
3. 캐싱으로 인한 월간 절감액 (하루 8시간, 월 22일 기준)
4. 호출 간격이 5분 이상인 경우 어떤 일이 발생하는가?

### 연습 4: 비용 대시보드 구축 (고급)

다음을 수행하는 Python 스크립트를 작성하세요:
1. API 사용 로그를 읽습니다 (생성된 데이터로 시뮬레이션)
2. 모델별, 일별, 엔드포인트별 비용을 계산합니다
3. 상위 3개 비용 원인을 식별합니다
4. 최적화 기회를 추천합니다 (모델 다운그레이드, 캐싱 후보)

### 연습 5: 계층화 아키텍처 설계 (고급)

다음을 처리하는 고객 지원 애플리케이션을 위한 모델 라우팅 시스템을 설계하세요:
- 간단한 FAQ 응답 (쿼리의 70%)
- 제품 문제 해결 (쿼리의 20%)
- 복잡한 불만 해결 (쿼리의 10%)

각 계층을 처리하는 모델을 지정하고, 월간 50,000개 쿼리에 대한 월간 비용을 추정하며, Sonnet만 사용하는 단일 모델 접근 방식과 비교하세요.

---

## 참고 자료

- Anthropic 가격 - https://www.anthropic.com/pricing
- 프롬프트 캐싱 문서 - https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
- 배치 API 문서 - https://docs.anthropic.com/en/docs/build-with-claude/message-batches
- Claude 모델 개요 - https://docs.anthropic.com/en/docs/about-claude/models

---

## 다음 단계

[20. 고급 개발 워크플로우](./20_Advanced_Workflows.md)에서는 다중 파일 리팩토링, Claude를 사용한 TDD, CI/CD 통합, 대규모 코드베이스 탐색 전략을 다룹니다. 이 비용 최적화된 모델들을 실제 개발 시나리오에 적용하는 방법을 배웁니다.
