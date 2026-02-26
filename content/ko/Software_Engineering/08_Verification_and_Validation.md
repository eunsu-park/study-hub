# 레슨 08: 검증과 확인 (Verification and Validation)

**이전**: [07. 소프트웨어 품질 보증](./07_Software_Quality_Assurance.md) | **다음**: [09. 형상 관리](./09_Configuration_Management.md)

---

"올바른 방식으로 제품을 만들고 있는가?" 그리고 "올바른 제품을 만들고 있는가?" 1979년 Barry Boehm이 제시한 이 두 질문은 검증(Verification)과 확인(Validation), 즉 V&V의 본질을 담고 있습니다. 이 두 가지는 함께 소프트웨어 공학의 품질 근간을 형성합니다. 소프트웨어가 기술적으로 올바르고 실질적으로 유용한지를 보장하는 체계적이고 규율 있는 접근 방식입니다. 이 레슨에서는 단위 테스트부터 형식적 증명(formal proof)까지 V&V 기법의 전반적인 스펙트럼을 살펴보고, 실제 프로젝트를 위한 포괄적인 테스트 전략을 설계할 수 있는 역량을 갖추도록 합니다.

**난이도**: ⭐⭐⭐

**선수 학습**:
- 소프트웨어 품질 보증(Software Quality Assurance) 기초 (레슨 07)
- 기본 프로그래밍 및 단위 테스트 경험
- 소프트웨어 개발 생명주기(SDLC) 이해 (레슨 02)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 검증(verification)과 확인(validation)을 구별하고 둘 다 필요한 이유를 설명한다
2. 블랙박스 기법(동등 분할, 경계값 분석, 결정 테이블, 상태 전이)을 사용하여 테스트 케이스를 설계한다
3. 화이트박스 테스팅을 위한 적절한 커버리지(coverage) 기준을 선택한다
4. 단위, 통합, 시스템, 인수 테스팅을 포괄하는 테스트 전략을 계획한다
5. 버그 생명주기를 이해하고 효과적인 결함 보고서를 작성한다
6. 고신뢰성(high-assurance) 소프트웨어에서 형식적 방법(formal methods)의 역할을 평가한다
7. 회귀 테스트 스위트를 구축하고 자동화 파이프라인에 통합한다

---

## 목차

1. [검증 vs 확인](#1-검증-vs-확인)
2. [테스팅 수준](#2-테스팅-수준)
3. [테스팅 유형](#3-테스팅-유형)
4. [블랙박스 테스팅 기법](#4-블랙박스-테스팅-기법)
5. [화이트박스 테스팅 기법](#5-화이트박스-테스팅-기법)
6. [테스트 계획 및 문서화](#6-테스트-계획-및-문서화)
7. [테스트 주도 개발](#7-테스트-주도-개발)
8. [회귀 테스팅 및 테스트 자동화](#8-회귀-테스팅-및-테스트-자동화)
9. [리뷰 및 검사](#9-리뷰-및-검사)
10. [형식적 검증](#10-형식적-검증)
11. [버그 생명주기](#11-버그-생명주기)
12. [요약](#12-요약)
13. [연습 문제](#13-연습-문제)
14. [더 읽을거리](#14-더-읽을거리)

---

## 1. 검증 vs 확인

### 1.1 핵심 차이점

| | 검증(Verification) | 확인(Validation) |
|--|--------------|------------|
| **질문** | 제품을 올바른 방식으로 만들고 있는가? | 올바른 제품을 만들고 있는가? |
| **기준** | 명세(specification), 설계 문서 | 사용자 요구, 비즈니스 목표 |
| **활동** | 리뷰, 검사, 정적 분석, 명세 대비 테스팅 | 사용자 인수 테스팅, 베타 테스팅, 프로토타이핑 |
| **단계** | 개발 전반에 걸쳐 | 주로 마일스톤과 납품 시점 |
| **발견** | 시스템이 명세를 따르는가? | 명세가 사용자의 실제 요구를 반영하는가? |

두 가지 모두 필수적입니다. 명세는 완벽하게 구현했지만 사용자 요구를 충족하지 못하는 시스템은 확인에 실패한 것입니다. 명세가 잘못된 경우입니다. 사용자 요구를 충족하지만 검증 없이 개발된 시스템은 기술 부채와 잠재적 결함의 집합체입니다.

### 1.2 V-모델

V-모델은 개발 단계(V의 왼쪽 다리)를 대응하는 테스트 단계(V의 오른쪽 다리)에 매핑하여 검증/확인 관계를 명확하게 보여줍니다:

```
요구사항 분석 ──────────────────────── 인수 테스팅
        │                                              │
   시스템 설계 ──────────────────── 시스템 테스팅  │
        │                                    │         │
  아키텍처 설계 ──── 통합 테스팅        │
        │                          │                   │
    상세 설계 ─── 단위 테스팅                   │
        │                    │                         │
      코딩 ────────────────┘                         │
        │                                              │
        └──────── 개발 ──────── 테스팅 ────────┘
```

각 테스트 단계는 대응하는 개발 단계의 산출물을 확인합니다. 인수 테스팅은 요구사항을 확인하고, 시스템 테스팅은 시스템 설계를 확인하며, 통합 테스팅은 아키텍처를 확인하고, 단위 테스팅은 상세 설계를 확인합니다.

### 1.3 V&V의 독립성

V&V에 관한 IEEE 표준(IEEE Std 1012)은 다음을 구분합니다:

- **독립적 V&V(IV&V, Independent V&V)**: 개발 결과에 이해관계가 없는 집단이 수행합니다. 흔히 별도의 조직이 담당하며, 안전 필수 시스템(의료기기, 항공우주, 원자력)에 요구됩니다.
- **비독립적 V&V**: 개발팀 또는 개발 조직이 수행합니다. 대부분의 상용 소프트웨어에 적합합니다.

연구에 따르면, 코드를 작성한 당사자가 아닌 검토자가 리뷰할 경우 사각지대가 없기 때문에 IV&V가 훨씬 더 높은 비율의 결함을 발견합니다.

---

## 2. 테스팅 수준

테스팅은 각기 다른 범위와 목적을 가진 네 가지 계층적 수준으로 구성됩니다.

### 2.1 단위 테스팅 (Unit Testing)

**범위**: 단일 함수, 메서드, 또는 클래스 — 가장 작은 테스트 가능 단위.

**수행자**: 코드를 작성한 개발자(TDD에서는 페어 파트너).

**목표**: 개별 단위가 격리된 상태에서 올바르게 동작하는지 검증합니다.

**특징**:
- 빠름 (테스트당 밀리초 단위)
- 외부 의존성 없음 (데이터베이스, 네트워크, 파일 시스템은 모의 객체(mock)로 대체)
- 고도로 집중됨; 실패한 테스트가 특정 함수를 지목

```python
# pytest example: testing a utility function in isolation
from decimal import Decimal
import pytest
from pricing import apply_discount

class TestApplyDiscount:
    def test_percentage_discount(self):
        price = Decimal("100.00")
        result = apply_discount(price, discount_pct=10)
        assert result == Decimal("90.00")

    def test_zero_discount(self):
        price = Decimal("50.00")
        result = apply_discount(price, discount_pct=0)
        assert result == Decimal("50.00")

    def test_hundred_percent_discount(self):
        result = apply_discount(Decimal("75.00"), discount_pct=100)
        assert result == Decimal("0.00")

    def test_negative_discount_raises(self):
        with pytest.raises(ValueError, match="discount must be non-negative"):
            apply_discount(Decimal("100.00"), discount_pct=-5)

    def test_discount_above_100_raises(self):
        with pytest.raises(ValueError, match="discount cannot exceed 100"):
            apply_discount(Decimal("100.00"), discount_pct=110)
```

### 2.2 통합 테스팅 (Integration Testing)

**범위**: 단위, 모듈, 또는 서브시스템 간의 상호작용.

**목표**: 컴포넌트들이 함께 올바르게 동작하는지 확인합니다. 인터페이스 불일치, 계약 위반, 통합 수준 버그를 발견합니다.

**접근 방식**:

| 접근 방식 | 설명 | 장점 | 단점 |
|----------|-------------|------|------|
| **빅뱅(Big Bang)** | 모든 것을 한 번에 통합한 후 테스트 | 설정이 간단 | 실패 위치 파악 어려움 |
| **하향식(Top-Down)** | 최상위 모듈부터 아래로 통합; 하위 모듈은 스텁(stub) 사용 | 상위 수준 로직을 조기에 테스트 | 스텁이 하위 수준 버그를 숨길 수 있음 |
| **상향식(Bottom-Up)** | 하위 모듈부터 위로 통합; 테스트 드라이버 사용 | 실제 하위 수준 동작을 조기에 테스트 | 상위 수준 로직 테스트가 늦어짐 |
| **샌드위치/혼합(Sandwich/Hybrid)** | 하향식과 상향식 동시 진행 | 양쪽의 균형 | 계획이 더 복잡 |
| **점진적(Incremental)** | 한 번에 하나의 컴포넌트씩 통합 | 실패 위치 파악 용이 | 통합 순서 계획에 더 많은 노력 필요 |

### 2.3 시스템 테스팅 (System Testing)

**범위**: 완전히 통합된 시스템 전체.

**목표**: 전체 시스템이 기능 요구사항과 비기능 요구사항을 모두 충족하는지 확인합니다.

시스템 테스팅은 기능 테스트(올바른 동작을 하는가?)와 비기능 테스트(충분히 빠른가? 충분히 안전한가? 충분한 가용성을 갖추는가?)를 모두 포함합니다. 일반적으로 개발자가 아닌 전담 QA팀이 수행합니다.

### 2.4 인수 테스팅 (Acceptance Testing)

**범위**: 사용자 관점에서 바라본 완전한 시스템.

**목표**: 시스템이 사용자 요구와 비즈니스 요구사항을 충족하는지 확인합니다.

| 유형 | 수행자 | 목적 |
|------|-----------------|---------|
| **사용자 인수 테스팅(UAT, User Acceptance Testing)** | 최종 사용자 또는 그 대리인 | 실제 업무에서 시스템이 작동함을 확인 |
| **알파 테스팅(Alpha Testing)** | 내부 사용자 (개발팀 외 회사 직원) | 외부 출시 전 버그 발견 |
| **베타 테스팅(Beta Testing)** | 선별된 외부 사용자 | 실제 환경에서 버그 발견 |
| **계약 인수 테스팅(Contract Acceptance Testing)** | 계약 조건에 따른 고객 | 계약상 의무 이행 확인 |
| **규제 인수 테스팅(Regulation Acceptance Testing)** | 규제 당국 | 규정 준수 여부 확인 |

---

## 3. 테스팅 유형

테스팅 유형은 수준을 가로질러 적용되며, 검증하는 속성에 따라 테스트를 분류합니다.

### 3.1 기능 테스팅 (Functional Testing)

시스템이 수행해야 할 동작을 수행하는지 검증합니다. 요구사항 대비 기능을 확인합니다.

- **스모크 테스팅(Smoke testing)**: 시스템이 시작하고 기본 동작을 수행할 수 있는지 확인하는 빠른 테스트 집합. 빌드마다 실행하여 심층 테스팅이 필요한지 판단합니다.
- **온전성 테스팅(Sanity testing)**: 특정 수정 사항이 작동하는지 확인하는 회귀 테스팅의 일부.
- **기능 테스팅(Feature testing)**: 요구사항에 기술된 모든 기능의 체계적 테스팅.

### 3.2 비기능 테스팅 (Non-Functional Testing)

| 유형 | 측정 대상 | 핵심 질문 |
|------|-----------------|--------------|
| **성능 테스팅(Performance testing)** | 속도, 처리량, 자원 사용 | 예상 부하를 처리할 수 있는가? |
| **부하 테스팅(Load testing)** | 예상 최대 부하에서의 동작 | 최대 부하 시 우아하게 성능이 저하되는가? |
| **스트레스 테스팅(Stress testing)** | 극한 또는 예상치 못한 부하에서의 동작 | 어디에서 무너지는가? 어떻게 실패하는가? |
| **지속 내구 테스팅(Soak/endurance testing)** | 정상 부하에서 장시간 동작 | 메모리 누수나 점진적 성능 저하가 있는가? |
| **확장성 테스팅(Scalability testing)** | 자원 추가에 따른 용량 증가 방식 | 서버 증가에 비례하여 성능이 확장되는가? |
| **보안 테스팅(Security testing)** | 공격에 대한 저항력 | OWASP Top 10에 취약한가? |
| **사용성 테스팅(Usability testing)** | 사용 편의성 | 사용자가 혼란 없이 작업을 완료할 수 있는가? |
| **접근성 테스팅(Accessibility testing)** | WCAG 준수 여부 | 장애가 있는 사용자도 시스템을 사용할 수 있는가? |
| **호환성 테스팅(Compatibility testing)** | 다양한 환경에서의 동작 | Chrome, Firefox, Safari, iOS, Android에서 작동하는가? |
| **복구 테스팅(Recovery testing)** | 장애 후 동작 | 충돌 후 올바르게 복구되는가? |

### 3.3 성능 테스팅 예제

```python
# locust: load testing tool for web applications
from locust import HttpUser, task, between

class WebsiteUser(HttpUser):
    wait_time = between(1, 3)  # simulate 1-3 seconds think time

    @task(3)  # weight: this task runs 3x more often than weight-1 tasks
    def browse_products(self):
        self.client.get("/api/products?page=1&limit=20")

    @task(1)
    def view_product_detail(self):
        product_id = 42
        self.client.get(f"/api/products/{product_id}")

    @task(1)
    def add_to_cart(self):
        self.client.post("/api/cart/items", json={
            "product_id": 42,
            "quantity": 1
        })

# Run: locust -f locustfile.py --headless -u 1000 -r 100 --host http://localhost:8000
# -u 1000: 1000 concurrent users
# -r 100: ramp up at 100 users/second
```

---

## 4. 블랙박스 테스팅 기법

블랙박스 기법은 내부 구현에 대한 지식 없이 명세로부터 테스트 케이스를 도출합니다. "어떤 입력을 시도해야 하는가?"에 답합니다.

### 4.1 동등 분할 (Equivalence Partitioning)

시스템이 모든 구성원에 대해 동일하게 동작해야 하는 클래스로 입력 도메인을 나눕니다. 각 클래스에서 하나의 값을 테스트합니다.

**예제**: 가격 등급을 결정하기 위해 나이(정수)를 받는 함수.
- 유효한 나이: 0–12 (어린이), 13–17 (청소년), 18–64 (성인), 65+ (노인)
- 유효하지 않음: 음수, 비정수, null

```
분할                      대표값          예상 결과
───────────────────────────────────────────────────────────────
유효: 어린이 (0–12)       8               "child" 등급
유효: 청소년 (13–17)      15              "teen" 등급
유효: 성인 (18–64)        30              "adult" 등급
유효: 노인 (65+)          70              "senior" 등급
유효하지 않음: 음수       -1              ValueError
유효하지 않음: null       None            TypeError
```

동등 분할은 고유한 동작의 커버리지를 유지하면서 테스트 공간을 무한에서 관리 가능한 수준으로 줄입니다.

### 4.2 경계값 분석 (Boundary Value Analysis)

버그는 동등 클래스의 경계에 집중됩니다. 각 경계에서, 바로 아래, 그리고 바로 위의 값을 테스트합니다.

나이 예제 사용:
```
경계        테스트할 값
────────────────────────────────────────
0 (어린이의 하한)           -1, 0, 1
12/13 (어린이/청소년 경계)  12, 13
17/18 (청소년/성인 경계)    17, 18
64/65 (성인/노인 경계)      64, 65
최대 (예: 150)              149, 150, 151
```

**경계가 중요한 이유**: 일대일 오류(`<` vs `<=`, `>` vs `>=`)는 가장 흔한 프로그래밍 실수 중 하나이며, 경계에서만 나타납니다.

### 4.3 결정 테이블 테스팅 (Decision Table Testing)

결정 테이블은 조건들의 조합과 그에 대응하는 행동을 체계적으로 열거합니다. "조합 맹점" — 두 조건 간의 상호작용을 놓치는 것 — 을 방지합니다.

**예제**: 세 가지 조건을 가진 대출 승인 시스템.

| | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 |
|--|----|----|----|----|----|----|----|----|
| 신용 점수 > 700 | T | T | T | T | F | F | F | F |
| 소득 > $50k | T | T | F | F | T | T | F | F |
| 부채 비율 < 40% | T | F | T | F | T | F | T | F |
| **행동** | 승인 | 검토 후 승인 | 검토 후 승인 | 거부 | 검토 후 승인 | 거부 | 거부 | 거부 |

각 열(규칙)이 테스트 케이스가 됩니다. 축약된 규칙(동일한 행동에 여러 조건)은 병합할 수 있지만, 각 고유 행동에는 최소 하나의 테스트가 필요합니다.

### 4.4 상태 전이 테스팅 (State Transition Testing)

고유한 상태를 가진 시스템의 경우, 상태 기계(state machine)로부터 테스트 케이스를 도출합니다. 유효한 전이, 유효하지 않은 전이, 경계 상태를 테스트합니다.

**예제**: 주문 상태 기계.

```
        ┌──────────────────────────────────────────────┐
        │                                              │
      [신규] ──── 결제 ────▶ [결제완료] ──── 발송 ───▶ [발송됨]
        │                    │                        │
     취소                  취소                   배달
        │                    │                        │
        ▼                    ▼                        ▼
   [취소됨]         [취소됨]            [배달완료]
                                                      │
                                                    반품
                                                      │
                                                      ▼
                                               [반품됨]
```

커버할 테스트 케이스:
1. 신규 → 결제 → 결제완료 (유효)
2. 신규 → 취소 → 취소됨 (유효)
3. 결제완료 → 발송 → 발송됨 (유효)
4. 결제완료 → 취소 → 취소됨 (유효)
5. 발송됨 → 배달 → 배달완료 (유효)
6. 배달완료 → 반품 → 반품됨 (유효)
7. 신규 → 발송 (유효하지 않은 전이 — 거부되어야 함)
8. 취소됨 → 결제 (종단 상태에서의 유효하지 않은 전이)

---

## 5. 화이트박스 테스팅 기법

화이트박스(구조적) 기법은 소스 코드의 내부 구조로부터 테스트 케이스를 도출합니다. 코드의 얼마나 많은 부분이 실행되는지를 측정하는 *커버리지(coverage)*를 측정합니다.

### 5.1 구문 커버리지 (Statement Coverage)

모든 실행 가능한 구문이 최소 한 번 실행됩니다.

```python
def classify(x):
    result = "unknown"          # Statement 1
    if x > 0:                   # Statement 2
        result = "positive"     # Statement 3
    elif x < 0:                 # Statement 4
        result = "negative"     # Statement 5
    else:
        result = "zero"         # Statement 6
    return result               # Statement 7
```

100% 구문 커버리지를 위해서는 최소 3개의 테스트 케이스가 필요합니다: x=1, x=-1, x=0.

**약점**: 구문 커버리지는 거짓(false) 분기를 테스트하지 않고도 달성할 수 있습니다.

### 5.2 분기 커버리지 (Branch Coverage / Decision Coverage)

모든 결정 지점에서 모든 분기가 최소 한 번 취해집니다(모든 `if`, `while`, `for`의 참(true)과 거짓(false) 결과 모두).

분기 커버리지는 구문 커버리지를 포함합니다: 100% 분기 커버리지는 100% 구문 커버리지를 의미하지만, 그 반대는 성립하지 않습니다.

```python
def validate_age(age):
    if age is None:             # Branch: True (age is None), False (age is not None)
        return False
    if age < 0 or age > 150:   # Branch: True, False; compound condition
        return False
    return True
```

100% 분기 커버리지를 위해:
- `validate_age(None)` — 첫 번째 if의 참(True) 분기
- `validate_age(25)` — 첫 번째 if의 거짓(False) 분기, 두 번째 if의 거짓(False) 분기
- `validate_age(-1)` — 두 번째 if의 참(True) 분기

### 5.3 조건 커버리지 (Condition Coverage)

모든 불리언 부분 조건(predicate)이 독립적으로 참(True)과 거짓(False) 모두로 평가됩니다.

`age < 0 or age > 150`과 같은 복합 조건의 경우:
- `age < 0`이 참(True)과 거짓(False) 모두여야 합니다
- `age > 150`이 참(True)과 거짓(False) 모두여야 합니다

### 5.4 경로 커버리지 (Path Coverage)

코드를 통과하는 모든 고유한 경로가 실행됩니다. `n`개의 독립적 결정 지점을 가진 함수에는 최대 `2^n`개의 경로가 있습니다.

경로 커버리지는 가장 강한 기준이지만 실제 함수에서는 일반적으로 비현실적입니다(지수적 폭발). 안전 필수 모듈에서 선택적으로 사용됩니다.

### 5.5 커버리지 요약

```
커버리지 기준        강도        실용적 용도
──────────────────────────────────────────────────────────────
구문(Statement)      가장 약함   최소 허용 수준 (80–90%)
분기(Branch)         중간        대부분의 프로젝트 표준
조건(Condition)      더 강함     보안 필수 코드
MC/DC*               강함        DO-178C (항공전자), 안전 시스템
경로(Path)           가장 강함   소규모 단위 외 비현실적
```

*MC/DC = 수정된 조건/결정 커버리지(Modified Condition/Decision Coverage), FAA가 항공전자 소프트웨어에 요구합니다.

### 5.6 실제 커버리지 측정

```bash
# Python: pytest + coverage
pip install pytest pytest-cov

pytest --cov=src --cov-report=html --cov-fail-under=80

# Output:
# Name                 Stmts   Miss  Cover
# ────────────────────────────────────────
# src/pricing.py          45      3    93%
# src/inventory.py        72     18    75%
# src/checkout.py         98     12    88%
# ────────────────────────────────────────
# TOTAL                  215     33    85%
```

```bash
# JavaScript: Jest
jest --coverage --coverageThreshold='{"global":{"branches":80,"lines":80}}'
```

---

## 6. 테스트 계획 및 문서화

### 6.1 테스트 계획서 (Test Plan)

테스트 계획서(IEEE Std 829)는 테스팅의 범위, 접근 방식, 자원, 일정을 문서화합니다. 주요 섹션:

| 섹션 | 내용 |
|---------|---------|
| **테스트 범위(Test Scope)** | 범위에 포함되거나 제외되는 기능/컴포넌트 |
| **테스트 접근 방식(Test Approach)** | 사용할 테스팅 수준, 유형, 기법 |
| **진입/완료 기준(Entry/Exit Criteria)** | 테스팅이 시작되고 완료되는 시점 |
| **테스트 환경(Test Environment)** | 하드웨어, OS, 브라우저, 네트워크 구성 |
| **자원(Resources)** | 누가 어떤 테스트를 수행하는가; 필요 도구 |
| **일정(Schedule)** | 각 테스팅 단계의 타임라인 |
| **위험 및 대응(Risk and Contingency)** | 테스트 작업의 위험 요소와 완화 계획 |
| **산출물(Deliverables)** | 테스트 케이스, 테스트 데이터, 결함 보고서, 테스트 요약 보고서 |

**진입 기준** 예제:
- 스프린트의 모든 코드가 병합되고 CI를 통과
- 단위 테스트 커버리지 ≥ 80%
- 이전 주기에서 열린 치명적/높음 수준 결함 없음

**완료 기준** 예제:
- 계획된 모든 테스트 케이스 실행 완료
- 열린 치명적 또는 높음 심각도 결함 없음
- 결함 제거 효율 ≥ 90%
- 성능 테스트 결과가 목표 대비 10% 이내

### 6.2 테스트 케이스 작성

좋은 테스트 케이스는 원자적이고, 독립적이며, 재현 가능합니다.

```
테스트 케이스 ID:    TC-CHECKOUT-042
제목:           결제 서비스 다운 시 장바구니 결제가 적절하게 실패
전제 조건:   - 사용자가 로그인되어 있음
                 - 장바구니에 총 $59.98의 항목 2개 있음
                 - 결제 서비스 모의 객체가 503을 반환하도록 구성
기능:         결제 / 주문
테스트 데이터:       User: testuser@example.com, Cart: [item_id:1, item_id:7]
단계:
  1. /cart로 이동
  2. "결제 진행" 클릭
  3. 유효한 신용카드 입력: 4111 1111 1111 1111
  4. "주문하기" 클릭
예상 결과: "결제 서비스를 일시적으로 사용할 수 없습니다.
                 장바구니가 저장되었습니다. 몇 분 후 다시 시도하세요."가 표시됨.
                 데이터베이스에 주문이 생성되지 않음.
                 장바구니 내용이 보존됨.
실제 결과:   (테스트 실행 중 기입)
상태:          통과 / 실패 / 차단
심각도:        높음
우선순위:        높음
작성자:          J. Smith
날짜:            2024-03-15
```

### 6.3 테스트 데이터 관리

좋은 테스트 데이터는:
- **대표성(Representative)**: 모든 동등 분할을 커버
- **재현성(Reproducible)**: 동일한 데이터가 동일한 결과를 생성
- **격리성(Isolated)**: 테스트가 가변 상태를 공유하지 않음
- **익명성(Anonymized)**: 실제 프로덕션 데이터가 아닌 합성 데이터 사용

```python
# Using factories for reproducible test data (Factory Boy library)
import factory
from factory.django import DjangoModelFactory
from myapp.models import User, Order

class UserFactory(DjangoModelFactory):
    class Meta:
        model = User

    username = factory.Sequence(lambda n: f"user_{n}")
    email = factory.LazyAttribute(lambda obj: f"{obj.username}@example.com")
    is_active = True

class OrderFactory(DjangoModelFactory):
    class Meta:
        model = Order

    user = factory.SubFactory(UserFactory)
    status = "new"
    total = factory.fuzzy.FuzzyDecimal(10.00, 500.00, precision=2)
```

---

## 7. 테스트 주도 개발

테스트 주도 개발(TDD, Test-Driven Development)은 전통적인 워크플로우를 역전시킵니다. 테스트를 테스트 대상 코드보다 *먼저* 작성합니다.

### 7.1 레드-그린-리팩터 주기 (Red-Green-Refactor Cycle)

```
        ┌─────────────────────────────────────────────┐
        │                                             │
        ▼                                             │
    레드(RED): 실패하는 테스트 작성                         │
    (테스트가 원하는 동작을 설명)             │
        │                                             │
        ▼                                             │
    그린(GREEN): 테스트를 통과하는 최소한의 코드 작성    │
    (그 이상도, 그 이하도 아님)                                │
        │                                             │
        ▼                                             │
    리팩터(REFACTOR): 코드 정리                       │
    (테스트 스위트가 아무것도 깨지지 않았음을          │
     보장)                                        │
        │                                             │
        └─────────────────────────────────────────────┘
```

### 7.2 TDD의 장점과 비용

| 장점 | 설명 |
|---------|-------------|
| 테스트가 명세로서의 역할 | 테스트가 코드가 수행해야 하는 것을 정확히 문서화 |
| 설계 압박 | 테스트하기 어려운 코드는 보통 설계가 나쁨; TDD가 이를 일찍 드러냄 |
| 리팩터링 신뢰도 | 녹색 테스트 스위트가 리팩터링으로 아무것도 깨지지 않았음을 증명 |
| 회귀 안전망 | 모든 버그 수정에 재발을 방지하는 테스트가 추가됨 |

| 비용 | 완화 방법 |
|------|-----------|
| 초기 개발 속도 저하 | 디버깅 시간 감소와 높은 품질로 상쇄 |
| 학습 곡선 | 팀 교육; 숙련된 실무자와의 페어 프로그래밍 |
| UI/통합 테스트가 더 어려움 | 단위 수준에서 TDD 적용; 별도의 통합 테스트 전략 사용 |

### 7.3 간단한 예제

```python
# Step 1: RED — write a failing test
def test_fizzbuzz_returns_fizz_for_multiples_of_3():
    assert fizzbuzz(3) == "Fizz"
    assert fizzbuzz(6) == "Fizz"
    assert fizzbuzz(9) == "Fizz"

# Step 2: GREEN — minimum code to pass
def fizzbuzz(n):
    if n % 3 == 0:
        return "Fizz"
    return str(n)

# Step 3: Add more tests → RED
def test_fizzbuzz_returns_buzz_for_multiples_of_5():
    assert fizzbuzz(5) == "Buzz"

# Step 4: GREEN
def fizzbuzz(n):
    if n % 15 == 0:
        return "FizzBuzz"
    if n % 3 == 0:
        return "Fizz"
    if n % 5 == 0:
        return "Buzz"
    return str(n)
```

TDD는 프로그래밍 토픽 (레슨 10: 테스팅과 TDD)에서 더 깊이 다룹니다.

---

## 8. 회귀 테스팅 및 테스트 자동화

### 8.1 회귀 테스팅 (Regression Testing)

회귀(regression)는 이전에 올바르게 작동하던 부분이 변경으로 인해 도입된 버그입니다. 회귀 테스팅은 변경 후 이전에 통과했던 테스트를 다시 실행하여 회귀를 감지합니다.

**회귀의 함정**: 시스템이 커질수록 수동 회귀 테스트 스위트는 불가능할 정도로 방대해집니다. 기능 100개에 각각 50개의 테스트가 있는 시스템 = 릴리스마다 5,000회의 수동 테스트 실행 — 실행 불가능합니다.

**해결책**: 회귀 스위트를 자동화합니다. 모든 버그 수정에 자동화된 테스트가 추가됩니다. 모든 기능에 자동화된 테스트가 추가됩니다. 스위트는 모든 커밋에서 실행됩니다.

### 8.2 테스트 자동화 피라미드

테스트 피라미드(Mike Cohn)는 테스트 유형의 올바른 균형을 규정합니다:

```
                     ╱╲
                    ╱  ╲
                   ╱ UI ╲
                  ╱ 테스트 ╲   ← 느리고, 취약하며, 비용이 높음
                 ╱──────────╲     적게 (스위트의 10–20%)
                ╱            ╲
               ╱   통합        ╲
              ╱    테스트        ╲  ← 중간 속도/비용
             ╱────────────────────╲    일부 (스위트의 20–30%)
            ╱                      ╲
           ╱      단위 테스트         ╲  ← 빠르고, 저렴하며, 정밀
          ╱────────────────────────────╲    많이 (스위트의 50–70%)
```

피라미드를 뒤집는 것(UI 테스트 많음, 단위 테스트 적음)은 느리고 취약한 테스트 스위트를 만들어 피드백이 나쁘고 유지보수 비용이 높아집니다.

### 8.3 CI/CD 통합

```yaml
# GitHub Actions: automated test pipeline
name: Test Suite

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: pip install -r requirements-dev.txt
      - name: Run unit tests with coverage
        run: |
          pytest tests/unit/ -v \
            --cov=src \
            --cov-report=xml \
            --cov-fail-under=85
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests  # only run if unit tests pass
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: testpass
          POSTGRES_DB: testdb
        ports:
          - 5432:5432
    steps:
      - uses: actions/checkout@v4
      - name: Run integration tests
        run: pytest tests/integration/ -v
        env:
          DATABASE_URL: postgresql://postgres:testpass@localhost/testdb
```

---

## 9. 리뷰 및 검사

정적 V&V(실행 없이 산출물 검토)는 특정 유형의 결함을 찾는 데 있어 테스팅보다 비용 효율적인 경우가 많습니다.

### 9.1 파간 검사 (Fagan Inspection)

1976년 IBM의 Michael Fagan이 개발했습니다. 공식적이고 역할 기반이며 매우 효과적입니다. Fagan의 원본 연구에서는 첫 번째 테스트 실행 이전에 결함의 82%를 발견했다고 보고했습니다.

**역할**:
| 역할 | 책임 |
|------|----------------|
| **조정자(Moderator)** | 계획 및 진행 관리; 프로세스가 준수되는지 확인 |
| **작성자(Author)** | 검토 대상 산출물을 작성; 질문에 답변 |
| **낭독자(Reader)** | 검사 중 산출물을 읽거나 재서술 |
| **검사자(Inspector)** | 결함 발견; 회의 전 독립적으로 준비 |
| **기록자(Scribe)** | 모든 결함과 결정 사항 기록 |

**단계**:
1. **계획(Planning)**: 조정자가 진입 기준 확인; 자료 배포
2. **개요(Overview)**: 작성자가 맥락과 설계 의도 설명
3. **준비(Preparation)**: 각 검사자가 독립적으로 검토하고 이슈 기록
4. **검사 회의(Inspection Meeting)**: 낭독자가 읽고; 검사자가 이슈 제기; 기록자가 기록
5. **재작업(Rework)**: 작성자가 기록된 모든 결함 수정
6. **후속 조치(Follow-up)**: 조정자가 모든 결함 처리 여부 확인; 완료 기준 점검

**최적 검사 속도**: 시간당 100–200 줄의 코드. 서두르면 결과가 미미합니다.

### 9.2 워크스루 (Walkthroughs)

파간 검사보다 덜 공식적입니다. 작성자가 산출물을 제시하고 팀을 안내하며 질문과 의견을 받습니다. 교육과 지식 이전에 적합합니다. 공식 검사보다 체계성이 낮습니다.

### 9.3 동료 코드 리뷰 (Peer Code Review / Pull Request Review)

현대 팀에서 가장 일반적인 코드 리뷰 형태입니다. 레슨 07 (소프트웨어 품질 보증, 섹션 10)에서 자세히 다룹니다.

### 9.4 검사 vs 테스팅 효과

| 기법 | 잘 찾는 것 | 찾기 어려운 것 |
|-----------|-----------------|---------------------|
| 검사(Inspection) | 논리 오류, 설계 문제, 표준 위반, 누락된 요구사항 | 타이밍 버그, 성능 문제 |
| 테스팅(Testing) | 통합 실패, 성능 문제, 타이밍 버그 | 누락된 기능 (없는 것은 테스트할 수 없음) |

두 기법은 상호 보완적입니다. 어느 하나만으로는 충분하지 않습니다.

---

## 10. 형식적 검증

형식적 검증(Formal Verification)은 수학적 증명을 사용하여 시스템이 명세를 만족한다는 것을 확립합니다. 가장 높은 수준의 보증을 제공하지만 비용이 높고 전문적인 전문 지식이 필요합니다.

### 10.1 모델 검사 (Model Checking)

모델 검사는 시스템의 유한 상태 모델(finite-state model)의 모든 가능한 상태를 완전히 탐색합니다. 모든 상태에서 속성이 유지됨을 확인하거나 반례 트레이스를 제공합니다.

**사용 사례**: 통신 프로토콜 (TLS 핸드셰이크), 동시성 시스템 (경쟁 조건 검사), 하드웨어 설계.

**도구**: SPIN, TLA+, Alloy, NuSMV.

```tla
(* TLA+ specification: a simple mutual exclusion algorithm *)
VARIABLES pc1, pc2, flag1, flag2

Init == pc1 = "start" /\ pc2 = "start"
     /\ flag1 = FALSE /\ flag2 = FALSE

(* Process 1 sets its flag and waits for process 2 to clear *)
Next1 == \/ /\ pc1 = "start"    /\ pc1' = "try"    /\ flag1' = TRUE  /\ UNCHANGED <<pc2, flag2>>
         \/ /\ pc1 = "try"     /\ ~flag2            /\ pc1' = "cs"    /\ UNCHANGED <<flag1, pc2, flag2>>
         \/ /\ pc1 = "cs"      /\ pc1' = "start"   /\ flag1' = FALSE  /\ UNCHANGED <<pc2, flag2>>

(* Safety: both processes cannot be in the critical section simultaneously *)
MutualExclusion == ~(pc1 = "cs" /\ pc2 = "cs")
```

### 10.2 정리 증명 (Theorem Proving)

정리 증명기(Coq, Isabelle/HOL, Lean)는 증명 구성을 위한 사람의 안내가 필요합니다. 모델 검사기가 처리할 수 없는 무한 상태 공간을 다룰 수 있습니다.

**주목할 만한 적용 사례**:
- seL4 마이크로커널: 완전한 형식적 정확성 증명을 갖춘 최초의 OS 커널
- CompCert C 컴파일러: C에서 올바른 기계어 코드를 생성함을 형식적으로 검증
- CryptoVerif: 암호화 프로토콜의 형식적 검증

### 10.3 형식적 방법의 적용 분야

```
보증 수준     비용 배율   일반적인 도메인
────────────────────────────────────────────────
테스팅         1x       모든 소프트웨어
코드 리뷰      1.5x     모든 소프트웨어
파간 검사      3x       고신뢰성 시스템
모델 검사      5–10x    프로토콜, 동시성 시스템
정리 증명      20–50x   안전 필수, 암호화
```

형식적 방법이 실용적인 경우:
- 작고 잘 정의된 컴포넌트 (스케줄러, 프로토콜 상태 기계)
- 보안 필수 알고리즘 (암호화 원시 함수)
- 실패 비용이 극단적인 시스템 (심박 조율기, 항공기 비행 제어)

---

## 11. 버그 생명주기

### 11.1 버그 상태

```
         발견됨
              │
              ▼
           [신규] ─── 중복? ──▶ [중복] → 종료
              │
              ▼
          [배정됨] ─── 버그 아님? ──▶ [거부됨] → 종료
              │
          개발자가
           작업 중
              │
              ▼
           [수정됨]
              │
           테스터가
          검증 중
              ├──── 여전히 실패 ──▶ [재오픈] ──▶ [배정됨]
              │
              ▼
          [검증됨]
              │
              ▼
           [종료됨]
```

### 11.2 버그 심각도 vs 우선순위

| 심각도 | 정의 | 예시 |
|----------|------------|---------|
| **치명적(Critical)** | 시스템 충돌, 데이터 손실, 보안 침해 — 우회 방법 없음 | 로그인 시 앱이 항상 충돌 |
| **높음(High)** | 주요 기능 고장, 우회 방법 없음 | 사용자가 결제를 완료할 수 없음 |
| **중간(Medium)** | 기능 고장이지만 우회 방법 존재 | PDF 내보내기 실패; 사용자가 복사-붙여넣기 가능 |
| **낮음(Low)** | 경미한 이슈; 외관상 문제 | 버튼이 2px 어긋나 있음 |

| 우선순위 | 정의 | 예시 |
|----------|------------|---------|
| **P1** | 즉시 수정 — 릴리스 중단 또는 롤백 | 결제 처리의 치명적 버그 |
| **P2** | 현재 스프린트에서 수정 | 핵심 사용자 워크플로우를 차단하는 높음 심각도 버그 |
| **P3** | 다음 스프린트에서 수정 | 알려진 우회 방법이 있는 중간 버그 |
| **P4** | 시간이 될 때 수정 | 낮은 심각도의 외관상 버그 |

심각도와 우선순위는 독립적입니다. 거의 사용되지 않는 관리자 기능의 치명적 버그는 P3일 수 있습니다. 모든 사용자가 보는 랜딩 페이지의 낮은 심각도 버그는 P2일 수 있습니다.

### 11.3 효과적인 버그 보고서 작성

```
제목:       비밀번호 재설정 링크가 1회 사용 후 만료됨 (예상: 10분)
ID:          BUG-2847
심각도:    높음
우선순위:    P2
보고자:    Q. Chen
담당자: R. Patel
버전:     v2.3.1
환경: 프로덕션 (스테이징에서도 재현됨)

재현 단계:
  1. 로그인 페이지에서 "비밀번호 찾기" 클릭
  2. 등록된 이메일 주소 입력
  3. 이메일 확인; 재설정 링크 클릭 → 재설정 폼으로 이동 ✓
  4. 새 비밀번호 설정 → 성공 메시지 ✓
  5. 10분 이내에 이메일에서 동일한 링크 다시 클릭

예상:
  폼에 "이 링크가 만료되었습니다"가 표시됨 (링크는 10분 동안 유효해야 함)

실제:
  서버가 HTTP 500 내부 서버 오류 반환

첨부 파일:
  - screenshot_500_error.png
  - server_error_log_2024-03-15_14:22:07.txt

추가 맥락:
  동일한 링크를 두 번째로 사용할 때만 발생함. 500 오류는
  토큰이 이미 삭제된 경우 토큰 삭제가 예외를 발생시킨다는 것을 시사함.
  삭제 전 "토큰 존재" 확인이 누락된 것으로 보임.
```

---

## 12. 요약

검증(Verification)과 확인(Validation)은 상호 보완적인 시스템을 형성합니다. 검증은 소프트웨어가 올바르게 개발되었는지(명세에 따라)를 보장하고, 확인은 올바른 제품을 개발하고 있는지(사용자 요구 충족)를 보장합니다.

핵심 요점:

- **네 가지 테스팅 수준** — 단위, 통합, 시스템, 인수 — 각각 고유한 범위와 목적을 가집니다. 네 가지 모두를 다루는 테스트 전략을 설계하세요.
- **블랙박스 기법** — 동등 분할, 경계값 분석, 결정 테이블, 상태 전이 — 소스 코드 접근 없이 명세 수준 동작의 체계적인 커버리지를 제공합니다.
- **화이트박스 기법** — 구문, 분기, 경로, 조건 커버리지 — 구조적 철저함의 정량적 측정을 제공합니다. 실용적인 최소값으로 분기 커버리지 ≥80%를 목표로 하세요.
- **테스트 피라미드** — 많은 빠른 단위 테스트, 더 적은 통합 테스트, 적은 엔드투엔드 테스트 — 빠르고 유지보수 가능한 테스트 스위트를 만듭니다. 피라미드를 뒤집으면 느리고 취약한 스위트가 됩니다.
- **리뷰와 검사**는 테스팅과 다른 결함을 발견합니다. 공식 파간 검사는 지금까지 측정된 가장 비용 효율적인 결함 발견 기법 중 하나입니다.
- **TDD**는 테스트 작성을 개발 리듬에 통합하고 무료 부산물로 회귀 스위트를 만듭니다.
- **형식적 검증**은 높은 비용으로 수학적 확실성을 제공합니다. 안전 필수 및 보안 필수 컴포넌트에 적합합니다.
- **버그 보고서**는 소통의 산출물입니다. 완전하고, 재현 가능하며, 구체적인 보고서가 더 빨리 수정됩니다.

---

## 13. 연습 문제

**연습 문제 1 — 동등 분할과 경계값 분석**

비밀번호 검증 함수는 다음을 요구합니다:
- 길이: 8–64자
- 대문자 하나 이상 포함
- 숫자 하나 이상 포함
- `!@#$%^&*()`에서 특수 문자 하나 이상 포함
- 공백 포함 불가

(a) 각 규칙에 대한 모든 동등 분할을 식별하세요.
(b) 길이 규칙에 대한 경계값 테스트 집합을 작성하세요.
(c) 결정 테이블을 사용하여 조건 위반의 모든 조합과 각각에 대한 예상 오류 메시지를 식별하세요.

**연습 문제 2 — 커버리지 분석**

다음 함수에 대해 제어 흐름 그래프를 그리세요. 그런 다음:
(a) 순환 복잡도(cyclomatic complexity)를 계산하세요.
(b) 100% 분기 커버리지를 달성하는 최소 테스트 집합을 식별하세요.
(c) 모든 독립 경로를 식별하세요(경로 커버리지를 위해). 경로 커버리지는 몇 개의 테스트 케이스를 필요로 하는가?

```python
def shipping_cost(weight_kg, express, country):
    if weight_kg <= 0:
        raise ValueError("Weight must be positive")
    base = weight_kg * 2.50
    if express:
        base *= 1.75
    if country == "domestic":
        return base
    elif country == "canada":
        return base * 1.20
    else:
        return base * 2.00
```

**연습 문제 3 — 테스트 계획서**

v2.0 릴리스 전 모바일 뱅킹 애플리케이션을 테스팅하고 있습니다. 이번 릴리스에는 다음이 포함됩니다:
- 새로운 생체 인증 (지문 / Face ID)
- P2P 결제 기능 (연락처에게 송금)
- 재설계된 거래 내역 화면

다음을 포함하는 테스트 계획서 개요를 작성하세요:
- 범위 (무엇이 포함되고 제외되는지)
- 테스팅 접근 방식 (어떤 수준과 유형의 테스트를, 그리고 그 이유)
- 진입 및 완료 기준
- 최소 3개의 위험과 그 완화 전략

**연습 문제 4 — 파간 검사**

팀이 150줄짜리 인증 모듈에 대한 파간 검사를 수행하려 합니다. 이 모듈은 시니어 개발자가 작성했으며 네 명의 엔지니어가 검토할 예정입니다.

(a) 권장 검사 속도를 고려할 때 준비 단계와 검사 회의 각각 얼마나 시간이 걸려야 하는가?
(b) 인증 코드에 특화된 최소 8개 항목이 포함된 검사 체크리스트를 설계하세요.
(c) 검사 중에 작성자가 검사자들이 이슈를 제기하기 전에 각 설계 결정을 설명하기 시작합니다. 이 상황에서 조정자의 책임은 무엇이며, 그 이유는 무엇인가?

**연습 문제 5 — 버그 보고서**

웹 애플리케이션에서 다음과 같은 동작을 발견했습니다: 제품 목록을 "가격: 낮은 순"으로 정렬하면, $0.00(무료) 가격의 항목이 목록 맨 위가 아닌 맨 아래에 표시됩니다.

섹션 11.3의 템플릿에 따라 완전하고 전문적인 버그 보고서를 작성하세요. 다음을 포함하세요:
- 설명적인 제목
- 적절한 심각도와 우선순위 등급 및 그 근거
- 정확한 재현 단계
- 예상 vs 실제 동작
- 근본 원인에 대한 최소 하나의 가설

---

## 14. 더 읽을거리

- **서적**:
  - *The Art of Software Testing* (3rd ed.) — Glenford Myers, Corey Sandler, Tom Badgett. 고전적인 입문서.
  - *Software Testing: A Craftsman's Approach* (4th ed.) — Paul Jorgensen. 모든 테스팅 기법의 포괄적 내용.
  - *Continuous Delivery* — Jez Humble and David Farley. 테스팅을 포함한 전체 전달 파이프라인의 자동화 방법.
  - *Introduction to the Theory of Computation* — Michael Sipser. 형식적 방법의 배경 지식.

- **표준**:
  - IEEE Std 829-2008 — 소프트웨어 및 시스템 테스트 문서화에 관한 IEEE 표준
  - IEEE Std 1012-2016 — 시스템, 소프트웨어 및 하드웨어 검증과 확인에 관한 IEEE 표준
  - ISO/IEC 29119 — 소프트웨어 테스팅 표준 (5부 시리즈)

- **도구**:
  - pytest — https://pytest.org/ (Python 테스팅 프레임워크)
  - Jest — https://jestjs.io/ (JavaScript 테스팅)
  - Locust — https://locust.io/ (부하 테스팅)
  - SPIN — http://spinroot.com/ (동시성 시스템을 위한 모델 검사기)
  - TLA+ — https://lamport.azurewebsites.net/tla/tla.html (형식적 명세 언어)

- **논문 및 기사**:
  - Fagan, M. E. (1976). "Design and Code Inspections to Reduce Errors in Program Development." *IBM Systems Journal*.
  - Myers, G. J. (1978). "A Controlled Experiment in Program Testing and Code Walkthroughs/Inspections." *Communications of the ACM*.
  - Boehm, B. (1979). "Guidelines for Verifying and Validating Software Requirements and Design Specifications." *EURO IFIP*.

---

## 연습 문제

### 연습 1: 블랙박스 테스트 케이스 설계

할인 계산 함수의 명세가 다음과 같습니다:

- 입력: `cart_total` (float, > 0이어야 함)과 `coupon_code` (문자열, 선택적)
- 유효한 쿠폰 코드: `"SAVE10"` (10% 할인), `"SAVE20"` (20% 할인), `"FREESHIP"` (가격 변동 없음, 무료 배송 플래그 설정)
- `cart_total` < 10.00인 경우, 유효한 쿠폰이 있어도 할인이 적용되지 않음
- `coupon_code`가 None이거나 비어 있으면 할인이 적용되지 않음
- 유효하지 않은 쿠폰 코드는 오류를 반환함

동치 분할(Equivalence Partitioning)과 경계값 분석(Boundary Value Analysis)을 사용하여 최소 테스트 집합을 도출하세요. 각 테스트 케이스에 대해 입력, 해당하는 동치 분할, 예상 출력을 명시하세요.

### 연습 2: 분기 커버리지(Branch Coverage) 달성

아래 함수에 대한 제어 흐름 그래프(Control Flow Graph)를 그리고, 100% 분기 커버리지를 달성하는 최소 테스트 집합을 식별하세요. 반드시 테스트해야 할 각 분기를 명시하세요.

```python
def classify_bmi(weight_kg, height_m):
    if height_m <= 0 or weight_kg <= 0:
        raise ValueError("Weight and height must be positive")
    bmi = weight_kg / (height_m ** 2)
    if bmi < 18.5:
        return "underweight"
    elif bmi < 25.0:
        return "normal"
    elif bmi < 30.0:
        return "overweight"
    else:
        return "obese"
```

(a) 몇 개의 분기가 존재합니까? 모두 나열하세요.
(b) 100% 분기 커버리지를 위해 필요한 최소 테스트 케이스 수는 얼마입니까?
(c) 이 테스트 집합이 100% 경로 커버리지(Path Coverage)도 달성합니까? 이유를 설명하세요.

### 연습 3: 통합 테스팅(Integration Testing) 계획 수립

마이크로서비스 주문 시스템에는 다음 서비스가 있습니다: `OrderService`, `InventoryService`, `PaymentService`, `NotificationService`. `OrderService`는 `InventoryService`와 `PaymentService`를 호출하고, `NotificationService`는 결제 성공 후 `OrderService`에 의해 호출됩니다.

(a) 통합 의존성 그래프를 스케치하세요.
(b) 점진적 상향식(Bottom-Up) 통합 테스트 계획을 설계하세요: 각 단계에서 통합되는 서비스, 필요한 스텁(Stub)/드라이버(Driver), 각 단계가 검증하는 인터페이스 계약을 명시하세요.
(c) 통합 수준에서만 발견할 수 있는 세 가지 통합 특화 실패 모드(단위 수준 버그가 아닌)를 식별하세요.

### 연습 4: 새로운 기능을 위한 테스트 계획 작성

모바일 뱅킹 앱이 PIN 입력 대안으로 생체 인증(지문 및 Face ID) 로그인을 추가하고 있습니다. 다음을 포함하는 한 페이지 분량의 테스트 계획 개요를 작성하세요:

- 범위(Scope): 이 기능의 테스트 주기에서 포함 및 제외 사항
- 테스트 수준(Test Levels): 적용되는 수준(단위, 통합, 시스템, 인수)과 각 수준에서 테스트하는 내용
- 테스팅 유형(Testing Types): 고려해야 할 비기능 유형 최소 네 가지와 각각이 중요한 이유
- 진입 및 종료 기준(Entry/Exit Criteria): 구체적이고 측정 가능한 조건
- 테스트 노력에 대한 상위 세 가지 위험과 각각의 완화 전략

### 연습 5: 버그 생명주기(Bug Lifecycle) 분석

테스터가 사용자가 10,000행 보고서를 PDF로 내보내려고 할 때 파일이 생성되는 대신 32초 후 504 게이트웨이 타임아웃(Gateway Timeout)이 반환되는 현상을 발견했습니다.

(a) 11.3절의 템플릿에 따라 완전한 버그 리포트를 작성하세요.
(b) 심각도(Severity)와 우선순위(Priority) 등급을 지정하고 각각 독립적으로 근거를 제시하세요.
(c) 근본 원인에 대한 두 가지 가설을 제안하세요 (하나는 애플리케이션 계층, 하나는 인프라 계층).
(d) 개발자가 버그를 수정됨(Fixed)으로 표시한 후 테스터가 수행해야 할 검증 단계를 설명하세요.

---

**이전**: [07. 소프트웨어 품질 보증](./07_Software_Quality_Assurance.md) | **다음**: [09. 형상 관리](./09_Configuration_Management.md)
