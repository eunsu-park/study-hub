# 레슨 07: 소프트웨어 품질 보증(Software Quality Assurance)

**이전**: [06. 추정](./06_Estimation.md) | **다음**: [08. 검증 및 확인](./08_Verification_and_Validation.md)

---

소프트웨어 품질은 우연히 생겨나지 않는다. 그것은 소프트웨어 개발 생명주기 전반에 걸쳐 의도적으로 적용되는 프로세스, 표준, 측정, 그리고 문화의 결과물이다. 이 레슨에서는 소프트웨어 품질 보증(SQA, Software Quality Assurance) 분야를 탐구한다 — 소프트웨어 제품과 프로세스의 품질을 정의하고, 측정하고, 체계적으로 개선한다는 것이 무엇을 의미하는지를 다룬다.

**난이도**: ⭐⭐⭐

**선수 학습**:
- 소프트웨어 개발 생명주기 기본 이해 (레슨 02)
- 소프트웨어 테스팅 개념 친숙도
- 기본 프로그래밍 지식

**학습 목표**:
- IEEE 및 ISO 25010 프레임워크를 사용하여 소프트웨어 품질 정의하기
- 품질 보증(Quality Assurance), 품질 관리(Quality Control), 테스팅의 차이 구별하기
- 소프트웨어 프로젝트에 품질 비용(Cost of Quality) 모델 적용하기
- 일반적인 소프트웨어 메트릭(순환 복잡도(Cyclomatic Complexity), 응집도(Cohesion), 결합도(Coupling)) 계산 및 해석하기
- 정적 분석(Static Analysis) 도구를 사용하여 코드 품질 측정 및 개선하기
- 기술 부채(Technical Debt) 식별 및 관리하기
- 효과적인 코드 리뷰(Code Review) 진행 및 참여하기

---

## 목차

1. [소프트웨어 품질 정의](#1-소프트웨어-품질-정의)
2. [품질 속성: ISO/IEC 25010](#2-품질-속성-isoiec-25010)
3. [QA vs QC vs 테스팅](#3-qa-vs-qc-vs-테스팅)
4. [품질 비용](#4-품질-비용)
5. [QA 프로세스](#5-qa-프로세스)
6. [소프트웨어 메트릭](#6-소프트웨어-메트릭)
7. [정적 분석 및 코드 품질 도구](#7-정적-분석-및-코드-품질-도구)
8. [품질 표준](#8-품질-표준)
9. [기술 부채](#9-기술-부채)
10. [코드 리뷰](#10-코드-리뷰)
11. [요약](#11-요약)
12. [연습 문제](#12-연습-문제)
13. [더 읽을거리](#13-더-읽을거리)

---

## 1. 소프트웨어 품질 정의

품질은 소프트웨어 공학에서 가장 논쟁이 많은 용어 중 하나다. 각각 다른 관점을 강조하는 여러 정의가 공존한다.

**IEEE 정의 (IEEE Std 730)**:
> "시스템, 구성 요소, 또는 프로세스가 명시된 요구사항을 충족하는 정도."

이는 *적합성(conformance) 기반* 관점이다: 품질은 요청된 것을 충족하는 것을 의미한다.

**크로스비(Crosby)의 정의**:
> "품질은 요구사항에 대한 적합성이다."

**주란(Juran)의 정의**:
> "사용 적합성(Fitness for use)."

이는 문서에서 사용자 필요로 초점을 이동시킨다 — 요구사항에 부합하더라도, 요구사항이 잘못된 경우 품질이 낮을 수 있다.

**ISO/IEC 25010 (SQuaRE)**:
제품 품질과 사용 품질(Quality in Use)에 대한 다차원 모델을 제공한다 (Section 2에서 자세히 다룸).

### 품질 딜레마

이러한 정의들은 실용적인 긴장을 만들어낸다:

```
관점                  질문                        무시할 경우 위험
──────────────────────────────────────────────────────────────
적합성(Conformance)   올바르게 만들었는가?          기술적 실패
사용 적합성           올바른 것을 만들었는가?        사용자 거부
가치(Value)           만들 가치가 있는가?            비즈니스 실패
탁월성(Excellence)    최선을 다했는가?              경쟁력 상실
```

실무에서는 완전한 품질 프로그램이 네 가지 모두를 다룬다. 레슨 08 (V&V)은 처음 두 가지를 심층적으로 다루고, 이 레슨은 네 가지 모두를 지원하는 프로세스와 측정에 초점을 맞춘다.

---

## 2. 품질 속성: ISO/IEC 25010

ISO/IEC 25010 (SQuaRE — 시스템 및 소프트웨어 품질 요구사항 및 평가(System and Software Quality Requirements and Evaluation) 시리즈의 일부)은 두 가지 품질 모델을 정의한다:

- **제품 품질 모델(Product quality model)**: 소프트웨어 산출물의 고유한 특성
- **사용 품질 모델(Quality in use model)**: 특정 맥락에서 시스템이 사용될 때의 결과

### 2.1 제품 품질 모델

| 품질 특성 | 하위 특성 | 의미 |
|-----------|-----------|------|
| **기능 적합성(Functional Suitability)** | 완전성, 정확성, 적합성 | 소프트웨어가 해야 할 일을 하는가? |
| **신뢰성(Reliability)** | 성숙도, 가용성, 결함 허용성, 회복성 | 예상 및 예상치 못한 조건에서 얼마나 잘 동작하는가? |
| **성능 효율성(Performance Efficiency)** | 시간 동작, 자원 활용, 용량 | 충분히 빠르고 자원 효율적인가? |
| **사용성(Usability)** | 적합성 인식성, 학습 용이성, 운용성, 사용자 오류 방지, UI 심미성, 접근성 | 사용자가 쉽게 목표를 달성할 수 있는가? |
| **보안성(Security)** | 기밀성, 무결성, 부인 방지, 책임 추적성, 진위성 | 데이터를 보호하고 공격에 저항하는가? |
| **호환성(Compatibility)** | 공존성, 상호운용성 | 다른 시스템과 함께 작동하는가? |
| **유지보수성(Maintainability)** | 모듈성, 재사용성, 분석성, 수정성, 시험성 | 과도한 노력 없이 변경 가능한가? |
| **이식성(Portability)** | 적응성, 설치성, 대체성 | 새로운 환경으로 이전 가능한가? |

### 2.2 사용 품질 모델

| 특성 | 설명 |
|------|------|
| **효과성(Effectiveness)** | 사용자가 완전하고 정확하게 목표를 달성할 수 있는가? |
| **효율성(Efficiency)** | 적절한 자원 소비로 목표를 달성하는가? |
| **만족도(Satisfaction)** | 사용자의 필요와 기대가 충족되는가? |
| **위험으로부터의 자유(Freedom from Risk)** | 경제, 안전, 환경에 대한 위험을 완화하는가? |
| **맥락 커버리지(Context Coverage)** | 대상 맥락 범위 전반에서 작동하는가? |

### 2.3 품질 속성 우선순위 결정

모든 속성이 모든 시스템에서 동등하게 중요한 것은 아니다. 심박 조율기 펌웨어는 무엇보다 신뢰성과 안전성을 우선시한다. 마케팅 랜딩 페이지는 사용성과 성능을 우선시한다. 뱅킹 API는 보안성과 신뢰성을 우선시한다.

품질 속성 우선순위는 **품질 속성 유틸리티 트리(Quality Attribute Utility Tree)** (아키텍처 실무에서 일반적)에 기록된다:

```
Root: System Quality
  ├── Performance
  │     ├── Response time < 200ms under 1000 concurrent users  [HIGH, HIGH]
  │     └── Throughput > 10,000 transactions/min               [HIGH, MEDIUM]
  ├── Security
  │     ├── No SQL injection vulnerabilities                    [HIGH, HIGH]
  │     └── Session tokens expire after 30 minutes             [MEDIUM, LOW]
  └── Maintainability
        └── New feature added in < 2 developer days            [MEDIUM, MEDIUM]
```

두 주석 `[중요도, 난이도]`는 팀이 아키텍처 결정의 우선순위를 정하는 데 도움을 준다.

---

## 3. QA vs QC vs 테스팅

이 세 용어는 종종 혼동된다. 차이를 이해하는 것이 일관된 품질 프로그램을 구축하는 데 필수적이다.

```
┌─────────────────────────────────────────────────────────────────────┐
│  품질 보증(Quality Assurance, QA)                                   │
│  프로세스 중심. 결함이 제품에 유입되는 것을 방지.                   │
│  예시: 코딩 표준, 프로세스 감사, 교육, 리뷰                        │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  품질 관리(Quality Control, QC)                               │  │
│  │  제품 중심. 제품에서 결함을 식별.                             │  │
│  │  예시: 리뷰, 검사, 워크스루                                   │  │
│  │                                                               │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │  테스팅(Testing)                                        │  │  │
│  │  │  실행 중심. 코드를 실행하여 장애를 발견.                │  │  │
│  │  │  예시: 단위 테스트, 통합 테스트, UAT                    │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

| 차원 | QA | QC | 테스팅 |
|------|----|----|--------|
| 초점 | 프로세스 | 제품 | 실행 |
| 목표 | 결함 방지 | 결함 발견 | 장애 발견 |
| 시기 | SDLC 전반 | 체크포인트 시점 | 코드 존재 후 |
| 산출물 | 표준, 프로세스 문서 | 결함 보고서, 리뷰 결과 | 테스트 결과, 버그 보고서 |
| 범위 | 모든 활동 | 산출물 및 납품물 | 실행 중인 소프트웨어 |

**핵심 통찰**: QA는 *관리 기능*이다. QA 팀은 단순히 테스트를 실행하는 것이 아니라 — 품질을 가능하게 하는 프로세스를 정의하고 시행한다. 테스팅은 품질에 필요하지만 충분하지 않다.

---

## 4. 품질 비용

필립 크로스비(Philip Crosby)의 **품질 비용(Cost of Quality, CoQ)** 모델은 품질 관련 비용이 네 가지 범주로 나뉜다고 주장한다. 이 모델은 직관에 반하는 진실을 드러낸다: *예방에 투자하면 총 비용이 감소한다*.

### 4.1 네 가지 CoQ 범주

```
품질 비용(Cost of Quality)
├── 적합 비용(Cost of Conformance) (낮은 품질 방지에 쓰는 비용)
│   ├── 예방 비용(Prevention Costs)
│   │   ├── 교육(Training)
│   │   ├── 프로세스 문서화
│   │   ├── 코딩 표준
│   │   ├── 정적 분석 도구
│   │   └── 아키텍처 리뷰
│   └── 평가 비용(Appraisal Costs)
│       ├── 코드 리뷰
│       ├── 테스트 실행
│       ├── 테스트 인프라
│       └── QA 감사
│
└── 부적합 비용(Cost of Non-Conformance) (낮은 품질 처리에 쓰는 비용)
    ├── 내부 실패 비용(Internal Failure Costs) (출시 전 발견)
    │   ├── 버그 수정
    │   ├── 재작업(Rework)
    │   ├── 수정 후 회귀 테스트
    │   └── 출시 지연
    └── 외부 실패 비용(External Failure Costs) (출시 후 고객 발견)
        ├── 고객 지원
        ├── 패치 및 핫픽스
        ├── 법적 책임
        ├── 평판 손상
        └── 고객 이탈
```

### 4.2 1-10-100 규칙

결함 수정 비용은 발견 시점이 늦을수록 극적으로 증가한다:

| 발견 단계 | 상대적 비용 |
|----------|------------|
| 요구사항 | 1x |
| 설계 | 5x |
| 코딩 | 10x |
| 단위 테스트 | 15x |
| 통합 테스트 | 25x |
| 시스템 테스트 | 50x |
| 프로덕션 | 100x+ |

이것이 QA에 대한 초기 투자(요구사항 리뷰, 설계 리뷰, 정적 분석)가 초기에 비용이 많이 드는 것처럼 느껴지더라도 높은 수익을 내는 이유다.

### 4.3 최적 품질 투자

고전적인 경제 모델은 총 CoQ가 0이 아닌 결함률에서 최솟값을 가진다는 것을 보여준다:

```
비용
 ▲
 │     총 비용(Total Cost)
 │       ╲     ╱
 │        ╲   ╱
 │  부적합  ╲ ╱ 비용
 │  비용     X
 │          ╱ ╲
 │         ╱   ╲ 예방/
 │        ╱     ╲ 평가 비용
 │───────────────────────────────▶ 결함률(Defect Rate)
         ↑
      최적점(Optimum)
```

그러나 현대의 애자일(Agile)과 데브옵스(DevOps) 실무는 이 곡선을 이동시킨다: 자동화가 예방을 저렴하게 만들어 최적점을 거의 0의 결함 쪽으로 밀어붙인다.

---

## 5. QA 프로세스

공식적인 SQA 프로그램에는 일반적으로 다음 활동들이 포함된다:

### 5.1 SQA 계획

**소프트웨어 품질 보증 계획(Software Quality Assurance Plan, SQAP)** (IEEE Std 730)은 다음을 문서화한다:
- 적용할 품질 표준 및 메트릭
- 수행할 리뷰, 감사, 테스팅
- 각 QA 활동에 대한 책임자
- 결함 보고 및 추적 절차
- 사용할 도구

### 5.2 표준 및 절차

SQA는 다음을 수립한다:
- **코딩 표준(Coding standards)**: 명명 규칙, 형식, 문서화 요구사항
- **프로세스 표준(Process standards)**: 요구사항 작성 방법, 설계 리뷰 방법
- **문서화 표준(Documentation standards)**: 필요한 문서와 그 형식

### 5.3 리뷰 및 감사

| 활동 | 목적 | 참여자 |
|------|------|--------|
| **요구사항 리뷰** | 완전성, 일관성, 시험성 검증 | 작성자, 분석가, 고객 |
| **설계 리뷰** | 아키텍처 및 설계 결정 평가 | 아키텍트, 개발자, QA |
| **코드 리뷰** | 결함 발견, 표준 시행 | 개발자, 동료 |
| **테스트 계획 리뷰** | 테스트 커버리지 및 접근 방식 검증 | QA, 개발자, PM |
| **프로세스 감사** | 팀이 정의된 프로세스를 따르는지 확인 | QA 리드, 경영진 |
| **제품 감사** | 납품물이 표준을 충족하는지 확인 | QA, 고객 |

### 5.4 메트릭 수집 및 분석

SQA는 프로세스 문제를 조기에 감지하기 위해 메트릭을 지속적으로 수집한다. 리팩토링 후 결함 밀도가 급증하면, 그 신호가 다음 출시 전에 조사를 촉발한다.

---

## 6. 소프트웨어 메트릭

소프트웨어 메트릭(Software Metrics)은 소프트웨어 제품 또는 프로세스의 측면을 정량화하여 객관적인 평가 및 추세 추적을 가능하게 한다.

### 6.1 제품 메트릭

#### 순환 복잡도(Cyclomatic Complexity, McCabe, 1976)

프로그램 소스 코드를 통한 선형 독립 경로의 수를 측정한다.

```
CC = E - N + 2P

여기서:
  E = 제어 흐름 그래프의 엣지(edge) 수
  N = 노드(node) 수
  P = 연결된 컴포넌트 수 (보통 1)
```

단일 함수의 경우, 이것은 단순화된다: **CC = 결정 지점 수 + 1**

```python
def classify_triangle(a, b, c):          # CC starts at 1
    if a <= 0 or b <= 0 or c <= 0:       # +1 → 2
        return "Invalid"
    if a + b <= c or b + c <= a or a + c <= b:  # +1, +1, +1 → 5
        return "Not a triangle"
    if a == b == c:                       # +1 → 6
        return "Equilateral"
    elif a == b or b == c or a == c:      # +1, +1, +1 → 9
        return "Isosceles"
    else:
        return "Scalene"
# CC = 9
```

| CC 값 | 위험도 | 권장 조치 |
|-------|--------|----------|
| 1–10 | 낮음 | 허용 가능 |
| 11–20 | 보통 | 리팩토링 고려 |
| 21–50 | 높음 | 리팩토링 필요 |
| > 50 | 매우 높음 | 반드시 리팩토링; 테스트 불가 |

#### 결합도와 응집도(Coupling and Cohesion)

이 두 메트릭은 품질 측면에서 서로 역관계다:

**응집도(Cohesion)** — 단일 모듈 내 책임들이 얼마나 강하게 연관되어 있는지.

```
높은 응집도 (Good)             낮은 응집도 (Bad)
─────────────────────          ─────────────────────
UserAuthenticator               UtilityHelper
  + login()                       + login()
  + logout()                      + formatDate()
  + resetPassword()               + sendEmail()
  + validateToken()               + parseCSV()
                                  + calculateTax()
```

**결합도(Coupling)** — 하나의 모듈이 다른 모듈에 얼마나 의존하는지.

```
낮은 결합도 (Good)             높은 결합도 (Bad)
──────────────────────         ──────────────────────
OrderService → IPayment        OrderService → PaypalPayment
(인터페이스에 의존)             (구체 클래스,
                                내부 상태, DB 스키마에 의존)
```

| 응집도 유형 | 설명 | 품질 |
|------------|------|------|
| 기능적(Functional) | 모든 요소가 하나의 잘 정의된 작업을 위해 작동 | 최선 |
| 순차적(Sequential) | 한 요소의 출력이 다음 요소로 전달 | 좋음 |
| 통신적(Communicational) | 요소들이 동일한 데이터에서 작동 | 보통 |
| 절차적(Procedural) | 요소들이 고정된 실행 순서를 따름 | 보통 |
| 시간적(Temporal) | 요소들이 실행 시점에 따라 그룹화 | 낮음 |
| 논리적(Logical) | 요소들이 제어 플래그로 선택 | 낮음 |
| 우연적(Coincidental) | 의미 있는 관계 없음 | 최악 |

#### 할스테드 메트릭(Halstead Metrics)

할스테드(Halstead, 1977)는 토큰 개수에 기반하여 소프트웨어를 측정한다:

```
n1 = 고유 연산자(distinct operators) 수
n2 = 고유 피연산자(distinct operands) 수
N1 = 연산자 총 출현 횟수
N2 = 피연산자 총 출현 횟수

어휘(Vocabulary):  n  = n1 + n2
길이(Length):      N  = N1 + N2
볼륨(Volume):      V  = N × log2(n)
난이도(Difficulty): D  = (n1/2) × (N2/n2)
노력(Effort):      E  = D × V
```

할스테드 볼륨은 모듈을 이해하고 수정하는 데 필요한 노력과 상관관계가 있다.

### 6.2 프로세스 메트릭

| 메트릭 | 공식 | 의미 |
|--------|------|------|
| **결함 밀도(Defect Density)** | 결함 수 / KLOC | 1000줄당 버그 비율 |
| **결함 제거 효율(Defect Removal Efficiency, DRE)** | 출시 전 발견된 결함 / (전 + 후) × 100% | 출시 전 잡힌 결함 비율 |
| **평균 고장 간격(Mean Time Between Failures, MTBF)** | 총 가동 시간 / 고장 횟수 | 배포된 시스템의 신뢰성 |
| **평균 수리 시간(Mean Time to Repair, MTTR)** | 총 수리 시간 / 고장 횟수 | 고장에 대한 대응 속도 |
| **테스트 커버리지(Test Coverage)** | 실행된 코드 / 전체 코드 × 100% | 테스트가 실행하는 코드 비율 |
| **이탈 결함률(Escaped Defect Rate)** | 프로덕션 버그 / 전체 버그 × 100% | 모든 QA 활동에서 놓친 결함 |

DRE의 업계 벤치마크: 세계적 수준의 조직은 95–99%를 달성한다.

---

## 7. 정적 분석 및 코드 품질 도구

정적 분석(Static Analysis)은 코드를 실행하지 않고 소스 코드를 검사하여 결함, 스타일 위반, 보안 취약점, 복잡도 문제를 찾아낸다.

### 7.1 정적 분석의 범주

```
정적 분석(Static Analysis)
├── 스타일 및 형식(Style and Formatting)
│   └── 코드가 합의된 규칙을 따르는가?
│   └── 도구: Prettier, Black, gofmt, clang-format
│
├── 린팅(Linting)
│   └── 일반적인 프로그래밍 오류나 안티 패턴이 있는가?
│   └── 도구: ESLint, Pylint, Flake8, RuboCop, Checkstyle
│
├── 복잡도 분석(Complexity Analysis)
│   └── 이해하거나 테스트하기에 너무 복잡한 함수가 있는가?
│   └── 도구: Lizard, Radon, SonarQube
│
├── 보안 스캐닝 (SAST, Static Application Security Testing)
│   └── 알려진 취약점 패턴(SQL 인젝션, XSS 등)이 있는가?
│   └── 도구: Bandit (Python), Semgrep, CodeQL, Checkmarx
│
└── 의존성 스캐닝(Dependency Scanning)
    └── 알려진 CVE가 있는 의존성이 있는가?
    └── 도구: Dependabot, Snyk, OWASP Dependency-Check
```

### 7.2 SonarQube: 통합 플랫폼

SonarQube는 많은 정적 분석 관심사를 하나의 대시보드에 통합한다. 주요 개념:

| SonarQube 개념 | 의미 |
|----------------|------|
| **버그(Bug)** | 잘못되어 런타임 오류를 일으킬 가능성이 높은 코드 |
| **취약점(Vulnerability)** | 공격에 취약한 코드 |
| **코드 스멜(Code Smell)** | 유지보수는 가능하지만 불필요하게 복잡하거나 혼란스러운 코드 |
| **보안 핫스팟(Security Hotspot)** | 보안 맥락에서 사람의 검토가 필요한 코드 |
| **기술 부채(Technical Debt)** | 모든 이슈에 대한 예상 수정 시간 |
| **품질 게이트(Quality Gate)** | CI/CD가 시행하는 합격/불합격 임계값 |

일반적인 품질 게이트 구성:

```yaml
# sonar-project.properties
sonar.qualitygate.wait=true

# Quality Gate conditions:
# - Coverage on new code >= 80%
# - Duplicated lines on new code < 3%
# - Maintainability rating on new code = A
# - Reliability rating on new code = A
# - Security rating on new code = A
```

### 7.3 CI/CD에 정적 분석 통합하기

```yaml
# GitHub Actions example
name: Code Quality

on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run linter
        run: |
          pip install flake8 pylint
          flake8 src/ --max-line-length=100
          pylint src/ --fail-under=8.0

      - name: Run security scan
        run: |
          pip install bandit
          bandit -r src/ -ll  # report medium and high severity only

      - name: Check complexity
        run: |
          pip install radon
          radon cc src/ -a -n C  # fail if average CC > C (10)

      - name: SonarQube Scan
        uses: SonarSource/sonarqube-scan-action@master
        env:
          SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
          SONAR_HOST_URL: ${{ secrets.SONAR_HOST_URL }}
```

---

## 8. 품질 표준

### 8.1 ISO 9001:2015

ISO 9001은 어떤 조직에도 적용 가능한 일반 품질 관리 표준이다. 소프트웨어 회사의 경우 다음을 요구한다:

- 문서화된 품질 관리 시스템
- 프로세스 준수 증거
- 시정 및 예방 조치 프로세스
- 품질 메트릭에 대한 경영진 검토
- 고객 중심 및 만족도 측정

ISO 9001 인증은 정부 기관이나 대기업과 비즈니스를 하기 위해 종종 요구된다.

### 8.2 ISO/IEC 25010 (SQuaRE)

이미 Section 2에서 다뤘다. 이 표준은 소프트웨어 제품의 품질 요구사항과 평가 기준을 정의하는 기반이다.

### 8.3 CMMI (역량 성숙도 모델 통합, Capability Maturity Model Integration)

CMMI는 카네기 멜론 대학교의 소프트웨어 공학 연구소(Software Engineering Institute)가 개발한 프로세스 개선 프레임워크다. 다섯 가지 성숙도 수준을 정의한다:

```
레벨 5: 최적화(Optimizing)    ← 지속적인 프로세스 개선
레벨 4: 정량적 관리(Quantitatively Managed) ← 측정 및 통제
레벨 3: 정의됨(Defined)       ← 문서화된 표준화된 프로세스
레벨 2: 관리됨(Managed)       ← 계획 및 추적
레벨 1: 초기(Initial)         ← 임시방편적, 혼란스러움
```

| 레벨 | 특성 | 일반적인 조직 |
|------|------|--------------|
| 1 초기 | 성공이 개인 역량에 의존; 반복 가능한 프로세스 없음 | 스타트업 |
| 2 관리됨 | 기본 프로젝트 관리; 프로젝트별 반복 가능 | 소규모 회사 |
| 3 정의됨 | 조직 전체 표준 프로세스; 프로젝트별 맞춤 | 중견 기업 |
| 4 정량적 관리 | 통계적 프로세스 통제; 예측 가능한 품질 | 성숙한 기업 |
| 5 최적화 | 데이터 기반 지속적 개선 | 세계적 수준 조직 |

대부분의 상업적 소프트웨어 조직은 레벨 2 또는 3에서 운영된다. 방위산업 및 항공우주 계약업체는 종종 레벨 3–5를 요구한다.

---

## 9. 기술 부채

워드 커닝엄(Ward Cunningham)은 1992년 "기술 부채(Technical Debt)"를 은유로 만들었다: 지금 지름길을 택하는 것은 돈을 빌리는 것과 같다 — 오늘 이익을 얻지만 시간이 지남에 따라 이자(속도 저하, 버그, 취약성)를 지불한다.

### 9.1 기술 부채의 유형

마틴 파울러(Martin Fowler)의 기술 부채 사분면(Technical Debt Quadrant):

```
                무모한(Reckless)        신중한(Prudent)
             ┌──────────────────┬──────────────────┐
  의도적      │ "설계할 시간이   │ "지금 출시해야   │
  (Deliberate)│  없다"           │  하고 결과는     │
              │                  │  나중에 처리"    │
              ├──────────────────┼──────────────────┤
  비의도적    │ "계층화가        │ "이제 어떻게     │
  (Inadvertent)│ 뭔데?"          │  했어야 했는지   │
              │                  │  알겠다"         │
              └──────────────────┴──────────────────┘
```

- **의도적 무모한 부채**: 위험 — 팀이 더 잘 알면서도 지름길을 택함
- **의도적 신중한 부채**: 허용 가능 — 상환 계획이 있는 의식적인 결정
- **비의도적 무모한 부채**: 일반적 — 팀이 기술이나 지식 부족
- **비의도적 신중한 부채**: 불가피 — 개발 중에 얻은 교훈

### 9.2 기술 부채 측정

SonarQube는 수정 시간으로 부채를 추정한다. 다른 접근 방식:

```python
# Simple debt estimation model
def estimate_debt_hours(metrics):
    debt = 0

    # Complex functions: 30 min per function above CC threshold
    complex_functions = sum(1 for cc in metrics['cyclomatic']
                           if cc > 10)
    debt += complex_functions * 0.5

    # Low test coverage: 20 min per uncovered function
    uncovered = metrics['total_functions'] * (1 - metrics['coverage'])
    debt += uncovered * 0.33

    # Duplicated code: 1 hour per duplicate block
    debt += metrics['duplicate_blocks'] * 1.0

    # Known code smells from static analysis
    debt += metrics['code_smells'] * 0.25

    return debt  # in hours
```

### 9.3 기술 부채 관리

실무에서 사용되는 전략:

| 전략 | 설명 | 사용 시기 |
|------|------|----------|
| **보이 스카우트 규칙(Boy Scout Rule)** | 발견한 것보다 코드를 더 깨끗하게 남겨두기; 지나치면서 작은 것들 수정 | 항상 |
| **부채 스프린트(Debt Sprints)** | 부채 감소를 위한 스프린트(또는 각 스프린트의 20%) 할당 | 정기적으로 |
| **기능 동결(Feature Freeze)** | 새 기능 중단; 품질을 위한 출시 주기 투자 | 품질이 심각하게 저하된 경우 |
| **스트랭글러 피그(Strangler Fig)** | 레거시 서브시스템을 깨끗한 코드로 점진적으로 교체 | 대형 레거시 시스템 |
| **터치 시 리팩토링(Refactor on Touch)** | 모듈에 새 기능을 추가하기 전에 리팩토링 | 모듈이 변경될 예정인 경우 |

핵심 원칙: **부채를 가시화하라**. 이슈 트래커에서 추적하라. 스토리 포인트를 부여하라. 속도 계산에 부채 감소를 포함시켜라.

---

## 10. 코드 리뷰

코드 리뷰(Code Review)는 가장 비용 효율적인 결함 예방 기법 중 하나다. 연구들은 코드 리뷰가 테스트 실행 전에 결함의 60–90%를 발견한다는 것을 일관되게 보여준다.

### 10.1 코드 리뷰의 유형

| 유형 | 공식성 | 노력 | 최적 사용 |
|------|--------|------|----------|
| **페어 프로그래밍(Pair Programming)** | 비공식 | 지속적 | 고위험 코드; 지식 이전 |
| **어깨 너머 리뷰(Over-the-Shoulder)** | 비공식 | 낮음 | 빠른 온전성 검사 |
| **도구 지원 리뷰(Tool-Assisted Review, PR 리뷰)** | 반공식 | 중간 | 일상적인 개발 |
| **워크스루(Walkthrough)** | 공식 | 중간 | 교육; 광범위한 리뷰 |
| **파건 검사(Fagan Inspection)** | 공식 | 높음 | 안전 필수 코드 |

### 10.2 풀 리퀘스트 리뷰 체크리스트

```
코드 정확성(Code Correctness)
  □ 티켓/스토리가 요구하는 것을 하는가?
  □ 엣지 케이스(null, 빈 값, 오버플로)가 처리되는가?
  □ 오류 처리가 적절하고 일관적인가?
  □ 동시 코드에 경쟁 조건이 있는가?

코드 품질(Code Quality)
  □ 주석 없이도 코드를 읽을 수 있는가?
  □ 변수/함수 이름이 설명적인가?
  □ 추출해야 할 중복 로직이 있는가?
  □ 순환 복잡도가 허용 가능한가 (함수당 < 10)?

테스팅(Testing)
  □ 새로운 동작에 대한 단위 테스트가 있는가?
  □ 테스트가 행복한 경로뿐 아니라 실패 케이스도 커버하는가?
  □ 테스트 커버리지가 유지되거나 향상되는가?

보안(Security)
  □ 사용자 입력이 검증/정제(sanitized)되는가?
  □ 코드에 시크릿/자격증명이 없는가?
  □ 권한 부여 검사가 올바른가?

성능(Performance)
  □ 명백한 N+1 쿼리 문제가 있는가?
  □ 비용이 많이 드는 작업이 적절히 캐시되는가?

문서화(Documentation)
  □ 공개 API가 문서화되었는가?
  □ 복잡한 알고리즘이 설명되어 있는가?
  □ CHANGELOG 또는 릴리즈 노트가 업데이트되었는가?
```

### 10.3 리뷰를 효과적으로 만들기

**리뷰어(reviewer)를 위해**:
- 집중된 세션으로 최대 60–90분 동안 리뷰 (이후 집중력 저하)
- 논리, 설계, 정확성에 집중 — 스타일 검사는 자동화
- 요구하기보다 질문하기 ("X가 null이면 어떻게 됩니까?" 라고 하되 "이것은 틀렸다"라고 하지 않기)
- 문제뿐 아니라 좋은 코드도 인정하기

**작성자(author)를 위해**:
- PR을 작게 유지 (순 신규 코드 400줄 미만이 일반적인 가이드라인)
- 명확한 설명 작성: 무엇을, 왜, 어떻게
- 리뷰 요청 전에 자기 검토
- 관련 이슈/티켓에 링크

**PR 크기 가이드라인**:
```
변경 줄 수            리뷰 품질 영향
< 50               철저한 리뷰; 거의 모든 버그 발견
50–400             좋은 리뷰; 대부분의 버그 발견
400–800            보통; 리뷰어가 긴 섹션을 훑어봄
> 800              형식적; 많은 버그 놓침; 리뷰어가 고통 종료를 위해 승인
```

---

## 11. 요약

소프트웨어 품질은 표준(ISO 25010)으로 정의되고, 메트릭(순환 복잡도, 결함 밀도, DRE)으로 측정되며, QA 프로세스를 통해 체계적으로 관리되는 다차원적 속성이다.

핵심 요점:

- **QA는 프로세스 중심** (결함 방지); QC는 제품 중심 (결함 감지); 테스팅은 실행 중심 (장애 발견).
- **조기에 투자하라**: 품질 비용 모델은 예방과 평가가 실패 비용보다, 특히 외부 실패보다 훨씬 저렴하다는 것을 보여준다.
- **중요한 것을 측정하라**: 순환 복잡도는 테스트 불가능한 코드를 표시하고; 결합도와 응집도는 설계 문제를 드러내고; DRE는 전반적인 QA 프로그램의 효과성을 측정한다.
- **도구를 자동화하라**: CI/CD에 통합된 정적 분석은 수동 노력 없이 많은 부류의 문제를 잡아낸다.
- **기술 부채는 실제 비용이다**: 가시화하고, 추적하고, 기능 작업과 함께 감소를 위한 역량을 할당하라.
- **코드 리뷰는 높은 ROI를 제공한다**: 명확한 체크리스트를 갖춘 소규모 집중 리뷰는 자동화된 도구가 놓치는 결함을 잡아낸다.

---

## 12. 연습 문제

**연습 문제 1 — 순환 복잡도**

다음 Python 함수의 순환 복잡도를 계산하라. 그런 다음 CC를 4 이하로 줄이도록 리팩토링하라.

```python
def process_order(order):
    if order is None:
        return None
    if order.status == "cancelled":
        return {"error": "Order is cancelled"}
    if order.items:
        for item in order.items:
            if item.quantity < 0:
                raise ValueError(f"Invalid quantity for {item.name}")
            if item.price < 0:
                raise ValueError(f"Invalid price for {item.name}")
            if item.quantity == 0:
                continue
            item.total = item.price * item.quantity
    if order.discount_code:
        if order.discount_code == "SAVE10":
            order.discount = 0.10
        elif order.discount_code == "SAVE20":
            order.discount = 0.20
        else:
            return {"error": "Invalid discount code"}
    order.total = sum(i.total for i in order.items if i.quantity > 0)
    if order.discount:
        order.total *= (1 - order.discount)
    return order
```

**연습 문제 2 — 품질 비용 분석**

소프트웨어 팀에는 다음과 같은 월별 비용 데이터가 있다:
- 코딩 표준 시행: $2,000
- 코드 리뷰 시간: $8,000
- 단위 테스트: $5,000
- 출시 전 버그 수정: $15,000
- 프로덕션 버그에 대한 고객 지원: $25,000
- 긴급 핫픽스 배포: $10,000

(a) 각 비용을 예방, 평가, 내부 실패, 외부 실패로 분류하라.
(b) 총 품질 비용과 각 범주의 비율을 계산하라.
(c) 예방에 월 $5,000을 추가 투자하면 내부 실패가 40%, 외부 실패가 25% 감소한다면, 팀이 이 투자를 해야 하는가? 논리를 보여라.

**연습 문제 3 — 품질 게이트 설계**

핀테크(fintech) API를 개발하는 팀의 QA 리드다. 이 시스템을 위한 SonarQube 품질 게이트를 설계하라. 도메인에 기반하여 각 선택을 정당화하면서 최소 6개의 조건을 임계값과 함께 명시하라.

**연습 문제 4 — 기술 부채 백로그**

레거시 전자상거래 애플리케이션에 대해 정적 분석 도구가 발견한 다음 코드 스멜을 검토하라:
- CC > 15인 함수 47개 (평균 CC = 22)
- 테스트 커버리지: 34%
- 중복 코드 블록 280개
- 잠재적 SQL 인젝션 취약점 3개
- 더 이상 사용되지 않는(deprecated) API 메서드 사용 12건

(a) 파울러의 기술 부채 사분면을 사용하여 각 항목을 분류하라.
(b) 부채 감소에 시간의 20%를 할당할 수 있는 두 명의 개발자가 있는 팀을 위해 목록의 우선순위를 정하라.
(c) 3개 스프린트 부채 감소 계획을 작성하라.

**연습 문제 5 — 코드 리뷰**

시니어 엔지니어처럼 다음 Python 함수를 리뷰하라. 정확성, 품질, 보안, 성능 차원에서 최소 다섯 가지 이슈를 식별하라. 각 이슈에 대해 무엇이 잘못될 수 있는지 설명하고 수정안을 제안하라.

```python
def get_user_data(user_id):
    conn = sqlite3.connect("app.db")
    query = "SELECT * FROM users WHERE id = " + str(user_id)
    result = conn.execute(query).fetchall()
    user = {}
    for row in result:
        user['id'] = row[0]
        user['name'] = row[1]
        user['email'] = row[2]
        user['password'] = row[3]
        user['admin'] = row[4]
    return user
```

---

## 13. 더 읽을거리

- **도서**:
  - *Code Complete* (2nd ed.) — Steve McConnell. 24–25장에서 품질 보증을 포괄적으로 다룬다.
  - *Clean Code* — Robert C. Martin. 유지보수 가능한 코드 작성에 대한 실용적인 지침.
  - *The Art of Software Testing* — Glenford Myers et al.
  - *Working Effectively with Legacy Code* — Michael Feathers. 기술 부채 관리에 대한 결정적인 가이드.

- **표준**:
  - ISO/IEC 25010:2011 — Systems and software Quality Requirements and Evaluation (SQuaRE)
  - IEEE Std 730-2014 — IEEE Standard for Software Quality Assurance Processes
  - CMMI Institute — https://cmmiinstitute.com/

- **도구**:
  - SonarQube Community Edition — https://www.sonarqube.org/
  - Radon (Python 복잡도) — https://radon.readthedocs.io/
  - Lizard (다언어 복잡도) — https://github.com/terryyin/lizard
  - Semgrep (패턴 기반 정적 분석) — https://semgrep.dev/

- **논문**:
  - McCabe, T. J. (1976). "A Complexity Measure." *IEEE Transactions on Software Engineering*.
  - Fagan, M. E. (1976). "Design and Code Inspections to Reduce Errors in Program Development." *IBM Systems Journal*.
  - Cunningham, W. (1992). "The WyCash Portfolio Management System." *OOPSLA*.

---

**이전**: [06. 추정](./06_Estimation.md) | **다음**: [08. 검증 및 확인](./08_Verification_and_Validation.md)
