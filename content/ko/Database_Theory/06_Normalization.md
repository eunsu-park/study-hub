# Lesson 06: 정규화 (1NF ~ BCNF)

**이전**: [05_Functional_Dependencies.md](./05_Functional_Dependencies.md) | **다음**: [07_Advanced_Normalization.md](./07_Advanced_Normalization.md)

---

> **주제**: 데이터베이스 이론(Database Theory)
> **레슨**: 16개 중 6번째
> **선수 지식**: 함수 종속성, 속성 폐쇄, 최소 커버 (Lesson 05)
> **목표**: 1NF부터 BCNF까지의 정규화 이해, 분해 알고리즘 마스터, 무손실 조인과 종속성 보존 속성 검증, 실제 스키마에 정규화 적용

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 비정규화 스키마에서 발생하는 세 가지 갱신 이상(삽입, 삭제, 갱신 이상)을 식별하고 정규화가 이를 어떻게 해결하는지 설명합니다.
2. 원자 값, 부분 종속성(Partial Dependency), 이행 종속성(Transitive Dependency)을 확인하여 릴레이션이 1NF, 2NF, 3NF, BCNF를 만족하는지 판단합니다.
3. BCNF 분해 알고리즘(BCNF Decomposition Algorithm)을 적용하여 스키마의 무손실 조인 분해를 생성합니다.
4. 3NF 합성 알고리즘(3NF Synthesis Algorithm)을 적용하여 무손실 조인 및 종속성 보존 분해를 생성합니다.
5. 형식적 테스트를 사용하여 주어진 분해의 무손실 조인(Lossless-Join) 및 종속성 보존(Dependency-Preservation) 속성을 검증합니다.
6. BCNF와 3NF 분해 전략 간의 트레이드오프를 평가하고 주어진 스키마에 적합한 정규화 목표를 선택합니다.

---

## 1. 소개(Introduction)

정규화(Normalization)는 관계형 데이터베이스 스키마를 중복을 줄이고 특정 유형의 데이터 이상 현상을 제거하기 위해 조직하는 과정입니다. Edgar F. Codd가 1970년에 도입한 이 방법은 스키마 설계에 대한 체계적이고 이론 기반의 접근 방식을 제공합니다.

### 1.1 문제: 잘못된 스키마 설계

대학을 위한 단일 릴레이션 설계를 고려해보세요:

```
UniversityCourse(
    student_id, student_name, student_addr,
    course_id, course_title, dept_name, dept_building,
    instructor_id, instructor_name,
    grade, semester
)
```

이 "범용 릴레이션"은 모든 것을 하나의 테이블에 저장합니다. 간단한 쿼리에는 작동하지만 심각한 문제를 겪습니다.

### 1.2 데이터 이상 현상(Data Anomalies)

#### 갱신 이상(Update Anomaly)

컴퓨터 과학부가 새 건물로 이사하면, `dept_name = 'Computer Science'`인 **모든 행**에서 `dept_building`을 업데이트해야 합니다. 단 하나의 행이라도 놓치면 데이터가 불일치해집니다.

```
변경 전:
| course_id | dept_name | dept_building |
|-----------|-----------|---------------|
| CS101     | CS        | Watson Hall   |
| CS201     | CS        | Watson Hall   |    ← 모든 행을 업데이트해야 함
| CS301     | CS        | Watson Hall   |

첫 번째 행만 업데이트하면:
| CS101     | CS        | Taylor Hall   |    ← 업데이트됨
| CS201     | CS        | Watson Hall   |    ← 불일치!
| CS301     | CS        | Watson Hall   |    ← 불일치!
```

#### 삽입 이상(Insertion Anomaly)

학생이 그 부서의 과목 중 하나에 등록하지 않는 한, 새 부서(예: 이름과 건물)를 기록할 수 없습니다. `student_id`가 기본 키의 일부이기 때문입니다.

#### 삭제 이상(Deletion Anomaly)

과목에 등록한 마지막 학생이 탈퇴하면, 등록 데이터뿐만 아니라 과목 제목, 강사 배정, 부서 정보도 잃게 됩니다.

### 1.3 근본 원인: FD 위반으로 인한 중복

세 가지 이상 현상은 모두 동일한 근본 원인에서 비롯됩니다: **키의 일부에만 의존하거나 비키 속성에 의존하는 속성이 중복되어 저장됩니다**. 정규화는 함수 종속성에 의해 안내되는 체계적인 분해를 통해 이러한 중복을 제거합니다.

### 1.4 정규화의 목표

1. **중복 제거**: 각 사실은 정확히 한 번만 저장됩니다
2. **이상 방지**: 업데이트, 삽입, 삭제가 깔끔합니다
3. **정보 보존**: 분해 중에 데이터가 손실되지 않습니다(무손실 조인)
4. **제약 조건 보존**: FD가 여전히 강제 가능합니다(종속성 보존)

---

## 2. 제1정규형(First Normal Form, 1NF)

### 2.1 정의

> **정의**: 릴레이션이 다음을 만족하면 **제1정규형(1NF)**에 있습니다:
> 1. 모든 속성이 **원자적(atomic)** (분할 불가능한) 값만 포함
> 2. **반복 그룹**이나 배열이 없음
> 3. 각 행이 고유하게 식별 가능 (기본 키 있음)

1NF는 관계형 모델에서 유효한 릴레이션이 되기 위한 기본 요구사항입니다.

### 2.2 위반과 수정

**위반 1: 비원자적 값**

```
| student_id | name       | phone_numbers          |
|------------|------------|------------------------|
| 101        | Alice      | 555-1234, 555-5678     |    ← 다중값!
| 102        | Bob        | 555-9999               |
```

**수정**: 각 전화번호에 대해 별도의 행을 만들거나 별도의 테이블 생성:

```
Student(student_id, name)
StudentPhone(student_id, phone_number)

| student_id | phone_number |
|------------|--------------|
| 101        | 555-1234     |
| 101        | 555-5678     |
| 102        | 555-9999     |
```

**위반 2: 반복 그룹**

```
| order_id | item1  | qty1 | item2  | qty2 | item3  | qty3 |
|----------|--------|------|--------|------|--------|------|
| 1001     | Pen    | 5    | Paper  | 10   | NULL   | NULL |
```

**수정**: 두 테이블로 정규화:

```
Order(order_id, order_date, customer_id)
OrderItem(order_id, item_name, quantity)
```

### 2.3 1NF와 관계형 모델

엄격한 관계형 이론에서 릴레이션은 정의상 1NF에 있습니다 — 관계형 모델은 비원자적 도메인을 허용하지 않습니다. 그러나 실제로 많은 시스템이 배열(PostgreSQL `int[]`), JSON 컬럼, 쉼표로 구분된 값을 허용합니다. 성능에는 유용할 수 있지만 이들은 1NF의 정신을 위반하고 FD 기반 추론을 어렵게 만듭니다.

---

## 3. 제2정규형(Second Normal Form, 2NF)

### 3.1 부분 종속성(Partial Dependency)

> **정의**: **부분 종속성**은 비주요 속성(어떤 후보 키의 일부가 아닌 속성)이 후보 키의 **진부분집합**에 함수적으로 종속될 때 존재합니다.

즉, 어떤 속성이 키 전체가 아닌 키의 *일부*에만 의존합니다.

### 3.2 정의

> **정의**: 릴레이션이 다음을 만족하면 **제2정규형(2NF)**에 있습니다:
> 1. 1NF에 있고,
> 2. 모든 비주요 속성이 모든 후보 키에 **완전 함수적으로 종속** (부분 종속성 없음)

참고: 2NF는 후보 키가 복합(하나 이상의 속성을 가짐)일 때만 관련이 있습니다. 모든 후보 키가 단일 속성이면 릴레이션은 자동으로 2NF에 있습니다.

### 3.3 예시

다음을 고려하세요:

```
StudentCourse(student_id, course_id, student_name, grade)
```

후보 키: {student_id, course_id}

FD:
- {student_id, course_id} → grade (완전 종속성)
- student_id → student_name (부분 종속성! student_name은 키의 일부에만 의존)

이것은 2NF를 위반합니다.

**2NF를 달성하기 위한 분해:**

```
Student(student_id, student_name)
    키: {student_id}
    FD: student_id → student_name

Enrollment(student_id, course_id, grade)
    키: {student_id, course_id}
    FD: {student_id, course_id} → grade
```

### 3.4 2NF에 대한 형식적 테스트

각 후보 키 K와 각 비주요 속성 A에 대해:
1. X ⊂ K인 진부분집합 X가 존재하여 X → A인지 확인
2. 그러한 부분 종속성이 존재하면 릴레이션은 2NF에 없음

---

## 4. 제3정규형(Third Normal Form, 3NF)

### 4.1 전이 종속성(Transitive Dependency)

> **정의**: **전이 종속성**은 비주요 속성 A가 다른 비주요 속성 B에 의존하고, B는 후보 키 K에 의존할 때 존재합니다:
>
> K → B → A, 여기서 B는 슈퍼키가 아니고 A는 어떤 후보 키의 일부가 아닙니다.

### 4.2 정의

> **정의**: 릴레이션 스키마 R이 A가 단일 속성인 모든 비자명한 함수 종속성 X → A에 대해 다음을 만족하면 **제3정규형(3NF)**에 있습니다:
> 1. X가 R의 슈퍼키, **또는**
> 2. A가 **주요 속성**(어떤 후보 키의 멤버)

동등하게: 어떤 비주요 속성도 어떤 후보 키에 전이적으로 종속되지 않습니다.

### 4.3 예시

다음을 고려하세요:

```
Employee(emp_id, dept_id, dept_name, dept_location)
```

후보 키: {emp_id}

FD:
- emp_id → dept_id, dept_name, dept_location
- dept_id → dept_name, dept_location

종속성 emp_id → dept_name은 dept_id를 통해 전이적입니다:
- emp_id → dept_id → dept_name

이것은 3NF를 위반합니다. dept_name이 dept_id(비슈퍼키)에 의존하고 dept_name이 주요 속성이 아니기 때문입니다.

**3NF를 달성하기 위한 분해:**

```
Employee(emp_id, dept_id)
    키: {emp_id}
    FD: emp_id → dept_id

Department(dept_id, dept_name, dept_location)
    키: {dept_id}
    FD: dept_id → dept_name, dept_location
```

### 4.4 "주요 속성" 예외의 중요성

3NF의 "A가 주요 속성"이라는 조건은 BCNF와 구별되는 것입니다. 이 예외는 결정자가 슈퍼키가 아니더라도 종속자가 후보 키의 일부인 특정 FD를 허용합니다. 이 예외가 3NF 분해를 항상 종속성 보존으로 만드는 것입니다(BCNF와는 달리).

---

## 5. Boyce-Codd 정규형(Boyce-Codd Normal Form, BCNF)

### 5.1 정의

> **정의**: 릴레이션 스키마 R이 모든 비자명한 함수 종속성 X → Y에 대해 다음을 만족하면 **Boyce-Codd 정규형(BCNF)**에 있습니다:
>
> X가 R의 슈퍼키입니다.

BCNF는 3NF보다 엄격하게 강합니다. "주요 속성" 예외를 제거합니다: **모든** 결정자가 슈퍼키여야 합니다. 마침표.

### 5.2 관계: 3NF vs BCNF

```
BCNF ⊂ 3NF ⊂ 2NF ⊂ 1NF

모든 BCNF 릴레이션은 3NF에 있습니다.
모든 3NF 릴레이션은 2NF에 있습니다.
모든 2NF 릴레이션은 1NF에 있습니다.

하지만 그 역은 아닙니다.
```

### 5.3 예시: 3NF이지만 BCNF가 아님

다음을 고려하세요:

```
TeachingAssignment(student_id, course, instructor)
```

비즈니스 규칙:
- 각 강사는 정확히 하나의 과목을 가르침: instructor → course
- 각 학생은 과목당 정확히 한 명의 강사를 가짐: {student_id, course} → instructor
- 학생은 주어진 과목에 대해 한 명의 강사만 가질 수 있음

후보 키: {student_id, course}와 {student_id, instructor}

FD:
- {student_id, course} → instructor
- {student_id, instructor} → course
- instructor → course

instructor → course에 대한 3NF 확인:
- instructor는 슈퍼키가 아님 ✗
- course는 주요 속성(후보 키 {student_id, course}의 일부) ✓

따라서 릴레이션은 **3NF**에 있습니다(3NF 정의의 조건 2가 만족됨).

instructor → course에 대한 BCNF 확인:
- instructor는 슈퍼키가 아님 ✗

따라서 릴레이션은 **BCNF에 없습니다**.

### 5.4 3NF와 BCNF가 다를 때

3NF와 BCNF는 다음의 경우에만 다릅니다:
1. 여러 개의 중첩된 후보 키가 있고,
2. 비슈퍼키 속성이 후보 키의 일부를 결정함

실제로 이 상황은 상대적으로 드뭅니다. 3NF에 있는 대부분의 릴레이션은 BCNF에도 있습니다.

---

## 6. 분해 속성(Decomposition Properties)

릴레이션을 더 높은 정규형을 달성하기 위해 분해할 때, 분해가 "좋은"지 확인해야 합니다. 두 가지 중요한 속성이 "좋음"의 의미를 정의합니다.

### 6.1 무손실 조인 속성(Lossless-Join Property)

> **정의**: R을 R₁, R₂, ..., Rₙ으로 분해할 때 R의 모든 합법적 인스턴스 r에 대해 다음을 만족하면 **무손실 조인 속성**을 가집니다:
>
> r = π_{R₁}(r) ⋈ π_{R₂}(r) ⋈ ... ⋈ π_{Rₙ}(r)

즉, 분해된 릴레이션을 자연 조인하여 원래 릴레이션을 재구성할 수 있습니다. 정보가 손실되지 않습니다.

#### 이진 분해 테스트

R을 R₁과 R₂로 분해하는 경우:

> **정리**: 분해가 무손실 조인이 되려면 다음 중 하나가 만족되어야 합니다:
>
> (R₁ ∩ R₂) → (R₁ - R₂) ∈ F⁺, 또는
> (R₁ ∩ R₂) → (R₂ - R₁) ∈ F⁺

공통 속성이 한 쪽의 모든 속성을 함수적으로 결정해야 합니다. 동등하게, 공통 속성이 분해된 릴레이션 중 적어도 하나의 슈퍼키여야 합니다.

#### 예시

Employee(emp_id, dept_id, dept_name)을 다음과 같이 분해:
- R₁(emp_id, dept_id)와 R₂(dept_id, dept_name)

공통 속성: {dept_id}
R₂ - R₁ = {dept_name}

dept_id → dept_name이 F⁺에 있는가? 예!

따라서 이 분해는 무손실 조인입니다. ✓

#### 무손실 조인이 중요한 이유

무손실 조인 속성이 없으면 분해된 테이블을 조인할 때 **허위 튜플(spurious tuples)** — 원래 릴레이션에 존재하지 않았던 행 — 이 생성됩니다:

```
원본:
| A | B | C |
|---|---|---|
| 1 | 2 | 3 |
| 4 | 2 | 5 |

R1(A,B)와 R2(B,C)로 분해:
R1:          R2:
| A | B |    | B | C |
|---|---|    |---|---|
| 1 | 2 |    | 2 | 3 |
| 4 | 2 |    | 2 | 5 |

R1 ⋈ R2:
| A | B | C |
|---|---|---|
| 1 | 2 | 3 |    ← 원본 ✓
| 1 | 2 | 5 |    ← 허위! ✗
| 4 | 2 | 3 |    ← 허위! ✗
| 4 | 2 | 5 |    ← 원본 ✓
```

여기서 B는 A나 C를 결정하지 않으므로 분해는 손실(무손실이 아님)입니다.

### 6.2 종속성 보존(Dependency Preservation)

> **정의**: R을 R₁, R₂, ..., Rₙ으로 분해할 때 다음을 만족하면 **종속성 보존**입니다:
>
> (F₁ ∪ F₂ ∪ ... ∪ Fₙ)⁺ = F⁺
>
> 여기서 Fᵢ는 Rᵢ의 속성만 포함하는 F⁺의 FD 집합입니다.

더 간단하게: 원래 F의 모든 FD를 테이블 조인 없이 개별 분해된 릴레이션 내의 제약 조건을 확인하여 검증할 수 있습니다.

#### 종속성 보존이 중요한 이유

분해가 종속성 보존이 아니면 일부 FD는 여러 테이블을 조인해야만 강제할 수 있습니다 — 이것은 비용이 많이 들고 종종 비실용적입니다. 종속성 보존이 없으면 DBMS는 데이터 무결성을 효율적으로 유지할 수 없습니다.

#### 예시

R(A, B, C)에 F = { A → B, B → C }가 있음

R₁(A, C)와 R₂(B, C)로 분해.

R₁의 FD: A → B를 강제할 수 있는가? 아니오 — B가 R₁에 없음.
R₂의 FD: A → B를 강제할 수 있는가? 아니오 — A가 R₂에 없음.

FD A → B가 **보존되지 않습니다**. 이를 확인하려면 R₁과 R₂를 조인해야 합니다.

더 나은 분해: R₁(A, B)와 R₂(B, C) — 이것은 A → B와 B → C 모두 보존합니다.

### 6.3 둘 다 가질 수 있는가?

| 정규형 | 무손실 조인 | 종속성 보존 |
|-------------|:------------:|:----------------------:|
| 3NF | 항상 달성 가능 ✓ | 항상 달성 가능 ✓ |
| BCNF | 항상 달성 가능 ✓ | **항상은 아님** ✗ |

이것이 핵심 트레이드오프입니다: BCNF는 더 엄격한 정규형이지만 달성하면 종속성 보존을 희생할 수 있습니다. 3NF는 두 속성을 모두 보장합니다.

---

## 7. 무손실 조인 테스트 알고리즘 (n-원 분해용)

두 개 이상의 릴레이션으로 분해하는 경우 테이블 알고리즘을 사용합니다.

### 7.1 알고리즘 (체이스 테스트, Chase Test)

```
ALGORITHM: LosslessJoinTest(R, F, {R₁, R₂, ..., Rₙ})
INPUT:  R = {A₁, A₂, ..., Aₘ} (m개 속성)
        F = FD 집합
        {R₁, R₂, ..., Rₙ} = 분해
OUTPUT: 무손실 조인이면 TRUE, 아니면 FALSE

단계 1: n × m 행렬 생성.
        행 i는 Rᵢ에 대응, 열 j는 속성 Aⱼ에 대응.

단계 2: 행렬 초기화:
        Aⱼ ∈ Rᵢ이면 entry[i][j] = aⱼ (구별 기호)
        아니면 entry[i][j] = bᵢⱼ (첨자 기호)

단계 3: REPEAT
            FOR EACH FD (X → Y) IN F DO
                X의 모든 열에서 일치하는 모든 행 찾기
                그러한 행들에 대해 Y-열을 동일하게 만들기:
                    Y의 어떤 열에 대해 어떤 행이 aⱼ를 가지면 모든 일치하는 행을 aⱼ로 설정
                    아니면 하나의 bᵢⱼ를 선택하고 모든 일치하는 행을 그 값으로 설정
            END FOR
        UNTIL 변경 발생하지 않음

단계 4: 어떤 행이 모든 구별 기호를 가지면 (a₁, a₂, ..., aₘ), TRUE 반환.
        아니면 FALSE 반환.
```

### 7.2 작업 예제

R(A, B, C, D, E), F = { A → C, B → C, C → D, DE → C, CE → A }

분해: R₁(A, D), R₂(A, B), R₃(B, E), R₄(C, D, E), R₅(A, E)

**단계 1-2: 초기 행렬**

|      | A   | B   | C   | D   | E   |
|------|-----|-----|-----|-----|-----|
| R₁   | a₁  | b₁₂ | b₁₃ | a₄  | b₁₅ |
| R₂   | a₁  | a₂  | b₂₃ | b₂₄ | b₂₅ |
| R₃   | b₃₁ | a₂  | b₃₃ | b₃₄ | a₅  |
| R₄   | b₄₁ | b₄₂ | a₃  | a₄  | a₅  |
| R₅   | a₁  | b₅₂ | b₅₃ | b₅₄ | a₅  |

**단계 3: FD를 반복적으로 적용**

A → C 적용: 동일한 A 값을 가진 행.
- R₁, R₂, R₅ 행이 A = a₁을 가짐. C 열을 동일하게 만들기.
  - R₁: b₁₃, R₂: b₂₃, R₅: b₅₃. 구별 기호 없음. 모두에 대해 b₁₃ 선택.
  - R₁: b₁₃, R₂: b₁₃, R₅: b₁₃

B → C 적용: R₂, R₃ 행이 B = a₂를 가짐.
- R₂: b₁₃, R₃: b₃₃. b₁₃ 선택.
- R₃: b₁₃

C → D 적용: 동일한 C 값을 가진 행.
- R₁, R₂, R₃, R₅ 행이 모두 C = b₁₃을 가짐. R₄는 C = a₃ (다름).
  - R₁: a₄, R₂: b₂₄, R₃: b₃₄, R₅: b₅₄. R₁이 구별된 a₄를 가짐. 모두 a₄로 설정.
  - R₂: a₄, R₃: a₄, R₅: a₄

DE → C 적용: 동일한 D와 E 값을 가진 행.
- R₃: D=a₄, E=a₅; R₄: D=a₄, E=a₅; R₅: D=a₄, E=a₅.
  - 이 행들이 DE에서 일치. C를 동일하게: R₃: b₁₃, R₄: a₃, R₅: b₁₃. R₄가 a₃를 가짐. 모두 a₃로 설정.
  - R₃: a₃, R₅: a₃.

CE → A 적용: 동일한 C와 E를 가진 행.
- R₃: C=a₃, E=a₅; R₄: C=a₃, E=a₅; R₅: C=a₃, E=a₅.
  - R₅가 A=a₁ (구별됨). R₃: a₁, R₄: a₁로 설정.

이제 A → C 재적용: R₁,R₂,R₃,R₄,R₅ 행이 A=a₁을 가짐.
- C 값: R₁=b₁₃, R₂=b₁₃, R₃=a₃, R₄=a₃, R₅=a₃. a₃가 있음. 모두 a₃로 설정.
- R₁=a₃, R₂=a₃.

C → D를 모든 행에 재적용 (이제 모두 C=a₃): 이미 모두 D=a₄. 변경 없음.

R₅ 행 확인: A=a₁, B=b₅₂, C=a₃, D=a₄, E=a₅. 여전히 B에 대해 b₅₂ 있음.

B → C 적용: R₂가 B=a₂, R₃이 B=a₂를 가짐. 둘 다 이미 C=a₃. 변경 없음.

확인: R₃ 행 = (a₁, a₂, a₃, a₄, a₅) — **모든 구별 기호!**

**결과: 분해는 무손실 조인입니다. ✓**

---

## 8. 3NF 합성 알고리즘

3NF 합성 알고리즘은 **무손실 조인**과 **종속성 보존** 모두를 가진 분해를 생성합니다.

### 8.1 알고리즘

```
ALGORITHM: 3NF_Synthesis(R, F)
INPUT:  R = 릴레이션 스키마, F = FD 집합
OUTPUT: 무손실 조인과 종속성 보존을 가진 3NF 분해 {R₁, R₂, ..., Rₙ}

단계 1: F의 최소 커버 F_min 계산.

단계 2: F_min의 각 FD X → A에 대해:
            릴레이션 스키마 Rᵢ = X ∪ {A} 생성
        동일한 LHS를 가진 FD 그룹화:
            X → A₁, X → A₂, ..., X → Aₖ가 모두 동일한 X를 가지면,
            하나의 스키마 Rᵢ = X ∪ {A₁, A₂, ..., Aₖ} 생성

단계 3: 스키마 중 어느 것도 R의 후보 키를 포함하지 않으면,
        스키마 Rₖ = R의 임의의 후보 키 추가.

단계 4: 다른 스키마 Rⱼ의 부분집합인 스키마 Rᵢ 제거.

RETURN {R₁, R₂, ..., Rₙ}
```

### 8.2 각 단계가 중요한 이유

- **단계 1** (최소 커버): 중복 FD가 추가 테이블을 생성하지 않도록 보장
- **단계 2** (LHS 그룹당 하나의 테이블): 각 FD를 직접 보존
- **단계 3** (필요한 경우 키 추가): 무손실 조인 속성 보장
- **단계 4** (부분집합 제거): 중복 테이블 제거

### 8.3 작업 예제

R(A, B, C, D, E, H)에 다음이 있음:

```
F = { A → BC,  E → HA,  B → D }
```

**단계 1: 최소 커버 계산**

RHS 분해:
```
F = { A → B, A → C, E → H, E → A, B → D }
```

불필요한 LHS 확인: 모든 LHS가 단일 속성. 단순화할 것 없음.

중복 FD 확인:
- A → B: 제거. 나머지 하에서 {A}⁺ = {A, C} ∪ ... = {A, C}. B ∉ {A}⁺. 유지.
- A → C: 제거. 나머지 하에서 {A}⁺ = {A, B, D} (A→B, B→D를 통해). C ∉ {A}⁺. 유지.
- E → H: 제거. 나머지 하에서 {E}⁺ = {E, A, B, C, D} (E→A, A→B, A→C, B→D를 통해). H ∉ {E}⁺. 유지.
- E → A: 제거. 나머지 하에서 {E}⁺ = {E, H}. A ∉ {E}⁺. 유지.
- B → D: 제거. 나머지 하에서 {B}⁺ = {B}. D ∉ {B}⁺. 유지.

최소 커버:
```
F_min = { A → B, A → C, E → H, E → A, B → D }
```

**단계 2: LHS로 그룹화하고 스키마 생성**

| LHS | FD | 스키마 |
|-----|-----|--------|
| A | A → B, A → C | R₁(A, B, C) |
| E | E → H, E → A | R₂(E, H, A) |
| B | B → D | R₃(B, D) |

**단계 3: 후보 키 확인**

{E}⁺ = {E, H, A, B, C, D} = 모든 속성 계산. 따라서 {E}가 후보 키.

E가 어떤 스키마에 있는가? R₂가 E를 포함. R₂의 키는 E. ✓ (후보 키 존재.)

**단계 4: 부분집합 스키마 제거**

어느 것도 다른 것의 부분집합이 아님.

**최종 분해:**
```
R₁(A, B, C)    — 키: {A}
R₂(E, H, A)    — 키: {E}
R₃(B, D)       — 키: {B}
```

모두 3NF ✓, 무손실 조인 ✓, 종속성 보존 ✓.

---

## 9. BCNF 분해 알고리즘

### 9.1 알고리즘

```
ALGORITHM: BCNF_Decomposition(R, F)
INPUT:  R = 릴레이션 스키마, F = FD 집합
OUTPUT: 무손실 조인을 가진 BCNF 릴레이션으로 분해

result ← {R}

WHILE result에 BCNF에 없는 Rᵢ가 존재 DO
    Rᵢ에서 BCNF를 위반하는 FD X → Y 찾기
    (즉, X가 Rᵢ의 슈퍼키가 아니고, Y ⊄ X)

    F에 대해 X⁺ 계산

    Rᵢ를 다음으로 대체:
        R₁ = X⁺ ∩ attributes(Rᵢ)    (Rᵢ 내에서 X가 결정하는 모든 것)
        R₂ = X ∪ (attributes(Rᵢ) - X⁺)   (X + 결정하지 않는 것)
END WHILE

RETURN result
```

핵심 분해 단계는 Rᵢ를 다음으로 분할합니다:
- R₁: (Rᵢ 내에서) X가 결정하는 모든 속성 — X는 R₁의 키
- R₂: X + 나머지 속성 — X는 R₁로 돌아가는 외래 키

이것은 무손실 조인을 보장합니다 (R₁ ∩ R₂ = X, 그리고 X → R₁).

### 9.2 작업 예제

R(A, B, C, D)에 F = { AB → C, C → B, AB → D }가 있음

**단계 1: R이 BCNF에 있는가?**

후보 키: {A, B} ({A,B}⁺ = {A,B,C,D}).
또한 {A, C}가 후보 키 ({A,C}⁺: C→B는 {A,B,C}를 제공, AB→D는 {A,B,C,D}를 제공).

FD 확인:
- AB → C: {A,B}는 슈퍼키. ✓
- C → B: {C}⁺ = {C, B}. C는 슈퍼키가 아님. **BCNF 위반!**
- AB → D: {A,B}는 슈퍼키. ✓

**단계 2: C → B에서 분해**

{C}⁺ ∩ {A,B,C,D} = {B, C} ∩ {A,B,C,D} = {B, C} 계산

- R₁ = {B, C}에 FD: C → B (키: C)
- R₂ = {C} ∪ ({A,B,C,D} - {B,C}) = {A, C, D}

R₂(A, C, D)에 FD 투영:
- AB → C는 관련 없음 (B ∉ R₂)
- C → B는 관련 없음 (B ∉ R₂)
- AB → D는... 확인 필요: 투영된 FD 하에서 R₂에 대한 {A,C}⁺. C→B는 R₂에 없음. 하지만 원본에서: AC → D? {A,C}⁺ = {A,C,B,D} (C→B, AB→D). 따라서 AC → D가 성립. R₂의 키는 {A, C}.
- 확인: AC가 R₂의 슈퍼키인가? R₂로 제한된 {A,C}⁺ = {A,C,D}. 예. ✓

**단계 3: R₁과 R₂ 확인**

- R₁(B, C), FD: C → B. 키: {C}. C는 슈퍼키. BCNF ✓
- R₂(A, C, D), FD: AC → D. 키: {A, C}. AC는 슈퍼키. BCNF ✓

**최종 BCNF 분해:**
```
R₁(B, C)      — 키: {C}
R₂(A, C, D)   — 키: {A, C}
```

**종속성 보존 확인:**
- AB → C: R₁이나 R₂ 단독으로 확인 불가 (A와 B가 같은 테이블에 없음). **보존되지 않음!**
- C → B: R₁에 있음. ✓
- AB → D: 직접 보존되지 않지만, AC → D는 R₂에 있음.

이것은 BCNF 트레이드오프를 보여줍니다: BCNF를 달성했지만 FD AB → C를 잃었습니다.

### 9.3 BCNF vs 3NF: 트레이드오프

| 속성 | 3NF 합성 | BCNF 분해 |
|----------|:------------:|:------------------:|
| 무손실 조인 | ✓ 항상 | ✓ 항상 |
| 종속성 보존 | ✓ 항상 | ✗ 항상은 아님 |
| 중복 제거 | 좋음 (최소) | 최상 (FD로부터 없음) |
| 선호하는 경우 | 종속성 보존이 중요할 때 | 최소 중복이 중요할 때 |

**실용적 지침**:
- BCNF 분해부터 시작
- 종속성 보존이 실패하면 3NF로 후퇴
- 실제로 대부분의 스키마는 종속성을 잃지 않고 BCNF를 달성

---

## 10. 완전한 작업 예제: 비정규화에서 BCNF까지

### 10.1 시나리오

회사가 프로젝트 배정을 추적:

```
ProjectAssignment(
    emp_id, emp_name, emp_phone,
    dept_id, dept_name, dept_budget,
    proj_id, proj_name, proj_budget,
    hours_worked, role
)
```

**비즈니스 규칙 (FD):**
```
F = {
    emp_id → emp_name, emp_phone, dept_id,
    dept_id → dept_name, dept_budget,
    proj_id → proj_name, proj_budget,
    {emp_id, proj_id} → hours_worked, role
}
```

**후보 키**: {emp_id, proj_id}

### 10.2 1NF 확인

모든 값이 원자적 (단일 값, 배열 없음). ✓ 1NF에 있음.

### 10.3 2NF 확인

비주요 속성: emp_name, emp_phone, dept_id, dept_name, dept_budget, proj_name, proj_budget, hours_worked, role.

부분 종속성 (속성이 키 {emp_id, proj_id}의 진부분집합에 의존):
- emp_id → emp_name, emp_phone, dept_id (부분: emp_id 단독에 의존)
- proj_id → proj_name, proj_budget (부분: proj_id 단독에 의존)

**2NF에 없음.** 분해:

```
Employee(emp_id, emp_name, emp_phone, dept_id)
    FD: emp_id → emp_name, emp_phone, dept_id

Project(proj_id, proj_name, proj_budget)
    FD: proj_id → proj_name, proj_budget

Assignment(emp_id, proj_id, hours_worked, role)
    FD: {emp_id, proj_id} → hours_worked, role
```

이제 2NF에 있음. ✓

### 10.4 3NF 확인

**Employee(emp_id, emp_name, emp_phone, dept_id)**

키: {emp_id}

전이 종속성이 있는가?
- emp_id → dept_id (직접) ✓
- 하지만 dept_name, dept_budget은 어디에? 제거되었음 — 하지만 원래 FD dept_id → dept_name, dept_budget이 있음. dept_name과 dept_budget이 더 이상 이 릴레이션에 없으므로 이 릴레이션 내에 전이 종속성이 존재하지 않음.

실제로 재고해봅시다. 원래 FD dept_id → dept_name, dept_budget은 Employee가 dept_name이나 dept_budget을 포함해서는 안 된다는 것을 의미합니다. 그리고 포함하지 않습니다 — 단계 2의 분해와 함께 갔습니다. 하지만 Department 테이블이 필요합니다:

```
Employee(emp_id, emp_name, emp_phone, dept_id)
```

남은 속성들 사이에 전이 종속성이 있는가? emp_id → emp_name, emp_phone, dept_id. 모두 키로부터의 직접 종속성. 이 테이블 내에서 비주요 속성이 다른 비주요 속성을 결정하지 않음 (emp_phone이 dept_id를 결정하지 않음 등).

3NF에 있음 ✓.

하지만 원래 스키마의 FD dept_id → dept_name, dept_budget은 "고아"입니다. Department 테이블이 필요합니다:

```
Department(dept_id, dept_name, dept_budget)
    키: {dept_id}
```

**이제 모든 릴레이션이 3NF에 있음:**
```
Employee(emp_id, emp_name, emp_phone, dept_id)    — 키: {emp_id}
Department(dept_id, dept_name, dept_budget)         — 키: {dept_id}
Project(proj_id, proj_name, proj_budget)           — 키: {proj_id}
Assignment(emp_id, proj_id, hours_worked, role)     — 키: {emp_id, proj_id}
```

### 10.5 BCNF 확인

각 릴레이션에 대해 확인: 모든 결정자가 슈퍼키인가?

- **Employee**: emp_id → (emp_name, emp_phone, dept_id). emp_id가 키. ✓
- **Department**: dept_id → (dept_name, dept_budget). dept_id가 키. ✓
- **Project**: proj_id → (proj_name, proj_budget). proj_id가 키. ✓
- **Assignment**: {emp_id, proj_id} → (hours_worked, role). {emp_id, proj_id}가 키. ✓

**모두 BCNF에 있음!** ✓

### 10.6 분해 요약

```
원본 (비정규화):
    ProjectAssignment(emp_id, emp_name, emp_phone, dept_id, dept_name,
                      dept_budget, proj_id, proj_name, proj_budget,
                      hours_worked, role)

최종 (BCNF):
    Employee(emp_id, emp_name, emp_phone, dept_id)
    Department(dept_id, dept_name, dept_budget)
    Project(proj_id, proj_name, proj_budget)
    Assignment(emp_id, proj_id, hours_worked, role)
```

이상 현상 제거:
- **갱신**: 부서 이름 변경시 Department의 한 행만 업데이트
- **삽입**: 직원 없이 부서 추가 가능
- **삭제**: 프로젝트의 모든 배정 제거해도 프로젝트 정보 손실 안됨

---

## 11. SQL에서의 정규화

### 11.1 정규화된 스키마 구현

```sql
CREATE TABLE Department (
    dept_id     INT PRIMARY KEY,
    dept_name   VARCHAR(100) NOT NULL,
    dept_budget DECIMAL(12, 2) NOT NULL
);

CREATE TABLE Employee (
    emp_id    INT PRIMARY KEY,
    emp_name  VARCHAR(100) NOT NULL,
    emp_phone VARCHAR(20),
    dept_id   INT NOT NULL,
    FOREIGN KEY (dept_id) REFERENCES Department(dept_id)
);

CREATE TABLE Project (
    proj_id     INT PRIMARY KEY,
    proj_name   VARCHAR(100) NOT NULL,
    proj_budget DECIMAL(12, 2) NOT NULL
);

CREATE TABLE Assignment (
    emp_id       INT NOT NULL,
    proj_id      INT NOT NULL,
    hours_worked DECIMAL(6, 2) NOT NULL DEFAULT 0,
    role         VARCHAR(50) NOT NULL,
    PRIMARY KEY (emp_id, proj_id),
    FOREIGN KEY (emp_id) REFERENCES Employee(emp_id),
    FOREIGN KEY (proj_id) REFERENCES Project(proj_id)
);
```

### 11.2 쿼리를 통한 정규화 검증

기존 데이터베이스의 잠재적 정규화 문제 확인:

```sql
-- 잠재적 2NF 위반 확인: 부분 종속성
-- 복합 키의 일부와 상관관계가 있는 비키 컬럼에 중복 값이 있는 경우
SELECT emp_id, COUNT(DISTINCT emp_name) AS name_count
FROM project_assignment_denormalized
GROUP BY emp_id
HAVING COUNT(DISTINCT emp_name) > 1;
-- 이것이 행을 반환하면 emp_name이 일관되지 않게 저장됨 (갱신 이상)

-- 잠재적 3NF 위반 확인: 전이 종속성
-- 함께 이동하는 컬럼들은 누락된 엔티티를 나타낼 수 있음
SELECT dept_id, COUNT(DISTINCT dept_name) AS names
FROM employee_denormalized
GROUP BY dept_id
HAVING COUNT(DISTINCT dept_name) > 1;
-- 이것이 행을 반환하면 주어진 dept_id에 대해 dept_name이 일관되지 않음
```

---

## 12. 정규형 요약

| 정규형 | 조건 | 제거하는 것 |
|-------------|-----------|-----------|
| **1NF** | 원자적 값, 반복 그룹 없음 | 비관계형 구조 |
| **2NF** | 1NF + 부분 종속성 없음 | 부분 키 종속성으로 인한 중복 |
| **3NF** | 2NF + 전이 종속성 없음 | 전이 종속성으로 인한 중복 |
| **BCNF** | 모든 결정자가 슈퍼키 | 모든 FD 기반 중복 |

### 의사결정 흐름도

```
시작: FD F를 가진 릴레이션 R

R이 1NF에 있는가?
├── 아니오 → 비원자적 값과 반복 그룹 제거
└── 예 ↓

R이 2NF에 있는가?
├── 아니오 → 부분 종속성 제거 (키의 일부에 의존하는
│         속성 분리)
└── 예 ↓

R이 3NF에 있는가?
├── 아니오 → 전이 종속성 제거 (비키 속성에 의존하는
│         속성 분리)
└── 예 ↓

R이 BCNF에 있는가?
├── 아니오 → 확인: 종속성 보존을 잃어도 되는가?
│   ├── 예 → BCNF 알고리즘으로 분해
│   └── 아니오 → 3NF에 머물기
└── 예 → 완료!
```

---

## 13. 연습 문제(Exercises)

### 연습 문제 1: 정규형 식별

각 릴레이션에 대해 가장 높은 정규형 (1NF, 2NF, 3NF, 또는 BCNF) 식별:

**a)** R(A, B, C, D), 키: {A, B}, FD: A → C, AB → D

**b)** R(A, B, C), 키: {A}, FD: A → B, B → C

**c)** R(A, B, C, D), 키: {A}, FD: A → BCD

**d)** R(A, B, C), 키: {A, B}와 {A, C}, FD: AB → C, AC → B, B → C, C → B

<details>
<summary>해답</summary>

**a)** A → C는 부분 종속성 (C가 키 {A,B}의 일부에 의존). **1NF** (2NF 아님).

**b)** A → B (키로부터 직접, OK), B → C (전이: A → B → C). 3NF 아님. 하지만 부분 종속성 없음 (단일 속성 키), 따라서 2NF. **2NF** (3NF 아님).

**c)** 키로부터만 FD. 모든 결정자 (A)가 슈퍼키. **BCNF**.

**d)** B → C: B는 슈퍼키가 아니지만 C는 주요 속성 (키 {A,C}의 일부). 따라서 3NF 성립. B는 슈퍼키가 아니므로 BCNF 실패. **3NF** (BCNF 아님).
</details>

### 연습 문제 2: 3NF 합성

다음에 3NF 합성 알고리즘 적용:

R(A, B, C, D, E)에 F = { A → B, BC → D, D → E, E → C }

<details>
<summary>해답</summary>

**단계 1: 최소 커버**

RHS 분해: 이미 단일 속성.

BC → D에서 불필요한 LHS 확인:
- B 제거: {C}⁺ = {C}. D ∉ {C}⁺. B 유지.
- C 제거: {B}⁺ = {B}. D ∉ {B}⁺. C 유지.

중복 FD 확인:
- A → B: A→B 없이 {A}⁺ = {A}. B ∉ {A}⁺. 유지.
- BC → D: BC→D 없이 {B,C}⁺ = {B,C}. D ∉ {B,C}⁺. 유지.
- D → E: D→E 없이 {D}⁺ = {D}. E ∉ {D}⁺. 유지.
- E → C: E→C 없이 {E}⁺ = {E}. C ∉ {E}⁺. 유지.

F_min = { A → B, BC → D, D → E, E → C }

**단계 2: 스키마 생성 (LHS로 그룹화)**

- R₁(A, B) from A → B
- R₂(B, C, D) from BC → D
- R₃(D, E) from D → E
- R₄(E, C) from E → C

**단계 3: 후보 키 확인**

{A}⁺ = {A, B}. 전체 아님.
{A, C}⁺ = {A, B, C, D, E}. 전체! 후보 키: {A, C}.
{A, E}⁺ = {A, B, C, D, E} (E→C, A→B, BC→D, D→E). 전체! 후보 키: {A, E}.
{A, D}⁺ = {A, B, D, E, C}. 전체! 후보 키: {A, D}.

R₁-R₄ 중 어느 것도 {A,C}, {A,E}, 또는 {A,D}를 전체 포함하지 않음.
- R₁ = {A,B}: 아니오
- R₂ = {B,C,D}: A 없음
- R₃ = {D,E}: A 없음
- R₄ = {E,C}: A 없음

R₅ = {A, C} (또는 {A, D} 또는 {A, E}) 추가.

**단계 4: 부분집합 제거**

R₄(E, C) ⊆ R₂(B, C, D)? 아니오 (E가 R₂에 없음). 제거할 부분집합 없음.

**최종 분해:**
```
R₁(A, B)       — 키: {A}
R₂(B, C, D)    — 키: {B, C}
R₃(D, E)       — 키: {D}
R₄(E, C)       — 키: {E}
R₅(A, C)       — 키: {A, C} (R의 후보 키)
```

모두 3NF ✓, 무손실 조인 ✓, 종속성 보존 ✓.
</details>

### 연습 문제 3: BCNF 분해

BCNF로 분해:

R(A, B, C, D)에 F = { AB → C, C → A, C → D }

<details>
<summary>해답</summary>

후보 키: {A,B}와 {B,C} (검증: {A,B}⁺ = {A,B,C,D}, {B,C}⁺ = {A,B,C,D}).

BCNF 확인:
- AB → C: {A,B}는 슈퍼키. ✓
- C → A: {C}⁺ = {A,C,D}. C는 슈퍼키가 아님. **BCNF 위반!**
- C → D: 같은 문제. **BCNF 위반!**

C → A에서 분해 (또는 C → AD):
- {C}⁺ ∩ {A,B,C,D} = {A,C,D}
- R₁ = (A, C, D) with key {C}
- R₂ = {C} ∪ ({A,B,C,D} - {A,C,D}) = (B, C) with key {B,C}

R₁(A, C, D) 확인:
- C → A: C는 R₁의 키. ✓
- C → D: C는 R₁의 키. ✓
- BCNF ✓

R₂(B, C) 확인:
- 슈퍼키가 아닌 결정자를 가진 비자명한 FD 없음.
- BCNF ✓

**BCNF 분해: R₁(A, C, D), R₂(B, C)**

종속성 보존: AB → C는 R₁과 R₂ 조인 필요. **보존 안됨.**
</details>

### 연습 문제 4: 무손실 조인(Lossless-Join) 검증

다음 분해가 무손실 조인(Lossless-Join) 속성을 갖는지 검증하시오:

R(A, B, C, D)에 F = { A → B, B → C }

분해: R₁(A, B), R₂(A, C), R₃(B, D)

<details>
<summary>해답</summary>

추적(chase) 테스트 사용:

초기 행렬:
|    | A  | B   | C   | D   |
|----|----|-----|-----|-----|
| R₁ | a₁ | a₂  | b₁₃ | b₁₄ |
| R₂ | a₁ | b₂₂ | a₃  | b₂₄ |
| R₃ | b₃₁| a₂  | b₃₃ | a₄  |

A → B 적용: R₁과 R₂가 A에서 일치(= a₁).
- R₁.B = a₂, R₂.B = b₂₂. R₁이 구별 기호를 가짐. R₂.B = a₂로 설정.

|    | A  | B  | C   | D   |
|----|----|----|-----|-----|
| R₁ | a₁ | a₂ | b₁₃ | b₁₄ |
| R₂ | a₁ | a₂ | a₃  | b₂₄ |
| R₃ | b₃₁| a₂ | b₃₃ | a₄  |

B → C 적용: R₁, R₂, R₃가 B에서 일치(= a₂).
- C 값: b₁₃, a₃, b₃₃. a₃ 존재. 모두 a₃로 설정.

|    | A  | B  | C  | D   |
|----|----|----|----| ----|
| R₁ | a₁ | a₂ | a₃ | b₁₄ |
| R₂ | a₁ | a₂ | a₃ | b₂₄ |
| R₃ | b₃₁| a₂ | a₃ | a₄  |

더 이상의 반복에서 변경 없음.

행 확인: 모든 구별 기호를 가진 행 없음. R₁은 b₁₄, R₂는 b₂₄, R₃는 b₃₁을 가짐.

**이 분해는 무손실 조인(Lossless-Join)이 아님.** ✗

문제점: R₃(B, D)는 다른 릴레이션과 B만 공유하며, B는 D의 결정 속성을 포함하는 어떤 릴레이션의 키도 아님.

올바른 분해: R₁(A, B), R₂(B, C, D) — B → C가 성립하고 {B}가 {B,C}에 제한된 R₂의 키이므로 무손실.

실제로 B → D는 주어지지 않음. FD는 A → B와 B → C뿐. 따라서 D에는 결정하는 FD가 없음. 재고해보면: {A}⁺ = {A,B,C}. 키는 D를 포함해야 함: 키 = {A, D}.

더 나은 분해: R₁(A, B, C)와 R₂(A, D). 공통 = {A}. A → {B,C}. {A}는 R₁의 키. 무손실 ✓.
</details>

### 연습 문제 5: 완전 정규화(Full Normalization)

합성 알고리즘을 사용하여 다음을 3NF로 정규화하시오:

```
Library(isbn, title, author_id, author_name, publisher_id,
        publisher_name, publisher_city, branch_id, branch_name, copies)
```

FD:
```
isbn → title, author_id, publisher_id
author_id → author_name
publisher_id → publisher_name, publisher_city
{isbn, branch_id} → copies
branch_id → branch_name
```

<details>
<summary>해답</summary>

**단계 1: 최소 커버(Minimal Cover)**

우변 분해:
```
isbn → title, isbn → author_id, isbn → publisher_id,
author_id → author_name,
publisher_id → publisher_name, publisher_id → publisher_city,
{isbn, branch_id} → copies,
branch_id → branch_name
```

이미 최소 (우변에 단일 속성, 불필요한 좌변 없음, 중복 FD 없음).

**단계 2: 좌변으로 그룹화**

- R₁(isbn, title, author_id, publisher_id) — isbn → title, author_id, publisher_id에서
- R₂(author_id, author_name) — author_id → author_name에서
- R₃(publisher_id, publisher_name, publisher_city) — publisher_id → publisher_name, publisher_city에서
- R₄(isbn, branch_id, copies) — {isbn, branch_id} → copies에서
- R₅(branch_id, branch_name) — branch_id → branch_name에서

**단계 3: 후보 키 = {isbn, branch_id}**

R₄가 {isbn, branch_id}를 포함함. ✓

**단계 4: 제거할 부분집합 없음.**

**최종 3NF 분해:**
```
Book(isbn, title, author_id, publisher_id)         — 키: {isbn}
Author(author_id, author_name)                      — 키: {author_id}
Publisher(publisher_id, publisher_name, publisher_city) — 키: {publisher_id}
BranchStock(isbn, branch_id, copies)                — 키: {isbn, branch_id}
Branch(branch_id, branch_name)                      — 키: {branch_id}
```

모든 결정자가 단일 속성 키(또는 BranchStock의 복합 키)이므로 BCNF이기도 함.
</details>

### 연습 문제 6: 이상(Anomaly) 식별

다음 릴레이션과 샘플 데이터를 보고, 구체적인 갱신, 삽입, 삭제 이상을 식별하시오:

```
CourseSection(course_id, section, semester, instructor, building, room)

FDs: {course_id, section, semester} → instructor, building, room
     building, room → capacity   (capacity도 속성으로 가정)
```

```
| course_id | section | semester | instructor | building | room | capacity |
|-----------|---------|----------|------------|----------|------|----------|
| CS101     | 1       | Fall24   | Dr. Smith  | Watson   | 101  | 50       |
| CS101     | 2       | Fall24   | Dr. Jones  | Watson   | 101  | 50       |
| CS201     | 1       | Fall24   | Dr. Smith  | Watson   | 201  | 30       |
| CS201     | 1       | Spr25    | Dr. Smith  | Taylor   | 105  | 40       |
```

<details>
<summary>해답</summary>

**갱신 이상(Update Anomaly)**: Watson 101의 수용 인원이 변경되면(예: 리모델링으로 좌석 추가), 여러 행(행 1과 2)을 갱신해야 함. 행 1만 갱신하면 행 1과 2가 불일치하게 됨.

**삽입 이상(Insertion Anomaly)**: 해당 강의실에 예정된 수업이 없으면 Taylor 302의 수용 인원이 60임을 기록할 수 없음. 강의실 정보를 독립적으로 저장할 방법이 없음.

**삭제 이상(Deletion Anomaly)**: CS201 Section 1 Spring 2025가 취소되면(행 4 삭제), Taylor 105의 수용 인원이 40임을 알 수 없게 됨(해당 강의실을 사용하는 다른 수업이 없다고 가정).

**근본 원인**: 전이 종속성(Transitive Dependency) {course_id, section, semester} → {building, room} → capacity가 중복을 만듦.

**해결**: 다음과 같이 분해:
```
CourseSection(course_id, section, semester, instructor, building, room)
Room(building, room, capacity)
```
</details>

---

## 14. 요약(Summary)

| 개념 | 핵심 아이디어 |
|---------|----------|
| **1NF** | 원자적 값만 — 관계형 모델의 기초 |
| **2NF** | 부분 종속성 없음 — 모든 비주요 속성이 전체 키에 의존 |
| **3NF** | 전이 종속성 없음 — 비주요 속성이 키에만 의존 |
| **BCNF** | 모든 결정자가 슈퍼키 — 가장 엄격한 FD 기반 형태 |
| **무손실 조인** | 자연 조인이 원래 데이터 복구 — 필수 |
| **종속성 보존** | 모든 FD를 조인 없이 확인 가능 — 바람직하지만 때때로 BCNF를 위해 희생 |
| **3NF 합성** | 무손실 조인과 종속성 보존 모두 보장 |
| **BCNF 분해** | 무손실 조인 보장; 종속성 보존 잃을 수 있음 |

BCNF까지의 정규화는 함수 종속성으로 인한 모든 중복을 처리합니다. 그러나 다른 유형의 종속성 — 다치 종속성과 조인 종속성 — 이 있어 더 높은 정규형이 필요합니다. 이는 다음 레슨에서 탐구합니다.

---

**이전**: [05_Functional_Dependencies.md](./05_Functional_Dependencies.md) | **다음**: [07_Advanced_Normalization.md](./07_Advanced_Normalization.md)
