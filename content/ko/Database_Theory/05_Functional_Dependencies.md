# Lesson 05: 함수 종속성(Functional Dependencies)

**이전**: [04_ER_Modeling.md](./04_ER_Modeling.md) | **다음**: [06_Normalization.md](./06_Normalization.md)

---

> **주제**: 데이터베이스 이론(Database Theory)
> **레슨**: 16개 중 5번째
> **선수 지식**: 관계형 모델 개념(릴레이션, 튜플, 속성, 키), 기본 집합론
> **목표**: 함수 종속성을 데이터베이스 정규화의 형식적 기반으로 이해하고, Armstrong의 공리 마스터하기, 속성 폐쇄 계산, 후보 키 도출, 최소 커버 계산

## 1. 소개(Introduction)

함수 종속성(FD: Functional Dependencies)은 관계형 데이터베이스 설계 이론에서 가장 중요한 개념입니다. 이는 "하나의 속성이 다른 속성을 고유하게 결정한다"는 개념을 형식화하며, 중복을 줄이고 이상 현상을 방지하기 위해 데이터베이스를 조직하는 과정인 정규화의 수학적 기초를 제공합니다.

함수 종속성이 등장하기 전에는 데이터베이스 설계자들이 직관과 경험에 의존했습니다. FD는 "좋은" 스키마가 무엇인지 엄밀하게 추론하고, 체계적으로 달성할 수 있는 알고리즘을 제공하는 엄격한 프레임워크를 제공했습니다.

### 1.1 함수 종속성이 중요한 이유

대학 데이터베이스의 단일 릴레이션을 고려해보세요:

```
StudentCourse(student_id, student_name, dept_name, course_id, course_title, grade)
```

직관적으로 우리는 다음을 알 수 있습니다:
- student_id는 student_name과 dept_name을 고유하게 식별합니다
- course_id는 course_title을 고유하게 식별합니다
- (student_id, course_id)의 조합은 grade를 고유하게 식별합니다

이러한 직관적 관찰이 바로 함수 종속성입니다. 이를 통해 다음을 수행할 수 있습니다:

1. **중복성 식별**: 동일한 student_id에 대해 student_name이 모든 행에 나타나면 중복이 있습니다
2. **이상 현상 탐지**: 한 행에서 학생의 학과를 업데이트하고 다른 행은 업데이트하지 않으면 불일치가 발생합니다
3. **분해 가이드**: FD는 릴레이션을 더 작고 잘 구조화된 조각으로 분할하는 정확한 방법을 알려줍니다
4. **키 제약 조건 검증**: FD는 속성 집합이 키인지 형식적으로 정의합니다

---

## 2. 형식적 정의(Formal Definition)

### 2.1 함수 종속성

R을 속성 집합 U를 가진 릴레이션 스키마, X, Y ⊆ U라고 하겠습니다.

> **정의**: **함수 종속성(Functional Dependency)** X → Y는 R의 모든 합법적 인스턴스 r에 대해, 두 튜플 t₁과 t₂가 X의 속성들에서 일치하면 Y의 속성들에서도 반드시 일치할 때 R에 대해 성립합니다. 형식적으로:
>
> X → Y ⟺ ∀ t₁, t₂ ∈ r : t₁[X] = t₂[X] ⟹ t₁[Y] = t₂[Y]

여기서:
- **X**는 **결정자(determinant)** (또는 좌변, LHS: left-hand side)라고 합니다
- **Y**는 **종속자(dependent)** (또는 우변, RHS: right-hand side)라고 합니다
- X → Y를 "X가 Y를 함수적으로 결정한다" 또는 "Y는 X에 함수적으로 종속된다"고 읽습니다

### 2.2 자명한 FD와 비자명한 FD

> **정의**: 함수 종속성 X → Y가 Y ⊆ X이면 **자명(trivial)**합니다.

예시:
- {student_id, course_id} → {student_id}는 자명합니다
- {student_id} → {student_id}는 자명합니다
- {student_id} → {student_name}는 **비자명**합니다 (student_name ∉ {student_id})

> **정의**: 함수 종속성 X → Y가 X ∩ Y = ∅이면 **완전 비자명(completely non-trivial)**합니다.

### 2.3 예시

다음 릴레이션 스키마를 고려하세요:

```
Employee(emp_id, emp_name, dept_id, dept_name, salary, manager_id)
```

일반적인 함수 종속성:
- emp_id → emp_name, dept_id, salary, manager_id (직원 ID는 모든 직원 속성을 결정)
- dept_id → dept_name (부서 ID는 부서 이름을 결정)
- emp_id → dept_name (dept_id를 통해 전이적으로)

**중요**: FD는 **의미론적(semantic)** 제약 조건입니다. 이는 현실 세계 데이터의 의미에 의해 결정되며, 특정 인스턴스를 검사하여 결정되지 않습니다. 데이터의 스냅샷을 보는 것은 FD를 **반증**할 수 있지만(반례를 찾아서), 일반적으로 성립함을 **증명**할 수는 없습니다.

### 2.4 FD vs. 키

함수 종속성과 키 사이에는 깊은 연관이 있습니다:

> **정의**: 속성 집합 K가 K → U일 때 릴레이션 R의 **슈퍼키(superkey)**입니다. 여기서 U는 R의 모든 속성의 집합입니다.

> **정의**: 속성 집합 K가 다음을 만족할 때 R의 **후보 키(candidate key)**입니다:
> 1. K → U (K는 슈퍼키), 그리고
> 2. K의 어떤 진부분집합도 U를 함수적으로 결정하지 않음 (K는 최소)

따라서 키는 단순히 우변이 전체 속성 집합이고 최소성 조건이 있는 함수 종속성입니다.

---

## 3. Armstrong의 공리(Armstrong's Axioms)

1974년 William W. Armstrong은 함수 종속성을 위한 추론 규칙 집합을 제안했습니다. 이러한 공리는 **건전(sound)**(올바른 FD만 도출)하고 **완전(complete)**(주어진 집합으로부터 논리적으로 따르는 모든 FD를 도출할 수 있음)합니다.

### 3.1 세 가지 공리

F를 릴레이션 스키마 R에 대한 함수 종속성 집합, X, Y, Z ⊆ attributes(R)라고 하겠습니다.

#### 공리 1: 재귀성(Reflexivity)

> Y ⊆ X이면 X → Y.

이 공리는 모든 자명한 FD를 생성합니다. 예를 들어:
- {A, B, C} → {A, B}
- {A, B, C} → {A}
- {A} → {A}

**건전성 증명**: Y ⊆ X이고 t₁[X] = t₂[X]이면, Y ⊆ X이므로 t₁[Y] = t₂[Y]입니다. ∎

#### 공리 2: 증가(Augmentation)

> X → Y이면 임의의 Z에 대해 XZ → YZ.

FD의 양쪽에 임의의 속성 집합을 "증가"시킬 수 있습니다. 예를 들어:
- A → B이면 AC → BC
- AB → C이면 ABD → CD

**건전성 증명**: X → Y이고 t₁[XZ] = t₂[XZ]라고 가정합니다. 그러면 t₁[X] = t₂[X]이고 t₁[Z] = t₂[Z]입니다. X → Y이므로 t₁[Y] = t₂[Y]입니다. 결합하면: t₁[YZ] = t₂[YZ]. ∎

#### 공리 3: 전이성(Transitivity)

> X → Y이고 Y → Z이면 X → Z.

이는 함수 종속성의 연쇄를 허용합니다. 예를 들어:
- student_id → dept_id이고 dept_id → dept_name이면 student_id → dept_name

**건전성 증명**: X → Y이고 Y → Z라고 가정합니다. t₁[X] = t₂[X]라고 하겠습니다. X → Y에 의해 t₁[Y] = t₂[Y]입니다. Y → Z에 의해 t₁[Z] = t₂[Z]입니다. 따라서 X → Z입니다. ∎

### 3.2 건전성과 완전성

> **정리 (Armstrong, 1974)**: Armstrong의 공리는 **건전하고 완전**합니다.
> - **건전성**: 이러한 공리를 사용하여 F로부터 도출 가능한 모든 FD는 F로부터 논리적으로 함의됩니다.
> - **완전성**: F로부터 논리적으로 함의되는 모든 FD는 이러한 공리를 사용하여 도출될 수 있습니다.

완전성 증명은 쉽지 않습니다. 핵심 아이디어는 Armstrong의 공리를 사용하여 F로부터 X → Y를 도출할 수 없다면, F를 만족하지만 X → Y를 위반하는 두 튜플 릴레이션이 존재함을 보이는 것입니다.

---

## 4. 도출된 추론 규칙(Derived Inference Rules)

Armstrong의 세 공리로부터 몇 가지 유용한 추가 규칙을 도출할 수 있습니다.

### 4.1 합집합 규칙(Union Rule)

> X → Y이고 X → Z이면 X → YZ.

**증명**:
1. X → Y (주어진)
2. X → Z (주어진)
3. X → XY (X로 단계 1을 증가; XX = X이므로 X → XY를 얻음)
4. XY → YZ (Y로 단계 2를 증가)
5. X → YZ (단계 3과 4에 전이성 적용) ∎

### 4.2 분해 규칙(Decomposition Rule)

> X → YZ이면 X → Y이고 X → Z.

**증명**:
1. X → YZ (주어진)
2. YZ → Y (재귀성, Y ⊆ YZ이므로)
3. X → Y (단계 1과 2에 전이성 적용)
4. YZ → Z (재귀성, Z ⊆ YZ이므로)
5. X → Z (단계 1과 4에 전이성 적용) ∎

### 4.3 의사전이성(Pseudotransitivity)

> X → Y이고 WY → Z이면 WX → Z.

**증명**:
1. X → Y (주어진)
2. WX → WY (W로 단계 1을 증가)
3. WY → Z (주어진)
4. WX → Z (단계 2와 3에 전이성 적용) ∎

### 4.4 자기결정(Self-Determination)

> 임의의 X에 대해 X → X.

이는 재귀성으로부터 직접 따라옵니다 (X ⊆ X).

### 4.5 축적(Accumulation)

> X → YZ이고 Z → BW이면 X → YBW.

**증명**:
1. X → YZ (주어진)
2. X → Z (단계 1의 분해)
3. Z → BW (주어진)
4. X → BW (단계 2와 3에 전이성 적용)
5. X → Y (단계 1의 분해)
6. X → YBW (단계 4와 5의 합집합) ∎

### 4.6 규칙 요약

| 규칙 | 명제 | 도출 근거 |
|------|-----------|-------------|
| 재귀성 | Y ⊆ X ⟹ X → Y | 공리 |
| 증가 | X → Y ⟹ XZ → YZ | 공리 |
| 전이성 | X → Y, Y → Z ⟹ X → Z | 공리 |
| 합집합 | X → Y, X → Z ⟹ X → YZ | 증가 + 전이성 |
| 분해 | X → YZ ⟹ X → Y, X → Z | 재귀성 + 전이성 |
| 의사전이성 | X → Y, WY → Z ⟹ WX → Z | 증가 + 전이성 |

---

## 5. 속성 집합의 폐쇄(Closure of an Attribute Set)

속성 집합의 폐쇄를 계산하는 것은 FD 이론의 가장 기본적인 알고리즘입니다. "주어진 FD 집합 F에서 주어진 집합 X에 의해 함수적으로 결정되는 속성은 무엇인가?"라는 질문에 답합니다.

### 5.1 정의

> **정의**: 함수 종속성 집합 F 하에서 **속성 집합 X의 폐쇄**는 X⁺ (또는 X⁺_F)로 표기하며, Armstrong의 공리를 사용하여 F로부터 X → A를 추론할 수 있는 모든 속성 A의 집합입니다.
>
> X⁺ = { A ∈ U | F ⊨ X → A }

### 5.2 알고리즘

다음 알고리즘은 X⁺을 효율적으로 계산합니다:

```
ALGORITHM: ComputeClosure(X, F)
INPUT:  X = 속성 집합
        F = 함수 종속성 집합
OUTPUT: X⁺ = F 하에서 X의 폐쇄

1.  result ← X
2.  REPEAT
3.      old_result ← result
4.      FOR EACH dependency (V → W) IN F DO
5.          IF V ⊆ result THEN
6.              result ← result ∪ W
7.          END IF
8.      END FOR
9.  UNTIL result = old_result
10. RETURN result
```

**시간 복잡도**: 최악의 경우 O(|F| × |U|), 여기서 |F|는 FD의 수이고 |U|는 속성의 수입니다.

### 5.3 작업 예제 1

R(A, B, C, D, E, F)에 다음 FD가 있다고 하겠습니다:

```
F = { A → BC,  CD → E,  B → D,  E → A }
```

**{A}⁺ 계산:**

| 반복 | result | 적용된 FD | 새 속성 |
|-----------|--------|-----------|----------------|
| 초기화 | {A} | — | — |
| 1 | {A, B, C} | A → BC | B, C |
| 2 | {A, B, C, D} | B → D | D |
| 3 | {A, B, C, D, E} | CD → E | E |
| 4 | {A, B, C, D, E} | E → A (A는 이미 result에 있음) | — |

반복 4에서 새 속성이 추가되지 않았으므로 중단합니다.

**{A}⁺ = {A, B, C, D, E}**

참고: F는 폐쇄에 없으므로 A는 R의 슈퍼키가 아닙니다(F를 결정하지 않음).

**{A, F}⁺ 계산:**

{A, F}로 시작하여 동일한 반복 후 F가 이미 존재함:

**{A, F}⁺ = {A, B, C, D, E, F}** = R의 모든 속성.

따라서 {A, F}는 슈퍼키입니다.

### 5.4 작업 예제 2

R(A, B, C, D, E)에 다음 FD가 있다고 하겠습니다:

```
F = { AB → C,  C → D,  D → E,  E → A }
```

**{B, C}⁺ 계산:**

| 반복 | result | 적용된 FD | 새 속성 |
|-----------|--------|-----------|----------------|
| 초기화 | {B, C} | — | — |
| 1 | {B, C, D} | C → D | D |
| 2 | {B, C, D, E} | D → E | E |
| 3 | {A, B, C, D, E} | E → A | A |
| 4 | {A, B, C, D, E} | AB → C (C는 이미 존재) | — |

**{B, C}⁺ = {A, B, C, D, E}** — 따라서 {B, C}는 슈퍼키입니다.

### 5.5 속성 폐쇄의 사용

폐쇄 알고리즘은 세 가지 주요 응용이 있습니다:

1. **X → Y가 성립하는지 테스트**: X⁺을 계산합니다. Y ⊆ X⁺이면 X → Y가 성립합니다.

2. **X가 슈퍼키인지 테스트**: X⁺을 계산합니다. X⁺ = U (모든 속성)이면 X는 슈퍼키입니다.

3. **F⁺ (F의 폐쇄) 계산**: 각 부분집합 X ⊆ U에 대해 X⁺을 계산하고 각 Y ⊆ X⁺에 대해 X → Y를 출력합니다. (이것은 지수적이고 실제로는 거의 실용적이지 않지만, 이론적으로 중요합니다.)

---

## 6. FD 집합의 폐쇄(Closure of a Set of FDs)

### 6.1 정의

> **정의**: 함수 종속성 집합 F의 **폐쇄**는 F⁺로 표기하며, F로부터 논리적으로 추론할 수 있는 모든 함수 종속성의 집합입니다.
>
> F⁺ = { X → Y | F ⊨ X → Y }

F⁺은 매우 클 수 있습니다. n개의 속성을 가진 릴레이션에 대해, X에 대해 2ⁿ개의 가능한 부분집합이 있고 Y에 대해 2ⁿ개가 있어 최대 2²ⁿ개의 가능한 FD가 있습니다. 실제로 우리는 F⁺을 직접 계산하지 않고 대신 속성 폐쇄 알고리즘을 사용합니다.

### 6.2 FD 집합의 동치

> **정의**: 두 FD 집합 F와 G가 F⁺ = G⁺일 때 **동치(equivalent)**(F ≡ G로 표기)입니다.

F ≡ G를 테스트하려면:
1. G의 모든 FD가 F로부터 도출될 수 있는지 확인: G의 각 X → Y에 대해 Y ⊆ X⁺_F인지 확인
2. F의 모든 FD가 G로부터 도출될 수 있는지 확인: F의 각 X → Y에 대해 Y ⊆ X⁺_G인지 확인

두 검사를 모두 통과하면 F ≡ G입니다.

### 6.3 예시

```
F = { A → B, B → C }
G = { A → B, B → C, A → C }
```

F ≡ G입니다. 왜냐하면 A → C는 전이성에 의해 F로부터 도출 가능하고, F의 모든 FD는 당연히 G에 있기 때문입니다.

---

## 7. 최소 커버(Minimal Cover / Canonical Cover)

최소 커버는 FD 집합의 "단순화된" 버전입니다 — 동일한 논리적 내용을 유지하면서 중복을 제거합니다. 이는 정규화 알고리즘에 필수적입니다.

### 7.1 정의

> **정의**: FD 집합 F_min이 다음을 만족할 때 F의 **최소 커버(minimal cover)** (또는 **정규 커버(canonical cover)**)입니다:
> 1. **동치**: F_min ≡ F (동일한 폐쇄)
> 2. **RHS에 단일 속성**: F_min의 모든 FD는 X → A 형태이며 여기서 A는 단일 속성
> 3. **중복 FD 없음**: F_min에서 어떤 FD를 제거해도 폐쇄가 변경됨
> 4. **LHS에 불필요한 속성 없음**: F_min의 각 FD X → A에 대해, X의 어떤 진부분집합도 F_min 하에서 A를 함수적으로 결정하지 않음

### 7.2 알고리즘

```
ALGORITHM: MinimalCover(F)
INPUT:  F = 함수 종속성 집합
OUTPUT: F_min = F의 최소 커버

단계 1: 우변 분해
    각 FD X → {A₁, A₂, ..., Aₙ}를
    X → A₁, X → A₂, ..., X → Aₙ으로 대체

단계 2: 좌변에서 불필요한 속성 제거
    FOR EACH FD (X → A) IN F where |X| > 1 DO
        FOR EACH attribute B IN X DO
            IF A ∈ closure(X - {B}, F) THEN
                (X → A)를 ((X - {B}) → A)로 대체
            END IF
        END FOR
    END FOR

단계 3: 중복 FD 제거
    FOR EACH FD (X → A) IN F DO
        IF A ∈ closure(X, F - {X → A}) THEN
            F에서 (X → A) 제거
        END IF
    END FOR

RETURN F
```

**중요**: 단계 2는 단계 3보다 먼저 와야 합니다. 중복 FD를 먼저 제거하면 불필요했던 일부 속성이 더 이상 불필요해 보이지 않을 수 있습니다.

### 7.3 작업 예제

R(A, B, C, D)에 다음 FD가 있다고 하겠습니다:

```
F = { A → BC,  B → C,  AB → D,  D → A }
```

**단계 1: RHS 분해**

```
F = { A → B,  A → C,  B → C,  AB → D,  D → A }
```

**단계 2: 불필요한 LHS 속성 제거**

AB → D 확인: A 또는 B를 제거할 수 있는가?
- B만으로 D를 결정하는지 확인: 현재 F 하에서 {B}⁺ 계산:
  - {B} → {B, C} (B → C를 통해) — 더 이상 없음. D ∉ {B}⁺. 따라서 B만으로는 작동하지 않음; A 유지.
- A만으로 D를 결정하는지 확인: 현재 F 하에서 {A}⁺ 계산:
  - {A} → {A, B, C} (A → B, A → C를 통해) → {A, B, C, D} (AB → D를 통해, A와 B가 모두 존재하므로)
  - D ∈ {A}⁺. 따라서 AB → D에서 B는 불필요합니다!

AB → D를 A → D로 대체:

```
F = { A → B,  A → C,  B → C,  A → D,  D → A }
```

**단계 3: 중복 FD 제거**

A → B 확인: 제거. F - {A → B} 하에서 {A}⁺ 계산:
- F - {A → B} = { A → C,  B → C,  A → D,  D → A }
- {A}⁺ = {A, C, D} (A → C, A → D, D → A를 통해). B ∉ {A}⁺. 따라서 A → B는 중복이 아닙니다. 유지.

A → C 확인: 제거. F - {A → C} 하에서 {A}⁺ 계산:
- {A}⁺: A → B는 B를 제공하고, B → C는 C를 제공. C ∈ {A}⁺. 따라서 A → C는 중복입니다. 제거.

```
F = { A → B,  B → C,  A → D,  D → A }
```

B → C 확인: 제거. F - {B → C} 하에서 {B}⁺ 계산:
- {B}⁺ = {B}. C ∉ {B}⁺. 중복이 아닙니다. 유지.

A → D 확인: 제거. F - {A → D} 하에서 {A}⁺ 계산:
- {A}⁺ = {A, B, C}. D ∉ {A}⁺. 중복이 아닙니다. 유지.

D → A 확인: 제거. F - {D → A} 하에서 {D}⁺ 계산:
- {D}⁺ = {D}. A ∉ {D}⁺. 중복이 아닙니다. 유지.

**최소 커버:**

```
F_min = { A → B,  B → C,  A → D,  D → A }
```

### 7.4 최소 커버의 비유일성

최소 커버는 **유일하지 않습니다**. 단계 2 또는 단계 3의 다른 순서는 다른 (하지만 동치인) 최소 커버를 생성할 수 있습니다. 예를 들어, 단계 2에서 속성을 다른 순서로 확인하면 다른 단순화로 이어질 수 있습니다.

---

## 8. FD를 사용하여 후보 키 찾기

### 8.1 속성 분류

후보 키를 효율적으로 찾기 위해 먼저 속성을 분류합니다:

| 범주 | 정의 | 키에서의 역할 |
|----------|-----------|-------------|
| **L-only** | FD의 LHS에만 나타남(RHS에는 절대 나타나지 않음) | 모든 키에 있어야 함 |
| **R-only** | RHS에만 나타남(LHS에는 절대 나타나지 않음) | 어떤 키에도 없음 |
| **Both** | LHS와 RHS 모두에 나타남 | 키에 있을 수도 있고 없을 수도 있음 |
| **Neither** | 어떤 FD에도 나타나지 않음 | 모든 키에 있어야 함 |

### 8.2 후보 키 찾기 알고리즘

```
ALGORITHM: FindCandidateKeys(R, F)
INPUT:  R = 릴레이션 스키마, F = FD 집합
OUTPUT: 모든 후보 키의 집합

단계 1: 속성을 L-only, R-only, Both, Neither로 분류.
단계 2: CORE = L-only ∪ Neither라고 하겠습니다.
        (CORE는 모든 후보 키의 일부여야 함.)
단계 3: CORE⁺을 계산.
        CORE⁺ = 모든 속성이면 CORE가 유일한 후보 키. 완료.
단계 4: 그렇지 않으면 "Both" 속성의 부분집합을 CORE에 추가 시도.
        단일 속성부터 시작하여 쌍 등으로 진행.
        "Both"의 각 부분집합 S에 대해:
            (CORE ∪ S)⁺ 계산
            모든 속성과 같고 CORE를 포함하는 (CORE ∪ S)의
            어떤 진부분집합도 슈퍼키가 아니면 CORE ∪ S는 후보 키.
```

### 8.3 작업 예제

R(A, B, C, D, E, F)에 다음 FD가 있다고 하겠습니다:

```
F = { AB → C,  C → D,  D → E,  CF → B }
```

**단계 1: 속성 분류**

| 속성 | LHS? | RHS? | 범주 |
|-----------|------|------|----------|
| A | Yes (AB→C) | No | L-only |
| B | Yes (AB→C) | Yes (CF→B) | Both |
| C | Yes (C→D, CF→B) | Yes (AB→C) | Both |
| D | Yes (D→E) | Yes (C→D) | Both |
| E | No | Yes (D→E) | R-only |
| F | Yes (CF→B) | No | L-only |

**단계 2: CORE = {A, F}** (L-only 속성; "Neither" 속성 없음)

**단계 3: {A, F}⁺ 계산**

- 시작: {A, F}
- LHS ⊆ {A, F}인 FD 없음 (AB는 B 필요, C→D는 C 필요 등)
- {A, F}⁺ = {A, F} ≠ 모든 속성

**단계 4: "Both" 속성 추가 시도 (B, C, D)**

B 추가 시도: {A, B, F}⁺
- AB → C: {A, B, F, C}
- C → D: {A, B, C, D, F}
- D → E: {A, B, C, D, E, F} = 모든 속성!
- {A, B, F}는 슈퍼키. 최소성 확인: CORE = {A, F} 단독은 작동하지 않음. 따라서 {A, B, F}는 후보 키.

C 추가 시도: {A, C, F}⁺
- CF → B: {A, B, C, F}
- AB → C: 이미 C 있음
- C → D: {A, B, C, D, F}
- D → E: {A, B, C, D, E, F} = 모든 속성!
- {A, C, F}는 슈퍼키. 최소성 확인: {A, F} 단독은 작동하지 않음. 따라서 {A, C, F}는 후보 키.

D 추가 시도: {A, D, F}⁺
- D → E: {A, D, E, F}
- 더 이상 적용 가능한 FD 없음. {A, D, E, F} ≠ 모든 속성. 슈퍼키가 아님.

**후보 키: {A, B, F}와 {A, C, F}**

우리는 이미 단일 추가로 후보 키를 찾았고 이들이 최소이므로 쌍({B, C}와 같은)을 확인할 필요가 없습니다.

---

## 9. 함의와 포함(Entailment and Implication)

### 9.1 논리적 함의

> **정의**: FD 집합 F가 FD X → Y를 **논리적으로 함의(logically implies)**(F ⊨ X → Y로 표기)한다는 것은 F의 모든 FD를 만족하는 모든 릴레이션 인스턴스가 X → Y도 만족함을 의미합니다.

### 9.2 함의 테스트

F ⊨ X → Y를 테스트하려면:
1. F 하에서 X⁺ 계산
2. Y ⊆ X⁺이면 F ⊨ X → Y

이것은 실용적인 작업 도구입니다: 공리 적용의 연쇄를 통해 추론하는 대신 폐쇄 알고리즘만 실행하면 됩니다.

### 9.3 예시

주어진 F = { A → B, B → C, CD → E }:

F ⊨ AD → E인가?

{A, D}⁺ 계산:
- A → B: {A, B, D}
- B → C: {A, B, C, D}
- CD → E: {A, B, C, D, E}

E ∈ {A, D}⁺이므로, 예, F ⊨ AD → E. ✓

F ⊨ A → E인가?

{A}⁺ 계산:
- A → B: {A, B}
- B → C: {A, B, C}
- 더 이상 적용 가능한 FD 없음.

E ∉ {A}⁺ = {A, B, C}. 따라서 F ⊭ A → E. ✗

---

## 10. 실제의 FD(FDs in Practice)

### 10.1 요구사항에서 FD 식별

실제 FD는 비즈니스 규칙과 도메인 지식에서 나옵니다:

| 비즈니스 규칙 | 함수 종속성 |
|--------------|----------------------|
| "각 직원은 정확히 하나의 부서를 가짐" | emp_id → dept_id |
| "각 부서는 정확히 하나의 이름을 가짐" | dept_id → dept_name |
| "각 학생은 과목당 하나의 성적을 받음" | {student_id, course_id} → grade |
| "각 ISBN은 하나의 책 제목을 식별함" | isbn → title |
| "특정 날짜의 비행편은 한 명의 조종사를 가짐" | {flight_num, date} → pilot_id |

### 10.2 FD와 NULL 값

표준 FD 이론은 NULL 값이 없다고 가정합니다. 실제로:
- NULL은 FD 추론을 복잡하게 만듭니다 (SQL에서 NULL ≠ NULL)
- SQL의 UNIQUE 제약 조건은 여러 NULL을 허용합니다 (PRIMARY KEY 제외)
- 일부 데이터베이스 시스템은 유일성 검사를 위해 NULL을 동일하게 취급하는 "NULLS NOT DISTINCT"를 제공합니다

### 10.3 데이터에서 FD 발견

FD는 의미론적 제약 조건(도메인 지식에 의해 결정되며 데이터가 아님)이지만, 데이터에서 **근사 FD를 발견**하는 알고리즘이 있습니다:

- **TANE** 알고리즘: 데이터셋에서 성립하는 모든 FD 발견
- **FUN** 알고리즘: 격자 기반 검색 사용
- **FDTool**: FD 발견을 위한 실용 도구

이들은 문서화가 잘 되지 않은 데이터베이스를 역공학하는 데 유용하지만, 발견된 FD는 항상 도메인 지식에 대해 검증되어야 합니다.

### 10.4 SQL의 FD

SQL은 직접적인 `FUNCTIONAL DEPENDENCY` 제약 조건이 없지만 FD는 다음을 통해 강제됩니다:

```sql
-- 기본 키는 emp_id → 다른 모든 속성을 강제
CREATE TABLE Employee (
    emp_id    INT PRIMARY KEY,
    emp_name  VARCHAR(100),
    dept_id   INT,
    salary    DECIMAL(10,2)
);

-- UNIQUE 제약 조건은 email → (키인 경우 암묵적으로 모든 속성)을 강제
ALTER TABLE Employee ADD CONSTRAINT uq_email UNIQUE(email);

-- FD dept_id → dept_name은 dept_id를 기본 키로 가진
-- 별도의 Departments 테이블을 통해 강제됨
CREATE TABLE Department (
    dept_id   INT PRIMARY KEY,
    dept_name VARCHAR(100)
);
```

---

## 11. 일반적인 함정(Common Pitfalls)

### 11.1 FD를 데이터 패턴과 혼동

일반적인 실수는 데이터를 보고 FD가 존재한다고 결론 내리는 것입니다:

```
| city        | state |
|-------------|-------|
| Springfield | IL    |
| Portland    | OR    |
| Austin      | TX    |
```

이 데이터는 city → state와 일치하지만, 실제로 많은 도시가 여러 주에 걸쳐 이름을 공유합니다(Springfield는 30개 이상의 주에 존재). FD city → state는 **성립하지 않습니다**.

### 11.2 방향이 중요함

X → Y는 Y → X를 함의하지 **않습니다**.

- dept_id → dept_name (부서는 하나의 이름을 가짐) ✓
- dept_name → dept_id (이름이 하나의 부서를 식별) — 이름이 유일한지에 따라 다름!

### 11.3 단일 속성에 대한 FD

{} → A와 같은 FD(빈 집합이 A를 결정)는 A가 모든 튜플에서 동일한 값을 가진다는 것을 의미합니다. 이것은 드물지만 유효한 FD입니다(예: 모든 직원이 같은 회사에 있는 테이블: {} → company_name).

### 11.4 최소 커버에서 연산 순서

단계 2(불필요한 LHS 속성 제거)는 단계 3(중복 FD 제거)보다 앞서야 합니다. 순서를 뒤바꾸면 잘못된 결과를 생성할 수 있습니다.

---

## 12. 증명과 이론(Proofs and Theory)

### 12.1 Armstrong 공리의 완전성 증명 (스케치)

**주장**: Armstrong의 공리를 사용하여 F ⊭ X → Y이면, F를 만족하지만 X → Y를 위반하는 릴레이션 인스턴스가 존재합니다.

**증명 스케치**: 두 튜플 릴레이션 r = {t₁, t₂}를 구성합니다:
- 모든 A ∈ X⁺에 대해 t₁[A] = t₂[A] = 1
- 모든 A ∉ X⁺에 대해 t₁[A] = 1, t₂[A] = 0

검증이 필요합니다:
1. r은 F의 모든 FD를 만족: F의 어떤 V → W에 대해서도, t₁[V] = t₂[V]이면 V ⊆ X⁺이므로, W ⊆ X⁺(폐쇄 알고리즘에 의해), 따라서 t₁[W] = t₂[W]. ✓
2. r은 X → Y를 위반(Y ⊄ X⁺라고 가정): t₁[X] = t₂[X] (모두 1)이지만 A ∉ X⁺인 일부 A ∈ Y가 존재하므로, t₁[A] ≠ t₂[A]. ✓

이것은 X → Y가 F로부터 논리적으로 함의된다는 가정과 모순되어 완전성을 증명합니다. ∎

### 12.2 복잡도 결과

| 문제 | 복잡도 |
|---------|-----------|
| X⁺ 계산 | O(\|F\| × \|U\|) — 다항식 |
| F ⊨ X → Y 테스트 | O(\|F\| × \|U\|) — 다항식 |
| F⁺ 계산 | 지수적 (2^(2n)일 수 있음) |
| 모든 후보 키 찾기 | 일반적으로 NP-완전 |
| 최소 커버 계산 | 다항식 |
| X가 슈퍼키인지 테스트 | O(\|F\| × \|U\|) — 다항식 |

---

## 13. 연습 문제(Exercises)

### 연습 문제 1: 속성 폐쇄

R(A, B, C, D, E)에 F = { AB → C, C → D, BD → E, A → B }가 있다고 하겠습니다.

다음 폐쇄를 계산하세요:
1. {A}⁺
2. {B, C}⁺
3. {A, D}⁺
4. {C, D}⁺

<details>
<summary>해답</summary>

1. **{A}⁺**: A → B는 {A, B}를 제공; AB → C는 {A, B, C}를 제공; C → D는 {A, B, C, D}를 제공; BD → E는 {A, B, C, D, E}를 제공. **{A}⁺ = {A, B, C, D, E}**

2. **{B, C}⁺**: C → D는 {B, C, D}를 제공; BD → E는 {B, C, D, E}를 제공. 더 이상 없음. **{B, C}⁺ = {B, C, D, E}**

3. **{A, D}⁺**: A → B는 {A, B, D}를 제공; AB → C는 {A, B, C, D}를 제공; BD → E는 {A, B, C, D, E}를 제공. **{A, D}⁺ = {A, B, C, D, E}**

4. **{C, D}⁺**: C → D (이미 D 있음). LHS ⊆ {C, D}인 다른 FD 없음. **{C, D}⁺ = {C, D}**
</details>

### 연습 문제 2: 후보 키 찾기

연습 문제 1의 릴레이션과 FD에 대해 모든 후보 키를 찾으세요.

<details>
<summary>해답</summary>

속성 분류:
- A: LHS only (AB→C, A→B에서) → L-only
- B: Both (LHS: AB→C, BD→E; RHS: A→B)
- C: Both (LHS: C→D; RHS: AB→C)
- D: Both (LHS: BD→E; RHS: C→D)
- E: RHS only (BD→E에서) → R-only

CORE = {A} (L-only; "Neither" 속성 없음).
{A}⁺ = {A, B, C, D, E} = 모든 속성.

**{A}가 유일한 후보 키.**
</details>

### 연습 문제 3: 최소 커버

다음에 대한 최소 커버를 찾으세요:

```
F = { A → BC,  B → C,  AB → D,  D → BC }
```

<details>
<summary>해답</summary>

**단계 1: RHS 분해**
```
F = { A → B, A → C, B → C, AB → D, D → B, D → C }
```

**단계 2: 불필요한 LHS 속성 제거**

AB → D 확인:
- A 제거 시도: {B}⁺ = {B, C}. D ∉ {B}⁺. A 유지.
- B 제거 시도: {A}⁺ = {A, B, C, D} (A→B, A→C를 통해, 그 다음 B가 포함되므로 AB→D, 그 다음 D→B, D→C). D ∈ {A}⁺. B 제거!

AB → D를 A → D로 대체.

```
F = { A → B, A → C, B → C, A → D, D → B, D → C }
```

**단계 3: 중복 FD 제거**

- A → B: 제거. F - {A→B} 하에서 {A}⁺ = {A, C, D, B, C} (A→C, A→D, D→B, D→C). B ∈ {A}⁺. **중복! 제거.**
- A → C: 제거. F - {A→C} 하에서 {A}⁺ = {A, D, B, C} (A→D, D→B, D→C). C ∈ {A}⁺. **중복! 제거.**
- B → C: 제거. F - {B→C} 하에서 {B}⁺ = {B}. C ∉ {B}⁺. **중복 아님. 유지.**
- A → D: 제거. F - {A→D} 하에서 {A}⁺ = {A}. D ∉ {A}⁺. **중복 아님. 유지.**
- D → B: 제거. F - {D→B} 하에서 {D}⁺ = {D, C}. B ∉ {D}⁺. **중복 아님. 유지.**
- D → C: 제거. F - {D→C} 하에서 {D}⁺ = {D, B, C}. C ∈ {D}⁺ (D→B, B→C를 통해). **중복! 제거.**

**최소 커버: F_min = { B → C, A → D, D → B }**
</details>

### 연습 문제 4: FD 증명

주어진 F = { A → B, B → C, C → D }, Armstrong의 공리를 사용하여 A → D를 증명하세요. 각 단계를 명시적으로 작성하세요.

<details>
<summary>해답</summary>

1. A → B (주어진)
2. B → C (주어진)
3. A → C (단계 1과 2에 전이성)
4. C → D (주어진)
5. A → D (단계 3과 4에 전이성) ∎
</details>

### 연습 문제 5: FD 함의

주어진 F = { A → B, BC → D, E → C }:

다음 FD가 F로부터 함의되는지 결정하세요:
1. AE → D
2. BE → D
3. A → D

<details>
<summary>해답</summary>

1. **AE → D**: {A, E}⁺ 계산 = {A, E} → {A, B, E} (A→B) → {A, B, C, E} (E→C) → {A, B, C, D, E} (BC→D). D ∈ {A,E}⁺. **예, F ⊨ AE → D.** ✓

2. **BE → D**: {B, E}⁺ 계산 = {B, E} → {B, C, E} (E→C) → {B, C, D, E} (BC→D). D ∈ {B,E}⁺. **예, F ⊨ BE → D.** ✓

3. **A → D**: {A}⁺ 계산 = {A} → {A, B} (A→B). 더 이상 적용 가능한 것 없음. D ∉ {A}⁺. **아니오, F ⊭ A → D.** ✗
</details>

### 연습 문제 6: FD 집합의 동치

다음 두 FD 집합이 동치인가요?

```
F = { A → B, B → C, A → C }
G = { A → B, B → C }
```

<details>
<summary>해답</summary>

F의 모든 FD가 G로부터 함의되는지 확인:
- A → B: G에 직접 있음. ✓
- B → C: G에 직접 있음. ✓
- A → C: {A}⁺_G = {A, B, C}. C ∈ {A}⁺. ✓

G의 모든 FD가 F로부터 함의되는지 확인:
- A → B: F에 직접 있음. ✓
- B → C: F에 직접 있음. ✓

**F ≡ G.** F의 FD A → C는 중복입니다; A → B와 B → C로부터 전이성에 의해 따라옵니다.
</details>

### 연습 문제 7: 여러 후보 키

R(A, B, C, D, E)에 F = { AB → CDE, C → A, D → B }가 있다고 하겠습니다.

모든 후보 키를 찾으세요.

<details>
<summary>해답</summary>

속성 분류:
- A: Both (LHS: AB→CDE; RHS: C→A)
- B: Both (LHS: AB→CDE; RHS: D→B)
- C: Both (LHS: C→A; RHS: AB→CDE)
- D: Both (LHS: D→B; RHS: AB→CDE)
- E: RHS only → R-only

L-only나 Neither 속성이 없으므로 CORE = {}.

단일 속성 시도:
- {A}⁺ = {A}. 슈퍼키 아님.
- {B}⁺ = {B}. 슈퍼키 아님.
- {C}⁺ = {C, A} = {A, C}. 슈퍼키 아님.
- {D}⁺ = {D, B} = {B, D}. 슈퍼키 아님.

쌍 시도:
- {A, B}⁺ = {A, B, C, D, E}. **슈퍼키!** 최소({A}나 {B} 단독은 작동하지 않음). **후보 키: {A, B}**
- {A, D}⁺ = {A, B, D} → {A, B, C, D, E}. **슈퍼키!** 확인: {A}⁺={A}, {D}⁺={B,D}. 최소. **후보 키: {A, D}**
- {B, C}⁺ = {A, B, C} → {A, B, C, D, E}. **슈퍼키!** 확인: {B}⁺={B}, {C}⁺={A,C}. 최소. **후보 키: {B, C}**
- {C, D}⁺ = {A, B, C, D} → {A, B, C, D, E}. **슈퍼키!** 확인: {C}⁺={A,C}, {D}⁺={B,D}. 최소. **후보 키: {C, D}**

**모든 후보 키: {A,B}, {A,D}, {B,C}, {C,D}**
</details>

### 연습 문제 8: 폐쇄 증명

합집합 규칙(X → Y, X → Z ⟹ X → YZ)이 Armstrong의 공리로부터 따라옴을 증명하세요.

<details>
<summary>해답</summary>

1. X → Y (주어진)
2. X → XY (X로 단계 1을 증가: XX → XY, 그리고 XX = X)
3. X → Z (주어진)
4. XY → YZ (Y로 단계 3을 증가: XY → ZY)
5. X → YZ (단계 2와 4에 전이성) ∎
</details>

---

## 14. 요약(Summary)

| 개념 | 핵심 사항 |
|---------|-----------|
| **함수 종속성** | X → Y는 X가 Y를 고유하게 결정함을 의미 |
| **Armstrong의 공리** | 재귀성, 증가, 전이성 — 건전하고 완전 |
| **도출된 규칙** | 합집합, 분해, 의사전이성 |
| **속성 폐쇄 X⁺** | X에 의해 결정되는 모든 속성 — 핵심 알고리즘 |
| **FD 집합 폐쇄 F⁺** | F로부터 함의되는 모든 FD — 일반적으로 X⁺을 통해 계산 |
| **최소 커버** | 단순화된 동치 FD 집합 — 정규화에 필요 |
| **후보 키** | 속성 분류와 폐쇄 계산으로 발견 |

함수 종속성은 관계형 데이터베이스 설계의 이론적 기반입니다. 이 레슨의 알고리즘 — 속성 폐쇄와 최소 커버 — 는 정규화에 관한 다음 레슨에서 광범위하게 사용되며, 여기서 FD를 적용하여 릴레이션을 잘 구조화된 스키마로 분해합니다.

---

**이전**: [04_ER_Modeling.md](./04_ER_Modeling.md) | **다음**: [06_Normalization.md](./06_Normalization.md)
