# Lesson 07: 고급 정규화(Advanced Normalization)

**이전**: [06_Normalization.md](./06_Normalization.md) | **다음**: [08_Query_Processing.md](./08_Query_Processing.md)

---

> **주제(Topic)**: Database Theory
> **레슨(Lesson)**: 7 of 16
> **선행 학습(Prerequisites)**: 함수 종속성(Functional dependencies, Lesson 05), BCNF까지의 정규화(normalization through BCNF, Lesson 06)
> **목표(Objective)**: 다치 종속성과 4NF(Multivalued dependencies and 4NF), 조인 종속성과 5NF(Join dependencies and 5NF), DKNF의 이론적 이상(theoretical ideal of DKNF), 그리고 실무 시스템을 위한 실용적 비정규화 전략(practical denormalization strategies)을 이해합니다

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 다치 종속성(Multivalued Dependency, MVD)을 정의하고 BCNF를 만족하는 릴레이션이 MVD로 인해 여전히 중복을 포함할 수 있는 이유를 설명합니다.
2. 비자명 다치 종속성(Non-trivial MVD)을 탐지하여 릴레이션이 제4정규형(4NF)을 만족하는지 판단합니다.
3. 조인 종속성(Join Dependency)을 정의하고 제5정규형(5NF)을 달성하기 위해 릴레이션을 분해해야 하는 조건을 설명합니다.
4. 도메인-키 정규형(Domain-Key Normal Form, DKNF)의 이론적 이상을 서술하고 실용적 한계를 설명합니다.
5. 비정규화(Denormalization)가 성능을 향상시키는 시나리오를 식별하고, 파생 컬럼, 요약 테이블, 사전 조인 테이블 등 일반적인 비정규화 패턴을 적용합니다.
6. 데이터베이스 스키마 설계 시 더 높은 정규형과 실무적 성능 요구사항 간의 트레이드오프를 평가합니다.

---

## 1. 소개(Introduction)

Lesson 06에서 우리는 함수 종속성이 BCNF까지의 정규화를 어떻게 이끄는지 살펴보았습니다. 그러나 BCNF는 **모든** 형태의 중복성을 제거하지 않습니다. BCNF에 있는 릴레이션이 여전히 다른 종류의 제약조건에 의해 야기된 중복 데이터를 포함하는 상황이 있습니다: **다치 종속성(multivalued dependencies)**과 **조인 종속성(join dependencies)**입니다.

이 레슨은 이러한 문제를 다루는 더 높은 정규형(4NF, 5NF, DKNF)을 다루고, 때로는 **덜 정규화하는 것이 더 나은** 실무적 현실 — 비정규화의 예술(art of denormalization) — 로 전환합니다.

### 1.1 정규형의 계층(The Hierarchy of Normal Forms)

```
DKNF  (Domain-Key Normal Form — 이론적 이상)
  ↑
5NF / PJNF  (Project-Join Normal Form)
  ↑
4NF  (Fourth Normal Form)
  ↑
BCNF  (Boyce-Codd Normal Form)
  ↑
3NF  (Third Normal Form)
  ↑
2NF  (Second Normal Form)
  ↑
1NF  (First Normal Form)
```

각 수준은 추가적인 중복성 클래스를 제거합니다. 실무에서 BCNF는 대부분의 애플리케이션에 충분합니다. 4NF는 가끔 필요합니다. 5NF와 DKNF는 주로 이론적 관심사입니다.

---

## 2. 다치 종속성(Multivalued Dependencies, MVDs)

### 2.1 동기(Motivation)

직원, 그들의 기술, 그리고 할당된 프로젝트를 추적하는 릴레이션을 고려하세요:

```
EmpSkillProject(emp_id, skill, project)
```

다음을 가정합니다:
- 직원의 기술은 그들의 프로젝트 할당과 **독립적**입니다
- 직원 E1은 기술 {Java, Python}을 가지고 있고 프로젝트 {Alpha, Beta}에서 일합니다

이를 올바르게 표현하려면 **모든 조합**이 필요합니다:

```
| emp_id | skill  | project |
|--------|--------|---------|
| E1     | Java   | Alpha   |
| E1     | Java   | Beta    |
| E1     | Python | Alpha   |
| E1     | Python | Beta    |
```

하나의 행(예: E1/Python/Beta)을 잊어버리면, 데이터는 E1의 Python 기술이 Beta 프로젝트와 연관되지 않았다고 잘못 암시합니다. 이 "모든 조합" 요구사항이 **다치 종속성**의 특징입니다.

참고: 이 릴레이션은 BCNF에 있습니다 (유일한 후보 키는 전체 집합 {emp_id, skill, project}이고, 비자명한 FD가 없습니다). 그러나 명백한 중복성이 있습니다 — 각 기술은 프로젝트당 한 번씩 나열되고, 그 반대도 마찬가지입니다.

### 2.2 정의(Definition)

> **정의**: 릴레이션 R에 대해 **다치 종속성(multivalued dependency, MVD)** X →→ Y가 성립한다는 것은, R의 모든 튜플 쌍 t₁과 t₂에 대해 t₁[X] = t₂[X]이면, R에 튜플 t₃와 t₄가 존재하여 다음을 만족함을 의미합니다:
>
> - t₃[X] = t₄[X] = t₁[X] = t₂[X]
> - t₃[Y] = t₁[Y] and t₃[Z] = t₂[Z]
> - t₄[Y] = t₂[Y] and t₄[Z] = t₁[Z]
>
> 여기서 Z = R - X - Y (나머지 모든 속성).

더 간단한 말로: X를 고정하면, Y 값의 집합은 다른 속성들과 독립적입니다. 모든 조합이 나타나야 합니다.

### 2.3 직관적 이해(Intuitive Understanding)

X →→ Y의 의미:

> "주어진 X 값에 대해, Y가 취하는 값들의 집합은 나머지 속성들(R - X - Y)이 취하는 값들과 **독립적**입니다."

우리 예제에서:
- emp_id →→ skill (주어진 직원에 대해, 기술은 프로젝트와 독립적입니다)
- emp_id →→ project (주어진 직원에 대해, 프로젝트는 기술과 독립적입니다)

### 2.4 MVD의 속성(Properties of MVDs)

**보완 규칙(Complementation rule)**: X →→ Y이면, X →→ Z (여기서 Z = R - X - Y).

이는 정의로부터 직접 따라옵니다. 우리 예제에서, emp_id →→ skill은 emp_id →→ project를 함의합니다.

**모든 FD는 MVD입니다**: X → Y이면, X →→ Y. (하지만 역은 성립하지 않습니다.)

증명: X → Y이고 t₁[X] = t₂[X]이면, t₁[Y] = t₂[Y]. "교환(swap)" 튜플 t₃와 t₄는 바로 t₁과 t₂ 자체입니다.

**MVD 추론 규칙(MVD inference rules)** (암스트롱 공리와 유사):

| 규칙 | 설명 |
|------|-----------|
| 보완(Complementation) | X →→ Y ⟹ X →→ (R - X - Y) |
| 확장(Augmentation) | X →→ Y and W ⊇ Z ⟹ XW →→ YZ |
| 이행(Transitivity) | X →→ Y and Y →→ Z ⟹ X →→ (Z - Y) |
| 복제(Replication) | X → Y ⟹ X →→ Y |
| 결합(Coalescence) | X →→ Y, Z ⊆ Y, W ∩ Y = ∅, W → Z ⟹ X → Z |

### 2.5 자명한 MVD(Trivial MVDs)

> **정의**: MVD X →→ Y가 **자명(trivial)**하다는 것은:
> - Y ⊆ X이거나,
> - X ∪ Y = R (릴레이션의 모든 속성)

자명한 MVD는 항상 성립하며 중복성을 야기하지 않습니다.

---

## 3. 제4정규형(Fourth Normal Form, 4NF)

### 3.1 정의(Definition)

> **정의**: 릴레이션 스키마 R이 **제4정규형(Fourth Normal Form, 4NF)**에 있다는 것은, R에 성립하는 모든 비자명한 다치 종속성 X →→ Y에 대해:
>
> X가 R의 슈퍼키입니다.

4NF는 BCNF보다 엄격히 더 강합니다. 다치 종속성에 의한 중복성을 제거합니다.

### 3.2 BCNF와의 관계(Relationship to BCNF)

모든 FD는 MVD이므로, 모든 비자명한 MVD가 슈퍼키 결정자를 가지면, 모든 비자명한 FD도 슈퍼키 결정자를 가집니다. 따라서:

```
4NF ⊂ BCNF ⊂ 3NF ⊂ 2NF ⊂ 1NF
```

4NF에 있는 릴레이션은 항상 BCNF에 있지만, BCNF에 있는 릴레이션은 4NF에 있지 않을 수 있습니다 (우리의 EmpSkillProject 예제가 보여주듯이).

### 3.3 4NF 분해 알고리즘(4NF Decomposition Algorithm)

```
ALGORITHM: 4NF_Decomposition(R, D)
INPUT:  R = 릴레이션 스키마
        D = FD와 MVD의 집합
OUTPUT: 무손실 조인을 가진 4NF 릴레이션들로의 분해

result ← {R}

WHILE result에 4NF에 있지 않은 Rᵢ가 존재 DO
    Rᵢ에서 4NF를 위반하는 비자명한 MVD X →→ Y를 찾습니다
    (즉, X가 Rᵢ의 슈퍼키가 아님)

    Rᵢ를 다음으로 교체:
        R₁ = X ∪ Y
        R₂ = Rᵢ - Y    (또는, X ∪ Z, 여기서 Z = Rᵢ - X - Y)
END WHILE

RETURN result
```

이는 BCNF 분해 알고리즘과 유사하지만 FD 대신 MVD를 사용합니다.

### 3.4 작업 예제(Worked Example)

**EmpSkillProject(emp_id, skill, project)**

MVDs:
- emp_id →→ skill
- emp_id →→ project

Key: {emp_id, skill, project} (FD가 존재하지 않으므로 전체 릴레이션)

4NF 확인: emp_id →→ skill은 비자명하고, {emp_id}는 슈퍼키가 아닙니다. **위반!**

emp_id →→ skill에 대해 분해:
- R₁ = {emp_id, skill}
- R₂ = {emp_id, project}

R₁(emp_id, skill) 확인:
- Key: {emp_id, skill}
- 자명한 MVD만 존재. 4NF ✓

R₂(emp_id, project) 확인:
- Key: {emp_id, project}
- 자명한 MVD만 존재. 4NF ✓

**최종 4NF 분해:**
```
EmpSkill(emp_id, skill)       — key: {emp_id, skill}
EmpProject(emp_id, project)   — key: {emp_id, project}
```

### 3.5 더 복잡한 예제(A More Complex Example)

다음을 고려하세요:

```
CourseBook(course, teacher, book)
```

다음을 가정합니다:
- 각 과목은 여러 교사가 가르칠 수 있습니다
- 각 과목은 여러 책을 사용합니다
- 과목에 사용되는 책은 누가 가르치는지와 관계없이 동일합니다

MVDs: course →→ teacher, course →→ book

하지만 다음도 있다고 가정합니다: teacher → book (각 교사는 특정 책을 사용하며, 이는 교사마다 다를 수 있습니다). 이는 MVD가 아니라 FD입니다. 이 경우, MVD course →→ book은 성립하지 않을 수 있고, 분석이 달라집니다.

이는 MVD를 식별하려면 실제 세계의 의미론을 주의 깊게 분석해야 함을 보여줍니다.

---

## 4. 제5정규형(Fifth Normal Form, 5NF / PJNF)

### 4.1 조인 종속성(Join Dependencies)

> **정의**: **조인 종속성(join dependency, JD)** ⋈{R₁, R₂, ..., Rₙ}이 릴레이션 R에 성립한다는 것은:
>
> R = π_{R₁}(R) ⋈ π_{R₂}(R) ⋈ ... ⋈ π_{Rₙ}(R)
>
> R의 모든 합법적인 인스턴스에 대해 성립함을 의미합니다.

조인 종속성은 R이 항상 R₁, R₂, ..., Rₙ으로 무손실 분해될 수 있다고 말합니다.

### 4.2 MVD를 JD의 특수 경우로(MVDs as Special Case of JDs)

R에 대한 모든 MVD X →→ Y는 조인 종속성 ⋈{XY, XZ} (여기서 Z = R - X - Y)와 동등합니다.

따라서 MVD는 이진 조인 종속성(정확히 두 개의 구성 요소로의 분해)입니다. 일반 조인 종속성은 세 개 이상의 구성 요소로의 분해를 요구할 수 있습니다.

### 4.3 5NF의 정의(Definition of 5NF)

> **정의**: 릴레이션 스키마 R이 **제5정규형(Fifth Normal Form, 5NF)** 또는 **Project-Join Normal Form (PJNF)**에 있다는 것은, R에 성립하는 모든 비자명한 조인 종속성 ⋈{R₁, R₂, ..., Rₙ}에 대해:
>
> 모든 Rᵢ가 R의 슈퍼키입니다.

5NF는 조인 종속성으로 정의된 가장 강력한 정규형입니다. 투영(projection)과 조인을 통해 감지할 수 있는 모든 중복성을 제거합니다.

### 4.4 예제: 5NF의 필요성(Example: Need for 5NF)

공급자, 부품, 프로젝트에 관한 릴레이션을 고려하세요:

```
SPJ(supplier, part, project)
```

다음 제약조건이 성립한다고 가정합니다: "공급자 S가 부품 P를 공급하고, 부품 P가 프로젝트 J에 사용되며, 공급자 S가 프로젝트 J에 어떤 부품을 공급한다면, 공급자 S는 프로젝트 J에 부품 P를 공급합니다."

이는 **순환 조인 종속성(cyclic join dependency)**입니다:

⋈{(supplier, part), (part, project), (supplier, project)}

이는 SPJ = π_{supplier,part}(SPJ) ⋈ π_{part,project}(SPJ) ⋈ π_{supplier,project}(SPJ)를 의미합니다

이 JD는 어떤 MVD에 의해서도 함의되지 않습니다 (세 방향 분해가 필요합니다). 이 JD가 성립하고 전체 속성 집합이 유일한 키라면:
- 릴레이션은 4NF에 있을 수 있습니다 (비자명한 MVD가 없음)
- 하지만 5NF에는 **없습니다** (구성 요소가 슈퍼키가 아닌 비자명한 JD)

**분해:**
```
SP(supplier, part)         — 어떤 공급자가 어떤 부품을 공급하는지
PJ(part, project)          — 어떤 부품이 어떤 프로젝트에 사용되는지
SJ(supplier, project)      — 어떤 공급자가 어떤 프로젝트에 공급하는지
```

### 4.5 조인 종속성 감지(Detecting Join Dependencies)

조인 종속성은 실무에서 감지하기 매우 어렵습니다:

1. 미묘합니다 — SPJ 예제의 순환 제약조건은 명백하지 않습니다
2. 속성 폐포 계산과 유사한 간단한 테스트가 없습니다
3. 데이터 검사가 아니라 도메인 지식으로부터 식별되어야 합니다

이것이 5NF가 주로 이론적 관심사인 이유입니다.

---

## 5. 도메인-키 정규형(Domain-Key Normal Form, DKNF)

### 5.1 정의(Definition)

> **정의 (Fagin, 1981)**: 릴레이션 스키마 R이 **도메인-키 정규형(Domain-Key Normal Form, DKNF)**에 있다는 것은, R의 모든 제약조건이 R의 도메인 제약조건과 키 제약조건의 논리적 귀결임을 의미합니다.
>
> - **도메인 제약조건(Domain constraint)**: 속성의 허용 값에 대한 제한 (예: age > 0, status ∈ {'active', 'inactive'})
> - **키 제약조건(Key constraint)**: 속성 집합에 대한 유일성 제약조건

### 5.2 중요성(Significance)

DKNF는 "궁극의" 정규형입니다. 릴레이션이 DKNF에 있으면, 알려진 모든 유형의 종속성(FD, MVD, JD 또는 기타)으로 특성화될 수 있는 중복성이 **없습니다**.

그러나 DKNF는 중요한 제한이 있습니다:

> **릴레이션을 DKNF로 변환하는 일반 알고리즘은 존재하지 않습니다.**

이는 DKNF를 이론적 이상으로 만듭니다 — 완벽한 정규화가 어떻게 보이는지 알려주지만, 모든 경우에 거기에 도달하는 방법을 알려주지 않습니다.

### 5.3 예제(Example)

```
Employee(emp_id, emp_name, dept, salary)
```

도메인 제약조건:
- emp_id: 양의 정수
- salary: 양의 십진수
- dept ∈ {'Engineering', 'Marketing', 'Sales', 'HR'}

키 제약조건: emp_id가 기본 키.

유일한 제약조건이 emp_id가 다른 모든 속성을 유일하게 결정한다는 것이고, 도메인 제약조건이 위와 같다면, 이 릴레이션은 DKNF에 있습니다 — 모든 제약조건이 키와 도메인 제약조건만으로부터 따라옵니다.

### 5.4 DKNF 실패 경우(When DKNF Fails)

릴레이션이 DKNF에 **없는** 경우는 도메인 또는 키 제약조건으로 표현할 수 없는 제약조건이 있을 때입니다. 예를 들어:

```
Employee(emp_id, emp_name, dept, dept_budget)
```

제약조건: dept → dept_budget (같은 부서의 모든 직원은 동일한 예산을 봅니다).

이는 키 제약조건이 **아닌** FD입니다 (dept는 키가 아닙니다). 따라서 이 릴레이션은 DKNF에 없습니다. 해결책: Employee(emp_id, emp_name, dept)와 Department(dept, dept_budget)로 분해하며, dept는 Department 테이블에서 키입니다.

### 5.5 실용적 관련성(Practical Relevance)

| 정규형 | 실무 사용 |
|-------------|---------------|
| 1NF - BCNF | 매우 일반적. 모든 데이터베이스 설계자가 알아야 합니다. |
| 4NF | 가끔 필요. 독립적인 다대다 관계. |
| 5NF | 드뭅니다. 특정 도메인의 순환 제약조건. |
| DKNF | 이론적. 실용적 알고리즘이 존재하지 않습니다. |

---

## 6. 모든 정규형의 요약(Summary of All Normal Forms)

| NF | 조건 | 제거하는 것 |
|----|-----------|-----------|
| **1NF** | 원자적 값 | 비관계형 구조 |
| **2NF** | 부분 FD 없음 | 부분 키 종속성 |
| **3NF** | 이행 FD 없음 (주 속성 예외 있음) | 이행 종속성 |
| **BCNF** | 모든 FD 결정자가 슈퍼키 | 모든 FD 기반 중복성 |
| **4NF** | 모든 비자명한 MVD 결정자가 슈퍼키 | MVD 기반 중복성 |
| **5NF** | 모든 비자명한 JD 구성 요소가 슈퍼키 | JD 기반 중복성 |
| **DKNF** | 모든 제약조건이 도메인 + 키로부터 따라옴 | 모든 가능한 중복성 |

---

## 7. 비정규화(Denormalization)

### 7.1 언제 비정규화할까(When to Denormalize)

정규화는 **데이터 무결성**과 **저장 효율성**을 최적화합니다. 하지만 실제 시스템은 **읽기 성능**도 필요합니다. 비정규화는 쿼리 성능을 향상시키기 위해 의도적으로 중복성을 도입하는 것입니다.

비정규화하는 일반적인 이유:

1. **비용이 많이 드는 조인**: 많은 테이블을 조인하는 자주 실행되는 쿼리
2. **집계 성능**: 사전 계산된 count, sum, average
3. **읽기 중심 워크로드**: 읽기가 쓰기보다 훨씬 많은 시스템 (예: 100:1 비율)
4. **보고 및 분석**: 과거 데이터에 대한 복잡한 분석 쿼리
5. **지연 시간 요구사항**: 밀리초 이하의 응답 시간

### 7.2 비정규화 기법(Denormalization Techniques)

#### 기법 1: 사전 조인된 테이블(Prejoined Tables)

여러 정규화된 테이블의 데이터를 단일 테이블에 저장:

```sql
-- 정규화됨: 조인 필요
SELECT o.order_id, o.order_date, c.customer_name, c.email
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id;

-- 비정규화됨: 조인 불필요
SELECT order_id, order_date, customer_name, customer_email
FROM orders_denormalized;
```

**트레이드오프**: 더 빠른 읽기, 하지만 customer_name과 customer_email이 모든 주문에 걸쳐 중복됩니다.

#### 기법 2: 파생/계산 열(Derived/Computed Columns)

사전 계산된 값 저장:

```sql
-- 매번 계산하는 대신:
SELECT order_id, SUM(quantity * price) AS total
FROM order_items
GROUP BY order_id;

-- 총액을 직접 저장:
ALTER TABLE orders ADD COLUMN total_amount DECIMAL(10,2);

-- 각 항목 변경 시 업데이트:
UPDATE orders SET total_amount = (
    SELECT SUM(quantity * price)
    FROM order_items WHERE order_items.order_id = orders.order_id
) WHERE order_id = ?;
```

#### 기법 3: 중복 열(Redundant Columns)

조인을 피하기 위해 자주 액세스되는 외래 테이블 열 추가:

```sql
-- 정규화됨
CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer_id INT REFERENCES customers(customer_id),
    order_date DATE
);

-- 비정규화됨: 표시 목적으로 customer_name 추가
CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer_id INT REFERENCES customers(customer_id),
    customer_name VARCHAR(100),  -- 중복이지만 조인을 피함
    order_date DATE
);
```

#### 기법 4: 요약/집계 테이블(Summary/Aggregate Tables)

사전 계산된 집계를 위한 별도의 테이블 생성:

```sql
CREATE TABLE daily_sales_summary (
    sale_date   DATE PRIMARY KEY,
    total_orders INT,
    total_revenue DECIMAL(12,2),
    avg_order_value DECIMAL(10,2)
);

-- 야간 배치 작업 또는 트리거로 채워짐
```

#### 기법 5: 구체화된 뷰(Materialized Views)

일부 데이터베이스는 구체화된 뷰를 지원합니다 — 테이블로 저장된 사전 계산 쿼리 결과:

```sql
-- PostgreSQL
CREATE MATERIALIZED VIEW product_sales_summary AS
SELECT
    p.product_id,
    p.product_name,
    p.category,
    COUNT(oi.order_id) AS times_ordered,
    SUM(oi.quantity) AS total_quantity,
    SUM(oi.quantity * oi.price) AS total_revenue
FROM products p
JOIN order_items oi ON p.product_id = oi.product_id
GROUP BY p.product_id, p.product_name, p.category;

-- 주기적으로 새로고침
REFRESH MATERIALIZED VIEW product_sales_summary;
```

### 7.3 비정규화된 데이터 관리(Managing Denormalized Data)

비정규화는 정규화가 방지하도록 설계된 바로 그 이상 현상들을 도입합니다. 이를 관리하는 전략:

**1. 애플리케이션 수준 강제(Application-level enforcement)**

애플리케이션이 쓰기 시 일관성을 보장:

```python
def update_customer_name(customer_id, new_name):
    # 정규화된 소스 업데이트
    db.execute("UPDATE customers SET name = ? WHERE id = ?",
               new_name, customer_id)

    # 모든 비정규화된 복사본 업데이트
    db.execute("UPDATE orders SET customer_name = ? WHERE customer_id = ?",
               new_name, customer_id)
    db.commit()
```

**2. 데이터베이스 트리거(Database triggers)**

데이터베이스가 일관성을 자동으로 유지하도록 합니다:

```sql
CREATE OR REPLACE FUNCTION sync_customer_name()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.name <> OLD.name THEN
        UPDATE orders
        SET customer_name = NEW.name
        WHERE customer_id = NEW.customer_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_customer_name_sync
AFTER UPDATE OF name ON customers
FOR EACH ROW EXECUTE FUNCTION sync_customer_name();
```

**3. 최종 일관성(Eventual consistency)**

비정규화된 데이터가 일시적으로 오래될 수 있음을 수용:

```sql
-- 신선도 추적을 위한 last_synced 타임스탬프 사용
ALTER TABLE orders_denormalized ADD COLUMN last_synced TIMESTAMP;

-- 백그라운드 작업이 주기적으로 새로고침
UPDATE orders_denormalized od
SET customer_name = c.name, last_synced = NOW()
FROM customers c
WHERE od.customer_id = c.customer_id
AND od.customer_name <> c.name;
```

### 7.4 정규화-비정규화 스펙트럼(The Normalization-Denormalization Spectrum)

```
완전 정규화 (BCNF/4NF)                완전 비정규화
├──────────────────────────────────────────────────┤
│                                                      │
│  ✓ 중복성 없음          ✗ 최대 중복성                │
│  ✓ 이상 현상 없음       ✗ 모든 이상 현상 가능        │
│  ✗ 많은 조인 필요        ✓ 조인 불필요                │
│  ✗ 복잡한 쿼리           ✓ 간단한 쿼리                │
│  ✓ 쓰기 최적화           ✓ 읽기 최적화                │
│                                                      │
│         OLTP 시스템 ←──────→ OLAP/분석              │
```

### 7.5 의사결정 프레임워크(Decision Framework)

비정규화하기 전에 다음 질문들을 하세요:

1. **조인이 정말 병목인가?** 먼저 프로파일링하세요. 종종 적절한 인덱싱이 비정규화의 필요성을 제거합니다.
2. **읽기:쓰기 비율은 얼마인가?** 비정규화는 읽기가 지배적일 때 (>10:1) 가장 도움이 됩니다.
3. **일관성이 얼마나 중요한가?** 금융 시스템은 완벽한 일관성을 요구합니다; 추천 엔진은 약간의 지연을 허용할 수 있습니다.
4. **구체화된 뷰로 충분한가?** 기본 스키마를 수정하지 않고 비정규화의 이점을 제공합니다.
5. **팀이 유지보수할 준비가 되어 있나?** 비정규화된 스키마는 규율이 필요합니다 — 모든 쓰기 경로가 모든 복사본을 업데이트해야 합니다.

---

## 8. 스타 스키마와 스노플레이크 스키마(Star Schema and Snowflake Schema)

데이터 웨어하우징은 분석 쿼리에 최적화된 특정 비정규화 패턴을 사용합니다.

### 8.1 스타 스키마(Star Schema)

스타 스키마는 가장 간단한 데이터 웨어하우스 패턴입니다. 다음으로 구성됩니다:
- 중심에 하나의 **팩트 테이블(fact table)** (측정/이벤트)
- 바깥으로 방사하는 여러 **차원 테이블(dimension tables)** (설명 속성)

```
                    ┌──────────────┐
                    │   dim_date   │
                    │──────────────│
                    │ date_key (PK)│
                    │ full_date    │
                    │ year         │
                    │ quarter      │
                    │ month        │
                    │ day_of_week  │
                    └──────┬───────┘
                           │
┌──────────────┐    ┌──────┴───────┐    ┌──────────────┐
│ dim_product  │    │  fact_sales  │    │ dim_customer │
│──────────────│    │──────────────│    │──────────────│
│ product_key  │◄───│ product_key  │    │ customer_key │
│ product_name │    │ customer_key │───►│ cust_name    │
│ category     │    │ date_key     │    │ city         │
│ subcategory  │    │ store_key    │    │ state        │
│ brand        │    │──────────────│    │ segment      │
└──────────────┘    │ quantity     │    └──────────────┘
                    │ unit_price   │
                    │ total_amount │    ┌──────────────┐
                    │ discount     │    │  dim_store   │
                    └──────┬───────┘    │──────────────│
                           │            │ store_key    │
                           └───────────►│ store_name   │
                                        │ city         │
                                        │ state        │
                                        │ region       │
                                        └──────────────┘
```

**주요 특징:**
- 팩트 테이블은 모든 차원 테이블에 대한 외래 키를 가집니다
- 차원 테이블은 **비정규화**되어 있습니다 (예: dim_product는 category와 subcategory를 하나의 평면 테이블에 가집니다)
- 쿼리는 일반적으로 차원에 필터링하고 팩트를 집계합니다

```sql
-- 전형적인 스타 스키마 쿼리: 제품 카테고리 및 분기별 총 판매액
SELECT
    d.category,
    dd.year,
    dd.quarter,
    SUM(f.total_amount) AS total_sales
FROM fact_sales f
JOIN dim_product d ON f.product_key = d.product_key
JOIN dim_date dd ON f.date_key = dd.date_key
WHERE dd.year = 2025
GROUP BY d.category, dd.year, dd.quarter
ORDER BY dd.quarter, total_sales DESC;
```

### 8.2 스노플레이크 스키마(Snowflake Schema)

스노플레이크 스키마는 차원 테이블을 하위 차원으로 정규화합니다:

```
                         ┌──────────────┐
                         │  dim_brand   │
                         │──────────────│
                         │ brand_key    │
                         │ brand_name   │
                         │ manufacturer │
                         └──────┬───────┘
                                │
┌──────────────┐    ┌───────────┴──┐    ┌──────────────┐
│dim_subcategory│   │ dim_product  │    │  fact_sales  │
│──────────────│    │──────────────│    │──────────────│
│ subcat_key   │◄───│ product_key  │◄───│ product_key  │
│ subcat_name  │    │ product_name │    │ customer_key │
│ category_key │    │ subcat_key   │    │ date_key     │
└──────┬───────┘    │ brand_key    │    │ quantity     │
       │            └──────────────┘    │ total_amount │
┌──────┴───────┐                        └──────────────┘
│dim_category  │
│──────────────│
│ category_key │
│ category_name│
└──────────────┘
```

**주요 특징:**
- 차원 테이블이 **정규화**되어 있습니다 (3NF 또는 BCNF)
- 큰 차원 테이블의 저장 공간 감소
- 쿼리에 더 많은 조인 필요

### 8.3 스타 vs 스노플레이크(Star vs Snowflake)

| 측면 | 스타 스키마 | 스노플레이크 스키마 |
|--------|-------------|------------------|
| **차원 구조** | 평면 (비정규화) | 정규화 (2NF-BCNF) |
| **쿼리 복잡성** | 더 간단 (더 적은 조인) | 더 복잡 (더 많은 조인) |
| **쿼리 성능** | 더 빠름 (더 적은 조인) | 더 느림 (더 많은 조인) |
| **저장 공간** | 더 많음 (차원의 중복성) | 더 적음 (중복성 없음) |
| **ETL 복잡성** | 더 간단 | 더 복잡 |
| **유지보수** | 업데이트가 더 많은 행에 영향 | 업데이트가 격리됨 |
| **업계 선호도** | 가장 일반적 | 차원이 매우 클 때 사용 |

### 8.4 팩트 테이블 유형(Fact Table Types)

| 유형 | 설명 | 예 |
|------|-------------|---------|
| **트랜잭션(Transaction)** | 이벤트당 하나의 행 | 각 판매, 클릭, 로그인 |
| **주기적 스냅샷(Periodic snapshot)** | 정기적 간격 | 일일 계좌 잔액, 월별 재고 |
| **누적 스냅샷(Accumulating snapshot)** | 생애주기 추적 | 주문: 주문됨 → 배송됨 → 배달됨 날짜 |
| **팩트리스 팩트(Factless fact)** | 측정값이 없는 이벤트 | 학생 출석 (키만) |

### 8.5 느리게 변하는 차원(Slowly Changing Dimensions, SCD)

차원 속성이 시간에 따라 변할 때:

**유형 1: 덮어쓰기(Overwrite)** — 단순히 값을 업데이트. 이력을 잃습니다.

```sql
UPDATE dim_customer SET city = 'New York' WHERE customer_key = 42;
```

**유형 2: 새 행 추가(Add new row)** — 유효 날짜가 있는 새 차원 행을 생성.

```sql
-- 전: 고객이 Boston에 살았음
| customer_key | cust_id | city    | valid_from | valid_to   | current |
|-------------|---------|---------|------------|------------|---------|
| 42          | C100    | Boston  | 2020-01-01 | 9999-12-31 | Y       |

-- 후: 고객이 New York으로 이사
| customer_key | cust_id | city     | valid_from | valid_to   | current |
|-------------|---------|----------|------------|------------|---------|
| 42          | C100    | Boston   | 2020-01-01 | 2025-06-30 | N       |
| 99          | C100    | New York | 2025-07-01 | 9999-12-31 | Y       |
```

**유형 3: 새 열 추가(Add new column)** — 이전 값을 추적.

```sql
| customer_key | city     | prev_city | city_change_date |
|-------------|----------|-----------|-----------------|
| 42          | New York | Boston    | 2025-07-01      |
```

---

## 9. 실용적 정규화 지침(Practical Normalization Guidelines)

### 9.1 경험 법칙(Rules of Thumb)

1. **정규화로 시작하고, 필요성이 입증되면 비정규화하세요**. 조기 비정규화는 흔한 실수입니다.

2. **OLTP 시스템**: 최소 3NF, 가급적 BCNF로 정규화하세요. 쓰기 중심 워크로드는 일관성을 요구합니다.

3. **OLAP/분석**: 스타 또는 스노플레이크 스키마를 사용하세요. 읽기 중심 분석 워크로드는 더 적은 조인으로 이득을 봅니다.

4. **마이크로서비스**: 각 서비스가 자체 데이터를 소유합니다. 서비스 내에서는 정규화하고; 서비스 간 중복성은 수용합니다.

5. **NoSQL 데이터베이스**: 문서 저장소(MongoDB)는 종종 내장/비정규화 구조를 사용합니다. 그래프 데이터베이스는 자체 모델링 패턴을 가집니다. 관계형 정규화 이론은 주로 관계형 데이터베이스에 적용됩니다.

### 9.2 일반적인 패턴(Common Patterns)

**패턴 1: 룩업 테이블(Lookup tables)** — 작고 거의 변하지 않는 참조 데이터.

```sql
-- 항상 정규화: 국가, 통화, 상태 코드
CREATE TABLE country (
    country_code CHAR(2) PRIMARY KEY,
    country_name VARCHAR(100) NOT NULL
);
```

**패턴 2: 감사/이력 테이블(Audit/history tables)** — 의도적으로 비정규화.

```sql
-- 이벤트 시점의 데이터 스냅샷 저장
CREATE TABLE order_audit (
    audit_id       SERIAL PRIMARY KEY,
    order_id       INT,
    customer_name  VARCHAR(100),  -- 스냅샷, FK 아님
    product_name   VARCHAR(100),  -- 스냅샷, FK 아님
    total_amount   DECIMAL(10,2),
    recorded_at    TIMESTAMP DEFAULT NOW()
);
```

**패턴 3: 캐시 테이블(Cache tables)** — 성능을 위해 사전 계산.

```sql
-- 정규화된 소스: orders + order_items로부터 계산
-- 캐시: 트리거 또는 배치 작업으로 업데이트
CREATE TABLE customer_stats (
    customer_id     INT PRIMARY KEY REFERENCES customers(customer_id),
    total_orders    INT DEFAULT 0,
    total_spent     DECIMAL(12,2) DEFAULT 0,
    last_order_date DATE,
    updated_at      TIMESTAMP DEFAULT NOW()
);
```

### 9.3 피해야 할 안티패턴(Anti-Patterns to Avoid)

**1. Entity-Attribute-Value (EAV)**

```sql
-- 피할 것: 유사-유연하지만 쿼리와 무결성에 끔찍함
CREATE TABLE attributes (
    entity_id   INT,
    attr_name   VARCHAR(100),
    attr_value  VARCHAR(500)
);
```

문제: 타입 안전성 없음, 참조 무결성 없음, 끔찍한 쿼리 성능.

**2. 과정규화(Over-normalization)**

`Address(street, city, state, zip)`와 같은 테이블을 `Address(street, zip_id)` + `ZipCode(zip_id, city, state)`로 분할하는 것은 기술적으로 올바르지만 (zip → city, state) 거의 유익하지 않습니다 — 우편번호 데이터는 극히 드물게 변합니다.

**3. One True Lookup Table (OTLT)**

```sql
-- 피할 것: 모든 참조 데이터를 하나의 테이블에 넣기
CREATE TABLE lookup (
    lookup_type  VARCHAR(50),
    lookup_code  VARCHAR(50),
    lookup_value VARCHAR(200)
);
```

문제: 외래 키 무결성 없음, 타입별 검증 없음, 혼란스러운 의미론.

---

## 10. 연습문제(Exercises)

### 연습문제 1: MVD 식별(Identifying MVDs)

R(student, course, hobby)가 주어지고, 학생의 과목들이 그들의 취미와 독립적이라면:

1. 어떤 MVD가 성립하나요?
2. 가장 높은 정규형은 무엇인가요?
3. 4NF로 분해하세요.

<details>
<summary>해답</summary>

1. **MVDs**: student →→ course 그리고 student →→ hobby (보완에 의해)

2. **가장 높은 NF**: 유일한 키는 {student, course, hobby}입니다. 비자명한 FD가 없으므로 BCNF가 성립합니다. 하지만 MVD student →→ course는 결정자 {student}가 슈퍼키가 아닙니다. **BCNF이지만 4NF는 아닙니다.**

3. **4NF 분해**:
   - R₁(student, course) — key: {student, course}
   - R₂(student, hobby) — key: {student, hobby}

   둘 다 4NF에 있습니다 (자명한 MVD만 남음). ✓
</details>

### 연습문제 2: MVD vs FD

R(A, B, C)가 주어지고 제약조건: A의 각 값에 대해 B 값의 집합이 C 값과 관계없이 고정됨.

1. 이것은 FD인가요 MVD인가요?
2. A → B가 성립하나요?
3. A →→ B가 성립하나요?

<details>
<summary>해답</summary>

1. 이것은 **MVD**입니다. 제약조건은 B 값과 C 값이 A가 주어졌을 때 독립적이라고 말합니다.

2. **A → B는 반드시 성립하지 않습니다.** FD A → B는 각 A 값이 정확히 하나의 B 값과 연관됨을 의미합니다. 제약조건은 A가 **집합**의 B 값과 연관되며, 이는 하나 이상의 요소를 가질 수 있다고 말합니다.

3. **A →→ B가 성립합니다.** 이것이 정확히 다치 종속성의 정의입니다: 주어진 A에 대해 B 값의 집합이 C 값과 독립적입니다.

예:
```
| A  | B  | C  |
|----|----|----|
| a1 | b1 | c1 |
| a1 | b1 | c2 |
| a1 | b2 | c1 |
| a1 | b2 | c2 |
```
A →→ B가 성립합니다 (모든 A-B 조합이 모든 C와 나타남). A → B는 성립하지 않습니다 (a1이 b1과 b2 모두에 매핑됨).
</details>

### 연습문제 3: 스타 스키마 설계(Star Schema Design)

도서관 대출 시스템을 위한 스타 스키마를 설계하세요. 팩트는 "책 대출 이벤트"입니다. 다음을 식별하세요:
1. 팩트 테이블과 그 측정값
2. 최소 3개의 차원 테이블
3. 스키마를 사용한 샘플 SQL 쿼리

<details>
<summary>해답</summary>

**팩트 테이블: fact_checkout**
```sql
CREATE TABLE fact_checkout (
    checkout_key    SERIAL PRIMARY KEY,
    date_key        INT REFERENCES dim_date(date_key),
    book_key        INT REFERENCES dim_book(book_key),
    patron_key      INT REFERENCES dim_patron(patron_key),
    branch_key      INT REFERENCES dim_branch(branch_key),
    -- 측정값
    days_borrowed   INT,
    is_returned     BOOLEAN,
    late_fee        DECIMAL(6,2)
);
```

**차원 테이블:**
```sql
CREATE TABLE dim_date (
    date_key    INT PRIMARY KEY,
    full_date   DATE,
    year        INT,
    month       INT,
    day_of_week VARCHAR(10),
    is_weekend  BOOLEAN
);

CREATE TABLE dim_book (
    book_key    INT PRIMARY KEY,
    isbn        VARCHAR(20),
    title       VARCHAR(200),
    author      VARCHAR(100),
    genre       VARCHAR(50),
    publisher   VARCHAR(100),
    pub_year    INT
);

CREATE TABLE dim_patron (
    patron_key  INT PRIMARY KEY,
    patron_id   VARCHAR(20),
    name        VARCHAR(100),
    membership  VARCHAR(20),  -- 'adult', 'student', 'senior'
    city        VARCHAR(50)
);

CREATE TABLE dim_branch (
    branch_key  INT PRIMARY KEY,
    branch_name VARCHAR(100),
    city        VARCHAR(50),
    state       VARCHAR(2)
);
```

**샘플 쿼리: 월별 가장 인기 있는 장르:**
```sql
SELECT
    dd.year,
    dd.month,
    db.genre,
    COUNT(*) AS checkouts
FROM fact_checkout f
JOIN dim_date dd ON f.date_key = dd.date_key
JOIN dim_book db ON f.book_key = db.book_key
WHERE dd.year = 2025
GROUP BY dd.year, dd.month, db.genre
ORDER BY dd.month, checkouts DESC;
```
</details>

### 연습문제 4: 4NF 분해(4NF Decomposition)

R(A, B, C, D)가 주어지고:
- A →→ B
- A → C

4NF로 분해하세요.

<details>
<summary>해답</summary>

먼저, 모든 종속성을 식별합니다:
- A →→ B는 A →→ CD를 함의합니다 (보완, R - A - B = {C, D}이므로)
- A → C (FD, A →→ C를 함의)

키: {A, B, D}⁺ 또는 실제 키를 찾으세요. A → C이므로 A로부터 C를 얻습니다. 따라서 B와 D를 결정하기 위해 A와 충분한 것이 필요합니다. A →→ B이면 B는 다치 값이므로 B는 키에 있어야 합니다. 마찬가지로 D가 키에 있을 수 있습니다.

키 후보: {A, B, D} (A가 C만 함수적으로 결정하고, B와 D가 A가 주어졌을 때 서로 독립이므로).

4NF 확인: A →→ B는 비자명하고, {A}는 슈퍼키가 아닙니다. 위반!

**A →→ B에 대해 분해:**
- R₁(A, B) — key: {A, B}
- R₂(A, C, D) — key: {A, D} (A → C이므로)

R₁(A, B) 확인: 자명한 MVD만. 4NF ✓
R₂(A, C, D) 확인: A → C. {A}가 슈퍼키인가? {A}⁺ = {A, C}. 아니요, D가 결정되지 않음. 키는 {A, D}입니다. A → C는 A가 슈퍼키가 아닌 FD입니다 — 이는 BCNF를 위반합니다!

**R₂를 A → C에 대해 분해:**
- R₃(A, C) — key: {A}
- R₄(A, D) — key: {A, D}

**최종 4NF 분해:**
```
R₁(A, B)    — key: {A, B}
R₃(A, C)    — key: {A}
R₄(A, D)    — key: {A, D}
```

모두 4NF에 있습니다 ✓.
</details>

### 연습문제 5: 비정규화 결정(Denormalization Decision)

각 시나리오에 대해 정규화할 것인지 비정규화할 것인지 결정하세요. 이유를 설명하세요.

1. 1000만 개의 제품이 있는 전자상거래 제품 카탈로그, 읽기:쓰기 비율 1000:1
2. 전신송금을 처리하는 은행 거래 시스템
3. 저자 이름과 프로필 사진이 있는 게시물을 보여주는 소셜 미디어 뉴스 피드
4. 매초 센서 판독값을 기록하는 과학 데이터 수집 시스템

<details>
<summary>해답</summary>

1. **제품 카탈로그 (1000만 개 제품, 1000:1 읽기:쓰기)**: **비정규화.** 압도적인 읽기 우세가 중복성을 정당화합니다. 카테고리 이름, 브랜드 이름 등이 직접 내장된 비정규화된 제품 테이블을 사용하세요. 검색/필터 페이지를 위한 구체화된 뷰를 고려하세요. 업데이트는 드물고 배치 처리될 수 있습니다.

2. **은행 전신송금**: **정규화 (BCNF).** 금융 데이터는 완벽한 일관성을 요구합니다. 모든 센트는 정확히 한 번 계상되어야 합니다. 조인의 성능 비용은 수용 가능합니다 — 정확성이 최우선입니다. 적절한 외래 키와 제약조건이 있는 정규화된 테이블을 사용하세요.

3. **소셜 미디어 뉴스 피드**: **비정규화.** 뉴스 피드는 읽기 중심이고 지연 시간에 민감합니다. 저자 이름과 프로필 사진 URL을 피드 항목에 직접 저장하세요 (또는 Redis 같은 캐시 계층 사용). 최종 일관성을 수용하세요 — 사용자가 프로필 사진을 변경하면 이전 게시물이 일시적으로 이전 사진을 보여주는 것이 수용 가능합니다.

4. **센서 데이터 수집**: **하이브리드.** 센서 메타데이터 (sensor_id, location, type)는 정규화되어야 합니다 (거의 변하지 않음). 시계열 판독값은 특수 패턴을 사용할 수 있습니다 — 시계열 데이터베이스 (InfluxDB, TimescaleDB) 또는 시간별로 파티션된 비정규화 와이드 테이블. 핵심 제약은 조인 복잡성이 아니라 쓰기 처리량입니다.
</details>

### 연습문제 6: 조인 종속성(Join Dependency)

R(A, B, C)가 조인 종속성 ⋈{(A,B), (B,C), (A,C)}를 가질 때:

1. 이 릴레이션이 반드시 4NF에 있나요?
2. 5NF에 있나요?
3. 5NF로 분해하세요.

<details>
<summary>해답</summary>

1. **4NF?** 아마도 예. JD ⋈{(A,B), (B,C), (A,C)}는 삼원 조인 종속성이며, 어떤 MVD에 의해서도 함의되지 않습니다. 비자명한 MVD가 없으면 릴레이션은 4NF에 있습니다.

2. **5NF?** 아니요. JD ⋈{(A,B), (B,C), (A,C)}는 비자명합니다 (후보 키에 의해 함의되지 않음). 유일한 키가 {A, B, C} (모든 속성)라면, {A,B}, {B,C}, {A,C}는 슈퍼키가 아닙니다. **5NF에 없습니다.**

3. **5NF 분해:**
   ```
   R₁(A, B)   — key: {A, B}
   R₂(B, C)   — key: {B, C}
   R₃(A, C)   — key: {A, C}
   ```

   JD는 이 분해가 무손실임을 보장합니다: R = R₁ ⋈ R₂ ⋈ R₃. 각 구성 요소는 자명한 JD만 가집니다. 모두 5NF에 있습니다 ✓.
</details>

---

## 11. 요약(Summary)

| 개념 | 핵심 포인트 |
|---------|-----------|
| **MVD (X →→ Y)** | 주어진 X에 대해, Y 값은 다른 속성들과 독립적 |
| **4NF** | 모든 비자명한 MVD 결정자가 슈퍼키 |
| **조인 종속성(Join Dependency)** | R이 여러 투영으로 무손실 분해 가능 |
| **5NF (PJNF)** | 모든 비자명한 JD 구성 요소가 슈퍼키 |
| **DKNF** | 모든 제약조건이 도메인과 키로부터 따라옴 (이론적 이상) |
| **비정규화(Denormalization)** | 읽기 성능을 위한 의도적 중복성 |
| **스타 스키마(Star Schema)** | 팩트 테이블 + 평면 차원 테이블 (데이터 웨어하우징) |
| **스노플레이크 스키마(Snowflake Schema)** | 팩트 테이블 + 정규화된 차원 테이블 |

핵심 실무 요점: **기본적으로 정규화하고(BCNF), 성능이 요구할 때 의도적이고 규율 있게 비정규화하세요**. 모든 비정규화 결정과 그 유지보수 전략을 문서화하세요.

다음 레슨은 쿼리 처리(query processing)를 다룹니다 — 데이터베이스 엔진이 이러한 스키마에 대한 쿼리를 실제로 어떻게 실행하는지, 그리고 옵티마이저가 효율적인 실행 계획을 어떻게 선택하는지 다룹니다.

---

**이전**: [06_Normalization.md](./06_Normalization.md) | **다음**: [08_Query_Processing.md](./08_Query_Processing.md)
