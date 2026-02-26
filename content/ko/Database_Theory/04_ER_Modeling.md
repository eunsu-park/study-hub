# ER 모델링(ER Modeling)

**이전**: [관계 대수](./03_Relational_Algebra.md) | **다음**: [함수 종속성](./05_Functional_Dependencies.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 개념적 데이터베이스 설계의 목적을 설명하고, 전체 데이터베이스 설계 프로세스에서 ER 모델링이 차지하는 위치를 서술합니다.
2. 표준 ER 표기법을 사용하여 개체 타입(Entity Type), 속성(단순, 복합, 다가, 유도), 관계 타입(Relationship Type)을 식별하고 모델링합니다.
3. 관계 타입에 대한 카디널리티 비율(1:1, 1:N, M:N)과 참여 제약조건(전체 참여, 부분 참여)을 명시합니다.
4. 약한 개체(Weak Entity)와 식별 관계(Identifying Relationship)를 모델링하고 일반 개체 타입과 구분합니다.
5. 향상된 ER(EER) 구조인 특수화(Specialization), 일반화(Generalization), 집합화(Aggregation)를 적용하여 복잡한 데이터 구조를 표현합니다.
6. ER-관계형 매핑 알고리즘(ER-to-Relational Mapping Algorithm)을 실행하여 ER 다이어그램을 완전한 관계형 스키마로 변환합니다.

---

1976년 Peter Chen이 소개한 개체-관계(Entity-Relationship, ER) 모델은 개념적 데이터베이스 설계에 가장 널리 사용되는 접근법입니다. 특정 DBMS와 무관하게 높은 수준의 추상화로 데이터 구조를 표현하기 위한 그래픽 표기법을 제공합니다. 이 강의에서는 ER 모델, 향상된 버전(EER), 그리고 ER 다이어그램을 관계 스키마로 변환하는 체계적인 알고리즘을 다룹니다.

## 목차

1. [개념적 설계 개요](#1-개념적-설계-개요)
2. [개체 타입과 개체 집합](#2-개체-타입과-개체-집합)
3. [속성](#3-속성)
4. [관계 타입](#4-관계-타입)
5. [카디널리티 제약조건](#5-카디널리티-제약조건)
6. [참여 제약조건](#6-참여-제약조건)
7. [약한 개체](#7-약한-개체)
8. [향상된 ER (EER) 모델](#8-향상된-er-eer-모델)
9. [ER-관계형 매핑 알고리즘](#9-er-관계형-매핑-알고리즘)
10. [설계 사례 연구: 대학 데이터베이스](#10-설계-사례-연구-대학-데이터베이스)
11. [일반적인 함정과 모범 사례](#11-일반적인-함정과-모범-사례)
12. [연습 문제](#12-연습-문제)

---

## 1. 개념적 설계 개요

데이터베이스 설계는 요구사항에서 구현까지 구조화된 프로세스를 따릅니다:

```
┌──────────────────┐
│  요구사항         │  "어떤 데이터가 필요? 어떤 쿼리?"
│  분석            │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  개념적          │  ER 다이어그램 (DBMS 독립적)
│  설계            │  ← 이 강의
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  논리적          │  관계 스키마 (테이블, 키, 제약조건)
│  설계            │  ← ER-관계형 매핑
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  물리적          │  인덱스, 저장소, 파티셔닝, SQL DDL
│  설계            │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  구현 및         │  CREATE TABLE, INSERT, 저장 프로시저
│  튜닝            │
└──────────────────┘
```

### 개념적 설계를 하는 이유는?

- **의사소통**: ER 다이어그램은 비기술 이해관계자도 이해 가능
- **추상화**: 구현을 걱정하지 않고 데이터 구조에 집중
- **정확성**: SQL을 작성하기 전 초기에 설계 오류 포착
- **문서화**: 데이터 모델의 살아있는 청사진 역할

### ER 다이어그램 표기법

이 강의에서는 원래의 **Chen 표기법(Chen notation)** (교과서에서 가장 일반적)을 사용합니다:

```
┌──────────────────────────────────────────────────────────────┐
│  기호 범례                                                     │
│                                                              │
│  ┌─────────┐       개체 타입 (강한)                          │
│  │  NAME   │                                                │
│  └─────────┘                                                │
│                                                              │
│  ┌═════════┐       개체 타입 (약한)                          │
│  ║  NAME   ║                                                │
│  └═════════┘                                                │
│                                                              │
│  ◇  또는  ◇───       관계 타입                              │
│  <WORKS_FOR>                                                │
│                                                              │
│  (속성)        속성 (타원)                                    │
│  ((유도된))    유도 속성 (점선 타원)                           │
│  {다중값}      다중값 속성 (이중 타원)                         │
│                                                              │
│  ─── 단일선    부분 참여                                      │
│  ═══ 이중선    전체 참여                                      │
│                                                              │
│  1, N, M            카디널리티 표시                          │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. 개체 타입과 개체 집합

### 개체

**개체(Entity)**는 다른 객체와 구별 가능한 현실 세계의 "사물" 또는 객체입니다. 물리적(사람, 책) 또는 개념적(과정, 은행 계좌)일 수 있습니다.

### 개체 타입

**개체 타입(Entity Type)**은 동일한 속성을 가진 개체의 집합을 정의합니다. 클래스나 템플릿과 같습니다.

### 개체 집합

**개체 집합(Entity Set)** (또는 개체 인스턴스 집합)은 특정 시점의 특정 타입의 모든 개체 집합입니다. 클래스의 모든 객체 집합과 같습니다.

```
개체 타입:     STUDENT
               (구조 정의: sid, name, year, dept)

개체 집합:     현재 학생 개체 집합:
               {(S001, Alice, 3, CS), (S002, Bob, 2, CS), ...}

개체 (인스턴스): 단일 학생, 예: (S001, Alice, 3, CS)
```

### ER 다이어그램에서의 표기법

```
              ┌───────────┐
              │  STUDENT  │
              └───────────┘
             /   |    |    \
          (sid) (name)(year)(dept)
           [PK]
```

---

## 3. 속성

속성은 개체 타입의 특성을 설명합니다. 여러 종류가 있습니다:

### 단순(원자) 속성

단순 속성은 더 작은 구성 요소로 나눌 수 없습니다.

```
예:
  - student_id: "S001"
  - year: 3
  - gpa: 3.85
```

### 복합 속성

복합 속성은 더 작은 하위 속성으로 나눌 수 있습니다.

```
              (name)
             /      \
       (first_name) (last_name)

              (address)
            /    |      \
      (street) (city) (zip_code)
                        |
                   (state) (country)
```

### 다중값 속성

다중값 속성은 단일 개체에 대해 여러 값을 가질 수 있습니다.

```
  {phone_numbers}    학생은 0, 1 또는 여러 전화번호를 가질 수 있음.
  {skills}           직원은 여러 기술을 가질 수 있음.
  {email_addresses}  사람은 여러 이메일 주소를 가질 수 있음.

표기법: 이중 타원 또는 중괄호 {속성}
```

### 유도 속성

유도 속성의 값은 다른 속성에서 계산할 수 있습니다.

```
  ((age))            생년월일과 현재 날짜에서 유도
  ((total_credits))  등록 과목의 학점 합산에서 유도
  ((employee_count)) 부서의 직원 수에서 유도

표기법: 점선 타원 또는 이중 괄호 ((속성))
```

### 키 속성

키 속성은 개체 집합에서 각 개체를 고유하게 식별합니다.

```
  STUDENT의 경우: student_id (다이어그램에서 밑줄)
  COURSE의 경우: course_id
  EMPLOYEE의 경우: employee_id 또는 ssn

표기법: 밑줄 친 속성 이름
```

### 복합 키

단일 속성으로 개체를 고유하게 식별할 수 없을 때, 속성 조합이 키를 형성합니다.

```
예: ENROLLMENT은 (student_id, course_id, semester)로 식별될 수 있음
```

### NULL 값

속성이 다음과 같은 경우 NULL 값을 가질 수 있습니다:
- 값이 **적용 불가능**(집의 아파트 번호)
- 값이 **알 수 없음**(제공되지 않은 전화번호)

### 속성 요약

```
속성 유형:
                                              ┌──────────────┐
                            ┌────────────────►│   단순        │
                            │                 │  (원자)       │
         ┌──────────┐      │                 └──────────────┘
         │ 구조     ├──────┤
         └──────────┘      │                 ┌──────────────┐
                            └────────────────►│  복합         │
                                              │  (나눌 수 있음)│
                                              └──────────────┘

                                              ┌──────────────┐
                            ┌────────────────►│ 단일값        │
                            │                 └──────────────┘
         ┌──────────┐      │
         │ 카디널리티├────┤
         └──────────┘      │                 ┌──────────────┐
                            └────────────────►│ 다중값        │
                                              │ {attr}       │
                                              └──────────────┘

                                              ┌──────────────┐
                            ┌────────────────►│   저장됨      │
                            │                 └──────────────┘
         ┌──────────┐      │
         │  출처     ├──────┤
         └──────────┘      │                 ┌──────────────┐
                            └────────────────►│  유도됨       │
                                              │ ((attr))     │
                                              └──────────────┘
```

### 모든 속성 유형이 포함된 ER 다이어그램

```
                        ┌───────────┐
                        │ EMPLOYEE  │
                        └───────────┘
                       / |  |  |   \  \
                     /   |  |  |    \   \
                   /     |  |  |     \    \
            (emp_id)  (name) | (hire_date) {phone}  ((age))
             [PK]    / \     |               다중값   유도됨
                   /     \ (salary)
            (first) (last)
            복합
  emp_id:    단순, 키
  name:      복합 (first + last)
  salary:    단순, 단일값
  hire_date: 단순, 저장됨
  phone:     단순, 다중값
  age:       단순, 유도됨 (birth_date에서)
```

---

## 4. 관계 타입

**관계 타입(Relationship Type)**은 개체 타입 간의 연관을 정의합니다. **관계 인스턴스(Relationship Instance)**는 특정 개체 인스턴스 간의 연관입니다.

### 이항 관계

**이항 관계(Binary Relationship)**는 두 개체 타입을 포함합니다 (가장 일반적인 경우).

```
  ┌──────────┐          ┌──────────┐
  │ STUDENT  │──<ENROLLS>──│  COURSE  │
  └──────────┘          └──────────┘

  관계 인스턴스:
    (Alice, CS101), (Alice, CS301), (Bob, CS101), ...
```

### 삼항 관계

**삼항 관계(Ternary Relationship)**는 세 개체 타입을 포함합니다.

```
  ┌──────────┐
  │ SUPPLIER │
  └──────────┘
       │
       │
  ◇ SUPPLIES ◇
  /           \
  │             │
  ┌──────────┐  ┌──────────┐
  │  PART    │  │ PROJECT  │
  └──────────┘  └──────────┘

  관계 인스턴스: (Supplier1, PartA, ProjectX)
  의미: Supplier1이 PartA를 ProjectX에 공급

  참고: 삼항 관계는 정보 손실 없이 항상 세 개의
  이항 관계로 분해할 수 없음!
```

### 재귀(단항) 관계

**재귀 관계(Recursive Relationship)**는 개체 타입을 자신과 연관시킵니다.

```
  ┌──────────┐
  │ EMPLOYEE │
  └────┬─────┘
       │    │
       │    │
    (감독자)
       │    │
       ├────┘
    <SUPERVISES>

  관계 인스턴스: (Manager_Alice, Employee_Bob)
  의미: Alice가 Bob을 감독

  역할 이름이 중요:
    EMPLOYEE (감독자로) ──<SUPERVISES>── EMPLOYEE (피감독자로)
```

### 관계 속성

관계는 자체 속성을 가질 수 있습니다:

```
  ┌──────────┐                              ┌──────────┐
  │ STUDENT  │────<ENROLLS_IN>────│  COURSE  │
  └──────────┘      │                       └──────────┘
                  (grade)
                  (semester)

  성적과 학기는 관계에 속하며, 어느 개체에도 속하지 않음.
  학생은 특정 과목에 대한 성적을 가지며, 일반적으로 가지지 않음.
```

### 관계의 차수

관계 타입의 **차수(Degree)**는 참여하는 개체 타입의 수입니다.

```
차수 1: 단항 (재귀)     EMPLOYEE가 EMPLOYEE를 감독
차수 2: 이항            STUDENT가 COURSE에 등록
차수 3: 삼항            SUPPLIER가 PART를 PROJECT에 공급
차수 n: n항 (드묾)      일반적으로 이항으로 분해됨
```

---

## 5. 카디널리티 제약조건

카디널리티 제약조건은 개체가 참여할 수 있는 관계 인스턴스의 수를 지정합니다. 이항 관계의 경우, 세 가지 기본 비율은 1:1, 1:N, M:N입니다.

### 일대일 (1:1)

A의 각 개체는 최대 하나의 B 개체와 연관되며, 그 반대도 마찬가지입니다.

```
  ┌──────────┐    1         1    ┌──────────┐
  │ EMPLOYEE │────<MANAGES>────│ DEPARTMENT│
  └──────────┘                   └──────────┘

  각 직원은 최대 하나의 부서를 관리.
  각 부서는 최대 하나의 직원에 의해 관리됨.

  인스턴스:
    Alice  ────  CS Department
    Bob    ────  EE Department
    Carol  ────  (관리하는 부서 없음)
    Dave   ────  ME Department

  매핑:
    A:  Alice ───► CS
    B:  Bob   ───► EE
    D:  Dave  ───► ME
```

### 일대다 (1:N)

A의 각 개체는 B의 여러 개체와 연관될 수 있지만, B의 각 개체는 최대 하나의 A 개체와 연관됩니다.

```
  ┌──────────┐    1         N    ┌──────────┐
  │DEPARTMENT│────<HAS>────│ EMPLOYEE │
  └──────────┘                   └──────────┘

  부서는 여러 직원을 가짐.
  직원은 최대 하나의 부서에 속함.

  인스턴스:
    CS ────┬──── Alice
           ├──── Bob
           └──── Eve
    EE ─────── Carol
    ME ─────── Dave
```

### 다대다 (M:N)

A의 각 개체는 B의 여러 개체와 연관될 수 있고, B의 각 개체는 A의 여러 개체와 연관될 수 있습니다.

```
  ┌──────────┐    M         N    ┌──────────┐
  │ STUDENT  │────<ENROLLS>────│  COURSE  │
  └──────────┘                   └──────────┘

  학생은 여러 과목에 등록할 수 있음.
  과목은 여러 학생을 가질 수 있음.

  인스턴스:
    Alice ──┬── CS101
            ├── CS301
            └── MA101
    Bob   ──┬── CS101
            └── CS301
    Carol ──┬── EE201
            └── CS101
```

### ER 다이어그램에서의 카디널리티

두 가지 주요 관례가 있습니다:

**관례 1: Chen의 표기법 (선 위의 레이블)**

```
  ┌──────────┐    1    ┌──────────┐    N    ┌──────────┐
  │DEPARTMENT│────────<WORKS_IN>────────│ EMPLOYEE │
  └──────────┘                               └──────────┘
```

**관례 2: (최소,최대) 표기법 (더 정확함)**

```
  ┌──────────┐  (1,1)   ┌────────────┐  (1,N)  ┌──────────┐
  │ EMPLOYEE │──────────<WORKS_IN>──────────│DEPARTMENT│
  └──────────┘                                  └──────────┘

  읽기:
    직원은 (1,1) 부서에서 일함 = 정확히 하나의 부서
    부서는 (1,N) 직원을 가짐 = 한 명 이상의 직원
```

### 카디널리티 제약조건 요약

```
┌─────────────────────────────────────────────────────────────────┐
│                    카디널리티 비율                                 │
│                                                                 │
│  1:1  ──  각 A는 최대 1개의 B에 매핑; 각 B는 최대 1개의 A에      │
│           예: Employee가 Department를 관리                        │
│                                                                 │
│  1:N  ──  각 A는 여러 B에 매핑; 각 B는 최대 1개의 A에           │
│           예: Department가 Employee를 가짐                        │
│                                                                 │
│  M:N  ──  각 A는 여러 B에 매핑; 각 B는 여러 A에 매핑            │
│           예: Student가 Course에 등록                            │
│                                                                 │
│  (최소,최대) 표기법:                                              │
│    (0,1)  ──  선택적, 최대 하나                                  │
│    (1,1)  ──  필수적, 정확히 하나                                │
│    (0,N)  ──  선택적, 무제한 다수                                │
│    (1,N)  ──  필수적, 최소 하나                                  │
│    (3,5)  ──  최소 3, 최대 5 (특정 범위)                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. 참여 제약조건

참여 제약조건은 모든 개체가 관계에 참여해야 하는지 또는 참여가 선택적인지 지정합니다.

### 전체 참여 (필수적)

개체 집합의 모든 개체는 최소 하나의 관계 인스턴스에 참여해야 합니다. **이중선** (===)으로 표시합니다.

```
  ┌──────────┐           ┌──────────────┐          ┌──────────┐
  │ EMPLOYEE │═══════════<WORKS_IN>──────────│DEPARTMENT│
  └──────────┘                                      └──────────┘

  모든 직원은 어떤 부서에서 일해야 함.
  (부서 없이 직원이 존재할 수 없음.)
```

### 부분 참여 (선택적)

개체는 관계에 참여할 수도 있고 참여하지 않을 수도 있습니다. **단일선** (---)으로 표시합니다.

```
  ┌──────────┐           ┌──────────────┐          ┌──────────┐
  │ EMPLOYEE │───────────<MANAGES>═══════════│DEPARTMENT│
  └──────────┘                                      └──────────┘

  모든 직원이 부서를 관리하는 것은 아님 (EMPLOYEE 쪽에 부분).
  모든 부서는 누군가에 의해 관리되어야 함 (DEPARTMENT 쪽에 전체).
```

### 카디널리티와 참여 결합

```
예: 대학 ER 다이어그램 (단편)

  ┌──────────┐  (1,1)  ┌──────────────┐  (1,N)  ┌──────────┐
  │ EMPLOYEE │═════════<WORKS_IN>════════════│DEPARTMENT│
  └──────────┘                                    └──────────┘

  (최소,최대) 읽기:
    Employee 쪽: (1,1) → 전체 참여, 정확히 하나의 부서
    Department 쪽: (1,N) → 전체 참여, 최소 한 명의 직원

  ┌──────────┐  (0,N)  ┌──────────────┐  (0,N)  ┌──────────┐
  │ STUDENT  │─────────<ENROLLS_IN>──────────│  COURSE  │
  └──────────┘                                    └──────────┘

  (최소,최대) 읽기:
    Student 쪽: (0,N) → 부분 (학생이 등록하지 않을 수 있음), 여러 과목
    Course 쪽: (0,N) → 부분 (과목에 학생이 없을 수 있음), 여러 학생
```

### 존재 종속성

개체의 존재가 다른 개체와의 관계에 종속될 때, 해당 관계에 **전체 참여**를 합니다.

```
예:
  DEPENDENT (가족 구성원)는 EMPLOYEE 없이 존재할 수 없음.
  따라서 DEPENDENT는 HAS_DEPENDENT 관계에 전체 참여.

  ┌──────────┐           ┌─────────────────┐          ┌═══════════┐
  │ EMPLOYEE │───────────<HAS_DEPENDENT>════════════║ DEPENDENT ║
  └──────────┘                                        └═══════════┘

  DEPENDENT는 약한 개체이기도 함 (다음에 논의).
```

---

## 7. 약한 개체

**약한 개체 타입(Weak Entity Type)**은 자신의 속성만으로는 고유하게 식별될 수 없는 개체 타입입니다. 관련된 **소유자(Owner)** (또는 **식별(Identifying)**) 개체 타입에 종속됩니다.

### 약한 개체의 특성

```
1. 자체 기본 키가 없음
2. 동일한 소유자 개체와 관련된 약한 개체를 구별하는
   부분 키(PARTIAL KEY, 식별자)를 가짐
3. 식별 관계에 항상 전체 참여
4. 소유자 개체에 존재-종속
```

### 표기법

```
  ┌──────────┐              ┌═══════════════┐
  │  OWNER   │══<IDENTIFIES>══║ WEAK ENTITY ║
  │ (강한)   │                ║             ║
  └──────────┘                └═══════════════┘
                                   |
                             (partial_key)
                              [점선 밑줄]

  이중 사각형: 약한 개체 타입
  이중 다이아몬드: 식별 관계 타입
  점선 밑줄: 부분 키 (식별자)
```

### 예: Employee와 Dependent

```
  ┌──────────┐                ┌═════════════┐
  │ EMPLOYEE │══<HAS_DEPENDENT>══║ DEPENDENT  ║
  └──────────┘     1:N          └═════════════┘
       |                          |    |    |
    (emp_id)              (dep_name) (birth) (relationship)
     [PK]                 [부분 키]

  EMPLOYEE는 기본 키: emp_id를 가짐
  DEPENDENT는 부분 키: dep_name을 가짐

  DEPENDENT의 완전한 식별:
    (소유자의 emp_id, dep_name)

  예:
    직원 E001 (Alice)의 피부양자:
      (E001, "Tom") → Alice의 아들 Tom
      (E001, "Sue") → Alice의 딸 Sue

    직원 E002 (Bob)의 피부양자:
      (E002, "Tom") → Bob의 아들 Tom (E001의 Tom과 다른 사람!)

  소유자의 키 없이 "Tom"만으로는 모호함.
```

### 약한 개체 vs. 강한 개체

```
┌──────────────────────────────┬──────────────────────────────────┐
│       강한 개체               │         약한 개체                 │
├──────────────────────────────┼──────────────────────────────────┤
│ 자체 기본 키를 가짐           │ 부분 키만 가짐                    │
│ 독립적으로 존재 가능          │ 소유자에 존재 종속                │
│ 단일 사각형                  │ 이중 사각형                       │
│ 부분 참여 가능               │ 전체 참여 필요                    │
│ 예: EMPLOYEE, COURSE         │ 예: DEPENDENT, ROOM              │
└──────────────────────────────┴──────────────────────────────────┘

약한 개체의 더 많은 예:
  BUILDING (강한) → ROOM (약한): room_number가 부분 키
  INVOICE (강한) → LINE_ITEM (약한): line_number가 부분 키
  COURSE (강한) → SECTION (약한): section_number가 부분 키
```

---

## 8. 향상된 ER (EER) 모델

**향상된 ER(Enhanced ER, EER)** 모델은 기본 ER 모델을 객체 지향 모델링에서 차용한 추가 개념으로 확장합니다: 특수화, 일반화, 상속.

### 특수화

**특수화(Specialization)**는 구별되는 특성을 기반으로 개체 타입의 하위 클래스를 정의하는 하향식 프로세스입니다.

```
                    ┌──────────┐
                    │  PERSON  │
                    └────┬─────┘
                         │
                        / \
                       / d \        d = 분리(disjoint)
                      /     \       o = 중첩(overlapping)
                     /       \
              ┌──────────┐  ┌──────────┐
              │ STUDENT  │  │ EMPLOYEE │
              └──────────┘  └──────────┘

  PERSON은 상위클래스(SUPERCLASS)
  STUDENT와 EMPLOYEE는 하위클래스(SUBCLASS)
  d/o가 있는 원은 제약조건을 지정
```

### 일반화

**일반화(Generalization)**는 여러 개체 타입의 공통 특징을 상위 수준(일반) 개체 타입으로 추상화하는 상향식 프로세스입니다.

```
  일반화 예:

  CAR와 TRUCK이 모두 다음을 가지고 있음을 관찰:
    - vehicle_id, make, model, year, color

  따라서 일반화:

                    ┌──────────┐
                    │ VEHICLE  │  ← 일반화된 상위클래스
                    └────┬─────┘
                         │
                        / \
                       / d \
                      /     \
              ┌──────────┐  ┌──────────┐
              │   CAR    │  │  TRUCK   │
              └──────────┘  └──────────┘
              (num_doors)   (payload_capacity)
              (trunk_size)  (num_axles)
```

### 특수화/일반화 제약조건

두 개의 직교 제약조건이 특수화를 지배합니다:

**제약조건 1: 분리성(Disjointness)**

```
분리(d):          개체는 최대 하나의 하위클래스에 속할 수 있음
                  예: 차량은 CAR 또는 TRUCK이며, 둘 다는 아님

중첩(o):          개체는 여러 하위클래스에 속할 수 있음
                  예: 사람은 STUDENT이면서 EMPLOYEE일 수 있음
```

**제약조건 2: 완전성(Completeness)**

```
전체:    모든 상위클래스 개체는 최소 하나의 하위클래스에 속해야 함
         상위클래스에서 특수화 원으로의 이중선
         예: 모든 VEHICLE은 CAR 또는 TRUCK이어야 함

부분:    상위클래스 개체가 어떤 하위클래스에도 속하지 않을 수 있음
         상위클래스에서 특수화 원으로의 단일선
         예: PERSON은 STUDENT도 EMPLOYEE도 아닐 수 있음
```

### 네 가지 조합

```
┌──────────────────────────────────────────────────────────────────┐
│              특수화 제약조건 조합                                   │
│                                                                  │
│  {분리, 전체}:      모든 개체가 정확히 하나의 하위클래스에         │
│                     예: VEHICLE → CAR xor TRUCK                  │
│                                                                  │
│  {분리, 부분}:      개체가 최대 하나의 하위클래스에               │
│                     예: ACCOUNT → SAVINGS xor CHECKING           │
│                     (일부 계좌는 둘 다 아닐 수 있음)              │
│                                                                  │
│  {중첩, 전체}:      개체가 하나 이상의 하위클래스에               │
│                     예: PERSON → STUDENT and/or                  │
│                     EMPLOYEE (그러나 최소 하나)                   │
│                                                                  │
│  {중첩, 부분}:      개체가 0개 이상의 하위클래스에                │
│                     예: PERSON → STUDENT and/or                  │
│                     EMPLOYEE (둘 다 아닐 수 있음)                │
└──────────────────────────────────────────────────────────────────┘
```

### 속성 상속

하위클래스는 상위클래스의 모든 속성을 **상속**하며 하위클래스에 특정한 추가 속성을 가질 수 있습니다.

```
            ┌──────────────────┐
            │      PERSON      │
            │──────────────────│
            │ person_id (PK)   │
            │ name             │
            │ date_of_birth    │
            │ email            │
            └────────┬─────────┘
                     │
                    / \
                   / o \     (중첩, 부분)
                  /     \
    ┌────────────────┐  ┌────────────────┐
    │    STUDENT     │  │   EMPLOYEE     │
    │────────────────│  │────────────────│
    │ + student_id   │  │ + emp_id       │
    │ + year         │  │ + salary       │
    │ + gpa          │  │ + hire_date    │
    │ + major        │  │ + department   │
    └────────────────┘  └────────────────┘

    STUDENT는 상속: person_id, name, date_of_birth, email
    그리고 추가: student_id, year, gpa, major

    STUDENT이면서 EMPLOYEE인 PERSON은 모든 속성을 가짐.
```

### 다중 상속과 범주 (합집합 타입)

**범주(Category)** (또는 합집합 타입)는 여러 가능한 상위클래스를 가진 하위클래스입니다:

```
            ┌──────────┐      ┌──────────┐      ┌──────────┐
            │  PERSON  │      │ COMPANY  │      │   BANK   │
            └────┬─────┘      └────┬─────┘      └────┬─────┘
                 │                 │                  │
                 └─────────────────┼──────────────────┘
                                   │
                                  (U)    ← 합집합 / 범주
                                   │
                            ┌──────────────┐
                            │   OWNER      │  (차량의)
                            └──────────────┘

    OWNER는 PERSON, COMPANY, 또는 BANK 중 하나일 수 있음.
    (특수화와 반대로, 하위클래스가 하나의 상위클래스를 공유.)
```

---

## 9. ER-관계형 매핑 알고리즘

이 섹션에서는 ER/EER 다이어그램을 관계 스키마로 변환하는 체계적인 **7단계 알고리즘**을 제시합니다.

### 1단계: 강한 개체 타입 매핑

각 강한(일반) 개체 타입 E에 대해, E의 모든 단순 속성을 포함하는 관계 R을 생성합니다. R의 기본 키를 선택합니다.

```
ER:
  ┌──────────┐
  │ EMPLOYEE │
  └──────────┘
  (emp_id), (name), (salary), (hire_date)

관계형:
  EMPLOYEE(emp_id, first_name, last_name, salary, hire_date)
  PK: emp_id

규칙:
  - 복합 속성: 리프 구성 요소만 포함
    (name → first_name, last_name)
  - 유도 속성: 생략 (쿼리 시 계산)
  - 다중값 속성: 6단계에서 처리
```

### 2단계: 약한 개체 타입 매핑

소유자 개체 E를 가진 각 약한 개체 타입 W에 대해, 다음을 포함하는 관계 R을 생성:
- W의 모든 단순 속성
- E의 기본 키를 외래 키로
- R의 기본 키 = E의 PK + W의 부분 키

```
ER:
  ┌──────────┐    1:N    ┌═══════════┐
  │ EMPLOYEE │══════════════║ DEPENDENT ║
  └──────────┘              └═══════════┘
  (emp_id)             (dep_name), (birth_date), (relationship)

관계형:
  DEPENDENT(emp_id, dep_name, birth_date, relationship)
  PK: (emp_id, dep_name)
  FK: emp_id → EMPLOYEE(emp_id) ON DELETE CASCADE
```

### 3단계: 이항 1:1 관계 타입 매핑

세 가지 접근법 (참여 제약조건에 따라 선택):

```
접근법 A: 외래 키 접근법 (선호됨)
  한 개체의 PK를 다른 개체에 FK로 추가.
  전체 참여가 있는 쪽에 FK 추가를 선호.

  ER: EMPLOYEE (0,1) ──<MANAGES>── (1,1) DEPARTMENT

  관계형:
    EMPLOYEE(emp_id, name, salary)
    DEPARTMENT(dept_id, dept_name, mgr_emp_id, mgr_start_date)
                                   ^^^^^^^^^
                                   FK → EMPLOYEE(emp_id)
    (DEPARTMENT가 전체 참여를 가지므로 DEPARTMENT에 FK:
     모든 부서는 관리자를 가져야 함)


접근법 B: 관계 병합
  두 개체 타입을 하나의 관계로 병합.
  양쪽이 모두 전체 참여를 가질 때만 가능.


접근법 C: 교차 참조 (관계 관계)
  관계를 위한 별도의 관계 생성.
  관계가 많은 속성을 가질 때 유용.
```

### 4단계: 이항 1:N 관계 타입 매핑

"1-쪽" 개체의 PK를 "N-쪽" 개체에 FK로 추가합니다.

```
ER: DEPARTMENT (1) ──<HAS>── (N) EMPLOYEE

관계형:
  DEPARTMENT(dept_id, dept_name, budget)
  EMPLOYEE(emp_id, name, salary, dept_id)
                                 ^^^^^^^
                                 FK → DEPARTMENT(dept_id)

  관계 속성은 N-쪽 개체와 함께 감:
  HAS가 (start_date)를 가지면, EMPLOYEE에 추가.
```

### 5단계: 이항 M:N 관계 타입 매핑

새로운 **관계 관계(Relationship Relation)** R을 생성합니다. 포함:
- 양쪽 참여 개체 타입의 PK를 FK로
- 관계의 모든 속성
- R의 PK = 양쪽 FK의 조합

```
ER: STUDENT (M) ──<ENROLLS>── (N) COURSE
    속성: grade, semester

관계형:
  STUDENT(student_id, name, year)
  COURSE(course_id, title, credits)
  ENROLLMENT(student_id, course_id, semester, grade)
  PK: (student_id, course_id, semester)
  FK: student_id → STUDENT(student_id)
      course_id → COURSE(course_id)
```

### 6단계: 다중값 속성 매핑

각 다중값 속성에 대해 새로운 관계를 생성합니다. 포함:
- 다중값 속성
- 개체의 PK를 FK로
- PK = FK + 다중값 속성

```
ER: EMPLOYEE가 다중값 속성 {phone_numbers}를 가짐

관계형:
  EMPLOYEE(emp_id, name, salary)
  EMPLOYEE_PHONE(emp_id, phone_number)
  PK: (emp_id, phone_number)
  FK: emp_id → EMPLOYEE(emp_id) ON DELETE CASCADE
```

### 7단계: 특수화/일반화 매핑

네 가지 옵션이 존재합니다. 최선의 선택은 제약조건에 따라 다릅니다:

**옵션 A: 타입 식별자를 사용한 단일 테이블**

```
PERSON(person_id, name, dob, email, person_type,
       -- STUDENT 속성 (학생이 아니면 NULL)
       student_id, year, gpa, major,
       -- EMPLOYEE 속성 (직원이 아니면 NULL)
       emp_id, salary, hire_date, department)

person_type IN ('S', 'E', 'SE', 'N')  -- Student, Employee, Both, Neither

장점: 간단, 조인 필요 없음
단점: 많은 NULL, 하위클래스 제약조건 시행 어려움
최적: 하위클래스가 적고, 모든 타입에 걸친 쿼리가 많음
```

**옵션 B: 각 하위클래스에 대한 별도 테이블 (상위클래스 PK 상속)**

```
PERSON(person_id, name, dob, email)
STUDENT(person_id, student_id, year, gpa, major)
  FK: person_id → PERSON(person_id)
EMPLOYEE(person_id, emp_id, salary, hire_date, department)
  FK: person_id → PERSON(person_id)

장점: NULL 없음, 깔끔한 분리
단점: 전체 데이터를 얻으려면 조인 필요
최적: 하위클래스 특정 속성이 많고, 중첩 허용
```

**옵션 C: 별도 테이블 (각각 전체 속성)**

```
STUDENT(person_id, name, dob, email, student_id, year, gpa, major)
EMPLOYEE(person_id, name, dob, email, emp_id, salary, hire_date, dept)

장점: 하위클래스 쿼리에 조인 필요 없음
단점: 중복 (공유 속성 중복), 모든 사람에 걸친 쿼리 어려움,
      중첩은 데이터 중복 필요
최적: 분리, 전체 특수화
```

**옵션 D: 하이브리드 (상위클래스 + 특수화 테이블)**

```
사용 패턴에 따라 선택:
  - 자주 함께 쿼리됨 → 옵션 A
  - 대부분 별도로 쿼리됨 → 옵션 C
  - 유연성 필요 → 옵션 B
```

### 매핑 결정 표

```
┌──────────────────────────┬───────────────────────────────────┐
│ ER 구성                  │ 관계형 매핑                        │
├──────────────────────────┼───────────────────────────────────┤
│ 강한 개체                │ 새 관계, 자체 PK                   │
│ 약한 개체                │ 새 관계, 복합 PK                   │
│ 1:1 관계                 │ 한 개체에 FK (전체 쪽)             │
│ 1:N 관계                 │ N-쪽 개체에 FK                     │
│ M:N 관계                 │ 새 관계 (브리지 테이블)            │
│ 다중값 속성              │ 새 관계                            │
│ 복합 속성                │ 구성 요소로 평탄화                 │
│ 유도 속성                │ 생략 (쿼리 시 계산)                │
│ 특수화/일반화            │ 옵션 A, B, C, 또는 D               │
│ 삼항 관계                │ 3개의 FK를 가진 새 관계            │
│ 재귀 관계                │ 자신의 테이블에 FK (또는 브리지)   │
└──────────────────────────┴───────────────────────────────────┘
```

---

## 10. 설계 사례 연구: 대학 데이터베이스

대학 데이터베이스를 위한 완전한 ER 다이어그램을 설계하고 관계 스키마로 매핑해 봅시다.

### 요구사항

```
1. 대학은 각각 이름, 건물, 예산을 가진 DEPARTMENT를 가짐.
2. 각 부서는 한 명의 CHAIRPERSON (교수 구성원)을 가짐.
3. FACULTY 구성원은 ID, 이름, 직급, 급여를 가짐. 각각 하나의 부서에 속함.
4. STUDENT는 ID, 이름, 학년, GPA를 가짐. 각각 전공 부서를 가짐.
5. COURSE는 ID, 제목, 학점을 가지며 부서에 속함.
6. 교수 구성원이 과목을 가르침(TEACH). 각 개설은 특정 학기에 있음.
   과목은 다른 교수가 가르치는 여러 섹션을 가질 수 있음.
7. 학생이 과목 섹션에 등록(ENROLL)하고 성적을 받음.
8. 학생은 여러 전화번호와 이메일 주소를 가질 수 있음.
9. 교수는 학생을 조언할 수 있음(ADVISE) (학생은 한 명의 조언자를 가짐).
```

### ER 다이어그램 (ASCII)

```
 {phone}  {email}
    \      /
     \    /
  ┌──────────┐ (0,1)   (1,1) ┌──────────┐
  │ STUDENT  │═════<ADVISES>═════│ FACULTY  │
  └──────────┘                   └──────────┘
  (sid)(name)                    (fid)(name)
  (year)(gpa)                    (rank)(salary)
       |                              |
    (1,N)|                         (1,N)|
       |                              |
  <ENROLLED_IN>                   <TEACHES>
       |                              |
    (1,N)|                         (1,N)|
       |                              |
  ┌══════════════┐               ┌══════════════┐
  ║   SECTION    ║               ║   SECTION    ║
  ║ (sec_number) ║               ║              ║
  └══════════════┘               └══════════════┘
       |                              |
    (1,1)|                            |
       |                              |
  ┌──────────┐     (1,N)    (1,1)  ┌──────────┐  (1,1)  ┌──────────┐
  │  COURSE  │═════<OFFERED_BY>══════│DEPARTMENT│════<CHAIRS>═════│ FACULTY  │
  └──────────┘                       └──────────┘                  (이미
  (cid)(title)                       (did)(name)                   표시됨)
  (credits)                          (building)
                                     (budget)
```

참고: 위는 단순화되었습니다. 더 정확한 표현은 SECTION을 COURSE의 약한 개체로 모델링하고, TEACHES는 FACULTY를 SECTION에, ENROLLED_IN은 STUDENT를 SECTION에 연결합니다.

### 정제된 ER 설계

```
개체:
  DEPARTMENT(dept_id, dept_name, building, budget)              강한
  FACULTY(fac_id, name, rank, salary)                           강한
  STUDENT(stu_id, name, year, gpa, {phone}, {email})           강한 + MV
  COURSE(course_id, title, credits)                             강한
  SECTION(sec_number, semester, year)                           약한 (소유자: COURSE)

관계:
  WORKS_IN:      FACULTY (N,1) --- DEPARTMENT     (1:N)
  MAJOR_IN:      STUDENT (N,1) --- DEPARTMENT     (1:N)
  CHAIRS:        FACULTY (1,0..1) --- DEPARTMENT  (1:1)
  OFFERS:        DEPARTMENT (1,N) --- COURSE      (1:N)
  HAS_SECTION:   COURSE (1,N) === SECTION         (식별, 1:N)
  TEACHES:       FACULTY (1,N) --- SECTION        (섹션당 1:1)
  ENROLLED_IN:   STUDENT (M) --- SECTION (N)      (M:N, 속성: grade)
  ADVISES:       FACULTY (1) --- STUDENT (N)      (1:N)
```

### 관계 스키마 (매핑됨)

```sql
-- 1단계: 강한 개체
CREATE TABLE department (
    dept_id     CHAR(4)      PRIMARY KEY,
    dept_name   VARCHAR(50)  NOT NULL UNIQUE,
    building    VARCHAR(30),
    budget      NUMERIC(12,2) CHECK (budget >= 0)
);

CREATE TABLE faculty (
    fac_id      CHAR(5)      PRIMARY KEY,
    name        VARCHAR(50)  NOT NULL,
    rank        VARCHAR(20)  CHECK (rank IN
                  ('Lecturer','Assistant','Associate','Full')),
    salary      NUMERIC(10,2) CHECK (salary > 0),
    dept_id     CHAR(4)      NOT NULL,  -- 4단계: 1:N WORKS_IN
    FOREIGN KEY (dept_id) REFERENCES department(dept_id)
);

CREATE TABLE student (
    stu_id      CHAR(5)      PRIMARY KEY,
    name        VARCHAR(50)  NOT NULL,
    year        SMALLINT     CHECK (year BETWEEN 1 AND 4),
    gpa         NUMERIC(3,2) CHECK (gpa >= 0.0 AND gpa <= 4.0),
    major_id    CHAR(4),                -- 4단계: 1:N MAJOR_IN
    advisor_id  CHAR(5),                -- 4단계: 1:N ADVISES
    FOREIGN KEY (major_id) REFERENCES department(dept_id),
    FOREIGN KEY (advisor_id) REFERENCES faculty(fac_id)
);

CREATE TABLE course (
    course_id   CHAR(6)      PRIMARY KEY,
    title       VARCHAR(100) NOT NULL,
    credits     SMALLINT     NOT NULL CHECK (credits BETWEEN 1 AND 5),
    dept_id     CHAR(4)      NOT NULL,  -- 4단계: 1:N OFFERS
    FOREIGN KEY (dept_id) REFERENCES department(dept_id)
);

-- 2단계: 약한 개체 (COURSE로 식별되는 SECTION)
CREATE TABLE section (
    course_id   CHAR(6)      NOT NULL,
    sec_number  SMALLINT     NOT NULL,
    semester    VARCHAR(10)  NOT NULL,
    sec_year    SMALLINT     NOT NULL,
    fac_id      CHAR(5),                -- 4단계: 1:N TEACHES
    PRIMARY KEY (course_id, sec_number, semester, sec_year),
    FOREIGN KEY (course_id) REFERENCES course(course_id)
        ON DELETE CASCADE,
    FOREIGN KEY (fac_id) REFERENCES faculty(fac_id)
);

-- 3단계: 1:1 CHAIRS (department 쪽에 FK, 전체 참여)
ALTER TABLE department
    ADD COLUMN chair_fac_id CHAR(5),
    ADD CONSTRAINT fk_chair
        FOREIGN KEY (chair_fac_id) REFERENCES faculty(fac_id);

-- 5단계: M:N ENROLLED_IN
CREATE TABLE enrollment (
    stu_id      CHAR(5)      NOT NULL,
    course_id   CHAR(6)      NOT NULL,
    sec_number  SMALLINT     NOT NULL,
    semester    VARCHAR(10)  NOT NULL,
    sec_year    SMALLINT     NOT NULL,
    grade       VARCHAR(2),
    PRIMARY KEY (stu_id, course_id, sec_number, semester, sec_year),
    FOREIGN KEY (stu_id) REFERENCES student(stu_id),
    FOREIGN KEY (course_id, sec_number, semester, sec_year)
        REFERENCES section(course_id, sec_number, semester, sec_year)
);

-- 6단계: 다중값 속성
CREATE TABLE student_phone (
    stu_id      CHAR(5)      NOT NULL,
    phone       VARCHAR(20)  NOT NULL,
    PRIMARY KEY (stu_id, phone),
    FOREIGN KEY (stu_id) REFERENCES student(stu_id) ON DELETE CASCADE
);

CREATE TABLE student_email (
    stu_id      CHAR(5)      NOT NULL,
    email       VARCHAR(100) NOT NULL,
    PRIMARY KEY (stu_id, email),
    FOREIGN KEY (stu_id) REFERENCES student(stu_id) ON DELETE CASCADE
);
```

### 스키마 다이어그램 요약

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  DEPARTMENT  │     │   FACULTY    │     │   STUDENT    │
├──────────────┤     ├──────────────┤     ├──────────────┤
│PK dept_id    │◄────│PK fac_id     │◄────│PK stu_id     │
│   dept_name  │  FK │   name       │  FK │   name       │
│   building   │dept │   rank       │adv  │   year       │
│   budget     │     │   salary     │     │   gpa        │
│FK chair_fac  │─────│FK dept_id ───┘     │FK major_id──►│ DEPARTMENT
└──────────────┘     └──────────────┘     │FK advisor_id─│►FACULTY
       │                    │             └──────────────┘
       │                    │                    │
       │              ┌─────┘                    │
       │              │                          │
  ┌──────────┐  ┌─────┴──────┐  ┌───────────────┴──┐
  │  COURSE  │  │  SECTION   │  │   ENROLLMENT     │
  ├──────────┤  ├────────────┤  ├──────────────────┤
  │PK crs_id │◄─│PK crs_id   │◄─│PK stu_id         │
  │   title  │  │PK sec_num  │  │PK crs_id         │
  │   credits│  │PK semester │  │PK sec_num        │
  │FK dept_id│  │PK sec_year │  │PK semester       │
  └──────────┘  │FK fac_id   │  │PK sec_year       │
                └────────────┘  │   grade           │
                                └──────────────────┘

  ┌─────────────────┐  ┌─────────────────┐
  │ STUDENT_PHONE   │  │ STUDENT_EMAIL   │
  ├─────────────────┤  ├─────────────────┤
  │PK,FK stu_id     │  │PK,FK stu_id     │
  │PK    phone      │  │PK    email      │
  └─────────────────┘  └─────────────────┘
```

---

## 11. 일반적인 함정과 모범 사례

### 함정 1: 팬 트랩

**팬 트랩(Fan Trap)**은 여러 1:N 관계를 통한 경로가 팬 아웃되어 모호한 연관을 만들 때 발생합니다.

```
문제:
  DEPARTMENT ─1:N─ EMPLOYEE
  DEPARTMENT ─1:N─ PROJECT

  쿼리: "어떤 직원이 어떤 프로젝트에서 일하는가?"
  경로는 DEPARTMENT → EMPLOYEE와 DEPARTMENT → PROJECT로 가지만
  EMPLOYEE와 PROJECT 간에 직접 링크가 없음!

해결책: EMPLOYEE와 PROJECT 간에 직접 WORKS_ON 관계 추가.
```

### 함정 2: 캐즘 트랩

**캐즘 트랩(Chasm Trap)**은 관련되어야 할 개체 타입 간에 경로가 존재하지 않을 때 발생합니다.

```
문제:
  DEPARTMENT ─1:N─ EMPLOYEE (부분)
  EMPLOYEE ─1:N─ PROJECT

  일부 부서에 직원이 없으면 DEPARTMENT에서 PROJECT로의 경로에
  "캐즘(chasm)" (간격)이 있음.

해결책: DEPARTMENT와 PROJECT 간에 직접 HAS_PROJECT 관계 추가.
```

### 함정 3: 다중값 속성의 과도한 사용

```
나쁨:
  PERSON with {address}   ← 주소가 자체 속성이 필요하면
                             (street, city, state, zip), 잘못됨

더 좋음:
  PERSON ─1:N─ ADDRESS
  ADDRESS(address_id, street, city, state, zip_code)
```

### 함정 4: 관계 속성 누락

```
나쁨:
  STUDENT가 grade 속성을 가짐    ← 어떤 과목의 성적?

더 좋음:
  STUDENT ─M:N─ COURSE with 관계 속성: grade
```

### 모범 사례

```
1. 개체부터 시작, 그 다음 관계, 그 다음 속성
2. 모든 개체는 키 속성을 가져야 함
3. 중복 관계 피하기 (다른 것에서 유도 가능)
4. 약한 개체는 진정으로 필요할 때만 사용
5. 가능하면 삼항보다 이항 관계 선호
6. 모든 가정과 제약조건 문서화
7. 구체적인 예제를 사용하여 이해관계자와 검증
8. 개체를 단수 명사로 명명 (STUDENTS가 아닌 STUDENT)
9. 관계를 동사로 명명 (ENROLLS_IN, TEACHES)
10. 정확한 제약조건을 위해 (min,max) 표기법 사용
```

---

## 12. 연습 문제

### 기본 개념

**연습 문제 4.1**: 다음 각각에 대해 개체 타입, 관계 타입, 또는 속성으로 모델링해야 하는지 식별하세요. 답을 정당화하세요.

- (a) 직원 이름
- (b) 부서
- (c) 결혼 (두 사람 간)
- (d) 학생 GPA
- (e) 책 ISBN
- (f) 과정 등록
- (g) 직원 기술 (직원이 많은 기술을 가질 수 있다고 가정)
- (h) 프로젝트 마감일

**연습 문제 4.2**: 다음 속성을 분류하세요:

| 속성 | 단순/복합 | 단일/다중 | 저장/유도 | 키? |
|-----------|-------------------|--------------|----------------|------|
| SSN | | | | |
| 전체 이름 (first + middle + last) | | | | |
| 전화번호 (여러) | | | | |
| 나이 (생년월일 주어짐) | | | | |
| 이메일 주소 | | | | |
| 주소 (street, city, state, zip) | | | | |

### ER 설계

**연습 문제 4.3**: 다음 요구사항을 가진 병원 시스템에 대한 ER 다이어그램 그리기:
- 환자는 ID, 이름, 생년월일, 혈액형을 가짐
- 의사는 ID, 이름, 전문 분야, 전화번호를 가짐
- 각 환자는 주치의에게 배정됨
- 의사는 환자에게 약을 처방할 수 있음 (날짜와 용량 기록)
- 약은 코드, 이름, 제조사를 가짐
- 환자는 여러 알레르기를 가질 수 있음

카디널리티와 참여 제약조건을 지정하세요.

**연습 문제 4.4**: 온라인 학습 플랫폼에 대한 ER 다이어그램 그리기:
- 강사가 과정을 생성. 과정은 제목, 설명, 가격을 가짐.
- 과정은 특정 순서로 여러 레슨을 포함.
- 학생은 과정에 등록하고 레슨별 진행률 추적 (완료 상태, 소요 시간).
- 학생은 과정을 평가하고 리뷰할 수 있음 (1-5 별, 텍스트).
- 각 레슨 끝에 퀴즈가 있음. 학생은 퀴즈를 시도하고 점수를 받음.

모든 개체 타입, 관계 타입, 속성, 키, 제약조건을 식별하세요.

### 카디널리티와 참여

**연습 문제 4.5**: 다음 각 시나리오에 대해 카디널리티 비율 (1:1, 1:N, M:N)과 참여 제약조건 (전체/부분) 결정:

- (a) 국가는 하나의 수도를 가짐; 수도는 하나의 국가에 속함
- (b) 학생은 하나의 기숙사 방에 거주; 방은 여러 학생을 수용 가능
- (c) 저자는 많은 책을 쓸 수 있음; 책은 많은 저자를 가질 수 있음
- (d) 직원은 여러 프로젝트에서 일함; 프로젝트는 여러 직원을 가짐
- (e) 사람은 하나의 여권을 가짐; 여권은 한 사람에게 속함 (모든 사람이 여권을 가지는 것은 아님)

### 약한 개체

**연습 문제 4.6**: 각 쌍에 대해 어느 것(있다면)이 약한 개체여야 하는지 결정:

- (a) Building과 Room
- (b) Invoice와 LineItem
- (c) Student와 Course
- (d) Bank와 Branch
- (e) Order와 OrderItem

각 약한 개체에 대해 부분 키와 식별 관계를 식별하세요.

### EER

**연습 문제 4.7**: 다음에 대한 EER 다이어그램 설계:

회사에 **직원**이 있습니다. 직원은 **관리자**, **엔지니어**, **비서**로 특수화됩니다. 엔지니어는 **소프트웨어 엔지니어**와 **하드웨어 엔지니어**로 더 특수화될 수 있습니다.

- 특수화가 분리인지 중첩인지 결정
- 전체인지 부분인지 결정
- 각 하위클래스에 대해 최소 두 개의 특정 속성 추가
- 완전한 EER 다이어그램 그리기

### ER-관계형 매핑

**연습 문제 4.8**: 다음 ER 다이어그램이 주어졌을 때, 7단계 매핑 알고리즘을 적용하여 SQL DDL과 함께 완전한 관계 스키마 생성:

```
                     {skill}
                       |
  ┌──────────┐  1:N  ┌──────────────┐  M:N   ┌──────────┐
  │DEPARTMENT│═══════<WORKS_IN>═══════│ EMPLOYEE │════<WORKS_ON>════│ PROJECT  │
  └──────────┘        └──────────────┘  |hours|  └──────────┘
  (dept_id)          (emp_id)(name)              (proj_id)(name)
  (name)             (salary)                    (budget)
  (budget)           (birth_date)                (location)
                          |
                     1:N  |
                          |
                ┌═══════════════┐
                ║  DEPENDENT   ║
                ║(dep_name)    ║
                ║(birth_date)  ║
                ║(relationship)║
                └═══════════════┘
```

포함: 모든 테이블, PK, FK, 도메인 제약조건, ON DELETE 동작.

**연습 문제 4.9**: 다음 특수화 계층을 세 가지 접근법 각각을 사용하여 관계 스키마로 매핑 (옵션 A: 단일 테이블, 옵션 B: 상위클래스 + 하위클래스 테이블, 옵션 C: 별도 테이블). 트레이드오프를 논의하세요.

```
            VEHICLE
           (vin, make, model, year, color)
              |
            /   \
          d,t
          /       \
       CAR          TRUCK
    (num_doors,   (payload_cap,
     trunk_vol)    num_axles,
                   cab_type)
```

### 설계 도전

**연습 문제 4.10**: 지역 도서관에 데이터베이스 시스템이 필요합니다. 완전한 개념 스키마 (ER/EER 다이어그램)를 설계하고 관계 스키마로 매핑하세요. 요구사항:

1. 도서관은 각각 이름, 주소, 전화번호를 가진 여러 **지점**을 가짐
2. **책**은 ISBN으로 식별되며 제목, 출판 연도, 판을 가짐
3. 각 책은 한 명 이상의 **저자**를 가짐
4. 책은 하나 이상의 **범주** (Fiction, Science, History 등)에 속함
5. 각 지점은 각 책의 여러 **사본**을 유지관리. 각 사본은 사본 번호와 상태를 가짐
6. **회원**은 카드 번호, 이름, 주소, 전화번호를 가짐. 회원은 특정 지점에 등록
7. 회원은 사본을 **대출**할 수 있음. 각 대출은 대출 날짜, 반납 예정일, 반납 날짜를 기록
8. 회원은 지점에서 책을 **예약**할 수 있음 (특정 사본이 아님)
9. **직원**은 지점에서 일함. **사서**와 **보조원**이 있음 (분리 특수화)
10. 각 지점은 **지점 관리자** 역할을 하는 한 명의 사서를 가짐

제출물:
- 모든 제약조건을 포함한 완전한 ER/EER 다이어그램
- 관계 스키마 (7단계 모두 적용)
- 최소 5개의 주요 테이블에 대한 SQL DDL
- 설계가 도서관의 요구를 지원함을 보여주는 세 가지 샘플 쿼리

---

**이전**: [관계 대수](./03_Relational_Algebra.md) | **다음**: [함수 종속성](./05_Functional_Dependencies.md)
