# 관계형 모델(The Relational Model)

**이전**: [데이터베이스 시스템 소개](./01_Introduction_to_Database_Systems.md) | **다음**: [관계 대수](./03_Relational_Algebra.md)

---

Edgar F. Codd가 1970년에 도입한 관계형 모델은 가장 널리 사용되는 데이터 모델이며 SQL 데이터베이스의 이론적 기반입니다. 이 모델은 데이터를 수학적 관계(테이블)로 표현하고 구조 정의, 무결성 강제, 데이터 조작을 위한 엄격한 프레임워크를 제공합니다. 이 레슨에서는 모든 데이터베이스 실무자가 이해해야 하는 형식적 정의, 핵심 개념, 무결성 제약조건, NULL의 미묘한 의미를 다룹니다.

## 목차(Table of Contents)

1. [역사적 맥락: Codd의 비전](#1-historical-context-codds-vision)
2. [Codd의 12가지 규칙](#2-codds-12-rules)
3. [형식적 정의](#3-formal-definitions)
4. [키](#4-keys)
5. [무결성 제약조건](#5-integrity-constraints)
6. [관계형 스키마 표기법](#6-relational-schema-notation)
7. [NULL 의미와 3치 논리](#7-null-semantics-and-three-valued-logic)
8. [실제 관계형 모델](#8-relational-model-in-practice)
9. [연습문제](#9-exercises)

---

## 1. 역사적 맥락: Codd의 비전

1970년, IBM San Jose Research Laboratory의 연구원 Edgar F. Codd는 Communications of the ACM에 "A Relational Model of Data for Large Shared Data Banks"를 발표했습니다. 이 논문은 데이터 관리에 대한 우리의 사고방식을 근본적으로 변화시켰습니다.

### Codd가 해결한 문제

관계형 모델 이전에는 두 가지 지배적인 접근법이 있었습니다:

```
계층 모델(IMS):              네트워크 모델(CODASYL):

     ROOT                          STUDENT ──── COURSE
    /    \                            │    \  /     │
  CHILD1  CHILD2                      │     \/      │
    |                                 │     /\      │
  CHILD3                              │    /  \     │
                                      ▼   /    \    ▼
                                   ADVISOR    ENROLLMENT

둘 다 다음을 요구했습니다:
  - 포인터 체인을 통한 탐색 프로그램
  - 물리적 데이터 레이아웃에 대한 지식
  - 구조 변경 시 코드 재작성
```

### Codd의 핵심 통찰

Codd는 데이터를 **관계(relations)**(수학적 개념)로 표현할 것을 제안했으며, 이는 테이블에 해당합니다:

```
대신:                             사용:
  "STUDENT 레코드에서              "SELECT student_name
   등록 체인을 따라                 FROM students
   과목 이름 찾기"                  JOIN courses
                                     ON students.id = enrollments.student_id
                                     WHERE course_name = 'Database Theory'"

  방법(HOW)을 탐색                  무엇(WHAT)을 선언
  (절차적)                          (선언적)
```

이러한 **절차적 탐색**에서 **선언적 명세**로의 전환은 혁명적이었습니다.

---

## 2. Codd의 12가지 규칙

1985년, Codd는 데이터베이스 관리 시스템이 진정으로 "관계형"이기 위해 필요한 것을 정의하는 12개의 규칙 (실제로는 0-12로 번호가 매겨진 13개)을 발표했습니다. 이러한 규칙은 RDBMS 구현을 평가하는 벤치마크 역할을 합니다.

### 규칙 0: 기초 규칙

> 관계형 DBMS는 관계형 기능만을 사용하여 저장된 데이터를 관리해야 합니다.

### 12가지 규칙

| # | 규칙 이름 | 설명 |
|---|-----------|-------------|
| 1 | **정보 규칙(Information Rule)** | 모든 데이터는 테이블(관계)의 값으로 표현됩니다. 여기에는 메타데이터(시스템 카탈로그)도 포함됩니다. |
| 2 | **보장된 접근 규칙(Guaranteed Access Rule)** | 모든 데이터는 테이블 이름, 열 이름, 기본 키 값을 지정하여 접근 가능합니다. 포인터가 필요하지 않습니다. |
| 3 | **NULL의 체계적 처리(Systematic Treatment of NULL)** | NULL은 누락되거나 적용 불가능한 데이터를 나타내며, 빈 문자열이나 0과 구별됩니다. 모든 데이터 타입에 대해 지원됩니다. |
| 4 | **동적 온라인 카탈로그(Dynamic Online Catalog)** | 데이터베이스 설명(메타데이터)은 테이블에 저장되고 사용자 데이터와 동일한 관계형 언어를 사용하여 쿼리 가능합니다. |
| 5 | **포괄적 데이터 부속어(Comprehensive Data Sublanguage)** | 최소한 하나의 언어가 데이터 정의, 조작, 무결성 제약조건, 권한 부여, 트랜잭션을 지원해야 합니다. (SQL이 이를 충족합니다.) |
| 6 | **뷰 업데이트 규칙(View Updating Rule)** | 이론적으로 업데이트 가능한 모든 뷰는 시스템에 의해 업데이트 가능해야 합니다. |
| 7 | **고수준 삽입, 업데이트, 삭제(High-Level Insert, Update, Delete)** | 집합 단위 연산 (행 단위만이 아님). 단일 문에서 여러 행을 삽입, 업데이트 또는 삭제할 수 있습니다. |
| 8 | **물리적 데이터 독립성(Physical Data Independence)** | 물리적 저장이나 접근 방법의 변경이 애플리케이션에 영향을 주지 않습니다. |
| 9 | **논리적 데이터 독립성(Logical Data Independence)** | 정보를 보존하는 개념 스키마의 변경이 애플리케이션에 영향을 주지 않습니다. |
| 10 | **무결성 독립성(Integrity Independence)** | 무결성 제약조건은 카탈로그에 정의되며 (애플리케이션 프로그램이 아님) 애플리케이션에 영향을 주지 않고 변경될 수 있습니다. |
| 11 | **분산 독립성(Distribution Independence)** | 데이터가 중앙집중식이든 분산되어 있든 애플리케이션은 동일하게 작동합니다. |
| 12 | **비전복 규칙(Nonsubversion Rule)** | 시스템이 저수준(레코드 단위) 인터페이스를 제공하는 경우, 관계형 무결성 제약조건을 우회하는 데 사용될 수 없습니다. |

### 실제 평가

어떤 상용 RDBMS도 12가지 규칙을 완전히 만족하지 않습니다. 다음은 대략적인 평가입니다:

```
규칙 준수 (대략):

                    PostgreSQL  MySQL   Oracle  SQLite
규칙 1 (Info)          ✓         ✓       ✓       ✓
규칙 2 (Access)        ✓         ✓       ✓       ✓
규칙 3 (NULL)          ✓         ✓       ✓       ~
규칙 4 (Catalog)       ✓         ✓       ✓       ~
규칙 5 (Language)      ✓         ✓       ✓       ✓
규칙 6 (View Update)   ~         ~       ~       ~
규칙 7 (Set Ops)       ✓         ✓       ✓       ✓
규칙 8 (Physical DI)   ✓         ✓       ✓       ~
규칙 9 (Logical DI)    ~         ~       ~       ~
규칙 10 (Integrity)    ✓         ~       ✓       ~
규칙 11 (Distribution) ~         ~       ~       ✗
규칙 12 (Nonsub.)      ✓         ~       ✓       ~

✓ = 대체로 준수  ~ = 부분적  ✗ = 미지원
```

---

## 3. 형식적 정의

관계형 모델은 집합론과 1차 술어 논리에 기반을 두고 있습니다. 형식적 정의를 이해하는 것은 데이터베이스 설계와 쿼리 작성에 필수적입니다.

### 도메인(Domain)

**도메인(Domain)** D는 원자적(나눌 수 없는) 값의 집합입니다. 각 도메인은 논리적 정의(값이 의미하는 것)와 데이터 타입을 가집니다.

```
도메인의 예:

D_StudentID  = {S001, S002, S003, ..., S999}
D_Name       = 길이 <= 50인 모든 문자열의 집합
D_GPA        = {x in R | 0.0 <= x <= 4.0}
D_Grade      = {A+, A, A-, B+, B, B-, C+, C, C-, D+, D, F}
D_Credits    = {1, 2, 3, 4, 5}
D_Boolean    = {TRUE, FALSE}
D_Date       = 모든 유효한 달력 날짜의 집합
```

### 관계 스키마(Relation Schema)

**관계 스키마(Relation schema)** R(A1, A2, ..., An)는 다음으로 구성됩니다:
- 관계 이름 R
- 속성 목록 A1, A2, ..., An
- 각 속성 Ai는 도메인 dom(Ai)을 가짐

```
형식 표기법:

  R(A1: D1, A2: D2, ..., An: Dn)

예:

  STUDENT(student_id: D_StudentID,
          name: D_Name,
          gpa: D_GPA)
```

관계의 **차수(degree)** (또는 원수, arity)는 속성 개수 n입니다.

### 관계(Relation, Instance)

관계 스키마 R(A1, A2, ..., An)의 **관계(relation)** r은 n-튜플의 집합입니다:

```
r(R) ⊆ dom(A1) × dom(A2) × ... × dom(An)
```

각 n-튜플 t는 순서가 있는 값의 목록입니다:

```
t = <v1, v2, ..., vn>   where vi ∈ dom(Ai) ∪ {NULL}
```

### 형식적 예

```
스키마:
  STUDENT(student_id: D_StudentID, name: D_Name, year: D_Year, gpa: D_GPA)

  where D_StudentID = {S001..S999}
        D_Name = 길이 <= 50인 문자열
        D_Year = {1, 2, 3, 4}
        D_GPA = {x ∈ R | 0.0 ≤ x ≤ 4.0}

인스턴스 (시간 t에서):
  r(STUDENT) = {
    <S001, "Alice Kim",  3, 3.85>,
    <S002, "Bob Park",   2, 3.42>,
    <S003, "Carol Lee",  4, 3.91>,
    <S004, "Dave Choi",  1, NULL>
  }

  차수 = 4
  카디널리티(튜플 개수) = 4
```

### 관계의 속성

| 속성 | 설명 |
|----------|-------------|
| **중복 튜플 없음(No duplicate tuples)** | 관계는 집합이므로, 두 개의 튜플이 동일할 수 없음 |
| **튜플은 순서가 없음(Tuples are unordered)** | 행의 고유한 순서가 없음 |
| **속성은 순서가 없음(Attributes are unordered)** | 열의 순서는 중요하지 않음 (SQL 구현은 일반적으로 선언 순서를 보존) |
| **속성 값은 원자적(Attribute values are atomic)** | 각 셀은 단일하고 나눌 수 없는 값을 포함 (제1정규형) |
| **각 속성은 고유한 이름을 가짐(Each attribute has a distinct name)** | 동일한 관계에서 두 개의 속성이 이름을 공유할 수 없음 |

### 관계 vs. 테이블

종종 서로 교환하여 사용되지만, 미묘한 차이가 있습니다:

```
┌─────────────────────────────┬────────────────────────────────┐
│     관계 (이론)              │        테이블 (SQL)             │
├─────────────────────────────┼────────────────────────────────┤
│ 튜플의 집합 (중복 없음)       │ 행의 다중집합 (중복 허용)        │
│ 튜플은 순서가 없음           │ 행은 물리적 순서를 가질 수 있음   │
│ 속성은 순서가 없음           │ 열은 선언된 순서를 가짐          │
│ 모든 값은 원자적             │ 일부에서 배열, JSON 허용         │
│ 명명된 관점만               │ 위치 접근 가능                   │
│ 도메인 기반 타이핑           │ SQL 데이터 타입                  │
└─────────────────────────────┴────────────────────────────────┘

SQL 테이블은 엄격하게 관계가 아닙니다. 왜냐하면:
  1. SQL은 중복 행을 허용합니다 (제약되지 않는 한)
  2. SQL은 열 순서를 보존합니다
  3. SQL은 추가 기능을 가집니다 (auto-increment 등)
```

### SQL의 속성 타입 및 도메인

```sql
-- 수학적 도메인을 SQL 타입으로 매핑

CREATE DOMAIN grade_domain AS VARCHAR(2)
    CHECK (VALUE IN ('A+','A','A-','B+','B','B-','C+','C','C-','D+','D','F'));

CREATE DOMAIN gpa_domain AS NUMERIC(3,2)
    CHECK (VALUE >= 0.0 AND VALUE <= 4.0);

CREATE TABLE student (
    student_id  CHAR(4)        NOT NULL,  -- D_StudentID
    name        VARCHAR(50)    NOT NULL,  -- D_Name
    year        SMALLINT       NOT NULL,  -- D_Year
    gpa         gpa_domain,               -- D_GPA (도메인 사용)
    CONSTRAINT pk_student PRIMARY KEY (student_id),
    CONSTRAINT ck_year CHECK (year BETWEEN 1 AND 4)
);
```

참고: `CREATE DOMAIN` 구문은 PostgreSQL에서 지원되지만 모든 RDBMS에서 지원되지는 않습니다.

---

## 4. 키(Keys)

키는 관계형 모델의 기본입니다. 튜플을 고유하게 식별하고 관계 간 관계를 설정하는 메커니즘을 제공합니다.

### 슈퍼키(Superkey)

관계 스키마 R의 **슈퍼키(superkey)**는 속성 집합 SK ⊆ R로, 유효한 관계 인스턴스 r(R)의 어떤 두 튜플도 SK에 대해 동일한 값을 가지지 않습니다.

형식적으로: r(R)의 임의의 두 개의 서로 다른 튜플 t1, t2에 대해:

```
t1[SK] ≠ t2[SK]
```

```
STUDENT(student_id, name, year, gpa)

STUDENT의 슈퍼키:
  {student_id}                          ← 최소
  {student_id, name}                    ← 최소가 아님 (student_id만으로 충분)
  {student_id, name, year}              ← 최소가 아님
  {student_id, name, year, gpa}         ← 자명한 슈퍼키 (모든 속성)
  {name, year}                          ← 고유한 경우 슈퍼키일 수 있음

참고: 모든 속성의 집합은 항상 슈퍼키입니다 (자명한 슈퍼키).
```

### 후보키(Candidate Key)

**후보키(candidate key)**는 **최소 슈퍼키(minimal superkey)**입니다 -- 고유성 속성을 잃지 않고 어떤 속성도 제거할 수 없는 슈퍼키입니다.

```
형식 정의:
  R의 슈퍼키 K는 K의 모든 진부분집합 K' ⊂ K에 대해,
  K'가 R의 슈퍼키가 아닐 때 후보키입니다.

예:
  ENROLLMENT(student_id, course_id, semester, grade)

  슈퍼키:
    {student_id, course_id, semester}             ← 최소 → 후보키
    {student_id, course_id, semester, grade}      ← 최소가 아님

  (name, year)가 STUDENT에 대해 고유하다고 가정하면:
    {name, year}는 또 다른 후보키입니다

  관계는 여러 개의 후보키를 가질 수 있습니다.
```

### 기본키(Primary Key)

**기본키(primary key)**는 데이터베이스 설계자가 튜플의 주요 식별자로 선택한 후보키입니다. 스키마 표기법에서 밑줄로 표시됩니다.

```
규약:
  - 기본키 속성에 밑줄 표시
  - 기본키 값은 NULL일 수 없음
  - 각 관계는 정확히 하나의 기본키를 가짐
  - 다른 후보키는 대체키(ALTERNATE KEYS)가 됨

  STUDENT(student_id, name, year, gpa)
         ^^^^^^^^^^
         기본키

  {student_id}와 {name, year} 둘 다 후보키인 경우:
    기본키: {student_id}       (설계자가 선택)
    대체키: {name, year}       (다른 후보키)
```

### 외래키(Foreign Key)

**외래키(foreign key)**는 한 관계에서 다른 (또는 동일한) 관계의 기본키를 참조하는 속성 집합입니다.

```
형식 정의:
  관계 R1의 속성 집합 FK는 다음 조건을 만족할 때
  관계 R2를 참조하는 외래키입니다:
    1. FK의 속성이 R2의 기본키 PK와 동일한 도메인을 가짐
    2. r(R1)의 튜플 t1에서 FK의 값이:
       (a) r(R2)의 어떤 튜플 t2의 PK 값으로 나타나거나, 또는
       (b) NULL (허용되는 경우)

예:
  STUDENT(student_id, name, year, gpa)
  COURSE(course_id, title, credits)
  ENROLLMENT(student_id, course_id, semester, grade)
             ^^^^^^^^^^  ^^^^^^^^^
             FK → STUDENT  FK → COURSE

다이어그램으로:

  STUDENT                    ENROLLMENT                    COURSE
  ┌───────────┐             ┌───────────────┐             ┌───────────┐
  │student_id │◄────────────│student_id (FK)│             │course_id  │
  │name       │             │course_id  (FK)│────────────►│title      │
  │year       │             │semester       │             │credits    │
  │gpa        │             │grade          │             └───────────┘
  └───────────┘             └───────────────┘
```

### 복합키(Composite Key)

여러 속성으로 구성된 키:

```
ENROLLMENT(student_id, course_id, semester, grade)

기본키: {student_id, course_id, semester}
  - 세 개의 속성으로 구성된 복합키
  - student_id만으로는 충분하지 않음 (학생이 여러 과목 수강)
  - (student_id, course_id)만으로는 충분하지 않음
    (학생이 다른 학기에 과목을 재수강할 수 있음)
```

### 대리키 vs. 자연키

```
자연키:                        대리키:
  실제 데이터 사용                시스템 생성 값 사용

  STUDENT(ssn, name, ...)            STUDENT(student_id, ssn, name, ...)
          ^^^                                 ^^^^^^^^^^
          자연 PK                             대리 PK (auto-increment)

자연키의 장점:                   대리키의 장점:
  - 의미 있음                      - 간결함 (정수)
  - 추가 열 불필요                 - 불변
  - 이미 고유함                    - 비즈니스 의미 변경 없음
  - 자체 문서화                    - 간단한 조인

단점:                            단점:
  - 변경될 수 있음 (이름 변경)     - 추가 열
  - 클 수 있음 (SSN, ISBN)         - 의미론적 의미 없음
  - 개인정보 문제 (SSN)            - 의미를 위한 조회 필요
```

### 키 타입 요약

```
┌─────────────────────────────────────────────────────────────┐
│                      키 계층                                 │
│                                                             │
│  슈퍼키                                                      │
│    │                                                        │
│    ├── 후보키 (최소 슈퍼키)                                  │
│    │     │                                                  │
│    │     ├── 기본키 (선택된 후보키)                           │
│    │     │                                                  │
│    │     └── 대체키 (선택되지 않은 후보키)                     │
│    │                                                        │
│    └── 비최소 슈퍼키                                         │
│                                                             │
│  외래키 (다른 관계의 기본키를 참조)                           │
│                                                             │
│  복합키 (여러 속성을 가진 키)                                │
│                                                             │
│  대리키 (시스템 생성, 비즈니스 의미 없음)                     │
│  자연키 (실제 데이터에서 파생)                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. 무결성 제약조건

무결성 제약조건은 데이터 정확성과 일관성을 보장하기 위해 데이터베이스의 값을 제한하는 규칙입니다. 관계형 모델은 여러 유형의 제약조건을 정의합니다.

### 도메인 제약조건

**도메인 제약조건(domain constraint)**은 각 속성 값이 속성의 도메인(또는 허용되는 경우 NULL)에서 나와야 한다고 지정합니다.

```sql
-- SQL의 도메인 제약조건

CREATE TABLE student (
    student_id  CHAR(4),
    name        VARCHAR(50)   NOT NULL,       -- 도메인 + NOT NULL
    year        SMALLINT      CHECK (year BETWEEN 1 AND 4),
    gpa         NUMERIC(3,2)  CHECK (gpa >= 0.0 AND gpa <= 4.0),
    email       VARCHAR(100)  CHECK (email LIKE '%@%.%')
);

-- 도메인 제약조건 위반 예:
INSERT INTO student VALUES ('S001', 'Alice', 5, 3.85, 'alice@univ.edu');
-- 오류: year=5가 CHECK 제약조건 위반 (year BETWEEN 1 AND 4)

INSERT INTO student VALUES ('S001', 'Alice', 3, 4.50, 'alice@univ.edu');
-- 오류: gpa=4.50이 CHECK 제약조건 위반 (gpa <= 4.0)

INSERT INTO student VALUES ('S001', NULL, 3, 3.85, 'alice@univ.edu');
-- 오류: name이 NOT NULL
```

### 엔티티 무결성 제약조건

**엔티티 무결성 제약조건(entity integrity constraint)**은 기본키 속성이 NULL일 수 없다고 명시합니다.

```
규칙: PK = {A1, A2, ..., Ak}가 R의 기본키인 경우,
      r(R)의 모든 튜플 t에 대해:
        t[Ai] ≠ NULL  for all i = 1, 2, ..., k

근거:
  - 기본키는 각 튜플을 고유하게 식별함
  - PK가 NULL이면, 그 튜플을 다른 튜플과 구별할 수 없음
  - SQL에서 NULL ≠ NULL (NULL을 무엇과 비교해도 UNKNOWN이 됨)
  - 따라서 NULL PK는 식별을 불가능하게 만듦

예:
  ENROLLMENT(student_id, course_id, semester, grade)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
             복합 PK - 이 중 어느 것도 NULL일 수 없음

  유효:   ('S001', 'CS301', 'Fall2025', 'A')
  무효:   ('S001', NULL, 'Fall2025', 'A')     -- course_id는 PK의 일부
  유효:   ('S001', 'CS301', 'Fall2025', NULL)  -- grade는 PK의 일부가 아님
```

### 참조 무결성 제약조건

**참조 무결성 제약조건(referential integrity constraint)**은 외래키 값이 참조된 관계의 기존 기본키 값과 일치하거나 NULL이어야 함을 보장합니다.

```
규칙: r(R1)의 모든 튜플 t1에 대해 R2를 참조하는 외래키 FK를 가진 경우:
      t1[FK]는 NULL이거나
      t1[FK] = t2[PK]인 r(R2)의 튜플 t2가 존재함

예:

  STUDENT:                           COURSE:
  ┌──────┬──────┐                    ┌──────┬───────────┐
  │ S001 │Alice │                    │CS101 │Intro CS   │
  │ S002 │Bob   │                    │CS301 │DB Theory  │
  │ S003 │Carol │                    │MA201 │Lin Alg    │
  └──────┴──────┘                    └──────┴───────────┘

  ENROLLMENT:
  ┌────────┬─────────┬───────┐
  │stu_id  │course_id│ grade │
  ├────────┼─────────┼───────┤
  │ S001   │ CS101   │  A    │  ✓ S001이 STUDENT에 존재, CS101이 COURSE에 존재
  │ S002   │ CS301   │  B+   │  ✓ S002가 STUDENT에 존재, CS301이 COURSE에 존재
  │ S004   │ CS101   │  A-   │  ✗ S004가 STUDENT에 존재하지 않음!
  │ S003   │ CS999   │  B    │  ✗ CS999가 COURSE에 존재하지 않음!
  └────────┴─────────┴───────┘
```

### 참조 무결성 동작

참조된 튜플이 삭제되거나 업데이트될 때, DBMS는 고아 외래키를 처리해야 합니다:

```sql
CREATE TABLE enrollment (
    student_id  CHAR(4),
    course_id   CHAR(5),
    semester    VARCHAR(10),
    grade       VARCHAR(2),

    PRIMARY KEY (student_id, course_id, semester),

    FOREIGN KEY (student_id) REFERENCES student(student_id)
        ON DELETE CASCADE           -- 학생 삭제 시 enrollment 삭제
        ON UPDATE CASCADE,          -- 학생 PK 변경 시 FK 업데이트

    FOREIGN KEY (course_id) REFERENCES course(course_id)
        ON DELETE RESTRICT          -- 수강생이 있으면 과목 삭제 방지
        ON UPDATE CASCADE           -- 과목 PK 변경 시 FK 업데이트
);
```

| 동작 | 행동 |
|--------|----------|
| **CASCADE** | 참조하는 튜플로 삭제/업데이트 전파 |
| **RESTRICT** (또는 NO ACTION) | 참조하는 튜플이 존재하면 삭제/업데이트 거부 |
| **SET NULL** | 참조하는 튜플에서 외래키를 NULL로 설정 |
| **SET DEFAULT** | 외래키를 기본값으로 설정 |

### 참조 무결성 동작 예

```
시나리오: DELETE FROM student WHERE student_id = 'S001';

ON DELETE CASCADE:
  → S001의 모든 enrollment가 자동으로 삭제됨

ON DELETE RESTRICT:
  → S001이 enrollment를 가지고 있으므로 DELETE가 거부됨

ON DELETE SET NULL:
  → S001의 행에 대해 enrollment.student_id가 NULL로 설정됨
  → (그러나 student_id가 enrollment에서 PK의 일부이므로
     SET NULL은 엔티티 무결성을 위반합니다!)

ON DELETE SET DEFAULT:
  → enrollment.student_id가 기본값으로 설정됨
```

### 키 제약조건

키 선언을 통해 강제되는 추가 제약조건:

```sql
-- UNIQUE 제약조건: 대체키
CREATE TABLE student (
    student_id  CHAR(4)     PRIMARY KEY,
    email       VARCHAR(100) UNIQUE NOT NULL,  -- 대체키
    ssn         CHAR(11)     UNIQUE            -- 또 다른 대체키 (nullable)
);

-- 복합 UNIQUE 제약조건
CREATE TABLE course_offering (
    offering_id   SERIAL PRIMARY KEY,          -- 대리키
    course_id     CHAR(5) NOT NULL,
    semester      VARCHAR(10) NOT NULL,
    instructor_id INT NOT NULL,
    UNIQUE (course_id, semester)                -- 자연키 제약조건
);
```

### 일반 제약조건 (의미론적 제약조건)

도메인, 키, 참조 제약조건만으로 표현할 수 없는 제약조건:

```sql
-- CHECK 제약조건 (튜플 수준)
CREATE TABLE course (
    course_id  CHAR(5) PRIMARY KEY,
    title      VARCHAR(100) NOT NULL,
    credits    SMALLINT CHECK (credits BETWEEN 1 AND 5),
    max_seats  INT CHECK (max_seats > 0),
    min_seats  INT CHECK (min_seats > 0),
    CONSTRAINT seats_check CHECK (max_seats >= min_seats)
);

-- ASSERTION (테이블 간 제약조건, SQL 표준이지만 거의 구현되지 않음)
-- "학생은 학기당 7개 이상의 과목을 수강할 수 없습니다"
CREATE ASSERTION max_courses_per_semester
    CHECK (NOT EXISTS (
        SELECT student_id, semester
        FROM enrollment
        GROUP BY student_id, semester
        HAVING COUNT(*) > 7
    ));

-- 실제로는 테이블 간 제약조건에 트리거를 사용합니다:
CREATE OR REPLACE FUNCTION check_max_courses()
RETURNS TRIGGER AS $$
BEGIN
    IF (SELECT COUNT(*) FROM enrollment
        WHERE student_id = NEW.student_id
        AND semester = NEW.semester) >= 7 THEN
        RAISE EXCEPTION 'Student cannot enroll in more than 7 courses per semester';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_max_courses
    BEFORE INSERT ON enrollment
    FOR EACH ROW EXECUTE FUNCTION check_max_courses();
```

### 제약조건 요약

```
┌─────────────────────────────────────────────────────────────────┐
│                    무결성 제약조건                               │
│                                                                 │
│  고유 모델 제약조건 (관계형 모델에 내장):                         │
│    - 중복 튜플 없음                                             │
│    - 원자적 속성 값 (1NF)                                       │
│                                                                 │
│  스키마 기반 제약조건 (DDL):                                    │
│    - 도메인 제약조건 (데이터 타입, CHECK)                        │
│    - 키 제약조건 (PRIMARY KEY, UNIQUE)                          │
│    - 엔티티 무결성 (PK NOT NULL)                                │
│    - 참조 무결성 (FOREIGN KEY)                                  │
│    - NOT NULL 제약조건                                          │
│                                                                 │
│  애플리케이션 기반 제약조건 (비즈니스 규칙):                     │
│    - 트리거                                                     │
│    - 애플리케이션 로직                                          │
│    - 어서션 (SQL에서 거의 지원되지 않음)                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. 관계형 스키마 표기법

문서와 교과서에서 관계형 스키마를 표현하는 몇 가지 표준 방법이 있습니다.

### 텍스트 표기법

```
STUDENT(student_id, name, year, gpa)
  PK: student_id
  FK: (없음)

COURSE(course_id, title, credits, dept_id)
  PK: course_id
  FK: dept_id → DEPARTMENT(dept_id)

ENROLLMENT(student_id, course_id, semester, grade)
  PK: (student_id, course_id, semester)
  FK: student_id → STUDENT(student_id)
      course_id → COURSE(course_id)

DEPARTMENT(dept_id, dept_name, building, budget)
  PK: dept_id

INSTRUCTOR(instructor_id, name, dept_id, salary)
  PK: instructor_id
  FK: dept_id → DEPARTMENT(dept_id)
```

### 밑줄 규약

기본키 속성에 밑줄을 표시합니다 (여기서는 밑줄로 표시):

```
STUDENT(_student_id_, name, year, gpa)
COURSE(_course_id_, title, credits, dept_id*)
ENROLLMENT(_student_id*_, _course_id*_, _semester_, grade)
DEPARTMENT(_dept_id_, dept_name, building, budget)
INSTRUCTOR(_instructor_id_, name, dept_id*, salary)

규약:
  _밑줄_ = 기본키 속성
  * = 외래키 속성
```

### 다이어그램 표기법

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│    DEPARTMENT     │     │    INSTRUCTOR     │     │      COURSE      │
├──────────────────┤     ├──────────────────┤     ├──────────────────┤
│ PK dept_id       │◄────│ PK instructor_id │     │ PK course_id     │
│    dept_name     │     │    name          │     │    title         │
│    building      │     │ FK dept_id    ───┘     │    credits       │
│    budget        │     │    salary       │     │ FK dept_id    ───┤
└──────────────────┘     └──────────────────┘     └────────┬─────────┘
                                                           │
                                                           │
┌──────────────────┐     ┌──────────────────────┐         │
│     STUDENT      │     │     ENROLLMENT       │         │
├──────────────────┤     ├──────────────────────┤         │
│ PK student_id    │◄────│ PK,FK student_id  ───┘         │
│    name          │     │ PK,FK course_id   ─────────────┘
│    year          │     │ PK    semester     │
│    gpa           │     │       grade        │
└──────────────────┘     └──────────────────────┘
```

### 스키마 정의로서의 SQL DDL

```sql
-- SQL로 완전한 스키마 정의

CREATE TABLE department (
    dept_id     CHAR(4)      PRIMARY KEY,
    dept_name   VARCHAR(50)  NOT NULL UNIQUE,
    building    VARCHAR(30),
    budget      NUMERIC(12,2) CHECK (budget >= 0)
);

CREATE TABLE instructor (
    instructor_id  SERIAL       PRIMARY KEY,
    name           VARCHAR(50)  NOT NULL,
    dept_id        CHAR(4)      NOT NULL,
    salary         NUMERIC(10,2) CHECK (salary > 0),
    FOREIGN KEY (dept_id) REFERENCES department(dept_id)
        ON DELETE RESTRICT ON UPDATE CASCADE
);

CREATE TABLE student (
    student_id  CHAR(4)      PRIMARY KEY,
    name        VARCHAR(50)  NOT NULL,
    year        SMALLINT     NOT NULL CHECK (year BETWEEN 1 AND 4),
    gpa         NUMERIC(3,2) CHECK (gpa >= 0.0 AND gpa <= 4.0)
);

CREATE TABLE course (
    course_id   CHAR(5)      PRIMARY KEY,
    title       VARCHAR(100) NOT NULL,
    credits     SMALLINT     NOT NULL CHECK (credits BETWEEN 1 AND 5),
    dept_id     CHAR(4)      NOT NULL,
    FOREIGN KEY (dept_id) REFERENCES department(dept_id)
        ON DELETE RESTRICT ON UPDATE CASCADE
);

CREATE TABLE enrollment (
    student_id  CHAR(4)      NOT NULL,
    course_id   CHAR(5)      NOT NULL,
    semester    VARCHAR(10)  NOT NULL,
    grade       VARCHAR(2),
    PRIMARY KEY (student_id, course_id, semester),
    FOREIGN KEY (student_id) REFERENCES student(student_id)
        ON DELETE CASCADE ON UPDATE CASCADE,
    FOREIGN KEY (course_id) REFERENCES course(course_id)
        ON DELETE RESTRICT ON UPDATE CASCADE
);
```

---

## 7. NULL 의미와 3치 논리

NULL은 관계형 모델에서 가장 미묘하고 자주 오해되는 개념 중 하나입니다. NULL 의미를 이해하는 것은 올바른 쿼리 작성에 중요합니다.

### NULL이 나타내는 것

NULL은 **값이 아닙니다**. 다음을 나타내는 **마커**입니다:

| 의미 | 예 |
|---------|---------|
| **누락(Missing)** (값이 존재하지만 알 수 없음) | 아직 기록되지 않은 학생의 전화번호 |
| **적용 불가능(Inapplicable)** (값이 의미가 없음) | 집에 사는 학생의 아파트 번호 |
| **보류(Withheld)** (값이 존재하지만 공개되지 않음) | 동료의 급여 |

```
중요한 구별:
  NULL ≠ 0       (0은 알려진 값)
  NULL ≠ ''      (빈 문자열은 알려진 값)
  NULL ≠ FALSE   (false는 알려진 불린 값)
  NULL ≠ NULL    (NULL과 NULL을 비교하면 UNKNOWN이지 TRUE가 아님!)
```

### 3치 논리(Three-Valued Logic, 3VL)

표준 불린 논리는 두 개의 값을 가집니다: TRUE와 FALSE. SQL은 TRUE, FALSE, **UNKNOWN**을 가진 **3치 논리**를 사용합니다.

NULL을 포함하는 비교는 UNKNOWN을 생성합니다:

```
5 > 3        → TRUE
5 > 7        → FALSE
5 > NULL     → UNKNOWN
NULL > NULL  → UNKNOWN
NULL = NULL  → UNKNOWN
NULL = 5     → UNKNOWN
NULL <> 5    → UNKNOWN
NULL <> NULL → UNKNOWN
```

### 3VL에 대한 진리표

**AND:**

```
         │  TRUE     FALSE    UNKNOWN
─────────┼──────────────────────────────
TRUE     │  TRUE     FALSE    UNKNOWN
FALSE    │  FALSE    FALSE    FALSE
UNKNOWN  │  UNKNOWN  FALSE    UNKNOWN
```

**OR:**

```
         │  TRUE     FALSE    UNKNOWN
─────────┼──────────────────────────────
TRUE     │  TRUE     TRUE     TRUE
FALSE    │  TRUE     FALSE    UNKNOWN
UNKNOWN  │  TRUE     UNKNOWN  UNKNOWN
```

**NOT:**

```
NOT TRUE    → FALSE
NOT FALSE   → TRUE
NOT UNKNOWN → UNKNOWN
```

### SQL 쿼리에 대한 함의

```sql
-- 설정
CREATE TABLE test (id INT, value INT);
INSERT INTO test VALUES (1, 10), (2, 20), (3, NULL);

-- 쿼리 1: WHERE value = 10
-- 반환: {(1, 10)}
-- 행 (3, NULL): NULL = 10 → UNKNOWN → 행 제외

-- 쿼리 2: WHERE value <> 10
-- 반환: {(2, 20)}
-- 행 (3, NULL): NULL <> 10 → UNKNOWN → 행 제외!
-- 놀라움: "value = 10"도 "value <> 10"도 NULL 행을 반환하지 않습니다!

-- 쿼리 3: WHERE value = NULL
-- 반환: 빈 집합!
-- NULL = NULL → UNKNOWN → NULL을 가진 모든 행이 제외됨
-- 이것은 일반적인 버그입니다. 대신 IS NULL을 사용하세요.

-- 쿼리 4: WHERE value IS NULL
-- 반환: {(3, NULL)}
-- IS NULL은 NULL을 올바르게 확인하는 특수 연산자입니다

-- 쿼리 5: WHERE value IS NOT NULL
-- 반환: {(1, 10), (2, 20)}
```

### 집계에서의 NULL

```sql
-- 집계 함수에서의 NULL 동작

CREATE TABLE scores (student_id CHAR(4), score INT);
INSERT INTO scores VALUES
    ('S001', 90), ('S002', 80), ('S003', NULL), ('S004', 70);

SELECT COUNT(*)        FROM scores;  -- 4 (모든 행 계산)
SELECT COUNT(score)    FROM scores;  -- 3 (NULL 제외)
SELECT SUM(score)      FROM scores;  -- 240 (NULL 제외)
SELECT AVG(score)      FROM scores;  -- 80 (240/3, NULL 제외)
SELECT MIN(score)      FROM scores;  -- 70 (NULL 제외)
SELECT MAX(score)      FROM scores;  -- 90 (NULL 제외)

-- 중요: AVG(score) = 80, 60이 아님!
-- AVG는 NULL을 무시하므로 240/3 = 80을 계산, 240/4 = 60이 아님
-- NULL을 0으로 처리하려면:
SELECT AVG(COALESCE(score, 0)) FROM scores;  -- 60 (240/4)
```

### 불린 표현식에서의 NULL

```sql
-- WHERE 절은 조건이 TRUE인 행만 유지합니다
-- 조건이 FALSE 또는 UNKNOWN인 행은 필터링됩니다

SELECT * FROM student WHERE gpa > 3.5;
-- gpa가 NULL인 경우: NULL > 3.5 → UNKNOWN → 행 제외

SELECT * FROM student WHERE gpa > 3.5 OR gpa <= 3.5;
-- gpa가 NULL인 경우:
--   NULL > 3.5 → UNKNOWN
--   NULL <= 3.5 → UNKNOWN
--   UNKNOWN OR UNKNOWN → UNKNOWN
--   논리적으로 gpa > 3.5 OR gpa <= 3.5가 모든 경우를 커버해야 하지만
--   행이 제외됩니다!

-- NULL을 포함하려면:
SELECT * FROM student WHERE gpa > 3.5 OR gpa IS NULL;
```

### DISTINCT 및 GROUP BY에서의 NULL

```sql
-- DISTINCT와 GROUP BY에서는 NULL이 동등한 것으로 간주됩니다
-- (이것은 "NULL ≠ NULL"에 대한 예외입니다)

SELECT DISTINCT dept FROM instructor;
-- 여러 instructor가 dept = NULL을 가지면,
-- 결과에 하나의 NULL만 나타납니다

SELECT dept, COUNT(*) FROM instructor GROUP BY dept;
-- dept = NULL인 모든 행이 함께 그룹화됩니다
```

### 조인에서의 NULL

```sql
-- 조인에서 NULL 값은 절대 일치하지 않습니다

-- STUDENT: (S001, 'Alice', 'CS'), (S002, 'Bob', NULL)
-- DEPARTMENT: ('CS', 'Computer Science'), ('EE', 'Electrical Engineering')

SELECT s.name, d.dept_name
FROM student s
JOIN department d ON s.dept = d.dept_id;

-- 결과: ('Alice', 'Computer Science')만
-- Bob은 NULL = 'CS' → UNKNOWN, NULL = 'EE' → UNKNOWN이므로 제외됩니다
```

### COALESCE 및 NULLIF

```sql
-- COALESCE: 첫 번째 비NULL 인자 반환
SELECT COALESCE(phone, email, 'No contact info') AS contact
FROM student;
-- phone이 NULL이고 email이 'alice@univ.edu'인 경우:
--   'alice@univ.edu' 반환
-- 둘 다 NULL인 경우:
--   'No contact info' 반환

-- NULLIF: 두 값이 같으면 NULL 반환
SELECT NULLIF(actual_score, 0) AS adjusted_score
FROM test_results;
-- actual_score = 0인 경우: NULL 반환 (0을 알 수 없는 것으로 처리)
-- actual_score = 85인 경우: 85 반환
```

### NULL에 대한 모범 사례

```
해야 할 것:
  ✓ NULL 확인에 IS NULL / IS NOT NULL 사용
  ✓ 기본값 제공에 COALESCE 사용
  ✓ WHERE 절 작성 시 NULL 동작 고려
  ✓ 집계 쿼리에서 NULL 처리에 대해 명시적으로 설명
  ✓ NULL을 허용하는 열과 그 이유를 문서화

하지 말아야 할 것:
  ✗ = NULL 또는 <> NULL 사용 (항상 UNKNOWN 반환)
  ✗ AVG가 NULL을 0으로 포함한다고 가정
  ✗ UNKNOWN과의 OR이 행을 제외할 수 있음을 잊지 말 것
  ✗ 기본키 열에서 NULL 허용
  ✗ NULL을 의미 있는 비즈니스 값으로 사용 (대신 플래그 열 사용)
```

---

## 8. 실제 관계형 모델

### 관계형 개념의 Python 구현

```python
"""
Python에서 관계형 모델 개념의 단순화된 구현.
교육 목적만을 위한 것입니다.
"""

from typing import Any, Optional
from dataclasses import dataclass


class Domain:
    """도메인(허용된 값의 집합)을 나타냅니다."""

    def __init__(self, name: str, check_fn=None):
        self.name = name
        self.check_fn = check_fn or (lambda x: True)

    def validate(self, value: Any) -> bool:
        if value is None:
            return True  # NULL은 명시적으로 제약되지 않는 한 허용됩니다
        return self.check_fn(value)


class Relation:
    """제약조건 확인을 가진 단순화된 관계(테이블)."""

    def __init__(self, name: str, attributes: list[str],
                 primary_key: list[str],
                 domains: Optional[dict] = None):
        self.name = name
        self.attributes = attributes
        self.primary_key = primary_key
        self.domains = domains or {}
        self.tuples: list[dict] = []

    def insert(self, values: dict) -> bool:
        """제약조건 확인과 함께 튜플을 삽입합니다."""
        # 확인: 모든 속성이 존재
        for attr in self.attributes:
            if attr not in values:
                raise ValueError(f"Missing attribute: {attr}")

        # 엔티티 무결성: PK는 NULL일 수 없음
        for pk_attr in self.primary_key:
            if values[pk_attr] is None:
                raise ValueError(
                    f"Entity integrity violation: "
                    f"PK attribute '{pk_attr}' cannot be NULL"
                )

        # 도메인 제약조건
        for attr, domain in self.domains.items():
            if not domain.validate(values.get(attr)):
                raise ValueError(
                    f"Domain violation: {values.get(attr)} "
                    f"not in domain {domain.name}"
                )

        # 키 제약조건: 중복 PK 없음
        pk_values = tuple(values[k] for k in self.primary_key)
        for existing in self.tuples:
            existing_pk = tuple(existing[k] for k in self.primary_key)
            if pk_values == existing_pk:
                raise ValueError(
                    f"Key violation: duplicate PK {pk_values}"
                )

        self.tuples.append(values)
        return True

    def select(self, predicate=None) -> list[dict]:
        """서술논리와 일치하는 튜플을 선택합니다 (sigma 연산)."""
        if predicate is None:
            return self.tuples.copy()
        return [t for t in self.tuples if predicate(t)]

    def project(self, attrs: list[str]) -> list[tuple]:
        """주어진 속성으로 프로젝트합니다 (pi 연산)."""
        result = set()
        for t in self.tuples:
            projected = tuple(t[a] for a in attrs)
            result.add(projected)
        return [dict(zip(attrs, p)) for p in result]

    def __repr__(self):
        header = " | ".join(f"{a:>12}" for a in self.attributes)
        separator = "-" * len(header)
        rows = []
        for t in self.tuples:
            row = " | ".join(
                f"{str(t.get(a, 'NULL')):>12}" for a in self.attributes
            )
            rows.append(row)
        return f"\n{self.name}\n{separator}\n{header}\n{separator}\n" + \
               "\n".join(rows) + f"\n{separator}\n"


# --- 데모 ---

# 도메인 정의
gpa_domain = Domain("GPA", lambda x: 0.0 <= x <= 4.0)
year_domain = Domain("Year", lambda x: x in {1, 2, 3, 4})

# 관계 생성
student = Relation(
    name="STUDENT",
    attributes=["student_id", "name", "year", "gpa"],
    primary_key=["student_id"],
    domains={"gpa": gpa_domain, "year": year_domain}
)

# 유효한 튜플 삽입
student.insert({"student_id": "S001", "name": "Alice", "year": 3, "gpa": 3.85})
student.insert({"student_id": "S002", "name": "Bob", "year": 2, "gpa": 3.42})
student.insert({"student_id": "S003", "name": "Carol", "year": 4, "gpa": None})

print(student)

# 엔티티 무결성 위반
try:
    student.insert({"student_id": None, "name": "Dave", "year": 1, "gpa": 3.0})
except ValueError as e:
    print(f"Caught: {e}")

# 도메인 제약조건 위반
try:
    student.insert({"student_id": "S004", "name": "Eve", "year": 5, "gpa": 3.0})
except ValueError as e:
    print(f"Caught: {e}")

# 키 제약조건 위반
try:
    student.insert({"student_id": "S001", "name": "Frank", "year": 1, "gpa": 2.5})
except ValueError as e:
    print(f"Caught: {e}")

# 선택
honors = student.select(lambda t: t["gpa"] is not None and t["gpa"] > 3.5)
print("Honors students:", honors)

# 프로젝션
names = student.project(["name", "year"])
print("Names and years:", names)
```

### 일반적인 관계형 모델 오해

```
오해                          현실
─────────────────────────────          ─────────────────────────────────
"테이블은 관계입니다"                 SQL 테이블은 다중집합이지 집합이 아님
                                       (중복을 허용)

"열 순서가 중요합니다"                 형식 모델에서는 중요하지 않음
                                       (SQL은 선언 순서를 보존)

"NULL은 빈 것이거나 0을 의미합니다"    NULL은 알 수 없거나 적용 불가능하거나
                                       누락을 의미 — 빈/0이 아님

"모든 테이블에 자동 증가 ID가           모델은 기본키를 요구하지만
 있어야 합니다"                        어떤 후보키든 가능

"외래키는 어떤 열이든 참조할 수 있음"  FK는 기본키를 참조
                                       (실제로는 UNIQUE 키)

"관계형 = SQL"                        SQL은 관계형 모델에서 여러 면에서
                                       벗어남 (중복, NULL 처리, 백 의미론)
```

---

## 9. 연습문제

### 개념적 질문

**연습문제 2.1**: Codd의 12가지 규칙을 자신의 말로 설명하세요. 각 규칙에 대해, 이를 만족하는 DBMS 기능의 예를 제시하세요.

**연습문제 2.2**: 다음의 차이를 설명하세요:
- (a) 슈퍼키와 후보키
- (b) 기본키와 대체키
- (c) 자연키와 대리키
- (d) 엔티티 무결성과 참조 무결성

**연습문제 2.3**: 관계형 모델이 기본키에서 NULL 값을 금지하는 이유는 무엇입니까? 이 제한이 완화되면 어떤 문제가 발생할까요?

**연습문제 2.4**: SQL에서 NULL = NULL이 UNKNOWN으로 평가되는 이유를 설명하고, TRUE가 아닌 이유를 설명하세요. NULL = NULL을 TRUE로 처리하면 잘못된 쿼리 결과를 생성할 시나리오를 설명하세요.

### 스키마 설계 질문

**연습문제 2.5**: 다음 도서관 시스템 요구사항이 주어졌을 때, 모든 관계, 그들의 속성, 기본키, 외래키, 도메인 제약조건을 식별하세요:

- 도서관에는 각각 ISBN, 제목, 저자, 출판사, 연도, 사본 수를 가진 책들이 있습니다
- 회원은 회원 ID, 이름, 주소, 전화번호를 가집니다
- 회원은 책을 빌릴 수 있습니다. 각 대출은 회원, 책, 대출 날짜, 반납 예정일, 반납 날짜를 기록합니다
- 각 책은 하나 이상의 카테고리에 속합니다 (예: Fiction, Science, History)

**연습문제 2.6**: 다음 관계 스키마에 대해, 모든 후보키를 식별하세요:

```
FLIGHT(flight_number, airline, departure_city, arrival_city,
       departure_time, arrival_time, gate, aircraft_id)

제약조건:
  - flight number는 항공편을 고유하게 식별합니다
  - 항공기는 한 번에 하나의 게이트에만 있을 수 있습니다
  - 게이트는 한 번에 하나의 항공기만 가질 수 있습니다
```

**연습문제 2.7**: 주어진 이 스키마에서, 모든 무결성 제약조건 위반을 식별하고 수정하세요:

```sql
CREATE TABLE department (
    dept_id   CHAR(4) PRIMARY KEY,
    dept_name VARCHAR(50)
);

CREATE TABLE employee (
    emp_id    INT PRIMARY KEY,
    name      VARCHAR(50) NOT NULL,
    dept_id   CHAR(4) REFERENCES department(dept_id),
    salary    NUMERIC(10,2),
    mgr_id    INT REFERENCES employee(emp_id)
);

-- 시도된 삽입:
INSERT INTO employee VALUES (1, 'Alice', 'CS01', 75000, NULL);
INSERT INTO department VALUES ('CS01', 'Computer Science');
INSERT INTO employee VALUES (2, NULL, 'CS01', 60000, 1);
INSERT INTO employee VALUES (3, 'Carol', 'EE01', 65000, 1);
INSERT INTO employee VALUES (NULL, 'Dave', 'CS01', 55000, 1);
```

### SQL 및 NULL 질문

**연습문제 2.8**: 각 쿼리의 출력을 예측하세요. 이유를 설명하세요.

```sql
CREATE TABLE t (a INT, b INT);
INSERT INTO t VALUES (1, 10), (2, NULL), (3, 30), (NULL, 40);

-- (a)
SELECT * FROM t WHERE b > 20;

-- (b)
SELECT * FROM t WHERE b > 20 OR b <= 20;

-- (c)
SELECT * FROM t WHERE a IN (1, 2, NULL);

-- (d)
SELECT COUNT(*), COUNT(a), COUNT(b), SUM(b), AVG(b) FROM t;

-- (e)
SELECT * FROM t WHERE b NOT IN (10, NULL);

-- (f)
SELECT COALESCE(a, 0) + COALESCE(b, 0) AS total FROM t;
```

**연습문제 2.9**: 다음에 대해 NULL을 올바르게 처리하는 SQL 쿼리를 작성하세요:

`EMPLOYEE(emp_id, name, dept, salary, bonus)` 주어짐, 여기서 `bonus`는 NULL일 수 있음:

- (a) 총 보상 (급여 + 보너스)이 100,000을 초과하는 직원 찾기
- (b) NULL 보너스를 0으로 처리하여 평균 보너스 찾기
- (c) 모든 직원이 비NULL 보너스를 가진 부서 찾기
- (d) 직원 'E001'과 다른 보너스를 가진 직원 찾기 (NULL 포함)

### 설계 연습

**연습문제 2.10**: 다음 요구사항으로 온라인 서점을 위한 관계형 스키마를 설계하세요:

- 책은 ISBN, 제목, 가격, 출판일, 페이지 수를 가집니다
- 저자는 ID, 이름, 전기를 가집니다. 한 책은 여러 저자를 가질 수 있고, 저자는 여러 책을 쓸 수 있습니다
- 고객은 ID, 이름, 이메일, 배송 주소를 가집니다
- 고객은 주문을 할 수 있습니다. 각 주문은 주문 ID, 주문 날짜, 상태를 가집니다
- 각 주문은 하나 이상의 책을 수량과 함께 포함합니다
- 고객은 책에 대한 리뷰를 작성할 수 있습니다 (평점 1-5, 텍스트, 날짜)

제공 사항:
1. 기본 및 외래키를 가진 모든 관계 스키마
2. 각 속성에 대한 도메인 제약조건
3. 추가 무결성 제약조건 (CHECK로 또는 영어로 작성)
4. 최소 두 개의 관계에 대한 유효하고 무효한 튜플의 예

---

**이전**: [데이터베이스 시스템 소개](./01_Introduction_to_Database_Systems.md) | **다음**: [관계 대수](./03_Relational_Algebra.md)
