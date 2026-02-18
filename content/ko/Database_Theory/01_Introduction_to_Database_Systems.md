# 데이터베이스 시스템 소개(Introduction to Database Systems)

**이전**: [개요](./00_Overview.md) | **다음**: [관계형 모델](./02_Relational_Model.md)

---

데이터베이스 시스템은 현대 컴퓨팅에서 가장 중요한 소프트웨어 인프라 중 하나입니다. 은행 거래와 항공 예약부터 소셜 미디어 피드와 과학 연구에 이르기까지, 데이터베이스는 지속적이고 공유되는 데이터를 관리하는 거의 모든 애플리케이션을 뒷받침합니다. 이 레슨에서는 데이터베이스 이론의 기초를 형성하는 기본 개념, 아키텍처, 용어를 소개합니다.

## 목차(Table of Contents)

1. [데이터베이스란 무엇인가?](#1-what-is-a-database)
2. [왜 파일만 사용하면 안 되는가?](#2-why-not-just-use-files)
3. [데이터베이스 관리 시스템](#3-database-management-systems)
4. [데이터베이스 시스템의 간략한 역사](#4-brief-history-of-database-systems)
5. [3단계 스키마 아키텍처](#5-three-schema-architecture)
6. [데이터 독립성](#6-data-independence)
7. [ANSI/SPARC 아키텍처](#7-ansisparc-architecture)
8. [데이터 모델](#8-data-models)
9. [데이터베이스 사용자 및 역할](#9-database-users-and-roles)
10. [데이터베이스 시스템 아키텍처](#10-database-system-architecture)
11. [연습문제](#11-exercises)

---

## 1. 데이터베이스란 무엇인가?

**데이터베이스(Database)**는 조직 내 여러 사용자의 정보 요구를 충족하도록 설계된, 논리적으로 관련된 데이터의 조직화된 집합입니다. 더 정확하게는:

> **데이터베이스**: 여러 사용자와 애플리케이션 간의 데이터 정의, 구축, 조작 및 공유를 위한 제어되고 신뢰할 수 있으며 효율적인 메커니즘을 제공하는, 공유되고 통합된 지속적 데이터의 집합입니다.

데이터베이스의 주요 속성:

- **지속성(Persistent)**: 데이터가 생성한 프로세스를 넘어 존속
- **공유(Shared)**: 여러 사용자와 애플리케이션이 데이터에 동시 접근 가능
- **통합(Integrated)**: 데이터가 통일된 구조로 수집되어 중복 최소화
- **관리(Managed)**: 규칙과 제약조건을 강제하는 소프트웨어가 접근 제어

### 데이터베이스 vs. 데이터

원시 **데이터(data)**와 **데이터베이스(database)**를 구분하는 것이 중요합니다:

```
데이터:       개별 사실 (예: "Alice", "29", "Engineering")

정보:         문맥과 의미를 가진 데이터
             ("Alice는 29세이고 Engineering에서 일합니다")

데이터베이스:    구조, 제약조건, 접근 제어를 갖춘
             조직화된 관련 데이터의 집합
```

### 간단한 예

학생, 과목, 수강 정보를 추적해야 하는 대학을 생각해봅시다:

```
STUDENT 테이블:
+--------+-----------+------+--------+
| Stu_ID | Name      | Year | GPA    |
+--------+-----------+------+--------+
| S001   | Alice Kim | 3    | 3.85   |
| S002   | Bob Park  | 2    | 3.42   |
| S003   | Carol Lee | 4    | 3.91   |
+--------+-----------+------+--------+

COURSE 테이블:
+-----------+---------------------+---------+
| Course_ID | Title               | Credits |
+-----------+---------------------+---------+
| CS101     | Intro to CS         | 3       |
| CS301     | Database Theory     | 3       |
| MA201     | Linear Algebra      | 4       |
+-----------+---------------------+---------+

ENROLLMENT 테이블:
+--------+-----------+-------+
| Stu_ID | Course_ID | Grade |
+--------+-----------+-------+
| S001   | CS101     | A     |
| S001   | CS301     | A+    |
| S002   | CS101     | B+    |
| S003   | MA201     | A     |
+--------+-----------+-------+
```

이러한 구조화된 표현은 다음과 같은 질의를 가능하게 합니다:
- "Alice가 수강 중인 과목은 무엇인가?"
- "CS101을 수강하는 학생은 몇 명인가?"
- "CS301 수강생의 평균 GPA는?"

---

## 2. 왜 파일만 사용하면 안 되는가?

데이터베이스가 존재하기 전에는 애플리케이션이 플랫 파일에 데이터를 저장했습니다. 파일은 간단하지만, 데이터베이스 시스템의 개발을 동기부여한 근본적인 한계를 가지고 있습니다.

### 파일 기반 접근법

```
                    ┌──────────────┐
                    │  Application │
                    │   Program 1  │──────► student_records.dat
                    └──────────────┘
                    ┌──────────────┐
                    │  Application │
                    │   Program 2  │──────► course_records.dat
                    └──────────────┘
                    ┌──────────────┐
                    │  Application │
                    │   Program 3  │──────► enrollment.dat
                    └──────────────┘

    각 애플리케이션이 독립적으로 자신의 파일을 관리합니다.
```

### 파일 기반 시스템의 문제점

| 문제 | 설명 | 예시 |
|---------|-------------|---------|
| **데이터 중복(Data Redundancy)** | 동일한 데이터가 여러 파일에 저장됨 | 학생 이름이 student 파일과 enrollment 파일에 모두 존재 |
| **데이터 불일치(Data Inconsistency)** | 중복된 사본이 동기화되지 않음 | 한 파일에서 이름이 변경되었지만 다른 파일에서는 변경되지 않음 |
| **프로그램-데이터 종속성(Program-Data Dependence)** | 파일 형식 변경이 프로그램 변경을 요구 | 레코드에 필드를 추가하면 기존 코드가 손상됨 |
| **제한된 데이터 공유(Limited Data Sharing)** | 각 애플리케이션이 자신의 파일을 가짐 | 등록 담당자와 재정 지원 부서가 데이터를 공유할 수 없음 |
| **동시 접근 불가(No Concurrent Access)** | 여러 사용자가 안전하게 동시 업데이트 불가 | 두 등록 담당자가 동일한 레코드를 편집 |
| **복구 메커니즘 없음(No Recovery Mechanism)** | 장애로 인한 데이터 손실이 영구적 | 파일 쓰기 중 정전이 데이터를 손상 |
| **보안 제어 없음(No Security Control)** | 파일 수준 접근만 가능, 세밀한 제어 불가 | 특정 필드에 대한 접근을 제한할 수 없음 |
| **무결성 강제 없음(No Integrity Enforcement)** | 유효한 데이터에 대한 중앙 집중식 규칙이 없음 | GPA 5.0 또는 음수 학점이 저장될 수 있음 |

### 중복 및 불일치의 구체적 예

등록 담당자와 재정 지원 부서가 각각 자신의 파일을 유지한다고 가정합시다:

```
등록 담당자의 파일 (students.txt):
S001, Alice Kim, Computer Science, 3.85

재정 지원 부서의 파일 (financial.txt):
S001, Alice Kim, Computer Science, Need-Based

Alice가 전공을 Data Science로 변경...
등록 담당자는 자신의 파일을 업데이트하지만, 재정 지원 부서의 파일은 여전히
"Computer Science"라고 표시됩니다. 이제 데이터가 불일치합니다.
```

### 데이터베이스 접근법

DBMS는 데이터 관리를 중앙집중화하여 이러한 문제를 해결합니다:

```
    ┌──────────────┐
    │  Application  │───┐
    │   Program 1   │   │
    └──────────────┘   │    ┌──────────┐    ┌──────────────┐
    ┌──────────────┐   ├───►│          │    │              │
    │  Application  │───┤   │   DBMS   │───►│   Database   │
    │   Program 2   │   ├───►│          │    │              │
    └──────────────┘   │    └──────────┘    └──────────────┘
    ┌──────────────┐   │
    │  Application  │───┘
    │   Program 3   │
    └──────────────┘

    모든 애플리케이션이 DBMS를 통해 단일 데이터베이스에 접근합니다.
```

---

## 3. 데이터베이스 관리 시스템

**데이터베이스 관리 시스템(Database Management System, DBMS)**은 사용자/애플리케이션과 저장된 데이터 사이에 위치하는 소프트웨어입니다. 데이터 생성, 검색, 업데이트 및 관리를 위한 체계적인 방법을 제공합니다.

### DBMS의 핵심 기능

1. **데이터 정의(Data Definition)**: 데이터베이스의 구조(스키마) 정의
2. **데이터 조작(Data Manipulation)**: 데이터 삽입, 업데이트, 삭제 및 검색
3. **데이터 제어(Data Control)**: 보안, 무결성, 동시 접근 관리
4. **데이터 관리(Data Administration)**: 백업, 복구, 성능 튜닝

### DBMS 구성 요소

```
┌─────────────────────────────────────────────────────────┐
│                    DBMS Software                         │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │    Query      │  │  Transaction │  │   Storage    │  │
│  │  Processor    │  │   Manager    │  │   Manager    │  │
│  │              │  │              │  │              │  │
│  │ - Parser     │  │ - Scheduler  │  │ - Buffer Mgr │  │
│  │ - Optimizer  │  │ - Lock Mgr   │  │ - File Mgr   │  │
│  │ - Executor   │  │ - Recovery   │  │ - Disk Mgr   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Catalog    │  │  Authorization│  │  Integrity   │  │
│  │   Manager    │  │   Manager    │  │   Manager    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                          │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
                ┌──────────────────┐
                │    Stored Data    │
                │  (Files on Disk)  │
                └──────────────────┘
```

### DBMS 접근법의 장점

| 장점 | 설명 |
|-----------|-------------|
| **데이터 추상화(Data Abstraction)** | 사용자는 논리적 구조를 보며, 물리적 저장 세부사항을 보지 않음 |
| **최소 중복(Minimal Redundancy)** | 제어된 중복을 가진 중앙집중식 저장 |
| **일관성(Consistency)** | DBMS가 무결성 제약조건 강제 |
| **데이터 공유(Data Sharing)** | 여러 사용자/앱이 동일한 데이터에 접근 |
| **동시성 제어(Concurrency Control)** | 잠금/MVCC를 통한 안전한 동시 접근 |
| **복구(Recovery)** | 장애로부터 자동 복구 |
| **보안(Security)** | 세밀한 접근 제어 (테이블, 행, 열 수준) |
| **표준 강제(Standards Enforcement)** | SQL 표준, 명명 규칙, 데이터 형식 |

### DBMS를 사용하지 않아야 할 때

DBMS가 항상 올바른 선택은 아닙니다. 다음과 같은 경우 더 간단한 대안을 고려하세요:

- 데이터베이스와 애플리케이션이 단순하고, 잘 정의되어 있으며, 변경될 것으로 예상되지 않을 때
- DBMS 오버헤드가 충족할 수 없는 엄격한 실시간 요구사항이 있을 때
- 다중 사용자 접근이 필요하지 않을 때
- 데이터 볼륨이 매우 작을 때 (몇 킬로바이트)
- 애플리케이션이 직접적인 저수준 저장 접근을 요구할 때

예시: 임베디드 센서 펌웨어, 간단한 구성 파일, 작은 스크립트.

---

## 4. 데이터베이스 시스템의 간략한 역사

데이터베이스가 어떻게 진화했는지 이해하면 현재 시스템이 왜 그렇게 작동하는지 이해하는 데 도움이 됩니다.

### 타임라인

```
1960s         1970s         1980s         1990s         2000s         2010s+
  │             │             │             │             │             │
  ▼             ▼             ▼             ▼             ▼             ▼
Flat Files  Relational    SQL Standard  Object-       NoSQL         NewSQL
Hierarchical Model        Commercial    Relational    Movement      Distributed
Network     (Codd 1970)   RDBMS Boom    DBMS          (2009+)       HTAP
(IMS, IDMS)  System R      Oracle,       PostgreSQL    MongoDB       CockroachDB
             INGRES        DB2, SQL      Informix      Cassandra     TiDB
                           Server                      Redis         Google Spanner
```

### 시대 1: 전(前)-관계형 (1960년대)

**계층 모델(Hierarchical Model)** (IBM IMS, 1966):
- 데이터를 트리 구조로 조직 (부모-자식 관계)
- 계층을 따른 사전 정의된 쿼리에 빠름
- 융통성 없음: 트리 구조 변경이 애플리케이션 재작성을 요구

```
        Department
       /          \
  Employee      Project
    |
  Dependent
```

**네트워크 모델(Network Model)** (CODASYL, 1969):
- 계층 모델의 일반화: 레코드가 여러 부모를 가질 수 있음
- 더 융통성 있지만, 복잡한 포인터 기반 탐색
- 프로그래머가 정확한 접근 경로를 알아야 함

```
  Student ──────── Course
     │    \  /  \    │
     │     \/    \   │
     │     /\     \  │
     ▼    /  \     ▼ ▼
  Advisor    Enrollment
```

### 시대 2: 관계형 혁명 (1970년대)

**Edgar F. Codd**는 1970년 "A Relational Model of Data for Large Shared Data Banks"를 발표하며, 데이터를 수학적 관계(테이블)로 표현할 것을 제안했습니다.

주요 혁신:
- **선언적 쿼리(Declarative queries)**: 어떻게 얻을지가 아닌 *무엇을* 원하는지 명시
- **데이터 독립성(Data independence)**: 물리적 저장 변경이 애플리케이션에 영향을 주지 않음
- **수학적 기초(Mathematical foundation)**: 관계 대수와 관계 계산법
- **단순성(Simplicity)**: 모든 데이터가 테이블로 균일하게 표현됨

**System R** (IBM, 1974-1979): 관계형 모델의 첫 구현. SQL(원래 SEQUEL)을 도입.

**INGRES** (UC Berkeley, 1973-1979): 동시 독립 구현. QUEL 쿼리 언어 사용.

### 시대 3: 상용 RDBMS (1980년대-1990년대)

관계형 모델이 가치를 입증하고 상용 시스템이 확산되었습니다:

| 시스템 | 연도 | 주요 특징 |
|--------|------|------------------|
| Oracle | 1979 | 최초의 상용 RDBMS |
| IBM DB2 | 1983 | System R 후속작 |
| SQL Server | 1989 | Microsoft의 RDBMS |
| PostgreSQL | 1996 | 오픈소스, 확장 가능 |
| MySQL | 1995 | 오픈소스, 웹 친화적 |

**SQL 표준화**:
- SQL-86: 첫 ANSI 표준
- SQL-92: 주요 개정 (서브쿼리, JOIN, CASE)
- SQL:1999: 재귀 쿼리, 트리거, 객체-관계형 기능
- SQL:2003: XML, 윈도우 함수, 시퀀스
- SQL:2011: 시간 데이터
- SQL:2016: JSON 지원
- SQL:2023: 속성 그래프 쿼리, 다차원 배열

### 시대 4: 객체-관계형 및 그 이상 (1990년대-2000년대)

**객체-관계형 DBMS(Object-Relational DBMS)**는 관계형 모델을 다음으로 확장했습니다:
- 사용자 정의 타입 및 함수
- 상속
- 복잡한 객체 (배열, 중첩 테이블)
- PostgreSQL이 대표적인 예

**객체 지향 DBMS(Object-Oriented DBMS, OODBMS)**:
- 객체를 직접 저장 (임피던스 불일치 없음)
- 주류로 채택되지 않음
- 예시: ObjectStore, db4o

### 시대 5: NoSQL 운동 (2009+)

대용량 데이터를 처리해야 하는 웹 규모 기업들에 의해 주도됨:

| 유형 | 예시 | 최적의 용도 |
|------|----------|----------|
| Key-Value | Redis, DynamoDB | 캐싱, 세션 |
| Document | MongoDB, CouchDB | 반정형 데이터 |
| Column-Family | Cassandra, HBase | 시계열, 분석 |
| Graph | Neo4j, JanusGraph | 연결된 데이터, 소셜 네트워크 |

주요 개념:
- **CAP 정리(CAP Theorem)**: 일관성(Consistency), 가용성(Availability), 분할 허용(Partition tolerance)을 동시에 보장할 수 없음
- **BASE**: Basically Available, Soft state, Eventually consistent (ACID와 대비)
- **스키마 유연성**: schema-on-read vs schema-on-write

### 시대 6: NewSQL 및 분산 SQL (2010년대+)

NoSQL의 확장성과 관계형 시스템의 보장을 결합:

| 시스템 | 접근법 |
|--------|----------|
| Google Spanner | 전역 분산, TrueTime |
| CockroachDB | 분산 PostgreSQL 호환 |
| TiDB | MySQL 호환, HTAP |
| YugabyteDB | PostgreSQL 호환 분산 |
| Vitess | MySQL 샤딩 미들웨어 |

---

## 5. 3단계 스키마 아키텍처

**3단계 스키마 아키텍처(Three-schema architecture)** (또는 3수준 아키텍처)는 데이터베이스 시스템을 세 개의 추상화 수준으로 분리합니다. 이 분리가 데이터 독립성의 기초입니다.

### 세 가지 수준

```
┌──────────────────────────────────────────────────────┐
│                  External Level                       │
│            (Individual User Views)                    │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │  View 1  │  │  View 2  │  │  View 3  │  ...     │
│  │(Students)│  │(Faculty) │  │(Finance) │          │
│  └──────────┘  └──────────┘  └──────────┘          │
│         │            │             │                 │
│         └────────────┼─────────────┘                 │
│                      │                               │
│            External/Conceptual Mapping               │
└──────────────────────┼───────────────────────────────┘
                       │
┌──────────────────────┼───────────────────────────────┐
│                      ▼                               │
│              Conceptual Level                         │
│         (Community User View)                         │
│                                                      │
│  Describes the WHAT:                                 │
│  - All entities, attributes, relationships           │
│  - Integrity constraints                             │
│  - Security and authorization rules                  │
│                                                      │
│            Conceptual/Internal Mapping               │
└──────────────────────┼───────────────────────────────┘
                       │
┌──────────────────────┼───────────────────────────────┐
│                      ▼                               │
│               Internal Level                          │
│          (Physical Storage View)                      │
│                                                      │
│  Describes the HOW:                                  │
│  - File organization (heap, sorted, hashed)          │
│  - Index structures (B+ tree, hash index)            │
│  - Record layout and compression                     │
│  - Buffer management policies                        │
│                                                      │
└──────────────────────────────────────────────────────┘
```

### 외부 수준(External Level, View Level)

**외부 수준**은 특정 사용자나 애플리케이션과 관련된 데이터베이스의 일부를 설명합니다. 서로 다른 사용자는 동일한 기본 데이터의 서로 다른 뷰를 봅니다.

```sql
-- 등록 담당자를 위한 뷰 (학업 정보 표시)
CREATE VIEW registrar_view AS
SELECT s.student_id, s.name, s.major, s.gpa,
       c.course_id, c.title, e.grade
FROM students s
JOIN enrollments e ON s.student_id = e.student_id
JOIN courses c ON e.course_id = c.course_id;

-- 재정 지원을 위한 뷰 (재정 정보 표시)
CREATE VIEW financial_aid_view AS
SELECT s.student_id, s.name, s.financial_status,
       s.scholarship_amount, s.loan_balance
FROM students s;

-- 학생 포털을 위한 뷰 (제한된 자기 뷰)
CREATE VIEW student_portal_view AS
SELECT s.name, s.gpa, c.title, e.grade
FROM students s
JOIN enrollments e ON s.student_id = e.student_id
JOIN courses c ON e.course_id = c.course_id
WHERE s.student_id = CURRENT_USER_ID();
```

### 개념 수준(Conceptual Level, Logical Level)

**개념 수준**은 모든 사용자를 위한 전체 데이터베이스의 논리적 구조를 설명합니다. 포함 내용:

- 모든 엔티티 타입과 그 속성
- 엔티티 간 관계
- 무결성 제약조건
- 보안 및 권한 부여 정보

```
개념 스키마:

STUDENT(student_id PK, name, major, gpa, financial_status,
        scholarship_amount, loan_balance)

COURSE(course_id PK, title, credits, department)

ENROLLMENT(student_id FK, course_id FK, grade, semester)
    PK(student_id, course_id, semester)

INSTRUCTOR(instructor_id PK, name, department, salary)

TEACHES(instructor_id FK, course_id FK, semester)
    PK(instructor_id, course_id, semester)

제약조건:
  - gpa BETWEEN 0.0 AND 4.0
  - credits > 0
  - grade IN ('A+','A','A-','B+','B','B-','C+','C','C-','D','F')
```

### 내부 수준(Internal Level, Physical Level)

**내부 수준**은 데이터가 디스크에 물리적으로 어떻게 저장되는지 설명합니다:

```
내부 스키마 (개념적 표현):

STUDENT 테이블:
  - 저장: 오버플로 페이지를 가진 힙 파일
  - 기본 인덱스: student_id의 B+ 트리 (클러스터형)
  - 보조 인덱스: name의 해시 인덱스
  - 레코드 형식: 고정 길이 (student_id: 8 bytes,
                   name: 50 bytes VARCHAR, major: 30 bytes, ...)
  - 압축: 'major' 열에 대한 사전 인코딩
  - 파티션: student_id의 범위 파티션

ENROLLMENT 테이블:
  - 저장: (student_id, course_id)로 정렬된 파일
  - 인덱스: (student_id, course_id, semester)의 복합 B+ 트리
  - 레코드 형식: 고정 길이, 레코드당 32 바이트
```

---

## 6. 데이터 독립성

**데이터 독립성(Data independence)**은 다음 상위 수준의 스키마를 변경하지 않고 한 수준의 스키마를 변경할 수 있는 능력입니다. 이것이 3단계 스키마 아키텍처의 주요 이점입니다.

### 논리적 데이터 독립성

외부 스키마나 애플리케이션 프로그램을 변경하지 않고 **개념 스키마**를 변경할 수 있는 능력.

```
예: STUDENT 테이블에 열 추가

이전:
  STUDENT(student_id, name, major, gpa)

이후:
  STUDENT(student_id, name, major, gpa, email, phone)

영향:
  - email/phone을 사용하지 않는 외부 뷰: 변경 불필요
  - name과 gpa만 쿼리하는 애플리케이션: 변경 불필요
  - email/phone이 필요한 애플리케이션만 업데이트해야 함
```

논리적 데이터 독립성의 이점을 받는 변경:
- 테이블에 속성 추가 또는 제거
- 테이블을 둘로 분할 (원래 모습을 유지하기 위한 뷰 사용)
- 두 테이블을 하나로 결합
- 새로운 관계 또는 엔티티 타입 추가

### 물리적 데이터 독립성

개념 스키마나 외부 뷰를 변경하지 않고 **내부 스키마**를 변경할 수 있는 능력.

```
예: 인덱스 구조 변경

이전:
  STUDENT.name이 B+ 트리로 인덱싱됨

이후:
  STUDENT.name이 해시 인덱스로 인덱싱됨

영향:
  - 개념 스키마: 변경 없음 (여전히 name 열을 가진 STUDENT 테이블)
  - 외부 뷰: 변경 없음
  - 애플리케이션 프로그램: 변경 없음
  - 쿼리 성능 특성만 변경됨
```

물리적 데이터 독립성의 이점을 받는 변경:
- 파일 조직 변경 (힙에서 정렬됨으로)
- 인덱스 추가 또는 제거
- 데이터를 다른 저장 장치로 이동
- 버퍼 관리 전략 변경
- 데이터를 다르게 압축하거나 파티셔닝

### 데이터 독립성이 중요한 이유

```
데이터 독립성 없이:          데이터 독립성과 함께:

  App 1 ──► 물리적 저장         App 1 ──► View 1 ─┐
  App 2 ──► 물리적 저장         App 2 ──► View 2 ─┤
  App 3 ──► 물리적 저장         App 3 ──► View 3 ─┤
                                                        ▼
  저장 형식 변경?               개념 스키마
  → 모든 애플리케이션 재작성!              │
                                           ▼
                                     내부 스키마

                                     저장 형식 변경?
                                     → 애플리케이션 영향 없음!
```

### 실제 현실

실제로 완전한 데이터 독립성을 달성하는 것은 어렵습니다:

- **논리적 독립성**은 물리적 독립성보다 달성하기 어려움
- 성능 고려사항이 종종 추상화 계층을 통해 누출됨
- ORM 프레임워크는 부분적인 논리적 독립성을 제공
- 뷰는 외부 수준 독립성을 제공하지만 성능 영향이 있을 수 있음

---

## 7. ANSI/SPARC 아키텍처

**ANSI/SPARC** (American National Standards Institute / Standards Planning and Requirements Committee) 아키텍처는 1975년에 제안되어 3단계 스키마 접근법을 공식화했습니다. 어떤 DBMS도 이것을 정확하게 구현하지는 않지만, 모든 현대 데이터베이스 시스템의 개념적 청사진으로 남아 있습니다.

### 전체 아키텍처 다이어그램

```
┌───────────────────────────────────────────────────────────────┐
│                         USERS                                  │
│                                                               │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐ │
│  │ Casual   │  │ App     │  │ Parametric│  │ DBA            │ │
│  │ User     │  │ Prog    │  │ User     │  │                │ │
│  └────┬─────┘  └────┬────┘  └─────┬────┘  └───────┬────────┘ │
│       │              │             │               │           │
└───────┼──────────────┼─────────────┼───────────────┼──────────┘
        │              │             │               │
        ▼              ▼             ▼               ▼
┌───────────────────────────────────────────────────────────────┐
│                 EXTERNAL LEVEL                                 │
│                                                               │
│  External     External     External     DDL              │
│  Schema 1     Schema 2     Schema 3     Compiler             │
│       │            │            │            │                │
│       └────────────┴────────────┘            │                │
│                    │                         │                │
│          External/Conceptual                 │                │
│              Mapping                         │                │
└────────────────────┼─────────────────────────┼────────────────┘
                     │                         │
                     ▼                         ▼
┌───────────────────────────────────────────────────────────────┐
│                CONCEPTUAL LEVEL                                │
│                                                               │
│            Conceptual Schema                                  │
│            (defined by DBA)                                   │
│                    │                                          │
│          Conceptual/Internal                                  │
│              Mapping                                          │
└────────────────────┼──────────────────────────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────────────────────────┐
│                 INTERNAL LEVEL                                 │
│                                                               │
│            Internal Schema                                    │
│       (storage structures, indexes)                           │
│                    │                                          │
│             Internal/Physical                                 │
│                Mapping                                        │
└────────────────────┼──────────────────────────────────────────┘
                     │
                     ▼
              ┌──────────────┐
              │   STORED     │
              │   DATABASE   │
              └──────────────┘
```

### 수준 간 매핑

아키텍처는 수준 간 요청과 결과를 변환하는 **매핑(mappings)**을 정의합니다:

1. **외부/개념 매핑(External/Conceptual Mapping)**: 사용자의 뷰와 개념 스키마 간 변환
2. **개념/내부 매핑(Conceptual/Internal Mapping)**: 논리적 구조와 물리적 저장 간 변환
3. **내부/물리적 매핑(Internal/Physical Mapping)**: DBMS 내부 구조와 OS 파일 시스템 간 변환

```python
# 매핑의 개념적 설명 (실제 DBMS 코드가 아님)

# 외부 스키마: 사용자가 보는 "student_summary" 뷰
class StudentSummaryView:
    student_name: str      # STUDENT.first_name + ' ' + STUDENT.last_name으로 매핑
    course_count: int      # ENROLLMENT의 COUNT(*)로 매핑
    average_grade: float   # AVG(ENROLLMENT.grade_points)로 매핑

# 개념 스키마: 실제 테이블
class Student:
    student_id: int        # PK
    first_name: str
    last_name: str
    major: str

class Enrollment:
    student_id: int        # FK -> Student
    course_id: str         # FK -> Course
    grade_points: float

# 내부 스키마: 물리적 저장
class StudentStorage:
    file_type: str = "B+ tree clustered on student_id"
    record_size: int = 128  # bytes
    page_size: int = 8192   # bytes
    records_per_page: int = 64
```

### 주요 인터페이스

| 인터페이스 | 사이 | 목적 |
|-----------|---------|---------|
| **DDL** (Data Definition Language) | DBA와 개념 수준 | 스키마 정의: CREATE TABLE, ALTER TABLE |
| **VDL** (View Definition Language) | 사용자와 외부 수준 | 뷰 정의: CREATE VIEW |
| **SDL** (Storage Definition Language) | DBA와 내부 수준 | 저장 정의: CREATE INDEX, TABLESPACE |
| **DML** (Data Manipulation Language) | 사용자와 데이터 | 데이터 조작: SELECT, INSERT, UPDATE, DELETE |

---

## 8. 데이터 모델

**데이터 모델(Data model)**은 데이터베이스의 구조, 연산, 제약조건을 설명하기 위한 개념의 집합입니다. 서로 다른 데이터 모델은 서로 다른 수준의 추상화를 제공합니다.

### 데이터 모델의 범주

```
높은 수준                                              낮은 수준
(개념적)                                           (물리적)
    │                                                      │
    ▼                                                      ▼
┌──────────┐    ┌──────────────┐    ┌──────────────┐   ┌──────────┐
│ ER Model │    │  Relational  │    │  Record-based│   │ Physical │
│ UML      │    │  Model       │    │  Models      │   │ Data     │
│ ORM      │    │              │    │  (Network,   │   │ Model    │
│          │    │              │    │   Hierarch.) │   │          │
└──────────┘    └──────────────┘    └──────────────┘   └──────────┘
  개념적          표현적               구현적             물리적
  (무엇)          (무엇 + 약간의 방법)    (방법)              (정확히 방법)
```

### 주요 데이터 모델 요약

| 데이터 모델 | 구조 | 쿼리 언어 | 시대 |
|------------|-----------|---------------|-----|
| **Hierarchical** | 트리 | DL/1 | 1960s |
| **Network** | 그래프 (CODASYL) | Navigational | 1960s |
| **Relational** | 테이블 (관계) | SQL | 1970s+ |
| **Entity-Relationship** | 엔티티 및 관계 | N/A (설계 도구) | 1976 |
| **Object-Oriented** | 객체, 클래스 | OQL | 1990s |
| **Object-Relational** | 확장 테이블 | SQL + extensions | 1990s |
| **Document** | JSON/BSON 문서 | MongoDB Query | 2000s |
| **Key-Value** | 키-값 쌍 | GET/SET | 2000s |
| **Column-Family** | 열 그룹 | CQL | 2000s |
| **Graph** | 노드 및 에지 | Cypher, SPARQL | 2000s |

### 관계형 모델 (미리보기)

관계형 모델이 이 과정의 초점이므로, 간략한 미리보기를 제공합니다:

```
관계(Relation, 테이블):
  - 튜플(행)의 집합
  - 각 튜플은 동일한 속성(열) 집합을 가짐
  - 각 속성은 도메인(허용 값)을 가짐
  - 튜플의 순서는 중요하지 않음
  - 속성의 순서는 중요하지 않음
  - 중복 튜플 없음
  - 각 셀은 원자 값을 포함

    ┌─────────────────────────────────────────┐
    │           EMPLOYEE (관계)                 │
    ├──────────┬──────────┬───────┬───────────┤
    │  emp_id  │   name   │  age  │   dept    │   ← 속성
    ├──────────┼──────────┼───────┼───────────┤
    │  E001    │  Alice   │  29   │  CS       │   ← 튜플 1
    │  E002    │  Bob     │  35   │  EE       │   ← 튜플 2
    │  E003    │  Carol   │  42   │  CS       │   ← 튜플 3
    └──────────┴──────────┴───────┴───────────┘

    Domain(emp_id) = {E001, E002, ..., E999}
    Domain(age) = positive integers
    Domain(dept) = {CS, EE, ME, CE, ...}
```

---

## 9. 데이터베이스 사용자 및 역할

데이터베이스 시스템은 다양한 요구와 기술적 전문성 수준을 가진 많은 유형의 사용자에게 서비스를 제공합니다.

### 사용자 분류

```
┌─────────────────────────────────────────────────────────┐
│                     Database Users                       │
│                                                         │
│  ┌─────────────────────┐  ┌──────────────────────────┐ │
│  │    Actors on the     │  │   Actors Behind the      │ │
│  │      Scene           │  │      Scene               │ │
│  │                     │  │                          │ │
│  │  - Database Admin   │  │  - DBMS Designers        │ │
│  │  - Database Designer│  │  - Tool Developers       │ │
│  │  - End Users        │  │  - System Administrators │ │
│  │  - App Programmers  │  │                          │ │
│  └─────────────────────┘  └──────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### 현장의 행위자

| 역할 | 책임 | 사용 도구 |
|------|-----------------|------------|
| **데이터베이스 관리자(Database Administrator, DBA)** | 스키마 정의, 저장 구조, 접근 제어, 백업/복구, 성능 튜닝 | DDL, DCL, 모니터링 도구 |
| **데이터베이스 설계자(Database Designer)** | 개념적 및 논리적 설계, ER 모델링, 정규화, 뷰 정의 | ER 도구, UML, CASE 도구 |
| **애플리케이션 프로그래머(Application Programmer)** | 데이터베이스에 접근하는 프로그램 작성, 호스트 언어에 SQL 임베드 | SQL, ORM, API |
| **최종 사용자(End Users)** | 애플리케이션 또는 임시 쿼리를 통해 데이터베이스 쿼리 및 업데이트 | 폼, 리포트, SQL |

### 최종 사용자의 유형

```
최종 사용자
    │
    ├── 임시 사용자(Casual Users)
    │     데이터베이스에 가끔 접근
    │     임시 쿼리 사용 (SQL 또는 GUI)
    │     예시: 월간 보고서를 실행하는 관리자
    │
    ├── 일반 사용자(Naive, Parametric Users)
    │     사전 작성된 애플리케이션을 반복적으로 사용
    │     쿼리를 작성하지 않음
    │     예시: 은행 창구 직원, 항공 예약 담당자
    │
    ├── 정교한 사용자(Sophisticated Users)
    │     DBMS 기능에 익숙
    │     복잡한 쿼리 작성
    │     예시: 데이터 분석가, 과학자
    │
    └── 독립 사용자(Standalone Users)
          개인 데이터베이스
          기성 소프트웨어 사용
          예시: 세금 준비 소프트웨어 사용자
```

### DBA 상세 설명

데이터베이스 관리자는 데이터베이스 관리의 중앙 권한입니다:

```python
# 일반적인 DBA 책임 (개념적)

class DBA:
    """데이터베이스 관리자 책임"""

    def schema_management(self):
        """데이터베이스 스키마 정의 및 수정"""
        # CREATE TABLE, ALTER TABLE, CREATE INDEX
        # 다양한 사용자 그룹을 위한 뷰 정의
        # 스키마 마이그레이션 관리

    def security_management(self):
        """데이터베이스에 대한 접근 제어"""
        # GRANT/REVOKE 권한
        # 역할 생성 및 사용자 할당
        # 접근 로그 감사

    def performance_tuning(self):
        """데이터베이스 성능 최적화"""
        # 쿼리 실행 계획 분석
        # 워크로드 기반 인덱스 생성/삭제
        # 버퍼 풀, 캐시 크기 구성
        # 대형 테이블 파티셔닝

    def backup_and_recovery(self):
        """데이터 지속성 보장"""
        # 정기 백업 예약 (전체, 증분)
        # 복구 절차 테스트
        # 트랜잭션 로그 관리
        # 재해 복구 처리

    def capacity_planning(self):
        """성장 계획"""
        # 디스크 사용 트렌드 모니터링
        # 미래 저장 요구 예측
        # 하드웨어 업그레이드 계획
```

---

## 10. 데이터베이스 시스템 아키텍처

현대 데이터베이스 시스템은 다양한 아키텍처 구성으로 배포될 수 있습니다.

### 중앙집중식 아키텍처

```
┌────────────────────────────────────────┐
│           Centralized System           │
│                                        │
│  ┌────────────────────────────────┐   │
│  │         Application +          │   │
│  │         DBMS Software          │   │
│  └────────────────────────────────┘   │
│                  │                     │
│  ┌────────────────────────────────┐   │
│  │          Database              │   │
│  └────────────────────────────────┘   │
│                                        │
│  사용자가 터미널을 통해 접근             │
└────────────────────────────────────────┘
```

### 클라이언트-서버 아키텍처

**2계층(Two-Tier):**
```
┌──────────┐  ┌──────────┐  ┌──────────┐
│ Client 1 │  │ Client 2 │  │ Client 3 │
│ (App +   │  │ (App +   │  │ (App +   │
│  UI)     │  │  UI)     │  │  UI)     │
└────┬─────┘  └────┬─────┘  └────┬─────┘
     │              │              │
     └──────────────┼──────────────┘
                    │  Network (SQL over TCP)
                    ▼
           ┌──────────────┐
           │   Database   │
           │    Server    │
           │  (DBMS +     │
           │   Database)  │
           └──────────────┘
```

**3계층(Three-Tier, 웹 아키텍처):**
```
┌──────────┐  ┌──────────┐  ┌──────────┐
│ Browser  │  │ Browser  │  │ Mobile   │
│ (Thin    │  │ (Thin    │  │ App      │
│  Client) │  │  Client) │  │          │
└────┬─────┘  └────┬─────┘  └────┬─────┘
     │              │              │
     └──────────────┼──────────────┘
                    │  HTTP/HTTPS
                    ▼
           ┌──────────────┐
           │ Application  │    계층 2: 비즈니스 로직
           │   Server     │    (웹 서버, API 서버)
           │ (Flask/      │
           │  Django/     │
           │  Express)    │
           └──────┬───────┘
                  │  SQL/Protocol
                  ▼
           ┌──────────────┐
           │   Database   │    계층 3: 데이터 관리
           │    Server    │
           │ (PostgreSQL/ │
           │  MySQL)      │
           └──────────────┘
```

### 분산 아키텍처

```
                    ┌──────────────┐
                    │   Client     │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   Router /   │
                    │   Coordinator│
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
      ┌──────────┐  ┌──────────┐  ┌──────────┐
      │  Node 1  │  │  Node 2  │  │  Node 3  │
      │ (Shard A)│  │ (Shard B)│  │ (Shard C)│
      │ +Replica │  │ +Replica │  │ +Replica │
      └──────────┘  └──────────┘  └──────────┘
              │            │            │
              └────────────┼────────────┘
                           │
                    Replication &
                    Consensus Protocol
```

### 클라우드 아키텍처

```
┌─────────────────────────────────────────────┐
│              Cloud Provider                  │
│                                             │
│  ┌─────────────────────────────────────┐   │
│  │         Managed Database            │   │
│  │  (RDS, Cloud SQL, Aurora, etc.)     │   │
│  │                                     │   │
│  │  ┌──────────┐  ┌──────────┐        │   │
│  │  │ Primary  │  │ Replica  │        │   │
│  │  │ Instance │──│ Instance │        │   │
│  │  └──────────┘  └──────────┘        │   │
│  │       │                             │   │
│  │  ┌──────────────────────────┐      │   │
│  │  │  Shared Storage Layer    │      │   │
│  │  │  (Distributed, Durable) │      │   │
│  │  └──────────────────────────┘      │   │
│  └─────────────────────────────────────┘   │
│                                             │
│  기능: 자동 백업, 스케일링,                   │
│  모니터링, 패칭, HA 장애조치                 │
└─────────────────────────────────────────────┘
```

---

## 11. 연습문제

### 개념적 질문

**연습문제 1.1**: 파일 기반 데이터 관리 접근법의 다섯 가지 단점을 나열하고 DBMS가 각각을 어떻게 해결하는지 설명하세요.

**연습문제 1.2**: 3단계 스키마 아키텍처 수준 간의 차이를 설명하세요. 대학 데이터베이스의 경우, 각 수준이 무엇을 설명하는지 예를 들어주세요.

**연습문제 1.3**: 다음 용어를 정의하세요:
- (a) 데이터 독립성(Data independence)
- (b) 데이터 추상화(Data abstraction)
- (c) 데이터 정의 언어(Data definition language)
- (d) 데이터 조작 언어(Data manipulation language)
- (e) 스키마 vs. 인스턴스(Schema vs. instance)

**연습문제 1.4**: 한 회사가 현재 모든 직원 데이터를 스프레드시트에 저장하고 있습니다. CEO는 데이터베이스 시스템으로 마이그레이션하고 싶어 합니다. 이 마이그레이션의 이점과 잠재적 과제를 설명하는 간단한 메모를 작성하세요 (5-7개 항목).

### 분석 질문

**연습문제 1.5**: 다음 변경 사항 각각을 (i) 외부, (ii) 개념, 또는 (iii) 내부 스키마의 수정이 필요한 것으로 분류하세요. 데이터 독립성이 보존되는지 설명하세요.

- (a) STUDENT 테이블에 새로운 인덱스가 추가됨
- (b) STUDENT 테이블에 새로운 열 `email`이 추가됨
- (c) STUDENT 테이블이 STUDENT_PERSONAL과 STUDENT_ACADEMIC으로 분할됨
- (d) 데이터베이스 파일이 HDD에서 SSD로 이동됨
- (e) 재정 지원 부서를 위한 새로운 뷰가 생성됨

**연습문제 1.6**: 각 데이터베이스 모델 쌍에 대해, 첫 번째 모델의 두 가지 장점과 두 가지 단점을 두 번째 모델과 비교하여 설명하세요:
- (a) 관계형 vs. 계층형
- (b) 문서 (NoSQL) vs. 관계형
- (c) 그래프 vs. 관계형

**연습문제 1.7**: 병원 데이터베이스 시스템의 다음 사용자를 역할별로 분류하세요 (DBA, 데이터베이스 설계자, 애플리케이션 프로그래머, 또는 최종 사용자 유형):
- (a) 환자 기록 시스템의 ER 다이어그램을 설계하는 사람
- (b) 태블릿 애플리케이션을 통해 환자 생체 신호를 입력하는 간호사
- (c) 야간 백업을 수행하고 쿼리 성능을 모니터링하는 IT 직원
- (d) 특정 진단을 가진 모든 환자를 찾기 위해 데이터베이스를 쿼리하는 의사
- (e) 환자 포털 웹 애플리케이션을 구축하는 프로그래머

### 실용 질문

**연습문제 1.8**: 세 가지 현대 DBMS (관계형 하나, 문서 하나, 그래프 하나)를 조사하고 비교하세요. 각각에 대해 다음을 식별하세요:
- 주요 데이터 모델
- 쿼리 언어
- ACID 지원 (전체, 부분, 또는 없음)
- 일반적인 사용 사례
- 3단계 스키마 아키텍처 지원 수준

**연습문제 1.9**: 다음 시나리오를 고려하세요:

> 전자상거래 회사가 현재 플랫 파일을 사용하여 제품 카탈로그 데이터 (product_catalog.csv), 고객 데이터 (customers.csv), 주문 데이터 (orders.csv)를 저장합니다. 하루에 약 1,000개의 주문을 처리하고 50,000개의 제품을 보유하고 있습니다.

이 데이터를 관계형 데이터베이스로 마이그레이션하기 위한 간단한 계획을 설계하세요. 포함 사항:
- 어떤 테이블을 생성할 것인가?
- 테이블 간 관계는 무엇인가?
- 파일 기반 접근법의 어떤 문제를 해결할 것인가?
- DBMS가 제공할 추가 기능은 무엇인가?

**연습문제 1.10**: ANSI/SPARC 아키텍처를 고려하세요. 사용자가 다음 SQL 쿼리를 실행할 때 각 수준에서 무엇이 일어나는지 설명하세요:

```sql
SELECT name, gpa FROM student_summary WHERE gpa > 3.5;
```

`student_summary`가 다음과 같이 정의된 뷰라고 가정하세요:
```sql
CREATE VIEW student_summary AS
SELECT student_id, first_name || ' ' || last_name AS name, gpa
FROM students;
```

외부 수준 (뷰 해결) -> 개념 수준 (논리적 계획) -> 내부 수준 (물리적 접근)을 통해 쿼리를 추적하세요.

---

**다음**: [관계형 모델](./02_Relational_Model.md)
