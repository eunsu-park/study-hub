# 관계 대수(Relational Algebra)

**이전**: [관계 모델](./02_Relational_Model.md) | **다음**: [ER 모델링](./04_ER_Modeling.md)

---

관계 대수는 관계 모델의 형식적 질의 언어입니다. 하나 또는 두 개의 관계를 입력으로 받아 새로운 관계를 출력으로 생성하는 연산자의 집합을 제공합니다. 관계 대수를 이해하는 것은 SQL 쿼리 처리의 기반이 되기 때문에 필수적입니다. 모든 SQL 쿼리는 내부적으로 관계 대수 표현식으로 변환되며, 쿼리 최적화기는 이를 변환하고 개선할 수 있습니다.

## 목차

1. [관계 대수 개요](#1-관계-대수-개요)
2. [단항 연산](#2-단항-연산)
3. [집합 연산](#3-집합-연산)
4. [이항 연산: 카티션 곱과 조인](#4-이항-연산-카티션-곱과-조인)
5. [나눗셈](#5-나눗셈)
6. [추가 연산](#6-추가-연산)
7. [쿼리 트리와 대수적 최적화](#7-쿼리-트리와-대수적-최적화)
8. [관계 해석 (간단한 소개)](#8-관계-해석-간단한-소개)
9. [SQL과의 동치성](#9-sql과의-동치성)
10. [완전한 작업 예제](#10-완전한-작업-예제)
11. [연습 문제](#11-연습-문제)

---

## 1. 관계 대수 개요

### 관계 대수란?

관계 대수는 **절차적(procedural)** 쿼리 언어입니다. 원하는 결과를 계산하기 위한 일련의 연산을 기술합니다. 각 연산은 하나 이상의 관계를 받아 새로운 관계를 생성합니다.

```
속성:
  - 폐쇄성(Closure): 모든 연산의 결과는 관계입니다
                     (연산의 조합을 가능하게 함)
  - 집합 의미론: 관계는 집합입니다 (중복 튜플 없음)
  - 기반: 집합 이론과 1차 논리를 기반으로 함
```

### 연산의 범주

```
┌─────────────────────────────────────────────────────────────┐
│              관계 대수 연산(Relational Algebra Operations)      │
│                                                             │
│  기본(다른 연산으로 표현 불가):                                 │
│    σ  선택(Selection)          (행 필터링)                    │
│    π  프로젝션(Projection)      (열 선택)                     │
│    ρ  재명명(Rename)            (관계/속성 이름 변경)          │
│    ∪  합집합(Union)             (두 관계 결합)                 │
│    −  차집합(Set Difference)    (R에 있지만 S에 없는 튜플)     │
│    ×  카티션 곱(Cartesian Product)  (모든 조합)              │
│                                                             │
│  유도(기본 연산으로 표현 가능):                                 │
│    ⋈  조인(Join) (다양한 유형)                                │
│    ∩  교집합(Intersection)                                   │
│    ÷  나눗셈(Division)                                       │
│    ←  대입(Assignment)                                      │
│    δ  중복 제거(Duplicate elimination) (백 의미론용)          │
│    γ  그룹화/집계(Grouping/Aggregation)                      │
│    τ  정렬(Sorting)                                         │
└─────────────────────────────────────────────────────────────┘
```

### 실행 예제 데이터베이스

이 강의 전체에서 다음 샘플 데이터베이스를 사용합니다:

```
STUDENT:
┌──────┬───────────┬──────┬──────┐
│ sid  │ name      │ year │ dept │
├──────┼───────────┼──────┼──────┤
│ S001 │ Alice     │  3   │ CS   │
│ S002 │ Bob       │  2   │ CS   │
│ S003 │ Carol     │  4   │ EE   │
│ S004 │ Dave      │  3   │ ME   │
│ S005 │ Eve       │  1   │ CS   │
└──────┴───────────┴──────┴──────┘

COURSE:
┌───────┬──────────────────────┬─────────┬──────┐
│ cid   │ title                │ credits │ dept │
├───────┼──────────────────────┼─────────┼──────┤
│ CS101 │ Intro to CS          │ 3       │ CS   │
│ CS301 │ Database Theory      │ 3       │ CS   │
│ CS401 │ Machine Learning     │ 4       │ CS   │
│ EE201 │ Circuit Analysis     │ 3       │ EE   │
│ MA101 │ Calculus I           │ 4       │ MA   │
└───────┴──────────────────────┴─────────┴──────┘

ENROLLMENT:
┌──────┬───────┬───────┐
│ sid  │ cid   │ grade │
├──────┼───────┼───────┤
│ S001 │ CS101 │ A     │
│ S001 │ CS301 │ A+    │
│ S001 │ MA101 │ B+    │
│ S002 │ CS101 │ B     │
│ S002 │ CS301 │ A-    │
│ S003 │ EE201 │ A     │
│ S003 │ CS101 │ B+    │
│ S004 │ CS101 │ C     │
│ S005 │ CS101 │ A     │
│ S005 │ MA101 │ A-    │
└──────┴───────┴───────┘

INSTRUCTOR:
┌──────┬──────────┬──────┬────────┐
│ iid  │ name     │ dept │ salary │
├──────┼──────────┼──────┼────────┤
│ I001 │ Prof. Kim│ CS   │ 95000  │
│ I002 │ Prof. Lee│ CS   │ 88000  │
│ I003 │ Prof. Park│ EE  │ 92000  │
│ I004 │ Prof. Choi│ MA  │ 85000  │
└──────┴──────────┴──────┴────────┘
```

---

## 2. 단항 연산

단항 연산은 단일 관계를 입력으로 받습니다.

### 선택 (sigma)

**선택(Selection)** 연산은 조건(predicate)을 기반으로 행을 필터링합니다.

```
표기법:   σ_조건(R)

출력:     조건을 만족하는 R의 튜플만을 포함하는 관계.

스키마:   R과 동일 (모든 속성 유지)
```

**형식적 정의:**

```
σ_조건(R) = { t | t ∈ R  AND  condition(t)이 TRUE }
```

**예제:**

```
1. CS 학생 선택:

   σ_{dept='CS'}(STUDENT)

   결과:
   ┌──────┬───────┬──────┬──────┐
   │ sid  │ name  │ year │ dept │
   ├──────┼───────┼──────┼──────┤
   │ S001 │ Alice │  3   │ CS   │
   │ S002 │ Bob   │  2   │ CS   │
   │ S005 │ Eve   │  1   │ CS   │
   └──────┴───────┴──────┴──────┘


2. 3학년 CS 학생 선택:

   σ_{dept='CS' AND year=3}(STUDENT)

   결과:
   ┌──────┬───────┬──────┬──────┐
   │ sid  │ name  │ year │ dept │
   ├──────┼───────┼──────┼──────┤
   │ S001 │ Alice │  3   │ CS   │
   └──────┴───────┴──────┴──────┘


3. 4학점 과목 선택:

   σ_{credits=4}(COURSE)

   결과:
   ┌───────┬──────────────────┬─────────┬──────┐
   │ cid   │ title            │ credits │ dept │
   ├───────┼──────────────────┼─────────┼──────┤
   │ CS401 │ Machine Learning │ 4       │ CS   │
   │ MA101 │ Calculus I       │ 4       │ MA   │
   └───────┴──────────────────┴─────────┴──────┘
```

**선택 조건에 사용 가능한 것:**
- 비교 연산자: =, ≠, <, >, ≤, ≥
- 논리 연결사: AND (∧), OR (∨), NOT (¬)
- 속성 이름과 상수

### 프로젝션 (pi)

**프로젝션(Projection)** 연산은 특정 열(속성)을 선택하고 중복을 제거합니다.

```
표기법:   π_{속성_목록}(R)

출력:     지정된 속성만을 포함하는 관계,
          중복 튜플 제거.

스키마:   속성_목록의 속성만
```

**형식적 정의:**

```
π_{A1, A2, ..., Ak}(R) = { <t[A1], t[A2], ..., t[Ak]> | t ∈ R }
```

**예제:**

```
1. 학생 이름과 학과 프로젝션:

   π_{name, dept}(STUDENT)

   결과:
   ┌───────┬──────┐
   │ name  │ dept │
   ├───────┼──────┤
   │ Alice │ CS   │
   │ Bob   │ CS   │
   │ Carol │ EE   │
   │ Dave  │ ME   │
   │ Eve   │ CS   │
   └───────┴──────┘


2. STUDENT에서 중복 없는 학과 프로젝션:

   π_{dept}(STUDENT)

   결과:
   ┌──────┐
   │ dept │
   ├──────┤
   │ CS   │
   │ EE   │
   │ ME   │
   └──────┘
   (참고: CS는 중복이 제거되어 한 번만 나타남)


3. 선택과 프로젝션 조합:

   "CS 학생의 이름 찾기"
   π_{name}(σ_{dept='CS'}(STUDENT))

   1단계: σ_{dept='CS'}(STUDENT) → {Alice/CS, Bob/CS, Eve/CS}
   2단계: π_{name}(...) → {Alice, Bob, Eve}

   결과:
   ┌───────┐
   │ name  │
   ├───────┤
   │ Alice │
   │ Bob   │
   │ Eve   │
   └───────┘
```

### 재명명 (rho)

**재명명(Rename)** 연산은 관계 및/또는 그 속성의 이름을 변경합니다.

```
표기법:   ρ_{S(B1, B2, ..., Bn)}(R)

          관계 R을 S로 재명명하고 속성을 B1, B2, ..., Bn으로 변경

축약형:   ρ_S(R)          -- 관계만 재명명
          ρ_{(B1,...,Bn)}(R) -- 속성만 재명명
```

**예제:**

```
1. STUDENT를 S로 재명명:

   ρ_S(STUDENT)
   → 동일한 튜플이지만 관계는 이제 S로 호출됨

2. 자기 조인 준비를 위한 재명명:

   ρ_{S1(sid1, name1, year1, dept1)}(STUDENT)
   ρ_{S2(sid2, name2, year2, dept2)}(STUDENT)

   이제 속성 이름 충돌 없이 S1과 S2를 조인할 수 있습니다.

3. 실용적 사용 - 같은 학과의 학생 쌍 찾기:

   σ_{S1.dept = S2.dept AND S1.sid < S2.sid}(
       ρ_{S1(sid1, name1, year1, dept1)}(STUDENT) ×
       ρ_{S2(sid2, name2, year2, dept2)}(STUDENT)
   )
```

---

## 3. 집합 연산

집합 연산은 **합병 호환(union-compatible)** (또는 타입 호환) 관계를 요구합니다. 동일한 수의 속성을 가져야 하며, 대응하는 속성들은 호환 가능한 도메인을 가져야 합니다.

```
합병 호환성:
  R(A1: D1, A2: D2, ..., An: Dn)
  S(B1: D1, B2: D2, ..., Bn: Dn)

  동일한 수의 속성 (n)과 호환 가능한 도메인.
  속성 이름은 다를 수 있음 (결과는 관례적으로 R의 이름 사용).
```

### 합집합

```
표기법:   R ∪ S

출력:     R, S, 또는 둘 다에 있는 모든 튜플.
          중복 제거됨.

R ∪ S = { t | t ∈ R  OR  t ∈ S }
```

**예제:**

```
CS_STUDENTS = π_{sid, name}(σ_{dept='CS'}(STUDENT))
┌──────┬───────┐
│ sid  │ name  │
├──────┼───────┤
│ S001 │ Alice │
│ S002 │ Bob   │
│ S005 │ Eve   │
└──────┴───────┘

YEAR3_STUDENTS = π_{sid, name}(σ_{year=3}(STUDENT))
┌──────┬───────┐
│ sid  │ name  │
├──────┼───────┤
│ S001 │ Alice │
│ S004 │ Dave  │
└──────┴───────┘

CS_STUDENTS ∪ YEAR3_STUDENTS =
┌──────┬───────┐
│ sid  │ name  │
├──────┼───────┤
│ S001 │ Alice │  ← 둘 다에 나타나지만 한 번만 나열
│ S002 │ Bob   │
│ S005 │ Eve   │
│ S004 │ Dave  │
└──────┴───────┘
```

### 차집합

```
표기법:   R − S  (또는 R \ S)

출력:     R에는 있지만 S에는 없는 튜플.

R − S = { t | t ∈ R  AND  t ∉ S }
```

**예제:**

```
"3학년이 아닌 CS 학생"

CS_STUDENTS − YEAR3_STUDENTS =
┌──────┬──────┐
│ sid  │ name │
├──────┼──────┤
│ S002 │ Bob  │
│ S005 │ Eve  │
└──────┴──────┘

참고: Alice (S001)는 YEAR3_STUDENTS에 있기 때문에 제거됨.
참고: R − S ≠ S − R  (차집합은 교환법칙이 성립하지 않음)

YEAR3_STUDENTS − CS_STUDENTS =
┌──────┬──────┐
│ sid  │ name │
├──────┼──────┤
│ S004 │ Dave │
└──────┴──────┘
```

### 교집합

```
표기법:   R ∩ S

출력:     R과 S 둘 다에 있는 튜플.

R ∩ S = { t | t ∈ R  AND  t ∈ S }

참고: R ∩ S = R − (R − S)  (교집합은 유도 가능)
```

**예제:**

```
"CS이면서 3학년인 학생"

CS_STUDENTS ∩ YEAR3_STUDENTS =
┌──────┬───────┐
│ sid  │ name  │
├──────┼───────┤
│ S001 │ Alice │
└──────┴───────┘
```

### 집합 연산 요약

```
     R           S                R ∪ S         R − S         R ∩ S

  ┌──────┐   ┌──────┐         ┌──────┐      ┌──────┐     ┌──────┐
  │ a    │   │ a    │         │ a    │      │ b    │     │ a    │
  │ b    │   │ a    │         │ b    │      │ c    │     │      │
  │ c    │   │ d    │         │ c    │      └──────┘     └──────┘
  └──────┘   └──────┘         │ d    │
                              └──────┘

  벤 다이어그램:
       ┌─────────────┐
       │  R          │
       │      ┌──────┼──────┐
       │      │ R∩S  │      │
       │      │      │   S  │
       └──────┼──────┘      │
              │             │
              └─────────────┘

  R ∪ S = 전체 음영 영역
  R − S = 왼쪽만 (겹치지 않음)
  R ∩ S = 겹치는 부분만
  S − R = 오른쪽만 (겹치지 않음)
```

---

## 4. 이항 연산: 카티션 곱과 조인

### 카티션 곱 (크로스 곱)

```
표기법:   R × S

출력:     R의 모든 튜플과 S의 모든 튜플의 결합.
          R이 n개의 튜플, S가 m개의 튜플이면, R × S는 n × m개의 튜플.
          R이 p개의 속성, S가 q개의 속성이면, R × S는 p + q개의 속성.
```

**형식적 정의:**

```
R × S = { <t_r, t_s> | t_r ∈ R  AND  t_s ∈ S }
```

**예제:**

```
A = {(a1, b1), (a2, b2)}     B = {(c1, d1), (c2, d2), (c3, d3)}

A × B =
┌────┬────┬────┬────┐
│ A  │ B  │ C  │ D  │
├────┼────┼────┼────┤
│ a1 │ b1 │ c1 │ d1 │
│ a1 │ b1 │ c2 │ d2 │
│ a1 │ b1 │ c3 │ d3 │
│ a2 │ b2 │ c1 │ d1 │
│ a2 │ b2 │ c2 │ d2 │
│ a2 │ b2 │ c3 │ d3 │
└────┴────┴────┴────┘

|A| = 2, |B| = 3, |A × B| = 2 × 3 = 6
```

카티션 곱 자체는 많은 의미 없는 조합을 생성하기 때문에 거의 유용하지 않습니다. 선택과 결합할 때(조인을 제공) 강력해집니다.

### 세타 조인

**세타 조인(Theta Join)**은 카티션 곱과 선택을 결합합니다:

```
표기법:   R ⋈_θ S   (여기서 θ는 조건)

정의:     R ⋈_θ S = σ_θ(R × S)

조건 θ는 =, ≠, <, >, ≤, ≥ 등의 비교를 사용할 수 있음
```

**예제:**

```
"같은 학과의 학생과 과목 찾기"

STUDENT ⋈_{STUDENT.dept = COURSE.dept} COURSE

동치: σ_{STUDENT.dept = COURSE.dept}(STUDENT × COURSE)

결과 (일부):
┌──────┬───────┬──────┬──────┬───────┬──────────────────┬─────────┬───────┐
│ sid  │ name  │ year │ s.dept│ cid  │ title            │ credits │c.dept │
├──────┼───────┼──────┼──────┼───────┼──────────────────┼─────────┼───────┤
│ S001 │ Alice │  3   │ CS   │ CS101 │ Intro to CS      │ 3       │ CS    │
│ S001 │ Alice │  3   │ CS   │ CS301 │ Database Theory  │ 3       │ CS    │
│ S001 │ Alice │  3   │ CS   │ CS401 │ Machine Learning │ 4       │ CS    │
│ S002 │ Bob   │  2   │ CS   │ CS101 │ Intro to CS      │ 3       │ CS    │
│ S002 │ Bob   │  2   │ CS   │ CS301 │ Database Theory  │ 3       │ CS    │
│ S002 │ Bob   │  2   │ CS   │ CS401 │ Machine Learning │ 4       │ CS    │
│ S003 │ Carol │  4   │ EE   │ EE201 │ Circuit Analysis │ 3       │ EE    │
│ S005 │ Eve   │  1   │ CS   │ CS101 │ Intro to CS      │ 3       │ CS    │
│ S005 │ Eve   │  1   │ CS   │ CS301 │ Database Theory  │ 3       │ CS    │
│ S005 │ Eve   │  1   │ CS   │ CS401 │ Machine Learning │ 4       │ CS    │
└──────┴───────┴──────┴──────┴───────┴──────────────────┴─────────┴───────┘
(Dave/ME는 일치하는 과목이 없어 결과에 없음)
```

### 동등 조인

**동등 조인(Equi-Join)**은 조건이 동등(=)만 사용하는 세타 조인입니다:

```
R ⋈_{R.A = S.B} S

모든 동등 조인은 세타 조인이지만, 모든 세타 조인이 동등 조인은 아님.
R.A > S.B를 사용하는 세타 조인은 동등 조인이 아님.
```

### 자연 조인

**자연 조인(Natural Join)**은 다음과 같은 특별한 동등 조인입니다:
1. 모든 공통 속성 이름에서 조인
2. 결과에서 중복 열 제거

```
표기법:   R ⋈ S   (아래 첨자 없음)

정의:     R과 S에서 같은 이름을 가진 모든 속성에서 조인한 후,
          중복 속성 열을 프로젝션으로 제거.
```

**예제:**

```
STUDENT ⋈ ENROLLMENT

공통 속성: sid

1단계: STUDENT.sid = ENROLLMENT.sid에서 동등 조인
2단계: 중복 sid 열 제거

결과:
┌──────┬───────┬──────┬──────┬───────┬───────┐
│ sid  │ name  │ year │ dept │ cid   │ grade │
├──────┼───────┼──────┼──────┼───────┼───────┤
│ S001 │ Alice │  3   │ CS   │ CS101 │ A     │
│ S001 │ Alice │  3   │ CS   │ CS301 │ A+    │
│ S001 │ Alice │  3   │ CS   │ MA101 │ B+    │
│ S002 │ Bob   │  2   │ CS   │ CS101 │ B     │
│ S002 │ Bob   │  2   │ CS   │ CS301 │ A-    │
│ S003 │ Carol │  4   │ EE   │ EE201 │ A     │
│ S003 │ Carol │  4   │ EE   │ CS101 │ B+    │
│ S004 │ Dave  │  3   │ ME   │ CS101 │ C     │
│ S005 │ Eve   │  1   │ CS   │ CS101 │ A     │
│ S005 │ Eve   │  1   │ CS   │ MA101 │ A-    │
└──────┴───────┴──────┴──────┴───────┴───────┘

경고: 관계가 의도치 않게 속성 이름을 공유할 때 자연 조인에 주의!

STUDENT(sid, name, year, dept) ⋈ COURSE(cid, title, credits, dept)
                          ^^^^                                ^^^^
이는 'dept'에서 조인하는데, 원하는 것이 아닐 수 있음!
학생을 그들의 학과의 과목과 매칭시킴.
```

### 외부 조인

표준 조인은 매칭되지 않는 튜플을 버립니다. **외부 조인(Outer Join)**은 NULL로 채워서 매칭되지 않는 튜플을 보존합니다.

```
세 가지 유형:

1. 왼쪽 외부 조인(LEFT OUTER JOIN) (⟕):
   왼쪽 관계의 모든 튜플 유지.
   오른쪽에 매칭이 없으면 NULL로 채움.

2. 오른쪽 외부 조인(RIGHT OUTER JOIN) (⟖):
   오른쪽 관계의 모든 튜플 유지.
   왼쪽에 매칭이 없으면 NULL로 채움.

3. 완전 외부 조인(FULL OUTER JOIN) (⟗):
   양쪽 관계의 모든 튜플 유지.
   필요에 따라 양쪽을 NULL로 채움.
```

**예제: 왼쪽 외부 조인**

```
STUDENT ⟕_{STUDENT.sid = ENROLLMENT.sid} ENROLLMENT

"등록이 없는 학생을 포함하여 모든 학생과 그들의 등록 찾기"

(데이터에서 S004/Dave가 등록이 없다고 가정.)

Dave가 등록이 없다면:

결과:
┌──────┬───────┬──────┬──────┬───────┬───────┐
│ sid  │ name  │ year │ dept │ cid   │ grade │
├──────┼───────┼──────┼──────┼───────┼───────┤
│ S001 │ Alice │  3   │ CS   │ CS101 │ A     │
│ S001 │ Alice │  3   │ CS   │ CS301 │ A+    │
│ S001 │ Alice │  3   │ CS   │ MA101 │ B+    │
│ S002 │ Bob   │  2   │ CS   │ CS101 │ B     │
│ S002 │ Bob   │  2   │ CS   │ CS301 │ A-    │
│ S003 │ Carol │  4   │ EE   │ EE201 │ A     │
│ S003 │ Carol │  4   │ EE   │ CS101 │ B+    │
│ S004 │ Dave  │  3   │ ME   │ NULL  │ NULL  │  ← NULL로 보존됨
│ S005 │ Eve   │  1   │ CS   │ CS101 │ A     │
│ S005 │ Eve   │  1   │ CS   │ MA101 │ A-    │
└──────┴───────┴──────┴──────┴───────┴───────┘
```

### 조인 비교 요약

```
┌──────────────────────────────────────────────────────────────────┐
│                        조인 유형                                   │
│                                                                  │
│  카티션 곱         R × S      모든 조합 (n × m 튜플)              │
│  세타 조인         R ⋈_θ S    카티션 + θ 선택                    │
│  동등 조인         R ⋈_{=} S  동등만 사용하는 세타 조인           │
│  자연 조인         R ⋈ S      공통 속성에서 동등 조인,            │
│                                중복 제거                          │
│  왼쪽 외부 조인    R ⟕ S      자연 + 매칭 안된 R 유지            │
│  오른쪽 외부 조인  R ⟖ S      자연 + 매칭 안된 S 유지            │
│  완전 외부 조인    R ⟗ S      자연 + 매칭 안된 모두 유지         │
│  세미 조인        R ⋉ S       S에 매칭이 있는 R의 튜플           │
│  안티 조인        R ▷ S       S에 매칭이 없는 R의 튜플           │
└──────────────────────────────────────────────────────────────────┘
```

### 세미 조인과 안티 조인

```
세미 조인:  R ⋉ S = π_{attrs(R)}(R ⋈ S)

"S에 매칭되는 튜플이 있는 R의 튜플 반환"
(결과에는 R의 속성만 나타남)

안티 조인:  R ▷ S = R − π_{attrs(R)}(R ⋈ S)

"S에 매칭되는 튜플이 없는 R의 튜플 반환"
```

**예제:**

```
"최소 한 과목에 등록한 학생" (세미 조인):
STUDENT ⋉ ENROLLMENT
→ ENROLLMENT에 나타나는 STUDENT의 모든 학생

"어떤 과목에도 등록하지 않은 학생" (안티 조인):
STUDENT ▷ ENROLLMENT
→ 매칭되는 등록이 없는 학생
```

---

## 5. 나눗셈

**나눗셈(Division)** 연산은 "모든(for all)" 쿼리에 답합니다. 관계 대수에서 가장 강력하면서도 가장 직관적이지 않은 연산 중 하나입니다.

```
표기법:   R ÷ S

주어진:   R(A1, A2, ..., An, B1, B2, ..., Bm)
          S(B1, B2, ..., Bm)

결과:     S의 모든 튜플 s에 대해, 튜플 <t, s>가 R에 있는
          {A1, ..., An}에 대한 튜플 t.

형식적으로:
  R ÷ S = { t | t ∈ π_{A}(R) AND ∀s ∈ S : <t,s> ∈ R }

  여기서 A = {A1, ..., An} (R에는 있지만 S에는 없는 속성)
```

### 기본 연산으로 표현된 나눗셈

```
R ÷ S = π_A(R) − π_A( (π_A(R) × S) − R )

설명:
  1. π_A(R)              = R의 모든 가능한 A-값
  2. π_A(R) × S          = 모든 A-값과 모든 S-튜플의 짝
  3. (π_A(R) × S) − R    = R에서 빠진 조합
  4. π_A(...)             = 최소 하나의 S-튜플이 빠진 A-값
  5. π_A(R) − ...         = S-튜플이 하나도 빠지지 않은 A-값
                          = 모든 S-튜플과 연관된 A-값
```

### 나눗셈 예제

```
"모든 CS 과목에 등록한 학생 찾기"

1단계: 피제수 정의 (학생-과목 쌍)
  R = π_{sid, cid}(ENROLLMENT)
  ┌──────┬───────┐
  │ sid  │ cid   │
  ├──────┼───────┤
  │ S001 │ CS101 │
  │ S001 │ CS301 │
  │ S001 │ MA101 │
  │ S002 │ CS101 │
  │ S002 │ CS301 │
  │ S003 │ EE201 │
  │ S003 │ CS101 │
  │ S004 │ CS101 │
  │ S005 │ CS101 │
  │ S005 │ MA101 │
  └──────┴───────┘

2단계: 제수 정의 (모든 CS 과목)
  S = π_{cid}(σ_{dept='CS'}(COURSE))
  ┌───────┐
  │ cid   │
  ├───────┤
  │ CS101 │
  │ CS301 │
  │ CS401 │
  └───────┘

3단계: R ÷ S = {CS101, CS301, CS401} 모두와 연관된 학생

  각 학생 확인:
    S001: {CS101, CS301}이 있지만 CS401이 없음 → ✗
    S002: {CS101, CS301}이 있지만 CS401이 없음 → ✗
    S003: {CS101}만 있음                       → ✗
    S004: {CS101}만 있음                       → ✗
    S005: {CS101}만 있음                       → ✗

  결과: 공집합 (세 CS 과목 모두에 등록한 학생 없음)

CS401이 과목 목록에 없다면 (CS101과 CS301만):
  S = {CS101, CS301}
  S001: {CS101, CS301} 있음 → ✓
  S002: {CS101, CS301} 있음 → ✓
  기타: ✗

  결과:
  ┌──────┐
  │ sid  │
  ├──────┤
  │ S001 │
  │ S002 │
  └──────┘
```

---

## 6. 추가 연산

### 집계와 그룹화

**그룹화/집계(Grouping/Aggregation)** 연산자는 집계 함수를 지원하도록 관계 대수를 확장합니다.

```
표기법:   _{G1, G2, ..., Gn} G _{F1(A1), F2(A2), ..., Fk(Ak)} (R)

            또는 더 일반적으로:

            γ_{G; F1(A1) AS name1, ...}(R)

여기서:
  G1, ..., Gn = 그룹화 속성
  F1, ..., Fk = 집계 함수 (COUNT, SUM, AVG, MIN, MAX)
  A1, ..., Ak = 집계할 속성
```

**예제:**

```
"학과별 학생 수"

γ_{dept; COUNT(sid) AS count}(STUDENT)

결과:
┌──────┬───────┐
│ dept │ count │
├──────┼───────┤
│ CS   │ 3     │
│ EE   │ 1     │
│ ME   │ 1     │
└──────┴───────┘


"학과별 평균 급여 (교수용)"

γ_{dept; AVG(salary) AS avg_sal, COUNT(*) AS num}(INSTRUCTOR)

결과:
┌──────┬─────────┬─────┐
│ dept │ avg_sal │ num │
├──────┼─────────┼─────┤
│ CS   │ 91500   │ 2   │
│ EE   │ 92000   │ 1   │
│ MA   │ 85000   │ 1   │
└──────┴─────────┴─────┘
```

### 대입

**대입(Assignment)** 연산자는 중간 결과를 저장합니다:

```
표기법:   temp ← 표현식

예제:
  CS_STUDENTS ← σ_{dept='CS'}(STUDENT)
  CS_NAMES ← π_{name}(CS_STUDENTS)
```

### 정렬

```
표기법:   τ_{A1 ASC, A2 DESC}(R)

참고: 정렬은 집합이 아닌 리스트를 생성. 엄밀히 말하면,
      순수 관계 대수(집합만 생성)를 넘어감.
```

---

## 7. 쿼리 트리와 대수적 최적화

### 쿼리 트리

**쿼리 트리(Query Tree)** (또는 연산자 트리)는 관계 대수 표현식을 트리로 표현한 것입니다:
- 리프 노드는 기본 관계
- 내부 노드는 관계 대수 연산
- 루트는 최종 결과 생성

**예제:** "CS301에 등록한 CS 학생의 이름 찾기"

```
관계 대수:
  π_{name}(σ_{dept='CS' AND cid='CS301'}(STUDENT ⋈ ENROLLMENT))

쿼리 트리:

              π_{name}
                 │
          σ_{dept='CS' AND cid='CS301'}
                 │
                ⋈_{sid=sid}
               / \
          STUDENT  ENROLLMENT
```

### 대수적 최적화

쿼리 최적화기는 **동치 규칙(equivalence rules)**을 사용하여 쿼리 트리를 변환해 더 효율적인 실행 계획을 찾습니다.

### 주요 동치 규칙

**규칙 1: 선택의 연쇄(Cascade of Selection)**

```
σ_{c1 AND c2}(R) = σ_{c1}(σ_{c2}(R))

연결된 선택은 일련의 선택으로 분리 가능.
```

**규칙 2: 선택의 교환법칙(Commutativity of Selection)**

```
σ_{c1}(σ_{c2}(R)) = σ_{c2}(σ_{c1}(R))

선택의 순서는 중요하지 않음.
```

**규칙 3: 프로젝션의 연쇄(Cascade of Projection)**

```
π_{L1}(π_{L2}(...π_{Ln}(R)...)) = π_{L1}(R)

가장 바깥쪽 프로젝션만 중요 (L1 ⊆ L2 ⊆ ... ⊆ Ln인 경우).
```

**규칙 4: 선택과 프로젝션의 교환(Commuting Selection with Projection)**

```
선택 조건 c가 L의 속성만 포함하는 경우:
  π_L(σ_c(R)) = σ_c(π_L(R))
```

**규칙 5: 조인의 교환법칙(Commutativity of Join)**

```
R ⋈ S = S ⋈ R
R × S = S × R
```

**규칙 6: 조인의 결합법칙(Associativity of Join)**

```
(R ⋈ S) ⋈ T = R ⋈ (S ⋈ T)
```

**규칙 7: 조인을 통한 선택 밀어내기(Pushing Selection Through Join)**

```
조건 c가 R의 속성만 포함하는 경우:
  σ_c(R ⋈ S) = σ_c(R) ⋈ S

이것이 가장 중요한 최적화 규칙!
중간 결과의 크기를 줄임.
```

**규칙 8: 집합 연산의 교환법칙(Commutativity of Set Operations)**

```
R ∪ S = S ∪ R
R ∩ S = S ∩ R
(하지만 R − S ≠ S − R)
```

### 최적화 예제

```
원본 쿼리 트리 (최적화 안됨):

              π_{name}
                 │
          σ_{dept='CS' AND cid='CS301'}
                 │
                ⋈_{sid=sid}
               / \
          STUDENT  ENROLLMENT
          (5 rows)  (10 rows)
          카티션: 필터 전 50 rows

최적화된 쿼리 트리 (선택 밀어내기):

              π_{name}
                 │
                ⋈_{sid=sid}
               / \
   σ_{dept='CS'}  σ_{cid='CS301'}
        |              |
     STUDENT       ENROLLMENT
     (→3 rows)     (→2 rows)
     조인: 6개 조합, ~2개 매칭

최적화된 트리:
  1. 먼저 STUDENT를 3명의 CS 학생으로 필터링
  2. 먼저 ENROLLMENT를 2개의 CS301 등록으로 필터링
  3. 5 × 10 = 50 대신 3 × 2 = 6개 조합만 조인
  4. 중간 결과 크기 대폭 감소
```

### 휴리스틱 최적화 규칙

```
1. 선택을 가능한 한 아래로 밀어내기
   (초기에 튜플 수 감소)

2. 프로젝션을 가능한 한 아래로 밀어내기
   (초기에 속성 수 감소, 단 조인 속성 유지)

3. 가장 제한적인 선택을 먼저 선택
   (가장 많은 튜플을 제거하는 선택)

4. 카티션 곱 피하기
   (항상 카티션 + 선택보다 조인 선호)

5. 중간 결과 크기를 최소화하는 조인 순서 선택
   (가장 어려운 최적화 문제 — 종종 NP-hard)
```

---

## 8. 관계 해석 (간단한 소개)

관계 대수가 **절차적(procedural)**(결과를 계산하는 방법 지정)인 반면, **관계 해석(Relational Calculus)**은 **선언적(declarative)**(결과가 무엇이어야 하는지 지정, 계산 방법은 아님)입니다.

### 튜플 관계 해석 (TRC)

TRC에서 쿼리는 관계에 대한 튜플 변수를 사용하여 표현됩니다.

```
일반 형식:
  { t | P(t) }

  "술어 P(t)가 참인 모든 튜플 t의 집합"

예제: "모든 CS 학생 찾기"

  { t | t ∈ STUDENT ∧ t.dept = 'CS' }

  "dept가 CS인 STUDENT의 튜플 t 집합"

예제: "3학년 또는 4학년 학생의 이름과 학과 찾기"

  { t.name, t.dept | t ∈ STUDENT ∧ (t.year = 3 ∨ t.year = 4) }

예제: "CS301에 등록한 학생 찾기"

  { t | t ∈ STUDENT ∧ ∃e ∈ ENROLLMENT(e.sid = t.sid ∧ e.cid = 'CS301') }

  "동일한 sid를 가지고 cid = CS301인 등록 e가 존재하는
   STUDENT의 튜플 t"
```

### 도메인 관계 해석 (DRC)

DRC에서 변수는 튜플이 아닌 개별 값(도메인)에 대한 범위를 갖습니다.

```
일반 형식:
  { <x1, x2, ..., xn> | P(x1, x2, ..., xn) }

예제: "CS 학생의 이름 찾기"

  { <n> | ∃s, y, d (STUDENT(s, n, y, d) ∧ d = 'CS') }

  "(s, n, y, d)가 STUDENT의 튜플이고 d가 CS인
   값 s, y, d가 존재하는 이름 값 n의 집합"
```

### 대수와 해석의 동치성

**코드의 정리(Codd's Theorem)** (1972): 관계 대수와 (안전한) 관계 해석은 동일한 표현력을 가집니다. 하나로 표현할 수 있는 모든 쿼리는 다른 것으로도 표현할 수 있습니다.

```
┌──────────────────────────────────────────────────────┐
│                                                      │
│  관계 대수(Relational Algebra)  ≡  안전한 튜플 관계 해석  │
│                      ≡  안전한 도메인 관계 해석          │
│                      ≡  SQL (핵심)                   │
│                                                      │
│  쿼리 언어가 관계 대수로 표현할 수 있는 모든 것을        │
│  표현할 수 있다면 "관계적으로 완전(relationally complete)"│
│                                                      │
│  SQL은 관계적으로 완전 (그 이상 — 집계, 정렬, 재귀 등)   │
└──────────────────────────────────────────────────────┘
```

### 표현식의 안전성

관계 해석 표현식이 유한한 결과를 보장하면 **안전(safe)**합니다. 안전하지 않은 표현식은 무한한 결과를 생성할 수 있습니다:

```
안전하지 않음:  { t | ¬(t ∈ STUDENT) }
         "STUDENT에 없는 모든 튜플" — 무한!

안전함:    { t | t ∈ STUDENT ∧ ¬(∃e ∈ ENROLLMENT(e.sid = t.sid)) }
         "어떤 과목에도 등록하지 않은 학생" — 유한 결과
```

---

## 9. SQL과의 동치성

모든 관계 대수 표현식은 동치인 SQL이 있습니다. 이 대응 관계를 이해하면 SQL 쿼리 작성 및 최적화에 도움이 됩니다.

### 연산별 매핑

```
┌────────────────────┬──────────────────────────────────────────┐
│ 관계 대수            │ SQL 동치                                  │
├────────────────────┼──────────────────────────────────────────┤
│ σ_{c}(R)           │ SELECT * FROM R WHERE c                  │
│ π_{A,B}(R)         │ SELECT DISTINCT A, B FROM R              │
│ ρ_{S}(R)           │ R AS S  (FROM 절에서)                    │
│ R ∪ S              │ SELECT * FROM R UNION SELECT * FROM S    │
│ R ∩ S              │ SELECT * FROM R INTERSECT SELECT * FROM S│
│ R − S              │ SELECT * FROM R EXCEPT SELECT * FROM S   │
│ R × S              │ SELECT * FROM R, S  (또는 CROSS JOIN)     │
│ R ⋈_{c} S          │ SELECT * FROM R JOIN S ON c              │
│ R ⋈ S              │ SELECT * FROM R NATURAL JOIN S           │
│ R ⟕ S              │ SELECT * FROM R LEFT OUTER JOIN S ON ... │
│ R ÷ S              │ (NOT EXISTS + 상관 부질의 필요)            │
│ γ_{G;F(A)}(R)      │ SELECT G, F(A) FROM R GROUP BY G        │
│ τ_{A}(R)           │ SELECT * FROM R ORDER BY A               │
└────────────────────┴──────────────────────────────────────────┘
```

### 상세한 SQL 동치

**선택:**

```
σ_{dept='CS' AND year>=3}(STUDENT)

SELECT *
FROM student
WHERE dept = 'CS' AND year >= 3;
```

**프로젝션:**

```
π_{name, dept}(STUDENT)

SELECT DISTINCT name, dept
FROM student;

참고: SQL은 기본적으로 중복을 제거하지 않음.
      관계 대수의 집합 의미론과 일치시키려면 DISTINCT를 사용해야 함.
```

**자연 조인:**

```
STUDENT ⋈ ENROLLMENT

-- 방법 1: NATURAL JOIN (모든 공통 열에서 매칭)
SELECT * FROM student NATURAL JOIN enrollment;

-- 방법 2: 명시적 JOIN (더 안전 — 어떤 열이 매칭되는지 제어)
SELECT s.sid, s.name, s.year, s.dept, e.cid, e.grade
FROM student s
JOIN enrollment e ON s.sid = e.sid;
```

**나눗셈 (SQL로 표현하기 가장 어려움):**

```
R ÷ S  =  "S의 모든 튜플과 연관된 R의 튜플 찾기"

-- "모든 CS 과목에 등록한 학생 찾기"
SELECT DISTINCT e.sid
FROM enrollment e
WHERE NOT EXISTS (
    SELECT c.cid
    FROM course c
    WHERE c.dept = 'CS'
    AND NOT EXISTS (
        SELECT 1
        FROM enrollment e2
        WHERE e2.sid = e.sid
        AND e2.cid = c.cid
    )
);

-- COUNT를 사용한 대안:
SELECT e.sid
FROM enrollment e
JOIN course c ON e.cid = c.cid
WHERE c.dept = 'CS'
GROUP BY e.sid
HAVING COUNT(DISTINCT e.cid) = (
    SELECT COUNT(*) FROM course WHERE dept = 'CS'
);
```

**조합 예제:**

```
"어떤 과목에서든 A를 받은 학생의 이름 찾기"

관계 대수:
  π_{name}(σ_{grade='A'}(STUDENT ⋈ ENROLLMENT))

SQL:
  SELECT DISTINCT s.name
  FROM student s
  JOIN enrollment e ON s.sid = e.sid
  WHERE e.grade = 'A';
```

---

## 10. 완전한 작업 예제

### 예제 1: 다단계 쿼리

**쿼리:** "CS 학과가 가르치는 과목에 등록했지만 CS 전공이 아닌 학생의 이름과 학과 찾기."

```
관계 대수:

  CS_COURSES ← σ_{dept='CS'}(COURSE)
  CS_ENROLLED ← ENROLLMENT ⋈_{cid} CS_COURSES
  NON_CS_STUDENTS ← σ_{dept≠'CS'}(STUDENT)
  RESULT ← π_{name, dept}(NON_CS_STUDENTS ⋈_{sid} CS_ENROLLED)

단계별:

1. CS_COURSES = σ_{dept='CS'}(COURSE)
   → {CS101, CS301, CS401}

2. CS_ENROLLED = π_{sid}(ENROLLMENT ⋈ CS_COURSES)
   → {S001, S002, S003, S004, S005} (CS101/CS301/CS401을 수강한 모두)

3. NON_CS_STUDENTS = σ_{dept≠'CS'}(STUDENT)
   → {S003/Carol/EE, S004/Dave/ME}

4. RESULT = π_{name, dept}(NON_CS_STUDENTS ⋈ CS_ENROLLED)
   → {(Carol, EE), (Dave, ME)}

SQL:
  SELECT DISTINCT s.name, s.dept
  FROM student s
  JOIN enrollment e ON s.sid = e.sid
  JOIN course c ON e.cid = c.cid
  WHERE c.dept = 'CS' AND s.dept <> 'CS';
```

### 예제 2: 나눗셈 쿼리

**쿼리:** "학생 S001이 수강한 모든 과목을 수강한 학생 찾기."

```
관계 대수:

  S001_COURSES ← π_{cid}(σ_{sid='S001'}(ENROLLMENT))
  ALL_ENROLLMENTS ← π_{sid, cid}(ENROLLMENT)
  RESULT ← ALL_ENROLLMENTS ÷ S001_COURSES

단계별:

1. S001_COURSES = {CS101, CS301, MA101}

2. 각 학생 확인:
   S001: {CS101, CS301, MA101} ⊇ {CS101, CS301, MA101} → ✓
   S002: {CS101, CS301}        ⊇ {CS101, CS301, MA101} → ✗ (MA101 빠짐)
   S003: {EE201, CS101}        ⊇ {CS101, CS301, MA101} → ✗
   S004: {CS101}               ⊇ {CS101, CS301, MA101} → ✗
   S005: {CS101, MA101}        ⊇ {CS101, CS301, MA101} → ✗ (CS301 빠짐)

3. RESULT = {S001}

SQL:
  SELECT e.sid
  FROM enrollment e
  WHERE e.cid IN (
      SELECT cid FROM enrollment WHERE sid = 'S001'
  )
  GROUP BY e.sid
  HAVING COUNT(DISTINCT e.cid) = (
      SELECT COUNT(DISTINCT cid) FROM enrollment WHERE sid = 'S001'
  );
```

### 예제 3: 외부 조인

**쿼리:** "등록이 없는 학생을 포함하여 모든 학생과 그들의 등록 수 나열."

```
관계 대수:

  JOINED ← STUDENT ⟕ ENROLLMENT       (sid에서 왼쪽 외부 조인)
  RESULT ← γ_{sid, name; COUNT(cid) AS num_courses}(JOINED)

Dave (S004)가 등록이 없다면:

SQL:
  SELECT s.sid, s.name, COUNT(e.cid) AS num_courses
  FROM student s
  LEFT OUTER JOIN enrollment e ON s.sid = e.sid
  GROUP BY s.sid, s.name
  ORDER BY num_courses DESC;

결과:
  ┌──────┬───────┬─────────────┐
  │ sid  │ name  │ num_courses │
  ├──────┼───────┼─────────────┤
  │ S001 │ Alice │ 3           │
  │ S002 │ Bob   │ 2           │
  │ S003 │ Carol │ 2           │
  │ S005 │ Eve   │ 2           │
  │ S004 │ Dave  │ 0           │  ← LEFT OUTER JOIN으로 보존됨
  └──────┴───────┴─────────────┘
```

---

## 11. 연습 문제

### 기본 연산

**연습 문제 3.1**: 샘플 데이터베이스를 사용하여 다음에 대한 관계 대수 표현식을 작성하세요:

- (a) 2학년 또는 3학년의 모든 학생
- (b) 4학점 이상의 과목 제목
- (c) EE201에 등록한 학생 ID
- (d) CS 학과가 아닌 학생의 이름

**연습 문제 3.2**: 다음 표현식을 단계별로 평가하고, 중간 결과를 보여주세요:

- (a) `π_{name}(σ_{year > 2}(STUDENT))`
- (b) `π_{sid}(σ_{grade='A'}(ENROLLMENT)) ∩ π_{sid}(σ_{dept='CS'}(STUDENT))`
- (c) `STUDENT ⋈ (σ_{cid='CS301'}(ENROLLMENT))`

### 조인 연산

**연습 문제 3.3**: 다음 각각에 대해 관계 대수 표현식과 동치 SQL을 작성하세요:

- (a) "Database Theory"에 등록한 학생의 이름 찾기
- (b) A 학점을 받은 학생이 최소 한 명 있는 과목 제목 찾기
- (c) 자신의 학과 외의 과목에 등록한 학생 찾기
- (d) 최소 한 과목을 공유하는 학생 쌍 찾기

**연습 문제 3.4**: 샘플 데이터에서 다음 세 쿼리 결과의 차이를 설명하세요:

```
Q1: STUDENT ⋈ ENROLLMENT
Q2: STUDENT ⟕ ENROLLMENT
Q3: STUDENT × ENROLLMENT
```

각각 몇 개의 튜플을 생성하나요? Q2에는 나타나지만 Q1에는 나타나지 않는 튜플은?

### 나눗셈

**연습 문제 3.5**: 나눗셈을 사용하여 다음에 대한 관계 대수 표현식을 작성하세요:

- (a) 모든 3학점 과목에 등록한 학생
- (b) Bob (S002)이 수강한 모든 과목을 수강한 학생

단계별 평가를 보여주세요.

**연습 문제 3.6**: 연습 문제 3.5의 각 나눗셈 쿼리를 다음을 사용하여 SQL로 표현하세요:
- (i) 이중 NOT EXISTS
- (ii) HAVING COUNT를 사용한 GROUP BY

### 최적화

**연습 문제 3.7**: 다음 쿼리가 주어졌을 때:

```
π_{name}(σ_{credits=3 AND grade='A'}(STUDENT ⋈ ENROLLMENT ⋈ COURSE))
```

- (a) 초기(최적화 안된) 쿼리 트리 그리기
- (b) 대수적 최적화 규칙을 적용하여 최적화된 쿼리 트리 생성
- (c) 어떤 규칙을 적용했는지, 왜 최적화된 트리가 더 나은지 설명
- (d) 중간 결과 크기의 감소 추정

**연습 문제 3.8**: 다음 동치를 증명하거나 반증하세요:

- (a) `σ_{c1}(R ∪ S) = σ_{c1}(R) ∪ σ_{c1}(S)`
- (b) `σ_{c1}(R − S) = σ_{c1}(R) − σ_{c1}(S)`
- (c) `π_A(R ∪ S) = π_A(R) ∪ π_A(S)`
- (d) `σ_{c1}(R × S) = σ_{c1}(R) × S`  (c1이 R의 속성만 포함하는 경우)

### 관계 해석

**연습 문제 3.9**: 다음을 튜플 관계 해석(TRC)으로 표현하세요:

- (a) CS 학과 학생의 이름
- (b) 최소 두 과목에 등록한 학생
- (c) 등록한 학생이 없는 과목
- (d) CS 학과가 제공하는 모든 과목에 등록한 학생

**연습 문제 3.10**: 다음 TRC 표현식 중 어느 것이 안전한지 결정하세요. 안전하지 않다면 그 이유를 설명하고 안전한 동치를 제공하세요:

- (a) `{ t | ¬(t ∈ STUDENT) }`
- (b) `{ t.name | t ∈ STUDENT ∧ t.gpa > 3.5 }`
- (c) `{ <x, y> | ∃t ∈ STUDENT(t.sid = x ∧ t.name = y) }`
- (d) `{ t | t.salary > 100000 }`

### 도전 문제

**연습 문제 3.11**: 관계 R(A, B, C)가 다음 튜플을 포함합니다:

```
{(a1, b1, c1), (a1, b2, c1), (a1, b2, c2), (a2, b1, c1), (a2, b1, c2)}
```

관계 S(B, C)는: `{(b1, c1), (b1, c2)}`를 포함합니다

R ÷ S를 단계별로 계산하세요. 결과의 각 튜플이 S의 모든 튜플과 연관되어 있는지 확인하여 답을 검증하세요.

**연습 문제 3.12**: 다음에 대한 단일 관계 대수 표현식(대입 없음) 작성:

"평균 교수 급여가 가장 높은 학과 찾기."

힌트: 집계와 비교 패턴이 필요합니다.

---

**이전**: [관계 모델](./02_Relational_Model.md) | **다음**: [ER 모델링](./04_ER_Modeling.md)
