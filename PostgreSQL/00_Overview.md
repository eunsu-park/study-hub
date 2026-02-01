# PostgreSQL 학습 가이드

## 소개

이 폴더는 PostgreSQL 관계형 데이터베이스를 학습하기 위한 자료를 담고 있습니다. SQL 기초부터 고급 기능, 운영까지 단계별로 학습할 수 있습니다.

**대상 독자**: SQL 입문자 ~ 중급자, 백엔드 개발자

---

## 학습 로드맵

```
[기초]                [중급]                 [고급]
  │                     │                      │
  ▼                     ▼                      ▼
PostgreSQL 기초 ──▶ JOIN ──────────▶ 함수/프로시저
  │                     │                      │
  ▼                     ▼                      ▼
DB 관리 ──────────▶ 집계와 그룹 ────▶ 트랜잭션
  │                     │                      │
  ▼                     ▼                      ▼
테이블/타입 ──────▶ 서브쿼리/CTE ───▶ 트리거
  │                     │                      │
  ▼                     ▼                      ▼
CRUD 기본 ────────▶ 뷰와 인덱스 ────▶ 백업/운영
  │
  ▼
조건과 정렬
```

---

## 선수 지식

- 기본적인 컴퓨터 사용법
- 터미널/명령줄 사용 경험
- (선택) Docker 기초 지식

---

## 파일 목록

| 파일명 | 난이도 | 주요 내용 |
|--------|--------|----------|
| [01_PostgreSQL_Basics.md](./01_PostgreSQL_Basics.md) | ⭐ | 개념, 설치, psql 기본 |
| [02_Database_Management.md](./02_Database_Management.md) | ⭐ | DB 생성/삭제, 사용자, 권한 |
| [03_Tables_and_Data_Types.md](./03_Tables_and_Data_Types.md) | ⭐⭐ | CREATE TABLE, 자료형, 제약조건 |
| [04_CRUD_Basics.md](./04_CRUD_Basics.md) | ⭐ | SELECT, INSERT, UPDATE, DELETE |
| [05_Conditions_and_Sorting.md](./05_Conditions_and_Sorting.md) | ⭐⭐ | WHERE, ORDER BY, LIMIT |
| [06_JOIN.md](./06_JOIN.md) | ⭐⭐ | INNER, LEFT, RIGHT, FULL JOIN |
| [07_Aggregation_and_Grouping.md](./07_Aggregation_and_Grouping.md) | ⭐⭐ | COUNT, SUM, GROUP BY, HAVING |
| [08_Subqueries_and_CTE.md](./08_Subqueries_and_CTE.md) | ⭐⭐⭐ | 서브쿼리, WITH 절 |
| [09_Views_and_Indexes.md](./09_Views_and_Indexes.md) | ⭐⭐⭐ | VIEW, INDEX, EXPLAIN |
| [10_Functions_and_Procedures.md](./10_Functions_and_Procedures.md) | ⭐⭐⭐ | PL/pgSQL, 사용자 정의 함수 |
| [11_Transactions.md](./11_Transactions.md) | ⭐⭐⭐ | ACID, BEGIN, COMMIT, 격리 수준 |
| [12_Triggers.md](./12_Triggers.md) | ⭐⭐⭐ | 트리거 생성 및 활용 |
| [13_Backup_and_Operations.md](./13_Backup_and_Operations.md) | ⭐⭐⭐⭐ | pg_dump, 모니터링, 운영 |
| [14_JSON_JSONB.md](./14_JSON_JSONB.md) | ⭐⭐⭐ | JSON 연산자, 인덱싱, 스키마 검증 |
| [15_Query_Optimization.md](./15_Query_Optimization.md) | ⭐⭐⭐⭐ | EXPLAIN ANALYZE, 인덱스 전략 |
| [16_Replication_HA.md](./16_Replication_HA.md) | ⭐⭐⭐⭐⭐ | 스트리밍 복제, 논리 복제, 페일오버 |
| [17_Window_Functions.md](./17_Window_Functions.md) | ⭐⭐⭐ | OVER, ROW_NUMBER, RANK, LEAD/LAG |
| [18_Table_Partitioning.md](./18_Table_Partitioning.md) | ⭐⭐⭐⭐ | Range/List/Hash 파티셔닝 |

---

## 추천 학습 순서

### 초급 (SQL 입문)
1. PostgreSQL 기초 → DB 관리 → 테이블/타입 → CRUD → 조건/정렬

### 중급 (데이터 분석)
2. JOIN → 집계와 그룹 → 서브쿼리/CTE → 뷰와 인덱스

### 고급 (DBA/백엔드)
3. 함수/프로시저 → 트랜잭션 → 트리거 → 백업/운영

### 심화 (전문가)
4. JSON/JSONB → 쿼리 최적화 심화 → 윈도우 함수 → 파티셔닝 → 복제와 고가용성

---

## 실습 환경

### Docker (권장)

```bash
# PostgreSQL 컨테이너 실행
docker run --name postgres-study \
  -e POSTGRES_PASSWORD=mypassword \
  -p 5432:5432 \
  -d postgres:16

# psql 접속
docker exec -it postgres-study psql -U postgres
```

### macOS (Homebrew)

```bash
brew install postgresql@16
brew services start postgresql@16
psql postgres
```

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo -u postgres psql
```

---

## psql 기본 명령어

| 명령어 | 설명 |
|--------|------|
| `\l` | 데이터베이스 목록 |
| `\c dbname` | 데이터베이스 전환 |
| `\dt` | 테이블 목록 |
| `\d tablename` | 테이블 구조 |
| `\q` | psql 종료 |

---

## 실습 예제

`examples/` 폴더에 핵심 SQL 문법을 실습할 수 있는 예제 파일이 있습니다.

### 예제 파일 목록

| 파일명 | 주제 | 핵심 내용 |
|--------|------|----------|
| [01_basic_crud.sql](./examples/01_basic_crud.sql) | CRUD 기본 | CREATE, INSERT, SELECT, UPDATE, DELETE |
| [02_joins.sql](./examples/02_joins.sql) | JOIN | INNER, LEFT, RIGHT, FULL, CROSS, SELF JOIN |
| [03_aggregation.sql](./examples/03_aggregation.sql) | 집계 함수 | COUNT, SUM, AVG, GROUP BY, HAVING, ROLLUP |
| [04_subqueries.sql](./examples/04_subqueries.sql) | 서브쿼리/CTE | 스칼라, 인라인 뷰, EXISTS, WITH, 재귀 CTE |
| [05_window_functions.sql](./examples/05_window_functions.sql) | 윈도우 함수 | ROW_NUMBER, RANK, LAG, LEAD, 프레임 절 |

### 예제 실행 방법

```bash
# Docker PostgreSQL 접속
docker exec -it postgres-study psql -U postgres

# 예제 파일 실행
\i examples/01_basic_crud.sql

# 또는 터미널에서 직접 실행
psql -U postgres -d study_db -f examples/01_basic_crud.sql
```

### 예제 사용 팁

1. **순서대로 실행**: 01부터 시작 (테이블 생성 포함)
2. **개별 실행**: 각 파일의 특정 섹션만 복사해서 실행 가능
3. **결과 확인**: 각 쿼리 실행 후 SELECT로 결과 확인
4. **초기화**: 테이블 삭제 후 다시 시작 가능

---

## 관련 자료

- [Docker 학습](../Docker/00_Overview.md) - PostgreSQL 컨테이너로 실행
- [공식 문서](https://www.postgresql.org/docs/)
