# PostgreSQL 기초

**다음**: [데이터베이스 관리](./02_Database_Management.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. PostgreSQL이 무엇인지 설명하고 주요 특징(ACID, MVCC, 확장성)을 기술할 수 있다
2. PostgreSQL과 다른 대표적인 데이터베이스(MySQL, SQLite)를 비교할 수 있다
3. Docker, Homebrew, 또는 Linux 패키지 관리자를 사용하여 PostgreSQL을 설치할 수 있다
4. `psql` 명령줄 클라이언트를 사용하여 PostgreSQL 서버에 접속할 수 있다
5. 필수 psql 메타 명령어(`\l`, `\dt`, `\d`, `\c`)를 식별하고 사용할 수 있다
6. 기본 SQL 구문(SELECT, CREATE DATABASE)을 작성하고 실행할 수 있다
7. SQL의 기본 문법 규칙(대소문자 구분, 주석, 세미콜론)을 적용할 수 있다
8. 일반적인 접속 및 시작 오류를 해결할 수 있다

---

PostgreSQL은 오늘날 사용 가능한 가장 발전된 오픈소스 관계형 데이터베이스(Relational Database) 중 하나입니다. 소규모 웹 애플리케이션을 구축하든 대규모 분석 플랫폼을 설계하든, PostgreSQL은 전문 개발자와 데이터 엔지니어가 의지하는 안정성, 확장성, 그리고 표준 준수를 제공합니다. 이 레슨은 설치부터 첫 번째 접속, 그리고 PostgreSQL 여정의 출발점이 되는 필수 명령어들을 단계별로 안내합니다.

---

## 1. PostgreSQL이란?

PostgreSQL(포스트그레스큐엘)은 오픈소스 관계형 데이터베이스 관리 시스템(RDBMS)입니다.

### 특징

- **오픈소스**: 무료로 사용 가능
- **표준 SQL 준수**: ANSI SQL 표준을 잘 따름
- **확장성**: JSON, 배열, 사용자 정의 타입 지원
- **ACID 준수**: 트랜잭션의 안정성 보장
- **동시성 제어**: MVCC(Multi-Version Concurrency Control)

### 왜 PostgreSQL을 사용할까?

```
┌─────────────────────────────────────────────────────────────┐
│                    PostgreSQL 장점                          │
├─────────────────────────────────────────────────────────────┤
│  • 복잡한 쿼리 처리 성능이 우수                              │
│  • JSON/JSONB 타입으로 NoSQL처럼 사용 가능                  │
│  • 풀텍스트 검색 내장                                       │
│  • 지리 데이터 지원 (PostGIS)                               │
│  • 대규모 데이터 처리에 적합                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 다른 데이터베이스와 비교

| 특징 | PostgreSQL | MySQL | SQLite |
|------|------------|-------|--------|
| 라이선스 | PostgreSQL License | GPL | Public Domain |
| JSON 지원 | JSONB (고성능) | JSON | JSON (제한적) |
| 동시성 | MVCC | InnoDB MVCC | 파일 잠금 |
| 확장성 | 매우 높음 | 높음 | 낮음 |
| 용도 | 엔터프라이즈, 분석 | 웹 애플리케이션 | 임베디드, 테스트 |

---

## 3. 설치 방법

### Docker (권장)

가장 빠르게 시작할 수 있는 방법입니다.

```bash
# Docker는 호스트 시스템으로부터 PostgreSQL을 격리 — 자유롭게 실험하고,
# 문제가 생기면 "docker rm"으로 초기화 가능. 다른 설치에 영향 없음.
# 또한 머신 간 동일한 환경을 보장 (재현성)
docker run --name postgres-study \
  -e POSTGRES_PASSWORD=mypassword \
  -e POSTGRES_USER=myuser \
  -e POSTGRES_DB=mydb \
  -p 5432:5432 \
  -d postgres:16

# 실행 확인
docker ps

# 컨테이너 내부에서 psql 접속
docker exec -it postgres-study psql -U myuser -d mydb
```

### macOS (Homebrew)

```bash
# PostgreSQL 설치
brew install postgresql@16

# 서비스 시작
brew services start postgresql@16

# 기본 데이터베이스 접속
psql postgres
```

### Linux (Ubuntu/Debian)

```bash
# 패키지 목록 업데이트
sudo apt update

# PostgreSQL 설치
sudo apt install postgresql postgresql-contrib

# 서비스 상태 확인
sudo systemctl status postgresql

# postgres 사용자로 접속
sudo -u postgres psql
```

### Linux (CentOS/RHEL)

```bash
# PostgreSQL 저장소 추가
sudo dnf install -y https://download.postgresql.org/pub/repos/yum/reporpms/EL-8-x86_64/pgdg-redhat-repo-latest.noarch.rpm

# PostgreSQL 설치
sudo dnf install -y postgresql16-server

# 데이터베이스 초기화
sudo /usr/pgsql-16/bin/postgresql-16-setup initdb

# 서비스 시작
sudo systemctl start postgresql-16
sudo systemctl enable postgresql-16
```

### Windows

1. [공식 다운로드 페이지](https://www.postgresql.org/download/windows/)에서 설치 프로그램 다운로드
2. 설치 마법사 실행
3. 비밀번호 설정
4. 기본 포트 5432 사용
5. pgAdmin 함께 설치 (GUI 도구)

---

## 4. 설치 확인

```bash
# PostgreSQL 버전 확인
psql --version
# 또는
postgres --version
```

출력 예시:
```
psql (PostgreSQL) 16.1
```

---

## 5. psql 클라이언트

psql은 PostgreSQL의 대화형 터미널 클라이언트입니다.

### 이론: 클라이언트/서버 프로세스 모델

PostgreSQL "서버"는 단일 프로세스가 아닙니다. 트리(tree) 구조입니다.

```
postmaster (parent)
├── backend for client #1   (연결당 프로세스 1개)
├── backend for client #2
├── background writer       (dirty buffer flush)
├── WAL writer              (WAL flush)
├── checkpointer            (consistent point 기록)
├── autovacuum launcher     (autovacuum worker spawn)
└── stats collector / logical replication launcher / ...
```

#### B.1 postmaster

`pg_ctl start`가 가장 먼저 띄우는 프로세스입니다. 설정된 TCP 포트(기본 5432)와 Unix-domain socket을 listen합니다. 자기 자신은 SQL을 실행하지 **않으며**, 연결을 받을 때마다 새 backend 프로세스를 `fork()`해서 socket을 자식에게 넘깁니다. 이로부터 다음이 따라옵니다:

- 연결 비용이 높습니다(연결당 OS 프로세스 하나). 그래서 PgBouncer 같은 **connection pooler**가 존재합니다.
- postmaster가 죽어도 실행 중인 쿼리가 반드시 죽는 것은 아닙니다 — 다만 재시작 전까지 신규 연결을 받을 수 없습니다.

#### B.2 Backend

클라이언트 하나당 OS 프로세스 하나. backend 안에 parser, planner, executor, MVCC visibility 검사, 그리고 연결-로컬 catalog cache가 모두 들어 있습니다. `work_mem`이나 `temp_buffers` 같은 메모리는 backend 단위로 할당되므로, `max_connections`가 크고 `work_mem`이 큰 조합은 시스템 RAM을 고갈시킬 수 있습니다.

#### B.3 Background worker

backend 안에서 동기적으로 처리하면 막힐 작업들을 별도 프로세스에서 처리합니다: dirty buffer 디스크 flush, WAL flush, autovacuum 실행, 병렬 인덱스 빌드, standby 복제 등. backend와 동일한 shared memory segment를 공유하며, 그것이 바로 `shared_buffers`입니다.

### 이론: 시스템 카탈로그 — 메타데이터를 테이블로

PostgreSQL은 자기 자신의 스키마를 PostgreSQL 테이블에 저장하며, 그 스키마 이름이 `pg_catalog`입니다. 모든 데이터베이스, 테이블, 컬럼, 인덱스, 함수, 역할(role), 테이블스페이스(tablespace)는 어느 카탈로그 테이블의 한 행입니다.

| 카탈로그 | 저장 내용 |
|---------|----------|
| `pg_database` | 클러스터 내 데이터베이스 한 행씩 |
| `pg_namespace` | 스키마 한 행씩 |
| `pg_class` | "relation"(테이블, 인덱스, 뷰, 시퀀스, 머티리얼라이즈드 뷰) 한 행씩 |
| `pg_attribute` | 모든 relation의 컬럼 한 행씩 |
| `pg_type` | 데이터 타입 한 행씩 |
| `pg_proc` | 함수/프로시저 한 행씩 |
| `pg_authid` | 역할 한 행씩 |

이 설계가 가져오는 두 가지 실용적 결과:

1. **`psql`이 보여주는 모든 것(`\l`, `\dt`, `\d`)은 단지 `pg_catalog`에 대한 SQL 쿼리입니다.** `psql`에서 `\set ECHO_HIDDEN on`을 켜고 `\d+`를 실행하면 실제로 발사되는 `SELECT` 문을 볼 수 있습니다.
2. **카탈로그 자체도 ACID와 MVCC를 따릅니다.** `CREATE TABLE`은 `pg_class`와 `pg_attribute`에 행을 삽입하는 트랜잭션입니다. DDL을 `BEGIN; ... ROLLBACK;`으로 감싸면, 그 테이블은 존재한 적이 없게 됩니다.

SQL 표준의 벤더 중립적 메타데이터 뷰는 별도 스키마인 `information_schema`로 제공됩니다. PostgreSQL 고유 기능을 들여다볼 때는 `pg_catalog`, 이식 가능한 도구를 만들 때는 `information_schema`를 사용합니다.

### 접속 방법

```bash
# 기본 접속 (로컬, 현재 사용자)
psql

# 특정 데이터베이스 접속
psql -d mydb

# 사용자 지정 접속
psql -U username -d dbname

# 호스트/포트 지정 접속
psql -h localhost -p 5432 -U username -d dbname

# Docker 컨테이너 접속
docker exec -it postgres-study psql -U myuser -d mydb
```

### 메타 명령어 (백슬래시 명령)

psql에서 `\`로 시작하는 명령어들입니다.

| 명령어 | 설명 |
|--------|------|
| `\l` | 데이터베이스 목록 (list) |
| `\c dbname` | 데이터베이스 전환 (connect) |
| `\dt` | 현재 DB의 테이블 목록 |
| `\dt+` | 테이블 목록 (상세) |
| `\d tablename` | 테이블 구조 확인 |
| `\d+ tablename` | 테이블 구조 (상세) |
| `\du` | 사용자(Role) 목록 |
| `\dn` | 스키마 목록 |
| `\df` | 함수 목록 |
| `\di` | 인덱스 목록 |
| `\x` | 확장 출력 모드 토글 |
| `\timing` | 쿼리 실행 시간 표시 토글 |
| `\i filename` | SQL 파일 실행 |
| `\o filename` | 출력을 파일로 저장 |
| `\q` | psql 종료 (quit) |
| `\?` | 메타 명령어 도움말 |
| `\h` | SQL 명령어 도움말 |
| `\h SELECT` | SELECT 문법 도움말 |

### 실습: 기본 명령어 사용

```sql
-- psql 접속 후

-- 데이터베이스 목록 확인
\l

-- 현재 연결 정보 확인
\conninfo

-- 테이블 목록 확인 (처음엔 비어있음)
\dt

-- 도움말 보기
\?
```

---

## 6. 첫 번째 쿼리 실행

### 이론: ACID, 한 글자씩 풀어보기

ACID는 동시성, 충돌(crash), 부분 실패(partial failure) 상황에서 무엇이 살아남는지를 규정한 계약입니다. 각 글자에는 PostgreSQL의 구체적인 메커니즘이 대응됩니다.

#### A.1 Atomicity(원자성) — 전부 아니면 전무

트랜잭션은 전체가 커밋되거나 어떠한 관측 가능한 효과도 남기지 않습니다. 내부적으로 모든 변경은 트랜잭션 ID(`xid`)와 함께 먼저 **WAL(Write-Ahead Log)** 에 기록됩니다. `COMMIT`이 일어나면 단일 WAL 레코드가 그 `xid`를 commit으로 표시하고, `ROLLBACK`이나 충돌이 발생하면 그 레코드가 끝까지 쓰이지 않으므로 변경 사항은 영원히 누구에게도 보이지 않습니다(이후 `VACUUM`에 의해 회수). "절반만 보이는" 상태는 존재하지 않으며, visibility는 단일 바이트로 뒤집힙니다.

#### A.2 Consistency(일관성) — 트랜잭션 경계에서 불변량 유지

트랜잭션이 시작되기 전에 모든 무결성 제약(`NOT NULL`, `CHECK`, 외래 키, unique index, deferred constraint)을 만족했다면, 커밋 후에도 만족합니다. PostgreSQL은 행이 삽입되는 시점에(또는 `DEFERRABLE` 제약의 경우 commit 시점에) 제약을 검사합니다. 어느 하나라도 실패하면 트랜잭션 전체가 abort되며 — 이는 다시 Atomicity로 이어집니다.

#### A.3 Isolation(격리성) — 동시 트랜잭션은 서로의 진행 중 상태를 보지 못함

PostgreSQL은 **MVCC**(Multi-Version Concurrency Control)를 사용합니다. 모든 행 버전(row version)은 `xmin`(그 행을 생성한 `xid`)과 `xmax`(그 행을 삭제/대체한 `xid`)를 들고 다닙니다. 어떤 reader는 `xmin`이 commit되었고 *자신의 snapshot에서 보이며* `xmax`가 그렇지 않은 경우에만 그 행을 봅니다. 따라서 reader는 writer를 절대 막지 않으며, writer 또한 reader를 막지 않습니다 — 락 기반 시스템이 따라올 수 없는 속성입니다. 11번 레슨에서 자세히 다룹니다.

#### A.4 Durability(영속성) — commit된 데이터는 충돌 후에도 살아남음

`COMMIT`은 WAL 레코드가 디스크에 `fsync`될 때까지 반환하지 않습니다(설정 가능하지만 기본값이 그렇습니다). 1ms 후 서버 전원이 나가도, 재시작 시 WAL을 replay해서 commit된 모든 변경을 복원합니다. heap 파일 자체는 lazy하게 쓰입니다 — 영속성은 로그에서 오는 것이지, 테이블 파일에서 오는 것이 아닙니다.

### 간단한 계산

```sql
-- 계산기처럼 사용
SELECT 1 + 1;
```

출력:
```
 ?column?
----------
        2
(1 row)
```

### 문자열 출력

```sql
SELECT 'Hello, PostgreSQL!';
```

출력:
```
      ?column?
--------------------
 Hello, PostgreSQL!
(1 row)
```

### 현재 시간 확인

```sql
SELECT NOW();
```

출력:
```
              now
-------------------------------
 2024-01-15 10:30:45.123456+09
(1 row)
```

### 버전 확인

```sql
SELECT version();
```

---

## 7. 기본 SQL 문법

### 대소문자

- SQL 키워드: 대소문자 구분 없음 (`SELECT` = `select`)
- 테이블/컬럼명: 기본적으로 소문자로 저장
- 문자열: 작은따옴표 사용 (`'Hello'`)

```sql
-- 이 세 쿼리는 동일
SELECT * FROM users;
select * from users;
Select * From Users;
```

### 주석

```sql
-- 한 줄 주석

/* 여러 줄
   주석 */

SELECT 1; -- 인라인 주석
```

### 문장 끝

- 세미콜론(`;`)으로 문장 종료
- psql에서 여러 줄 입력 후 `;`로 실행

```sql
SELECT
    id,
    name,
    email
FROM users
WHERE active = true;
```

---

## 8. 데이터베이스 생성 및 삭제

### 이론: OID — 객체 식별자

`pg_catalog`의 모든 객체는 4바이트 unsigned integer 주키(primary key)인 **OID**(Object Identifier)를 가집니다. `SELECT * FROM users`를 작성하면 parser는 *문자열* `"users"`를 executor에 넘기지 않습니다 — `pg_class` 행의 OID로 이름을 해석한 뒤, 이후 모든 단계에서 OID를 사용합니다.

OID 덕분에 테이블 rename이 저렴합니다(`pg_class`의 한 행만 바뀌고, 이를 참조하는 인덱스/뷰/외래 키는 동일한 `relid`를 그대로 유지). 또 OID가 클러스터 단위가 아니라 데이터베이스 단위로 unique이기 때문에 cross-database 객체 참조가 허용되지 않는 이유이기도 합니다.

사용자 테이블은 PostgreSQL 12부터 기본적으로 `OID` 시스템 컬럼을 갖지 않게 되었습니다(`WITH OIDS`는 deprecated). 시스템 카탈로그의 행은 여전히 OID를 가지며 의사 컬럼(pseudo-column) `oid`로 접근합니다 — `SELECT oid, datname FROM pg_database;`가 이를 읽는 표준적인 방법입니다.

### 데이터베이스 생성

```sql
-- 각 데이터베이스는 격리된 네임스페이스 — 한 DB의 테이블은 다른 DB의 테이블을 볼 수 없음.
-- 개발/테스트/프로덕션용 별도 DB를 만들어 실수로 인한 교차 오염 방지
CREATE DATABASE mydb;

-- 인코딩과 로케일은 생성 후 변경 불가이므로 미리 지정.
-- UTF8은 모든 언어 지원; 로케일은 정렬 순서와 문자열 비교에 영향
CREATE DATABASE mydb
    ENCODING 'UTF8'
    LC_COLLATE 'ko_KR.UTF-8'
    LC_CTYPE 'ko_KR.UTF-8';
```

### 데이터베이스 전환

```sql
-- psql 메타 명령
\c mydb
```

출력:
```
You are now connected to database "mydb" as user "postgres".
```

### 데이터베이스 삭제

```sql
DROP DATABASE mydb;

-- 존재하는 경우에만 삭제
DROP DATABASE IF EXISTS mydb;
```

---

## 9. 실습 예제

### 실습 1: 환경 설정 확인

```sql
-- 1. PostgreSQL 버전 확인
SELECT version();

-- 2. 현재 사용자 확인
SELECT current_user;

-- 3. 현재 데이터베이스 확인
SELECT current_database();

-- 4. 현재 시간 확인
SELECT NOW();

-- 5. 서버 설정 확인
SHOW server_version;
SHOW data_directory;
```

### 실습 2: 첫 데이터베이스 만들기

```sql
-- 1. 학습용 데이터베이스 생성
CREATE DATABASE study_db;

-- 2. 데이터베이스 목록 확인
\l

-- 3. 새 데이터베이스로 전환
\c study_db

-- 4. 연결 정보 확인
\conninfo
```

### 실습 3: 간단한 테이블 만들기

```sql
-- 1. 테이블 생성
CREATE TABLE hello (
    id SERIAL PRIMARY KEY,
    message TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 2. 데이터 삽입
INSERT INTO hello (message) VALUES ('Hello, PostgreSQL!');
INSERT INTO hello (message) VALUES ('첫 번째 테이블입니다.');

-- 3. 데이터 조회
SELECT * FROM hello;

-- 4. 테이블 구조 확인
\d hello
```

출력 예시:
```
 id |        message        |         created_at
----+-----------------------+----------------------------
  1 | Hello, PostgreSQL!    | 2024-01-15 10:30:45.123456
  2 | 첫 번째 테이블입니다. | 2024-01-15 10:30:50.654321
(2 rows)
```

---

## 10. 문제 해결

### 접속 오류

**오류**: `psql: error: connection refused`
```bash
# 서비스 실행 확인
sudo systemctl status postgresql

# 서비스 시작
sudo systemctl start postgresql
```

**오류**: `FATAL: password authentication failed`
```bash
# pg_hba.conf 확인 및 수정 필요
# 또는 올바른 비밀번호 사용
```

**오류**: `FATAL: database "username" does not exist`
```bash
# 데이터베이스 지정하여 접속
psql -d postgres
```

### Docker 관련

```bash
# 컨테이너 상태 확인
docker ps -a

# 컨테이너 로그 확인
docker logs postgres-study

# 컨테이너 재시작
docker restart postgres-study
```

---

**다음**: [데이터베이스 관리](./02_Database_Management.md)
