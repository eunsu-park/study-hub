# PostgreSQL 기초

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
# PostgreSQL 16 컨테이너 실행
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

### 데이터베이스 생성

```sql
-- 기본 생성
CREATE DATABASE mydb;

-- 옵션 지정
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

## 다음 단계

[02_Database_Management.md](./02_Database_Management.md)에서 데이터베이스와 사용자 관리를 자세히 다뤄봅시다!
