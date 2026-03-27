# 데이터베이스 관리

**이전**: [PostgreSQL 기초](./01_PostgreSQL_Basics.md) | **다음**: [테이블과 데이터 타입](./03_Tables_and_Data_Types.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. PostgreSQL 서버의 계층 구조(서버, 데이터베이스, 스키마, 테이블)를 설명할 수 있다
2. 적절한 옵션을 사용하여 데이터베이스를 생성, 조회, 이름 변경, 삭제할 수 있다
3. 템플릿 데이터베이스(`template0`, `template1`)의 역할을 설명할 수 있다
4. 특정 권한을 가진 롤(Role)(사용자 및 그룹)을 생성하고 관리할 수 있다
5. GRANT/REVOKE 시스템을 적용하여 데이터베이스, 스키마, 테이블 수준에서 접근을 제어할 수 있다
6. 스키마(Schema)를 구성하여 데이터베이스 내 객체를 논리적으로 정리할 수 있다
7. 권한 모델을 설계할 때 최소 권한 원칙(Principle of Least Privilege)을 적용할 수 있다

---

운영 환경에서는 올바른 SQL을 작성하는 것만큼이나 데이터베이스, 사용자, 권한을 적절히 관리하는 것이 중요합니다. 잘못 구성된 롤(Role)이나 과도하게 허용된 권한은 민감한 데이터를 노출시키거나 우발적인 삭제를 허용할 수 있습니다. 이 레슨에서는 처음부터 안전하고 체계적인 PostgreSQL 환경을 구축하는 데 필요한 관리 명령어들을 다룹니다.

## 1. 데이터베이스 기본 개념

PostgreSQL에서 데이터베이스는 테이블, 뷰, 함수 등을 담는 최상위 컨테이너입니다.

```
┌─────────────────────────────────────────────────────┐
│                PostgreSQL 서버                       │
├─────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │   DB 1   │  │   DB 2   │  │   DB 3   │          │
│  │ ┌──────┐ │  │ ┌──────┐ │  │ ┌──────┐ │          │
│  │ │Schema│ │  │ │Schema│ │  │ │Schema│ │          │
│  │ │┌────┐│ │  │ │┌────┐│ │  │ │┌────┐│ │          │
│  │ ││Table│ │  │ ││Table│ │  │ ││Table│ │          │
│  │ │└────┘│ │  │ │└────┘│ │  │ │└────┘│ │          │
│  │ └──────┘ │  │ └──────┘ │  │ └──────┘ │          │
│  └──────────┘  └──────────┘  └──────────┘          │
└─────────────────────────────────────────────────────┘
```

---

## 2. 데이터베이스 생성

### 기본 생성

```sql
CREATE DATABASE mydb;
```

### 옵션과 함께 생성

```sql
CREATE DATABASE mydb
    WITH
    OWNER = myuser
    ENCODING = 'UTF8'
    LC_COLLATE = 'ko_KR.UTF-8'
    LC_CTYPE = 'ko_KR.UTF-8'
    TEMPLATE = template0
    CONNECTION LIMIT = 100;
```

### 주요 옵션

| 옵션 | 설명 |
|------|------|
| `OWNER` | 데이터베이스 소유자 |
| `ENCODING` | 문자 인코딩 (UTF8 권장) |
| `LC_COLLATE` | 정렬 순서 로케일 |
| `LC_CTYPE` | 문자 분류 로케일 |
| `TEMPLATE` | 템플릿 데이터베이스 |
| `CONNECTION LIMIT` | 최대 동시 연결 수 (-1은 무제한) |

### 템플릿 데이터베이스

```sql
-- template1은 기본 템플릿 — template1에 추가한 확장이나 객체가
-- 모든 새 데이터베이스에 자동 포함됨. 조직 전체 기본값 설정에 활용
CREATE DATABASE mydb TEMPLATE template1;

-- template0은 수정되지 않은 원본 템플릿 — template1이 클러스터의
-- 원래 설정을 상속하므로, 다른 인코딩/로케일이 필요할 때 사용
CREATE DATABASE mydb TEMPLATE template0 ENCODING 'UTF8';
```

---

## 3. 데이터베이스 목록 및 정보

### 데이터베이스 목록

```sql
-- psql 메타 명령
\l

-- 상세 정보
\l+

-- SQL 쿼리
SELECT datname, datdba, encoding, datcollate
FROM pg_database;
```

### 현재 데이터베이스 확인

```sql
SELECT current_database();
```

### 데이터베이스 크기 확인

```sql
-- 특정 데이터베이스 크기
SELECT pg_size_pretty(pg_database_size('mydb'));

-- 모든 데이터베이스 크기
SELECT
    datname,
    pg_size_pretty(pg_database_size(datname)) AS size
FROM pg_database
ORDER BY pg_database_size(datname) DESC;
```

---

## 4. 데이터베이스 전환 및 수정

### 데이터베이스 전환

```sql
-- psql에서만 사용 가능
\c mydb

-- 또는
\connect mydb
```

### 데이터베이스 이름 변경

```sql
-- 해당 DB에 연결된 세션이 없어야 함
ALTER DATABASE oldname RENAME TO newname;
```

### 데이터베이스 소유자 변경

```sql
ALTER DATABASE mydb OWNER TO newowner;
```

### 연결 제한 변경

```sql
ALTER DATABASE mydb CONNECTION LIMIT 50;
```

---

## 5. 데이터베이스 삭제

```sql
-- 기본 삭제
DROP DATABASE mydb;

-- 존재하는 경우에만 삭제
DROP DATABASE IF EXISTS mydb;

-- 강제 삭제 (연결된 세션 종료)
DROP DATABASE mydb WITH (FORCE);  -- PostgreSQL 13+
```

### 연결된 세션 확인 및 종료

```sql
-- 연결된 세션 확인
SELECT pid, usename, application_name, client_addr
FROM pg_stat_activity
WHERE datname = 'mydb';

-- 특정 세션 종료
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE datname = 'mydb' AND pid <> pg_backend_pid();
```

---

## 6. 사용자(Role) 관리

PostgreSQL에서는 사용자와 그룹을 모두 "Role"이라고 합니다.

### Role 생성

```sql
-- 기본 사용자 생성
CREATE ROLE myuser LOGIN PASSWORD 'mypassword';

-- CREATE USER는 LOGIN이 기본으로 포함됨
CREATE USER myuser WITH PASSWORD 'mypassword';

-- 다양한 옵션
CREATE ROLE admin_user WITH
    LOGIN
    PASSWORD 'securepassword'
    CREATEDB
    CREATEROLE
    VALID UNTIL '2025-12-31';
```

### Role 옵션

| 옵션 | 설명 |
|------|------|
| `LOGIN` | 로그인 가능 |
| `SUPERUSER` | 슈퍼유저 권한 |
| `CREATEDB` | 데이터베이스 생성 권한 |
| `CREATEROLE` | Role 생성 권한 |
| `INHERIT` | 그룹 권한 상속 |
| `REPLICATION` | 복제 권한 |
| `PASSWORD 'xxx'` | 비밀번호 설정 |
| `VALID UNTIL 'timestamp'` | 계정 만료일 |
| `CONNECTION LIMIT n` | 최대 연결 수 |

### Role 목록 확인

```sql
-- psql 메타 명령
\du

-- 상세 정보
\du+

-- SQL 쿼리
SELECT rolname, rolsuper, rolcreatedb, rolcreaterole, rolcanlogin
FROM pg_roles;
```

### Role 수정

```sql
-- 비밀번호 변경
ALTER ROLE myuser WITH PASSWORD 'newpassword';

-- 권한 추가
ALTER ROLE myuser CREATEDB;

-- 권한 제거
ALTER ROLE myuser NOCREATEDB;

-- 이름 변경
ALTER ROLE oldname RENAME TO newname;
```

### Role 삭제

```sql
DROP ROLE myuser;

-- 존재하는 경우에만 삭제
DROP ROLE IF EXISTS myuser;
```

---

## 7. 권한 관리

### 데이터베이스 권한

```sql
-- 데이터베이스 연결 권한 부여
GRANT CONNECT ON DATABASE mydb TO myuser;

-- 데이터베이스의 모든 권한 부여
GRANT ALL PRIVILEGES ON DATABASE mydb TO myuser;

-- 권한 회수
REVOKE CONNECT ON DATABASE mydb FROM myuser;
```

### 스키마 권한

```sql
-- 스키마 사용 권한
GRANT USAGE ON SCHEMA public TO myuser;

-- 스키마 내 객체 생성 권한
GRANT CREATE ON SCHEMA public TO myuser;
```

### 테이블 권한

```sql
-- 특정 테이블 SELECT 권한
GRANT SELECT ON TABLE users TO myuser;

-- 특정 테이블 모든 권한
GRANT ALL PRIVILEGES ON TABLE users TO myuser;

-- 스키마 내 모든 테이블 권한
GRANT SELECT ON ALL TABLES IN SCHEMA public TO myuser;

-- DEFAULT PRIVILEGES 없이는 새 테이블마다 수동 GRANT 필요.
-- 자동으로 새 테이블에 권한이 상속되도록 설정
ALTER DEFAULT PRIVILEGES IN SCHEMA public
GRANT SELECT ON TABLES TO myuser;
```

### 권한 종류

| 권한 | 적용 대상 | 설명 |
|------|-----------|------|
| `SELECT` | 테이블, 뷰 | 데이터 조회 |
| `INSERT` | 테이블 | 데이터 삽입 |
| `UPDATE` | 테이블 | 데이터 수정 |
| `DELETE` | 테이블 | 데이터 삭제 |
| `TRUNCATE` | 테이블 | 테이블 비우기 |
| `REFERENCES` | 테이블 | 외래키 생성 |
| `TRIGGER` | 테이블 | 트리거 생성 |
| `CREATE` | DB, 스키마 | 객체 생성 |
| `CONNECT` | DB | 연결 |
| `USAGE` | 스키마, 시퀀스 | 사용 |
| `EXECUTE` | 함수 | 실행 |

### 권한 확인

```sql
-- 테이블 권한 확인
\dp users

-- 또는
SELECT grantee, privilege_type
FROM information_schema.table_privileges
WHERE table_name = 'users';
```

---

## 8. 스키마 관리

스키마는 데이터베이스 내에서 테이블을 논리적으로 그룹화합니다.

### 스키마 생성

```sql
-- 기본 생성
CREATE SCHEMA myschema;

-- 소유자 지정
CREATE SCHEMA myschema AUTHORIZATION myuser;
```

### 스키마 목록

```sql
-- psql 메타 명령
\dn

-- SQL 쿼리
SELECT schema_name FROM information_schema.schemata;
```

### 스키마 사용

```sql
-- 테이블 생성 시 스키마 지정
CREATE TABLE myschema.users (
    id SERIAL PRIMARY KEY,
    name TEXT
);

-- 검색 경로 설정
SET search_path TO myschema, public;

-- 검색 경로 확인
SHOW search_path;
```

### 스키마 삭제

```sql
-- 빈 스키마 삭제
DROP SCHEMA myschema;

-- 내용물 포함 삭제
DROP SCHEMA myschema CASCADE;
```

---

## 9. 실습 예제

### 실습 1: 프로젝트용 데이터베이스 구성

```sql
-- 1. 데이터베이스 생성
CREATE DATABASE project_db;

-- 2. 데이터베이스 전환
\c project_db

-- 3. 애플리케이션용 사용자 생성
CREATE USER app_user WITH PASSWORD 'app_password';

-- 4. 읽기 전용 사용자 생성
CREATE USER readonly_user WITH PASSWORD 'readonly_password';

-- 5. 스키마 생성
CREATE SCHEMA app_schema;
CREATE SCHEMA report_schema;

-- 6. 권한 설정
-- app_user: 전체 권한
GRANT ALL PRIVILEGES ON DATABASE project_db TO app_user;
GRANT ALL PRIVILEGES ON SCHEMA app_schema TO app_user;

-- readonly_user: 읽기 전용
GRANT CONNECT ON DATABASE project_db TO readonly_user;
GRANT USAGE ON SCHEMA app_schema TO readonly_user;
GRANT SELECT ON ALL TABLES IN SCHEMA app_schema TO readonly_user;

-- 7. 향후 테이블에도 권한 적용
ALTER DEFAULT PRIVILEGES IN SCHEMA app_schema
GRANT SELECT ON TABLES TO readonly_user;
```

### 실습 2: 사용자별 권한 테스트

```sql
-- postgres 사용자로 테이블 생성
CREATE TABLE app_schema.products (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    price NUMERIC(10,2)
);

INSERT INTO app_schema.products (name, price) VALUES
('노트북', 1500000),
('마우스', 35000);

-- readonly_user로 접속하여 테스트
-- psql -U readonly_user -d project_db

-- SELECT는 성공
SELECT * FROM app_schema.products;

-- INSERT는 실패 (권한 없음)
INSERT INTO app_schema.products (name, price) VALUES ('키보드', 80000);
-- ERROR: permission denied for table products
```

### 실습 3: 데이터베이스 정보 조회

```sql
-- 모든 데이터베이스 크기
SELECT
    datname AS database,
    pg_size_pretty(pg_database_size(datname)) AS size
FROM pg_database
WHERE datistemplate = false
ORDER BY pg_database_size(datname) DESC;

-- 현재 연결 정보
SELECT
    pid,
    usename,
    datname,
    client_addr,
    state,
    query
FROM pg_stat_activity
WHERE datname = current_database();

-- Role별 권한 요약
SELECT
    r.rolname,
    r.rolsuper AS superuser,
    r.rolcreatedb AS can_create_db,
    r.rolcreaterole AS can_create_role,
    r.rolcanlogin AS can_login
FROM pg_roles r
WHERE r.rolname NOT LIKE 'pg_%'
ORDER BY r.rolname;
```

---

## 10. 보안 모범 사례

### 최소 권한 원칙

```sql
-- 최소 권한: 앱이 실제로 수행하는 작업만 부여.
-- app_user가 DELETE/DROP 권한을 가지면 공격자도 그 권한을 상속
GRANT SELECT, INSERT, UPDATE ON users TO app_user;

-- ALL PRIVILEGES는 TRUNCATE, REFERENCES, TRIGGER까지 포함 —
-- 대부분의 애플리케이션 사용자에게 불필요한 권한
-- GRANT ALL PRIVILEGES ON ... -- 비권장
```

### 슈퍼유저 사용 최소화

```sql
-- 일반 작업은 일반 사용자로
-- 관리 작업만 슈퍼유저로
```

### 비밀번호 정책

```sql
-- 강력한 비밀번호 사용
CREATE USER myuser WITH PASSWORD 'C0mplex!P@ssw0rd';

-- 계정 만료일 설정
ALTER ROLE myuser VALID UNTIL '2025-12-31';
```

---

**이전**: [PostgreSQL 기초](./01_PostgreSQL_Basics.md) | **다음**: [테이블과 데이터 타입](./03_Tables_and_Data_Types.md)
