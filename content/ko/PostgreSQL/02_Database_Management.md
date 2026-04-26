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

관리 명령어로 들어가기 전, [**이론과 원리**](#이론과-원리) 절을 먼저 읽으세요 — "데이터베이스"란 디스크 위에서 실제로 무엇인지, `pg_catalog`가 자기 자신의 메타데이터를 어떻게 저장하는지, 테이블스페이스(tablespace)의 역할, 그리고 모든 쓰기가 왜 WAL을 먼저 통과해야 하는지를 다룹니다.

---

## 이론과 원리

`CREATE DATABASE` 한 줄은 설정 파일에 이름이 하나 더 추가되는 일이 아닙니다. 물리 디렉터리 트리를 할당하고, 그 안을 template 데이터베이스의 파일로 채우고, 클러스터 전역 카탈로그인 `pg_database`에 행을 삽입하며, 이후 그 데이터베이스의 모든 변경이 영속적으로 남도록 WAL 스트림에 슬롯을 예약합니다. 아래 네 가지 — 클러스터/데이터베이스/스키마 계층, 카탈로그 구조, 물리 저장 계층인 테이블스페이스, WAL — 를 이해하면 "관리 명령어"가 예측 가능하고 검증 가능한 관측 변화로 바뀝니다.

이 절에서 다루는 내용:

- **(A)** 클러스터 → 데이터베이스 → 스키마 → relation 계층과 파일시스템 매핑.
- **(B)** `pg_catalog`가 클러스터 메타데이터를 저장하는 방식, 그리고 어떤 카탈로그가 어떤 `\` 명령의 backing store인가.
- **(C)** 테이블스페이스 — 데이터베이스의 위치를 데이터 디렉터리에서 분리하는 물리 계층.
- **(D)** WAL — `CREATE DATABASE` 같은 DDL을 포함한 모든 변경이 적용되기 전에 먼저 로그에 기록되어야 하는 이유.

### A. 클러스터, 데이터베이스, 스키마, Relation

이 네 단어는 동의어가 아닙니다. 각각 PostgreSQL 네임스페이스 계층의 별개의 층입니다.

| 계층 | 의미 | 조회 방법 |
|------|------|----------|
| **클러스터(Cluster)** | 실행 중인 postmaster 하나 + 데이터 디렉터리(`PGDATA`) 하나. 여러 데이터베이스를 담음. | `pg_lsclusters` (Debian) 또는 `SHOW data_directory;` |
| **데이터베이스(Database)** | 자체 카탈로그 테이블, 인코딩, collation을 가진 격리된 네임스페이스. | `\l` (`pg_database` 조회) |
| **스키마(Schema)** | 한 데이터베이스 *안에서* relation을 묶는 논리 그룹. 기본값은 `public`. | `\dn` (`pg_namespace` 조회) |
| **Relation** | 테이블, 인덱스, 뷰, 시퀀스, 머티리얼라이즈드 뷰. | `\dt` (`pg_class WHERE relkind = 'r'` 조회) |

#### A.1 파일시스템 매핑

`PGDATA/base/` 안에서 모든 데이터베이스는 자기 `pg_database` 행의 OID를 이름으로 갖는 디렉터리입니다. 그 안에서 모든 relation은 자기 `pg_class` 행의 OID를 이름으로 갖는 파일(또는 1 GB 단위로 분할된 파일들)입니다. 그래서:

```
PGDATA/
├── base/
│   ├── 16384/        ← database OID
│   │   ├── 24576     ← relation OID (heap)
│   │   ├── 24576.1   ← 두 번째 1 GB segment
│   │   ├── 24579     ← associated index
│   │   └── ...
│   └── 1/            ← template1 database
└── pg_wal/           ← Write-Ahead Log segments
```

`DROP DATABASE`를 실행하면 PostgreSQL은 두 가지를 합니다 — `pg_database`에서 행을 제거하고, 디렉터리를 `unlink()`합니다. 데이터베이스에는 마법이 없습니다. 본질적으로 디렉터리 하나 + 카탈로그 행 하나입니다.

#### A.2 스키마가 존재하는 이유

만든 모든 것이 `public`에 산다면, 이름 충돌이 금세 문제가 됩니다. 스키마는 하위 네임스페이스를 제공합니다 — `app.users`와 `analytics.users`는 서로 다른 두 테이블입니다. `search_path` 설정은 unqualified 이름을 어떤 스키마에서 찾을지 결정합니다. `SET search_path = app, public;`은 `users`가 `app.users`로 먼저 해석되고, 없으면 `public.users`로 해석된다는 뜻입니다.

### B. `pg_catalog` 내부

`psql`의 모든 관리용 `\` 메타 명령은 `pg_catalog`에 대한 SQL 쿼리의 wrapper입니다. `\set ECHO_HIDDEN on`을 켜면 실제 쿼리를 볼 수 있습니다. 이 레슨에 관련된 카탈로그:

- **`pg_database`** — 데이터베이스 한 행씩. 컬럼: `oid`, `datname`, `datdba`(owner), `encoding`, `datcollate`, `dattablespace`, `datallowconn`, `datistemplate`. `\l`의 backing store.
- **`pg_authid`** — 클러스터 전역 role 한 행씩. 컬럼: `oid`, `rolname`, `rolsuper`, `rolcanlogin`, `rolpassword`(hash). `\du`의 backing store.
- **`pg_roles`** — `pg_authid`에 대한 view로, password 컬럼을 숨김. 비-superuser에게 안전.
- **`pg_tablespace`** — 테이블스페이스 한 행씩. 컬럼: `oid`, `spcname`, `spcowner`, `spcacl`. `\db`의 backing store.
- **`pg_namespace`** — 스키마 한 행씩. `\dn`의 backing store.

카탈로그 테이블 자체도 일반 테이블입니다 — 다만 다른 모든 쿼리를 plan하기 위해 엔진이 읽는 테이블일 뿐입니다. ACID를 따릅니다. `CREATE ROLE alice;`는 정확히 `INSERT INTO pg_authid`(과 약간의 부수 효과)이며, 둘러싼 트랜잭션이 abort되면 함께 rollback됩니다.

#### B.1 클러스터 전역 vs 데이터베이스 로컬

미묘한 점 하나 — 어떤 카탈로그는 *클러스터 전체에서 공유*되고, 어떤 카탈로그는 *데이터베이스 단위*입니다. role, database, tablespace는 클러스터 전역(어느 데이터베이스에 접속하든 동일한 `alice`로 로그인합니다). 테이블, 스키마, 함수, 인덱스는 데이터베이스 로컬(각 데이터베이스가 자기 `pg_class`를 가짐). 그래서 두 데이터베이스를 `JOIN`할 수 없습니다 — 카탈로그를 공유하지 않기 때문입니다.

### C. 테이블스페이스

기본적으로 모든 데이터베이스는 `PGDATA/base/` 아래에 삽니다. **테이블스페이스**는 "이 데이터베이스(또는 이 테이블, 이 인덱스)는 다른 파일시스템 경로 아래에 저장하라"고 말하게 해 줍니다. 내부적으로는 단지 `PGDATA/pg_tblspc/<oid>`에서 설정된 디렉터리로의 심볼릭 링크입니다.

#### C.1 사용 시나리오

1. **hot 데이터는 SSD, cold 데이터는 HDD.** 파티션 테이블은 빠른 테이블스페이스에, 과거 archive는 느린 테이블스페이스에.
2. **테넌트별 디스크 quota.** 각 테넌트가 자체 size limit을 가진 자체 파일시스템 위의 테이블스페이스를 가짐.
3. **I/O를 여러 볼륨에 분산** — 단일 디스크가 병목일 때.

`CREATE TABLESPACE archive LOCATION '/mnt/cold/pg';`는 세 가지를 합니다 — 디렉터리가 존재하고 비어 있는지 검증하고, `pg_tablespace`에 행을 삽입하고, 심볼릭 링크를 생성합니다. 이후 `CREATE TABLE foo (...) TABLESPACE archive;`는 `foo`의 파일들을 그 경로에 둡니다.

#### C.2 하지 않는 것

테이블스페이스는 보안 경계가 *아니고*, 별개의 데이터베이스가 *아니며*, 클러스터 간에 파일을 공유하는 방법도 *아닙니다*. 서로 다른 두 클러스터는 동일한 테이블스페이스 디렉터리를 공유할 수 없습니다 — OID 공간이 충돌하기 때문입니다.

### D. WAL — Write-Ahead Log

근본 내구성 규칙 — **대응되는 로그 레코드가 `pg_wal/`에 디스크로 내려가기 전까지는 어떤 변경도 heap page에 적용되지 않는다**. 이것이 "write-ahead"입니다. 로그가 먼저 쓰이고, 데이터 파일은 lazy하게 갱신됩니다.

이유 — 서버가 충돌하면 재시작 시 마지막 checkpoint부터 WAL을 replay해서, heap 파일에 아직 도달하지 못한 commit된 변경을 복원할 수 있습니다. 로그가 없다면 commit마다 모든 dirty 데이터 페이지를 `fsync()`해야 하는데, 데이터 페이지는 디스크에 흩어져 있는 반면 로그는 sequential하므로 그쪽이 훨씬 느립니다.

#### D.1 WAL 레코드의 내용

WAL 레코드는 *물리적* 변경을 기술합니다 — "relation 24576의 page 17의 32-40 바이트를 이 값으로 설정", 그리고 그 변경을 일으킨 트랜잭션 ID. commit, abort, prepared transaction 상태, full-page image(checkpoint 이후 페이지에 대한 첫 변경은 torn write로 인한 손상에서 복구 가능하도록 페이지 전체를 기록) 같은 논리 레코드도 있습니다.

#### D.2 Checkpoint

주기적으로(매 `checkpoint_timeout`, 기본 5분, 또는 `max_wal_size` 만큼의 WAL이 생성되면) checkpointer 프로세스가 모든 dirty buffer를 데이터 파일로 flush하고, *checkpoint 레코드*를 WAL에 씁니다. 충돌 후 복구는 마지막 checkpoint 이후의 WAL만 replay하면 되며, 그 이전 WAL은 재활용되거나 archive됩니다.

#### D.3 DDL과 `CREATE DATABASE`도 WAL을 거치는 이유

모든 카탈로그 변경은 heap 변경이기도 합니다(`pg_class`, `pg_database` 등에 쓰기). 따라서 WAL을 생성합니다. 그래서 `CREATE DATABASE`가 standby로 복제될 수 있습니다 — standby는 디렉터리 생성과 template 복사 작업까지 포함된 WAL 레코드를 단순히 replay합니다.

### 이론에서 아래 명령으로

이어지는 각 절은 위 개념들이 구체화된 형태입니다:

- **`CREATE DATABASE`, `DROP DATABASE`** — `pg_database`(§B)를 변경하고 `base/` 아래 디렉터리를 생성/제거(§A.1)하며, 그 과정에서 WAL 레코드를 생성(§D.3).
- **`CREATE ROLE`, `GRANT`** — 클러스터 전역 `pg_authid`와 `pg_class`의 객체별 ACL 컬럼을 변경(§B.1).
- **`CREATE SCHEMA`, `search_path`** — `pg_namespace`와 세션별 이름 해석 규칙에 대해 동작(§A.2).
- **`CREATE TABLESPACE`** — `pg_tablespace`에 행을 추가하고 `pg_tblspc/` 아래에 심볼릭 링크 생성(§C).
- **`pg_dumpall`, `pg_dump`** — `pg_catalog`를 읽어 DDL을 재구성. `pg_dumpall`은 `pg_dump`가 건너뛰는 클러스터 전역 카탈로그(role, tablespace)까지 포함(§B.1).

---

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
