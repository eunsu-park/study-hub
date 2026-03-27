# 20. 보안과 접근 제어

**이전**: [전문 검색](./19_Full_Text_Search.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 그룹 역할(group role), 로그인 역할(login role), 최소 권한 원칙(principle of least privilege)을 활용한 역할 계층 구조를 설계할 수 있다
2. 스키마, 테이블, 시퀀스, 함수에 대한 객체 수준 및 컬럼 수준 권한을 부여하고 취소할 수 있다
3. 멀티 테넌트 격리 및 부서 기반 접근 제어를 위한 행 수준 보안(Row-Level Security, RLS) 정책을 구현할 수 있다
4. 적절한 인증 방법(scram-sha-256, peer, cert)으로 pg_hba.conf 인증 규칙을 구성할 수 있다
5. 클라이언트 인증서 인증을 포함한 SSL/TLS 암호화 연결을 설정할 수 있다
6. PostgreSQL 내장 로깅과 pgAudit 확장을 활용하여 감사 로깅(audit logging)을 활성화할 수 있다
7. SQL 인젝션 방지, pgcrypto를 활용한 데이터 암호화, 연결 보안 강화 등 보안 모범 사례를 적용할 수 있다

---

보안은 나중에 덧붙이는 기능이 아닙니다 — 처음부터 데이터베이스 설계에 포함되어야 합니다. 잘못 구성된 PostgreSQL 인스턴스는 민감한 데이터를 노출시키거나, 권한 상승을 허용하거나, 공격자의 진입점이 될 수 있습니다. 이 레슨은 PostgreSQL의 심층 방어(defense-in-depth) 보안 모델을 다룹니다: 네트워크 수준 제어와 인증(authentication)부터 세밀한 인가(authorization)와 행 수준 보안, 암호화와 감사 로깅까지. 멀티 테넌트 SaaS 애플리케이션을 구축하든, 기존 배포를 강화하든, 이 기법들은 안전한 PostgreSQL 환경의 기반이 됩니다.

## 목차
1. [보안 개요](#1-보안-개요)
2. [역할과 권한](#2-역할과-권한)
3. [행 수준 보안(RLS)](#3-행-수준-보안rls)
4. [인증(pg_hba.conf)](#4-인증pg_hbaconf)
5. [SSL/TLS 연결](#5-ssltls-연결)
6. [감사 로깅](#6-감사-로깅)
7. [보안 모범 사례](#7-보안-모범-사례)
8. [연습 문제](#8-연습-문제)

---

## 1. 보안 개요

### 1.1 PostgreSQL 보안 계층

```
┌─────────────────────────────────────────────────────────────────┐
│              PostgreSQL 보안 아키텍처                            │
│                                                                 │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  계층 1: 네트워크 보안                                │       │
│  │  ┌──────────────────────────────────────────────┐    │       │
│  │  │  방화벽, listen_addresses, 포트               │    │       │
│  │  └──────────────────────────────────────────────┘    │       │
│  │                                                      │       │
│  │  계층 2: 인증 (pg_hba.conf)                          │       │
│  │  ┌──────────────────────────────────────────────┐    │       │
│  │  │  누가 연결할 수 있나? 어디서? 어떻게?          │    │       │
│  │  └──────────────────────────────────────────────┘    │       │
│  │                                                      │       │
│  │  계층 3: 인가 (GRANT/REVOKE)                         │       │
│  │  ┌──────────────────────────────────────────────┐    │       │
│  │  │  객체 수준 권한                               │    │       │
│  │  └──────────────────────────────────────────────┘    │       │
│  │                                                      │       │
│  │  계층 4: 행 수준 보안 (RLS)                          │       │
│  │  ┌──────────────────────────────────────────────┐    │       │
│  │  │  행 수준 접근 제어 정책                       │    │       │
│  │  └──────────────────────────────────────────────┘    │       │
│  │                                                      │       │
│  │  계층 5: 컬럼 수준 보안                              │       │
│  │  ┌──────────────────────────────────────────────┐    │       │
│  │  │  컬럼별 GRANT                                 │    │       │
│  │  └──────────────────────────────────────────────┘    │       │
│  │                                                      │       │
│  │  계층 6: 암호화                                      │       │
│  │  ┌──────────────────────────────────────────────┐    │       │
│  │  │  SSL/TLS, pgcrypto, 저장 데이터 암호화         │    │       │
│  │  └──────────────────────────────────────────────┘    │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 역할과 권한

### 2.1 역할 관리

```sql
-- 역할 생성
CREATE ROLE app_reader LOGIN PASSWORD 'secure_password';
CREATE ROLE app_writer LOGIN PASSWORD 'secure_password';
CREATE ROLE app_admin LOGIN PASSWORD 'secure_password' CREATEROLE;

-- 그룹 역할 (LOGIN 없음)
CREATE ROLE readonly_group NOLOGIN;
CREATE ROLE readwrite_group NOLOGIN;
CREATE ROLE admin_group NOLOGIN;

-- 역할 멤버십
GRANT readonly_group TO app_reader;
GRANT readwrite_group TO app_writer;
GRANT admin_group TO app_admin;

-- 역할 속성
ALTER ROLE app_reader SET statement_timeout = '30s';
ALTER ROLE app_writer CONNECTION LIMIT 10;
ALTER ROLE app_admin VALID UNTIL '2026-12-31';
```

### 2.2 객체 권한

```sql
-- 스키마 권한
GRANT USAGE ON SCHEMA public TO readonly_group;
GRANT CREATE ON SCHEMA public TO admin_group;

-- 테이블 권한
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly_group;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO readwrite_group;

-- 미래 객체에 대한 기본 권한
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT ON TABLES TO readonly_group;

ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO readwrite_group;

-- 시퀀스 권한
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO readwrite_group;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT USAGE ON SEQUENCES TO readwrite_group;

-- 함수 권한
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO readwrite_group;
```

### 2.3 컬럼 수준 권한

```sql
-- 특정 컬럼에만 접근 권한 부여
CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    salary NUMERIC(10,2),
    ssn TEXT
);

-- HR은 모든 것을 볼 수 있음
GRANT SELECT ON employees TO hr_role;

-- 관리자는 민감하지 않은 컬럼만 볼 수 있음
GRANT SELECT (id, name, email) ON employees TO manager_role;

-- 접근 권한 취소
REVOKE SELECT (salary, ssn) ON employees FROM manager_role;
```

### 2.4 권한 검사

```sql
-- 테이블 권한 확인
SELECT grantee, privilege_type
FROM information_schema.role_table_grants
WHERE table_name = 'employees';

-- 컬럼 권한 확인
SELECT grantee, column_name, privilege_type
FROM information_schema.column_privileges
WHERE table_name = 'employees';

-- 역할 멤버십 목록
SELECT r.rolname AS role, m.rolname AS member
FROM pg_auth_members am
JOIN pg_roles r ON r.oid = am.roleid
JOIN pg_roles m ON m.oid = am.member;

-- 현재 사용자의 권한 확인
SELECT has_table_privilege('app_reader', 'employees', 'SELECT');
SELECT has_schema_privilege('app_reader', 'public', 'USAGE');
```

---

## 3. 행 수준 보안(RLS)

### 3.1 RLS 기초

```sql
-- 테이블에서 RLS 활성화
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT,
    owner_name TEXT NOT NULL DEFAULT current_user,
    department TEXT,
    is_public BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);

ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

-- 정책: 사용자는 자신의 문서 또는 공개 문서만 볼 수 있음
CREATE POLICY documents_select ON documents
    FOR SELECT
    USING (owner_name = current_user OR is_public = TRUE);

-- 정책: 사용자는 자신의 문서만 수정할 수 있음
CREATE POLICY documents_modify ON documents
    FOR ALL
    USING (owner_name = current_user)
    WITH CHECK (owner_name = current_user);
```

### 3.2 정책 유형

```sql
-- USING: SELECT, UPDATE(기존 행), DELETE에 대한 행 필터링
-- WITH CHECK: INSERT, UPDATE(새 행)에 대한 행 검증

-- 삽입 정책
CREATE POLICY documents_insert ON documents
    FOR INSERT
    WITH CHECK (owner_name = current_user);

-- 업데이트 정책 (USING과 WITH CHECK 분리)
CREATE POLICY documents_update ON documents
    FOR UPDATE
    USING (owner_name = current_user)           -- 자신의 행만 볼 수 있음
    WITH CHECK (owner_name = current_user);     -- 자신만 소유자로 설정 가능

-- 삭제 정책
CREATE POLICY documents_delete ON documents
    FOR DELETE
    USING (owner_name = current_user);
```

### 3.3 멀티 테넌트 RLS

```sql
-- 애플리케이션 설정 변수를 사용한 테넌트 격리
CREATE TABLE tenant_data (
    id SERIAL PRIMARY KEY,
    tenant_id INT NOT NULL,
    data JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

ALTER TABLE tenant_data ENABLE ROW LEVEL SECURITY;

-- 세션 변수를 사용하는 정책
CREATE POLICY tenant_isolation ON tenant_data
    FOR ALL
    USING (tenant_id = current_setting('app.tenant_id')::INT)
    WITH CHECK (tenant_id = current_setting('app.tenant_id')::INT);

-- 애플리케이션이 테넌트 컨텍스트 설정
SET app.tenant_id = '42';
SELECT * FROM tenant_data;  -- 테넌트 42의 데이터만 표시

-- 관리자 역할에 대한 RLS 우회
ALTER TABLE tenant_data FORCE ROW LEVEL SECURITY;  -- 테이블 소유자에게도 적용
CREATE POLICY admin_bypass ON tenant_data
    FOR ALL
    TO admin_group
    USING (TRUE)
    WITH CHECK (TRUE);
```

### 3.4 부서 기반 접근

```sql
-- 사용자는 자신의 부서 + 공개 문서를 볼 수 있음
CREATE TABLE user_departments (
    username TEXT PRIMARY KEY,
    department TEXT NOT NULL
);

INSERT INTO user_departments VALUES
('alice', 'engineering'),
('bob', 'marketing'),
('charlie', 'engineering');

CREATE POLICY dept_access ON documents
    FOR SELECT
    USING (
        is_public = TRUE
        OR owner_name = current_user
        OR department IN (
            SELECT department FROM user_departments
            WHERE username = current_user
        )
    );
```

---

## 4. 인증(pg_hba.conf)

### 4.1 pg_hba.conf 구조

```
# TYPE  DATABASE  USER       ADDRESS        METHOD

# 로컬 연결
local   all       postgres                  peer
local   all       all                       scram-sha-256

# IPv4 연결
host    all       all        127.0.0.1/32   scram-sha-256
host    mydb      app_user   10.0.0.0/8     scram-sha-256
host    all       all        0.0.0.0/0      reject

# IPv6 연결
host    all       all        ::1/128        scram-sha-256

# SSL 필수
hostssl mydb      app_user   10.0.0.0/8     scram-sha-256
hostnossl all     all        0.0.0.0/0      reject
```

### 4.2 인증 방법

```
┌────────────────┬──────────────────────────────────────────────────┐
│ 방법            │ 설명                                             │
├────────────────┼──────────────────────────────────────────────────┤
│ trust          │ 인증 없음 (프로덕션에서 절대 사용 금지)            │
│ reject         │ 항상 연결 거부                                    │
│ scram-sha-256  │ 챌린지-응답 (권장)                                │
│ md5            │ MD5 해시 (레거시, scram-sha-256 사용 권장)        │
│ password       │ 평문 (절대 사용 금지)                             │
│ peer           │ OS 사용자 = PG 사용자 (로컬만)                    │
│ ident          │ OS 사용자 매핑 (TCP/IP)                           │
│ cert           │ SSL 클라이언트 인증서                             │
│ ldap           │ LDAP 서버 인증                                    │
│ gss            │ Kerberos/GSSAPI                                   │
└────────────────┴──────────────────────────────────────────────────┘
```

### 4.3 비밀번호 관리

```sql
-- scram-sha-256 사용 (PG 14+ 기본값)
SET password_encryption = 'scram-sha-256';

-- 암호화된 비밀번호로 사용자 생성
CREATE ROLE app_user LOGIN PASSWORD 'strong_password_here';

-- 비밀번호 변경 강제
ALTER ROLE app_user VALID UNTIL '2026-03-01';

-- 비밀번호 암호화 방법 확인
SELECT rolname, rolpassword ~ '^SCRAM-SHA-256' AS is_scram
FROM pg_authid
WHERE rolcanlogin;
```

### 4.4 구성 다시 로드

```bash
# pg_hba.conf 편집 후 다시 로드:
pg_ctl reload -D /path/to/data

# 또는 SQL에서:
# SELECT pg_reload_conf();

# 현재 설정 확인
# SELECT * FROM pg_hba_file_rules;  -- PG 15+
```

---

## 5. SSL/TLS 연결

### 5.1 SSL 활성화

```bash
# 자체 서명 인증서 생성
openssl req -new -x509 -days 365 -nodes \
    -out server.crt -keyout server.key \
    -subj "/CN=postgres-server"

chmod 600 server.key
```

```
# postgresql.conf
ssl = on
ssl_cert_file = 'server.crt'
ssl_key_file = 'server.key'
ssl_min_protocol_version = 'TLSv1.2'
ssl_ciphers = 'HIGH:MEDIUM:+3DES:!aNULL'
```

### 5.2 클라이언트 인증서 인증

```bash
# CA 생성
openssl req -new -x509 -days 3650 -nodes \
    -out root.crt -keyout root.key \
    -subj "/CN=MyCA"

# 클라이언트 인증서 생성
openssl req -new -nodes \
    -out client.csr -keyout client.key \
    -subj "/CN=app_user"

openssl x509 -req -in client.csr -days 365 \
    -CA root.crt -CAkey root.key -CAcreateserial \
    -out client.crt
```

```
# pg_hba.conf — 클라이언트 인증서 필수
hostssl mydb  app_user  10.0.0.0/8  cert  clientcert=verify-full
```

### 5.3 SSL 연결 확인

```sql
-- 현재 연결이 SSL을 사용하는지 확인
SELECT ssl, version, cipher
FROM pg_stat_ssl
WHERE pid = pg_backend_pid();

-- 모든 SSL 연결 확인
SELECT s.pid, s.ssl, s.version, s.cipher, a.usename, a.client_addr
FROM pg_stat_ssl s
JOIN pg_stat_activity a ON s.pid = a.pid
WHERE s.ssl = TRUE;
```

---

## 6. 감사 로깅

### 6.1 기본 로깅 (postgresql.conf)

```
# postgresql.conf
log_statement = 'all'           # none, ddl, mod, all
log_min_duration_statement = 0  # 모든 명령문을 시간과 함께 로그
log_connections = on
log_disconnections = on
log_line_prefix = '%t [%p] %u@%d '  # 타임스탬프, pid, 사용자, 데이터베이스
```

### 6.2 pgAudit 확장

```sql
-- pgAudit 설치 (shared_preload_libraries에 있어야 함)
-- postgresql.conf: shared_preload_libraries = 'pgaudit'

CREATE EXTENSION pgaudit;

-- 감사 로깅 구성
SET pgaudit.log = 'write, ddl';        -- 쓰기와 DDL 로그
SET pgaudit.log_catalog = off;          -- 시스템 카탈로그 쿼리 생략
SET pgaudit.log_relation = on;          -- 객체 이름 로그
SET pgaudit.log_statement_once = on;    -- 명령문을 한 번만 로그

-- 역할 기반 감사
CREATE ROLE auditor NOLOGIN;
SET pgaudit.role = 'auditor';

-- 특정 테이블에 대한 감사 권한 부여
GRANT SELECT, INSERT, UPDATE, DELETE ON employees TO auditor;
-- 이제 employees에 대한 모든 DML이 감사됨
```

### 6.3 사용자 정의 감사 테이블

```sql
-- 간단한 감사 추적
CREATE TABLE audit_log (
    id BIGSERIAL PRIMARY KEY,
    table_name TEXT NOT NULL,
    operation TEXT NOT NULL,  -- INSERT, UPDATE, DELETE
    old_data JSONB,
    new_data JSONB,
    changed_by TEXT DEFAULT current_user,
    changed_at TIMESTAMP DEFAULT NOW()
);

-- 범용 감사 트리거
CREATE OR REPLACE FUNCTION audit_trigger_func()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO audit_log (table_name, operation, new_data)
        VALUES (TG_TABLE_NAME, 'INSERT', to_jsonb(NEW));
        RETURN NEW;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit_log (table_name, operation, old_data, new_data)
        VALUES (TG_TABLE_NAME, 'UPDATE', to_jsonb(OLD), to_jsonb(NEW));
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO audit_log (table_name, operation, old_data)
        VALUES (TG_TABLE_NAME, 'DELETE', to_jsonb(OLD));
        RETURN OLD;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- 테이블에 적용
CREATE TRIGGER employees_audit
    AFTER INSERT OR UPDATE OR DELETE ON employees
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_func();
```

---

## 7. 보안 모범 사례

### 7.1 최소 권한 원칙

```sql
-- 1. 기본 공개 접근 취소
REVOKE ALL ON SCHEMA public FROM PUBLIC;
REVOKE CREATE ON SCHEMA public FROM PUBLIC;

-- 2. 애플리케이션별 스키마 생성
CREATE SCHEMA app_schema;
GRANT USAGE ON SCHEMA app_schema TO app_role;

-- 3. 기능별로 역할 분리
CREATE ROLE migration_role LOGIN PASSWORD '...';  -- DDL만
CREATE ROLE app_role LOGIN PASSWORD '...';         -- DML만
CREATE ROLE report_role LOGIN PASSWORD '...';      -- SELECT만

GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA app_schema TO app_role;
GRANT SELECT ON ALL TABLES IN SCHEMA app_schema TO report_role;
```

### 7.2 SQL 인젝션 방지

```sql
-- 문자열 연결로 쿼리를 만들면 안 됨
-- 나쁨:
-- EXECUTE 'SELECT * FROM users WHERE name = ''' || user_input || '''';

-- 좋음: 매개변수화된 쿼리 사용
CREATE OR REPLACE FUNCTION safe_search(search_name TEXT)
RETURNS SETOF employees AS $$
BEGIN
    RETURN QUERY
    SELECT * FROM employees WHERE name = search_name;
END;
$$ LANGUAGE plpgsql;

-- 좋음: 리터럴에 %L, 식별자에 %I를 사용한 format() 사용
CREATE OR REPLACE FUNCTION dynamic_query(table_name TEXT, col TEXT, val TEXT)
RETURNS SETOF RECORD AS $$
BEGIN
    RETURN QUERY EXECUTE format(
        'SELECT * FROM %I WHERE %I = %L',
        table_name, col, val
    );
END;
$$ LANGUAGE plpgsql;
```

### 7.3 데이터 암호화

```sql
-- pgcrypto 활성화
CREATE EXTENSION pgcrypto;

-- 민감한 데이터 암호화
INSERT INTO users (name, ssn_encrypted)
VALUES ('Alice', pgp_sym_encrypt('123-45-6789', 'encryption_key'));

-- 복호화
SELECT name, pgp_sym_decrypt(ssn_encrypted, 'encryption_key') AS ssn
FROM users;

-- 비밀번호 해시 (bcrypt 사용)
INSERT INTO users (name, password_hash)
VALUES ('Alice', crypt('user_password', gen_salt('bf', 10)));

-- 비밀번호 확인
SELECT name FROM users
WHERE password_hash = crypt('user_password', password_hash);
```

### 7.4 연결 보안 체크리스트

```
┌─────────────────────────────────────────────────────────────────┐
│              보안 체크리스트                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  네트워크:                                                      │
│  ☐ listen_addresses 제한 ('*' 아님)                             │
│  ☐ 방화벽이 포트 5432 제한                                      │
│  ☐ SSL/TLS 활성화 및 필수화                                     │
│                                                                 │
│  인증:                                                          │
│  ☐ 모든 비밀번호 인증에 scram-sha-256                           │
│  ☐ trust 또는 password 방법 없음                                │
│  ☐ pg_hba.conf가 특정 주소 사용 (0.0.0.0/0 아님)                │
│                                                                 │
│  인가:                                                          │
│  ☐ PUBLIC 스키마 권한 취소됨                                    │
│  ☐ 최소 권한 역할                                               │
│  ☐ 멀티 테넌트 데이터에 RLS 활성화                              │
│                                                                 │
│  모니터링:                                                      │
│  ☐ 감사 로깅 활성화                                             │
│  ☐ 실패한 로그인 모니터링                                       │
│  ☐ 의심스러운 쿼리 탐지                                         │
│                                                                 │
│  데이터:                                                        │
│  ☐ 민감한 데이터 암호화 (pgcrypto)                              │
│  ☐ bcrypt로 비밀번호 해시                                       │
│  ☐ 정기적인 백업 암호화                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. 연습 문제

### 연습 1: 멀티 테넌트 애플리케이션
테넌트 격리가 있는 SaaS 애플리케이션을 위한 RLS를 설정하세요.

```sql
-- 예시 답안
CREATE TABLE tenant_orders (
    id SERIAL PRIMARY KEY,
    tenant_id INT NOT NULL,
    product TEXT NOT NULL,
    amount NUMERIC(10,2),
    created_at TIMESTAMP DEFAULT NOW()
);

ALTER TABLE tenant_orders ENABLE ROW LEVEL SECURITY;
ALTER TABLE tenant_orders FORCE ROW LEVEL SECURITY;

-- 테넌트 격리 정책
CREATE POLICY tenant_orders_isolation ON tenant_orders
    FOR ALL
    USING (tenant_id = current_setting('app.tenant_id')::INT)
    WITH CHECK (tenant_id = current_setting('app.tenant_id')::INT);

-- 관리자 우회
CREATE POLICY tenant_orders_admin ON tenant_orders
    FOR ALL
    TO admin_group
    USING (TRUE);

-- 테스트
SET app.tenant_id = '1';
INSERT INTO tenant_orders (tenant_id, product, amount) VALUES (1, 'Widget', 29.99);
SELECT * FROM tenant_orders;  -- 테넌트 1의 주문만
```

### 연습 2: 감사 시스템
employees 테이블을 위한 포괄적인 감사 시스템을 생성하세요.

```sql
-- 예시 답안
CREATE TABLE employee_audit (
    id BIGSERIAL PRIMARY KEY,
    operation TEXT NOT NULL,
    employee_id INT,
    old_values JSONB,
    new_values JSONB,
    changed_fields TEXT[],
    user_name TEXT DEFAULT current_user,
    client_ip INET DEFAULT inet_client_addr(),
    occurred_at TIMESTAMP DEFAULT NOW()
);

CREATE OR REPLACE FUNCTION employee_audit_trigger()
RETURNS TRIGGER AS $$
DECLARE
    changed TEXT[] := '{}';
    col TEXT;
BEGIN
    IF TG_OP = 'UPDATE' THEN
        FOR col IN SELECT column_name FROM information_schema.columns
                   WHERE table_name = 'employees' LOOP
            EXECUTE format('SELECT ($1).%I IS DISTINCT FROM ($2).%I', col, col)
            INTO STRICT changed USING NEW, OLD;
        END LOOP;
    END IF;

    INSERT INTO employee_audit (operation, employee_id, old_values, new_values)
    VALUES (
        TG_OP,
        COALESCE(NEW.id, OLD.id),
        CASE WHEN TG_OP != 'INSERT' THEN to_jsonb(OLD) END,
        CASE WHEN TG_OP != 'DELETE' THEN to_jsonb(NEW) END
    );

    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_employee_audit
    AFTER INSERT OR UPDATE OR DELETE ON employees
    FOR EACH ROW EXECUTE FUNCTION employee_audit_trigger();
```

### 연습 3: 역할 계층 구조
웹 애플리케이션을 위한 역할 계층 구조를 생성하세요.

```sql
-- 예시 답안
-- 기본 역할
CREATE ROLE web_anonymous NOLOGIN;
CREATE ROLE web_user NOLOGIN;
CREATE ROLE web_admin NOLOGIN;

-- 상속 계층 구조
GRANT web_anonymous TO web_user;
GRANT web_user TO web_admin;

-- 권한 (상속을 통해 누적)
GRANT USAGE ON SCHEMA public TO web_anonymous;
GRANT SELECT ON public.products, public.categories TO web_anonymous;

GRANT SELECT, INSERT, UPDATE ON public.orders TO web_user;
GRANT SELECT, INSERT ON public.reviews TO web_user;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO web_user;

GRANT ALL ON ALL TABLES IN SCHEMA public TO web_admin;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO web_admin;

-- 로그인 역할
CREATE ROLE api_anon LOGIN PASSWORD '...' IN ROLE web_anonymous;
CREATE ROLE api_user LOGIN PASSWORD '...' IN ROLE web_user;
CREATE ROLE api_admin LOGIN PASSWORD '...' IN ROLE web_admin;
```

---

**이전**: [전문 검색](./19_Full_Text_Search.md)

## 참고 자료
- [PostgreSQL Roles](https://www.postgresql.org/docs/current/user-manag.html)
- [Row-Level Security](https://www.postgresql.org/docs/current/ddl-rowsecurity.html)
- [pg_hba.conf](https://www.postgresql.org/docs/current/auth-pg-hba-conf.html)
- [SSL Support](https://www.postgresql.org/docs/current/ssl-tcp.html)
- [pgAudit](https://www.pgaudit.org/)
