# 20. Security and Access Control

**Previous**: [Full-Text Search](./19_Full_Text_Search.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Design a role hierarchy with group roles, login roles, and the principle of least privilege
2. Grant and revoke object-level and column-level privileges on schemas, tables, sequences, and functions
3. Implement Row-Level Security (RLS) policies for multi-tenant isolation and department-based access control
4. Configure pg_hba.conf authentication rules with appropriate methods (scram-sha-256, peer, cert)
5. Set up SSL/TLS encrypted connections including client certificate authentication
6. Enable audit logging using both built-in PostgreSQL logging and the pgAudit extension
7. Apply security best practices including SQL injection prevention, data encryption with pgcrypto, and connection security hardening

---

Security is not a feature you bolt on at the end -- it must be designed into your database from the start. A misconfigured PostgreSQL instance can expose sensitive data, allow privilege escalation, or become an entry point for attackers. This lesson walks through PostgreSQL's defense-in-depth security model: from network-level controls and authentication, through fine-grained authorization and row-level security, to encryption and audit logging. Whether you are building a multi-tenant SaaS application or hardening an existing deployment, these techniques form the foundation of a secure PostgreSQL environment.

## Table of Contents
1. [Security Overview](#1-security-overview)
2. [Roles and Privileges](#2-roles-and-privileges)
3. [Row-Level Security (RLS)](#3-row-level-security-rls)
4. [Authentication (pg_hba.conf)](#4-authentication-pg_hbaconf)
5. [SSL/TLS Connections](#5-ssltls-connections)
6. [Audit Logging](#6-audit-logging)
7. [Security Best Practices](#7-security-best-practices)
8. [Practice Problems](#8-practice-problems)

---

## 1. Security Overview

### 1.1 PostgreSQL Security Layers

```
┌─────────────────────────────────────────────────────────────────┐
│              PostgreSQL Security Architecture                    │
│                                                                 │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  Layer 1: Network Security                           │       │
│  │  ┌──────────────────────────────────────────────┐    │       │
│  │  │  Firewall, listen_addresses, port             │    │       │
│  │  └──────────────────────────────────────────────┘    │       │
│  │                                                      │       │
│  │  Layer 2: Authentication (pg_hba.conf)               │       │
│  │  ┌──────────────────────────────────────────────┐    │       │
│  │  │  Who can connect? From where? How?            │    │       │
│  │  └──────────────────────────────────────────────┘    │       │
│  │                                                      │       │
│  │  Layer 3: Authorization (GRANT/REVOKE)               │       │
│  │  ┌──────────────────────────────────────────────┐    │       │
│  │  │  Object-level privileges                      │    │       │
│  │  └──────────────────────────────────────────────┘    │       │
│  │                                                      │       │
│  │  Layer 4: Row-Level Security (RLS)                   │       │
│  │  ┌──────────────────────────────────────────────┐    │       │
│  │  │  Row-level access control policies            │    │       │
│  │  └──────────────────────────────────────────────┘    │       │
│  │                                                      │       │
│  │  Layer 5: Column-Level Security                      │       │
│  │  ┌──────────────────────────────────────────────┐    │       │
│  │  │  Column-specific GRANT                        │    │       │
│  │  └──────────────────────────────────────────────┘    │       │
│  │                                                      │       │
│  │  Layer 6: Encryption                                 │       │
│  │  ┌──────────────────────────────────────────────┐    │       │
│  │  │  SSL/TLS, pgcrypto, data-at-rest              │    │       │
│  │  └──────────────────────────────────────────────┘    │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Roles and Privileges

### 2.1 Role Management

```sql
-- Create roles
CREATE ROLE app_reader LOGIN PASSWORD 'secure_password';
CREATE ROLE app_writer LOGIN PASSWORD 'secure_password';
CREATE ROLE app_admin LOGIN PASSWORD 'secure_password' CREATEROLE;

-- Group roles (no LOGIN)
CREATE ROLE readonly_group NOLOGIN;
CREATE ROLE readwrite_group NOLOGIN;
CREATE ROLE admin_group NOLOGIN;

-- Role membership
GRANT readonly_group TO app_reader;
GRANT readwrite_group TO app_writer;
GRANT admin_group TO app_admin;

-- Role attributes
ALTER ROLE app_reader SET statement_timeout = '30s';
ALTER ROLE app_writer CONNECTION LIMIT 10;
ALTER ROLE app_admin VALID UNTIL '2026-12-31';
```

### 2.2 Object Privileges

```sql
-- Schema privileges
GRANT USAGE ON SCHEMA public TO readonly_group;
GRANT CREATE ON SCHEMA public TO admin_group;

-- Table privileges
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly_group;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO readwrite_group;

-- Default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT ON TABLES TO readonly_group;

ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO readwrite_group;

-- Sequence privileges
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO readwrite_group;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT USAGE ON SEQUENCES TO readwrite_group;

-- Function privileges
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO readwrite_group;
```

### 2.3 Column-Level Privileges

```sql
-- Grant access to specific columns only
CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    salary NUMERIC(10,2),
    ssn TEXT
);

-- HR can see everything
GRANT SELECT ON employees TO hr_role;

-- Managers can see non-sensitive columns
GRANT SELECT (id, name, email) ON employees TO manager_role;

-- Revoke access
REVOKE SELECT (salary, ssn) ON employees FROM manager_role;
```

### 2.4 Inspecting Privileges

```sql
-- Check table privileges
SELECT grantee, privilege_type
FROM information_schema.role_table_grants
WHERE table_name = 'employees';

-- Check column privileges
SELECT grantee, column_name, privilege_type
FROM information_schema.column_privileges
WHERE table_name = 'employees';

-- List role memberships
SELECT r.rolname AS role, m.rolname AS member
FROM pg_auth_members am
JOIN pg_roles r ON r.oid = am.roleid
JOIN pg_roles m ON m.oid = am.member;

-- Check current user's privileges
SELECT has_table_privilege('app_reader', 'employees', 'SELECT');
SELECT has_schema_privilege('app_reader', 'public', 'USAGE');
```

---

## 3. Row-Level Security (RLS)

### 3.1 RLS Basics

```sql
-- Enable RLS on a table
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

-- Policy: users can only see their own documents or public ones
CREATE POLICY documents_select ON documents
    FOR SELECT
    USING (owner_name = current_user OR is_public = TRUE);

-- Policy: users can only modify their own documents
CREATE POLICY documents_modify ON documents
    FOR ALL
    USING (owner_name = current_user)
    WITH CHECK (owner_name = current_user);
```

### 3.2 Policy Types

```sql
-- USING: filters rows for SELECT, UPDATE (existing rows), DELETE
-- WITH CHECK: validates rows for INSERT, UPDATE (new rows)

-- Insert policy
CREATE POLICY documents_insert ON documents
    FOR INSERT
    WITH CHECK (owner_name = current_user);

-- Update policy (separate USING and WITH CHECK)
CREATE POLICY documents_update ON documents
    FOR UPDATE
    USING (owner_name = current_user)           -- can only see own rows
    WITH CHECK (owner_name = current_user);     -- can only set self as owner

-- Delete policy
CREATE POLICY documents_delete ON documents
    FOR DELETE
    USING (owner_name = current_user);
```

### 3.3 Multi-Tenant RLS

```sql
-- Tenant isolation using application-set variable
CREATE TABLE tenant_data (
    id SERIAL PRIMARY KEY,
    tenant_id INT NOT NULL,
    data JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

ALTER TABLE tenant_data ENABLE ROW LEVEL SECURITY;

-- Policy using session variable
CREATE POLICY tenant_isolation ON tenant_data
    FOR ALL
    USING (tenant_id = current_setting('app.tenant_id')::INT)
    WITH CHECK (tenant_id = current_setting('app.tenant_id')::INT);

-- Application sets tenant context
SET app.tenant_id = '42';
SELECT * FROM tenant_data;  -- only sees tenant 42's data

-- Bypass RLS for admin role
ALTER TABLE tenant_data FORCE ROW LEVEL SECURITY;  -- apply to table owner too
CREATE POLICY admin_bypass ON tenant_data
    FOR ALL
    TO admin_group
    USING (TRUE)
    WITH CHECK (TRUE);
```

### 3.4 Department-Based Access

```sql
-- Users see documents from their department + public ones
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

## 4. Authentication (pg_hba.conf)

### 4.1 pg_hba.conf Structure

```
# TYPE  DATABASE  USER       ADDRESS        METHOD

# Local connections
local   all       postgres                  peer
local   all       all                       scram-sha-256

# IPv4 connections
host    all       all        127.0.0.1/32   scram-sha-256
host    mydb      app_user   10.0.0.0/8     scram-sha-256
host    all       all        0.0.0.0/0      reject

# IPv6 connections
host    all       all        ::1/128        scram-sha-256

# SSL required
hostssl mydb      app_user   10.0.0.0/8     scram-sha-256
hostnossl all     all        0.0.0.0/0      reject
```

### 4.2 Authentication Methods

```
┌────────────────┬──────────────────────────────────────────────────┐
│ Method         │ Description                                      │
├────────────────┼──────────────────────────────────────────────────┤
│ trust          │ No authentication (NEVER in production)           │
│ reject         │ Always reject connection                          │
│ scram-sha-256  │ Challenge-response (recommended)                  │
│ md5            │ MD5 hash (legacy, use scram-sha-256 instead)     │
│ password       │ Cleartext (NEVER use)                             │
│ peer           │ OS user = PG user (local only)                   │
│ ident          │ OS user mapping (TCP/IP)                          │
│ cert           │ SSL client certificate                            │
│ ldap           │ LDAP server authentication                        │
│ gss            │ Kerberos/GSSAPI                                   │
└────────────────┴──────────────────────────────────────────────────┘
```

### 4.3 Password Management

```sql
-- Use scram-sha-256 (default in PG 14+)
SET password_encryption = 'scram-sha-256';

-- Create user with encrypted password
CREATE ROLE app_user LOGIN PASSWORD 'strong_password_here';

-- Force password change
ALTER ROLE app_user VALID UNTIL '2026-03-01';

-- Check password encryption method
SELECT rolname, rolpassword ~ '^SCRAM-SHA-256' AS is_scram
FROM pg_authid
WHERE rolcanlogin;
```

### 4.4 Reload Configuration

```bash
# After editing pg_hba.conf, reload:
pg_ctl reload -D /path/to/data

# Or from SQL:
# SELECT pg_reload_conf();

# Verify current settings
# SELECT * FROM pg_hba_file_rules;  -- PG 15+
```

---

## 5. SSL/TLS Connections

### 5.1 Enable SSL

```bash
# Generate self-signed certificate
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

### 5.2 Client Certificate Authentication

```bash
# Generate CA
openssl req -new -x509 -days 3650 -nodes \
    -out root.crt -keyout root.key \
    -subj "/CN=MyCA"

# Generate client certificate
openssl req -new -nodes \
    -out client.csr -keyout client.key \
    -subj "/CN=app_user"

openssl x509 -req -in client.csr -days 365 \
    -CA root.crt -CAkey root.key -CAcreateserial \
    -out client.crt
```

```
# pg_hba.conf — require client certificate
hostssl mydb  app_user  10.0.0.0/8  cert  clientcert=verify-full
```

### 5.3 Verify SSL Connection

```sql
-- Check if current connection uses SSL
SELECT ssl, version, cipher
FROM pg_stat_ssl
WHERE pid = pg_backend_pid();

-- Check all SSL connections
SELECT s.pid, s.ssl, s.version, s.cipher, a.usename, a.client_addr
FROM pg_stat_ssl s
JOIN pg_stat_activity a ON s.pid = a.pid
WHERE s.ssl = TRUE;
```

---

## 6. Audit Logging

### 6.1 Basic Logging (postgresql.conf)

```
# postgresql.conf
log_statement = 'all'           # none, ddl, mod, all
log_min_duration_statement = 0  # log all statements with duration
log_connections = on
log_disconnections = on
log_line_prefix = '%t [%p] %u@%d '  # timestamp, pid, user, database
```

### 6.2 pgAudit Extension

```sql
-- Install pgAudit (must be in shared_preload_libraries)
-- postgresql.conf: shared_preload_libraries = 'pgaudit'

CREATE EXTENSION pgaudit;

-- Configure audit logging
SET pgaudit.log = 'write, ddl';        -- log writes and DDL
SET pgaudit.log_catalog = off;          -- skip system catalog queries
SET pgaudit.log_relation = on;          -- log object names
SET pgaudit.log_statement_once = on;    -- log statement only once

-- Role-based auditing
CREATE ROLE auditor NOLOGIN;
SET pgaudit.role = 'auditor';

-- Grant audit on specific tables
GRANT SELECT, INSERT, UPDATE, DELETE ON employees TO auditor;
-- Now all DML on employees is audited
```

### 6.3 Custom Audit Table

```sql
-- Simple audit trail
CREATE TABLE audit_log (
    id BIGSERIAL PRIMARY KEY,
    table_name TEXT NOT NULL,
    operation TEXT NOT NULL,  -- INSERT, UPDATE, DELETE
    old_data JSONB,
    new_data JSONB,
    changed_by TEXT DEFAULT current_user,
    changed_at TIMESTAMP DEFAULT NOW()
);

-- Generic audit trigger
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

-- Apply to tables
CREATE TRIGGER employees_audit
    AFTER INSERT OR UPDATE OR DELETE ON employees
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_func();
```

---

## 7. Security Best Practices

### 7.1 Principle of Least Privilege

```sql
-- 1. Revoke default public access
REVOKE ALL ON SCHEMA public FROM PUBLIC;
REVOKE CREATE ON SCHEMA public FROM PUBLIC;

-- 2. Create application-specific schemas
CREATE SCHEMA app_schema;
GRANT USAGE ON SCHEMA app_schema TO app_role;

-- 3. Separate roles by function
CREATE ROLE migration_role LOGIN PASSWORD '...';  -- DDL only
CREATE ROLE app_role LOGIN PASSWORD '...';         -- DML only
CREATE ROLE report_role LOGIN PASSWORD '...';      -- SELECT only

GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA app_schema TO app_role;
GRANT SELECT ON ALL TABLES IN SCHEMA app_schema TO report_role;
```

### 7.2 SQL Injection Prevention

```sql
-- NEVER build queries with string concatenation
-- BAD:
-- EXECUTE 'SELECT * FROM users WHERE name = ''' || user_input || '''';

-- GOOD: Use parameterized queries
CREATE OR REPLACE FUNCTION safe_search(search_name TEXT)
RETURNS SETOF employees AS $$
BEGIN
    RETURN QUERY
    SELECT * FROM employees WHERE name = search_name;
END;
$$ LANGUAGE plpgsql;

-- GOOD: Use format() with %L for literals, %I for identifiers
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

### 7.3 Data Encryption

```sql
-- Enable pgcrypto
CREATE EXTENSION pgcrypto;

-- Encrypt sensitive data
INSERT INTO users (name, ssn_encrypted)
VALUES ('Alice', pgp_sym_encrypt('123-45-6789', 'encryption_key'));

-- Decrypt
SELECT name, pgp_sym_decrypt(ssn_encrypted, 'encryption_key') AS ssn
FROM users;

-- Hash passwords (use bcrypt)
INSERT INTO users (name, password_hash)
VALUES ('Alice', crypt('user_password', gen_salt('bf', 10)));

-- Verify password
SELECT name FROM users
WHERE password_hash = crypt('user_password', password_hash);
```

### 7.4 Connection Security Checklist

```
┌─────────────────────────────────────────────────────────────────┐
│              Security Checklist                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Network:                                                       │
│  ☐ listen_addresses limited (not '*')                           │
│  ☐ Firewall restricts port 5432                                 │
│  ☐ SSL/TLS enabled and required                                 │
│                                                                 │
│  Authentication:                                                │
│  ☐ scram-sha-256 for all password auth                          │
│  ☐ No trust or password method                                  │
│  ☐ pg_hba.conf uses specific addresses (not 0.0.0.0/0)         │
│                                                                 │
│  Authorization:                                                 │
│  ☐ PUBLIC schema privileges revoked                             │
│  ☐ Least-privilege roles                                        │
│  ☐ RLS enabled for multi-tenant data                            │
│                                                                 │
│  Monitoring:                                                    │
│  ☐ Audit logging enabled                                        │
│  ☐ Failed login monitoring                                      │
│  ☐ Suspicious query detection                                   │
│                                                                 │
│  Data:                                                          │
│  ☐ Sensitive data encrypted (pgcrypto)                          │
│  ☐ Passwords hashed with bcrypt                                 │
│  ☐ Regular backup encryption                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Practice Problems

### Exercise 1: Multi-Tenant Application
Set up RLS for a SaaS application with tenant isolation.

```sql
-- Example answer
CREATE TABLE tenant_orders (
    id SERIAL PRIMARY KEY,
    tenant_id INT NOT NULL,
    product TEXT NOT NULL,
    amount NUMERIC(10,2),
    created_at TIMESTAMP DEFAULT NOW()
);

ALTER TABLE tenant_orders ENABLE ROW LEVEL SECURITY;
ALTER TABLE tenant_orders FORCE ROW LEVEL SECURITY;

-- Tenant isolation policy
CREATE POLICY tenant_orders_isolation ON tenant_orders
    FOR ALL
    USING (tenant_id = current_setting('app.tenant_id')::INT)
    WITH CHECK (tenant_id = current_setting('app.tenant_id')::INT);

-- Admin bypass
CREATE POLICY tenant_orders_admin ON tenant_orders
    FOR ALL
    TO admin_group
    USING (TRUE);

-- Test
SET app.tenant_id = '1';
INSERT INTO tenant_orders (tenant_id, product, amount) VALUES (1, 'Widget', 29.99);
SELECT * FROM tenant_orders;  -- only tenant 1's orders
```

### Exercise 2: Audit System
Create a comprehensive audit system for the employees table.

```sql
-- Example answer
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

### Exercise 3: Role Hierarchy
Create a role hierarchy for a web application.

```sql
-- Example answer
-- Base roles
CREATE ROLE web_anonymous NOLOGIN;
CREATE ROLE web_user NOLOGIN;
CREATE ROLE web_admin NOLOGIN;

-- Inheritance hierarchy
GRANT web_anonymous TO web_user;
GRANT web_user TO web_admin;

-- Privileges (cumulative through inheritance)
GRANT USAGE ON SCHEMA public TO web_anonymous;
GRANT SELECT ON public.products, public.categories TO web_anonymous;

GRANT SELECT, INSERT, UPDATE ON public.orders TO web_user;
GRANT SELECT, INSERT ON public.reviews TO web_user;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO web_user;

GRANT ALL ON ALL TABLES IN SCHEMA public TO web_admin;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO web_admin;

-- Login roles
CREATE ROLE api_anon LOGIN PASSWORD '...' IN ROLE web_anonymous;
CREATE ROLE api_user LOGIN PASSWORD '...' IN ROLE web_user;
CREATE ROLE api_admin LOGIN PASSWORD '...' IN ROLE web_admin;
```

---

## References
- [PostgreSQL Roles](https://www.postgresql.org/docs/current/user-manag.html)
- [Row-Level Security](https://www.postgresql.org/docs/current/ddl-rowsecurity.html)
- [pg_hba.conf](https://www.postgresql.org/docs/current/auth-pg-hba-conf.html)
- [SSL Support](https://www.postgresql.org/docs/current/ssl-tcp.html)
- [pgAudit](https://www.pgaudit.org/)

---

**Previous**: [Full-Text Search](./19_Full_Text_Search.md)
