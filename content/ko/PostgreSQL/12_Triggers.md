# 트리거

**이전**: [트랜잭션](./11_Transactions.md) | **다음**: [백업과 운영](./13_Backup_and_Operations.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 트리거(Trigger)의 목적과 데이터 변경 이벤트에 대한 자동 응답 방식을 설명할 수 있습니다
2. PL/pgSQL에서 TRIGGER 타입을 반환하는 트리거 함수(Trigger Function)를 생성할 수 있습니다
3. BEFORE 트리거와 AFTER 트리거를 구분하고 주어진 사용 사례에 적합한 실행 시점을 선택할 수 있습니다
4. 트리거 함수 내에서 NEW와 OLD 레코드 변수를 사용하여 행 데이터에 접근할 수 있습니다
5. FOR EACH ROW와 FOR EACH STATEMENT 트리거의 실행 단위(Granularity)를 비교할 수 있습니다
6. WHEN 절을 사용하여 조건부 트리거(Conditional Trigger)를 구현할 수 있습니다
7. 감사 로그(Audit Log), 타임스탬프 자동 갱신, 재고 관리 등 실용적인 트리거 기반 솔루션을 구축할 수 있습니다
8. 트리거를 목록 조회, 활성화, 비활성화, 삭제하는 방식으로 관리할 수 있습니다

---

트리거(Trigger)를 사용하면 비즈니스 규칙을 데이터베이스 계층에 직접 내장할 수 있어, 감사 추적 유지, 데이터 유효성 검사, 파생 컬럼 갱신 같은 중요한 로직이 데이터 변경 시 자동으로 실행됩니다. 모든 애플리케이션이 올바른 함수를 호출하는 것을 기억하는 대신, 데이터베이스 자체가 일관성을 강제합니다. 이는 트리거를 모든 운영 PostgreSQL 시스템에서 데이터 무결성(Data Integrity)을 위한 필수 도구로 만듭니다.

---

## 1. 트리거 개념

트리거는 특정 이벤트(INSERT, UPDATE, DELETE)가 발생할 때 자동으로 실행되는 함수입니다.

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   INSERT    │ ──▶ │   TRIGGER   │ ──▶ │  자동 실행  │
│   UPDATE    │     │   (감시)    │     │  (트리거    │
│   DELETE    │     │             │     │   함수)     │
└─────────────┘     └─────────────┘     └─────────────┘
```

---

## 2. 트리거 구성요소

1. **트리거 함수**: 실행할 로직
2. **트리거**: 언제, 어떤 테이블에서 함수를 실행할지 정의

### 트리거 함수 생성

```sql
CREATE FUNCTION trigger_function_name()
RETURNS TRIGGER
AS $$
BEGIN
    -- 로직
    RETURN NEW;  -- 또는 RETURN OLD; 또는 RETURN NULL;
END;
$$ LANGUAGE plpgsql;
```

### 트리거 생성

```sql
CREATE TRIGGER trigger_name
{BEFORE | AFTER | INSTEAD OF} {INSERT | UPDATE | DELETE}
ON table_name
[FOR EACH ROW | FOR EACH STATEMENT]
EXECUTE FUNCTION trigger_function_name();
```

---

## 3. BEFORE vs AFTER

### 이론: 트리거 매트릭스

두 timing × 두 level × 네 event = 잠재적으로 16 조합, 그러나 TRUNCATE는 statement-level에서만 발사되므로 14 valid cell. 가장 흔한 12개:

|       | INSERT | UPDATE | DELETE |
|-------|--------|--------|--------|
| **BEFORE ROW** | NEW 존재, NEW 수정 가능, 행을 건너뛰려면 NULL 반환 | NEW + OLD, NEW 수정 가능 | OLD 존재, delete를 abort하려면 NULL 반환 |
| **AFTER ROW** | NEW 동결 | NEW + OLD 동결 | OLD 동결 |
| **BEFORE STATEMENT** | 모든 행 전에 1번 실행 | 1번 실행 | 1번 실행 |
| **AFTER STATEMENT** | 모든 행 후에 1번 실행 | 1번 실행 | 1번 실행 |

#### A.1 BEFORE ROW — "intercept" 트리거

*각 행에 대해, 엔진이 변경을 적용하기 전에* 발사. 트리거 함수는 곧 쓰일 행(`NEW`)을, UPDATE의 경우 변경 전 행(`OLD`)도 받음. 함수는:

- **`NEW`를 in-place로 수정**하고 `RETURN NEW;` 가능 — 수정된 행이 쓰임.
- `RETURN NULL;` 가능 — 엔진이 **이 행을 통째로 건너뜀**(INSERT/UPDATE 발생 안 함).
- `RAISE EXCEPTION` 가능 — 전체 statement abort(잡히지 않으면 트랜잭션 abort).

BEFORE ROW는 입력 검증, 파생 컬럼 채우기(`NEW.normalized_email := lower(NEW.email)`), 조건부 행 억제에 사용.

#### A.2 AFTER ROW — "react" 트리거

*각 행에 대해, 엔진이 변경을 적용한 후* 발사. `NEW`와 `OLD`는 read-only — 행은 이미 디스크로 가는 중. 트리거 반환값 무시됨.

AFTER ROW는 audit 로깅(`change_log` 테이블에 INSERT), 캐시 무효화, NOTIFY 전송, 또는 변경이 실제로 commit된 쓰기에 성공했을 때만 일어나야 하는 부수 효과에 사용.

#### A.3 BEFORE/AFTER STATEMENT — statement당 1번

statement가 영향준 행 수와 무관하게 1번 발사 — statement가 0행에 영향을 줘도. `NEW`와 `OLD`는 NULL — 특정 행이 없기 때문. PostgreSQL 10+는 영향받은 행 집합을 **transition table**로 노출:

```sql
CREATE TRIGGER ... AFTER UPDATE ON orders
REFERENCING OLD TABLE AS old_rows NEW TABLE AS new_rows
FOR EACH STATEMENT
EXECUTE FUNCTION audit_changes();
```

트리거 함수 안에서 `old_rows`와 `new_rows`는 영향받은 모든 행을 담은 임시 테이블처럼 query 가능. "행마다 1개가 아니라 1개의 로그 항목으로 batch UPDATE를 audit"에 유용.

### BEFORE 트리거

이벤트 발생 **전**에 실행됩니다. 데이터를 검증하거나 수정할 수 있습니다.

```sql
-- 가격이 0 이하면 오류 발생
CREATE FUNCTION check_price()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.price <= 0 THEN
        RAISE EXCEPTION '가격은 0보다 커야 합니다.';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER before_insert_product
BEFORE INSERT ON products
FOR EACH ROW
EXECUTE FUNCTION check_price();
```

### AFTER 트리거

이벤트 발생 **후**에 실행됩니다. 감사 로그, 알림 등에 사용합니다.

```sql
-- 주문 생성 후 재고 차감
CREATE FUNCTION update_stock()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE products
    SET stock = stock - NEW.quantity
    WHERE id = NEW.product_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER after_insert_order_item
AFTER INSERT ON order_items
FOR EACH ROW
EXECUTE FUNCTION update_stock();
```

---

## 4. NEW vs OLD

| 변수 | INSERT | UPDATE | DELETE |
|------|--------|--------|--------|
| `NEW` | 새 행 | 새 행 | 없음 |
| `OLD` | 없음 | 기존 행 | 삭제될 행 |

```sql
-- UPDATE 시 변경 전후 값 비교
CREATE FUNCTION log_price_change()
RETURNS TRIGGER AS $$
BEGIN
    IF OLD.price <> NEW.price THEN
        INSERT INTO price_history (product_id, old_price, new_price)
        VALUES (NEW.id, OLD.price, NEW.price);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER after_update_price
AFTER UPDATE OF price ON products
FOR EACH ROW
EXECUTE FUNCTION log_price_change();
```

---

### 이론: `NEW`와 `OLD`가 무엇인가

둘 다 트리거가 붙은 테이블의 행과 같은 타입의 *record*. 모든 컬럼 + 시스템 컬럼(`tableoid`, `xmin`, `xmax`, `ctid`)을 가짐.

| Event | `NEW` | `OLD` |
|-------|-------|-------|
| INSERT | 곧 삽입될 행(BEFORE) 또는 방금 삽입된 행(AFTER) | undefined / NULL |
| UPDATE | 변경 후 행 | 변경 전 행 |
| DELETE | undefined / NULL | 곧/방금 삭제될 행 |

#### B.1 BEFORE INSERT 예 — `NEW`는 mutable

```sql
CREATE FUNCTION normalize_email() RETURNS trigger AS $$
BEGIN
    NEW.email := lower(trim(NEW.email));
    IF NEW.email = '' THEN
        RETURN NULL;  -- email이 비어 있는 행을 조용히 건너뜀
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_normalize BEFORE INSERT ON users
FOR EACH ROW EXECUTE FUNCTION normalize_email();
```

결국 디스크에 닿는 행은 정규화된 email을 가짐. 애플리케이션 코드가 알 필요 없음.

#### B.2 "반드시 반환" 규칙

BEFORE ROW 트리거는 `RETURN NEW`, `RETURN OLD`(DELETE의 경우), 또는 `RETURN NULL` 해야 함. 반환을 잊는 것이 가장 흔한 PL/pgSQL 버그 중 하나 — 엔진이 반환을 NULL로 읽고 행을 조용히 drop.

AFTER ROW 트리거는 `RETURN NULL` 또는 `RETURN NEW` 가능 — 값 무시됨. 관례적 스타일은 INSERT/UPDATE에 `RETURN NEW;`, DELETE에 `RETURN OLD;`.

## 5. FOR EACH ROW vs FOR EACH STATEMENT

### FOR EACH ROW

각 행마다 트리거가 실행됩니다.

```sql
-- 각 행에 대해 실행
CREATE TRIGGER row_trigger
AFTER INSERT ON products
FOR EACH ROW
EXECUTE FUNCTION my_function();

-- INSERT INTO products VALUES (...), (...), (...);
-- → 3번 실행
```

### FOR EACH STATEMENT

문장당 한 번만 실행됩니다.

```sql
-- 문장당 한 번만 실행
CREATE TRIGGER statement_trigger
AFTER INSERT ON products
FOR EACH STATEMENT
EXECUTE FUNCTION my_function();

-- INSERT INTO products VALUES (...), (...), (...);
-- → 1번 실행
```

---

## 6. 조건부 트리거 (WHEN)

```sql
-- 가격이 100만원 이상일 때만 실행
CREATE TRIGGER high_price_alert
AFTER INSERT ON products
FOR EACH ROW
WHEN (NEW.price >= 1000000)
EXECUTE FUNCTION send_alert();
```

---

### 이론: 다중 트리거, 재귀, `WHEN`

#### D.1 트리거 순서

같은 테이블의 같은 event에 두 트리거가 발사되면, PostgreSQL은 **트리거 이름의 알파벳 순으로** 발사. `t01_validate`, `t02_normalize`, `t03_audit` 같은 이름은 생성 순서에 의존하지 않고 순서를 통제하는 관례적 방식.

#### D.2 재귀

트리거 body가 같은 또는 다른 트리거를 발사하는 INSERT/UPDATE/DELETE를 issue할 수 있음. Unbounded 재귀가 가능(흔한 버그), PostgreSQL은 일반 statement nesting limit 외에 내장 재귀 깊이 제한이 없음. 방어적 코드 — 재진입 검출에 `pg_trigger_depth()` 사용, 또는 cycle을 깨기 위해 세션 변수 유지.

#### D.3 `WHEN` 절 — 트리거를 통째로 건너뛰기

```sql
CREATE TRIGGER ... AFTER UPDATE ON orders
FOR EACH ROW
WHEN (OLD.status IS DISTINCT FROM NEW.status)
EXECUTE FUNCTION on_status_change();
```

`WHEN` 술어는 트리거 시스템에 의해 함수 호출 *전*에, `NEW`와 `OLD`가 사용 가능한 상태로 평가됨. false면 함수가 아예 호출되지 않음. 함수에 들어가서 즉시 반환하는 것보다 훨씬 저렴 — 변경의 작은 부분집합만 신경 쓰는 트리거에 유용.

## 7. 실습 예제

### 실습 1: 자동 타임스탬프

```sql
-- updated_at 자동 갱신
CREATE FUNCTION update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 테이블에 적용
CREATE TABLE articles (
    id SERIAL PRIMARY KEY,
    title VARCHAR(200),
    content TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TRIGGER set_updated_at
BEFORE UPDATE ON articles
FOR EACH ROW
EXECUTE FUNCTION update_timestamp();

-- 테스트
INSERT INTO articles (title, content) VALUES ('제목', '내용');
SELECT * FROM articles;

UPDATE articles SET content = '수정된 내용' WHERE id = 1;
SELECT * FROM articles;  -- updated_at 자동 갱신됨
```

### 실습 2: 감사 로그

```sql
-- 감사 로그 테이블
CREATE TABLE audit_log (
    id SERIAL PRIMARY KEY,
    table_name VARCHAR(50),
    operation VARCHAR(10),
    old_data JSONB,
    new_data JSONB,
    changed_by VARCHAR(100),
    changed_at TIMESTAMP DEFAULT NOW()
);

-- 감사 트리거 함수
CREATE FUNCTION audit_trigger()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO audit_log (table_name, operation, new_data, changed_by)
        VALUES (TG_TABLE_NAME, 'INSERT', row_to_json(NEW)::JSONB, current_user);
        RETURN NEW;

    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit_log (table_name, operation, old_data, new_data, changed_by)
        VALUES (TG_TABLE_NAME, 'UPDATE', row_to_json(OLD)::JSONB, row_to_json(NEW)::JSONB, current_user);
        RETURN NEW;

    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO audit_log (table_name, operation, old_data, changed_by)
        VALUES (TG_TABLE_NAME, 'DELETE', row_to_json(OLD)::JSONB, current_user);
        RETURN OLD;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- 트리거 적용
CREATE TRIGGER users_audit
AFTER INSERT OR UPDATE OR DELETE ON users
FOR EACH ROW
EXECUTE FUNCTION audit_trigger();

-- 테스트
INSERT INTO users (name, email) VALUES ('감사테스트', 'audit@test.com');
UPDATE users SET name = '감사수정' WHERE email = 'audit@test.com';
DELETE FROM users WHERE email = 'audit@test.com';

SELECT * FROM audit_log;
```

### 실습 3: 재고 관리

```sql
-- 재고 테이블
CREATE TABLE inventory (
    product_id INTEGER PRIMARY KEY,
    quantity INTEGER DEFAULT 0,
    reserved INTEGER DEFAULT 0
);

-- 주문 시 재고 예약
CREATE FUNCTION reserve_stock()
RETURNS TRIGGER AS $$
DECLARE
    available INTEGER;
BEGIN
    SELECT quantity - reserved INTO available
    FROM inventory
    WHERE product_id = NEW.product_id;

    IF available < NEW.quantity THEN
        RAISE EXCEPTION '재고 부족: 가용 재고 %, 요청 %', available, NEW.quantity;
    END IF;

    UPDATE inventory
    SET reserved = reserved + NEW.quantity
    WHERE product_id = NEW.product_id;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER before_order_item
BEFORE INSERT ON order_items
FOR EACH ROW
EXECUTE FUNCTION reserve_stock();

-- 주문 완료 시 실제 재고 차감
CREATE FUNCTION complete_stock()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status = 'completed' AND OLD.status <> 'completed' THEN
        UPDATE inventory
        SET quantity = quantity - oi.quantity,
            reserved = reserved - oi.quantity
        FROM order_items oi
        WHERE oi.order_id = NEW.id
          AND inventory.product_id = oi.product_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER after_order_complete
AFTER UPDATE ON orders
FOR EACH ROW
EXECUTE FUNCTION complete_stock();
```

### 실습 4: 데이터 유효성 검사

```sql
-- 이메일 중복 검사 (대소문자 무시)
CREATE FUNCTION check_email_unique()
RETURNS TRIGGER AS $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM users
        WHERE LOWER(email) = LOWER(NEW.email)
          AND id <> COALESCE(NEW.id, -1)
    ) THEN
        RAISE EXCEPTION '이메일이 이미 존재합니다: %', NEW.email;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER before_user_email
BEFORE INSERT OR UPDATE OF email ON users
FOR EACH ROW
EXECUTE FUNCTION check_email_unique();
```

---

## 8. 트리거 관리

### 트리거 목록 확인

```sql
-- 테이블의 트리거 확인
SELECT tgname, tgtype, proname
FROM pg_trigger t
JOIN pg_proc p ON t.tgfoid = p.oid
WHERE tgrelid = 'users'::regclass;

-- 또는
\dS users
```

### 트리거 비활성화/활성화

```sql
-- 특정 트리거 비활성화
ALTER TABLE users DISABLE TRIGGER users_audit;

-- 모든 트리거 비활성화
ALTER TABLE users DISABLE TRIGGER ALL;

-- 활성화
ALTER TABLE users ENABLE TRIGGER users_audit;
ALTER TABLE users ENABLE TRIGGER ALL;
```

### 트리거 삭제

```sql
DROP TRIGGER trigger_name ON table_name;
DROP TRIGGER IF EXISTS trigger_name ON table_name;
```

---

## 9. 트리거 TG_ 변수

| 변수 | 설명 |
|------|------|
| `TG_NAME` | 트리거 이름 |
| `TG_TABLE_NAME` | 테이블 이름 |
| `TG_TABLE_SCHEMA` | 스키마 이름 |
| `TG_OP` | 작업 (INSERT, UPDATE, DELETE) |
| `TG_WHEN` | BEFORE 또는 AFTER |
| `TG_LEVEL` | ROW 또는 STATEMENT |

```sql
CREATE FUNCTION debug_trigger()
RETURNS TRIGGER AS $$
BEGIN
    RAISE NOTICE 'Trigger: %, Table: %, Op: %, When: %',
        TG_NAME, TG_TABLE_NAME, TG_OP, TG_WHEN;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
```

---

## 10. 주의사항

### 이론: 트랜잭션과 lock scope

트리거는 발사 statement와 **같은 트랜잭션 안에서** 실행. 이는 협상 불가.

#### C.1 같은 xid, 같은 snapshot

트리거는 호출 statement와 같은 MVCC snapshot을 봄. statement에 visible한 행은 트리거에도 visible, invisible한 행은 invisible. 트리거의 쓰기는 같은 xid 사용 — 호출 statement의 쓰기와 함께 commit되거나 rollback됨.

이로부터:

- **트리거는 `COMMIT`이나 `BEGIN` 불가**(트랜잭션 안에 있음, 이유는 10번 레슨 §D.1).
- **트리거의 쓰기는 발사 statement의 쓰기와 atomic**. 실패한 트리거는 전체 statement를 abort.
- **트리거가 또 다른 트리거를 발사(cascade)** — 같은 트랜잭션 안에서 실행. "트리거 트랜잭션" 경계 없음.

#### C.2 Constraint 트리거 — DEFERRED vs IMMEDIATE

대부분의 트리거는 statement 동안 즉시 발사. **Constraint 트리거**는 `DEFERRABLE INITIALLY DEFERRED`일 수 있고, 그 경우 commit 시점에 발사. 트랜잭션 중간에는 성립할 수 없는 inter-table 불변량을 강제하는 데 사용(예 — 두 INSERT로 set up되어야 하는 순환 외래 키).

#### C.3 Lock 획득

트리거가 다른 테이블을 읽거나 쓰면, 호출 트랜잭션의 lock 집합 안에서 적절한 lock을 획득. Deadlock 검출(11번 레슨 §D.3)은 트리거의 lock을 다른 lock과 동일하게 봄 — 발사마다 두 테이블을 일관되지 않은 순서로 update하는 트리거는 deadlock 대기 중.

### 무한 루프 방지

```sql
-- 나쁜 예: 트리거가 자신을 다시 호출
CREATE FUNCTION bad_trigger()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE same_table SET ...;  -- 같은 테이블 UPDATE → 무한 루프!
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
```

### 성능 고려

```sql
-- 트리거는 모든 작업에 오버헤드 추가
-- 대량 데이터 처리 시 트리거 비활성화 고려

ALTER TABLE users DISABLE TRIGGER ALL;
-- 대량 INSERT/UPDATE
ALTER TABLE users ENABLE TRIGGER ALL;
```

### 디버깅

```sql
-- RAISE NOTICE로 디버깅
CREATE FUNCTION debug_function()
RETURNS TRIGGER AS $$
BEGIN
    RAISE NOTICE 'OLD: %, NEW: %', OLD, NEW;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
```

---

**이전**: [트랜잭션](./11_Transactions.md) | **다음**: [백업과 운영](./13_Backup_and_Operations.md)
