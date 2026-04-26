# Triggers

**Previous**: [Transactions](./11_Transactions.md) | **Next**: [Backup and Operations](./13_Backup_and_Operations.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the purpose of triggers and how they automate responses to data modification events
2. Create trigger functions that return the TRIGGER type in PL/pgSQL
3. Distinguish between BEFORE and AFTER triggers and choose the appropriate timing for a given use case
4. Use the NEW and OLD record variables to access row data within trigger functions
5. Compare FOR EACH ROW and FOR EACH STATEMENT trigger granularity
6. Implement conditional triggers using the WHEN clause
7. Build practical trigger-based solutions such as audit logs, timestamp updates, and inventory management
8. Manage triggers by listing, enabling, disabling, and dropping them

---

Triggers let you embed business rules directly into the database layer, ensuring that critical logic -- such as maintaining audit trails, validating data, or updating derived columns -- executes automatically whenever data changes. Instead of relying on every application to remember to call the right function, the database itself enforces consistency. This makes triggers an indispensable tool for data integrity in any production PostgreSQL system.

Before the syntax, read [**Theory & Principles**](#theory--principles) — the row-level vs statement-level distinction, the BEFORE/AFTER timing rules, what `NEW` and `OLD` actually contain, and the transaction scope a trigger inherits from its caller.

---

## Theory & Principles

A trigger looks like a callback in application code, but it runs in a fundamentally different environment. It executes inside the same transaction as the statement that fired it, sees the same MVCC snapshot, can modify the row about to be written (BEFORE) or react to the row that was just written (AFTER), and can either run *once per affected row* or *once per statement* regardless of row count. The choice of timing (BEFORE/AFTER), level (ROW/STATEMENT), and event (INSERT/UPDATE/DELETE/TRUNCATE) gives you 24 distinct trigger flavors — most apparent bugs in trigger code come from picking the wrong combination.

This section covers:

- **(A)** The 12-cell trigger matrix: BEFORE/AFTER × ROW/STATEMENT × INSERT/UPDATE/DELETE.
- **(B)** What `NEW` and `OLD` are in each cell, and how a BEFORE ROW trigger can change what gets written.
- **(C)** Trigger and transaction scope: same xact, same snapshot, same locks; deferred vs immediate.
- **(D)** Trigger order, recursion, and `WHEN` predicates.

### A. The Trigger Matrix

Two timings × two levels × four events = potentially 16 combinations, but TRUNCATE only fires statement-level, leaving 14 valid cells. The 12 most common are:

|       | INSERT | UPDATE | DELETE |
|-------|--------|--------|--------|
| **BEFORE ROW** | NEW exists; can modify NEW; return NULL to skip row | NEW + OLD; can modify NEW | OLD exists; return NULL to abort delete |
| **AFTER ROW** | NEW frozen | NEW + OLD frozen | OLD frozen |
| **BEFORE STATEMENT** | runs once before any row | runs once | runs once |
| **AFTER STATEMENT** | runs once after all rows | runs once | runs once |

#### A.1 BEFORE ROW — the "intercept" trigger

Fires *for each row, before the engine applies the change*. The trigger function receives the row that is about to be written (`NEW`) and, for UPDATE, the row before the change (`OLD`). The function:

- Can **modify `NEW`** in place and `RETURN NEW;` — the modified row is what gets written.
- Can `RETURN NULL;` — the engine **skips this row entirely** (no INSERT/UPDATE happens).
- Can `RAISE EXCEPTION` — aborts the entire statement (and, unless caught, the transaction).

Use BEFORE ROW for input validation, derived-column population (`NEW.normalized_email := lower(NEW.email)`), and conditional row suppression.

#### A.2 AFTER ROW — the "react" trigger

Fires *for each row, after the engine has applied the change*. `NEW` and `OLD` are read-only — the row is already on its way to disk. The trigger return value is ignored.

Use AFTER ROW for audit logging (insert into a `change_log` table), cache invalidation, sending NOTIFY, or any side effect that should happen only when the change actually committed-to-write succeeded.

#### A.3 BEFORE/AFTER STATEMENT — once per statement

Fire once regardless of how many rows the statement affected — even if the statement affected zero rows. `NEW` and `OLD` are NULL because there is no specific row. PostgreSQL 10+ exposes the affected row sets through **transition tables**:

```sql
CREATE TRIGGER ... AFTER UPDATE ON orders
REFERENCING OLD TABLE AS old_rows NEW TABLE AS new_rows
FOR EACH STATEMENT
EXECUTE FUNCTION audit_changes();
```

Inside the trigger function, `old_rows` and `new_rows` are queryable like temp tables containing all affected rows. Useful for "audit a batch UPDATE in one log entry instead of one per row".

### B. What `NEW` and `OLD` Are

Both are *records* of the same type as the row of the table the trigger is on. They have all the columns plus the system columns (`tableoid`, `xmin`, `xmax`, `ctid`).

| Event | `NEW` | `OLD` |
|-------|-------|-------|
| INSERT | row about to be inserted (BEFORE) or just inserted (AFTER) | undefined / NULL |
| UPDATE | row after change | row before change |
| DELETE | undefined / NULL | row about to be / just deleted |

#### B.1 BEFORE INSERT example — `NEW` is mutable

```sql
CREATE FUNCTION normalize_email() RETURNS trigger AS $$
BEGIN
    NEW.email := lower(trim(NEW.email));
    IF NEW.email = '' THEN
        RETURN NULL;  -- silently skip rows with empty email
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_normalize BEFORE INSERT ON users
FOR EACH ROW EXECUTE FUNCTION normalize_email();
```

The row that finally hits disk has the normalized email. The application code did not need to know.

#### B.2 The "must return" rule

BEFORE ROW triggers must `RETURN NEW`, `RETURN OLD` (for DELETE), or `RETURN NULL`. Forgetting the return is one of the most common PL/pgSQL bugs — the engine reads the return as NULL and silently drops the row.

AFTER ROW triggers can `RETURN NULL` or `RETURN NEW` — the value is ignored. Conventional style is `RETURN NEW;` for INSERT/UPDATE and `RETURN OLD;` for DELETE.

### C. Transaction and Locking Scope

A trigger runs **inside the same transaction** as the firing statement. This is non-negotiable.

#### C.1 Same xid, same snapshot

The trigger sees the same MVCC snapshot as the calling statement. Any row visible to the statement is visible to the trigger; any row invisible is invisible. The trigger's writes use the same xid — they commit or roll back together with the calling statement's writes.

This means:

- **A trigger cannot `COMMIT` or `BEGIN`** (it is inside a transaction; see lesson 10 §D.1 for why).
- **A trigger's writes are atomic with the firing statement's writes**. A failed trigger aborts the whole statement.
- **A trigger that itself fires another trigger (cascade)** runs within the same transaction. There is no "trigger transaction" boundary.

#### C.2 Constraint triggers — DEFERRED vs IMMEDIATE

Most triggers fire immediately during the statement. **Constraint triggers** can be `DEFERRABLE INITIALLY DEFERRED`, in which case they fire at commit time instead. Used to enforce inter-table invariants that cannot hold mid-transaction (e.g., circular foreign keys that need to be set up in two INSERTs).

#### C.3 Lock acquisition

If the trigger reads or writes other tables, it acquires the appropriate locks within the calling transaction's lock set. Deadlock detection (lesson 11 §D.3) sees the trigger's locks the same as any other lock — a trigger that updates two tables in inconsistent order across firings is a deadlock waiting to happen.

### D. Multiple Triggers, Recursion, and `WHEN`

#### D.1 Trigger ordering

If two triggers fire on the same event of the same table, PostgreSQL fires them **in alphabetical order of trigger name**. Naming triggers `t01_validate`, `t02_normalize`, `t03_audit` is the conventional way to control order without relying on creation order.

#### D.2 Recursion

A trigger's body can issue an INSERT/UPDATE/DELETE that fires the same or another trigger. Unbounded recursion is possible (and is a common bug); PostgreSQL has no built-in recursion depth limit beyond the general statement nesting limit. Defensive code: use `pg_trigger_depth()` to detect re-entry, or maintain a session variable to break the cycle.

#### D.3 `WHEN` clause — skip the trigger entirely

```sql
CREATE TRIGGER ... AFTER UPDATE ON orders
FOR EACH ROW
WHEN (OLD.status IS DISTINCT FROM NEW.status)
EXECUTE FUNCTION on_status_change();
```

The `WHEN` predicate is evaluated by the trigger system *before* invoking the function, with `NEW` and `OLD` available. If false, the function is not called at all. This is much cheaper than entering the function and immediately returning — useful for triggers that only care about a small subset of changes.

### From Theory to the SQL Below

Each of the following sections is one of these mechanisms made concrete:

- **`CREATE FUNCTION ... RETURNS trigger`** — the trigger function, with access to `NEW`/`OLD` (§B).
- **`CREATE TRIGGER ... BEFORE / AFTER ... FOR EACH ROW / STATEMENT`** — picks one of the §A cells.
- **`RETURN NEW` / `RETURN NULL`** — controls whether the row is written (§B.2).
- **`REFERENCING OLD TABLE / NEW TABLE`** — transition tables for STATEMENT-level triggers (§A.3).
- **`WHEN (...)`** — predicate that skips the function call entirely (§D.3).
- **`pg_trigger_depth()`** — recursion guard inside trigger functions (§D.2).
- **`DEFERRABLE INITIALLY DEFERRED`** — constraint triggers that fire at commit (§C.2).

---

## 1. Trigger Concept

A trigger is a function that automatically executes when a specific event (INSERT, UPDATE, DELETE) occurs.

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   INSERT    │ ──▶ │   TRIGGER   │ ──▶ │  Auto-exec  │
│   UPDATE    │     │  (Monitor)  │     │  (Trigger   │
│   DELETE    │     │             │     │  Function)  │
└─────────────┘     └─────────────┘     └─────────────┘
```

---

## 2. Trigger Components

1. **Trigger Function**: Logic to execute
2. **Trigger**: Defines when and on which table to execute the function

### Creating Trigger Functions

```sql
CREATE FUNCTION trigger_function_name()
RETURNS TRIGGER
AS $$
BEGIN
    -- Logic
    RETURN NEW;  -- Or RETURN OLD; or RETURN NULL;
END;
$$ LANGUAGE plpgsql;
```

### Creating Triggers

```sql
CREATE TRIGGER trigger_name
{BEFORE | AFTER | INSTEAD OF} {INSERT | UPDATE | DELETE}
ON table_name
[FOR EACH ROW | FOR EACH STATEMENT]
EXECUTE FUNCTION trigger_function_name();
```

---

## 3. BEFORE vs AFTER

### BEFORE Trigger

Executes **before** the event. Can validate or modify data.

```sql
-- Raise error if price is 0 or less
CREATE FUNCTION check_price()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.price <= 0 THEN
        RAISE EXCEPTION 'Price must be greater than 0';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER before_insert_product
BEFORE INSERT ON products
FOR EACH ROW
EXECUTE FUNCTION check_price();
```

### AFTER Trigger

Executes **after** the event. Used for audit logs, notifications, etc.

```sql
-- Reduce stock after order creation
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

| Variable | INSERT | UPDATE | DELETE |
|------|--------|--------|--------|
| `NEW` | New row | New row | None |
| `OLD` | None | Old row | Deleted row |

```sql
-- Compare old and new values on UPDATE
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

## 5. FOR EACH ROW vs FOR EACH STATEMENT

### FOR EACH ROW

Trigger executes for each row.

```sql
-- Execute for each row
CREATE TRIGGER row_trigger
AFTER INSERT ON products
FOR EACH ROW
EXECUTE FUNCTION my_function();

-- INSERT INTO products VALUES (...), (...), (...);
-- → Executes 3 times
```

### FOR EACH STATEMENT

Executes once per statement.

```sql
-- Execute once per statement
CREATE TRIGGER statement_trigger
AFTER INSERT ON products
FOR EACH STATEMENT
EXECUTE FUNCTION my_function();

-- INSERT INTO products VALUES (...), (...), (...);
-- → Executes 1 time
```

---

## 6. Conditional Triggers (WHEN)

```sql
-- Execute only when price is 1,000,000 or more
CREATE TRIGGER high_price_alert
AFTER INSERT ON products
FOR EACH ROW
WHEN (NEW.price >= 1000000)
EXECUTE FUNCTION send_alert();
```

---

## 7. Practice Examples

### Practice 1: Auto Timestamp

```sql
-- Auto-update updated_at
CREATE FUNCTION update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply to table
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

-- Test
INSERT INTO articles (title, content) VALUES ('Title', 'Content');
SELECT * FROM articles;

UPDATE articles SET content = 'Modified content' WHERE id = 1;
SELECT * FROM articles;  -- updated_at automatically updated
```

### Practice 2: Audit Log

```sql
-- Audit log table
CREATE TABLE audit_log (
    id SERIAL PRIMARY KEY,
    table_name VARCHAR(50),
    operation VARCHAR(10),
    old_data JSONB,
    new_data JSONB,
    changed_by VARCHAR(100),
    changed_at TIMESTAMP DEFAULT NOW()
);

-- Audit trigger function
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

-- Apply trigger
CREATE TRIGGER users_audit
AFTER INSERT OR UPDATE OR DELETE ON users
FOR EACH ROW
EXECUTE FUNCTION audit_trigger();

-- Test
INSERT INTO users (name, email) VALUES ('Audit Test', 'audit@test.com');
UPDATE users SET name = 'Audit Modified' WHERE email = 'audit@test.com';
DELETE FROM users WHERE email = 'audit@test.com';

SELECT * FROM audit_log;
```

### Practice 3: Inventory Management

```sql
-- Inventory table
CREATE TABLE inventory (
    product_id INTEGER PRIMARY KEY,
    quantity INTEGER DEFAULT 0,
    reserved INTEGER DEFAULT 0
);

-- Reserve stock on order
CREATE FUNCTION reserve_stock()
RETURNS TRIGGER AS $$
DECLARE
    available INTEGER;
BEGIN
    SELECT quantity - reserved INTO available
    FROM inventory
    WHERE product_id = NEW.product_id;

    IF available < NEW.quantity THEN
        RAISE EXCEPTION 'Insufficient stock: available %, requested %', available, NEW.quantity;
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

-- Deduct actual stock on order completion
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

### Practice 4: Data Validation

```sql
-- Email uniqueness check (case-insensitive)
CREATE FUNCTION check_email_unique()
RETURNS TRIGGER AS $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM users
        WHERE LOWER(email) = LOWER(NEW.email)
          AND id <> COALESCE(NEW.id, -1)
    ) THEN
        RAISE EXCEPTION 'Email already exists: %', NEW.email;
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

## 8. Trigger Management

### List Triggers

```sql
-- Check table's triggers
SELECT tgname, tgtype, proname
FROM pg_trigger t
JOIN pg_proc p ON t.tgfoid = p.oid
WHERE tgrelid = 'users'::regclass;

-- Or
\dS users
```

### Disable/Enable Triggers

```sql
-- Disable specific trigger
ALTER TABLE users DISABLE TRIGGER users_audit;

-- Disable all triggers
ALTER TABLE users DISABLE TRIGGER ALL;

-- Enable
ALTER TABLE users ENABLE TRIGGER users_audit;
ALTER TABLE users ENABLE TRIGGER ALL;
```

### Drop Triggers

```sql
DROP TRIGGER trigger_name ON table_name;
DROP TRIGGER IF EXISTS trigger_name ON table_name;
```

---

## 9. Trigger TG_ Variables

| Variable | Description |
|------|------|
| `TG_NAME` | Trigger name |
| `TG_TABLE_NAME` | Table name |
| `TG_TABLE_SCHEMA` | Schema name |
| `TG_OP` | Operation (INSERT, UPDATE, DELETE) |
| `TG_WHEN` | BEFORE or AFTER |
| `TG_LEVEL` | ROW or STATEMENT |

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

## 10. Precautions

### Prevent Infinite Loops

```sql
-- Bad example: Trigger calls itself
CREATE FUNCTION bad_trigger()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE same_table SET ...;  -- UPDATE same table → infinite loop!
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
```

### Performance Considerations

```sql
-- Triggers add overhead to all operations
-- Consider disabling triggers for bulk data processing

ALTER TABLE users DISABLE TRIGGER ALL;
-- Bulk INSERT/UPDATE
ALTER TABLE users ENABLE TRIGGER ALL;
```

### Debugging

```sql
-- Debug with RAISE NOTICE
CREATE FUNCTION debug_function()
RETURNS TRIGGER AS $$
BEGIN
    RAISE NOTICE 'OLD: %, NEW: %', OLD, NEW;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
```

---

---

**Previous**: [Transactions](./11_Transactions.md) | **Next**: [Backup and Operations](./13_Backup_and_Operations.md)
