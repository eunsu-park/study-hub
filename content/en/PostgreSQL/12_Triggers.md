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
