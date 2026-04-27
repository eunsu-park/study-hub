# Functions and Procedures

**Previous**: [Views and Indexes](./09_Views_and_Indexes.md) | **Next**: [Transactions](./11_Transactions.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Use PostgreSQL built-in functions for strings, numbers, and dates
2. Create simple SQL functions with CREATE FUNCTION ... LANGUAGE SQL
3. Write PL/pgSQL functions with variables, IF-ELSE branching, CASE, and loops
4. Return table data from functions using RETURNS TABLE, RETURNS SETOF, and OUT parameters
5. Implement exception handling with BEGIN ... EXCEPTION blocks and RAISE
6. Create stored procedures (PROCEDURE) and explain how they differ from functions
7. Apply CREATE OR REPLACE to modify existing functions without dropping them
8. List, inspect, and drop user-defined functions and procedures

---

Built-in functions handle the most common transformations, but every application eventually needs custom logic that lives inside the database itself. User-defined functions and procedures let you encapsulate business rules -- tax calculations, tier classifications, input validation -- right next to the data, reducing round trips and ensuring that the logic is applied consistently no matter which client connects.

---

## 1. Built-in Functions

PostgreSQL provides various built-in functions.

### String Functions

| Function | Description | Example | Result |
|------|------|------|------|
| `LENGTH()` | String length | `LENGTH('Hello')` | 5 |
| `UPPER()` | Convert to uppercase | `UPPER('hello')` | HELLO |
| `LOWER()` | Convert to lowercase | `LOWER('HELLO')` | hello |
| `TRIM()` | Remove whitespace | `TRIM('  hi  ')` | hi |
| `SUBSTRING()` | Extract substring | `SUBSTRING('Hello', 1, 3)` | Hel |
| `REPLACE()` | Replace string | `REPLACE('Hello', 'l', 'L')` | HeLLo |
| `CONCAT()` | Concatenate strings | `CONCAT('A', 'B', 'C')` | ABC |
| `SPLIT_PART()` | Split by delimiter | `SPLIT_PART('a,b,c', ',', 2)` | b |

```sql
SELECT
    LENGTH('PostgreSQL') AS len,
    UPPER('hello') AS upper,
    LOWER('WORLD') AS lower,
    TRIM('  text  ') AS trimmed,
    SUBSTRING('PostgreSQL', 1, 8) AS sub,
    REPLACE('Hello', 'l', 'L') AS replaced,
    CONCAT('Post', 'gre', 'SQL') AS concat;
```

### Numeric Functions

| Function | Description | Example | Result |
|------|------|------|------|
| `ROUND()` | Round | `ROUND(3.567, 2)` | 3.57 |
| `FLOOR()` | Floor | `FLOOR(3.9)` | 3 |
| `CEIL()` | Ceiling | `CEIL(3.1)` | 4 |
| `ABS()` | Absolute value | `ABS(-5)` | 5 |
| `MOD()` | Modulo | `MOD(10, 3)` | 1 |
| `POWER()` | Power | `POWER(2, 3)` | 8 |
| `SQRT()` | Square root | `SQRT(16)` | 4 |
| `RANDOM()` | Random 0~1 | `RANDOM()` | 0.xxx |

```sql
SELECT
    ROUND(123.456, 2),
    FLOOR(9.9),
    CEIL(1.1),
    ABS(-100),
    MOD(17, 5),
    POWER(2, 10),
    ROUND(RANDOM() * 100);
```

### Date/Time Functions

| Function | Description |
|------|------|
| `NOW()` | Current timestamp |
| `CURRENT_DATE` | Current date |
| `CURRENT_TIME` | Current time |
| `DATE_TRUNC()` | Truncate date |
| `EXTRACT()` | Extract date part |
| `AGE()` | Date difference |
| `TO_CHAR()` | Format date |

```sql
SELECT
    NOW(),
    CURRENT_DATE,
    DATE_TRUNC('month', NOW()),
    EXTRACT(YEAR FROM NOW()),
    EXTRACT(DOW FROM NOW()),  -- 0=Sunday
    AGE('2024-12-31', '2024-01-01'),
    TO_CHAR(NOW(), 'YYYY-MM-DD HH24:MI:SS');
```

---

## 2. User-Defined Function Basics

### Theory: Function Languages and Inlining

PostgreSQL functions can be written in many languages. The two most common in user code:

#### A.1 SQL functions

`CREATE FUNCTION add_one(x int) RETURNS int LANGUAGE sql AS 'SELECT x + 1';`

The body is a (possibly multi-statement) SQL block. If the function is **simple enough** — single SELECT, IMMUTABLE/STABLE, no complex parameters — the planner can **inline** it: every call is replaced with the body during planning, just like a view. There is then no function-call overhead at runtime.

A SQL function is not always inlinable. Multi-statement bodies, VOLATILE marking, set-returning functions in unusual contexts — any of these blocks inlining and forces real per-call execution.

#### A.2 PL/pgSQL functions

`CREATE FUNCTION ... LANGUAGE plpgsql AS $$ DECLARE ... BEGIN ... END $$;`

PL/pgSQL is a procedural language layered on top of SQL. The body has variables, control flow (`IF`, `LOOP`, `FOR`, `WHILE`), exception handling, and explicit `RETURN`. PL/pgSQL functions are **never inlined** — every call is a real call, with its own variable scope, and its body is parsed once but executed each time.

The cost: function call overhead per invocation (microseconds, but it adds up if called per row). The benefit: anything you cannot express in pure SQL.

### Theory: Volatility Classes

Every function declares (or defaults) one of three **volatility** classes. This is metadata that tells the planner what optimizations are safe.

#### B.1 IMMUTABLE — same input, same output, forever

`abs(x)`, `length(s)`, `pi()`. The planner is free to:

- **Constant-fold** the call at planning time if all arguments are constants. `WHERE x = abs(-5)` becomes `WHERE x = 5` once.
- **Use the function in an expression index** (`CREATE INDEX ON t (immutable_func(col))`). PostgreSQL refuses to build an index on a STABLE or VOLATILE function because the index would silently go stale.

#### B.2 STABLE — same input, same output, within one statement

`now()`, `current_user`, `current_setting('foo')`. Inside one query, calling `now()` ten times returns the same value (PostgreSQL guarantees this). Across queries, it does not.

The planner can:

- **Use the function in an index scan as the bound** (`WHERE x > now()`). The function is called once at the start of the scan; the result is treated as a constant for the duration.
- **Not** constant-fold across statements.

#### B.3 VOLATILE — anything goes

`random()`, `nextval('seq')`, any function that performs `INSERT` or `UPDATE`. The planner must call it every time, in source order, and cannot move it across iteration boundaries.

#### B.4 Why getting this wrong silently breaks things

Marking `now()` as IMMUTABLE would let the planner constant-fold "today" into a value that never changes in your prepared statement — same query, same answer for the rest of the connection's life. Marking `random()` as STABLE would let the planner reuse one random value across an entire SELECT. PostgreSQL trusts your declaration; bad declarations produce wrong answers, not errors.

The default is VOLATILE, which is safe but pessimistic. Always think about the right class when defining a custom function.

### SQL Functions

```sql
-- Simple function
CREATE FUNCTION add_numbers(a INTEGER, b INTEGER)
RETURNS INTEGER
AS $$
    SELECT a + b;
$$ LANGUAGE SQL;

-- Usage
SELECT add_numbers(5, 3);  -- 8
```

### Dropping Functions

```sql
DROP FUNCTION add_numbers(INTEGER, INTEGER);
DROP FUNCTION IF EXISTS add_numbers(INTEGER, INTEGER);
```

---

## 3. PL/pgSQL Functions

PL/pgSQL is PostgreSQL's procedural language.

### Theory: PL/pgSQL — Parse, Plan, and Cache

A PL/pgSQL function body is processed in three phases.

#### C.1 First call — full parse

The first time a session calls the function, PL/pgSQL parses the entire body into an internal tree. Every embedded SQL statement (`SELECT`, `INSERT`, `EXECUTE`) becomes a node.

#### C.2 First execution of each SQL statement — plan and cache

Each embedded SQL statement is planned the first time it is executed and the plan is **cached for the lifetime of the session**. The next call to the function reuses the cached plan — no re-planning.

This makes PL/pgSQL fast for repeated calls but creates a subtle trap: the cached plan is built using the parameter values from the *first* call (PostgreSQL uses partial generic-plan logic). If subsequent calls have very different parameter selectivities, the cached plan can be poor for them. PostgreSQL switches between custom and generic plans after a few calls based on cost estimates, but pathological cases exist.

#### C.3 `EXECUTE` — explicit re-planning

Inside PL/pgSQL, plain SQL statements use the cached-plan mechanism. The dynamic form `EXECUTE 'SELECT ...';` re-parses and re-plans every time. Use it for queries where the parameter values radically change selectivity, or where the query text itself is built dynamically.

### Basic Structure

```sql
CREATE FUNCTION function_name(parameters)
RETURNS return_type
AS $$
DECLARE
    -- Variable declarations
BEGIN
    -- Function body
    RETURN value;
END;
$$ LANGUAGE plpgsql;
```

### Variables and Assignment

```sql
CREATE FUNCTION calculate_tax(price NUMERIC)
RETURNS NUMERIC
AS $$
DECLARE
    tax_rate NUMERIC := 0.1;  -- 10%
    tax_amount NUMERIC;
BEGIN
    tax_amount := price * tax_rate;
    RETURN tax_amount;
END;
$$ LANGUAGE plpgsql;

SELECT calculate_tax(10000);  -- 1000
```

### IF-ELSE

```sql
CREATE FUNCTION get_grade(score INTEGER)
RETURNS VARCHAR
AS $$
BEGIN
    IF score >= 90 THEN
        RETURN 'A';
    ELSIF score >= 80 THEN
        RETURN 'B';
    ELSIF score >= 70 THEN
        RETURN 'C';
    ELSIF score >= 60 THEN
        RETURN 'D';
    ELSE
        RETURN 'F';
    END IF;
END;
$$ LANGUAGE plpgsql;

SELECT get_grade(85);  -- B
```

### CASE Statement

```sql
CREATE FUNCTION day_name(day_num INTEGER)
RETURNS VARCHAR
AS $$
BEGIN
    RETURN CASE day_num
        WHEN 0 THEN 'Sunday'
        WHEN 1 THEN 'Monday'
        WHEN 2 THEN 'Tuesday'
        WHEN 3 THEN 'Wednesday'
        WHEN 4 THEN 'Thursday'
        WHEN 5 THEN 'Friday'
        WHEN 6 THEN 'Saturday'
        ELSE 'Invalid input'
    END;
END;
$$ LANGUAGE plpgsql;
```

### Loops

```sql
-- LOOP
CREATE FUNCTION factorial(n INTEGER)
RETURNS BIGINT
AS $$
DECLARE
    result BIGINT := 1;
    i INTEGER := 1;
BEGIN
    LOOP
        EXIT WHEN i > n;
        result := result * i;
        i := i + 1;
    END LOOP;
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- FOR loop
CREATE FUNCTION sum_to_n(n INTEGER)
RETURNS INTEGER
AS $$
DECLARE
    total INTEGER := 0;
BEGIN
    FOR i IN 1..n LOOP
        total := total + i;
    END LOOP;
    RETURN total;
END;
$$ LANGUAGE plpgsql;

-- WHILE
CREATE FUNCTION count_digits(num INTEGER)
RETURNS INTEGER
AS $$
DECLARE
    n INTEGER := ABS(num);
    count INTEGER := 0;
BEGIN
    WHILE n > 0 LOOP
        n := n / 10;
        count := count + 1;
    END LOOP;
    RETURN CASE WHEN count = 0 THEN 1 ELSE count END;
END;
$$ LANGUAGE plpgsql;
```

---

## 4. Returning Table Data

### RETURNS TABLE

```sql
CREATE FUNCTION get_users_by_city(p_city VARCHAR)
RETURNS TABLE (
    user_id INTEGER,
    user_name VARCHAR,
    user_email VARCHAR
)
AS $$
BEGIN
    RETURN QUERY
    SELECT id, name, email
    FROM users
    WHERE city = p_city;
END;
$$ LANGUAGE plpgsql;

-- Usage
SELECT * FROM get_users_by_city('Seoul');
```

### RETURNS SETOF

```sql
CREATE FUNCTION get_expensive_products(min_price NUMERIC)
RETURNS SETOF products
AS $$
BEGIN
    RETURN QUERY
    SELECT * FROM products WHERE price >= min_price;
END;
$$ LANGUAGE plpgsql;

-- Usage
SELECT * FROM get_expensive_products(100000);
```

### OUT Parameters

```sql
CREATE FUNCTION get_user_stats(
    IN p_user_id INTEGER,
    OUT order_count INTEGER,
    OUT total_amount NUMERIC
)
AS $$
BEGIN
    SELECT COUNT(*), COALESCE(SUM(amount), 0)
    INTO order_count, total_amount
    FROM orders
    WHERE user_id = p_user_id;
END;
$$ LANGUAGE plpgsql;

-- Usage
SELECT * FROM get_user_stats(1);
```

---

## 5. Exception Handling

```sql
CREATE FUNCTION safe_divide(a NUMERIC, b NUMERIC)
RETURNS NUMERIC
AS $$
BEGIN
    IF b = 0 THEN
        RAISE EXCEPTION 'Cannot divide by zero';
    END IF;
    RETURN a / b;
EXCEPTION
    WHEN division_by_zero THEN
        RAISE NOTICE 'Division by zero attempted';
        RETURN NULL;
    WHEN OTHERS THEN
        RAISE NOTICE 'Exception occurred: %', SQLERRM;
        RETURN NULL;
END;
$$ LANGUAGE plpgsql;
```

### RAISE Levels

```sql
RAISE DEBUG 'Debug message';
RAISE LOG 'Log message';
RAISE INFO 'Info message';
RAISE NOTICE 'Notice message';     -- Default output
RAISE WARNING 'Warning message';
RAISE EXCEPTION 'Error message';   -- Aborts execution
```

---

## 6. PROCEDURE

Procedures do not return values, they perform actions (PostgreSQL 11+).

### Theory: Functions vs Procedures — Transaction Scope

Until PostgreSQL 11, only functions existed. PG 11 added procedures (`CREATE PROCEDURE`), and the difference is purely about transaction control:

| Feature | Function | Procedure |
|---------|----------|-----------|
| Returns a value | Yes (`RETURNS type`) | No (use `OUT` parameters) |
| Called by | `SELECT func()` (inside expressions) | `CALL proc()` (top-level) |
| `COMMIT` / `ROLLBACK` inside body | **No** | **Yes** |
| Can run in parallel queries | If marked `PARALLEL SAFE` | No |

#### D.1 Why a function cannot COMMIT

A function call is part of a SQL statement. The statement is part of a transaction. Allowing the function to commit mid-call would mean the calling statement straddles a commit boundary — incoherent for MVCC visibility, undefined for triggers, impossible for the executor to clean up after on error.

Procedures are called from the top level, not from inside an expression. They can `COMMIT; ... BEGIN;` and start a new transaction inside the body. Use them for batch jobs that need to chunk work across multiple transactions (e.g., "process 1000 rows, commit, repeat").

#### D.2 Anonymous code — `DO` blocks

`DO $$ DECLARE x int; BEGIN ... END $$;` runs an anonymous PL/pgSQL block. Useful for one-off scripts. The block runs in the calling transaction; like functions, it cannot commit.

### Creating Procedures

```sql
CREATE PROCEDURE update_user_status(p_user_id INTEGER, p_status VARCHAR)
AS $$
BEGIN
    UPDATE users SET status = p_status WHERE id = p_user_id;
    RAISE NOTICE 'User % status changed to %', p_user_id, p_status;
END;
$$ LANGUAGE plpgsql;

-- Calling
CALL update_user_status(1, 'active');
```

### Transaction Control

```sql
CREATE PROCEDURE transfer_money(
    from_account INTEGER,
    to_account INTEGER,
    amount NUMERIC
)
AS $$
BEGIN
    UPDATE accounts SET balance = balance - amount WHERE id = from_account;
    UPDATE accounts SET balance = balance + amount WHERE id = to_account;
    COMMIT;
EXCEPTION
    WHEN OTHERS THEN
        ROLLBACK;
        RAISE;
END;
$$ LANGUAGE plpgsql;
```

---

## 7. Functions vs Procedures

| Feature | FUNCTION | PROCEDURE |
|------|-----------------|----------------------|
| Return value | Must return | No return |
| In SELECT | Can use | Cannot use |
| Call method | SELECT func() | CALL proc() |
| Transaction | External transaction | Can have own transaction |
| COMMIT/ROLLBACK | Not allowed | Allowed |

---

## 8. Practice Examples

### Practice 1: Utility Functions

```sql
-- 1. Extract email domain
CREATE FUNCTION get_email_domain(email VARCHAR)
RETURNS VARCHAR
AS $$
BEGIN
    RETURN SPLIT_PART(email, '@', 2);
END;
$$ LANGUAGE plpgsql;

SELECT get_email_domain('user@gmail.com');  -- gmail.com

-- 2. Calculate age
CREATE FUNCTION calculate_age(birth_date DATE)
RETURNS INTEGER
AS $$
BEGIN
    RETURN EXTRACT(YEAR FROM AGE(CURRENT_DATE, birth_date));
END;
$$ LANGUAGE plpgsql;

SELECT calculate_age('1990-05-15');  -- 34 (as of 2024)

-- 3. Format price
CREATE FUNCTION format_price(price NUMERIC)
RETURNS VARCHAR
AS $$
BEGIN
    RETURN TO_CHAR(price, 'FM999,999,999') || ' KRW';
END;
$$ LANGUAGE plpgsql;

SELECT format_price(1500000);  -- 1,500,000 KRW
```

### Practice 2: Business Logic Functions

```sql
-- 1. Calculate order total
CREATE FUNCTION calculate_order_total(p_order_id INTEGER)
RETURNS NUMERIC
AS $$
DECLARE
    total NUMERIC;
BEGIN
    SELECT SUM(p.price * oi.quantity)
    INTO total
    FROM order_items oi
    JOIN products p ON oi.product_id = p.id
    WHERE oi.order_id = p_order_id;

    RETURN COALESCE(total, 0);
END;
$$ LANGUAGE plpgsql;

-- 2. Determine user tier
CREATE FUNCTION get_user_tier(p_user_id INTEGER)
RETURNS VARCHAR
AS $$
DECLARE
    total_spent NUMERIC;
BEGIN
    SELECT COALESCE(SUM(amount), 0)
    INTO total_spent
    FROM orders
    WHERE user_id = p_user_id;

    RETURN CASE
        WHEN total_spent >= 1000000 THEN 'VIP'
        WHEN total_spent >= 500000 THEN 'Gold'
        WHEN total_spent >= 100000 THEN 'Silver'
        ELSE 'Bronze'
    END;
END;
$$ LANGUAGE plpgsql;
```

### Practice 3: Data Validation Functions

```sql
-- 1. Email validation
CREATE FUNCTION is_valid_email(email VARCHAR)
RETURNS BOOLEAN
AS $$
BEGIN
    RETURN email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$';
END;
$$ LANGUAGE plpgsql;

SELECT is_valid_email('test@email.com');  -- true
SELECT is_valid_email('invalid-email');   -- false

-- 2. Format phone number
CREATE FUNCTION format_phone(phone VARCHAR)
RETURNS VARCHAR
AS $$
DECLARE
    cleaned VARCHAR;
BEGIN
    cleaned := REGEXP_REPLACE(phone, '[^0-9]', '', 'g');
    IF LENGTH(cleaned) = 11 THEN
        RETURN SUBSTRING(cleaned, 1, 3) || '-' ||
               SUBSTRING(cleaned, 4, 4) || '-' ||
               SUBSTRING(cleaned, 8, 4);
    ELSE
        RETURN phone;
    END IF;
END;
$$ LANGUAGE plpgsql;

SELECT format_phone('01012345678');  -- 010-1234-5678
```

---

## 9. Function Management

### List Functions

```sql
-- psql command
\df

-- SQL query
SELECT routine_name, routine_type
FROM information_schema.routines
WHERE routine_schema = 'public';
```

### View Function Definition

```sql
-- View function source code
\sf function_name

-- Or
SELECT prosrc FROM pg_proc WHERE proname = 'function_name';
```

### Modify Functions

```sql
CREATE OR REPLACE FUNCTION function_name(...)
RETURNS ...
AS $$
    -- Modified content
$$ LANGUAGE plpgsql;
```

---

**Previous**: [Views and Indexes](./09_Views_and_Indexes.md) | **Next**: [Transactions](./11_Transactions.md)
