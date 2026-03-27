"""
Exercises for Lesson 16: Database Design Case Study
Topic: Database_Theory

Solutions to practice problems from the lesson.
Covers schema extension, normalization analysis, index design,
transaction design, and schema critique/fix.
"""


# === Exercise 1: Complete E-Commerce Schema ===
# Problem: Extend ShopWave schema with product variants, order status history, seller payouts.

def exercise_1():
    """Extend e-commerce schema with new features."""
    print("=== Extension 1: Product Variants ===")
    print()
    variant_ddl = """
    -- Product variants (e.g., "Blue, Large", "Red, Small")
    CREATE TABLE product_variants (
        id          BIGSERIAL PRIMARY KEY,
        product_id  BIGINT NOT NULL REFERENCES products(id) ON DELETE CASCADE,
        sku         VARCHAR(50) NOT NULL UNIQUE,
        name        VARCHAR(100) NOT NULL,  -- e.g., "Blue, Large"
        price       DECIMAL(10,2) NOT NULL CHECK (price > 0),
        stock       INT NOT NULL DEFAULT 0 CHECK (stock >= 0),
        -- Variant attributes stored as JSONB for flexibility
        attributes  JSONB NOT NULL DEFAULT '{}',
        -- e.g., {"color": "blue", "size": "L"}
        is_active   BOOLEAN NOT NULL DEFAULT TRUE,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE INDEX idx_variant_product ON product_variants(product_id);
    CREATE INDEX idx_variant_attrs ON product_variants USING gin(attributes);

    -- Update order_items to reference variants instead of products
    ALTER TABLE order_items ADD COLUMN variant_id BIGINT REFERENCES product_variants(id);
    """
    print(variant_ddl)

    print("=== Extension 2: Order Status History ===")
    print()
    status_ddl = """
    CREATE TABLE order_status_history (
        id          BIGSERIAL PRIMARY KEY,
        order_id    BIGINT NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
        old_status  VARCHAR(20),
        new_status  VARCHAR(20) NOT NULL,
        changed_by  BIGINT REFERENCES users(id),
        changed_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        notes       TEXT
    );

    CREATE INDEX idx_status_history_order ON order_status_history(order_id, changed_at DESC);

    -- Trigger to auto-record status changes
    -- CREATE TRIGGER trg_order_status_change
    -- AFTER UPDATE OF status ON orders
    -- FOR EACH ROW EXECUTE FUNCTION record_status_change();
    """
    print(status_ddl)

    print("=== Extension 3: Seller Payouts ===")
    print()
    payout_ddl = """
    CREATE TABLE seller_commissions (
        id              BIGSERIAL PRIMARY KEY,
        order_item_id   BIGINT NOT NULL REFERENCES order_items(id),
        seller_id       BIGINT NOT NULL REFERENCES sellers(id),
        item_total      DECIMAL(10,2) NOT NULL,
        commission_rate DECIMAL(5,4) NOT NULL DEFAULT 0.10,  -- 10% default
        commission_amt  DECIMAL(10,2) GENERATED ALWAYS AS (item_total * commission_rate) STORED,
        payout_amount   DECIMAL(10,2) GENERATED ALWAYS AS (item_total - item_total * commission_rate) STORED,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE TABLE seller_payouts (
        id          BIGSERIAL PRIMARY KEY,
        seller_id   BIGINT NOT NULL REFERENCES sellers(id),
        amount      DECIMAL(12,2) NOT NULL CHECK (amount > 0),
        period_start DATE NOT NULL,
        period_end  DATE NOT NULL,
        status      VARCHAR(20) NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'processing', 'paid', 'failed')),
        paid_at     TIMESTAMPTZ,
        bank_ref    VARCHAR(100),
        created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE INDEX idx_payouts_seller ON seller_payouts(seller_id, period_start DESC);
    """
    print(payout_ddl)


# === Exercise 2: Normalization Analysis ===
# Problem: Analyze a denormalized flat_orders table.

def exercise_2():
    """Normalization analysis of flat_orders table."""
    print("flat_orders(order_id, customer_name, customer_email, product_name,")
    print("            product_sku, product_price, quantity, order_total, order_date,")
    print("            shipping_street, shipping_city, shipping_zip)")
    print()

    # 1. Functional dependencies
    print("1. Functional Dependencies:")
    fds = [
        "order_id -> customer_name, customer_email, order_total, order_date, shipping_street, shipping_city, shipping_zip",
        "customer_email -> customer_name",
        "product_sku -> product_name, product_price",
        "{order_id, product_sku} -> quantity",
        "shipping_zip -> shipping_city  (assuming US ZIP codes)",
    ]
    for fd in fds:
        print(f"  {fd}")
    print()

    # 2. Candidate keys
    print("2. Candidate key(s):")
    print("  {order_id, product_sku}")
    print("  (Together they uniquely identify each row: which product in which order)")
    print()

    # 3. Normal form analysis
    print("3. Normal form analysis:")
    print("  1NF? YES -- all attributes are atomic (no repeating groups)")
    print()
    print("  2NF? NO -- partial dependencies exist:")
    print("    order_id -> customer_name (depends on part of the key)")
    print("    product_sku -> product_name (depends on part of the key)")
    print()
    print("  Therefore: 1NF (not 2NF)")
    print()

    # 4. BCNF Decomposition
    print("4. BCNF Decomposition:")
    print()
    print("  Step 1: Decompose on customer_email -> customer_name")
    print("    R1(customer_email, customer_name)  -- key: {customer_email}")
    print("    Remaining: remove customer_name from original")
    print()
    print("  Step 2: Decompose on product_sku -> product_name, product_price")
    print("    R2(product_sku, product_name, product_price)  -- key: {product_sku}")
    print("    Remaining: remove product_name, product_price")
    print()
    print("  Step 3: Decompose on shipping_zip -> shipping_city")
    print("    R3(shipping_zip, shipping_city)  -- key: {shipping_zip}")
    print("    Remaining: remove shipping_city")
    print()
    print("  Step 4: Decompose on order_id -> customer_email, order_total, order_date, ...")
    print("    R4(order_id, customer_email, order_total, order_date, shipping_street, shipping_zip)")
    print("    R5(order_id, product_sku, quantity)  -- key: {order_id, product_sku}")
    print()

    print("  Final BCNF decomposition:")
    tables = [
        ("Customers", "customer_email, customer_name", "{customer_email}"),
        ("Products", "product_sku, product_name, product_price", "{product_sku}"),
        ("ZipCodes", "shipping_zip, shipping_city", "{shipping_zip}"),
        ("Orders", "order_id, customer_email, order_total, order_date, shipping_street, shipping_zip", "{order_id}"),
        ("OrderItems", "order_id, product_sku, quantity", "{order_id, product_sku}"),
    ]
    for name, attrs, key in tables:
        print(f"    {name}({attrs}) -- key: {key}")
    print()

    # 5. Lossless-join verification
    print("5. Lossless-join verification:")
    print("  Each decomposition step splits on an FD X -> Y where X is a key of one component.")
    print("  - customer_email -> customer_name: customer_email is key of R1")
    print("  - product_sku -> product_name, product_price: product_sku is key of R2")
    print("  - shipping_zip -> shipping_city: shipping_zip is key of R3")
    print("  - order_id -> (order attributes): order_id is key of R4")
    print("  All splits satisfy the lossless-join condition. Verified.")


# === Exercise 4: Transaction Design ===
# Problem: Design transaction for bulk discount with audit logging.

def exercise_4():
    """Transaction design for bulk discount application."""
    print("Business operation: 15% discount on Electronics, capped at $50/product")
    print()

    print("Requirements:")
    print("  1. No partial visibility (atomic)")
    print("  2. Don't block reads for > 1 second")
    print("  3. Log all changes with before/after values")
    print()

    transaction_sql = """
    -- Use REPEATABLE READ to see consistent snapshot of products
    -- but allow concurrent readers
    BEGIN ISOLATION LEVEL REPEATABLE READ;

    -- Create audit log entries in batch
    -- This captures before-values before any updates
    INSERT INTO price_audit_log (product_id, old_price, new_price, change_type, changed_at)
    SELECT
        p.id,
        p.price AS old_price,
        GREATEST(
            p.price - LEAST(p.price * 0.15, 50.00),
            0.01  -- never go below $0.01
        ) AS new_price,
        'bulk_discount_electronics_15pct',
        NOW()
    FROM products p
    JOIN categories c ON p.category_id = c.id
    WHERE c.name = 'Electronics'
      AND p.is_active = TRUE;

    -- Apply the discount
    UPDATE products p
    SET
        sale_price = GREATEST(
            p.price - LEAST(p.price * 0.15, 50.00),
            0.01
        ),
        updated_at = NOW()
    FROM categories c
    WHERE p.category_id = c.id
      AND c.name = 'Electronics'
      AND p.is_active = TRUE;

    COMMIT;
    """
    print("Transaction SQL:")
    print(transaction_sql)

    print("Design notes:")
    notes = [
        "Isolation level: REPEATABLE READ ensures consistent snapshot for the batch.",
        "Atomicity: Both INSERT (audit) and UPDATE (price) are in one transaction.",
        "Performance: UPDATE uses a single statement (not row-by-row) for speed.",
        "Blocking: REPEATABLE READ + MVCC means readers see old prices until commit.",
        "  PostgreSQL's MVCC lets SELECT queries proceed without blocking.",
        "Cap logic: LEAST(price * 0.15, 50.00) ensures discount <= $50.",
        "GREATEST(..., 0.01) prevents zero or negative prices.",
        "Alternative for very large catalogs: batch in chunks of 1000 with",
        "  advisory locks to limit lock duration."
    ]
    for n in notes:
        print(f"  - {n}")


# === Exercise 6: Critique and Fix Schema ===
# Problem: Find and fix 8+ problems in a poorly designed schema.

def exercise_6():
    """Find and fix schema design problems."""
    print("Original (problematic) schema:")
    print("""
    CREATE TABLE users (
        id INT,
        name TEXT,
        email TEXT,
        password TEXT,
        created DATE
    );

    CREATE TABLE products (
        id INT,
        title TEXT,
        price FLOAT,
        category TEXT,
        tags TEXT,  -- comma-separated: "electronics,sale,new"
        stock INT
    );

    CREATE TABLE orders (
        id INT,
        user_email TEXT,
        product_ids TEXT,  -- comma-separated: "1,5,12"
        quantities TEXT,   -- comma-separated: "2,1,3"
        total FLOAT,
        status TEXT,
        date TEXT
    );
    """)

    problems = [
        {
            "problem": "1. No PRIMARY KEY constraints",
            "why": "Without PKs, duplicate rows can exist. No referential integrity possible.",
            "fix": "ALTER TABLE users ADD PRIMARY KEY (id);\n"
                  "    ALTER TABLE products ADD PRIMARY KEY (id);\n"
                  "    ALTER TABLE orders ADD PRIMARY KEY (id);"
        },
        {
            "problem": "2. password stored in plaintext",
            "why": "Security vulnerability. If database is breached, all passwords are exposed.",
            "fix": "Rename to password_hash VARCHAR(255) NOT NULL.\n"
                  "    Store bcrypt/argon2 hashed passwords only."
        },
        {
            "problem": "3. FLOAT for price and total (precision loss)",
            "why": "FLOAT causes rounding errors for monetary values. 0.1 + 0.2 != 0.3.",
            "fix": "Change to DECIMAL(10,2) NOT NULL for exact decimal arithmetic."
        },
        {
            "problem": "4. tags as comma-separated TEXT (1NF violation)",
            "why": "Cannot query individual tags efficiently. No index support.",
            "fix": "Create separate product_tags(product_id FK, tag VARCHAR(50)).\n"
                  "    Or use JSONB/array type with GIN index."
        },
        {
            "problem": "5. product_ids and quantities as comma-separated TEXT (1NF violation)",
            "why": "Violates 1NF. Cannot enforce FK constraints. Cannot SUM or JOIN.",
            "fix": "Create order_items(order_id FK, product_id FK, quantity INT, unit_price DECIMAL)."
        },
        {
            "problem": "6. user_email instead of user_id FK",
            "why": "No referential integrity. If user changes email, orders break.",
            "fix": "Change to user_id INT NOT NULL REFERENCES users(id)."
        },
        {
            "problem": "7. date stored as TEXT",
            "why": "Cannot do date arithmetic, comparison, or indexing. '2025-1-5' != '2025-01-05'.",
            "fix": "Change to created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()."
        },
        {
            "problem": "8. No NOT NULL constraints",
            "why": "All columns accept NULL by default. A user with NULL email is invalid.",
            "fix": "Add NOT NULL to required columns: name, email, password_hash, price, title."
        },
        {
            "problem": "9. No UNIQUE constraint on email",
            "why": "Multiple users could have the same email address.",
            "fix": "ALTER TABLE users ADD CONSTRAINT uq_email UNIQUE (email);"
        },
        {
            "problem": "10. status as unconstrained TEXT",
            "why": "Any string value allowed. 'shipped', 'SHIPPED', 'Ship' are all different.",
            "fix": "Add CHECK (status IN ('pending', 'confirmed', 'shipped', 'delivered', 'cancelled'))."
        },
        {
            "problem": "11. stock has no CHECK constraint",
            "why": "Negative stock values are possible.",
            "fix": "ALTER TABLE products ADD CHECK (stock >= 0);"
        }
    ]

    print("Problems found and fixes:\n")
    for p in problems:
        print(f"  {p['problem']}")
        print(f"    Why: {p['why']}")
        print(f"    Fix: {p['fix']}")
        print()

    print("Corrected schema:")
    corrected_sql = """
    CREATE TABLE users (
        id              SERIAL PRIMARY KEY,
        name            VARCHAR(100) NOT NULL,
        email           VARCHAR(255) NOT NULL UNIQUE,
        password_hash   VARCHAR(255) NOT NULL,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE TABLE categories (
        id    SERIAL PRIMARY KEY,
        name  VARCHAR(50) NOT NULL UNIQUE
    );

    CREATE TABLE products (
        id          SERIAL PRIMARY KEY,
        title       VARCHAR(255) NOT NULL,
        price       DECIMAL(10,2) NOT NULL CHECK (price > 0),
        category_id INT REFERENCES categories(id),
        stock       INT NOT NULL DEFAULT 0 CHECK (stock >= 0),
        created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE TABLE product_tags (
        product_id  INT NOT NULL REFERENCES products(id) ON DELETE CASCADE,
        tag         VARCHAR(50) NOT NULL,
        PRIMARY KEY (product_id, tag)
    );

    CREATE TABLE orders (
        id          SERIAL PRIMARY KEY,
        user_id     INT NOT NULL REFERENCES users(id),
        total       DECIMAL(10,2) NOT NULL CHECK (total >= 0),
        status      VARCHAR(20) NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending','confirmed','shipped','delivered','cancelled')),
        created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE TABLE order_items (
        id          SERIAL PRIMARY KEY,
        order_id    INT NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
        product_id  INT NOT NULL REFERENCES products(id),
        quantity    INT NOT NULL CHECK (quantity > 0),
        unit_price  DECIMAL(10,2) NOT NULL CHECK (unit_price > 0)
    );
    """
    print(corrected_sql)


# === Exercise 7: Scaling Strategy ===
# Problem: Design scaling strategy for grown e-commerce platform.

def exercise_7():
    """Scaling strategy for large e-commerce platform."""
    print("ShopWave at scale:")
    print("  50M customers, 10M products, 500M orders, 1.5B order items, 100M reviews")
    print()

    strategies = [
        {
            "area": "1. Table partitioning",
            "tables": {
                "orders": "Range partition by created_at (monthly). Most queries filter by date range.",
                "order_items": "Range partition by created_at (monthly), matching orders for efficient joins.",
                "reviews": "Range partition by created_at (quarterly). Older reviews queried less.",
            }
        },
        {
            "area": "2. Read replicas",
            "tables": {
                "products": "2-3 read replicas. Product pages are read-heavy. Tolerate ~100ms lag.",
                "reviews": "2 read replicas. Review display is read-heavy, writes infrequent.",
                "orders": "1 read replica for reporting/analytics (separate from OLTP).",
            }
        },
        {
            "area": "3. Redis caching layer",
            "tables": {
                "Product catalog": "Cache product details, category listings (TTL: 5 min).",
                "Session data": "User sessions, cart contents (TTL: 30 min).",
                "Hot counters": "View counts, cart item counts (write-behind to PostgreSQL).",
                "Rate limiting": "API rate limits per user/IP.",
            }
        },
        {
            "area": "4. Specialized databases",
            "tables": {
                "Elasticsearch": "Full-text search for products (name, description, tags). "
                                "Faceted search, autocomplete, fuzzy matching.",
                "TimescaleDB": "Analytics events, clickstream data (time-series pattern).",
            }
        },
        {
            "area": "5. Zero-downtime migration plan",
            "tables": {
                "Step 1": "Set up read replicas first (no schema change needed).",
                "Step 2": "Add Redis caching with cache-aside pattern (app change only).",
                "Step 3": "Add Elasticsearch as secondary index (dual-write from app).",
                "Step 4": "Partition tables using pg_partman (online, minimal locking).",
                "Step 5": "Migrate analytics queries to TimescaleDB (separate service).",
            }
        }
    ]

    for s in strategies:
        print(f"{s['area']}:")
        for table, strategy in s["tables"].items():
            print(f"  {table}: {strategy}")
        print()


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Schema Extensions ===")
    print("=" * 70)
    exercise_1()

    print("=" * 70)
    print("=== Exercise 2: Normalization Analysis ===")
    print("=" * 70)
    exercise_2()

    print("=" * 70)
    print("=== Exercise 4: Transaction Design ===")
    print("=" * 70)
    exercise_4()

    print("=" * 70)
    print("=== Exercise 6: Schema Critique and Fix ===")
    print("=" * 70)
    exercise_6()

    print("=" * 70)
    print("=== Exercise 7: Scaling Strategy ===")
    print("=" * 70)
    exercise_7()

    print("\nAll exercises completed!")
