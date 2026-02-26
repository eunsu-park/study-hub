"""
Database Design Case Study: E-Commerce System

Demonstrates:
- Requirements analysis → ER model → Relational schema
- Normalization (1NF → 3NF)
- Index strategy
- Query optimization considerations

Theory:
- Database design follows: requirements → conceptual (ER) →
  logical (relational) → physical (indexes, partitions).
- Normalization eliminates redundancy and update anomalies.
- Denormalization may be needed for read-heavy workloads.
- Index design balances read speed vs write overhead.

Adapted from Database Theory Lesson 16.
"""

import sqlite3
from collections import defaultdict


# ── Schema Design ──────────────────────────────────────────────────────

SCHEMA = """
-- Users table
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Products table
CREATE TABLE IF NOT EXISTS products (
    product_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    price REAL NOT NULL CHECK(price > 0),
    stock INTEGER NOT NULL DEFAULT 0 CHECK(stock >= 0),
    category TEXT NOT NULL
);

-- Orders table
CREATE TABLE IF NOT EXISTS orders (
    order_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(user_id),
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK(status IN ('pending', 'confirmed', 'shipped', 'delivered', 'cancelled')),
    total REAL NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Order Items (junction table)
CREATE TABLE IF NOT EXISTS order_items (
    item_id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER NOT NULL REFERENCES orders(order_id),
    product_id INTEGER NOT NULL REFERENCES products(product_id),
    quantity INTEGER NOT NULL CHECK(quantity > 0),
    unit_price REAL NOT NULL,
    UNIQUE(order_id, product_id)
);

-- Reviews
CREATE TABLE IF NOT EXISTS reviews (
    review_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(user_id),
    product_id INTEGER NOT NULL REFERENCES products(product_id),
    rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
    comment TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, product_id)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_orders_user ON orders(user_id);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_order_items_order ON order_items(order_id);
CREATE INDEX IF NOT EXISTS idx_products_category ON products(category);
CREATE INDEX IF NOT EXISTS idx_reviews_product ON reviews(product_id);
"""


def create_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA)
    return conn


def seed_data(conn: sqlite3.Connection) -> None:
    """Insert sample data."""
    cur = conn.cursor()

    # Users
    users = [
        ("alice@example.com", "Alice"),
        ("bob@example.com", "Bob"),
        ("charlie@example.com", "Charlie"),
    ]
    cur.executemany("INSERT INTO users (email, name) VALUES (?, ?)", users)

    # Products
    products = [
        ("Laptop", "High-performance laptop", 999.99, 50, "Electronics"),
        ("Mouse", "Wireless mouse", 29.99, 200, "Electronics"),
        ("Keyboard", "Mechanical keyboard", 79.99, 150, "Electronics"),
        ("Python Book", "Learn Python", 39.99, 100, "Books"),
        ("DB Book", "Database Systems", 49.99, 80, "Books"),
        ("Headphones", "Noise-cancelling", 199.99, 75, "Electronics"),
    ]
    cur.executemany(
        "INSERT INTO products (name, description, price, stock, category) "
        "VALUES (?, ?, ?, ?, ?)", products
    )

    # Orders
    orders = [
        (1, "delivered", 1029.98),  # Alice: laptop + mouse
        (1, "shipped", 79.99),      # Alice: keyboard
        (2, "pending", 89.98),      # Bob: 2 books
        (3, "confirmed", 229.98),   # Charlie: mouse + headphones
    ]
    cur.executemany(
        "INSERT INTO orders (user_id, status, total) VALUES (?, ?, ?)",
        orders
    )

    # Order items
    items = [
        (1, 1, 1, 999.99),  # Order 1: 1 laptop
        (1, 2, 1, 29.99),   # Order 1: 1 mouse
        (2, 3, 1, 79.99),   # Order 2: 1 keyboard
        (3, 4, 1, 39.99),   # Order 3: 1 Python book
        (3, 5, 1, 49.99),   # Order 3: 1 DB book
        (4, 2, 1, 29.99),   # Order 4: 1 mouse
        (4, 6, 1, 199.99),  # Order 4: 1 headphones
    ]
    cur.executemany(
        "INSERT INTO order_items (order_id, product_id, quantity, unit_price) "
        "VALUES (?, ?, ?, ?)", items
    )

    # Reviews
    reviews = [
        (1, 1, 5, "Excellent laptop!"),
        (1, 2, 4, "Good mouse, decent battery"),
        (2, 4, 5, "Great Python resource"),
        (3, 6, 3, "OK sound quality"),
    ]
    cur.executemany(
        "INSERT INTO reviews (user_id, product_id, rating, comment) "
        "VALUES (?, ?, ?, ?)", reviews
    )
    conn.commit()


# ── Demos ──────────────────────────────────────────────────────────────

def demo_schema():
    print("=" * 60)
    print("E-COMMERCE SCHEMA DESIGN")
    print("=" * 60)

    print(f"""
  Entity-Relationship Model:

    Users ──1:N── Orders ──1:N── OrderItems ──N:1── Products
      │                                               │
      └─────────────────N:1── Reviews ──N:1───────────┘

  Tables:
    users(user_id PK, email UQ, name)
    products(product_id PK, name, price, stock, category)
    orders(order_id PK, user_id FK, status, total)
    order_items(item_id PK, order_id FK, product_id FK, qty, unit_price)
    reviews(review_id PK, user_id FK, product_id FK, rating, comment)

  Normalization:
    1NF: ✓ All columns atomic, no repeating groups
    2NF: ✓ No partial dependencies (all non-key columns depend on full PK)
    3NF: ✓ No transitive dependencies
         (unit_price in order_items captures price at time of order,
          not derived from products.price)""")


def demo_queries(conn: sqlite3.Connection):
    print("\n" + "=" * 60)
    print("COMMON QUERIES")
    print("=" * 60)

    cur = conn.cursor()

    # 1. User's order history
    print(f"\n  1. Alice's order history:")
    cur.execute("""
        SELECT o.order_id, o.status, o.total, GROUP_CONCAT(p.name)
        FROM orders o
        JOIN order_items oi ON o.order_id = oi.order_id
        JOIN products p ON oi.product_id = p.product_id
        WHERE o.user_id = 1
        GROUP BY o.order_id
    """)
    print(f"    {'Order':>6} {'Status':<12} {'Total':>8} Products")
    print(f"    {'-'*6} {'-'*12} {'-'*8} {'-'*25}")
    for row in cur.fetchall():
        print(f"    {row[0]:>6} {row[1]:<12} ${row[2]:>7.2f} {row[3]}")

    # 2. Product with average rating
    print(f"\n  2. Products with ratings:")
    cur.execute("""
        SELECT p.name, p.price,
               COALESCE(AVG(r.rating), 0) as avg_rating,
               COUNT(r.review_id) as review_count
        FROM products p
        LEFT JOIN reviews r ON p.product_id = r.product_id
        GROUP BY p.product_id
        ORDER BY avg_rating DESC
    """)
    print(f"    {'Product':<15} {'Price':>8} {'Avg Rating':>11} {'Reviews':>8}")
    print(f"    {'-'*15} {'-'*8} {'-'*11} {'-'*8}")
    for row in cur.fetchall():
        stars = "★" * int(row[2]) + "☆" * (5 - int(row[2])) if row[2] > 0 else "No ratings"
        print(f"    {row[0]:<15} ${row[1]:>7.2f} {row[2]:>5.1f} {stars} {row[3]:>3}")

    # 3. Revenue by category
    print(f"\n  3. Revenue by category:")
    cur.execute("""
        SELECT p.category,
               SUM(oi.quantity * oi.unit_price) as revenue,
               COUNT(DISTINCT o.order_id) as orders
        FROM order_items oi
        JOIN products p ON oi.product_id = p.product_id
        JOIN orders o ON oi.order_id = o.order_id
        GROUP BY p.category
        ORDER BY revenue DESC
    """)
    print(f"    {'Category':<15} {'Revenue':>10} {'Orders':>7}")
    print(f"    {'-'*15} {'-'*10} {'-'*7}")
    for row in cur.fetchall():
        print(f"    {row[0]:<15} ${row[1]:>9.2f} {row[2]:>7}")


def demo_index_strategy(conn: sqlite3.Connection):
    print("\n" + "=" * 60)
    print("INDEX STRATEGY")
    print("=" * 60)

    cur = conn.cursor()

    # Show query plans
    queries = [
        ("Find user by email",
         "SELECT * FROM users WHERE email = 'alice@example.com'"),
        ("Orders by user",
         "SELECT * FROM orders WHERE user_id = 1"),
        ("Products by category",
         "SELECT * FROM products WHERE category = 'Electronics'"),
        ("Full table scan (no index)",
         "SELECT * FROM products WHERE description LIKE '%laptop%'"),
    ]

    for desc, sql in queries:
        cur.execute(f"EXPLAIN QUERY PLAN {sql}")
        plan = cur.fetchone()
        uses_index = "USING INDEX" in str(plan) or "USING COVERING INDEX" in str(plan)
        method = "Index scan" if uses_index else "Table scan"
        print(f"\n  {desc}:")
        print(f"    SQL: {sql}")
        print(f"    Plan: {plan[-1]}")
        print(f"    Method: {method}")

    print(f"""
  Index Design Guidelines:
    ✓ Index foreign keys (user_id, product_id, order_id)
    ✓ Index frequently filtered columns (status, category)
    ✓ Index unique constraints (email)
    ✗ Don't index rarely queried columns
    ✗ Don't over-index (slows writes)
    ⚠ Consider composite indexes for common multi-column queries""")


if __name__ == "__main__":
    demo_schema()
    conn = create_db()
    seed_data(conn)
    demo_queries(conn)
    demo_index_strategy(conn)
    conn.close()
