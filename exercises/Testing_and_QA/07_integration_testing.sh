#!/bin/bash
# Exercises for Lesson 07: Integration Testing
# Topic: Testing_and_QA
# Solutions to practice problems from the lesson.

# === Exercise 1: Database Integration Tests ===
# Problem: Write integration tests for a user repository that
# interacts with SQLite. Test CRUD operations with real DB queries.
exercise_1() {
    echo "=== Exercise 1: Database Integration Tests ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import pytest
import sqlite3

class UserRepository:
    """Repository pattern wrapping database operations."""

    def __init__(self, connection):
        self.conn = connection

    def create(self, name: str, email: str) -> int:
        cursor = self.conn.execute(
            "INSERT INTO users (name, email) VALUES (?, ?)",
            (name, email)
        )
        self.conn.commit()
        return cursor.lastrowid

    def find_by_id(self, user_id: int) -> dict | None:
        row = self.conn.execute(
            "SELECT id, name, email FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if row:
            return {"id": row[0], "name": row[1], "email": row[2]}
        return None

    def find_by_email(self, email: str) -> dict | None:
        row = self.conn.execute(
            "SELECT id, name, email FROM users WHERE email = ?", (email,)
        ).fetchone()
        return {"id": row[0], "name": row[1], "email": row[2]} if row else None

    def delete(self, user_id: int) -> bool:
        cursor = self.conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        self.conn.commit()
        return cursor.rowcount > 0

# --- Test fixtures ---

@pytest.fixture
def db_conn():
    """Create an in-memory SQLite database for each test."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL
        )
    """)
    yield conn
    conn.close()

@pytest.fixture
def repo(db_conn):
    return UserRepository(db_conn)

# --- Integration tests ---

def test_create_and_find_user(repo):
    """Test the full create-find cycle with a real database."""
    user_id = repo.create("Alice", "alice@test.com")
    user = repo.find_by_id(user_id)

    assert user is not None
    assert user["name"] == "Alice"
    assert user["email"] == "alice@test.com"

def test_find_nonexistent_user(repo):
    assert repo.find_by_id(999) is None

def test_find_by_email(repo):
    repo.create("Bob", "bob@test.com")
    user = repo.find_by_email("bob@test.com")
    assert user["name"] == "Bob"

def test_unique_email_constraint(repo):
    """Test that the database enforces uniqueness."""
    repo.create("Alice", "alice@test.com")
    with pytest.raises(sqlite3.IntegrityError):
        repo.create("Another Alice", "alice@test.com")

def test_delete_user(repo):
    user_id = repo.create("Charlie", "charlie@test.com")
    assert repo.delete(user_id) is True
    assert repo.find_by_id(user_id) is None

def test_delete_nonexistent(repo):
    assert repo.delete(999) is False
SOLUTION
}

# === Exercise 2: Service Layer Integration ===
# Problem: Test a service that coordinates between two repositories.
exercise_2() {
    echo "=== Exercise 2: Service Layer Integration ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import pytest
import sqlite3

class OrderService:
    """Service that coordinates between user and order repositories."""

    def __init__(self, conn):
        self.conn = conn

    def place_order(self, user_id: int, product: str, amount: float) -> int:
        # Verify user exists
        user = self.conn.execute(
            "SELECT id FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if not user:
            raise ValueError(f"User {user_id} not found")

        cursor = self.conn.execute(
            "INSERT INTO orders (user_id, product, amount) VALUES (?, ?, ?)",
            (user_id, product, amount)
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_user_orders(self, user_id: int) -> list:
        rows = self.conn.execute(
            "SELECT id, product, amount FROM orders WHERE user_id = ?",
            (user_id,)
        ).fetchall()
        return [{"id": r[0], "product": r[1], "amount": r[2]} for r in rows]

    def get_user_total(self, user_id: int) -> float:
        row = self.conn.execute(
            "SELECT COALESCE(SUM(amount), 0) FROM orders WHERE user_id = ?",
            (user_id,)
        ).fetchone()
        return row[0]

@pytest.fixture
def service_db():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("""
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER REFERENCES users(id),
            product TEXT NOT NULL,
            amount REAL NOT NULL
        )
    """)
    conn.execute("INSERT INTO users VALUES (1, 'Alice')")
    conn.execute("INSERT INTO users VALUES (2, 'Bob')")
    conn.commit()
    yield conn
    conn.close()

@pytest.fixture
def order_service(service_db):
    return OrderService(service_db)

def test_place_order_for_valid_user(order_service):
    order_id = order_service.place_order(1, "Book", 29.99)
    assert order_id is not None
    assert order_id > 0

def test_place_order_for_invalid_user(order_service):
    with pytest.raises(ValueError, match="User 999 not found"):
        order_service.place_order(999, "Book", 29.99)

def test_get_user_orders(order_service):
    order_service.place_order(1, "Book", 29.99)
    order_service.place_order(1, "Pen", 3.50)
    orders = order_service.get_user_orders(1)
    assert len(orders) == 2

def test_get_user_total(order_service):
    order_service.place_order(1, "Book", 29.99)
    order_service.place_order(1, "Pen", 3.50)
    total = order_service.get_user_total(1)
    assert total == pytest.approx(33.49)

def test_different_users_isolated(order_service):
    """Orders for one user don't appear in another user's orders."""
    order_service.place_order(1, "Book", 29.99)
    order_service.place_order(2, "Pen", 3.50)
    assert len(order_service.get_user_orders(1)) == 1
    assert len(order_service.get_user_orders(2)) == 1
SOLUTION
}

# === Exercise 3: Integration Test Strategies ===
# Problem: Compare different strategies for integration testing:
# in-memory DB, test containers, and transaction rollback.
exercise_3() {
    echo "=== Exercise 3: Integration Test Strategies ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Strategy 1: In-Memory Database (used in exercises above)
# Pros: Fast, no external dependencies, perfect isolation
# Cons: May differ from production DB (SQLite vs PostgreSQL)
# Best for: Simple schemas, rapid development

# Strategy 2: Transaction Rollback
# Pros: Uses real database engine, fast cleanup
# Cons: Can't test commit/rollback behavior itself
@pytest.fixture
def db_session(real_db_connection):
    """Wrap each test in a transaction, then roll back."""
    transaction = real_db_connection.begin()
    yield real_db_connection
    transaction.rollback()  # Undo everything the test did

# Strategy 3: Test Containers (Docker)
# Pros: Identical to production, tests real constraints/features
# Cons: Slower startup, requires Docker
# Best for: CI/CD pipelines, complex queries
# pip install testcontainers

# from testcontainers.postgres import PostgresContainer
# @pytest.fixture(scope="session")
# def postgres():
#     with PostgresContainer("postgres:15") as pg:
#         yield pg.get_connection_url()

# Strategy 4: Shared Test Database with Cleanup
# Pros: Fast after initial setup
# Cons: Tests must clean up after themselves
@pytest.fixture
def clean_tables(db_conn):
    """Delete data from all tables before each test."""
    yield db_conn
    for table in ["orders", "users"]:
        db_conn.execute(f"DELETE FROM {table}")
    db_conn.commit()

# RECOMMENDATION:
# - Unit tests: Mock the database entirely
# - Integration tests: In-memory SQLite for speed
# - Pre-deploy tests: Test containers with real DB engine
# - Never test against production database!
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 07: Integration Testing"
echo "======================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
