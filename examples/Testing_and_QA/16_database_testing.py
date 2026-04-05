#!/usr/bin/env python3
"""Example: Database Testing

Demonstrates test fixtures for databases, transaction rollback isolation,
factory patterns for test data, and repository testing strategies.
Related lesson: 16_Database_Testing.md
"""

# =============================================================================
# WHY DATABASE TESTING IS HARD
#
# Databases introduce statefulness — the enemy of test isolation.
# Key challenges:
#   1. Tests must start with a known state (fixtures / factories)
#   2. Tests must not leak state to other tests (transaction rollback)
#   3. Schema must match production (migrations)
#   4. Tests must be fast (in-memory SQLite or transaction rollback)
#
# Strategy: Use SQLite for unit tests, real DB for integration tests.
# Transaction rollback keeps tests fast and isolated without teardown logic.
# =============================================================================

import pytest
import sqlite3
from dataclasses import dataclass, field
from typing import Optional
from contextlib import contextmanager


# =============================================================================
# PRODUCTION CODE — REPOSITORY PATTERN
# =============================================================================

@dataclass
class User:
    """Domain model for a user."""
    id: Optional[int] = None
    name: str = ""
    email: str = ""
    active: bool = True


class UserRepository:
    """Data access layer — isolates SQL from business logic.
    In production, this would use SQLAlchemy or similar ORM."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def create_table(self):
        """Create the users table (idempotent)."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                active BOOLEAN NOT NULL DEFAULT 1
            )
        """)
        self.conn.commit()

    def insert(self, user: User) -> User:
        """Insert a new user and return it with the generated ID."""
        cursor = self.conn.execute(
            "INSERT INTO users (name, email, active) VALUES (?, ?, ?)",
            (user.name, user.email, user.active),
        )
        self.conn.commit()
        user.id = cursor.lastrowid
        return user

    def find_by_id(self, user_id: int) -> Optional[User]:
        """Find a user by primary key."""
        row = self.conn.execute(
            "SELECT id, name, email, active FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        if row is None:
            return None
        return User(id=row[0], name=row[1], email=row[2], active=bool(row[3]))

    def find_by_email(self, email: str) -> Optional[User]:
        """Find a user by email address."""
        row = self.conn.execute(
            "SELECT id, name, email, active FROM users WHERE email = ?",
            (email,),
        ).fetchone()
        if row is None:
            return None
        return User(id=row[0], name=row[1], email=row[2], active=bool(row[3]))

    def find_active(self) -> list[User]:
        """Find all active users."""
        rows = self.conn.execute(
            "SELECT id, name, email, active FROM users WHERE active = 1"
        ).fetchall()
        return [User(id=r[0], name=r[1], email=r[2], active=bool(r[3])) for r in rows]

    def update(self, user: User) -> None:
        """Update an existing user."""
        self.conn.execute(
            "UPDATE users SET name = ?, email = ?, active = ? WHERE id = ?",
            (user.name, user.email, user.active, user.id),
        )
        self.conn.commit()

    def delete(self, user_id: int) -> bool:
        """Delete a user. Returns True if a row was deleted."""
        cursor = self.conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    def count(self) -> int:
        """Count total users."""
        return self.conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]


# =============================================================================
# FACTORY PATTERN FOR TEST DATA
# =============================================================================

class UserFactory:
    """Factory for creating test users with sensible defaults.
    Override only the fields that matter for your specific test.

    This is a simplified version of what factory_boy provides."""

    _counter = 0

    @classmethod
    def build(cls, **overrides) -> User:
        """Create a User instance (not saved to DB)."""
        cls._counter += 1
        defaults = {
            "name": f"User {cls._counter}",
            "email": f"user{cls._counter}@test.com",
            "active": True,
        }
        defaults.update(overrides)
        return User(**defaults)

    @classmethod
    def create(cls, repo: UserRepository, **overrides) -> User:
        """Create a User and persist it via the repository."""
        user = cls.build(**overrides)
        return repo.insert(user)

    @classmethod
    def create_batch(cls, repo: UserRepository, count: int, **overrides) -> list[User]:
        """Create multiple users at once."""
        return [cls.create(repo, **overrides) for _ in range(count)]

    @classmethod
    def reset(cls):
        """Reset the counter between test sessions."""
        cls._counter = 0


# =============================================================================
# FIXTURES — DATABASE SETUP AND TEARDOWN
# =============================================================================

@pytest.fixture
def db_conn():
    """Fresh in-memory SQLite database per test.
    Each test gets a completely isolated database — no cleanup needed."""
    conn = sqlite3.connect(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def repo(db_conn):
    """UserRepository with schema already created."""
    repo = UserRepository(db_conn)
    repo.create_table()
    return repo


@pytest.fixture(autouse=True)
def reset_factory():
    """Reset factory counter before each test for predictable names."""
    UserFactory.reset()
    yield


# =============================================================================
# TESTS — BASIC CRUD
# =============================================================================

class TestUserCRUD:
    """Test Create, Read, Update, Delete operations."""

    def test_insert_and_find(self, repo):
        user = UserFactory.create(repo, name="Alice", email="alice@example.com")
        found = repo.find_by_id(user.id)

        assert found is not None
        assert found.name == "Alice"
        assert found.email == "alice@example.com"
        assert found.active is True

    def test_find_nonexistent_returns_none(self, repo):
        assert repo.find_by_id(999) is None

    def test_find_by_email(self, repo):
        UserFactory.create(repo, email="bob@example.com")
        found = repo.find_by_email("bob@example.com")
        assert found is not None
        assert found.email == "bob@example.com"

    def test_update_user(self, repo):
        user = UserFactory.create(repo, name="Carol")
        user.name = "Carol Updated"
        repo.update(user)

        found = repo.find_by_id(user.id)
        assert found.name == "Carol Updated"

    def test_delete_user(self, repo):
        user = UserFactory.create(repo)
        assert repo.delete(user.id) is True
        assert repo.find_by_id(user.id) is None

    def test_delete_nonexistent_returns_false(self, repo):
        assert repo.delete(999) is False

    def test_unique_email_constraint(self, repo):
        UserFactory.create(repo, email="same@example.com")
        with pytest.raises(sqlite3.IntegrityError):
            UserFactory.create(repo, email="same@example.com")


# =============================================================================
# TESTS — FACTORY PATTERN
# =============================================================================

class TestFactory:
    """Demonstrate factory pattern benefits."""

    def test_factory_defaults(self, repo):
        """Factory provides sensible defaults — test only specifies what matters."""
        user = UserFactory.create(repo)
        assert user.id is not None
        assert user.name.startswith("User")
        assert "@test.com" in user.email

    def test_factory_override(self, repo):
        """Override only the fields relevant to the test."""
        user = UserFactory.create(repo, name="Custom Name", active=False)
        assert user.name == "Custom Name"
        assert user.active is False

    def test_batch_creation(self, repo):
        """Create multiple records for list/filter tests."""
        users = UserFactory.create_batch(repo, 5)
        assert len(users) == 5
        assert repo.count() == 5


# =============================================================================
# TESTS — QUERY LOGIC
# =============================================================================

class TestQueryLogic:
    """Test filtering and query operations."""

    def test_find_active_users(self, repo):
        UserFactory.create(repo, active=True)
        UserFactory.create(repo, active=True)
        UserFactory.create(repo, active=False)

        active = repo.find_active()
        assert len(active) == 2
        assert all(u.active for u in active)

    def test_count(self, repo):
        assert repo.count() == 0
        UserFactory.create_batch(repo, 3)
        assert repo.count() == 3


# =============================================================================
# TESTS — ISOLATION VERIFICATION
# =============================================================================

class TestIsolation:
    """Verify that tests don't leak state to each other."""

    def test_first_test_creates_data(self, repo):
        """This test creates data..."""
        UserFactory.create_batch(repo, 10)
        assert repo.count() == 10

    def test_second_test_starts_clean(self, repo):
        """...but this test starts with a fresh database."""
        assert repo.count() == 0


# =============================================================================
# TRANSACTION ROLLBACK PATTERN (REFERENCE)
# =============================================================================

@contextmanager
def transactional_test(conn: sqlite3.Connection):
    """Context manager that rolls back after the test.
    In frameworks like SQLAlchemy, this is done with nested transactions
    (SAVEPOINT) so the test can commit within its scope."""
    conn.execute("BEGIN")
    try:
        yield conn
    finally:
        conn.rollback()


def test_transaction_rollback_pattern(db_conn):
    """Demonstrate the rollback pattern for frameworks that share connections."""
    repo = UserRepository(db_conn)
    repo.create_table()

    with transactional_test(db_conn) as conn:
        repo_in_tx = UserRepository(conn)
        repo_in_tx.insert(User(name="Temp", email="temp@test.com"))
        # Data exists within the transaction
        assert repo_in_tx.count() == 1

    # After rollback, data is gone
    assert repo.count() == 0


# =============================================================================
# RUNNING THIS FILE
# =============================================================================
# pytest 16_database_testing.py -v

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
