#!/bin/bash
# Exercises for Lesson 03: Test Fixtures and Setup
# Topic: Testing_and_QA
# Solutions to practice problems from the lesson.

# === Exercise 1: Fixture Scopes ===
# Problem: Create fixtures with different scopes for a database-backed application.
# Show how scope affects fixture lifecycle.
exercise_1() {
    echo "=== Exercise 1: Fixture Scopes ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import pytest
import sqlite3

# SESSION scope: created once for the entire test run.
# Use for expensive setup like database connections.
@pytest.fixture(scope="session")
def db_engine():
    """Simulate creating a database engine (expensive operation)."""
    print("\n[SESSION] Creating database engine")
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    yield conn
    print("\n[SESSION] Closing database engine")
    conn.close()

# MODULE scope: created once per test file.
# Use for shared state within a module.
@pytest.fixture(scope="module")
def schema(db_engine):
    """Ensure schema exists. Created once per module."""
    print("\n[MODULE] Verifying schema")
    # Schema already created by db_engine; this fixture
    # demonstrates module-level dependency on session fixture.
    return db_engine

# FUNCTION scope (default): created for each test function.
# Use for data that tests might mutate.
@pytest.fixture
def clean_db(schema):
    """Clean table before each test — prevents data leaks."""
    print("\n[FUNCTION] Cleaning database")
    schema.execute("DELETE FROM users")
    schema.commit()
    return schema

def test_insert_user(clean_db):
    clean_db.execute("INSERT INTO users (name) VALUES (?)", ("Alice",))
    clean_db.commit()
    row = clean_db.execute("SELECT name FROM users").fetchone()
    assert row[0] == "Alice"

def test_empty_after_clean(clean_db):
    """This test gets a clean DB even though test_insert_user added data."""
    count = clean_db.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    assert count == 0  # clean_db deleted all rows

# Run with: pytest -v --setup-show  to see fixture lifecycle
# Output shows:
#   SETUP S db_engine       (S = Session — once)
#   SETUP M schema          (M = Module — once per file)
#   SETUP F clean_db        (F = Function — before each test)
#   TEARDOWN F clean_db     (after each test)
SOLUTION
}

# === Exercise 2: Yield Fixtures with Cleanup ===
# Problem: Create a fixture that sets up a temporary directory with files
# and cleans up afterward, even if the test fails.
exercise_2() {
    echo "=== Exercise 2: Yield Fixtures with Cleanup ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import pytest
import tempfile
import os
from pathlib import Path

@pytest.fixture
def project_dir():
    """Create a temporary project directory with standard structure.
    Cleanup happens after yield, even if the test raises an exception."""
    # SETUP: create directory structure
    base = Path(tempfile.mkdtemp(prefix="test_project_"))
    (base / "src").mkdir()
    (base / "tests").mkdir()
    (base / "src" / "__init__.py").touch()
    (base / "src" / "main.py").write_text("def hello(): return 'world'")
    (base / "tests" / "test_main.py").write_text("def test_hello(): pass")
    (base / "README.md").write_text("# Test Project")

    yield base

    # TEARDOWN: clean up temp directory
    # This runs even if the test fails — like a finally block
    import shutil
    shutil.rmtree(base, ignore_errors=True)

def test_project_structure(project_dir):
    """Verify the fixture created the expected structure."""
    assert (project_dir / "src" / "main.py").exists()
    assert (project_dir / "tests" / "test_main.py").exists()
    assert (project_dir / "README.md").read_text() == "# Test Project"

def test_project_is_isolated(project_dir):
    """Each test gets its own directory — modifications are safe."""
    # Create extra files — won't affect other tests
    (project_dir / "extra.txt").write_text("temporary")
    assert (project_dir / "extra.txt").exists()

# After each test, the temp directory is deleted automatically.
# Verify with: pytest -v --setup-show
SOLUTION
}

# === Exercise 3: Factory Fixtures ===
# Problem: Create a factory fixture that generates user objects
# with customizable properties.
exercise_3() {
    echo "=== Exercise 3: Factory Fixtures ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import pytest
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class User:
    id: int
    username: str
    email: str
    is_active: bool = True
    roles: list = field(default_factory=lambda: ["viewer"])
    created_at: datetime = field(default_factory=datetime.utcnow)

@pytest.fixture
def make_user():
    """Factory fixture: returns a function that creates users.

    Benefits over static fixtures:
    - Tests specify only relevant fields (less noise)
    - Auto-incrementing IDs prevent collisions
    - Multiple users per test without multiple fixtures
    """
    _counter = 0

    def _factory(
        username=None,
        email=None,
        is_active=True,
        roles=None,
    ):
        nonlocal _counter
        _counter += 1
        uname = username or f"user_{_counter}"
        return User(
            id=_counter,
            username=uname,
            email=email or f"{uname}@test.com",
            is_active=is_active,
            roles=roles or ["viewer"],
        )

    return _factory

def test_create_default_user(make_user):
    user = make_user()
    assert user.username.startswith("user_")
    assert user.is_active is True
    assert user.roles == ["viewer"]

def test_create_admin(make_user):
    admin = make_user(username="admin", roles=["admin", "editor"])
    assert admin.username == "admin"
    assert "admin" in admin.roles

def test_create_multiple_unique_users(make_user):
    users = [make_user() for _ in range(5)]
    ids = [u.id for u in users]
    # All IDs are unique
    assert len(set(ids)) == 5

def test_inactive_user(make_user):
    user = make_user(is_active=False)
    assert user.is_active is False
    # Email is auto-generated from username
    assert "@test.com" in user.email
SOLUTION
}

# === Exercise 4: Conftest Patterns ===
# Problem: Design a conftest.py hierarchy for a multi-layer application
# with shared and layer-specific fixtures.
exercise_4() {
    echo "=== Exercise 4: Conftest Patterns ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Project structure:
# tests/
# ├── conftest.py            <- Root conftest (shared by ALL tests)
# ├── unit/
# │   ├── conftest.py        <- Unit-specific fixtures
# │   └── test_services.py
# └── integration/
#     ├── conftest.py        <- Integration-specific fixtures
#     └── test_api.py

# === tests/conftest.py ===
import pytest

@pytest.fixture(scope="session")
def app_config():
    """Configuration shared by all test types."""
    return {
        "TESTING": True,
        "DATABASE_URL": "sqlite:///:memory:",
        "SECRET_KEY": "test-secret",
    }

@pytest.fixture
def sample_data():
    """Basic test data available everywhere."""
    return {
        "users": [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ]
    }

# === tests/unit/conftest.py ===
import pytest
from unittest.mock import Mock

@pytest.fixture
def mock_db():
    """Unit tests use a mock database — no real DB needed."""
    db = Mock()
    db.query.return_value = []
    db.save.return_value = True
    return db

@pytest.fixture
def mock_email_service():
    """Unit tests mock external services."""
    service = Mock()
    service.send.return_value = {"status": "sent"}
    return service

# === tests/integration/conftest.py ===
import pytest

@pytest.fixture(scope="module")
def test_db(app_config):
    """Integration tests use a real (test) database."""
    import sqlite3
    conn = sqlite3.connect(app_config["DATABASE_URL"].replace("sqlite:///", ""))
    conn.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER, name TEXT)")
    yield conn
    conn.close()

@pytest.fixture
def seeded_db(test_db):
    """Pre-populate database for integration tests."""
    test_db.execute("DELETE FROM users")
    test_db.execute("INSERT INTO users VALUES (1, 'Alice')")
    test_db.execute("INSERT INTO users VALUES (2, 'Bob')")
    test_db.commit()
    return test_db

# Key insight: conftest.py files form a hierarchy.
# - Inner conftest.py can use fixtures from outer conftest.py
# - Inner fixtures override outer ones with the same name
# - No imports needed — pytest auto-discovers conftest.py files
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 03: Test Fixtures and Setup"
echo "==========================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
