#!/usr/bin/env python3
"""Example: Fixtures and Conftest

Demonstrates pytest fixtures: scope, yield cleanup, factory fixtures,
parameterized fixtures, autouse, and conftest.py patterns.
Related lesson: 03_Test_Fixtures_and_Setup.md
"""

# =============================================================================
# WHY FIXTURES?
# Fixtures solve the "test setup" problem without setUp/tearDown boilerplate.
# They use dependency injection: declare what you need as a parameter, and
# pytest provides it. This makes tests declarative and composable.
# =============================================================================

import pytest
import tempfile
import json
import sqlite3
from pathlib import Path
from dataclasses import dataclass, field
from typing import List


# =============================================================================
# BASIC FIXTURES
# =============================================================================

@pytest.fixture
def sample_user():
    """A simple fixture that returns test data.
    Any test that declares 'sample_user' as a parameter receives this dict.
    This is better than duplicating the dict in every test."""
    return {
        "username": "testuser",
        "email": "test@example.com",
        "age": 30,
        "roles": ["user", "editor"],
    }


def test_user_has_email(sample_user):
    """The fixture is injected by name — no manual setup needed."""
    assert "@" in sample_user["email"]


def test_user_has_roles(sample_user):
    """Each test gets a FRESH copy of the fixture (function scope by default),
    so mutations in one test never leak into another."""
    sample_user["roles"].append("admin")
    assert "admin" in sample_user["roles"]


def test_user_roles_unchanged(sample_user):
    """Proves isolation: the 'admin' role added above is NOT present here."""
    assert "admin" not in sample_user["roles"]


# =============================================================================
# YIELD FIXTURES — SETUP AND TEARDOWN
# =============================================================================
# Code before yield is setup; code after yield is teardown.
# This replaces setUp/tearDown with a cleaner, more Pythonic pattern.
# Teardown ALWAYS runs, even if the test fails (like a finally block).

@pytest.fixture
def temp_json_file():
    """Create a temporary JSON file, yield its path, then clean up.
    This pattern ensures test artifacts never accumulate on disk."""
    data = {"key": "value", "numbers": [1, 2, 3]}
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    )
    json.dump(data, tmp)
    tmp.close()

    yield Path(tmp.name)  # <-- test runs here

    # Teardown: clean up the temp file
    Path(tmp.name).unlink(missing_ok=True)


def test_read_json(temp_json_file):
    """The fixture handles creation AND cleanup — the test only tests."""
    data = json.loads(temp_json_file.read_text())
    assert data["key"] == "value"
    assert len(data["numbers"]) == 3


# =============================================================================
# FIXTURE SCOPES
# =============================================================================
# Scope controls how often the fixture is created/destroyed:
#   function (default) — once per test function
#   class    — once per test class
#   module   — once per test file
#   session  — once per entire test run
#
# Use broader scopes for expensive setup (DB connections, servers).
# Rule of thumb: use the narrowest scope that gives acceptable performance.

@pytest.fixture(scope="module")
def database_connection():
    """Expensive setup: create DB connection once per module.
    All tests in this file share the same connection."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL
        )
    """)
    conn.commit()

    yield conn

    conn.close()


@pytest.fixture
def populated_db(database_connection):
    """Fixtures can depend on other fixtures — pytest resolves the graph.
    This fixture uses the module-scoped connection but resets data per test."""
    database_connection.execute("DELETE FROM users")
    database_connection.execute(
        "INSERT INTO users (name, email) VALUES (?, ?)",
        ("Alice", "alice@example.com")
    )
    database_connection.execute(
        "INSERT INTO users (name, email) VALUES (?, ?)",
        ("Bob", "bob@example.com")
    )
    database_connection.commit()
    return database_connection


def test_count_users(populated_db):
    cursor = populated_db.execute("SELECT COUNT(*) FROM users")
    assert cursor.fetchone()[0] == 2


def test_find_user_by_email(populated_db):
    cursor = populated_db.execute(
        "SELECT name FROM users WHERE email = ?",
        ("alice@example.com",)
    )
    row = cursor.fetchone()
    assert row is not None
    assert row[0] == "Alice"


# =============================================================================
# FACTORY FIXTURES
# =============================================================================
# When tests need different variations of test data, a factory fixture
# returns a callable that creates data on demand with custom overrides.

@dataclass
class Product:
    name: str
    price: float
    tags: List[str] = field(default_factory=list)
    in_stock: bool = True


@pytest.fixture
def make_product():
    """Factory fixture: returns a function that creates products.
    Tests call it with only the fields they care about — sensible defaults
    handle the rest, reducing noise in test code."""
    _counter = 0

    def _factory(name=None, price=9.99, tags=None, in_stock=True):
        nonlocal _counter
        _counter += 1
        return Product(
            name=name or f"Product-{_counter}",
            price=price,
            tags=tags or [],
            in_stock=in_stock,
        )

    return _factory


def test_default_product(make_product):
    """Factory provides sensible defaults — test focuses on what matters."""
    product = make_product()
    assert product.price == 9.99
    assert product.in_stock is True


def test_discounted_product(make_product):
    """Override only the fields relevant to this test scenario."""
    product = make_product(name="Sale Item", price=4.99)
    assert product.price < 5.00
    assert product.name == "Sale Item"


def test_multiple_products(make_product):
    """Create as many instances as needed — each gets a unique name."""
    products = [make_product(price=i * 10) for i in range(5)]
    assert len(products) == 5
    assert all(p.name.startswith("Product-") for p in products)


# =============================================================================
# PARAMETERIZED FIXTURES
# =============================================================================
# Like @pytest.mark.parametrize but for fixtures.
# Every test using this fixture runs once per parameter value.

@pytest.fixture(params=["sqlite", "json", "csv"])
def storage_backend(request):
    """Each test using this fixture runs 3 times — once per backend.
    request.param gives access to the current parameter value."""
    backend = request.param
    # In real code, you would initialize the actual backend here
    yield backend
    # Teardown per backend if needed


def test_storage_backend_name(storage_backend):
    """This test runs 3 times: with 'sqlite', 'json', and 'csv'."""
    assert storage_backend in ("sqlite", "json", "csv")


# =============================================================================
# AUTOUSE FIXTURES
# =============================================================================
# autouse=True applies the fixture to EVERY test in scope without explicitly
# requesting it. Use sparingly — implicit behavior can confuse readers.
# Good use cases: resetting global state, timing all tests.

@pytest.fixture(autouse=True)
def reset_environment(monkeypatch):
    """Automatically clear a hypothetical environment variable for every test.
    monkeypatch is a built-in fixture that auto-reverts changes."""
    monkeypatch.delenv("DEBUG_MODE", raising=False)


# =============================================================================
# BUILT-IN FIXTURES
# =============================================================================

def test_tmp_path(tmp_path):
    """tmp_path (built-in): provides a unique temporary directory per test.
    Automatically cleaned up after the test session."""
    file = tmp_path / "data.txt"
    file.write_text("hello from test")
    assert file.read_text() == "hello from test"
    assert file.parent == tmp_path


def test_capsys(capsys):
    """capsys (built-in): captures stdout/stderr output.
    Essential for testing CLI tools and print-based interfaces."""
    print("captured output")
    captured = capsys.readouterr()
    assert "captured" in captured.out
    assert captured.err == ""


def test_monkeypatch(monkeypatch):
    """monkeypatch (built-in): temporarily modify objects, dicts, env vars.
    Changes are automatically reverted after the test."""
    monkeypatch.setenv("API_KEY", "test-key-123")
    import os
    assert os.environ["API_KEY"] == "test-key-123"


# =============================================================================
# CONFTEST.PY PATTERN
# =============================================================================
# In a real project, shared fixtures go in conftest.py files:
#
#   tests/
#   ├── conftest.py          <-- session/module fixtures shared by all tests
#   ├── unit/
#   │   ├── conftest.py      <-- fixtures specific to unit tests
#   │   └── test_models.py
#   └── integration/
#       ├── conftest.py      <-- fixtures specific to integration tests
#       └── test_api.py
#
# Fixtures in conftest.py are auto-discovered — no import needed.
# Closer conftest.py files override more distant ones (like CSS specificity).
#
# Example conftest.py content:
#
#   @pytest.fixture(scope="session")
#   def app():
#       """Create application instance once for the entire test session."""
#       from myapp import create_app
#       app = create_app(testing=True)
#       yield app
#
#   @pytest.fixture
#   def client(app):
#       """Create a test client for each test function."""
#       return app.test_client()


# =============================================================================
# RUNNING THIS FILE
# =============================================================================
# pytest 02_fixtures_and_conftest.py -v
# pytest 02_fixtures_and_conftest.py -v --setup-show  # see fixture setup/teardown

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--setup-show"])
