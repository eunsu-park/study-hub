# 11. Testing & Quality Assurance

**Previous**: [Performance Optimization](./10_Performance_Optimization.md) | **Next**: [Packaging & Distribution](./12_Packaging_and_Distribution.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Write and organize tests using the pytest framework, including test discovery conventions and execution options
2. Implement fixtures for test setup and teardown with appropriate scope management
3. Apply parametrized testing to cover multiple input scenarios with minimal code duplication
4. Use `unittest.mock` and `pytest-mock` to isolate units under test from external dependencies
5. Measure code coverage with `pytest-cov` and configure coverage thresholds and exclusions
6. Distinguish between the Arrange-Act-Assert and Given-When-Then test patterns
7. Apply test markers to categorize, skip, and selectively run test subsets

---

Automated testing is the single most effective practice for maintaining software quality over time. Without tests, every refactor is a gamble and every deployment is a source of anxiety. A well-structured test suite not only catches regressions early but also serves as living documentation of how your code is intended to behave.

## Table of Contents
1. [pytest Basics](#1-pytest-basics)
2. [Fixtures](#2-fixtures)
3. [Parametrize](#3-parametrize)
4. [Mocking](#4-mocking)
5. [Coverage](#5-coverage)
6. [Test Patterns](#6-test-patterns)
7. [Practice Problems](#7-practice-problems)

---

## 1. pytest Basics

### 1.1 pytest Installation and Structure

```bash
# Installation
pip install pytest pytest-cov pytest-mock pytest-asyncio

# Project structure
myproject/
├── src/
│   └── mypackage/
│       ├── __init__.py
│       ├── calculator.py
│       └── utils.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py          # Shared fixtures
│   ├── test_calculator.py
│   └── test_utils.py
├── pyproject.toml
└── pytest.ini               # Or in pyproject.toml
```

### 1.2 Writing Basic Tests

```python
# src/mypackage/calculator.py
class Calculator:
    def add(self, a: float, b: float) -> float:
        return a + b

    def divide(self, a: float, b: float) -> float:
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

    def is_positive(self, n: float) -> bool:
        return n > 0
```

```python
# tests/test_calculator.py
import pytest
from mypackage.calculator import Calculator


class TestCalculator:
    """Calculator class tests"""

    def setup_method(self):
        """Run before each test method"""
        self.calc = Calculator()

    def test_add_positive_numbers(self):
        """Test adding positive numbers"""
        result = self.calc.add(2, 3)
        assert result == 5

    def test_add_negative_numbers(self):
        """Test adding negative numbers"""
        result = self.calc.add(-2, -3)
        assert result == -5

    def test_divide_normal(self):
        """Normal division"""
        result = self.calc.divide(10, 2)
        assert result == 5.0

    def test_divide_by_zero_raises_error(self):
        """Test division by zero exception"""
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            self.calc.divide(10, 0)

    def test_is_positive_true(self):
        assert self.calc.is_positive(5) is True

    def test_is_positive_false(self):
        assert self.calc.is_positive(-5) is False
```

### 1.3 pytest Execution Options

```bash
# Basic execution
pytest

# Verbose output
pytest -v

# Specific file/directory
pytest tests/test_calculator.py
pytest tests/

# Specific test function
pytest tests/test_calculator.py::TestCalculator::test_add_positive_numbers

# Filter by keyword
pytest -k "add"      # Only tests with 'add'
pytest -k "not slow" # Tests without 'slow'

# Stop at first failure
pytest -x

# Rerun last failed tests only
pytest --lf

# Start with failed tests
pytest --ff

# Parallel execution (needs pytest-xdist)
pip install pytest-xdist
pytest -n auto
```

### 1.4 Configuration File

```ini
# pytest.ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
markers =
    slow: marks tests as slow
    integration: marks tests as integration tests
```

```toml
# pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --tb=short"
markers = [
    "slow: marks tests as slow",
    "integration: marks tests as integration tests",
]
```

---

## 2. Fixtures

### 2.1 Basic Fixtures

```python
# tests/conftest.py
import pytest
from mypackage.calculator import Calculator
from mypackage.database import Database


@pytest.fixture
def calculator():
    """Provide Calculator instance"""
    return Calculator()


@pytest.fixture
def sample_data():
    """Sample test data"""
    return {
        "users": [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ],
        "products": [
            {"id": 1, "price": 100},
            {"id": 2, "price": 200},
        ]
    }
```

```python
# tests/test_with_fixtures.py
def test_add_with_fixture(calculator):
    """Use fixture as argument"""
    assert calculator.add(2, 3) == 5


def test_sample_data(sample_data):
    """Sample data fixture"""
    assert len(sample_data["users"]) == 2
```

### 2.2 Fixture Scope

```python
@pytest.fixture(scope="function")  # Default: per test function
def func_fixture():
    print("Setup function fixture")
    yield "function"
    print("Teardown function fixture")


@pytest.fixture(scope="class")  # Once per test class
def class_fixture():
    print("Setup class fixture")
    yield "class"
    print("Teardown class fixture")


@pytest.fixture(scope="module")  # Once per module
def module_fixture():
    print("Setup module fixture")
    yield "module"
    print("Teardown module fixture")


@pytest.fixture(scope="session")  # Once per test session
def session_fixture():
    print("Setup session fixture")
    yield "session"
    print("Teardown session fixture")
```

### 2.3 Setup/Teardown Pattern

```python
@pytest.fixture
def database():
    """Database connection fixture with cleanup"""
    # Setup
    db = Database()
    db.connect()
    db.create_tables()

    yield db  # Provide to test

    # Teardown (runs automatically after test)
    db.drop_tables()
    db.disconnect()


@pytest.fixture
def temp_file(tmp_path):
    """Create temporary file (uses pytest built-in tmp_path)"""
    file_path = tmp_path / "test.txt"
    file_path.write_text("test content")
    yield file_path
    # tmp_path is automatically cleaned up
```

### 2.4 Factory Fixtures

```python
@pytest.fixture
def user_factory():
    """User object creation factory"""
    created_users = []

    def _create_user(name: str, age: int = 20):
        user = {"name": name, "age": age, "id": len(created_users) + 1}
        created_users.append(user)
        return user

    yield _create_user

    # Teardown: clean up all created users
    created_users.clear()


def test_with_factory(user_factory):
    user1 = user_factory("Alice")
    user2 = user_factory("Bob", age=30)

    assert user1["name"] == "Alice"
    assert user2["age"] == 30
```

### 2.5 Fixture Dependencies

```python
@pytest.fixture
def config():
    return {"db_url": "sqlite:///test.db"}


@pytest.fixture
def database(config):  # Depends on config fixture
    db = Database(config["db_url"])
    db.connect()
    yield db
    db.disconnect()


@pytest.fixture
def user_service(database):  # Depends on database fixture
    return UserService(database)
```

---

## 3. Parametrize

### 3.1 Basic Parametrization

```python
@pytest.mark.parametrize("a, b, expected", [
    (2, 3, 5),
    (0, 0, 0),
    (-1, 1, 0),
    (100, 200, 300),
])
def test_add_parametrized(calculator, a, b, expected):
    assert calculator.add(a, b) == expected


@pytest.mark.parametrize("dividend, divisor, expected", [
    (10, 2, 5.0),
    (9, 3, 3.0),
    (7, 2, 3.5),
])
def test_divide_parametrized(calculator, dividend, divisor, expected):
    assert calculator.divide(dividend, divisor) == expected
```

### 3.2 Specifying IDs

```python
@pytest.mark.parametrize("input_val, expected", [
    pytest.param(1, True, id="positive"),
    pytest.param(0, False, id="zero"),
    pytest.param(-1, False, id="negative"),
])
def test_is_positive_with_ids(calculator, input_val, expected):
    assert calculator.is_positive(input_val) == expected
```

### 3.3 Multiple Parametrization

```python
@pytest.mark.parametrize("x", [1, 2])
@pytest.mark.parametrize("y", [10, 20])
def test_multiply(x, y):
    """Creates 4 test cases: (1,10), (1,20), (2,10), (2,20)"""
    result = x * y
    assert result == x * y
```

### 3.4 Exception Test Parametrization

```python
@pytest.mark.parametrize("dividend, divisor, error_match", [
    (10, 0, "Cannot divide by zero"),
    (5, 0, "Cannot divide by zero"),
])
def test_divide_errors(calculator, dividend, divisor, error_match):
    with pytest.raises(ValueError, match=error_match):
        calculator.divide(dividend, divisor)
```

---

## 4. Mocking

### 4.1 pytest-mock Basics

```python
# src/mypackage/weather.py
import requests


class WeatherService:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.weather.com"

    def get_temperature(self, city: str) -> float:
        response = requests.get(
            f"{self.base_url}/current",
            params={"city": city, "key": self.api_key}
        )
        response.raise_for_status()
        return response.json()["temperature"]
```

```python
# tests/test_weather.py
import pytest
from unittest.mock import Mock, patch
from mypackage.weather import WeatherService


def test_get_temperature_with_mock(mocker):
    """Using mocker fixture (pytest-mock)"""
    # Mock setup
    mock_response = Mock()
    mock_response.json.return_value = {"temperature": 25.5}
    mock_response.raise_for_status = Mock()

    mocker.patch("mypackage.weather.requests.get", return_value=mock_response)

    # Test
    service = WeatherService("fake-api-key")
    temp = service.get_temperature("Seoul")

    assert temp == 25.5


def test_get_temperature_with_patch():
    """Using @patch decorator"""
    with patch("mypackage.weather.requests.get") as mock_get:
        mock_get.return_value.json.return_value = {"temperature": 30.0}
        mock_get.return_value.raise_for_status = Mock()

        service = WeatherService("fake-key")
        temp = service.get_temperature("Tokyo")

        assert temp == 30.0
        mock_get.assert_called_once()
```

### 4.2 Mock Object Details

```python
from unittest.mock import Mock, MagicMock, PropertyMock


def test_mock_methods():
    mock = Mock()

    # Set return value
    mock.method.return_value = 42
    assert mock.method() == 42

    # side_effect for raising exception
    mock.error_method.side_effect = ValueError("Error!")
    with pytest.raises(ValueError):
        mock.error_method()

    # side_effect for sequential returns
    mock.sequence.side_effect = [1, 2, 3]
    assert mock.sequence() == 1
    assert mock.sequence() == 2
    assert mock.sequence() == 3


def test_mock_assertions():
    mock = Mock()

    mock.method("arg1", kwarg="value")

    # Call verification
    mock.method.assert_called()
    mock.method.assert_called_once()
    mock.method.assert_called_with("arg1", kwarg="value")
    mock.method.assert_called_once_with("arg1", kwarg="value")

    # Call count
    assert mock.method.call_count == 1


def test_magic_mock():
    """MagicMock: supports magic methods"""
    mock = MagicMock()
    mock.__len__.return_value = 5
    assert len(mock) == 5

    mock.__getitem__.return_value = "item"
    assert mock[0] == "item"
```

### 4.3 Mocking Async Code

```python
import pytest
from unittest.mock import AsyncMock


@pytest.mark.asyncio
async def test_async_mock():
    # Using AsyncMock
    mock_async = AsyncMock(return_value={"data": "value"})

    result = await mock_async()
    assert result == {"data": "value"}


@pytest.mark.asyncio
async def test_async_service(mocker):
    from mypackage.async_service import AsyncDataService

    mock_fetch = mocker.patch.object(
        AsyncDataService,
        "fetch_data",
        new_callable=AsyncMock,
        return_value={"status": "ok"}
    )

    service = AsyncDataService()
    result = await service.fetch_data()

    assert result["status"] == "ok"
    mock_fetch.assert_awaited_once()
```

### 4.4 Mocking Patterns

```python
# Mock environment variables
def test_with_env_var(mocker):
    mocker.patch.dict("os.environ", {"API_KEY": "test-key"})
    import os
    assert os.environ["API_KEY"] == "test-key"


# Mock time
def test_with_frozen_time(mocker):
    from datetime import datetime
    mock_now = datetime(2024, 1, 15, 12, 0, 0)
    mocker.patch("mypackage.service.datetime")
    mocker.patch("mypackage.service.datetime.now", return_value=mock_now)


# Mock filesystem
def test_file_read(mocker):
    mock_open = mocker.mock_open(read_data="file content")
    mocker.patch("builtins.open", mock_open)

    with open("dummy.txt") as f:
        content = f.read()

    assert content == "file content"
```

---

## 5. Coverage

### 5.1 pytest-cov Setup

```bash
# Installation
pip install pytest-cov

# Run with coverage
pytest --cov=mypackage tests/

# Generate HTML report
pytest --cov=mypackage --cov-report=html tests/

# Terminal output with missing lines
pytest --cov=mypackage --cov-report=term-missing tests/
```

### 5.2 Configuration File

```ini
# .coveragerc
[run]
source = src/mypackage
omit =
    */tests/*
    */__pycache__/*
    */migrations/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise NotImplementedError
    if TYPE_CHECKING:

fail_under = 80

[html]
directory = htmlcov
```

```toml
# pyproject.toml
[tool.coverage.run]
source = ["src/mypackage"]
omit = ["*/tests/*", "*/__pycache__/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
]
fail_under = 80
```

### 5.3 Excluding from Coverage

```python
def debug_function():  # pragma: no cover
    """Debug-only function - exclude from coverage"""
    print("Debug info")


def main():
    if __name__ == "__main__":  # pragma: no cover
        run_app()
```

---

## 6. Test Patterns

### 6.1 Arrange-Act-Assert (AAA)

```python
def test_user_registration():
    # Arrange: setup
    user_data = {"name": "Alice", "email": "alice@example.com"}
    service = UserService()

    # Act: execute
    user = service.register(user_data)

    # Assert: verify
    assert user.name == "Alice"
    assert user.email == "alice@example.com"
    assert user.id is not None
```

### 6.2 Given-When-Then (BDD Style)

```python
def test_withdraw_sufficient_balance():
    """
    Given: Account has 1000 won
    When: Withdraw 500 won
    Then: Balance becomes 500 won
    """
    # Given
    account = Account(balance=1000)

    # When
    account.withdraw(500)

    # Then
    assert account.balance == 500
```

### 6.3 Test Markers

```python
import pytest


@pytest.mark.slow
def test_large_data_processing():
    """Slow test"""
    pass


@pytest.mark.integration
def test_database_connection():
    """Integration test"""
    pass


@pytest.mark.skip(reason="Feature not implemented")
def test_future_feature():
    pass


@pytest.mark.skipif(
    condition=True,
    reason="Skip under specific conditions"
)
def test_conditional():
    pass


@pytest.mark.xfail(reason="Known bug")
def test_known_bug():
    assert False  # Test passes even if it fails


# Run: pytest -m "not slow"  # Exclude slow
# Run: pytest -m "integration"  # Only integration
```

### 6.4 Test Grouping

```python
class TestUserCreation:
    """User creation test group"""

    def test_create_with_valid_data(self):
        pass

    def test_create_with_invalid_email(self):
        pass

    def test_create_duplicate_user(self):
        pass


class TestUserAuthentication:
    """Authentication test group"""

    def test_login_success(self):
        pass

    def test_login_wrong_password(self):
        pass
```

---

## 7. Practice Problems

### Exercise 1: Write Basic Tests
Write tests for the following function.

```python
# Test target
def validate_email(email: str) -> bool:
    """Email validation"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


# Write tests
@pytest.mark.parametrize("email, expected", [
    ("user@example.com", True),
    ("invalid-email", False),
    ("user@domain", False),
    ("user.name+tag@example.co.kr", True),
    ("", False),
])
def test_validate_email(email, expected):
    assert validate_email(email) == expected
```

### Exercise 2: Use Fixtures
Write and use a database connection fixture.

```python
# Sample solution
@pytest.fixture
def db_connection():
    """In-memory test database"""
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    yield conn
    conn.close()


def test_insert_user(db_connection):
    db_connection.execute("INSERT INTO users (name) VALUES (?)", ("Alice",))
    result = db_connection.execute("SELECT name FROM users").fetchone()
    assert result[0] == "Alice"
```

### Exercise 3: Mocking Practice
Test a function that calls an external API using mocking.

```python
# Test target
class PaymentService:
    def process_payment(self, amount: float) -> dict:
        # Actually calls external payment API
        import requests
        response = requests.post(
            "https://payment.api/charge",
            json={"amount": amount}
        )
        return response.json()


# Test (sample solution)
def test_process_payment(mocker):
    mock_post = mocker.patch("requests.post")
    mock_post.return_value.json.return_value = {
        "status": "success",
        "transaction_id": "txn_123"
    }

    service = PaymentService()
    result = service.process_payment(100.0)

    assert result["status"] == "success"
    mock_post.assert_called_once()
```

---

## References
- [pytest Documentation](https://docs.pytest.org/)
- [pytest-mock](https://pytest-mock.readthedocs.io/)
- [pytest-cov](https://pytest-cov.readthedocs.io/)
- [unittest.mock](https://docs.python.org/3/library/unittest.mock.html)

---

**Previous**: [Performance Optimization](./10_Performance_Optimization.md) | **Next**: [Packaging & Distribution](./12_Packaging_and_Distribution.md)
