# Mocking and Patching

**Previous**: [Test Fixtures and Parameterization](./03_Test_Fixtures_and_Parameterization.md) | **Next**: [Test Coverage and Quality](./05_Test_Coverage_and_Quality.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Use `unittest.mock.Mock` and `MagicMock` to create test doubles
2. Patch functions, methods, and objects with `patch` and `patch.object`
3. Control mock behavior with `side_effect` and `return_value`
4. Use `spec` to create type-safe mocks that catch interface errors
5. Apply pytest's `monkeypatch` for lightweight patching
6. Recognize when mocking is appropriate and when it is harmful

---

## Why Mock?

Unit tests should run in isolation, without external dependencies like databases, APIs, or file systems. Mocking lets you replace those dependencies with controlled stand-ins so you can:

- **Test logic in isolation** without spinning up infrastructure
- **Simulate edge cases** like network timeouts or API errors that are hard to reproduce
- **Speed up tests** by eliminating I/O
- **Verify interactions** — confirm your code calls dependencies correctly

But mocking has a dark side: over-mocking creates tests that pass even when the real code is broken. The key is knowing *when* to mock and *what* to mock.

---

## Mock Basics

### Mock and MagicMock

`Mock` creates an object that accepts any attribute access or method call. `MagicMock` extends `Mock` with default implementations of magic methods (`__len__`, `__iter__`, etc.).

```python
from unittest.mock import Mock, MagicMock

# Mock accepts any attribute or method call
m = Mock()
m.some_method(1, 2, 3)           # Does not raise
m.nested.deeply.nested.attr      # Does not raise
print(m.some_method.called)      # True
print(m.some_method.call_args)   # call(1, 2, 3)

# MagicMock supports magic methods
mm = MagicMock()
mm.__len__.return_value = 5
print(len(mm))                   # 5
mm.__getitem__.return_value = "hello"
print(mm[0])                     # "hello"
```

### Configuring Return Values

```python
from unittest.mock import Mock

# Simple return value
api_client = Mock()
api_client.get_user.return_value = {"id": 1, "name": "Alice"}

result = api_client.get_user(user_id=1)
assert result == {"id": 1, "name": "Alice"}

# Nested return values
api_client.get_user.return_value.get.return_value = "Alice"
```

### Asserting Calls

```python
from unittest.mock import Mock, call

notifier = Mock()

# Call the mock
notifier.send("alice@test.com", "Hello!")
notifier.send("bob@test.com", "Hi!")

# Assert it was called
notifier.send.assert_called()                    # At least once
notifier.send.assert_called_once()                # FAILS — called twice
notifier.send.assert_any_call("alice@test.com", "Hello!")  # Any call matches

# Check call count
assert notifier.send.call_count == 2

# Check all calls in order
notifier.send.assert_has_calls([
    call("alice@test.com", "Hello!"),
    call("bob@test.com", "Hi!"),
])

# Check the most recent call
notifier.send.assert_called_with("bob@test.com", "Hi!")
```

---

## side_effect: Dynamic Mock Behavior

`side_effect` lets you make a mock raise exceptions, return different values on successive calls, or run custom logic.

### Raising Exceptions

```python
from unittest.mock import Mock
import pytest

http_client = Mock()
http_client.get.side_effect = ConnectionError("Network is unreachable")

with pytest.raises(ConnectionError, match="unreachable"):
    http_client.get("https://api.example.com/data")
```

### Sequential Return Values

```python
from unittest.mock import Mock

# Return different values on each call
token_generator = Mock()
token_generator.generate.side_effect = ["token-aaa", "token-bbb", "token-ccc"]

assert token_generator.generate() == "token-aaa"
assert token_generator.generate() == "token-bbb"
assert token_generator.generate() == "token-ccc"
```

### Custom Logic

```python
from unittest.mock import Mock

def fake_lookup(user_id):
    users = {1: "Alice", 2: "Bob"}
    if user_id not in users:
        raise KeyError(f"User {user_id} not found")
    return users[user_id]

db = Mock()
db.find_user.side_effect = fake_lookup

assert db.find_user(1) == "Alice"
assert db.find_user(2) == "Bob"
with pytest.raises(KeyError):
    db.find_user(999)
```

---

## patch: Replacing Real Objects

`patch` temporarily replaces an object in the module where it is *used* (not where it is defined). This is the most common source of patching bugs.

### The Import Path Rule

```python
# myapp/service.py
from myapp.client import ApiClient  # ApiClient is imported HERE

class UserService:
    def __init__(self):
        self.client = ApiClient()

    def get_user_name(self, user_id):
        data = self.client.fetch(f"/users/{user_id}")
        return data["name"]
```

```python
# test_service.py
from unittest.mock import patch, MagicMock

# CORRECT: patch where ApiClient is USED (myapp.service)
@patch("myapp.service.ApiClient")
def test_get_user_name(MockApiClient):
    # MockApiClient is now the mock class
    mock_instance = MockApiClient.return_value
    mock_instance.fetch.return_value = {"name": "Alice"}

    service = UserService()
    assert service.get_user_name(1) == "Alice"
    mock_instance.fetch.assert_called_once_with("/users/1")


# WRONG: patching where ApiClient is DEFINED
# @patch("myapp.client.ApiClient")  # This would NOT affect myapp.service
```

### patch as Context Manager

```python
from unittest.mock import patch

def test_with_context_manager():
    with patch("myapp.service.ApiClient") as MockApiClient:
        mock_instance = MockApiClient.return_value
        mock_instance.fetch.return_value = {"name": "Bob"}

        service = UserService()
        assert service.get_user_name(2) == "Bob"
    # Patch is automatically removed after the with block
```

### patch.object

Patch a specific attribute of an object. Useful for patching methods on existing instances.

```python
from unittest.mock import patch

class EmailSender:
    def send(self, to: str, body: str) -> bool:
        # Real implementation connects to SMTP server
        ...

def test_notification_sends_email():
    sender = EmailSender()

    with patch.object(sender, "send", return_value=True) as mock_send:
        result = sender.send("alice@test.com", "Hello!")
        assert result is True
        mock_send.assert_called_once_with("alice@test.com", "Hello!")
```

---

## spec: Type-Safe Mocks

Without `spec`, mocks accept any attribute access silently. This means typos in your test go undetected:

```python
from unittest.mock import Mock

# Without spec — typos are invisible
mock_user = Mock()
mock_user.nmae = "Alice"  # Typo! Should be 'name'. No error raised.
```

With `spec`, the mock validates attribute access against a real class:

```python
from unittest.mock import Mock

class User:
    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email

    def deactivate(self):
        pass

# With spec — typos raise AttributeError
mock_user = Mock(spec=User)
mock_user.name = "Alice"       # OK
mock_user.nmae = "Alice"       # AttributeError: Mock object has no attribute 'nmae'
mock_user.deactivate()         # OK
mock_user.nonexistent()        # AttributeError
```

**Best practice**: Always use `spec` (or `spec_set` for even stricter checking) when mocking concrete classes.

---

## pytest monkeypatch

pytest's `monkeypatch` fixture provides a simpler alternative for common patching scenarios. Unlike `unittest.mock.patch`, it automatically undoes changes after each test.

### Patching Environment Variables

```python
def get_database_url() -> str:
    import os
    return os.environ.get("DATABASE_URL", "sqlite:///default.db")


def test_database_url_from_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/test")
    assert get_database_url() == "postgresql://localhost/test"

def test_database_url_default(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert get_database_url() == "sqlite:///default.db"
```

### Patching Attributes

```python
import myapp.config as config

def test_debug_mode(monkeypatch):
    monkeypatch.setattr(config, "DEBUG", True)
    assert config.DEBUG is True

def test_custom_timeout(monkeypatch):
    monkeypatch.setattr(config, "REQUEST_TIMEOUT", 5)
    assert config.REQUEST_TIMEOUT == 5
```

### Patching Functions

```python
from datetime import datetime

def get_greeting() -> str:
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning"
    elif hour < 18:
        return "Good afternoon"
    else:
        return "Good evening"


def test_morning_greeting(monkeypatch):
    class FakeDatetime:
        @staticmethod
        def now():
            return datetime(2024, 1, 1, 9, 0, 0)  # 9 AM

    # Patch datetime in the module where get_greeting is defined
    import myapp.greetings
    monkeypatch.setattr(myapp.greetings, "datetime", FakeDatetime)
    assert get_greeting() == "Good morning"
```

---

## Dependency Injection for Testability

The best way to make code testable is to *design* it for testing. Dependency injection eliminates the need for patching in many cases.

### Before: Hard to Test

```python
# Hard to test — creates its own dependencies
class OrderProcessor:
    def __init__(self):
        self.db = PostgresDatabase()       # Hardcoded
        self.mailer = SmtpMailer()         # Hardcoded
        self.logger = FileLogger("/var/log/orders.log")  # Hardcoded

    def process(self, order):
        self.db.save(order)
        self.mailer.send(order.customer_email, "Order confirmed")
        self.logger.log(f"Processed order {order.id}")
```

### After: Easy to Test

```python
# Easy to test — dependencies are injected
class OrderProcessor:
    def __init__(self, db, mailer, logger):
        self.db = db
        self.mailer = mailer
        self.logger = logger

    def process(self, order):
        self.db.save(order)
        self.mailer.send(order.customer_email, "Order confirmed")
        self.logger.log(f"Processed order {order.id}")


# In tests — inject mocks directly, no patching needed
def test_process_order():
    db = Mock()
    mailer = Mock()
    logger = Mock()

    processor = OrderProcessor(db, mailer, logger)
    order = Mock(id=42, customer_email="alice@test.com")

    processor.process(order)

    db.save.assert_called_once_with(order)
    mailer.send.assert_called_once_with("alice@test.com", "Order confirmed")
    logger.log.assert_called_once()
```

---

## When NOT to Mock

Mocking is a tool, not a religion. Here are situations where mocking hurts more than it helps:

### 1. Do Not Mock What You Own (Sometimes)

If you wrote the class and it is fast, consider using the real thing:

```python
# Unnecessary mock — the real Calculator is fast and has no I/O
def test_report_total_unnecessary_mock():
    calc = Mock()
    calc.add.return_value = 15
    report = Report(calc)
    report.generate()
    calc.add.assert_called()  # Tests the mock, not the code

# Better — use the real Calculator
def test_report_total():
    calc = Calculator()
    report = Report(calc)
    result = report.generate()
    assert result.total == 15
```

### 2. Do Not Mock Data Structures

```python
# BAD: mocking a dictionary
mock_config = Mock()
mock_config.__getitem__ = Mock(return_value="value")

# GOOD: use a real dictionary
config = {"key": "value", "debug": True}
```

### 3. Do Not Mock Everything

If a test requires 5+ mocks, the code under test probably has too many responsibilities. Refactor the code, do not add more mocks.

### When to Mock

- External HTTP APIs (use `responses` or `pytest-httpx`)
- Database calls in unit tests
- File system operations in unit tests
- Time-dependent functions (`datetime.now`, `time.time`)
- Non-deterministic functions (random, UUIDs)
- Slow operations (network, heavy computation)

---

## Practical Example: Mocking a Weather Service

```python
# weather_service.py
import httpx

class WeatherService:
    BASE_URL = "https://api.weather.com/v1"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_temperature(self, city: str) -> float:
        response = httpx.get(
            f"{self.BASE_URL}/current",
            params={"city": city, "key": self.api_key},
        )
        response.raise_for_status()
        data = response.json()
        return data["temperature"]

    def is_freezing(self, city: str) -> bool:
        return self.get_temperature(city) <= 0.0
```

```python
# test_weather_service.py
from unittest.mock import patch, Mock
import pytest
import httpx
from weather_service import WeatherService


@pytest.fixture
def service():
    return WeatherService(api_key="test-key-123")


class TestGetTemperature:
    @patch("weather_service.httpx.get")
    def test_returns_temperature(self, mock_get, service):
        mock_response = Mock()
        mock_response.json.return_value = {"temperature": 22.5}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        temp = service.get_temperature("Seoul")
        assert temp == 22.5
        mock_get.assert_called_once_with(
            "https://api.weather.com/v1/current",
            params={"city": "Seoul", "key": "test-key-123"},
        )

    @patch("weather_service.httpx.get")
    def test_raises_on_api_error(self, mock_get, service):
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found", request=Mock(), response=Mock(status_code=404)
        )
        mock_get.return_value = mock_response

        with pytest.raises(httpx.HTTPStatusError):
            service.get_temperature("Atlantis")


class TestIsFreezing:
    @patch.object(WeatherService, "get_temperature")
    def test_freezing_temperature(self, mock_temp, service):
        mock_temp.return_value = -5.0
        assert service.is_freezing("Moscow") is True

    @patch.object(WeatherService, "get_temperature")
    def test_warm_temperature(self, mock_temp, service):
        mock_temp.return_value = 25.0
        assert service.is_freezing("Miami") is False

    @patch.object(WeatherService, "get_temperature")
    def test_exactly_zero_is_freezing(self, mock_temp, service):
        mock_temp.return_value = 0.0
        assert service.is_freezing("Berlin") is True
```

---

## Exercises

1. **Mock a file reader**: Write a function `count_lines(filepath)` that reads a file and returns the number of lines. Write tests using `mock_open` from `unittest.mock` without creating real files.

2. **Dependency injection refactor**: Given a class `NotificationService` that directly creates an `SmtpClient` inside `__init__`, refactor it to accept the client as a parameter. Write tests using a `Mock` client.

3. **side_effect chain**: Write a `RetryClient` that retries HTTP requests up to 3 times on failure. Use `side_effect` to simulate two failures followed by a success. Verify the client makes exactly 3 calls and returns the successful result.

---

**License**: CC BY-NC 4.0
