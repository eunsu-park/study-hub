# pytest를 이용한 단위 테스팅 (Unit Testing with pytest)

**이전**: [테스팅 기초](./01_Testing_Fundamentals.md) | **다음**: [테스트 Fixture와 매개변수화](./03_Test_Fixtures_and_Parameterization.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. pytest 규칙에 따라 테스트 함수를 작성하고 실행할 수 있다
2. pytest의 어서션 인트로스펙션을 활용하여 명확한 실패 메시지를 얻을 수 있다
3. `pytest.raises`로 예상되는 예외를 테스트할 수 있다
4. mark, skip, xfail로 테스트 실행을 제어할 수 있다
5. 출력된 내용과 로그 메시지를 캡처하고 어서트할 수 있다

---

## 왜 pytest인가?

Python은 표준 라이브러리에 `unittest`를 포함하고 있지만, pytest가 Python 테스팅의 사실상 표준이 되었습니다. 그 이유는 다음과 같습니다:

- **일반 함수** — `TestCase`를 상속하거나 `self.assertEqual` 메서드를 기억할 필요 없음
- **어서션 인트로스펙션** — 일반 `assert` 문을 작성하면 pytest가 상세한 실패 정보를 보여주도록 재작성
- **풍부한 플러그인 생태계** — 병렬 실행부터 스냅샷 테스팅까지 800개 이상의 플러그인
- **Fixture 시스템** — 강력하고 조합 가능한 의존성 주입 (레슨 03에서 다룸)
- **더 나은 출력** — 색상 구분, 간결, 설정 가능

```python
# unittest style — verbose
import unittest

class TestMath(unittest.TestCase):
    def test_add(self):
        self.assertEqual(1 + 1, 2)

# pytest style — clean
def test_add():
    assert 1 + 1 == 2
```

pytest 설치:

```bash
pip install pytest
```

---

## 첫 번째 테스트 작성

### 테스트 탐색 규칙

pytest는 다음 규칙을 사용하여 자동으로 테스트를 탐색합니다:

1. `test_*.py` 또는 `*_test.py` 형식의 파일명
2. `test_` 접두사가 붙은 함수
3. `Test` 접두사가 붙은 클래스 (`__init__` 메서드 없음)
4. 테스트 클래스 내에서 `test_` 접두사가 붙은 메서드

```python
# test_strings.py

def test_upper():
    assert "hello".upper() == "HELLO"

def test_strip():
    assert "  spaces  ".strip() == "spaces"

class TestSplit:
    def test_simple_split(self):
        assert "a,b,c".split(",") == ["a", "b", "c"]

    def test_split_with_limit(self):
        assert "a,b,c".split(",", 1) == ["a", "b,c"]
```

테스트 실행:

```bash
# Run all tests in current directory (recursive)
pytest

# Run a specific file
pytest test_strings.py

# Run a specific test function
pytest test_strings.py::test_upper

# Run a specific test class method
pytest test_strings.py::TestSplit::test_simple_split

# Verbose output
pytest -v

# Short traceback
pytest --tb=short

# Stop on first failure
pytest -x

# Run last failed tests only
pytest --lf
```

---

## 어서션 인트로스펙션 (Assertion Introspection)

pytest는 import 시점에 `assert` 문을 재작성하여 풍부한 실패 메시지를 제공합니다. 특별한 어서션 메서드가 필요하지 않습니다.

```python
def test_list_equality():
    expected = [1, 2, 3, 4, 5]
    actual = [1, 2, 3, 4, 6]
    assert actual == expected
```

실패 출력:

```
FAILED test_example.py::test_list_equality
    assert actual == expected
E    AssertionError: assert [1, 2, 3, 4, 6] == [1, 2, 3, 4, 5]
E      At index 4 diff: 6 != 5
```

### 일반적인 어서션 패턴

```python
def test_equality():
    assert calculate_total([10, 20, 30]) == 60

def test_truthiness():
    assert is_valid_email("user@example.com")

def test_membership():
    result = get_supported_formats()
    assert "pdf" in result

def test_identity():
    singleton_a = get_config()
    singleton_b = get_config()
    assert singleton_a is singleton_b

def test_approximate_equality():
    # For floating-point comparisons
    assert 0.1 + 0.2 == pytest.approx(0.3)
    assert calculate_pi() == pytest.approx(3.14159, rel=1e-3)

def test_string_contains():
    error_msg = get_error_message(404)
    assert "not found" in error_msg.lower()

def test_length():
    items = fetch_page(page=1, size=10)
    assert len(items) == 10

def test_type():
    result = parse_config("app.yaml")
    assert isinstance(result, dict)
```

---

## pytest.raises로 예외 테스트

`pytest.raises`를 컨텍스트 매니저로 사용하여 코드가 예상된 예외를 발생시키는지 검증합니다.

```python
import pytest

def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Divisor cannot be zero")
    return a / b


def test_divide_by_zero_raises_value_error():
    with pytest.raises(ValueError):
        divide(10, 0)

def test_divide_by_zero_error_message():
    with pytest.raises(ValueError, match="Divisor cannot be zero"):
        divide(10, 0)

def test_divide_by_zero_inspect_exception():
    with pytest.raises(ValueError) as exc_info:
        divide(10, 0)
    assert "zero" in str(exc_info.value)
    assert exc_info.type is ValueError
```

### 여러 예외 유형 테스트

```python
def parse_age(value: str) -> int:
    if not value:
        raise ValueError("Age cannot be empty")
    age = int(value)  # May raise ValueError for non-numeric
    if age < 0:
        raise ValueError("Age cannot be negative")
    if age > 150:
        raise ValueError("Age seems unrealistic")
    return age


def test_empty_age():
    with pytest.raises(ValueError, match="empty"):
        parse_age("")

def test_negative_age():
    with pytest.raises(ValueError, match="negative"):
        parse_age("-5")

def test_non_numeric_age():
    with pytest.raises(ValueError):
        parse_age("abc")

def test_unrealistic_age():
    with pytest.raises(ValueError, match="unrealistic"):
        parse_age("200")
```

---

## 클래스로 테스트 구조화

pytest는 일반 함수와 잘 동작하지만, 클래스를 사용하면 관련 테스트를 그룹화하는 데 도움이 됩니다:

```python
class TestShoppingCart:
    """Tests for the ShoppingCart class."""

    def test_new_cart_is_empty(self):
        cart = ShoppingCart()
        assert len(cart.items) == 0

    def test_add_item(self):
        cart = ShoppingCart()
        cart.add("apple", quantity=3, price=1.50)
        assert len(cart.items) == 1

    def test_total_with_multiple_items(self):
        cart = ShoppingCart()
        cart.add("apple", quantity=2, price=1.50)
        cart.add("bread", quantity=1, price=3.00)
        assert cart.total() == 6.00

    def test_remove_item(self):
        cart = ShoppingCart()
        cart.add("apple", quantity=1, price=1.50)
        cart.remove("apple")
        assert len(cart.items) == 0

    def test_remove_nonexistent_item_raises(self):
        cart = ShoppingCart()
        with pytest.raises(KeyError):
            cart.remove("banana")
```

참고: `unittest.TestCase`와 달리, pytest 테스트 클래스는 `__init__` 메서드를 가져서는 **안 됩니다**. `setUp`/`tearDown` 대신 fixture를 사용하세요 (레슨 03).

---

## Mark: 테스트 분류와 제어

Mark는 테스트에 메타데이터를 부착하는 데코레이터입니다. pytest는 mark를 사용하여 테스트를 필터링, 건너뛰기, 또는 동작을 수정합니다.

### 내장 Mark

```python
import pytest
import sys

# Skip unconditionally
@pytest.mark.skip(reason="Feature not implemented yet")
def test_future_feature():
    assert fancy_new_thing() == 42

# Skip conditionally
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Unix-only test"
)
def test_unix_permissions():
    import os
    assert os.access("/tmp", os.W_OK)

# Expected failure — test is known to fail
@pytest.mark.xfail(reason="Bug #1234 not yet fixed")
def test_known_bug():
    assert broken_function() == "correct"

# Strict xfail — fail the test if it unexpectedly passes
@pytest.mark.xfail(strict=True, reason="Should fail until fix is merged")
def test_strict_known_bug():
    assert broken_function() == "correct"
```

### 커스텀 Mark

자신만의 mark를 정의하여 테스트를 분류할 수 있습니다:

```python
# conftest.py or pyproject.toml to register marks
# [tool.pytest.ini_options]
# markers = [
#     "slow: marks tests as slow (deselect with '-m \"not slow\"')",
#     "integration: marks tests requiring external services",
# ]

@pytest.mark.slow
def test_large_dataset_processing():
    data = generate_large_dataset(size=1_000_000)
    result = process(data)
    assert len(result) > 0

@pytest.mark.integration
def test_database_connection():
    conn = connect_to_db()
    assert conn.is_alive()
```

mark 필터로 실행:

```bash
# Run only slow tests
pytest -m slow

# Run everything EXCEPT slow tests
pytest -m "not slow"

# Run tests marked slow OR integration
pytest -m "slow or integration"
```

---

## 출력 캡처

pytest는 기본적으로 `stdout`과 `stderr`를 캡처합니다. 실패한 테스트에 대해서만 출력을 보여줍니다. `capsys` fixture를 사용하여 출력된 내용을 명시적으로 테스트할 수 있습니다.

```python
def greet(name: str) -> None:
    """Print a greeting message."""
    print(f"Hello, {name}!")


def test_greet_output(capsys):
    greet("World")
    captured = capsys.readouterr()
    assert captured.out == "Hello, World!\n"
    assert captured.err == ""


def warn_user(message: str) -> None:
    """Print a warning to stderr."""
    import sys
    print(f"WARNING: {message}", file=sys.stderr)


def test_warn_output(capsys):
    warn_user("disk almost full")
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "disk almost full" in captured.err
```

### 캡처 동작 제어

```bash
# Disable capture (see all print output in real time)
pytest -s

# Equivalent
pytest --capture=no

# Default (capture both stdout and stderr)
pytest --capture=fd
```

---

## 로그 출력 테스트

`caplog` fixture를 사용하여 로그 메시지를 테스트합니다:

```python
import logging

logger = logging.getLogger(__name__)

def process_order(order_id: int) -> str:
    logger.info(f"Processing order {order_id}")
    if order_id <= 0:
        logger.error(f"Invalid order ID: {order_id}")
        raise ValueError("Order ID must be positive")
    logger.info(f"Order {order_id} processed successfully")
    return "success"


def test_process_order_logs(caplog):
    with caplog.at_level(logging.INFO):
        result = process_order(42)

    assert result == "success"
    assert "Processing order 42" in caplog.text
    assert "processed successfully" in caplog.text


def test_invalid_order_logs_error(caplog):
    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError):
            process_order(-1)

    assert "Invalid order ID: -1" in caplog.text
```

---

## 실전 예제: 비밀번호 유효성 검사기 테스팅

여러 pytest 기능을 현실적인 시나리오에서 통합하여 활용합니다:

```python
# password_validator.py
import re
from dataclasses import dataclass


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str]


def validate_password(password: str) -> ValidationResult:
    """Validate a password against security rules."""
    errors = []

    if len(password) < 8:
        errors.append("Password must be at least 8 characters")
    if len(password) > 128:
        errors.append("Password must be at most 128 characters")
    if not re.search(r"[A-Z]", password):
        errors.append("Password must contain an uppercase letter")
    if not re.search(r"[a-z]", password):
        errors.append("Password must contain a lowercase letter")
    if not re.search(r"[0-9]", password):
        errors.append("Password must contain a digit")
    if not re.search(r"[!@#$%^&*]", password):
        errors.append("Password must contain a special character")

    return ValidationResult(is_valid=len(errors) == 0, errors=errors)
```

```python
# test_password_validator.py
import pytest
from password_validator import validate_password


class TestValidPassword:
    def test_strong_password_is_valid(self):
        result = validate_password("MyStr0ng!Pass")
        assert result.is_valid is True
        assert result.errors == []

    def test_minimum_valid_password(self):
        result = validate_password("Aa1!xxxx")
        assert result.is_valid is True


class TestPasswordLength:
    def test_too_short(self):
        result = validate_password("Aa1!")
        assert result.is_valid is False
        assert "at least 8" in result.errors[0]

    def test_too_long(self):
        result = validate_password("A" * 129 + "a1!")
        assert result.is_valid is False
        assert any("at most 128" in e for e in result.errors)

    def test_exactly_eight_characters(self):
        result = validate_password("Aa1!xxxx")
        assert result.is_valid is True


class TestPasswordCharacterRequirements:
    def test_missing_uppercase(self):
        result = validate_password("mystr0ng!pass")
        assert not result.is_valid
        assert any("uppercase" in e for e in result.errors)

    def test_missing_lowercase(self):
        result = validate_password("MYSTR0NG!PASS")
        assert not result.is_valid
        assert any("lowercase" in e for e in result.errors)

    def test_missing_digit(self):
        result = validate_password("MyStrong!Pass")
        assert not result.is_valid
        assert any("digit" in e for e in result.errors)

    def test_missing_special_character(self):
        result = validate_password("MyStr0ngPass1")
        assert not result.is_valid
        assert any("special" in e for e in result.errors)


class TestMultipleErrors:
    def test_empty_password_reports_all_errors(self):
        result = validate_password("")
        assert not result.is_valid
        assert len(result.errors) >= 4  # length + missing char types
```

---

## 유용한 커맨드 라인 옵션

```bash
# Show slowest N tests
pytest --durations=10

# Run tests matching a keyword expression
pytest -k "password and not slow"

# Run tests in parallel (requires pytest-xdist)
pytest -n auto

# Generate JUnit XML report (for CI)
pytest --junitxml=results.xml

# Show local variables in tracebacks
pytest -l

# Drop into debugger on failure
pytest --pdb

# Collect tests without running them
pytest --collect-only

# Show print output for all tests (not just failures)
pytest -s -v
```

---

## 연습 문제

1. **단위 테스트 작성**: `temperature.py` 모듈을 만들고 `celsius_to_fahrenheit(c)`와 `fahrenheit_to_celsius(f)` 함수를 작성하세요. 정상 값, 빙점, 끓는점, 음수 온도를 포함하여 최소 6개의 테스트를 작성하세요. 부동소수점 비교에는 `pytest.approx`를 사용하세요.

2. **예외 테스트**: `"rgb(255, 128, 0)"` 같은 문자열을 파싱하여 튜플 `(255, 128, 0)`을 반환하는 `parse_rgb(color_string)` 함수를 작성하세요. 잘못된 형식과 0-255 범위 밖의 값에 대해 `ValueError`를 발생시켜야 합니다. `match` 패턴과 함께 `pytest.raises`를 사용하여 테스트를 작성하세요.

3. **Mark와 필터**: 최소 8개의 테스트가 있는 테스트 파일을 만드세요. 일부에는 `@pytest.mark.slow`를, 일부에는 `@pytest.mark.smoke`를 표시하세요. `-m` 표현식을 사용하여 다른 부분 집합을 실행하는 연습을 하세요.

---

**License**: CC BY-NC 4.0
