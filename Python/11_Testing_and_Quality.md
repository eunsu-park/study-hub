# 11. 테스트 및 품질 관리 (Testing & Quality Assurance)

## 학습 목표
- pytest 프레임워크의 기본과 고급 기능 마스터
- 효과적인 테스트 작성 패턴 이해
- 모킹(Mocking)을 활용한 격리 테스트
- 코드 커버리지 측정 및 개선
- 테스트 자동화와 CI/CD 통합

## 목차
1. [pytest 기초](#1-pytest-기초)
2. [Fixtures](#2-fixtures)
3. [Parametrize](#3-parametrize)
4. [Mocking](#4-mocking)
5. [커버리지](#5-커버리지)
6. [테스트 패턴](#6-테스트-패턴)
7. [연습 문제](#7-연습-문제)

---

## 1. pytest 기초

### 1.1 pytest 설치 및 구조

```bash
# 설치
pip install pytest pytest-cov pytest-mock pytest-asyncio

# 프로젝트 구조
myproject/
├── src/
│   └── mypackage/
│       ├── __init__.py
│       ├── calculator.py
│       └── utils.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py          # 공유 fixtures
│   ├── test_calculator.py
│   └── test_utils.py
├── pyproject.toml
└── pytest.ini               # 또는 pyproject.toml에 설정
```

### 1.2 기본 테스트 작성

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
    """Calculator 클래스 테스트"""

    def setup_method(self):
        """각 테스트 메서드 전에 실행"""
        self.calc = Calculator()

    def test_add_positive_numbers(self):
        """양수 덧셈 테스트"""
        result = self.calc.add(2, 3)
        assert result == 5

    def test_add_negative_numbers(self):
        """음수 덧셈 테스트"""
        result = self.calc.add(-2, -3)
        assert result == -5

    def test_divide_normal(self):
        """정상 나눗셈"""
        result = self.calc.divide(10, 2)
        assert result == 5.0

    def test_divide_by_zero_raises_error(self):
        """0으로 나누기 예외 테스트"""
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            self.calc.divide(10, 0)

    def test_is_positive_true(self):
        assert self.calc.is_positive(5) is True

    def test_is_positive_false(self):
        assert self.calc.is_positive(-5) is False
```

### 1.3 pytest 실행 옵션

```bash
# 기본 실행
pytest

# 상세 출력
pytest -v

# 특정 파일/디렉토리
pytest tests/test_calculator.py
pytest tests/

# 특정 테스트 함수
pytest tests/test_calculator.py::TestCalculator::test_add_positive_numbers

# 키워드로 필터링
pytest -k "add"      # 'add'가 포함된 테스트만
pytest -k "not slow" # 'slow'가 없는 테스트만

# 첫 번째 실패에서 중단
pytest -x

# 마지막 실패한 테스트만 재실행
pytest --lf

# 실패한 테스트부터 시작
pytest --ff

# 병렬 실행 (pytest-xdist 필요)
pip install pytest-xdist
pytest -n auto
```

### 1.4 설정 파일

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

### 2.1 기본 Fixture

```python
# tests/conftest.py
import pytest
from mypackage.calculator import Calculator
from mypackage.database import Database


@pytest.fixture
def calculator():
    """Calculator 인스턴스 제공"""
    return Calculator()


@pytest.fixture
def sample_data():
    """테스트용 샘플 데이터"""
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
    """fixture를 인자로 받아 사용"""
    assert calculator.add(2, 3) == 5


def test_sample_data(sample_data):
    """샘플 데이터 fixture"""
    assert len(sample_data["users"]) == 2
```

### 2.2 Fixture Scope

```python
@pytest.fixture(scope="function")  # 기본값: 각 테스트 함수마다
def func_fixture():
    print("Setup function fixture")
    yield "function"
    print("Teardown function fixture")


@pytest.fixture(scope="class")  # 테스트 클래스당 한 번
def class_fixture():
    print("Setup class fixture")
    yield "class"
    print("Teardown class fixture")


@pytest.fixture(scope="module")  # 모듈당 한 번
def module_fixture():
    print("Setup module fixture")
    yield "module"
    print("Teardown module fixture")


@pytest.fixture(scope="session")  # 전체 테스트 세션당 한 번
def session_fixture():
    print("Setup session fixture")
    yield "session"
    print("Teardown session fixture")
```

### 2.3 Setup/Teardown 패턴

```python
@pytest.fixture
def database():
    """데이터베이스 연결 fixture with cleanup"""
    # Setup
    db = Database()
    db.connect()
    db.create_tables()

    yield db  # 테스트에 제공

    # Teardown (테스트 후 자동 실행)
    db.drop_tables()
    db.disconnect()


@pytest.fixture
def temp_file(tmp_path):
    """임시 파일 생성 (pytest 내장 tmp_path 활용)"""
    file_path = tmp_path / "test.txt"
    file_path.write_text("test content")
    yield file_path
    # tmp_path는 자동 정리됨
```

### 2.4 Factory Fixture

```python
@pytest.fixture
def user_factory():
    """사용자 객체 생성 팩토리"""
    created_users = []

    def _create_user(name: str, age: int = 20):
        user = {"name": name, "age": age, "id": len(created_users) + 1}
        created_users.append(user)
        return user

    yield _create_user

    # Teardown: 생성된 모든 사용자 정리
    created_users.clear()


def test_with_factory(user_factory):
    user1 = user_factory("Alice")
    user2 = user_factory("Bob", age=30)

    assert user1["name"] == "Alice"
    assert user2["age"] == 30
```

### 2.5 Fixture 의존성

```python
@pytest.fixture
def config():
    return {"db_url": "sqlite:///test.db"}


@pytest.fixture
def database(config):  # config fixture에 의존
    db = Database(config["db_url"])
    db.connect()
    yield db
    db.disconnect()


@pytest.fixture
def user_service(database):  # database fixture에 의존
    return UserService(database)
```

---

## 3. Parametrize

### 3.1 기본 파라미터화

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

### 3.2 ID 지정

```python
@pytest.mark.parametrize("input_val, expected", [
    pytest.param(1, True, id="positive"),
    pytest.param(0, False, id="zero"),
    pytest.param(-1, False, id="negative"),
])
def test_is_positive_with_ids(calculator, input_val, expected):
    assert calculator.is_positive(input_val) == expected
```

### 3.3 다중 파라미터화

```python
@pytest.mark.parametrize("x", [1, 2])
@pytest.mark.parametrize("y", [10, 20])
def test_multiply(x, y):
    """4개 테스트 케이스 생성: (1,10), (1,20), (2,10), (2,20)"""
    result = x * y
    assert result == x * y
```

### 3.4 예외 테스트 파라미터화

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

### 4.1 pytest-mock 기초

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
    """mocker fixture 사용 (pytest-mock)"""
    # Mock 설정
    mock_response = Mock()
    mock_response.json.return_value = {"temperature": 25.5}
    mock_response.raise_for_status = Mock()

    mocker.patch("mypackage.weather.requests.get", return_value=mock_response)

    # 테스트
    service = WeatherService("fake-api-key")
    temp = service.get_temperature("Seoul")

    assert temp == 25.5


def test_get_temperature_with_patch():
    """@patch 데코레이터 사용"""
    with patch("mypackage.weather.requests.get") as mock_get:
        mock_get.return_value.json.return_value = {"temperature": 30.0}
        mock_get.return_value.raise_for_status = Mock()

        service = WeatherService("fake-key")
        temp = service.get_temperature("Tokyo")

        assert temp == 30.0
        mock_get.assert_called_once()
```

### 4.2 Mock 객체 상세

```python
from unittest.mock import Mock, MagicMock, PropertyMock


def test_mock_methods():
    mock = Mock()

    # 반환값 설정
    mock.method.return_value = 42
    assert mock.method() == 42

    # side_effect로 예외 발생
    mock.error_method.side_effect = ValueError("Error!")
    with pytest.raises(ValueError):
        mock.error_method()

    # side_effect로 순차적 반환
    mock.sequence.side_effect = [1, 2, 3]
    assert mock.sequence() == 1
    assert mock.sequence() == 2
    assert mock.sequence() == 3


def test_mock_assertions():
    mock = Mock()

    mock.method("arg1", kwarg="value")

    # 호출 검증
    mock.method.assert_called()
    mock.method.assert_called_once()
    mock.method.assert_called_with("arg1", kwarg="value")
    mock.method.assert_called_once_with("arg1", kwarg="value")

    # 호출 횟수
    assert mock.method.call_count == 1


def test_magic_mock():
    """MagicMock: 매직 메서드 지원"""
    mock = MagicMock()
    mock.__len__.return_value = 5
    assert len(mock) == 5

    mock.__getitem__.return_value = "item"
    assert mock[0] == "item"
```

### 4.3 비동기 코드 모킹

```python
import pytest
from unittest.mock import AsyncMock


@pytest.mark.asyncio
async def test_async_mock():
    # AsyncMock 사용
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

### 4.4 모킹 패턴

```python
# 환경 변수 모킹
def test_with_env_var(mocker):
    mocker.patch.dict("os.environ", {"API_KEY": "test-key"})
    import os
    assert os.environ["API_KEY"] == "test-key"


# 시간 모킹
def test_with_frozen_time(mocker):
    from datetime import datetime
    mock_now = datetime(2024, 1, 15, 12, 0, 0)
    mocker.patch("mypackage.service.datetime")
    mocker.patch("mypackage.service.datetime.now", return_value=mock_now)


# 파일 시스템 모킹
def test_file_read(mocker):
    mock_open = mocker.mock_open(read_data="file content")
    mocker.patch("builtins.open", mock_open)

    with open("dummy.txt") as f:
        content = f.read()

    assert content == "file content"
```

---

## 5. 커버리지

### 5.1 pytest-cov 설정

```bash
# 설치
pip install pytest-cov

# 커버리지 측정 실행
pytest --cov=mypackage tests/

# HTML 리포트 생성
pytest --cov=mypackage --cov-report=html tests/

# 터미널에 상세 출력
pytest --cov=mypackage --cov-report=term-missing tests/
```

### 5.2 설정 파일

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

### 5.3 커버리지 제외

```python
def debug_function():  # pragma: no cover
    """디버그 전용 함수 - 커버리지 제외"""
    print("Debug info")


def main():
    if __name__ == "__main__":  # pragma: no cover
        run_app()
```

---

## 6. 테스트 패턴

### 6.1 Arrange-Act-Assert (AAA)

```python
def test_user_registration():
    # Arrange: 준비
    user_data = {"name": "Alice", "email": "alice@example.com"}
    service = UserService()

    # Act: 실행
    user = service.register(user_data)

    # Assert: 검증
    assert user.name == "Alice"
    assert user.email == "alice@example.com"
    assert user.id is not None
```

### 6.2 Given-When-Then (BDD 스타일)

```python
def test_withdraw_sufficient_balance():
    """
    Given: 계좌에 1000원이 있을 때
    When: 500원을 출금하면
    Then: 잔액은 500원이 된다
    """
    # Given
    account = Account(balance=1000)

    # When
    account.withdraw(500)

    # Then
    assert account.balance == 500
```

### 6.3 테스트 마커

```python
import pytest


@pytest.mark.slow
def test_large_data_processing():
    """느린 테스트"""
    pass


@pytest.mark.integration
def test_database_connection():
    """통합 테스트"""
    pass


@pytest.mark.skip(reason="기능 미구현")
def test_future_feature():
    pass


@pytest.mark.skipif(
    condition=True,
    reason="특정 조건에서 스킵"
)
def test_conditional():
    pass


@pytest.mark.xfail(reason="알려진 버그")
def test_known_bug():
    assert False  # 실패해도 테스트 통과


# 실행: pytest -m "not slow"  # slow 제외
# 실행: pytest -m "integration"  # integration만
```

### 6.4 테스트 그룹화

```python
class TestUserCreation:
    """사용자 생성 관련 테스트 그룹"""

    def test_create_with_valid_data(self):
        pass

    def test_create_with_invalid_email(self):
        pass

    def test_create_duplicate_user(self):
        pass


class TestUserAuthentication:
    """인증 관련 테스트 그룹"""

    def test_login_success(self):
        pass

    def test_login_wrong_password(self):
        pass
```

---

## 7. 연습 문제

### 연습 1: 기본 테스트 작성
다음 함수에 대한 테스트를 작성하세요.

```python
# 테스트 대상
def validate_email(email: str) -> bool:
    """이메일 유효성 검사"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


# 테스트 작성
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

### 연습 2: Fixture 활용
데이터베이스 연결 fixture를 작성하고 사용하세요.

```python
# 예시 답안
@pytest.fixture
def db_connection():
    """테스트용 인메모리 DB"""
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

### 연습 3: Mocking 실습
외부 API를 호출하는 함수를 모킹하여 테스트하세요.

```python
# 테스트 대상
class PaymentService:
    def process_payment(self, amount: float) -> dict:
        # 실제로는 외부 결제 API 호출
        import requests
        response = requests.post(
            "https://payment.api/charge",
            json={"amount": amount}
        )
        return response.json()


# 테스트 (예시 답안)
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

## 다음 단계
- [12. 패키징 및 배포](./12_Packaging_and_Distribution.md)
- [13. 데이터클래스](./13_Dataclasses.md)

## 참고 자료
- [pytest Documentation](https://docs.pytest.org/)
- [pytest-mock](https://pytest-mock.readthedocs.io/)
- [pytest-cov](https://pytest-cov.readthedocs.io/)
- [unittest.mock](https://docs.python.org/3/library/unittest.mock.html)
