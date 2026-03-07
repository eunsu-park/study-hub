# Mocking과 Patching (Mocking and Patching)

**이전**: [테스트 Fixture와 매개변수화](./03_Test_Fixtures_and_Parameterization.md) | **다음**: [테스트 커버리지와 품질](./05_Test_Coverage_and_Quality.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `unittest.mock.Mock`과 `MagicMock`을 사용하여 테스트 더블을 생성할 수 있다
2. `patch`와 `patch.object`로 함수, 메서드, 객체를 패치할 수 있다
3. `side_effect`와 `return_value`로 mock 동작을 제어할 수 있다
4. `spec`을 사용하여 인터페이스 오류를 잡는 타입 안전한 mock을 생성할 수 있다
5. pytest의 `monkeypatch`를 경량 패칭에 적용할 수 있다
6. mocking이 적절한 경우와 해로운 경우를 구분할 수 있다

---

## 왜 Mock을 사용하는가?

단위 테스트는 데이터베이스, API, 파일 시스템 같은 외부 의존성 없이 격리된 상태에서 실행되어야 합니다. Mocking을 사용하면 이러한 의존성을 제어된 대역으로 교체하여 다음을 할 수 있습니다:

- **로직을 격리하여 테스트** — 인프라를 구동하지 않아도 됨
- **엣지 케이스 시뮬레이션** — 네트워크 타임아웃이나 API 오류 같이 재현하기 어려운 경우
- **테스트 속도 향상** — I/O 제거
- **상호작용 검증** — 코드가 의존성을 올바르게 호출하는지 확인

하지만 mocking에는 어두운 면이 있습니다: 과도한 mocking은 실제 코드가 깨져 있어도 통과하는 테스트를 만듭니다. 핵심은 *언제* mock하고 *무엇을* mock할지 아는 것입니다.

---

## Mock 기초

### Mock과 MagicMock

`Mock`은 모든 속성 접근이나 메서드 호출을 수용하는 객체를 생성합니다. `MagicMock`은 매직 메서드(`__len__`, `__iter__` 등)의 기본 구현을 포함하여 `Mock`을 확장합니다.

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

### 반환값 설정

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

### 호출 어서트

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

## side_effect: 동적 Mock 동작

`side_effect`를 사용하면 mock이 예외를 발생시키거나, 연속 호출에서 다른 값을 반환하거나, 커스텀 로직을 실행하도록 할 수 있습니다.

### 예외 발생

```python
from unittest.mock import Mock
import pytest

http_client = Mock()
http_client.get.side_effect = ConnectionError("Network is unreachable")

with pytest.raises(ConnectionError, match="unreachable"):
    http_client.get("https://api.example.com/data")
```

### 순차적 반환값

```python
from unittest.mock import Mock

# Return different values on each call
token_generator = Mock()
token_generator.generate.side_effect = ["token-aaa", "token-bbb", "token-ccc"]

assert token_generator.generate() == "token-aaa"
assert token_generator.generate() == "token-bbb"
assert token_generator.generate() == "token-ccc"
```

### 커스텀 로직

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

## patch: 실제 객체 교체

`patch`는 객체가 *정의된* 모듈이 아니라 *사용되는* 모듈에서 객체를 일시적으로 교체합니다. 이것이 패칭 버그의 가장 흔한 원인입니다.

### Import 경로 규칙

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

### 컨텍스트 매니저로 사용하는 patch

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

객체의 특정 속성을 패치합니다. 기존 인스턴스의 메서드를 패치할 때 유용합니다.

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

## spec: 타입 안전한 Mock

`spec` 없이 mock은 모든 속성 접근을 묵묵히 수용합니다. 이는 테스트에서 오타가 감지되지 않는다는 의미입니다:

```python
from unittest.mock import Mock

# Without spec — typos are invisible
mock_user = Mock()
mock_user.nmae = "Alice"  # Typo! Should be 'name'. No error raised.
```

`spec`을 사용하면 mock이 실제 클래스에 대해 속성 접근을 검증합니다:

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

**모범 사례**: 구체적인 클래스를 mocking할 때는 항상 `spec` (또는 더 엄격한 검사를 위한 `spec_set`)을 사용하세요.

---

## pytest monkeypatch

pytest의 `monkeypatch` fixture는 일반적인 패칭 시나리오에 대한 더 간단한 대안을 제공합니다. `unittest.mock.patch`와 달리, 각 테스트 후 변경 사항을 자동으로 되돌립니다.

### 환경 변수 패칭

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

### 속성 패칭

```python
import myapp.config as config

def test_debug_mode(monkeypatch):
    monkeypatch.setattr(config, "DEBUG", True)
    assert config.DEBUG is True

def test_custom_timeout(monkeypatch):
    monkeypatch.setattr(config, "REQUEST_TIMEOUT", 5)
    assert config.REQUEST_TIMEOUT == 5
```

### 함수 패칭

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

## 테스트 용이성을 위한 의존성 주입

코드를 테스트 가능하게 만드는 가장 좋은 방법은 테스팅을 위해 *설계*하는 것입니다. 의존성 주입은 많은 경우에서 패칭의 필요성을 제거합니다.

### Before: 테스트하기 어려움

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

### After: 테스트하기 쉬움

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

## Mock을 사용하지 말아야 할 때

Mocking은 도구이지 종교가 아닙니다. 다음은 mocking이 도움보다 해를 끼치는 상황들입니다:

### 1. 소유한 것을 Mock하지 마라 (때때로)

직접 작성한 클래스이고 빠르다면, 실제 객체를 사용하는 것을 고려하세요:

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

### 2. 데이터 구조를 Mock하지 마라

```python
# BAD: mocking a dictionary
mock_config = Mock()
mock_config.__getitem__ = Mock(return_value="value")

# GOOD: use a real dictionary
config = {"key": "value", "debug": True}
```

### 3. 모든 것을 Mock하지 마라

테스트에 5개 이상의 mock이 필요하다면, 테스트 대상 코드의 책임이 너무 많을 가능성이 있습니다. 더 많은 mock을 추가하지 말고 코드를 리팩토링하세요.

### Mock을 사용해야 할 때

- 외부 HTTP API (`responses` 또는 `pytest-httpx` 사용)
- 단위 테스트에서의 데이터베이스 호출
- 단위 테스트에서의 파일 시스템 작업
- 시간 의존 함수 (`datetime.now`, `time.time`)
- 비결정론적 함수 (random, UUID)
- 느린 작업 (네트워크, 무거운 계산)

---

## 실전 예제: 날씨 서비스 Mocking

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

## 연습 문제

1. **파일 리더 Mock하기**: 파일을 읽고 줄 수를 반환하는 `count_lines(filepath)` 함수를 작성하세요. 실제 파일을 생성하지 않고 `unittest.mock`의 `mock_open`을 사용하여 테스트를 작성하세요.

2. **의존성 주입 리팩토링**: `__init__` 내부에서 `SmtpClient`를 직접 생성하는 `NotificationService` 클래스가 주어졌을 때, 클라이언트를 매개변수로 받도록 리팩토링하세요. `Mock` 클라이언트를 사용하여 테스트를 작성하세요.

3. **side_effect 체인**: 실패 시 최대 3번까지 HTTP 요청을 재시도하는 `RetryClient`를 작성하세요. `side_effect`를 사용하여 두 번의 실패 후 성공을 시뮬레이션하세요. 클라이언트가 정확히 3번 호출하고 성공한 결과를 반환하는지 검증하세요.

---

**License**: CC BY-NC 4.0
