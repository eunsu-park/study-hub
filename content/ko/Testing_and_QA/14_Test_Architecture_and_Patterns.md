# 레슨 14: 테스트 아키텍처와 패턴

**이전**: [CI/CD Integration](./13_CI_CD_Integration.md) | **다음**: [Testing Async Code](./15_Testing_Async_Code.md)

---

개별 테스트를 작성하는 것은 기술입니다. 코드베이스가 수백 개에서 수천 개의 테스트로 성장할 때 유지보수 가능하고, 읽기 쉽고, 신뢰할 수 있는 테스트 스위트를 설계하는 것은 아키텍처 문제입니다. 이 레슨은 개발을 가속화하는 테스트 스위트와 부담이 되는 테스트 스위트를 구분짓는 기본적인 패턴과 구조적 결정을 다룹니다.

**난이도**: ⭐⭐⭐

**사전 요구사항**:
- pytest로 테스트를 작성한 경험 (레슨 02-04)
- mocking 개념에 대한 이해 (레슨 06)
- 객체지향 설계에 대한 익숙함

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 다섯 가지 유형의 테스트 대역(dummy, stub, spy, mock, fake)을 분류하고 올바르게 사용할 수 있다
2. Arrange-Act-Assert(AAA) 패턴을 사용하여 테스트를 구조화할 수 있다
3. Given-When-Then을 사용하여 행동 주도(BDD) 스타일의 테스트를 작성할 수 있다
4. Builder 패턴을 적용하여 읽기 쉬운 테스트 데이터를 생성할 수 있다
5. UI/API 테스트 추상화를 위한 Page Object 패턴을 구현할 수 있다
6. 실무에서의 테스팅 피라미드에 대해 합리적인 결정을 내릴 수 있다

---

## 1. 테스트 대역 분류

"테스트 대역(Test double)"은 테스트에서 실제 의존성을 대신하는 모든 객체를 가리키는 총칭입니다. 이 용어는 영화 산업의 "스턴트 대역(stunt double)"에서 유래했습니다. Gerard Meszaros는 각각 다른 목적을 가진 다섯 가지 유형을 정의했습니다.

### 1.1 Dummy

Dummy는 전달되지만 실제로 사용되지 않습니다. 테스트와 무관한 필수 매개변수를 채울 때 사용합니다.

```python
class DummyLogger:
    """Satisfies the logger parameter without doing anything."""
    def info(self, msg): pass
    def error(self, msg): pass
    def warning(self, msg): pass


def test_user_creation_does_not_require_logging():
    dummy_logger = DummyLogger()
    user = UserService(logger=dummy_logger)
    result = user.create("alice", "alice@example.com")
    assert result.name == "alice"
```

테스트는 로깅에 대해 관심이 없습니다. dummy는 누락된 인자로 인한 `TypeError`를 방지합니다.

### 1.2 Stub

Stub은 테스트 중 호출에 대해 미리 정해진 응답을 제공합니다. 아무것도 기록하지 않으며 -- 단지 사전에 결정된 값을 반환합니다.

```python
class StubPriceService:
    """Returns fixed prices regardless of input."""

    def get_price(self, product_id: str) -> float:
        return 10.00  # Always $10

    def get_tax_rate(self, region: str) -> float:
        return 0.08  # Always 8%


def test_order_total_calculation():
    stub_prices = StubPriceService()
    order = OrderService(price_service=stub_prices)

    total = order.calculate_total(product_id="abc", quantity=3, region="CA")

    # 3 * $10.00 * 1.08 = $32.40
    assert total == 32.40
```

Stub은 다음 질문에 답합니다: "의존성이 이 값을 반환하면, 내 코드가 올바르게 처리하는가?"

### 1.3 Spy

Spy는 호출된 방식에 대한 정보를 기록하며, 테스트 이후에 이를 검사할 수 있습니다. 실제 구현에 위임할 수도 있습니다.

```python
class SpyEmailSender:
    """Records all emails sent."""

    def __init__(self):
        self.sent_emails = []

    def send(self, to: str, subject: str, body: str):
        self.sent_emails.append({
            "to": to,
            "subject": subject,
            "body": body
        })


def test_registration_sends_welcome_email():
    spy_sender = SpyEmailSender()
    service = RegistrationService(email_sender=spy_sender)

    service.register("bob@example.com", "password123")

    assert len(spy_sender.sent_emails) == 1
    assert spy_sender.sent_emails[0]["to"] == "bob@example.com"
    assert "welcome" in spy_sender.sent_emails[0]["subject"].lower()
```

Spy는 다음에 답합니다: "내 코드가 올바른 인자로 올바른 횟수만큼 호출되었는가?"

### 1.4 Mock

Mock은 기대값이 내장된 spy입니다. 예상되는 호출이 미리 프로그래밍되어 있으며, 기대값이 충족되지 않으면 테스트가 실패합니다. Python의 `unittest.mock.Mock`은 기술적으로 spy-mock 하이브리드입니다.

```python
from unittest.mock import Mock


def test_payment_processing():
    mock_gateway = Mock()
    mock_gateway.charge.return_value = {"status": "success", "tx_id": "abc123"}

    service = PaymentService(gateway=mock_gateway)
    result = service.process_payment(amount=50.00, card_token="tok_visa")

    # Verify the call was made with expected arguments
    mock_gateway.charge.assert_called_once_with(
        amount=50.00,
        token="tok_visa",
        currency="USD"
    )
    assert result["tx_id"] == "abc123"
```

### 1.5 Fake

Fake는 프로덕션에는 적합하지 않은 간소화된 방식을 취하는 실제 동작하는 구현입니다. Fake는 stub보다 현실적이지만 실제 의존성보다 단순합니다.

```python
class FakeUserRepository:
    """In-memory implementation of the user repository."""

    def __init__(self):
        self._users = {}
        self._next_id = 1

    def save(self, user):
        if user.id is None:
            user.id = self._next_id
            self._next_id += 1
        self._users[user.id] = user
        return user

    def find_by_id(self, user_id):
        return self._users.get(user_id)

    def find_by_email(self, email):
        for user in self._users.values():
            if user.email == email:
                return user
        return None

    def delete(self, user_id):
        return self._users.pop(user_id, None)


def test_user_workflow_with_fake_repo():
    repo = FakeUserRepository()
    service = UserService(repo=repo)

    user = service.create_user("alice", "alice@example.com")
    assert user.id == 1

    found = service.get_user(user.id)
    assert found.name == "alice"

    service.delete_user(user.id)
    assert service.get_user(user.id) is None
```

Fake는 가장 강력한 테스트 대역입니다. 외부 의존성 없이 현실적인 통합 스타일의 테스트를 가능하게 합니다.

### 1.6 올바른 테스트 대역 선택

| 테스트 대역 | 사용 시점 | 검증 방식 |
|---|---|---|
| **Dummy** | 매개변수가 필요하지만 무관할 때 | 없음 |
| **Stub** | 제어된 반환값이 필요할 때 | SUT의 상태 |
| **Spy** | 사후에 상호작용을 검증해야 할 때 | 기록된 호출 |
| **Mock** | 특정 상호작용을 검증해야 할 때 | 내장된 기대값 |
| **Fake** | 실제 인프라 없이 현실적인 동작이 필요할 때 | SUT + fake의 상태 |

---

## 2. Arrange-Act-Assert (AAA)

AAA 패턴은 단위 테스트의 표준 구조입니다. 모든 테스트는 정확히 세 단계로 구성됩니다:

```python
def test_discount_applied_to_order():
    # Arrange — 테스트 컨텍스트 설정
    catalog = FakeCatalog()
    catalog.add_product("widget", price=100.00)
    order = Order(catalog=catalog)
    order.add_item("widget", quantity=2)
    discount = PercentageDiscount(10)

    # Act — 테스트 대상 동작 수행
    order.apply_discount(discount)

    # Assert — 예상 결과 검증
    assert order.total == 180.00  # 200 - 10%
    assert order.discount_amount == 20.00
```

### 2.1 AAA 가이드라인

1. **테스트당 하나의 Act**: 여러 Act 단계가 있다면 여러 테스트로 분리합니다
2. **Arrange는 공유 가능**: 공통 설정에 fixture를 사용합니다
3. **구현이 아닌 동작을 Assert**: 어떻게가 아닌 무엇이 일어났는지 검사합니다

```python
# BAD: 여러 act (정확히 무엇을 테스트하는 것인지?)
def test_user_lifecycle():
    user = create_user("alice")     # Act 1
    assert user.id is not None
    update_user(user.id, "bob")     # Act 2
    assert get_user(user.id).name == "bob"
    delete_user(user.id)            # Act 3
    assert get_user(user.id) is None

# GOOD: 테스트당 하나의 act
def test_create_user_assigns_id():
    user = create_user("alice")
    assert user.id is not None

def test_update_user_changes_name():
    user = create_user("alice")
    update_user(user.id, "bob")
    assert get_user(user.id).name == "bob"

def test_delete_user_removes_from_database():
    user = create_user("alice")
    delete_user(user.id)
    assert get_user(user.id) is None
```

---

## 3. Given-When-Then (BDD 스타일)

Given-When-Then은 행동 주도 개발(BDD)에서 비롯되었습니다. 의미적으로 AAA와 동일하지만 도메인 언어를 사용합니다.

```python
def test_customer_with_premium_membership_gets_free_shipping():
    # Given a customer with premium membership
    customer = Customer(membership="premium")
    cart = ShoppingCart(customer=customer)
    cart.add_item(Product("laptop", price=999.99))

    # When the customer proceeds to checkout
    checkout = cart.checkout()

    # Then shipping should be free
    assert checkout.shipping_cost == 0.00
    assert checkout.total == 999.99
```

### 3.1 Given-When-Then 사용 시점

Given-When-Then이 적합한 경우:
- 테스트가 비개발자도 이해해야 하는 비즈니스 규칙을 설명할 때
- 테스트 이름이 명세서처럼 읽힐 때
- 이해관계자와의 협업으로 BDD를 실천할 때

AAA가 바람직한 경우:
- 순수하게 기술적인 테스트(알고리즘 정확성, 자료구조 동작)
- 대상이 전적으로 개발자일 때

---

## 4. 테스트 데이터를 위한 Builder 패턴

많은 필드를 가진 테스트 객체를 생성하는 것은 장황하고 취약합니다. Builder 패턴은 합리적인 기본값과 함께 테스트 데이터를 구성하는 유연한 API를 제공합니다.

### 4.1 문제점

```python
# Builder 없이 — 모든 테스트가 모든 필드를 반복
def test_premium_user_gets_discount():
    user = User(
        name="alice",
        email="alice@example.com",
        age=30,
        country="US",
        membership="premium",    # 이 필드만 중요
        created_at=datetime.now(),
        is_active=True,
        avatar_url=None,
        phone=None,
    )
    assert calculate_discount(user) == 0.15
```

### 4.2 해결책

```python
class UserBuilder:
    """Builds User objects with sensible defaults."""

    def __init__(self):
        self._name = "default_user"
        self._email = "default@example.com"
        self._age = 25
        self._country = "US"
        self._membership = "free"
        self._created_at = datetime(2024, 1, 1)
        self._is_active = True
        self._avatar_url = None
        self._phone = None

    def with_name(self, name):
        self._name = name
        return self

    def with_membership(self, membership):
        self._membership = membership
        return self

    def with_country(self, country):
        self._country = country
        return self

    def inactive(self):
        self._is_active = False
        return self

    def build(self):
        return User(
            name=self._name,
            email=self._email,
            age=self._age,
            country=self._country,
            membership=self._membership,
            created_at=self._created_at,
            is_active=self._is_active,
            avatar_url=self._avatar_url,
            phone=self._phone,
        )


# 깔끔하고 핵심에 집중된 테스트
def test_premium_user_gets_discount():
    user = UserBuilder().with_membership("premium").build()
    assert calculate_discount(user) == 0.15

def test_inactive_users_get_no_discount():
    user = UserBuilder().with_membership("premium").inactive().build()
    assert calculate_discount(user) == 0.00
```

### 4.3 pytest Fixture로서의 Builder

```python
@pytest.fixture
def user_builder():
    return UserBuilder()


def test_premium_discount(user_builder):
    user = user_builder.with_membership("premium").build()
    assert calculate_discount(user) == 0.15
```

---

## 5. Page Object 패턴

Page Object 패턴은 UI 또는 API 상호작용 세부 사항을 깔끔한 인터페이스 뒤에 추상화합니다. 원래 Selenium 테스팅에서 시작되었지만, API 테스트에도 동일하게 적용됩니다.

### 5.1 웹 UI 테스트를 위한 Page Object

```python
class LoginPage:
    """Encapsulates all interactions with the login page."""

    def __init__(self, client):
        self.client = client

    def login(self, username, password):
        response = self.client.post("/login", data={
            "username": username,
            "password": password,
        })
        return response

    def is_error_displayed(self, response):
        return "Invalid credentials" in response.text

    def get_redirect_url(self, response):
        return response.headers.get("Location")


class DashboardPage:
    """Encapsulates interactions with the dashboard."""

    def __init__(self, client):
        self.client = client

    def get_welcome_message(self):
        response = self.client.get("/dashboard")
        # Parse the welcome message from HTML
        return extract_text(response.text, ".welcome-message")

    def get_recent_items(self):
        response = self.client.get("/dashboard")
        return extract_items(response.text, ".recent-items li")


# 테스트는 page object를 사용 — 테스트 코드에서 HTML 파싱 없음
def test_successful_login_redirects_to_dashboard(client):
    login_page = LoginPage(client)
    response = login_page.login("alice", "correct_password")
    assert login_page.get_redirect_url(response) == "/dashboard"


def test_failed_login_shows_error(client):
    login_page = LoginPage(client)
    response = login_page.login("alice", "wrong_password")
    assert login_page.is_error_displayed(response)
```

### 5.2 API 테스트를 위한 Page Object

```python
class UserAPI:
    """Encapsulates the User REST API."""

    def __init__(self, client, base_url="/api/v1/users"):
        self.client = client
        self.base_url = base_url

    def create(self, name, email):
        response = self.client.post(self.base_url, json={
            "name": name,
            "email": email
        })
        return response.json()

    def get(self, user_id):
        response = self.client.get(f"{self.base_url}/{user_id}")
        return response.json() if response.status_code == 200 else None

    def list_all(self, page=1, per_page=20):
        response = self.client.get(self.base_url, params={
            "page": page, "per_page": per_page
        })
        return response.json()

    def delete(self, user_id):
        return self.client.delete(f"{self.base_url}/{user_id}")


# 동작에 집중한 깔끔한 테스트
def test_create_and_retrieve_user(client):
    api = UserAPI(client)
    created = api.create("alice", "alice@example.com")
    retrieved = api.get(created["id"])
    assert retrieved["name"] == "alice"
    assert retrieved["email"] == "alice@example.com"
```

### 5.3 Page Object의 장점

1. **DRY**: API 상호작용 로직이 한 번만 정의됨
2. **유지보수성**: API가 변경되면 50개 테스트가 아닌 하나의 클래스만 수정
3. **가독성**: 테스트가 HTTP 메커니즘이 아닌 의도를 표현
4. **재사용성**: page object를 조합하여 복잡한 워크플로우 구성 가능

---

## 6. 실무에서의 테스팅 피라미드

테스팅 피라미드는 각 수준에서 얼마나 많은 테스트를 작성해야 하는지에 대한 가이드라인입니다:

```
        ┌──────┐
        │  E2E │  소수 (10-20)
       ┌┴──────┴┐
       │ 통합    │  적당한 수 (50-100)
       │ 테스트  │
      ┌┴─────────┴┐
      │  단위 테스트│  다수 (500+)
      └────────────┘
```

### 6.1 이 형태인 이유

| 수준 | 속도 | 안정성 | 유지보수 비용 | 신뢰도 |
|---|---|---|---|---|
| 단위 | 빠름 (ms) | 매우 안정적 | 낮음 | 로직 정확성 |
| 통합 | 보통 (s) | 대체로 안정적 | 보통 | 컴포넌트 간 협력 |
| E2E | 느림 (min) | 종종 불안정 | 높음 | 시스템 전체 동작 |

### 6.2 실무에서의 피라미드

이상적인 비율은 시스템에 따라 달라집니다:

- **라이브러리/프레임워크**: 90% 단위, 10% 통합, 0% E2E
- **웹 API**: 60% 단위, 30% 통합, 10% E2E
- **프론트엔드 앱**: 40% 단위, 30% 통합, 30% E2E
- **데이터 파이프라인**: 30% 단위, 60% 통합, 10% E2E

### 6.3 안티패턴

**아이스크림 콘** (역피라미드): E2E 테스트가 많고 단위 테스트가 적습니다. 느리고, 불안정하며, 유지보수 비용이 높습니다.

**모래시계**: 단위 테스트와 E2E 테스트는 많지만 통합 테스트가 적습니다. 컴포넌트 간 상호작용 버그를 놓칩니다.

**100% 단위 커버리지**: 모든 함수가 격리 상태에서 테스트되지만, 컴포넌트를 함께 테스트하지 않아 시스템이 동작하지 않습니다.

---

## 7. 테스트 파일 구성

### 7.1 소스 구조를 미러링

```
myapp/
├── services/
│   ├── auth.py
│   └── payment.py
├── models/
│   └── user.py
└── api/
    └── routes.py

tests/
├── unit/
│   ├── services/
│   │   ├── test_auth.py
│   │   └── test_payment.py
│   └── models/
│       └── test_user.py
├── integration/
│   └── test_payment_flow.py
├── e2e/
│   └── test_checkout.py
├── conftest.py
└── builders/
    └── user_builder.py
```

### 7.2 공유 테스트 유틸리티

```python
# tests/conftest.py — shared fixtures
import pytest

from tests.builders.user_builder import UserBuilder


@pytest.fixture
def user_builder():
    return UserBuilder()


@pytest.fixture
def fake_repo():
    return FakeUserRepository()
```

---

## 연습 문제

1. **테스트 대역 분류**: 다섯 가지 테스트 시나리오(예: "등록 후 이메일이 발송되는지 테스트")가 주어졌을 때, 각각에 가장 적합한 테스트 대역 유형을 식별하고 구현하십시오.

2. **Builder 패턴**: 전자상거래 도메인을 위한 `ProductBuilder`와 `OrderBuilder`를 작성하십시오. `OrderBuilder`는 `ProductBuilder` 인스턴스를 받을 수 있어야 합니다. builder를 사용하여 세 개의 테스트를 작성하십시오.

3. **Page Object 리팩토링**: 직접 HTTP 호출을 하는 API 테스트 세트를 가져와서 Page Object를 사용하도록 리팩토링하십시오. 리팩토링 전후의 가독성을 비교하십시오.

4. **피라미드 감사**: 기존 프로젝트의 모든 테스트를 단위, 통합, E2E로 분류하십시오. 결과 형태를 그리십시오. 격차를 식별하고 균형을 개선하기 위한 테스트를 작성하십시오.

5. **AAA 규율**: 테스트 파일을 리뷰하여 단일 Act 규칙을 위반하는 테스트를 식별하십시오. 올바르게 구조화된 AAA 테스트로 리팩토링하십시오.

---

**License**: CC BY-NC 4.0
