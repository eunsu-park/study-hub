# Lesson 14: Test Architecture and Patterns

**Previous**: [CI/CD Integration](./13_CI_CD_Integration.md) | **Next**: [Testing Async Code](./15_Testing_Async_Code.md)

---

Writing individual tests is a skill. Designing a test suite that remains maintainable, readable, and trustworthy as a codebase grows from hundreds to thousands of tests is an architecture problem. This lesson covers the foundational patterns and structural decisions that make the difference between a test suite that accelerates development and one that becomes a burden.

**Difficulty**: ⭐⭐⭐

**Prerequisites**:
- Experience writing tests with pytest (Lessons 02–04)
- Understanding of mocking concepts (Lesson 06)
- Familiarity with object-oriented design

## Learning Objectives

After completing this lesson, you will be able to:

1. Classify and correctly use the five types of test doubles (dummy, stub, spy, mock, fake)
2. Structure tests using the Arrange-Act-Assert (AAA) pattern
3. Write behavior-driven tests using Given-When-Then
4. Apply the Builder pattern to create readable test data
5. Implement the Page Object pattern for UI/API test abstraction
6. Make informed decisions about the testing pyramid in practice

---

## 1. Test Doubles Taxonomy

"Test double" is the generic term for any object that stands in for a real dependency in a test. The term comes from the film industry's "stunt double." Gerard Meszaros defined five distinct types, each serving a different purpose.

### 1.1 Dummy

A dummy is passed around but never actually used. It fills a required parameter that is irrelevant to the test.

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

The test does not care about logging. The dummy prevents a `TypeError` from a missing argument.

### 1.2 Stub

A stub provides canned answers to calls made during the test. It does not record anything — it just returns predetermined values.

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

Stubs answer the question: "If the dependency returns this value, does my code handle it correctly?"

### 1.3 Spy

A spy records information about how it was called, which you can inspect after the test. It may also delegate to the real implementation.

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

Spies answer: "Was my code called with the right arguments the right number of times?"

### 1.4 Mock

A mock is a spy with built-in expectations. It is pre-programmed with expected calls and fails the test if those expectations are not met. Python's `unittest.mock.Mock` is technically a spy-mock hybrid.

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

A fake is a working implementation that takes shortcuts unsuitable for production. Fakes are more realistic than stubs but simpler than real dependencies.

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

Fakes are the most powerful test double. They enable realistic integration-style tests without external dependencies.

### 1.6 Choosing the Right Test Double

| Test Double | Use When | Verification |
|---|---|---|
| **Dummy** | Parameter required but irrelevant | None |
| **Stub** | Need controlled return values | State of SUT |
| **Spy** | Need to verify interactions after the fact | Recorded calls |
| **Mock** | Need to verify specific interactions | Built-in expectations |
| **Fake** | Need realistic behavior without real infra | State of SUT + fake |

---

## 2. Arrange-Act-Assert (AAA)

The AAA pattern is the standard structure for unit tests. Every test has exactly three phases:

```python
def test_discount_applied_to_order():
    # Arrange — set up the test context
    catalog = FakeCatalog()
    catalog.add_product("widget", price=100.00)
    order = Order(catalog=catalog)
    order.add_item("widget", quantity=2)
    discount = PercentageDiscount(10)

    # Act — perform the action being tested
    order.apply_discount(discount)

    # Assert — verify the expected outcome
    assert order.total == 180.00  # 200 - 10%
    assert order.discount_amount == 20.00
```

### 2.1 AAA Guidelines

1. **One Act per test**: If you have multiple Act phases, split into multiple tests
2. **Arrange can be shared**: Use fixtures for common setup
3. **Assert on behavior, not implementation**: Check what happened, not how

```python
# BAD: Multiple acts (what exactly is being tested?)
def test_user_lifecycle():
    user = create_user("alice")     # Act 1
    assert user.id is not None
    update_user(user.id, "bob")     # Act 2
    assert get_user(user.id).name == "bob"
    delete_user(user.id)            # Act 3
    assert get_user(user.id) is None

# GOOD: One act per test
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

## 3. Given-When-Then (BDD Style)

Given-When-Then originates from Behavior-Driven Development (BDD). It is semantically identical to AAA but uses domain language.

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

### 3.1 When to Use Given-When-Then

Given-When-Then shines when:
- Tests describe business rules that non-developers should understand
- The test name reads like a specification
- You are practicing BDD with stakeholder collaboration

AAA is preferable when:
- Tests are purely technical (algorithm correctness, data structure behavior)
- The audience is exclusively developers

---

## 4. Builder Pattern for Test Data

Creating test objects with many fields is verbose and brittle. The Builder pattern provides a fluent API for constructing test data with sensible defaults.

### 4.1 The Problem

```python
# Without a builder — every test repeats all fields
def test_premium_user_gets_discount():
    user = User(
        name="alice",
        email="alice@example.com",
        age=30,
        country="US",
        membership="premium",    # Only this field matters
        created_at=datetime.now(),
        is_active=True,
        avatar_url=None,
        phone=None,
    )
    assert calculate_discount(user) == 0.15
```

### 4.2 The Solution

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


# Clean, focused tests
def test_premium_user_gets_discount():
    user = UserBuilder().with_membership("premium").build()
    assert calculate_discount(user) == 0.15

def test_inactive_users_get_no_discount():
    user = UserBuilder().with_membership("premium").inactive().build()
    assert calculate_discount(user) == 0.00
```

### 4.3 Builder as a pytest Fixture

```python
@pytest.fixture
def user_builder():
    return UserBuilder()


def test_premium_discount(user_builder):
    user = user_builder.with_membership("premium").build()
    assert calculate_discount(user) == 0.15
```

---

## 5. Page Object Pattern

The Page Object pattern abstracts UI or API interaction details behind a clean interface. Originally from Selenium testing, it applies equally to API testing.

### 5.1 Page Object for Web UI Testing

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


# Tests use page objects — no HTML parsing in test code
def test_successful_login_redirects_to_dashboard(client):
    login_page = LoginPage(client)
    response = login_page.login("alice", "correct_password")
    assert login_page.get_redirect_url(response) == "/dashboard"


def test_failed_login_shows_error(client):
    login_page = LoginPage(client)
    response = login_page.login("alice", "wrong_password")
    assert login_page.is_error_displayed(response)
```

### 5.2 Page Object for API Testing

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


# Clean tests focused on behavior
def test_create_and_retrieve_user(client):
    api = UserAPI(client)
    created = api.create("alice", "alice@example.com")
    retrieved = api.get(created["id"])
    assert retrieved["name"] == "alice"
    assert retrieved["email"] == "alice@example.com"
```

### 5.3 Benefits of Page Objects

1. **DRY**: API interaction logic is defined once
2. **Maintainable**: When the API changes, update one class, not 50 tests
3. **Readable**: Tests express intent, not HTTP mechanics
4. **Reusable**: Page objects compose into complex workflows

---

## 6. The Testing Pyramid in Practice

The testing pyramid is a guideline for how many tests to write at each level:

```
        ┌──────┐
        │  E2E │  Few (10-20)
       ┌┴──────┴┐
       │ Integr- │  Moderate (50-100)
       │ ation   │
      ┌┴─────────┴┐
      │    Unit    │  Many (500+)
      └────────────┘
```

### 6.1 Why This Shape?

| Level | Speed | Reliability | Maintenance | Confidence |
|---|---|---|---|---|
| Unit | Fast (ms) | Very stable | Low | Logic correctness |
| Integration | Medium (s) | Mostly stable | Medium | Components work together |
| E2E | Slow (min) | Often flaky | High | System works end-to-end |

### 6.2 The Pyramid in Practice

The ideal ratio depends on your system:

- **Library/Framework**: 90% unit, 10% integration, 0% E2E
- **Web API**: 60% unit, 30% integration, 10% E2E
- **Frontend App**: 40% unit, 30% integration, 30% E2E
- **Data Pipeline**: 30% unit, 60% integration, 10% E2E

### 6.3 Anti-Patterns

**The Ice Cream Cone** (inverted pyramid): Many E2E tests, few unit tests. Slow, flaky, expensive to maintain.

**The Hourglass**: Many unit tests, many E2E tests, few integration tests. Misses component interaction bugs.

**100% Unit Coverage**: Every function tested in isolation, but the system does not work because the components were never tested together.

---

## 7. Organizing Test Files

### 7.1 Mirror the Source Structure

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

### 7.2 Shared Test Utilities

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

## Exercises

1. **Test Double Classification**: Given five test scenarios (e.g., "test that an email is sent after registration"), identify which test double type is most appropriate for each and implement it.

2. **Builder Pattern**: Create a `ProductBuilder` and an `OrderBuilder` for an e-commerce domain. The `OrderBuilder` should accept `ProductBuilder` instances. Write three tests using the builders.

3. **Page Object Refactoring**: Take a set of API tests that make direct HTTP calls and refactor them to use Page Objects. Compare the before and after readability.

4. **Pyramid Audit**: Categorize all tests in an existing project into unit, integration, and E2E. Draw the resulting shape. Identify gaps and write tests to improve the balance.

5. **AAA Discipline**: Review a test file and identify any tests that violate the single-Act rule. Refactor them into properly structured AAA tests.

---

**License**: CC BY-NC 4.0
