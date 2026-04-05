# 11. API Testing and Validation

**Previous**: [API Documentation](./10_API_Documentation.md) | **Next**: [Webhooks and Callbacks](./12_Webhooks_and_Callbacks.md)

**Difficulty**: ⭐⭐⭐

---

## Learning Objectives

- Implement contract testing with Pact and Schemathesis to verify API behavior against specifications
- Design schema validation strategies that catch malformed requests before they reach business logic
- Apply fuzz testing to discover edge cases and vulnerabilities in API endpoints
- Write integration test patterns for multi-service API architectures
- Use API mocking to isolate components during testing
- Manage test environments and test data for reliable, repeatable API tests

---

## Table of Contents

1. [API Testing Pyramid](#1-api-testing-pyramid)
2. [Schema Validation](#2-schema-validation)
3. [Contract Testing](#3-contract-testing)
4. [Integration Testing Patterns](#4-integration-testing-patterns)
5. [Fuzz Testing](#5-fuzz-testing)
6. [API Mocking](#6-api-mocking)
7. [Test Environment Management](#7-test-environment-management)
8. [Exercises](#8-exercises)
9. [References](#9-references)

---

## 1. API Testing Pyramid

API testing spans multiple levels. Each level catches different categories of bugs and has different cost/speed characteristics.

```
                    ┌─────────────┐
                    │   E2E Tests  │  Slow, expensive, catch integration bugs
                    │  (few)       │
                   ┌┴─────────────┴┐
                   │  Contract Tests │  Medium speed, verify API agreements
                   │  (moderate)     │
                  ┌┴─────────────────┴┐
                  │  Integration Tests  │  Test service with real dependencies
                  │  (moderate)         │
                 ┌┴─────────────────────┴┐
                 │  Unit Tests             │  Fast, test individual functions
                 │  (many)                 │
                 └─────────────────────────┘
```

| Level | What It Tests | Speed | Reliability | Example |
|-------|--------------|-------|-------------|---------|
| Unit | Individual functions, validators, serializers | Fast | High | Test that `validate_isbn()` rejects invalid formats |
| Integration | Service + database, service + cache | Medium | Medium | Test that `POST /books` creates a row in the database |
| Contract | API shape matches consumer expectations | Medium | High | Test that response schema matches the OpenAPI spec |
| E2E | Full request flow through all services | Slow | Low | Test that creating an order triggers payment and notification |

---

## 2. Schema Validation

Schema validation ensures that incoming requests conform to the expected structure before reaching business logic. FastAPI does this automatically with Pydantic, but you can also validate responses.

### Request Validation with Pydantic

```python
from pydantic import BaseModel, Field, field_validator
from datetime import date


class BookCreate(BaseModel):
    """Validates incoming book creation requests.

    Pydantic automatically rejects requests that don't match
    this schema, returning a 422 Unprocessable Entity response.
    """

    title: str = Field(..., min_length=1, max_length=500)
    isbn: str = Field(..., pattern=r"^978-\d{10}$")
    publication_date: date
    price: float = Field(..., gt=0, le=10000)
    genres: list[str] = Field(..., min_length=1, max_length=5)

    @field_validator("genres")
    @classmethod
    def validate_genres(cls, v):
        allowed = {"fiction", "non-fiction", "science", "history", "biography"}
        invalid = set(v) - allowed
        if invalid:
            raise ValueError(
                f"Invalid genres: {invalid}. Allowed: {allowed}"
            )
        return v

    @field_validator("publication_date")
    @classmethod
    def validate_not_future(cls, v):
        if v > date.today():
            raise ValueError("Publication date cannot be in the future")
        return v
```

### Response Validation

Validate your own responses to catch serialization bugs before they reach clients:

```python
from pydantic import BaseModel, ConfigDict


class BookResponse(BaseModel):
    """Response model — also validates outgoing data.

    FastAPI uses this model to:
    1. Filter out fields not in the model (security)
    2. Validate that all required fields are present
    3. Generate the OpenAPI response schema
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    isbn: str
    price: float
    genres: list[str]


@app.get("/books/{book_id}", response_model=BookResponse)
async def get_book(book_id: int):
    book = await db.get(Book, book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    # Pydantic validates the response before sending it
    return book
```

### Testing Validation

```python
import pytest
from pydantic import ValidationError


class TestBookCreateValidation:
    """Test that the schema rejects invalid data."""

    def test_valid_book(self):
        book = BookCreate(
            title="Test Book",
            isbn="978-1234567890",
            publication_date=date(2025, 1, 1),
            price=29.99,
            genres=["fiction"],
        )
        assert book.title == "Test Book"

    def test_rejects_empty_title(self):
        with pytest.raises(ValidationError) as exc_info:
            BookCreate(
                title="",
                isbn="978-1234567890",
                publication_date=date(2025, 1, 1),
                price=29.99,
                genres=["fiction"],
            )
        assert "min_length" in str(exc_info.value)

    def test_rejects_invalid_isbn(self):
        with pytest.raises(ValidationError):
            BookCreate(
                title="Test",
                isbn="invalid-isbn",
                publication_date=date(2025, 1, 1),
                price=29.99,
                genres=["fiction"],
            )

    def test_rejects_negative_price(self):
        with pytest.raises(ValidationError):
            BookCreate(
                title="Test",
                isbn="978-1234567890",
                publication_date=date(2025, 1, 1),
                price=-5.0,
                genres=["fiction"],
            )

    def test_rejects_future_publication_date(self):
        with pytest.raises(ValidationError):
            BookCreate(
                title="Test",
                isbn="978-1234567890",
                publication_date=date(2099, 12, 31),
                price=29.99,
                genres=["fiction"],
            )

    def test_rejects_invalid_genre(self):
        with pytest.raises(ValidationError) as exc_info:
            BookCreate(
                title="Test",
                isbn="978-1234567890",
                publication_date=date(2025, 1, 1),
                price=29.99,
                genres=["horror"],  # Not in allowed set
            )
        assert "Invalid genres" in str(exc_info.value)
```

---

## 3. Contract Testing

Contract testing verifies that the API producer and consumer agree on the shape and behavior of the API. It catches breaking changes before they reach production.

### Schemathesis: Property-Based API Testing

Schemathesis generates test cases from your OpenAPI specification and tests every endpoint with random valid and invalid inputs.

```bash
# Install
pip install schemathesis

# Run against a live API
schemathesis run http://localhost:8000/openapi.json

# Run against a specific endpoint
schemathesis run http://localhost:8000/openapi.json \
    --endpoint /books \
    --method POST

# Run with more thorough testing
schemathesis run http://localhost:8000/openapi.json \
    --hypothesis-max-examples 500 \
    --stateful=links
```

### Schemathesis in Python Tests

```python
import schemathesis
from app.main import app

# Generate test cases from the FastAPI app's OpenAPI schema
schema = schemathesis.from_asgi("/openapi.json", app)


@schema.parametrize()
def test_api_conformance(case):
    """Property-based test: every valid request according to the schema
    should receive a response that matches the documented response schema.

    Schemathesis will:
    1. Generate random valid inputs matching parameter schemas
    2. Send requests to each endpoint
    3. Verify response status codes match documented ones
    4. Verify response bodies match documented schemas
    """
    response = case.call_asgi()
    case.validate_response(response)


@schema.parametrize(endpoint="/books", method="POST")
def test_book_creation_contract(case):
    """Focused contract test for book creation endpoint."""
    response = case.call_asgi()
    case.validate_response(response)

    if response.status_code == 201:
        data = response.json()
        assert "id" in data
        assert "title" in data
```

### Pact: Consumer-Driven Contract Testing

Pact lets API consumers define the contract. The consumer writes a test specifying what it expects from the provider, and the provider verifies it fulfills that contract.

```python
# Consumer side: define expected interactions
# tests/test_book_consumer.py
import atexit
import unittest
from pact import Consumer, Provider

pact = Consumer("BookshelfUI").has_pact_with(
    Provider("BookstoreAPI"),
    pact_dir="./pacts",
)
pact.start_service()
atexit.register(pact.stop_service)


class TestBookConsumer(unittest.TestCase):
    def test_get_book(self):
        """Define what the consumer expects from GET /books/1."""
        expected_body = {
            "id": 1,
            "title": "API Design Patterns",
            "isbn": "978-1617295850",
            "price": 49.99,
        }

        (
            pact
            .given("a book with ID 1 exists")
            .upon_receiving("a request for book 1")
            .with_request("GET", "/books/1")
            .will_respond_with(200, body=expected_body)
        )

        with pact:
            # Consumer code makes the actual request
            result = book_client.get_book(1)
            assert result["title"] == "API Design Patterns"


# Provider side: verify the pact
# tests/test_book_provider.py
from pact import Verifier


def test_provider_honors_pact():
    """Verify the provider fulfills all consumer contracts."""
    verifier = Verifier(
        provider="BookstoreAPI",
        provider_base_url="http://localhost:8000",
    )

    output, _ = verifier.verify_pacts(
        "./pacts/bookshelfui-bookstoreapi.json",
        provider_states_setup_url="http://localhost:8000/_pact/setup",
    )
    assert output == 0
```

### Contract Testing Workflow

```
Consumer Team                           Provider Team
     │                                       │
     │  1. Write consumer test               │
     │     (define expected interactions)     │
     │                                       │
     │  2. Generate pact file ──────────────>│
     │     (JSON contract)                   │
     │                                       │  3. Verify pact against
     │                                       │     actual implementation
     │                                       │
     │                                       │  4. Publish verification
     │  <────────────────────────────────────│     results
     │                                       │
     │  Both teams deploy with confidence    │
```

---

## 4. Integration Testing Patterns

### FastAPI TestClient

```python
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.database import get_db, Base


# Use an in-memory SQLite database for tests
SQLALCHEMY_TEST_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_TEST_URL)
TestSessionLocal = sessionmaker(bind=engine)


@pytest.fixture(autouse=True)
def setup_database():
    """Create tables before each test, drop after."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session():
    """Provide a transactional database session for tests."""
    session = TestSessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def client(db_session):
    """FastAPI test client with database override."""
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


class TestBookAPI:
    """Integration tests for the Book API."""

    def test_create_book(self, client):
        response = client.post(
            "/books",
            json={
                "title": "Test Book",
                "isbn": "978-1234567890",
                "publication_date": "2025-01-01",
                "price": 29.99,
                "genres": ["fiction"],
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["title"] == "Test Book"
        assert "id" in data

    def test_create_duplicate_isbn_returns_409(self, client):
        # Create first book
        client.post(
            "/books",
            json={
                "title": "Book A",
                "isbn": "978-1234567890",
                "publication_date": "2025-01-01",
                "price": 19.99,
                "genres": ["fiction"],
            },
        )

        # Attempt duplicate
        response = client.post(
            "/books",
            json={
                "title": "Book B",
                "isbn": "978-1234567890",
                "publication_date": "2025-06-01",
                "price": 24.99,
                "genres": ["non-fiction"],
            },
        )
        assert response.status_code == 409

    def test_get_book_not_found(self, client):
        response = client.get("/books/99999")
        assert response.status_code == 404

    def test_list_books_with_pagination(self, client):
        # Create 5 books
        for i in range(5):
            client.post(
                "/books",
                json={
                    "title": f"Book {i}",
                    "isbn": f"978-{i:010d}",
                    "publication_date": "2025-01-01",
                    "price": 10.0 + i,
                    "genres": ["fiction"],
                },
            )

        # Fetch first page
        response = client.get("/books?limit=2&offset=0")
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) == 2
        assert data["pagination"]["total"] == 5

    def test_validation_error_returns_422(self, client):
        response = client.post(
            "/books",
            json={
                "title": "",           # Too short
                "isbn": "invalid",     # Wrong format
                "price": -5,           # Negative
                "genres": [],          # Empty
            },
        )
        assert response.status_code == 422
        errors = response.json()
        assert "errors" in errors or "detail" in errors
```

### Async Integration Tests

```python
import pytest
import httpx
from app.main import app


@pytest.mark.anyio
async def test_create_and_retrieve_book():
    """Test the full lifecycle: create then retrieve."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        # Create
        create_response = await client.post(
            "/books",
            json={
                "title": "Async Test Book",
                "isbn": "978-9999999999",
                "publication_date": "2025-01-01",
                "price": 35.00,
                "genres": ["science"],
            },
        )
        assert create_response.status_code == 201
        book_id = create_response.json()["id"]

        # Retrieve
        get_response = await client.get(f"/books/{book_id}")
        assert get_response.status_code == 200
        assert get_response.json()["title"] == "Async Test Book"
```

---

## 5. Fuzz Testing

Fuzz testing sends random, malformed, or unexpected input to your API to discover crashes, unhandled exceptions, and security vulnerabilities.

### Schemathesis as a Fuzzer

Schemathesis can be configured to send boundary values, extremely long strings, and other edge cases:

```bash
# Aggressive fuzzing with many examples
schemathesis run http://localhost:8000/openapi.json \
    --hypothesis-max-examples 1000 \
    --hypothesis-seed 42 \
    --checks all \
    --validate-schema true
```

### Custom Fuzz Testing

```python
import pytest
import string
import random
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestFuzzing:
    """Fuzz tests to discover edge cases and crashes."""

    @pytest.mark.parametrize("title", [
        "",                          # Empty string
        " ",                         # Whitespace only
        "A" * 10000,                 # Very long string
        "'; DROP TABLE books;--",    # SQL injection
        "<script>alert(1)</script>", # XSS payload
        "null",                      # String "null"
        "\x00\x01\x02",             # Null bytes and control characters
        "🎉📚✨",                     # Unicode emoji
        "书名测试",                    # CJK characters
        "\n\r\t",                    # Whitespace characters
        "A" * 501,                   # Just over max length
    ])
    def test_title_edge_cases(self, title):
        """The API should handle unusual titles without crashing.

        It may reject them (422) or accept them, but it should
        never return a 500 Internal Server Error.
        """
        response = client.post(
            "/books",
            json={
                "title": title,
                "isbn": "978-1234567890",
                "publication_date": "2025-01-01",
                "price": 29.99,
                "genres": ["fiction"],
            },
        )
        assert response.status_code != 500, (
            f"Server crashed on title: {repr(title)}"
        )

    @pytest.mark.parametrize("book_id", [
        0,
        -1,
        2**31,       # 32-bit integer overflow
        2**63,       # 64-bit integer overflow
    ])
    def test_id_boundary_values(self, book_id):
        """Path parameters with extreme values should not crash the server."""
        response = client.get(f"/books/{book_id}")
        assert response.status_code in (400, 404, 422)

    @pytest.mark.parametrize("price", [
        0, 0.001, -0.01, 99999999.99, float("inf"), float("nan"),
    ])
    def test_price_edge_cases(self, price):
        """Numeric edge cases should be handled gracefully."""
        response = client.post(
            "/books",
            json={
                "title": "Price Test",
                "isbn": "978-1234567890",
                "publication_date": "2025-01-01",
                "price": price,
                "genres": ["fiction"],
            },
        )
        assert response.status_code != 500

    def test_random_json_bodies(self):
        """Send 100 random JSON bodies. None should cause a 500."""
        for _ in range(100):
            body = generate_random_json()
            response = client.post("/books", json=body)
            assert response.status_code != 500, (
                f"Server crashed on body: {body}"
            )


def generate_random_json(depth=0):
    """Generate a random JSON-compatible Python object."""
    if depth > 3:
        return random.choice([None, True, False, 42, "test"])

    choice = random.randint(0, 4)
    if choice == 0:
        return None
    elif choice == 1:
        return random.choice([True, False, random.randint(-1000, 1000)])
    elif choice == 2:
        return "".join(random.choices(string.printable, k=random.randint(0, 100)))
    elif choice == 3:
        return [generate_random_json(depth + 1) for _ in range(random.randint(0, 5))]
    else:
        return {
            "".join(random.choices(string.ascii_lowercase, k=5)): generate_random_json(depth + 1)
            for _ in range(random.randint(0, 5))
        }
```

---

## 6. API Mocking

Mocking lets you test components in isolation by replacing real dependencies with controlled substitutes.

### Mocking External Services

```python
from unittest.mock import AsyncMock, patch
import pytest
from fastapi.testclient import TestClient
from app.main import app


class TestPaymentIntegration:
    """Tests for endpoints that call external payment services."""

    @patch("app.services.payment.stripe_client")
    def test_create_payment_success(self, mock_stripe, client):
        """Mock Stripe to test the happy path without real charges."""
        mock_stripe.PaymentIntent.create = AsyncMock(
            return_value={
                "id": "pi_test_123",
                "status": "succeeded",
                "amount": 4999,
            }
        )

        response = client.post(
            "/orders/1/pay",
            json={"payment_method": "pm_card_visa"},
        )

        assert response.status_code == 200
        assert response.json()["payment_id"] == "pi_test_123"

    @patch("app.services.payment.stripe_client")
    def test_create_payment_declined(self, mock_stripe, client):
        """Mock a card decline to test error handling."""
        mock_stripe.PaymentIntent.create = AsyncMock(
            side_effect=Exception("Card declined")
        )

        response = client.post(
            "/orders/1/pay",
            json={"payment_method": "pm_card_declined"},
        )

        assert response.status_code == 402
        assert "declined" in response.json()["detail"].lower()
```

### FastAPI Dependency Override

```python
from fastapi import Depends
from app.services.email import EmailService


# Production dependency
async def get_email_service():
    return EmailService(smtp_host="smtp.example.com")


@app.post("/users")
async def create_user(
    user: UserCreate,
    email_service: EmailService = Depends(get_email_service),
):
    new_user = await save_user(user)
    await email_service.send_welcome(new_user.email)
    return new_user


# In tests: override the dependency
class FakeEmailService:
    """Records sent emails for assertion without sending real emails."""

    def __init__(self):
        self.sent_emails: list[dict] = []

    async def send_welcome(self, email: str):
        self.sent_emails.append({"to": email, "type": "welcome"})


@pytest.fixture
def fake_email():
    service = FakeEmailService()
    app.dependency_overrides[get_email_service] = lambda: service
    yield service
    app.dependency_overrides.clear()


def test_create_user_sends_welcome_email(client, fake_email):
    response = client.post(
        "/users",
        json={"name": "Alice", "email": "alice@example.com"},
    )
    assert response.status_code == 201
    assert len(fake_email.sent_emails) == 1
    assert fake_email.sent_emails[0]["to"] == "alice@example.com"
```

### Mock Server with responses Library

```python
import responses
import httpx


@responses.activate
def test_external_api_call():
    """Mock an external HTTP API call using the responses library."""
    responses.add(
        responses.GET,
        "https://api.weather.com/v1/current",
        json={"temperature": 22.5, "unit": "celsius"},
        status=200,
    )

    # Your code that calls the external API
    response = httpx.get(
        "https://api.weather.com/v1/current",
        params={"city": "Seoul"},
    )
    assert response.json()["temperature"] == 22.5


@responses.activate
def test_external_api_timeout():
    """Simulate a timeout from an external service."""
    responses.add(
        responses.GET,
        "https://api.weather.com/v1/current",
        body=ConnectionError("Connection timed out"),
    )

    with pytest.raises(ConnectionError):
        httpx.get("https://api.weather.com/v1/current")
```

---

## 7. Test Environment Management

### Test Configuration

```python
# app/config.py
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://user:pass@localhost/myapp"
    redis_url: str = "redis://localhost:6379/0"
    stripe_api_key: str = ""
    environment: str = "development"

    model_config = {"env_file": ".env"}


# tests/conftest.py
import pytest
from app.config import Settings


@pytest.fixture(scope="session")
def test_settings():
    """Override settings for the test environment."""
    return Settings(
        database_url="sqlite:///./test.db",
        redis_url="redis://localhost:6379/15",  # Dedicated test DB
        stripe_api_key="sk_test_fake",
        environment="testing",
    )
```

### Database Fixtures with Factories

```python
import factory
from app.models import Book, User
from datetime import date


class UserFactory(factory.Factory):
    """Generate realistic test users."""

    class Meta:
        model = User

    id = factory.Sequence(lambda n: n + 1)
    name = factory.Faker("name")
    email = factory.Faker("email")
    role = "user"


class BookFactory(factory.Factory):
    """Generate realistic test books."""

    class Meta:
        model = Book

    id = factory.Sequence(lambda n: n + 1)
    title = factory.Faker("sentence", nb_words=4)
    isbn = factory.LazyFunction(
        lambda: f"978-{factory.Faker('numerify', text='##########').evaluate(None, None, {'locale': 'en'})}"
    )
    price = factory.Faker("pydecimal", left_digits=2, right_digits=2, positive=True)
    publication_date = factory.Faker("date_between", start_date="-5y")
    genres = factory.LazyFunction(lambda: ["fiction"])


@pytest.fixture
def sample_books(db_session):
    """Create a set of test books in the database."""
    books = BookFactory.create_batch(10)
    for book in books:
        db_session.add(book)
    db_session.commit()
    return books
```

### Test Isolation

```python
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker


@pytest.fixture(scope="function")
def db_session(engine):
    """Each test runs in a transaction that is rolled back after the test.

    This ensures complete isolation between tests without the overhead
    of creating and dropping tables for each test.
    """
    connection = engine.connect()
    transaction = connection.begin()
    session = sessionmaker(bind=connection)()

    # Begin a nested transaction (savepoint)
    nested = connection.begin_nested()

    @event.listens_for(session, "after_transaction_end")
    def restart_savepoint(sess, trans):
        nonlocal nested
        if trans.nested and not trans._parent.nested:
            nested = connection.begin_nested()

    yield session

    session.close()
    transaction.rollback()
    connection.close()
```

---

## 8. Exercises

### Exercise 1: Comprehensive Test Suite

Write a complete test suite for a `POST /users` endpoint that:

- Creates a user with valid data (verify 201 response and response body)
- Rejects duplicate email addresses (verify 409 response)
- Rejects invalid email formats (verify 422 response with field-level error)
- Rejects passwords shorter than 12 characters
- Verifies the password is hashed (not stored in plaintext)
- Verifies a welcome email is sent (mock the email service)

Use pytest fixtures for database setup, test client, and dependency overrides.

### Exercise 2: Contract Test with Schemathesis

Set up Schemathesis to test a FastAPI application. Write a test file that:

1. Loads the OpenAPI schema from the FastAPI app
2. Runs property-based tests against all endpoints
3. Adds custom checks: no response should take longer than 500ms, no response body should exceed 1MB
4. Generates a test report showing which endpoints were tested and which failed

### Exercise 3: Fuzz Testing Campaign

Design a fuzz testing campaign for a user registration endpoint. Create parametrized tests for:

- Name field: empty strings, 10,000 characters, null bytes, SQL injection payloads, Unicode edge cases
- Email field: missing `@`, double `@`, 255+ character local part, IDN domains
- Password field: all spaces, all Unicode, control characters, 1MB string

For each test, assert that the response is never a 500 and always includes a machine-readable error format.

### Exercise 4: Mock External Service

Write tests for an endpoint that fetches weather data from an external API and caches it in Redis. Your tests should:

- Mock the external API to return controlled data
- Verify the caching behavior (second call should hit cache)
- Simulate a timeout from the external API and verify graceful degradation
- Simulate a malformed response and verify error handling

Use `unittest.mock.patch` and `fakeredis` for the Redis mock.

### Exercise 5: Integration Test Fixture System

Build a pytest fixture system for a multi-table application (users, books, orders, reviews) that:

- Creates all tables in a test database
- Provides factory functions for each model
- Supports transactional isolation (rollback after each test)
- Provides a pre-populated "seed" fixture with a realistic dataset of 50+ records
- Cleans up all resources after the test session

---

## 9. References

- [Schemathesis Documentation](https://schemathesis.readthedocs.io/)
- [Pact Documentation](https://docs.pact.io/)
- [FastAPI Testing Guide](https://fastapi.tiangolo.com/tutorial/testing/)
- [pytest Documentation](https://docs.pytest.org/)
- [Hypothesis (Property-Based Testing)](https://hypothesis.readthedocs.io/)
- [factory_boy Documentation](https://factoryboy.readthedocs.io/)
- [OWASP Testing Guide](https://owasp.org/www-project-web-security-testing-guide/)
- [responses Library](https://github.com/getsentry/responses)

---

**Previous**: [API Documentation](./10_API_Documentation.md) | [Overview](./00_Overview.md) | **Next**: [Webhooks and Callbacks](./12_Webhooks_and_Callbacks.md)

**License**: CC BY-NC 4.0
