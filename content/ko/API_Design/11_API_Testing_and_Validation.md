# 11. API 테스팅과 검증

**이전**: [API 문서화](./10_API_Documentation.md) | **다음**: [Webhooks와 Callbacks](./12_Webhooks_and_Callbacks.md)

**난이도**: ⭐⭐⭐

---

## 학습 목표

- Pact와 Schemathesis를 사용한 계약 테스팅을 구현하여 API 동작을 명세 대비 검증하기
- 비즈니스 로직에 도달하기 전에 잘못된 요청을 잡아내는 스키마 검증 전략 설계하기
- Fuzz 테스팅을 적용하여 API 엔드포인트의 엣지 케이스와 취약점 발견하기
- 멀티서비스 API 아키텍처를 위한 통합 테스트 패턴 작성하기
- API 모킹을 사용하여 테스트 시 컴포넌트를 격리하기
- 신뢰할 수 있고 반복 가능한 API 테스트를 위한 테스트 환경과 테스트 데이터 관리하기

---

## 목차

1. [API 테스팅 피라미드](#1-api-테스팅-피라미드)
2. [스키마 검증](#2-스키마-검증)
3. [계약 테스팅](#3-계약-테스팅)
4. [통합 테스팅 패턴](#4-통합-테스팅-패턴)
5. [Fuzz 테스팅](#5-fuzz-테스팅)
6. [API 모킹](#6-api-모킹)
7. [테스트 환경 관리](#7-테스트-환경-관리)
8. [연습 문제](#8-연습-문제)
9. [참고 자료](#9-참고-자료)

---

## 1. API 테스팅 피라미드

API 테스팅은 여러 수준에 걸쳐 이루어집니다. 각 수준은 서로 다른 범주의 버그를 잡아내며 비용/속도 특성이 다릅니다.

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

| 수준 | 테스트 대상 | 속도 | 신뢰성 | 예시 |
|-------|--------------|-------|-------------|---------|
| Unit | 개별 함수, 검증기, 직렬화기 | 빠름 | 높음 | `validate_isbn()`이 잘못된 형식을 거부하는지 테스트 |
| Integration | 서비스 + 데이터베이스, 서비스 + 캐시 | 중간 | 중간 | `POST /books`가 데이터베이스에 행을 생성하는지 테스트 |
| Contract | API 형태가 소비자 기대와 일치 | 중간 | 높음 | 응답 스키마가 OpenAPI 명세와 일치하는지 테스트 |
| E2E | 모든 서비스를 통한 전체 요청 흐름 | 느림 | 낮음 | 주문 생성이 결제와 알림을 트리거하는지 테스트 |

---

## 2. 스키마 검증

스키마 검증은 수신 요청이 비즈니스 로직에 도달하기 전에 예상된 구조를 따르는지 확인합니다. FastAPI는 Pydantic으로 이를 자동으로 수행하지만, 응답도 검증할 수 있습니다.

### Pydantic을 사용한 요청 검증

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

### 응답 검증

자체 응답을 검증하여 클라이언트에 도달하기 전에 직렬화 버그를 잡아내십시오:

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

### 검증 테스팅

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

## 3. 계약 테스팅

계약 테스팅은 API 제공자와 소비자가 API의 형태와 동작에 대해 합의하고 있는지 검증합니다. 프로덕션에 도달하기 전에 호환성 깨짐(breaking changes)을 잡아냅니다.

### Schemathesis: 속성 기반 API 테스팅

Schemathesis는 OpenAPI 명세에서 테스트 케이스를 생성하고 무작위 유효/무효 입력으로 모든 엔드포인트를 테스트합니다.

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

### Python 테스트에서 Schemathesis 사용

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

### Pact: 소비자 주도 계약 테스팅

Pact는 API 소비자가 계약을 정의하도록 합니다. 소비자가 제공자에게 기대하는 것을 명시하는 테스트를 작성하고, 제공자는 해당 계약을 이행하는지 검증합니다.

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

### 계약 테스팅 워크플로우

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

## 4. 통합 테스팅 패턴

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

### 비동기 통합 테스트

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

## 5. Fuzz 테스팅

Fuzz 테스팅은 무작위, 잘못된 형식, 또는 예상치 못한 입력을 API에 보내 크래시, 처리되지 않은 예외, 보안 취약점을 발견합니다.

### Fuzzer로서의 Schemathesis

Schemathesis는 경계값, 극단적으로 긴 문자열, 기타 엣지 케이스를 보내도록 설정할 수 있습니다:

```bash
# Aggressive fuzzing with many examples
schemathesis run http://localhost:8000/openapi.json \
    --hypothesis-max-examples 1000 \
    --hypothesis-seed 42 \
    --checks all \
    --validate-schema true
```

### 커스텀 Fuzz 테스팅

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

## 6. API 모킹

모킹은 실제 의존성을 통제된 대체물로 교체하여 컴포넌트를 격리된 상태로 테스트할 수 있게 합니다.

### 외부 서비스 모킹

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

### responses 라이브러리를 사용한 Mock 서버

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

## 7. 테스트 환경 관리

### 테스트 설정

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

### 팩토리를 사용한 데이터베이스 픽스처

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

### 테스트 격리

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

## 8. 연습 문제

### 연습 문제 1: 종합 테스트 스위트

`POST /users` 엔드포인트에 대한 완전한 테스트 스위트를 작성하십시오:

- 유효한 데이터로 사용자 생성 (201 응답 및 응답 본문 검증)
- 중복 이메일 주소 거부 (409 응답 검증)
- 잘못된 이메일 형식 거부 (필드 수준 오류가 포함된 422 응답 검증)
- 12자 미만의 비밀번호 거부
- 비밀번호가 해시 처리되었는지 검증 (평문으로 저장되지 않음)
- 환영 이메일이 전송되었는지 검증 (이메일 서비스 모킹)

데이터베이스 설정, 테스트 클라이언트, dependency override를 위한 pytest 픽스처를 사용하십시오.

### 연습 문제 2: Schemathesis를 사용한 계약 테스트

FastAPI 애플리케이션을 테스트하기 위해 Schemathesis를 설정하십시오. 다음을 수행하는 테스트 파일을 작성하십시오:

1. FastAPI 앱에서 OpenAPI 스키마 로드
2. 모든 엔드포인트에 대해 속성 기반 테스트 실행
3. 커스텀 체크 추가: 응답이 500ms를 초과하지 않아야 하고, 응답 본문이 1MB를 초과하지 않아야 함
4. 어떤 엔드포인트가 테스트되었고 어떤 것이 실패했는지 보여주는 테스트 리포트 생성

### 연습 문제 3: Fuzz 테스팅 캠페인

사용자 등록 엔드포인트를 위한 fuzz 테스팅 캠페인을 설계하십시오. 다음에 대한 파라미터화된 테스트를 작성하십시오:

- Name 필드: 빈 문자열, 10,000자, null 바이트, SQL 인젝션 페이로드, Unicode 엣지 케이스
- Email 필드: `@` 누락, 이중 `@`, 255자 이상의 local part, IDN 도메인
- Password 필드: 모두 공백, 모두 Unicode, 제어 문자, 1MB 문자열

각 테스트에 대해 응답이 절대 500이 아니고 항상 기계 판독 가능한 오류 형식을 포함하는지 확인하십시오.

### 연습 문제 4: 외부 서비스 모킹

외부 API에서 날씨 데이터를 가져오고 Redis에 캐시하는 엔드포인트에 대한 테스트를 작성하십시오. 테스트는 다음을 수행해야 합니다:

- 외부 API를 모킹하여 통제된 데이터 반환
- 캐싱 동작 검증 (두 번째 호출은 캐시에서 가져와야 함)
- 외부 API의 타임아웃을 시뮬레이션하고 점진적 성능 저하 검증
- 잘못된 형식의 응답을 시뮬레이션하고 오류 처리 검증

Redis 모킹에는 `unittest.mock.patch`와 `fakeredis`를 사용하십시오.

### 연습 문제 5: 통합 테스트 픽스처 시스템

멀티 테이블 애플리케이션(users, books, orders, reviews)을 위한 pytest 픽스처 시스템을 구축하십시오:

- 테스트 데이터베이스에 모든 테이블 생성
- 각 모델에 대한 팩토리 함수 제공
- 트랜잭션 격리 지원 (각 테스트 후 롤백)
- 50개 이상의 레코드로 구성된 현실적인 데이터셋의 사전 채워진 "seed" 픽스처 제공
- 테스트 세션 후 모든 리소스 정리

---

## 9. 참고 자료

- [Schemathesis Documentation](https://schemathesis.readthedocs.io/)
- [Pact Documentation](https://docs.pact.io/)
- [FastAPI Testing Guide](https://fastapi.tiangolo.com/tutorial/testing/)
- [pytest Documentation](https://docs.pytest.org/)
- [Hypothesis (Property-Based Testing)](https://hypothesis.readthedocs.io/)
- [factory_boy Documentation](https://factoryboy.readthedocs.io/)
- [OWASP Testing Guide](https://owasp.org/www-project-web-security-testing-guide/)
- [responses Library](https://github.com/getsentry/responses)

---

**이전**: [API 문서화](./10_API_Documentation.md) | [개요](./00_Overview.md) | **다음**: [Webhooks와 Callbacks](./12_Webhooks_and_Callbacks.md)

**License**: CC BY-NC 4.0
