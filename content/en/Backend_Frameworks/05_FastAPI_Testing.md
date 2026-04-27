# 05. FastAPI Testing

**Previous**: [FastAPI Database](./04_FastAPI_Database.md) | **Next**: [Express Basics](./06_Express_Basics.md)

**Difficulty**: ⭐⭐⭐

---

## Learning Objectives

- Write synchronous tests using FastAPI's `TestClient` and asynchronous tests using `httpx.AsyncClient`
- Design pytest fixtures and `conftest.py` for reusable test setup with proper isolation
- Implement database test fixtures that create isolated test databases with transaction rollback
- Override FastAPI dependencies to inject mocks and test doubles during testing
- Configure test coverage reporting to identify untested code paths

---

## Table of Contents

1. [TestClient for Synchronous Testing](#1-testclient-for-synchronous-testing)
2. [httpx.AsyncClient for Async Testing](#2-httpxasyncclient-for-async-testing)
3. [pytest Fixtures and conftest.py](#3-pytest-fixtures-and-conftestpy)
4. [Database Fixtures](#4-database-fixtures)
5. [Mocking Dependencies with dependency_overrides](#5-mocking-dependencies-with-dependency_overrides)
6. [Testing Authentication Flows](#6-testing-authentication-flows)
7. [Coverage Reporting](#7-coverage-reporting)
8. [Practice Problems](#8-practice-problems)
9. [References](#9-references)

---

## 1. TestClient for Synchronous Testing

FastAPI's `TestClient` wraps `httpx` to send requests to your app without running a real server. It is synchronous, so you do not need `async/await` in your tests.

### Theory: The In-Process ASGI Transport

In production, an HTTP request travels: client → kernel → TCP → kernel → uvicorn → ASGI → FastAPI → handler. In tests, that whole chain is overkill. The `TestClient` (and `httpx.AsyncClient` with `ASGITransport`) collapses it: client → ASGI dict → FastAPI → handler. Same code path inside FastAPI; no socket involved.

#### A.1 What TestClient actually does

`TestClient(app)` is `httpx.Client` configured with a special `WSGITransport`/`ASGITransport` that does not open a network socket. When you call `client.get("/items/")`, the transport:

1. Builds an ASGI `scope` dict from the URL, method, and headers (the same shape from Lesson 02 §A.1).
2. Calls `await app(scope, receive, send)` directly, where `receive` yields the request body and `send` collects the response messages.
3. Reassembles the collected messages into an `httpx.Response` object.

The whole exchange runs in the test process, in memory. Round-trip latency is microseconds, not milliseconds. There is no port to allocate, no listener thread, no TLS handshake.

#### A.2 Why this matters for fidelity

The same FastAPI application instance handles the request. All middleware, all `Depends`, all serialization runs. The only thing skipped is the *network*. That makes ASGI in-process tests genuine integration tests — a passing test means the request would also succeed in production, modulo network/TLS issues.

The cost: tests cannot exercise behaviors that depend on the network layer. If you need to test reconnect logic, real timeouts, or proxy behavior, you need a real server. For 99% of API tests, in-process is what you want.

#### A.3 Sync vs async clients: the same transport

`TestClient` is the sync wrapper; under the hood it runs the async app on an internal event loop. For tests of `async def` handlers that await fixtures, you usually want `httpx.AsyncClient` directly with `ASGITransport(app=app)` — that way both your test code and the handler share the same event loop, and you can `await` fixture resources naturally.

The choice is ergonomic, not performance: both flow through the same ASGI mechanism.

### Basic Setup

```python
# tests/test_basic.py
from fastapi import FastAPI
from fastapi.testclient import TestClient

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello, World!"}

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    return {"user_id": user_id, "name": f"User {user_id}"}

# TestClient sends requests to the app in-process (no network I/O)
client = TestClient(app)

def test_root():
    """Test the root endpoint returns 200 with expected message."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello, World!"}

def test_get_user():
    """Test path parameter parsing and response structure."""
    response = client.get("/users/42")
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == 42
    assert data["name"] == "User 42"

def test_get_user_invalid_id():
    """FastAPI returns 422 when path parameter type validation fails.
    Sending a string where an int is expected triggers automatic validation."""
    response = client.get("/users/not-a-number")
    assert response.status_code == 422
    # The error body includes details about which field failed
    assert "detail" in response.json()
```

### Testing POST Requests

```python
from pydantic import BaseModel
from fastapi import status

class ItemCreate(BaseModel):
    name: str
    price: float

@app.post("/items", status_code=status.HTTP_201_CREATED)
async def create_item(item: ItemCreate):
    return {"id": 1, **item.model_dump()}

def test_create_item():
    """Test creating an item with valid data."""
    response = client.post(
        "/items",
        json={"name": "Widget", "price": 9.99},  # json= auto-sets Content-Type
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Widget"
    assert data["price"] == 9.99
    assert "id" in data

def test_create_item_missing_field():
    """Test that missing required fields return 422 with field details."""
    response = client.post("/items", json={"name": "Widget"})
    assert response.status_code == 422
    errors = response.json()["detail"]
    # Find the error for the missing 'price' field
    price_error = next(e for e in errors if "price" in e["loc"])
    assert price_error["type"] == "missing"

def test_create_item_invalid_type():
    """Test that wrong types return 422."""
    response = client.post(
        "/items",
        json={"name": "Widget", "price": "not-a-number"},
    )
    assert response.status_code == 422
```

### Testing Headers and Query Parameters

```python
def test_with_headers():
    """Test endpoints that require specific headers."""
    response = client.get(
        "/protected",
        headers={"X-API-Key": "valid-key-123"},
    )
    assert response.status_code == 200

def test_with_query_params():
    """Query parameters are passed as a dict to the params argument."""
    response = client.get(
        "/users",
        params={"skip": 0, "limit": 5, "role": "admin"},
    )
    assert response.status_code == 200
```

---

## 2. httpx.AsyncClient for Async Testing

When your endpoints use `async` dependencies (like async database sessions), you need an async test client. The `TestClient` works for many cases, but `httpx.AsyncClient` gives you true async testing.

### Setup

```bash
pip install httpx pytest-asyncio
```

### Async Test Examples

```python
# tests/test_async.py
import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app

# pytest-asyncio provides the async test infrastructure
# mode="auto" means all async test functions are automatically
# treated as async tests without needing @pytest.mark.asyncio on each one
pytestmark = pytest.mark.anyio

@pytest.fixture
async def async_client():
    """Create an async test client.
    ASGITransport sends requests directly to the ASGI app
    without starting a real server or making network calls."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

async def test_root(async_client: AsyncClient):
    """Async test -- uses await for the request."""
    response = await async_client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello, World!"}

async def test_create_and_get_user(async_client: AsyncClient):
    """Test a full create-then-read workflow.
    Async tests can naturally chain multiple await calls."""
    # Create
    create_response = await async_client.post(
        "/api/users",
        json={"username": "testuser", "email": "test@example.com", "password": "secret123"},
    )
    assert create_response.status_code == 201
    user_id = create_response.json()["id"]

    # Read back
    get_response = await async_client.get(f"/api/users/{user_id}")
    assert get_response.status_code == 200
    assert get_response.json()["username"] == "testuser"
```

### When to Use Which

| Scenario | Use `TestClient` | Use `AsyncClient` |
|----------|-----------------|-------------------|
| Simple endpoint tests | Yes | Yes |
| Async database operations | Works (wraps async) | Better (native async) |
| WebSocket testing | `TestClient` has built-in support | Manual |
| Speed | Slightly faster | Slightly slower |
| Debugging | Easier (synchronous stack) | Harder (async stack) |

---

## 3. pytest Fixtures and conftest.py

Fixtures are reusable test setup/teardown functions. `conftest.py` makes fixtures available to all tests in the directory without importing them.

### Theory: Fixture Scopes: A Lifecycle Hierarchy

pytest fixtures have four built-in scopes, ordered from narrowest to widest:

| Scope | Built when | Torn down when | Cost amortized over |
|-------|------------|----------------|---------------------|
| `function` (default) | Each test starts | Each test ends | One test |
| `class` | First test in a class needs it | Last test in the class ends | All tests in the class |
| `module` | First test in a file needs it | Last test in the file ends | All tests in the file |
| `session` | First test in the run needs it | The whole run ends | The entire test suite |

The choice is a tradeoff between **isolation** and **speed**:

- Building an entire SQLAlchemy engine on `function` scope is correct but extremely slow — you pay engine setup × N tests.
- Building it on `session` scope is fast but risks state leaking between tests if you do not reset it.

The conventional pattern: heavyweight, read-only resources at `session` (the engine, the FastAPI app instance). Mutable per-test state at `function` (a database session, an HTTP client, the dependency override). This minimizes setup cost while keeping each test's mutations invisible to its neighbors.

#### B.1 Fixture composition

A fixture can request other fixtures by name:

```python
@pytest.fixture(scope="session")
def engine(): ...

@pytest.fixture(scope="function")
def db_session(engine): ...

@pytest.fixture(scope="function")
def client(db_session): ...
```

pytest builds a DAG (just like the FastAPI Depends DAG from Lesson 03), respects scope (`function` cannot depend on a wider scope being narrower than itself, but the reverse is fine), and tears down in reverse build order.

#### B.2 `yield` for setup/teardown

The `yield` pattern in fixtures matches the Depends pattern from Lesson 03:

```python
@pytest.fixture
def db_session(engine):
    connection = engine.connect()
    transaction = connection.begin()
    session = SessionLocal(bind=connection)
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()
```

Setup before yield, teardown after. The teardown runs even if the test fails. This is the foundation of test isolation in §C.

### Project Structure

```
tests/
├── conftest.py         # Shared fixtures (available to all tests)
├── test_users.py       # User endpoint tests
├── test_posts.py       # Post endpoint tests
└── test_auth.py        # Authentication tests
```

### conftest.py

```python
# tests/conftest.py
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.main import app
from app.models import Base
from app.dependencies import get_db

# Use an in-memory SQLite database for tests
# This is fast and isolated -- each test run starts fresh
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

@pytest.fixture(scope="session")
def anyio_backend():
    """Tell pytest-asyncio to use asyncio (not trio)."""
    return "asyncio"

@pytest.fixture(scope="session")
async def test_engine():
    """Create a test database engine once per test session.
    scope='session' means this runs once, not per-test."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)

    # Create all tables defined in your models
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Teardown: drop all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()

@pytest.fixture
async def db_session(test_engine):
    """Create a new database session for each test.
    Uses a transaction that is rolled back after the test,
    ensuring complete isolation between tests."""
    session_factory = async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with session_factory() as session:
        # Begin a nested transaction (savepoint)
        async with session.begin():
            yield session
            # Rollback after test -- no data persists between tests
            await session.rollback()

@pytest.fixture
async def async_client(db_session: AsyncSession):
    """Create an async test client with the test database injected.
    This overrides the real get_db dependency so tests use the
    test database instead of the production one."""

    async def override_get_db():
        yield db_session

    # Replace the real database dependency with our test version
    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    # Clean up: remove the override so it doesn't affect other tests
    app.dependency_overrides.clear()

@pytest.fixture
def sample_user_data() -> dict:
    """Reusable test data. Using a fixture instead of a constant
    ensures each test gets a fresh copy (dicts are mutable)."""
    return {
        "username": "testuser",
        "email": "testuser@example.com",
        "password": "securepassword123",
        "full_name": "Test User",
    }
```

### Fixture Scopes

| Scope | Created | Destroyed | Use Case |
|-------|---------|-----------|----------|
| `function` (default) | Per test function | After test ends | DB sessions, test data |
| `class` | Per test class | After class ends | Shared class setup |
| `module` | Per test module | After module ends | Expensive setup per file |
| `session` | Once per test run | After all tests | Engine, connection pool |

---

## 4. Database Fixtures

### Theory: Test Isolation: The Three Mechanisms

The hardest property to maintain in a test suite is **isolation**: test B should not see anything test A did. Three mechanisms work together to provide it.

#### C.1 Transactional rollback

Instead of dropping and recreating the database for every test, wrap each test in a transaction that is rolled back at the end:

```
BEGIN
  ... test runs, INSERTs, UPDATEs ...
ROLLBACK
```

The database goes back to its starting state, but you never paid for `CREATE TABLE` per test. This is the standard pattern with SQLAlchemy: pin the session to a connection, begin a transaction, hand the session to the test, roll back when done.

The subtlety: the test code must not call `session.commit()` directly, because that would close the outer transaction. The fix is a **savepoint** (`SAVEPOINT` / `ROLLBACK TO SAVEPOINT`) — a nested transaction. SQLAlchemy provides this with `connection.begin_nested()` and a hook that re-opens a new savepoint after every commit. The test sees normal commit semantics; the outer transaction still gets rolled back.

#### C.2 dependency_overrides for fakes

`app.dependency_overrides[get_db] = override_get_db` (from Lesson 03 §A.4) lets the test substitute a fake implementation of any dependency. For database tests, the override returns the test's transactional session. For external API tests, it returns a fake client. The handler is unmodified, but it now receives the test's controlled doubles.

The override must be set up *before* `TestClient` makes its first request, and torn down after — usually in a fixture's setup/teardown. Otherwise an override leaks into the next test.

#### C.3 Database per worker, not per test

When tests run in parallel (`pytest-xdist`), all workers must not share one SQLite file or one PostgreSQL schema — they will trample each other. The pattern is **one database per worker**: pytest exposes `worker_id` ("gw0", "gw1", ...), and you suffix the database name with it. Within a worker, tests still serialize, so transactional rollback still works.

The combination — session-scoped engine per worker, function-scoped transactional session, dependency-overridden handlers — gives the right balance: each test is isolated, the suite runs in seconds, and parallelism scales linearly with worker count.

### Strategy: Transaction Rollback

The most effective pattern for database test isolation is **transaction rollback**: wrap each test in a transaction and roll it back afterward. This is faster than creating/dropping tables for each test.

```
Test Start
    │
    ▼
┌─────────────────────────┐
│  BEGIN TRANSACTION       │  ◀── Savepoint created
│                          │
│  INSERT user ...         │  ◀── Test creates data
│  SELECT * FROM users ... │  ◀── Test reads data
│  assert len(users) == 1  │  ◀── Test passes
│                          │
│  ROLLBACK                │  ◀── All changes undone
└─────────────────────────┘
    │
    ▼
Test End (database unchanged)
```

### Seeding Test Data

```python
# tests/conftest.py (continued)
from app.models import User, Post
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

@pytest.fixture
async def sample_user(db_session: AsyncSession) -> User:
    """Create a user in the test database.
    This fixture depends on db_session, so the user is
    automatically rolled back after the test."""
    user = User(
        username="alice",
        email="alice@example.com",
        hashed_password=pwd_context.hash("password123"),
        full_name="Alice Johnson",
    )
    db_session.add(user)
    await db_session.flush()  # Generate the ID
    await db_session.refresh(user)
    return user

@pytest.fixture
async def sample_posts(db_session: AsyncSession, sample_user: User) -> list[Post]:
    """Create multiple posts for testing list/filter endpoints.
    Depends on sample_user to satisfy the foreign key constraint."""
    posts = [
        Post(title="First Post", content="Content 1", author_id=sample_user.id, is_published=True),
        Post(title="Second Post", content="Content 2", author_id=sample_user.id, is_published=True),
        Post(title="Draft Post", content="Content 3", author_id=sample_user.id, is_published=False),
    ]
    db_session.add_all(posts)
    await db_session.flush()
    for post in posts:
        await db_session.refresh(post)
    return posts
```

### Using Data Fixtures in Tests

```python
# tests/test_posts.py
import pytest

pytestmark = pytest.mark.anyio

async def test_list_published_posts(async_client, sample_posts):
    """sample_posts fixture creates 3 posts (2 published, 1 draft).
    The list endpoint should only return published posts."""
    response = await async_client.get("/api/posts?published=true")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert all(post["is_published"] for post in data)

async def test_get_nonexistent_post(async_client):
    """Without the sample_posts fixture, the database is empty.
    Requesting a non-existent post should return 404."""
    response = await async_client.get("/api/posts/999")
    assert response.status_code == 404
```

---

## 5. Mocking Dependencies with dependency_overrides

`dependency_overrides` is FastAPI's built-in mechanism for replacing dependencies during tests. It replaces a dependency callable with a test double.

### Basic Override

```python
from app.dependencies import get_current_user
from app.models import User

# Create a fake user that bypasses real authentication
mock_user = User(
    id=1,
    username="testadmin",
    email="admin@test.com",
    hashed_password="not-a-real-hash",
    is_active=True,
)

async def mock_get_current_user():
    """Return a fake user without checking JWT tokens.
    This lets us test protected endpoints without dealing
    with real authentication in every test."""
    return mock_user

# Override before tests run
app.dependency_overrides[get_current_user] = mock_get_current_user
```

### Using Overrides in Fixtures

```python
# tests/conftest.py
import pytest
from app.main import app
from app.dependencies import get_current_user, get_current_active_user

@pytest.fixture
def authenticated_client(async_client, sample_user):
    """Client that automatically authenticates as sample_user.
    Overrides the auth dependency so protected endpoints work."""

    async def override_auth():
        return sample_user

    app.dependency_overrides[get_current_active_user] = override_auth
    yield async_client
    # Clean up after the test
    del app.dependency_overrides[get_current_active_user]

@pytest.fixture
def admin_client(async_client):
    """Client authenticated as an admin user."""

    async def override_auth():
        return User(
            id=99,
            username="admin",
            email="admin@test.com",
            hashed_password="",
            is_active=True,
            role="admin",
        )

    app.dependency_overrides[get_current_active_user] = override_auth
    yield async_client
    del app.dependency_overrides[get_current_active_user]
```

### Testing with Overrides

```python
# tests/test_protected.py
pytestmark = pytest.mark.anyio

async def test_get_my_profile(authenticated_client):
    """Test the /users/me endpoint with mocked authentication."""
    response = await authenticated_client.get("/api/users/me")
    assert response.status_code == 200
    assert response.json()["username"] == "alice"

async def test_admin_only_endpoint(admin_client):
    """Test that admin endpoints work with admin override."""
    response = await admin_client.get("/api/admin/stats")
    assert response.status_code == 200

async def test_unauthorized_without_override(async_client):
    """Without the auth override, protected endpoints return 401."""
    response = await async_client.get("/api/users/me")
    assert response.status_code == 401
```

### Overriding External Services

```python
from app.dependencies import get_email_service

class MockEmailService:
    """Mock that records sent emails instead of sending them.
    Useful for asserting that the right emails were triggered."""
    def __init__(self):
        self.sent_emails: list[dict] = []

    async def send(self, to: str, subject: str, body: str):
        self.sent_emails.append({"to": to, "subject": subject, "body": body})

@pytest.fixture
def mock_email(async_client):
    """Inject a mock email service and expose it for assertions."""
    service = MockEmailService()

    async def override():
        return service

    app.dependency_overrides[get_email_service] = override
    yield service
    del app.dependency_overrides[get_email_service]

async def test_registration_sends_welcome_email(async_client, mock_email):
    """Verify that user registration triggers a welcome email."""
    await async_client.post(
        "/api/users",
        json={"username": "newuser", "email": "new@test.com", "password": "secret123"},
    )
    assert len(mock_email.sent_emails) == 1
    assert mock_email.sent_emails[0]["to"] == "new@test.com"
    assert "welcome" in mock_email.sent_emails[0]["subject"].lower()
```

---

## 6. Testing Authentication Flows

### Testing the Login Flow End-to-End

```python
# tests/test_auth.py
import pytest

pytestmark = pytest.mark.anyio

async def test_login_success(async_client, sample_user):
    """Test the full login flow: send credentials, receive JWT."""
    # OAuth2 password flow uses form data, not JSON
    response = await async_client.post(
        "/token",
        data={  # data= sends form-encoded, not JSON
            "username": "alice",
            "password": "password123",
        },
    )
    assert response.status_code == 200
    token_data = response.json()
    assert "access_token" in token_data
    assert token_data["token_type"] == "bearer"

async def test_login_wrong_password(async_client, sample_user):
    """Wrong password should return 401, not 200 with an error body."""
    response = await async_client.post(
        "/token",
        data={"username": "alice", "password": "wrongpassword"},
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Incorrect username or password"

async def test_login_nonexistent_user(async_client):
    """Non-existent user should return the same error as wrong password
    to prevent username enumeration attacks."""
    response = await async_client.post(
        "/token",
        data={"username": "nonexistent", "password": "anything"},
    )
    assert response.status_code == 401

async def test_protected_endpoint_with_token(async_client, sample_user):
    """Full integration test: login, get token, use it to access protected route."""
    # Step 1: Login
    login_response = await async_client.post(
        "/token",
        data={"username": "alice", "password": "password123"},
    )
    token = login_response.json()["access_token"]

    # Step 2: Access protected endpoint with the token
    response = await async_client.get(
        "/api/users/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.json()["username"] == "alice"

async def test_expired_token(async_client):
    """Test that expired tokens are properly rejected."""
    # Create a token that expired in the past
    from app.auth import create_access_token
    from datetime import timedelta

    expired_token = create_access_token(
        data={"sub": "alice"},
        expires_delta=timedelta(seconds=-1),  # Already expired
    )
    response = await async_client.get(
        "/api/users/me",
        headers={"Authorization": f"Bearer {expired_token}"},
    )
    assert response.status_code == 401

async def test_malformed_token(async_client):
    """Completely invalid tokens should return 401."""
    response = await async_client.get(
        "/api/users/me",
        headers={"Authorization": "Bearer not.a.valid.jwt.token"},
    )
    assert response.status_code == 401

async def test_missing_auth_header(async_client):
    """No Authorization header at all should return 401."""
    response = await async_client.get("/api/users/me")
    assert response.status_code == 401
```

### Helper Fixture for Authenticated Requests

```python
# tests/conftest.py
@pytest.fixture
async def auth_headers(async_client, sample_user) -> dict[str, str]:
    """Login as sample_user and return auth headers.
    Use this fixture when you need real JWT tokens, not mocked auth."""
    response = await async_client.post(
        "/token",
        data={"username": "alice", "password": "password123"},
    )
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

# Usage in tests:
async def test_create_post_authenticated(async_client, auth_headers):
    response = await async_client.post(
        "/api/posts",
        json={"title": "My Post", "content": "Hello world"},
        headers=auth_headers,
    )
    assert response.status_code == 201
```

---

## 7. Coverage Reporting

Code coverage tells you which lines of code are executed during your tests. It helps identify untested paths, but 100% coverage does not guarantee correctness.

### Setup

```bash
pip install pytest-cov
```

### Running Tests with Coverage

```bash
# Run all tests with coverage reporting
pytest --cov=app --cov-report=term-missing tests/

# Output example:
# Name                    Stmts   Miss  Cover   Missing
# -------------------------------------------------------
# app/__init__.py             0      0   100%
# app/main.py                15      0   100%
# app/models.py              32      2    94%   45-46
# app/routers/users.py       48      5    90%   72-76
# app/dependencies.py        18      0   100%
# -------------------------------------------------------
# TOTAL                     113      7    94%
```

### Configuration in pyproject.toml

```toml
# pyproject.toml
[tool.pytest.ini_options]
# Automatically use asyncio mode for async tests
asyncio_mode = "auto"
# Default pytest arguments -- always run with coverage
addopts = [
    "--cov=app",
    "--cov-report=term-missing",
    "--cov-report=html:htmlcov",  # Generate HTML report
    "-v",                          # Verbose output
]
# Test file discovery patterns
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]

[tool.coverage.run]
# Source directories to measure
source = ["app"]
# Omit files that don't need coverage
omit = [
    "app/config.py",       # Configuration -- tested implicitly
    "alembic/*",           # Migration scripts
    "tests/*",             # Don't measure test code itself
]

[tool.coverage.report]
# Fail if coverage drops below this threshold
fail_under = 85
# Lines that are intentionally not tested
exclude_lines = [
    "pragma: no cover",
    "if __name__ == .__main__.",
    "pass",
    "raise NotImplementedError",
]
# Show lines that are not covered
show_missing = true
```

### HTML Coverage Report

```bash
# Generate and open HTML report
pytest --cov=app --cov-report=html tests/
open htmlcov/index.html  # macOS
# xdg-open htmlcov/index.html  # Linux
```

The HTML report highlights covered lines in green and missed lines in red, making it easy to spot untested branches.

### Coverage Best Practices

| Practice | Reason |
|----------|--------|
| Aim for 85-90% coverage | Diminishing returns above this; 100% often wastes effort on trivial code |
| Focus on branch coverage | Line coverage misses untested `if/else` branches |
| Cover error paths | `except` blocks and `raise HTTPException` are often missed |
| Don't test framework code | You don't need to test that FastAPI returns 422 on bad input |
| Use `# pragma: no cover` sparingly | Only for code that truly cannot be tested (platform-specific, debug-only) |

---

## 8. Practice Problems

### Problem 1: CRUD Test Suite

Write a complete test suite for a `POST /items`, `GET /items`, `GET /items/{id}`, `PUT /items/{id}`, `DELETE /items/{id}` API. Include:
- Happy path tests for each endpoint
- Edge cases: create with missing fields, get non-existent item, delete twice
- Data isolation: prove that one test's data doesn't leak into another

### Problem 2: Database Test Isolation

Implement the transaction-rollback pattern for test isolation:
1. Create a `conftest.py` with engine, session, and client fixtures
2. Write two tests that both create a user with the same username
3. Prove they don't conflict (both pass because each test's data is rolled back)
4. Write a test that verifies the database is empty at the start of each test

### Problem 3: Dependency Override Testing

Given an endpoint that depends on an external weather API:

```python
async def get_weather_service():
    return RealWeatherService(api_key="...")

@app.get("/forecast/{city}")
async def get_forecast(city: str, weather=Depends(get_weather_service)):
    return await weather.get_forecast(city)
```

Write tests that:
1. Override `get_weather_service` with a mock that returns predictable data
2. Test the sunny-day scenario (valid city, good data)
3. Test the rainy-day scenario (mock raises an API error)
4. Test that the real service is never called during tests

### Problem 4: Authentication Test Matrix

Create a test matrix covering all authentication scenarios:

| Scenario | Expected Status |
|----------|----------------|
| Valid token, active user | 200 |
| Valid token, disabled user | 400 |
| Expired token | 401 |
| Malformed token | 401 |
| Missing Authorization header | 401 |
| Wrong token type ("Basic" instead of "Bearer") | 401 |
| Token with wrong secret key | 401 |

Write a parameterized test using `@pytest.mark.parametrize` for the invalid token scenarios.

### Problem 5: Coverage Gap Analysis

Given a FastAPI app with the following untested lines:

```
app/routers/users.py    48      5    90%   72-76
app/routers/posts.py    65     12    82%   34-39, 88-93
```

1. Explain what information the "Missing" column gives you
2. Describe a strategy to decide which missing lines to prioritize
3. Write test cases that would cover lines 72-76 (assume they are an error handling block for duplicate email registration)
4. Explain when it is acceptable to leave lines uncovered

---

## 9. References

- [FastAPI Testing Guide](https://fastapi.tiangolo.com/tutorial/testing/)
- [pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [httpx Documentation](https://www.python-httpx.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [Coverage.py Configuration](https://coverage.readthedocs.io/en/latest/config.html)
- [FastAPI Dependency Overrides](https://fastapi.tiangolo.com/advanced/testing-dependencies/)

---

**Previous**: [FastAPI Database](./04_FastAPI_Database.md) | **Next**: [Express Basics](./06_Express_Basics.md)
