# Lesson 15: Testing Async Code

**Previous**: [Test Architecture and Patterns](./14_Test_Architecture_and_Patterns.md) | **Next**: [Database Testing](./16_Database_Testing.md)

---

Asynchronous code is everywhere in modern Python: web frameworks (FastAPI, aiohttp), database drivers (asyncpg, motor), HTTP clients (httpx, aiohttp), and task queues. Testing async code introduces challenges that synchronous tests do not face — event loops, coroutine scheduling, concurrent operations, and timing-dependent behavior. Without the right tools and patterns, async tests become unreliable, hard to debug, or simply do not run at all.

This lesson covers the practical patterns for testing async Python code with pytest-asyncio, including async fixtures, mocking coroutines, and avoiding common pitfalls.

**Difficulty**: ⭐⭐⭐⭐

**Prerequisites**:
- Solid understanding of Python's `async`/`await` syntax
- Experience with pytest fixtures (Lessons 02–04)
- Familiarity with mocking (Lesson 06)
- Basic understanding of the event loop

## Learning Objectives

After completing this lesson, you will be able to:

1. Set up pytest-asyncio and write async test functions
2. Create async fixtures for database connections, HTTP clients, and other resources
3. Mock async functions using `AsyncMock`
4. Test aiohttp and FastAPI applications
5. Avoid common async testing pitfalls (event loop conflicts, timing issues, resource leaks)

---

## 1. Setting Up pytest-asyncio

[pytest-asyncio](https://pytest-asyncio.readthedocs.io/) is a pytest plugin that provides support for testing async code.

### 1.1 Installation

```bash
pip install pytest-asyncio
```

### 1.2 Configuration

Configure pytest-asyncio mode in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"  # Automatically detect async tests
```

The three modes are:
- **`auto`**: Any `async def test_*` function is automatically treated as async. Recommended.
- **`strict`**: Requires explicit `@pytest.mark.asyncio` on every async test.
- **`legacy`** (deprecated): Old behavior with implicit loop creation.

### 1.3 Your First Async Test

```python
# With asyncio_mode = "auto", no decorator needed
async def test_basic_async():
    result = await some_async_function()
    assert result == "expected"


# With asyncio_mode = "strict", decorator required
import pytest

@pytest.mark.asyncio
async def test_basic_async_strict():
    result = await some_async_function()
    assert result == "expected"
```

---

## 2. Async Fixtures

Fixtures in async testing follow the same patterns as synchronous fixtures, but they can use `await` and async context managers.

### 2.1 Basic Async Fixtures

```python
import pytest
import aiohttp


@pytest.fixture
async def http_session():
    """Create and clean up an aiohttp session."""
    session = aiohttp.ClientSession()
    yield session
    await session.close()


async def test_fetch_data(http_session):
    async with http_session.get("https://httpbin.org/json") as response:
        data = await response.json()
        assert response.status == 200
        assert "slideshow" in data
```

### 2.2 Async Fixtures with Setup and Teardown

```python
import asyncpg


@pytest.fixture
async def db_connection():
    """Create a database connection for testing."""
    conn = await asyncpg.connect(
        host="localhost",
        port=5432,
        user="test",
        password="test",
        database="testdb"
    )

    # Setup: create test schema
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS test_users (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL
        )
    """)

    yield conn

    # Teardown: clean up
    await conn.execute("DROP TABLE IF EXISTS test_users")
    await conn.close()


async def test_insert_user(db_connection):
    await db_connection.execute(
        "INSERT INTO test_users (name, email) VALUES ($1, $2)",
        "alice", "alice@example.com"
    )
    row = await db_connection.fetchrow(
        "SELECT * FROM test_users WHERE email = $1",
        "alice@example.com"
    )
    assert row["name"] == "alice"
```

### 2.3 Fixture Scoping

```python
@pytest.fixture(scope="session")
async def app():
    """Create the application once for the entire test session."""
    app = create_app(testing=True)
    await app.startup()
    yield app
    await app.shutdown()


@pytest.fixture(scope="function")
async def clean_db(db_connection):
    """Reset the database before each test."""
    yield db_connection
    await db_connection.execute("DELETE FROM test_users")
```

---

## 3. Mocking Async Functions with AsyncMock

Python 3.8 introduced `AsyncMock` in `unittest.mock`, designed specifically for mocking coroutines.

### 3.1 Basic AsyncMock

```python
from unittest.mock import AsyncMock, patch


async def test_service_calls_api():
    # Create an async mock
    mock_client = AsyncMock()
    mock_client.get.return_value = {"status": "ok", "data": [1, 2, 3]}

    service = DataService(client=mock_client)
    result = await service.fetch_data("users")

    mock_client.get.assert_awaited_once_with("/api/users")
    assert result == [1, 2, 3]
```

### 3.2 AsyncMock vs Mock

```python
from unittest.mock import Mock, AsyncMock

# Mock for synchronous functions
sync_mock = Mock(return_value=42)
result = sync_mock()  # Returns 42

# AsyncMock for async functions
async_mock = AsyncMock(return_value=42)
result = await async_mock()  # Returns 42

# AsyncMock also works for sync calls
# but Mock does NOT work for await (raises TypeError)
```

### 3.3 Patching Async Methods

```python
import httpx


class WeatherService:
    async def get_temperature(self, city: str) -> float:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://api.weather.com/temp?city={city}"
            )
            return response.json()["temperature"]


async def test_weather_service_with_patch():
    mock_response = AsyncMock()
    mock_response.json.return_value = {"temperature": 22.5}
    mock_response.status_code = 200

    with patch("httpx.AsyncClient.get", return_value=mock_response):
        service = WeatherService()
        temp = await service.get_temperature("London")
        assert temp == 22.5
```

### 3.4 Mocking Async Context Managers

```python
from unittest.mock import AsyncMock, MagicMock
from contextlib import asynccontextmanager


class AsyncDatabasePool:
    async def acquire(self):
        ...

    async def release(self, conn):
        ...


# Mock the async context manager pattern
def create_mock_pool():
    mock_conn = AsyncMock()
    mock_conn.execute.return_value = [{"id": 1, "name": "alice"}]

    mock_pool = MagicMock()
    mock_pool.acquire = AsyncMock(return_value=mock_conn)
    mock_pool.release = AsyncMock()

    return mock_pool, mock_conn


async def test_repository_uses_connection_pool():
    mock_pool, mock_conn = create_mock_pool()
    repo = UserRepository(pool=mock_pool)

    users = await repo.find_all()

    mock_pool.acquire.assert_awaited_once()
    mock_conn.execute.assert_awaited_once_with("SELECT * FROM users")
    assert len(users) == 1
```

### 3.5 Mocking Async Iterators

```python
class MockAsyncIterator:
    """Mock for async for loops."""

    def __init__(self, items):
        self.items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.items)
        except StopIteration:
            raise StopAsyncIteration


async def test_processing_async_stream():
    mock_stream = MockAsyncIterator([
        {"event": "start"},
        {"event": "data", "value": 42},
        {"event": "end"},
    ])

    processor = StreamProcessor()
    results = await processor.process(mock_stream)
    assert len(results) == 3
```

---

## 4. Testing aiohttp Applications

### 4.1 Using aiohttp's Test Utilities

```python
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop


# Application under test
async def handle_hello(request):
    name = request.match_info.get("name", "World")
    return web.json_response({"message": f"Hello, {name}!"})


def create_app():
    app = web.Application()
    app.router.add_get("/hello/{name}", handle_hello)
    return app


# Test with pytest-aiohttp
async def test_hello_endpoint(aiohttp_client):
    client = await aiohttp_client(create_app())
    response = await client.get("/hello/Alice")
    assert response.status == 200
    data = await response.json()
    assert data["message"] == "Hello, Alice!"
```

### 4.2 Testing with pytest-aiohttp Fixture

```bash
pip install pytest-aiohttp
```

```python
import pytest
from aiohttp import web


@pytest.fixture
def app():
    """Create the aiohttp application."""
    app = web.Application()
    app.router.add_get("/api/status", handle_status)
    app.router.add_post("/api/data", handle_data)
    return app


async def test_status_endpoint(aiohttp_client, app):
    client = await aiohttp_client(app)
    response = await client.get("/api/status")
    assert response.status == 200


async def test_post_data(aiohttp_client, app):
    client = await aiohttp_client(app)
    response = await client.post("/api/data", json={"key": "value"})
    assert response.status == 201
```

---

## 5. Testing FastAPI Applications

FastAPI provides excellent testing support through its `TestClient` (synchronous) and `httpx.AsyncClient` (async).

### 5.1 Synchronous Testing (Simpler)

```python
from fastapi import FastAPI
from fastapi.testclient import TestClient

app = FastAPI()


@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id, "name": f"Item {item_id}"}


# TestClient handles the event loop internally
def test_read_item():
    client = TestClient(app)
    response = client.get("/items/42")
    assert response.status_code == 200
    assert response.json() == {"item_id": 42, "name": "Item 42"}
```

### 5.2 Async Testing (Full Async Support)

```python
import httpx
import pytest
from fastapi import FastAPI, Depends

app = FastAPI()


async def get_db():
    """Dependency that provides a database session."""
    db = await create_session()
    try:
        yield db
    finally:
        await db.close()


@app.get("/users/{user_id}")
async def get_user(user_id: int, db=Depends(get_db)):
    user = await db.fetch_one("SELECT * FROM users WHERE id = $1", user_id)
    return user


# Override dependencies in tests
@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.fetch_one.return_value = {"id": 1, "name": "alice"}
    return db


@pytest.fixture
def test_app(mock_db):
    app.dependency_overrides[get_db] = lambda: mock_db
    yield app
    app.dependency_overrides.clear()


async def test_get_user(test_app):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=test_app),
        base_url="http://test"
    ) as client:
        response = await client.get("/users/1")
        assert response.status_code == 200
        assert response.json()["name"] == "alice"
```

---

## 6. Testing WebSockets

### 6.1 WebSocket Testing with FastAPI

```python
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient

app = FastAPI()


@app.websocket("/ws/echo")
async def echo_websocket(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        if data == "close":
            await websocket.close()
            break
        await websocket.send_text(f"Echo: {data}")


def test_websocket_echo():
    client = TestClient(app)
    with client.websocket_connect("/ws/echo") as websocket:
        websocket.send_text("hello")
        response = websocket.receive_text()
        assert response == "Echo: hello"

        websocket.send_text("world")
        response = websocket.receive_text()
        assert response == "Echo: world"
```

### 6.2 Testing WebSocket with Multiple Clients

```python
import asyncio
from unittest.mock import AsyncMock


async def test_chat_broadcast():
    """Test that messages are broadcast to all connected clients."""
    chat_room = ChatRoom()

    # Create mock WebSocket connections
    ws1 = AsyncMock()
    ws2 = AsyncMock()

    await chat_room.connect(ws1)
    await chat_room.connect(ws2)

    await chat_room.broadcast("Hello everyone!")

    ws1.send_text.assert_awaited_once_with("Hello everyone!")
    ws2.send_text.assert_awaited_once_with("Hello everyone!")
```

---

## 7. Common Async Testing Pitfalls

### 7.1 Forgetting to `await`

```python
# BUG: Missing await — test passes because it asserts on a coroutine object
async def test_broken():
    result = some_async_function()  # Returns coroutine, not result!
    assert result  # Coroutine is truthy — always passes!

# FIX: Always await
async def test_correct():
    result = await some_async_function()
    assert result
```

### 7.2 Resource Leaks

```python
# BAD: Connection is never closed
async def test_leaky():
    conn = await create_connection()
    result = await conn.query("SELECT 1")
    assert result == 1
    # Connection leaked!

# GOOD: Use async context manager or fixture
async def test_clean():
    async with create_connection() as conn:
        result = await conn.query("SELECT 1")
        assert result == 1
```

### 7.3 Event Loop Already Running

```python
# BAD: Mixing asyncio.run() with pytest-asyncio
async def test_double_loop():
    # This fails because pytest-asyncio already provides an event loop
    result = asyncio.run(some_coroutine())  # RuntimeError!

# GOOD: Just await directly
async def test_correct():
    result = await some_coroutine()
```

### 7.4 Timing-Dependent Tests

```python
# FRAGILE: Depends on exact timing
async def test_fragile_timeout():
    start = time.time()
    await asyncio.sleep(0.1)
    elapsed = time.time() - start
    assert elapsed < 0.15  # May fail under CI load!

# ROBUST: Use asyncio.wait_for for timeout testing
async def test_robust_timeout():
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(slow_operation(), timeout=0.5)
```

### 7.5 Shared Mutable State Between Tests

```python
# BAD: Module-level mutable state persists across tests
_cache = {}

async def test_first():
    _cache["key"] = "value"
    assert _cache["key"] == "value"

async def test_second():
    # This may or may not find "key" depending on test order!
    assert "key" not in _cache  # FAILS if test_first runs first

# GOOD: Use fixtures for per-test state
@pytest.fixture
def cache():
    return {}

async def test_isolated(cache):
    cache["key"] = "value"
    assert cache["key"] == "value"
```

---

## 8. Testing Concurrent Operations

### 8.1 Testing with `asyncio.gather`

```python
async def test_concurrent_requests_all_succeed():
    """Verify that multiple concurrent operations complete correctly."""
    service = DataService()

    results = await asyncio.gather(
        service.fetch("resource_1"),
        service.fetch("resource_2"),
        service.fetch("resource_3"),
    )

    assert len(results) == 3
    assert all(r["status"] == "ok" for r in results)


async def test_concurrent_requests_handle_partial_failure():
    """Verify graceful handling when some concurrent operations fail."""
    service = DataService()

    results = await asyncio.gather(
        service.fetch("valid_resource"),
        service.fetch("invalid_resource"),
        return_exceptions=True,
    )

    assert results[0]["status"] == "ok"
    assert isinstance(results[1], Exception)
```

### 8.2 Testing Rate Limiting

```python
async def test_rate_limiter():
    """Verify that the rate limiter throttles concurrent requests."""
    limiter = RateLimiter(max_concurrent=3)
    call_count = 0

    async def tracked_operation():
        nonlocal call_count
        call_count += 1
        current = call_count
        await asyncio.sleep(0.1)
        return current

    # Launch 10 operations, only 3 should run concurrently
    tasks = [limiter.execute(tracked_operation) for _ in range(10)]
    results = await asyncio.gather(*tasks)

    assert len(results) == 10
    assert set(results) == set(range(1, 11))
```

---

## Exercises

1. **Async Fixture Chain**: Create three async fixtures — a database connection, a table setup (depends on connection), and a data seeder (depends on table). Write a test that uses the seeded data.

2. **AsyncMock Practice**: Write an async service that calls two external APIs and combines their results. Test it using `AsyncMock` to mock both API calls, including one that raises an exception.

3. **WebSocket Test**: Implement a simple async chat server and write tests that verify: (a) a client can connect, (b) messages are echoed back, (c) multiple clients receive broadcasts.

4. **FastAPI Dependency Override**: Create a FastAPI app with a database dependency. Write tests that override the dependency with an `AsyncMock`, testing both success and error paths.

5. **Concurrency Bug**: Write a test that demonstrates a race condition in a shared counter (no lock), then fix it with `asyncio.Lock` and verify the test passes.

---

**License**: CC BY-NC 4.0
