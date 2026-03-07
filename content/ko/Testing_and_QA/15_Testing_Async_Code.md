# 레슨 15: 비동기 코드 테스트

**이전**: [Test Architecture and Patterns](./14_Test_Architecture_and_Patterns.md) | **다음**: [Database Testing](./16_Database_Testing.md)

---

비동기 코드는 현대 Python 어디에나 있습니다: 웹 프레임워크(FastAPI, aiohttp), 데이터베이스 드라이버(asyncpg, motor), HTTP 클라이언트(httpx, aiohttp), 태스크 큐 등. 비동기 코드를 테스트하면 동기 테스트에서는 직면하지 않는 문제가 발생합니다 -- 이벤트 루프, 코루틴 스케줄링, 동시 작업, 타이밍 의존적 동작 등. 올바른 도구와 패턴 없이는 비동기 테스트가 불안정하거나, 디버깅이 어렵거나, 아예 실행되지 않을 수 있습니다.

이 레슨은 pytest-asyncio를 사용하여 비동기 Python 코드를 테스트하는 실용적인 패턴을 다룹니다. 비동기 fixture, 코루틴 mocking, 일반적인 함정 회피 등을 포함합니다.

**난이도**: ⭐⭐⭐⭐

**사전 요구사항**:
- Python의 `async`/`await` 문법에 대한 확실한 이해
- pytest fixture 경험 (레슨 02-04)
- mocking에 대한 익숙함 (레슨 06)
- 이벤트 루프에 대한 기본적인 이해

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. pytest-asyncio를 설정하고 비동기 테스트 함수를 작성할 수 있다
2. 데이터베이스 연결, HTTP 클라이언트 및 기타 리소스를 위한 비동기 fixture를 생성할 수 있다
3. `AsyncMock`을 사용하여 비동기 함수를 mock할 수 있다
4. aiohttp 및 FastAPI 애플리케이션을 테스트할 수 있다
5. 일반적인 비동기 테스트 함정(이벤트 루프 충돌, 타이밍 문제, 리소스 누수)을 피할 수 있다

---

## 1. pytest-asyncio 설정

[pytest-asyncio](https://pytest-asyncio.readthedocs.io/)는 비동기 코드 테스트를 지원하는 pytest 플러그인입니다.

### 1.1 설치

```bash
pip install pytest-asyncio
```

### 1.2 구성

`pyproject.toml`에서 pytest-asyncio 모드를 구성합니다:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"  # Automatically detect async tests
```

세 가지 모드가 있습니다:
- **`auto`**: 모든 `async def test_*` 함수가 자동으로 비동기로 처리됩니다. 권장.
- **`strict`**: 모든 비동기 테스트에 명시적으로 `@pytest.mark.asyncio`가 필요합니다.
- **`legacy`** (deprecated): 암묵적 루프 생성을 수행하는 이전 동작.

### 1.3 첫 번째 비동기 테스트

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

## 2. 비동기 Fixture

비동기 테스트에서의 fixture는 동기 fixture와 같은 패턴을 따르지만, `await`와 비동기 컨텍스트 매니저를 사용할 수 있습니다.

### 2.1 기본 비동기 Fixture

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

### 2.2 설정과 해제가 있는 비동기 Fixture

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

### 2.3 Fixture 스코프

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

## 3. AsyncMock을 사용한 비동기 함수 Mocking

Python 3.8에서 코루틴을 mock하기 위해 특별히 설계된 `AsyncMock`이 `unittest.mock`에 도입되었습니다.

### 3.1 기본 AsyncMock

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

### 3.3 비동기 메서드 패칭

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

### 3.4 비동기 컨텍스트 매니저 Mocking

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

### 3.5 비동기 이터레이터 Mocking

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

## 4. aiohttp 애플리케이션 테스트

### 4.1 aiohttp 테스트 유틸리티 사용

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

### 4.2 pytest-aiohttp Fixture를 사용한 테스트

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

## 5. FastAPI 애플리케이션 테스트

FastAPI는 `TestClient`(동기)와 `httpx.AsyncClient`(비동기)를 통해 우수한 테스트 지원을 제공합니다.

### 5.1 동기 테스트 (더 간단한 방식)

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

### 5.2 비동기 테스트 (완전한 비동기 지원)

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

## 6. WebSocket 테스트

### 6.1 FastAPI에서의 WebSocket 테스트

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

### 6.2 여러 클라이언트로 WebSocket 테스트

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

## 7. 일반적인 비동기 테스트 함정

### 7.1 `await` 누락

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

### 7.2 리소스 누수

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

### 7.3 이벤트 루프가 이미 실행 중

```python
# BAD: Mixing asyncio.run() with pytest-asyncio
async def test_double_loop():
    # This fails because pytest-asyncio already provides an event loop
    result = asyncio.run(some_coroutine())  # RuntimeError!

# GOOD: Just await directly
async def test_correct():
    result = await some_coroutine()
```

### 7.4 타이밍 의존적 테스트

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

### 7.5 테스트 간 공유 가변 상태

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

## 8. 동시 작업 테스트

### 8.1 `asyncio.gather`를 사용한 테스트

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

### 8.2 속도 제한 테스트

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

## 연습 문제

1. **비동기 Fixture 체인**: 세 개의 비동기 fixture -- 데이터베이스 연결, 테이블 설정(연결에 의존), 데이터 시더(테이블에 의존) -- 를 생성하십시오. 시드된 데이터를 사용하는 테스트를 작성하십시오.

2. **AsyncMock 실습**: 두 개의 외부 API를 호출하고 결과를 결합하는 비동기 서비스를 작성하십시오. `AsyncMock`을 사용하여 두 API 호출을 mock하여 테스트하십시오. 하나가 예외를 발생시키는 경우를 포함하십시오.

3. **WebSocket 테스트**: 간단한 비동기 채팅 서버를 구현하고 다음을 검증하는 테스트를 작성하십시오: (a) 클라이언트가 연결할 수 있음, (b) 메시지가 에코됨, (c) 여러 클라이언트가 브로드캐스트를 수신함.

4. **FastAPI Dependency Override**: 데이터베이스 의존성이 있는 FastAPI 앱을 생성하십시오. `AsyncMock`으로 의존성을 오버라이드하여 성공 및 오류 경로를 테스트하십시오.

5. **동시성 버그**: 공유 카운터에서 경쟁 조건(race condition)을 보여주는 테스트를 작성하고(잠금 없이), `asyncio.Lock`으로 수정한 후 테스트가 통과하는지 검증하십시오.

---

**License**: CC BY-NC 4.0
