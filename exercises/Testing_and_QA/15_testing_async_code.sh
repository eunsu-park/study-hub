#!/bin/bash
# Exercises for Lesson 15: Testing Async Code
# Topic: Testing_and_QA
# Solutions to practice problems from the lesson.

# === Exercise 1: Async Fixture Chain ===
# Problem: Create three async fixtures — a database connection, a table
# setup (depends on connection), and a data seeder (depends on table).
# Write a test that uses the seeded data.
exercise_1() {
    echo "=== Exercise 1: Async Fixture Chain ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import pytest
import aiosqlite

@pytest.fixture
async def db_connection(tmp_path):
    """Layer 1: Create an async database connection."""
    db_path = tmp_path / "test.db"
    conn = await aiosqlite.connect(str(db_path))
    conn.row_factory = aiosqlite.Row
    yield conn
    await conn.close()

@pytest.fixture
async def db_with_tables(db_connection):
    """Layer 2: Create tables (depends on connection)."""
    await db_connection.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL
        )
    """)
    await db_connection.execute("""
        CREATE TABLE posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER REFERENCES users(id),
            title TEXT NOT NULL,
            body TEXT
        )
    """)
    await db_connection.commit()
    yield db_connection

@pytest.fixture
async def seeded_db(db_with_tables):
    """Layer 3: Seed test data (depends on tables)."""
    await db_with_tables.execute(
        "INSERT INTO users (name, email) VALUES (?, ?)",
        ("Alice", "alice@example.com")
    )
    await db_with_tables.execute(
        "INSERT INTO users (name, email) VALUES (?, ?)",
        ("Bob", "bob@example.com")
    )
    await db_with_tables.execute(
        "INSERT INTO posts (user_id, title, body) VALUES (?, ?, ?)",
        (1, "First Post", "Hello world")
    )
    await db_with_tables.commit()
    yield db_with_tables

@pytest.mark.asyncio
async def test_seeded_users_exist(seeded_db):
    cursor = await seeded_db.execute("SELECT COUNT(*) FROM users")
    row = await cursor.fetchone()
    assert row[0] == 2

@pytest.mark.asyncio
async def test_seeded_post_belongs_to_user(seeded_db):
    cursor = await seeded_db.execute(
        "SELECT u.name, p.title FROM posts p "
        "JOIN users u ON p.user_id = u.id"
    )
    row = await cursor.fetchone()
    assert row["name"] == "Alice"
    assert row["title"] == "First Post"

@pytest.mark.asyncio
async def test_empty_after_fresh_fixture(db_with_tables):
    """Each test gets a fresh database — no bleed from seeded_db."""
    cursor = await db_with_tables.execute("SELECT COUNT(*) FROM users")
    row = await cursor.fetchone()
    assert row[0] == 0
SOLUTION
}

# === Exercise 2: AsyncMock Practice ===
# Problem: Write an async service that calls two external APIs and
# combines their results. Test it using AsyncMock to mock both API
# calls, including one that raises an exception.
exercise_2() {
    echo "=== Exercise 2: AsyncMock Practice ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import pytest
from unittest.mock import AsyncMock

class WeatherClient:
    async def get_temperature(self, city: str) -> float:
        raise NotImplementedError

class GeoClient:
    async def get_coordinates(self, city: str) -> dict:
        raise NotImplementedError

class LocationWeatherService:
    """Service that combines data from two async APIs."""

    def __init__(self, weather: WeatherClient, geo: GeoClient):
        self.weather = weather
        self.geo = geo

    async def get_city_info(self, city: str) -> dict:
        temp = await self.weather.get_temperature(city)
        coords = await self.geo.get_coordinates(city)
        return {
            "city": city,
            "temperature": temp,
            "latitude": coords["lat"],
            "longitude": coords["lng"],
        }

@pytest.fixture
def mock_weather():
    client = AsyncMock(spec=WeatherClient)
    client.get_temperature.return_value = 22.5
    return client

@pytest.fixture
def mock_geo():
    client = AsyncMock(spec=GeoClient)
    client.get_coordinates.return_value = {"lat": 51.5, "lng": -0.12}
    return client

@pytest.fixture
def service(mock_weather, mock_geo):
    return LocationWeatherService(weather=mock_weather, geo=mock_geo)

@pytest.mark.asyncio
async def test_get_city_info_combines_apis(service, mock_weather, mock_geo):
    result = await service.get_city_info("London")

    assert result["city"] == "London"
    assert result["temperature"] == 22.5
    assert result["latitude"] == 51.5
    mock_weather.get_temperature.assert_awaited_once_with("London")
    mock_geo.get_coordinates.assert_awaited_once_with("London")

@pytest.mark.asyncio
async def test_weather_api_failure(service, mock_weather):
    """When the weather API fails, the error propagates."""
    mock_weather.get_temperature.side_effect = ConnectionError("Weather API down")

    with pytest.raises(ConnectionError, match="Weather API down"):
        await service.get_city_info("London")

@pytest.mark.asyncio
async def test_geo_api_failure(service, mock_geo):
    """When the geo API fails, the error propagates."""
    mock_geo.get_coordinates.side_effect = TimeoutError("Geo API timeout")

    with pytest.raises(TimeoutError, match="Geo API timeout"):
        await service.get_city_info("London")
SOLUTION
}

# === Exercise 3: WebSocket Test ===
# Problem: Implement a simple async chat server and write tests that
# verify: (a) a client can connect, (b) messages are echoed back,
# (c) multiple clients receive broadcasts.
exercise_3() {
    echo "=== Exercise 3: WebSocket Test ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import pytest
from unittest.mock import AsyncMock

class ChatRoom:
    """Simple async chat room supporting connect, disconnect, broadcast."""

    def __init__(self):
        self._clients = []

    async def connect(self, websocket):
        self._clients.append(websocket)
        await websocket.send_text("Welcome!")

    async def disconnect(self, websocket):
        self._clients.remove(websocket)

    async def echo(self, websocket, message: str):
        await websocket.send_text(f"Echo: {message}")

    async def broadcast(self, message: str):
        for client in self._clients:
            await client.send_text(message)

    @property
    def client_count(self):
        return len(self._clients)

@pytest.fixture
def chat_room():
    return ChatRoom()

def make_ws_mock():
    """Helper to create a mock WebSocket."""
    ws = AsyncMock()
    ws.send_text = AsyncMock()
    ws.receive_text = AsyncMock()
    return ws

@pytest.mark.asyncio
async def test_client_can_connect(chat_room):
    ws = make_ws_mock()
    await chat_room.connect(ws)

    assert chat_room.client_count == 1
    ws.send_text.assert_awaited_once_with("Welcome!")

@pytest.mark.asyncio
async def test_echo_sends_back(chat_room):
    ws = make_ws_mock()
    await chat_room.connect(ws)
    ws.send_text.reset_mock()  # Clear the Welcome message

    await chat_room.echo(ws, "hello")

    ws.send_text.assert_awaited_once_with("Echo: hello")

@pytest.mark.asyncio
async def test_broadcast_reaches_all_clients(chat_room):
    ws1 = make_ws_mock()
    ws2 = make_ws_mock()
    ws3 = make_ws_mock()

    await chat_room.connect(ws1)
    await chat_room.connect(ws2)
    await chat_room.connect(ws3)

    # Reset to clear Welcome messages
    for ws in [ws1, ws2, ws3]:
        ws.send_text.reset_mock()

    await chat_room.broadcast("Hello everyone!")

    ws1.send_text.assert_awaited_once_with("Hello everyone!")
    ws2.send_text.assert_awaited_once_with("Hello everyone!")
    ws3.send_text.assert_awaited_once_with("Hello everyone!")

@pytest.mark.asyncio
async def test_disconnect_removes_client(chat_room):
    ws1 = make_ws_mock()
    ws2 = make_ws_mock()
    await chat_room.connect(ws1)
    await chat_room.connect(ws2)
    assert chat_room.client_count == 2

    await chat_room.disconnect(ws1)
    assert chat_room.client_count == 1

    # Broadcast should only reach ws2
    for ws in [ws1, ws2]:
        ws.send_text.reset_mock()

    await chat_room.broadcast("After disconnect")
    ws1.send_text.assert_not_awaited()
    ws2.send_text.assert_awaited_once_with("After disconnect")
SOLUTION
}

# === Exercise 4: FastAPI Dependency Override ===
# Problem: Create a FastAPI app with a database dependency. Write tests
# that override the dependency with an AsyncMock, testing both success
# and error paths.
exercise_4() {
    echo "=== Exercise 4: FastAPI Dependency Override ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import pytest
from unittest.mock import AsyncMock
from fastapi import FastAPI, Depends, HTTPException

# --- Application code ---

app = FastAPI()

async def get_db():
    """Dependency: async database session."""
    raise NotImplementedError("Real DB not available in tests")

@app.get("/users/{user_id}")
async def get_user(user_id: int, db=Depends(get_db)):
    user = await db.fetch_one(
        "SELECT * FROM users WHERE id = $1", user_id
    )
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return {"id": user["id"], "name": user["name"]}

@app.get("/users")
async def list_users(db=Depends(get_db)):
    rows = await db.fetch_all("SELECT * FROM users")
    return [{"id": r["id"], "name": r["name"]} for r in rows]

# --- Tests ---

import httpx

@pytest.fixture
def mock_db():
    db = AsyncMock()
    return db

@pytest.fixture
def test_app(mock_db):
    """Override the DB dependency with a mock."""
    app.dependency_overrides[get_db] = lambda: mock_db
    yield app
    app.dependency_overrides.clear()

@pytest.mark.asyncio
async def test_get_user_success(test_app, mock_db):
    mock_db.fetch_one.return_value = {"id": 1, "name": "Alice"}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=test_app),
        base_url="http://test"
    ) as client:
        response = await client.get("/users/1")

    assert response.status_code == 200
    assert response.json() == {"id": 1, "name": "Alice"}

@pytest.mark.asyncio
async def test_get_user_not_found(test_app, mock_db):
    mock_db.fetch_one.return_value = None

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=test_app),
        base_url="http://test"
    ) as client:
        response = await client.get("/users/999")

    assert response.status_code == 404
    assert response.json()["detail"] == "User not found"

@pytest.mark.asyncio
async def test_list_users(test_app, mock_db):
    mock_db.fetch_all.return_value = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
    ]

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=test_app),
        base_url="http://test"
    ) as client:
        response = await client.get("/users")

    assert response.status_code == 200
    assert len(response.json()) == 2
SOLUTION
}

# === Exercise 5: Concurrency Bug ===
# Problem: Write a test that demonstrates a race condition in a shared
# counter (no lock), then fix it with asyncio.Lock and verify the test
# passes.
exercise_5() {
    echo "=== Exercise 5: Concurrency Bug ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import pytest
import asyncio

class BuggyCounter:
    """Counter WITHOUT a lock — susceptible to race conditions."""

    def __init__(self):
        self.count = 0

    async def increment(self):
        current = self.count
        await asyncio.sleep(0)  # Yield control — simulates async I/O
        self.count = current + 1

class SafeCounter:
    """Counter WITH asyncio.Lock — race-condition-proof."""

    def __init__(self):
        self.count = 0
        self._lock = asyncio.Lock()

    async def increment(self):
        async with self._lock:
            current = self.count
            await asyncio.sleep(0)
            self.count = current + 1

@pytest.mark.asyncio
async def test_buggy_counter_race_condition():
    """Demonstrate that BuggyCounter loses increments under concurrency."""
    counter = BuggyCounter()
    n = 100

    tasks = [counter.increment() for _ in range(n)]
    await asyncio.gather(*tasks)

    # Without a lock, concurrent reads of 'current' before any write
    # causes lost updates. The final count will be less than n.
    assert counter.count < n, (
        f"Expected lost updates but count == {counter.count}"
    )

@pytest.mark.asyncio
async def test_safe_counter_no_race_condition():
    """Verify that SafeCounter handles concurrency correctly."""
    counter = SafeCounter()
    n = 100

    tasks = [counter.increment() for _ in range(n)]
    await asyncio.gather(*tasks)

    assert counter.count == n

@pytest.mark.asyncio
async def test_safe_counter_sequential():
    """Verify basic sequential behavior."""
    counter = SafeCounter()

    await counter.increment()
    await counter.increment()
    await counter.increment()

    assert counter.count == 3
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 15: Testing Async Code"
echo "======================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
echo ""
exercise_5
