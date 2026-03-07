#!/bin/bash
# Exercises for Lesson 10: Testing Async Code
# Topic: Testing_and_QA
# Solutions to practice problems from the lesson.

# === Exercise 1: Basic Async Test Patterns ===
# Problem: Write async tests for an async key-value store.
exercise_1() {
    echo "=== Exercise 1: Basic Async Test Patterns ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import pytest
import asyncio
from typing import Optional

class AsyncKeyValueStore:
    def __init__(self):
        self._data = {}

    async def get(self, key: str) -> Optional[str]:
        await asyncio.sleep(0)  # Simulate async I/O
        return self._data.get(key)

    async def set(self, key: str, value: str) -> None:
        await asyncio.sleep(0)
        self._data[key] = value

    async def delete(self, key: str) -> bool:
        await asyncio.sleep(0)
        if key in self._data:
            del self._data[key]
            return True
        return False

    async def exists(self, key: str) -> bool:
        await asyncio.sleep(0)
        return key in self._data

@pytest.fixture
def store():
    return AsyncKeyValueStore()

# Mark async tests with @pytest.mark.asyncio
@pytest.mark.asyncio
async def test_set_and_get(store):
    await store.set("name", "Alice")
    value = await store.get("name")
    assert value == "Alice"

@pytest.mark.asyncio
async def test_get_missing_key(store):
    value = await store.get("nonexistent")
    assert value is None

@pytest.mark.asyncio
async def test_delete_existing(store):
    await store.set("temp", "data")
    result = await store.delete("temp")
    assert result is True
    assert await store.exists("temp") is False

@pytest.mark.asyncio
async def test_delete_missing(store):
    result = await store.delete("nonexistent")
    assert result is False

@pytest.mark.asyncio
async def test_exists(store):
    assert await store.exists("key") is False
    await store.set("key", "value")
    assert await store.exists("key") is True
SOLUTION
}

# === Exercise 2: Mocking Async Dependencies ===
# Problem: Test an async service that depends on an async HTTP client.
exercise_2() {
    echo "=== Exercise 2: Mocking Async Dependencies ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import pytest
from unittest.mock import AsyncMock, patch

class AsyncHttpClient:
    async def get(self, url: str) -> dict:
        raise NotImplementedError("Real HTTP call")

    async def post(self, url: str, data: dict) -> dict:
        raise NotImplementedError("Real HTTP call")

class UserProfileService:
    def __init__(self, http_client: AsyncHttpClient):
        self.client = http_client

    async def get_profile(self, user_id: int) -> dict:
        data = await self.client.get(f"/api/users/{user_id}")
        return {
            "id": data["id"],
            "display_name": f"{data['first_name']} {data['last_name']}",
            "email": data["email"],
        }

    async def update_bio(self, user_id: int, bio: str) -> bool:
        result = await self.client.post(
            f"/api/users/{user_id}/bio",
            {"bio": bio}
        )
        return result.get("success", False)

@pytest.fixture
def mock_client():
    """AsyncMock is essential for mocking async methods."""
    client = AsyncMock(spec=AsyncHttpClient)
    return client

@pytest.fixture
def service(mock_client):
    return UserProfileService(mock_client)

@pytest.mark.asyncio
async def test_get_profile(service, mock_client):
    """Configure AsyncMock return value, then verify transformation."""
    mock_client.get.return_value = {
        "id": 1,
        "first_name": "Alice",
        "last_name": "Smith",
        "email": "alice@example.com"
    }

    profile = await service.get_profile(1)

    assert profile["display_name"] == "Alice Smith"
    assert profile["email"] == "alice@example.com"
    mock_client.get.assert_awaited_once_with("/api/users/1")

@pytest.mark.asyncio
async def test_update_bio_success(service, mock_client):
    mock_client.post.return_value = {"success": True}

    result = await service.update_bio(1, "Hello world")

    assert result is True
    mock_client.post.assert_awaited_once_with(
        "/api/users/1/bio",
        {"bio": "Hello world"}
    )

@pytest.mark.asyncio
async def test_get_profile_api_error(service, mock_client):
    """Test error propagation from async dependency."""
    mock_client.get.side_effect = ConnectionError("API down")

    with pytest.raises(ConnectionError):
        await service.get_profile(1)
SOLUTION
}

# === Exercise 3: Testing Concurrent Operations ===
# Problem: Test that concurrent async operations execute correctly
# and handle race conditions.
exercise_3() {
    echo "=== Exercise 3: Testing Concurrent Operations ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import pytest
import asyncio

class AsyncCounter:
    """Thread-safe async counter using asyncio.Lock."""

    def __init__(self):
        self._count = 0
        self._lock = asyncio.Lock()

    async def increment(self):
        async with self._lock:
            current = self._count
            await asyncio.sleep(0)  # Simulate async work
            self._count = current + 1

    async def get_count(self) -> int:
        return self._count

@pytest.mark.asyncio
async def test_concurrent_increments():
    """Verify counter handles concurrent increments correctly."""
    counter = AsyncCounter()

    # Launch 100 concurrent increments
    tasks = [counter.increment() for _ in range(100)]
    await asyncio.gather(*tasks)

    count = await counter.get_count()
    assert count == 100  # Lock prevents lost updates

@pytest.mark.asyncio
async def test_gather_collects_results():
    """asyncio.gather returns results in order."""
    async def delayed_value(value, delay):
        await asyncio.sleep(delay)
        return value

    results = await asyncio.gather(
        delayed_value("a", 0.03),
        delayed_value("b", 0.01),
        delayed_value("c", 0.02),
    )

    # Results are in submission order, not completion order
    assert results == ["a", "b", "c"]

@pytest.mark.asyncio
async def test_gather_handles_exceptions():
    """Test error handling in concurrent operations."""
    async def failing_task():
        raise ValueError("task failed")

    async def succeeding_task():
        return "ok"

    with pytest.raises(ValueError):
        await asyncio.gather(
            succeeding_task(),
            failing_task(),
        )

    # With return_exceptions=True, errors are returned as values
    results = await asyncio.gather(
        succeeding_task(),
        failing_task(),
        return_exceptions=True,
    )
    assert results[0] == "ok"
    assert isinstance(results[1], ValueError)

@pytest.mark.asyncio
async def test_timeout_handling():
    """Verify timeout behavior for slow async operations."""
    async def slow_operation():
        await asyncio.sleep(10)

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(slow_operation(), timeout=0.01)
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 10: Testing Async Code"
echo "======================================================"
exercise_1
echo ""
exercise_2
echo ""
exercise_3
