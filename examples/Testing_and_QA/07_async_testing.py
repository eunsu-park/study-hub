#!/usr/bin/env python3
"""Example: Async Testing

Demonstrates pytest-asyncio patterns for testing asynchronous Python code:
async fixtures, async test functions, mocking async calls, and testing
concurrent operations.
Related lesson: 10_Testing_Async_Code.md
"""

# =============================================================================
# WHY ASYNC TESTING?
# Modern Python applications often use async/await for I/O-bound operations
# (HTTP requests, database queries, file I/O). Testing async code requires
# special support because:
#   1. Tests must run in an event loop
#   2. Fixtures may need to be async (e.g., async DB setup)
#   3. Mocking must handle coroutines, not just regular functions
#   4. Concurrency bugs (race conditions) need specific test patterns
# =============================================================================

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from dataclasses import dataclass
from typing import Optional, List, Dict

try:
    import pytest_asyncio
    ASYNCIO_AVAILABLE = True
except ImportError:
    ASYNCIO_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not ASYNCIO_AVAILABLE,
    reason="pytest-asyncio not installed (pip install pytest-asyncio)"
)


# =============================================================================
# PRODUCTION CODE — Async services to test
# =============================================================================

class AsyncCache:
    """An async in-memory cache with TTL support."""

    def __init__(self):
        self._store: Dict[str, any] = {}

    async def get(self, key: str) -> Optional[any]:
        """Simulate async cache lookup (e.g., Redis)."""
        await asyncio.sleep(0)  # Yield control to event loop
        return self._store.get(key)

    async def set(self, key: str, value: any) -> None:
        """Simulate async cache write."""
        await asyncio.sleep(0)
        self._store[key] = value

    async def delete(self, key: str) -> bool:
        """Simulate async cache deletion."""
        await asyncio.sleep(0)
        if key in self._store:
            del self._store[key]
            return True
        return False

    async def clear(self) -> None:
        """Clear all cached entries."""
        await asyncio.sleep(0)
        self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)


class AsyncUserService:
    """Service that fetches user data from an async API."""

    def __init__(self, api_client, cache: AsyncCache):
        self.api_client = api_client
        self.cache = cache

    async def get_user(self, user_id: int) -> dict:
        """Get user, checking cache first (cache-aside pattern)."""
        cache_key = f"user:{user_id}"

        # Try cache first
        cached = await self.cache.get(cache_key)
        if cached:
            return cached

        # Cache miss — fetch from API
        user = await self.api_client.fetch_user(user_id)
        if user:
            await self.cache.set(cache_key, user)
        return user

    async def get_users_batch(self, user_ids: List[int]) -> List[dict]:
        """Fetch multiple users concurrently using asyncio.gather.
        This is faster than sequential fetches for I/O-bound operations."""
        tasks = [self.get_user(uid) for uid in user_ids]
        return await asyncio.gather(*tasks)

    async def update_user(self, user_id: int, data: dict) -> dict:
        """Update user and invalidate cache."""
        result = await self.api_client.update_user(user_id, data)
        # Invalidate cache after update — stale cache is a common bug
        await self.cache.delete(f"user:{user_id}")
        return result


async def async_retry(coro_func, max_retries: int = 3, delay: float = 0.01):
    """Generic async retry with exponential backoff."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return await coro_func()
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                await asyncio.sleep(delay * (2 ** attempt))
    raise last_error


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def cache():
    """Fresh async cache for each test."""
    return AsyncCache()


@pytest.fixture
def mock_api():
    """Mock API client with async methods.
    AsyncMock is essential — regular Mock won't work with await."""
    api = AsyncMock()
    api.fetch_user = AsyncMock()
    api.update_user = AsyncMock()
    return api


@pytest.fixture
def user_service(mock_api, cache):
    """User service with mocked dependencies."""
    return AsyncUserService(api_client=mock_api, cache=cache)


# =============================================================================
# 1. BASIC ASYNC TESTS
# =============================================================================
# Mark async test functions with @pytest.mark.asyncio.
# pytest-asyncio creates and manages the event loop for you.

@pytest.mark.asyncio
async def test_cache_set_and_get(cache):
    """Basic async test: set a value, then get it back.
    The 'await' keyword makes this an async test — pytest-asyncio handles the loop."""
    await cache.set("key1", "value1")
    result = await cache.get("key1")
    assert result == "value1"


@pytest.mark.asyncio
async def test_cache_get_missing_key(cache):
    """Getting a non-existent key should return None, not raise."""
    result = await cache.get("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_cache_delete(cache):
    """Test the full set-delete-verify lifecycle."""
    await cache.set("temp", "data")
    assert await cache.delete("temp") is True
    assert await cache.get("temp") is None


@pytest.mark.asyncio
async def test_cache_delete_missing(cache):
    """Deleting a non-existent key should return False."""
    assert await cache.delete("nonexistent") is False


@pytest.mark.asyncio
async def test_cache_clear(cache):
    """Clear should remove all entries."""
    await cache.set("a", 1)
    await cache.set("b", 2)
    await cache.clear()
    assert cache.size == 0


# =============================================================================
# 2. TESTING WITH ASYNC MOCKS
# =============================================================================

@pytest.mark.asyncio
async def test_get_user_cache_miss(user_service, mock_api, cache):
    """On cache miss, service should fetch from API and populate cache."""
    mock_api.fetch_user.return_value = {"id": 1, "name": "Alice"}

    user = await user_service.get_user(1)

    assert user["name"] == "Alice"
    mock_api.fetch_user.assert_awaited_once_with(1)

    # Verify the cache was populated
    cached = await cache.get("user:1")
    assert cached["name"] == "Alice"


@pytest.mark.asyncio
async def test_get_user_cache_hit(user_service, mock_api, cache):
    """On cache hit, service should NOT call the API.
    This verifies the caching behavior reduces API calls."""
    # Pre-populate cache
    await cache.set("user:1", {"id": 1, "name": "Alice"})

    user = await user_service.get_user(1)

    assert user["name"] == "Alice"
    # API should NOT have been called — cache served the request
    mock_api.fetch_user.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_user_invalidates_cache(user_service, mock_api, cache):
    """After update, the cache entry must be invalidated.
    Stale cache after writes is a classic bug in cache-aside pattern."""
    # Set up cache and mock
    await cache.set("user:1", {"id": 1, "name": "Alice"})
    mock_api.update_user.return_value = {"id": 1, "name": "Alice Updated"}

    await user_service.update_user(1, {"name": "Alice Updated"})

    # Cache should be invalidated
    assert await cache.get("user:1") is None


# =============================================================================
# 3. TESTING CONCURRENT OPERATIONS
# =============================================================================

@pytest.mark.asyncio
async def test_batch_fetch(user_service, mock_api):
    """Test concurrent fetching with asyncio.gather."""
    mock_api.fetch_user.side_effect = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
        {"id": 3, "name": "Charlie"},
    ]

    users = await user_service.get_users_batch([1, 2, 3])

    assert len(users) == 3
    assert users[0]["name"] == "Alice"
    assert users[2]["name"] == "Charlie"
    assert mock_api.fetch_user.await_count == 3


@pytest.mark.asyncio
async def test_concurrent_cache_operations(cache):
    """Verify cache handles concurrent reads and writes correctly."""
    # Write many values concurrently
    write_tasks = [cache.set(f"key{i}", i) for i in range(100)]
    await asyncio.gather(*write_tasks)

    assert cache.size == 100

    # Read them all concurrently
    read_tasks = [cache.get(f"key{i}") for i in range(100)]
    results = await asyncio.gather(*read_tasks)

    assert all(results[i] == i for i in range(100))


# =============================================================================
# 4. TESTING ASYNC ERROR HANDLING
# =============================================================================

@pytest.mark.asyncio
async def test_api_error_propagation(user_service, mock_api):
    """Verify that API errors propagate correctly through async code."""
    mock_api.fetch_user.side_effect = ConnectionError("API unreachable")

    with pytest.raises(ConnectionError, match="API unreachable"):
        await user_service.get_user(1)


@pytest.mark.asyncio
async def test_async_retry_succeeds_eventually():
    """Test retry logic: fails twice, succeeds on third attempt."""
    call_count = 0

    async def flaky_operation():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError(f"Attempt {call_count} failed")
        return "success"

    result = await async_retry(flaky_operation, max_retries=3, delay=0.001)
    assert result == "success"
    assert call_count == 3


@pytest.mark.asyncio
async def test_async_retry_exhausted():
    """Test retry logic: all attempts fail, last error is raised."""
    async def always_fails():
        raise ValueError("permanent failure")

    with pytest.raises(ValueError, match="permanent failure"):
        await async_retry(always_fails, max_retries=3, delay=0.001)


# =============================================================================
# 5. TESTING TIMEOUTS
# =============================================================================

@pytest.mark.asyncio
async def test_operation_timeout():
    """Test that slow operations are properly cancelled with timeout."""
    async def slow_operation():
        await asyncio.sleep(10)  # Way too slow
        return "done"

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(slow_operation(), timeout=0.01)


@pytest.mark.asyncio
async def test_operation_within_timeout():
    """Test that fast operations complete within timeout."""
    async def fast_operation():
        await asyncio.sleep(0.001)
        return "done"

    result = await asyncio.wait_for(fast_operation(), timeout=1.0)
    assert result == "done"


# =============================================================================
# 6. ASYNC CONTEXT MANAGER TESTING
# =============================================================================

class AsyncDatabaseConnection:
    """Async context manager for database connections."""

    def __init__(self):
        self.connected = False
        self.queries = []

    async def __aenter__(self):
        await asyncio.sleep(0)  # Simulate connection time
        self.connected = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.connected = False
        return False  # Don't suppress exceptions

    async def execute(self, query: str) -> str:
        if not self.connected:
            raise RuntimeError("Not connected")
        self.queries.append(query)
        return f"Executed: {query}"


@pytest.mark.asyncio
async def test_async_context_manager():
    """Test async with statement lifecycle."""
    db = AsyncDatabaseConnection()
    assert not db.connected

    async with db as conn:
        assert conn.connected
        result = await conn.execute("SELECT 1")
        assert "Executed" in result

    # After exiting context, connection should be closed
    assert not db.connected


@pytest.mark.asyncio
async def test_async_context_manager_on_error():
    """Verify cleanup happens even when an exception occurs."""
    db = AsyncDatabaseConnection()

    with pytest.raises(ValueError):
        async with db as conn:
            await conn.execute("SELECT 1")
            raise ValueError("Something went wrong")

    # Connection must be closed even after error
    assert not db.connected
    assert len(db.queries) == 1


# =============================================================================
# RUNNING THIS FILE
# =============================================================================
# pip install pytest pytest-asyncio
# pytest 07_async_testing.py -v

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
