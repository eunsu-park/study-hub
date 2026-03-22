"""
Exercises for Lesson 05: Closures and Scope
Topic: Python

Solutions to practice problems from the lesson.
"""

import time
from typing import Callable, Any, Tuple


# === Exercise 1: Counter Factory ===
# Problem: Create a counter factory that allows setting start value and increment.
# counter = make_counter(start=10, step=5)
# counter() -> 10, counter() -> 15, counter() -> 20

def make_counter(start: int = 0, step: int = 1) -> Callable[[], int]:
    """Return a closure that counts from `start` by `step` on each call.

    The current value lives in the enclosing scope. We use `nonlocal`
    to mutate it from within the inner function, because assignment
    would otherwise create a new local variable.
    """
    current = start

    def counter() -> int:
        nonlocal current
        value = current
        current += step
        return value

    return counter


def exercise_1():
    """Demonstrate counter factory."""
    counter = make_counter(start=10, step=5)
    print(f"counter() = {counter()}")  # 10
    print(f"counter() = {counter()}")  # 15
    print(f"counter() = {counter()}")  # 20

    # Different counter with default args
    c2 = make_counter()
    print(f"\nc2() = {c2()}")  # 0
    print(f"c2() = {c2()}")    # 1
    print(f"c2() = {c2()}")    # 2


# === Exercise 2: Function Call History ===
# Problem: Create a closure that records function call history.
# tracked_add, get_history = track_calls(add)
# tracked_add(1, 2)
# tracked_add(3, 4)
# get_history() -> [(1, 2, 3), (3, 4, 7)]

def track_calls(func: Callable) -> Tuple[Callable, Callable[[], list]]:
    """Wrap a function to record its call history.

    Returns two functions:
      - tracked: calls the original function and records (args..., result)
      - get_history: returns the accumulated call log

    The history list is shared between both closures via the
    enclosing scope. This is a lightweight alternative to a class-based
    approach when you only need call tracking.
    """
    history: list[tuple] = []

    def tracked(*args, **kwargs):
        result = func(*args, **kwargs)
        # Store positional args + result as a flat tuple
        history.append((*args, result))
        return result

    def get_history() -> list[tuple]:
        return history.copy()  # Return a copy to prevent external mutation

    return tracked, get_history


def exercise_2():
    """Demonstrate function call history tracking."""
    def add(a: int, b: int) -> int:
        return a + b

    tracked_add, get_history = track_calls(add)
    tracked_add(1, 2)
    tracked_add(3, 4)
    tracked_add(10, 20)

    print(f"History: {get_history()}")
    # [(1, 2, 3), (3, 4, 7), (10, 20, 30)]


# === Exercise 3: Rate Limiter ===
# Problem: Create a closure that limits calls per second.

def rate_limiter(max_calls: int, per_seconds: float = 1.0) -> Callable[[Callable], Callable]:
    """Create a rate-limiting wrapper.

    Tracks timestamps of recent calls in a sliding window. If the number
    of calls within the window exceeds `max_calls`, the wrapper raises
    a RuntimeError instead of calling the function.
    """
    def decorator(func: Callable) -> Callable:
        call_times: list[float] = []

        def wrapper(*args, **kwargs):
            now = time.time()
            # Remove timestamps outside the sliding window
            while call_times and call_times[0] < now - per_seconds:
                call_times.pop(0)

            if len(call_times) >= max_calls:
                raise RuntimeError(
                    f"Rate limit exceeded: {max_calls} calls per {per_seconds}s"
                )

            call_times.append(now)
            return func(*args, **kwargs)

        return wrapper
    return decorator


def exercise_3():
    """Demonstrate rate limiter."""
    @rate_limiter(max_calls=3, per_seconds=1.0)
    def api_call(endpoint: str) -> str:
        return f"Response from {endpoint}"

    # First 3 calls succeed
    for i in range(3):
        result = api_call(f"/endpoint/{i}")
        print(f"Call {i + 1}: {result}")

    # 4th call within the same second should fail
    try:
        api_call("/endpoint/4")
    except RuntimeError as e:
        print(f"Call 4: {e}")

    # Wait for the window to expire, then call again
    print("\nWaiting 1 second for rate limit to reset...")
    time.sleep(1.1)
    result = api_call("/endpoint/5")
    print(f"Call 5 (after wait): {result}")


if __name__ == "__main__":
    print("=== Exercise 1: Counter Factory ===")
    exercise_1()

    print("\n=== Exercise 2: Function Call History ===")
    exercise_2()

    print("\n=== Exercise 3: Rate Limiter ===")
    exercise_3()

    print("\nAll exercises completed!")
