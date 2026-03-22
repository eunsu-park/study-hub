"""
Exercises for Lesson 03: Context Managers
Topic: Python

Solutions to practice problems from the lesson.
"""

import signal
import logging
import time
from contextlib import contextmanager


# === Exercise 1: Timeout Context Manager ===
# Problem: Create a context manager that raises TimeoutError after a specified time.

@contextmanager
def timeout(seconds: float):
    """Context manager that raises TimeoutError after `seconds`.

    Uses SIGALRM on Unix systems. For a cross-platform alternative,
    a threading-based approach would be needed (see lesson 02 exercise).
    Falls back to a no-op warning on platforms without SIGALRM.
    """
    if not hasattr(signal, "SIGALRM"):
        print(f"[WARNING] SIGALRM not available; timeout will not be enforced")
        yield
        return

    def handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds}s")

    # Save old handler, install ours, set the alarm
    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        # Cancel alarm and restore original handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def exercise_1():
    """Demonstrate timeout context manager."""
    # Fast operation -- completes before timeout
    try:
        with timeout(2):
            total = sum(range(1_000_000))
            print(f"Fast operation completed: sum = {total}")
    except TimeoutError as e:
        print(f"TimeoutError: {e}")

    # Slow operation -- exceeds timeout
    try:
        with timeout(1):
            print("Starting slow operation...")
            time.sleep(3)
            print("This should not be printed")
    except TimeoutError as e:
        print(f"TimeoutError: {e}")


# === Exercise 2: Log Level Change ===
# Problem: Create a context manager that temporarily changes the logging level
# and then restores it.

@contextmanager
def log_level(logger: logging.Logger, level: int):
    """Temporarily change a logger's level, then restore the original.

    Useful for enabling verbose logging in a specific code section
    without permanently changing the configuration.
    """
    old_level = logger.level
    logger.setLevel(level)
    try:
        yield logger
    finally:
        logger.setLevel(old_level)


def exercise_2():
    """Demonstrate log level context manager."""
    # Set up logger with a handler so we can see output
    logger = logging.getLogger("demo")
    logger.setLevel(logging.WARNING)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("  [%(levelname)s] %(message)s"))
    logger.addHandler(handler)

    print(f"Logger level before: {logging.getLevelName(logger.level)}")
    logger.debug("This DEBUG message is hidden (level=WARNING)")
    logger.warning("This WARNING message is visible")

    with log_level(logger, logging.DEBUG) as lg:
        print(f"Logger level inside: {logging.getLevelName(lg.level)}")
        lg.debug("This DEBUG message is now visible (level=DEBUG)")

    print(f"Logger level after: {logging.getLevelName(logger.level)}")
    logger.debug("This DEBUG message is hidden again")

    # Clean up handler to avoid duplicate output in later runs
    logger.removeHandler(handler)


# === Exercise 3: Test Double ===
# Problem: Create a context manager that temporarily replaces a function
# for testing purposes.

@contextmanager
def mock_function(obj, attr_name: str, replacement):
    """Temporarily replace an attribute on an object with a mock.

    This is a simplified version of unittest.mock.patch. It saves the
    original attribute, installs the replacement, and restores the
    original when the block exits -- even if an exception occurs.
    """
    original = getattr(obj, attr_name)
    setattr(obj, attr_name, replacement)
    try:
        yield replacement
    finally:
        setattr(obj, attr_name, original)


# A sample module-level function to demonstrate mocking
class WeatherService:
    """Simulates a service that fetches weather data."""

    @staticmethod
    def get_temperature(city: str) -> float:
        """In production, this would call an external API."""
        raise NotImplementedError("Real API not available in demo")


def exercise_3():
    """Demonstrate test double context manager."""
    # Without mock -- the real function raises an error
    try:
        temp = WeatherService.get_temperature("Seoul")
    except NotImplementedError as e:
        print(f"Without mock: {e}")

    # With mock -- the replacement returns a fixed value
    def fake_temperature(city: str) -> float:
        return {"Seoul": 22.5, "Tokyo": 18.3}.get(city, 20.0)

    with mock_function(WeatherService, "get_temperature", staticmethod(fake_temperature)):
        temp = WeatherService.get_temperature("Seoul")
        print(f"With mock: Seoul temperature = {temp}")

        temp = WeatherService.get_temperature("Tokyo")
        print(f"With mock: Tokyo temperature = {temp}")

    # After the context, the original is restored
    try:
        WeatherService.get_temperature("Seoul")
    except NotImplementedError:
        print("After mock: original function restored (raises NotImplementedError)")


if __name__ == "__main__":
    print("=== Exercise 1: Timeout Context Manager ===")
    exercise_1()

    print("\n=== Exercise 2: Log Level Change ===")
    exercise_2()

    print("\n=== Exercise 3: Test Double ===")
    exercise_3()

    print("\nAll exercises completed!")
