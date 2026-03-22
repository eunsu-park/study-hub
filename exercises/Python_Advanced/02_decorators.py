"""
Exercises for Lesson 02: Decorators
Topic: Python

Solutions to practice problems from the lesson.
"""

import functools
import time
import threading
import logging
import io


# === Exercise 1: Execution Time Limit ===
# Problem: Create a decorator that raises TimeoutError if the function
# doesn't complete within a specified time.

def timeout(seconds: float):
    """Decorator that raises TimeoutError if function exceeds time limit.

    Uses a daemon thread to run the function and joins with a timeout.
    If the thread is still alive after the timeout, we raise TimeoutError.
    This approach avoids platform-specific signals (SIGALRM is Unix-only).
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e

            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            thread.join(timeout=seconds)

            if thread.is_alive():
                raise TimeoutError(
                    f"{func.__name__} did not complete within {seconds}s"
                )
            if exception[0] is not None:
                raise exception[0]
            return result[0]

        return wrapper
    return decorator


def exercise_1():
    """Demonstrate timeout decorator."""
    @timeout(1)
    def fast_function():
        time.sleep(0.1)
        return "done quickly"

    @timeout(0.5)
    def slow_function():
        time.sleep(2)
        return "done slowly"

    print(f"fast_function() = {fast_function()}")

    try:
        slow_function()
    except TimeoutError as e:
        print(f"TimeoutError: {e}")


# === Exercise 2: Result Logging ===
# Problem: Create a decorator that logs both input and output of a function to a file.

def log_results(log_file: str = "function_log.txt"):
    """Decorator that logs function calls (args, kwargs, result) to a file.

    Uses Python's logging module with a FileHandler so log output is
    appended rather than overwriting previous entries.
    """
    def decorator(func):
        # Set up a dedicated logger per function to avoid cross-contamination
        logger = logging.getLogger(f"log_results.{func.__name__}")
        logger.setLevel(logging.INFO)

        handler = logging.FileHandler(log_file)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(message)s")
        )
        logger.addHandler(handler)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            logger.info(
                f"CALL {func.__name__}(args={args}, kwargs={kwargs}) -> {result}"
            )
            return result

        return wrapper
    return decorator


def exercise_2():
    """Demonstrate result logging decorator (prints to a StringIO instead of file)."""
    # For demonstration, we use a StreamHandler with StringIO
    stream = io.StringIO()

    def log_to_stream(func):
        """Simplified version that logs to a stream for demonstration."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            log_entry = f"CALL {func.__name__}(args={args}, kwargs={kwargs}) -> {result}"
            stream.write(log_entry + "\n")
            print(f"  [LOG] {log_entry}")
            return result
        return wrapper

    @log_to_stream
    def add(a, b):
        return a + b

    @log_to_stream
    def greet(name, greeting="Hello"):
        return f"{greeting}, {name}!"

    add(3, 5)
    greet("Alice")
    greet("Bob", greeting="Hi")

    print(f"\nFull log contents:\n{stream.getvalue()}")


# === Exercise 3: Debug Mode ===
# Problem: Create a decorator that outputs debug information only when
# a DEBUG flag is True.

DEBUG = False  # Global flag


def debug_mode(func):
    """Decorator that prints call details only when DEBUG is True.

    Checks the global DEBUG flag at call time (not at decoration time),
    so toggling DEBUG mid-program takes effect immediately.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if DEBUG:
            print(f"[DEBUG] Calling {func.__name__}")
            print(f"[DEBUG]   args={args}, kwargs={kwargs}")

        result = func(*args, **kwargs)

        if DEBUG:
            print(f"[DEBUG]   returned {result}")

        return result
    return wrapper


def exercise_3():
    """Demonstrate debug mode decorator."""
    global DEBUG

    @debug_mode
    def multiply(a, b):
        return a * b

    print("With DEBUG=False:")
    result = multiply(3, 4)
    print(f"  multiply(3, 4) = {result}")

    DEBUG = True
    print("\nWith DEBUG=True:")
    result = multiply(5, 6)
    print(f"  multiply(5, 6) = {result}")

    DEBUG = False  # Reset


if __name__ == "__main__":
    print("=== Exercise 1: Execution Time Limit ===")
    exercise_1()

    print("\n=== Exercise 2: Result Logging ===")
    exercise_2()

    print("\n=== Exercise 3: Debug Mode ===")
    exercise_3()

    print("\nAll exercises completed!")
