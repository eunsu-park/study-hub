"""
12 Exceptions
=============
Demonstrates try/except/else/finally patterns, custom exceptions,
exception chaining, EAFP vs LBYL, and practical error handling.
"""


def basic_try_except():
    """Basic exception handling patterns."""
    # Catch specific exception
    try:
        result = 10 / 0
    except ZeroDivisionError:
        print("Caught ZeroDivisionError: division by zero")

    # Catch with exception object
    try:
        numbers = [1, 2, 3]
        print(numbers[10])
    except IndexError as e:
        print(f"Caught IndexError: {e}")

    # Multiple except clauses
    test_cases = [("10", "valid"), ("abc", "invalid"), (None, "None")]
    for value, label in test_cases:
        try:
            result = int(value)
            print(f"  int({value!r}) = {result}")
        except ValueError:
            print(f"  int({value!r}) -> ValueError")
        except TypeError:
            print(f"  int({value!r}) -> TypeError")

    # Catch multiple exception types in one clause
    try:
        d = {}
        d["missing"]
    except (KeyError, IndexError) as e:
        print(f"\nCaught {type(e).__name__}: {e}")


def else_and_finally():
    """The else and finally clauses."""

    def divide(a, b):
        try:
            result = a / b
        except ZeroDivisionError:
            print(f"  {a}/{b}: Cannot divide by zero")
            return None
        else:
            # Runs only if no exception occurred
            print(f"  {a}/{b} = {result:.2f} (else block ran)")
            return result
        finally:
            # Always runs, even after return
            print(f"  {a}/{b}: finally block (always runs)")

    divide(10, 3)
    print()
    divide(10, 0)

    # finally guarantees cleanup
    print("\nFinally for resource cleanup:")

    class FakeConnection:
        def __init__(self):
            print("  Connection opened")

        def close(self):
            print("  Connection closed")

    conn = FakeConnection()
    try:
        print("  Doing work...")
        raise RuntimeError("Something went wrong")
    except RuntimeError as e:
        print(f"  Handled: {e}")
    finally:
        conn.close()


def custom_exceptions():
    """Define and use custom exception classes."""

    class AppError(Exception):
        """Base exception for our application."""

    class ValidationError(AppError):
        """Raised when input validation fails."""
        def __init__(self, field, message):
            self.field = field
            self.message = message
            super().__init__(f"{field}: {message}")

    class NotFoundError(AppError):
        """Raised when a resource is not found."""
        def __init__(self, resource, identifier):
            self.resource = resource
            self.identifier = identifier
            super().__init__(f"{resource} '{identifier}' not found")

    class RateLimitError(AppError):
        """Raised when rate limit is exceeded."""
        def __init__(self, retry_after=60):
            self.retry_after = retry_after
            super().__init__(f"Rate limited. Retry after {retry_after}s")

    # Using custom exceptions
    def validate_age(age):
        if not isinstance(age, int):
            raise ValidationError("age", "must be an integer")
        if age < 0 or age > 150:
            raise ValidationError("age", f"must be 0-150, got {age}")
        return age

    test_ages = [25, -5, "abc", 200]
    for age in test_ages:
        try:
            result = validate_age(age)
            print(f"  validate_age({age!r}) = {result}")
        except ValidationError as e:
            print(f"  validate_age({age!r}) -> ValidationError: {e}")

    # Catch hierarchy
    print("\nCatching base class catches all subclasses:")
    errors = [
        ValidationError("email", "invalid format"),
        NotFoundError("User", "42"),
        RateLimitError(30),
    ]
    for err in errors:
        try:
            raise err
        except AppError as e:
            print(f"  {type(e).__name__}: {e}")


def exception_chaining():
    """Exception chaining with 'from' and implicit chaining."""

    def fetch_config(key):
        config = {"host": "localhost", "port": "8080"}
        try:
            return config[key]
        except KeyError as e:
            raise ValueError(f"Missing config key: {key}") from e

    try:
        fetch_config("database")
    except ValueError as e:
        print(f"Caught: {e}")
        print(f"Caused by: {e.__cause__}")

    # Suppress chaining with 'from None'
    def parse_int(s):
        try:
            return int(s)
        except ValueError:
            raise TypeError(f"Cannot parse {s!r} as integer") from None

    try:
        parse_int("abc")
    except TypeError as e:
        print(f"\nSuppressed chain: {e}")
        print(f"__cause__ is None: {e.__cause__ is None}")


def eafp_vs_lbyl():
    """EAFP (Easier to Ask Forgiveness) vs LBYL (Look Before You Leap)."""

    data = {"users": [{"name": "Alice", "age": 30}]}

    # LBYL: Check everything before accessing
    print("LBYL approach:")
    if "users" in data and len(data["users"]) > 0 and "name" in data["users"][0]:
        name = data["users"][0]["name"]
        print(f"  Name: {name}")
    else:
        print("  Data not available")

    # EAFP: Try and handle failure (more Pythonic)
    print("\nEAFP approach (Pythonic):")
    try:
        name = data["users"][0]["name"]
        print(f"  Name: {name}")
    except (KeyError, IndexError, TypeError):
        print("  Data not available")

    # EAFP for file existence
    print("\nFile access:")
    from pathlib import Path

    # LBYL
    path = Path("/tmp/test_file.txt")
    if path.exists():
        content = path.read_text()
    else:
        print("  LBYL: File not found")

    # EAFP (handles race conditions too)
    try:
        content = path.read_text()
    except FileNotFoundError:
        print("  EAFP: File not found")

    # EAFP for type conversion
    print("\nType conversion:")
    values = ["42", "3.14", "hello", "0xFF"]
    for v in values:
        try:
            n = int(v, 0) if v.startswith("0") else int(v)
            print(f"  {v!r:>8} -> int: {n}")
        except ValueError:
            try:
                n = float(v)
                print(f"  {v!r:>8} -> float: {n}")
            except ValueError:
                print(f"  {v!r:>8} -> not a number")


def context_manager_exception():
    """How context managers handle exceptions."""

    class ManagedResource:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            print(f"  Acquired: {self.name}")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            print(f"  Released: {self.name}")
            if exc_type is not None:
                print(f"  Exception in context: {exc_type.__name__}: {exc_val}")
            # Return False (or None) to propagate, True to suppress
            return False

    # Normal usage
    print("Normal:")
    with ManagedResource("DB Connection") as r:
        print(f"  Using {r.name}")

    # With exception
    print("\nWith exception:")
    try:
        with ManagedResource("DB Connection") as r:
            print(f"  Using {r.name}")
            raise RuntimeError("Query failed")
    except RuntimeError:
        print("  Exception propagated and handled outside")


def practical_patterns():
    """Real-world exception handling patterns."""

    # Retry pattern
    def retry(func, max_attempts=3, exceptions=(Exception,)):
        for attempt in range(1, max_attempts + 1):
            try:
                return func()
            except exceptions as e:
                print(f"    Attempt {attempt} failed: {e}")
                if attempt == max_attempts:
                    raise

    call_count = 0

    def flaky_operation():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError(f"Connection refused (attempt {call_count})")
        return "Success!"

    print("Retry pattern:")
    try:
        result = retry(flaky_operation, max_attempts=3, exceptions=(ConnectionError,))
        print(f"  Result: {result}")
    except ConnectionError:
        print("  All attempts failed")

    # Collecting errors
    print("\nCollecting errors:")
    items = ["10", "abc", "20", "xyz", "30"]
    results = []
    errors = []
    for item in items:
        try:
            results.append(int(item))
        except ValueError as e:
            errors.append((item, str(e)))

    print(f"  Parsed: {results}")
    print(f"  Errors: {errors}")


if __name__ == "__main__":
    sections = [
        ("Basic try/except", basic_try_except),
        ("else and finally", else_and_finally),
        ("Custom Exceptions", custom_exceptions),
        ("Exception Chaining", exception_chaining),
        ("EAFP vs LBYL", eafp_vs_lbyl),
        ("Context Manager + Exceptions", context_manager_exception),
        ("Practical Patterns", practical_patterns),
    ]

    for title, func in sections:
        print(f"\n{'=' * 50}")
        print(f"  {title}")
        print('=' * 50)
        func()
