"""
Exercises for Lesson 09: Error Handling
Topic: Programming

Solutions to practice problems from the lesson.
"""
import json
import random
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Tuple, Union


# === Exercise 1: Refactor Poor Error Handling ===
# Problem: Fix a file processing function with no error handling.

def exercise_1():
    """Solution: Robust file processing with proper error handling."""

    # Original (broken):
    # def process_file(filename):
    #     file = open(filename, 'r')       # No error if file missing
    #     data = file.read()               # File not closed on error
    #     file.close()                     # Skipped if exception above
    #     result = json.loads(data)         # No error if invalid JSON
    #     return result['value'] * 2        # No error if 'value' missing or non-numeric

    def process_file(filename):
        """
        Read a JSON file and double the 'value' field.

        Uses context manager (with) to guarantee file closure even on errors.
        Each potential failure point has explicit handling with informative messages.
        """
        try:
            with open(filename, "r") as file:
                data = file.read()
        except FileNotFoundError:
            print(f"    Error: File '{filename}' not found")
            return None
        except PermissionError:
            print(f"    Error: No permission to read '{filename}'")
            return None

        try:
            result = json.loads(data)
        except json.JSONDecodeError as e:
            print(f"    Error: Invalid JSON in '{filename}': {e}")
            return None

        if "value" not in result:
            print(f"    Error: Missing 'value' key in JSON data")
            return None

        value = result["value"]
        if not isinstance(value, (int, float)):
            print(f"    Error: 'value' is not numeric (got {type(value).__name__})")
            return None

        return value * 2

    # Test with simulated scenarios (we can't create files, so demonstrate the logic)
    import tempfile
    import os

    # Test 1: Valid file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"value": 21}, f)
        valid_file = f.name

    print(f"  Valid file: {process_file(valid_file)}")
    os.unlink(valid_file)

    # Test 2: Missing file
    print(f"  Missing file: {process_file('nonexistent.json')}")

    # Test 3: Invalid JSON
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("not valid json{{{")
        bad_json = f.name

    print(f"  Bad JSON: {process_file(bad_json)}")
    os.unlink(bad_json)

    # Test 4: Missing key
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"other_key": 42}, f)
        missing_key = f.name

    print(f"  Missing key: {process_file(missing_key)}")
    os.unlink(missing_key)


# === Exercise 2: Design an Error Strategy ===
# Problem: Design error handling for a REST API payment system.

def exercise_2():
    """Solution: Error handling strategy for a payment API."""

    # Custom exception hierarchy for the payment domain.
    # Each exception type maps to a specific HTTP status code.

    class PaymentError(Exception):
        """Base class for payment-related errors."""
        status_code = 500

    class ValidationError(PaymentError):
        """Invalid request data (400 Bad Request)."""
        status_code = 400

        def __init__(self, field, message):
            self.field = field
            super().__init__(f"Validation error on '{field}': {message}")

    class InsufficientFundsError(PaymentError):
        """Not enough balance (422 Unprocessable Entity)."""
        status_code = 422

    class GatewayTimeoutError(PaymentError):
        """Payment gateway did not respond (504 Gateway Timeout)."""
        status_code = 504

    class GatewayRejectionError(PaymentError):
        """Payment gateway rejected the transaction (502 Bad Gateway)."""
        status_code = 502

    class DatabaseError(PaymentError):
        """Database connection failure (503 Service Unavailable)."""
        status_code = 503

    def format_error_response(error):
        """Create a standardized JSON error response."""
        return {
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "status_code": error.status_code,
            }
        }

    def process_payment(request):
        """Simulate payment processing with comprehensive error handling."""
        # Step 1: Validate request
        if "amount" not in request:
            raise ValidationError("amount", "Amount is required")
        if not isinstance(request["amount"], (int, float)):
            raise ValidationError("amount", "Must be a number")
        if request["amount"] <= 0:
            raise ValidationError("amount", "Must be positive")

        # Step 2: Check balance (simulated)
        balance = 100.0
        if request["amount"] > balance:
            raise InsufficientFundsError(
                f"Requested ${request['amount']:.2f}, available ${balance:.2f}"
            )

        return {"status": "success", "transaction_id": "TXN-001"}

    # Demonstrate each error scenario
    test_requests = [
        ({"amount": 50.0}, "Valid payment"),
        ({}, "Missing amount"),
        ({"amount": "fifty"}, "Non-numeric amount"),
        ({"amount": -10}, "Negative amount"),
        ({"amount": 500.0}, "Insufficient funds"),
    ]

    for request, description in test_requests:
        try:
            result = process_payment(request)
            print(f"  {description}: {result}")
        except PaymentError as e:
            response = format_error_response(e)
            print(f"  {description}: [{e.status_code}] {response['error']['message']}")


# === Exercise 3: Implement Retry Logic ===
# Problem: Retry with exponential backoff and circuit breaker.

def exercise_3():
    """Solution: Retry with exponential backoff and circuit breaker."""

    class CircuitBreakerOpen(Exception):
        """Raised when the circuit breaker is open (too many failures)."""
        pass

    class CircuitBreaker:
        """
        Circuit breaker pattern: stop calling a failing service after
        consecutive failures, then periodically test if it recovered.
        """
        def __init__(self, failure_threshold=3, recovery_timeout=5.0):
            self._failure_count = 0
            self._failure_threshold = failure_threshold
            self._recovery_timeout = recovery_timeout
            self._last_failure_time = 0
            self._state = "closed"  # closed = normal, open = blocking

        @property
        def state(self):
            if self._state == "open":
                # Check if recovery timeout has passed
                if time.time() - self._last_failure_time >= self._recovery_timeout:
                    self._state = "half-open"
            return self._state

        def record_success(self):
            self._failure_count = 0
            self._state = "closed"

        def record_failure(self):
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._failure_count >= self._failure_threshold:
                self._state = "open"

    def retry_with_circuit_breaker(func, max_retries=3, base_delay=0.01,
                                    circuit_breaker=None):
        """
        Retry a function with exponential backoff.

        Backoff doubles the delay each retry: 0.01s, 0.02s, 0.04s, ...
        Circuit breaker opens after consecutive failures to avoid
        hammering a down service.
        """
        cb = circuit_breaker or CircuitBreaker(failure_threshold=max_retries)

        for attempt in range(1, max_retries + 1):
            if cb.state == "open":
                return (False, f"Circuit breaker OPEN after {cb._failure_count} failures")

            try:
                result = func()
                cb.record_success()
                return (True, result)
            except Exception as e:
                cb.record_failure()
                delay = base_delay * (2 ** (attempt - 1))
                print(f"    Attempt {attempt} failed: {e} (retry in {delay:.3f}s)")
                if attempt < max_retries:
                    time.sleep(delay)

        return (False, f"All {max_retries} attempts failed")

    # Test with a function that fails randomly
    call_count = 0

    def unreliable_service():
        """Simulate a service that fails 60% of the time."""
        nonlocal call_count
        call_count += 1
        if random.random() < 0.6:
            raise ConnectionError("Service unavailable")
        return f"Success on call #{call_count}"

    random.seed(42)  # Reproducible results
    print("  Testing retry with exponential backoff:")
    for i in range(3):
        call_count = 0
        success, result = retry_with_circuit_breaker(
            unreliable_service, max_retries=4, base_delay=0.001
        )
        status = "OK" if success else "FAIL"
        print(f"  Run {i + 1}: [{status}] {result}")


# === Exercise 4: Error Message Improvement ===
# Problem: Improve four bad error messages.

def exercise_4():
    """Solution: Transform cryptic errors into helpful messages."""

    improvements = [
        {
            "original": "Error 42",
            "developer": "DatabaseConnectionError(code=42): Connection pool exhausted. "
                         "Active connections: 100/100. Oldest idle: 45s. "
                         "Check max_connections setting and connection leak in UserService.query().",
            "user": "We're experiencing high demand right now. Please try again in a few moments. "
                    "If this persists, contact support with reference code ERR-42.",
        },
        {
            "original": "Invalid input",
            "developer": "ValidationError: Field 'email' value 'john@' failed regex validation. "
                         "Expected format: user@domain.tld. Input length: 5, min required: 6.",
            "user": "The email address 'john@' doesn't appear to be valid. "
                    "Please enter a complete email address (e.g., john@example.com).",
        },
        {
            "original": "NullPointerException at line 127",
            "developer": "NullPointerException in UserService.java:127 - user.getAddress().getCity() "
                         "returned null. user_id=12345 has no address record. "
                         "Stack: UserService.getCity() -> AddressFormatter.format().",
            "user": "We couldn't load your profile information. "
                    "Please update your address in Settings and try again.",
        },
        {
            "original": "Cannot process request",
            "developer": "PaymentGatewayTimeout: Stripe API did not respond within 30s. "
                         "Request ID: req_abc123. Endpoint: POST /v1/charges. "
                         "Retries exhausted (3/3). Last error: ETIMEDOUT.",
            "user": "Your payment is taking longer than expected. "
                    "You have NOT been charged. Please wait 5 minutes and try again, "
                    "or use a different payment method.",
        },
    ]

    for item in improvements:
        print(f"  Original: \"{item['original']}\"")
        print(f"  Developer: {item['developer']}")
        print(f"  User: {item['user']}")
        print()


# === Exercise 5: Exception vs Result Type ===
# Problem: Parse a date string using both approaches.

def exercise_5():
    """Solution: Date parsing with exceptions and Result type."""

    # Approach 1: Exception-based
    def parse_date_exception(text):
        """
        Parse date string (YYYY-MM-DD) using exceptions.

        Pro: Clean happy path, standard Python pattern.
        Con: Caller might forget to catch the exception.
        """
        try:
            return datetime.strptime(text, "%Y-%m-%d").date()
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid date '{text}': expected YYYY-MM-DD format") from e

    # Approach 2: Result type (tuple-based)
    @dataclass
    class Result:
        """Lightweight Result type: holds either a value or an error message."""
        success: bool
        value: object = None
        error: str = ""

        @staticmethod
        def ok(value):
            return Result(success=True, value=value)

        @staticmethod
        def err(message):
            return Result(success=False, error=message)

    def parse_date_result(text):
        """
        Parse date string (YYYY-MM-DD) using Result type.

        Pro: Caller is forced to check success before using value.
        Con: More verbose calling code.
        """
        if not isinstance(text, str):
            return Result.err(f"Expected string, got {type(text).__name__}")
        try:
            date = datetime.strptime(text, "%Y-%m-%d").date()
            return Result.ok(date)
        except ValueError:
            return Result.err(f"Invalid date '{text}': expected YYYY-MM-DD format")

    # Test both approaches
    test_inputs = ["2026-02-27", "2026-13-01", "not-a-date", "", None]

    print("  Exception-based approach:")
    for text in test_inputs:
        try:
            date = parse_date_exception(text)
            print(f"    '{text}' -> {date}")
        except (ValueError, TypeError) as e:
            print(f"    '{text}' -> Error: {e}")

    print("\n  Result-type approach:")
    for text in test_inputs:
        result = parse_date_result(text)
        if result.success:
            print(f"    '{text}' -> {result.value}")
        else:
            print(f"    '{text}' -> Error: {result.error}")

    print("\n  When to use each:")
    print("    Exceptions: When failure is truly exceptional (file not found, network error)")
    print("    Result type: When failure is expected (user input parsing, validation)")
    print("    Result type: When composing multiple operations that each might fail")


if __name__ == "__main__":
    print("=== Exercise 1: Refactor Poor Error Handling ===")
    exercise_1()
    print("\n=== Exercise 2: Design an Error Strategy ===")
    exercise_2()
    print("\n=== Exercise 3: Implement Retry Logic ===")
    exercise_3()
    print("\n=== Exercise 4: Error Message Improvement ===")
    exercise_4()
    print("\n=== Exercise 5: Exception vs Result Type ===")
    exercise_5()
    print("\nAll exercises completed!")
