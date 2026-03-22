"""
Exercise 12: Exceptions

Practice custom exceptions, retry logic, and input validation.
"""

import time


# --- Exercise 1: Custom Exception Hierarchy ---

class AppError(Exception):
    """Base exception for the application."""
    # TODO: Implement with a message attribute
    pass


class ValidationError(AppError):
    """Raised when input validation fails.

    Attributes:
        field: Name of the field that failed validation.
        message: Description of the error.
    """
    # TODO: Implement with field and message attributes
    # __str__ should return "ValidationError({field}): {message}"
    pass


class NotFoundError(AppError):
    """Raised when a resource is not found.

    Attributes:
        resource: Type of resource (e.g., "User").
        resource_id: ID of the missing resource.
    """
    # TODO: Implement with resource and resource_id attributes
    # __str__ should return "{resource} not found: {resource_id}"
    pass


# --- Exercise 2: Validate User Input ---

def validate_user(name, age, email):
    """Validate user registration data.

    Rules:
    - name: must be non-empty string (raise ValidationError field="name")
    - age: must be int between 0 and 150 (raise ValidationError field="age")
    - email: must contain '@' (raise ValidationError field="email")

    Args:
        name: User's name.
        age: User's age.
        email: User's email.

    Returns:
        dict with validated data: {"name": name, "age": age, "email": email}

    Raises:
        ValidationError: If any field is invalid.
    """
    # TODO: Implement validation with specific error messages
    pass


# --- Exercise 3: Retry Decorator ---

def retry(max_attempts=3, exceptions=(Exception,)):
    """Decorator that retries a function on failure.

    Args:
        max_attempts: Maximum number of attempts.
        exceptions: Tuple of exception types to catch and retry.

    Returns:
        Decorator function.

    If all attempts fail, re-raise the last exception.
    """
    # TODO: Implement the decorator
    pass


# --- Exercise 4: Safe Division ---

def safe_divide(a, b):
    """Safely divide a by b, handling various error types.

    Returns:
        A dict with either:
            {"success": True, "result": value}
            {"success": False, "error": "error description"}

    Handle: ZeroDivisionError, TypeError
    """
    # TODO: Implement this
    pass


# --- Exercise 5: Context-like Resource Manager ---

def process_with_cleanup(data, processor):
    """Process data with guaranteed cleanup.

    Call processor(data). Whether it succeeds or fails,
    return a dict:
        {"processed": True/False, "result": value_or_None,
         "error": error_msg_or_None, "cleaned_up": True}

    Args:
        data: Data to process.
        processor: Callable that processes data.

    Returns:
        Result dict with cleanup flag always True.
    """
    # TODO: Use try/except/finally pattern
    pass


# === Tests ===

# Custom exceptions
assert issubclass(ValidationError, AppError), "ValidationError inherits AppError"
assert issubclass(NotFoundError, AppError), "NotFoundError inherits AppError"

ve = ValidationError("email", "must contain @")
assert str(ve) == "ValidationError(email): must contain @", "ValidationError str"
nf = NotFoundError("User", 42)
assert str(nf) == "User not found: 42", "NotFoundError str"

# Validate user
user = validate_user("Alice", 30, "alice@test.com")
assert user == {"name": "Alice", "age": 30, "email": "alice@test.com"}, "Valid user"
try:
    validate_user("", 30, "alice@test.com")
    assert False, "Empty name should fail"
except ValidationError as e:
    assert e.field == "name", "Error on name field"
try:
    validate_user("Alice", -1, "alice@test.com")
    assert False, "Negative age should fail"
except ValidationError as e:
    assert e.field == "age", "Error on age field"
try:
    validate_user("Alice", 30, "invalid")
    assert False, "Invalid email should fail"
except ValidationError as e:
    assert e.field == "email", "Error on email field"

# Retry decorator
call_count = 0

@retry(max_attempts=3, exceptions=(ValueError,))
def flaky_function():
    global call_count
    call_count += 1
    if call_count < 3:
        raise ValueError("not ready")
    return "success"

call_count = 0
assert flaky_function() == "success", "Retry succeeds on 3rd attempt"
assert call_count == 3, "Called 3 times"

# Safe divide
assert safe_divide(10, 2) == {"success": True, "result": 5.0}, "Normal divide"
assert safe_divide(10, 0)["success"] is False, "Divide by zero"
assert safe_divide("a", 2)["success"] is False, "Type error"

# Process with cleanup
result = process_with_cleanup(5, lambda x: x * 2)
assert result == {"processed": True, "result": 10, "error": None, "cleaned_up": True}

result = process_with_cleanup(5, lambda x: 1 / 0)
assert result["processed"] is False, "Failed processing"
assert result["cleaned_up"] is True, "Cleanup happened"
assert result["error"] is not None, "Has error message"

print("All tests passed!")
