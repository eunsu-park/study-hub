"""
Exercises for Lesson 11: Testing and Quality
Topic: Python

Solutions to practice problems from the lesson.

Note: These solutions demonstrate testing concepts inline. In a real project,
tests would live in a separate test_*.py file and be run with pytest.
"""

import re
import sqlite3
from unittest.mock import MagicMock, patch


# === Exercise 1: Write Basic Tests ===
# Problem: Write tests for an email validation function.

def validate_email(email: str) -> bool:
    """Email validation using regex."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def exercise_1():
    """Demonstrate basic test writing with parametrize-style test cases.

    In a real pytest setup, you would use @pytest.mark.parametrize.
    Here we simulate the same pattern with a loop over test cases.
    """
    # Each tuple: (input_email, expected_result, description)
    test_cases = [
        ("user@example.com", True, "valid basic email"),
        ("invalid-email", False, "missing @ and domain"),
        ("user@domain", False, "missing TLD"),
        ("user.name+tag@example.co.kr", True, "valid with dots, plus, subdomain"),
        ("", False, "empty string"),
        ("@example.com", False, "missing local part"),
        ("user@.com", False, "missing domain name"),
        ("user@example.", False, "incomplete TLD"),
        ("a@b.cd", True, "minimal valid email"),
        ("user name@example.com", False, "space in local part"),
    ]

    passed = 0
    failed = 0
    for email, expected, description in test_cases:
        result = validate_email(email)
        status = "PASS" if result == expected else "FAIL"
        if status == "PASS":
            passed += 1
        else:
            failed += 1
        print(f"  [{status}] validate_email({email!r}) = {result} (expected {expected}) -- {description}")

    print(f"\n  Results: {passed} passed, {failed} failed out of {len(test_cases)}")


# === Exercise 2: Use Fixtures ===
# Problem: Write and use a database connection fixture.

class DatabaseFixture:
    """Simulates a pytest fixture for an in-memory SQLite database.

    In pytest, this would be a @pytest.fixture function that yields
    the connection and closes it in the teardown (after yield).
    """

    def __init__(self):
        self.conn = None

    def setup(self) -> sqlite3.Connection:
        """Set up an in-memory database with a users table."""
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)"
        )
        return self.conn

    def teardown(self):
        """Clean up the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None


def exercise_2():
    """Demonstrate fixture pattern for database testing."""
    fixture = DatabaseFixture()
    db = fixture.setup()

    try:
        # Test 1: Insert and retrieve a user
        db.execute("INSERT INTO users (name, email) VALUES (?, ?)", ("Alice", "alice@example.com"))
        result = db.execute("SELECT name, email FROM users WHERE name = ?", ("Alice",)).fetchone()
        assert result == ("Alice", "alice@example.com"), f"Expected Alice, got {result}"
        print("  [PASS] Insert and retrieve user")

        # Test 2: Insert multiple users and count
        db.execute("INSERT INTO users (name, email) VALUES (?, ?)", ("Bob", "bob@example.com"))
        db.execute("INSERT INTO users (name, email) VALUES (?, ?)", ("Charlie", "charlie@example.com"))
        count = db.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        assert count == 3, f"Expected 3 users, got {count}"
        print("  [PASS] Multiple inserts and count")

        # Test 3: Update a user
        db.execute("UPDATE users SET email = ? WHERE name = ?", ("alice@newdomain.com", "Alice"))
        new_email = db.execute("SELECT email FROM users WHERE name = ?", ("Alice",)).fetchone()[0]
        assert new_email == "alice@newdomain.com", f"Expected new email, got {new_email}"
        print("  [PASS] Update user email")

        # Test 4: Delete a user
        db.execute("DELETE FROM users WHERE name = ?", ("Bob",))
        count = db.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        assert count == 2, f"Expected 2 users after delete, got {count}"
        print("  [PASS] Delete user")

    finally:
        # Teardown always runs, even if a test fails -- like a pytest fixture
        fixture.teardown()
        print("  [INFO] Database fixture cleaned up")


# === Exercise 3: Mocking Practice ===
# Problem: Test a function that calls an external API using mocking.

class PaymentService:
    """Service that processes payments via an external API.

    The _http_post method is separated from the business logic so it
    can be easily mocked in tests without needing the requests library.
    """

    def _http_post(self, url: str, json_data: dict) -> dict:
        """Send an HTTP POST request. In production, uses requests library.

        Separated into its own method so tests can mock just this part,
        avoiding the need to install or patch the requests module.
        """
        # In real code: import requests; return requests.post(url, json=json_data).json()
        raise NotImplementedError("HTTP client not configured")

    def process_payment(self, amount: float) -> dict:
        """Process a payment by calling the external payment API."""
        return self._http_post(
            "https://payment.api/charge",
            {"amount": amount},
        )

    def refund(self, transaction_id: str, amount: float) -> dict:
        """Process a refund for a transaction."""
        return self._http_post(
            "https://payment.api/refund",
            {"transaction_id": transaction_id, "amount": amount},
        )


def exercise_3():
    """Demonstrate mocking external API calls.

    Uses unittest.mock.patch.object to replace the _http_post method
    on PaymentService instances. This approach works without installing
    the requests library, while still teaching the core mocking concepts:
    controlling return values and verifying call arguments.
    """
    # Test 1: Successful payment
    service = PaymentService()
    with patch.object(service, "_http_post") as mock_post:
        # Configure the mock to return a success response
        mock_post.return_value = {
            "status": "success",
            "transaction_id": "txn_123",
        }

        result = service.process_payment(100.0)

        assert result["status"] == "success"
        assert result["transaction_id"] == "txn_123"
        # Verify the mock was called with correct URL and payload
        mock_post.assert_called_once_with(
            "https://payment.api/charge",
            {"amount": 100.0},
        )
        print("  [PASS] Successful payment processing")

    # Test 2: Failed payment (API error)
    service = PaymentService()
    with patch.object(service, "_http_post") as mock_post:
        mock_post.return_value = {
            "status": "error",
            "message": "Insufficient funds",
        }

        result = service.process_payment(999999.0)

        assert result["status"] == "error"
        assert "Insufficient funds" in result["message"]
        print("  [PASS] Failed payment handling")

    # Test 3: Refund processing
    service = PaymentService()
    with patch.object(service, "_http_post") as mock_post:
        mock_post.return_value = {
            "status": "refunded",
            "refund_id": "ref_456",
        }

        result = service.refund("txn_123", 50.0)

        assert result["status"] == "refunded"
        mock_post.assert_called_once_with(
            "https://payment.api/refund",
            {"transaction_id": "txn_123", "amount": 50.0},
        )
        print("  [PASS] Refund processing")

    # Test 4: Demonstrate MagicMock for more complex scenarios
    print("\n  --- MagicMock demonstration ---")
    mock_service = MagicMock(spec=PaymentService)
    mock_service.process_payment.return_value = {"status": "success", "id": "txn_999"}

    result = mock_service.process_payment(50.0)
    print(f"  MagicMock result: {result}")
    print(f"  Called with: {mock_service.process_payment.call_args}")
    mock_service.process_payment.assert_called_once_with(50.0)
    print("  [PASS] MagicMock spec-based mocking")


if __name__ == "__main__":
    print("=== Exercise 1: Write Basic Tests ===")
    exercise_1()

    print("\n=== Exercise 2: Use Fixtures ===")
    exercise_2()

    print("\n=== Exercise 3: Mocking Practice ===")
    exercise_3()

    print("\nAll exercises completed!")
