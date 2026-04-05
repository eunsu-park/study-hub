#!/bin/bash
# Exercises for Lesson 04: Mocking and Test Doubles
# Topic: Testing_and_QA
# Solutions to practice problems from the lesson.

# === Exercise 1: Types of Test Doubles ===
# Problem: Implement stub, spy, and mock versions of a payment gateway
# interface and write tests using each.
exercise_1() {
    echo "=== Exercise 1: Types of Test Doubles ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import pytest
from unittest.mock import Mock, call

# Interface to be doubled
class PaymentGateway:
    def charge(self, amount: float, card_token: str) -> dict:
        raise NotImplementedError("Real payment processing")

# --- STUB: Returns pre-configured responses ---
# Use when you need controlled return values but don't care about how
# the dependency was called.
class StubPaymentGateway(PaymentGateway):
    def __init__(self, should_succeed=True):
        self.should_succeed = should_succeed

    def charge(self, amount, card_token):
        if self.should_succeed:
            return {"status": "success", "transaction_id": "txn_stub_123"}
        return {"status": "failed", "error": "Declined"}

def test_order_with_stub():
    gateway = StubPaymentGateway(should_succeed=True)
    result = gateway.charge(99.99, "tok_visa")
    assert result["status"] == "success"

# --- SPY: Records calls for later verification ---
# Use when you need to verify interactions (was it called? how many times?)
class SpyPaymentGateway(PaymentGateway):
    def __init__(self):
        self.calls = []

    def charge(self, amount, card_token):
        self.calls.append({"amount": amount, "card_token": card_token})
        return {"status": "success", "transaction_id": "txn_spy_123"}

def test_order_with_spy():
    gateway = SpyPaymentGateway()
    gateway.charge(50.00, "tok_visa")
    gateway.charge(25.00, "tok_mastercard")

    assert len(gateway.calls) == 2
    assert gateway.calls[0]["amount"] == 50.00
    assert gateway.calls[1]["card_token"] == "tok_mastercard"

# --- MOCK (unittest.mock): Combines stub + spy + expectations ---
# Use for most cases — it's the most flexible approach.
def test_order_with_mock():
    gateway = Mock(spec=PaymentGateway)
    gateway.charge.return_value = {"status": "success", "transaction_id": "txn_mock"}

    result = gateway.charge(75.00, "tok_amex")

    assert result["status"] == "success"
    gateway.charge.assert_called_once_with(75.00, "tok_amex")
SOLUTION
}

# === Exercise 2: Patch Targets ===
# Problem: Use @patch correctly to mock dependencies. The key rule:
# "Patch where it's LOOKED UP, not where it's DEFINED."
exercise_2() {
    echo "=== Exercise 2: Patch Targets ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from unittest.mock import patch, Mock

# --- Production code (order_service.py) ---
# from payment_gateway import PaymentGateway  # <-- imported here
#
# class OrderService:
#     def __init__(self):
#         self.gateway = PaymentGateway()
#
#     def place_order(self, amount):
#         return self.gateway.charge(amount, "default_token")

# WRONG: patching where the class is defined
# @patch("payment_gateway.PaymentGateway")  # <-- This patches the source module

# RIGHT: patching where the class is looked up
# @patch("order_service.PaymentGateway")    # <-- This patches the import

# Example demonstrating the principle:
import os

def get_config_path():
    """Uses os.path.expanduser — we must patch 'os.path.expanduser',
    not some other module's reference to it."""
    return os.path.expanduser("~/.config/myapp")

@patch("os.path.expanduser")
def test_config_path(mock_expand):
    mock_expand.return_value = "/home/testuser/.config/myapp"
    result = get_config_path()
    assert result == "/home/testuser/.config/myapp"
    mock_expand.assert_called_once_with("~/.config/myapp")

# Another example: patching built-in open
def read_config():
    with open("/etc/myapp.conf") as f:
        return f.read()

@patch("builtins.open", create=True)
def test_read_config(mock_open):
    mock_open.return_value.__enter__ = Mock(return_value=Mock(read=Mock(return_value="key=value")))
    mock_open.return_value.__exit__ = Mock(return_value=False)
    # In practice, use mock_open helper from unittest.mock
    from unittest.mock import mock_open as mo
    with patch("builtins.open", mo(read_data="key=value")):
        result = read_config()
        assert result == "key=value"
SOLUTION
}

# === Exercise 3: Side Effects and Exceptions ===
# Problem: Use side_effect to test retry logic and error handling
# in a service that calls an unreliable API.
exercise_3() {
    echo "=== Exercise 3: Side Effects and Exceptions ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import pytest
from unittest.mock import Mock, patch
import time

class UnreliableAPI:
    def fetch_data(self, endpoint: str) -> dict:
        raise NotImplementedError

class DataService:
    def __init__(self, api: UnreliableAPI, max_retries: int = 3):
        self.api = api
        self.max_retries = max_retries

    def get_data_with_retry(self, endpoint: str) -> dict:
        """Retry on failure with increasing delay."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return self.api.fetch_data(endpoint)
            except ConnectionError as e:
                last_error = e
                # In production: time.sleep(2 ** attempt)
        raise last_error

# Test: API succeeds on first try
def test_success_no_retry():
    mock_api = Mock(spec=UnreliableAPI)
    mock_api.fetch_data.return_value = {"data": "success"}

    service = DataService(mock_api)
    result = service.get_data_with_retry("/users")

    assert result == {"data": "success"}
    assert mock_api.fetch_data.call_count == 1

# Test: API fails twice, then succeeds (tests retry logic)
def test_retry_then_success():
    mock_api = Mock(spec=UnreliableAPI)
    mock_api.fetch_data.side_effect = [
        ConnectionError("Timeout"),        # Attempt 1: fail
        ConnectionError("Connection reset"),  # Attempt 2: fail
        {"data": "finally worked"},           # Attempt 3: success
    ]

    service = DataService(mock_api, max_retries=3)
    result = service.get_data_with_retry("/users")

    assert result == {"data": "finally worked"}
    assert mock_api.fetch_data.call_count == 3

# Test: All retries exhausted
def test_all_retries_fail():
    mock_api = Mock(spec=UnreliableAPI)
    mock_api.fetch_data.side_effect = ConnectionError("Persistent failure")

    service = DataService(mock_api, max_retries=3)

    with pytest.raises(ConnectionError, match="Persistent failure"):
        service.get_data_with_retry("/users")

    assert mock_api.fetch_data.call_count == 3

# Test: Different exceptions on each attempt
def test_different_errors():
    mock_api = Mock(spec=UnreliableAPI)
    mock_api.fetch_data.side_effect = [
        ConnectionError("DNS resolution failed"),
        ConnectionError("SSL handshake timeout"),
        ConnectionError("Connection refused"),
    ]

    service = DataService(mock_api, max_retries=3)

    with pytest.raises(ConnectionError, match="Connection refused"):
        service.get_data_with_retry("/users")
SOLUTION
}

# === Exercise 4: Spec and Autospec ===
# Problem: Demonstrate why spec= is important for catching interface
# mismatches, and show the difference between Mock, Mock(spec=), and create_autospec.
exercise_4() {
    echo "=== Exercise 4: Spec and Autospec ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import pytest
from unittest.mock import Mock, create_autospec

class EmailService:
    def send(self, to: str, subject: str, body: str) -> bool:
        """Send an email. Returns True on success."""
        raise NotImplementedError

# --- Without spec: DANGEROUS ---
def test_without_spec_allows_typos():
    """BAD: Mock without spec silently accepts any attribute."""
    mock_email = Mock()

    # This should fail but DOESN'T — 'sent' instead of 'send'
    mock_email.sent("user@test.com", "Hi", "Hello")  # typo: 'sent' not 'send'
    mock_email.sent.assert_called_once()  # "Passes" but tests nothing useful

# --- With spec: SAFE ---
def test_with_spec_catches_typos():
    """GOOD: Mock with spec only allows real attributes."""
    mock_email = Mock(spec=EmailService)

    # This correctly raises AttributeError — typo caught!
    with pytest.raises(AttributeError):
        mock_email.sent("user@test.com", "Hi", "Hello")

    # Correct method name works fine
    mock_email.send.return_value = True
    result = mock_email.send("user@test.com", "Hi", "Hello")
    assert result is True

# --- create_autospec: SAFEST ---
def test_autospec_checks_signatures():
    """BEST: autospec also validates argument count and types."""
    mock_email = create_autospec(EmailService)

    # This raises TypeError — wrong number of arguments!
    with pytest.raises(TypeError):
        mock_email.send("only_one_arg")  # Missing subject and body

    # Correct call works
    mock_email.send.return_value = True
    result = mock_email.send("user@test.com", "Subject", "Body")
    assert result is True

# RECOMMENDATION: Always use spec= or create_autospec.
# The small extra effort prevents false-positive tests that
# silently accept incorrect mock usage.
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 04: Mocking and Test Doubles"
echo "============================================================"
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
