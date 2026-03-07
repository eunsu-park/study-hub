#!/usr/bin/env python3
"""Example: Mocking and Patching

Demonstrates unittest.mock: Mock, MagicMock, patch, side_effect, spec,
and best practices for isolating units under test.
Related lesson: 04_Mocking_and_Test_Doubles.md
"""

# =============================================================================
# WHY MOCK?
# Tests should verify YOUR code, not third-party services. Mocking replaces
# real dependencies (APIs, databases, file systems) with controlled substitutes
# so tests are fast, deterministic, and isolated.
#
# Key principle: Mock at the boundary. Don't mock internal implementation
# details — mock the external interface your code depends on.
# =============================================================================

import pytest
from unittest.mock import Mock, MagicMock, patch, call, PropertyMock
from dataclasses import dataclass
from typing import Optional
import json


# =============================================================================
# PRODUCTION CODE (what we are testing)
# =============================================================================

class WeatherAPI:
    """External weather service — we don't want real HTTP calls in tests."""

    def get_temperature(self, city: str) -> float:
        # In production, this would make an HTTP request
        raise NotImplementedError("Real API call")

    def get_forecast(self, city: str, days: int) -> list:
        raise NotImplementedError("Real API call")


class NotificationService:
    """External email/SMS service."""

    def send_email(self, to: str, subject: str, body: str) -> bool:
        raise NotImplementedError("Real email send")

    def send_sms(self, to: str, message: str) -> bool:
        raise NotImplementedError("Real SMS send")


@dataclass
class WeatherAlert:
    """Business logic that depends on external services.
    This is the class we want to test."""
    weather_api: WeatherAPI
    notification: NotificationService
    threshold: float = 35.0

    def check_and_alert(self, city: str, email: str) -> Optional[str]:
        """Check temperature and send alert if above threshold."""
        temp = self.weather_api.get_temperature(city)

        if temp > self.threshold:
            message = f"Heat alert for {city}: {temp}C"
            self.notification.send_email(
                to=email,
                subject=f"Weather Alert: {city}",
                body=message,
            )
            return message
        return None


# =============================================================================
# 1. BASIC MOCK USAGE
# =============================================================================

def test_mock_method_return():
    """Mock replaces real methods with controlled return values.
    This lets us test WeatherAlert WITHOUT making real API calls."""
    # Create mocks for the dependencies
    mock_api = Mock(spec=WeatherAPI)
    mock_notif = Mock(spec=NotificationService)

    # Configure the mock to return a specific value
    mock_api.get_temperature.return_value = 40.0
    mock_notif.send_email.return_value = True

    # Now test the business logic
    alert = WeatherAlert(weather_api=mock_api, notification=mock_notif)
    result = alert.check_and_alert("Seoul", "user@example.com")

    # Verify the result
    assert result == "Heat alert for Seoul: 40.0C"

    # Verify the mock was called correctly
    mock_api.get_temperature.assert_called_once_with("Seoul")
    mock_notif.send_email.assert_called_once()


def test_no_alert_below_threshold():
    """When temperature is below threshold, no notification should be sent.
    Mock lets us verify that send_email was NOT called."""
    mock_api = Mock(spec=WeatherAPI)
    mock_notif = Mock(spec=NotificationService)

    mock_api.get_temperature.return_value = 20.0

    alert = WeatherAlert(weather_api=mock_api, notification=mock_notif)
    result = alert.check_and_alert("Seoul", "user@example.com")

    assert result is None
    # Crucially: verify the email was NOT sent
    mock_notif.send_email.assert_not_called()


# =============================================================================
# 2. SPEC — TYPE-SAFE MOCKS
# =============================================================================
# Always use spec= to create mocks that mirror the real class interface.
# Without spec, typos in method names silently succeed (a dangerous false pass).

def test_spec_prevents_typos():
    """spec=WeatherAPI means only real methods of WeatherAPI are allowed."""
    mock_api = Mock(spec=WeatherAPI)

    # This works — get_temperature is a real method
    mock_api.get_temperature.return_value = 25.0
    assert mock_api.get_temperature("Seoul") == 25.0

    # This would raise AttributeError — typo caught!
    with pytest.raises(AttributeError):
        mock_api.get_temprature("Seoul")  # Typo: "temprature"


# =============================================================================
# 3. SIDE_EFFECT — DYNAMIC RESPONSES
# =============================================================================

def test_side_effect_function():
    """side_effect lets the mock compute its return value dynamically.
    Use it when the return depends on the input arguments."""
    mock_api = Mock(spec=WeatherAPI)

    # Different cities return different temperatures
    city_temps = {"Seoul": 38.0, "London": 15.0, "Dubai": 45.0}
    mock_api.get_temperature.side_effect = lambda city: city_temps[city]

    assert mock_api.get_temperature("Seoul") == 38.0
    assert mock_api.get_temperature("London") == 15.0
    assert mock_api.get_temperature("Dubai") == 45.0


def test_side_effect_exception():
    """side_effect can raise exceptions to test error handling paths.
    This is crucial — you MUST test what happens when dependencies fail."""
    mock_api = Mock(spec=WeatherAPI)
    mock_api.get_temperature.side_effect = ConnectionError("API is down")

    with pytest.raises(ConnectionError, match="API is down"):
        mock_api.get_temperature("Seoul")


def test_side_effect_sequence():
    """side_effect with a list returns values in order, then raises StopIteration.
    Useful for testing retry logic or pagination."""
    mock_api = Mock(spec=WeatherAPI)

    # First call fails, second succeeds (simulating retry)
    mock_api.get_temperature.side_effect = [
        ConnectionError("timeout"),
        25.0,
    ]

    with pytest.raises(ConnectionError):
        mock_api.get_temperature("Seoul")  # First call: raises

    result = mock_api.get_temperature("Seoul")  # Second call: returns 25.0
    assert result == 25.0


# =============================================================================
# 4. CALL ASSERTIONS
# =============================================================================

def test_call_count_and_args():
    """Verify not just that a mock was called, but how many times
    and with what arguments."""
    mock_notif = Mock(spec=NotificationService)
    mock_notif.send_email.return_value = True

    # Simulate sending multiple emails
    mock_notif.send_email(to="a@test.com", subject="Alert 1", body="Hot")
    mock_notif.send_email(to="b@test.com", subject="Alert 2", body="Cold")

    assert mock_notif.send_email.call_count == 2

    # Check all calls were made with expected arguments
    mock_notif.send_email.assert_any_call(
        to="a@test.com", subject="Alert 1", body="Hot"
    )

    # Check the exact sequence of calls
    expected_calls = [
        call(to="a@test.com", subject="Alert 1", body="Hot"),
        call(to="b@test.com", subject="Alert 2", body="Cold"),
    ]
    mock_notif.send_email.assert_has_calls(expected_calls, any_order=False)


# =============================================================================
# 5. PATCH DECORATOR AND CONTEXT MANAGER
# =============================================================================
# patch replaces objects in the namespace WHERE THEY ARE LOOKED UP,
# not where they are defined. This is the most common mock mistake.

def fetch_user_data(user_id: int) -> dict:
    """Production function that calls an external API."""
    import urllib.request
    # In real code: response = urllib.request.urlopen(f"https://api/users/{user_id}")
    raise NotImplementedError("Real HTTP call")


# Patch as a decorator — the mock is passed as an extra argument
@patch("urllib.request.urlopen")
def test_fetch_user_with_decorator(mock_urlopen):
    """patch replaces urlopen in the urllib.request module.
    The mock is injected as the LAST parameter of the test function."""
    # Configure the mock response
    mock_response = Mock()
    mock_response.read.return_value = json.dumps(
        {"id": 1, "name": "Alice"}
    ).encode()
    mock_response.__enter__ = Mock(return_value=mock_response)
    mock_response.__exit__ = Mock(return_value=False)
    mock_urlopen.return_value = mock_response

    # Now urllib.request.urlopen is mocked
    import urllib.request
    with urllib.request.urlopen("https://api/users/1") as resp:
        data = json.loads(resp.read())

    assert data["name"] == "Alice"
    mock_urlopen.assert_called_once_with("https://api/users/1")


def test_patch_context_manager():
    """patch as a context manager — useful when you need the mock
    only for part of the test."""
    with patch("os.path.exists") as mock_exists:
        mock_exists.return_value = True
        import os.path
        assert os.path.exists("/fake/path") is True

    # Outside the context, os.path.exists is restored to the real function


# =============================================================================
# 6. PATCHING OBJECT ATTRIBUTES
# =============================================================================

class Config:
    DEBUG = False
    DATABASE_URL = "postgres://prod:5432/mydb"

    @property
    def is_production(self):
        return not self.DEBUG


def test_patch_object():
    """patch.object targets a specific attribute of a specific object.
    More precise than patch() — less chance of patching the wrong thing."""
    config = Config()

    with patch.object(config, "DATABASE_URL", "sqlite:///:memory:"):
        assert config.DATABASE_URL == "sqlite:///:memory:"

    # After the context, original value is restored
    assert config.DATABASE_URL == "postgres://prod:5432/mydb"


def test_patch_property():
    """Properties require PropertyMock to patch correctly."""
    with patch.object(
        Config, "is_production", new_callable=PropertyMock, return_value=False
    ):
        config = Config()
        assert config.is_production is False


# =============================================================================
# 7. MAGICMOCK — MAGIC METHOD SUPPORT
# =============================================================================

def test_magic_mock():
    """MagicMock pre-configures magic methods (__len__, __iter__, etc.).
    Regular Mock does not support magic methods out of the box."""
    mock_collection = MagicMock()

    # Configure magic methods
    mock_collection.__len__.return_value = 5
    mock_collection.__getitem__.return_value = "item"
    mock_collection.__contains__.return_value = True

    assert len(mock_collection) == 5
    assert mock_collection[0] == "item"
    assert "anything" in mock_collection


def test_magic_mock_context_manager():
    """MagicMock supports context manager protocol automatically."""
    mock_file = MagicMock()
    mock_file.__enter__.return_value = mock_file
    mock_file.read.return_value = "file contents"

    with mock_file as f:
        content = f.read()

    assert content == "file contents"
    mock_file.__exit__.assert_called_once()


# =============================================================================
# 8. BEST PRACTICES
# =============================================================================

def test_dont_mock_what_you_dont_own():
    """GOOD: Mock your own adapter/wrapper, not the third-party library directly.
    This way, if the library API changes, only your adapter needs updating."""

    # Instead of mocking requests.get directly, mock your own wrapper:
    class HttpClient:
        def get(self, url: str) -> dict:
            raise NotImplementedError

    mock_client = Mock(spec=HttpClient)
    mock_client.get.return_value = {"status": "ok"}

    result = mock_client.get("https://api.example.com")
    assert result["status"] == "ok"


def test_verify_behavior_not_implementation():
    """GOOD: Assert on outcomes and interactions, not internal details.
    Tests that mirror implementation become brittle and break on refactoring."""
    mock_api = Mock(spec=WeatherAPI)
    mock_notif = Mock(spec=NotificationService)
    mock_api.get_temperature.return_value = 40.0
    mock_notif.send_email.return_value = True

    alert = WeatherAlert(weather_api=mock_api, notification=mock_notif)
    result = alert.check_and_alert("Seoul", "user@test.com")

    # GOOD: verify the observable behavior
    assert result is not None
    assert "Seoul" in result

    # GOOD: verify critical interaction (email was sent)
    mock_notif.send_email.assert_called_once()

    # BAD (not shown): asserting on internal variable values or call order
    # of non-critical methods


# =============================================================================
# RUNNING THIS FILE
# =============================================================================
# pytest 03_mocking_examples.py -v

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
