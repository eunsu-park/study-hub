#!/usr/bin/env python3
"""Example: Test Architecture

Demonstrates the test pyramid, page object pattern, test doubles
taxonomy (dummy, stub, spy, mock, fake), and layered test organization.
Related lesson: 14_Test_Architecture.md
"""

# =============================================================================
# TEST PYRAMID
#
#        /  E2E  \        Few, slow, expensive — verify full workflows
#       /  Integ. \       Medium count — verify component interactions
#      /   Unit    \      Many, fast, cheap — verify individual functions
#     /______________\
#
# Invert this (ice cream cone) and your CI takes forever and tests are brittle.
# The pyramid is not dogma but a cost/speed/confidence trade-off guide.
# =============================================================================

import pytest
from dataclasses import dataclass, field
from typing import Protocol, Optional
from unittest.mock import MagicMock


# =============================================================================
# TEST DOUBLES TAXONOMY
# =============================================================================
# The 5 types of test doubles (Gerard Meszaros terminology):
#
#   Dummy  — Passed but never used. Satisfies a parameter requirement.
#   Stub   — Returns canned answers. No logic, no recording.
#   Spy    — Records calls for later verification. May also return values.
#   Mock   — Pre-programmed with expectations. Fails if expectations unmet.
#   Fake   — Working implementation with shortcuts (in-memory DB, etc.)

# --- Contracts (Protocols) ---

class EmailService(Protocol):
    """Protocol for sending emails — production code depends on this."""
    def send(self, to: str, subject: str, body: str) -> bool: ...


class UserRepository(Protocol):
    """Protocol for user persistence."""
    def find_by_id(self, user_id: int) -> Optional[dict]: ...
    def save(self, user: dict) -> None: ...


# --- Production Code ---

@dataclass
class UserService:
    """Business logic that depends on external services via protocols."""
    repo: UserRepository
    email: EmailService

    def register(self, user_id: int, name: str, email_addr: str) -> dict:
        existing = self.repo.find_by_id(user_id)
        if existing:
            raise ValueError(f"User {user_id} already exists")

        user = {"id": user_id, "name": name, "email": email_addr, "active": True}
        self.repo.save(user)
        self.email.send(
            to=email_addr,
            subject="Welcome!",
            body=f"Hello {name}, your account is ready.",
        )
        return user

    def deactivate(self, user_id: int) -> dict:
        user = self.repo.find_by_id(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")
        user["active"] = False
        self.repo.save(user)
        return user


# --- Test Doubles ---

class DummyEmailService:
    """DUMMY — satisfies the interface but is never expected to be called."""
    def send(self, to: str, subject: str, body: str) -> bool:
        raise RuntimeError("Dummy should not be called")


class StubUserRepository:
    """STUB — returns pre-configured data, no recording."""
    def __init__(self, users: dict = None):
        self._users = users or {}

    def find_by_id(self, user_id: int) -> Optional[dict]:
        return self._users.get(user_id)

    def save(self, user: dict) -> None:
        pass  # Intentionally does nothing


class SpyEmailService:
    """SPY — records all calls for later assertion."""
    def __init__(self):
        self.calls: list[dict] = []

    def send(self, to: str, subject: str, body: str) -> bool:
        self.calls.append({"to": to, "subject": subject, "body": body})
        return True


class FakeUserRepository:
    """FAKE — working in-memory implementation, no real database."""
    def __init__(self):
        self._store: dict[int, dict] = {}

    def find_by_id(self, user_id: int) -> Optional[dict]:
        return self._store.get(user_id)

    def save(self, user: dict) -> None:
        self._store[user["id"]] = user


# =============================================================================
# TESTS — TEST DOUBLES IN ACTION
# =============================================================================

class TestWithStub:
    """STUB example: control what the dependency returns."""

    def test_deactivate_existing_user(self):
        stub_repo = StubUserRepository(users={
            1: {"id": 1, "name": "Alice", "email": "a@b.com", "active": True}
        })
        # Deactivate doesn't send email, so use a Dummy
        service = UserService(repo=stub_repo, email=DummyEmailService())

        # The stub provides canned data; we don't verify save was called
        # (that would be a mock/spy concern)
        user = service.deactivate(1)
        assert user["active"] is False

    def test_deactivate_missing_user_raises(self):
        stub_repo = StubUserRepository()  # empty — find_by_id returns None
        service = UserService(repo=stub_repo, email=DummyEmailService())

        with pytest.raises(ValueError, match="not found"):
            service.deactivate(999)


class TestWithSpy:
    """SPY example: verify interactions after the fact."""

    def test_register_sends_welcome_email(self):
        fake_repo = FakeUserRepository()
        spy_email = SpyEmailService()
        service = UserService(repo=fake_repo, email=spy_email)

        service.register(1, "Alice", "alice@example.com")

        # Verify the spy recorded the expected call
        assert len(spy_email.calls) == 1
        assert spy_email.calls[0]["to"] == "alice@example.com"
        assert "Welcome" in spy_email.calls[0]["subject"]


class TestWithMock:
    """MOCK example: using unittest.mock for pre-programmed expectations."""

    def test_register_calls_save(self):
        mock_repo = MagicMock(spec=FakeUserRepository)
        mock_repo.find_by_id.return_value = None  # no existing user

        mock_email = MagicMock()
        mock_email.send.return_value = True

        service = UserService(repo=mock_repo, email=mock_email)
        service.register(1, "Bob", "bob@example.com")

        # Mock verifies that save was called with the right arguments
        mock_repo.save.assert_called_once()
        saved_user = mock_repo.save.call_args[0][0]
        assert saved_user["name"] == "Bob"


class TestWithFake:
    """FAKE example: in-memory implementation for integration-style tests."""

    def test_register_then_deactivate(self):
        """Full workflow using a Fake — closer to integration test."""
        fake_repo = FakeUserRepository()
        spy_email = SpyEmailService()
        service = UserService(repo=fake_repo, email=spy_email)

        # Register
        user = service.register(1, "Carol", "carol@example.com")
        assert user["active"] is True

        # Verify persistence (Fake actually stores data)
        stored = fake_repo.find_by_id(1)
        assert stored is not None

        # Deactivate
        deactivated = service.deactivate(1)
        assert deactivated["active"] is False

        # Verify persistence updated
        assert fake_repo.find_by_id(1)["active"] is False

    def test_register_duplicate_raises(self):
        fake_repo = FakeUserRepository()
        spy_email = SpyEmailService()
        service = UserService(repo=fake_repo, email=spy_email)

        service.register(1, "Dave", "dave@example.com")
        with pytest.raises(ValueError, match="already exists"):
            service.register(1, "Dave Again", "dave2@example.com")


# =============================================================================
# PAGE OBJECT PATTERN (WEB UI TESTING ABSTRACTION)
# =============================================================================

class LoginPage:
    """Page Object — encapsulates UI interaction details.

    In real code, this wraps Selenium/Playwright selectors.
    Tests read like user stories instead of DOM manipulation."""

    def __init__(self, driver=None):
        self.driver = driver or {}
        self._submitted = False
        self._username = ""
        self._password = ""

    def enter_username(self, username: str) -> "LoginPage":
        self._username = username
        return self  # fluent API

    def enter_password(self, password: str) -> "LoginPage":
        self._password = password
        return self

    def submit(self) -> "DashboardPage":
        self._submitted = True
        if self._username == "admin" and self._password == "secret":
            return DashboardPage(authenticated=True, username=self._username)
        return DashboardPage(authenticated=False, username="")

    @property
    def is_displayed(self) -> bool:
        return not self._submitted


class DashboardPage:
    """Page Object for the dashboard — returned after login."""

    def __init__(self, authenticated: bool, username: str):
        self.authenticated = authenticated
        self.username = username

    @property
    def welcome_message(self) -> str:
        if self.authenticated:
            return f"Welcome, {self.username}"
        return "Login failed"


class TestPageObjectPattern:
    """Tests using Page Objects read like user stories."""

    def test_successful_login(self):
        page = LoginPage()
        dashboard = (
            page.enter_username("admin")
                .enter_password("secret")
                .submit()
        )
        assert dashboard.authenticated
        assert "Welcome" in dashboard.welcome_message

    def test_failed_login(self):
        page = LoginPage()
        dashboard = (
            page.enter_username("wrong")
                .enter_password("wrong")
                .submit()
        )
        assert not dashboard.authenticated
        assert "failed" in dashboard.welcome_message


# =============================================================================
# RUNNING THIS FILE
# =============================================================================
# pytest 14_test_architecture.py -v

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
