#!/usr/bin/env python3
"""Example: Security Testing

Demonstrates SAST/DAST concepts, dependency scanning patterns,
input validation testing, and basic fuzzing techniques.
Related lesson: 12_Security_Testing.md
"""

# =============================================================================
# WHY SECURITY TESTING?
#
# Security bugs are functional bugs with adversarial consequences.
# The cost of finding a vulnerability in production is 10-100x the cost
# of catching it in tests. Security testing shifts left:
#
#   SAST  — Static Application Security Testing (scan source code)
#   DAST  — Dynamic Application Security Testing (probe running app)
#   SCA   — Software Composition Analysis (scan dependencies)
#   Fuzz  — Feed random/malformed inputs to find crashes
# =============================================================================

import pytest
import re
import html
import hashlib
import secrets
from urllib.parse import urlparse


# =============================================================================
# PRODUCTION CODE — COMMON VULNERABILITY PATTERNS
# =============================================================================

class InputValidator:
    """Validates and sanitizes user inputs against common attack vectors."""

    EMAIL_PATTERN = re.compile(
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )

    @staticmethod
    def sanitize_html(user_input: str) -> str:
        """Escape HTML to prevent XSS attacks.
        Never trust user input — always escape before rendering."""
        return html.escape(user_input, quote=True)

    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format. Rejects inputs that could enable injection."""
        if len(email) > 254:
            return False
        return bool(InputValidator.EMAIL_PATTERN.match(email))

    @staticmethod
    def validate_url(url: str, allowed_schemes: tuple = ("https",)) -> bool:
        """Validate URL against allowed schemes to prevent SSRF.
        Only HTTPS by default — no file://, javascript://, etc."""
        try:
            parsed = urlparse(url)
            return (
                parsed.scheme in allowed_schemes
                and bool(parsed.netloc)
                and ".." not in parsed.path
            )
        except Exception:
            return False

    @staticmethod
    def validate_path(path: str) -> bool:
        """Reject path traversal attempts."""
        dangerous_patterns = ["..", "~", "\x00"]
        return not any(p in path for p in dangerous_patterns)


class PasswordPolicy:
    """Enforce password strength requirements."""

    MIN_LENGTH = 12
    REQUIRED_PATTERNS = [
        (r'[A-Z]', "uppercase letter"),
        (r'[a-z]', "lowercase letter"),
        (r'[0-9]', "digit"),
        (r'[!@#$%^&*(),.?":{}|<>]', "special character"),
    ]

    COMMON_PASSWORDS = {"password123", "admin123", "letmein", "qwerty123"}

    @classmethod
    def validate(cls, password: str) -> list[str]:
        """Return list of policy violations. Empty list = valid password."""
        errors = []
        if len(password) < cls.MIN_LENGTH:
            errors.append(f"Must be at least {cls.MIN_LENGTH} characters")
        for pattern, desc in cls.REQUIRED_PATTERNS:
            if not re.search(pattern, password):
                errors.append(f"Must contain at least one {desc}")
        if password.lower() in cls.COMMON_PASSWORDS:
            errors.append("Password is too common")
        return errors

    @staticmethod
    def hash_password(password: str) -> tuple[str, str]:
        """Hash password with a random salt. Returns (hash, salt)."""
        salt = secrets.token_hex(16)
        hashed = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
        return hashed.hex(), salt


class SqlQueryBuilder:
    """Demonstrates safe vs unsafe query building."""

    @staticmethod
    def unsafe_query(username: str) -> str:
        """BAD: String interpolation — vulnerable to SQL injection.
        Never do this in production!"""
        return f"SELECT * FROM users WHERE username = '{username}'"

    @staticmethod
    def safe_query(username: str) -> tuple[str, tuple]:
        """GOOD: Parameterized query — immune to SQL injection.
        Always use parameter binding."""
        return ("SELECT * FROM users WHERE username = ?", (username,))


# =============================================================================
# TESTS — XSS PREVENTION
# =============================================================================

class TestXSSPrevention:
    """Verify HTML sanitization prevents cross-site scripting."""

    @pytest.mark.parametrize("malicious_input, expected_safe", [
        ('<script>alert("xss")</script>',
         '&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;'),
        ('<img src=x onerror=alert(1)>',
         '&lt;img src=x onerror=alert(1)&gt;'),
        ('"><svg onload=alert(1)>',
         '&quot;&gt;&lt;svg onload=alert(1)&gt;'),
        ("'; DROP TABLE users; --",
         "&#x27;; DROP TABLE users; --"),
    ])
    def test_html_sanitization(self, malicious_input, expected_safe):
        """Each XSS vector must be fully escaped."""
        result = InputValidator.sanitize_html(malicious_input)
        assert "<script" not in result
        assert "onerror" not in result or "&lt;" in result
        assert result == expected_safe

    def test_safe_input_unchanged(self):
        """Normal text should pass through with minimal changes."""
        safe = "Hello, this is a normal comment."
        assert InputValidator.sanitize_html(safe) == safe


# =============================================================================
# TESTS — SQL INJECTION PREVENTION
# =============================================================================

class TestSQLInjectionPrevention:
    """Verify parameterized queries prevent SQL injection."""

    SQL_INJECTION_PAYLOADS = [
        "' OR '1'='1",
        "'; DROP TABLE users; --",
        "' UNION SELECT password FROM admin --",
        "1; EXEC xp_cmdshell('whoami')",
    ]

    @pytest.mark.parametrize("payload", SQL_INJECTION_PAYLOADS)
    def test_unsafe_query_is_injectable(self, payload):
        """Show that string interpolation allows injection."""
        query = SqlQueryBuilder.unsafe_query(payload)
        # The payload becomes part of the SQL structure — dangerous!
        assert payload in query

    @pytest.mark.parametrize("payload", SQL_INJECTION_PAYLOADS)
    def test_safe_query_prevents_injection(self, payload):
        """Parameterized queries treat payload as data, not SQL."""
        query, params = SqlQueryBuilder.safe_query(payload)
        # The query template never contains the payload
        assert payload not in query
        # The payload is passed separately as a parameter
        assert params == (payload,)
        assert "?" in query


# =============================================================================
# TESTS — INPUT VALIDATION
# =============================================================================

class TestInputValidation:
    """Test input validation against common attack patterns."""

    @pytest.mark.parametrize("email, valid", [
        ("user@example.com", True),
        ("user+tag@example.co.uk", True),
        ("invalid", False),
        ("@example.com", False),
        ("user@", False),
        ("user@.com", False),
        ("a" * 255 + "@example.com", False),  # too long
    ])
    def test_email_validation(self, email, valid):
        assert InputValidator.validate_email(email) == valid

    @pytest.mark.parametrize("url, valid", [
        ("https://example.com", True),
        ("https://example.com/path", True),
        ("http://example.com", False),       # http not allowed by default
        ("file:///etc/passwd", False),        # SSRF vector
        ("javascript:alert(1)", False),       # XSS vector
        ("https://evil.com/../../../etc/passwd", False),  # traversal
    ])
    def test_url_validation(self, url, valid):
        assert InputValidator.validate_url(url) == valid

    @pytest.mark.parametrize("path, valid", [
        ("documents/report.pdf", True),
        ("../../../etc/passwd", False),       # path traversal
        ("files/../../secret", False),
        ("file\x00.txt", False),              # null byte injection
        ("~/secret", False),
    ])
    def test_path_validation(self, path, valid):
        assert InputValidator.validate_path(path) == valid


# =============================================================================
# TESTS — PASSWORD POLICY
# =============================================================================

class TestPasswordPolicy:
    """Verify password policy enforcement."""

    def test_strong_password_passes(self):
        errors = PasswordPolicy.validate("C0mpl3x!Pass#2024")
        assert errors == []

    def test_short_password_rejected(self):
        errors = PasswordPolicy.validate("Sh0rt!")
        assert any("at least" in e for e in errors)

    def test_missing_uppercase(self):
        errors = PasswordPolicy.validate("alllowercase123!")
        assert any("uppercase" in e for e in errors)

    def test_common_password_rejected(self):
        errors = PasswordPolicy.validate("password123")
        assert any("common" in e for e in errors)

    def test_password_hashing_unique_salts(self):
        """Each hash must use a unique salt — same password, different hashes."""
        hash1, salt1 = PasswordPolicy.hash_password("MyP@ssw0rd!!")
        hash2, salt2 = PasswordPolicy.hash_password("MyP@ssw0rd!!")
        assert salt1 != salt2
        assert hash1 != hash2


# =============================================================================
# BASIC FUZZING PATTERN
# =============================================================================

class TestFuzzingPatterns:
    """Demonstrate simple fuzz-style testing — feed unexpected inputs."""

    FUZZ_STRINGS = [
        "",                         # empty
        " " * 1000,                 # whitespace flood
        "A" * 10_000,               # long string
        "\x00\x01\x02\xff",        # binary data
        "🎉" * 100,                # unicode
        "<>&\"'\\",                # special chars
        "\n\r\t",                  # control characters
        "null",                    # literal null string
        "-1",                      # negative number string
        "0" * 50,                  # numeric overflow string
    ]

    @pytest.mark.parametrize("fuzz_input", FUZZ_STRINGS)
    def test_sanitize_does_not_crash(self, fuzz_input):
        """The sanitizer must handle ANY input without raising."""
        result = InputValidator.sanitize_html(fuzz_input)
        assert isinstance(result, str)

    @pytest.mark.parametrize("fuzz_input", FUZZ_STRINGS)
    def test_email_validation_does_not_crash(self, fuzz_input):
        """Validation must return True/False, never crash."""
        result = InputValidator.validate_email(fuzz_input)
        assert isinstance(result, bool)


# =============================================================================
# RUNNING THIS FILE
# =============================================================================
# Basic run:
#   pytest 12_security_testing.py -v
#
# SAST tools (run on source code):
#   pip install bandit && bandit -r src/
#   pip install semgrep && semgrep --config auto src/
#
# Dependency scanning:
#   pip install safety && safety check
#   pip install pip-audit && pip-audit

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
