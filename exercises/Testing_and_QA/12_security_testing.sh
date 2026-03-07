#!/bin/bash
# Exercises for Lesson 12: Security Testing
# Topic: Testing_and_QA
# Solutions to practice problems from the lesson.

# === Exercise 1: Bandit SAST Integration ===
# Problem: Write a pytest test that runs Bandit against a project
# and fails if any high-severity issues are found. Include parsing
# of the JSON report to produce actionable error messages.
exercise_1() {
    echo "=== Exercise 1: Bandit SAST Integration ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import json
import subprocess
import sys

import pytest


def test_bandit_finds_no_high_severity_issues():
    """Run Bandit SAST scan and fail on medium+ severity findings."""
    result = subprocess.run(
        [
            sys.executable, "-m", "bandit",
            "-r", "myapp/",
            "-ll",            # Medium+ severity
            "-ii",            # Medium+ confidence
            "-f", "json",
            "--exit-zero",    # Always exit 0 so we can parse output
        ],
        capture_output=True, text=True
    )

    report = json.loads(result.stdout)
    issues = report.get("results", [])

    if issues:
        messages = []
        for issue in issues:
            messages.append(
                f"  [{issue['issue_severity']}/{issue['issue_confidence']}] "
                f"{issue['filename']}:{issue['line_number']} "
                f"({issue['test_id']}) {issue['issue_text']}"
            )
        pytest.fail(
            f"Bandit found {len(issues)} security issue(s):\n"
            + "\n".join(messages)
        )


def test_bandit_no_sql_injection():
    """Specifically check for SQL injection patterns (B608)."""
    result = subprocess.run(
        [
            sys.executable, "-m", "bandit",
            "-r", "myapp/",
            "-t", "B608",     # Only SQL injection test
            "-f", "json",
            "--exit-zero",
        ],
        capture_output=True, text=True
    )

    report = json.loads(result.stdout)
    issues = report.get("results", [])

    if issues:
        locations = [
            f"  {i['filename']}:{i['line_number']}: {i['issue_text']}"
            for i in issues
        ]
        pytest.fail(
            "SQL injection risk detected:\n" + "\n".join(locations)
            + "\n\nFix: Use parameterized queries instead of string formatting."
        )
SOLUTION
}

# === Exercise 2: Dependency Vulnerability Audit ===
# Problem: Run pip-audit programmatically and generate a structured
# report of vulnerabilities with recommended fix versions.
exercise_2() {
    echo "=== Exercise 2: Dependency Vulnerability Audit ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import json
import subprocess
from dataclasses import dataclass


@dataclass
class Vulnerability:
    package: str
    version: str
    vuln_id: str
    fix_versions: str
    description: str


def run_dependency_audit(requirements_file: str = None) -> list[Vulnerability]:
    """Run pip-audit and return structured vulnerability data."""
    cmd = ["pip-audit", "-f", "json", "--progress-spinner=off"]
    if requirements_file:
        cmd.extend(["-r", requirements_file])

    result = subprocess.run(cmd, capture_output=True, text=True)
    vulnerabilities = []

    if result.returncode != 0:
        try:
            data = json.loads(result.stdout)
            for dep in data.get("dependencies", []):
                for vuln in dep.get("vulns", []):
                    vulnerabilities.append(Vulnerability(
                        package=dep["name"],
                        version=dep["version"],
                        vuln_id=vuln["id"],
                        fix_versions=", ".join(vuln.get("fix_versions", [])),
                        description=vuln.get("description", "N/A"),
                    ))
        except (json.JSONDecodeError, KeyError):
            pass

    return vulnerabilities


def test_no_vulnerable_dependencies():
    """Ensure no installed packages have known CVEs."""
    vulns = run_dependency_audit()

    if vulns:
        report_lines = []
        for v in vulns:
            report_lines.append(
                f"  {v.package}=={v.version}: {v.vuln_id} "
                f"(fix: {v.fix_versions or 'no fix available'})"
            )
        raise AssertionError(
            f"Found {len(vulns)} vulnerable dependencies:\n"
            + "\n".join(report_lines)
            + "\n\nRun 'pip-audit --fix' to auto-upgrade."
        )


def test_no_critical_vulnerabilities():
    """Allow low-severity vulns but block critical ones."""
    vulns = run_dependency_audit()
    # Filter for critical advisory IDs (CVE or PYSEC)
    critical = [v for v in vulns if v.vuln_id.startswith("CVE")]

    if critical:
        lines = [f"  {v.package}=={v.version}: {v.vuln_id}" for v in critical]
        raise AssertionError(
            f"Critical vulnerabilities found:\n" + "\n".join(lines)
        )
SOLUTION
}

# === Exercise 3: Secrets Detection Setup ===
# Problem: Set up detect-secrets with a baseline, demonstrate
# detection of various secret patterns, and configure a pre-commit hook.
exercise_3() {
    echo "=== Exercise 3: Secrets Detection Setup ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import json
import subprocess
import tempfile
from pathlib import Path


def test_detect_secrets_catches_api_key():
    """Verify detect-secrets catches a planted API key."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a file with a fake secret
        test_file = Path(tmpdir) / "config.py"
        test_file.write_text(
            'API_KEY = "sk_live_EXAMPLE_KEY_DO_NOT_USE"\n'
            'DATABASE_URL = "postgresql://admin:s3cret@db.example.com/app"\n'
            'DEBUG = True\n'
        )

        # Run detect-secrets scan
        result = subprocess.run(
            ["detect-secrets", "scan", str(test_file)],
            capture_output=True, text=True
        )
        report = json.loads(result.stdout)
        results = report.get("results", {})

        # Verify at least one secret was detected
        detected_files = list(results.keys())
        assert len(detected_files) > 0, "detect-secrets should find the planted secret"

        # Count total secrets found
        total_secrets = sum(len(v) for v in results.values())
        assert total_secrets >= 1, f"Expected at least 1 secret, found {total_secrets}"


def test_no_secrets_in_codebase():
    """Scan the project for hardcoded secrets."""
    result = subprocess.run(
        [
            "detect-secrets", "scan",
            "--baseline", ".secrets.baseline",
        ],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        report = json.loads(result.stdout)
        new_secrets = report.get("results", {})
        if new_secrets:
            findings = []
            for filename, secrets in new_secrets.items():
                for s in secrets:
                    findings.append(
                        f"  {filename}:{s['line_number']} — {s['type']}"
                    )
            raise AssertionError(
                "New secrets detected (not in baseline):\n"
                + "\n".join(findings)
            )


# Pre-commit configuration (.pre-commit-config.yaml):
#
# repos:
#   - repo: https://github.com/Yelp/detect-secrets
#     rev: v1.4.0
#     hooks:
#       - id: detect-secrets
#         args: ['--baseline', '.secrets.baseline']
#
# Setup steps:
# 1. pip install detect-secrets pre-commit
# 2. detect-secrets scan > .secrets.baseline
# 3. detect-secrets audit .secrets.baseline
# 4. pre-commit install
SOLUTION
}

# === Exercise 4: SQL Injection Vulnerability Test ===
# Problem: Demonstrate a SQL injection vulnerability, write a test
# that proves it exists, then fix the code and verify the fix.
exercise_4() {
    echo "=== Exercise 4: SQL Injection Vulnerability Test ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import sqlite3

import pytest


def get_user_vulnerable(db, username: str) -> dict | None:
    """VULNERABLE: Uses string formatting for SQL query."""
    cursor = db.execute(
        f"SELECT id, username, email FROM users WHERE username = '{username}'"
    )
    row = cursor.fetchone()
    if row:
        return {"id": row[0], "username": row[1], "email": row[2]}
    return None


def get_user_safe(db, username: str) -> dict | None:
    """SAFE: Uses parameterized query."""
    cursor = db.execute(
        "SELECT id, username, email FROM users WHERE username = ?",
        (username,)
    )
    row = cursor.fetchone()
    if row:
        return {"id": row[0], "username": row[1], "email": row[2]}
    return None


@pytest.fixture
def db():
    """Create an in-memory SQLite database with test data."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT, email TEXT)"
    )
    conn.execute(
        "INSERT INTO users (username, email) VALUES ('alice', 'alice@example.com')"
    )
    conn.execute(
        "INSERT INTO users (username, email) VALUES ('bob', 'bob@example.com')"
    )
    conn.commit()
    yield conn
    conn.close()


def test_vulnerable_query_allows_injection(db):
    """Demonstrate that the vulnerable function leaks data via injection."""
    # Normal usage works
    user = get_user_vulnerable(db, "alice")
    assert user["username"] == "alice"

    # SQL injection: ' OR '1'='1 returns the first user regardless
    injected = get_user_vulnerable(db, "' OR '1'='1")
    assert injected is not None, "Injection should return a row"


def test_safe_query_blocks_injection(db):
    """Verify parameterized query prevents SQL injection."""
    # Normal usage works
    user = get_user_safe(db, "alice")
    assert user["username"] == "alice"

    # Same injection attempt returns nothing
    injected = get_user_safe(db, "' OR '1'='1")
    assert injected is None, "Parameterized query should block injection"


def test_safe_query_handles_special_characters(db):
    """Parameterized queries safely handle special characters."""
    result = get_user_safe(db, "O'Brien")
    assert result is None  # No user with that name, but no crash

    result = get_user_safe(db, "'; DROP TABLE users; --")
    assert result is None  # Injection attempt fails silently
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 12: Security Testing"
echo "======================================================"
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
