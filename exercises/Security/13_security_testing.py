"""
Exercise Solutions: Security Testing
=====================================
Lesson 13 from Security topic.

Covers SAST analysis (Bandit), custom Semgrep rules, dependency auditing,
property-based testing, CI/CD pipeline design, and security code review.
"""

import hashlib
import os
import re
import secrets
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Exercise 1: Bandit Scan Analysis (Fixed Code)
# ---------------------------------------------------------------------------

def exercise_1_bandit_fixed():
    """
    Provide the fixed version of the intentionally vulnerable code.
    Each fix addresses a specific Bandit finding.
    """
    print("Bandit Findings and Fixes:")
    print("=" * 60)

    findings = [
        {
            "bandit_id": "B105",
            "code": 'SECRET_KEY = "my-super-secret-key-12345"',
            "severity": "LOW",
            "confidence": "MEDIUM",
            "fix": "SECRET_KEY = os.environ['SECRET_KEY']",
            "reason": "Hardcoded password string detected",
        },
        {
            "bandit_id": "B608",
            "code": 'f"SELECT * FROM products WHERE name LIKE \'%{query}%\'"',
            "severity": "MEDIUM",
            "confidence": "MEDIUM",
            "fix": "cursor.execute('SELECT * FROM products WHERE name LIKE ?', (f'%{query}%',))",
            "reason": "SQL injection via string formatting",
        },
        {
            "bandit_id": "B602",
            "code": "subprocess.check_output(cmd, shell=True)",
            "severity": "HIGH",
            "confidence": "HIGH",
            "fix": "REMOVED ENTIRELY — executing user commands is a backdoor",
            "reason": "Shell injection via subprocess with shell=True",
        },
        {
            "bandit_id": "B301",
            "code": "pickle.loads(data)",
            "severity": "MEDIUM",
            "confidence": "HIGH",
            "fix": "json.loads(data) — use JSON for untrusted data",
            "reason": "Deserialization of untrusted data (RCE vector)",
        },
        {
            "bandit_id": "B303",
            "code": "hashlib.md5(password.encode()).hexdigest()",
            "severity": "MEDIUM",
            "confidence": "HIGH",
            "fix": "Use bcrypt or argon2 for password hashing",
            "reason": "MD5 is cryptographically broken",
        },
        {
            "bandit_id": "B201",
            "code": "app.run(host='0.0.0.0', debug=True)",
            "severity": "HIGH",
            "confidence": "MEDIUM",
            "fix": "app.run(host='127.0.0.1', debug=False) in production",
            "reason": "Debug mode and binding to all interfaces in production",
        },
    ]

    for f in findings:
        print(f"\n  [{f['bandit_id']}] {f['severity']}/{f['confidence']}")
        print(f"  Code:   {f['code']}")
        print(f"  Reason: {f['reason']}")
        print(f"  Fix:    {f['fix']}")

    # Show complete fixed version
    fixed_code = '''
# Fixed version — all Bandit findings resolved

import os
import json
import hashlib
import sqlite3
from flask import Flask, request, jsonify

app = Flask(__name__)

# B105 Fix: Read secret from environment
SECRET_KEY = os.environ.get("SECRET_KEY")

@app.route("/search")
def search():
    query = request.args.get("q", "")
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()
    # B608 Fix: Parameterized query
    cursor.execute(
        "SELECT * FROM products WHERE name LIKE ?",
        (f"%{query}%",),
    )
    return jsonify(cursor.fetchall())

# B602 Fix: /run endpoint REMOVED (was a command injection backdoor)

@app.route("/load", methods=["POST"])
def load_data():
    data = request.get_data()
    # B301 Fix: Use JSON instead of pickle
    obj = json.loads(data)
    return jsonify(obj)

@app.route("/hash", methods=["POST"])
def hash_password():
    password = request.json.get("password", "")
    # B303 Fix: Use proper password hashing
    # In production: argon2.PasswordHasher().hash(password)
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 600_000)
    return jsonify({"hash": f"{salt.hex()}:{dk.hex()}"})

if __name__ == "__main__":
    # B201 Fix: Secure defaults
    app.run(host="127.0.0.1", debug=False)
'''
    print("\n--- Complete Fixed Code ---")
    print(fixed_code)


# ---------------------------------------------------------------------------
# Exercise 2: Custom Semgrep Rules
# ---------------------------------------------------------------------------

def exercise_2_semgrep_rules():
    """Design custom Semgrep rules for common security patterns."""
    rules = [
        {
            "id": "no-os-system-concat",
            "description": "Detect os.system() with string concatenation",
            "pattern_concept": "os.system(f'...' + ...) or os.system('...' + variable)",
            "semgrep_yaml": '''
  - id: no-os-system-concat
    patterns:
      - pattern: os.system(...)
      - pattern-not: os.system("...")
    message: "os.system() with dynamic input — use subprocess with list args"
    severity: ERROR
    languages: [python]
''',
        },
        {
            "id": "flask-post-no-content-type",
            "description": "Flask POST routes that don't validate Content-Type",
            "pattern_concept": "@app.route(..., methods=['POST']) without Content-Type check",
            "semgrep_yaml": '''
  - id: flask-post-check-content-type
    patterns:
      - pattern: |
          @app.route($PATH, methods=[..., "POST", ...])
          def $FUNC(...):
              ...
              $DATA = request.json
              ...
      - pattern-not: |
          @app.route($PATH, methods=[..., "POST", ...])
          def $FUNC(...):
              ...
              request.content_type
              ...
    message: "POST handler accesses request.json without Content-Type validation"
    severity: WARNING
    languages: [python]
''',
        },
        {
            "id": "no-eval-in-handler",
            "description": "Detect eval/exec in HTTP request handlers",
            "pattern_concept": "eval() or exec() inside Flask route handlers",
            "semgrep_yaml": '''
  - id: no-eval-in-handler
    patterns:
      - pattern-inside: |
          @app.route(...)
          def $FUNC(...):
              ...
      - pattern-either:
          - pattern: eval(...)
          - pattern: exec(...)
    message: "eval/exec in HTTP handler — potential code injection"
    severity: ERROR
    languages: [python]
''',
        },
        {
            "id": "no-hardcoded-db-string",
            "description": "Detect hardcoded database connection strings",
            "pattern_concept": "String matching database URL patterns in source code",
            "semgrep_yaml": '''
  - id: no-hardcoded-db-string
    pattern-regex: "(postgresql|mysql|mongodb|redis)://\\w+:\\w+@"
    message: "Hardcoded database connection string with credentials"
    severity: ERROR
    languages: [python]
''',
        },
    ]

    print("Custom Semgrep Security Rules:")
    print("=" * 60)
    for rule in rules:
        print(f"\n  Rule: {rule['id']}")
        print(f"  Description: {rule['description']}")
        print(f"  Concept: {rule['pattern_concept']}")
        print(f"  YAML:{rule['semgrep_yaml']}")


# ---------------------------------------------------------------------------
# Exercise 3: Dependency Audit
# ---------------------------------------------------------------------------

def exercise_3_dependency_audit():
    """
    Simulated dependency audit showing vulnerable packages and fixes.
    """
    audit_results = [
        {
            "package": "flask",
            "installed": "2.0.1",
            "min_safe": "2.3.2",
            "cves": ["CVE-2023-30861"],
            "description": "Session cookie vulnerability",
        },
        {
            "package": "requests",
            "installed": "2.25.1",
            "min_safe": "2.31.0",
            "cves": ["CVE-2023-32681"],
            "description": "Proxy credential leak on redirect",
        },
        {
            "package": "django",
            "installed": "3.2.0",
            "min_safe": "3.2.25",
            "cves": ["CVE-2024-24680", "CVE-2023-46695"],
            "description": "Multiple security fixes",
        },
        {
            "package": "pyyaml",
            "installed": "5.3.1",
            "min_safe": "6.0.1",
            "cves": ["CVE-2020-14343"],
            "description": "Arbitrary code execution via yaml.load()",
        },
        {
            "package": "pillow",
            "installed": "8.0.0",
            "min_safe": "10.2.0",
            "cves": ["CVE-2023-44271", "CVE-2023-50447"],
            "description": "Multiple buffer overflow vulnerabilities",
        },
    ]

    print("Dependency Audit Report")
    print("=" * 60)
    for pkg in audit_results:
        print(f"\n  {pkg['package']} {pkg['installed']} -> {pkg['min_safe']}")
        print(f"  CVEs: {', '.join(pkg['cves'])}")
        print(f"  Issue: {pkg['description']}")

    print("\n--- requirements-secure.txt ---")
    for pkg in audit_results:
        print(f"  {pkg['package']}>={pkg['min_safe']}")


# ---------------------------------------------------------------------------
# Exercise 4: Property-Based Testing
# ---------------------------------------------------------------------------

def exercise_4_property_based_testing():
    """
    Design property-based tests using Hypothesis concepts.
    Shows the test structure and strategies.
    """
    # Password validator to test
    def validate_password(password: str) -> bool:
        """Password must have uppercase, lowercase, digit, special char, min 8."""
        if len(password) < 8:
            return False
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()-_=+[]{}|;:,.<>?" for c in password)
        return has_upper and has_lower and has_digit and has_special

    # URL sanitizer to test
    def sanitize_url(url: str) -> str:
        """Remove dangerous URL schemes."""
        url = url.strip()
        if re.match(r"^(javascript|data|vbscript):", url, re.IGNORECASE):
            return ""
        if not re.match(r"^https?://", url, re.IGNORECASE):
            return ""
        return url

    # HTML tag stripper to test
    def strip_html(text: str) -> str:
        """Remove all HTML tags but preserve text content."""
        return re.sub(r"<[^>]+>", "", text)

    print("Property-Based Testing Strategies:")
    print("=" * 60)

    # Test the password validator
    print("\n1. Password Validator Tests:")
    test_cases = [
        ("Abc1!xyz", True, "Minimum valid password"),
        ("abc", False, "Too short, missing requirements"),
        ("ABCDEFGH", False, "Missing lowercase, digit, special"),
        ("abcdefgh", False, "Missing uppercase, digit, special"),
        ("Abcdefg1", False, "Missing special character"),
        ("A" * 100 + "a1!", True, "Very long password"),
        ("", False, "Empty string"),
    ]
    for pwd, expected, desc in test_cases:
        result = validate_password(pwd)
        status = "PASS" if result == expected else "FAIL"
        print(f"  [{status}] {desc}: validate('{pwd[:20]}...') = {result}")

    # Test the URL sanitizer
    print("\n2. URL Sanitizer Tests:")
    url_tests = [
        ("https://example.com", "https://example.com"),
        ("javascript:alert(1)", ""),
        ("data:text/html,<script>", ""),
        ("JAVASCRIPT:void(0)", ""),
        ("ftp://files.com", ""),
        ("http://safe.com", "http://safe.com"),
    ]
    for url, expected in url_tests:
        result = sanitize_url(url)
        status = "PASS" if result == expected else "FAIL"
        print(f"  [{status}] sanitize('{url}') = '{result}'")

    # Test the HTML stripper
    print("\n3. HTML Stripper Tests:")
    html_tests = [
        ("<b>bold</b>", "bold"),
        ("<script>alert(1)</script>", "alert(1)"),
        ("no tags", "no tags"),
        ('<a href="x">link</a>', "link"),
    ]
    for html, expected in html_tests:
        result = strip_html(html)
        status = "PASS" if result == expected else "FAIL"
        print(f"  [{status}] strip('{html}') = '{result}'")

    print("\nHypothesis test structure (conceptual):")
    print("""
  @given(st.text(min_size=0, max_size=100))
  def test_password_validator_never_crashes(password):
      # Property: validator never raises an exception
      result = validate_password(password)
      assert isinstance(result, bool)

  @given(st.text())
  def test_sanitized_url_is_safe(url):
      # Property: output never starts with javascript: or data:
      result = sanitize_url(url)
      assert not result.lower().startswith("javascript:")
      assert not result.lower().startswith("data:")

  @given(st.text())
  def test_strip_html_removes_all_tags(text):
      # Property: output contains no < or > from tags
      result = strip_html(text)
      assert "<" not in result or ">" not in result  # simplified
""")


# ---------------------------------------------------------------------------
# Exercise 5: CI/CD Security Pipeline (GitHub Actions)
# ---------------------------------------------------------------------------

def exercise_5_cicd_pipeline():
    """Design a CI/CD security pipeline in GitHub Actions."""
    workflow = '''
name: Security Pipeline
on: [pull_request]

permissions:
  contents: read
  security-events: write
  pull-requests: write

jobs:
  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: pip install bandit pip-audit

      # Step 1: SAST with Bandit
      - name: Run Bandit
        run: |
          bandit -r src/ -f sarif -o bandit.sarif || true

      - name: Upload Bandit SARIF
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: bandit.sarif

      # Step 2: Dependency audit
      - name: Run pip-audit
        run: |
          pip-audit -r requirements.txt --format json --output audit.json || true

      # Step 3: Secret scanning
      - name: Run gitleaks
        uses: gitleaks/gitleaks-action@v2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

      # Step 4: Check for critical findings
      - name: Gate check
        run: |
          python -c "
          import json, sys
          # Check Bandit
          with open('bandit.sarif') as f:
              sarif = json.load(f)
              critical = sum(1 for run in sarif.get('runs', [])
                           for result in run.get('results', [])
                           if result.get('level') in ('error',))
          if critical > 0:
              print(f'BLOCKED: {critical} critical Bandit findings')
              sys.exit(1)
          print('All security checks passed')
          "

      # Step 5: Comment on PR
      - name: Post summary
        if: always()
        uses: actions/github-script@v7
        with:
          script: |
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: 'Security scan completed. Check the Actions tab for details.'
            })
'''
    print("GitHub Actions Security Pipeline:")
    print("=" * 60)
    print(workflow)


# ---------------------------------------------------------------------------
# Exercise 6: Security Code Review
# ---------------------------------------------------------------------------

def exercise_6_security_code_review():
    """Identify all security vulnerabilities in the given code."""
    vulnerabilities = [
        {
            "location": "GET /api/users/<user_id>",
            "vuln": "SQL injection (f-string in query)",
            "fix": "Use parameterized query: WHERE id = ?",
        },
        {
            "location": "GET /api/users/<user_id>",
            "vuln": "No authentication — anyone can access any user",
            "fix": "Add @require_auth decorator",
        },
        {
            "location": "GET /api/users/<user_id> response",
            "vuln": "SSN exposed in response — sensitive PII leak",
            "fix": "Never return SSN in API. Redact or remove.",
        },
        {
            "location": "GET /api/users/<user_id> error",
            "vuln": "User enumeration via error message",
            "fix": "Return generic 'Not found' without echoing user_id",
        },
        {
            "location": "POST /api/login",
            "vuln": "SQL injection (f-string in WHERE clause)",
            "fix": "Use parameterized query",
        },
        {
            "location": "POST /api/login",
            "vuln": "Plaintext password comparison (no hashing)",
            "fix": "Hash passwords with argon2/bcrypt, compare hashes",
        },
        {
            "location": "POST /api/login",
            "vuln": "Weak JWT secret ('secret123')",
            "fix": "Use 256-bit random secret from environment variable",
        },
        {
            "location": "POST /api/login",
            "vuln": "JWT has no expiration claim",
            "fix": "Add 'exp' claim with short TTL (15 minutes)",
        },
        {
            "location": "GET /api/redirect",
            "vuln": "Open redirect — attacker can redirect to phishing site",
            "fix": "Validate URL against allowlist of trusted domains",
        },
        {
            "location": "POST /api/upload",
            "vuln": "Path traversal via filename (../../etc/passwd)",
            "fix": "Use secure_filename() and validate against base directory",
        },
        {
            "location": "POST /api/upload",
            "vuln": "No file type validation — can upload .exe, .sh, etc.",
            "fix": "Validate file extension against allowlist",
        },
        {
            "location": "POST /api/upload",
            "vuln": "No file size limit — denial of service via large upload",
            "fix": "Set MAX_CONTENT_LENGTH in Flask config",
        },
    ]

    print("Security Code Review — Vulnerability Report")
    print("=" * 60)
    for i, v in enumerate(vulnerabilities, 1):
        print(f"\n  {i}. [{v['location']}]")
        print(f"     Vulnerability: {v['vuln']}")
        print(f"     Fix: {v['fix']}")

    print(f"\nTotal: {len(vulnerabilities)} distinct vulnerabilities found")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Exercise 1: Bandit Scan Analysis (Fixed Code)")
    print("=" * 70)
    exercise_1_bandit_fixed()

    print("\n" + "=" * 70)
    print("Exercise 2: Custom Semgrep Rules")
    print("=" * 70)
    exercise_2_semgrep_rules()

    print("\n" + "=" * 70)
    print("Exercise 3: Dependency Audit")
    print("=" * 70)
    exercise_3_dependency_audit()

    print("\n" + "=" * 70)
    print("Exercise 4: Property-Based Testing")
    print("=" * 70)
    exercise_4_property_based_testing()

    print("\n" + "=" * 70)
    print("Exercise 5: CI/CD Security Pipeline")
    print("=" * 70)
    exercise_5_cicd_pipeline()

    print("\n" + "=" * 70)
    print("Exercise 6: Security Code Review")
    print("=" * 70)
    exercise_6_security_code_review()
