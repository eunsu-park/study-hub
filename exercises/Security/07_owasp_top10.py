"""
Exercise Solutions: OWASP Top 10 (2021)
=======================================
Lesson 07 from Security topic.

Covers vulnerability identification, secure application design,
security audit reporting, vulnerability fixes, and breach mapping.
"""

import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Exercise 1: Vulnerability Identification
# ---------------------------------------------------------------------------

def exercise_1_vulnerability_identification():
    """
    Identify OWASP Top 10 categories for each vulnerability in the
    given Flask application code.
    """
    issues = [
        {
            "code": "app.config['DEBUG'] = True",
            "owasp": "A05: Security Misconfiguration",
            "explanation": (
                "Debug mode in production exposes stack traces, enables "
                "interactive debugger (Werkzeug), and leaks internal details."
            ),
        },
        {
            "code": "app.config['SECRET_KEY'] = 'development'",
            "owasp": "A02: Cryptographic Failures",
            "explanation": (
                "Hardcoded, predictable secret key allows session forgery. "
                "An attacker can sign their own session cookies."
            ),
        },
        {
            "code": "f\"SELECT * FROM products WHERE name LIKE '%{q}%'\"",
            "owasp": "A03: Injection",
            "explanation": (
                "SQL injection via string interpolation. An attacker can "
                "extract data, modify records, or drop tables."
            ),
        },
        {
            "code": "def get_user(user_id): # No auth check",
            "owasp": "A01: Broken Access Control",
            "explanation": (
                "No authentication or authorization check. Any user can "
                "access any other user's data by changing the ID (IDOR)."
            ),
        },
        {
            "code": "pickle.loads(request.data)",
            "owasp": "A08: Software and Data Integrity Failures",
            "explanation": (
                "Deserializing untrusted pickle data allows arbitrary code "
                "execution. Pickle is a known RCE vector."
            ),
        },
        {
            "code": "requests.get(url)  # url from user input",
            "owasp": "A10: Server-Side Request Forgery (SSRF)",
            "explanation": (
                "User-controlled URL allows the server to make requests to "
                "internal services (e.g., cloud metadata at 169.254.169.254)."
            ),
        },
        {
            "code": "yaml.load(request.data)  # without Loader",
            "owasp": "A08: Software and Data Integrity Failures",
            "explanation": (
                "yaml.load() without SafeLoader allows arbitrary Python "
                "object construction, leading to RCE. Use yaml.safe_load()."
            ),
        },
        {
            "code": "user.password == password  # plaintext comparison",
            "owasp": "A02: Cryptographic Failures",
            "explanation": (
                "Passwords stored/compared in plaintext. Must use bcrypt "
                "or argon2 for hashing."
            ),
        },
        {
            "code": 'f"User {username} not found or wrong password"',
            "owasp": "A07: Identification and Authentication Failures",
            "explanation": (
                "Error message reveals whether the username exists "
                "(user enumeration). Also leaks the username in the response."
            ),
        },
        {
            "code": "traceback.format_exc() in error response",
            "owasp": "A05: Security Misconfiguration",
            "explanation": (
                "Full stack traces in error responses reveal internal "
                "file paths, library versions, and application structure."
            ),
        },
    ]

    print("OWASP Top 10 Vulnerability Identification")
    print("=" * 60)
    for i, item in enumerate(issues, 1):
        print(f"\nIssue {i}: {item['code']}")
        print(f"  OWASP:   {item['owasp']}")
        print(f"  Explanation: {item['explanation']}")


# ---------------------------------------------------------------------------
# Exercise 2: Secure Application Design (File Sharing)
# ---------------------------------------------------------------------------

class SecureFileSharing:
    """Secure file-sharing application addressing all OWASP Top 10 categories."""

    ALLOWED_EXTENSIONS = {".pdf", ".txt", ".png", ".jpg", ".csv", ".docx"}
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
    UPLOAD_DIR = "/secure-uploads"

    def __init__(self):
        self.files: dict[str, dict] = {}
        self.shares: dict[str, dict] = {}  # file_id -> {user_id: perms}

    def upload_file(self, user_id: str, file_data: bytes,
                    filename: str) -> dict:
        """
        Secure file upload.
        A03: Sanitize filename (prevent injection via name)
        A04: Enforce file size limits (insecure design prevention)
        A05: Validate file type (security misconfiguration)
        A08: Compute integrity hash
        """
        # A04: Size limit
        if len(file_data) > self.MAX_FILE_SIZE:
            return {"error": f"File exceeds {self.MAX_FILE_SIZE} byte limit"}

        # A03: Sanitize filename — remove path components, allow only safe chars
        safe_name = re.sub(r"[^\w\-.]", "_", os.path.basename(filename))
        if not safe_name or safe_name.startswith("."):
            return {"error": "Invalid filename"}

        # A05: Validate extension
        _, ext = os.path.splitext(safe_name)
        if ext.lower() not in self.ALLOWED_EXTENSIONS:
            return {"error": f"File type '{ext}' not allowed"}

        # A08: Compute integrity hash
        file_hash = hashlib.sha256(file_data).hexdigest()

        # Generate secure file ID (not sequential)
        file_id = secrets.token_urlsafe(16)

        self.files[file_id] = {
            "id": file_id,
            "owner": user_id,
            "filename": safe_name,
            "size": len(file_data),
            "hash": file_hash,
        }
        self.shares[file_id] = {user_id: ["read", "write", "delete"]}

        # A09: Log the upload event
        print(f"  [AUDIT] User {user_id} uploaded {safe_name} ({file_id})")

        return {"file_id": file_id, "hash": file_hash}

    def share_file(self, owner_id: str, file_id: str,
                   target_user_id: str, permissions: list[str]) -> bool:
        """
        Share a file with another user.
        A01: Verify ownership before sharing
        A04: Limit sharing to prevent abuse
        """
        # A01: Check ownership
        file_info = self.files.get(file_id)
        if not file_info or file_info["owner"] != owner_id:
            return False

        # A04: Validate permission values (no arbitrary strings)
        valid_perms = {"read", "write"}
        filtered = [p for p in permissions if p in valid_perms]
        if not filtered:
            return False

        self.shares.setdefault(file_id, {})[target_user_id] = filtered

        # A09: Log
        print(f"  [AUDIT] User {owner_id} shared {file_id} "
              f"with {target_user_id} ({filtered})")
        return True

    def download_file(self, user_id: str, file_id: str) -> Optional[dict]:
        """
        Download a file.
        A01: Verify access control
        A09: Log download events
        """
        # A01: Check access
        file_shares = self.shares.get(file_id, {})
        if user_id not in file_shares:
            print(f"  [AUDIT] DENIED: User {user_id} tried to download {file_id}")
            return None

        if "read" not in file_shares[user_id]:
            return None

        # A09: Log
        print(f"  [AUDIT] User {user_id} downloaded {file_id}")
        return self.files.get(file_id)

    @staticmethod
    def fetch_external_file(url: str) -> Optional[bytes]:
        """
        Fetch a file from an external URL.
        A10: SSRF prevention
        A06: Use validated, updated libraries
        """
        from urllib.parse import urlparse

        parsed = urlparse(url)

        # A10: Block internal/private URLs
        blocked_hosts = {"localhost", "127.0.0.1", "0.0.0.0", "169.254.169.254"}
        if parsed.hostname in blocked_hosts:
            print(f"  [AUDIT] BLOCKED SSRF attempt to {parsed.hostname}")
            return None

        # Block private IP ranges
        if parsed.hostname and parsed.hostname.startswith(("10.", "172.", "192.168.")):
            print(f"  [AUDIT] BLOCKED SSRF attempt to private IP {parsed.hostname}")
            return None

        # Only allow HTTPS
        if parsed.scheme != "https":
            return None

        print(f"  [AUDIT] External fetch: {url}")
        # In production: return httpx.get(url, timeout=10, follow_redirects=False).content
        return b"simulated-content"


def exercise_2_secure_file_sharing():
    """Demonstrate the secure file sharing system."""
    app = SecureFileSharing()

    # Upload
    result = app.upload_file("alice", b"Hello, World!", "report.pdf")
    print(f"Upload result: {result}")

    # Malicious filename
    result = app.upload_file("alice", b"data", "../../../etc/passwd")
    print(f"Path traversal attempt: {result}")

    # Blocked file type
    result = app.upload_file("alice", b"data", "malware.exe")
    print(f"Blocked extension: {result}")

    # Share and download
    file_id = list(app.files.keys())[0]
    app.share_file("alice", file_id, "bob", ["read"])
    downloaded = app.download_file("bob", file_id)
    print(f"Bob downloaded: {downloaded is not None}")

    # Unauthorized access
    downloaded = app.download_file("eve", file_id)
    print(f"Eve access denied: {downloaded is None}")

    # SSRF prevention
    app.fetch_external_file("https://169.254.169.254/latest/meta-data/")


# ---------------------------------------------------------------------------
# Exercise 3: Security Audit Report
# ---------------------------------------------------------------------------

def exercise_3_security_audit_report():
    """Generate a simulated security audit report."""
    findings = [
        {"category": "A01: Broken Access Control", "risk": "Critical",
         "detail": "JWT authentication present but no object-level authorization. "
                   "Users can access other users' files by changing file_id.",
         "remediation": "Add ownership check on every resource access endpoint.",
         "effort": "2-3 days"},
        {"category": "A02: Cryptographic Failures", "risk": "High",
         "detail": "JWT secret is only 20 characters. File storage lacks "
                   "encryption at rest.",
         "remediation": "Use 256-bit secret. Enable S3 SSE or EBS encryption.",
         "effort": "1 day"},
        {"category": "A03: Injection", "risk": "Critical",
         "detail": "PostgreSQL queries use string formatting in 2 endpoints.",
         "remediation": "Convert to parameterized queries using SQLAlchemy.",
         "effort": "1-2 days"},
        {"category": "A04: Insecure Design", "risk": "Medium",
         "detail": "No file size limits on upload. No rate limiting on any endpoint.",
         "remediation": "Add file size limit (50MB). Add rate limiting middleware.",
         "effort": "1 day"},
        {"category": "A05: Security Misconfiguration", "risk": "Medium",
         "detail": "Debug mode enabled. Default error pages expose stack traces.",
         "remediation": "Set DEBUG=False. Configure custom error handlers.",
         "effort": "0.5 days"},
        {"category": "A06: Vulnerable Components", "risk": "High",
         "detail": "15 dependencies not audited in 6 months. 3 have known CVEs.",
         "remediation": "Run pip-audit. Set up Dependabot. Pin dependency versions.",
         "effort": "1 day"},
        {"category": "A07: Auth Failures", "risk": "High",
         "detail": "No rate limiting on login. No MFA option. Weak password policy.",
         "remediation": "Add rate limiting (5/min). Implement TOTP 2FA. "
                        "Require 12+ char passwords.",
         "effort": "3-4 days"},
        {"category": "A08: Integrity Failures", "risk": "Medium",
         "detail": "No file integrity verification on download. "
                   "Dependencies not pinned to hashes.",
         "remediation": "Add checksum verification. Use pip install --require-hashes.",
         "effort": "1 day"},
        {"category": "A09: Logging Failures", "risk": "High",
         "detail": "Logs written to local files only. No structured logging. "
                   "No alerting on security events.",
         "remediation": "Ship logs to centralized system. Add security event alerting.",
         "effort": "2-3 days"},
        {"category": "A10: SSRF", "risk": "Critical",
         "detail": "Webhook integration fetches user-supplied URLs without "
                   "validation. Can reach AWS metadata endpoint.",
         "remediation": "Validate URLs against allowlist. Block private IPs. "
                        "Use network segmentation.",
         "effort": "1-2 days"},
    ]

    print("SECURITY AUDIT REPORT")
    print("=" * 60)
    print(f"Date: 2025-01-15")
    print(f"Application: File Sharing Web App")
    print(f"Auditor: Security Exercise Lab")
    print()

    critical = sum(1 for f in findings if f["risk"] == "Critical")
    high = sum(1 for f in findings if f["risk"] == "High")
    medium = sum(1 for f in findings if f["risk"] == "Medium")
    print(f"Summary: {critical} Critical, {high} High, {medium} Medium\n")

    for i, f in enumerate(findings, 1):
        print(f"{i}. [{f['risk'].upper()}] {f['category']}")
        print(f"   Finding: {f['detail']}")
        print(f"   Remediation: {f['remediation']}")
        print(f"   Effort: {f['effort']}")
        print()


# ---------------------------------------------------------------------------
# Exercise 4: Fix the Vulnerable Application
# ---------------------------------------------------------------------------

def exercise_4_fix_vulnerable_app():
    """
    Provide the corrected version of the Exercise 1 vulnerable app.
    """
    corrected = '''
import os
import secrets
import traceback
import logging
from flask import Flask, request, jsonify, session
import sqlite3
import json  # Use json instead of pickle for deserialization
import yaml

app = Flask(__name__)
app.config["DEBUG"] = False                                    # Fix 1: A05
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY",
                            secrets.token_hex(32))             # Fix 2: A02

logger = logging.getLogger("security")

@app.route("/api/search")
def search():
    q = request.args.get("q", "")
    conn = sqlite3.connect("app.db")
    cursor = conn.execute(
        "SELECT * FROM products WHERE name LIKE ?",           # Fix 3: A03
        (f"%{q}%",),
    )
    return jsonify(cursor.fetchall())

@app.route("/api/user/<int:user_id>")
@require_auth                                                  # Fix 4: A01
def get_user(user_id):
    if current_user.id != user_id and not current_user.is_admin:
        return jsonify({"error": "Forbidden"}), 403
    user = db.get_user(user_id)
    return jsonify(user)

@app.route("/api/import", methods=["POST"])
@require_auth
def import_data():
    data = json.loads(request.data)                            # Fix 5: A08
    return jsonify({"status": "imported"})

@app.route("/api/fetch", methods=["POST"])
@require_auth
def fetch():
    url = request.json.get("url", "")
    parsed = urlparse(url)
    # Fix 6: A10 — validate URL
    if parsed.hostname in BLOCKED_HOSTS or not url.startswith("https://"):
        return jsonify({"error": "URL not allowed"}), 400
    resp = httpx.get(url, timeout=10, follow_redirects=False)
    return resp.text

@app.route("/api/config", methods=["POST"])
@require_auth
@require_role("admin")
def load_config():
    config = yaml.safe_load(request.data)                      # Fix 7: A08
    return jsonify(config)

@app.route("/login", methods=["POST"])
@rate_limit("5/minute")
def login():
    username = request.json.get("username", "")
    password = request.json.get("password", "")
    user = db.find_user(username)
    if user and password_hasher.verify(user.password_hash, password):  # Fix 8: A02
        session.regenerate()
        session["user_id"] = user.id
        logger.info(f"Login success: user_id={user.id}")
        return jsonify({"status": "ok"})
    # Fix 9: A07 — generic error message
    return jsonify({"error": "Invalid credentials"}), 401

@app.errorhandler(500)
def error(e):
    logger.error(f"Server error: {e}")
    return jsonify({
        "error": "Internal server error",                      # Fix 10: A05
    }), 500
'''
    print("Corrected Application (all OWASP categories addressed):")
    print(corrected)


# ---------------------------------------------------------------------------
# Exercise 5: OWASP Top 10 Mapping (Real-World Breaches)
# ---------------------------------------------------------------------------

def exercise_5_breach_mapping():
    """Map real-world breaches to OWASP Top 10 categories."""
    breaches = [
        {
            "name": "Equifax (2017)",
            "description": "Unpatched Apache Struts vulnerability",
            "owasp": "A06: Vulnerable and Outdated Components",
            "explanation": (
                "The breach exploited CVE-2017-5638 in Apache Struts, a "
                "known vulnerability with a patch available months before. "
                "Failure to update dependencies is a classic A06 failure."
            ),
        },
        {
            "name": "Capital One (2019)",
            "description": "SSRF to access AWS metadata",
            "owasp": "A10: Server-Side Request Forgery (SSRF)",
            "explanation": (
                "Attacker exploited a misconfigured WAF to perform SSRF, "
                "accessing the AWS metadata service at 169.254.169.254 to "
                "obtain IAM credentials."
            ),
        },
        {
            "name": "SolarWinds (2020)",
            "description": "Compromised build pipeline",
            "owasp": "A08: Software and Data Integrity Failures",
            "explanation": (
                "Attackers compromised the build process to inject malware "
                "into the Orion software update. This is a supply chain "
                "integrity failure."
            ),
        },
        {
            "name": "Facebook (2019)",
            "description": "540M user records in unprotected S3",
            "owasp": "A05: Security Misconfiguration",
            "explanation": (
                "Third-party app developers stored Facebook user data in "
                "publicly accessible S3 buckets. Misconfigured cloud "
                "storage permissions."
            ),
        },
        {
            "name": "Uber (2016)",
            "description": "Hardcoded AWS credentials in GitHub repo",
            "owasp": "A02: Cryptographic Failures (also A05)",
            "explanation": (
                "Developers committed AWS access keys to a public GitHub "
                "repo. Attackers used them to access S3 buckets with 57M "
                "user records."
            ),
        },
        {
            "name": "British Airways (2018)",
            "description": "Magecart XSS in payment page",
            "owasp": "A03: Injection (XSS)",
            "explanation": (
                "Attackers injected malicious JavaScript into the payment "
                "page, skimming credit card details from 380,000 customers."
            ),
        },
        {
            "name": "Marriott (2018)",
            "description": "Breach undetected for 4 years",
            "owasp": "A09: Security Logging and Monitoring Failures",
            "explanation": (
                "Attackers had access since 2014 but were not detected "
                "until 2018. Inadequate logging and monitoring allowed "
                "years of undetected data exfiltration."
            ),
        },
        {
            "name": "Yahoo (2013-2014)",
            "description": "Weak/no encryption on user data",
            "owasp": "A02: Cryptographic Failures",
            "explanation": (
                "Yahoo used MD5 for password hashing (no salt) and failed "
                "to encrypt sensitive user data at rest. 3 billion accounts "
                "were compromised."
            ),
        },
    ]

    print("Real-World Breach Mapping to OWASP Top 10")
    print("=" * 60)
    for breach in breaches:
        print(f"\n{breach['name']}: {breach['description']}")
        print(f"  OWASP Category: {breach['owasp']}")
        print(f"  Analysis: {breach['explanation']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Exercise 1: Vulnerability Identification")
    print("=" * 70)
    exercise_1_vulnerability_identification()

    print("\n" + "=" * 70)
    print("Exercise 2: Secure File Sharing")
    print("=" * 70)
    exercise_2_secure_file_sharing()

    print("\n" + "=" * 70)
    print("Exercise 3: Security Audit Report")
    print("=" * 70)
    exercise_3_security_audit_report()

    print("\n" + "=" * 70)
    print("Exercise 4: Fix the Vulnerable Application")
    print("=" * 70)
    exercise_4_fix_vulnerable_app()

    print("\n" + "=" * 70)
    print("Exercise 5: Breach Mapping")
    print("=" * 70)
    exercise_5_breach_mapping()
