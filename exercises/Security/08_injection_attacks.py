"""
Exercise Solutions: Injection Attacks and Prevention
====================================================
Lesson 08 from Security topic.

Demonstrates SECURE coding patterns to prevent injection attacks.
NEVER includes actual exploit payloads — focuses on defense.
"""

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Exercise 1: SQL Injection Prevention (Fixed Code)
# ---------------------------------------------------------------------------

def exercise_1_sql_injection_prevention():
    """
    Identify injection points in the vulnerable code and demonstrate
    the secure version using parameterized queries.
    """
    # Analysis of the vulnerable code
    injection_points = [
        {
            "parameter": "category",
            "type": "String injection in WHERE clause",
            "why": "Wrapped in single quotes but not escaped — classic SQLi",
        },
        {
            "parameter": "min_price / max_price",
            "type": "Numeric injection in BETWEEN clause",
            "why": "No type validation — string values injected directly into SQL",
        },
        {
            "parameter": "sort",
            "type": "ORDER BY injection",
            "why": "Column name cannot be parameterized — must use allowlist",
        },
    ]

    print("SQL Injection Points Found:")
    for i, ip in enumerate(injection_points, 1):
        print(f"  {i}. {ip['parameter']}: {ip['type']}")
        print(f"     Why: {ip['why']}")

    # Secure implementation
    print("\n--- Secure Implementation ---")

    # Create in-memory test database
    db = sqlite3.connect(":memory:")
    db.execute("""
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT, price REAL, description TEXT, category TEXT
        )
    """)
    db.execute("INSERT INTO products VALUES (1, 'Widget', 9.99, 'A widget', 'tools')")
    db.execute("INSERT INTO products VALUES (2, 'Gadget', 29.99, 'A gadget', 'electronics')")
    db.commit()

    # Secure query with parameterized values and allowlisted sort column
    ALLOWED_SORT_COLUMNS = {"name", "price", "id"}

    def search_products_secure(category: str, min_price: str,
                               max_price: str, sort: str):
        """Secure product search with parameterized queries."""
        # Validate and coerce numeric inputs
        try:
            min_p = float(min_price)
            max_p = float(max_price)
        except (ValueError, TypeError):
            return {"error": "Invalid price values"}

        # Allowlist sort column (cannot parameterize ORDER BY)
        if sort not in ALLOWED_SORT_COLUMNS:
            sort = "name"

        query = f"""
            SELECT id, name, price, description
            FROM products
            WHERE category = ?
            AND price BETWEEN ? AND ?
            ORDER BY {sort}
        """
        results = db.execute(query, (category, min_p, max_p)).fetchall()
        return results

    results = search_products_secure("tools", "0", "100", "name")
    print(f"  Secure query results: {results}")

    # Attempted injection blocked
    results = search_products_secure("' OR 1=1 --", "0", "100", "name; DROP TABLE--")
    print(f"  Injection attempt results: {results}")  # Empty, not exploited
    db.close()


# ---------------------------------------------------------------------------
# Exercise 2: XSS Prevention (Analysis)
# ---------------------------------------------------------------------------

def exercise_2_xss_prevention():
    """
    Identify all XSS vectors in the vulnerable template and explain fixes.
    """
    xss_vectors = [
        {
            "location": "highlight_color in <style> tag",
            "type": "Reflected XSS (CSS context)",
            "attack": "Input like: red}</style><script>alert(1)</script><style>.x{color:blue",
            "fix": "Validate against allowlist of color names/hex codes. "
                   "Never inject user input into CSS.",
        },
        {
            "location": "search_query in paragraph (string concatenation)",
            "type": "Reflected XSS (HTML context)",
            "attack": "String concatenation bypasses Jinja2 auto-escaping.",
            "fix": "Use {{ search_query }} (Jinja2 variable, not string concat). "
                   "Jinja2 auto-escapes in {{ }} but NOT in concatenated strings.",
        },
        {
            "location": "post_content with |safe filter",
            "type": "Stored XSS",
            "attack": "Content from DB rendered as raw HTML. If user submitted "
                      "<script> tags, they execute.",
            "fix": "Use a sanitization library (e.g., bleach or nh3) to strip "
                   "dangerous tags while allowing safe HTML formatting.",
        },
        {
            "location": "share_url in javascript: href",
            "type": "Reflected XSS (JavaScript URL context)",
            "attack": "Input like: '); alert(1); //",
            "fix": "URL-encode the value. Better: use data attributes and "
                   "addEventListener instead of inline JavaScript.",
        },
        {
            "location": "current_user in <script> tag",
            "type": "Reflected XSS (JavaScript string context)",
            "attack": "Input like: '; alert(1); var x='",
            "fix": "Use json.dumps() to safely embed values in JavaScript. "
                   "Or pass via data attributes and read with JS.",
        },
        {
            "location": "innerHTML assignment in <script>",
            "type": "DOM-based XSS",
            "attack": "If userName contains HTML, innerHTML parses and executes it.",
            "fix": "Use textContent instead of innerHTML for text display.",
        },
    ]

    print("XSS Vectors Identified:")
    for i, v in enumerate(xss_vectors, 1):
        print(f"\n  {i}. Location: {v['location']}")
        print(f"     Type: {v['type']}")
        print(f"     Attack vector: {v['attack']}")
        print(f"     Fix: {v['fix']}")

    # CSP recommendation
    print("\nRecommended Content-Security-Policy header:")
    csp = (
        "default-src 'self'; "
        "script-src 'self' 'nonce-{random}'; "  # Use nonces for inline scripts
        "style-src 'self' 'nonce-{random}'; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self'"
    )
    print(f"  {csp}")


# ---------------------------------------------------------------------------
# Exercise 3: CSRF Protection
# ---------------------------------------------------------------------------

class CSRFProtection:
    """CSRF token generation and validation."""

    def __init__(self, secret_key: str):
        self._secret = secret_key.encode()

    def generate_token(self, session_id: str) -> str:
        """Generate a CSRF token bound to the session."""
        random_part = secrets.token_hex(16)
        signature = hmac.new(
            self._secret,
            f"{session_id}:{random_part}".encode(),
            hashlib.sha256,
        ).hexdigest()
        return f"{random_part}.{signature}"

    def validate_token(self, session_id: str, token: str) -> bool:
        """Validate a CSRF token."""
        if not token or "." not in token:
            return False
        random_part, signature = token.rsplit(".", 1)
        expected = hmac.new(
            self._secret,
            f"{session_id}:{random_part}".encode(),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(signature, expected)


def exercise_3_csrf_protection():
    """Demonstrate CSRF token generation and validation."""
    csrf = CSRFProtection(secret_key=secrets.token_hex(32))
    session_id = "session-abc-123"

    # Generate token
    token = csrf.generate_token(session_id)
    print(f"CSRF Token: {token[:40]}...")

    # Validate correct token
    valid = csrf.validate_token(session_id, token)
    print(f"Valid token: {valid}")

    # Validate wrong session
    valid = csrf.validate_token("different-session", token)
    print(f"Wrong session: {valid}")

    # Validate forged token
    valid = csrf.validate_token(session_id, "forged-token.signature")
    print(f"Forged token: {valid}")

    print("\nCSRF Protection Checklist:")
    print("  1. Generate unique token per session")
    print("  2. Include in forms: <input type='hidden' name='csrf_token' value='...'>")
    print("  3. For AJAX: send in X-CSRF-Token header")
    print("  4. Validate on every state-changing request")
    print("  5. Set cookies with SameSite=Lax or Strict")
    print("  6. Check Origin/Referer header as defense-in-depth")


# ---------------------------------------------------------------------------
# Exercise 4: Command Injection Prevention
# ---------------------------------------------------------------------------

def exercise_4_command_injection_prevention():
    """
    Rewrite file management endpoints using Python libraries
    instead of shell commands.
    """
    BASE_DIR = Path("/tmp/safe-uploads")

    def list_files_secure(path_str: str) -> dict:
        """List files using pathlib instead of os.popen('ls')."""
        requested = (BASE_DIR / path_str).resolve()
        # Path traversal prevention
        if not str(requested).startswith(str(BASE_DIR)):
            return {"error": "Access denied: path traversal detected"}
        if not requested.is_dir():
            return {"error": "Not a directory"}
        files = []
        for item in requested.iterdir():
            files.append({
                "name": item.name,
                "size": item.stat().st_size if item.is_file() else 0,
                "is_dir": item.is_dir(),
            })
        return {"files": files}

    def search_files_secure(path_str: str, pattern: str) -> dict:
        """Search files using pathlib.glob instead of os.popen('find')."""
        requested = (BASE_DIR / path_str).resolve()
        if not str(requested).startswith(str(BASE_DIR)):
            return {"error": "Access denied"}
        # Sanitize pattern — only allow alphanumeric, *, ?, .
        safe_pattern = re.sub(r"[^\w*?.]", "", pattern)
        results = [str(p.relative_to(BASE_DIR)) for p in requested.glob(safe_pattern)]
        return {"results": results}

    def compress_files_secure(files: list[str], output_name: str) -> dict:
        """Compress files using Python's tarfile instead of os.system('tar')."""
        import tarfile

        # Sanitize output name
        safe_output = re.sub(r"[^\w\-.]", "_", output_name)
        if not safe_output.endswith(".tar.gz"):
            safe_output += ".tar.gz"

        output_path = BASE_DIR / safe_output

        # Validate all file paths
        safe_files = []
        for f in files:
            full_path = (BASE_DIR / f).resolve()
            if not str(full_path).startswith(str(BASE_DIR)):
                return {"error": f"Access denied: {f}"}
            if full_path.exists():
                safe_files.append(full_path)

        # Use Python tarfile module (no shell involved)
        with tarfile.open(output_path, "w:gz") as tar:
            for f in safe_files:
                tar.add(f, arcname=f.name)

        return {"status": "compressed", "output": str(output_path)}

    print("Command Injection Prevention — Secure Implementations")
    print("=" * 60)
    print("\n1. list_files: Uses pathlib.iterdir() instead of 'ls -la'")
    print("   - Path traversal check: requested.resolve() must start with BASE_DIR")
    print("   - No shell execution at all")
    print("\n2. search_files: Uses pathlib.glob() instead of 'find'")
    print("   - Pattern sanitized to alphanumeric + wildcards only")
    print("   - No shell execution at all")
    print("\n3. compress_files: Uses tarfile module instead of os.system('tar')")
    print("   - All paths validated against BASE_DIR")
    print("   - Output name sanitized")
    print("\n4. system_info: REMOVED entirely")
    print("   - An endpoint that runs arbitrary commands is a backdoor")
    print("   - If system info is needed, use specific Python calls (platform, etc.)")


# ---------------------------------------------------------------------------
# Exercise 5: Full Application Security Review (Fixed Version)
# ---------------------------------------------------------------------------

def exercise_5_full_security_review():
    """
    Provide the corrected version of the vulnerable application
    that had SQL injection, XSS, CSRF, command injection, and SSTI.
    """
    print("Vulnerability Analysis and Fixes:")
    print("=" * 60)

    fixes = [
        {
            "vuln": "SQL Injection in /search",
            "fix": "Use parameterized query: db.execute('... WHERE title LIKE ?', (f'%{q}%',))",
        },
        {
            "vuln": "Reflected XSS in /search (render_template_string with f-string)",
            "fix": "Pass q as template variable: render_template_string(template, q=q, results=results)",
        },
        {
            "vuln": "SQL Injection in /comment (INSERT with f-string)",
            "fix": "Use parameterized: db.execute('INSERT INTO comments (text) VALUES (?)', (text,))",
        },
        {
            "vuln": "Stored XSS (comments rendered without escaping)",
            "fix": "Jinja2 auto-escapes {{ }} variables. Ensure |safe is never used on user content.",
        },
        {
            "vuln": "SSTI in /preview (user input as template)",
            "fix": "Never pass user input as template code. Use: render_template_string('<p>{{ content }}</p>', content=user_input)",
        },
        {
            "vuln": "Command Injection in /export (os.system with filename)",
            "fix": "Use shutil.copy() with pathlib. Validate filename against allowlist.",
        },
        {
            "vuln": "CSRF on /profile (no token validation)",
            "fix": "Add CSRF token validation. Use Flask-WTF or custom middleware.",
        },
        {
            "vuln": "SQL Injection in /profile (UPDATE with f-string)",
            "fix": "Parameterized: db.execute('UPDATE users SET bio = ? WHERE id = ?', (bio, user_id))",
        },
        {
            "vuln": "Weak secret key ('dev-key')",
            "fix": "Use os.environ['SECRET_KEY'] with a 256-bit random value.",
        },
    ]

    for i, f in enumerate(fixes, 1):
        print(f"\n{i}. {f['vuln']}")
        print(f"   Fix: {f['fix']}")

    # Defense-in-depth measures
    print("\n\nDefense-in-Depth Measures:")
    print("  1. Content-Security-Policy header on all responses")
    print("  2. Input validation (length, type, format) on all endpoints")
    print("  3. Output encoding (Jinja2 auto-escape, never use |safe)")
    print("  4. CSRF tokens on all state-changing endpoints")
    print("  5. Rate limiting on all endpoints")
    print("  6. Structured logging for security events")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Exercise 1: SQL Injection Prevention")
    print("=" * 70)
    exercise_1_sql_injection_prevention()

    print("\n" + "=" * 70)
    print("Exercise 2: XSS Prevention")
    print("=" * 70)
    exercise_2_xss_prevention()

    print("\n" + "=" * 70)
    print("Exercise 3: CSRF Protection")
    print("=" * 70)
    exercise_3_csrf_protection()

    print("\n" + "=" * 70)
    print("Exercise 4: Command Injection Prevention")
    print("=" * 70)
    exercise_4_command_injection_prevention()

    print("\n" + "=" * 70)
    print("Exercise 5: Full Application Security Review")
    print("=" * 70)
    exercise_5_full_security_review()
