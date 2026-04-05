# Security Testing

**Previous**: [12. Container and Cloud Security](./12_Container_Security.md) | **Next**: [14. Incident Response and Forensics](14_Incident_Response.md)

---

Security testing is the systematic process of finding vulnerabilities in software before attackers do. This lesson covers the major categories of security testing -- Static Application Security Testing (SAST), Dynamic Application Security Testing (DAST), Software Composition Analysis (SCA), and fuzzing -- along with penetration testing methodology and CI/CD integration. By the end, you will be able to build a comprehensive security testing pipeline for your projects.

## Learning Objectives

- Understand the differences between SAST, DAST, SCA, and fuzzing
- Use Bandit and Semgrep to find vulnerabilities in Python code
- Write custom Semgrep rules for project-specific patterns
- Integrate security scanning into CI/CD pipelines
- Apply penetration testing methodology
- Conduct effective security code reviews

---

## 1. Security Testing Overview

### 1.1 The Security Testing Pyramid

```
┌─────────────────────────────────────────────────────────────────┐
│                  Security Testing Pyramid                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                         /\                                       │
│                        /  \        Manual Penetration Testing    │
│                       / PT \       (Most expensive, most         │
│                      /      \       thorough for logic flaws)    │
│                     /--------\                                   │
│                    /   DAST   \     Dynamic testing against      │
│                   /            \    running application           │
│                  /--------------\                                 │
│                 /    Fuzzing     \   Automated input mutation    │
│                /                  \  for crash discovery          │
│               /--------------------\                             │
│              /        SCA           \  Dependency vulnerability  │
│             /                        \ scanning                  │
│            /--------------------------\                          │
│           /           SAST             \ Static code analysis   │
│          /                              \ (Cheapest, fastest,   │
│         /________________________________\ most automatable)    │
│                                                                  │
│  ◄── Cost/Effort increases going up                             │
│  ◄── Automation decreases going up                              │
│  ◄── Each layer catches different vulnerability classes          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 When to Apply Each Testing Type

```
┌──────────────────────────────────────────────────────────────────┐
│                    SDLC Security Testing Map                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Code Write ──► Commit ──► Build ──► Deploy ──► Production       │
│      │            │          │         │            │              │
│      ▼            ▼          ▼         ▼            ▼              │
│    IDE         Pre-commit   CI/CD    Staging     Continuous       │
│  Linting       Hooks       Pipeline  Testing     Monitoring       │
│                                                                   │
│  ┌──────┐   ┌──────────┐  ┌──────┐  ┌──────┐  ┌──────────┐     │
│  │ SAST │   │SAST + SCA│  │ All  │  │ DAST │  │ Runtime  │     │
│  │(IDE) │   │(pre-push)│  │Types │  │  PT  │  │ Scanning │     │
│  └──────┘   └──────────┘  └──────┘  └──────┘  └──────────┘     │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 1.3 Comparison of Testing Approaches

| Feature | SAST | DAST | SCA | Fuzzing | PT |
|---------|------|------|-----|---------|-----|
| **Requires running app** | No | Yes | No | Sometimes | Yes |
| **Language-specific** | Yes | No | Yes | Varies | No |
| **False positive rate** | High | Medium | Low | Low | Very Low |
| **Finds logic flaws** | Rarely | Sometimes | No | Rarely | Yes |
| **Automation level** | Full | Full | Full | Full | Partial |
| **Speed** | Fast | Slow | Fast | Medium | Very Slow |
| **Coverage** | Code paths | Attack surface | Dependencies | Input space | Targeted |

---

## 2. Static Application Security Testing (SAST)

### 2.1 How SAST Works

SAST tools analyze source code (or bytecode) without executing it. They build an abstract syntax tree (AST) or control/data flow graph and match patterns that indicate potential vulnerabilities.

```
┌─────────────────────────────────────────────────────────────────┐
│                      SAST Analysis Pipeline                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Source Code                                                     │
│      │                                                           │
│      ▼                                                           │
│  ┌──────────┐                                                    │
│  │  Parser   │ ──► Abstract Syntax Tree (AST)                   │
│  └──────────┘                                                    │
│      │                                                           │
│      ▼                                                           │
│  ┌──────────────────┐                                            │
│  │  Control Flow     │ ──► CFG: execution paths                 │
│  │  Analysis         │                                           │
│  └──────────────────┘                                            │
│      │                                                           │
│      ▼                                                           │
│  ┌──────────────────┐                                            │
│  │  Data Flow        │ ──► Taint tracking: source → sink        │
│  │  Analysis         │                                           │
│  └──────────────────┘                                            │
│      │                                                           │
│      ▼                                                           │
│  ┌──────────────────┐                                            │
│  │  Pattern Matching │ ──► Known vulnerability patterns         │
│  └──────────────────┘                                            │
│      │                                                           │
│      ▼                                                           │
│  Vulnerability Report                                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Bandit: Python Security Linter

Bandit is the most popular SAST tool for Python. It checks for common security issues like hardcoded passwords, use of `eval()`, insecure hash functions, and SQL injection patterns.

#### Installation and Basic Usage

```bash
# Install Bandit
pip install bandit

# Scan a single file
bandit target_file.py

# Scan an entire directory
bandit -r ./myproject/

# Scan with specific severity level
bandit -r ./myproject/ -ll  # Medium and above
bandit -r ./myproject/ -lll  # High severity only

# Output formats
bandit -r ./myproject/ -f json -o bandit_report.json
bandit -r ./myproject/ -f html -o bandit_report.html
bandit -r ./myproject/ -f csv -o bandit_report.csv

# Exclude specific tests
bandit -r ./myproject/ --skip B101,B601

# Include only specific tests
bandit -r ./myproject/ --tests B301,B302,B303
```

#### Bandit Test Categories

```python
"""
Bandit test categories and what they detect.
Each test has an ID like B101, B102, etc.
"""

# ─── B1xx: General security issues ───

# B101: assert_used - asserts are removed with -O flag
assert user.is_admin  # WARNING: B101

# B102: exec_used
exec(user_input)  # WARNING: B102

# B103: set_bad_file_permissions
import os
os.chmod('/etc/shadow', 0o777)  # WARNING: B103

# B104: hardcoded_bind_all_interfaces
app.run(host='0.0.0.0')  # WARNING: B104

# B105-B107: hardcoded passwords
password = "SuperSecret123"  # WARNING: B105
config = {"password": "admin123"}  # WARNING: B106


# ─── B2xx: Cryptographic issues ───

# B301: pickle usage (deserialization attack)
import pickle
data = pickle.loads(user_data)  # WARNING: B301

# B303: insecure hash function
import hashlib
h = hashlib.md5(password.encode())  # WARNING: B303

# B304-B305: insecure cipher
from Crypto.Cipher import DES
cipher = DES.new(key, DES.MODE_ECB)  # WARNING: B304


# ─── B3xx: Injection issues ───

# B601: paramiko shell injection
import paramiko
client.exec_command(user_input)  # WARNING: B601

# B602: subprocess with shell=True
import subprocess
subprocess.call(user_input, shell=True)  # WARNING: B602

# B608: SQL injection via string formatting
query = "SELECT * FROM users WHERE id = %s" % user_id  # WARNING: B608


# ─── B5xx: Cryptographic and SSL issues ───

# B501: request with verify=False
import requests
requests.get(url, verify=False)  # WARNING: B501

# B502: ssl with no version check
import ssl
context = ssl._create_unverified_context()  # WARNING: B502


# ─── B6xx: Injection issues (continued) ───

# B610-B611: Django SQL injection
Entry.objects.extra(where=[user_input])  # WARNING: B610

# B701: Jinja2 autoescape disabled
from jinja2 import Environment
env = Environment(autoescape=False)  # WARNING: B701
```

#### Bandit Configuration File

```yaml
# .bandit.yaml (or setup.cfg [bandit] section)

# Tests to skip
skips:
  - B101  # assert_used (acceptable in test files)
  - B601  # paramiko (we sanitize inputs)

# Paths to exclude
exclude_dirs:
  - tests
  - venv
  - .tox
  - migrations

# Set severity threshold
# Only report issues of this severity or higher
severity: LOW

# Set confidence threshold
confidence: LOW
```

#### Interpreting Bandit Output

```bash
# Run bandit on a sample vulnerable file
$ bandit -r vulnerable_app.py

Run started:2025-01-15 10:30:00

Test results:
>> Issue: [B608:hardcoded_sql_expressions] Possible SQL injection vector
   through string-based query construction.
   Severity: Medium   Confidence: Low
   CWE: CWE-89 (https://cwe.mitre.org/data/definitions/89.html)
   Location: vulnerable_app.py:42:0
   More Info: https://bandit.readthedocs.io/en/latest/plugins/b608...
41	    user_id = request.args.get('id')
42	    query = f"SELECT * FROM users WHERE id = '{user_id}'"
43	    cursor.execute(query)

>> Issue: [B105:hardcoded_password_string] Possible hardcoded password
   Severity: Low   Confidence: Medium
   CWE: CWE-259 (https://cwe.mitre.org/data/definitions/259.html)
   Location: vulnerable_app.py:15:0
14
15	    DATABASE_PASSWORD = "admin123"
16

--------------------------------------------------
Code scanned:
    Total lines of code: 156
    Total lines skipped (#nosec): 0

Run metrics:
    Total issues (by severity):
        Undefined: 0
        Low: 1
        Medium: 1
        High: 0
    Total issues (by confidence):
        Undefined: 0
        Low: 1
        Medium: 1
        High: 0
Files skipped (0):
```

#### Suppressing False Positives

```python
# Method 1: Inline suppression with #nosec
import hashlib
# This MD5 is for non-security checksum, not password hashing
checksum = hashlib.md5(file_content).hexdigest()  # nosec B303

# Method 2: Inline with specific test ID
password_hash = hashlib.sha256(salt + password)  # nosec B303

# Method 3: Using a baseline file
# Generate baseline (captures current issues)
# bandit -r ./myproject/ -f json -o baseline.json
# Run against baseline (only shows NEW issues)
# bandit -r ./myproject/ -b baseline.json
```

### 2.3 Semgrep: Multi-Language Static Analysis

Semgrep is a fast, open-source SAST tool that supports 30+ languages and uses a pattern-matching approach that is easy to understand and extend.

#### Installation and Basic Usage

```bash
# Install Semgrep
pip install semgrep

# Run with default rules
semgrep --config auto .

# Run with specific rulesets
semgrep --config p/python .
semgrep --config p/flask .
semgrep --config p/django .
semgrep --config p/owasp-top-ten .
semgrep --config p/security-audit .

# Run with a local rule file
semgrep --config my_rules.yaml .

# Output formats
semgrep --config auto . --json > report.json
semgrep --config auto . --sarif > report.sarif
```

#### Writing Custom Semgrep Rules

```yaml
# custom_rules.yaml

rules:
  # Rule 1: Detect SQL injection via f-strings
  - id: sql-injection-fstring
    patterns:
      - pattern: |
          $CURSOR.execute(f"...{$VAR}...")
    message: >
      Potential SQL injection via f-string interpolation.
      Use parameterized queries instead:
      cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    languages: [python]
    severity: ERROR
    metadata:
      cwe:
        - CWE-89
      owasp:
        - A03:2021 Injection
      category: security
      technology:
        - python

  # Rule 2: Detect hardcoded JWT secrets
  - id: hardcoded-jwt-secret
    patterns:
      - pattern: |
          jwt.encode($PAYLOAD, "...", ...)
      - pattern-not: |
          jwt.encode($PAYLOAD, $CONFIG, ...)
    message: >
      JWT token is signed with a hardcoded secret.
      Use environment variable or secret management service.
    languages: [python]
    severity: ERROR
    metadata:
      cwe:
        - CWE-798

  # Rule 3: Detect missing rate limiting on login
  - id: login-no-rate-limit
    patterns:
      - pattern: |
          @$APP.route("/login", ...)
          def $FUNC(...):
              ...
      - pattern-not-inside: |
          @limiter.limit(...)
          @$APP.route("/login", ...)
          def $FUNC(...):
              ...
    message: >
      Login endpoint without rate limiting. Add @limiter.limit()
      to prevent brute force attacks.
    languages: [python]
    severity: WARNING

  # Rule 4: Detect eval/exec with user input
  - id: dangerous-eval-user-input
    patterns:
      - pattern-either:
          - pattern: eval(request.$METHOD.get(...))
          - pattern: exec(request.$METHOD.get(...))
          - pattern: |
              $X = request.$METHOD.get(...)
              ...
              eval($X)
          - pattern: |
              $X = request.$METHOD.get(...)
              ...
              exec($X)
    message: >
      User input is passed to eval()/exec(). This allows
      arbitrary code execution. Never use eval/exec with
      untrusted input.
    languages: [python]
    severity: ERROR
    metadata:
      cwe:
        - CWE-95

  # Rule 5: Detect missing CSRF protection in Flask forms
  - id: flask-form-no-csrf
    patterns:
      - pattern: |
          @$APP.route("...", methods=[..., "POST", ...])
          def $FUNC(...):
              ...
              $X = request.form[...]
              ...
      - pattern-not-inside: |
          @csrf.exempt
          ...
    message: >
      POST endpoint processes form data. Ensure CSRF protection
      is enabled via Flask-WTF or manual token validation.
    languages: [python]
    severity: WARNING
```

#### Running Custom Rules

```bash
# Test a custom rule
semgrep --config custom_rules.yaml ./myproject/

# Combine custom rules with standard rulesets
semgrep --config custom_rules.yaml --config p/python ./myproject/

# Test rule against a specific file
semgrep --config custom_rules.yaml target_file.py

# Validate rule syntax
semgrep --validate --config custom_rules.yaml
```

#### Advanced Semgrep Patterns

```yaml
rules:
  # Taint tracking: trace data from source to sink
  - id: flask-ssrf
    mode: taint
    pattern-sources:
      - patterns:
          - pattern: request.$METHOD.get(...)
    pattern-sinks:
      - patterns:
          - pattern: requests.get($URL, ...)
    message: >
      User input flows into an HTTP request, possible SSRF.
    languages: [python]
    severity: ERROR

  # Metavariable comparison
  - id: weak-rsa-key
    patterns:
      - pattern: rsa.generate_private_key(public_exponent=65537, key_size=$SIZE)
      - metavariable-comparison:
          metavariable: $SIZE
          comparison: $SIZE < 2048
    message: RSA key size $SIZE is too small. Use at least 2048 bits.
    languages: [python]
    severity: ERROR

  # Pattern with focus on specific metavariable
  - id: unvalidated-redirect
    patterns:
      - pattern: redirect($URL)
      - pattern-not: redirect(url_for(...))
      - focus-metavariable: $URL
    message: Potential open redirect. Use url_for() for safe redirects.
    languages: [python]
    severity: WARNING
```

### 2.4 SonarQube Overview

SonarQube is an enterprise-grade platform for continuous code quality and security inspection.

```
┌─────────────────────────────────────────────────────────────────┐
│                    SonarQube Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Developer Machine              SonarQube Server                │
│  ┌──────────────┐              ┌──────────────────┐             │
│  │  Source Code  │    scan     │  ┌─────────────┐  │             │
│  │      +        │ ────────►  │  │   Analyzer   │  │             │
│  │ sonar-scanner │             │  │   Engine     │  │             │
│  └──────────────┘              │  └──────┬──────┘  │             │
│                                │         │         │             │
│  CI/CD Pipeline                │         ▼         │             │
│  ┌──────────────┐              │  ┌─────────────┐  │             │
│  │  Build Step   │   report   │  │   Database   │  │             │
│  │  + Scanner    │ ────────►  │  │ (PostgreSQL) │  │             │
│  └──────────────┘              │  └──────┬──────┘  │             │
│                                │         │         │             │
│                                │         ▼         │             │
│  Web Browser                   │  ┌─────────────┐  │             │
│  ┌──────────────┐              │  │  Web UI /    │  │             │
│  │  Dashboard    │ ◄────────  │  │  Dashboard   │  │             │
│  │  & Reports    │             │  └─────────────┘  │             │
│  └──────────────┘              └──────────────────┘             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

```bash
# Running SonarQube with Docker
docker run -d --name sonarqube \
  -p 9000:9000 \
  sonarqube:community

# Configure project: sonar-project.properties
# sonar.projectKey=my-python-project
# sonar.sources=src
# sonar.python.version=3.11
# sonar.exclusions=**/tests/**,**/migrations/**

# Run scanner
sonar-scanner \
  -Dsonar.projectKey=my-python-project \
  -Dsonar.sources=. \
  -Dsonar.host.url=http://localhost:9000 \
  -Dsonar.token=your_token_here
```

---

## 3. Dynamic Application Security Testing (DAST)

### 3.1 How DAST Works

DAST tools test a running application by sending crafted requests and analyzing responses for vulnerabilities. They act like an automated attacker.

```
┌─────────────────────────────────────────────────────────────────┐
│                     DAST Testing Flow                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────┐     Crawl      ┌───────────────┐                │
│  │   DAST    │ ─────────────► │   Running      │                │
│  │   Tool    │                │   Application  │                │
│  │           │ ◄───────────── │   (Target)     │                │
│  │           │    Responses   │               │                │
│  │           │                └───────────────┘                │
│  │           │                                                  │
│  │  Phase 1: │  Spider/crawl to discover endpoints             │
│  │  Discover │  Find forms, parameters, API endpoints          │
│  │           │                                                  │
│  │  Phase 2: │  Send malicious payloads:                       │
│  │  Attack   │  - SQL injection strings                        │
│  │           │  - XSS payloads                                  │
│  │           │  - Path traversal attempts                       │
│  │           │  - Command injection                             │
│  │           │                                                  │
│  │  Phase 3: │  Analyze responses for:                         │
│  │  Analyze  │  - Error messages revealing info                │
│  │           │  - Reflected input (XSS)                        │
│  │           │  - Unexpected behavior                           │
│  └───────────┘                                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 OWASP ZAP

OWASP ZAP (Zed Attack Proxy) is the most widely used free DAST tool.

```bash
# Run ZAP in Docker
docker run -t owasp/zap2docker-stable zap-baseline.py \
  -t http://target-app:8080

# Full scan (more thorough, slower)
docker run -t owasp/zap2docker-stable zap-full-scan.py \
  -t http://target-app:8080

# API scan (for REST APIs)
docker run -t owasp/zap2docker-stable zap-api-scan.py \
  -t http://target-app:8080/openapi.json \
  -f openapi

# Generate HTML report
docker run -v $(pwd):/zap/wrk/:rw \
  -t owasp/zap2docker-stable zap-baseline.py \
  -t http://target-app:8080 \
  -r report.html
```

#### ZAP Python API

```python
"""
Using OWASP ZAP's Python API for automated scanning.
Requires: pip install python-owasp-zap-v2.4
ZAP must be running as a daemon.
"""

from zapv2 import ZAPv2
import time

# Connect to ZAP
zap = ZAPv2(
    apikey='your-api-key',
    proxies={
        'http': 'http://127.0.0.1:8080',
        'https': 'http://127.0.0.1:8080'
    }
)

target = 'http://target-app:5000'

def run_zap_scan(target_url: str) -> dict:
    """Run a ZAP scan against target and return results."""

    print(f"[*] Spidering target: {target_url}")
    scan_id = zap.spider.scan(target_url)

    # Wait for spider to complete
    while int(zap.spider.status(scan_id)) < 100:
        print(f"    Spider progress: {zap.spider.status(scan_id)}%")
        time.sleep(2)

    print(f"[*] Spider found {len(zap.spider.results(scan_id))} URLs")

    # Run active scan
    print(f"[*] Starting active scan...")
    scan_id = zap.ascan.scan(target_url)

    while int(zap.ascan.status(scan_id)) < 100:
        print(f"    Active scan progress: {zap.ascan.status(scan_id)}%")
        time.sleep(5)

    # Get alerts
    alerts = zap.core.alerts(baseurl=target_url)

    # Categorize by risk level
    results = {
        'High': [],
        'Medium': [],
        'Low': [],
        'Informational': []
    }

    for alert in alerts:
        risk = alert.get('risk', 'Informational')
        results[risk].append({
            'name': alert.get('name'),
            'url': alert.get('url'),
            'description': alert.get('description'),
            'solution': alert.get('solution'),
            'cweid': alert.get('cweid'),
        })

    return results


def print_results(results: dict) -> None:
    """Print scan results in a readable format."""
    for risk_level in ['High', 'Medium', 'Low', 'Informational']:
        alerts = results[risk_level]
        if alerts:
            print(f"\n{'='*60}")
            print(f"  {risk_level} Risk Alerts: {len(alerts)}")
            print(f"{'='*60}")
            for alert in alerts:
                print(f"\n  [{alert['cweid']}] {alert['name']}")
                print(f"  URL: {alert['url']}")
                print(f"  Solution: {alert['solution'][:100]}...")


if __name__ == '__main__':
    results = run_zap_scan(target)
    print_results(results)
```

### 3.3 Burp Suite Concepts

Burp Suite is a commercial (with free Community edition) web security testing platform.

```
┌─────────────────────────────────────────────────────────────────┐
│                   Burp Suite Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Browser ◄──────► Burp Proxy ◄──────► Target Server             │
│                      │                                           │
│           ┌──────────┼──────────────────┐                       │
│           │          │                  │                        │
│           ▼          ▼                  ▼                        │
│      ┌─────────┐ ┌─────────┐    ┌───────────┐                  │
│      │ Spider  │ │Repeater │    │  Scanner   │                  │
│      │ (Crawl) │ │ (Manual │    │(Automated) │                  │
│      │         │ │  test)  │    │            │                  │
│      └─────────┘ └─────────┘    └───────────┘                  │
│           │          │                  │                        │
│           ▼          ▼                  ▼                        │
│      ┌─────────┐ ┌─────────┐    ┌───────────┐                  │
│      │Sequencer│ │Intruder │    │  Decoder   │                  │
│      │ (Token  │ │(Payload │    │(Encode/    │                  │
│      │  test)  │ │  fuzzer)│    │ Decode)    │                  │
│      └─────────┘ └─────────┘    └───────────┘                  │
│                                                                  │
│  Key Capabilities:                                               │
│  - Intercept & modify HTTP/HTTPS traffic                        │
│  - Automated vulnerability scanning                              │
│  - Manual testing with Repeater & Intruder                      │
│  - Session token analysis with Sequencer                        │
│  - Extensible via BApp Store                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Software Composition Analysis (SCA)

### 4.1 Why SCA Matters

Most modern applications consist of 70-90% third-party code. SCA tools scan your dependencies for known vulnerabilities.

```
┌─────────────────────────────────────────────────────────────────┐
│                Your Application's Code Composition                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                                                          │   │
│  │  ██████████████████████████████████████████████████████  │   │
│  │  ██████████████████████████████████████████████████████  │   │
│  │  ██████████████ Third-party libraries ████████████████  │   │
│  │  ██████████████   (70-90% of code)    ████████████████  │   │
│  │  ██████████████████████████████████████████████████████  │   │
│  │  ██████████████████████████████████████████████████████  │   │
│  │  ██████████████████████████████████████████████████████  │   │
│  │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │   │
│  │  ░░░░░ Your code (10-30%) ░░░░░░░░░░░░░░░░░░░░░░░░░░  │   │
│  │  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │   │
│  │                                                          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Each third-party library may have its OWN dependencies         │
│  (transitive dependencies), creating a deep dependency tree.    │
│  A vulnerability anywhere in this tree affects YOUR app.        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 pip-audit

```bash
# Install pip-audit
pip install pip-audit

# Scan current environment
pip-audit

# Scan a requirements file
pip-audit -r requirements.txt

# Output in JSON format
pip-audit -f json -o audit_report.json

# Auto-fix vulnerabilities (upgrade packages)
pip-audit --fix

# Scan with specific vulnerability database
pip-audit --vulnerability-service osv  # Google OSV (default)
pip-audit --vulnerability-service pypi  # PyPI Advisory DB

# Strict mode: exit with error if any vulnerabilities found
pip-audit --strict
```

#### Example pip-audit Output

```
$ pip-audit -r requirements.txt

Found 3 known vulnerabilities in 2 packages

Name       Version  ID                  Fix Versions
---------- -------- ------------------- ---------------
flask      2.0.1    PYSEC-2023-62       2.3.2
requests   2.25.1   PYSEC-2023-74       2.31.0
requests   2.25.1   GHSA-j8r2-6x86-q33q 2.32.0
```

### 4.3 Safety

```bash
# Install safety
pip install safety

# Check current environment
safety check

# Check a requirements file
safety check -r requirements.txt

# Output in JSON format
safety check --output json

# Use in CI (exit code 1 if vulnerabilities found)
safety check --full-report
```

### 4.4 Python Script: Dependency Vulnerability Scanner

```python
"""
dependency_scanner.py - A comprehensive dependency vulnerability scanner.
Combines pip-audit results with additional checks.
"""

import subprocess
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Vulnerability:
    """Represents a single vulnerability in a dependency."""
    package: str
    version: str
    vuln_id: str
    description: str = ""
    fix_version: Optional[str] = None
    severity: str = "UNKNOWN"
    aliases: list[str] = field(default_factory=list)


@dataclass
class ScanResult:
    """Results from a dependency scan."""
    vulnerabilities: list[Vulnerability] = field(default_factory=list)
    scanned_packages: int = 0
    scan_tool: str = ""
    errors: list[str] = field(default_factory=list)


def run_pip_audit(requirements_file: Optional[str] = None) -> ScanResult:
    """Run pip-audit and parse results."""
    cmd = ["pip-audit", "-f", "json", "--desc"]
    if requirements_file:
        cmd.extend(["-r", requirements_file])

    result = ScanResult(scan_tool="pip-audit")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        data = json.loads(proc.stdout)

        # Parse dependencies
        for dep in data.get("dependencies", []):
            result.scanned_packages += 1
            for vuln in dep.get("vulns", []):
                v = Vulnerability(
                    package=dep["name"],
                    version=dep["version"],
                    vuln_id=vuln["id"],
                    description=vuln.get("description", ""),
                    fix_version=vuln.get("fix_versions", [None])[0]
                        if vuln.get("fix_versions") else None,
                    aliases=vuln.get("aliases", [])
                )
                result.vulnerabilities.append(v)

    except FileNotFoundError:
        result.errors.append("pip-audit not installed. Run: pip install pip-audit")
    except subprocess.TimeoutExpired:
        result.errors.append("pip-audit timed out after 120 seconds")
    except json.JSONDecodeError as e:
        result.errors.append(f"Failed to parse pip-audit output: {e}")

    return result


def check_requirements_pinning(requirements_file: str) -> list[str]:
    """
    Check if dependencies are properly pinned with exact versions.
    Unpinned dependencies are a security risk because they can
    silently pull in vulnerable versions.
    """
    warnings = []
    req_path = Path(requirements_file)

    if not req_path.exists():
        return [f"Requirements file not found: {requirements_file}"]

    for line_num, line in enumerate(req_path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue

        # Check for unpinned dependencies
        if "==" not in line:
            if ">=" in line:
                warnings.append(
                    f"Line {line_num}: '{line}' uses >= (unpinned upper bound). "
                    f"Use == for exact pinning."
                )
            elif line.isidentifier() or "." in line:
                warnings.append(
                    f"Line {line_num}: '{line}' has no version pin. "
                    f"Use == to pin exact version."
                )

    return warnings


def check_known_malicious_packages(requirements_file: str) -> list[str]:
    """
    Check for known typosquatting / malicious package names.
    This is a simplified check - real scanners use larger databases.
    """
    # Known typosquatting examples (simplified list)
    SUSPICIOUS_PATTERNS = {
        "python-dateutil": "dateutil",       # common confusion
        "beautifulsoup4": "beautifulsoup",   # old version
        # These are examples of KNOWN malicious packages (now removed from PyPI)
        "colourama": "colorama",
        "python3-dateutil": "python-dateutil",
        "jeIlyfish": "jellyfish",            # capital I vs lowercase l
    }

    warnings = []
    req_path = Path(requirements_file)

    if not req_path.exists():
        return []

    for line in req_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        pkg_name = line.split("==")[0].split(">=")[0].split("<=")[0].strip()

        if pkg_name.lower() in [k.lower() for k in SUSPICIOUS_PATTERNS]:
            correct = SUSPICIOUS_PATTERNS.get(pkg_name, "unknown")
            warnings.append(
                f"WARNING: '{pkg_name}' may be a typosquat of '{correct}'. "
                f"Verify the package name is correct."
            )

    return warnings


def generate_report(scan_result: ScanResult, pinning_warnings: list[str],
                    typosquat_warnings: list[str]) -> str:
    """Generate a formatted security report."""
    lines = []
    lines.append("=" * 60)
    lines.append("  Dependency Security Scan Report")
    lines.append("=" * 60)
    lines.append(f"\nTool: {scan_result.scan_tool}")
    lines.append(f"Packages scanned: {scan_result.scanned_packages}")
    lines.append(f"Vulnerabilities found: {len(scan_result.vulnerabilities)}")

    if scan_result.errors:
        lines.append(f"\nErrors:")
        for err in scan_result.errors:
            lines.append(f"  [!] {err}")

    if scan_result.vulnerabilities:
        lines.append(f"\n{'─' * 60}")
        lines.append("  Vulnerabilities")
        lines.append(f"{'─' * 60}")

        for vuln in scan_result.vulnerabilities:
            lines.append(f"\n  Package: {vuln.package} {vuln.version}")
            lines.append(f"  ID:      {vuln.vuln_id}")
            if vuln.aliases:
                lines.append(f"  Aliases: {', '.join(vuln.aliases)}")
            if vuln.fix_version:
                lines.append(f"  Fix:     Upgrade to {vuln.fix_version}")
            if vuln.description:
                desc = vuln.description[:200]
                lines.append(f"  Detail:  {desc}")

    if pinning_warnings:
        lines.append(f"\n{'─' * 60}")
        lines.append("  Version Pinning Warnings")
        lines.append(f"{'─' * 60}")
        for w in pinning_warnings:
            lines.append(f"  [!] {w}")

    if typosquat_warnings:
        lines.append(f"\n{'─' * 60}")
        lines.append("  Typosquatting Warnings")
        lines.append(f"{'─' * 60}")
        for w in typosquat_warnings:
            lines.append(f"  [!] {w}")

    lines.append(f"\n{'=' * 60}")

    # Determine exit recommendation
    if scan_result.vulnerabilities or typosquat_warnings:
        lines.append("  RESULT: FAIL - Issues found that require attention")
    elif pinning_warnings:
        lines.append("  RESULT: WARN - Consider fixing pinning issues")
    else:
        lines.append("  RESULT: PASS - No issues found")

    lines.append("=" * 60)

    return "\n".join(lines)


if __name__ == "__main__":
    req_file = sys.argv[1] if len(sys.argv) > 1 else "requirements.txt"

    print(f"[*] Scanning dependencies from: {req_file}")

    # Run all checks
    scan_result = run_pip_audit(req_file)
    pinning_warnings = check_requirements_pinning(req_file)
    typosquat_warnings = check_known_malicious_packages(req_file)

    # Generate and print report
    report = generate_report(scan_result, pinning_warnings, typosquat_warnings)
    print(report)

    # Exit with appropriate code for CI
    if scan_result.vulnerabilities or typosquat_warnings:
        sys.exit(1)
    elif scan_result.errors:
        sys.exit(2)
    else:
        sys.exit(0)
```

### 4.5 Dependabot Configuration (GitHub)

```yaml
# .github/dependabot.yml

version: 2
updates:
  # Python pip dependencies
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "09:00"
      timezone: "America/New_York"
    open-pull-requests-limit: 10
    reviewers:
      - "security-team"
    labels:
      - "dependencies"
      - "security"
    # Group minor/patch updates together
    groups:
      minor-and-patch:
        update-types:
          - "minor"
          - "patch"
    # Ignore specific packages
    ignore:
      - dependency-name: "boto3"
        update-types: ["version-update:semver-patch"]
    # Security updates only (no version updates)
    # Uncomment below and remove schedule for security-only
    # open-pull-requests-limit: 0

  # Docker base images
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"

  # GitHub Actions
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
```

---

## 5. Fuzzing

### 5.1 What is Fuzzing?

Fuzzing is an automated testing technique that feeds random, malformed, or unexpected input to a program to find crashes, hangs, or security vulnerabilities.

```
┌─────────────────────────────────────────────────────────────────┐
│                      Fuzzing Feedback Loop                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────┐     ┌────────────┐     ┌────────────────┐       │
│  │   Seed     │     │  Mutation  │     │    Target      │       │
│  │   Corpus   │────►│  Engine    │────►│   Program      │       │
│  │  (initial  │     │            │     │                │       │
│  │   inputs)  │     │ - bit flip │     │ Parse input    │       │
│  └────────────┘     │ - insert   │     │ Process data   │       │
│       ▲             │ - delete   │     │ Return result  │       │
│       │             │ - replace  │     └───────┬────────┘       │
│       │             └────────────┘             │                │
│       │                                        │                │
│       │         ┌──────────────┐               │                │
│       │         │  Coverage    │◄──────────────┘                │
│       └─────────│  Monitor    │  (code coverage feedback)      │
│  (save inputs   │             │                                 │
│   that find     └──────────────┘                                │
│   new paths)          │                                         │
│                       ▼                                         │
│                ┌──────────────┐                                  │
│                │  Crash / Bug │                                  │
│                │  Detection   │                                  │
│                └──────────────┘                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 AFL (American Fuzzy Lop)

AFL is the most influential coverage-guided fuzzer for C/C++ programs.

```bash
# Install AFL++
sudo apt-get install afl++  # Debian/Ubuntu

# Compile target with AFL instrumentation
afl-cc -o target_program target_program.c

# Create seed corpus directory
mkdir -p seeds
echo "valid input" > seeds/seed1.txt

# Run AFL
afl-fuzz -i seeds -o findings ./target_program @@

# @@ is replaced by the input file path
# -i: seed directory
# -o: output directory (crashes, hangs, queue)

# Monitor AFL status
afl-whatsup findings/
```

#### AFL Output Directory Structure

```
findings/
├── crashes/         # Inputs that caused crashes
│   ├── id:000000,...  # Crash-triggering inputs
│   └── README.txt
├── hangs/           # Inputs that caused hangs/timeouts
├── queue/           # Interesting inputs (new coverage)
└── fuzzer_stats     # Current fuzzing statistics
```

### 5.3 Hypothesis: Property-Based Testing for Python

Hypothesis is a Python library for property-based testing. While not a traditional fuzzer, it automatically generates test inputs to find edge cases.

```python
"""
Property-based testing with Hypothesis.
Install: pip install hypothesis
"""

from hypothesis import given, assume, settings, HealthCheck
from hypothesis import strategies as st
import json
import re


# ─── Basic Example: Testing a function with generated inputs ───

def encode_decode_round_trip(data: str) -> bool:
    """Encoding then decoding should return original data."""
    encoded = data.encode('utf-8')
    decoded = encoded.decode('utf-8')
    return decoded == data


@given(st.text())
def test_encode_decode_roundtrip(s):
    """Test that UTF-8 encode/decode is a perfect round trip."""
    assert encode_decode_round_trip(s)


# ─── Testing JSON parsing robustness ───

@given(st.text())
def test_json_loads_doesnt_crash(s):
    """
    json.loads should either parse successfully or raise
    ValueError/JSONDecodeError - never crash or hang.
    """
    try:
        json.loads(s)
    except (json.JSONDecodeError, ValueError):
        pass  # Expected for invalid JSON


# ─── Testing input validation functions ───

def validate_email(email: str) -> bool:
    """Simple email validation (intentionally buggy for demo)."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


@given(st.emails())
def test_valid_emails_pass_validation(email):
    """All valid emails should pass our validator."""
    # This will likely FAIL, exposing gaps in our regex
    assert validate_email(email), f"Valid email rejected: {email}"


# ─── Testing with structured data ───

# Strategy for generating user registration data
user_strategy = st.fixed_dictionaries({
    'username': st.text(
        alphabet=st.characters(whitelist_categories=('L', 'N')),
        min_size=1,
        max_size=50
    ),
    'email': st.emails(),
    'password': st.text(min_size=8, max_size=128),
    'age': st.integers(min_value=0, max_value=200),
})


def process_registration(data: dict) -> dict:
    """Process user registration. Should handle any valid input."""
    if len(data['username']) < 1:
        raise ValueError("Username too short")
    if len(data['password']) < 8:
        raise ValueError("Password too short")
    if data['age'] < 13:
        raise ValueError("Must be at least 13 years old")

    return {
        'username': data['username'].lower(),
        'email': data['email'].lower(),
        'status': 'registered'
    }


@given(user_strategy)
def test_registration_never_crashes(user_data):
    """Registration should either succeed or raise ValueError."""
    try:
        result = process_registration(user_data)
        assert 'username' in result
        assert 'email' in result
    except ValueError:
        pass  # Expected for invalid data


# ─── Fuzzing a URL parser ───

from urllib.parse import urlparse, urlunparse

@given(st.from_regex(
    r'https?://[a-z0-9.-]{1,50}(:[0-9]{1,5})?(/[a-z0-9._-]{0,20}){0,5}(\?[a-z0-9=&]{0,50})?',
    fullmatch=True
))
def test_url_parse_roundtrip(url):
    """URL parse then unparse should preserve the URL."""
    parsed = urlparse(url)
    reconstructed = urlunparse(parsed)
    # Re-parse both to compare components (normalization may differ)
    assert urlparse(reconstructed).netloc == parsed.netloc


# ─── Testing security-sensitive functions ───

def sanitize_filename(filename: str) -> str:
    """Remove dangerous characters from a filename."""
    # Remove path separators and null bytes
    sanitized = filename.replace('/', '').replace('\\', '')
    sanitized = sanitized.replace('\x00', '')
    sanitized = sanitized.replace('..', '')
    # Remove leading dots (hidden files)
    sanitized = sanitized.lstrip('.')
    return sanitized or 'unnamed'


@given(st.text(min_size=1, max_size=255))
def test_sanitized_filename_is_safe(filename):
    """Sanitized filenames should never contain path traversal."""
    result = sanitize_filename(filename)
    assert '/' not in result, f"Path separator in: {result}"
    assert '\\' not in result, f"Backslash in: {result}"
    assert '\x00' not in result, f"Null byte in: {result}"
    assert not result.startswith('.'), f"Hidden file: {result}"
    assert '..' not in result, f"Path traversal in: {result}"
    assert len(result) > 0, "Empty filename after sanitization"


# ─── Advanced: Stateful testing ───

from hypothesis.stateful import RuleBasedStateMachine, rule, initialize

class ShoppingCartStateMachine(RuleBasedStateMachine):
    """
    Stateful testing for a shopping cart.
    Hypothesis will generate sequences of operations
    and check invariants after each step.
    """

    def __init__(self):
        super().__init__()
        self.cart = {}
        self.total = 0.0

    @initialize()
    def init_cart(self):
        self.cart = {}
        self.total = 0.0

    @rule(
        item=st.text(min_size=1, max_size=20),
        price=st.floats(min_value=0.01, max_value=10000, allow_nan=False),
        quantity=st.integers(min_value=1, max_value=100)
    )
    def add_item(self, item, price, quantity):
        """Add an item to the cart."""
        if item in self.cart:
            self.cart[item]['quantity'] += quantity
        else:
            self.cart[item] = {'price': price, 'quantity': quantity}
        self._recalculate_total()

    @rule(item=st.text(min_size=1, max_size=20))
    def remove_item(self, item):
        """Remove an item from the cart."""
        if item in self.cart:
            del self.cart[item]
            self._recalculate_total()

    def _recalculate_total(self):
        self.total = sum(
            v['price'] * v['quantity']
            for v in self.cart.values()
        )

    def teardown(self):
        """Invariant: total should never be negative."""
        assert self.total >= 0, f"Negative total: {self.total}"
        assert len(self.cart) >= 0


# Create a test from the state machine
TestShoppingCart = ShoppingCartStateMachine.TestCase


# ─── Running Hypothesis with settings ───

@settings(
    max_examples=1000,        # Number of test cases to generate
    deadline=None,            # No time limit per test
    suppress_health_check=[   # Suppress specific health checks
        HealthCheck.too_slow,
        HealthCheck.filter_too_much,
    ],
)
@given(st.binary(min_size=1, max_size=1024))
def test_binary_processing(data):
    """Test that our binary processor handles any input."""
    # Your binary processing function here
    try:
        result = data.decode('utf-8', errors='replace')
        assert isinstance(result, str)
    except Exception as e:
        # Should never reach here with errors='replace'
        raise AssertionError(f"Unexpected error: {e}")
```

### 5.4 Fuzzing Network Protocols

```python
"""
Simple protocol fuzzer for educational purposes.
Generates malformed inputs for protocol testing.
"""

import random
import struct
import socket
from typing import Generator


def mutate_bytes(data: bytes, num_mutations: int = 5) -> bytes:
    """Apply random mutations to a byte string."""
    data = bytearray(data)

    for _ in range(num_mutations):
        mutation_type = random.choice([
            'bit_flip', 'byte_replace', 'insert', 'delete',
            'duplicate', 'overflow'
        ])

        if len(data) == 0:
            data = bytearray(random.randbytes(10))
            continue

        pos = random.randint(0, max(0, len(data) - 1))

        if mutation_type == 'bit_flip':
            bit = random.randint(0, 7)
            data[pos] ^= (1 << bit)

        elif mutation_type == 'byte_replace':
            # Replace with interesting values
            interesting = [0x00, 0x01, 0x7F, 0x80, 0xFF, 0xFE]
            data[pos] = random.choice(interesting)

        elif mutation_type == 'insert':
            insert_data = random.randbytes(random.randint(1, 10))
            data[pos:pos] = insert_data

        elif mutation_type == 'delete':
            del_len = random.randint(1, min(5, len(data) - pos))
            del data[pos:pos + del_len]

        elif mutation_type == 'duplicate':
            chunk = data[pos:pos + random.randint(1, 10)]
            data[pos:pos] = chunk

        elif mutation_type == 'overflow':
            # Insert a very long string
            overflow = b'A' * random.choice([256, 1024, 4096, 65536])
            data[pos:pos] = overflow

    return bytes(data)


def generate_http_fuzz_requests(host: str, port: int) -> Generator[bytes, None, None]:
    """Generate fuzzed HTTP requests."""

    base_requests = [
        f"GET / HTTP/1.1\r\nHost: {host}\r\n\r\n".encode(),
        f"POST /login HTTP/1.1\r\nHost: {host}\r\nContent-Length: 10\r\n\r\nuser=admin".encode(),
        f"GET /{'A' * 5000} HTTP/1.1\r\nHost: {host}\r\n\r\n".encode(),
    ]

    # Yield original requests
    for req in base_requests:
        yield req

    # Yield mutated versions
    for _ in range(100):
        base = random.choice(base_requests)
        yield mutate_bytes(base)

    # Special cases
    yield b"\x00" * 1024                        # Null bytes
    yield b"GET / HTTP/9.9\r\n\r\n"             # Invalid version
    yield b"XYZZY / HTTP/1.1\r\n\r\n"           # Invalid method
    yield b"GET / HTTP/1.1\r\n" + b"X: Y\r\n" * 10000 + b"\r\n"  # Header bomb
    yield b"GET / HTTP/1.1\r\nContent-Length: -1\r\n\r\n"         # Negative length
    yield b"GET / HTTP/1.1\r\nContent-Length: 999999999\r\n\r\n"  # Huge length


def fuzz_target(host: str, port: int, timeout: float = 2.0) -> None:
    """
    Send fuzzed requests to a target server.

    WARNING: Only use against servers you own or have explicit
    permission to test. Unauthorized testing is illegal.
    """
    print(f"[*] Fuzzing {host}:{port}")
    crash_count = 0
    total_sent = 0

    for payload in generate_http_fuzz_requests(host, port):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect((host, port))
            sock.sendall(payload)

            try:
                response = sock.recv(4096)
                total_sent += 1
            except socket.timeout:
                print(f"  [!] Timeout on payload #{total_sent} "
                      f"(length={len(payload)})")
                total_sent += 1

        except ConnectionRefusedError:
            crash_count += 1
            print(f"  [!!!] Connection refused after payload #{total_sent}. "
                  f"Server may have crashed!")
            print(f"       Payload (first 100 bytes): {payload[:100]}")
            # Save crash-triggering payload
            with open(f"crash_{crash_count}.bin", "wb") as f:
                f.write(payload)

        except Exception as e:
            print(f"  [!] Error: {e}")

        finally:
            sock.close()

    print(f"\n[*] Fuzzing complete. Sent {total_sent} payloads. "
          f"Crashes detected: {crash_count}")
```

---

## 6. Penetration Testing Methodology

### 6.1 The Penetration Testing Process

```
┌─────────────────────────────────────────────────────────────────┐
│              Penetration Testing Methodology                     │
│              (Based on PTES / OWASP)                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────┐                                       │
│  │ 1. Planning & Scope  │  Define targets, rules of engagement │
│  │    (Pre-engagement)  │  Legal authorization, boundaries      │
│  └──────────┬───────────┘                                       │
│             │                                                    │
│             ▼                                                    │
│  ┌──────────────────────┐                                       │
│  │ 2. Reconnaissance    │  Passive: OSINT, DNS, WHOIS          │
│  │    (Information       │  Active: port scan, service enum     │
│  │     Gathering)        │                                      │
│  └──────────┬───────────┘                                       │
│             │                                                    │
│             ▼                                                    │
│  ┌──────────────────────┐                                       │
│  │ 3. Vulnerability     │  Automated scanning (Nessus, ZAP)    │
│  │    Assessment         │  Manual testing, misconfigurations   │
│  └──────────┬───────────┘                                       │
│             │                                                    │
│             ▼                                                    │
│  ┌──────────────────────┐                                       │
│  │ 4. Exploitation      │  Attempt to exploit vulnerabilities  │
│  │                       │  Gain access, escalate privileges    │
│  └──────────┬───────────┘                                       │
│             │                                                    │
│             ▼                                                    │
│  ┌──────────────────────┐                                       │
│  │ 5. Post-Exploitation │  Lateral movement, data exfil test   │
│  │                       │  Persistence mechanisms              │
│  └──────────┬───────────┘                                       │
│             │                                                    │
│             ▼                                                    │
│  ┌──────────────────────┐                                       │
│  │ 6. Reporting         │  Executive summary, technical detail │
│  │                       │  Remediation recommendations         │
│  └──────────────────────┘                                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Web Application Penetration Testing Checklist

```
┌──────────────────────────────────────────────────────────────────┐
│              Web Application Pentest Checklist                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Authentication:                                                  │
│  [ ] Brute force protection (account lockout, rate limiting)     │
│  [ ] Password complexity enforcement                              │
│  [ ] Multi-factor authentication                                  │
│  [ ] Session management (timeout, rotation, secure flags)        │
│  [ ] Password reset mechanism                                     │
│  [ ] Default credentials                                          │
│                                                                   │
│  Authorization:                                                   │
│  [ ] Horizontal privilege escalation (access other users' data)  │
│  [ ] Vertical privilege escalation (admin functions)             │
│  [ ] IDOR (Insecure Direct Object Reference)                    │
│  [ ] Missing function-level access control                       │
│                                                                   │
│  Input Validation:                                                │
│  [ ] SQL Injection (all input points)                            │
│  [ ] XSS (Reflected, Stored, DOM-based)                          │
│  [ ] Command Injection                                            │
│  [ ] Path Traversal / LFI / RFI                                 │
│  [ ] XML External Entity (XXE)                                   │
│  [ ] Server-Side Request Forgery (SSRF)                          │
│  [ ] Template Injection (SSTI)                                   │
│                                                                   │
│  Configuration:                                                   │
│  [ ] HTTPS enforcement                                            │
│  [ ] Security headers (CSP, HSTS, X-Frame-Options, etc.)        │
│  [ ] CORS policy                                                  │
│  [ ] Error handling (no stack traces in production)              │
│  [ ] Directory listing disabled                                   │
│  [ ] Unnecessary features/pages removed                          │
│                                                                   │
│  Business Logic:                                                  │
│  [ ] Race conditions                                              │
│  [ ] Price manipulation                                           │
│  [ ] Workflow bypass                                              │
│  [ ] Mass assignment                                              │
│                                                                   │
│  API-Specific:                                                    │
│  [ ] API key exposure                                             │
│  [ ] Rate limiting                                                │
│  [ ] Excessive data exposure                                     │
│  [ ] Lack of resource limits                                     │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 6.3 Python Penetration Testing Helpers

```python
"""
Penetration testing helper functions.
For authorized testing only.
"""

import requests
import urllib.parse
from typing import Optional


# ─── SQL Injection Testing ───

SQL_INJECTION_PAYLOADS = [
    "' OR '1'='1",
    "' OR '1'='1' --",
    "' OR '1'='1' /*",
    "1; DROP TABLE users --",
    "' UNION SELECT NULL,NULL,NULL --",
    "' AND 1=1 --",
    "' AND 1=2 --",
    "admin'--",
    "1' ORDER BY 1 --",
    "1' ORDER BY 100 --",
    "-1 OR 1=1",
    "' OR ''='",
    "'; WAITFOR DELAY '0:0:5' --",  # Time-based blind SQLi (MSSQL)
    "' OR SLEEP(5) --",              # Time-based blind SQLi (MySQL)
]


def test_sql_injection(url: str, param_name: str,
                       method: str = "GET") -> list[dict]:
    """
    Test a URL parameter for SQL injection vulnerabilities.
    Returns list of potentially vulnerable payloads.
    """
    results = []

    # First, get a baseline response
    if method == "GET":
        baseline = requests.get(url, params={param_name: "1"}, timeout=10)
    else:
        baseline = requests.post(url, data={param_name: "1"}, timeout=10)

    baseline_length = len(baseline.text)
    baseline_time = baseline.elapsed.total_seconds()

    for payload in SQL_INJECTION_PAYLOADS:
        try:
            if method == "GET":
                resp = requests.get(
                    url, params={param_name: payload}, timeout=15
                )
            else:
                resp = requests.post(
                    url, data={param_name: payload}, timeout=15
                )

            # Check for signs of SQL injection
            indicators = []

            # Error-based: SQL error messages in response
            sql_errors = [
                "sql syntax", "mysql", "sqlite", "postgresql",
                "ora-", "unclosed quotation", "unterminated string",
                "syntax error"
            ]
            for err in sql_errors:
                if err in resp.text.lower():
                    indicators.append(f"SQL error message: '{err}'")

            # Boolean-based: significant length difference
            length_diff = abs(len(resp.text) - baseline_length)
            if length_diff > baseline_length * 0.3:
                indicators.append(
                    f"Response length changed: {baseline_length} -> {len(resp.text)}"
                )

            # Time-based: response took significantly longer
            time_diff = resp.elapsed.total_seconds() - baseline_time
            if time_diff > 4.0:
                indicators.append(
                    f"Response delayed: {resp.elapsed.total_seconds():.1f}s "
                    f"(baseline: {baseline_time:.1f}s)"
                )

            if indicators:
                results.append({
                    'payload': payload,
                    'status_code': resp.status_code,
                    'indicators': indicators,
                    'response_length': len(resp.text),
                    'response_time': resp.elapsed.total_seconds()
                })

        except requests.exceptions.Timeout:
            results.append({
                'payload': payload,
                'status_code': None,
                'indicators': ['Request timed out (possible time-based SQLi)'],
                'response_length': 0,
                'response_time': 15.0
            })
        except requests.exceptions.RequestException as e:
            pass  # Connection error, skip

    return results


# ─── XSS Testing ───

XSS_PAYLOADS = [
    '<script>alert(1)</script>',
    '<img src=x onerror=alert(1)>',
    '<svg onload=alert(1)>',
    '"><script>alert(1)</script>',
    "'-alert(1)-'",
    '<body onload=alert(1)>',
    '{{7*7}}',  # Template injection test
    '${7*7}',   # Template injection test
    'javascript:alert(1)',
    '<iframe src="javascript:alert(1)">',
]


def test_reflected_xss(url: str, param_name: str) -> list[dict]:
    """
    Test for reflected XSS by checking if payloads appear
    unescaped in the response.
    """
    results = []

    for payload in XSS_PAYLOADS:
        try:
            resp = requests.get(
                url, params={param_name: payload}, timeout=10
            )

            # Check if payload is reflected without encoding
            if payload in resp.text:
                results.append({
                    'payload': payload,
                    'reflected': True,
                    'encoded': False,
                    'status_code': resp.status_code,
                })
            # Check for HTML-encoded version (partial protection)
            encoded = (payload.replace('<', '&lt;')
                              .replace('>', '&gt;')
                              .replace('"', '&quot;'))
            if encoded in resp.text and payload not in resp.text:
                results.append({
                    'payload': payload,
                    'reflected': True,
                    'encoded': True,
                    'status_code': resp.status_code,
                })

        except requests.exceptions.RequestException:
            pass

    return results


# ─── Security Header Checker ───

SECURITY_HEADERS = {
    'Strict-Transport-Security': {
        'description': 'HSTS - Forces HTTPS',
        'recommended': 'max-age=31536000; includeSubDomains',
        'severity': 'HIGH',
    },
    'Content-Security-Policy': {
        'description': 'CSP - Prevents XSS and injection',
        'recommended': "default-src 'self'",
        'severity': 'HIGH',
    },
    'X-Content-Type-Options': {
        'description': 'Prevents MIME sniffing',
        'recommended': 'nosniff',
        'severity': 'MEDIUM',
    },
    'X-Frame-Options': {
        'description': 'Prevents clickjacking',
        'recommended': 'DENY',
        'severity': 'MEDIUM',
    },
    'X-XSS-Protection': {
        'description': 'Legacy XSS filter',
        'recommended': '0',  # Modern guidance: disable, use CSP
        'severity': 'LOW',
    },
    'Referrer-Policy': {
        'description': 'Controls referrer information',
        'recommended': 'strict-origin-when-cross-origin',
        'severity': 'LOW',
    },
    'Permissions-Policy': {
        'description': 'Controls browser features',
        'recommended': 'camera=(), microphone=(), geolocation=()',
        'severity': 'MEDIUM',
    },
}


def check_security_headers(url: str) -> dict:
    """Check security headers of a URL."""
    resp = requests.get(url, timeout=10, allow_redirects=True)
    results = {
        'url': url,
        'status_code': resp.status_code,
        'headers_present': {},
        'headers_missing': {},
    }

    for header, info in SECURITY_HEADERS.items():
        value = resp.headers.get(header)
        if value:
            results['headers_present'][header] = {
                'value': value,
                'description': info['description'],
            }
        else:
            results['headers_missing'][header] = {
                'recommended': info['recommended'],
                'description': info['description'],
                'severity': info['severity'],
            }

    return results
```

---

## 7. Security Code Review Checklist

### 7.1 Code Review Process

```
┌──────────────────────────────────────────────────────────────────┐
│              Security Code Review Process                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Step 1: Understand the Change                                   │
│  ├── What does this code do?                                     │
│  ├── What data does it process?                                  │
│  └── What are the trust boundaries?                              │
│                                                                   │
│  Step 2: Check Input/Output                                      │
│  ├── All external input validated?                               │
│  ├── Output properly encoded?                                    │
│  └── File operations use safe paths?                             │
│                                                                   │
│  Step 3: Authentication & Authorization                          │
│  ├── Auth checks on all protected endpoints?                     │
│  ├── Proper session management?                                  │
│  └── Least privilege applied?                                    │
│                                                                   │
│  Step 4: Data Protection                                         │
│  ├── Sensitive data encrypted at rest?                           │
│  ├── Sensitive data encrypted in transit?                        │
│  ├── No secrets in source code?                                  │
│  └── PII handled correctly?                                      │
│                                                                   │
│  Step 5: Error Handling                                          │
│  ├── Errors don't leak information?                              │
│  ├── Proper exception handling?                                  │
│  └── Fail securely (deny by default)?                           │
│                                                                   │
│  Step 6: Dependencies                                            │
│  ├── New dependencies reviewed?                                  │
│  ├── Versions pinned?                                            │
│  └── Known vulnerabilities checked?                              │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 7.2 Python-Specific Security Review Checklist

```python
"""
Python Security Code Review Checklist with Examples.
Each section shows a VULNERABLE and SECURE version.
"""

# ─── 1. Input Validation ───

# VULNERABLE: No validation
def get_user_bad(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return db.execute(query)

# SECURE: Parameterized query + type validation
def get_user_good(user_id: int):
    if not isinstance(user_id, int) or user_id < 0:
        raise ValueError("Invalid user ID")
    return db.execute("SELECT * FROM users WHERE id = ?", (user_id,))


# ─── 2. Authentication ───

# VULNERABLE: Timing attack on password comparison
def check_password_bad(stored: str, provided: str) -> bool:
    return stored == provided  # String comparison short-circuits

# SECURE: Constant-time comparison
import hmac
def check_password_good(stored: str, provided: str) -> bool:
    return hmac.compare_digest(stored.encode(), provided.encode())


# ─── 3. Serialization ───

# VULNERABLE: pickle with untrusted data
import pickle
def load_data_bad(data: bytes):
    return pickle.loads(data)  # Arbitrary code execution!

# SECURE: Use JSON or validated schemas
import json
def load_data_good(data: str):
    parsed = json.loads(data)
    # Validate schema
    if not isinstance(parsed, dict):
        raise ValueError("Expected JSON object")
    return parsed


# ─── 4. File Operations ───

# VULNERABLE: Path traversal
import os
def read_file_bad(filename: str):
    with open(f"/uploads/{filename}") as f:
        return f.read()

# SECURE: Resolve and validate path
from pathlib import Path
UPLOAD_DIR = Path("/uploads").resolve()

def read_file_good(filename: str):
    file_path = (UPLOAD_DIR / filename).resolve()
    if not file_path.is_relative_to(UPLOAD_DIR):
        raise ValueError("Path traversal detected")
    if not file_path.is_file():
        raise FileNotFoundError("File not found")
    return file_path.read_text()


# ─── 5. Cryptography ───

# VULNERABLE: Weak hashing
import hashlib
def hash_password_bad(password: str) -> str:
    return hashlib.md5(password.encode()).hexdigest()

# SECURE: Proper password hashing
from argon2 import PasswordHasher
ph = PasswordHasher()

def hash_password_good(password: str) -> str:
    return ph.hash(password)

def verify_password_good(hash: str, password: str) -> bool:
    try:
        return ph.verify(hash, password)
    except Exception:
        return False


# ─── 6. Subprocess ───

# VULNERABLE: Shell injection
import subprocess
def run_command_bad(filename: str):
    subprocess.run(f"cat {filename}", shell=True)

# SECURE: No shell, use list
def run_command_good(filename: str):
    # Validate filename first
    if not Path(filename).name == filename:  # No path separators
        raise ValueError("Invalid filename")
    subprocess.run(["cat", filename], shell=False, check=True)


# ─── 7. Logging ───

# VULNERABLE: Logging sensitive data
import logging
logger = logging.getLogger(__name__)

def login_bad(username: str, password: str):
    logger.info(f"Login attempt: {username} / {password}")  # Logs password!

# SECURE: Never log secrets
def login_good(username: str, password: str):
    logger.info(f"Login attempt: user={username}")
    # Use placeholders for sensitive fields
    logger.debug("Login attempt: user=%s password=<REDACTED>", username)


# ─── 8. Regular Expressions ───

# VULNERABLE: ReDoS (Regular expression Denial of Service)
import re
def validate_email_bad(email: str) -> bool:
    # This pattern is vulnerable to catastrophic backtracking
    pattern = r'^([a-zA-Z0-9]+)*@[a-zA-Z0-9]+\.[a-zA-Z]+$'
    return bool(re.match(pattern, email, re.TIMEOUT))

# SECURE: Use a well-tested library or simple pattern
def validate_email_good(email: str) -> bool:
    # Simple, non-backtracking pattern
    if len(email) > 254:
        return False
    pattern = r'^[^@\s]+@[^@\s]+\.[^@\s]+$'
    return bool(re.match(pattern, email))
```

---

## 8. CI/CD Security Pipeline Integration

### 8.1 GitHub Actions Security Pipeline

```yaml
# .github/workflows/security.yml

name: Security Scanning Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    # Run weekly on Monday at 9 AM UTC
    - cron: '0 9 * * 1'

permissions:
  contents: read
  security-events: write  # For SARIF upload

jobs:
  # ─── Stage 1: SAST ───
  sast-bandit:
    name: "SAST: Bandit"
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install Bandit
        run: pip install bandit[toml]

      - name: Run Bandit
        run: |
          bandit -r src/ \
            -f sarif \
            -o bandit-results.sarif \
            --severity-level medium \
            --confidence-level medium \
            --exclude tests/
        continue-on-error: true

      - name: Upload SARIF
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: bandit-results.sarif

  sast-semgrep:
    name: "SAST: Semgrep"
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run Semgrep
        uses: semgrep/semgrep-action@v1
        with:
          config: >-
            p/python
            p/flask
            p/owasp-top-ten
            .semgrep/custom_rules.yaml
        env:
          SEMGREP_APP_TOKEN: ${{ secrets.SEMGREP_APP_TOKEN }}

  # ─── Stage 2: SCA ───
  sca-dependencies:
    name: "SCA: Dependency Check"
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pip-audit safety

      - name: Run pip-audit
        run: |
          pip-audit \
            -r requirements.txt \
            -f json \
            -o pip-audit-results.json \
            --desc
        continue-on-error: true

      - name: Run Safety
        run: |
          safety check \
            -r requirements.txt \
            --full-report \
            --output json > safety-results.json
        continue-on-error: true

      - name: Check for critical vulnerabilities
        run: |
          python -c "
          import json, sys
          with open('pip-audit-results.json') as f:
              data = json.load(f)
          vulns = []
          for dep in data.get('dependencies', []):
              vulns.extend(dep.get('vulns', []))
          if vulns:
              print(f'Found {len(vulns)} vulnerabilities!')
              for v in vulns:
                  print(f'  - {v[\"id\"]}: {v.get(\"description\", \"\")[:80]}')
              sys.exit(1)
          print('No vulnerabilities found.')
          "

  # ─── Stage 3: Secret Scanning ───
  secret-scanning:
    name: "Secret Scanning"
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for secret scanning

      - name: Run Gitleaks
        uses: gitleaks/gitleaks-action@v2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

      - name: Run TruffleHog
        uses: trufflesecurity/trufflehog@main
        with:
          extra_args: --only-verified

  # ─── Stage 4: Container Scanning ───
  container-scan:
    name: "Container Scan"
    runs-on: ubuntu-latest
    if: github.event_name == 'push'
    steps:
      - uses: actions/checkout@v4

      - name: Build Docker image
        run: docker build -t myapp:${{ github.sha }} .

      - name: Run Trivy
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: myapp:${{ github.sha }}
          format: 'sarif'
          output: 'trivy-results.sarif'
          severity: 'CRITICAL,HIGH'

      - name: Upload Trivy SARIF
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: trivy-results.sarif

  # ─── Stage 5: DAST (on staging) ───
  dast-zap:
    name: "DAST: ZAP Baseline"
    runs-on: ubuntu-latest
    needs: [sast-bandit, sast-semgrep, sca-dependencies]
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4

      - name: Start application
        run: |
          docker-compose up -d
          sleep 10  # Wait for app to start

      - name: Run ZAP Baseline Scan
        uses: zaproxy/action-baseline@v0.12.0
        with:
          target: 'http://localhost:8080'
          rules_file_name: '.zap/rules.tsv'
          cmd_options: '-a'

      - name: Stop application
        if: always()
        run: docker-compose down

  # ─── Security Gate ───
  security-gate:
    name: "Security Gate"
    runs-on: ubuntu-latest
    needs: [sast-bandit, sast-semgrep, sca-dependencies, secret-scanning]
    steps:
      - name: Check results
        run: |
          echo "All security checks passed!"
          echo "Review the Security tab for detailed findings."
```

### 8.2 Pre-commit Hooks for Security

```yaml
# .pre-commit-config.yaml

repos:
  # Bandit - Python security linter
  - repo: https://github.com/PyCQA/bandit
    rev: '1.7.7'
    hooks:
      - id: bandit
        args: ['-c', 'pyproject.toml', '-ll']
        additional_dependencies: ['bandit[toml]']

  # Detect secrets
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline']

  # Check for private keys
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: detect-private-key
      - id: check-added-large-files
        args: ['--maxkb=500']

  # Gitleaks
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.1
    hooks:
      - id: gitleaks
```

### 8.3 GitLab CI Security Pipeline

```yaml
# .gitlab-ci.yml

stages:
  - test
  - security
  - deploy

variables:
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.cache/pip"

# ─── SAST Stage ───
bandit-sast:
  stage: security
  image: python:3.12-slim
  script:
    - pip install bandit
    - bandit -r src/ -f json -o gl-sast-report.json --severity-level medium || true
  artifacts:
    reports:
      sast: gl-sast-report.json

semgrep-sast:
  stage: security
  image: semgrep/semgrep:latest
  script:
    - semgrep --config auto --sarif -o semgrep-results.sarif .
  artifacts:
    reports:
      sast: semgrep-results.sarif

# ─── Dependency Scanning ───
dependency-check:
  stage: security
  image: python:3.12-slim
  script:
    - pip install pip-audit
    - pip-audit -r requirements.txt --strict
  allow_failure: true

# ─── Secret Detection ───
secret-detection:
  stage: security
  image:
    name: zricethezav/gitleaks:latest
    entrypoint: [""]
  script:
    - gitleaks detect --source . --report-path gitleaks-report.json
  artifacts:
    reports:
      secret_detection: gitleaks-report.json
```

---

## 9. Comprehensive Security Scanning Script

```python
"""
security_scanner.py - Unified security scanning orchestrator.
Runs multiple security tools and generates a combined report.

Usage:
    python security_scanner.py --project-dir ./myproject
    python security_scanner.py --project-dir ./myproject --output report.json
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class Finding:
    """A single security finding from any tool."""
    tool: str
    severity: str           # CRITICAL, HIGH, MEDIUM, LOW, INFO
    category: str           # SAST, SCA, SECRET, CONFIG
    title: str
    description: str
    file: Optional[str] = None
    line: Optional[int] = None
    cwe: Optional[str] = None
    fix: Optional[str] = None


@dataclass
class ScanReport:
    """Combined report from all scanners."""
    project: str
    scan_date: str = ""
    scan_duration_seconds: float = 0.0
    findings: list[Finding] = field(default_factory=list)
    tools_run: list[str] = field(default_factory=list)
    tools_failed: list[str] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.scan_date:
            self.scan_date = datetime.now().isoformat()


class SecurityScanner:
    """Orchestrates multiple security scanning tools."""

    def __init__(self, project_dir: str):
        self.project_dir = Path(project_dir).resolve()
        self.report = ScanReport(project=str(self.project_dir))
        self.start_time = time.time()

    def run_all(self) -> ScanReport:
        """Run all available security scanners."""
        print(f"[*] Starting security scan of: {self.project_dir}")
        print(f"[*] Scan started at: {self.report.scan_date}")
        print()

        self._run_bandit()
        self._run_pip_audit()
        self._run_secret_check()
        self._check_security_configs()

        self.report.scan_duration_seconds = time.time() - self.start_time
        self._generate_summary()

        return self.report

    def _run_bandit(self) -> None:
        """Run Bandit SAST scanner."""
        print("[*] Running Bandit (SAST)...")
        try:
            result = subprocess.run(
                [
                    "bandit", "-r", str(self.project_dir),
                    "-f", "json",
                    "--severity-level", "low",
                    "-x", "tests,venv,.tox"
                ],
                capture_output=True, text=True, timeout=120
            )

            data = json.loads(result.stdout)
            for issue in data.get("results", []):
                self.report.findings.append(Finding(
                    tool="bandit",
                    severity=issue["issue_severity"].upper(),
                    category="SAST",
                    title=f"[{issue['test_id']}] {issue['test_name']}",
                    description=issue["issue_text"],
                    file=issue["filename"],
                    line=issue["line_number"],
                    cwe=issue.get("issue_cwe", {}).get("id"),
                ))

            self.report.tools_run.append("bandit")
            print(f"    Found {len(data.get('results', []))} issues")

        except FileNotFoundError:
            print("    [!] Bandit not installed")
            self.report.tools_failed.append("bandit")
        except Exception as e:
            print(f"    [!] Bandit failed: {e}")
            self.report.tools_failed.append("bandit")

    def _run_pip_audit(self) -> None:
        """Run pip-audit SCA scanner."""
        print("[*] Running pip-audit (SCA)...")

        req_file = self.project_dir / "requirements.txt"
        if not req_file.exists():
            print("    [!] No requirements.txt found, skipping")
            return

        try:
            result = subprocess.run(
                ["pip-audit", "-r", str(req_file), "-f", "json", "--desc"],
                capture_output=True, text=True, timeout=120
            )

            data = json.loads(result.stdout)
            vuln_count = 0
            for dep in data.get("dependencies", []):
                for vuln in dep.get("vulns", []):
                    vuln_count += 1
                    fix_versions = vuln.get("fix_versions", [])
                    self.report.findings.append(Finding(
                        tool="pip-audit",
                        severity="HIGH",
                        category="SCA",
                        title=f"{dep['name']} {dep['version']}: {vuln['id']}",
                        description=vuln.get("description", ""),
                        fix=f"Upgrade to {fix_versions[0]}"
                            if fix_versions else "No fix available",
                    ))

            self.report.tools_run.append("pip-audit")
            print(f"    Found {vuln_count} vulnerable dependencies")

        except FileNotFoundError:
            print("    [!] pip-audit not installed")
            self.report.tools_failed.append("pip-audit")
        except Exception as e:
            print(f"    [!] pip-audit failed: {e}")
            self.report.tools_failed.append("pip-audit")

    def _run_secret_check(self) -> None:
        """Check for hardcoded secrets in source files."""
        print("[*] Checking for hardcoded secrets...")
        import re

        secret_patterns = [
            (r'(?i)(api[_-]?key|apikey)\s*[:=]\s*["\'][a-zA-Z0-9]{20,}["\']',
             "Possible API key"),
            (r'(?i)(password|passwd|pwd)\s*[:=]\s*["\'][^"\']{6,}["\']',
             "Possible hardcoded password"),
            (r'(?i)(secret|token)\s*[:=]\s*["\'][a-zA-Z0-9+/=]{20,}["\']',
             "Possible hardcoded secret/token"),
            (r'-----BEGIN (?:RSA |DSA |EC )?PRIVATE KEY-----',
             "Private key detected"),
            (r'(?i)aws[_-]?(?:access[_-]?key[_-]?id|secret[_-]?access[_-]?key)\s*[:=]\s*["\']?[A-Z0-9]{16,}',
             "Possible AWS credential"),
        ]

        findings_count = 0
        for py_file in self.project_dir.rglob("*.py"):
            # Skip virtual environments and test fixtures
            rel_path = py_file.relative_to(self.project_dir)
            if any(part in str(rel_path) for part in
                   ['venv', '.tox', 'node_modules', '__pycache__']):
                continue

            try:
                content = py_file.read_text(errors='ignore')
                for line_num, line in enumerate(content.splitlines(), 1):
                    # Skip comments with nosec
                    if '# nosec' in line or '# noqa' in line:
                        continue
                    for pattern, description in secret_patterns:
                        if re.search(pattern, line):
                            findings_count += 1
                            self.report.findings.append(Finding(
                                tool="secret-scanner",
                                severity="HIGH",
                                category="SECRET",
                                title=description,
                                description=f"Potential secret found in source code",
                                file=str(rel_path),
                                line=line_num,
                                cwe="CWE-798",
                                fix="Move secrets to environment variables or "
                                    "a secret management service",
                            ))
            except Exception:
                pass

        self.report.tools_run.append("secret-scanner")
        print(f"    Found {findings_count} potential secrets")

    def _check_security_configs(self) -> None:
        """Check for security-related configuration issues."""
        print("[*] Checking security configurations...")

        findings_count = 0

        # Check for DEBUG mode in Flask/Django settings
        for py_file in self.project_dir.rglob("*.py"):
            rel_path = py_file.relative_to(self.project_dir)
            if any(part in str(rel_path) for part in ['venv', '.tox']):
                continue

            try:
                content = py_file.read_text(errors='ignore')

                # Flask DEBUG
                if 'app.run(debug=True)' in content:
                    findings_count += 1
                    self.report.findings.append(Finding(
                        tool="config-checker",
                        severity="MEDIUM",
                        category="CONFIG",
                        title="Flask debug mode enabled",
                        description="Debug mode should not be enabled in production",
                        file=str(rel_path),
                        fix="Use environment variable: "
                            "app.run(debug=os.getenv('FLASK_DEBUG', False))",
                    ))

                # Django DEBUG
                if 'DEBUG = True' in content and 'settings' in str(rel_path):
                    findings_count += 1
                    self.report.findings.append(Finding(
                        tool="config-checker",
                        severity="MEDIUM",
                        category="CONFIG",
                        title="Django DEBUG mode enabled in settings",
                        description="DEBUG should be False in production",
                        file=str(rel_path),
                        fix="Use: DEBUG = os.getenv('DJANGO_DEBUG', 'False') == 'True'",
                    ))

            except Exception:
                pass

        self.report.tools_run.append("config-checker")
        print(f"    Found {findings_count} configuration issues")

    def _generate_summary(self) -> None:
        """Generate summary statistics."""
        severity_counts = {
            'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'INFO': 0
        }
        category_counts = {
            'SAST': 0, 'SCA': 0, 'SECRET': 0, 'CONFIG': 0
        }

        for f in self.report.findings:
            severity_counts[f.severity] = severity_counts.get(f.severity, 0) + 1
            category_counts[f.category] = category_counts.get(f.category, 0) + 1

        self.report.summary = {
            'total_findings': len(self.report.findings),
            'by_severity': severity_counts,
            'by_category': category_counts,
            'tools_run': len(self.report.tools_run),
            'tools_failed': len(self.report.tools_failed),
        }


def print_report(report: ScanReport) -> None:
    """Print a formatted text report."""
    print("\n" + "=" * 65)
    print("  SECURITY SCAN REPORT")
    print("=" * 65)
    print(f"  Project:  {report.project}")
    print(f"  Date:     {report.scan_date}")
    print(f"  Duration: {report.scan_duration_seconds:.1f}s")
    print(f"  Tools:    {', '.join(report.tools_run)}")
    if report.tools_failed:
        print(f"  Failed:   {', '.join(report.tools_failed)}")

    s = report.summary
    print(f"\n  Total findings: {s['total_findings']}")
    print(f"  By severity: "
          f"CRITICAL={s['by_severity']['CRITICAL']} "
          f"HIGH={s['by_severity']['HIGH']} "
          f"MEDIUM={s['by_severity']['MEDIUM']} "
          f"LOW={s['by_severity']['LOW']}")
    print(f"  By category: "
          f"SAST={s['by_category']['SAST']} "
          f"SCA={s['by_category']['SCA']} "
          f"SECRET={s['by_category']['SECRET']} "
          f"CONFIG={s['by_category']['CONFIG']}")

    # Print findings grouped by severity
    for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO']:
        findings = [f for f in report.findings if f.severity == severity]
        if not findings:
            continue

        print(f"\n{'─' * 65}")
        print(f"  {severity} ({len(findings)} findings)")
        print(f"{'─' * 65}")

        for f in findings:
            print(f"\n  [{f.tool}] {f.title}")
            if f.file:
                loc = f"    File: {f.file}"
                if f.line:
                    loc += f":{f.line}"
                print(loc)
            print(f"    {f.description[:100]}")
            if f.fix:
                print(f"    Fix: {f.fix[:100]}")

    print(f"\n{'=' * 65}")
    if s['by_severity']['CRITICAL'] > 0 or s['by_severity']['HIGH'] > 0:
        print("  RESULT: FAIL - Critical/High issues found")
    elif s['total_findings'] > 0:
        print("  RESULT: WARN - Issues found, review recommended")
    else:
        print("  RESULT: PASS - No issues found")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Unified Security Scanner")
    parser.add_argument("--project-dir", required=True,
                        help="Path to project directory")
    parser.add_argument("--output", help="Output JSON report file")
    parser.add_argument("--fail-on", default="high",
                        choices=["critical", "high", "medium", "low"],
                        help="Severity level that causes non-zero exit")
    args = parser.parse_args()

    scanner = SecurityScanner(args.project_dir)
    report = scanner.run_all()

    # Print text report
    print_report(report)

    # Save JSON report
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(asdict(report), f, indent=2)
        print(f"\nJSON report saved to: {args.output}")

    # Exit code based on findings
    severity_order = ['low', 'medium', 'high', 'critical']
    threshold_idx = severity_order.index(args.fail_on)
    fail_severities = [s.upper() for s in severity_order[threshold_idx:]]

    for finding in report.findings:
        if finding.severity in fail_severities:
            sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
```

---

## 10. Exercises

### Exercise 1: Bandit Scan Analysis

Run Bandit on this intentionally vulnerable code and fix all findings:

```python
"""vulnerable_app.py - Fix all security issues found by Bandit."""
import os
import pickle
import hashlib
import subprocess
import sqlite3
from flask import Flask, request

app = Flask(__name__)
SECRET_KEY = "my-super-secret-key-12345"

@app.route('/search')
def search():
    query = request.args.get('q')
    conn = sqlite3.connect('app.db')
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM products WHERE name LIKE '%{query}%'")
    return str(cursor.fetchall())

@app.route('/run')
def run_command():
    cmd = request.args.get('cmd')
    result = subprocess.check_output(cmd, shell=True)
    return result

@app.route('/load')
def load_data():
    data = request.get_data()
    obj = pickle.loads(data)
    return str(obj)

@app.route('/hash')
def hash_password():
    password = request.args.get('pw')
    return hashlib.md5(password.encode()).hexdigest()

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
```

**Tasks:**
1. Run `bandit vulnerable_app.py` and document all findings
2. Fix each vulnerability
3. Run Bandit again to verify all issues are resolved

### Exercise 2: Write Custom Semgrep Rules

Write Semgrep rules that detect:
1. Use of `os.system()` with any string concatenation
2. Flask routes that accept POST but do not validate `Content-Type`
3. Any use of `eval()` or `exec()` within a function that handles HTTP requests
4. Hardcoded database connection strings

### Exercise 3: Dependency Audit

Create a `requirements.txt` with intentionally old, vulnerable packages:
```
flask==2.0.1
requests==2.25.1
django==3.2.0
pyyaml==5.3.1
pillow==8.0.0
```

1. Run `pip-audit -r requirements.txt` and document all CVEs found
2. Determine the minimum safe version for each package
3. Create a `requirements-secure.txt` with fixed versions

### Exercise 4: Property-Based Testing

Write Hypothesis tests for:
1. A password strength validator (must have uppercase, lowercase, digit, special char, min 8 chars)
2. A URL sanitizer that should prevent `javascript:` and `data:` schemes
3. An HTML tag stripper that should remove all HTML but preserve text content

### Exercise 5: CI/CD Security Pipeline

Design and implement a GitHub Actions workflow that:
1. Runs Bandit with SARIF output
2. Runs pip-audit on requirements.txt
3. Checks for secrets using gitleaks
4. Fails the pipeline if any HIGH or CRITICAL findings exist
5. Posts a comment on the PR with a summary of findings

### Exercise 6: Security Code Review

Review this code and identify all security issues:

```python
from flask import Flask, request, jsonify, redirect
import jwt
import sqlite3
import os

app = Flask(__name__)

@app.route('/api/users/<user_id>')
def get_user(user_id):
    db = sqlite3.connect('users.db')
    cursor = db.execute(
        f"SELECT id, name, email, ssn FROM users WHERE id = {user_id}"
    )
    user = cursor.fetchone()
    if user:
        return jsonify({
            'id': user[0], 'name': user[1],
            'email': user[2], 'ssn': user[3]
        })
    return jsonify({'error': f'User {user_id} not found'}), 404

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    db = sqlite3.connect('users.db')
    cursor = db.execute(
        f"SELECT * FROM users WHERE email = '{data['email']}' "
        f"AND password = '{data['password']}'"
    )
    user = cursor.fetchone()
    if user:
        token = jwt.encode(
            {'user_id': user[0], 'role': user[4]},
            'secret123',
            algorithm='HS256'
        )
        return jsonify({'token': token})
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/api/redirect')
def handle_redirect():
    url = request.args.get('url')
    return redirect(url)

@app.route('/api/upload', methods=['POST'])
def upload():
    f = request.files['file']
    f.save(os.path.join('/uploads', f.filename))
    return jsonify({'status': 'uploaded'})
```

Document at least 10 distinct security vulnerabilities and provide fixes for each one.

---

## Summary

```
┌─────────────────────────────────────────────────────────────────┐
│              Security Testing Key Takeaways                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Layer your defenses: Use SAST + SCA + DAST + fuzzing       │
│  2. Shift left: Find bugs as early as possible in SDLC         │
│  3. Automate: Integrate all tools into CI/CD pipeline          │
│  4. Custom rules: Write project-specific Semgrep rules         │
│  5. False positives: Manage them with baselines and #nosec     │
│  6. Dependencies: Scan and update regularly (Dependabot/SCA)   │
│  7. Code review: Security is a human + tool collaboration      │
│  8. Fuzzing: Finds bugs that other methods miss                │
│  9. Penetration testing: Validates all other findings          │
│ 10. Continuous: Security testing is ongoing, not one-time      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

**Previous**: [12. Container and Cloud Security](./12_Container_Security.md) | **Next**: [14. Incident Response and Forensics](14_Incident_Response.md)
