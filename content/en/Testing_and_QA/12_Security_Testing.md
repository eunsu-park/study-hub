# Lesson 12: Security Testing

**Previous**: [Performance Testing](./11_Performance_Testing.md) | **Next**: [CI/CD Integration](./13_CI_CD_Integration.md)

---

Security vulnerabilities are not theoretical concerns — they are exploited daily at scale. A SQL injection in a login form, a hardcoded API key pushed to GitHub, a dependency with a known CVE — any of these can compromise an entire system. Security testing is the practice of systematically finding these vulnerabilities before attackers do. Unlike functional testing, which asks "does this work?", security testing asks "can this be abused?"

This lesson focuses on automated security testing tools that integrate into a developer's workflow, not on manual penetration testing or red-team exercises.

**Difficulty**: ⭐⭐⭐

**Prerequisites**:
- Python development experience (Lesson 01)
- Familiarity with pytest and CI/CD concepts (Lessons 02, 13)
- Basic understanding of common security vulnerabilities (SQL injection, XSS)

## Learning Objectives

After completing this lesson, you will be able to:

1. Use SAST tools (Bandit) to find security issues in Python source code
2. Audit Python dependencies for known vulnerabilities using pip-audit and Safety
3. Detect hardcoded secrets in code repositories using detect-secrets and TruffleHog
4. Understand DAST concepts and when to apply them
5. Integrate automated security checks into CI/CD pipelines

---

## 1. The Security Testing Landscape

Security testing spans a spectrum from fully automated to fully manual:

```
Automated ◄─────────────────────────────────────► Manual

SAST         Dependency    Secrets      DAST       Penetration
(Static)     Scanning      Detection    (Dynamic)  Testing
│            │             │            │          │
Bandit       pip-audit     detect-      OWASP      Human
Semgrep      Safety        secrets      ZAP        experts
             Trivy         TruffleHog   Burp Suite
```

This lesson covers the automated end of the spectrum — tools you can run in CI alongside your unit tests.

### 1.1 SAST vs DAST

| Aspect | SAST (Static) | DAST (Dynamic) |
|---|---|---|
| When | Analyzes source code | Tests running application |
| Finds | Code patterns, unsafe calls | Runtime vulnerabilities |
| Speed | Fast (seconds to minutes) | Slower (minutes to hours) |
| False positives | Higher | Lower |
| Coverage | All code paths | Only exercised paths |
| Examples | Bandit, Semgrep | OWASP ZAP, Burp Suite |

A mature security testing strategy uses both. SAST catches issues early and cheaply; DAST catches issues that only manifest at runtime.

---

## 2. Static Application Security Testing with Bandit

[Bandit](https://bandit.readthedocs.io/) is the standard SAST tool for Python. It analyzes your source code's abstract syntax tree (AST) to find common security issues.

### 2.1 Installation and Basic Usage

```bash
pip install bandit

# Scan a single file
bandit myapp/views.py

# Scan an entire project (recursive)
bandit -r myapp/

# Scan with specific severity/confidence
bandit -r myapp/ -ll -ii  # Only high severity + high confidence
```

### 2.2 What Bandit Detects

Bandit has over 30 built-in checks organized by category:

```python
# B101: Use of assert (stripped in optimized bytecode)
def verify_admin(user):
    assert user.is_admin  # INSECURE: assert is removed with -O flag
    return sensitive_data()

# B102: Use of exec()
def process_template(template_str, context):
    exec(template_str)  # INSECURE: arbitrary code execution

# B301: Use of pickle (arbitrary code execution on deserialization)
import pickle
data = pickle.loads(user_input)  # INSECURE

# B303: Use of insecure hash functions
import hashlib
password_hash = hashlib.md5(password.encode()).hexdigest()  # INSECURE

# B608: SQL injection via string formatting
def get_user(username):
    query = f"SELECT * FROM users WHERE name = '{username}'"  # INSECURE
    cursor.execute(query)

# B105: Hardcoded password
DB_PASSWORD = "supersecret123"  # INSECURE
```

### 2.3 Configuration

Create `.bandit` or `pyproject.toml` configuration:

```toml
# pyproject.toml
[tool.bandit]
exclude_dirs = ["tests", "venv"]
skips = ["B101"]  # Skip assert warnings (acceptable in tests)
targets = ["myapp"]

# Per-test configuration
[tool.bandit.assert_used]
skips = ["*/test_*.py", "*/conftest.py"]
```

### 2.4 Handling False Positives

```python
import subprocess

# Bandit flags subprocess usage. If you've validated the input,
# suppress with a nosec comment explaining WHY it's safe:
result = subprocess.run(
    ["git", "log", "--oneline", "-5"],  # nosec B603 — hardcoded safe command
    capture_output=True, text=True
)
```

Always include a justification with `nosec`. A bare `nosec` hides the reasoning and becomes technical debt.

### 2.5 Integrating Bandit with pytest

```python
# tests/test_security.py
import subprocess
import sys


def test_bandit_finds_no_issues():
    """Run Bandit as part of the test suite."""
    result = subprocess.run(
        [
            sys.executable, "-m", "bandit",
            "-r", "myapp/",
            "-ll",  # Medium+ severity
            "-ii",  # Medium+ confidence
            "-f", "json",
            "--exit-zero"
        ],
        capture_output=True, text=True
    )
    import json
    report = json.loads(result.stdout)
    issues = report.get("results", [])

    if issues:
        messages = []
        for issue in issues:
            messages.append(
                f"{issue['filename']}:{issue['line_number']} "
                f"[{issue['test_id']}] {issue['issue_text']}"
            )
        raise AssertionError(
            f"Bandit found {len(issues)} security issues:\n"
            + "\n".join(messages)
        )
```

---

## 3. Dependency Vulnerability Scanning

Your code might be secure, but your dependencies might not be. The average Python project has dozens of transitive dependencies, any of which could contain known vulnerabilities (CVEs).

### 3.1 pip-audit

[pip-audit](https://github.com/pypa/pip-audit) checks installed packages against the Python Packaging Advisory Database and OSV:

```bash
pip install pip-audit

# Scan installed packages
pip-audit

# Scan a requirements file
pip-audit -r requirements.txt

# JSON output for CI processing
pip-audit -f json -o audit-results.json

# Fix vulnerabilities automatically
pip-audit --fix
```

Example output:

```
Name        Version  ID                  Fix Versions
----------  -------  ------------------  ------------
requests    2.25.0   PYSEC-2023-74       2.31.0
cryptography 3.4.6   CVE-2023-38325      41.0.2
```

### 3.2 Safety

[Safety](https://safetycli.com/) is an alternative dependency scanner:

```bash
pip install safety

# Check installed packages
safety check

# Check a requirements file
safety check -r requirements.txt

# JSON output
safety check --output json
```

### 3.3 Automating Dependency Checks

```python
# tests/test_dependencies.py
import subprocess
import json


def test_no_vulnerable_dependencies():
    """Ensure no installed packages have known vulnerabilities."""
    result = subprocess.run(
        ["pip-audit", "-f", "json", "--progress-spinner=off"],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        try:
            data = json.loads(result.stdout)
            vulnerabilities = data.get("dependencies", [])
            vuln_list = []
            for dep in vulnerabilities:
                for vuln in dep.get("vulns", []):
                    vuln_list.append(
                        f"  {dep['name']}=={dep['version']}: "
                        f"{vuln['id']} (fix: {vuln.get('fix_versions', 'N/A')})"
                    )
            raise AssertionError(
                f"Found {len(vuln_list)} vulnerable dependencies:\n"
                + "\n".join(vuln_list)
            )
        except json.JSONDecodeError:
            raise AssertionError(
                f"pip-audit failed:\n{result.stderr}"
            )
```

---

## 4. Secrets Detection

Hardcoded secrets — API keys, passwords, tokens — are one of the most common and damaging security mistakes. Once a secret is pushed to a repository, it must be considered compromised, even if the commit is later removed (it remains in Git history).

### 4.1 detect-secrets

[detect-secrets](https://github.com/Yelp/detect-secrets) by Yelp uses heuristics and entropy analysis to find secrets:

```bash
pip install detect-secrets

# Generate a baseline (initial scan)
detect-secrets scan > .secrets.baseline

# Audit the baseline (mark false positives interactively)
detect-secrets audit .secrets.baseline

# Check for new secrets (CI mode)
detect-secrets scan --baseline .secrets.baseline
```

### 4.2 Using detect-secrets as a Pre-commit Hook

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline']
```

### 4.3 TruffleHog

[TruffleHog](https://github.com/trufflesecurity/trufflehog) scans Git history, not just the current state:

```bash
# Install
pip install trufflehog

# Or use the Go binary (faster)
# brew install trufflehog

# Scan a local repository's entire history
trufflehog git file://. --only-verified

# Scan a GitHub repository
trufflehog github --repo=https://github.com/org/repo

# Scan only recent commits
trufflehog git file://. --since-commit=HEAD~10
```

### 4.4 What Gets Detected

```python
# These patterns trigger secrets detection:

# AWS credentials
AWS_ACCESS_KEY_ID = "AKIAIOSFODNN7EXAMPLE"        # DETECTED
AWS_SECRET_ACCESS_KEY = "wJalrXUtnFEMI/K7MDENG..."  # DETECTED

# API keys
STRIPE_KEY = "sk_live_EXAMPLE_KEY_DO_NOT_USE"       # DETECTED
GITHUB_TOKEN = "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"   # DETECTED

# Database URLs
DATABASE_URL = "postgresql://user:pass@host/db"     # DETECTED

# Private keys
PRIVATE_KEY = "-----BEGIN RSA PRIVATE KEY-----..."  # DETECTED
```

### 4.5 Best Practices for Secret Management

```python
# WRONG: Hardcoded secret
API_KEY = "sk-1234567890abcdef"

# RIGHT: Environment variable
import os
API_KEY = os.environ["API_KEY"]

# RIGHT: Configuration file (excluded from Git)
# .gitignore: config/secrets.yaml
import yaml
with open("config/secrets.yaml") as f:
    secrets = yaml.safe_load(f)

# RIGHT: Secrets manager (production)
import boto3
client = boto3.client("secretsmanager")
secret = client.get_secret_value(SecretId="myapp/api-key")
```

Always add secrets files to `.gitignore`:

```gitignore
# .gitignore
.env
*.pem
*.key
config/secrets.yaml
config/secrets.json
```

---

## 5. DAST: Dynamic Application Security Testing

DAST tools test a running application by sending malicious requests and observing responses. They find vulnerabilities that static analysis cannot — like misconfigurations, authentication flaws, and injection vulnerabilities that only manifest at runtime.

### 5.1 OWASP ZAP

[OWASP ZAP](https://www.zaproxy.org/) (Zed Attack Proxy) is the most widely used open-source DAST tool. It acts as a proxy between your browser and the application:

```bash
# Run ZAP in daemon mode for automated scanning
docker run -t ghcr.io/zaproxy/zaproxy:stable zap-baseline.py \
    -t http://your-app:8000 \
    -r report.html

# Full scan (more thorough, slower)
docker run -t ghcr.io/zaproxy/zaproxy:stable zap-full-scan.py \
    -t http://your-app:8000 \
    -r full-report.html
```

### 5.2 Integrating ZAP with Python Tests

```python
# tests/test_dast.py
import subprocess

import pytest


@pytest.mark.dast
def test_zap_baseline_scan():
    """Run OWASP ZAP baseline scan against the staging application."""
    result = subprocess.run(
        [
            "docker", "run", "--rm", "--network=host",
            "ghcr.io/zaproxy/zaproxy:stable",
            "zap-baseline.py",
            "-t", "http://localhost:8000",
            "-J", "zap-report.json"
        ],
        capture_output=True, text=True, timeout=300
    )
    # ZAP returns 0 for pass, 1 for warnings, 2 for failures
    assert result.returncode < 2, (
        f"ZAP found security issues:\n{result.stdout}"
    )
```

---

## 6. Integrating Security Checks in CI

The most effective security testing is automated and continuous. Here is a comprehensive CI configuration:

### 6.1 GitHub Actions Security Workflow

```yaml
# .github/workflows/security.yml
name: Security Checks

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  sast:
    name: Static Analysis (Bandit)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install Bandit
        run: pip install bandit

      - name: Run Bandit
        run: bandit -r myapp/ -ll -ii -f json -o bandit-report.json

      - name: Upload report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: bandit-report
          path: bandit-report.json

  dependency-scan:
    name: Dependency Vulnerabilities
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
          pip install pip-audit

      - name: Run pip-audit
        run: pip-audit --strict --desc

  secrets-scan:
    name: Secrets Detection
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for TruffleHog

      - name: Run TruffleHog
        uses: trufflesecurity/trufflehog@main
        with:
          extra_args: --only-verified
```

### 6.2 Pre-commit Integration

Run security checks before code is even committed:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/PyCQA/bandit
    rev: '1.7.7'
    hooks:
      - id: bandit
        args: ['-ll', '-ii']
        exclude: tests/

  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline']
```

---

## 7. Building a Security Testing Strategy

### 7.1 The Security Testing Pyramid

```
          ┌─────────┐
          │  Manual  │  ← Penetration testing (quarterly)
          │ Pen Test │
         ┌┴─────────┴┐
         │   DAST     │  ← Against staging (weekly/nightly)
         │  (ZAP)     │
        ┌┴────────────┴┐
        │  Dependency   │  ← Every build
        │  Scanning     │
       ┌┴──────────────┴┐
       │   SAST + Secrets │  ← Every commit
       │   (Bandit + etc) │
       └─────────────────┘
```

The bottom layers run on every commit and are fully automated. The top layer is expensive and infrequent but catches issues that automation cannot.

### 7.2 Prioritizing Findings

Not all security findings are equal. Prioritize by:

1. **Severity**: How much damage if exploited? (Critical > High > Medium > Low)
2. **Exploitability**: How easy to exploit? (Network-accessible > requires authentication)
3. **Confidence**: Is this a real issue or a false positive?
4. **Exposure**: Is the affected code in a public-facing endpoint?

A critical SQL injection in a login endpoint is more urgent than a low-severity use of `assert` in a CLI script.

---

## Exercises

1. **Bandit Scan**: Run Bandit against a Python project you have written. Fix any high-severity issues it finds. For each `nosec` suppression you add, document why the finding is a false positive.

2. **Dependency Audit**: Run `pip-audit` against a project with at least 10 dependencies. Create a plan to address any vulnerabilities found (upgrade, replace, or accept risk with justification).

3. **Secrets Baseline**: Set up `detect-secrets` on a repository. Generate a baseline, audit it, and configure it as a pre-commit hook. Test that it catches a deliberately planted fake API key.

4. **CI Security Pipeline**: Create a GitHub Actions workflow that runs Bandit, pip-audit, and detect-secrets on every pull request. The workflow should fail the PR if any tool finds issues.

5. **Vulnerability Analysis**: Given the Bandit finding `B608: SQL injection via string formatting` in a codebase, write a test that demonstrates the vulnerability, then fix the code using parameterized queries and verify the test passes.

---

**License**: CC BY-NC 4.0
