# 레슨 12: 보안 테스트 (Security Testing)

**이전**: [성능 테스트](./11_Performance_Testing.md) | **다음**: [CI/CD 통합](./13_CI_CD_Integration.md)

---

보안 취약점은 이론적인 우려가 아닙니다 — 매일 대규모로 악용되고 있습니다. 로그인 폼의 SQL 인젝션, GitHub에 푸시된 하드코딩된 API 키, 알려진 CVE가 있는 종속성 — 이 중 어느 하나라도 전체 시스템을 위험에 빠뜨릴 수 있습니다. 보안 테스트는 공격자보다 먼저 이러한 취약점을 체계적으로 찾는 실행 방법입니다. "이것이 작동하는가?"를 묻는 기능 테스트와 달리, 보안 테스트는 "이것이 악용될 수 있는가?"를 묻습니다.

이 레슨은 수동 침투 테스트나 레드팀 훈련이 아닌, 개발자의 워크플로우에 통합되는 자동화된 보안 테스트 도구에 집중합니다.

**난이도**: ⭐⭐⭐

**선수 조건**:
- Python 개발 경험 (레슨 01)
- pytest 및 CI/CD 개념에 대한 익숙함 (레슨 02, 13)
- 일반적인 보안 취약점(SQL 인젝션, XSS)에 대한 기본 이해

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. SAST 도구(Bandit)를 사용하여 Python 소스 코드에서 보안 문제를 찾을 수 있다
2. pip-audit와 Safety를 사용하여 Python 종속성의 알려진 취약점을 감사할 수 있다
3. detect-secrets와 TruffleHog를 사용하여 코드 저장소에서 하드코딩된 시크릿을 감지할 수 있다
4. DAST 개념과 적용 시점을 이해할 수 있다
5. 자동화된 보안 검사를 CI/CD 파이프라인에 통합할 수 있다

---

## 1. 보안 테스트 환경

보안 테스트는 완전 자동화부터 완전 수동까지 스펙트럼을 가집니다:

```
Automated ◄─────────────────────────────────────► Manual

SAST         Dependency    Secrets      DAST       Penetration
(Static)     Scanning      Detection    (Dynamic)  Testing
│            │             │            │          │
Bandit       pip-audit     detect-      OWASP      Human
Semgrep      Safety        secrets      ZAP        experts
             Trivy         TruffleHog   Burp Suite
```

이 레슨은 스펙트럼의 자동화 쪽을 다룹니다 — 단위 테스트와 함께 CI에서 실행할 수 있는 도구들입니다.

### 1.1 SAST vs DAST

| 측면 | SAST (정적) | DAST (동적) |
|---|---|---|
| 시점 | 소스 코드 분석 | 실행 중인 애플리케이션 테스트 |
| 발견 대상 | 코드 패턴, 안전하지 않은 호출 | 런타임 취약점 |
| 속도 | 빠름 (초~분) | 느림 (분~시간) |
| 오탐(false positive) | 높음 | 낮음 |
| 커버리지 | 모든 코드 경로 | 실행된 경로만 |
| 예시 | Bandit, Semgrep | OWASP ZAP, Burp Suite |

성숙한 보안 테스트 전략은 양쪽 모두를 사용합니다. SAST는 문제를 일찍, 저렴하게 잡고; DAST는 런타임에만 나타나는 문제를 잡습니다.

---

## 2. Bandit을 사용한 정적 애플리케이션 보안 테스트

[Bandit](https://bandit.readthedocs.io/)은 Python의 표준 SAST 도구입니다. 소스 코드의 추상 구문 트리(AST)를 분석하여 일반적인 보안 문제를 찾습니다.

### 2.1 설치 및 기본 사용법

```bash
pip install bandit

# Scan a single file
bandit myapp/views.py

# Scan an entire project (recursive)
bandit -r myapp/

# Scan with specific severity/confidence
bandit -r myapp/ -ll -ii  # Only high severity + high confidence
```

### 2.2 Bandit이 감지하는 것

Bandit은 범주별로 구성된 30개 이상의 내장 검사를 갖추고 있습니다:

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

### 2.3 설정

`.bandit` 또는 `pyproject.toml` 설정을 생성합니다:

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

### 2.4 오탐 처리

```python
import subprocess

# Bandit flags subprocess usage. If you've validated the input,
# suppress with a nosec comment explaining WHY it's safe:
result = subprocess.run(
    ["git", "log", "--oneline", "-5"],  # nosec B603 — hardcoded safe command
    capture_output=True, text=True
)
```

항상 `nosec`에 정당한 이유를 포함합니다. `nosec`만 단독으로 사용하면 근거가 숨겨지고 기술 부채가 됩니다.

### 2.5 Bandit을 pytest와 통합하기

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

## 3. 종속성 취약점 스캐닝

작성한 코드는 안전할 수 있지만, 종속성은 그렇지 않을 수 있습니다. 평균적인 Python 프로젝트는 수십 개의 전이적 종속성을 가지며, 이 중 어느 것이든 알려진 취약점(CVE)을 포함할 수 있습니다.

### 3.1 pip-audit

[pip-audit](https://github.com/pypa/pip-audit)는 설치된 패키지를 Python Packaging Advisory Database 및 OSV와 대조하여 확인합니다:

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

출력 예시:

```
Name        Version  ID                  Fix Versions
----------  -------  ------------------  ------------
requests    2.25.0   PYSEC-2023-74       2.31.0
cryptography 3.4.6   CVE-2023-38325      41.0.2
```

### 3.2 Safety

[Safety](https://safetycli.com/)는 대안적인 종속성 스캐너입니다:

```bash
pip install safety

# Check installed packages
safety check

# Check a requirements file
safety check -r requirements.txt

# JSON output
safety check --output json
```

### 3.3 종속성 검사 자동화

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

## 4. 시크릿 감지

하드코딩된 시크릿 — API 키, 패스워드, 토큰 — 은 가장 흔하고 피해가 큰 보안 실수 중 하나입니다. 시크릿이 한번 저장소에 푸시되면, 나중에 커밋이 제거되더라도(Git 히스토리에 남아 있으므로) 유출된 것으로 간주해야 합니다.

### 4.1 detect-secrets

[detect-secrets](https://github.com/Yelp/detect-secrets)는 Yelp가 만든 도구로 휴리스틱과 엔트로피 분석을 사용하여 시크릿을 찾습니다:

```bash
pip install detect-secrets

# Generate a baseline (initial scan)
detect-secrets scan > .secrets.baseline

# Audit the baseline (mark false positives interactively)
detect-secrets audit .secrets.baseline

# Check for new secrets (CI mode)
detect-secrets scan --baseline .secrets.baseline
```

### 4.2 detect-secrets를 pre-commit 훅으로 사용하기

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

[TruffleHog](https://github.com/trufflesecurity/trufflehog)는 현재 상태뿐만 아니라 Git 히스토리를 스캔합니다:

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

### 4.4 감지 대상

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

### 4.5 시크릿 관리 모범 사례

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

항상 시크릿 파일을 `.gitignore`에 추가합니다:

```gitignore
# .gitignore
.env
*.pem
*.key
config/secrets.yaml
config/secrets.json
```

---

## 5. DAST: 동적 애플리케이션 보안 테스트

DAST 도구는 악의적인 요청을 보내고 응답을 관찰하여 실행 중인 애플리케이션을 테스트합니다. 정적 분석으로 찾을 수 없는 취약점 — 설정 오류, 인증 결함, 런타임에만 나타나는 인젝션 취약점 등을 발견합니다.

### 5.1 OWASP ZAP

[OWASP ZAP](https://www.zaproxy.org/) (Zed Attack Proxy)는 가장 널리 사용되는 오픈소스 DAST 도구입니다. 브라우저와 애플리케이션 사이에서 프록시 역할을 합니다:

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

### 5.2 ZAP을 Python 테스트와 통합하기

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

## 6. CI에 보안 검사 통합하기

가장 효과적인 보안 테스트는 자동화되고 지속적입니다. 종합적인 CI 설정은 다음과 같습니다:

### 6.1 GitHub Actions 보안 워크플로우

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

### 6.2 Pre-commit 통합

코드가 커밋되기 전에 보안 검사를 실행합니다:

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

## 7. 보안 테스트 전략 수립

### 7.1 보안 테스트 피라미드

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

아래 계층은 매 커밋마다 실행되며 완전히 자동화됩니다. 최상위 계층은 비용이 높고 빈도가 낮지만 자동화로 잡을 수 없는 문제를 발견합니다.

### 7.2 발견 사항 우선순위 정하기

모든 보안 발견 사항이 동일하지는 않습니다. 다음 기준으로 우선순위를 정합니다:

1. **심각도**: 악용 시 얼마나 큰 피해가 발생하는가? (치명적 > 높음 > 중간 > 낮음)
2. **악용 가능성**: 얼마나 쉽게 악용할 수 있는가? (네트워크 접근 가능 > 인증 필요)
3. **신뢰도**: 실제 문제인가 오탐인가?
4. **노출도**: 영향받는 코드가 공개 엔드포인트에 있는가?

로그인 엔드포인트의 치명적 SQL 인젝션은 CLI 스크립트에서의 낮은 심각도 `assert` 사용보다 더 긴급합니다.

---

## 연습 문제

1. **Bandit 스캔**: 작성한 Python 프로젝트에 Bandit을 실행합니다. 발견된 높은 심각도 문제를 수정합니다. 추가하는 각 `nosec` 억제에 대해 해당 발견이 오탐인 이유를 문서화합니다.

2. **종속성 감사**: 최소 10개의 종속성이 있는 프로젝트에 `pip-audit`를 실행합니다. 발견된 취약점을 해결하기 위한 계획(업그레이드, 대체, 또는 정당한 이유와 함께 위험 수용)을 수립합니다.

3. **시크릿 베이스라인**: 저장소에 `detect-secrets`를 설정합니다. 베이스라인을 생성하고, 감사하고, pre-commit 훅으로 설정합니다. 의도적으로 심어둔 가짜 API 키를 감지하는지 테스트합니다.

4. **CI 보안 파이프라인**: 모든 풀 리퀘스트에서 Bandit, pip-audit, detect-secrets를 실행하는 GitHub Actions 워크플로우를 생성합니다. 어떤 도구라도 문제를 발견하면 PR이 실패해야 합니다.

5. **취약점 분석**: 코드베이스에서 `B608: SQL injection via string formatting`이라는 Bandit 발견이 주어졌을 때, 취약점을 시연하는 테스트를 작성한 다음, 매개변수화된 쿼리를 사용하여 코드를 수정하고 테스트가 통과하는지 확인합니다.

---

**License**: CC BY-NC 4.0
