# Web Security Headers and CSP

**Previous**: [08. Injection Attacks and Prevention](./08_Injection_Attacks.md) | **Next**: [10_API_Security.md](./10_API_Security.md)

---

HTTP security headers are the first line of defense for web applications. They instruct browsers to enforce security policies that mitigate entire classes of attacks — from cross-site scripting and clickjacking to protocol downgrade attacks and data exfiltration. A single missing header can leave an otherwise well-coded application vulnerable. This lesson provides a comprehensive guide to every major security header, with practical configuration examples for Flask and Django.

## Learning Objectives

- Understand the purpose and directives of Content-Security-Policy (CSP)
- Configure HSTS to enforce HTTPS connections
- Apply X-Content-Type-Options, X-Frame-Options, and Referrer-Policy headers
- Implement Permissions-Policy to restrict browser features
- Configure Cross-Origin policies (CORP, COEP, COOP)
- Use Subresource Integrity (SRI) to verify external resources
- Set up security headers in Flask and Django applications
- Test and audit headers using command-line tools and scanners

---

## 1. Security Headers Overview

### 1.1 Why Security Headers Matter

```
┌─────────────────────────────────────────────────────────────────────┐
│                  HTTP Response Security Headers                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Server ──── HTTP Response ────▶ Browser                             │
│              │                                                       │
│              ├── Content-Security-Policy                              │
│              ├── Strict-Transport-Security                            │
│              ├── X-Content-Type-Options                               │
│              ├── X-Frame-Options                                      │
│              ├── Referrer-Policy                                      │
│              ├── Permissions-Policy                                   │
│              ├── Cross-Origin-Resource-Policy                         │
│              ├── Cross-Origin-Embedder-Policy                         │
│              └── Cross-Origin-Opener-Policy                           │
│                                                                      │
│  These headers instruct the browser to:                              │
│  • Block inline scripts and unauthorized resources (CSP)             │
│  • Always use HTTPS (HSTS)                                           │
│  • Prevent MIME-type sniffing (X-Content-Type-Options)               │
│  • Block framing / clickjacking (X-Frame-Options)                    │
│  • Control referrer information leakage (Referrer-Policy)            │
│  • Restrict access to browser APIs (Permissions-Policy)              │
│  • Isolate cross-origin resources (CORP/COEP/COOP)                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Defense in Depth

Security headers are not a replacement for secure coding practices — they are an additional layer. Even with a perfectly coded application, security headers provide protection against:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Defense in Depth Layers                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Layer 5:  Security Headers          (browser-enforced policies)     │
│  Layer 4:  Application Logic         (input validation, auth)        │
│  Layer 3:  Framework Protections     (CSRF tokens, ORM)              │
│  Layer 2:  Network Security          (TLS, firewalls, WAF)           │
│  Layer 1:  Infrastructure            (OS hardening, patching)        │
│                                                                      │
│  Each layer catches what the layer below might miss.                 │
│  Security headers catch what application code might miss.            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 Quick Reference Table

| Header | Mitigates | Typical Value |
|--------|-----------|---------------|
| Content-Security-Policy | XSS, data injection | `default-src 'self'` |
| Strict-Transport-Security | Protocol downgrade, cookie hijacking | `max-age=31536000; includeSubDomains` |
| X-Content-Type-Options | MIME-type confusion | `nosniff` |
| X-Frame-Options | Clickjacking | `DENY` |
| Referrer-Policy | Information leakage | `strict-origin-when-cross-origin` |
| Permissions-Policy | Unauthorized API access | `camera=(), microphone=()` |
| Cross-Origin-Resource-Policy | Cross-origin data leaks | `same-origin` |
| Cross-Origin-Embedder-Policy | Spectre-style attacks | `require-corp` |
| Cross-Origin-Opener-Policy | Cross-window attacks | `same-origin` |

---

## 2. Content-Security-Policy (CSP)

### 2.1 What is CSP?

Content-Security-Policy is arguably the most powerful security header. It defines an allowlist of content sources that the browser should trust, effectively preventing XSS attacks by blocking unauthorized script execution.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CSP Enforcement Model                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Browser receives CSP header:                                        │
│    Content-Security-Policy: default-src 'self'; script-src 'self'    │
│                                                                      │
│  ┌──────────────────┐     ┌──────────────────┐                       │
│  │ <script src=      │     │ <script>          │                      │
│  │  "/app.js">       │     │  alert('XSS')     │                      │
│  │                   │     │ </script>          │                      │
│  │  Source: self      │     │  Source: inline    │                      │
│  │  ✓ ALLOWED        │     │  ✗ BLOCKED         │                      │
│  └──────────────────┘     └──────────────────┘                       │
│                                                                      │
│  ┌──────────────────┐     ┌──────────────────┐                       │
│  │ <script src=      │     │ <img src=          │                      │
│  │  "https://cdn     │     │  "/logo.png">      │                      │
│  │  .example.com     │     │                    │                      │
│  │  /lib.js">        │     │  Source: self       │                      │
│  │  Source: cdn       │     │  ✓ ALLOWED         │                      │
│  │  ✗ BLOCKED        │     │                    │                      │
│  └──────────────────┘     └──────────────────┘                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 CSP Directives

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CSP Directive Categories                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Fetch Directives (control where resources load from):               │
│  ├── default-src     Fallback for all other fetch directives         │
│  ├── script-src      JavaScript sources                              │
│  ├── style-src       CSS sources                                     │
│  ├── img-src         Image sources                                   │
│  ├── font-src        Web font sources                                │
│  ├── connect-src     XHR, fetch, WebSocket, EventSource              │
│  ├── media-src       Audio/video sources                             │
│  ├── object-src      <object>, <embed>, <applet>                     │
│  ├── child-src       Web workers and nested contexts                 │
│  ├── worker-src      Worker, SharedWorker, ServiceWorker             │
│  └── manifest-src    App manifest                                    │
│                                                                      │
│  Document Directives:                                                │
│  ├── base-uri        Restrict <base> element                         │
│  ├── sandbox         Apply sandbox restrictions                      │
│  └── plugin-types    Restrict plugin MIME types (deprecated)         │
│                                                                      │
│  Navigation Directives:                                              │
│  ├── form-action     Restrict form submission targets                │
│  ├── frame-ancestors Restrict who can embed this page                │
│  └── navigate-to     Restrict navigation targets (limited support)   │
│                                                                      │
│  Reporting Directives:                                               │
│  ├── report-uri      Send violation reports (deprecated)             │
│  └── report-to       Send violation reports (modern)                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.3 Source Values

| Source | Meaning | Example |
|--------|---------|---------|
| `'self'` | Same origin (scheme + host + port) | `script-src 'self'` |
| `'none'` | Block all sources | `object-src 'none'` |
| `'unsafe-inline'` | Allow inline scripts/styles | `style-src 'unsafe-inline'` |
| `'unsafe-eval'` | Allow eval(), Function(), etc. | `script-src 'unsafe-eval'` |
| `'strict-dynamic'` | Trust scripts loaded by trusted scripts | `script-src 'strict-dynamic'` |
| `'nonce-<base64>'` | Allow specific inline by nonce | `script-src 'nonce-abc123'` |
| `'sha256-<hash>'` | Allow specific inline by hash | `script-src 'sha256-...'` |
| `https:` | Any HTTPS source | `img-src https:` |
| `data:` | Data URIs | `img-src data:` |
| `blob:` | Blob URIs | `worker-src blob:` |
| `*.example.com` | Wildcard subdomain | `script-src *.cdn.com` |

### 2.4 Building a CSP Policy Step by Step

```python
"""
Building CSP policies progressively — from permissive to strict.
"""

# ── Level 1: Basic CSP (blocks obvious XSS) ──────────────────────
# Start with default-src 'self' and loosen as needed
csp_level1 = "default-src 'self'"

# ── Level 2: Allow specific external resources ───────────────────
csp_level2 = (
    "default-src 'self'; "
    "script-src 'self' https://cdn.jsdelivr.net; "
    "style-src 'self' https://fonts.googleapis.com; "
    "font-src 'self' https://fonts.gstatic.com; "
    "img-src 'self' data: https:; "
    "connect-src 'self' https://api.example.com; "
    "object-src 'none'; "
    "base-uri 'self'; "
    "form-action 'self'"
)

# ── Level 3: Nonce-based CSP (recommended for modern apps) ──────
# Server generates a random nonce per request
import secrets

def generate_csp_with_nonce():
    nonce = secrets.token_urlsafe(32)
    csp = (
        f"default-src 'self'; "
        f"script-src 'self' 'nonce-{nonce}'; "
        f"style-src 'self' 'nonce-{nonce}'; "
        f"img-src 'self' data:; "
        f"font-src 'self'; "
        f"connect-src 'self'; "
        f"object-src 'none'; "
        f"base-uri 'self'; "
        f"form-action 'self'; "
        f"frame-ancestors 'none'"
    )
    return csp, nonce

csp_header, nonce = generate_csp_with_nonce()
# In HTML: <script nonce="<nonce>">...</script>

# ── Level 4: Hash-based CSP (for static inline scripts) ─────────
import hashlib
import base64

def compute_csp_hash(script_content: str) -> str:
    """Compute SHA-256 hash for CSP inline script allowlisting."""
    digest = hashlib.sha256(script_content.encode('utf-8')).digest()
    b64 = base64.b64encode(digest).decode('utf-8')
    return f"'sha256-{b64}'"

inline_script = "console.log('Hello, World!');"
script_hash = compute_csp_hash(inline_script)
print(f"CSP hash: {script_hash}")
# Output: CSP hash: 'sha256-TWupyvVdPa1DyFqLnQMqRpuUWdS3nKPnz70IcS/1o3Q='

csp_level4 = f"script-src 'self' {script_hash}"

# ── Level 5: strict-dynamic (for apps that load scripts dynamically)
csp_level5 = (
    "script-src 'strict-dynamic' 'nonce-{nonce}'; "
    "object-src 'none'; "
    "base-uri 'self'"
)
# With strict-dynamic, scripts loaded by a nonced script
# are automatically trusted, regardless of their origin.
```

### 2.5 CSP Reporting

```python
"""
CSP violation reporting — detect policy violations without blocking.
"""

# ── Report-Only mode (monitor without enforcement) ───────────────
# Use Content-Security-Policy-Report-Only header first
csp_report_only = (
    "default-src 'self'; "
    "script-src 'self'; "
    "report-uri /csp-report; "
    "report-to csp-endpoint"
)

# The Report-To header (companion for report-to directive)
report_to = {
    "group": "csp-endpoint",
    "max_age": 86400,
    "endpoints": [
        {"url": "https://example.com/csp-report"}
    ]
}

# ── Flask endpoint to receive CSP violation reports ──────────────
from flask import Flask, request, jsonify
import json
import logging

app = Flask(__name__)
logger = logging.getLogger('csp_reports')

@app.route('/csp-report', methods=['POST'])
def csp_report():
    """Receive and log CSP violation reports."""
    try:
        # CSP reports come as application/csp-report
        report = request.get_json(force=True)
        violation = report.get('csp-report', {})

        logger.warning(
            "CSP Violation: blocked_uri=%s, "
            "violated_directive=%s, "
            "document_uri=%s, "
            "source_file=%s, "
            "line_number=%s",
            violation.get('blocked-uri', 'N/A'),
            violation.get('violated-directive', 'N/A'),
            violation.get('document-uri', 'N/A'),
            violation.get('source-file', 'N/A'),
            violation.get('line-number', 'N/A'),
        )

        return jsonify({"status": "received"}), 204
    except Exception as e:
        logger.error(f"Error processing CSP report: {e}")
        return jsonify({"error": "invalid report"}), 400

# ── Example CSP violation report JSON ────────────────────────────
example_report = {
    "csp-report": {
        "document-uri": "https://example.com/page",
        "referrer": "",
        "violated-directive": "script-src 'self'",
        "effective-directive": "script-src",
        "original-policy": "default-src 'self'; script-src 'self'",
        "blocked-uri": "https://evil.com/malicious.js",
        "status-code": 200,
        "source-file": "https://example.com/page",
        "line-number": 15,
        "column-number": 2
    }
}
```

### 2.6 Common CSP Mistakes

```python
"""
Common CSP mistakes and how to fix them.
"""

# ── MISTAKE 1: Using 'unsafe-inline' with script-src ─────────────
# This defeats the purpose of CSP for XSS prevention
bad_csp = "script-src 'self' 'unsafe-inline'"
# FIX: Use nonces or hashes instead
good_csp = "script-src 'self' 'nonce-{random}'"

# ── MISTAKE 2: Wildcard in script-src ────────────────────────────
# Allows loading scripts from any subdomain
bad_csp = "script-src 'self' *.googleapis.com"
# FIX: Use specific hostnames
good_csp = "script-src 'self' https://ajax.googleapis.com"

# ── MISTAKE 3: Missing default-src ───────────────────────────────
# Without default-src, unlisted directives default to allow-all
bad_csp = "script-src 'self'"
# FIX: Always include default-src as fallback
good_csp = "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'"

# ── MISTAKE 4: Allowing data: in script-src ──────────────────────
# data: URIs can contain executable JavaScript
bad_csp = "script-src 'self' data:"
# FIX: Only use data: for images/fonts where needed
good_csp = "script-src 'self'; img-src 'self' data:"

# ── MISTAKE 5: Forgetting object-src ────────────────────────────
# Flash and Java applets can execute scripts
bad_csp = "default-src 'self'"  # object-src falls back to 'self'
# FIX: Explicitly block object-src
good_csp = "default-src 'self'; object-src 'none'"

# ── MISTAKE 6: Overly permissive connect-src ────────────────────
# Allows data exfiltration to any HTTPS endpoint
bad_csp = "connect-src https:"
# FIX: List specific API endpoints
good_csp = "connect-src 'self' https://api.example.com"
```

---

## 3. Strict-Transport-Security (HSTS)

### 3.1 How HSTS Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HSTS Protection Flow                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  WITHOUT HSTS:                                                       │
│  User ──http://──▶ Server ──301 Redirect──▶ https://                │
│        ↑                                                             │
│        └── MITM can intercept this plain HTTP request                │
│                                                                      │
│  WITH HSTS:                                                          │
│  User ──http://──▶ Browser intercepts (307 Internal Redirect)        │
│                    Browser ──https://──▶ Server                      │
│                    (no network request on HTTP)                       │
│                                                                      │
│  First visit:                                                        │
│  1. Browser connects via HTTP                                        │
│  2. Server responds with 301 + HSTS header                           │
│  3. Browser stores HSTS policy for domain                            │
│                                                                      │
│  Subsequent visits:                                                   │
│  1. Browser automatically upgrades to HTTPS                          │
│  2. No HTTP request ever leaves the browser                          │
│  3. Invalid certificates cause hard failure (no bypass)              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 HSTS Directives

```
Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
```

| Directive | Purpose | Recommended Value |
|-----------|---------|-------------------|
| `max-age` | Duration (seconds) to remember HSTS | `31536000` (1 year) |
| `includeSubDomains` | Apply to all subdomains | Include for full protection |
| `preload` | Request inclusion in browser preload list | Include after testing |

```python
"""
HSTS configuration considerations.
"""

# ── Deployment strategy: gradual rollout ─────────────────────────
# Start with a short max-age and increase over time

# Step 1: Test with 5 minutes
hsts_test = "max-age=300"

# Step 2: Increase to 1 week
hsts_week = "max-age=604800"

# Step 3: Increase to 1 month with subdomains
hsts_month = "max-age=2592000; includeSubDomains"

# Step 4: Full deployment (1 year + preload)
hsts_full = "max-age=31536000; includeSubDomains; preload"

# WARNING: Setting a long max-age before ensuring all
# subdomains support HTTPS can lock users out.
# includeSubDomains means ALL subdomains MUST have valid TLS.

# ── HSTS Preload List ───────────────────────────────────────────
# The HSTS preload list is baked into browsers.
# Domains on this list ALWAYS use HTTPS, even on first visit.
# Submit at: https://hstspreload.org/
#
# Requirements for preload:
# 1. Valid certificate
# 2. Redirect all HTTP to HTTPS on same host
# 3. HSTS header with:
#    - max-age >= 31536000 (1 year)
#    - includeSubDomains
#    - preload
# 4. HTTPS redirect must also include HSTS header
```

---

## 4. X-Content-Type-Options

### 4.1 MIME Sniffing Attack

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MIME Sniffing Attack                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Attacker uploads: malicious.jpg                                     │
│  (actually contains JavaScript, not image data)                      │
│                                                                      │
│  Server serves: Content-Type: image/jpeg                             │
│                                                                      │
│  WITHOUT nosniff:                                                    │
│  Browser sniffs content ──▶ "This looks like JavaScript"             │
│  Browser executes the file as JavaScript ──▶ XSS!                    │
│                                                                      │
│  WITH nosniff:                                                       │
│  Browser trusts Content-Type ──▶ "This is image/jpeg"                │
│  Browser renders as image (fails) ──▶ No script execution            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Configuration

```
X-Content-Type-Options: nosniff
```

This is the simplest security header — it has exactly one valid value: `nosniff`. There is no reason not to include it on every response. It prevents:

- Scripts being loaded from files with non-script MIME types
- Stylesheets being loaded from files with non-CSS MIME types
- MIME-type confusion attacks when serving user-uploaded content

---

## 5. X-Frame-Options and frame-ancestors

### 5.1 Clickjacking Prevention

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Clickjacking Attack                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Attacker's page:                                                    │
│  ┌─────────────────────────────────────────┐                         │
│  │  "Click here to win a prize!"           │                         │
│  │                                          │                        │
│  │  ┌─────────────────────────────────┐     │                        │
│  │  │  Invisible iframe (opacity: 0)  │     │                        │
│  │  │  ┌────────────────────────┐     │     │                        │
│  │  │  │ Your Banking App       │     │     │                        │
│  │  │  │                        │     │     │                        │
│  │  │  │  [Transfer $1000] ◄────┼─────┼──── User clicks here       │
│  │  │  │                        │     │     │                        │
│  │  │  └────────────────────────┘     │     │                        │
│  │  └─────────────────────────────────┘     │                        │
│  └─────────────────────────────────────────┘                         │
│                                                                      │
│  The user thinks they are clicking on the prize button,              │
│  but they are actually clicking the "Transfer" button                │
│  in the invisible iframe containing their bank's website.            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 X-Frame-Options Values

```
# Deny all framing
X-Frame-Options: DENY

# Allow only same origin
X-Frame-Options: SAMEORIGIN

# Allow specific origin (deprecated, not recommended)
X-Frame-Options: ALLOW-FROM https://trusted.example.com
```

### 5.3 CSP frame-ancestors (Modern Replacement)

```python
"""
frame-ancestors is the CSP replacement for X-Frame-Options.
It provides more granular control and supports multiple origins.
"""

# ── Block all framing (equivalent to X-Frame-Options: DENY) ─────
csp_no_frame = "frame-ancestors 'none'"

# ── Allow same-origin only ───────────────────────────────────────
csp_same_origin = "frame-ancestors 'self'"

# ── Allow specific origins ───────────────────────────────────────
csp_specific = "frame-ancestors 'self' https://trusted.example.com https://partner.example.com"

# ── Key differences from X-Frame-Options ────────────────────────
#
# | Feature              | X-Frame-Options      | frame-ancestors        |
# |---------------------|----------------------|------------------------|
# | Multiple origins    | No                   | Yes                    |
# | Wildcard subdomains | No                   | Yes (*.example.com)    |
# | scheme restriction  | No                   | Yes (https:)           |
# | CSP integration     | Separate header      | Part of CSP            |
# | Browser support     | Universal            | Modern browsers        |
#
# Recommendation: Set BOTH for maximum compatibility
# X-Frame-Options: DENY
# Content-Security-Policy: frame-ancestors 'none'
```

---

## 6. Referrer-Policy

### 6.1 Referrer Information Leakage

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Referrer Leakage Scenarios                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  User is on: https://bank.com/accounts/12345/transfer?amount=500     │
│                                                                      │
│  They click a link to: https://analytics.example.com                 │
│                                                                      │
│  WITHOUT Referrer-Policy:                                            │
│  Referer: https://bank.com/accounts/12345/transfer?amount=500        │
│  ↑ Full URL including path and query params leaked!                  │
│                                                                      │
│  WITH Referrer-Policy: strict-origin-when-cross-origin               │
│  Referer: https://bank.com                                           │
│  ↑ Only origin sent, no sensitive path or params                     │
│                                                                      │
│  WITH Referrer-Policy: no-referrer                                   │
│  Referer: (empty)                                                    │
│  ↑ No referrer information sent at all                               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Policy Values

| Policy | Same-Origin | Cross-Origin (HTTPS→HTTPS) | Downgrade (HTTPS→HTTP) |
|--------|-------------|---------------------------|----------------------|
| `no-referrer` | Nothing | Nothing | Nothing |
| `no-referrer-when-downgrade` | Full URL | Full URL | Nothing |
| `origin` | Origin only | Origin only | Origin only |
| `origin-when-cross-origin` | Full URL | Origin only | Origin only |
| `same-origin` | Full URL | Nothing | Nothing |
| `strict-origin` | Origin only | Origin only | Nothing |
| `strict-origin-when-cross-origin` | Full URL | Origin only | Nothing |
| `unsafe-url` | Full URL | Full URL | Full URL |

```python
"""
Recommended Referrer-Policy configurations.
"""

# ── Default recommendation ──────────────────────────────────────
# strict-origin-when-cross-origin is the browser default in modern browsers
# and a good balance of functionality and privacy
referrer_policy = "strict-origin-when-cross-origin"

# ── Maximum privacy ─────────────────────────────────────────────
# no-referrer strips all referrer information
# Downside: breaks analytics and some CSRF protections
referrer_max_privacy = "no-referrer"

# ── For sites with sensitive URLs ───────────────────────────────
# same-origin sends full referrer only within the same site
referrer_sensitive = "same-origin"

# ── Per-element override ────────────────────────────────────────
# You can also set referrer policy on individual elements:
# <a href="..." referrerpolicy="no-referrer">External Link</a>
# <img src="..." referrerpolicy="no-referrer">
# <script src="..." referrerpolicy="no-referrer">
```

---

## 7. Permissions-Policy

### 7.1 Restricting Browser Features

Permissions-Policy (formerly Feature-Policy) controls which browser features and APIs can be used by the page and its embedded content.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Permissions-Policy                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Controls access to browser APIs:                                    │
│                                                                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐    │
│  │  Camera    │  │ Microphone │  │ Geolocation│  │  Payment   │    │
│  │  camera    │  │ microphone │  │ geolocation│  │  payment   │    │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘    │
│                                                                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐    │
│  │ Fullscreen │  │  Autoplay  │  │  USB       │  │  Bluetooth │    │
│  │ fullscreen │  │  autoplay  │  │  usb       │  │  bluetooth │    │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘    │
│                                                                      │
│  Syntax:                                                             │
│  Permissions-Policy: feature=(allowlist)                             │
│                                                                      │
│  Allowlist values:                                                   │
│  *            = Allow all origins                                    │
│  self         = Allow same origin only                               │
│  (empty)      = () means block entirely                              │
│  "origin"     = Allow specific origin                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2 Configuration Examples

```python
"""
Permissions-Policy header configuration.
"""

# ── Restrictive policy (recommended for most sites) ─────────────
permissions_policy = (
    "camera=(), "                              # Block camera
    "microphone=(), "                          # Block microphone
    "geolocation=(), "                         # Block geolocation
    "payment=(), "                             # Block Payment API
    "usb=(), "                                 # Block WebUSB
    "bluetooth=(), "                           # Block Web Bluetooth
    "magnetometer=(), "                        # Block magnetometer
    "gyroscope=(), "                           # Block gyroscope
    "accelerometer=(), "                       # Block accelerometer
    'autoplay=(self), '                        # Allow autoplay same-origin only
    'fullscreen=(self), '                      # Allow fullscreen same-origin only
    'picture-in-picture=(self)'                # Allow PiP same-origin only
)

# ── Policy for a video conferencing app ─────────────────────────
permissions_video_app = (
    'camera=(self "https://meet.example.com"), '
    'microphone=(self "https://meet.example.com"), '
    "geolocation=(), "
    'fullscreen=(self), '
    'display-capture=(self)'
)

# ── Policy for an e-commerce site ───────────────────────────────
permissions_ecommerce = (
    "camera=(), "
    "microphone=(), "
    'geolocation=(self), '                     # For store locator
    'payment=(self), '                         # For Payment Request API
    "usb=(), "
    "bluetooth=()"
)
```

---

## 8. Cross-Origin Policies (CORP, COEP, COOP)

### 8.1 Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Cross-Origin Isolation                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Three headers work together for cross-origin isolation:             │
│                                                                      │
│  CORP (Cross-Origin-Resource-Policy)                                 │
│  ├── Set on RESOURCES (images, scripts, etc.)                        │
│  ├── Controls who can load this resource                             │
│  └── Values: same-site, same-origin, cross-origin                    │
│                                                                      │
│  COEP (Cross-Origin-Embedder-Policy)                                 │
│  ├── Set on the DOCUMENT that embeds resources                       │
│  ├── Requires all resources to opt in (via CORP or CORS)             │
│  └── Values: unsafe-none, require-corp, credentialless               │
│                                                                      │
│  COOP (Cross-Origin-Opener-Policy)                                   │
│  ├── Set on the DOCUMENT                                             │
│  ├── Controls window.opener relationships                            │
│  └── Values: unsafe-none, same-origin, same-origin-allow-popups      │
│                                                                      │
│  When COEP: require-corp + COOP: same-origin are both set:           │
│  ──▶ Page is "cross-origin isolated"                                 │
│  ──▶ Enables SharedArrayBuffer, high-resolution timers               │
│  ──▶ Protects against Spectre-style side-channel attacks             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.2 Cross-Origin-Resource-Policy (CORP)

```python
"""
CORP controls which origins can load a resource.
Set this on your API responses, images, scripts, etc.
"""

# ── Values ───────────────────────────────────────────────────────
# same-origin:  Only requests from the same origin
# same-site:    Requests from the same site (includes subdomains)
# cross-origin: Any origin can load this resource

# For private API endpoints
corp_api = "same-origin"

# For public CDN resources
corp_cdn = "cross-origin"

# For resources shared across subdomains
corp_shared = "same-site"
```

### 8.3 Cross-Origin-Embedder-Policy (COEP)

```python
"""
COEP ensures that all resources loaded by a page have
explicitly opted in to being loaded cross-origin.
"""

# ── require-corp: strictest mode ────────────────────────────────
# All cross-origin resources must either:
# 1. Be served with CORP: cross-origin header
# 2. Be loaded with crossorigin attribute (CORS)
coep_strict = "require-corp"

# In HTML, resources need crossorigin attribute:
# <img src="https://cdn.example.com/image.jpg" crossorigin>
# <script src="https://cdn.example.com/lib.js" crossorigin>

# ── credentialless: more practical ──────────────────────────────
# Cross-origin requests are made without credentials (cookies)
# Resources don't need CORP header
coep_credentialless = "credentialless"

# ── unsafe-none: no restriction (default) ───────────────────────
coep_none = "unsafe-none"
```

### 8.4 Cross-Origin-Opener-Policy (COOP)

```python
"""
COOP controls the relationship between a page and its opener
(the page that opened it via window.open or a link).
"""

# ── same-origin: full isolation ─────────────────────────────────
# Breaks window.opener references to cross-origin pages
# Prevents cross-origin pages from accessing this window
coop_strict = "same-origin"

# ── same-origin-allow-popups ────────────────────────────────────
# Same as same-origin, but popups opened by this page
# can retain their opener reference
coop_popups = "same-origin-allow-popups"

# ── unsafe-none: no restriction (default) ───────────────────────
coop_none = "unsafe-none"

# ── Achieving cross-origin isolation ────────────────────────────
# Both headers required:
# Cross-Origin-Embedder-Policy: require-corp
# Cross-Origin-Opener-Policy: same-origin
#
# Verify in JavaScript:
# if (crossOriginIsolated) {
#   // SharedArrayBuffer is available
#   // Performance.now() has full precision
# }
```

---

## 9. Subresource Integrity (SRI)

### 9.1 How SRI Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Subresource Integrity (SRI)                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Problem: CDN compromise                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                       │
│  │ Your App │───▶│   CDN    │───▶│ Browser  │                       │
│  └──────────┘    └──────────┘    └──────────┘                       │
│                       ↑                                              │
│                   Attacker modifies                                   │
│                   jquery.min.js                                       │
│                   on the CDN                                         │
│                                                                      │
│  Solution: SRI hash verification                                     │
│  <script src="https://cdn/jquery.js"                                │
│          integrity="sha384-abc123..."                                │
│          crossorigin="anonymous">                                    │
│  </script>                                                           │
│                                                                      │
│  Browser downloads file ──▶ Computes hash ──▶ Compares to integrity  │
│  Match?    ✓ Execute     ✗ Block and report error                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 9.2 Generating SRI Hashes

```bash
# Generate SRI hash from a local file
cat jquery-3.7.1.min.js | openssl dgst -sha384 -binary | openssl base64 -A
# Output: oQVuAfEn...

# Generate SRI hash using shasum
shasum -b -a 384 jquery-3.7.1.min.js | awk '{ print $1 }' | xxd -r -p | base64

# Using curl to generate hash from remote file
curl -s https://code.jquery.com/jquery-3.7.1.min.js | \
    openssl dgst -sha384 -binary | openssl base64 -A
```

```python
"""
Generating and verifying SRI hashes in Python.
"""
import hashlib
import base64
import requests

def generate_sri_hash(content: bytes, algorithm: str = 'sha384') -> str:
    """Generate an SRI hash for the given content."""
    hash_func = getattr(hashlib, algorithm)
    digest = hash_func(content).digest()
    b64 = base64.b64encode(digest).decode('utf-8')
    return f"{algorithm}-{b64}"

def generate_sri_from_url(url: str) -> str:
    """Download a resource and generate its SRI hash."""
    response = requests.get(url)
    response.raise_for_status()
    return generate_sri_hash(response.content)

# Example usage
url = "https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"
sri_hash = generate_sri_from_url(url)
print(f'<link rel="stylesheet" href="{url}" '
      f'integrity="{sri_hash}" crossorigin="anonymous">')

# ── SRI in HTML ──────────────────────────────────────────────────
# For scripts:
# <script src="https://cdn.example.com/lib.js"
#         integrity="sha384-oqVuAfXRKap7fdgcCY5uykM6+R9GqQ8K/uxy9rx7HNQlGYl1kPzQho1wx4JwY8wC"
#         crossorigin="anonymous"></script>
#
# For stylesheets:
# <link rel="stylesheet"
#       href="https://cdn.example.com/style.css"
#       integrity="sha384-abc123..."
#       crossorigin="anonymous">
#
# Multiple hashes (for migration):
# <script src="https://cdn.example.com/lib.js"
#         integrity="sha384-oldHash... sha384-newHash..."
#         crossorigin="anonymous"></script>
# Browser accepts if ANY hash matches

# ── Important notes ──────────────────────────────────────────────
# 1. crossorigin="anonymous" is REQUIRED for cross-origin SRI
# 2. SRI only works with <script> and <link rel="stylesheet">
# 3. Hash must match byte-for-byte (whitespace matters)
# 4. If CDN updates the file, the hash breaks (this is by design)
```

---

## 10. Flask Security Headers Configuration

### 10.1 Manual Header Setting

```python
"""
Setting security headers in Flask applications.
"""
from flask import Flask, request, make_response, g
import secrets

app = Flask(__name__)

# ── Method 1: after_request decorator ────────────────────────────
@app.after_request
def set_security_headers(response):
    """Add security headers to every response."""

    # Generate nonce for CSP
    nonce = getattr(g, 'csp_nonce', secrets.token_urlsafe(32))

    # Content-Security-Policy
    response.headers['Content-Security-Policy'] = (
        f"default-src 'self'; "
        f"script-src 'self' 'nonce-{nonce}'; "
        f"style-src 'self' 'nonce-{nonce}' https://fonts.googleapis.com; "
        f"font-src 'self' https://fonts.gstatic.com; "
        f"img-src 'self' data:; "
        f"connect-src 'self'; "
        f"object-src 'none'; "
        f"base-uri 'self'; "
        f"form-action 'self'; "
        f"frame-ancestors 'none'"
    )

    # HSTS (only set over HTTPS in production)
    response.headers['Strict-Transport-Security'] = (
        'max-age=31536000; includeSubDomains; preload'
    )

    # Prevent MIME sniffing
    response.headers['X-Content-Type-Options'] = 'nosniff'

    # Clickjacking protection
    response.headers['X-Frame-Options'] = 'DENY'

    # Referrer control
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'

    # Permissions policy
    response.headers['Permissions-Policy'] = (
        'camera=(), microphone=(), geolocation=(), payment=()'
    )

    # Cross-origin policies
    response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
    response.headers['Cross-Origin-Resource-Policy'] = 'same-origin'

    # Remove server identification
    response.headers.pop('Server', None)

    return response


@app.before_request
def generate_nonce():
    """Generate a CSP nonce for each request."""
    g.csp_nonce = secrets.token_urlsafe(32)


# ── Method 2: Using Flask-Talisman ──────────────────────────────
# pip install flask-talisman
from flask_talisman import Talisman

app2 = Flask(__name__)

csp = {
    'default-src': "'self'",
    'script-src': "'self'",
    'style-src': "'self' https://fonts.googleapis.com",
    'font-src': "'self' https://fonts.gstatic.com",
    'img-src': "'self' data:",
    'object-src': "'none'",
}

talisman = Talisman(
    app2,
    content_security_policy=csp,
    content_security_policy_nonce_in=['script-src', 'style-src'],
    force_https=True,
    strict_transport_security=True,
    strict_transport_security_max_age=31536000,
    strict_transport_security_include_subdomains=True,
    strict_transport_security_preload=True,
    frame_options='DENY',
    referrer_policy='strict-origin-when-cross-origin',
    permissions_policy={
        'camera': '()',
        'microphone': '()',
        'geolocation': '()',
    },
    session_cookie_secure=True,
    session_cookie_http_only=True,
)

# In Jinja2 templates, use the nonce:
# <script nonce="{{ csp_nonce() }}">
#     // Inline JavaScript here
# </script>
```

### 10.2 Per-Route Header Overrides

```python
"""
Different security headers for different routes.
"""
from flask import Flask, make_response
from functools import wraps

app = Flask(__name__)

def custom_csp(csp_string):
    """Decorator to override CSP for specific routes."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            response = make_response(f(*args, **kwargs))
            response.headers['Content-Security-Policy'] = csp_string
            return response
        return decorated_function
    return decorator


@app.route('/admin')
@custom_csp("default-src 'self'; script-src 'self'; frame-ancestors 'none'")
def admin_panel():
    """Admin panel with strict CSP."""
    return "Admin Panel"


@app.route('/embed-widget')
@custom_csp(
    "default-src 'self'; "
    "frame-ancestors 'self' https://partner.example.com"
)
def embeddable_widget():
    """Widget that can be embedded by specific partners."""
    return "Widget"


@app.route('/public-api')
def public_api():
    """API endpoint with relaxed CORP for cross-origin access."""
    response = make_response({"data": "value"})
    response.headers['Cross-Origin-Resource-Policy'] = 'cross-origin'
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response
```

---

## 11. Django Security Headers Configuration

### 11.1 Django Settings

```python
"""
Django security header configuration in settings.py.
"""

# ── Built-in Django security settings ────────────────────────────

# HSTS
SECURE_HSTS_SECONDS = 31536000           # 1 year
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True

# HTTPS redirect
SECURE_SSL_REDIRECT = True               # Redirect HTTP to HTTPS
SECURE_REDIRECT_EXEMPT = []              # Paths exempt from redirect

# Content-Type sniffing
SECURE_CONTENT_TYPE_NOSNIFF = True       # X-Content-Type-Options: nosniff

# Clickjacking protection
X_FRAME_OPTIONS = 'DENY'                 # Built-in middleware

# Referrer policy
SECURE_REFERRER_POLICY = 'strict-origin-when-cross-origin'

# Cross-origin opener policy
SECURE_CROSS_ORIGIN_OPENER_POLICY = 'same-origin'

# Cookie security
SESSION_COOKIE_SECURE = True
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = 'Lax'
CSRF_COOKIE_SECURE = True
CSRF_COOKIE_HTTPONLY = True

# ── Required middleware ──────────────────────────────────────────
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',       # MUST be first
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    # ... other middleware
]
```

### 11.2 Django CSP with django-csp

```python
"""
CSP configuration using django-csp package.
pip install django-csp
"""

# settings.py
MIDDLEWARE = [
    'csp.middleware.CSPMiddleware',
    # ... other middleware
]

# CSP directives
CSP_DEFAULT_SRC = ("'self'",)
CSP_SCRIPT_SRC = ("'self'",)
CSP_STYLE_SRC = ("'self'", "https://fonts.googleapis.com")
CSP_FONT_SRC = ("'self'", "https://fonts.gstatic.com")
CSP_IMG_SRC = ("'self'", "data:")
CSP_CONNECT_SRC = ("'self'",)
CSP_OBJECT_SRC = ("'none'",)
CSP_BASE_URI = ("'self'",)
CSP_FORM_ACTION = ("'self'",)
CSP_FRAME_ANCESTORS = ("'none'",)

# Enable nonce for script-src and style-src
CSP_INCLUDE_NONCE_IN = ['script-src', 'style-src']

# Report violations
CSP_REPORT_URI = '/csp-report/'

# Report-only mode (for testing)
# CSP_REPORT_ONLY = True


# ── In Django templates ──────────────────────────────────────────
# {% load csp %}
#
# <script nonce="{% csp_nonce %}">
#     // This inline script will be allowed by the nonce
#     console.log('Hello from a nonced script');
# </script>
#
# ── Per-view CSP override ────────────────────────────────────────
# from csp.decorators import csp_update, csp_replace, csp_exempt
#
# @csp_update(SCRIPT_SRC=("'self'", "https://cdn.example.com"))
# def my_view(request):
#     ...
#
# @csp_exempt  # Disables CSP entirely for this view
# def legacy_view(request):
#     ...
```

---

## 12. Testing Security Headers

### 12.1 Testing with curl

```bash
# ── View all response headers ────────────────────────────────────
curl -I https://example.com

# ── Check specific security headers ─────────────────────────────
curl -sI https://example.com | grep -iE \
  '(content-security|strict-transport|x-content-type|x-frame|referrer-policy|permissions-policy|cross-origin)'

# ── Verbose output to see TLS details too ───────────────────────
curl -vI https://example.com 2>&1 | head -40

# ── Test HSTS ───────────────────────────────────────────────────
curl -sI https://example.com | grep -i strict-transport

# ── Test CSP in report-only mode ────────────────────────────────
curl -sI https://example.com | grep -i content-security-policy

# ── Check for missing headers ───────────────────────────────────
HEADERS_TO_CHECK=(
    "Content-Security-Policy"
    "Strict-Transport-Security"
    "X-Content-Type-Options"
    "X-Frame-Options"
    "Referrer-Policy"
    "Permissions-Policy"
)

URL="https://example.com"
echo "Checking security headers for $URL"
echo "=================================="

for header in "${HEADERS_TO_CHECK[@]}"; do
    result=$(curl -sI "$URL" | grep -i "^$header:")
    if [ -n "$result" ]; then
        echo "[PASS] $result"
    else
        echo "[FAIL] Missing: $header"
    fi
done
```

### 12.2 Python Security Header Scanner

```python
"""
A simple security header scanner.
"""
import requests
from dataclasses import dataclass
from typing import Optional

@dataclass
class HeaderCheck:
    name: str
    present: bool
    value: Optional[str]
    severity: str  # 'critical', 'high', 'medium', 'low'
    recommendation: str

def scan_security_headers(url: str) -> list[HeaderCheck]:
    """Scan a URL for security headers and return findings."""

    response = requests.get(url, allow_redirects=True, timeout=10)
    headers = response.headers
    results = []

    # ── Content-Security-Policy ──────────────────────────────────
    csp = headers.get('Content-Security-Policy')
    results.append(HeaderCheck(
        name='Content-Security-Policy',
        present=csp is not None,
        value=csp,
        severity='critical',
        recommendation=(
            "Add CSP header. Start with: "
            "Content-Security-Policy: default-src 'self'; "
            "object-src 'none'; base-uri 'self'"
        ) if not csp else _analyze_csp(csp)
    ))

    # ── Strict-Transport-Security ────────────────────────────────
    hsts = headers.get('Strict-Transport-Security')
    results.append(HeaderCheck(
        name='Strict-Transport-Security',
        present=hsts is not None,
        value=hsts,
        severity='critical',
        recommendation=(
            "Add HSTS header: "
            "Strict-Transport-Security: max-age=31536000; "
            "includeSubDomains; preload"
        ) if not hsts else _analyze_hsts(hsts)
    ))

    # ── X-Content-Type-Options ───────────────────────────────────
    xcto = headers.get('X-Content-Type-Options')
    results.append(HeaderCheck(
        name='X-Content-Type-Options',
        present=xcto is not None,
        value=xcto,
        severity='high',
        recommendation=(
            "Add header: X-Content-Type-Options: nosniff"
        ) if not xcto else "OK"
    ))

    # ── X-Frame-Options ─────────────────────────────────────────
    xfo = headers.get('X-Frame-Options')
    results.append(HeaderCheck(
        name='X-Frame-Options',
        present=xfo is not None,
        value=xfo,
        severity='high',
        recommendation=(
            "Add header: X-Frame-Options: DENY "
            "(or SAMEORIGIN if framing is needed)"
        ) if not xfo else "OK"
    ))

    # ── Referrer-Policy ──────────────────────────────────────────
    rp = headers.get('Referrer-Policy')
    results.append(HeaderCheck(
        name='Referrer-Policy',
        present=rp is not None,
        value=rp,
        severity='medium',
        recommendation=(
            "Add header: Referrer-Policy: "
            "strict-origin-when-cross-origin"
        ) if not rp else "OK"
    ))

    # ── Permissions-Policy ───────────────────────────────────────
    pp = headers.get('Permissions-Policy')
    results.append(HeaderCheck(
        name='Permissions-Policy',
        present=pp is not None,
        value=pp,
        severity='medium',
        recommendation=(
            "Add header: Permissions-Policy: "
            "camera=(), microphone=(), geolocation=()"
        ) if not pp else "OK"
    ))

    # ── Check for dangerous headers ─────────────────────────────
    server = headers.get('Server')
    if server:
        results.append(HeaderCheck(
            name='Server',
            present=True,
            value=server,
            severity='low',
            recommendation=(
                f"Server header reveals: '{server}'. "
                "Consider removing or obfuscating."
            )
        ))

    x_powered = headers.get('X-Powered-By')
    if x_powered:
        results.append(HeaderCheck(
            name='X-Powered-By',
            present=True,
            value=x_powered,
            severity='medium',
            recommendation=(
                f"X-Powered-By reveals: '{x_powered}'. "
                "Remove this header to avoid information disclosure."
            )
        ))

    return results


def _analyze_csp(csp: str) -> str:
    """Analyze a CSP policy for common weaknesses."""
    issues = []
    if "'unsafe-inline'" in csp and 'script-src' in csp:
        issues.append("script-src allows 'unsafe-inline' (weakens XSS protection)")
    if "'unsafe-eval'" in csp:
        issues.append("Policy allows 'unsafe-eval' (enables eval()-based attacks)")
    if "default-src" not in csp:
        issues.append("Missing default-src fallback directive")
    if "object-src" not in csp and "default-src 'none'" not in csp:
        issues.append("Missing object-src directive (Flash/plugin risks)")
    if "base-uri" not in csp:
        issues.append("Missing base-uri (can enable dangling markup injection)")
    return "; ".join(issues) if issues else "OK"


def _analyze_hsts(hsts: str) -> str:
    """Analyze HSTS header for weaknesses."""
    issues = []
    hsts_lower = hsts.lower()
    if 'max-age=' in hsts_lower:
        # Extract max-age value
        import re
        match = re.search(r'max-age=(\d+)', hsts_lower)
        if match:
            max_age = int(match.group(1))
            if max_age < 31536000:
                issues.append(
                    f"max-age={max_age} is less than 1 year (31536000)"
                )
    if 'includesubdomains' not in hsts_lower:
        issues.append("Missing includeSubDomains")
    if 'preload' not in hsts_lower:
        issues.append("Missing preload (not eligible for browser preload list)")
    return "; ".join(issues) if issues else "OK"


# ── Usage ────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys

    url = sys.argv[1] if len(sys.argv) > 1 else 'https://example.com'
    print(f"\nScanning: {url}\n{'=' * 60}")

    results = scan_security_headers(url)

    for check in results:
        status = "PASS" if check.present and check.recommendation == "OK" else "FAIL"
        icon = "[+]" if status == "PASS" else "[-]"
        print(f"\n{icon} {check.name} [{check.severity.upper()}]")
        print(f"    Value: {check.value or '(not set)'}")
        if check.recommendation != "OK":
            print(f"    Recommendation: {check.recommendation}")
```

### 12.3 Online Scanners

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Security Header Scanners                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. SecurityHeaders.com                                              │
│     URL: https://securityheaders.com                                 │
│     Grades: A+ to F                                                  │
│     Checks all major security headers                                │
│                                                                      │
│  2. Mozilla Observatory                                              │
│     URL: https://observatory.mozilla.org                             │
│     Comprehensive scan including CSP analysis                        │
│     Provides remediation advice                                      │
│                                                                      │
│  3. CSP Evaluator (Google)                                           │
│     URL: https://csp-evaluator.withgoogle.com                        │
│     Specialized CSP analysis                                         │
│     Identifies bypasses and weaknesses                               │
│                                                                      │
│  4. Hardenize                                                        │
│     URL: https://www.hardenize.com                                   │
│     Tests headers + TLS + DNS + email security                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 13. Complete Security Headers Template

### 13.1 Nginx Configuration

```nginx
# /etc/nginx/conf.d/security-headers.conf
# Include this in your server block

# Content-Security-Policy
add_header Content-Security-Policy
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'"
    always;

# HSTS (only enable when you are sure HTTPS works everywhere)
add_header Strict-Transport-Security
    "max-age=31536000; includeSubDomains; preload"
    always;

# Prevent MIME sniffing
add_header X-Content-Type-Options "nosniff" always;

# Clickjacking protection
add_header X-Frame-Options "DENY" always;

# Referrer policy
add_header Referrer-Policy "strict-origin-when-cross-origin" always;

# Permissions policy
add_header Permissions-Policy
    "camera=(), microphone=(), geolocation=(), payment=()"
    always;

# Cross-origin policies
add_header Cross-Origin-Opener-Policy "same-origin" always;
add_header Cross-Origin-Resource-Policy "same-origin" always;
add_header Cross-Origin-Embedder-Policy "require-corp" always;

# Remove server version info
server_tokens off;

# Remove X-Powered-By (if set by upstream app)
proxy_hide_header X-Powered-By;
```

### 13.2 Apache Configuration

```apache
# .htaccess or httpd.conf

# Content-Security-Policy
Header always set Content-Security-Policy "\
default-src 'self'; \
script-src 'self'; \
style-src 'self' 'unsafe-inline'; \
img-src 'self' data:; \
font-src 'self'; \
connect-src 'self'; \
object-src 'none'; \
base-uri 'self'; \
form-action 'self'; \
frame-ancestors 'none'"

# HSTS
Header always set Strict-Transport-Security \
    "max-age=31536000; includeSubDomains; preload"

# Other security headers
Header always set X-Content-Type-Options "nosniff"
Header always set X-Frame-Options "DENY"
Header always set Referrer-Policy "strict-origin-when-cross-origin"
Header always set Permissions-Policy \
    "camera=(), microphone=(), geolocation=(), payment=()"
Header always set Cross-Origin-Opener-Policy "same-origin"
Header always set Cross-Origin-Resource-Policy "same-origin"

# Remove server information
ServerTokens Prod
Header always unset X-Powered-By
```

---

## 14. Exercises

### Exercise 1: CSP Policy Analysis

Analyze the following CSP policy and identify all security weaknesses:

```
Content-Security-Policy:
    default-src *;
    script-src 'self' 'unsafe-inline' 'unsafe-eval' https:;
    style-src 'self' 'unsafe-inline';
    img-src *;
    connect-src *;
    font-src *
```

Questions:
1. How many distinct security issues can you identify?
2. For each issue, explain the attack vector it enables.
3. Rewrite the policy to be secure while allowing Google Fonts and a single CDN (`cdn.example.com`).

### Exercise 2: Flask Security Headers Middleware

Build a Flask middleware class that:

1. Sets all recommended security headers
2. Generates a unique CSP nonce per request
3. Makes the nonce available to Jinja2 templates
4. Allows per-route CSP overrides via a decorator
5. Supports a "report-only" mode for CSP testing
6. Logs any header configuration errors

### Exercise 3: Header Scanner Enhancement

Extend the Python security header scanner from section 12.2 to:

1. Check for `Cross-Origin-Embedder-Policy` and `Cross-Origin-Opener-Policy`
2. Verify that HSTS max-age is at least 1 year
3. Detect if `X-Powered-By` or `Server` headers reveal version information
4. Parse and analyze CSP for `unsafe-inline` and `unsafe-eval` usage
5. Generate a grade (A+ to F) based on the scan results
6. Output results in both text and JSON format

### Exercise 4: SRI Hash Generator

Write a Python script that:

1. Takes a list of CDN URLs as input
2. Downloads each resource
3. Computes SHA-384 integrity hashes
4. Generates the complete HTML `<script>` or `<link>` tags with integrity attributes
5. Optionally writes an HTML file with all the tags
6. Handles errors gracefully (network failures, 404s, etc.)

### Exercise 5: HSTS Preload Readiness Check

Write a tool that checks whether a domain is ready for HSTS preload submission:

1. Verify the domain has a valid TLS certificate
2. Check that HTTP redirects to HTTPS
3. Verify the HSTS header includes `max-age >= 31536000`
4. Check for `includeSubDomains` and `preload` directives
5. Test common subdomains (www, mail, api) for HTTPS support
6. Generate a readiness report

### Exercise 6: Cross-Origin Isolation Audit

For a given web application URL:

1. Check if `Cross-Origin-Embedder-Policy` is set
2. Check if `Cross-Origin-Opener-Policy` is set
3. Identify all cross-origin resources loaded by the page
4. For each cross-origin resource, check if it has `Cross-Origin-Resource-Policy` set
5. Report which resources would break if COEP: require-corp were enabled

---

## Summary

| Header | Attack Prevented | Must Have? |
|--------|-----------------|------------|
| Content-Security-Policy | XSS, injection | Yes |
| Strict-Transport-Security | Protocol downgrade | Yes (HTTPS sites) |
| X-Content-Type-Options | MIME confusion | Yes |
| X-Frame-Options | Clickjacking | Yes |
| Referrer-Policy | Information leak | Yes |
| Permissions-Policy | Feature abuse | Recommended |
| CORP/COEP/COOP | Spectre, cross-origin | Recommended |

### Key Takeaways

1. **Start with report-only mode** — deploy CSP in report-only first, fix violations, then enforce
2. **Use nonces over unsafe-inline** — nonces provide per-request authorization for inline scripts
3. **HSTS needs gradual rollout** — start with short max-age, increase after verifying all subdomains support HTTPS
4. **Defense in depth** — security headers complement, not replace, secure coding practices
5. **Test regularly** — use automated scanners in CI/CD to catch header regressions

---

**Previous**: [08. Injection Attacks and Prevention](./08_Injection_Attacks.md) | **Next**: [10_API_Security.md](./10_API_Security.md)
