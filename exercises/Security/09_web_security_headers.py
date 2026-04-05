"""
Exercise Solutions: Web Security Headers and CSP
================================================
Lesson 09 from Security topic.

Covers CSP policy analysis, security headers middleware, header scanning,
SRI hash generation, HSTS preload readiness, and cross-origin isolation.
"""

import base64
import hashlib
import json
import os
import re
import secrets
import ssl
import socket
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# Exercise 1: CSP Policy Analysis
# ---------------------------------------------------------------------------

def exercise_1_csp_analysis():
    """Analyze a CSP policy and identify security weaknesses."""
    policy = (
        "default-src *; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval' https:; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src *; "
        "connect-src *; "
        "font-src *"
    )

    issues = [
        {
            "directive": "default-src *",
            "issue": "Allows loading resources from ANY origin as fallback",
            "attack": "Any directive not explicitly set falls back to *, "
                      "allowing loading from attacker-controlled domains.",
        },
        {
            "directive": "script-src 'unsafe-inline'",
            "issue": "Allows inline <script> tags and event handlers",
            "attack": "XSS payloads in inline scripts execute freely. "
                      "This defeats the primary purpose of CSP.",
        },
        {
            "directive": "script-src 'unsafe-eval'",
            "issue": "Allows eval(), Function(), setTimeout(string), etc.",
            "attack": "Attacker can execute arbitrary JavaScript through eval().",
        },
        {
            "directive": "script-src https:",
            "issue": "Allows scripts from ANY HTTPS origin",
            "attack": "Attacker hosts malicious JS on any HTTPS site and "
                      "includes it via XSS.",
        },
        {
            "directive": "style-src 'unsafe-inline'",
            "issue": "Allows inline styles",
            "attack": "CSS injection can leak data via attribute selectors "
                      "and background-image URLs.",
        },
        {
            "directive": "img-src *",
            "issue": "Allows images from any origin",
            "attack": "Data exfiltration via image requests to attacker server. "
                      "CSS-based data leaks via background images.",
        },
        {
            "directive": "connect-src *",
            "issue": "Allows fetch/XHR/WebSocket to any origin",
            "attack": "XSS payload can exfiltrate data to attacker's server "
                      "via fetch() or WebSocket.",
        },
        {
            "directive": "font-src *",
            "issue": "Allows fonts from any origin",
            "attack": "Minor risk — font-based fingerprinting or homoglyph attacks.",
        },
        {
            "directive": "Missing frame-ancestors",
            "issue": "No clickjacking protection from CSP",
            "attack": "Page can be embedded in an iframe on a malicious site.",
        },
    ]

    print("CSP Policy Analysis")
    print(f"Policy: {policy}\n")
    for i, item in enumerate(issues, 1):
        print(f"  {i}. [{item['directive']}]")
        print(f"     Issue: {item['issue']}")
        print(f"     Attack: {item['attack']}")

    # Rewritten secure policy
    secure_policy = (
        "default-src 'self'; "
        "script-src 'self' 'nonce-{RANDOM}'; "
        "style-src 'self' 'nonce-{RANDOM}' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' https://cdn.example.com data:; "
        "connect-src 'self' https://cdn.example.com; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self'; "
        "upgrade-insecure-requests"
    )
    print(f"\nSecure rewritten policy:\n  {secure_policy}")


# ---------------------------------------------------------------------------
# Exercise 2: Flask Security Headers Middleware
# ---------------------------------------------------------------------------

class SecurityHeadersMiddleware:
    """
    Flask-compatible middleware that sets all recommended security headers.
    Usage (conceptual — not tied to a running Flask app):
        app = Flask(__name__)
        SecurityHeadersMiddleware(app)
    """

    DEFAULT_CSP = (
        "default-src 'self'; "
        "script-src 'self' 'nonce-{nonce}'; "
        "style-src 'self' 'nonce-{nonce}'; "
        "img-src 'self' data:; "
        "font-src 'self'; "
        "connect-src 'self'; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self'"
    )

    def __init__(self, report_only: bool = False):
        self.report_only = report_only
        self._route_overrides: dict[str, str] = {}

    def generate_nonce(self) -> str:
        """Generate a unique CSP nonce for each request."""
        return base64.b64encode(secrets.token_bytes(16)).decode()

    def get_headers(self, route: str = "/") -> dict[str, str]:
        """Generate all security headers for a response."""
        nonce = self.generate_nonce()

        # Use route-specific CSP or default
        csp_template = self._route_overrides.get(route, self.DEFAULT_CSP)
        csp_value = csp_template.replace("{nonce}", nonce)

        csp_header = ("Content-Security-Policy-Report-Only"
                      if self.report_only
                      else "Content-Security-Policy")

        headers = {
            csp_header: csp_value,
            "Strict-Transport-Security": "max-age=63072000; includeSubDomains; preload",
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "0",  # Disabled — CSP is the modern replacement
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": (
                "camera=(), microphone=(), geolocation=(), "
                "payment=(), usb=()"
            ),
            "Cross-Origin-Opener-Policy": "same-origin",
            "Cross-Origin-Embedder-Policy": "require-corp",
        }

        return headers, nonce

    def set_route_csp(self, route: str, csp: str):
        """Override CSP for a specific route."""
        self._route_overrides[route] = csp


def exercise_2_security_headers_middleware():
    """Demonstrate the security headers middleware."""
    middleware = SecurityHeadersMiddleware(report_only=False)

    headers, nonce = middleware.get_headers("/")
    print("Security Headers:")
    for key, value in headers.items():
        print(f"  {key}: {value[:80]}{'...' if len(value) > 80 else ''}")
    print(f"\n  CSP Nonce for templates: {nonce}")

    # Report-only mode
    middleware_ro = SecurityHeadersMiddleware(report_only=True)
    headers_ro, _ = middleware_ro.get_headers("/")
    has_report_only = "Content-Security-Policy-Report-Only" in headers_ro
    print(f"\n  Report-only mode: {'Content-Security-Policy-Report-Only' if has_report_only else 'Content-Security-Policy'}")


# ---------------------------------------------------------------------------
# Exercise 3: Header Scanner
# ---------------------------------------------------------------------------

@dataclass
class HeaderScanResult:
    url: str
    grade: str = "F"
    score: int = 0
    findings: list[dict] = field(default_factory=list)
    headers_found: dict = field(default_factory=dict)


def scan_security_headers(host: str) -> HeaderScanResult:
    """Scan a host for security headers and generate a grade."""
    result = HeaderScanResult(url=host)
    score = 0

    try:
        import http.client
        conn = http.client.HTTPSConnection(host, timeout=10)
        conn.request("HEAD", "/")
        resp = conn.getresponse()
        headers = dict(resp.getheaders())
        result.headers_found = headers
        conn.close()

        # Check each security header
        checks = [
            ("strict-transport-security", 20, "HSTS"),
            ("content-security-policy", 25, "CSP"),
            ("x-content-type-options", 10, "X-Content-Type-Options"),
            ("x-frame-options", 10, "X-Frame-Options"),
            ("referrer-policy", 10, "Referrer-Policy"),
            ("permissions-policy", 5, "Permissions-Policy"),
        ]

        for header, points, name in checks:
            if header in headers:
                score += points
                result.findings.append({
                    "header": name, "status": "PRESENT",
                    "value": headers[header][:60]
                })

                # Additional HSTS checks
                if header == "strict-transport-security":
                    hsts_val = headers[header]
                    match = re.search(r"max-age=(\d+)", hsts_val)
                    if match:
                        max_age = int(match.group(1))
                        if max_age < 31536000:
                            result.findings.append({
                                "header": "HSTS max-age",
                                "status": "WARNING",
                                "value": f"max-age={max_age} (should be >= 31536000)"
                            })
                            score -= 5

                # CSP checks
                if header == "content-security-policy":
                    csp_val = headers[header]
                    if "unsafe-inline" in csp_val:
                        result.findings.append({
                            "header": "CSP unsafe-inline",
                            "status": "WARNING",
                            "value": "Contains 'unsafe-inline'"
                        })
                        score -= 10
                    if "unsafe-eval" in csp_val:
                        result.findings.append({
                            "header": "CSP unsafe-eval",
                            "status": "WARNING",
                            "value": "Contains 'unsafe-eval'"
                        })
                        score -= 10
            else:
                result.findings.append({
                    "header": name, "status": "MISSING", "value": ""
                })

        # Check for information leakage
        for leak_header in ["server", "x-powered-by"]:
            if leak_header in headers:
                result.findings.append({
                    "header": leak_header,
                    "status": "INFO_LEAK",
                    "value": headers[leak_header],
                })
                score -= 5

        # COEP/COOP checks
        if "cross-origin-embedder-policy" in headers:
            score += 5
        if "cross-origin-opener-policy" in headers:
            score += 5

    except Exception as e:
        result.findings.append({"header": "CONNECTION", "status": "ERROR",
                                "value": str(e)})

    # Assign grade
    result.score = max(0, score)
    if score >= 90:
        result.grade = "A+"
    elif score >= 80:
        result.grade = "A"
    elif score >= 65:
        result.grade = "B"
    elif score >= 50:
        result.grade = "C"
    elif score >= 35:
        result.grade = "D"
    else:
        result.grade = "F"

    return result


def exercise_3_header_scanner():
    """Scan security headers for a website."""
    hosts = ["google.com", "github.com"]
    for host in hosts:
        result = scan_security_headers(host)
        print(f"\n{result.url}: Grade {result.grade} (Score: {result.score})")
        for f in result.findings:
            status_icon = {"PRESENT": "+", "MISSING": "-",
                           "WARNING": "!", "INFO_LEAK": "~",
                           "ERROR": "X"}.get(f["status"], "?")
            print(f"  [{status_icon}] {f['header']}: {f.get('value', '')}")


# ---------------------------------------------------------------------------
# Exercise 4: SRI Hash Generator
# ---------------------------------------------------------------------------

def generate_sri_hash(content: bytes) -> str:
    """Generate an SRI integrity hash (SHA-384)."""
    digest = hashlib.sha384(content).digest()
    return "sha384-" + base64.b64encode(digest).decode()


def exercise_4_sri_generator():
    """Generate SRI hashes for CDN resources."""
    # Simulated CDN resources (in production, fetch via HTTP)
    resources = [
        {
            "url": "https://cdn.example.com/js/app.min.js",
            "content": b"(function(){console.log('app')})();",
            "tag_type": "script",
        },
        {
            "url": "https://cdn.example.com/css/style.min.css",
            "content": b"body{margin:0;font-family:sans-serif}",
            "tag_type": "link",
        },
    ]

    print("SRI Hash Generator")
    print("=" * 60)
    html_tags = []

    for res in resources:
        integrity = generate_sri_hash(res["content"])

        if res["tag_type"] == "script":
            tag = (f'<script src="{res["url"]}" '
                   f'integrity="{integrity}" '
                   f'crossorigin="anonymous"></script>')
        else:
            tag = (f'<link rel="stylesheet" href="{res["url"]}" '
                   f'integrity="{integrity}" '
                   f'crossorigin="anonymous">')

        html_tags.append(tag)
        print(f"\n  URL: {res['url']}")
        print(f"  Integrity: {integrity}")
        print(f"  Tag: {tag}")

    return html_tags


# ---------------------------------------------------------------------------
# Exercise 5: HSTS Preload Readiness Check
# ---------------------------------------------------------------------------

@dataclass
class HSTSReadiness:
    domain: str
    has_valid_cert: bool = False
    redirects_to_https: bool = False
    has_hsts: bool = False
    max_age_sufficient: bool = False
    has_include_subdomains: bool = False
    has_preload: bool = False
    issues: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return all([
            self.has_valid_cert,
            self.has_hsts,
            self.max_age_sufficient,
            self.has_include_subdomains,
            self.has_preload,
        ])


def check_hsts_preload(domain: str) -> HSTSReadiness:
    """Check if a domain is ready for HSTS preload submission."""
    result = HSTSReadiness(domain=domain)

    try:
        # Check TLS certificate
        context = ssl.create_default_context()
        with socket.create_connection((domain, 443), timeout=10) as sock:
            with context.wrap_socket(sock, server_hostname=domain) as ssock:
                result.has_valid_cert = True
    except ssl.SSLError as e:
        result.issues.append(f"TLS error: {e}")
    except Exception as e:
        result.issues.append(f"Connection error: {e}")

    # Check HSTS header
    try:
        import http.client
        conn = http.client.HTTPSConnection(domain, timeout=10)
        conn.request("HEAD", "/")
        resp = conn.getresponse()
        hsts = resp.getheader("strict-transport-security", "")
        conn.close()

        if hsts:
            result.has_hsts = True

            # Check max-age
            match = re.search(r"max-age=(\d+)", hsts)
            if match:
                max_age = int(match.group(1))
                if max_age >= 31536000:
                    result.max_age_sufficient = True
                else:
                    result.issues.append(
                        f"max-age={max_age} (need >= 31536000)"
                    )

            # Check includeSubDomains
            if "includesubdomains" in hsts.lower():
                result.has_include_subdomains = True
            else:
                result.issues.append("Missing includeSubDomains")

            # Check preload
            if "preload" in hsts.lower():
                result.has_preload = True
            else:
                result.issues.append("Missing preload directive")
        else:
            result.issues.append("No HSTS header found")

    except Exception as e:
        result.issues.append(f"HTTP error: {e}")

    return result


def exercise_5_hsts_preload():
    """Check HSTS preload readiness for domains."""
    domains = ["google.com", "github.com"]
    for domain in domains:
        result = check_hsts_preload(domain)
        status = "READY" if result.ready else "NOT READY"
        print(f"\n{domain}: {status}")
        print(f"  Valid cert: {result.has_valid_cert}")
        print(f"  HSTS present: {result.has_hsts}")
        print(f"  max-age >= 1yr: {result.max_age_sufficient}")
        print(f"  includeSubDomains: {result.has_include_subdomains}")
        print(f"  preload: {result.has_preload}")
        if result.issues:
            for issue in result.issues:
                print(f"  Issue: {issue}")


# ---------------------------------------------------------------------------
# Exercise 6: Cross-Origin Isolation Audit
# ---------------------------------------------------------------------------

def exercise_6_cross_origin_audit():
    """
    Explain cross-origin isolation requirements and auditing approach.
    """
    print("Cross-Origin Isolation Audit")
    print("=" * 60)

    print("""
Cross-origin isolation requires three headers working together:

1. Cross-Origin-Embedder-Policy (COEP): require-corp
   - Forces all cross-origin resources to explicitly opt in
   - Resources must set Cross-Origin-Resource-Policy (CORP)
   - Or be loaded with crossorigin="anonymous" with proper CORS

2. Cross-Origin-Opener-Policy (COOP): same-origin
   - Isolates the browsing context from cross-origin popups
   - Prevents cross-origin window references

3. Cross-Origin-Resource-Policy (CORP): same-origin | cross-origin
   - Set on individual resources to allow/deny cross-origin loading

Audit Steps:
  1. Check if COEP is set on the main page
  2. Check if COOP is set on the main page
  3. Identify all cross-origin resources (scripts, images, fonts, etc.)
  4. For each cross-origin resource, check if CORP is set
  5. Report which resources would break under COEP: require-corp

Benefits of cross-origin isolation:
  - Enables SharedArrayBuffer (required for some WASM workloads)
  - Prevents Spectre-like side-channel attacks
  - Stronger security boundary between origins

Common resources that may break:
  - Third-party analytics scripts (Google Analytics)
  - CDN-hosted fonts (Google Fonts)
  - Social media embeds (Twitter, YouTube)
  - Ad network scripts
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Exercise 1: CSP Policy Analysis")
    print("=" * 70)
    exercise_1_csp_analysis()

    print("\n" + "=" * 70)
    print("Exercise 2: Security Headers Middleware")
    print("=" * 70)
    exercise_2_security_headers_middleware()

    print("\n" + "=" * 70)
    print("Exercise 3: Header Scanner")
    print("=" * 70)
    exercise_3_header_scanner()

    print("\n" + "=" * 70)
    print("Exercise 4: SRI Hash Generator")
    print("=" * 70)
    exercise_4_sri_generator()

    print("\n" + "=" * 70)
    print("Exercise 5: HSTS Preload Readiness")
    print("=" * 70)
    exercise_5_hsts_preload()

    print("\n" + "=" * 70)
    print("Exercise 6: Cross-Origin Isolation Audit")
    print("=" * 70)
    exercise_6_cross_origin_audit()
