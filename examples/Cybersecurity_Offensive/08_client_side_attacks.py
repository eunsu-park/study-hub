"""
Example: Client-Side Attacks
==============================
DOM XSS sink identification, clickjacking check, open redirect analysis,
and CSP policy evaluator.

IMPORTANT: For authorized security testing and CTF only.
"""

from dataclasses import dataclass, field
from urllib.parse import urlparse, parse_qs


# ---------------------------------------------------------------------------
# DOM XSS Sinks and Sources
# ---------------------------------------------------------------------------

DOM_SOURCES = [
    "document.URL", "document.documentURI", "document.referrer",
    "location.href", "location.search", "location.hash",
    "window.name", "postMessage data",
]

DOM_SINKS = {
    "innerHTML": {"type": "html_injection", "risk": "high"},
    "outerHTML": {"type": "html_injection", "risk": "high"},
    "document.write": {"type": "html_injection", "risk": "high"},
    "document.writeln": {"type": "html_injection", "risk": "high"},
    "eval": {"type": "code_execution", "risk": "critical"},
    "setTimeout(string)": {"type": "code_execution", "risk": "critical"},
    "setInterval(string)": {"type": "code_execution", "risk": "critical"},
    "Function()": {"type": "code_execution", "risk": "critical"},
    "location.href": {"type": "redirect", "risk": "medium"},
    "location.assign": {"type": "redirect", "risk": "medium"},
    "textContent": {"type": "safe", "risk": "none"},
    "innerText": {"type": "safe", "risk": "none"},
}


# ---------------------------------------------------------------------------
# Clickjacking Protection Check
# ---------------------------------------------------------------------------

@dataclass
class ClickjackResult:
    url: str
    protected: bool
    method: str
    details: str


def check_clickjacking(url: str, headers: dict[str, str]) -> ClickjackResult:
    """Check if response headers protect against clickjacking."""
    xfo = headers.get("X-Frame-Options", "").upper()
    csp = headers.get("Content-Security-Policy", "")

    if "frame-ancestors" in csp:
        ancestors = csp.split("frame-ancestors")[1].split(";")[0].strip()
        if "'none'" in ancestors or "'self'" in ancestors:
            return ClickjackResult(url, True, "CSP frame-ancestors", ancestors)

    if xfo in ("DENY", "SAMEORIGIN"):
        return ClickjackResult(url, True, "X-Frame-Options", xfo)

    if xfo.startswith("ALLOW-FROM"):
        return ClickjackResult(url, False, "X-Frame-Options (weak)",
                               "ALLOW-FROM is deprecated and ignored by modern browsers")

    return ClickjackResult(url, False, "None", "No clickjacking protection found")


# ---------------------------------------------------------------------------
# Open Redirect Detection
# ---------------------------------------------------------------------------

def detect_open_redirect(url: str) -> dict:
    """Analyze URL for open redirect vulnerability."""
    parsed = urlparse(url)
    params = parse_qs(parsed.query)

    redirect_params = ["next", "url", "redirect", "return", "returnTo",
                       "goto", "continue", "dest", "destination"]
    findings = []

    for param in redirect_params:
        values = params.get(param, [])
        for val in values:
            if val.startswith("http://") or val.startswith("https://"):
                findings.append({"param": param, "value": val,
                                 "type": "absolute_url", "vulnerable": True})
            elif val.startswith("//"):
                findings.append({"param": param, "value": val,
                                 "type": "protocol_relative", "vulnerable": True})
            elif val.startswith("/\\") or val.startswith("/\t"):
                findings.append({"param": param, "value": val,
                                 "type": "backslash_bypass", "vulnerable": True})
            elif val.startswith("/"):
                findings.append({"param": param, "value": val,
                                 "type": "relative_path", "vulnerable": False})

    return {"url": url, "findings": findings,
            "vulnerable": any(f["vulnerable"] for f in findings)}


# ---------------------------------------------------------------------------
# CSP Evaluator
# ---------------------------------------------------------------------------

CSP_WEAKNESSES = {
    "'unsafe-inline'": "Allows inline scripts, defeats XSS protection",
    "'unsafe-eval'": "Allows eval(), reduces protection",
    "*": "Wildcard allows any source",
    "data:": "Allows data: URIs for script injection",
    "http:": "Allows any HTTP source (no HTTPS requirement)",
}


def evaluate_csp(policy: str) -> dict:
    """Evaluate a CSP header for weaknesses."""
    directives = {}
    for part in policy.split(";"):
        part = part.strip()
        if not part:
            continue
        tokens = part.split()
        directives[tokens[0]] = tokens[1:] if len(tokens) > 1 else []

    weaknesses = []
    for directive, values in directives.items():
        for val in values:
            if val in CSP_WEAKNESSES:
                weaknesses.append(f"{directive}: {val} - {CSP_WEAKNESSES[val]}")

    has_script = "script-src" in directives or "default-src" in directives
    return {
        "directives": directives,
        "weaknesses": weaknesses,
        "has_script_policy": has_script,
        "overall_rating": "weak" if weaknesses else "strong",
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    print("Client-Side Attacks Examples")
    print("=" * 50)

    print("\nDOM XSS Dangerous Sinks:")
    for sink, info in DOM_SINKS.items():
        if info["risk"] != "none":
            print(f"  [{info['risk']:8s}] {sink} ({info['type']})")

    print("\nClickjacking Check:")
    tests = [
        ("https://bank.com", {}),
        ("https://shop.com", {"X-Frame-Options": "DENY"}),
        ("https://app.com", {"Content-Security-Policy": "frame-ancestors 'none'"}),
    ]
    for url, headers in tests:
        result = check_clickjacking(url, headers)
        status = "PROTECTED" if result.protected else "VULNERABLE"
        print(f"  {url}: {status} ({result.method})")

    print("\nCSP Evaluation:")
    policies = [
        "default-src 'self'; script-src 'unsafe-inline'",
        "default-src 'self'; script-src 'self'; style-src 'self'",
    ]
    for p in policies:
        result = evaluate_csp(p)
        print(f"  Policy: {p}")
        print(f"    Rating: {result['overall_rating']}, Weaknesses: {len(result['weaknesses'])}")


if __name__ == "__main__":
    demo()
