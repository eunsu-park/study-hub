"""
HTTP Security Headers Checker

Demonstrates:
- Common security headers and their purpose
- Header parsing and validation
- Security score calculation
- Best-practice recommendations

Theory:
- HTTP security headers instruct browsers to enforce security policies.
- Content-Security-Policy (CSP): restricts resource loading sources,
  mitigates XSS and data injection attacks.
- Strict-Transport-Security (HSTS): forces HTTPS connections,
  prevents protocol downgrade attacks.
- X-Content-Type-Options: prevents MIME-type sniffing.
- X-Frame-Options: prevents clickjacking by controlling framing.
- Referrer-Policy: controls referrer information leakage.
- Permissions-Policy: restricts browser feature access (camera, mic).

Adapted from Security Lesson 09.
"""

from dataclasses import dataclass


@dataclass
class HeaderCheck:
    name: str
    present: bool
    value: str
    severity: str       # "critical", "high", "medium", "low"
    recommendation: str
    score: int          # Points if present (0-10)


SECURITY_HEADERS = {
    "Strict-Transport-Security": {
        "severity": "critical",
        "score": 10,
        "recommended": "max-age=31536000; includeSubDomains; preload",
        "description": "Forces HTTPS, prevents downgrade attacks",
    },
    "Content-Security-Policy": {
        "severity": "critical",
        "score": 10,
        "recommended": "default-src 'self'; script-src 'self'",
        "description": "Restricts resource loading, mitigates XSS",
    },
    "X-Content-Type-Options": {
        "severity": "high",
        "score": 5,
        "recommended": "nosniff",
        "description": "Prevents MIME-type sniffing",
    },
    "X-Frame-Options": {
        "severity": "high",
        "score": 5,
        "recommended": "DENY",
        "description": "Prevents clickjacking via iframes",
    },
    "Referrer-Policy": {
        "severity": "medium",
        "score": 5,
        "recommended": "strict-origin-when-cross-origin",
        "description": "Controls referrer information leakage",
    },
    "Permissions-Policy": {
        "severity": "medium",
        "score": 5,
        "recommended": "camera=(), microphone=(), geolocation=()",
        "description": "Restricts browser feature access",
    },
    "X-XSS-Protection": {
        "severity": "low",
        "score": 3,
        "recommended": "0",
        "description": "Deprecated — CSP is preferred. Set to 0 to disable.",
    },
    "Cross-Origin-Opener-Policy": {
        "severity": "medium",
        "score": 5,
        "recommended": "same-origin",
        "description": "Isolates browsing context from cross-origin windows",
    },
    "Cross-Origin-Resource-Policy": {
        "severity": "medium",
        "score": 5,
        "recommended": "same-origin",
        "description": "Restricts cross-origin resource loading",
    },
}


class SecurityHeaderChecker:
    """Analyze HTTP response headers for security best practices."""

    def __init__(self, headers: dict[str, str]):
        self.headers = {k.lower(): v for k, v in headers.items()}
        self.results: list[HeaderCheck] = []

    def check_all(self) -> list[HeaderCheck]:
        self.results = []
        for name, info in SECURITY_HEADERS.items():
            value = self.headers.get(name.lower(), "")
            present = bool(value)
            if present:
                recommendation = "Present — verify value is correct"
            else:
                recommendation = f"Add: {name}: {info['recommended']}"

            self.results.append(HeaderCheck(
                name=name,
                present=present,
                value=value if present else "(missing)",
                severity=info["severity"],
                recommendation=recommendation,
                score=info["score"] if present else 0,
            ))
        return self.results

    def check_dangerous_headers(self) -> list[str]:
        """Check for headers that should NOT be present."""
        dangerous = []
        if "server" in self.headers:
            dangerous.append(
                f"Server: {self.headers['server']} — reveals server software")
        if "x-powered-by" in self.headers:
            dangerous.append(
                f"X-Powered-By: {self.headers['x-powered-by']} "
                f"— reveals technology stack")
        if "x-aspnet-version" in self.headers:
            dangerous.append("X-AspNet-Version — reveals framework version")
        return dangerous

    def get_score(self) -> tuple[int, int]:
        """Return (achieved, maximum) score."""
        if not self.results:
            self.check_all()
        achieved = sum(r.score for r in self.results)
        maximum = sum(info["score"] for info in SECURITY_HEADERS.values())
        return achieved, maximum

    def get_grade(self) -> str:
        achieved, maximum = self.get_score()
        pct = (achieved / maximum) * 100 if maximum else 0
        if pct >= 90:
            return "A"
        elif pct >= 75:
            return "B"
        elif pct >= 60:
            return "C"
        elif pct >= 40:
            return "D"
        return "F"


def parse_csp(csp: str) -> dict[str, list[str]]:
    """Parse Content-Security-Policy into directive map."""
    directives: dict[str, list[str]] = {}
    for part in csp.split(";"):
        part = part.strip()
        if not part:
            continue
        tokens = part.split()
        if tokens:
            directives[tokens[0]] = tokens[1:]
    return directives


def validate_csp(csp: str) -> list[str]:
    """Check CSP for common weaknesses."""
    issues = []
    directives = parse_csp(csp)

    if "default-src" not in directives:
        issues.append("Missing 'default-src' fallback directive")

    for directive, sources in directives.items():
        if "'unsafe-inline'" in sources:
            issues.append(
                f"'{directive}' allows 'unsafe-inline' — XSS risk")
        if "'unsafe-eval'" in sources:
            issues.append(
                f"'{directive}' allows 'unsafe-eval' — code injection risk")
        if "*" in sources:
            issues.append(
                f"'{directive}' uses wildcard '*' — too permissive")
        if "http:" in sources:
            issues.append(
                f"'{directive}' allows 'http:' — mixed content risk")

    return issues


def validate_hsts(hsts: str) -> list[str]:
    """Check HSTS header for best practices."""
    issues = []
    parts = [p.strip().lower() for p in hsts.split(";")]

    max_age = None
    for part in parts:
        if part.startswith("max-age="):
            try:
                max_age = int(part.split("=")[1])
            except ValueError:
                issues.append("Invalid max-age value")

    if max_age is None:
        issues.append("Missing max-age directive")
    elif max_age < 31536000:
        issues.append(f"max-age={max_age} is less than 1 year (31536000)")

    if "includesubdomains" not in parts:
        issues.append("Missing includeSubDomains — subdomains not protected")

    if "preload" not in parts:
        issues.append("Missing preload — not eligible for HSTS preload list")

    return issues


# ── Demos ─────────────────────────────────────────────────────────────

def demo_good_headers():
    print("=" * 60)
    print("WELL-CONFIGURED SECURITY HEADERS")
    print("=" * 60)

    headers = {
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
        "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'",
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
        "X-XSS-Protection": "0",
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Resource-Policy": "same-origin",
    }

    checker = SecurityHeaderChecker(headers)
    results = checker.check_all()

    for r in results:
        status = "PASS" if r.present else "FAIL"
        print(f"  [{status}] {r.name}")
        if r.present:
            print(f"        Value: {r.value}")

    score, maximum = checker.get_score()
    grade = checker.get_grade()
    print(f"\n  Score: {score}/{maximum} (Grade: {grade})")


def demo_poor_headers():
    print("\n" + "=" * 60)
    print("POORLY-CONFIGURED HEADERS")
    print("=" * 60)

    headers = {
        "X-Content-Type-Options": "nosniff",
        "Server": "Apache/2.4.41 (Ubuntu)",
        "X-Powered-By": "Express",
    }

    checker = SecurityHeaderChecker(headers)
    results = checker.check_all()

    print("\n  Missing headers:")
    for r in results:
        if not r.present:
            print(f"  [{r.severity.upper():>8}] {r.name}")
            print(f"            → {r.recommendation}")

    dangerous = checker.check_dangerous_headers()
    if dangerous:
        print("\n  Information leakage:")
        for d in dangerous:
            print(f"    ⚠ {d}")

    score, maximum = checker.get_score()
    grade = checker.get_grade()
    print(f"\n  Score: {score}/{maximum} (Grade: {grade})")


def demo_csp_analysis():
    print("\n" + "=" * 60)
    print("CSP ANALYSIS")
    print("=" * 60)

    policies = [
        ("Strict CSP",
         "default-src 'self'; script-src 'self'; style-src 'self'; "
         "img-src 'self' data:; font-src 'self'"),
        ("Permissive CSP",
         "default-src *; script-src * 'unsafe-inline' 'unsafe-eval'; "
         "style-src * 'unsafe-inline'"),
        ("Mixed content CSP",
         "default-src https:; script-src https: http:; "
         "img-src *"),
    ]

    for name, csp in policies:
        print(f"\n  {name}:")
        print(f"    Policy: {csp[:60]}...")
        directives = parse_csp(csp)
        print(f"    Directives: {list(directives.keys())}")
        issues = validate_csp(csp)
        if issues:
            print(f"    Issues:")
            for issue in issues:
                print(f"      - {issue}")
        else:
            print(f"    No issues found")


def demo_hsts_analysis():
    print("\n" + "=" * 60)
    print("HSTS ANALYSIS")
    print("=" * 60)

    values = [
        ("Complete", "max-age=31536000; includeSubDomains; preload"),
        ("Short max-age", "max-age=86400"),
        ("No preload", "max-age=31536000; includeSubDomains"),
    ]

    for name, value in values:
        print(f"\n  {name}: {value}")
        issues = validate_hsts(value)
        if issues:
            for issue in issues:
                print(f"    - {issue}")
        else:
            print(f"    All checks passed")


if __name__ == "__main__":
    demo_good_headers()
    demo_poor_headers()
    demo_csp_analysis()
    demo_hsts_analysis()
