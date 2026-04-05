"""
Dockerfile Security Linter

Demonstrates:
- Common Dockerfile security anti-patterns
- Static analysis of Dockerfiles
- Security scoring and recommendations
- Best practice enforcement

Theory:
- Container images inherit all vulnerabilities of their base image.
  Use minimal bases (alpine, distroless) to reduce attack surface.
- Running as root inside containers is a major risk.
  Always use USER directive with a non-root user.
- Secrets (API keys, passwords) should never be in image layers.
  Use build secrets or runtime injection instead.
- Pinning versions (base image tags, package versions) ensures
  reproducible and auditable builds.
- Multi-stage builds reduce final image size and attack surface.

Adapted from Security Lesson 12.
"""

from dataclasses import dataclass
import re


@dataclass
class Finding:
    line_number: int
    severity: str       # "critical", "high", "medium", "low", "info"
    rule_id: str
    message: str
    fix: str


class DockerfileLinter:
    """Security-focused Dockerfile linter."""

    def __init__(self, content: str):
        self.lines = content.strip().split("\n")
        self.findings: list[Finding] = []

    def lint(self) -> list[Finding]:
        self.findings = []
        for i, line in enumerate(self.lines, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            self._check_root_user(i, stripped)
            self._check_latest_tag(i, stripped)
            self._check_secrets(i, stripped)
            self._check_add_vs_copy(i, stripped)
            self._check_apt_cleanup(i, stripped)
            self._check_curl_wget(i, stripped)
            self._check_sudo(i, stripped)
            self._check_healthcheck(i, stripped)

        self._check_global_rules()
        return self.findings

    def _check_root_user(self, line_num: int, line: str) -> None:
        if line.upper().startswith("USER") and "root" in line.lower():
            self.findings.append(Finding(
                line_number=line_num,
                severity="critical",
                rule_id="SEC001",
                message="Container runs as root user",
                fix="USER nonroot:nonroot",
            ))

    def _check_latest_tag(self, line_num: int, line: str) -> None:
        if line.upper().startswith("FROM"):
            image = line.split()[1] if len(line.split()) > 1 else ""
            if ":" not in image or image.endswith(":latest"):
                self.findings.append(Finding(
                    line_number=line_num,
                    severity="high",
                    rule_id="SEC002",
                    message=f"Unpinned base image: {image}",
                    fix="Pin to a specific version, e.g., python:3.12-slim",
                ))

    def _check_secrets(self, line_num: int, line: str) -> None:
        secret_patterns = [
            (r"(?i)(password|passwd|pwd)\s*=", "password"),
            (r"(?i)(api[_-]?key|apikey)\s*=", "API key"),
            (r"(?i)(secret|token)\s*=", "secret/token"),
            (r"(?i)AWS_(ACCESS_KEY|SECRET)", "AWS credential"),
        ]
        for pattern, name in secret_patterns:
            if re.search(pattern, line):
                self.findings.append(Finding(
                    line_number=line_num,
                    severity="critical",
                    rule_id="SEC003",
                    message=f"Possible {name} in image layer",
                    fix="Use --mount=type=secret or runtime env vars",
                ))

    def _check_add_vs_copy(self, line_num: int, line: str) -> None:
        if line.upper().startswith("ADD") and "http" not in line.lower():
            self.findings.append(Finding(
                line_number=line_num,
                severity="medium",
                rule_id="SEC004",
                message="ADD used instead of COPY for local files",
                fix="Use COPY unless you need ADD's tar extraction",
            ))

    def _check_apt_cleanup(self, line_num: int, line: str) -> None:
        if "apt-get install" in line and "rm -rf /var/lib/apt" not in line:
            if "&&" not in line or "apt-get clean" not in line:
                self.findings.append(Finding(
                    line_number=line_num,
                    severity="medium",
                    rule_id="SEC005",
                    message="apt-get install without cleanup",
                    fix="Add && apt-get clean && rm -rf /var/lib/apt/lists/*",
                ))

    def _check_curl_wget(self, line_num: int, line: str) -> None:
        if ("curl" in line or "wget" in line) and "| sh" in line:
            self.findings.append(Finding(
                line_number=line_num,
                severity="high",
                rule_id="SEC006",
                message="Piping remote script directly to shell",
                fix="Download, verify checksum, then execute separately",
            ))

    def _check_sudo(self, line_num: int, line: str) -> None:
        if "sudo" in line.lower().split():
            self.findings.append(Finding(
                line_number=line_num,
                severity="medium",
                rule_id="SEC007",
                message="sudo used in Dockerfile (runs as root already)",
                fix="Remove sudo — RUN commands execute as current USER",
            ))

    def _check_healthcheck(self, line_num: int, line: str) -> None:
        if line.upper().startswith("HEALTHCHECK NONE"):
            self.findings.append(Finding(
                line_number=line_num,
                severity="low",
                rule_id="SEC008",
                message="HEALTHCHECK explicitly disabled",
                fix="Add a meaningful HEALTHCHECK for container orchestration",
            ))

    def _check_global_rules(self) -> None:
        has_user = any(l.strip().upper().startswith("USER")
                       for l in self.lines
                       if not l.strip().startswith("#"))
        if not has_user:
            self.findings.append(Finding(
                line_number=0,
                severity="high",
                rule_id="SEC009",
                message="No USER directive — container runs as root",
                fix="Add USER nonroot before CMD/ENTRYPOINT",
            ))

        has_healthcheck = any(l.strip().upper().startswith("HEALTHCHECK")
                              for l in self.lines
                              if not l.strip().startswith("#"))
        if not has_healthcheck:
            self.findings.append(Finding(
                line_number=0,
                severity="low",
                rule_id="SEC010",
                message="No HEALTHCHECK directive",
                fix="Add HEALTHCHECK CMD curl -f http://localhost/ || exit 1",
            ))

    def get_score(self) -> int:
        """Calculate security score (100 = perfect, 0 = worst)."""
        if not self.findings:
            self.lint()
        deductions = {
            "critical": 25,
            "high": 15,
            "medium": 8,
            "low": 3,
            "info": 0,
        }
        total = sum(deductions.get(f.severity, 0) for f in self.findings)
        return max(0, 100 - total)


# ── Demos ─────────────────────────────────────────────────────────────

GOOD_DOCKERFILE = """\
# Multi-stage build
FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /install /usr/local
COPY . .

RUN groupadd -r appuser && useradd -r -g appuser appuser
USER appuser:appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD curl -f http://localhost:8000/health || exit 1
CMD ["python", "app.py"]
"""

BAD_DOCKERFILE = """\
FROM python
ADD . /app
WORKDIR /app
ENV API_KEY=sk-12345-secret-key
ENV DB_PASSWORD=mysecretpassword
RUN apt-get update && apt-get install -y curl vim netcat
RUN curl https://example.com/setup.sh | sh
RUN sudo chmod 777 /app
CMD ["python", "app.py"]
"""

MODERATE_DOCKERFILE = """\
FROM python:3.12
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN groupadd -r app && useradd -r -g app app
USER app
EXPOSE 8000
CMD ["python", "app.py"]
"""


def print_findings(findings: list[Finding]) -> None:
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    sorted_findings = sorted(findings, key=lambda f: severity_order.get(f.severity, 5))

    for f in sorted_findings:
        loc = f"L{f.line_number}" if f.line_number else "Global"
        print(f"  [{f.severity.upper():>8}] {f.rule_id} ({loc}): {f.message}")
        print(f"            Fix: {f.fix}")


def demo_good_dockerfile():
    print("=" * 60)
    print("SECURE DOCKERFILE")
    print("=" * 60)

    print("\n  Dockerfile:")
    for i, line in enumerate(GOOD_DOCKERFILE.strip().split("\n"), 1):
        print(f"    {i:>2}| {line}")

    linter = DockerfileLinter(GOOD_DOCKERFILE)
    findings = linter.lint()

    print(f"\n  Findings ({len(findings)}):")
    if findings:
        print_findings(findings)
    else:
        print("  None — all checks passed!")

    print(f"\n  Security Score: {linter.get_score()}/100")


def demo_bad_dockerfile():
    print("\n" + "=" * 60)
    print("INSECURE DOCKERFILE")
    print("=" * 60)

    print("\n  Dockerfile:")
    for i, line in enumerate(BAD_DOCKERFILE.strip().split("\n"), 1):
        print(f"    {i:>2}| {line}")

    linter = DockerfileLinter(BAD_DOCKERFILE)
    findings = linter.lint()

    print(f"\n  Findings ({len(findings)}):")
    print_findings(findings)

    print(f"\n  Security Score: {linter.get_score()}/100")


def demo_moderate_dockerfile():
    print("\n" + "=" * 60)
    print("MODERATE DOCKERFILE")
    print("=" * 60)

    print("\n  Dockerfile:")
    for i, line in enumerate(MODERATE_DOCKERFILE.strip().split("\n"), 1):
        print(f"    {i:>2}| {line}")

    linter = DockerfileLinter(MODERATE_DOCKERFILE)
    findings = linter.lint()

    print(f"\n  Findings ({len(findings)}):")
    if findings:
        print_findings(findings)
    else:
        print("  None")

    print(f"\n  Security Score: {linter.get_score()}/100")


def demo_best_practices():
    print("\n" + "=" * 60)
    print("CONTAINER SECURITY BEST PRACTICES")
    print("=" * 60)

    print(f"""
  Build Phase:
    1. Use minimal base images (alpine, distroless, slim)
    2. Pin base image versions and package versions
    3. Use multi-stage builds to minimize final image
    4. Don't install unnecessary packages (curl, vim, netcat)
    5. Use COPY instead of ADD for local files

  Runtime Phase:
    6. Never run as root — always set USER
    7. Never embed secrets in ENV or COPY
    8. Use read-only filesystems where possible
    9. Drop all capabilities, add only what's needed
   10. Set resource limits (memory, CPU)

  Supply Chain:
    11. Scan images for CVEs (Trivy, Grype)
    12. Sign images (Cosign, Notary)
    13. Use private registries with access control
    14. Regularly rebuild to pick up security patches

  Dockerfile Security Checklist:
    {'Rule':<35} {'Impact':<12}
    {'-'*35} {'-'*12}
    {'Pin base image version':<35} {'High':<12}
    {'Run as non-root user':<35} {'Critical':<12}
    {'No secrets in layers':<35} {'Critical':<12}
    {'Multi-stage build':<35} {'Medium':<12}
    {'Clean up package caches':<35} {'Low':<12}
    {'Use COPY over ADD':<35} {'Medium':<12}
    {'Add HEALTHCHECK':<35} {'Low':<12}
    {'Minimize installed packages':<35} {'Medium':<12}""")


if __name__ == "__main__":
    demo_good_dockerfile()
    demo_bad_dockerfile()
    demo_moderate_dockerfile()
    demo_best_practices()
