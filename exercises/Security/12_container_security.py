"""
Exercise Solutions: Container and Cloud Security
================================================
Lesson 12 from Security topic.

Covers secure Dockerfiles, Kubernetes hardening, container scanning,
IAM auditing, Terraform security scanning, and supply chain security.
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Exercise 1: Secure Dockerfile (Fixed Version)
# ---------------------------------------------------------------------------

def exercise_1_secure_dockerfile():
    """
    Analyze the insecure Dockerfile and provide the secured version.
    """
    issues = [
        "FROM ubuntu:latest — unpinned base, bloated, includes unnecessary packages",
        "No multi-stage build — dev dependencies end up in production image",
        "No non-root user — container runs as root",
        "Secrets in ENV — DATABASE_URL and API_SECRET baked into image layers",
        "Exposes unnecessary ports (22, 5432) — SSH and direct DB access",
        "No health check — orchestrator cannot determine container health",
        "No .dockerignore — .git, __pycache__, .env may be copied",
        "No pinned package versions — pip install may pull breaking changes",
    ]

    print("Issues Found in Insecure Dockerfile:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")

    secure_dockerfile = '''
# ---- Build Stage ----
FROM python:3.12-slim AS builder

WORKDIR /build

# Install dependencies in a virtual environment
COPY requirements.txt .
RUN python -m venv /opt/venv && \\
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ---- Production Stage ----
FROM python:3.12-slim

# Pin the base image digest for reproducibility
# FROM python:3.12-slim@sha256:<digest>

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copy only the virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=appuser:appuser app.py .

# No secrets in the image — pass via environment at runtime
# ENV DATABASE_URL=...  <-- NEVER do this

# Only expose the application port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \\
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Run as non-root
USER appuser

# Read-only filesystem (writable volumes mounted at runtime)
# Use --read-only flag when running: docker run --read-only --tmpfs /tmp

CMD ["python", "app.py"]
'''

    dockerignore = '''
.git
.gitignore
__pycache__
*.pyc
.env
.env.local
*.md
tests/
docs/
.vscode/
'''

    print("\n--- Secure Dockerfile ---")
    print(secure_dockerfile)
    print("--- .dockerignore ---")
    print(dockerignore)


# ---------------------------------------------------------------------------
# Exercise 2: Kubernetes Security Manifests
# ---------------------------------------------------------------------------

def exercise_2_k8s_security():
    """Generate security-hardened Kubernetes manifests."""
    manifests = {
        "namespace": {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {
                "name": "secure-app",
                "labels": {
                    "pod-security.kubernetes.io/enforce": "restricted",
                    "pod-security.kubernetes.io/audit": "restricted",
                    "pod-security.kubernetes.io/warn": "restricted",
                },
            },
        },
        "serviceaccount": {
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": {
                "name": "secure-app-sa",
                "namespace": "secure-app",
            },
            "automountServiceAccountToken": False,
        },
        "deployment_spec": {
            "containers": [{
                "name": "web",
                "image": "myapp:1.0.0@sha256:abc123",
                "ports": [{"containerPort": 8080}],
                "securityContext": {
                    "runAsNonRoot": True,
                    "runAsUser": 1000,
                    "readOnlyRootFilesystem": True,
                    "allowPrivilegeEscalation": False,
                    "capabilities": {"drop": ["ALL"]},
                    "seccompProfile": {"type": "RuntimeDefault"},
                },
                "resources": {
                    "limits": {"cpu": "500m", "memory": "256Mi"},
                    "requests": {"cpu": "100m", "memory": "128Mi"},
                },
                "livenessProbe": {
                    "httpGet": {"path": "/health", "port": 8080},
                    "initialDelaySeconds": 10,
                },
                "readinessProbe": {
                    "httpGet": {"path": "/ready", "port": 8080},
                    "initialDelaySeconds": 5,
                },
            }],
        },
        "networkpolicy": {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {"name": "secure-app-netpol", "namespace": "secure-app"},
            "spec": {
                "podSelector": {"matchLabels": {"app": "secure-app"}},
                "policyTypes": ["Ingress", "Egress"],
                "ingress": [{
                    "from": [{"podSelector": {"matchLabels": {"role": "ingress"}}}],
                    "ports": [{"port": 8080, "protocol": "TCP"}],
                }],
                "egress": [
                    {
                        "to": [{"podSelector": {"matchLabels": {"app": "api-service"}}}],
                        "ports": [{"port": 8080, "protocol": "TCP"}],
                    },
                    {
                        "to": [],
                        "ports": [{"port": 53, "protocol": "UDP"},
                                  {"port": 53, "protocol": "TCP"}],
                    },
                ],
            },
        },
    }

    print("Kubernetes Security Manifests:")
    print("=" * 60)
    for name, manifest in manifests.items():
        print(f"\n--- {name} ---")
        print(json.dumps(manifest, indent=2))


# ---------------------------------------------------------------------------
# Exercise 3: Container Image Scanner (Simulated)
# ---------------------------------------------------------------------------

@dataclass
class VulnFinding:
    package: str
    installed_version: str
    fixed_version: str
    severity: str
    cve_id: str


def exercise_3_image_scanner():
    """Simulate container image vulnerability scanning."""
    # Simulated Trivy scan results
    findings = [
        VulnFinding("libssl3", "3.0.2-0ubuntu1.7", "3.0.2-0ubuntu1.12",
                     "CRITICAL", "CVE-2024-0727"),
        VulnFinding("libc6", "2.35-0ubuntu3.1", "2.35-0ubuntu3.6",
                     "HIGH", "CVE-2023-6246"),
        VulnFinding("python3.12", "3.12.0-1", "3.12.1-1",
                     "MEDIUM", "CVE-2023-27043"),
        VulnFinding("curl", "7.81.0-1ubuntu1.7", "7.81.0-1ubuntu1.16",
                     "HIGH", "CVE-2023-46218"),
    ]

    print("Container Image Scan Report")
    print("=" * 60)
    print(f"Image: myapp:1.0.0")
    print(f"Scanner: Trivy (simulated)")
    print(f"Total findings: {len(findings)}")

    critical = sum(1 for f in findings if f.severity == "CRITICAL")
    high = sum(1 for f in findings if f.severity == "HIGH")
    print(f"Critical: {critical}, High: {high}")

    for f in findings:
        print(f"\n  [{f.severity}] {f.cve_id}")
        print(f"  Package: {f.package} ({f.installed_version})")
        print(f"  Fixed in: {f.fixed_version}")

    # CI/CD gate decision
    gate_pass = critical == 0 and high == 0
    print(f"\nCI/CD Gate: {'PASS' if gate_pass else 'FAIL'}")
    if not gate_pass:
        print("  Recommendation: Update base image and rebuild")


# ---------------------------------------------------------------------------
# Exercise 4: Cloud IAM Audit Tool (Simulated)
# ---------------------------------------------------------------------------

def exercise_4_iam_audit():
    """Simulate an AWS IAM audit."""
    # Simulated IAM data
    users = [
        {"name": "alice", "mfa": True, "access_key_age_days": 30,
         "policies": ["ReadOnly"], "last_login_days": 2},
        {"name": "bob", "mfa": False, "access_key_age_days": 120,
         "policies": ["AdministratorAccess"], "last_login_days": 5},
        {"name": "ci-deploy", "mfa": False, "access_key_age_days": 200,
         "policies": ["PowerUser"], "last_login_days": 1},
        {"name": "old-contractor", "mfa": False, "access_key_age_days": 365,
         "policies": ["ReadOnly"], "last_login_days": 180},
    ]

    findings = []

    for user in users:
        # Check MFA
        if not user["mfa"]:
            findings.append({
                "user": user["name"], "severity": "HIGH",
                "finding": "MFA not enabled",
                "remediation": "Enable MFA immediately",
            })

        # Check access key age
        if user["access_key_age_days"] > 90:
            findings.append({
                "user": user["name"], "severity": "MEDIUM",
                "finding": f"Access key is {user['access_key_age_days']} days old",
                "remediation": "Rotate access key",
            })

        # Check for admin access
        if "AdministratorAccess" in user["policies"]:
            findings.append({
                "user": user["name"], "severity": "HIGH",
                "finding": "Has AdministratorAccess policy",
                "remediation": "Replace with least-privilege policy",
            })

        # Check for inactive users
        if user["last_login_days"] > 90:
            findings.append({
                "user": user["name"], "severity": "MEDIUM",
                "finding": f"Inactive for {user['last_login_days']} days",
                "remediation": "Disable or remove account",
            })

    print("AWS IAM Audit Report")
    print("=" * 60)
    for f in findings:
        print(f"  [{f['severity']}] {f['user']}: {f['finding']}")
        print(f"    Remediation: {f['remediation']}")


# ---------------------------------------------------------------------------
# Exercise 5: Terraform Security Scanner
# ---------------------------------------------------------------------------

class TerraformScanner:
    """Scan Terraform configurations for security issues."""

    def scan_content(self, content: str, filename: str) -> list[dict]:
        findings = []

        # Check 1: Open security groups
        if re.search(r'cidr_blocks\s*=\s*\["0\.0\.0\.0/0"\]', content):
            # Check if it's on sensitive ports
            for port in ["22", "3389", "5432", "3306"]:
                if f'from_port.*{port}' in content or f'to_port.*{port}' in content:
                    findings.append({
                        "file": filename, "severity": "CRITICAL",
                        "rule": f"Security group open to 0.0.0.0/0 on port {port}",
                        "fix": "Restrict to specific CIDR ranges",
                    })
                    break
            else:
                if "0.0.0.0/0" in content:
                    findings.append({
                        "file": filename, "severity": "HIGH",
                        "rule": "Security group open to 0.0.0.0/0",
                        "fix": "Restrict to specific CIDR ranges",
                    })

        # Check 2: Unencrypted S3
        if "aws_s3_bucket" in content and "server_side_encryption" not in content:
            findings.append({
                "file": filename, "severity": "HIGH",
                "rule": "S3 bucket without encryption",
                "fix": "Add server_side_encryption_configuration block",
            })

        # Check 3: S3 without public access block
        if "aws_s3_bucket" in content and "aws_s3_bucket_public_access_block" not in content:
            findings.append({
                "file": filename, "severity": "HIGH",
                "rule": "S3 bucket without public access block",
                "fix": "Add aws_s3_bucket_public_access_block resource",
            })

        # Check 4: Hardcoded credentials
        if re.search(r'(?:access_key|secret_key|password)\s*=\s*"[^"]*"', content):
            findings.append({
                "file": filename, "severity": "CRITICAL",
                "rule": "Hardcoded credentials in Terraform",
                "fix": "Use variables, environment variables, or a secrets manager",
            })

        # Check 5: Missing tags
        if "aws_instance" in content or "aws_s3_bucket" in content:
            if "Environment" not in content or "Owner" not in content:
                findings.append({
                    "file": filename, "severity": "LOW",
                    "rule": "Missing required tags (Environment, Owner)",
                    "fix": "Add required tags to all resources",
                })

        return findings


def exercise_5_terraform_scanner():
    """Demonstrate Terraform security scanning."""
    scanner = TerraformScanner()

    sample_tf = '''
resource "aws_security_group" "web" {
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_s3_bucket" "data" {
  bucket = "my-data-bucket"
}

resource "aws_db_instance" "main" {
  engine   = "postgres"
  password = "hardcoded_password_123"
}
'''

    findings = scanner.scan_content(sample_tf, "main.tf")

    print("Terraform Security Scan")
    print("=" * 60)
    for f in findings:
        print(f"  [{f['severity']}] {f['file']}: {f['rule']}")
        print(f"    Fix: {f['fix']}")


# ---------------------------------------------------------------------------
# Exercise 6: Supply Chain Security Pipeline (Design)
# ---------------------------------------------------------------------------

def exercise_6_supply_chain_pipeline():
    """Design a secure CI/CD pipeline for supply chain security."""
    pipeline_stages = [
        {
            "stage": "1. Source Code Scan (SAST)",
            "tools": "Bandit (Python), Semgrep (multi-language)",
            "gate": "No HIGH/CRITICAL findings",
            "output": "SARIF report",
        },
        {
            "stage": "2. Dependency Scan (SCA)",
            "tools": "pip-audit, safety, Snyk",
            "gate": "No known CVEs in CRITICAL/HIGH",
            "output": "CVE report with fixed versions",
        },
        {
            "stage": "3. Build Container Image",
            "tools": "Docker multi-stage build, BuildKit",
            "gate": "Build succeeds with --no-cache for release",
            "output": "Container image with pinned base",
        },
        {
            "stage": "4. Image Vulnerability Scan",
            "tools": "Trivy, Grype",
            "gate": "No CRITICAL vulnerabilities, HIGH < 5",
            "output": "Vulnerability report",
        },
        {
            "stage": "5. Image Signing",
            "tools": "cosign (Sigstore)",
            "gate": "Image signed with verified identity",
            "output": "Signature stored in registry",
        },
        {
            "stage": "6. SBOM Generation",
            "tools": "syft (SPDX/CycloneDX format)",
            "gate": "SBOM attached to image",
            "output": "SBOM artifact in SPDX format",
        },
        {
            "stage": "7. Admission Control",
            "tools": "Kyverno, OPA Gatekeeper",
            "gate": "Only signed images from allowed registries",
            "output": "Deployment approved/rejected",
        },
        {
            "stage": "8. Deploy",
            "tools": "Kubernetes, ArgoCD",
            "gate": "All previous gates passed",
            "output": "Running in production",
        },
    ]

    print("Supply Chain Security Pipeline")
    print("=" * 60)
    for stage in pipeline_stages:
        print(f"\n  {stage['stage']}")
        print(f"    Tools: {stage['tools']}")
        print(f"    Gate: {stage['gate']}")
        print(f"    Output: {stage['output']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Exercise 1: Secure Dockerfile")
    print("=" * 70)
    exercise_1_secure_dockerfile()

    print("\n" + "=" * 70)
    print("Exercise 2: Kubernetes Security Manifests")
    print("=" * 70)
    exercise_2_k8s_security()

    print("\n" + "=" * 70)
    print("Exercise 3: Container Image Scanner")
    print("=" * 70)
    exercise_3_image_scanner()

    print("\n" + "=" * 70)
    print("Exercise 4: IAM Audit")
    print("=" * 70)
    exercise_4_iam_audit()

    print("\n" + "=" * 70)
    print("Exercise 5: Terraform Security Scanner")
    print("=" * 70)
    exercise_5_terraform_scanner()

    print("\n" + "=" * 70)
    print("Exercise 6: Supply Chain Security Pipeline")
    print("=" * 70)
    exercise_6_supply_chain_pipeline()
