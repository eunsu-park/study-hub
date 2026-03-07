#!/bin/bash
# Exercises for Lesson 11: Security in DevOps (DevSecOps)
# Topic: DevOps
# Solutions to practice problems from the lesson.

# === Exercise 1: Secret Management ===
# Problem: Implement secret management patterns — never hardcode secrets,
# use vault or environment injection, and rotate credentials.
exercise_1() {
    echo "=== Exercise 1: Secret Management ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import os
import json
from dataclasses import dataclass

# BAD: Hardcoded secrets (NEVER do this)
# DATABASE_URL = "postgresql://admin:P@ssw0rd@db.prod.internal:5432/orders"
# API_KEY = "sk-live-abc123def456"

# GOOD: Read from environment variables
DATABASE_URL = os.environ.get("DATABASE_URL")
API_KEY = os.environ.get("STRIPE_API_KEY")

# BETTER: Use a secret manager with caching
@dataclass
class SecretManager:
    """Abstraction over vault/cloud secret managers."""
    backend: str  # "env", "vault", "aws_ssm", "gcp_secret_manager"
    cache_ttl_seconds: int = 300

    def get_secret(self, key: str) -> str:
        """Retrieve a secret from the configured backend."""
        if self.backend == "env":
            value = os.environ.get(key)
            if not value:
                raise ValueError(f"Secret '{key}' not found in environment")
            return value
        elif self.backend == "vault":
            # HashiCorp Vault: vault kv get -field=value secret/myapp/{key}
            # In production, use hvac Python client
            pass
        elif self.backend == "aws_ssm":
            # AWS Systems Manager Parameter Store
            # import boto3
            # ssm = boto3.client('ssm')
            # response = ssm.get_parameter(Name=key, WithDecryption=True)
            # return response['Parameter']['Value']
            pass
        return ""

# Secret management in Kubernetes:
# 1. Create secret: kubectl create secret generic db-creds --from-literal=password=xxx
# 2. Mount as env var in pod spec:
#    env:
#      - name: DB_PASSWORD
#        valueFrom:
#          secretKeyRef:
#            name: db-creds
#            key: password

# Secret management best practices:
practices = [
    "Never commit secrets to Git (use .gitignore, git-secrets, gitleaks)",
    "Rotate secrets on a schedule (90 days for passwords, 365 for API keys)",
    "Use short-lived credentials (IAM roles, OIDC tokens) over long-lived keys",
    "Encrypt secrets at rest (Vault auto-unseal, AWS KMS, SOPS)",
    "Audit secret access (who accessed what secret and when)",
    "Use Sealed Secrets or External Secrets Operator in Kubernetes",
]
for p in practices:
    print(f"  - {p}")
SOLUTION
}

# === Exercise 2: CI/CD Security Scanning ===
# Problem: Design a security scanning pipeline that includes SAST,
# dependency scanning, container scanning, and secret detection.
exercise_2() {
    echo "=== Exercise 2: CI/CD Security Scanning ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Security scanning stages in CI/CD:

# .github/workflows/security.yml
name: Security Scanning
on: [push, pull_request]

jobs:
  # Stage 1: Secret Detection (prevent leaked credentials)
  secrets:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0    # Full history for scanning all commits
      - name: Detect secrets with gitleaks
        uses: gitleaks/gitleaks-action@v2
        # Scans commit diffs for passwords, API keys, tokens
        # Configure: .gitleaks.toml for allowlists

  # Stage 2: SAST — Static Application Security Testing
  sast:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Bandit (Python SAST)
        run: |
          pip install bandit
          bandit -r src/ -f json -o bandit-report.json || true
          # Checks for: SQL injection, hardcoded passwords, eval(), etc.
      - name: Run Semgrep
        uses: returntocorp/semgrep-action@v1
        # Language-agnostic SAST with community rules

  # Stage 3: Dependency Scanning (known CVEs in libraries)
  dependencies:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Check Python dependencies
        run: |
          pip install pip-audit
          pip-audit -r requirements.txt --fix --dry-run
          # Checks PyPI packages against the OSV vulnerability database
      - name: SBOM Generation
        run: |
          pip install cyclonedx-bom
          cyclonedx-py requirements -i requirements.txt -o sbom.json
          # Software Bill of Materials — inventory of all dependencies

  # Stage 4: Container Scanning
  container:
    needs: [secrets, sast, dependencies]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: docker build -t myapp:scan .
      - name: Trivy container scan
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: myapp:scan
          severity: CRITICAL,HIGH
          exit-code: 1

# Scanning tool summary:
tools = {
    "gitleaks":   "Secret detection in Git history",
    "Bandit":     "Python SAST (security anti-patterns)",
    "Semgrep":    "Multi-language SAST with custom rules",
    "pip-audit":  "Python dependency vulnerability scanning",
    "Trivy":      "Container image + filesystem + IaC scanning",
    "Snyk":       "SCA + container + IaC (commercial + free tier)",
}

print("Security Scanning Tools:")
for tool, desc in tools.items():
    print(f"  {tool:12s}: {desc}")
SOLUTION
}

# === Exercise 3: Supply Chain Security ===
# Problem: Implement software supply chain security measures:
# SBOM generation, image signing, and dependency pinning.
exercise_3() {
    echo "=== Exercise 3: Supply Chain Security ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Supply chain attacks target the build/deploy pipeline, not the app itself.
# Examples: compromised npm packages, poisoned Docker base images, CI tampering.

supply_chain_controls = {
    "Dependency Pinning": {
        "what": "Lock exact versions of all dependencies",
        "python": "pip freeze > requirements.txt  # Pin exact versions",
        "docker": "FROM python:3.12.1-slim@sha256:abc123...  # Pin by digest",
        "why": "Prevents silent upgrades to compromised versions",
    },
    "SBOM (Software Bill of Materials)": {
        "what": "Machine-readable inventory of all components",
        "formats": "CycloneDX (JSON/XML), SPDX (JSON/RDF)",
        "generate": "cyclonedx-py requirements -o sbom.json",
        "why": "Know exactly what's in your software for CVE response",
    },
    "Image Signing": {
        "what": "Cryptographic signature proving image authenticity",
        "tool": "cosign (from Sigstore project)",
        "sign": "cosign sign --key cosign.key ghcr.io/myorg/myapp:v1.0",
        "verify": "cosign verify --key cosign.pub ghcr.io/myorg/myapp:v1.0",
        "why": "Prevents deploying tampered images",
    },
    "Provenance Attestation": {
        "what": "Proof of where and how an artifact was built",
        "standard": "SLSA (Supply-chain Levels for Software Artifacts)",
        "levels": "L1: documented build, L2: hosted build, L3: hardened build",
        "why": "Ensures builds are reproducible and tamper-resistant",
    },
}

for control, details in supply_chain_controls.items():
    print(f"\n{control}")
    print(f"  What: {details['what']}")
    print(f"  Why:  {details['why']}")

# Kubernetes admission control (enforce signed images):
# - Use Kyverno or OPA Gatekeeper to reject unsigned images
# - Policy: all images in production namespace must have cosign signature
SOLUTION
}

# === Exercise 4: OWASP Top 10 Mapping ===
# Problem: Map the OWASP Top 10 to DevOps controls that mitigate each risk.
exercise_4() {
    echo "=== Exercise 4: OWASP Top 10 Mapping ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
owasp_devops_mapping = [
    ("A01: Broken Access Control",
     "RBAC policies, least-privilege IAM, automated access reviews"),
    ("A02: Cryptographic Failures",
     "Enforce TLS everywhere (cert-manager), encrypt at rest (KMS)"),
    ("A03: Injection",
     "SAST scanning (Bandit, Semgrep), parameterized queries, WAF rules"),
    ("A04: Insecure Design",
     "Threat modeling in design phase, security requirements in backlog"),
    ("A05: Security Misconfiguration",
     "IaC scanning (tfsec, checkov), CIS benchmarks, policy-as-code"),
    ("A06: Vulnerable Components",
     "Dependency scanning (pip-audit, Snyk), SBOM, automated updates"),
    ("A07: Auth Failures",
     "MFA enforcement, short-lived tokens, secret rotation automation"),
    ("A08: Data Integrity Failures",
     "Image signing (cosign), SLSA provenance, CI/CD pipeline security"),
    ("A09: Logging Failures",
     "Structured logging, centralized aggregation, log-based alerting"),
    ("A10: SSRF",
     "Network segmentation, egress controls, private subnets, WAF"),
]

print("OWASP Top 10 -> DevOps Controls")
print("-" * 70)
for risk, controls in owasp_devops_mapping:
    print(f"\n  {risk}")
    print(f"    Controls: {controls}")
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 11: Security in DevOps"
echo "====================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
