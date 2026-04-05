# 클라우드 보안 테스트

**이전**: [16. 무선 보안](./16_Wireless_Security.md) | **다음**: [18. 악성코드 분석](./18_Malware_Analysis.md)

---

조직이 클라우드 플랫폼으로 마이그레이션함에 따라 공격 표면이 온프레미스 네트워크에서 클라우드 서비스와 API로 이동한다. 이 레슨에서는 IAM 설정 오류, 메타데이터 서비스 익스플로잇, 스토리지 버킷 열거, 서버리스 함수 공격 등 AWS, GCP, Azure의 고유한 보안 과제를 다룬다.

> **중요**: 소유하거나 명시적으로 평가를 인가받은 클라우드 리소스에 대해서만 테스트한다.

**난이도**: ⭐⭐⭐⭐

## 학습 목표

1. 클라우드 보안의 공동 책임 모델(Shared Responsibility Model) 이해
2. AWS에서 IAM 설정 오류 열거 및 익스플로잇
3. IMDS를 익스플로잇하여 크리덴셜 도용 및 권한 상승
4. 공개적으로 접근 가능한 S3 버킷 발견 및 익스플로잇
5. 서버리스 함수(Lambda, Cloud Functions) 공격
6. 클라우드 인프라 설정 오류 식별
7. Terraform 및 IaC 구성의 보안 이슈 테스트
8. 클라우드 보안 모범 사례 구현

---

## 목차

1. [클라우드 보안 기초](#1-클라우드-보안-기초)
2. [AWS IAM 익스플로잇](#2-aws-iam-익스플로잇)
3. [인스턴스 메타데이터 서비스(IMDS) 공격](#3-인스턴스-메타데이터-서비스imds-공격)
4. [S3 버킷 열거 및 익스플로잇](#4-s3-버킷-열거-및-익스플로잇)
5. [Lambda 및 서버리스 공격](#5-lambda-및-서버리스-공격)
6. [GCP 보안 테스트](#6-gcp-보안-테스트)
7. [Azure 보안 테스트](#7-azure-보안-테스트)
8. [클라우드 크리덴셜 도용](#8-클라우드-크리덴셜-도용)
9. [Terraform 및 IaC 설정 오류](#9-terraform-및-iac-설정-오류)
10. [클라우드 보안 도구 및 프레임워크](#10-클라우드-보안-도구-및-프레임워크)
11. [연습 문제](#11-연습-문제)
12. [요약](#12-요약)
13. [참고 자료](#13-참고-자료)

---

## 1. 클라우드 보안 기초

### 1.1 공동 책임 모델

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

## 2. AWS IAM 익스플로잇

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

## 3. 인스턴스 메타데이터 서비스(IMDS) 공격

IMDS는 IAM 크리덴셜을 포함한 인스턴스 메타데이터를 `http://169.254.169.254/`에서 제공한다.

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

### 3.1 SSRF를 통한 IMDS

가장 일반적인 클라우드 익스플로잇 경로: SSRF 취약점 → IMDS → 크리덴셜 도용 → 권한 상승.

---

## 4. S3 버킷 열거 및 익스플로잇

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

## 5. Lambda 및 서버리스 공격

### 5.1 Lambda 공격 벡터

- **이벤트 인젝션(Event Injection)**: Lambda 이벤트 데이터에 악의적 입력 삽입
- **의존성 혼동(Dependency Confusion)**: Lambda 레이어에 대한 공급망 공격
- **환경 변수 유출**: 환경 변수에 저장된 시크릿
- **과도한 권한**: 광범위한 IAM 권한을 가진 Lambda 역할
- **코드 인젝션(Code Injection)**: Lambda가 사용자 제어 코드를 처리하는 경우

---

## 6. GCP 보안 테스트

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

## 7. Azure 보안 테스트

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

## 8. 클라우드 크리덴셜 도용

클라우드 크리덴셜이 저장되는 일반적인 위치:

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

## 9. Terraform 및 IaC 설정 오류

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

## 10. 클라우드 보안 도구 및 프레임워크

| 도구 | 용도 | 클라우드 |
|------|------|----------|
| ScoutSuite | 멀티 클라우드 보안 감사 | AWS, GCP, Azure |
| Prowler | AWS 보안 평가 | AWS |
| CloudSploit | 클라우드 설정 오류 스캐너 | 멀티 클라우드 |
| Pacu | AWS 익스플로잇 프레임워크 | AWS |
| enumerate-iam | IAM 권한 열거 | AWS |
| tfsec | Terraform 보안 스캐너 | 전체 |
| Steampipe | 클라우드 API에 대한 SQL 쿼리 | 멀티 클라우드 |

---

## 11. 연습 문제

1. **S3 열거**: 테스트 조직에 속한 공개적으로 접근 가능한 S3 버킷을 검색한다.
2. **IMDS 익스플로잇**: 랩 환경에서 SSRF 취약점을 익스플로잇하여 AWS 메타데이터 서비스에 접근한다.
3. **IAM 분석**: IAM 정책을 열거하고 Pacu를 사용하여 권한 상승 경로를 식별한다.
4. **IaC 스캐닝**: Terraform 프로젝트에 대해 tfsec을 실행하고 모든 발견 사항을 수정한다.
5. **클라우드 감사**: 자신의 AWS 계정에서 ScoutSuite를 실행하고 보안 보고서를 생성한다.
6. **전체 체인**: SSRF → IMDS → 크리덴셜 도용 → 권한 상승 체인을 익스플로잇한다.

---

## 12. 요약

클라우드 보안 테스트는 현대의 공격 표면을 대상으로 한다:

- **공동 책임 모델**은 고객과 제공자의 보안 의무를 정의한다
- **IAM 설정 오류**는 가장 일반적인 클라우드 취약점이다
- **IMDS 공격**은 클라우드 인스턴스에서 크리덴셜을 도용한다
- **공개 S3 버킷**은 민감한 데이터를 노출한다
- **서버리스 함수**는 새로운 공격 벡터를 도입한다
- **IaC 스캐닝**은 배포 전에 설정 오류를 포착한다
- 방어에는 최소 권한 IAM, IMDSv2, 지속적 모니터링이 필요하다

---

## 13. 참고 자료

- AWS Security Documentation: https://docs.aws.amazon.com/security/
- Pacu (AWS Exploitation): https://github.com/RhinoSecurityLabs/pacu
- ScoutSuite: https://github.com/nccgroup/ScoutSuite
- HackTricks Cloud: https://cloud.hacktricks.xyz/
- Prowler: https://github.com/prowler-cloud/prowler
- OWASP Cloud Security: https://owasp.org/www-project-cloud-security/
