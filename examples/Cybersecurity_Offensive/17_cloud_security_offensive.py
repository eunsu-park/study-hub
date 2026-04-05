"""
Example: Cloud Security (Offensive)
=====================================
IAM policy analyzer, S3 bucket checker, cloud metadata paths,
and privilege escalation chains in AWS.

IMPORTANT: For authorized security testing and CTF only.
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# IAM Policy Risk Analyzer
# ---------------------------------------------------------------------------

HIGH_RISK_ACTIONS = {
    "iam:PassRole": "Can pass roles to services for privilege escalation",
    "iam:CreatePolicyVersion": "Can create new policy version with admin access",
    "iam:AttachUserPolicy": "Can attach admin policy to own user",
    "iam:CreateUser": "Can create new users with arbitrary permissions",
    "iam:CreateAccessKey": "Can create access keys for any user",
    "sts:AssumeRole": "Can assume other roles (potentially admin)",
    "lambda:CreateFunction": "Combined with iam:PassRole enables escalation",
    "ec2:RunInstances": "Combined with iam:PassRole enables escalation",
    "*": "Full admin access to all services",
}


@dataclass
class PolicyAnalysis:
    policy_name: str
    risk_level: str
    issues: list[str] = field(default_factory=list)
    escalation_paths: list[str] = field(default_factory=list)


def analyze_iam_policy(name: str, statements: list[dict]) -> PolicyAnalysis:
    """Analyze an IAM policy for security risks."""
    issues = []
    escalation_paths = []

    for stmt in statements:
        if stmt.get("Effect") != "Allow":
            continue
        actions = stmt.get("Action", [])
        if isinstance(actions, str):
            actions = [actions]
        resource = stmt.get("Resource", "")

        for action in actions:
            if action in HIGH_RISK_ACTIONS:
                issues.append(f"{action}: {HIGH_RISK_ACTIONS[action]}")
            if action == "*" and resource == "*":
                issues.append("CRITICAL: Full admin access (Action:* Resource:*)")
                escalation_paths.append("Direct admin access")

        # Escalation combos
        action_set = set(actions)
        if "iam:PassRole" in action_set and "lambda:CreateFunction" in action_set:
            escalation_paths.append(
                "PassRole + Lambda: Create Lambda with admin role")
        if "iam:PassRole" in action_set and "ec2:RunInstances" in action_set:
            escalation_paths.append(
                "PassRole + EC2: Launch instance with admin role")

    risk = "low"
    if escalation_paths:
        risk = "critical"
    elif len(issues) > 2:
        risk = "high"
    elif issues:
        risk = "medium"

    return PolicyAnalysis(name, risk, issues, escalation_paths)


# ---------------------------------------------------------------------------
# S3 Bucket Security Check
# ---------------------------------------------------------------------------

@dataclass
class BucketFinding:
    bucket: str
    publicly_readable: bool
    publicly_writable: bool
    encrypted: bool
    versioned: bool
    issues: list[str] = field(default_factory=list)


def check_bucket(name: str, acl: str, policy: dict | None,
                 encryption: str | None, versioning: bool) -> BucketFinding:
    """Check S3 bucket for misconfigurations."""
    issues = []
    pub_read = acl in ("public-read", "public-read-write")
    pub_write = acl == "public-read-write"

    if pub_read:
        issues.append("Bucket ACL allows public read")
    if pub_write:
        issues.append("CRITICAL: Bucket ACL allows public write")
    if not encryption:
        issues.append("No server-side encryption configured")
    if not versioning:
        issues.append("Versioning disabled — no protection against deletion")

    # Check bucket policy for public access
    if policy:
        for stmt in policy.get("Statement", []):
            if stmt.get("Principal") == "*" and stmt.get("Effect") == "Allow":
                issues.append("Bucket policy grants access to Principal *")
                pub_read = True

    return BucketFinding(name, pub_read, pub_write,
                         encryption is not None, versioning, issues)


# ---------------------------------------------------------------------------
# Cloud Metadata Endpoints
# ---------------------------------------------------------------------------

AWS_METADATA_PATHS = {
    "/latest/meta-data/iam/security-credentials/": {
        "value": "IAM role temporary credentials (AccessKey, SecretKey, Token)",
        "risk": "critical",
    },
    "/latest/meta-data/instance-id": {
        "value": "EC2 instance identifier",
        "risk": "low",
    },
    "/latest/user-data": {
        "value": "Instance startup script (may contain secrets)",
        "risk": "high",
    },
    "/latest/meta-data/local-ipv4": {
        "value": "Internal IP address",
        "risk": "medium",
    },
    "/latest/meta-data/public-keys/": {
        "value": "SSH public keys configured for instance",
        "risk": "low",
    },
    "/latest/meta-data/network/interfaces/macs/": {
        "value": "VPC, subnet, and security group info",
        "risk": "medium",
    },
}

GCP_METADATA_PATHS = {
    "/computeMetadata/v1/instance/service-accounts/default/token": {
        "value": "GCP service account OAuth token",
        "risk": "critical",
        "header_required": "Metadata-Flavor: Google",
    },
    "/computeMetadata/v1/project/project-id": {
        "value": "GCP project ID",
        "risk": "low",
        "header_required": "Metadata-Flavor: Google",
    },
}


# ---------------------------------------------------------------------------
# AWS Privilege Escalation Paths
# ---------------------------------------------------------------------------

AWS_PRIVESC_PATHS = [
    {
        "name": "PassRole + Lambda",
        "required_permissions": ["iam:PassRole", "lambda:CreateFunction",
                                 "lambda:InvokeFunction"],
        "steps": [
            "Create Lambda function with AdminAccess role",
            "Lambda code calls IAM API to attach admin policy to attacker",
            "Invoke the Lambda function",
        ],
    },
    {
        "name": "CreatePolicyVersion",
        "required_permissions": ["iam:CreatePolicyVersion"],
        "steps": [
            "Create new version of attached managed policy",
            'Set policy document to {"Effect":"Allow","Action":"*","Resource":"*"}',
            "Set as default version",
        ],
    },
    {
        "name": "AssumeRole Chain",
        "required_permissions": ["sts:AssumeRole"],
        "steps": [
            "Enumerate roles with permissive trust policies",
            "Assume a role with higher privileges",
            "Chain role assumptions until reaching admin",
        ],
    },
]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    print("Cloud Security (Offensive) Examples")
    print("=" * 50)

    # IAM policy analysis
    print("\nIAM Policy Analysis:")
    result = analyze_iam_policy("DangerousPolicy", [
        {"Effect": "Allow", "Action": ["iam:PassRole", "lambda:*"],
         "Resource": "*"},
    ])
    print(f"  Policy: {result.policy_name} [{result.risk_level}]")
    for issue in result.issues:
        print(f"    Issue: {issue}")
    for path in result.escalation_paths:
        print(f"    Escalation: {path}")

    # S3 bucket check
    print("\nS3 Bucket Security:")
    buckets = [
        ("public-assets", "public-read", None, None, False),
        ("backups", "private", None, "AES256", True),
        ("dev-temp", "public-read-write", None, None, False),
    ]
    for name, acl, policy, enc, ver in buckets:
        finding = check_bucket(name, acl, policy, enc, ver)
        status = "VULNERABLE" if finding.issues else "OK"
        print(f"  [{status:10s}] {name}: {len(finding.issues)} issues")

    # Metadata endpoints
    print("\nAWS Metadata Endpoints (via SSRF):")
    for path, info in AWS_METADATA_PATHS.items():
        print(f"  [{info['risk']:8s}] {path}")

    # Privilege escalation paths
    print("\nAWS Privilege Escalation Paths:")
    for pe in AWS_PRIVESC_PATHS:
        print(f"  {pe['name']}:")
        print(f"    Requires: {', '.join(pe['required_permissions'])}")


if __name__ == "__main__":
    demo()
