"""
Exercises for Lesson 17: Cloud Security (Offensive)
Topic: Cybersecurity_Offensive

Practice problems covering cloud misconfiguration detection,
IAM policy analysis, metadata service exploitation, and S3 enumeration.
"""


# === Exercise 1: IAM Policy Analyzer ===
# Problem: Analyze AWS IAM policies for overly permissive configurations.

def exercise_1():
    """
    iam_policies = [
        {"name": "AdminAccess",
         "statement": [{"Effect": "Allow", "Action": "*", "Resource": "*"}]},
        {"name": "S3ReadOnly",
         "statement": [{"Effect": "Allow", "Action": "s3:GetObject",
                        "Resource": "arn:aws:s3:::public-bucket/*"}]},
        {"name": "EC2FullAccess",
         "statement": [{"Effect": "Allow", "Action": "ec2:*",
                        "Resource": "*"}]},
        {"name": "LambdaInvoke",
         "statement": [{"Effect": "Allow", "Action": ["lambda:InvokeFunction"],
                        "Resource": "arn:aws:lambda:*:*:function:myFunc"}]},
        {"name": "PassRole",
         "statement": [{"Effect": "Allow",
                        "Action": ["iam:PassRole", "sts:AssumeRole"],
                        "Resource": "*"}]},
    ]
    For each policy, determine:
      - risk_level: high/medium/low
      - issues: list of specific security concerns
      - escalation_potential: bool (can this lead to privilege escalation?)
    Return list of analysis dicts.
    """
    # TODO: Analyze IAM policies for security issues
    pass


# === Exercise 2: S3 Bucket Misconfiguration Checker ===
# Problem: Given S3 bucket ACL and policy data, identify misconfigurations.

def exercise_2():
    """
    buckets = [
        {"name": "company-public-assets",
         "acl": "public-read", "policy": None,
         "versioning": False, "encryption": None},
        {"name": "company-backups",
         "acl": "private",
         "policy": {"Statement": [{"Effect": "Allow", "Principal": "*",
                    "Action": "s3:GetObject", "Resource": "arn:aws:s3:::company-backups/*"}]},
         "versioning": True, "encryption": "AES256"},
        {"name": "company-logs",
         "acl": "private", "policy": None,
         "versioning": False, "encryption": "aws:kms"},
        {"name": "dev-temp",
         "acl": "public-read-write", "policy": None,
         "versioning": False, "encryption": None},
    ]
    For each bucket:
      - publicly_accessible: bool
      - issues: list[str]
      - risk: str
      - remediation: list[str]
    """
    # TODO: Check S3 buckets for misconfigurations
    pass


# === Exercise 3: Cloud Metadata Exploitation ===
# Problem: Given SSRF access to a cloud instance, determine what
# metadata endpoints to query and what data can be extracted.

def exercise_3():
    """
    cloud_provider = "AWS"  # Could also be "GCP" or "Azure"

    For AWS IMDSv1 (http://169.254.169.254/latest/meta-data/):
    Build a list of valuable metadata endpoints to query:
      - IAM role credentials
      - Instance identity
      - User data (startup scripts, may contain secrets)
      - Network configuration

    Return {
        "endpoints": [{"path": str, "value_type": str, "risk": str}],
        "exploitation_steps": list[str],
        "mitigations": list[str]  # IMDSv2, etc.
    }
    """
    # TODO: Map cloud metadata exploitation paths
    pass


# === Exercise 4: Cloud Attack Path Mapper ===
# Problem: Given compromised cloud credentials, map potential
# attack paths based on permissions.

def exercise_4():
    """
    compromised_role = {
        "name": "LambdaExecutionRole",
        "permissions": [
            "lambda:*",
            "s3:GetObject", "s3:PutObject",
            "logs:CreateLogGroup", "logs:PutLogEvents",
            "iam:PassRole",
            "sts:AssumeRole",
        ],
        "trust_policy": {"Service": "lambda.amazonaws.com"},
    }
    Determine:
      1. What resources can be accessed directly?
      2. Can we escalate privileges? (iam:PassRole + lambda:* is dangerous)
      3. What is the attack chain to full account compromise?
    Return {"direct_access": list, "escalation_path": list[str],
            "max_impact": str}
    """
    # TODO: Map cloud attack paths from compromised role
    pass


if __name__ == "__main__":
    print("=== Exercise 1: IAM Policy Analyzer ===")
    print(exercise_1())
    print("\n=== Exercise 2: S3 Bucket Misconfiguration ===")
    print(exercise_2())
    print("\n=== Exercise 3: Cloud Metadata Exploitation ===")
    print(exercise_3())
    print("\n=== Exercise 4: Cloud Attack Path Mapper ===")
    print(exercise_4())
