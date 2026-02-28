"""
Exercises for Lesson 13: Identity and Access Management
Topic: Cloud_Computing

Solutions to practice problems from the lesson.
Simulates IAM policy writing, role design, cross-account access, and security analysis.
"""

import json


# === Exercise 1: Least Privilege Policy Writing ===
def exercise_1():
    """Write a minimal IAM policy for a Lambda function."""

    print("Least Privilege IAM Policy for Lambda:")
    print("=" * 70)
    print()
    print("  Requirements:")
    print("    - Read objects from S3 bucket 'my-app-data' (any object)")
    print("    - Write logs to CloudWatch Log group '/aws/lambda/my-function'")
    print()

    # The policy grants only the minimum permissions needed.
    # Why s3:GetObject only: The function reads data, it does not write or delete.
    # Why specific Resource ARNs: Prevents the function from accessing other buckets
    # or log groups if compromised.
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "ReadFromS3Bucket",
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject"
                ],
                "Resource": "arn:aws:s3:::my-app-data/*"
            },
            {
                "Sid": "WriteCloudWatchLogs",
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogStream",
                    "logs:PutLogEvents"
                ],
                "Resource": "arn:aws:logs:*:*:log-group:/aws/lambda/my-function:*"
            }
        ]
    }

    print("  Policy Document:")
    print(json.dumps(policy, indent=4))
    print()

    print("  Why this is least privilege:")
    reasons = [
        "s3:GetObject only -- not s3:* or s3:PutObject. Function only reads.",
        "S3 resource scoped to my-app-data/* -- not * (all buckets).",
        "CloudWatch actions limited to CreateLogStream + PutLogEvents.",
        "Log group resource scoped to the specific function's group.",
    ]
    for r in reasons:
        print(f"    - {r}")
    print()
    print("  Common mistake: Using 'Resource': '*' for CloudWatch logs")
    print("  grants access to ALL log groups, not just this function's group.")


# === Exercise 2: IAM Role vs IAM User for EC2 ===
def exercise_2():
    """Explain why IAM roles are preferred over IAM users for EC2."""

    print("IAM Role vs IAM User for EC2 S3 Access:")
    print("=" * 70)
    print()

    approaches = {
        "Developer A (IAM User + Access Keys)": {
            "method": "Create IAM user, generate access keys, hard-code in app config.",
            "verdict": "INCORRECT",
            "problems": [
                "Long-lived credentials: Access keys don't expire automatically. "
                "If compromised, usable indefinitely until manually rotated.",
                "Credential exposure risk: Hard-coded keys can leak into Git, "
                "Docker images, environment variables, or application logs.",
                "Key rotation overhead: Rotating keys requires updating every "
                "server and config that uses them -- manual, error-prone.",
            ],
        },
        "Developer B (IAM Role + Instance Profile)": {
            "method": "Create IAM role with S3 permissions, attach to EC2 instance profile.",
            "verdict": "CORRECT",
            "advantages": [
                "Temporary credentials: EC2 metadata service provides 1-hour STS "
                "tokens that auto-rotate. Stolen tokens expire quickly.",
                "No secrets to manage: AWS SDK default credential chain fetches "
                "credentials from metadata service. No keys to store or rotate.",
                "No code changes: Switch to a more restrictive role by updating "
                "the role's permissions -- no app changes or deployments.",
            ],
        },
    }

    for approach, details in approaches.items():
        print(f"  {approach}:")
        print(f"    Method: {details['method']}")
        print(f"    Verdict: {details['verdict']}")
        if "problems" in details:
            print("    Problems:")
            for p in details["problems"]:
                print(f"      - {p}")
        if "advantages" in details:
            print("    Advantages:")
            for a in details["advantages"]:
                print(f"      + {a}")
        print()

    print("  Usage with AWS SDK (no credentials needed):")
    print("    import boto3")
    print("    s3 = boto3.client('s3')  # SDK auto-uses instance role")
    print("    s3.upload_file('local_file.txt', 'my-bucket', 'uploaded_file.txt')")


# === Exercise 3: Cross-Account Role Assumption ===
def exercise_3():
    """Describe the complete setup for cross-account S3 access."""

    account_a = "111111111111"  # Resource account (has S3 bucket)
    account_b = "222222222222"  # Consumer account (has Lambda function)

    print("Cross-Account Role Assumption Setup:")
    print("=" * 70)
    print(f"\n  Account A ({account_a}): Has S3 bucket with data")
    print(f"  Account B ({account_b}): Has Lambda that needs to read from A's bucket")
    print()

    # Step 1: Create role in Account A
    # Why a trust policy: Account A must explicitly allow Account B to assume
    # the role. Without this, cross-account access is impossible.
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {
                "AWS": f"arn:aws:iam::{account_b}:root"
            },
            "Action": "sts:AssumeRole"
        }]
    }

    print("  Step 1: Create role in Account A (resource account)")
    print(f"    Role name: CrossAccountS3ReadRole")
    print(f"    Trust policy (who can assume this role):")
    print(f"    {json.dumps(trust_policy, indent=4)}")
    print(f"    Attach policy: AmazonS3ReadOnlyAccess")
    print()

    # Step 2: Grant Lambda permission to assume the role
    assume_policy = {
        "Effect": "Allow",
        "Action": "sts:AssumeRole",
        "Resource": f"arn:aws:iam::{account_a}:role/CrossAccountS3ReadRole"
    }

    print("  Step 2: In Account B, add to Lambda execution role:")
    print(f"    {json.dumps(assume_policy, indent=4)}")
    print()

    # Step 3: Lambda code
    print("  Step 3: Lambda code assumes the role:")
    print("    sts = boto3.client('sts')")
    print("    assumed = sts.assume_role(")
    print(f"        RoleArn='arn:aws:iam::{account_a}:role/CrossAccountS3ReadRole',")
    print("        RoleSessionName='LambdaCrossAccountSession'")
    print("    )")
    print("    # Use temporary credentials to access Account A's S3")
    print("    s3 = boto3.client('s3',")
    print("        aws_access_key_id=assumed['Credentials']['AccessKeyId'],")
    print("        aws_secret_access_key=assumed['Credentials']['SecretAccessKey'],")
    print("        aws_session_token=assumed['Credentials']['SessionToken'])")
    print()

    print("  Security: Temporary credentials expire in 1 hour (configurable).")
    print("  Account A maintains full control -- can revoke the role at any time.")


# === Exercise 4: IAM Policy Analysis ===
def exercise_4():
    """Analyze an IAM policy for security concerns."""

    # This is the "PowerUser minus IAM" pattern -- commonly seen but problematic.
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": "*",
                "Resource": "*"
            },
            {
                "Effect": "Deny",
                "Action": [
                    "iam:*",
                    "organizations:*"
                ],
                "Resource": "*"
            }
        ]
    }

    print("IAM Policy Analysis:")
    print("=" * 70)
    print()
    print("  Policy to analyze:")
    print(json.dumps(policy, indent=4))
    print()

    print("  What this policy does:")
    print("    Statement 1: Grants full access (*) to all AWS services and resources.")
    print("    Statement 2: Explicitly denies all IAM and Organizations actions.")
    print("    Net effect: User can do everything EXCEPT manage IAM/Organizations.")
    print("    (Deny always overrides Allow in IAM.)")
    print()

    # Security concerns
    # Why this is dangerous: Even without IAM access, the user can delete
    # databases, terminate instances, empty S3 buckets, deploy Lambda functions,
    # access secrets -- essentially everything except modifying permissions.
    print("  Security Concerns:")
    concerns = [
        ("Still dangerously over-privileged",
         "User can delete databases, terminate EC2, empty S3 buckets, "
         "deploy Lambda functions, access secrets -- everything except IAM."),
        ("IAM deny is insufficient protection",
         "Cannot create new roles, but can still abuse existing permissions "
         "to cause significant damage or exfiltrate data."),
        ("Privilege escalation still possible",
         "User could abuse Lambda, EC2 user data, or CloudFormation to "
         "execute code with higher permissions."),
    ]
    for name, detail in concerns:
        print(f"    - {name}:")
        print(f"      {detail}")
    print()

    print("  Better approach: Start with NO permissions and add only what is needed")
    print("  (allowlist), not start with ALL and try to deny specifics (denylist).")
    print("  This policy violates the principle of least privilege.")


# === Exercise 5: GCP Service Account Best Practices ===
def exercise_5():
    """Configure minimal GCP service account for a Compute Engine app."""

    print("GCP Service Account Configuration:")
    print("=" * 70)
    print()
    print("  Application needs:")
    print("    - Write metrics to Cloud Storage")
    print("    - Read configuration from Secret Manager")
    print()

    steps = [
        {
            "step": "Create a dedicated service account (one per application)",
            "command": (
                "gcloud iam service-accounts create my-app-sa \\\n"
                '    --display-name="My Application Service Account" \\\n'
                "    --project=my-project-id"
            ),
        },
        {
            "step": "Grant minimal required roles",
            "commands": [
                {
                    "role": "roles/storage.objectCreator",
                    "reason": "Write new objects only -- cannot read or delete others.",
                    "command": (
                        "gcloud projects add-iam-policy-binding my-project-id \\\n"
                        '    --member="serviceAccount:my-app-sa@my-project-id.'
                        'iam.gserviceaccount.com" \\\n'
                        '    --role="roles/storage.objectCreator"'
                    ),
                },
                {
                    "role": "roles/secretmanager.secretAccessor",
                    "reason": "Read secret values only -- cannot create or modify secrets.",
                    "command": (
                        "gcloud projects add-iam-policy-binding my-project-id \\\n"
                        '    --member="serviceAccount:my-app-sa@my-project-id.'
                        'iam.gserviceaccount.com" \\\n'
                        '    --role="roles/secretmanager.secretAccessor"'
                    ),
                },
            ],
        },
        {
            "step": "Assign service account to Compute Engine instance",
            "command": (
                "gcloud compute instances create my-instance \\\n"
                "    --service-account=my-app-sa@my-project-id."
                "iam.gserviceaccount.com \\\n"
                "    --scopes=cloud-platform \\\n"
                "    --zone=asia-northeast3-a"
            ),
        },
    ]

    for i, s in enumerate(steps, 1):
        print(f"  Step {i}: {s['step']}")
        if "command" in s:
            print(f"    {s['command']}")
        if "commands" in s:
            for c in s["commands"]:
                print(f"    Role: {c['role']}")
                print(f"      Why: {c['reason']}")
                print(f"      {c['command']}")
                print()
        print()

    # Best practices
    # Why one SA per app: If compromised, only that application's permissions
    # are exposed. Shared SAs create a blast radius across all apps.
    print("  Best Practices:")
    practices = [
        "One service account per application -- limits blast radius if compromised.",
        "Use IAM conditions to scope to specific resources (e.g., specific bucket).",
        "objectCreator instead of storage.admin -- write only, not delete/read.",
        "secretAccessor instead of secretmanager.admin -- read only, not create/modify.",
        "Avoid downloading key files -- use metadata server for CE instances instead.",
    ]
    for p in practices:
        print(f"    - {p}")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Least Privilege Policy Writing ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: IAM Role vs IAM User for EC2 ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Cross-Account Role Assumption ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: IAM Policy Analysis ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: GCP Service Account Best Practices ===")
    print("=" * 70)
    exercise_5()

    print("\nAll exercises completed!")
