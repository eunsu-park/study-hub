# IAM (Identity and Access Management)

**Previous**: [NoSQL Databases](./12_NoSQL_Databases.md) | **Next**: [Security Services](./14_Security_Services.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the purpose of IAM and the principle of least privilege
2. Compare AWS IAM and GCP IAM in scope, policy models, and role-binding mechanisms
3. Create users, groups, and roles with appropriate permission policies
4. Write and attach IAM policies using JSON policy documents (AWS) or role bindings (GCP)
5. Configure service accounts for machine-to-machine authentication
6. Implement cross-account access using IAM role assumption
7. Audit IAM configurations to identify over-permissioned principals

---

Identity and Access Management is the cornerstone of cloud security. Every API call, every resource access, and every service interaction is governed by IAM policies. A single misconfigured permission can expose sensitive data or grant attackers a foothold. Mastering IAM is not optional -- it is the first and most important security control in any cloud environment.

## 1. IAM Overview

### 1.1 What is IAM?

IAM is a service that securely controls access to cloud resources.

**Core Questions:**
- **Who**: Users, groups, roles
- **What**: Resources
- **How**: Permissions (allow/deny)

### 1.2 AWS vs GCP IAM Comparison

| Item | AWS IAM | GCP IAM |
|------|---------|---------|
| Scope | Account level | Organization/project level |
| Policy Attachment | To users/groups/roles | To resources |
| Roles | Assume role (AssumeRole) | Role binding |
| Service Account | Role + instance profile | Service account |

---

## 2. AWS IAM

### 2.1 Core Concepts

```
┌─────────────────────────────────────────────────────────────┐
│  AWS Account                                                │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  IAM                                                    ││
│  │  ┌───────────────┐  ┌───────────────┐                   ││
│  │  │    Users      │  │     Groups    │                   ││
│  │  │               │  │               │                   ││
│  │  └───────────────┘  └───────────────┘                   ││
│  │         ↓                  ↓                            ││
│  │  ┌─────────────────────────────────────────────┐        ││
│  │  │              Policies                       │        ││
│  │  │  { "Effect": "Allow",                       │        ││
│  │  │    "Action": "s3:*",                        │        ││
│  │  │    "Resource": "*" }                        │        ││
│  │  └─────────────────────────────────────────────┘        ││
│  │                     ↓                                   ││
│  │  ┌───────────────────────────────────────────────────┐  ││
│  │  │              Roles                                │  ││
│  │  │  - EC2 instance role                             │  ││
│  │  │  - Lambda execution role                         │  ││
│  │  │  - Cross-account role                            │  ││
│  │  └───────────────────────────────────────────────────┘  ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Users and Groups

```bash
# Create user
aws iam create-user --user-name john

# Set login password
aws iam create-login-profile \
    --user-name john \
    --password 'TempPassword123!' \
    --password-reset-required

# Create access key (programmatic access)
aws iam create-access-key --user-name john

# Create group
aws iam create-group --group-name Developers

# Add user to group
aws iam add-user-to-group --group-name Developers --user-name john

# List group members
aws iam get-group --group-name Developers
```

### 2.3 Policies

**Policy Structure:**
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AllowS3Read",
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::my-bucket",
                "arn:aws:s3:::my-bucket/*"
            ],
            "Condition": {
                "IpAddress": {
                    "aws:SourceIp": "203.0.113.0/24"
                }
            }
        }
    ]
}
```

```bash
# Attach managed policy
aws iam attach-user-policy \
    --user-name john \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess

# Create custom policy
aws iam create-policy \
    --policy-name MyS3Policy \
    --policy-document file://policy.json

# Attach policy to group
aws iam attach-group-policy \
    --group-name Developers \
    --policy-arn arn:aws:iam::123456789012:policy/MyS3Policy

# Add inline policy
aws iam put-user-policy \
    --user-name john \
    --policy-name InlinePolicy \
    --policy-document file://inline-policy.json
```

### 2.4 Roles

**EC2 Instance Role:**
```bash
# Trust policy (who can assume the role)
cat > trust-policy.json << 'EOF'
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"Service": "ec2.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }
    ]
}
EOF

# Create role
aws iam create-role \
    --role-name EC2-S3-Access \
    --assume-role-policy-document file://trust-policy.json

# Attach policy
aws iam attach-role-policy \
    --role-name EC2-S3-Access \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess

# Create instance profile and add role
aws iam create-instance-profile --instance-profile-name EC2-S3-Profile
aws iam add-role-to-instance-profile \
    --instance-profile-name EC2-S3-Profile \
    --role-name EC2-S3-Access

# Attach instance profile to EC2
aws ec2 associate-iam-instance-profile \
    --instance-id i-1234567890abcdef0 \
    --iam-instance-profile Name=EC2-S3-Profile
```

**Cross-Account Role:**
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"AWS": "arn:aws:iam::OTHER_ACCOUNT_ID:root"},
            "Action": "sts:AssumeRole"
        }
    ]
}
```

```bash
# Assume role from another account
aws sts assume-role \
    --role-arn arn:aws:iam::TARGET_ACCOUNT:role/CrossAccountRole \
    --role-session-name MySession
```

---

## 3. GCP IAM

### 3.1 Core Concepts

```
┌─────────────────────────────────────────────────────────────┐
│  Organization                                               │
│  ├── Folder                                                 │
│  │   └── Project                                            │
│  │       └── Resource                                       │
│  └─────────────────────────────────────────────────────────│
│                                                             │
│  IAM Binding:                                               │
│  Member + Role = Permission on Resource                     │
│                                                             │
│  Example: user:john@example.com + roles/storage.admin      │
│      → Admin permission on gs://my-bucket                   │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Role Types

| Type | Description | Example |
|------|------|------|
| **Basic Roles** | Broad permissions | Owner, Editor, Viewer |
| **Predefined Roles** | Service-specific granular | roles/storage.admin |
| **Custom Roles** | User-defined | my-custom-role |

### 3.3 Role Bindings

```bash
# Grant project-level role
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="user:john@example.com" \
    --role="roles/compute.admin"

# Grant bucket-level role
gsutil iam ch user:john@example.com:objectViewer gs://my-bucket

# View role bindings
gcloud projects get-iam-policy PROJECT_ID

# Remove role
gcloud projects remove-iam-policy-binding PROJECT_ID \
    --member="user:john@example.com" \
    --role="roles/compute.admin"
```

### 3.4 Service Accounts

```bash
# Create service account
gcloud iam service-accounts create my-service-account \
    --display-name="My Service Account"

# Grant role
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:my-service-account@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.admin"

# Create key file (programmatic access)
gcloud iam service-accounts keys create key.json \
    --iam-account=my-service-account@PROJECT_ID.iam.gserviceaccount.com

# Attach service account to Compute Engine
gcloud compute instances create my-instance \
    --service-account=my-service-account@PROJECT_ID.iam.gserviceaccount.com \
    --scopes=cloud-platform
```

### 3.5 Workload Identity (GKE)

```bash
# Enable workload identity pool
gcloud container clusters update my-cluster \
    --region=asia-northeast3 \
    --workload-pool=PROJECT_ID.svc.id.goog

# Bind Kubernetes service account to GCP service account
gcloud iam service-accounts add-iam-policy-binding \
    my-gcp-sa@PROJECT_ID.iam.gserviceaccount.com \
    --role=roles/iam.workloadIdentityUser \
    --member="serviceAccount:PROJECT_ID.svc.id.goog[NAMESPACE/K8S_SA]"
```

---

## 4. Principle of Least Privilege

### 4.1 Principle

```
Least Privilege = Grant only minimum permissions needed for the task

Bad Examples:
- Admin permissions to all users
- Permissions on * (all resources)

Good Examples:
- Specify only required Actions
- Permissions on specific resources
- Conditional access
```

### 4.2 AWS Policy Examples

**Bad Example:**
```json
{
    "Effect": "Allow",
    "Action": "*",
    "Resource": "*"
}
```

**Good Example:**
```json
{
    "Effect": "Allow",
    "Action": [
        "s3:GetObject",
        "s3:PutObject"
    ],
    "Resource": "arn:aws:s3:::my-bucket/uploads/*",
    "Condition": {
        "StringEquals": {
            "s3:x-amz-acl": "private"
        }
    }
}
```

### 4.3 GCP Role Selection

```bash
# Too broad roles (avoid)
roles/owner
roles/editor

# Appropriate roles
roles/storage.objectViewer  # Read objects only
roles/compute.instanceAdmin.v1  # Manage instances only
roles/cloudsql.client  # SQL connection only
```

---

## 5. Conditional Access

### 5.1 AWS Conditions

```json
{
    "Effect": "Allow",
    "Action": "s3:*",
    "Resource": "*",
    "Condition": {
        "IpAddress": {
            "aws:SourceIp": "203.0.113.0/24"
        },
        "Bool": {
            "aws:MultiFactorAuthPresent": "true"
        },
        "DateGreaterThan": {
            "aws:CurrentTime": "2024-01-01T00:00:00Z"
        },
        "StringEquals": {
            "aws:RequestedRegion": "ap-northeast-2"
        }
    }
}
```

### 5.2 GCP Conditions

```bash
# Conditional role binding
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="user:john@example.com" \
    --role="roles/compute.admin" \
    --condition='expression=request.time < timestamp("2024-12-31T23:59:59Z"),title=Temporary Access,description=Access until end of year'

# IP-based condition (with VPC Service Controls)
expression: 'resource.name.startsWith("projects/PROJECT_ID/zones/asia-northeast3")'
```

---

## 6. Permission Analysis

### 6.1 AWS IAM Access Analyzer

```bash
# Create Access Analyzer
aws accessanalyzer create-analyzer \
    --analyzer-name my-analyzer \
    --type ACCOUNT

# View findings
aws accessanalyzer list-findings --analyzer-arn arn:aws:access-analyzer:...:analyzer/my-analyzer

# Validate policy
aws accessanalyzer validate-policy \
    --policy-document file://policy.json \
    --policy-type IDENTITY_POLICY
```

### 6.2 GCP Policy Analyzer

```bash
# Analyze IAM policy
gcloud asset analyze-iam-policy \
    --organization=ORG_ID \
    --identity="user:john@example.com"

# Check permissions
gcloud projects get-iam-policy PROJECT_ID \
    --flatten="bindings[].members" \
    --filter="bindings.members:john@example.com" \
    --format="table(bindings.role)"
```

---

## 7. MFA (Multi-Factor Authentication)

### 7.1 AWS MFA

```bash
# Enable virtual MFA
aws iam create-virtual-mfa-device \
    --virtual-mfa-device-name john-mfa \
    --outfile qrcode.png \
    --bootstrap-method QRCodePNG

# Attach MFA device
aws iam enable-mfa-device \
    --user-name john \
    --serial-number arn:aws:iam::123456789012:mfa/john-mfa \
    --authentication-code1 123456 \
    --authentication-code2 789012

# MFA required policy
{
    "Effect": "Deny",
    "Action": "*",
    "Resource": "*",
    "Condition": {
        "BoolIfExists": {
            "aws:MultiFactorAuthPresent": "false"
        }
    }
}
```

### 7.2 GCP 2-Step Verification

```bash
# Enforce 2FA at organization level (in Admin Console)
# Google Workspace Admin → Security → 2-Step Verification

# Service accounts don't support MFA → Instead:
# - Secure key file management
# - Use workload identity
# - Use short-lived tokens
```

---

## 8. Common Role Patterns

### 8.1 AWS Common Roles

| Role | Permission | Use Case |
|------|------|------|
| AdministratorAccess | Full | Administrator |
| PowerUserAccess | All except IAM | Developer |
| ReadOnlyAccess | Read-only | Auditor/Viewer |
| AmazonS3FullAccess | S3 full | Storage management |
| AmazonEC2FullAccess | EC2 full | Compute management |

### 8.2 GCP Common Roles

| Role | Permission | Use Case |
|------|------|------|
| roles/owner | Full | Administrator |
| roles/editor | Edit except IAM | Developer |
| roles/viewer | Read-only | Viewer |
| roles/compute.admin | Compute full | Infrastructure management |
| roles/storage.admin | Storage full | Storage management |

---

## 9. Security Best Practices

```
□ Don't use Root/Owner account for daily tasks
□ Enable MFA on Root/Owner account
□ Apply principle of least privilege
□ Manage permissions via groups/roles (not individual users)
□ Regular permission review (remove unused permissions)
□ Secure service account key files
□ Use temporary credentials (STS, workload identity)
□ Use conditional access (IP, time, MFA)
□ Enable audit logs (CloudTrail, Cloud Audit Logs)
□ Set up policy change notifications
```

---

## 10. Next Steps

- [14_Security_Services.md](./14_Security_Services.md) - Security Services
- [02_AWS_GCP_Account_Setup.md](./02_AWS_GCP_Account_Setup.md) - Initial Account Setup

---

## Exercises

### Exercise 1: Least Privilege Policy Writing

A Lambda function needs to:
- Read objects from S3 bucket `my-app-data` (any object)
- Write logs to a specific CloudWatch Log group `/aws/lambda/my-function`

Write the minimal IAM policy document (JSON) that follows the principle of least privilege.

<details>
<summary>Show Answer</summary>

```json
{
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
```

**Why this is least privilege**:
- `s3:GetObject` only — not `s3:*` or `s3:PutObject`. The function should only read, not write or delete.
- S3 resource is scoped to `my-app-data/*` — not `*` (all buckets).
- CloudWatch actions are limited to creating streams and putting events — not creating log groups or describing resources.
- Log group resource is scoped to the specific function's log group.

**Common mistake**: Using `"Resource": "*"` for CloudWatch logs — this grants access to all log groups in the account, not just the Lambda function's group.

</details>

### Exercise 2: IAM Role vs IAM User for EC2

An EC2 instance needs to upload files to an S3 bucket. Two developers suggest different approaches:
- **Developer A**: Create an IAM user with S3 permissions, generate access keys, and hard-code them in the application config.
- **Developer B**: Create an IAM role with S3 permissions and attach it to the EC2 instance profile.

Which approach is correct? Explain the security risks of the incorrect approach.

<details>
<summary>Show Answer</summary>

**Developer B is correct** — Use an IAM role attached to the instance profile.

**Problems with Developer A's approach (IAM user + access keys)**:

1. **Long-lived credentials**: Access keys don't expire automatically. If compromised (leaked in code, config files, or logs), they can be used indefinitely until manually rotated.

2. **Credential exposure risk**: Hard-coding credentials in config files means they can end up in version control (Git), Docker images, container environment variables, or application logs.

3. **Key rotation overhead**: Rotating access keys requires updating every server and config that uses them — a manual, error-prone process.

**Why IAM roles are better**:
1. **Temporary credentials**: The EC2 metadata service provides short-lived credentials (1-hour STS tokens) that automatically rotate. If stolen, they expire quickly.
2. **No secrets to manage**: The application uses the AWS SDK's default credential chain, which automatically fetches credentials from the instance metadata service. No keys to store, rotate, or expose.
3. **No code changes needed**: Switch to a more restrictive role by updating the role's permissions — no application changes or deployments required.

**How to use with AWS SDK**:
```python
import boto3

# No credentials needed - SDK automatically uses the instance role
s3 = boto3.client('s3')
s3.upload_file('local_file.txt', 'my-bucket', 'uploaded_file.txt')
```

</details>

### Exercise 3: Cross-Account Role Assumption

Account A (ID: `111111111111`) has an S3 bucket with important data. Account B (ID: `222222222222`) has a Lambda function that needs to read from that bucket.

Describe the complete setup required for cross-account access using IAM role assumption.

<details>
<summary>Show Answer</summary>

**Step 1: Create a role in Account A (the resource account)**

Create a role that Account B's Lambda can assume:

```bash
# In Account A
aws iam create-role \
    --role-name CrossAccountS3ReadRole \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::222222222222:root"
            },
            "Action": "sts:AssumeRole"
        }]
    }'

# Attach S3 read policy to the role in Account A
aws iam attach-role-policy \
    --role-name CrossAccountS3ReadRole \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess
```

**Step 2: Grant Account B's Lambda permission to assume the role**

In Account B, attach a policy to the Lambda execution role:
```json
{
    "Effect": "Allow",
    "Action": "sts:AssumeRole",
    "Resource": "arn:aws:iam::111111111111:role/CrossAccountS3ReadRole"
}
```

**Step 3: Lambda code assumes the role**

```python
import boto3

def lambda_handler(event, context):
    # Assume the cross-account role
    sts = boto3.client('sts')
    assumed = sts.assume_role(
        RoleArn='arn:aws:iam::111111111111:role/CrossAccountS3ReadRole',
        RoleSessionName='LambdaCrossAccountSession'
    )

    # Use temporary credentials to access Account A's S3
    s3 = boto3.client('s3',
        aws_access_key_id=assumed['Credentials']['AccessKeyId'],
        aws_secret_access_key=assumed['Credentials']['SecretAccessKey'],
        aws_session_token=assumed['Credentials']['SessionToken']
    )

    response = s3.get_object(Bucket='account-a-bucket', Key='data.json')
    return response['Body'].read()
```

</details>

### Exercise 4: IAM Policy Analysis

Analyze the following IAM policy and answer: what does this policy allow and deny? Are there any security concerns?

```json
{
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
```

<details>
<summary>Show Answer</summary>

**What this policy does**:
- **Statement 1**: Grants full access (`*`) to all AWS services and resources — equivalent to `AdministratorAccess`.
- **Statement 2**: Explicitly denies all IAM actions and all Organizations actions.

**Net effect**: The user can do anything in AWS EXCEPT manage IAM (create/delete users, roles, policies) and manage AWS Organizations. Deny always overrides Allow in IAM.

**Security concerns**:

1. **Still dangerously over-privileged**: Even without IAM access, the user can delete databases, terminate EC2 instances, empty S3 buckets, deploy Lambda functions, modify VPCs, access secrets — essentially everything except modifying permissions.

2. **The IAM deny is insufficient protection**: The user cannot create new roles, but they can still abuse their existing permissions to cause significant damage or exfiltrate data.

3. **The "prevent privilege escalation by denying IAM" pattern is fragile**: A user with broad permissions could potentially still escalate privileges through other means (e.g., abusing Lambda, EC2 user data, or CloudFormation to execute code with higher permissions).

**Better approach**: Start with no permissions and add only what is needed (allowlist), rather than starting with all permissions and trying to deny specific actions (denylist). This violates the principle of least privilege.

</details>

### Exercise 5: GCP Service Account Best Practices

A GCP Compute Engine instance runs a Python application that writes metrics to Cloud Storage and reads configuration from Secret Manager. Create the minimal service account configuration and bind the correct predefined roles.

<details>
<summary>Show Answer</summary>

```bash
# Step 1: Create a dedicated service account (one per application)
gcloud iam service-accounts create my-app-sa \
    --display-name="My Application Service Account" \
    --project=my-project-id

# Step 2: Grant minimal required roles

# Storage Object Creator allows writing objects to Cloud Storage (not reading, not deleting buckets)
gcloud projects add-iam-policy-binding my-project-id \
    --member="serviceAccount:my-app-sa@my-project-id.iam.gserviceaccount.com" \
    --role="roles/storage.objectCreator" \
    --condition="expression=resource.name.startsWith('projects/_/buckets/my-metrics-bucket'),title=only-metrics-bucket"

# Secret Accessor allows reading secret values (not creating or managing secrets)
gcloud projects add-iam-policy-binding my-project-id \
    --member="serviceAccount:my-app-sa@my-project-id.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"

# Step 3: Assign the service account to the Compute Engine instance
gcloud compute instances create my-instance \
    --service-account=my-app-sa@my-project-id.iam.gserviceaccount.com \
    --scopes=cloud-platform \
    --zone=asia-northeast3-a
```

**Best practice notes**:
- **One service account per application** — Never share service accounts across applications. If one is compromised, only that application's permissions are exposed.
- **Use IAM conditions** where possible to scope to specific resources (e.g., specific bucket only).
- **`roles/storage.objectCreator`** instead of `roles/storage.admin` — Only write new objects, cannot delete or read other objects.
- **`roles/secretmanager.secretAccessor`** instead of `roles/secretmanager.admin` — Only read secret values, cannot create or modify secrets.
- **Avoid downloading service account key files** — Use the metadata server for Compute Engine instances instead. Key files are long-lived credentials that must be manually rotated.

</details>

---

## References

- [AWS IAM Documentation](https://docs.aws.amazon.com/iam/)
- [AWS IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
- [GCP IAM Documentation](https://cloud.google.com/iam/docs)
- [GCP IAM Best Practices](https://cloud.google.com/iam/docs/using-iam-securely)
