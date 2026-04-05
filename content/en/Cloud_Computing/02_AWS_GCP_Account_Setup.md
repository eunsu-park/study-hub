# AWS & GCP Account Setup

**Previous**: [Cloud Computing Overview](./01_Cloud_Computing_Overview.md) | **Next**: [Regions and Availability Zones](./03_Regions_Availability_Zones.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Create and configure an AWS account with proper security settings
2. Create and configure a GCP project with billing enabled
3. Enable multi-factor authentication (MFA) on root and admin accounts
4. Set up billing alerts and budget thresholds on both platforms
5. Configure IAM admin users to avoid day-to-day root account usage
6. Navigate the AWS Management Console and GCP Cloud Console

---

Before you can deploy any cloud resource, you need a properly secured account. Poor account setup is one of the most common sources of security breaches and unexpected bills. This lesson walks through the exact steps to create, secure, and configure both AWS and GCP accounts so you start with a solid foundation from day one.

## 1. AWS Account Creation

### 1.1 Account Creation Steps

1. **Access AWS Sign-up Page**
   - Click "Create an AWS Account" at https://aws.amazon.com/

2. **Enter Account Information**
   - Email address (for Root account)
   - AWS account name
   - Password

3. **Contact Information**
   - Account type: Personal or Business
   - Name, address, phone number

4. **Billing Information**
   - Credit card registration (required even for free tier)
   - $1 verification charge (refunded)

5. **Identity Verification**
   - PIN verification via SMS or voice call

6. **Select Support Plan**
   - Recommended: Basic Support (free)

### 1.2 Root Account Security

The Root account has full permissions, so security hardening is essential.

```
⚠️ Root Account Security Checklist
□ Set strong password (16+ characters with special characters)
□ Enable MFA (Multi-Factor Authentication)
□ Do not create access keys
□ Do not use Root account for daily tasks
□ Create and use IAM users instead
```

### 1.3 MFA Setup (AWS)

**Activate MFA from Console:**

1. AWS Console → Top-right account name → "Security credentials"
2. "Multi-factor authentication (MFA)" section
3. Click "Activate MFA"
4. Select MFA device type:
   - **Virtual MFA device**: Use Google Authenticator, Authy app
   - **Hardware TOTP token**: Physical token
   - **Security key**: FIDO security key

**Virtual MFA Setup:**
```
1. Install app: Google Authenticator or Authy
2. Scan QR code in app
3. Enter two consecutive MFA codes
4. Click "Assign MFA"
```

---

## 2. GCP Account Creation

### 2.1 Account Creation Steps

1. **Access GCP Console**
   - https://console.cloud.google.com/

2. **Google Account Login**
   - Use existing Google account or create new one

3. **Accept GCP Terms**
   - Select country
   - Agree to terms of service

4. **Set Up Billing Account** (for free trial)
   - Enter credit card information
   - Activate $300 free credit (90 days)

5. **Create First Project**
   - Specify project name
   - Select organization (select "No organization" for personal use)

### 2.2 GCP Security Settings

**Strengthen Google Account Security:**

```
⚠️ GCP Account Security Checklist
□ Enable 2-Step Verification for Google account
□ Strengthen password security
□ Set up recovery email/phone number
□ Review organization policies (for business)
□ Use service accounts (recommended)
```

### 2.3 2-Step Verification Setup (GCP)

1. Google Account Settings → "Security"
2. Enable "2-Step Verification"
3. Select authentication method:
   - Google prompts
   - Authenticator app (Google Authenticator)
   - Security key
   - Backup codes

---

## 3. Console Navigation

### 3.1 AWS Management Console

**Main UI Components:**

```
┌─────────────────────────────────────────────────────────────┐
│  [AWS Logo]  Service Search Bar        Region ▼  Account ▼  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [Services Menu]                                            │
│   ├── Compute (EC2, Lambda, ECS...)                         │
│   ├── Storage (S3, EBS, EFS...)                             │
│   ├── Database (RDS, DynamoDB...)                           │
│   ├── Networking (VPC, Route 53...)                         │
│   ├── Security (IAM, KMS...)                                │
│   └── Management (CloudWatch, CloudFormation...)            │
│                                                             │
│  [Recently Visited Services]                                │
│  [Favorite Services]                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Useful Features:**
- **Service Search**: Enter service name in top search bar
- **Region Selection**: Select working region from top-right
- **CloudShell**: Browser-based terminal (AWS CLI pre-installed)
- **Resource Groups**: Group resources and manage tags

### 3.2 GCP Console

**Main UI Components:**

```
┌─────────────────────────────────────────────────────────────┐
│  [GCP Logo]  [Project Select ▼]  Search Bar    [Account Icon]│
├──────────────┬──────────────────────────────────────────────┤
│  [Navigation]│                                              │
│   │          │  [Dashboard]                                 │
│   ├─ Compute │   ├── Project info                           │
│   ├─ Storage │   ├── Resource summary                       │
│   ├─ Networking│  ├── API activity                          │
│   ├─ Database│   └── Billing summary                        │
│   ├─ Security│                                              │
│   ├─ Tools   │                                              │
│   └─ Billing │                                              │
│              │                                              │
└──────────────┴──────────────────────────────────────────────┘
```

**Useful Features:**
- **Project Selection**: Switch projects from top-left
- **Cloud Shell**: Terminal icon at top-right (gcloud pre-installed)
- **Pin Services**: Pin frequently used services to menu
- **APIs & Services**: Manage API activation

---

## 4. First Project/Resource Group

### 4.1 AWS: Resource Management with Tags

AWS manages resources with **tags** instead of a project concept.

```bash
# Example of tagging resources during creation
aws ec2 run-instances \
    --image-id ami-12345678 \
    --instance-type t2.micro \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Project,Value=MyApp},{Key=Environment,Value=dev}]'
```

**Tag Best Practices:**

| Tag Key | Example Value | Purpose |
|--------|--------|------|
| Project | MyApp | Track costs by project |
| Environment | dev, staging, prod | Distinguish environments |
| Owner | john@example.com | Identify owner |
| CostCenter | IT-001 | Assign cost center |

### 4.2 GCP: Project Creation

GCP isolates resources and billing by **project**.

**Create Project:**
1. Console top → Project selection dropdown
2. Click "New Project"
3. Enter project name (unique ID auto-generated)
4. Link billing account
5. Click "Create"

```bash
# Create project with gcloud
gcloud projects create my-project-id \
    --name="My Project" \
    --labels=env=dev

# Switch project
gcloud config set project my-project-id
```

**Recommended Project Structure:**

```
Organization (optional)
├── Folder: Development
│   ├── Project: dev-frontend
│   └── Project: dev-backend
├── Folder: Production
│   ├── Project: prod-frontend
│   └── Project: prod-backend
└── Folder: Shared
    └── Project: shared-services
```

---

## 5. Cost Alerting Setup

### 5.1 AWS Budget Alerts

**Set Up AWS Budgets:**

1. AWS Console → "Billing and Cost Management" → "Budgets"
2. Click "Create budget"
3. Select budget type: "Cost budget"
4. Configure budget:
   - Name: "Monthly Budget"
   - Amount: Desired limit (e.g., $50)
   - Period: Monthly

5. Alert conditions:
   - Alert when actual cost reaches 80% of budget
   - Alert when forecasted cost exceeds 100%

6. Alert recipients:
   - Enter email addresses
   - Link SNS topic (optional)

```bash
# Create budget with AWS CLI
aws budgets create-budget \
    --account-id 123456789012 \
    --budget '{
        "BudgetName": "Monthly-50USD",
        "BudgetLimit": {"Amount": "50", "Unit": "USD"},
        "TimeUnit": "MONTHLY",
        "BudgetType": "COST"
    }' \
    --notifications-with-subscribers '[{
        "Notification": {
            "NotificationType": "ACTUAL",
            "ComparisonOperator": "GREATER_THAN",
            "Threshold": 80
        },
        "Subscribers": [{
            "SubscriptionType": "EMAIL",
            "Address": "your@email.com"
        }]
    }]'
```

### 5.2 GCP Budget Alerts

**Set Up GCP Billing Budget:**

1. Console → "Billing" → "Budgets & alerts"
2. Click "Create budget"
3. Configure budget:
   - Name: "Monthly Budget"
   - Projects: All or specific projects
   - Amount: Specified amount (e.g., $50)

4. Alert thresholds:
   - Set alerts at 50%, 90%, 100%

5. Alert channels:
   - Email recipients
   - Cloud Monitoring (optional)
   - Pub/Sub topic (for automation)

```bash
# Create budget with gcloud
gcloud billing budgets create \
    --billing-account=BILLING_ACCOUNT_ID \
    --display-name="Monthly Budget" \
    --budget-amount=50USD \
    --threshold-rule=percent=0.5 \
    --threshold-rule=percent=0.9 \
    --threshold-rule=percent=1.0
```

---

## 6. Free Tier Utilization

### 6.1 AWS Free Tier

| Type | Service | Free Limit |
|------|--------|----------|
| **12 Months Free** | EC2 | t2.micro 750 hours/month |
| | S3 | 5GB storage |
| | RDS | db.t2.micro 750 hours/month |
| | CloudFront | 50GB data transfer |
| **Always Free** | Lambda | 1M requests/month |
| | DynamoDB | 25GB storage, 25 WCU/RCU |
| | SNS | 1M requests/month |
| | CloudWatch | Basic monitoring |

**Monitor Free Tier:**
- Console → "Billing" → "Free Tier" tab to check usage

### 6.2 GCP Free Tier

| Type | Service | Free Limit |
|------|--------|----------|
| **$300 Credit** | All services | 90 days (new accounts) |
| **Always Free** | Compute Engine | 1 e2-micro (specific regions) |
| | Cloud Storage | 5GB (US region) |
| | Cloud Functions | 2M invocations/month |
| | BigQuery | 1TB queries/month, 10GB storage |
| | Cloud Run | 2M requests/month |
| | Firestore | 1GB storage, 50K reads/day |

**Always Free Region Restrictions:**
- Compute Engine e2-micro: Only us-west1, us-central1, us-east1

---

## 7. Initial Security Setup Summary

### 7.1 AWS Initial Security Checklist

```
□ Enable Root account MFA
□ Verify Root access keys are deleted
□ Create IAM users and enable MFA
□ Strengthen IAM password policy
□ Enable CloudTrail (audit logs)
□ Set up budget alerts
□ Verify S3 public access block settings
```

### 7.2 GCP Initial Security Checklist

```
□ Enable 2-Step Verification for Google account
□ Review organization policies (if applicable)
□ Create service accounts (for applications)
□ Grant least privilege IAM roles
□ Enable Cloud Audit Logs
□ Set up budget alerts
□ Review VPC firewall rules
```

---

## 8. Next Steps

- [03_Regions_Availability_Zones.md](./03_Regions_Availability_Zones.md) - Understanding regions and availability zones
- [13_Identity_Access_Management.md](./13_Identity_Access_Management.md) - Detailed IAM setup

---

## Exercises

### Exercise 1: Root Account Security Audit

You have just created a new AWS account and want to lock it down before using it. List all the steps you should take immediately, in the correct priority order, and explain why each step is important.

<details>
<summary>Show Answer</summary>

1. **Enable MFA on the Root account** — The root user has unrestricted access to everything. If compromised without MFA, an attacker can delete all resources, transfer data, and rack up charges with no recourse.
2. **Delete or do not create Root access keys** — Access keys allow programmatic access. Root access keys are especially dangerous because they bypass permission policies.
3. **Create an IAM admin user** — Never use the root account for day-to-day tasks. Create an IAM user with `AdministratorAccess` policy and use that instead.
4. **Enable MFA on the IAM admin user** — Admin IAM users are also high-value targets; MFA adds a critical second factor.
5. **Set up billing alerts** — Prevents surprise charges from accidental resource creation or account compromise.
6. **Enable CloudTrail** — Creates an audit log of all API activity, which is essential for detecting unauthorized access.

</details>

### Exercise 2: GCP vs AWS Account Structure

A team is setting up cloud infrastructure for three separate environments: development, staging, and production.

1. How would you organize these environments in GCP? Describe the recommended project structure.
2. How would you handle the same separation in AWS? What mechanism replaces GCP projects?

<details>
<summary>Show Answer</summary>

1. **GCP structure** — Use a folder-and-project hierarchy:
   ```
   Organization
   ├── Folder: Development
   │   └── Project: myapp-dev
   ├── Folder: Staging
   │   └── Project: myapp-staging
   └── Folder: Production
       └── Project: myapp-prod
   ```
   Each project has its own billing, IAM policies, and resource quotas, providing strong isolation.

2. **AWS structure** — AWS uses **tags** and optionally **AWS Organizations** (separate accounts) for isolation:
   - For lightweight separation within a single account: apply consistent tags (`Environment=dev/staging/prod`) to all resources. Use IAM policies and resource groups to enforce boundaries.
   - For strong isolation: use **AWS Organizations** with separate AWS accounts per environment (recommended for production). Each account has its own IAM, billing, and resource limits.
   - Key difference: GCP's project is a built-in first-class concept; AWS requires deliberate tagging discipline or separate accounts to achieve equivalent isolation.

</details>

### Exercise 3: Budget Alert Configuration

A developer is learning cloud on AWS and wants to ensure they never accidentally spend more than $20 in a month. Write the AWS CLI command to create a budget that sends an email alert at 80% of the budget limit. Use `your@email.com` as the recipient.

<details>
<summary>Show Answer</summary>

```bash
aws budgets create-budget \
    --account-id YOUR_ACCOUNT_ID \
    --budget '{
        "BudgetName": "Monthly-20USD",
        "BudgetLimit": {"Amount": "20", "Unit": "USD"},
        "TimeUnit": "MONTHLY",
        "BudgetType": "COST"
    }' \
    --notifications-with-subscribers '[{
        "Notification": {
            "NotificationType": "ACTUAL",
            "ComparisonOperator": "GREATER_THAN",
            "Threshold": 80
        },
        "Subscribers": [{
            "SubscriptionType": "EMAIL",
            "Address": "your@email.com"
        }]
    }]'
```

**Key parameters explained**:
- `"Amount": "20"` — sets the $20 limit
- `"TimeUnit": "MONTHLY"` — resets each calendar month
- `"NotificationType": "ACTUAL"` — alerts on actual (not forecasted) spend
- `"Threshold": 80` — fires at 80% ($16)

To also add a 100% (forecasted) alert, add a second object in the `notifications-with-subscribers` array with `"NotificationType": "FORECASTED"` and `"Threshold": 100`.

</details>

### Exercise 4: Free Tier Planning

You are building a personal project and want to stay entirely within the free tier for the first year. Your application needs: a web server, a relational database, and file storage for user uploads.

For AWS, which specific services and instance types/configurations would you choose to stay within the free tier? List any limitations you need to be aware of.

<details>
<summary>Show Answer</summary>

| Need | AWS Service | Free Tier Limit | Key Limitations |
|------|-------------|-----------------|-----------------|
| Web server | EC2 **t2.micro** | 750 hours/month (12 months) | Only 1 GB RAM; must stop the instance if running multiple to stay within 750 hours |
| Relational database | RDS **db.t2.micro** | 750 hours/month (12 months) | 20 GB storage; Single-AZ only; MySQL, PostgreSQL, MariaDB, or SQL Server Express supported |
| File storage | S3 | 5 GB standard storage, 20,000 GET requests, 2,000 PUT requests/month (12 months) | Watch request counts for high-traffic apps; egress (outbound) data transfer is NOT free beyond 100 GB/month |

**Important caveats**:
- Free tier limits are per account, not per service instance. If you run two t2.micro instances, you consume 1,500 hours — exceeding the 750-hour limit.
- The 12-month free tier begins at account creation, not when you first use the service.
- Always enable a Free Tier Usage Alert in AWS Billing to get notified before charges occur.

</details>

### Exercise 5: MFA Method Comparison

Compare the three MFA device types available in AWS (Virtual MFA, Hardware TOTP token, Security Key). For each, describe a use case where it would be the most appropriate choice.

<details>
<summary>Show Answer</summary>

| MFA Type | How It Works | Best Use Case |
|----------|-------------|---------------|
| **Virtual MFA device** (e.g., Google Authenticator, Authy) | Time-based one-time password (TOTP) generated in a smartphone app | Individual developers and personal accounts. Free, convenient, and requires no additional hardware. Best for learning and development environments. |
| **Hardware TOTP token** (e.g., Gemalto token) | Dedicated physical device generating TOTP codes | Corporate environments where employees should not use personal phones for work MFA, or situations requiring a dedicated non-networked device for security reasons. |
| **Security key** (FIDO2/WebAuthn, e.g., YubiKey) | Physical USB/NFC key that performs cryptographic authentication | High-security accounts (root account, break-glass admin accounts) where phishing resistance is critical. Security keys are immune to phishing because they validate the domain. Best for production root accounts. |

**Recommendation for most teams**: Use a virtual MFA device for IAM users during development, and a hardware security key (YubiKey) for the root account and any privileged admin accounts.

</details>

---

## References

- [AWS Account Creation Guide](https://docs.aws.amazon.com/accounts/latest/reference/manage-acct-creating.html)
- [GCP Getting Started](https://cloud.google.com/docs/get-started)
- [AWS Free Tier](https://aws.amazon.com/free/)
- [GCP Free Tier](https://cloud.google.com/free)
