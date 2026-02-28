"""
Exercises for Lesson 02: AWS & GCP Account Setup
Topic: Cloud_Computing

Solutions to practice problems from the lesson.
Simulates account setup procedures using Python data structures.
"""


# === Exercise 1: Root Account Security Audit ===
# Problem: List all steps to lock down a new AWS account in priority order.

def exercise_1():
    """Simulate a root account security audit checklist."""

    # Security steps ordered by criticality -- the most dangerous exposure first.
    # Root is the most powerful identity; compromising it means total account loss.
    security_steps = [
        {
            "step": "Enable MFA on the Root account",
            "priority": 1,
            "reason": (
                "The root user has unrestricted access to everything. "
                "If compromised without MFA, an attacker can delete all resources, "
                "transfer data, and rack up charges with no recourse."
            ),
        },
        {
            "step": "Delete or do not create Root access keys",
            "priority": 2,
            "reason": (
                "Access keys allow programmatic access. Root access keys are "
                "especially dangerous because they bypass permission policies."
            ),
        },
        {
            "step": "Create an IAM admin user",
            "priority": 3,
            "reason": (
                "Never use the root account for day-to-day tasks. Create an IAM "
                "user with AdministratorAccess policy and use that instead."
            ),
        },
        {
            "step": "Enable MFA on the IAM admin user",
            "priority": 4,
            "reason": (
                "Admin IAM users are also high-value targets; MFA adds a "
                "critical second factor."
            ),
        },
        {
            "step": "Set up billing alerts",
            "priority": 5,
            "reason": (
                "Prevents surprise charges from accidental resource creation "
                "or account compromise."
            ),
        },
        {
            "step": "Enable CloudTrail",
            "priority": 6,
            "reason": (
                "Creates an audit log of all API activity, essential for "
                "detecting unauthorized access."
            ),
        },
    ]

    print("Root Account Security Audit -- Priority Order:")
    print("-" * 60)
    for item in security_steps:
        status = "[x]"  # Simulating all steps completed
        print(f"  {status} Priority {item['priority']}: {item['step']}")
        print(f"        Why: {item['reason']}")
        print()


# === Exercise 2: GCP vs AWS Account Structure ===
# Problem: Design environment separation for dev/staging/prod on both platforms.

def exercise_2():
    """Compare GCP project structure vs AWS tagging/Organizations approach."""

    # GCP uses a folder-and-project hierarchy for isolation.
    # Each project has its own billing, IAM, and resource quotas.
    gcp_structure = {
        "Organization": {
            "Folder: Development": {"Project: myapp-dev": {}},
            "Folder: Staging": {"Project: myapp-staging": {}},
            "Folder: Production": {"Project: myapp-prod": {}},
        }
    }

    # AWS uses tags (lightweight) or separate accounts via Organizations (strong isolation).
    aws_lightweight = {
        "Single Account": {
            "Tag: Environment=dev": ["EC2 instances", "RDS", "S3 buckets"],
            "Tag: Environment=staging": ["EC2 instances", "RDS", "S3 buckets"],
            "Tag: Environment=prod": ["EC2 instances", "RDS", "S3 buckets"],
        }
    }

    aws_strong = {
        "AWS Organization": {
            "Account: myapp-dev (111111111111)": "Development resources",
            "Account: myapp-staging (222222222222)": "Staging resources",
            "Account: myapp-prod (333333333333)": "Production resources",
        }
    }

    print("GCP: Folder + Project Hierarchy")
    print("-" * 50)
    _print_tree(gcp_structure, indent=2)
    print()
    print("  Each project has its own billing, IAM, and resource quotas.")
    print()

    print("AWS Option A: Tags within a Single Account (lightweight)")
    print("-" * 50)
    for account, envs in aws_lightweight.items():
        print(f"  {account}")
        for tag, resources in envs.items():
            print(f"    {tag}: {', '.join(resources)}")
    print()

    print("AWS Option B: Separate Accounts via AWS Organizations (strong)")
    print("-" * 50)
    for org, accounts in aws_strong.items():
        print(f"  {org}")
        for acct, desc in accounts.items():
            print(f"    {acct} -> {desc}")
    print()

    print("Key difference: GCP's project is a built-in first-class concept;")
    print("AWS requires deliberate tagging discipline or separate accounts.")


def _print_tree(d, indent=0):
    """Helper to print nested dict as a tree."""
    for key, value in d.items():
        print(" " * indent + str(key))
        if isinstance(value, dict):
            _print_tree(value, indent + 2)


# === Exercise 3: Budget Alert Configuration ===
# Problem: Create a $20 monthly budget alert at 80% threshold.

def exercise_3():
    """Simulate AWS Budget alert configuration."""

    # Budget configuration parameters -- simulating the AWS CLI command structure.
    budget_config = {
        "BudgetName": "Monthly-20USD",
        "BudgetLimit": {"Amount": "20", "Unit": "USD"},
        "TimeUnit": "MONTHLY",
        "BudgetType": "COST",
    }

    notification = {
        "NotificationType": "ACTUAL",
        "ComparisonOperator": "GREATER_THAN",
        "Threshold": 80,  # Fires at 80% of $20 = $16
    }

    subscriber = {
        "SubscriptionType": "EMAIL",
        "Address": "your@email.com",
    }

    print("AWS Budget Configuration:")
    print(f"  Budget Name: {budget_config['BudgetName']}")
    print(f"  Limit: ${budget_config['BudgetLimit']['Amount']} {budget_config['BudgetLimit']['Unit']}")
    print(f"  Period: {budget_config['TimeUnit']}")
    print(f"  Type: {budget_config['BudgetType']}")
    print()
    print("Notification Rule:")
    print(f"  Trigger: When {notification['NotificationType']} spend is "
          f"{notification['ComparisonOperator'].replace('_', ' ').lower()} "
          f"{notification['Threshold']}%")
    alert_amount = float(budget_config["BudgetLimit"]["Amount"]) * notification["Threshold"] / 100
    print(f"  Alert fires at: ${alert_amount:.2f}")
    print(f"  Recipient: {subscriber['Address']}")
    print()

    # Simulate the equivalent CLI command
    print("Equivalent AWS CLI command:")
    print("  aws budgets create-budget \\")
    print(f"    --account-id YOUR_ACCOUNT_ID \\")
    print(f"    --budget '{{\"BudgetName\": \"{budget_config['BudgetName']}\", ...}}' \\")
    print(f"    --notifications-with-subscribers '[{{\"Notification\": ...}}]'")


# === Exercise 4: Free Tier Planning ===
# Problem: Plan a personal project within the AWS Free Tier.

def exercise_4():
    """Plan AWS Free Tier usage for a web server + database + storage project."""

    free_tier_services = [
        {
            "need": "Web server",
            "service": "EC2 t2.micro",
            "free_limit": "750 hours/month (12 months)",
            "limitations": [
                "Only 1 GB RAM",
                "Must stop if running multiple to stay within 750 hours",
            ],
        },
        {
            "need": "Relational database",
            "service": "RDS db.t2.micro",
            "free_limit": "750 hours/month (12 months)",
            "limitations": [
                "20 GB storage",
                "Single-AZ only",
                "MySQL, PostgreSQL, MariaDB, or SQL Server Express",
            ],
        },
        {
            "need": "File storage",
            "service": "S3",
            "free_limit": "5 GB standard, 20K GET, 2K PUT/month (12 months)",
            "limitations": [
                "Watch request counts for high-traffic apps",
                "Egress data transfer NOT free beyond 100 GB/month",
            ],
        },
    ]

    print("Free Tier Project Plan:")
    print("=" * 60)
    for svc in free_tier_services:
        print(f"\n  Need: {svc['need']}")
        print(f"  Service: {svc['service']}")
        print(f"  Free Limit: {svc['free_limit']}")
        print(f"  Limitations:")
        for lim in svc["limitations"]:
            print(f"    - {lim}")

    # Compute total monthly cost within free tier
    print("\n\nTotal Monthly Cost (within free tier): $0.00")
    print()
    print("Important Caveats:")
    caveats = [
        "Free tier limits are per account, not per service instance.",
        "The 12-month free tier begins at account creation, not first use.",
        "Always enable a Free Tier Usage Alert in AWS Billing.",
    ]
    for c in caveats:
        print(f"  - {c}")


# === Exercise 5: MFA Method Comparison ===
# Problem: Compare MFA device types and their best use cases.

def exercise_5():
    """Compare MFA device types available in AWS."""

    mfa_types = [
        {
            "type": "Virtual MFA device",
            "examples": "Google Authenticator, Authy",
            "mechanism": "Time-based one-time password (TOTP) in a smartphone app",
            "best_use_case": (
                "Individual developers and personal accounts. Free, convenient, "
                "requires no additional hardware. Best for learning/dev."
            ),
            "cost": "Free",
        },
        {
            "type": "Hardware TOTP token",
            "examples": "Gemalto token",
            "mechanism": "Dedicated physical device generating TOTP codes",
            "best_use_case": (
                "Corporate environments where employees should not use personal "
                "phones for work MFA, or situations requiring a non-networked device."
            ),
            "cost": "$10-50 per device",
        },
        {
            "type": "Security key (FIDO2/WebAuthn)",
            "examples": "YubiKey",
            "mechanism": "Physical USB/NFC key with cryptographic authentication",
            "best_use_case": (
                "High-security accounts (root, break-glass admin). Immune to "
                "phishing because it validates the domain. Best for prod root."
            ),
            "cost": "$25-75 per key",
        },
    ]

    print("MFA Device Type Comparison:")
    print("=" * 70)
    for mfa in mfa_types:
        print(f"\n  Type: {mfa['type']}")
        print(f"  Examples: {mfa['examples']}")
        print(f"  How it works: {mfa['mechanism']}")
        print(f"  Best use case: {mfa['best_use_case']}")
        print(f"  Cost: {mfa['cost']}")

    print("\n\nRecommendation for most teams:")
    print("  - Virtual MFA for IAM users during development")
    print("  - Hardware security key (YubiKey) for root and privileged admin accounts")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Root Account Security Audit ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: GCP vs AWS Account Structure ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Budget Alert Configuration ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: Free Tier Planning ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: MFA Method Comparison ===")
    print("=" * 70)
    exercise_5()

    print("\nAll exercises completed!")
