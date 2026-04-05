"""
Exercises for Lesson 14: Security Services
Topic: Cloud_Computing

Solutions to practice problems from the lesson.
Simulates defense-in-depth mapping, KMS encryption, WAF rules, and audit trails.
"""


# === Exercise 1: Defense-in-Depth Layer Mapping ===
def exercise_1():
    """Match security concerns to appropriate AWS services."""

    print("Defense-in-Depth Layer Mapping:")
    print("=" * 70)
    print()
    print("  Architecture: Web app on EC2 behind ALB")
    print()

    # Each security concern is addressed by a specific service/feature.
    # Defense-in-depth means each layer handles a different threat vector.
    mappings = [
        {
            "concern": "Block SQL injection attacks from the internet",
            "service": "AWS WAF with AWSManagedRulesSQLiRuleSet",
            "layer": "Application edge",
        },
        {
            "concern": "Detect suspicious API calls (e.g., unusual IAM activity)",
            "service": "Amazon GuardDuty",
            "layer": "Account level",
        },
        {
            "concern": "Encrypt the RDS database at rest with a customer-managed key",
            "service": "AWS KMS (customer managed key) + RDS encryption",
            "layer": "Data at rest",
        },
        {
            "concern": "Prevent direct internet access to the EC2 instances",
            "service": "Private subnet (no public IP) + Security Group",
            "layer": "Network access",
        },
        {
            "concern": "Rotate the database password automatically every 30 days",
            "service": "AWS Secrets Manager with automatic rotation",
            "layer": "Credential hygiene",
        },
    ]

    print(f"  {'Security Concern':<55} {'AWS Service/Feature'}")
    print("  " + "-" * 90)
    for m in mappings:
        print(f"  {m['concern']:<55} {m['service']}")
    print()

    print("  Defense-in-Depth Layers:")
    for m in mappings:
        print(f"    {m['layer']}: {m['service']}")
    print()
    print("  Each layer handles a different threat vector -- no single service")
    print("  provides complete security on its own.")


# === Exercise 2: KMS Encryption Workflow ===
def exercise_2():
    """Create a KMS key, encrypt a file, and configure S3 default encryption."""

    print("KMS Encryption Workflow:")
    print("=" * 70)
    print()
    print("  Goal: Encrypt config.json before storing in S3")
    print()

    steps = [
        {
            "step": "Create customer managed KMS key with alias",
            "commands": [
                (
                    "aws kms create-key \\\n"
                    '    --description "Config file encryption key" \\\n'
                    "    --key-usage ENCRYPT_DECRYPT \\\n"
                    "    --origin AWS_KMS"
                ),
                (
                    "# Note the KeyId from output, then create alias\n"
                    "aws kms create-alias \\\n"
                    "    --alias-name alias/config-key \\\n"
                    "    --target-key-id <KeyId-from-above>"
                ),
            ],
            # Why customer managed key: Gives you control over the key policy
            # (who can use it), rotation schedule, and deletion. AWS-managed
            # keys cannot be configured or deleted.
            "note": "Customer managed keys give control over key policy, rotation, and deletion.",
        },
        {
            "step": "Encrypt the file using the KMS key",
            "commands": [
                (
                    "aws kms encrypt \\\n"
                    "    --key-id alias/config-key \\\n"
                    "    --plaintext fileb://config.json \\\n"
                    "    --output text \\\n"
                    "    --query CiphertextBlob | base64 --decode > config.json.enc"
                ),
            ],
            "note": "Output is base64-encoded; must decode before storage.",
        },
        {
            "step": "Configure S3 bucket default encryption with SSE-KMS",
            "commands": [
                (
                    "aws s3api put-bucket-encryption \\\n"
                    "    --bucket my-config-bucket \\\n"
                    "    --server-side-encryption-configuration '{\n"
                    '        "Rules": [{\n'
                    '            "ApplyServerSideEncryptionByDefault": {\n'
                    '                "SSEAlgorithm": "aws:kms",\n'
                    '                "KMSMasterKeyID": "alias/config-key"\n'
                    "            }\n"
                    "        }]\n"
                    "    }'"
                ),
            ],
            "note": "SSE-KMS encrypts objects automatically on upload; no manual encryption needed.",
        },
    ]

    for i, s in enumerate(steps, 1):
        print(f"  Step {i}: {s['step']}")
        for cmd in s["commands"]:
            print(f"    {cmd}")
            print()
        print(f"    Note: {s['note']}")
        print()


# === Exercise 3: WAF Rule Design ===
def exercise_3():
    """Design a GCP Cloud Armor policy to protect checkout endpoint from bots."""

    print("GCP Cloud Armor WAF Rule Design:")
    print("=" * 70)
    print()
    print("  Problem: POST /api/checkout receiving thousands of fake orders/min")
    print("  from automated bots.")
    print()

    rules = [
        {
            "priority": 1000,
            "description": "Block known attack patterns (SQL injection)",
            "command": (
                "gcloud compute security-policies rules create 1000 \\\n"
                "    --security-policy=checkout-protection \\\n"
                '    --expression="evaluatePreconfiguredWaf(\'sqli-v33-stable\')" \\\n'
                "    --action=deny-403"
            ),
            # Why priority 1000: Lower number = higher priority. SQL injection
            # blocking is evaluated first, before rate limiting.
            "rationale": "Evaluated first (lowest priority number). Blocks SQL injection payloads.",
        },
        {
            "priority": 2000,
            "description": "Rate limit: max 10 requests/min per IP to checkout",
            "command": (
                "gcloud compute security-policies rules create 2000 \\\n"
                "    --security-policy=checkout-protection \\\n"
                '    --expression="request.path.matches(\'/api/checkout\')" \\\n'
                "    --action=rate-based-ban \\\n"
                "    --rate-limit-threshold-count=10 \\\n"
                "    --rate-limit-threshold-interval-sec=60 \\\n"
                "    --ban-duration-sec=300"
            ),
            "rationale": "Temporarily blocks IPs exceeding 10 req/min for 5 minutes.",
        },
    ]

    print("  Step 1: Create security policy")
    print("    gcloud compute security-policies create checkout-protection \\")
    print('        --description="Protect checkout endpoint from abuse"')
    print()

    for rule in rules:
        print(f"  Rule (priority {rule['priority']}): {rule['description']}")
        print(f"    {rule['command']}")
        print(f"    Rationale: {rule['rationale']}")
        print()

    print("  Step 3: Attach to backend service")
    print("    gcloud compute backend-services update checkout-backend \\")
    print("        --security-policy=checkout-protection \\")
    print("        --global")
    print()

    print("  Design Notes:")
    print("    - Lower priority number = higher evaluation priority")
    print("    - Rate-based banning reduces bot traffic without blocking all users")
    print("    - For sophisticated bots, consider reCAPTCHA integration")


# === Exercise 4: Secrets Manager vs KMS ===
def exercise_4():
    """Explain when to use Secrets Manager vs KMS directly."""

    print("Secrets Manager vs KMS -- When to Use Which:")
    print("=" * 70)
    print()
    print("  Question: Store a third-party API key for a Lambda function.")
    print("  Answer: Use AWS Secrets Manager.")
    print()

    # Comparison table
    features = [
        ("Purpose", "Store and retrieve secret values", "Encrypt/decrypt data; manage keys"),
        ("Secret storage", "Yes -- stores the secret value", "No -- only manages keys"),
        ("Auto rotation", "Yes -- native Lambda rotation", "No -- build rotation yourself"),
        ("Access control", "IAM policy on the secret", "Key policy + IAM policy"),
        ("Cost", "$0.40/secret/month + API calls", "$1/key/month + API calls"),
    ]

    print(f"  {'Feature':<18} {'Secrets Manager':<35} {'KMS (direct)'}")
    print("  " + "-" * 75)
    for feat, sm, kms in features:
        print(f"  {feat:<18} {sm:<35} {kms}")
    print()

    # Recommended approach
    print("  Recommended Implementation:")
    print("    # Store the API key")
    print("    aws secretsmanager create-secret \\")
    print("        --name /lambda/third-party-api-key \\")
    print('        --secret-string \'{"api_key": "sk-abc123..."}\'')
    print()
    print("    # Lambda retrieves it at runtime")
    print("    import boto3, json")
    print("    def get_api_key():")
    print("        client = boto3.client('secretsmanager')")
    print("        response = client.get_secret_value(")
    print("            SecretId='/lambda/third-party-api-key')")
    print("        return json.loads(response['SecretString'])['api_key']")
    print()

    # Why Secrets Manager wraps KMS: You are not choosing between them.
    # Secrets Manager uses KMS internally to encrypt the stored secret.
    # It adds a higher-level API for storing, retrieving, and rotating secrets.
    print("  Key insight: Secrets Manager USES KMS internally to encrypt secrets.")
    print("  You're not choosing between them -- Secrets Manager wraps KMS to")
    print("  provide a higher-level secret storage service.")
    print()
    print("  Use KMS directly only when you need to encrypt your own data blobs")
    print("  (e.g., application-level encryption of database fields).")


# === Exercise 5: Security Audit Trail ===
def exercise_5():
    """Configure CloudTrail logging, retention, and console login alerts."""

    print("Security Audit Trail Configuration:")
    print("=" * 70)
    print()
    print("  Requirements:")
    print("    - Log ALL AWS API calls in the account")
    print("    - Retain logs for 1 year")
    print("    - Protect from deletion")
    print("    - Alert on Console logins")
    print()

    steps = [
        {
            "step": "Create S3 bucket for log storage with versioning",
            "commands": [
                "aws s3api create-bucket \\\n"
                "    --bucket my-cloudtrail-logs \\\n"
                "    --region us-east-1",
                "# Enable versioning (protects against accidental deletion)\n"
                "aws s3api put-bucket-versioning \\\n"
                "    --bucket my-cloudtrail-logs \\\n"
                "    --versioning-configuration Status=Enabled",
                "# Lifecycle: Glacier after 90 days, expire after 1 year\n"
                "aws s3api put-bucket-lifecycle-configuration \\\n"
                "    --bucket my-cloudtrail-logs \\\n"
                "    --lifecycle-configuration '{\n"
                '        "Rules": [{\n'
                '            "Status": "Enabled",\n'
                '            "Transitions": [{"Days": 90, "StorageClass": "GLACIER"}],\n'
                '            "Expiration": {"Days": 365},\n'
                '            "Filter": {"Prefix": ""}\n'
                "        }]\n"
                "    }'",
            ],
        },
        {
            "step": "Create multi-region CloudTrail with log file validation",
            "commands": [
                # Why multi-region: Captures API activity from ALL regions in one trail.
                # Why log file validation: Creates a digest file to detect tampering.
                "aws cloudtrail create-trail \\\n"
                "    --name org-audit-trail \\\n"
                "    --s3-bucket-name my-cloudtrail-logs \\\n"
                "    --is-multi-region-trail \\\n"
                "    --enable-log-file-validation",
                "aws cloudtrail start-logging --name org-audit-trail",
            ],
        },
        {
            "step": "Create CloudWatch metric filter for Console logins",
            "commands": [
                "aws logs put-metric-filter \\\n"
                "    --log-group-name CloudTrail/logs \\\n"
                "    --filter-name ConsoleLoginFilter \\\n"
                "    --filter-pattern '{ $.eventName = \"ConsoleLogin\" }' \\\n"
                "    --metric-transformations "
                "metricName=ConsoleLoginCount,metricNamespace=Security,metricValue=1",
            ],
        },
        {
            "step": "Create alarm to alert on Console logins",
            "commands": [
                "aws cloudwatch put-metric-alarm \\\n"
                "    --alarm-name ConsoleLoginAlert \\\n"
                "    --metric-name ConsoleLoginCount \\\n"
                "    --namespace Security \\\n"
                "    --statistic Sum \\\n"
                "    --period 300 \\\n"
                "    --threshold 1 \\\n"
                "    --comparison-operator GreaterThanOrEqualToThreshold \\\n"
                "    --evaluation-periods 1 \\\n"
                "    --alarm-actions arn:aws:sns:...:security-alerts",
            ],
        },
    ]

    for i, s in enumerate(steps, 1):
        print(f"  Step {i}: {s['step']}")
        for cmd in s["commands"]:
            print(f"    {cmd}")
            print()

    print("  Key Points:")
    points = [
        "--is-multi-region-trail captures API activity from ALL regions.",
        "--enable-log-file-validation creates digest files to detect tampering.",
        "Use S3 Object Lock (WORM mode) for stricter immutability (compliance).",
        "CloudTrail + CloudWatch metric filters = near-real-time security alerting.",
    ]
    for p in points:
        print(f"    - {p}")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Defense-in-Depth Layer Mapping ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: KMS Encryption Workflow ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: WAF Rule Design ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: Secrets Manager vs KMS ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: Security Audit Trail ===")
    print("=" * 70)
    exercise_5()

    print("\nAll exercises completed!")
