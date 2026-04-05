"""
Exercises for Lesson 07: Object Storage (S3 / Cloud Storage)
Topic: Cloud_Computing

Solutions to practice problems from the lesson.
Simulates storage class decisions, lifecycle policies, and cost calculations.
"""


# === Exercise 1: Storage Class Selection ===
# Problem: Choose the most cost-effective S3 storage class for each data category.

def exercise_1():
    """Select optimal S3 storage classes for different access patterns."""

    # Storage class selection depends on access frequency and retrieval requirements.
    # The key trade-off: cheaper storage = more expensive or slower retrieval.
    storage_classes = {
        "S3 Standard": {"storage_cost": 0.025, "retrieval_fee": 0.0, "access": "Frequent"},
        "S3 Intelligent-Tiering": {"storage_cost": 0.025, "retrieval_fee": 0.0, "access": "Variable"},
        "S3 Standard-IA": {"storage_cost": 0.0138, "retrieval_fee": 0.01, "access": "Infrequent"},
        "S3 Glacier Deep Archive": {"storage_cost": 0.00099, "retrieval_fee": 0.02, "access": "Rare"},
    }

    selections = [
        {
            "data": "Thumbnail images for active social media feed (thousands of accesses/day)",
            "class": "S3 Standard",
            "reason": (
                "High-frequency access. IA/Glacier retrieval fees would far "
                "exceed storage savings for thousands of accesses per day."
            ),
        },
        {
            "data": "Raw video footage (frequent first 48hrs, then rarely accessed)",
            "class": "S3 Intelligent-Tiering",
            "reason": (
                "Access pattern shifts dramatically. Intelligent-Tiering "
                "auto-moves objects between tiers without retrieval fees."
            ),
        },
        {
            "data": "Quarterly financial reports retained 7 years for compliance",
            "class": "S3 Glacier Deep Archive",
            "reason": (
                "Accessed < once/year, held 7 years. At $0.00099/GB/month, "
                "25x cheaper than Standard. 12-hour retrieval is acceptable."
            ),
        },
        {
            "data": "System log files (occasional debugging first 30 days, never after)",
            "class": "S3 Standard-IA",
            "reason": (
                "Occasionally accessed first 30 days. Lower storage cost vs "
                "Standard. Add lifecycle rule to Glacier after 30 days."
            ),
        },
    ]

    print("S3 Storage Classes ($/GB/month, Seoul region):")
    print(f"  {'Class':<28} {'Storage':<12} {'Retrieval':<12} {'Access Pattern'}")
    print("  " + "-" * 65)
    for name, info in storage_classes.items():
        print(f"  {name:<28} ${info['storage_cost']:<11} ${info['retrieval_fee']:<11} {info['access']}")
    print()

    for i, s in enumerate(selections, 1):
        print(f"Data {i}: {s['data']}")
        print(f"  Class: {s['class']}")
        print(f"  Reason: {s['reason']}")
        print()


# === Exercise 2: Lifecycle Policy Design ===
# Problem: Design a lifecycle policy for application log files.

def exercise_2():
    """Design an S3 lifecycle policy with transitions and expiration."""

    # Lifecycle policies automate storage class transitions to reduce costs.
    # Each transition moves data to a cheaper tier as access frequency decreases.
    lifecycle_rules = {
        "ID": "log-lifecycle",
        "Status": "Enabled",
        "Filter": {"Prefix": "logs/"},
        "Transitions": [
            {"Days": 7, "StorageClass": "STANDARD_IA",
             "reason": "After 7 days, logs are accessed only for debugging."},
            {"Days": 30, "StorageClass": "GLACIER_FLEXIBLE_RETRIEVAL",
             "reason": "After 30 days, compliance retention only."},
        ],
        "Expiration": {"Days": 365, "reason": "After 1 year, delete to eliminate cost."},
    }

    print("S3 Lifecycle Policy for Application Logs:")
    print("=" * 60)
    print(f"  Rule ID: {lifecycle_rules['ID']}")
    print(f"  Filter: {lifecycle_rules['Filter']}")
    print()

    print("  Timeline:")
    print(f"  Day 0-7:    S3 Standard        (active analysis, fast free retrieval)")
    print(f"  Day 7-30:   S3 Standard-IA      (occasional debugging, per-request fee)")
    print(f"  Day 30-365: Glacier Flexible     (compliance retention, 1-12hr retrieval)")
    print(f"  Day 365+:   DELETED              (automatic expiration)")
    print()

    # Visual timeline
    timeline = [
        (0, 7, "Standard", "#"),
        (7, 30, "Std-IA", "="),
        (30, 365, "Glacier", "-"),
        (365, 400, "Deleted", " "),
    ]
    print("  Visual timeline (each char = ~10 days):")
    bar = ""
    for start, end, label, char in timeline:
        segment_len = max(1, (end - start) // 10)
        bar += char * segment_len
    print(f"  [{bar}]")
    print(f"   {'Standard':>8} {'Std-IA':>5}  {'Glacier':>15}  {'Del':>5}")
    print()

    # JSON representation
    print("  Lifecycle Configuration (JSON):")
    print("  {")
    print('    "Rules": [{')
    print(f'      "ID": "{lifecycle_rules["ID"]}",')
    print(f'      "Status": "Enabled",')
    print(f'      "Filter": {{"Prefix": "logs/"}},')
    print(f'      "Transitions": [')
    for t in lifecycle_rules["Transitions"]:
        print(f'        {{"Days": {t["Days"]}, "StorageClass": "{t["StorageClass"]}"}},')
    print(f'      ],')
    print(f'      "Expiration": {{"Days": {lifecycle_rules["Expiration"]["Days"]}}}')
    print("    }]")
    print("  }")


# === Exercise 3: Bucket Versioning and Public Access ===
# Problem: Protect config files from deletion and ensure no public access.

def exercise_3():
    """Configure versioning and public access block for S3 bucket."""

    bucket = "my-config-bucket"

    commands = [
        {
            "purpose": "Enable versioning (prevent accidental deletion)",
            "command": (
                "aws s3api put-bucket-versioning \\\n"
                f"    --bucket {bucket} \\\n"
                "    --versioning-configuration Status=Enabled"
            ),
            "effect": (
                "Deleted objects get a 'delete marker' (not removed). "
                "Overwritten files keep the old version."
            ),
        },
        {
            "purpose": "Block all public access",
            "command": (
                "aws s3api put-public-access-block \\\n"
                f"    --bucket {bucket} \\\n"
                "    --public-access-block-configuration \\\n"
                "        BlockPublicAcls=true,\\\n"
                "        IgnorePublicAcls=true,\\\n"
                "        BlockPublicPolicy=true,\\\n"
                "        RestrictPublicBuckets=true"
            ),
            "effect": "Even if someone adds a public policy, it is blocked.",
        },
    ]

    print(f"Securing S3 Bucket: {bucket}")
    print("=" * 60)
    for cmd in commands:
        print(f"\n  Purpose: {cmd['purpose']}")
        print(f"  Command:\n    {cmd['command']}")
        print(f"  Effect: {cmd['effect']}")

    print()
    print("  Effect of versioning on 'aws s3 rm':")
    print("    -> Creates a delete marker; the object is hidden but recoverable.")
    print("    -> To permanently delete, you must specify the version ID.")


# === Exercise 4: Pre-Signed URL Use Case ===
# Problem: Generate a pre-signed URL for private object download.

def exercise_4():
    """Generate and explain pre-signed URL for private S3 object access."""

    bucket = "my-reports-bucket"
    key = "reports/q3-summary.pdf"
    expires_in = 3600  # 1 hour

    print("Pre-Signed URL for Private Object Access:")
    print("=" * 50)
    print(f"  Bucket: {bucket}")
    print(f"  Object: {key}")
    print(f"  Expiry: {expires_in} seconds ({expires_in // 3600} hour)")
    print()

    print("  Command:")
    print(f"    aws s3 presign s3://{bucket}/{key} --expires-in {expires_in}")
    print()

    print("  Generated URL (example):")
    print(f"    https://{bucket}.s3.amazonaws.com/{key}"
          "?X-Amz-Algorithm=AWS4-HMAC-SHA256"
          "&X-Amz-Credential=...&X-Amz-Expires=3600&X-Amz-Signature=...")
    print()

    print("  How it works:")
    steps = [
        "AWS signs the URL using the credentials of the generating IAM entity.",
        "The signature encodes: bucket, key, expiry time, and identity.",
        "Anyone with the URL can access the object via HTTP GET -- no AWS creds.",
        f"After {expires_in} seconds, the signature expires -> 403 Forbidden.",
    ]
    for i, step in enumerate(steps, 1):
        print(f"    {i}. {step}")

    print()
    print("  Security note: If the generating role loses S3 read permission")
    print("  before expiry, the URL also stops working.")


# === Exercise 5: Cross-Region Replication Setup ===
# Problem: Describe prerequisites and steps for S3 CRR configuration.

def exercise_5():
    """Configure S3 Cross-Region Replication for disaster recovery."""

    source_bucket = "source-bucket-seoul"
    source_region = "ap-northeast-2"
    dest_bucket = "destination-bucket-virginia"
    dest_region = "us-east-1"

    print("S3 Cross-Region Replication (CRR) Setup:")
    print("=" * 60)
    print(f"  Source: {source_bucket} ({source_region})")
    print(f"  Destination: {dest_bucket} ({dest_region})")
    print()

    print("Prerequisites:")
    prereqs = [
        "Versioning must be enabled on BOTH source and destination buckets.",
        "An IAM role that grants S3 permission to read source and write destination.",
    ]
    for p in prereqs:
        print(f"  - {p}")
    print()

    steps = [
        {
            "step": "Enable versioning on source bucket",
            "command": (
                f"aws s3api put-bucket-versioning \\\n"
                f"    --bucket {source_bucket} \\\n"
                f"    --versioning-configuration Status=Enabled"
            ),
        },
        {
            "step": "Create destination bucket and enable versioning",
            "command": (
                f"aws s3api create-bucket --bucket {dest_bucket} --region {dest_region}\n"
                f"aws s3api put-bucket-versioning \\\n"
                f"    --bucket {dest_bucket} \\\n"
                f"    --versioning-configuration Status=Enabled"
            ),
        },
        {
            "step": "Configure replication rule",
            "command": (
                f"aws s3api put-bucket-replication \\\n"
                f"    --bucket {source_bucket} \\\n"
                f"    --replication-configuration '{{...}}'"
            ),
        },
    ]

    print("Steps:")
    for i, s in enumerate(steps, 1):
        print(f"  Step {i}: {s['step']}")
        print(f"    {s['command']}")
        print()

    print("Important Notes:")
    notes = [
        "CRR only replicates NEW objects after the rule is configured.",
        "Existing objects are NOT replicated -- use S3 Batch Operations.",
        "Delete markers are not replicated by default (configurable).",
        "Data transfer from Seoul to Virginia incurs egress charges.",
    ]
    for n in notes:
        print(f"  - {n}")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Storage Class Selection ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: Lifecycle Policy Design ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Bucket Versioning and Public Access ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: Pre-Signed URL Use Case ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: Cross-Region Replication Setup ===")
    print("=" * 70)
    exercise_5()

    print("\nAll exercises completed!")
