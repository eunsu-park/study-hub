"""
Exercises for Lesson 03: Regions and Availability Zones
Topic: Cloud_Computing

Solutions to practice problems from the lesson.
Simulates cloud infrastructure concepts using Python data structures.
"""


# === Exercise 1: Infrastructure Hierarchy Identification ===
# Problem: Identify the scope (Global, Regional, or AZ) for each AWS resource.

def exercise_1():
    """Classify AWS resources by their infrastructure scope."""

    # AWS resources exist at different scopes in the infrastructure hierarchy.
    # Understanding scope is essential for designing HA architectures.
    resources = [
        {
            "resource": "An EC2 instance running in ap-northeast-2a",
            "scope": "Availability Zone",
            "reason": "EC2 instances are tied to a specific AZ.",
        },
        {
            "resource": "An S3 bucket named my-data-bucket",
            "scope": "Regional",
            "reason": (
                "S3 buckets are regional; AWS replicates data across AZs "
                "within the region automatically."
            ),
        },
        {
            "resource": "An IAM role named EC2ReadOnlyRole",
            "scope": "Global",
            "reason": (
                "IAM is a global service. Roles, users, and policies "
                "apply across all regions in an account."
            ),
        },
        {
            "resource": "An RDS Multi-AZ instance",
            "scope": "Regional",
            "reason": (
                "Multi-AZ RDS is a regional resource; it manages a standby "
                "replica in a different AZ within the same region."
            ),
        },
        {
            "resource": "An EBS volume in us-east-1b",
            "scope": "Availability Zone",
            "reason": (
                "EBS volumes are tied to the AZ where they are created. "
                "They cannot be directly attached to an instance in another AZ."
            ),
        },
        {
            "resource": "A Route 53 hosted zone",
            "scope": "Global",
            "reason": (
                "Route 53 is a global service. Hosted zones and DNS records "
                "are accessible globally."
            ),
        },
    ]

    # Group by scope for a clear summary
    scopes = {"Global": [], "Regional": [], "Availability Zone": []}
    for r in resources:
        scopes[r["scope"]].append(r["resource"])

    print("AWS Resource Scope Classification:")
    print("-" * 60)
    for r in resources:
        print(f"  {r['resource']}")
        print(f"    Scope: {r['scope']}")
        print(f"    Reason: {r['reason']}")
        print()

    print("\nSummary by Scope:")
    for scope, items in scopes.items():
        print(f"  {scope}: {len(items)} resources")
        for item in items:
            print(f"    - {item}")


# === Exercise 2: Region Selection Decision ===
# Problem: Choose regions for a Korean healthcare startup with PIPA compliance.

def exercise_2():
    """Evaluate region selection based on compliance and latency requirements."""

    # Compliance requirements drive region selection decisions.
    # Data residency laws override latency and cost considerations.
    decisions = [
        {
            "question": "Primary region for main application and database?",
            "answer": "ap-northeast-2 (Seoul)",
            "reason": (
                "Patient data must remain within South Korea per PIPA. Seoul is "
                "the only AWS region on South Korean soil, so it is mandatory."
            ),
        },
        {
            "question": "Can us-east-1 be used for any part of the system?",
            "answer": "Yes, for global/non-regulated services",
            "reason": (
                "us-east-1 hosts global services (IAM, CloudFront distributions). "
                "Acceptable for non-patient data: anonymized analytics, internal "
                "tooling, static marketing assets."
            ),
        },
        {
            "question": "What region consideration applies for EU customers?",
            "answer": "eu-west-1 (Ireland) or eu-central-1 (Frankfurt)",
            "reason": (
                "GDPR may require EU patients' data to stay within the EU. "
                "Multi-region architecture: ap-northeast-2 for Korean patients, "
                "EU region for EU patients."
            ),
        },
    ]

    print("Region Selection for Healthcare Startup:")
    print("=" * 60)
    print("Context: South Korean healthcare startup, PIPA compliance required")
    print()
    for i, d in enumerate(decisions, 1):
        print(f"  Q{i}: {d['question']}")
        print(f"  A:  {d['answer']}")
        print(f"      {d['reason']}")
        print()


# === Exercise 3: Multi-AZ Architecture Design ===
# Problem: Design a highly available web application across two AZs.

def exercise_3():
    """Design a Multi-AZ architecture for a web application."""

    # Simulate the architecture with nested data structures.
    # Each component is placed in a specific AZ for high availability.
    architecture = {
        "region": "ap-northeast-2 (Seoul)",
        "load_balancer": {
            "type": "Application Load Balancer",
            "scope": "Regional (spans all AZs)",
        },
        "availability_zones": {
            "ap-northeast-2a": {
                "EC2 App Server": "App-1 (Auto Scaling Group)",
                "RDS": "Primary (active, synchronous replication to AZ-b)",
            },
            "ap-northeast-2b": {
                "EC2 App Server": "App-2 (Auto Scaling Group)",
                "RDS": "Standby (passive, automatic failover in ~2 min)",
            },
        },
    }

    print("Multi-AZ Architecture Design:")
    print("=" * 60)
    print(f"Region: {architecture['region']}")
    print()
    print(f"[{architecture['load_balancer']['type']}]")
    print(f"  Scope: {architecture['load_balancer']['scope']}")
    print()

    for az, components in architecture["availability_zones"].items():
        print(f"  {az}:")
        for component, detail in components.items():
            print(f"    {component}: {detail}")
        print()

    print("Component Explanations:")
    explanations = [
        (
            "ALB",
            "Routes traffic to healthy instances across both AZs. "
            "If AZ-a fails, it only routes to AZ-b instances."
        ),
        (
            "EC2 in Auto Scaling Group",
            "Spread across AZs. If one AZ fails, ASG launches "
            "replacement instances in the surviving AZ."
        ),
        (
            "RDS Multi-AZ",
            "Primary in AZ-a, synchronous standby in AZ-b. "
            "Failover completes in ~2 minutes automatically."
        ),
    ]
    for name, explanation in explanations:
        print(f"  - {name}: {explanation}")

    print()
    print("Why two AZs? Independent power, cooling, and networking means")
    print("an outage in one AZ does not affect the other.")


# === Exercise 4: AWS vs GCP VPC Scope Difference ===
# Problem: Explain the fundamental difference between AWS and GCP VPC scope.

def exercise_4():
    """Compare AWS (regional) vs GCP (global) VPC models."""

    # AWS VPCs are regional; GCP VPCs are global. This has significant
    # implications for multi-region networking complexity.
    aws_vpc = {
        "scope": "Regional",
        "model": {
            "VPC in us-east-1": {
                "Subnet-a (us-east-1a)": "10.0.1.0/24",
                "Subnet-b (us-east-1b)": "10.0.2.0/24",
            },
            "VPC in ap-northeast-2 (separate)": {
                "Subnet-a (ap-northeast-2a)": "10.1.1.0/24",
            },
        },
        "cross_region": "Requires VPC Peering or Transit Gateway",
    }

    gcp_vpc = {
        "scope": "Global",
        "model": {
            "Single VPC (global)": {
                "Subnet-us (us-central1)": "10.0.1.0/24",
                "Subnet-asia (asia-northeast3)": "10.0.2.0/24",
                "Subnet-eu (europe-west1)": "10.0.3.0/24",
            },
        },
        "cross_region": "Automatic private IP communication within the same VPC",
    }

    print("AWS VPC Model (Regional):")
    print("-" * 50)
    for vpc, subnets in aws_vpc["model"].items():
        print(f"  {vpc}")
        for subnet, cidr in subnets.items():
            print(f"    {subnet}: {cidr}")
    print(f"  Cross-region: {aws_vpc['cross_region']}")
    print()

    print("GCP VPC Model (Global):")
    print("-" * 50)
    for vpc, subnets in gcp_vpc["model"].items():
        print(f"  {vpc}")
        for subnet, cidr in subnets.items():
            print(f"    {subnet}: {cidr}")
    print(f"  Cross-region: {gcp_vpc['cross_region']}")
    print()

    print("Scenario where GCP is simpler:")
    print("  A global microservices application with services in multiple regions.")
    print("  In GCP, all services share one VPC -- a database in asia-northeast3")
    print("  can be reached via private IP from an API server in us-central1")
    print("  without any peering setup.")
    print()
    print("  In AWS, you would need VPC Peering or Transit Gateway between")
    print("  regional VPCs and must manage CIDR ranges to avoid overlap.")


# === Exercise 5: CLI Commands for Infrastructure Discovery ===
# Problem: Write CLI commands to query region/AZ information.

def exercise_5():
    """Show CLI commands for infrastructure discovery on AWS and GCP."""

    # These commands would be run in a real terminal. Here we document
    # the exact command syntax and what each does.
    commands = [
        {
            "task": "List all AZs in Tokyo region (ap-northeast-1) with state (AWS)",
            "command": (
                "aws ec2 describe-availability-zones \\\n"
                "    --region ap-northeast-1 \\\n"
                "    --query 'AvailabilityZones[*].[ZoneName,State]' \\\n"
                "    --output table"
            ),
            "expected_output": [
                ("ap-northeast-1a", "available"),
                ("ap-northeast-1c", "available"),
                ("ap-northeast-1d", "available"),
            ],
        },
        {
            "task": "List all zones in Seoul region (asia-northeast3) (GCP)",
            "command": (
                "gcloud compute zones list \\\n"
                "    --filter=\"region:asia-northeast3\" \\\n"
                "    --format=\"table(name,status)\""
            ),
            "expected_output": [
                ("asia-northeast3-a", "UP"),
                ("asia-northeast3-b", "UP"),
                ("asia-northeast3-c", "UP"),
            ],
        },
        {
            "task": "Set default region to Seoul (AWS)",
            "command": "aws configure set region ap-northeast-2",
            "expected_output": [],
        },
    ]

    for i, cmd in enumerate(commands, 1):
        print(f"Task {i}: {cmd['task']}")
        print(f"  Command:\n    {cmd['command']}")
        if cmd["expected_output"]:
            print(f"  Expected Output:")
            for row in cmd["expected_output"]:
                print(f"    {row[0]:<25} {row[1]}")
        else:
            print(f"  (No output -- updates configuration)")
        print()


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Infrastructure Hierarchy Identification ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: Region Selection Decision ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Multi-AZ Architecture Design ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: AWS vs GCP VPC Scope Difference ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: CLI Commands for Infrastructure Discovery ===")
    print("=" * 70)
    exercise_5()

    print("\nAll exercises completed!")
