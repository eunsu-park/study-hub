"""
Exercises for Lesson 04: Virtual Machines (EC2 / Compute Engine)
Topic: Cloud_Computing

Solutions to practice problems from the lesson.
Simulates VM provisioning and pricing decisions using Python.
"""


# === Exercise 1: Instance Type Selection ===
# Problem: Match workloads to appropriate EC2 instance families.

def exercise_1():
    """Select correct instance families for different workloads."""

    # EC2 instance families are optimized for specific workload patterns.
    # Choosing the right family avoids over-provisioning costs.
    instance_families = {
        "t": {"purpose": "Burstable General Purpose", "strength": "Variable CPU with credit system"},
        "m": {"purpose": "Balanced General Purpose", "strength": "Balanced CPU/memory"},
        "c": {"purpose": "Compute Optimized", "strength": "Highest CPU performance per dollar"},
        "r": {"purpose": "Memory Optimized", "strength": "High RAM-to-vCPU ratio"},
        "p/g": {"purpose": "GPU Instances", "strength": "GPU acceleration (training/inference)"},
    }

    workloads = [
        {
            "description": "Web app with variable traffic, 10x spike during promotions",
            "family": "t",
            "instance": "t3.medium or t3.large",
            "reason": (
                "Burstable instances accumulate CPU credits during quiet periods "
                "and spend them during spikes. Cost-effective for variable workloads."
            ),
        },
        {
            "description": "In-memory analytics engine loading 200 GB dataset into RAM",
            "family": "r",
            "instance": "r5.8xlarge (256 GB RAM)",
            "reason": (
                "Memory-optimized instances provide high RAM-to-vCPU ratios. "
                "200 GB dataset requires at least r5.8xlarge (256 GB)."
            ),
        },
        {
            "description": "Mathematical simulations (pure CPU, no GPU) for several hours",
            "family": "c",
            "instance": "c5.2xlarge or c6i.4xlarge",
            "reason": (
                "Compute-optimized instances offer the highest vCPU performance "
                "per dollar for CPU-bound workloads like scientific modeling."
            ),
        },
        {
            "description": "Deep learning model training requiring GPU acceleration",
            "family": "p/g",
            "instance": "p3.2xlarge (V100) or g4dn.xlarge (T4)",
            "reason": (
                "GPU instances: 'p' for high-end training (p3/p4 with V100/A100), "
                "'g' for cost-effective training of smaller models and inference."
            ),
        },
    ]

    print("EC2 Instance Family Reference:")
    print(f"{'Family':<8} {'Purpose':<30} {'Strength'}")
    print("-" * 75)
    for fam, info in instance_families.items():
        print(f"{fam:<8} {info['purpose']:<30} {info['strength']}")
    print()

    for i, w in enumerate(workloads, 1):
        print(f"Workload {i}: {w['description']}")
        print(f"  Family: {w['family']} -> {w['instance']}")
        print(f"  Reason: {w['reason']}")
        print()


# === Exercise 2: SSH Key Pair Creation and Instance Connection ===
# Problem: Describe the complete CLI sequence for key pair + EC2 launch + SSH.

def exercise_2():
    """Simulate the EC2 key pair creation and SSH connection workflow."""

    # SSH key pairs are required for accessing EC2 instances.
    # The private key must be securely stored with restrictive permissions.
    steps = [
        {
            "step": "Create key pair and save private key",
            "command": (
                "aws ec2 create-key-pair \\\n"
                "    --key-name my-web-key \\\n"
                "    --query 'KeyMaterial' \\\n"
                "    --output text > my-web-key.pem\n"
                "chmod 400 my-web-key.pem"
            ),
            "note": "chmod 400 is mandatory; SSH refuses world-readable key files.",
        },
        {
            "step": "Launch instance with key pair",
            "command": (
                "aws ec2 run-instances \\\n"
                "    --image-id ami-0c55b159cbfafe1f0 \\\n"
                "    --instance-type t3.micro \\\n"
                "    --key-name my-web-key \\\n"
                "    --tag-specifications "
                "'ResourceType=instance,Tags=[{Key=Name,Value=WebServer}]'"
            ),
            "note": "Security group must allow inbound TCP port 22.",
        },
        {
            "step": "Get public IP and SSH into the instance",
            "command": (
                "aws ec2 describe-instances \\\n"
                "    --filters \"Name=tag:Name,Values=WebServer\" \\\n"
                "    --query 'Reservations[0].Instances[0].PublicIpAddress'\n\n"
                "ssh -i my-web-key.pem ec2-user@<PUBLIC_IP>"
            ),
            "note": "Default user: ec2-user (Amazon Linux), ubuntu (Ubuntu).",
        },
    ]

    for i, s in enumerate(steps, 1):
        print(f"Step {i}: {s['step']}")
        print(f"  Command:\n    {s['command']}")
        print(f"  Note: {s['note']}")
        print()


# === Exercise 3: Auto Scaling Group Scenario ===
# Problem: Design ASG configuration for predictable weekday traffic pattern.

def exercise_3():
    """Design Auto Scaling Group configuration for time-based traffic."""

    # Auto Scaling uses scheduled policies for predictable patterns
    # and target tracking for unexpected spikes as a safety net.
    asg_config = {
        "min_capacity": 1,
        "max_capacity": 5,
        "desired_capacity": 1,
    }

    scheduled_actions = [
        {
            "name": "scale-up-weekday",
            "schedule": "0 9 * * 1-5",  # 9 AM Mon-Fri
            "desired": 4,
            "description": "Scale up to 4 instances at 9 AM on weekdays",
        },
        {
            "name": "scale-down-weekday",
            "schedule": "0 18 * * 1-5",  # 6 PM Mon-Fri
            "desired": 1,
            "description": "Scale back to 1 instance at 6 PM on weekdays",
        },
    ]

    print("Auto Scaling Group Configuration:")
    print(f"  Minimum capacity: {asg_config['min_capacity']} (always at least 1 instance)")
    print(f"  Maximum capacity: {asg_config['max_capacity']} (prevent runaway costs)")
    print(f"  Desired capacity: {asg_config['desired_capacity']} (initial)")
    print()

    print("Scaling Policy: Scheduled Scaling (predictable time-based pattern)")
    print("-" * 60)
    for action in scheduled_actions:
        print(f"  Action: {action['name']}")
        print(f"    Cron: {action['schedule']}")
        print(f"    Desired: {action['desired']}")
        print(f"    {action['description']}")
        print()

    # Simulate traffic pattern
    hours = list(range(24))
    for h in hours:
        if 9 <= h < 18:
            instances = 4
        else:
            instances = 1
        bar = "#" * instances
        print(f"  {h:02d}:00 [{bar:<5}] {instances} instances")

    print()
    print("Safety net: Add Target Tracking on CPU (maintain 60% avg) for spikes.")


# === Exercise 4: Pricing Model Decision ===
# Problem: Choose pricing model for a nightly batch processing cluster.

def exercise_4():
    """Compare pricing models for a batch processing workload."""

    # Spot Instances are ideal for interruptible, checkpointed batch jobs.
    # The 90% discount makes a massive difference at scale.
    on_demand_rate = 0.34  # c5.2xlarge per hour
    spot_rate = 0.034       # ~90% discount
    instances = 20
    hours_per_night = 6
    nights_per_month = 30

    # Calculate costs
    on_demand_monthly = instances * on_demand_rate * hours_per_night * nights_per_month
    spot_monthly = instances * spot_rate * hours_per_night * nights_per_month
    savings_monthly = on_demand_monthly - spot_monthly
    savings_yearly = savings_monthly * 12

    print("Batch Processing Cluster Pricing Analysis")
    print("=" * 60)
    print(f"  Instance type: c5.2xlarge")
    print(f"  Count: {instances} instances")
    print(f"  Run time: {hours_per_night} hours/night, {nights_per_month} nights/month")
    print(f"  Job characteristic: checkpointed, can tolerate interruption")
    print()

    print("Cost Comparison:")
    print(f"  On-Demand ({on_demand_rate:.3f}/hr): ${on_demand_monthly:,.2f}/month")
    print(f"  Spot ({spot_rate:.3f}/hr):       ${spot_monthly:,.2f}/month")
    print(f"  Monthly savings:           ${savings_monthly:,.2f}")
    print(f"  Yearly savings:            ${savings_yearly:,.2f}")
    print()

    print("Recommendation: Spot Instances")
    print("  - Job is checkpointed -> interruptions can be tolerated")
    print("  - Up to 90% discount vs On-Demand")
    print("  - Use Spot Fleet with multiple instance types/AZs to minimize interruption")
    print()
    print("Why not Reserved? RI requires 1-3 year commitment and is priced for")
    print("continuous usage. These instances only run 6/24 hours (25% utilization).")


# === Exercise 5: GCP Custom Machine Type vs AWS ===
# Problem: Create a custom machine type on GCP and compare to AWS approach.

def exercise_5():
    """Compare GCP custom machine types vs AWS predefined instance catalog."""

    # GCP allows arbitrary CPU/memory combinations; AWS requires choosing
    # from a predefined catalog, which may mean over-provisioning.
    required_vcpu = 6
    required_memory_gb = 20

    gcp_solution = {
        "command": (
            "gcloud compute instances create custom-instance \\\n"
            "    --zone=asia-northeast3-a \\\n"
            f"    --custom-cpu={required_vcpu} \\\n"
            f"    --custom-memory={required_memory_gb}GB \\\n"
            "    --image-family=ubuntu-2204-lts \\\n"
            "    --image-project=ubuntu-os-cloud"
        ),
        "exact_match": True,
    }

    # AWS has no custom types -- find closest predefined matches
    aws_candidates = [
        {"type": "c5.xlarge", "vcpu": 4, "memory_gb": 8, "fit": "too small"},
        {"type": "m5.2xlarge", "vcpu": 8, "memory_gb": 32, "fit": "over-provisioned"},
        {"type": "c5.2xlarge", "vcpu": 8, "memory_gb": 16, "fit": "over on CPU, under on RAM"},
    ]

    print(f"Requirement: {required_vcpu} vCPUs, {required_memory_gb} GB RAM")
    print("=" * 60)
    print()

    print("GCP Solution: Custom Machine Type (exact match)")
    print(f"  Command:\n    {gcp_solution['command']}")
    print(f"  Exact match: {gcp_solution['exact_match']}")
    print(f"  Note: Memory must be a multiple of 256 MB. {required_memory_gb} GB is valid.")
    print()

    print("AWS: No Custom Types -- Closest Predefined Matches:")
    print(f"  {'Type':<15} {'vCPU':<6} {'RAM (GB)':<10} {'Assessment'}")
    print("  " + "-" * 50)
    for c in aws_candidates:
        print(f"  {c['type']:<15} {c['vcpu']:<6} {c['memory_gb']:<10} {c['fit']}")

    print()
    print("Why no custom types in AWS?")
    print("  AWS optimizes hardware/hypervisor around specific instance families for")
    print("  predictable performance. GCP uses a more flexible allocation model.")
    print()
    print("Practical impact: GCP lets you right-size precisely and avoid paying")
    print("for unused vCPUs or RAM. In AWS, you typically over-provision.")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Instance Type Selection ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: SSH Key Pair and Instance Connection ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Auto Scaling Group Scenario ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: Pricing Model Decision ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: GCP Custom Machine Type vs AWS ===")
    print("=" * 70)
    exercise_5()

    print("\nAll exercises completed!")
