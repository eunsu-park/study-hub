"""
Exercises for Lesson 08: Block and File Storage
Topic: Cloud_Computing

Solutions to practice problems from the lesson.
Simulates storage decisions, cost calculations, and EBS/EFS architecture design.
"""


# === Exercise 1: Storage Type Selection ===
# Problem: Select the most appropriate storage type for each workload.

def exercise_1():
    """Choose between block, file, and object storage for various workloads."""

    workloads = [
        {
            "description": "PostgreSQL database on EC2 requiring low-latency random I/O",
            "storage_type": "Block storage (EBS)",
            "reason": (
                "Databases require block-level access for random I/O operations. "
                "EBS gp3 or io2 provides the low latency and high IOPS needed."
            ),
        },
        {
            "description": "CMS across 10 EC2 web servers sharing uploaded media files",
            "storage_type": "File storage (EFS)",
            "reason": (
                "Multiple instances mounting the same filesystem simultaneously "
                "is the defining use case for NFS-based file storage. EBS can "
                "only attach to one instance in read-write mode."
            ),
        },
        {
            "description": "Log aggregation archiving 500 GB/day, rarely read",
            "storage_type": "Object storage (S3)",
            "reason": (
                "Write-once, read-rarely, massive volume. S3's pay-per-GB "
                "pricing and lifecycle rules (auto-transition to Glacier) "
                "make it most cost-effective."
            ),
        },
        {
            "description": "Temporary scratch space for intermediate computation files",
            "storage_type": "Instance Store (local SSD)",
            "reason": (
                "Highest IOPS/throughput with no additional cost. Data loss on "
                "stop is acceptable for temporary scratch. Avoids EBS cost for "
                "ephemeral data."
            ),
        },
    ]

    for i, w in enumerate(workloads, 1):
        print(f"Workload {i}: {w['description']}")
        print(f"  Storage: {w['storage_type']}")
        print(f"  Reason: {w['reason']}")
        print()


# === Exercise 2: EBS Volume Type Selection ===
# Problem: Select EBS volume type for a high-frequency trading database.

def exercise_2():
    """Select and create the right EBS volume type for HFT workloads."""

    requirements = {
        "iops": 50_000,
        "throughput_mbps": 500,
        "capacity_tb": 2,
        "latency": "sub-millisecond",
    }

    # Compare volume types against requirements
    volume_types = [
        {"type": "gp3", "max_iops": 16_000, "suitable": False,
         "reason": "Max 16,000 IOPS -- insufficient for 50,000 IOPS."},
        {"type": "io2", "max_iops": 64_000, "suitable": True,
         "reason": "Supports up to 64,000 IOPS (256,000 with Block Express)."},
        {"type": "st1", "max_iops": 500, "suitable": False,
         "reason": "HDD-based, unsuitable for random I/O with latency requirements."},
    ]

    print("Requirements:")
    print(f"  IOPS: {requirements['iops']:,}")
    print(f"  Throughput: {requirements['throughput_mbps']} MB/s")
    print(f"  Capacity: {requirements['capacity_tb']} TB")
    print(f"  Latency: {requirements['latency']}")
    print()

    print("Volume Type Analysis:")
    for vt in volume_types:
        status = "OK" if vt["suitable"] else "FAIL"
        print(f"  [{status:>4}] {vt['type']}: max {vt['max_iops']:,} IOPS -- {vt['reason']}")
    print()

    print("Selected: io2 (Provisioned IOPS SSD)")
    print("  Command:")
    print("    aws ec2 create-volume \\")
    print("        --availability-zone ap-northeast-2a \\")
    print(f"        --size {requirements['capacity_tb'] * 1000} \\")
    print("        --volume-type io2 \\")
    print(f"        --iops {requirements['iops']}")


# === Exercise 3: EBS Snapshot and Restore ===
# Problem: Create snapshot, list snapshots, and restore to a different AZ.

def exercise_3():
    """Demonstrate EBS snapshot creation, listing, and cross-AZ restore."""

    volume_id = "vol-0abc123"

    steps = [
        {
            "step": "Create snapshot with description",
            "command": (
                f"aws ec2 create-snapshot \\\n"
                f"    --volume-id {volume_id} \\\n"
                f"    --description \"Pre-OS-update backup\" \\\n"
                f"    --tag-specifications 'ResourceType=snapshot,"
                "Tags=[{{Key=Name,Value=pre-os-update-backup}}]'"
            ),
        },
        {
            "step": "List snapshots for this volume",
            "command": (
                f"aws ec2 describe-snapshots \\\n"
                f"    --owner-ids self \\\n"
                f"    --filters \"Name=volume-id,Values={volume_id}\" \\\n"
                f"    --query 'Snapshots[*].[SnapshotId,StartTime,State]' \\\n"
                f"    --output table"
            ),
        },
        {
            "step": "Restore to new volume in different AZ (ap-northeast-2b)",
            "command": (
                "aws ec2 create-volume \\\n"
                "    --snapshot-id snap-0xyz456 \\\n"
                "    --availability-zone ap-northeast-2b \\\n"
                "    --volume-type gp3"
            ),
        },
    ]

    print("EBS Snapshot and Restore Workflow:")
    print("=" * 60)
    for i, s in enumerate(steps, 1):
        print(f"\n  Step {i}: {s['step']}")
        print(f"    {s['command']}")

    print()
    print("Key insight: Snapshots are REGIONAL resources. You can restore a")
    print("snapshot to any AZ within the same region, enabling cross-AZ DR.")


# === Exercise 4: EFS vs EBS Architecture Decision ===
# Problem: Analyze EBS-per-instance vs shared EFS for web server file storage.

def exercise_4():
    """Compare EBS-per-instance vs shared EFS architecture."""

    comparison = {
        "Data consistency": {
            "EBS per instance": (
                "Files uploaded to one server are NOT visible on others. "
                "Users get different responses depending on which server "
                "handles their request -- a critical bug."
            ),
            "Shared EFS": (
                "All servers see the same filesystem. A file uploaded "
                "through Server 1 is immediately readable on Server 2."
            ),
        },
        "Scaling": {
            "EBS per instance": (
                "New instances start with empty volumes. Files must be "
                "manually synchronized."
            ),
            "Shared EFS": (
                "New instances mount EFS and have access to all existing "
                "files immediately. Scaling out is seamless."
            ),
        },
        "Cost": {
            "EBS per instance": (
                "EBS gp3: ~$0.10/GB/month. 100 GB x 10 servers = "
                "1 TB total = ~$100/month"
            ),
            "Shared EFS": (
                "EFS Standard: ~$0.30/GB/month for 100 GB shared = "
                "~$30/month (one copy for all servers)"
            ),
        },
    }

    servers = 10
    data_gb = 100
    ebs_cost = data_gb * servers * 0.10 / 1000 * 1000  # $100
    efs_cost = data_gb * 0.30  # $30

    print("EBS per Instance vs Shared EFS:")
    print("=" * 70)
    for aspect, options in comparison.items():
        print(f"\n  {aspect}:")
        print(f"    Option A (EBS): {options['EBS per instance']}")
        print(f"    Option B (EFS): {options['Shared EFS']}")

    print(f"\n  Cost Summary ({servers} servers, {data_gb} GB data):")
    print(f"    EBS: ${ebs_cost:.0f}/month (duplicated across {servers} servers)")
    print(f"    EFS: ${efs_cost:.0f}/month (single shared copy)")
    print()
    print("  Conclusion: EFS is the correct architecture for shared web server files.")
    print("  Best practice: For truly high-scale apps, upload directly to S3")
    print("  (pre-signed POST) and serve via CloudFront.")


# === Exercise 5: Storage Cost Comparison ===
# Problem: Calculate monthly costs for 10 TB with infrequent access.

def exercise_5():
    """Calculate and compare storage costs across different options."""

    data_tb = 10
    data_gb = data_tb * 1000
    monthly_reads = 2
    read_size_gb = 100

    options = [
        {
            "name": "EBS st1 (Throughput-optimized HDD)",
            "storage_per_gb": 0.045,
            "retrieval_per_gb": 0.0,
            "retrieval_note": "Attached storage, no per-access fee",
        },
        {
            "name": "S3 Standard-IA",
            "storage_per_gb": 0.0138,
            "retrieval_per_gb": 0.01,
            "retrieval_note": "Per-GB retrieval fee",
        },
        {
            "name": "S3 Glacier Flexible (Expedited ~15 min)",
            "storage_per_gb": 0.005,
            "retrieval_per_gb": 0.03,
            "retrieval_note": "Expedited retrieval (1-5 min, meets 15-min SLA)",
        },
    ]

    print(f"Storage Cost Comparison: {data_tb} TB, {monthly_reads} reads of "
          f"{read_size_gb} GB/month")
    print("Requirement: Available within 15 minutes")
    print("=" * 65)
    print()

    results = []
    for opt in options:
        storage_cost = data_gb * opt["storage_per_gb"]
        retrieval_cost = monthly_reads * read_size_gb * opt["retrieval_per_gb"]
        total = storage_cost + retrieval_cost
        results.append((opt["name"], storage_cost, retrieval_cost, total))

        print(f"  {opt['name']}:")
        print(f"    Storage: {data_gb:,} GB x ${opt['storage_per_gb']}/GB = ${storage_cost:,.2f}")
        print(f"    Retrieval: {monthly_reads} x {read_size_gb} GB x "
              f"${opt['retrieval_per_gb']}/GB = ${retrieval_cost:,.2f}")
        print(f"    Total: ${total:,.2f}/month")
        print()

    # Find winner
    winner = min(results, key=lambda x: x[3])
    ebs_cost = results[0][3]
    savings_pct = (1 - winner[3] / ebs_cost) * 100

    print(f"Winner: {winner[0]} at ${winner[3]:,.2f}/month")
    print(f"  {savings_pct:.0f}% cheaper than EBS while meeting the 15-min SLA.")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Storage Type Selection ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: EBS Volume Type Selection ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: EBS Snapshot and Restore ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: EFS vs EBS Architecture Decision ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: Storage Cost Comparison ===")
    print("=" * 70)
    exercise_5()

    print("\nAll exercises completed!")
