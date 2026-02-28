"""
Exercises for Lesson 11: Managed Relational Databases
Topic: Cloud_Computing

Solutions to practice problems from the lesson.
Simulates database architecture decisions and security configurations.
"""


# === Exercise 1: Managed DB vs Self-Hosted Decision ===
def exercise_1():
    """Compare RDS managed vs EC2 self-hosted PostgreSQL."""

    automated_tasks = [
        {
            "task": "Automated backups and PITR",
            "rds": "Daily snapshots + transaction logs to S3. PITR to any second within retention.",
            "self_managed": "Write/maintain backup scripts, manage storage, test restores, monitor failures.",
        },
        {
            "task": "Multi-AZ failover",
            "rds": "Synchronous standby in different AZ. Automatic promotion in ~1-2 min.",
            "self_managed": "Manual streaming replication + repmgr/Pacemaker/Corosync failover scripts.",
        },
        {
            "task": "Minor version patching",
            "rds": "Auto-applied during maintenance windows.",
            "self_managed": "Monitor releases, test compatibility, schedule downtime, apply manually.",
        },
    ]

    print("RDS Managed vs Self-Hosted PostgreSQL:")
    print("=" * 70)
    for t in automated_tasks:
        print(f"\n  Task: {t['task']}")
        print(f"    RDS:          {t['rds']}")
        print(f"    Self-managed: {t['self_managed']}")

    print("\n  When self-managed EC2 is better:")
    print("    - Need OS-level access (custom kernel params, huge pages)")
    print("    - Require PostgreSQL extensions not available on RDS")
    print("    - Need complete control over exact version and configuration")


# === Exercise 2: RDS Multi-AZ vs Read Replica ===
def exercise_2():
    """Distinguish Multi-AZ (HA) from Read Replicas (read scaling)."""

    features = {
        "Multi-AZ": {
            "purpose": "High availability and automatic failover",
            "replication": "Synchronous",
            "serves_reads": False,
            "failover": "Automatic (~1-2 min)",
        },
        "Read Replica": {
            "purpose": "Read scaling and offloading read traffic",
            "replication": "Asynchronous",
            "serves_reads": True,
            "failover": "Manual promotion",
        },
    }

    print("Multi-AZ vs Read Replica:")
    print(f"  {'Feature':<18} {'Multi-AZ':<35} {'Read Replica'}")
    print("  " + "-" * 70)
    for feat in ["purpose", "replication", "serves_reads", "failover"]:
        print(f"  {feat:<18} {str(features['Multi-AZ'][feat]):<35} "
              f"{str(features['Read Replica'][feat])}")
    print()

    scenarios = [
        ("Heavy reporting queries slowing production",
         "Read Replica", "Offload reports to replica, restoring primary performance."),
        ("Primary AZ hardware failure, DB unreachable",
         "Multi-AZ", "Auto-promotes standby in ~1-2 min. DNS endpoint updated."),
    ]
    for desc, answer, reason in scenarios:
        print(f"  Scenario: {desc}")
        print(f"    Answer: {answer} -- {reason}")
        print()


# === Exercise 3: Database Security Configuration ===
def exercise_3():
    """List and explain key RDS security configurations."""

    configs = [
        {
            "config": "Private subnet, publicly-accessible=false",
            "why": "No public IP. Only VPC resources can connect.",
        },
        {
            "config": "Dedicated security group (port 5432 from app-SG only)",
            "why": "Only application servers can reach the database.",
        },
        {
            "config": "Encryption at rest (--storage-encrypted)",
            "why": "All data, backups, snapshots encrypted with AES-256. Required for PCI/HIPAA.",
        },
        {
            "config": "Automated backups (retention >= 7 days)",
            "why": "Enables PITR. Essential for recovering from accidental data corruption.",
        },
        {
            "config": "Master password in Secrets Manager with auto-rotation",
            "why": "Never hardcode credentials. IAM roles grant application access.",
        },
        {
            "config": "Deletion protection enabled",
            "why": "Prevents accidental deletion of the database instance.",
        },
    ]

    print("RDS Production Security Checklist:")
    print("=" * 60)
    for i, c in enumerate(configs, 1):
        print(f"  {i}. {c['config']}")
        print(f"     Why: {c['why']}")


# === Exercise 4: Aurora vs RDS PostgreSQL ===
def exercise_4():
    """Evaluate Aurora PostgreSQL migration for a growing SaaS company."""

    aurora_advantages = [
        ("Read scaling", "Up to 15 low-latency replicas (vs 5 for RDS). "
         "Shared storage means <100ms replica lag."),
        ("Cross-region DR", "Aurora Global Database: sub-second replication. "
         "Failover in <1 min."),
        ("Storage scalability", "Auto-grows in 10 GB increments up to 128 TB."),
        ("Performance", "Up to 3x throughput of standard PostgreSQL."),
    ]

    considerations = [
        "20-40% more expensive per vCPU-hour than equivalent RDS.",
        "Not all PostgreSQL extensions available on Aurora.",
        "Migration requires testing period for compatibility.",
    ]

    print("Aurora vs RDS PostgreSQL Evaluation:")
    print("=" * 60)
    print("  Context: Growing SaaS, 2 read replicas, needs cross-region DR")
    print()
    print("  Aurora Advantages:")
    for name, detail in aurora_advantages:
        print(f"    + {name}: {detail}")
    print()
    print("  Considerations:")
    for c in considerations:
        print(f"    ! {c}")
    print()
    print("  Recommendation: Migrate to Aurora PostgreSQL.")
    print("    The need for multiple read replicas AND cross-region DR makes")
    print("    Aurora the better fit despite higher per-instance cost.")


# === Exercise 5: Point-in-Time Recovery Scenario ===
def exercise_5():
    """Demonstrate PITR for recovering from accidental data corruption."""

    source_db = "my-production-db"
    target_db = "my-production-db-restored"
    restore_time = "2026-02-24T14:30:00Z"
    error_time = "14:32 UTC"

    print("Point-in-Time Recovery:")
    print("=" * 60)
    print(f"  Error at: {error_time} (erroneous DELETE statement)")
    print(f"  Restore to: {restore_time} (2 min before error)")
    print()

    print("  Command:")
    print(f"    aws rds restore-db-instance-to-point-in-time \\")
    print(f"        --source-db-instance-identifier {source_db} \\")
    print(f"        --target-db-instance-identifier {target_db} \\")
    print(f"        --restore-time {restore_time} \\")
    print(f"        --db-instance-class db.r5.2xlarge \\")
    print(f"        --multi-az")
    print()

    print("  What happens after restore:")
    steps = [
        f"AWS creates a NEW instance ({target_db}) -- original is unchanged.",
        f"Restore applies transaction logs up to {restore_time} exactly.",
        "New instance gets a new endpoint (DNS hostname).",
        "Use restored instance to export/import only affected rows.",
        "Delete restored instance after recovery to avoid double cost.",
    ]
    for i, step in enumerate(steps, 1):
        print(f"    {i}. {step}")

    print()
    print("  Prerequisite: backup-retention-period >= 1 day, automated backups enabled.")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Managed DB vs Self-Hosted ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: Multi-AZ vs Read Replica ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Database Security Configuration ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: Aurora vs RDS PostgreSQL ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: Point-in-Time Recovery ===")
    print("=" * 70)
    exercise_5()

    print("\nAll exercises completed!")
