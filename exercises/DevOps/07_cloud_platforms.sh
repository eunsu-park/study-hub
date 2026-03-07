#!/bin/bash
# Exercises for Lesson 07: Cloud Platforms
# Topic: DevOps
# Solutions to practice problems from the lesson.

# === Exercise 1: Cloud Service Comparison ===
# Problem: Map common infrastructure needs to the equivalent services
# across AWS, GCP, and Azure.
exercise_1() {
    echo "=== Exercise 1: Cloud Service Comparison ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
service_map = {
    "Compute (VMs)":            {"AWS": "EC2",              "GCP": "Compute Engine",      "Azure": "Virtual Machines"},
    "Containers (managed K8s)": {"AWS": "EKS",              "GCP": "GKE",                 "Azure": "AKS"},
    "Serverless functions":     {"AWS": "Lambda",           "GCP": "Cloud Functions",     "Azure": "Azure Functions"},
    "Serverless containers":    {"AWS": "Fargate",          "GCP": "Cloud Run",           "Azure": "Container Apps"},
    "Object storage":           {"AWS": "S3",               "GCP": "Cloud Storage",       "Azure": "Blob Storage"},
    "Relational DB (managed)":  {"AWS": "RDS",              "GCP": "Cloud SQL",           "Azure": "Azure SQL"},
    "NoSQL (document)":         {"AWS": "DynamoDB",         "GCP": "Firestore",           "Azure": "Cosmos DB"},
    "Message queue":            {"AWS": "SQS",              "GCP": "Pub/Sub",             "Azure": "Service Bus"},
    "CDN":                      {"AWS": "CloudFront",       "GCP": "Cloud CDN",           "Azure": "Azure CDN"},
    "DNS":                      {"AWS": "Route 53",         "GCP": "Cloud DNS",           "Azure": "Azure DNS"},
    "IAM":                      {"AWS": "IAM",              "GCP": "IAM",                 "Azure": "Entra ID"},
    "Monitoring":               {"AWS": "CloudWatch",       "GCP": "Cloud Monitoring",    "Azure": "Azure Monitor"},
    "IaC":                      {"AWS": "CloudFormation",   "GCP": "Deployment Manager",  "Azure": "ARM/Bicep"},
    "Secret management":        {"AWS": "Secrets Manager",  "GCP": "Secret Manager",      "Azure": "Key Vault"},
}

print(f"{'Need':<28} {'AWS':<20} {'GCP':<20} {'Azure':<20}")
print("-" * 88)
for need, services in service_map.items():
    print(f"{need:<28} {services['AWS']:<20} {services['GCP']:<20} {services['Azure']:<20}")
SOLUTION
}

# === Exercise 2: Well-Architected Framework ===
# Problem: For a given architecture, evaluate it against the five pillars
# of the AWS Well-Architected Framework and suggest improvements.
exercise_2() {
    echo "=== Exercise 2: Well-Architected Framework ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
pillars = {
    "Operational Excellence": {
        "principles": [
            "Perform operations as code (IaC, GitOps)",
            "Make frequent, small, reversible changes",
            "Refine operations procedures frequently",
            "Anticipate failure (runbooks, game days)",
            "Learn from all operational events",
        ],
        "checklist": [
            "All infrastructure defined in Terraform/CloudFormation",
            "CI/CD pipeline deploys to all environments",
            "Runbooks exist for common failure scenarios",
            "Monitoring dashboards cover all critical paths",
        ],
    },
    "Security": {
        "principles": [
            "Implement a strong identity foundation (least privilege)",
            "Enable traceability (CloudTrail, audit logs)",
            "Apply security at all layers (network, host, app)",
            "Automate security best practices",
            "Protect data in transit and at rest",
        ],
        "checklist": [
            "MFA enabled for all human users",
            "Service accounts use IAM roles, not access keys",
            "All data encrypted at rest (KMS)",
            "Network segmented with private subnets",
        ],
    },
    "Reliability": {
        "principles": [
            "Automatically recover from failure",
            "Test recovery procedures",
            "Scale horizontally to increase aggregate availability",
            "Stop guessing capacity",
            "Manage change in automation",
        ],
        "checklist": [
            "Multi-AZ deployment for databases and compute",
            "Auto-scaling configured with health checks",
            "Backups tested with regular restore drills",
            "Circuit breakers for external dependencies",
        ],
    },
    "Performance Efficiency": {
        "principles": [
            "Democratize advanced technologies (managed services)",
            "Go global in minutes (CDN, multi-region)",
            "Use serverless architectures where appropriate",
            "Experiment more often (A/B test infrastructure)",
        ],
        "checklist": [
            "Right-sized instances based on actual metrics",
            "CDN for static assets",
            "Database read replicas for read-heavy workloads",
            "Caching layer (ElastiCache/Redis) for hot data",
        ],
    },
    "Cost Optimization": {
        "principles": [
            "Implement cloud financial management",
            "Adopt a consumption model (pay for what you use)",
            "Measure overall efficiency",
            "Stop spending on undifferentiated heavy lifting",
        ],
        "checklist": [
            "Reserved instances for steady-state workloads",
            "Spot instances for batch/fault-tolerant workloads",
            "Auto-scaling to zero for dev/staging at night",
            "Cost allocation tags on all resources",
        ],
    },
}

for pillar, details in pillars.items():
    print(f"\n{pillar}")
    print(f"  Key principles:")
    for p in details["principles"][:3]:
        print(f"    - {p}")
    print(f"  Checklist:")
    for c in details["checklist"][:3]:
        print(f"    [x] {c}")
SOLUTION
}

# === Exercise 3: Cost Optimization ===
# Problem: Given a monthly AWS bill, identify the top cost-saving
# opportunities and estimate savings.
exercise_3() {
    echo "=== Exercise 3: Cost Optimization ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
monthly_costs = [
    {"service": "EC2 On-Demand",   "cost": 8500, "instances": 20, "avg_cpu": 15},
    {"service": "RDS Multi-AZ",    "cost": 3200, "class": "db.r6g.xlarge"},
    {"service": "S3 Storage",      "cost": 1800, "tb": 60, "access_pattern": "rare"},
    {"service": "NAT Gateway",     "cost": 1200, "data_gb": 400},
    {"service": "EBS Volumes",     "cost": 900,  "unused_gb": 500},
    {"service": "CloudWatch Logs", "cost": 600,  "retention_days": "forever"},
]

savings = [
    {
        "action": "Convert 70% EC2 to Reserved Instances (1yr, no upfront)",
        "current": 8500,
        "projected": 8500 * 0.3 + 8500 * 0.7 * 0.6,  # 30% on-demand + 70% at 40% discount
        "rationale": "Steady-state workloads running 24/7 with 15% avg CPU",
    },
    {
        "action": "Right-size RDS to db.r6g.large (half the current)",
        "current": 3200,
        "projected": 1600,
        "rationale": "CPU/memory metrics show <30% utilization",
    },
    {
        "action": "Move 50TB to S3 Glacier Deep Archive",
        "current": 1800,
        "projected": 1800 - (50 * 23) + (50 * 1),  # $23/TB/mo -> $1/TB/mo
        "rationale": "Rarely accessed data (access pattern: rare)",
    },
    {
        "action": "Use VPC endpoints instead of NAT Gateway for S3/DynamoDB",
        "current": 1200,
        "projected": 400,
        "rationale": "Gateway endpoints are free; reduces NAT data processing",
    },
    {
        "action": "Delete 500GB unused EBS volumes",
        "current": 900,
        "projected": 900 - (500 * 0.08),
        "rationale": "Unattached volumes still incur storage charges",
    },
    {
        "action": "Set CloudWatch Logs retention to 90 days",
        "current": 600,
        "projected": 200,
        "rationale": "Export to S3 for long-term; reduce live storage",
    },
]

total_current = sum(s["current"] for s in savings)
total_projected = sum(s["projected"] for s in savings)
total_saved = total_current - total_projected

print(f"{'Action':<55} {'Current':>8} {'After':>8} {'Saved':>8}")
print("-" * 85)
for s in savings:
    saved = s["current"] - s["projected"]
    print(f"{s['action']:<55} ${s['current']:>6,.0f} ${s['projected']:>6,.0f} ${saved:>6,.0f}")
print("-" * 85)
print(f"{'Total':<55} ${total_current:>6,.0f} ${total_projected:>6,.0f} ${total_saved:>6,.0f}")
print(f"\nMonthly savings: ${total_saved:,.0f} ({total_saved/total_current:.0%} reduction)")
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 07: Cloud Platforms"
echo "=================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
