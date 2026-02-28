"""
Exercises for Lesson 17: Monitoring, Logging, and Cost Management
Topic: Cloud_Computing

Solutions to practice problems from the lesson.
Simulates monitoring service selection, alarm design, log queries, budget alerts,
and cost optimization analysis.
"""


# === Exercise 1: Monitoring Service Mapping ===
def exercise_1():
    """Map observability requirements to AWS and GCP services."""

    print("Monitoring Service Mapping:")
    print("=" * 70)
    print()

    requirements = [
        {
            "requirement": "Track Lambda function execution duration",
            "aws": "CloudWatch Metrics (Lambda/Duration)",
            "gcp": "Cloud Monitoring (cloudfunctions.googleapis.com/function/execution_times)",
        },
        {
            "requirement": "Store and search application log output",
            "aws": "CloudWatch Logs",
            "gcp": "Cloud Logging",
        },
        {
            "requirement": "Automated notification when disk > 90%",
            "aws": "CloudWatch Alarm + SNS",
            "gcp": "Cloud Monitoring Alerting Policy + Notification Channel",
        },
        {
            "requirement": "Trace request through microservices to find bottleneck",
            "aws": "AWS X-Ray",
            "gcp": "Cloud Trace",
        },
        {
            "requirement": "Query last 24h of error logs with SQL-like syntax",
            "aws": "CloudWatch Logs Insights",
            "gcp": "Cloud Logging (Log Analytics / BigQuery export)",
        },
    ]

    print(f"  {'Requirement':<50} {'AWS':<30} {'GCP'}")
    print("  " + "-" * 100)
    for r in requirements:
        print(f"  {r['requirement']:<50} {r['aws']:<30} {r['gcp']}")
    print()

    print("  Note: For GCP log querying, Cloud Logging uses its own filter syntax")
    print("  (not SQL). Export to BigQuery for SQL-based analysis.")
    print("  CloudWatch Logs Insights has its own query language resembling SQL.")


# === Exercise 2: CloudWatch Alarm Design ===
def exercise_2():
    """Design CloudWatch alarms for three SLO requirements."""

    sns_topic = "arn:aws:sns:ap-northeast-2:123456789012:prod-alerts"

    print("CloudWatch Alarm Design for SLOs:")
    print("=" * 70)
    print()
    print("  SLO Requirements:")
    print("    - Availability: Health checks respond within 5 seconds")
    print("    - Latency: p95 response time < 2 seconds")
    print("    - Error rate: HTTP 5xx < 1% of requests")
    print()

    alarms = [
        {
            "name": "prod-unhealthy-hosts",
            "slo": "Availability",
            "command": (
                "aws cloudwatch put-metric-alarm \\\n"
                '    --alarm-name "prod-unhealthy-hosts" \\\n'
                '    --alarm-description "ALB has unhealthy targets" \\\n'
                "    --metric-name UnHealthyHostCount \\\n"
                "    --namespace AWS/ApplicationELB \\\n"
                "    --dimensions Name=LoadBalancer,"
                "Value=app/my-alb/1234567890abcdef \\\n"
                "    --statistic Average \\\n"
                "    --period 60 \\\n"
                "    --threshold 1 \\\n"
                "    --comparison-operator GreaterThanOrEqualToThreshold \\\n"
                "    --evaluation-periods 2 \\\n"
                f"    --alarm-actions {sns_topic}"
            ),
            "explanation": (
                "Triggers when any ALB target is unhealthy for 2 consecutive "
                "1-minute periods. evaluation-periods=2 prevents flapping."
            ),
        },
        {
            "name": "prod-high-latency-p95",
            "slo": "Latency (p95 < 2s)",
            # Why extended-statistic p95: Standard statistics (Average, Sum)
            # miss tail latency. p95 catches the worst 5% of requests,
            # which are often what users complain about.
            "command": (
                "aws cloudwatch put-metric-alarm \\\n"
                '    --alarm-name "prod-high-latency-p95" \\\n'
                '    --alarm-description "p95 latency exceeds 2 seconds" \\\n'
                "    --metric-name TargetResponseTime \\\n"
                "    --namespace AWS/ApplicationELB \\\n"
                "    --dimensions Name=LoadBalancer,"
                "Value=app/my-alb/1234567890abcdef \\\n"
                "    --extended-statistic p95 \\\n"
                "    --period 300 \\\n"
                "    --threshold 2 \\\n"
                "    --comparison-operator GreaterThanThreshold \\\n"
                "    --evaluation-periods 3 \\\n"
                f"    --alarm-actions {sns_topic}"
            ),
            "explanation": (
                "Uses extended-statistic p95 (not Average) to catch tail "
                "latency. 3 evaluation periods of 5 min each = 15 min "
                "sustained before alerting."
            ),
        },
        {
            "name": "prod-high-5xx-rate",
            "slo": "Error rate (5xx < 1%)",
            # Why metric math: Using raw 5xx count would fire during low
            # traffic (3 errors out of 10 requests = 30%). Metric math
            # computes the percentage ratio for accurate SLO measurement.
            "command": (
                "aws cloudwatch put-metric-alarm \\\n"
                '    --alarm-name "prod-high-5xx-rate" \\\n'
                '    --alarm-description "5xx error rate exceeds 1%" \\\n'
                "    --metrics '[\n"
                '        {"Id":"e1","Expression":"m2/m1*100","Label":"5xx Rate %"},\n'
                '        {"Id":"m1","MetricStat":{"Metric":{"Namespace":'
                '"AWS/ApplicationELB","MetricName":"RequestCount",...},'
                '"Period":300,"Stat":"Sum"},"ReturnData":false},\n'
                '        {"Id":"m2","MetricStat":{"Metric":{"Namespace":'
                '"AWS/ApplicationELB","MetricName":"HTTPCode_Target_5XX_Count",...},'
                '"Period":300,"Stat":"Sum"},"ReturnData":false}\n'
                "    ]' \\\n"
                "    --comparison-operator GreaterThanThreshold \\\n"
                "    --threshold 1 \\\n"
                "    --evaluation-periods 2 \\\n"
                f"    --alarm-actions {sns_topic}"
            ),
            "explanation": (
                "Uses metric math (m2/m1*100) to compute 5xx percentage. "
                "Avoids false alarms from raw counts during low-traffic periods."
            ),
        },
    ]

    for alarm in alarms:
        print(f"  Alarm: {alarm['name']} (SLO: {alarm['slo']})")
        print(f"    {alarm['command']}")
        print(f"    Explanation: {alarm['explanation']}")
        print()


# === Exercise 3: Log Query with CloudWatch Logs Insights ===
def exercise_3():
    """Write a Logs Insights query to find payment errors."""

    print("CloudWatch Logs Insights Query:")
    print("=" * 70)
    print()
    print("  Context: Customer reported error at ~14:30 UTC on 2024-03-15.")
    print("  Goal: Find 20 most recent ERROR entries containing 'payment'.")
    print()

    # Query command
    print("  Command:")
    print("    aws logs start-query \\")
    print("        --log-group-name /myapp/production \\")
    print('        --start-time $(date -d "2024-03-15T14:00:00Z" +%s) \\')
    print('        --end-time $(date -d "2024-03-15T15:00:00Z" +%s) \\')
    print("        --query-string '")
    print("            fields @timestamp, requestId, message")
    print('            | filter level = "ERROR" and message like /payment/')
    print("            | sort @timestamp desc")
    print("            | limit 20")
    print("        '")
    print()
    print("    # Note the queryId from output, then retrieve results:")
    print("    aws logs get-query-results --query-id <QUERY_ID>")
    print()

    # Syntax breakdown
    print("  Logs Insights Query Syntax:")
    clauses = [
        ("fields", "Select which fields to display"),
        ("filter", "Filter rows (like SQL WHERE); supports and, or, not, like /regex/"),
        ("sort", "Order results; @timestamp is a built-in field"),
        ("limit", "Cap the number of results returned"),
    ]
    for clause, explanation in clauses:
        print(f"    {clause}: {explanation}")
    print()

    # Common patterns
    print("  Common Logs Insights Patterns:")
    print()
    print("    # Count errors by type:")
    print("    fields errorType")
    print('    | filter level = "ERROR"')
    print("    | stats count(*) as errorCount by errorType")
    print("    | sort errorCount desc")
    print()
    print("    # 95th percentile latency by endpoint:")
    print("    fields endpoint, duration")
    print("    | stats pct(duration, 95) as p95 by endpoint")
    print("    | sort p95 desc")


# === Exercise 4: AWS Budget Alert ===
def exercise_4():
    """Configure multi-threshold budget alerts including forecast."""

    budget_amount = 500
    thresholds = [
        {"type": "ACTUAL", "pct": 60},
        {"type": "ACTUAL", "pct": 80},
        {"type": "ACTUAL", "pct": 100},
        {"type": "FORECASTED", "pct": 110},
    ]

    print(f"AWS Budget Alert Configuration (${budget_amount}/month):")
    print("=" * 70)
    print()
    print(f"  Budget: ${budget_amount}/month")
    print("  Alert thresholds:")
    for t in thresholds:
        dollar_amount = budget_amount * t["pct"] / 100
        label = "actual spend" if t["type"] == "ACTUAL" else "forecasted spend"
        print(f"    - {t['pct']}% ({label}): ${dollar_amount:.0f}")
    print()

    # CLI command
    print("  CLI Command:")
    print("    aws budgets create-budget \\")
    print("        --account-id 123456789012 \\")
    print("        --budget '{")
    print(f'            "BudgetName": "Monthly-{budget_amount}USD",')
    print("            \"BudgetLimit\": {")
    print(f'                "Amount": "{budget_amount}",')
    print('                "Unit": "USD"')
    print("            },")
    print('            "TimeUnit": "MONTHLY",')
    print('            "BudgetType": "COST"')
    print("        }' \\")
    print("        --notifications-with-subscribers '[")

    for i, t in enumerate(thresholds):
        print("            {")
        print("                \"Notification\": {")
        print(f'                    "NotificationType": "{t["type"]}",')
        print('                    "ComparisonOperator": "GREATER_THAN",')
        print(f'                    "Threshold": {t["pct"]},')
        print('                    "ThresholdType": "PERCENTAGE"')
        print("                },")
        print("                \"Subscribers\": [")
        print('                    {"SubscriptionType": "EMAIL", '
              '"Address": "team@example.com"}')
        print("                ]")
        comma = "," if i < len(thresholds) - 1 else ""
        print(f"            }}{comma}")
    print("        ]'")
    print()

    # Key points
    # Why FORECASTED: Gives early warning before you actually overspend.
    # AWS predicts month-end spend based on current trajectory.
    print("  Key Points:")
    points = [
        "'ACTUAL' triggers when real charges exceed the threshold.",
        "'FORECASTED' triggers when AWS predicts month-end spend will exceed "
        "the threshold -- gives EARLY WARNING before overspending.",
        "Up to 5 notifications per budget, each with up to 10 subscribers.",
        "Budgets can also trigger SNS topics for automated remediation "
        "(e.g., shutting down dev instances).",
    ]
    for p in points:
        print(f"    - {p}")


# === Exercise 5: Cost Optimization Analysis ===
def exercise_5():
    """Investigate an unexpectedly high AWS bill."""

    expected = {"EC2": 200, "S3": 50, "Data Transfer": 30}
    actual = {"EC2": 800, "S3": 150, "Data Transfer": 200}
    total_expected = sum(expected.values())
    total_actual = sum(actual.values())

    print("Cost Optimization Analysis:")
    print("=" * 70)
    print()
    print(f"  Last month's bill: ${total_actual:,} (expected: ${total_expected:,})")
    print()

    print(f"  {'Service':<18} {'Expected':<12} {'Actual':<12} {'Over by'}")
    print("  " + "-" * 50)
    for service in expected:
        over = actual[service] / expected[service]
        print(f"  {service:<18} ${expected[service]:<11} ${actual[service]:<11} {over:.1f}x")
    print()

    # Investigation per service
    investigations = [
        {
            "service": "EC2 ($800 vs $200 -- 4x over)",
            "causes": [
                "Forgotten running instances (especially large instance types)",
                "Instances not using Reserved Instances or Savings Plans",
                "Wrong instance type (someone launched m5.4xlarge by mistake)",
            ],
            "commands": [
                ("List all running instances with type and launch time",
                 "aws ec2 describe-instances \\\n"
                 '    --filters "Name=instance-state-name,Values=running" \\\n'
                 "    --query 'Reservations[*].Instances[*]."
                 "[InstanceId,InstanceType,LaunchTime]' \\\n"
                 "    --output table"),
                ("Get rightsizing recommendations",
                 "aws compute-optimizer get-ec2-instance-recommendations \\\n"
                 "    --query 'instanceRecommendations[?finding==`OVER_PROVISIONED`]'"),
            ],
        },
        {
            "service": "S3 ($150 vs $50 -- 3x over)",
            "causes": [
                "Objects in STANDARD that should be in STANDARD_IA or GLACIER",
                "Bucket versioning with many old versions accumulating",
                "Expired multipart uploads not cleaned up",
            ],
            "commands": [
                ("Check lifecycle policies",
                 "aws s3api get-bucket-lifecycle-configuration --bucket my-bucket"),
            ],
        },
        {
            "service": "Data Transfer ($200 vs $30 -- 6.7x over)",
            # Why data transfer is often overlooked: It's charged per GB
            # egressing AWS. Cross-AZ traffic, NAT Gateway processing,
            # and internet egress all add up silently.
            "causes": [
                "Internet egress (~$0.09/GB -- most expensive transfer type)",
                "Cross-AZ traffic (EC2 <-> RDS in different AZs)",
                "NAT Gateway traffic (~$0.045/GB processed)",
            ],
            "commands": [
                ("Review cost by usage type",
                 "aws ce get-cost-and-usage \\\n"
                 "    --time-period Start=2024-01-01,End=2024-01-31 \\\n"
                 "    --granularity MONTHLY \\\n"
                 "    --metrics BlendedCost \\\n"
                 "    --group-by Type=DIMENSION,Key=USAGE_TYPE"),
            ],
        },
    ]

    for inv in investigations:
        print(f"  {inv['service']}:")
        print("    Likely causes:")
        for c in inv["causes"]:
            print(f"      - {c}")
        print("    Investigation commands:")
        for desc, cmd in inv["commands"]:
            print(f"      # {desc}")
            print(f"      {cmd}")
            print()

    # General cost hygiene
    print("  General Cost Hygiene Checklist:")
    checklist = [
        "Enable AWS Cost Anomaly Detection (automatic spending spike alerts).",
        "Tag all resources with Project, Environment, Owner for cost attribution.",
        "Set up budget alerts BEFORE launching production workloads.",
        "Review Cost Explorer weekly -- don't wait for the monthly bill.",
    ]
    for item in checklist:
        print(f"    - {item}")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Monitoring Service Mapping ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: CloudWatch Alarm Design ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Log Query with CloudWatch Logs Insights ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: AWS Budget Alert ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: Cost Optimization Analysis ===")
    print("=" * 70)
    exercise_5()

    print("\nAll exercises completed!")
