"""
Exercises for Lesson 05: Serverless Functions (Lambda / Cloud Functions)
Topic: Cloud_Computing

Solutions to practice problems from the lesson.
Simulates serverless concepts and cost calculations using Python.
"""


# === Exercise 1: Serverless vs VM Trade-off Analysis ===
# Problem: Determine serverless or VM for each use case.

def exercise_1():
    """Evaluate serverless vs VM suitability for different workloads."""

    # Serverless is best for short, event-driven, bursty workloads.
    # VMs are better for long-running, continuous, or high-customization needs.
    use_cases = [
        {
            "description": "REST API with ~500 requests/day, each < 200ms",
            "choice": "Serverless",
            "reason": (
                "500 requests/day is extremely low. A VM would sit idle 99.9% "
                "of the time. Lambda's free tier covers this at ~$0."
            ),
        },
        {
            "description": "Video transcoding processing 4K files for up to 30 minutes",
            "choice": "VM",
            "reason": (
                "Lambda max is 15 min; Cloud Functions (1st gen) is 9 min. "
                "A 30-min job cannot run on either platform."
            ),
        },
        {
            "description": "Event-driven pipeline processing queue messages on new orders",
            "choice": "Serverless",
            "reason": (
                "Queue processing is a textbook serverless use case. Native "
                "integrations with SQS/Pub-Sub. Scales to zero when queue is empty."
            ),
        },
        {
            "description": "ML training job running for 6 hours continuously",
            "choice": "VM",
            "reason": (
                "6-hour run exceeds all serverless time limits. Requires "
                "persistent state and dedicated GPU resources."
            ),
        },
    ]

    # Serverless constraints that drive the decision
    limits = {
        "Lambda max execution": "15 minutes",
        "Cloud Functions (1st gen)": "9 minutes",
        "Cloud Functions (2nd gen)": "60 minutes",
    }

    print("Serverless Execution Limits:")
    for service, limit in limits.items():
        print(f"  {service}: {limit}")
    print()

    for i, uc in enumerate(use_cases, 1):
        print(f"Use Case {i}: {uc['description']}")
        print(f"  Choice: {uc['choice']}")
        print(f"  Reason: {uc['reason']}")
        print()


# === Exercise 2: Cold Start Root Cause and Mitigation ===
# Problem: Diagnose and mitigate cold start latency in a Lambda auth function.

def exercise_2():
    """Analyze cold start causes and propose mitigation strategies."""

    # Cold starts occur when no warm container exists. The entire
    # initialization pipeline must run: container -> runtime -> imports -> init.
    cold_start_phases = [
        ("Spin up new container", "~500ms"),
        ("Initialize language runtime", "~200ms"),
        ("Import dependencies (boto3, JWT libs)", "~1-2s"),
        ("Run global initialization code", "~200ms"),
    ]

    print("Cold Start Mechanism:")
    print("  First-morning request path:")
    total_cold = 0
    for phase, duration in cold_start_phases:
        print(f"    -> {phase}: {duration}")
    print(f"  Total cold start: ~3-4 seconds")
    print()
    print("  Warm request path:")
    print(f"    -> Reuse initialized container -> Execute handler: ~50ms")
    print()

    # Mitigation strategies with trade-offs
    strategies = [
        {
            "name": "Provisioned Concurrency",
            "mechanism": "Pre-initialize N containers that stay permanently warm",
            "trade_off": (
                "Eliminates cold starts for up to N concurrent invocations, "
                "but you pay for provisioned capacity even with no requests."
            ),
            "command": (
                "aws lambda put-provisioned-concurrency-config \\\n"
                "    --function-name my-auth-function \\\n"
                "    --qualifier prod \\\n"
                "    --provisioned-concurrent-executions 5"
            ),
        },
        {
            "name": "Scheduled Warm-up (CloudWatch Events)",
            "mechanism": "Ping the function every 5 minutes to keep containers warm",
            "trade_off": (
                "Low cost (stays within free tier), but only keeps a small "
                "number of containers warm. Large spikes will still cold start."
            ),
            "command": (
                "# Create CloudWatch Events rule to ping every 5 minutes\n"
                "aws events put-rule --name warmup-auth \\\n"
                "    --schedule-expression 'rate(5 minutes)'"
            ),
        },
    ]

    print("Mitigation Strategies:")
    for s in strategies:
        print(f"\n  Strategy: {s['name']}")
        print(f"  Mechanism: {s['mechanism']}")
        print(f"  Trade-off: {s['trade_off']}")
        print(f"  Command:\n    {s['command']}")


# === Exercise 3: Lambda Cost Estimation ===
# Problem: Calculate monthly Lambda cost for given parameters.

def exercise_3():
    """Calculate AWS Lambda monthly cost step by step."""

    # Lambda charges for requests and compute (GB-seconds).
    # Free tier: 1M requests + 400K GB-seconds per month.
    memory_gb = 0.5     # 512 MB
    duration_s = 0.4    # 400ms
    invocations = 5_000_000
    price_per_million_requests = 0.20
    price_per_gb_second = 0.0000166667
    free_requests = 1_000_000
    free_gb_seconds = 400_000

    # Step 1: Request cost
    billable_requests = invocations - free_requests
    request_cost = (billable_requests / 1_000_000) * price_per_million_requests

    # Step 2: Compute cost (GB-seconds)
    gb_seconds_per_invocation = memory_gb * duration_s
    total_gb_seconds = invocations * gb_seconds_per_invocation
    billable_gb_seconds = total_gb_seconds - free_gb_seconds
    compute_cost = billable_gb_seconds * price_per_gb_second

    total_cost = request_cost + compute_cost

    print("Lambda Cost Estimation")
    print("=" * 50)
    print(f"  Memory: {int(memory_gb * 1024)} MB ({memory_gb} GB)")
    print(f"  Duration: {int(duration_s * 1000)}ms ({duration_s}s)")
    print(f"  Invocations: {invocations:,}/month")
    print(f"  Architecture: x86")
    print()

    print("Step 1: Request Cost")
    print(f"  Total invocations:    {invocations:>12,}")
    print(f"  Free tier:            {free_requests:>12,}")
    print(f"  Billable invocations: {billable_requests:>12,}")
    print(f"  Cost: {billable_requests:,} / 1,000,000 x ${price_per_million_requests} = ${request_cost:.2f}")
    print()

    print("Step 2: Compute Cost (GB-seconds)")
    print(f"  GB-seconds/invocation: {memory_gb} GB x {duration_s}s = {gb_seconds_per_invocation}")
    print(f"  Total GB-seconds:      {invocations:,} x {gb_seconds_per_invocation} = {total_gb_seconds:,.0f}")
    print(f"  Free tier:             {free_gb_seconds:,}")
    print(f"  Billable GB-seconds:   {billable_gb_seconds:,.0f}")
    print(f"  Cost: {billable_gb_seconds:,.0f} x ${price_per_gb_second} = ${compute_cost:.2f}")
    print()

    print(f"Total Monthly Cost: ${request_cost:.2f} + ${compute_cost:.2f} = ${total_cost:.2f}")
    print()

    # ARM optimization
    arm_discount = 0.20
    arm_compute_cost = compute_cost * (1 - arm_discount)
    arm_total = request_cost + arm_compute_cost
    print(f"Optimization: ARM (Graviton2) saves 20% on compute:")
    print(f"  ARM compute cost: ${arm_compute_cost:.2f}")
    print(f"  ARM total: ${arm_total:.2f}/month (saves ${total_cost - arm_total:.2f}/month)")


# === Exercise 4: Event Source Configuration ===
# Problem: Configure S3-to-Lambda trigger and prevent infinite loop.

def exercise_4():
    """Design S3 event trigger with infinite loop prevention."""

    # The infinite loop problem is a critical gotcha: if the Lambda writes
    # thumbnails to the same bucket, the write triggers another invocation.
    print("S3 -> Lambda Image Thumbnail Pipeline")
    print("=" * 50)
    print()

    print("Event source: S3 Event Notification")
    print("  Event type: s3:ObjectCreated:*")
    print("  Trigger: Every new object upload invokes Lambda")
    print()

    print("DANGER: Infinite Loop Problem")
    print("  If Lambda writes thumbnail to the SAME bucket:")
    print("    Upload image -> Lambda runs -> writes thumbnail -> triggers Lambda")
    print("    -> writes thumbnail -> triggers Lambda -> ... (infinite!)")
    print()

    solutions = [
        {
            "name": "Separate output bucket",
            "description": "Write thumbnails to a different S3 bucket.",
            "prevention": "Only configure trigger on the source bucket.",
        },
        {
            "name": "Prefix/suffix filtering + code guard",
            "description": (
                "Configure S3 trigger only for 'uploads/' prefix. "
                "Write thumbnails to 'thumbnails/' prefix."
            ),
            "prevention": (
                "Both S3-level filtering and code-level guard:\n"
                "    if key.startswith('thumbnails/'):\n"
                "        return  # Skip -- defense in depth"
            ),
        },
    ]

    print("Solutions:")
    for i, sol in enumerate(solutions, 1):
        print(f"\n  Solution {i}: {sol['name']}")
        print(f"  Description: {sol['description']}")
        print(f"  Prevention: {sol['prevention']}")


# === Exercise 5: Serverless Architecture Design ===
# Problem: Design serverless e-commerce order processing system.

def exercise_5():
    """Design a serverless architecture for order processing."""

    # Fan-out pattern: one event triggers multiple downstream processors.
    # SNS acts as the hub; each subscriber processes independently.
    architecture = {
        "entry_point": {
            "service": "API Gateway",
            "role": "HTTP endpoint for order requests",
        },
        "order_handler": {
            "service": "Lambda (Order Handler)",
            "actions": ["Save order to DynamoDB", "Publish to SNS 'order-created'"],
        },
        "fan_out_hub": {
            "service": "SNS Topic: order-created",
            "role": "Delivers event to multiple downstream Lambdas simultaneously",
        },
        "downstream": [
            {
                "service": "Lambda (Inventory)",
                "action": "Check and decrement inventory in DynamoDB",
            },
            {
                "service": "Lambda (Email)",
                "action": "Send confirmation email via SES",
            },
            {
                "service": "Lambda (Analytics)",
                "action": "Record event to Kinesis -> S3 data lake",
            },
        ],
    }

    print("Serverless E-Commerce Order Processing Architecture:")
    print("=" * 60)
    print()
    print(f"Customer -> [{architecture['entry_point']['service']}]")
    print(f"              -> [{architecture['order_handler']['service']}]")
    for action in architecture["order_handler"]["actions"]:
        print(f"                   |-- {action}")
    print(f"              -> [{architecture['fan_out_hub']['service']}]")
    print(f"                   |")
    for ds in architecture["downstream"]:
        print(f"                   +-> [{ds['service']}] -> {ds['action']}")

    print()
    print("Why serverless is appropriate:")
    reasons = [
        "Each step is event-driven and short-lived (< 15 minutes).",
        "Traffic is bursty; serverless scales to zero and back automatically.",
        "Loose coupling via SNS: each function can fail independently.",
        "Cost is proportional to actual order volume -- no idle capacity.",
    ]
    for r in reasons:
        print(f"  - {r}")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Serverless vs VM Trade-off ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: Cold Start Root Cause and Mitigation ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Lambda Cost Estimation ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: Event Source Configuration ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: Serverless Architecture Design ===")
    print("=" * 70)
    exercise_5()

    print("\nAll exercises completed!")
