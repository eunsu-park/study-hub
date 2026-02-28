"""
Exercises for Lesson 10: Load Balancing and CDN
Topic: Cloud_Computing

Solutions to practice problems from the lesson.
Simulates load balancer selection, health checks, CDN caching, and routing rules.
"""


# === Exercise 1: ALB vs NLB Selection ===
# Problem: Choose between ALB and NLB for each scenario.

def exercise_1():
    """Select the appropriate load balancer type for various scenarios."""

    scenarios = [
        {
            "description": "REST API with path-based routing: /api/users/* and /api/orders/*",
            "choice": "ALB",
            "reason": "Path-based routing is ALB-exclusive. ALB inspects HTTP paths "
                      "and routes to different target groups. NLB operates at L4.",
        },
        {
            "description": "Real-time gaming server using UDP, needs static IP",
            "choice": "NLB",
            "reason": "NLB supports UDP (ALB is HTTP/HTTPS only) and provides "
                      "static IPs per AZ for firewall whitelisting.",
        },
        {
            "description": "HTTPS web app needing SSL cert management and sticky sessions",
            "choice": "ALB",
            "reason": "ALB supports SSL termination, ACM certificate management, "
                      "and cookie-based sticky sessions for web applications.",
        },
        {
            "description": "Financial trading platform, sub-ms latency, millions of TCP connections",
            "choice": "NLB",
            "reason": "NLB is built for extreme performance: millions of req/s "
                      "with ~100us latency vs ~1ms for ALB.",
        },
    ]

    # Quick reference
    print("Load Balancer Quick Reference:")
    print(f"  {'Feature':<25} {'ALB (Layer 7)':<25} {'NLB (Layer 4)'}")
    print("  " + "-" * 72)
    features = [
        ("Protocol", "HTTP, HTTPS, WebSocket", "TCP, UDP, TLS"),
        ("Routing", "Path, host, header-based", "Port-based only"),
        ("Latency", "~1ms", "~100us"),
        ("Static IP", "No (use Global Accelerator)", "Yes (per AZ)"),
        ("SSL termination", "Yes (ACM integrated)", "TLS passthrough"),
    ]
    for feat, alb, nlb in features:
        print(f"  {feat:<25} {alb:<25} {nlb}")
    print()

    for i, s in enumerate(scenarios, 1):
        print(f"Scenario {i}: {s['description']}")
        print(f"  Choice: {s['choice']}")
        print(f"  Reason: {s['reason']}")
        print()


# === Exercise 2: Health Check Configuration ===
# Problem: Configure ALB health checks for a web application.

def exercise_2():
    """Design ALB health check configuration with failure behavior."""

    config = {
        "protocol": "HTTP",
        "port": 80,
        "path": "/health",
        "interval_seconds": 15,
        "timeout_seconds": 5,
        "healthy_threshold": 2,
        "unhealthy_threshold": 3,
        "matcher": "200",
    }

    print("Health Check Configuration:")
    print("=" * 50)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    print("Configuration Explained:")
    explanations = [
        ("/health path", "Use dedicated health endpoint, not root (may redirect)."),
        ("15s interval", "Balance between responsiveness and traffic overhead."),
        ("2 healthy threshold", "2 consecutive passes to mark healthy (prevents flapping)."),
        ("3 unhealthy threshold", "3 consecutive failures before removal (tolerates transients)."),
        ("200 matcher", "Only HTTP 200 accepted; 503 signals unhealthy."),
    ]
    for name, explanation in explanations:
        print(f"  - {name}: {explanation}")
    print()

    # Simulate failure timeline
    check_interval = config["interval_seconds"]
    unhealthy_checks = config["unhealthy_threshold"]
    detection_time = check_interval * unhealthy_checks
    drain_time = 300  # default connection draining

    print("Failure Timeline:")
    print(f"  t=0s:    Instance starts failing health checks")
    print(f"  t={detection_time}s:  Marked unhealthy after {unhealthy_checks} failures "
          f"({unhealthy_checks} x {check_interval}s)")
    print(f"  t={detection_time}s:  ALB stops sending NEW requests")
    print(f"  t={detection_time}-{detection_time + drain_time}s: Existing connections drain")
    print(f"  t={detection_time}s+: ASG detects unhealthy -> terminates -> launches replacement")


# === Exercise 3: CDN Cache Configuration ===
# Problem: Configure CloudFront for global thumbnail delivery.

def exercise_3():
    """Design CDN caching strategy for media thumbnail delivery."""

    print("CloudFront CDN for Global Thumbnail Delivery:")
    print("=" * 60)
    print()
    print("  Service: Amazon CloudFront")
    print("  Origin: S3 bucket (media-bucket) in ap-northeast-2")
    print("  Edge locations: 400+ worldwide")
    print()

    print("  TTL Recommendation: 86400 seconds (24 hours)")
    print("  Why long TTL:")
    print("    - Thumbnails are typically static, rarely change")
    print("    - Edge serves from cache for 24h without contacting origin")
    print("    - Reduces S3 load and cost significantly")
    print()

    print("  For updated thumbnails, use versioned filenames:")
    print("    user-123-v1.jpg -> user-123-v2.jpg (new URL = immediate cache bypass)")
    print()

    print("  Cache Invalidation (when needed):")
    print("    # Invalidate specific thumbnail")
    print("    aws cloudfront create-invalidation \\")
    print("        --distribution-id DIST_ID \\")
    print('        --paths "/thumbnails/user-123.jpg"')
    print()
    print("    # Invalidate all thumbnails")
    print("    aws cloudfront create-invalidation \\")
    print("        --distribution-id DIST_ID \\")
    print('        --paths "/thumbnails/*"')
    print()
    print("  Cost: $0.005 per 1,000 invalidation paths (first 1,000/month free).")
    print("  Prefer URL versioning over invalidations for frequently updated assets.")


# === Exercise 4: Path-Based Routing Rule ===
# Problem: Route /api/v2/* to new microservice, everything else to legacy.

def exercise_4():
    """Configure ALB path-based routing for strangler fig migration."""

    # Strangler fig pattern: incrementally migrate monolith to microservices
    # using the load balancer as a traffic router.
    rules = [
        {
            "priority": 10,
            "condition": "path-pattern = /api/v2/*",
            "target": "microservice-api",
            "action": "forward",
        },
        {
            "priority": "default",
            "condition": "all other requests",
            "target": "legacy-app",
            "action": "forward",
        },
    ]

    print("ALB Path-Based Routing (Strangler Fig Pattern):")
    print("=" * 60)
    print()
    print("  Routing Rules:")
    print(f"  {'Priority':<12} {'Condition':<30} {'Target Group'}")
    print("  " + "-" * 55)
    for rule in rules:
        print(f"  {str(rule['priority']):<12} {rule['condition']:<30} {rule['target']}")
    print()

    print("  CLI Command:")
    print("    aws elbv2 create-rule \\")
    print("        --listener-arn <LISTENER_ARN> \\")
    print("        --priority 10 \\")
    print("        --conditions '[{\"Field\":\"path-pattern\",\"Values\":[\"/api/v2/*\"]}]' \\")
    print("        --actions '[{\"Type\":\"forward\",\"TargetGroupArn\":\"<MICROSERVICE_TG>\"}]'")
    print()
    print("  How it works:")
    print("    1. Rules evaluated in priority order (lowest number = highest priority)")
    print("    2. Priority 10: /api/v2/* -> microservice-api")
    print("    3. Default rule (always last): everything else -> legacy-app")
    print()
    print("  This pattern enables incremental monolith-to-microservice migration.")


# === Exercise 5: Load Balancer Troubleshooting ===
# Problem: Diagnose ALB returning 502 when EC2 responds correctly.

def exercise_5():
    """Troubleshoot ALB 502 Bad Gateway with working EC2 instance."""

    causes = [
        {
            "cause": "Instance not registered or marked unhealthy",
            "diagnose": (
                "aws elbv2 describe-target-health --target-group-arn <ARN>"
            ),
            "fix": "Register instance if missing. Fix health check if unhealthy.",
        },
        {
            "cause": "Security group blocking ALB-to-instance traffic",
            "diagnose": (
                "Check instance SG allows inbound from ALB SG on app port."
            ),
            "fix": "Add inbound rule: source = ALB SG, port = app port.",
        },
        {
            "cause": "Health check path returns non-200 status code",
            "diagnose": (
                "Test manually: curl http://<INSTANCE_IP>/health"
            ),
            "fix": "Fix endpoint to return 200, or update health check matcher.",
        },
        {
            "cause": "Application listening on wrong port",
            "diagnose": (
                "On instance: ss -tlnp | grep <port>"
            ),
            "fix": "Ensure app binds to the same port as target group config.",
        },
    ]

    print("ALB 502 Bad Gateway Troubleshooting:")
    print("=" * 60)
    print("Symptom: EC2 app responds on public IP, but ALB returns 502")
    print()

    for i, c in enumerate(causes, 1):
        print(f"  Cause {i}: {c['cause']}")
        print(f"    Diagnose: {c['diagnose']}")
        print(f"    Fix: {c['fix']}")
        print()


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: ALB vs NLB Selection ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: Health Check Configuration ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: CDN Cache Configuration ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: Path-Based Routing Rule ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: Load Balancer Troubleshooting ===")
    print("=" * 70)
    exercise_5()

    print("\nAll exercises completed!")
