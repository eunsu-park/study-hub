"""
Exercises for Lesson 06: Container Services
Topic: Cloud_Computing

Solutions to practice problems from the lesson.
Simulates container service decisions and workflows using Python.
"""


# === Exercise 1: Container vs VM Trade-off ===
# Problem: Compare containers (Fargate) vs VMs (EC2) for a monolith migration.

def exercise_1():
    """Analyze container vs VM trade-offs for application migration."""

    container_advantages = [
        {
            "advantage": "Portability and consistency",
            "detail": (
                "Docker image encapsulates application and all dependencies. "
                "Same image runs identically across dev, CI/CD, and production."
            ),
        },
        {
            "advantage": "Density and cost efficiency",
            "detail": (
                "Multiple containers share the host OS kernel, using far less "
                "memory than VMs (each with its own OS). Fargate charges only "
                "for reserved CPU/memory."
            ),
        },
    ]

    vm_better_scenarios = [
        {
            "scenario": "Applications requiring full OS access or kernel customization",
            "detail": (
                "Custom kernel modules, specialized hardware drivers, or apps "
                "that modify /proc and /sys require root-level OS access."
            ),
        },
        {
            "scenario": "Legacy apps that cannot be containerized",
            "detail": (
                "Hard dependencies on specific OS versions, registry entries "
                "(Windows), or COM objects make containerization impractical."
            ),
        },
    ]

    print("Container Advantages (Fargate):")
    for adv in container_advantages:
        print(f"  + {adv['advantage']}")
        print(f"    {adv['detail']}")
    print()

    print("Scenarios Where VM (EC2) is Better:")
    for s in vm_better_scenarios:
        print(f"  - {s['scenario']}")
        print(f"    {s['detail']}")


# === Exercise 2: Service Selection ===
# Problem: Choose the right AWS container service for each scenario.

def exercise_2():
    """Select appropriate AWS container service for various scenarios."""

    scenarios = [
        {
            "description": "Small startup, simple REST API, zero infra management, auto HTTPS",
            "service": "AWS App Runner",
            "reason": (
                "Zero infrastructure management: no clusters, no task definitions, "
                "no load balancer. Handles HTTPS, scaling, deployments automatically."
            ),
        },
        {
            "description": "50+ microservices, advanced networking, custom admission controllers",
            "service": "Amazon EKS",
            "reason": (
                "Kubernetes is the standard for complex microservice platforms. "
                "CRDs, admission webhooks, service mesh (Istio) are K8s-native. "
                "Multi-cloud portability."
            ),
        },
        {
            "description": "GPU-accelerated ML training as container batch jobs",
            "service": "ECS Fargate with GPU / EKS with GPU node groups",
            "reason": (
                "Batch jobs that start, process, and exit suit ECS tasks or "
                "Kubernetes Jobs. Both support GPU instances."
            ),
        },
        {
            "description": "Custom EC2 instance types (storage-optimized i3) required",
            "service": "ECS on EC2",
            "reason": (
                "Specific instance types that Fargate doesn't support require "
                "managing EC2 nodes. ECS on EC2 provides orchestration with "
                "full instance type control."
            ),
        },
    ]

    # Service comparison matrix
    services = {
        "App Runner": {"complexity": "Low", "k8s": False, "custom_instances": False},
        "ECS Fargate": {"complexity": "Medium", "k8s": False, "custom_instances": False},
        "ECS on EC2": {"complexity": "Medium-High", "k8s": False, "custom_instances": True},
        "EKS": {"complexity": "High", "k8s": True, "custom_instances": True},
    }

    print("AWS Container Service Comparison:")
    print(f"  {'Service':<15} {'Complexity':<15} {'K8s':<6} {'Custom Instances'}")
    print("  " + "-" * 55)
    for name, info in services.items():
        print(f"  {name:<15} {info['complexity']:<15} {str(info['k8s']):<6} {info['custom_instances']}")
    print()

    for i, s in enumerate(scenarios, 1):
        print(f"Scenario {i}: {s['description']}")
        print(f"  Service: {s['service']}")
        print(f"  Reason: {s['reason']}")
        print()


# === Exercise 3: Container Registry Workflow ===
# Problem: Complete ECR authentication, build, tag, and push workflow.

def exercise_3():
    """Describe the ECR container registry workflow commands."""

    account_id = "123456789012"
    region = "ap-northeast-2"
    repo = "my-app"
    tag = "v1.0"

    steps = [
        {
            "step": "Authenticate Docker to ECR",
            "command": (
                f"aws ecr get-login-password --region {region} | \\\n"
                f"    docker login --username AWS --password-stdin \\\n"
                f"    {account_id}.dkr.ecr.{region}.amazonaws.com"
            ),
        },
        {
            "step": "Build and tag the image",
            "command": (
                f"docker build -t {repo} .\n"
                f"docker tag {repo}:latest \\\n"
                f"    {account_id}.dkr.ecr.{region}.amazonaws.com/{repo}:{tag}\n"
                f"docker tag {repo}:latest \\\n"
                f"    {account_id}.dkr.ecr.{region}.amazonaws.com/{repo}:latest"
            ),
        },
        {
            "step": "Push to ECR",
            "command": (
                f"docker push {account_id}.dkr.ecr.{region}.amazonaws.com/{repo}:{tag}\n"
                f"docker push {account_id}.dkr.ecr.{region}.amazonaws.com/{repo}:latest"
            ),
        },
    ]

    print("ECR Container Registry Workflow:")
    print("=" * 60)
    print(f"  Account: {account_id}")
    print(f"  Region: {region}")
    print(f"  Repository: {repo}")
    print(f"  Tag: {tag}")
    print()

    # Prerequisite: create the repo if it doesn't exist
    print("  Prerequisite:")
    print(f"    aws ecr create-repository --repository-name {repo} --region {region}")
    print()

    for i, s in enumerate(steps, 1):
        print(f"  Step {i}: {s['step']}")
        print(f"    {s['command']}")
        print()


# === Exercise 4: Cloud Run Traffic Splitting ===
# Problem: Write gcloud command for canary deployment (10/90 split).

def exercise_4():
    """Demonstrate Cloud Run canary deployment via traffic splitting."""

    service = "payment-service"
    region = "asia-northeast3"
    new_revision = "payment-service-00003-xyz"
    old_revision = "payment-service-00002-abc"

    # Canary deployment: route only 10% to the new version
    canary_split = {new_revision: 10, old_revision: 90}

    print("Cloud Run Canary Deployment:")
    print("=" * 50)
    print(f"  Service: {service}")
    print(f"  Region: {region}")
    print(f"  New revision: {new_revision} (10%)")
    print(f"  Old revision: {old_revision} (90%)")
    print()

    revisions_arg = ",".join(f"{rev}={pct}" for rev, pct in canary_split.items())
    print(f"  Command:")
    print(f"    gcloud run services update-traffic {service} \\")
    print(f"        --region={region} \\")
    print(f"        --to-revisions={revisions_arg}")
    print()

    print("Why canary deployment is useful:")
    reasons = [
        "Risk reduction: only 10% of users hit the new version.",
        "Real traffic testing: synthetic tests may not reproduce all conditions.",
        "Gradual rollout: increase to 50%, then 100% if metrics look good.",
        "Instant rollback: route 100% back to stable if problems arise.",
    ]
    for r in reasons:
        print(f"  - {r}")
    print()

    print("  Promote to 100%:")
    print(f"    gcloud run services update-traffic {service} \\")
    print(f"        --region={region} --to-latest")


# === Exercise 5: Fargate vs Cloud Run Comparison ===
# Problem: Architectural differences for a stateless HTTP microservice.

def exercise_5():
    """Compare Fargate and Cloud Run for a bursty HTTP microservice."""

    comparison = [
        ("Scaling model", "Pre-configured min/max tasks", "Scales to zero; request-based"),
        ("Startup time", "30-60 seconds for new task", "Few seconds"),
        ("Pricing", "Per vCPU-hr/GB-hr (even idle)", "Per vCPU-sec/GB-sec (active only)"),
        ("Cluster needed", "Yes (ECS cluster config)", "No cluster concept"),
        ("Traffic splitting", "ALB weighted target groups", "Built-in revision-based"),
    ]

    print("Fargate vs Cloud Run Comparison:")
    print("=" * 75)
    print(f"  {'Aspect':<20} {'AWS Fargate':<28} {'GCP Cloud Run'}")
    print("  " + "-" * 72)
    for aspect, fargate, cloud_run in comparison:
        print(f"  {aspect:<20} {fargate:<28} {cloud_run}")

    print()
    print("Scenario: Stateless HTTP microservice, ~100 req/s with 10x spikes")
    print()
    print("Recommendation: GCP Cloud Run")
    reasons = [
        ("Cost", "Per-request billing is cheaper for bursty workloads. "
         "Fargate charges for idle time between spikes."),
        ("Scaling", "Fast scale-out handles 10x spikes well."),
        ("Simplicity", "No cluster management, no task definitions, no ALB needed."),
    ]
    for name, detail in reasons:
        print(f"  {name}: {detail}")
    print()
    print("For AWS: App Runner is the closer equivalent to Cloud Run.")
    print("Use Fargate for fine-grained VPC networking or ECS service integration.")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Container vs VM Trade-off ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: Service Selection ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Container Registry Workflow ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: Cloud Run Traffic Splitting ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: Fargate vs Cloud Run Comparison ===")
    print("=" * 70)
    exercise_5()

    print("\nAll exercises completed!")
