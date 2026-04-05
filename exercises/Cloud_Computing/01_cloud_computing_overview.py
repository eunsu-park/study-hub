"""
Exercises for Lesson 01: Cloud Computing Overview
Topic: Cloud_Computing

Solutions to practice problems from the lesson.
Simulates cloud concepts using Python data structures.
"""


# === Exercise 1: Classify the Service Model ===
# Problem: For each scenario, identify whether it is IaaS, PaaS, or SaaS.

def exercise_1():
    """Classify cloud service models for given scenarios."""

    # Define the responsibility layers for each service model.
    # In IaaS the customer manages from the OS up; in PaaS only code/data;
    # in SaaS the customer only manages data and access.
    service_models = {
        "IaaS": {
            "provider_manages": [
                "Networking", "Storage", "Servers", "Virtualization"
            ],
            "customer_manages": [
                "OS", "Middleware", "Runtime", "Data", "Application"
            ],
        },
        "PaaS": {
            "provider_manages": [
                "Networking", "Storage", "Servers", "Virtualization",
                "OS", "Middleware", "Runtime"
            ],
            "customer_manages": ["Data", "Application"],
        },
        "SaaS": {
            "provider_manages": [
                "Networking", "Storage", "Servers", "Virtualization",
                "OS", "Middleware", "Runtime", "Application"
            ],
            "customer_manages": ["Data"],
        },
    }

    # Scenarios to classify
    scenarios = [
        {
            "description": "A startup deploys their Node.js web app to Heroku via git push.",
            "answer": "PaaS",
            "reason": "Heroku manages the runtime, OS, and infrastructure. "
                      "The developer only provides application code.",
        },
        {
            "description": "A company rents EC2 instances and installs their own database software.",
            "answer": "IaaS",
            "reason": "EC2 provides raw virtual machines. The company manages "
                      "the OS, installed software, and configuration.",
        },
        {
            "description": "A sales team uses Salesforce CRM to manage their customer pipeline.",
            "answer": "SaaS",
            "reason": "Salesforce is fully managed software delivered over the internet. "
                      "Users consume it without managing any underlying infrastructure.",
        },
        {
            "description": "A data engineering team uses Google App Engine to host a Python ETL service.",
            "answer": "PaaS",
            "reason": "App Engine manages the runtime environment. The team deploys "
                      "code; Google handles scaling, OS patching, and infrastructure.",
        },
    ]

    for i, scenario in enumerate(scenarios, 1):
        model = scenario["answer"]
        print(f"Scenario {i}: {scenario['description']}")
        print(f"  Classification: {model}")
        print(f"  Reason: {scenario['reason']}")
        # Show what the provider vs customer manages under this model
        print(f"  Provider manages: {', '.join(service_models[model]['provider_manages'])}")
        print(f"  Customer manages: {', '.join(service_models[model]['customer_manages'])}")
        print()


# === Exercise 2: Shared Responsibility Mapping ===
# Problem: Match each security task to the correct responsible party (AWS or Customer)
# for an EC2-based (IaaS) web application.

def exercise_2():
    """Map security responsibilities for an EC2 deployment."""

    # The shared responsibility model defines who handles what.
    # For IaaS (EC2), AWS handles infrastructure "OF the cloud";
    # the customer handles everything "IN the cloud" (OS and above).
    responsibility_map = {
        "Patching the host hypervisor": {
            "party": "AWS",
            "reason": "AWS manages the virtualization layer beneath EC2 instances.",
        },
        "Installing OS security updates on the EC2 instance": {
            "party": "Customer",
            "reason": "The OS on IaaS is the customer's responsibility.",
        },
        "Encrypting application data stored in the database": {
            "party": "Customer",
            "reason": "Data protection is always the customer's responsibility.",
        },
        "Ensuring physical access controls at the data center": {
            "party": "AWS",
            "reason": "Physical data center security is AWS's responsibility.",
        },
        "Configuring security group firewall rules": {
            "party": "Customer",
            "reason": "Network/firewall configuration on IaaS is customer-managed.",
        },
    }

    print("Security Task Responsibility Mapping (EC2 / IaaS):")
    print("-" * 70)
    for task, info in responsibility_map.items():
        print(f"  Task: {task}")
        print(f"    Responsible Party: {info['party']}")
        print(f"    Reason: {info['reason']}")
        print()

    print("Key insight: In IaaS, AWS is responsible OF the cloud (physical")
    print("infrastructure and virtualization), while the customer is responsible")
    print("IN the cloud (OS and everything above it).")


# === Exercise 3: Disaster Recovery Strategy Selection ===
# Problem: Choose DR strategy for a fintech company with real-time payments.

def exercise_3():
    """Select appropriate DR strategy based on business requirements."""

    # Define DR strategies with their characteristics
    dr_strategies = {
        "Backup & Restore": {
            "rpo": "Hours",
            "rto": "Hours",
            "cost": "$",
            "description": "Regular backups to another region; restore on disaster.",
        },
        "Pilot Light": {
            "rpo": "Minutes",
            "rto": "Tens of minutes",
            "cost": "$$",
            "description": "Minimal core infrastructure always running; scale up on disaster.",
        },
        "Warm Standby": {
            "rpo": "Seconds to minutes",
            "rto": "Minutes",
            "cost": "$$$",
            "description": "Scaled-down but fully functional copy of production.",
        },
        "Multi-Site Active-Active": {
            "rpo": "Near-zero",
            "rto": "Near-zero",
            "cost": "$$$$",
            "description": "Full production in 2+ regions; traffic distributed across all.",
        },
    }

    # Business requirements
    requirements = {
        "outage_cost_per_30min": 500_000,  # $500,000 per 30 minutes
        "data_loss_tolerance": "zero",      # Regulators require no transaction data loss
        "budget": "substantial",
    }

    print("Business Requirements:")
    print(f"  30-min outage cost: ${requirements['outage_cost_per_30min']:,}")
    print(f"  Data loss tolerance: {requirements['data_loss_tolerance']}")
    print(f"  Budget: {requirements['budget']}")
    print()

    print("DR Strategy Comparison:")
    print(f"{'Strategy':<25} {'RPO':<20} {'RTO':<20} {'Cost':<8}")
    print("-" * 73)
    for name, info in dr_strategies.items():
        print(f"{name:<25} {info['rpo']:<20} {info['rto']:<20} {info['cost']:<8}")

    # Selection logic: zero data loss requires RPO ~0, high outage cost requires RTO ~0
    selected = "Multi-Site Active-Active"
    print(f"\nRecommended Strategy: {selected}")
    print(f"  RPO requirement: No data loss -> RPO ~ 0. Only Active-Active meets this.")
    print(f"  RTO requirement: $500K/30min outage -> RTO ~ 0. Active-Active routes")
    print(f"    traffic away from a failed region in seconds.")
    print(f"\n  Key AWS services:")
    services = [
        "Route 53 (latency-based routing / health check failover)",
        "DynamoDB Global Tables (multi-region, multi-active replication)",
        "Aurora Global Database (sub-second replication lag)",
        "AWS Global Accelerator (consistent, low-latency global routing)",
        "S3 Cross-Region Replication (object storage assets)",
    ]
    for svc in services:
        print(f"    - {svc}")


# === Exercise 4: NIST Characteristics in Practice ===
# Problem: Identify which of the five NIST essential characteristics is demonstrated.

def exercise_4():
    """Identify NIST cloud computing characteristics from scenarios."""

    # NIST's 5 Essential Characteristics
    nist_characteristics = {
        "On-Demand Self-Service": (
            "Users provision resources directly without human intervention."
        ),
        "Broad Network Access": (
            "Access through standard mechanisms over the network."
        ),
        "Resource Pooling": (
            "Resources shared through multi-tenant model with physical location abstraction."
        ),
        "Rapid Elasticity": (
            "Automatic scaling up/down based on demand. Resources appear unlimited."
        ),
        "Measured Service": (
            "Usage monitoring, reporting, and transparent billing."
        ),
    }

    scenarios = [
        {
            "description": (
                "An e-commerce site automatically adds 50 more servers during a "
                "Black Friday sale and scales back down afterward."
            ),
            "characteristic": "Rapid Elasticity",
            "reason": "The system scales resources up and down automatically to match demand.",
        },
        {
            "description": (
                "A developer provisions a new PostgreSQL database in 3 minutes "
                "through the AWS Console without contacting anyone."
            ),
            "characteristic": "On-Demand Self-Service",
            "reason": "The developer provisions resources without human interaction with the provider.",
        },
        {
            "description": (
                "AWS charges a customer based on exact GB-hours of storage used "
                "each month, visible in a detailed billing dashboard."
            ),
            "characteristic": "Measured Service",
            "reason": (
                "Usage is monitored, controlled, and reported transparently, "
                "enabling pay-per-use billing."
            ),
        },
        {
            "description": (
                "Multiple customers' workloads run on the same physical servers, "
                "but each customer sees logically isolated resources."
            ),
            "characteristic": "Resource Pooling",
            "reason": (
                "Physical resources are shared across multiple tenants "
                "(multi-tenancy) with logical isolation."
            ),
        },
    ]

    print("NIST Essential Characteristics of Cloud Computing")
    print("=" * 60)
    for name, desc in nist_characteristics.items():
        print(f"  {name}: {desc}")
    print()

    for i, s in enumerate(scenarios, 1):
        print(f"Scenario {i}: {s['description']}")
        print(f"  -> {s['characteristic']}")
        print(f"     {s['reason']}")
        print()


# === Exercise 5: Cost Model Analysis ===
# Problem: Recommend the best pricing option for each workload.

def exercise_5():
    """Analyze cloud pricing models for different workload patterns."""

    pricing_options = {
        "On-Demand": {
            "discount": "0%",
            "commitment": "None",
            "interruptible": False,
            "best_for": "Unpredictable, short-term workloads",
        },
        "Reserved Instances": {
            "discount": "Up to 72%",
            "commitment": "1-3 years",
            "interruptible": False,
            "best_for": "Steady-state, predictable workloads",
        },
        "Spot Instances": {
            "discount": "Up to 90%",
            "commitment": "None",
            "interruptible": True,
            "best_for": "Fault-tolerant, flexible workloads",
        },
    }

    workloads = [
        {
            "description": (
                "A production web server that must run 24/7 with no interruption "
                "for at least the next 2 years."
            ),
            "recommendation": "Reserved Instances",
            "reason": (
                "The workload runs continuously for a known duration. Committing "
                "upfront saves up to 72% compared to On-Demand. Since it cannot "
                "be interrupted, Spot is not viable."
            ),
        },
        {
            "description": (
                "A batch data processing job that runs nightly for 3 hours and "
                "can be restarted if interrupted."
            ),
            "recommendation": "Spot Instances",
            "reason": (
                "Batch jobs that can tolerate interruption and be restarted are "
                "the ideal use case for Spot. The up-to-90% discount significantly "
                "reduces cost."
            ),
        },
        {
            "description": (
                "A new microservice being tested in development that will run "
                "unpredictably over the next 2 weeks."
            ),
            "recommendation": "On-Demand",
            "reason": (
                "Development/test workloads have unpredictable and short-term "
                "usage patterns. No commitment benefit from Reserved Instances."
            ),
        },
    ]

    print("Pricing Model Comparison:")
    print(f"{'Option':<22} {'Discount':<15} {'Commitment':<12} {'Interruptible'}")
    print("-" * 65)
    for name, info in pricing_options.items():
        print(f"{name:<22} {info['discount']:<15} {info['commitment']:<12} {info['interruptible']}")
    print()

    for i, w in enumerate(workloads, 1):
        print(f"Workload {i}: {w['description']}")
        print(f"  Recommendation: {w['recommendation']}")
        print(f"  Reason: {w['reason']}")
        print()


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: Classify the Service Model ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: Shared Responsibility Mapping ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Disaster Recovery Strategy Selection ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: NIST Characteristics in Practice ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: Cost Model Analysis ===")
    print("=" * 70)
    exercise_5()

    print("\nAll exercises completed!")
