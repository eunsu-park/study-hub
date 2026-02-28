"""
Exercises for Lesson 09: Virtual Private Cloud (VPC)
Topic: Cloud_Computing

Solutions to practice problems from the lesson.
Simulates VPC design, CIDR planning, security groups, and NAT configuration.
"""

import ipaddress


# === Exercise 1: CIDR Block Planning ===
# Problem: Design subnet layout for a 3-tier app across 3 AZs.

def exercise_1():
    """Design a VPC subnet layout with CIDR block allocation."""

    vpc_cidr = ipaddress.IPv4Network("10.100.0.0/16")
    total_addresses = vpc_cidr.num_addresses

    # Subnet plan: /24 subnets give 256 addresses each (251 usable after AWS reserves 5).
    # Public subnets use .1-.3 range, private subnets use .11-.13 range.
    subnets = [
        {"name": "public-subnet-a", "az": "ap-northeast-2a", "cidr": "10.100.1.0/24", "purpose": "Web (public)"},
        {"name": "public-subnet-b", "az": "ap-northeast-2b", "cidr": "10.100.2.0/24", "purpose": "Web (public)"},
        {"name": "public-subnet-c", "az": "ap-northeast-2c", "cidr": "10.100.3.0/24", "purpose": "Web (public)"},
        {"name": "private-subnet-a", "az": "ap-northeast-2a", "cidr": "10.100.11.0/24", "purpose": "App/DB (private)"},
        {"name": "private-subnet-b", "az": "ap-northeast-2b", "cidr": "10.100.12.0/24", "purpose": "App/DB (private)"},
        {"name": "private-subnet-c", "az": "ap-northeast-2c", "cidr": "10.100.13.0/24", "purpose": "App/DB (private)"},
    ]

    print(f"VPC CIDR: {vpc_cidr} ({total_addresses:,} addresses)")
    print()

    print("Subnet Layout:")
    print(f"  {'Name':<22} {'AZ':<22} {'CIDR':<20} {'Purpose'}")
    print("  " + "-" * 80)
    for s in subnets:
        # Validate the subnet is within the VPC CIDR
        subnet_net = ipaddress.IPv4Network(s["cidr"])
        assert subnet_net.subnet_of(vpc_cidr), f"{s['cidr']} not in {vpc_cidr}"
        usable = subnet_net.num_addresses - 5  # AWS reserves 5 IPs per subnet
        print(f"  {s['name']:<22} {s['az']:<22} {s['cidr']:<20} {s['purpose']} ({usable} usable)")

    print()
    print("Design Principles:")
    print("  - Public subnets: .1.x-.3.x range for easy identification")
    print("  - Private subnets: .11.x-.13.x range, grouped away from public")
    print(f"  - 6 subnets use only 6 /24s of {total_addresses:,} addresses")
    print("  - AWS reserves 5 IPs/subnet: first 4 + last 1")


# === Exercise 2: Security Group vs Network ACL ===
# Problem: Write security group rules for an application server tier.

def exercise_2():
    """Configure security group rules for a three-tier architecture."""

    # Security groups are stateful (return traffic auto-allowed);
    # NACLs are stateless (both directions must be explicitly allowed).
    sg_rules = [
        {
            "rule": 1,
            "type": "Custom TCP",
            "protocol": "TCP",
            "port": 8080,
            "source": "sg-web-server (Web Server SG ID)",
            "description": "Allow traffic only from web servers",
        },
        {
            "rule": 2,
            "type": "SSH",
            "protocol": "TCP",
            "port": 22,
            "source": "sg-bastion (Bastion SG ID)",
            "description": "Allow admin SSH from bastion host",
        },
    ]

    print("Application Server Security Group -- Inbound Rules:")
    print(f"  {'Rule':<6} {'Type':<12} {'Port':<8} {'Source':<35} {'Description'}")
    print("  " + "-" * 80)
    for rule in sg_rules:
        print(f"  {rule['rule']:<6} {rule['type']:<12} {rule['port']:<8} "
              f"{rule['source']:<35} {rule['description']}")
    print()

    # Key difference between SG and NACL
    print("Security Group vs Network ACL:")
    print(f"  {'Feature':<15} {'Security Group':<30} {'Network ACL'}")
    print("  " + "-" * 70)
    comparisons = [
        ("State", "Stateful (return auto-allowed)", "Stateless (both dirs explicit)"),
        ("Scope", "Per instance/ENI", "Per subnet"),
        ("Rules", "Allow only (no deny)", "Allow and Deny (ordered)"),
    ]
    for feature, sg, nacl in comparisons:
        print(f"  {feature:<15} {sg:<30} {nacl}")

    print()
    print("Practical implication: With a NACL allowing inbound HTTP (port 80),")
    print("you must ALSO add an outbound rule for ephemeral ports (1024-65535).")


# === Exercise 3: NAT Gateway Purpose and Configuration ===
# Problem: Enable outbound-only internet access for private subnet instances.

def exercise_3():
    """Configure NAT Gateway for outbound-only internet access."""

    print("NAT Gateway Configuration:")
    print("=" * 60)
    print()
    print("Purpose: Outbound-only internet for private subnet instances.")
    print("  - Private instances CAN download packages from the internet.")
    print("  - Internet CANNOT initiate connections to private instances.")
    print()

    steps = [
        {
            "step": "Allocate an Elastic IP for the NAT Gateway",
            "command": "aws ec2 allocate-address --domain vpc",
        },
        {
            "step": "Create NAT Gateway in the PUBLIC subnet",
            "command": (
                "aws ec2 create-nat-gateway \\\n"
                "    --subnet-id subnet-public-a \\\n"
                "    --allocation-id eipalloc-xxx"
            ),
        },
        {
            "step": "Add default route in private route table to NAT Gateway",
            "command": (
                "aws ec2 create-route \\\n"
                "    --route-table-id rtb-private \\\n"
                "    --destination-cidr-block 0.0.0.0/0 \\\n"
                "    --nat-gateway-id nat-xxx"
            ),
        },
    ]

    for i, s in enumerate(steps, 1):
        print(f"  Step {i}: {s['step']}")
        print(f"    {s['command']}")
        print()

    print("Cost note: NAT Gateways charge ~$0.045/hour + per GB processed.")
    print("For dev environments, consider NAT Instances (free-tier eligible).")


# === Exercise 4: VPC Peering Scenario ===
# Problem: Establish VPC Peering between two VPCs in the same region.

def exercise_4():
    """Set up VPC Peering for cross-VPC private communication."""

    vpc_a = {"name": "VPC-A", "cidr": "10.0.0.0/16", "owner": "Company A"}
    vpc_b = {"name": "VPC-B", "cidr": "172.16.0.0/16", "owner": "Company B"}

    # Verify no CIDR overlap -- required for peering
    net_a = ipaddress.IPv4Network(vpc_a["cidr"])
    net_b = ipaddress.IPv4Network(vpc_b["cidr"])
    overlap = net_a.overlaps(net_b)

    print("VPC Peering Setup:")
    print("=" * 60)
    print(f"  {vpc_a['name']}: {vpc_a['cidr']} ({vpc_a['owner']})")
    print(f"  {vpc_b['name']}: {vpc_b['cidr']} ({vpc_b['owner']})")
    print(f"  CIDR overlap: {overlap} {'(peering blocked!)' if overlap else '(peering OK)'}")
    print()

    print("Key Limitations:")
    limitations = [
        "No transitive routing: A-B + B-C does NOT enable A-C.",
        "No overlapping CIDR blocks allowed.",
        "Cross-region peering adds latency and data transfer costs.",
    ]
    for lim in limitations:
        print(f"  - {lim}")
    print()

    steps = [
        "Create peering request (VPC-A -> VPC-B)",
        "Accept the peering request (VPC-B owner)",
        "Update route table in VPC-A: 172.16.0.0/16 -> pcx-xxx",
        "Update route table in VPC-B: 10.0.0.0/16 -> pcx-xxx",
        "Update security groups on both sides to allow peer CIDR",
    ]
    print("Steps:")
    for i, step in enumerate(steps, 1):
        print(f"  {i}. {step}")


# === Exercise 5: Three-Tier Architecture Design ===
# Problem: Design complete VPC architecture for HA three-tier web app.

def exercise_5():
    """Design a complete three-tier VPC architecture with HA."""

    architecture = {
        "VPC": "10.0.0.0/16",
        "Internet Gateway": "Attached to VPC",
        "AZs": {
            "ap-northeast-2a": {
                "Public Subnet (10.0.1.0/24)": ["ALB node", "NAT Gateway (EIP)"],
                "Private Subnet - App (10.0.11.0/24)": ["EC2 App Servers (ASG)"],
                "Private Subnet - DB (10.0.21.0/24)": ["RDS Primary"],
            },
            "ap-northeast-2b": {
                "Public Subnet (10.0.2.0/24)": ["ALB node", "NAT Gateway (redundant)"],
                "Private Subnet - App (10.0.12.0/24)": ["EC2 App Servers (ASG)"],
                "Private Subnet - DB (10.0.22.0/24)": ["RDS Standby (Multi-AZ)"],
            },
        },
        "Security Groups": {
            "ALB-SG": "Inbound 80/443 from 0.0.0.0/0",
            "App-SG": "Inbound 8080 from ALB-SG only",
            "DB-SG": "Inbound 5432 from App-SG only",
        },
        "Route Tables": {
            "Public": "0.0.0.0/0 -> Internet Gateway",
            "Private-App": "0.0.0.0/0 -> NAT Gateway",
            "Private-DB": "Local only (no internet route)",
        },
    }

    print("Three-Tier VPC Architecture:")
    print("=" * 65)
    print(f"VPC: {architecture['VPC']}")
    print(f"IGW: {architecture['Internet Gateway']}")
    print()

    for az, subnets in architecture["AZs"].items():
        print(f"  {az}:")
        for subnet, resources in subnets.items():
            print(f"    {subnet}")
            for r in resources:
                print(f"      - {r}")
    print()

    print("Security Groups:")
    for sg, rule in architecture["Security Groups"].items():
        print(f"  {sg}: {rule}")
    print()

    print("Route Tables:")
    for rt, route in architecture["Route Tables"].items():
        print(f"  {rt}: {route}")
    print()

    print("Design Rationale:")
    rationale = [
        "Each tier in separate subnet enables distinct security policies.",
        "Two NAT Gateways (one per AZ) ensure HA for private internet access.",
        "DB subnets have NO internet route -- zero internet exposure.",
        "ALB spans both public subnets for cross-AZ traffic distribution.",
    ]
    for r in rationale:
        print(f"  - {r}")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: CIDR Block Planning ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: Security Group vs Network ACL ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: NAT Gateway Configuration ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: VPC Peering Scenario ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: Three-Tier Architecture Design ===")
    print("=" * 70)
    exercise_5()

    print("\nAll exercises completed!")
