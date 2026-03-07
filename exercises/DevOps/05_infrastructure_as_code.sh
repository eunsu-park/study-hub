#!/bin/bash
# Exercises for Lesson 05: Infrastructure as Code
# Topic: DevOps
# Solutions to practice problems from the lesson.

# === Exercise 1: Terraform Resource Definition ===
# Problem: Write Terraform configuration for a VPC with public/private
# subnets, NAT gateway, and security groups.
exercise_1() {
    echo "=== Exercise 1: Terraform Resource Definition ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# main.tf — VPC with public/private subnets

terraform {
  required_version = ">= 1.7"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  backend "s3" {
    bucket         = "myorg-terraform-state"
    key            = "network/vpc/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "terraform-locks"
    encrypt        = true
  }
}

variable "environment" {
  type        = string
  description = "Environment name (dev, staging, prod)"
}

variable "vpc_cidr" {
  type    = string
  default = "10.0.0.0/16"
}

locals {
  common_tags = {
    Environment = var.environment
    ManagedBy   = "terraform"
    Project     = "platform"
  }
}

resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = merge(local.common_tags, { Name = "${var.environment}-vpc" })
}

resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, count.index)
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = merge(local.common_tags, {
    Name = "${var.environment}-public-${count.index}"
    Tier = "public"
  })
}

resource "aws_subnet" "private" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, count.index + 10)
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = merge(local.common_tags, {
    Name = "${var.environment}-private-${count.index}"
    Tier = "private"
  })
}

resource "aws_internet_gateway" "gw" {
  vpc_id = aws_vpc.main.id
  tags   = merge(local.common_tags, { Name = "${var.environment}-igw" })
}

resource "aws_nat_gateway" "nat" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public[0].id
  tags          = merge(local.common_tags, { Name = "${var.environment}-nat" })
}

resource "aws_eip" "nat" {
  domain = "vpc"
  tags   = merge(local.common_tags, { Name = "${var.environment}-nat-eip" })
}

output "vpc_id" {
  value = aws_vpc.main.id
}

output "public_subnet_ids" {
  value = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  value = aws_subnet.private[*].id
}

data "aws_availability_zones" "available" {
  state = "available"
}

# Key IaC principles demonstrated:
# 1. Remote backend (S3 + DynamoDB) — team collaboration + state locking
# 2. Variables — reusable across environments
# 3. Locals — DRY tags
# 4. count + cidrsubnet — dynamic subnet creation
# 5. Outputs — compose with other modules
SOLUTION
}

# === Exercise 2: Terraform State Management ===
# Problem: Explain state file concepts and demonstrate state operations.
exercise_2() {
    echo "=== Exercise 2: Terraform State Management ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Terraform State — Key Concepts and Operations

# 1. WHY STATE?
# Terraform maps real-world resources to your configuration.
# State tracks: resource IDs, attributes, dependencies, metadata.
# Without state, Terraform can't know what it's managing.

# 2. STATE BACKENDS
backends = {
    "local":     "Default. terraform.tfstate file on disk. No locking. No sharing.",
    "s3":        "AWS S3 bucket + DynamoDB for locking. Most common for AWS teams.",
    "gcs":       "Google Cloud Storage. Built-in locking via bucket object metadata.",
    "azurerm":   "Azure Blob Storage. Supports locking via blob leases.",
    "consul":    "HashiCorp Consul. Good for multi-cloud or on-premise.",
    "terraform_cloud": "Managed by HashiCorp. UI, RBAC, remote runs included.",
}

# 3. ESSENTIAL STATE COMMANDS
commands = {
    "terraform state list": "List all resources in state",
    "terraform state show aws_vpc.main": "Show details of one resource",
    "terraform state mv aws_vpc.main module.network.aws_vpc.main":
        "Rename/move a resource (e.g., when refactoring to modules)",
    "terraform state rm aws_vpc.main":
        "Remove from state WITHOUT destroying (adopt externally managed resource)",
    "terraform import aws_vpc.main vpc-12345":
        "Import existing resource INTO state",
    "terraform state pull > backup.json":
        "Download state for inspection",
}

for cmd, desc in commands.items():
    print(f"  $ {cmd}")
    print(f"    {desc}\n")

# 4. STATE LOCKING
# Problem: Two engineers run `terraform apply` simultaneously
# Solution: DynamoDB lock table (S3 backend) prevents concurrent writes
# If lock is stuck: terraform force-unlock LOCK_ID

# 5. STATE FILE SECURITY
# State may contain secrets (database passwords, API keys)
# ALWAYS: encrypt at rest (S3 SSE, GCS encryption)
# NEVER: commit terraform.tfstate to Git
# Use: terraform_remote_state data source to share outputs between stacks
SOLUTION
}

# === Exercise 3: Terraform Modules ===
# Problem: Refactor a flat Terraform configuration into reusable modules.
exercise_3() {
    echo "=== Exercise 3: Terraform Modules ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Module structure:
# modules/
#   vpc/
#     main.tf       # Resource definitions
#     variables.tf  # Input variables
#     outputs.tf    # Output values
#   ec2/
#     main.tf
#     variables.tf
#     outputs.tf
# environments/
#   dev/main.tf     # Uses modules with dev-specific vars
#   prod/main.tf    # Uses modules with prod-specific vars

# --- modules/vpc/variables.tf ---
# variable "environment" { type = string }
# variable "vpc_cidr"    { type = string; default = "10.0.0.0/16" }

# --- modules/vpc/main.tf ---
# resource "aws_vpc" "main" { ... }
# resource "aws_subnet" "public" { ... }

# --- modules/vpc/outputs.tf ---
# output "vpc_id" { value = aws_vpc.main.id }
# output "public_subnet_ids" { value = aws_subnet.public[*].id }

# --- environments/dev/main.tf ---
# module "vpc" {
#   source      = "../../modules/vpc"
#   environment = "dev"
#   vpc_cidr    = "10.0.0.0/16"
# }
#
# module "app" {
#   source     = "../../modules/ec2"
#   vpc_id     = module.vpc.vpc_id
#   subnet_ids = module.vpc.public_subnet_ids
#   instance_type = "t3.micro"   # Dev uses smaller instances
# }

# Module design best practices:
module_rules = [
    "One module = one logical component (VPC, ECS cluster, RDS)",
    "Expose only necessary outputs (don't leak internal resource IDs)",
    "Use variable validation blocks for input constraints",
    "Pin module versions: source = 'git::...?ref=v1.2.0'",
    "Document with README.md and examples/ directory",
    "Test modules with terratest or terraform test (v1.6+)",
]

print("Module Design Best Practices:")
for rule in module_rules:
    print(f"  - {rule}")
SOLUTION
}

# === Exercise 4: Drift Detection ===
# Problem: Implement a drift detection workflow that compares Terraform
# plan output and alerts on unexpected changes.
exercise_4() {
    echo "=== Exercise 4: Drift Detection ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import json

def parse_terraform_plan(plan_json: dict) -> dict:
    """Parse terraform plan JSON output and categorize changes."""
    changes = {"create": [], "update": [], "delete": [], "no-op": []}

    for rc in plan_json.get("resource_changes", []):
        address = rc["address"]
        actions = rc["change"]["actions"]

        if actions == ["no-op"]:
            changes["no-op"].append(address)
        elif actions == ["create"]:
            changes["create"].append(address)
        elif actions == ["delete"]:
            changes["delete"].append(address)
        elif "update" in actions:
            changes["update"].append(address)
        elif actions == ["delete", "create"]:
            changes["update"].append(f"{address} (replace)")

    return changes

def detect_drift(changes: dict) -> dict:
    """Identify drift: unexpected updates or deletes not from code changes."""
    drift_indicators = {
        "drift_detected": len(changes["update"]) > 0 or len(changes["delete"]) > 0,
        "unexpected_updates": changes["update"],
        "unexpected_deletes": changes["delete"],
        "severity": "none",
    }

    if changes["delete"]:
        drift_indicators["severity"] = "critical"
    elif changes["update"]:
        drift_indicators["severity"] = "warning"

    return drift_indicators

# Example: scheduled drift check in CI
# .github/workflows/drift-check.yml
# schedule:
#   - cron: "0 8 * * *"     # Daily at 8am
# jobs:
#   drift-check:
#     steps:
#       - run: terraform init
#       - run: terraform plan -detailed-exitcode -out=plan.tfplan
#         # Exit code: 0=no changes, 1=error, 2=changes detected
#       - run: terraform show -json plan.tfplan > plan.json
#       - run: python detect_drift.py plan.json
#       - if: failure()
#         uses: slackapi/slack-github-action@v1
#         with:
#           payload: '{"text": "Terraform drift detected in production!"}'

# Drift detection commands:
# terraform plan -refresh-only    # Check for drift without applying
# terraform plan -detailed-exitcode  # Exit 2 if changes needed
# terraform plan -json            # Machine-parsable output

print("Drift Detection Strategy:")
print("  1. Schedule daily terraform plan -refresh-only")
print("  2. Parse plan JSON for unexpected changes")
print("  3. Alert on updates/deletes not from recent commits")
print("  4. Auto-remediate by re-applying desired state (GitOps)")
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 05: Infrastructure as Code"
echo "========================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
