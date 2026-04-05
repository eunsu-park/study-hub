"""
Exercises for Lesson 16: Infrastructure as Code
Topic: Cloud_Computing

Solutions to practice problems from the lesson.
Simulates IaC tool selection, Terraform workflows, state management, and module design.
"""


# === Exercise 1: IaC Tool Selection ===
def exercise_1():
    """Evaluate IaC tools for different scenarios."""

    print("IaC Tool Selection:")
    print("=" * 70)
    print()

    scenarios = [
        {
            "scenario": "Manage resources on both AWS and GCP from one codebase",
            "tool": "Terraform",
            "reason": (
                "Multi-cloud support; single HCL codebase with provider "
                "plugins for each cloud."
            ),
        },
        {
            "scenario": "Team only uses AWS, wants native JSON/YAML support",
            "tool": "CloudFormation",
            "reason": (
                "Native AWS service; no extra tooling needed; deep "
                "integration with AWS services."
            ),
        },
        {
            "scenario": "Configure OS packages and software on existing VMs",
            "tool": "Ansible",
            # Why not Terraform: Terraform handles provisioning (creating
            # infrastructure), while Ansible handles configuration management
            # (installing software, configuring services on existing servers).
            "reason": (
                "Procedural tool designed for configuration management "
                "(SSH-based, agentless). Terraform provisions infrastructure; "
                "Ansible configures it."
            ),
        },
        {
            "scenario": "Write infrastructure in TypeScript instead of HCL",
            "tool": "Pulumi",
            "reason": (
                "Supports general-purpose languages (Python, TypeScript, Go, "
                "C#); full IDE support, type safety."
            ),
        },
    ]

    print(f"  {'Scenario':<55} {'Best Tool':<15} {'Reason'}")
    print("  " + "-" * 95)
    for s in scenarios:
        print(f"  {s['scenario']:<55} {s['tool']:<15}")
        print(f"  {'':>55} {s['reason']}")
        print()

    print("  Note: Terraform and CloudFormation are declarative (define desired state).")
    print("  Ansible is procedural (define steps to execute).")
    print("  Common combo: Terraform (provisioning) + Ansible (configuration).")


# === Exercise 2: Terraform Workflow ===
def exercise_2():
    """List and explain the Terraform command workflow from init to apply."""

    print("Terraform Workflow (init to apply):")
    print("=" * 70)
    print()

    steps = [
        {
            "command": "terraform init",
            "description": "Initialize the working directory",
            "details": (
                "Downloads provider plugins (hashicorp/aws) and sets up "
                "the backend. Run once per new project or after changing "
                "backend/provider configuration."
            ),
        },
        {
            "command": "terraform validate",
            "description": "Validate configuration syntax",
            "details": (
                "Catches HCL syntax errors and type mismatches before "
                "touching real infrastructure. Fast, local-only check."
            ),
        },
        {
            "command": "terraform plan",
            "description": "Preview changes (dry run)",
            "details": (
                "Shows what resources will be created/modified/destroyed. "
                "Does NOT change anything. Review this output carefully."
            ),
        },
        {
            "command": "terraform apply",
            "description": "Apply the configuration",
            "details": (
                "Creates the actual resources. Prompts for confirmation "
                "unless -auto-approve is used."
            ),
        },
        {
            "command": "terraform destroy",
            "description": "Destroy all managed resources (optional)",
            "details": (
                "Removes all resources managed by this configuration. "
                "Use with extreme caution in production."
            ),
        },
    ]

    for i, s in enumerate(steps, 1):
        print(f"  Step {i}: {s['command']}")
        print(f"    Purpose: {s['description']}")
        print(f"    Details: {s['details']}")
        print()

    # What happens internally during apply
    print("  What happens internally during 'terraform apply':")
    internals = [
        "Reads the current state (.tfstate file or remote backend).",
        "Calls the cloud API to check the current real-world state.",
        "Calculates the diff between desired state (.tf) and actual state.",
        "Executes API calls to create/update/delete in dependency order.",
        "Writes the new state to the backend.",
    ]
    for i, step in enumerate(internals, 1):
        print(f"    {i}. {step}")
    print()

    # Critical warning
    print("  CRITICAL: Never manually edit .tfstate files.")
    print("  Use 'terraform state' commands instead. Corrupting the state")
    print("  file is the most common Terraform disaster.")


# === Exercise 3: Remote State and Locking ===
def exercise_3():
    """Configure remote backend with state locking for team safety."""

    print("Remote State and Locking:")
    print("=" * 70)
    print()
    print("  Problem: 3 engineers run 'terraform apply' on the same infrastructure.")
    print()

    # Problems with local state
    problems = [
        ("State conflicts",
         "Two engineers apply simultaneously -> both read old state -> "
         "one overwrites the other's changes, causing drift."),
        ("State loss",
         "State file on one laptop -> engineer leaves or laptop breaks -> "
         "state lost -> Terraform no longer knows what it manages."),
        ("No history",
         "No audit trail of who changed what and when."),
    ]
    print("  Problems with local state in a team:")
    for name, detail in problems:
        print(f"    - {name}: {detail}")
    print()

    # Solution: Remote backend with locking
    print("  Solution: Remote backend with state locking")
    print()
    print("  backend.tf:")
    print("    terraform {")
    print('      backend "s3" {')
    print('        bucket         = "mycompany-terraform-state"')
    print('        key            = "production/terraform.tfstate"')
    print('        region         = "ap-northeast-2"')
    print("        encrypt        = true")
    print('        dynamodb_table = "terraform-state-locks"  # Lock table')
    print("      }")
    print("    }")
    print()

    # DynamoDB lock table setup
    # Why DynamoDB for locking: It provides atomic conditional writes,
    # which guarantee only one terraform apply can hold the lock at a time.
    print("  Create DynamoDB lock table (one-time setup):")
    print("    aws dynamodb create-table \\")
    print("        --table-name terraform-state-locks \\")
    print("        --attribute-definitions AttributeName=LockID,AttributeType=S \\")
    print("        --key-schema AttributeName=LockID,KeyType=HASH \\")
    print("        --billing-mode PAY_PER_REQUEST")
    print()

    # How locking works
    print("  How locking works:")
    locking_steps = [
        "Engineer A runs 'terraform apply' -> acquires lock in DynamoDB.",
        "Engineer B runs 'terraform apply' -> lock is held -> "
        "Terraform exits: 'Error acquiring the state lock'.",
        "Engineer A finishes -> lock is released.",
        "Engineer B can now proceed safely.",
    ]
    for i, step in enumerate(locking_steps, 1):
        print(f"    {i}. {step}")
    print()

    # GCP alternative
    print("  GCP alternative (locking is built-in via object versioning):")
    print("    terraform {")
    print('      backend "gcs" {')
    print('        bucket = "mycompany-terraform-state"')
    print('        prefix = "production/terraform.tfstate"')
    print("      }")
    print("    }")


# === Exercise 4: Variables and Outputs ===
def exercise_4():
    """Refactor a hardcoded RDS resource to use variables and outputs."""

    print("Terraform Variables and Outputs:")
    print("=" * 70)
    print()

    print("  Original (hardcoded) resource:")
    print('    resource "aws_db_instance" "main" {')
    print('      identifier        = "myapp-db"')
    print('      engine            = "mysql"')
    print('      instance_class    = "db.t3.micro"')
    print('      allocated_storage = 20')
    print('      username          = "admin"')
    print('      password          = "SuperSecret123!"  # DANGER: hardcoded!')
    print("    }")
    print()

    print("  Refactored -- variables.tf:")
    variables = [
        ('db_identifier', 'string', '"myapp-db"',
         "The name of the RDS instance", False),
        ('db_instance_class', 'string', '"db.t3.micro"',
         "The instance type of the RDS instance", False),
        ('db_allocated_storage', 'number', '20',
         "The allocated storage size in gigabytes", False),
        ('db_username', 'string', '"admin"',
         "The master username for the database", False),
        ('db_password', 'string', None,
         "The master password for the database", True),
    ]

    for name, vtype, default, desc, sensitive in variables:
        print(f'    variable "{name}" {{')
        print(f'      description = "{desc}"')
        print(f'      type        = {vtype}')
        if default:
            print(f'      default     = {default}')
        if sensitive:
            # Why sensitive: Redacted in plan/apply output and state list.
            # Prevents accidental exposure in CI/CD logs.
            print(f'      sensitive   = true  # Never shown in plan/apply output')
        print(f'    }}')
        print()

    print("  Refactored -- main.tf:")
    print('    resource "aws_db_instance" "main" {')
    print("      identifier        = var.db_identifier")
    print('      engine            = "mysql"')
    print('      engine_version    = "8.0"')
    print("      instance_class    = var.db_instance_class")
    print("      allocated_storage = var.db_allocated_storage")
    print("      username          = var.db_username")
    print("      password          = var.db_password")
    print("    }")
    print()

    print("  Refactored -- outputs.tf:")
    print('    output "db_endpoint" {')
    print('      description = "The connection endpoint"')
    print("      value       = aws_db_instance.main.endpoint")
    print("    }")
    print('    output "db_port" {')
    print('      description = "The database port"')
    print("      value       = aws_db_instance.main.port")
    print("    }")
    print()

    # How to pass the sensitive password
    print("  Set sensitive password via environment variable (never in .tfvars):")
    print('    export TF_VAR_db_password="SuperSecret123!"')
    print("    terraform apply")
    print()
    print("  Key points:")
    print("    - sensitive = true redacts values in logs and CLI output.")
    print("    - Never hardcode credentials in .tf files committed to Git.")
    print("    - Pass via TF_VAR_ env vars or Vault provider integration.")


# === Exercise 5: Terraform Module Design ===
def exercise_5():
    """Design module structure for deploying three-tier architecture to dev and prod."""

    print("Terraform Module Design for Multi-Environment:")
    print("=" * 70)
    print()
    print("  Goal: Same three-tier architecture (VPC + web + DB) for dev and prod.")
    print("  Dev uses smaller, cheaper instances.")
    print()

    # Directory structure
    print("  Directory Structure:")
    structure = [
        "terraform/",
        "  modules/",
        "    vpc/",
        "      main.tf       # VPC, subnets, IGW, route tables",
        "      variables.tf  # cidr_block, name, etc.",
        "      outputs.tf    # vpc_id, public_subnet_ids, private_subnet_ids",
        "    web/",
        "      main.tf       # EC2 + ASG + ALB",
        "      variables.tf  # instance_type, subnet_ids",
        "      outputs.tf    # alb_dns_name",
        "    database/",
        "      main.tf       # RDS + subnet group + security group",
        "      variables.tf  # db_instance_class, subnet_ids",
        "      outputs.tf    # db_endpoint, db_port",
        "  environments/",
        "    dev/",
        "      main.tf",
        "      terraform.tfvars",
        "    prod/",
        "      main.tf",
        "      terraform.tfvars",
    ]
    for line in structure:
        print(f"    {line}")
    print()

    # Dev environment
    # Why different CIDR blocks: Prevents conflicts if VPC peering is needed
    # between dev and prod later.
    print("  environments/dev/main.tf:")
    print('    module "vpc" {')
    print('      source     = "../../modules/vpc"')
    print('      name       = "myapp-dev"')
    print('      cidr_block = "10.0.0.0/16"')
    print("    }")
    print('    module "web" {')
    print('      source        = "../../modules/web"')
    print('      instance_type = "t3.micro"        # Cheap dev instance')
    print("      subnet_ids    = module.vpc.public_subnet_ids")
    print("    }")
    print('    module "database" {')
    print('      source            = "../../modules/database"')
    print('      db_instance_class = "db.t3.micro"  # Smallest RDS')
    print("      subnet_ids        = module.vpc.private_subnet_ids")
    print("    }")
    print()

    # Prod environment
    print("  environments/prod/main.tf:")
    print('    module "vpc" {')
    print('      source     = "../../modules/vpc"')
    print('      name       = "myapp-prod"')
    print('      cidr_block = "10.1.0.0/16"')
    print("    }")
    print('    module "web" {')
    print('      source        = "../../modules/web"')
    print('      instance_type = "t3.large"        # Production-grade')
    print("      subnet_ids    = module.vpc.public_subnet_ids")
    print("    }")
    print('    module "database" {')
    print('      source            = "../../modules/database"')
    print('      db_instance_class = "db.r6g.large" # Production RDS')
    print("      multi_az          = true            # High availability")
    print("      subnet_ids        = module.vpc.private_subnet_ids")
    print("    }")
    print()

    # Benefits
    print("  Benefits of This Structure:")
    benefits = [
        "Module code written once and reused across environments -- DRY principle.",
        "Environment-specific params isolated in each environment's main.tf/tfvars.",
        "Different state files per environment prevent dev destroy from affecting prod.",
        "Adding a new environment (staging) is just a new directory calling existing modules.",
    ]
    for b in benefits:
        print(f"    - {b}")


if __name__ == "__main__":
    print("=" * 70)
    print("=== Exercise 1: IaC Tool Selection ===")
    print("=" * 70)
    exercise_1()

    print("\n" + "=" * 70)
    print("=== Exercise 2: Terraform Workflow ===")
    print("=" * 70)
    exercise_2()

    print("\n" + "=" * 70)
    print("=== Exercise 3: Remote State and Locking ===")
    print("=" * 70)
    exercise_3()

    print("\n" + "=" * 70)
    print("=== Exercise 4: Variables and Outputs ===")
    print("=" * 70)
    exercise_4()

    print("\n" + "=" * 70)
    print("=== Exercise 5: Terraform Module Design ===")
    print("=" * 70)
    exercise_5()

    print("\nAll exercises completed!")
