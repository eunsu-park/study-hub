# Lesson 6: Terraform Advanced

**Previous**: [Infrastructure as Code](./05_Infrastructure_as_Code.md) | **Next**: [Configuration Management](./07_Configuration_Management.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Design and publish reusable Terraform modules with proper input/output interfaces
2. Use workspaces to manage multiple environments from a single configuration
3. Leverage data sources and provisioners for complex infrastructure scenarios
4. Detect and remediate configuration drift between Terraform state and actual infrastructure
5. Import existing infrastructure into Terraform management using the import command
6. Apply advanced patterns including Terragrunt for DRY configurations and Terratest for infrastructure testing

---

The basics of Terraform -- providers, resources, variables, and state -- get you started, but production-grade infrastructure requires advanced patterns. Modules prevent code duplication across projects. Workspaces manage multiple environments without separate codebases. Drift detection catches unauthorized manual changes. Testing frameworks validate infrastructure code before it touches real cloud resources. This lesson covers the patterns and tools that separate hobby Terraform from production Terraform.

> **Analogy -- Software Libraries:** Terraform modules are like software libraries. Instead of copying the same networking code into every project, you publish a module once and import it everywhere. Just as a well-designed library has a clean API (function signatures, return types), a well-designed module has clear inputs (variables), outputs, and documentation.

## 1. Modules

A Terraform module is a reusable, self-contained package of Terraform configuration.

### Module Structure

```
modules/
└── vpc/
    ├── main.tf          # Resources
    ├── variables.tf     # Input interface
    ├── outputs.tf       # Output interface
    ├── versions.tf      # Provider requirements
    └── README.md        # Usage documentation
```

### Writing a Module

```hcl
# modules/vpc/variables.tf
variable "name" {
  description = "Name prefix for all VPC resources"
  type        = string
}

variable "cidr_block" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24"]
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.10.0/24", "10.0.11.0/24"]
}

variable "enable_nat_gateway" {
  description = "Create a NAT Gateway for private subnet internet access"
  type        = bool
  default     = true
}

variable "tags" {
  description = "Additional tags for all resources"
  type        = map(string)
  default     = {}
}
```

```hcl
# modules/vpc/main.tf
data "aws_availability_zones" "available" {
  state = "available"
}

locals {
  azs = slice(data.aws_availability_zones.available.names, 0,
              min(length(var.public_subnet_cidrs), length(data.aws_availability_zones.available.names)))
}

resource "aws_vpc" "this" {
  cidr_block           = var.cidr_block
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = merge(var.tags, {
    Name = "${var.name}-vpc"
  })
}

resource "aws_internet_gateway" "this" {
  vpc_id = aws_vpc.this.id

  tags = merge(var.tags, {
    Name = "${var.name}-igw"
  })
}

resource "aws_subnet" "public" {
  count = length(var.public_subnet_cidrs)

  vpc_id                  = aws_vpc.this.id
  cidr_block              = var.public_subnet_cidrs[count.index]
  availability_zone       = local.azs[count.index]
  map_public_ip_on_launch = true

  tags = merge(var.tags, {
    Name = "${var.name}-public-${count.index + 1}"
    Tier = "public"
  })
}

resource "aws_subnet" "private" {
  count = length(var.private_subnet_cidrs)

  vpc_id            = aws_vpc.this.id
  cidr_block        = var.private_subnet_cidrs[count.index]
  availability_zone = local.azs[count.index]

  tags = merge(var.tags, {
    Name = "${var.name}-private-${count.index + 1}"
    Tier = "private"
  })
}

resource "aws_eip" "nat" {
  count  = var.enable_nat_gateway ? 1 : 0
  domain = "vpc"

  tags = merge(var.tags, {
    Name = "${var.name}-nat-eip"
  })
}

resource "aws_nat_gateway" "this" {
  count = var.enable_nat_gateway ? 1 : 0

  allocation_id = aws_eip.nat[0].id
  subnet_id     = aws_subnet.public[0].id

  tags = merge(var.tags, {
    Name = "${var.name}-nat-gw"
  })

  depends_on = [aws_internet_gateway.this]
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.this.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.this.id
  }

  tags = merge(var.tags, {
    Name = "${var.name}-public-rt"
  })
}

resource "aws_route_table_association" "public" {
  count = length(var.public_subnet_cidrs)

  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table" "private" {
  count  = var.enable_nat_gateway ? 1 : 0
  vpc_id = aws_vpc.this.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.this[0].id
  }

  tags = merge(var.tags, {
    Name = "${var.name}-private-rt"
  })
}

resource "aws_route_table_association" "private" {
  count = var.enable_nat_gateway ? length(var.private_subnet_cidrs) : 0

  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private[0].id
}
```

```hcl
# modules/vpc/outputs.tf
output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.this.id
}

output "public_subnet_ids" {
  description = "IDs of public subnets"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "IDs of private subnets"
  value       = aws_subnet.private[*].id
}

output "nat_gateway_ip" {
  description = "Elastic IP of the NAT Gateway"
  value       = var.enable_nat_gateway ? aws_eip.nat[0].public_ip : null
}
```

### Using a Module

```hcl
# Root module (project that uses the vpc module)
module "vpc" {
  source = "./modules/vpc"         # Local path

  name                 = "myapp-production"
  cidr_block           = "10.0.0.0/16"
  public_subnet_cidrs  = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  private_subnet_cidrs = ["10.0.10.0/24", "10.0.11.0/24", "10.0.12.0/24"]
  enable_nat_gateway   = true

  tags = {
    Environment = "production"
    Team        = "platform"
  }
}

# Reference module outputs
resource "aws_instance" "web" {
  ami           = data.aws_ami.ubuntu.id
  instance_type = "t3.micro"
  subnet_id     = module.vpc.public_subnet_ids[0]

  tags = {
    Name = "web-server"
  }
}

output "vpc_id" {
  value = module.vpc.vpc_id
}
```

### Module Sources

```hcl
# Local path
module "vpc" {
  source = "./modules/vpc"
}

# Terraform Registry (public or private)
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "5.5.1"
}

# GitHub
module "vpc" {
  source = "github.com/org/terraform-modules//vpc?ref=v1.2.0"
}

# S3 bucket
module "vpc" {
  source = "s3::https://s3-us-east-1.amazonaws.com/my-modules/vpc.zip"
}

# Git over SSH
module "vpc" {
  source = "git::ssh://git@github.com/org/modules.git//vpc?ref=v1.2.0"
}
```

---

## 2. Workspaces

Workspaces allow a single Terraform configuration to manage multiple environments (dev, staging, production).

### Workspace Commands

```bash
# List workspaces (default workspace always exists)
terraform workspace list
# * default

# Create a new workspace
terraform workspace new dev
terraform workspace new staging
terraform workspace new production

# Switch workspace
terraform workspace select dev

# Show current workspace
terraform workspace show
# dev

# Delete a workspace (must switch away first)
terraform workspace select default
terraform workspace delete dev
```

### Using Workspaces in Configuration

```hcl
# Use terraform.workspace to vary configuration by environment
locals {
  environment = terraform.workspace

  instance_types = {
    dev        = "t3.micro"
    staging    = "t3.small"
    production = "t3.large"
  }

  instance_counts = {
    dev        = 1
    staging    = 2
    production = 3
  }

  instance_type = local.instance_types[local.environment]
  instance_count = local.instance_counts[local.environment]
}

resource "aws_instance" "web" {
  count         = local.instance_count
  ami           = data.aws_ami.ubuntu.id
  instance_type = local.instance_type

  tags = {
    Name        = "web-${local.environment}-${count.index + 1}"
    Environment = local.environment
  }
}
```

### Workspaces with Remote Backend

```hcl
# Each workspace gets its own state file in the backend
terraform {
  backend "s3" {
    bucket = "mycompany-terraform-state"
    key    = "myapp/terraform.tfstate"      # Workspaces add prefix automatically
    region = "us-east-1"

    # State files created:
    # env:/dev/myapp/terraform.tfstate
    # env:/staging/myapp/terraform.tfstate
    # env:/production/myapp/terraform.tfstate
  }
}
```

### Workspaces vs Separate Directories

```
Workspaces (same code, different state):
  ✓ DRY -- single codebase
  ✓ Easy to keep environments in sync
  ✗ All environments share exact same structure
  ✗ Risk of applying to wrong workspace
  ✗ Limited per-environment customization

Separate Directories (copy code per environment):
  environments/
  ├── dev/
  │   ├── main.tf
  │   └── terraform.tfvars
  ├── staging/
  │   ├── main.tf
  │   └── terraform.tfvars
  └── production/
      ├── main.tf
      └── terraform.tfvars
  ✓ Full customization per environment
  ✓ Clear separation -- no accidental cross-env applies
  ✗ Code duplication
  ✗ Environments can drift apart
```

---

## 3. for_each and Dynamic Blocks

### for_each

```hcl
# Create multiple resources from a map
variable "subnets" {
  type = map(object({
    cidr_block        = string
    availability_zone = string
    public            = bool
  }))
  default = {
    "public-1" = {
      cidr_block        = "10.0.1.0/24"
      availability_zone = "us-east-1a"
      public            = true
    }
    "public-2" = {
      cidr_block        = "10.0.2.0/24"
      availability_zone = "us-east-1b"
      public            = true
    }
    "private-1" = {
      cidr_block        = "10.0.10.0/24"
      availability_zone = "us-east-1a"
      public            = false
    }
  }
}

resource "aws_subnet" "this" {
  for_each = var.subnets

  vpc_id                  = aws_vpc.main.id
  cidr_block              = each.value.cidr_block
  availability_zone       = each.value.availability_zone
  map_public_ip_on_launch = each.value.public

  tags = {
    Name = "${var.name}-${each.key}"   # each.key = "public-1", "public-2", etc.
    Tier = each.value.public ? "public" : "private"
  }
}

# Reference: aws_subnet.this["public-1"].id
```

### Dynamic Blocks

```hcl
# Generate repeated nested blocks dynamically
variable "ingress_rules" {
  type = list(object({
    port        = number
    protocol    = string
    cidr_blocks = list(string)
    description = string
  }))
  default = [
    { port = 80, protocol = "tcp", cidr_blocks = ["0.0.0.0/0"], description = "HTTP" },
    { port = 443, protocol = "tcp", cidr_blocks = ["0.0.0.0/0"], description = "HTTPS" },
    { port = 22, protocol = "tcp", cidr_blocks = ["10.0.0.0/8"], description = "SSH internal" },
  ]
}

resource "aws_security_group" "web" {
  name   = "${var.name}-web-sg"
  vpc_id = aws_vpc.main.id

  dynamic "ingress" {
    for_each = var.ingress_rules
    content {
      from_port   = ingress.value.port
      to_port     = ingress.value.port
      protocol    = ingress.value.protocol
      cidr_blocks = ingress.value.cidr_blocks
      description = ingress.value.description
    }
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
```

---

## 4. Drift Detection

Configuration drift occurs when actual infrastructure diverges from the Terraform state, usually due to manual changes.

### Detecting Drift

```bash
# The plan command detects drift automatically
terraform plan

# If someone manually changed the instance type from t3.micro to t3.small:
# ~ resource "aws_instance" "web" {
#     ~ instance_type = "t3.small" -> "t3.micro"   # Terraform will revert it
#   }

# Refresh state to match reality (without changing infrastructure)
terraform apply -refresh-only

# This updates the state file to reflect actual infrastructure
# Useful when you WANT to accept manual changes
```

### Drift Detection in CI/CD

```yaml
# .github/workflows/drift-detection.yml
name: Terraform Drift Detection

on:
  schedule:
    - cron: '0 8 * * *'        # Daily at 8 AM UTC

jobs:
  detect-drift:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: 1.7.0

      - name: Terraform Init
        run: terraform init

      - name: Detect Drift
        id: plan
        run: terraform plan -detailed-exitcode -out=drift.plan
        continue-on-error: true
        # Exit code 0 = no changes (no drift)
        # Exit code 1 = error
        # Exit code 2 = changes detected (drift!)

      - name: Notify on Drift
        if: steps.plan.outcome == 'failure'
        run: |
          curl -X POST "${{ secrets.SLACK_WEBHOOK }}" \
            -d '{"text": "Terraform drift detected! Review the plan."}'
```

---

## 5. Import

Import brings existing infrastructure under Terraform management.

### Import Workflow

```bash
# Step 1: Write the resource configuration (empty or partial)
# Add to main.tf:
# resource "aws_instance" "legacy_server" {
#   # Configuration will be filled after import
# }

# Step 2: Import the resource
terraform import aws_instance.legacy_server i-0abc123def456789

# Step 3: Run terraform plan to see what config is needed
terraform plan
# Terraform shows what attributes need to be set

# Step 4: Fill in the configuration to match the imported resource
# Update main.tf with the actual values from plan output

# Step 5: Verify no changes needed
terraform plan
# No changes. Infrastructure is up-to-date.
```

### Import Block (Terraform 1.5+)

```hcl
# Declarative import -- preferred for Terraform 1.5+
import {
  to = aws_instance.legacy_server
  id = "i-0abc123def456789"
}

# Generate configuration automatically
# terraform plan -generate-config-out=generated.tf
# This creates a generated.tf with the full resource configuration
```

### Bulk Import with a Script

```bash
#!/bin/bash
# import-existing.sh -- import multiple resources

set -euo pipefail

# List of resources to import: "resource_address cloud_id"
IMPORTS=(
  "aws_instance.web-1 i-0abc123def456789"
  "aws_instance.web-2 i-0def456abc789012"
  "aws_security_group.web sg-0123456789abcdef0"
  "aws_vpc.main vpc-0fedcba9876543210"
)

for import in "${IMPORTS[@]}"; do
  read -r address id <<< "$import"
  echo "Importing $address ($id)..."
  terraform import "$address" "$id"
done

echo "Import complete. Run 'terraform plan' to verify."
```

---

## 6. Terragrunt

Terragrunt is a thin wrapper around Terraform that provides extra tools for keeping configurations DRY and managing multiple environments.

### Terragrunt Project Structure

```
infrastructure/
├── terragrunt.hcl                    # Root config (backend, provider defaults)
├── modules/                          # Reusable Terraform modules
│   ├── vpc/
│   ├── ecs/
│   └── rds/
└── environments/
    ├── dev/
    │   ├── terragrunt.hcl            # Environment-level config
    │   ├── vpc/
    │   │   └── terragrunt.hcl        # Component-level config
    │   ├── ecs/
    │   │   └── terragrunt.hcl
    │   └── rds/
    │       └── terragrunt.hcl
    ├── staging/
    │   ├── terragrunt.hcl
    │   ├── vpc/
    │   │   └── terragrunt.hcl
    │   ...
    └── production/
        ├── terragrunt.hcl
        ├── vpc/
        │   └── terragrunt.hcl
        ...
```

### Root terragrunt.hcl

```hcl
# infrastructure/terragrunt.hcl
# Shared configuration inherited by all children

remote_state {
  backend = "s3"
  generate = {
    path      = "backend.tf"
    if_exists = "overwrite_terragrunt"
  }
  config = {
    bucket         = "mycompany-terraform-state"
    key            = "${path_relative_to_include()}/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-locks"
  }
}

generate "provider" {
  path      = "provider.tf"
  if_exists = "overwrite_terragrunt"
  contents  = <<EOF
provider "aws" {
  region = var.aws_region
  default_tags {
    tags = {
      ManagedBy = "terragrunt"
    }
  }
}
EOF
}
```

### Component terragrunt.hcl

```hcl
# environments/dev/vpc/terragrunt.hcl
include "root" {
  path = find_in_parent_folders()     # Inherit root config
}

terraform {
  source = "../../../modules/vpc"      # Point to module
}

inputs = {
  name                 = "dev"
  cidr_block           = "10.0.0.0/16"
  public_subnet_cidrs  = ["10.0.1.0/24", "10.0.2.0/24"]
  private_subnet_cidrs = ["10.0.10.0/24", "10.0.11.0/24"]
  enable_nat_gateway   = false          # Save cost in dev
  aws_region           = "us-east-1"
}
```

### Terragrunt Commands

```bash
# Apply a single component
cd environments/dev/vpc
terragrunt apply

# Apply all components in an environment
cd environments/dev
terragrunt run-all apply

# Plan all components
terragrunt run-all plan

# Destroy all components (reverse dependency order)
terragrunt run-all destroy

# Show dependency graph
terragrunt graph-dependencies
```

---

## 7. Testing with Terratest

Terratest is a Go library for testing Terraform code by deploying real infrastructure, validating it, and then destroying it.

### Basic Test Structure

```go
// test/vpc_test.go
package test

import (
    "testing"

    "github.com/gruntwork-io/terratest/modules/aws"
    "github.com/gruntwork-io/terratest/modules/terraform"
    "github.com/stretchr/testify/assert"
)

func TestVpcModule(t *testing.T) {
    t.Parallel()

    // Define Terraform options
    terraformOptions := &terraform.Options{
        TerraformDir: "../modules/vpc",

        Vars: map[string]interface{}{
            "name":                 "test-vpc",
            "cidr_block":           "10.99.0.0/16",
            "public_subnet_cidrs":  []string{"10.99.1.0/24"},
            "private_subnet_cidrs": []string{"10.99.10.0/24"},
            "enable_nat_gateway":   false,
        },

        EnvVars: map[string]string{
            "AWS_DEFAULT_REGION": "us-east-1",
        },
    }

    // Destroy resources after test completes
    defer terraform.Destroy(t, terraformOptions)

    // Deploy infrastructure
    terraform.InitAndApply(t, terraformOptions)

    // Validate outputs
    vpcID := terraform.Output(t, terraformOptions, "vpc_id")
    assert.NotEmpty(t, vpcID)

    publicSubnetIDs := terraform.OutputList(t, terraformOptions, "public_subnet_ids")
    assert.Equal(t, 1, len(publicSubnetIDs))

    // Validate using AWS SDK
    vpc := aws.GetVpcById(t, vpcID, "us-east-1")
    assert.Equal(t, "10.99.0.0/16", *vpc.CidrBlock)
}
```

### Running Tests

```bash
# Run all tests
cd test
go test -v -timeout 30m

# Run a specific test
go test -v -timeout 30m -run TestVpcModule

# Run with shorter timeout for faster feedback
go test -v -timeout 10m -run TestVpcModule
```

### Terraform Validate and Plan Tests (No Deploy)

```bash
# Fast validation without deploying (no cloud costs)
# Use in CI for every PR

terraform init -backend=false        # Skip backend initialization
terraform validate                   # Syntax and consistency check
terraform plan -input=false          # Plan without apply
```

---

## 8. Best Practices Summary

### Module Design

```
Module Best Practices:
──────────────────────
✓ One module = one logical component (VPC, ECS cluster, RDS instance)
✓ Expose necessary variables, hide implementation details
✓ Provide sensible defaults for optional variables
✓ Use validation blocks for input constraints
✓ Output everything consumers might need
✓ Version your modules with Git tags
✓ Document with README.md and examples/
✗ Don't make modules too granular (one resource per module)
✗ Don't hardcode values that should be variables
✗ Don't use provisioners in modules (prefer user_data or cloud-init)
```

### State Management

```
State Best Practices:
─────────────────────
✓ Always use remote state for team projects
✓ Enable state locking (DynamoDB for S3 backend)
✓ Enable state encryption at rest
✓ Enable state file versioning for rollback
✓ Never manually edit the state file
✓ Use separate state per environment
✗ Never commit terraform.tfstate to Git
✗ Never share state files via email or chat
```

---

## Exercises

### Exercise 1: Create a Reusable Module

Build a Terraform module for an application load balancer (ALB):
1. Accept inputs: name, VPC ID, subnet IDs, health check path, target port
2. Create the ALB, target group, listener (HTTP and HTTPS redirect), and security group
3. Output the ALB DNS name, ARN, and target group ARN
4. Use the module from a root configuration that also creates EC2 instances registered to the target group
5. Add input validation for the health check path (must start with "/")

### Exercise 2: Multi-Environment with Workspaces

Set up a Terraform configuration that uses workspaces:
1. Create dev, staging, and production workspaces
2. Configure different instance types and counts per workspace
3. Use different CIDR blocks per workspace to avoid conflicts
4. Show how to switch between workspaces and apply
5. Discuss what safeguards you would add to prevent accidental production changes

### Exercise 3: Import Legacy Infrastructure

You have an existing AWS setup (created via console) with:
- 1 VPC, 2 subnets, 1 security group, 2 EC2 instances
1. Write the Terraform configuration for these resources
2. Write the import commands for each resource
3. Describe how you would verify the import was successful
4. What would you do if `terraform plan` shows changes after import?

### Exercise 4: Drift Detection Pipeline

Design a CI/CD pipeline for Terraform that includes:
1. On PR: run `terraform fmt`, `terraform validate`, `terraform plan`
2. On merge to main: run `terraform apply` automatically for dev
3. Scheduled daily: run drift detection with Slack notification
4. Write the complete GitHub Actions workflow YAML
5. Include appropriate safeguards for the production environment

---

**Previous**: [Infrastructure as Code](./05_Infrastructure_as_Code.md) | [Overview](00_Overview.md) | **Next**: [Configuration Management](./07_Configuration_Management.md)

**License**: CC BY-NC 4.0
