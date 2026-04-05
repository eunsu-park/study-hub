# Lesson 5: Infrastructure as Code

**Previous**: [GitHub Actions Deep Dive](./04_GitHub_Actions_Deep_Dive.md) | **Next**: [Terraform Advanced](./06_Terraform_Advanced.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the principles of Infrastructure as Code (IaC) and why it replaces manual infrastructure provisioning
2. Write Terraform configurations using HCL syntax including providers, resources, variables, and outputs
3. Execute the Terraform workflow: init, plan, apply, and destroy
4. Manage Terraform state and understand why remote state backends are essential for team collaboration
5. Organize Terraform code using input variables, output values, and local values for maintainability

---

Infrastructure as Code (IaC) is the practice of managing and provisioning infrastructure through machine-readable configuration files rather than through manual processes or interactive tools. Before IaC, provisioning a server meant clicking through a cloud console, running ad-hoc scripts, or filing tickets for an operations team. These approaches are slow, error-prone, and impossible to reproduce reliably. IaC brings the same rigor to infrastructure that version control brought to application code: every change is tracked, reviewed, tested, and reproducible.

> **Analogy -- Architecture Blueprint:** Manual infrastructure management is like building a house by giving verbal instructions to a construction crew. Every house comes out slightly different, and replicating a house from memory is unreliable. IaC is the architectural blueprint: precise, version-controlled, and usable by any construction crew to produce identical results.

## 1. Why Infrastructure as Code?

### The Problem with Manual Infrastructure

```
Manual Provisioning Problems:
─────────────────────────────
1. Snowflake servers
   "This server was set up 3 years ago by someone who left the company.
    Nobody knows all the configuration changes that were made."

2. Configuration drift
   "Production and staging are supposed to be identical, but production
    has a firewall rule nobody documented."

3. Slow provisioning
   "We need a new environment. File a ticket, wait 2 weeks."

4. No audit trail
   "Who changed the database security group? When? Why?"

5. Disaster recovery
   "Can we recreate our entire infrastructure from scratch? Probably not."
```

### IaC Benefits

| Benefit | Description |
|---------|-------------|
| **Reproducibility** | Create identical environments every time |
| **Version control** | Track every infrastructure change in Git |
| **Code review** | Review infrastructure changes via pull requests |
| **Self-service** | Developers provision their own environments |
| **Speed** | Create complex environments in minutes |
| **Documentation** | The code IS the documentation |
| **Disaster recovery** | Recreate entire infrastructure from code |
| **Cost control** | Destroy unused environments, audit resources |

---

## 2. IaC Approaches

### Declarative vs Imperative

```
Declarative (What):
  "I want 3 web servers behind a load balancer."
  The tool figures out HOW to achieve this state.
  Tools: Terraform, CloudFormation, Pulumi (optional)

Imperative (How):
  "Create server 1. Create server 2. Create server 3.
   Create a load balancer. Register all three servers."
  You specify exact steps.
  Tools: Bash scripts, AWS CLI, Ansible (partially)
```

```hcl
# Declarative (Terraform) -- describe desired state
resource "aws_instance" "web" {
  count         = 3
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t3.micro"
  tags = {
    Name = "web-${count.index + 1}"
  }
}

# If you change count from 3 to 5, Terraform creates 2 MORE servers.
# If you change count from 3 to 1, Terraform destroys 2 servers.
# Terraform handles the diff automatically.
```

```bash
# Imperative (bash script) -- describe exact steps
#!/bin/bash
for i in 1 2 3; do
  aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \
    --instance-type t3.micro \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=web-$i}]"
done

# If you want 5 servers now, you need a NEW script.
# If you want 1 server, you need ANOTHER script to delete 2.
# You must track what exists and compute the diff yourself.
```

### Mutable vs Immutable Infrastructure

```
Mutable Infrastructure:
  Create server ──▶ SSH in ──▶ Update packages ──▶ Modify config
  (Server changes in-place over time -- configuration drift happens)

Immutable Infrastructure:
  Build new image ──▶ Deploy new servers ──▶ Destroy old servers
  (Servers are never modified -- replaced entirely)
  Tools: Docker, Packer, AMI builders
```

---

## 3. IaC Tool Landscape

| Tool | Type | Language | Provider | State |
|------|------|----------|----------|-------|
| **Terraform** | Declarative | HCL | Multi-cloud | Explicit state file |
| **OpenTofu** | Declarative | HCL | Multi-cloud | Explicit state file |
| **AWS CloudFormation** | Declarative | JSON/YAML | AWS only | Managed by AWS |
| **Pulumi** | Declarative | Python/TS/Go | Multi-cloud | Managed or self-hosted |
| **Ansible** | Imperative/Declarative | YAML | Multi-cloud | Stateless (agentless) |
| **AWS CDK** | Imperative | Python/TS/Java | AWS only | CloudFormation stack |
| **Crossplane** | Declarative | YAML (K8s CRDs) | Multi-cloud | Kubernetes state |

This lesson focuses on **Terraform** because it is the most widely adopted multi-cloud IaC tool.

---

## 4. Terraform Fundamentals

### Installation

```bash
# macOS
brew install terraform

# Linux (Ubuntu/Debian)
wget -O- https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
sudo apt update && sudo apt install terraform

# Verify installation
terraform --version
# Terraform v1.7.x
```

### HCL (HashiCorp Configuration Language)

```hcl
# HCL is Terraform's configuration language
# It is human-readable and machine-parseable

# Block syntax: <BLOCK_TYPE> "<BLOCK_LABEL>" "<BLOCK_NAME>" { ... }
resource "aws_instance" "example" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t3.micro"
}

# Arguments: key = value
instance_type = "t3.micro"

# Strings, numbers, booleans
name    = "my-instance"
count   = 3
enabled = true

# Lists
tags = ["web", "production"]

# Maps
labels = {
  environment = "production"
  team        = "platform"
}

# Multi-line strings (heredoc)
user_data = <<-EOF
  #!/bin/bash
  apt-get update
  apt-get install -y nginx
EOF

# String interpolation
name = "web-${var.environment}"

# Comments
# This is a single-line comment
/* This is a
   multi-line comment */
```

---

## 5. Terraform Configuration Structure

### Project Layout

```
terraform-project/
├── main.tf           # Primary resource definitions
├── variables.tf      # Input variable declarations
├── outputs.tf        # Output value declarations
├── providers.tf      # Provider configuration
├── terraform.tfvars  # Variable values (often gitignored)
├── versions.tf       # Required provider versions
└── backend.tf        # State backend configuration
```

### Provider Configuration

```hcl
# providers.tf -- configure cloud providers

terraform {
  required_version = ">= 1.7.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"       # >= 5.0, < 6.0
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "my-app"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

# Multiple provider instances with aliases
provider "aws" {
  alias  = "us_west"
  region = "us-west-2"
}
```

### Resources

```hcl
# main.tf -- define infrastructure resources

# VPC
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "${var.project}-vpc"
  }
}

# Subnet
resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.main.id        # Reference another resource
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "${var.aws_region}a"
  map_public_ip_on_launch = true

  tags = {
    Name = "${var.project}-public-subnet"
  }
}

# Security Group
resource "aws_security_group" "web" {
  name        = "${var.project}-web-sg"
  description = "Allow HTTP and HTTPS traffic"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"                # All traffic
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# EC2 Instance
resource "aws_instance" "web" {
  ami                    = var.ami_id
  instance_type          = var.instance_type
  subnet_id              = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.web.id]

  user_data = <<-EOF
    #!/bin/bash
    apt-get update
    apt-get install -y nginx
    systemctl start nginx
    systemctl enable nginx
    echo "<h1>Hello from Terraform</h1>" > /var/www/html/index.html
  EOF

  tags = {
    Name = "${var.project}-web-server"
  }
}
```

### Variables

```hcl
# variables.tf -- declare input variables

variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string

  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "Environment must be dev, staging, or production."
  }
}

variable "project" {
  description = "Project name used for resource naming"
  type        = string
  default     = "myapp"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.micro"
}

variable "ami_id" {
  description = "AMI ID for EC2 instances"
  type        = string
}

variable "allowed_cidr_blocks" {
  description = "List of CIDR blocks allowed to access the instance"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "extra_tags" {
  description = "Additional tags for resources"
  type        = map(string)
  default     = {}
}

# Sensitive variable (masked in output)
variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}
```

```hcl
# terraform.tfvars -- provide variable values
aws_region    = "us-east-1"
environment   = "dev"
project       = "myapp"
instance_type = "t3.small"
ami_id        = "ami-0c55b159cbfafe1f0"

allowed_cidr_blocks = [
  "10.0.0.0/8",
  "172.16.0.0/12",
]

extra_tags = {
  Team   = "platform"
  CostCenter = "engineering"
}
```

### Outputs

```hcl
# outputs.tf -- expose values after apply

output "instance_id" {
  description = "ID of the EC2 instance"
  value       = aws_instance.web.id
}

output "instance_public_ip" {
  description = "Public IP address of the web server"
  value       = aws_instance.web.public_ip
}

output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "web_url" {
  description = "URL of the web server"
  value       = "http://${aws_instance.web.public_ip}"
}

# Sensitive output (hidden in CLI unless explicitly queried)
output "db_connection_string" {
  description = "Database connection string"
  value       = "postgresql://admin:${var.db_password}@${aws_instance.web.private_ip}:5432/mydb"
  sensitive   = true
}
```

---

## 6. Terraform Workflow

### The Core Commands

```
terraform init ──▶ terraform plan ──▶ terraform apply ──▶ terraform destroy
     │                    │                   │                    │
  Download            Show what           Execute the         Tear down all
  providers           will change         planned changes     resources
  Initialize          (dry run)           (asks for           (asks for
  backend                                 confirmation)       confirmation)
```

### Step-by-Step

```bash
# 1. Initialize -- download providers and modules, initialize backend
terraform init

# Output:
# Initializing the backend...
# Initializing provider plugins...
# - Finding hashicorp/aws versions matching "~> 5.0"...
# - Installing hashicorp/aws v5.31.0...
# Terraform has been successfully initialized!

# 2. Validate -- check syntax and internal consistency
terraform validate

# Output:
# Success! The configuration is valid.

# 3. Format -- auto-format HCL files
terraform fmt -recursive

# 4. Plan -- preview changes (dry run)
terraform plan -out=tfplan

# Output shows:
# + resource "aws_instance" "web" {   (+ means CREATE)
#     ami           = "ami-0c55..."
#     instance_type = "t3.micro"
#   }
#
# Plan: 4 to add, 0 to change, 0 to destroy.

# 5. Apply -- execute the plan
terraform apply tfplan

# Or apply directly (will show plan and ask for confirmation):
terraform apply

# 6. Show current state
terraform show

# 7. List resources in state
terraform state list

# Output:
# aws_instance.web
# aws_security_group.web
# aws_subnet.public
# aws_vpc.main

# 8. Destroy -- remove all managed resources
terraform destroy

# Or destroy specific resources:
terraform destroy -target=aws_instance.web
```

### Plan Output Symbols

```
+ create           A new resource will be created
- destroy          An existing resource will be destroyed
~ update in-place  An existing resource will be modified
-/+ replace        A resource will be destroyed and recreated
<= read            A data source will be read
```

---

## 7. State Management

Terraform state is a JSON file that maps your configuration to real-world resources. It is the single most critical artifact in a Terraform project.

### What State Tracks

```json
{
  "version": 4,
  "terraform_version": "1.7.0",
  "resources": [
    {
      "type": "aws_instance",
      "name": "web",
      "provider": "provider[\"registry.terraform.io/hashicorp/aws\"]",
      "instances": [
        {
          "attributes": {
            "id": "i-0abc123def456789",
            "ami": "ami-0c55b159cbfafe1f0",
            "instance_type": "t3.micro",
            "public_ip": "54.123.45.67"
          }
        }
      ]
    }
  ]
}
```

### Local vs Remote State

```
Local State (terraform.tfstate):
  ✗ Single developer only -- no team collaboration
  ✗ No locking -- concurrent applies corrupt state
  ✗ State file may contain secrets in plaintext
  ✗ Lost laptop = lost state = orphaned resources

Remote State (S3, GCS, Azure Blob, Terraform Cloud):
  ✓ Team collaboration -- shared state
  ✓ Locking -- prevents concurrent modifications
  ✓ Encryption at rest
  ✓ Versioning -- rollback to previous state
  ✓ Access control
```

### Remote Backend Configuration (AWS S3)

```hcl
# backend.tf
terraform {
  backend "s3" {
    bucket         = "mycompany-terraform-state"
    key            = "projects/myapp/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-locks"      # State locking via DynamoDB
  }
}
```

```hcl
# Bootstrap: create the S3 bucket and DynamoDB table for state storage
# (This is a chicken-and-egg problem -- use a separate config or create manually)

resource "aws_s3_bucket" "terraform_state" {
  bucket = "mycompany-terraform-state"

  lifecycle {
    prevent_destroy = true    # Prevent accidental deletion
  }
}

resource "aws_s3_bucket_versioning" "terraform_state" {
  bucket = aws_s3_bucket.terraform_state.id

  versioning_configuration {
    status = "Enabled"        # Keep history of all state versions
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "terraform_state" {
  bucket = aws_s3_bucket.terraform_state.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
  }
}

resource "aws_dynamodb_table" "terraform_locks" {
  name         = "terraform-locks"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "LockID"

  attribute {
    name = "LockID"
    type = "S"
  }
}
```

### State Commands

```bash
# List all resources in state
terraform state list

# Show detailed state for a specific resource
terraform state show aws_instance.web

# Move a resource (rename without destroy/recreate)
terraform state mv aws_instance.web aws_instance.application

# Remove a resource from state (Terraform forgets it, resource still exists)
terraform state rm aws_instance.web

# Pull remote state to local file (for inspection)
terraform state pull > state.json

# Import an existing resource into state
terraform import aws_instance.web i-0abc123def456789
```

---

## 8. Locals and Data Sources

### Local Values

```hcl
# Use locals to compute values and avoid repetition

locals {
  name_prefix = "${var.project}-${var.environment}"

  common_tags = {
    Project     = var.project
    Environment = var.environment
    ManagedBy   = "terraform"
  }

  # Conditional logic
  instance_type = var.environment == "production" ? "t3.large" : "t3.micro"

  # Computed from other resources
  public_subnet_ids = aws_subnet.public[*].id
}

resource "aws_instance" "web" {
  ami           = data.aws_ami.ubuntu.id
  instance_type = local.instance_type       # Use local value

  tags = merge(local.common_tags, {
    Name = "${local.name_prefix}-web"
  })
}
```

### Data Sources

Data sources let you query existing resources or external data.

```hcl
# Look up the latest Ubuntu AMI
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]    # Canonical (Ubuntu publisher)

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# Look up existing VPC by tag
data "aws_vpc" "existing" {
  filter {
    name   = "tag:Name"
    values = ["main-vpc"]
  }
}

# Look up current AWS account info
data "aws_caller_identity" "current" {}

# Look up available AZs in the region
data "aws_availability_zones" "available" {
  state = "available"
}

# Use data sources in resources
resource "aws_instance" "web" {
  ami               = data.aws_ami.ubuntu.id
  instance_type     = "t3.micro"
  availability_zone = data.aws_availability_zones.available.names[0]

  tags = {
    Account = data.aws_caller_identity.current.account_id
  }
}
```

---

## 9. Resource Dependencies and Lifecycle

### Implicit Dependencies

```hcl
# Terraform automatically infers dependencies from references
resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"
}

resource "aws_subnet" "public" {
  vpc_id     = aws_vpc.main.id    # Implicit dependency on VPC
  cidr_block = "10.0.1.0/24"
}
# Terraform creates VPC first, then subnet (because subnet references VPC ID)
```

### Explicit Dependencies

```hcl
# Use depends_on when there is no reference-based dependency
resource "aws_instance" "web" {
  ami           = data.aws_ami.ubuntu.id
  instance_type = "t3.micro"

  depends_on = [
    aws_iam_role_policy.web_policy   # Ensure IAM policy exists first
  ]
}
```

### Lifecycle Rules

```hcl
resource "aws_instance" "web" {
  ami           = data.aws_ami.ubuntu.id
  instance_type = "t3.micro"

  lifecycle {
    # Create new resource before destroying old one (zero-downtime)
    create_before_destroy = true

    # Prevent accidental deletion of critical resources
    prevent_destroy = true

    # Ignore changes made outside of Terraform
    ignore_changes = [
      tags["LastModified"],   # Ignore tag changes made by automation
      user_data,              # Ignore user_data changes (would force replacement)
    ]
  }
}
```

---

## Exercises

### Exercise 1: First Terraform Project

Create a Terraform configuration that provisions a simple web server setup:
1. Create `providers.tf` with AWS provider configuration
2. Create `variables.tf` with variables for region, instance type, and project name
3. Create `main.tf` with a VPC, public subnet, security group (ports 80, 443, 22), and an EC2 instance with nginx user_data
4. Create `outputs.tf` that shows the instance's public IP and the URL
5. Run `terraform init`, `terraform validate`, `terraform plan`
6. Review the plan output and explain what each resource does

### Exercise 2: State Management

Starting from Exercise 1:
1. Create an S3 backend configuration for remote state
2. Explain what happens if two developers run `terraform apply` simultaneously without state locking
3. Run `terraform state list` and `terraform state show` for one resource
4. Describe a scenario where you would use `terraform state mv` and `terraform state rm`

### Exercise 3: Variables and Environments

Extend the Exercise 1 configuration to support multiple environments:
1. Use variables to control instance type (micro for dev, large for production)
2. Create separate `.tfvars` files for dev and production
3. Add a validation rule that restricts instance types to an allowed list
4. Add a sensitive variable for a database password
5. Show how to apply with environment-specific variables: `terraform apply -var-file=dev.tfvars`

### Exercise 4: Data Sources

Add data sources to your configuration:
1. Use `aws_ami` data source to find the latest Amazon Linux 2023 AMI instead of hardcoding
2. Use `aws_availability_zones` to distribute resources across AZs
3. Use `aws_caller_identity` to tag resources with the account ID
4. Explain the difference between a `resource` and a `data` block

---

**Previous**: [GitHub Actions Deep Dive](./04_GitHub_Actions_Deep_Dive.md) | [Overview](00_Overview.md) | **Next**: [Terraform Advanced](./06_Terraform_Advanced.md)

**License**: CC BY-NC 4.0
