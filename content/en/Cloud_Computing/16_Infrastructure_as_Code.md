# Infrastructure as Code (Terraform)

**Previous**: [CLI and SDK](./15_CLI_and_SDK.md) | **Next**: [Monitoring, Logging, and Cost Management](./17_Monitoring_Logging_Cost.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the principles of Infrastructure as Code and its advantages over manual provisioning
2. Compare declarative (Terraform, CloudFormation) and procedural (Ansible) IaC approaches
3. Write Terraform configurations using HCL to define cloud resources
4. Execute the Terraform workflow: init, plan, apply, and destroy
5. Manage Terraform state files and configure remote backends for team collaboration
6. Use variables, modules, and outputs to create reusable, maintainable infrastructure code
7. Implement a multi-resource deployment with proper dependency management

---

Manually clicking through a console to create resources is slow, error-prone, and impossible to reproduce consistently. Infrastructure as Code treats your cloud environment like software -- version-controlled, peer-reviewed, tested, and deployed through automated pipelines. It is the foundation of reliable, auditable, and scalable cloud operations.

> **Analogy — Blueprints, Not Bricklaying**: Manually configuring cloud resources is like building a house by giving verbal instructions to each worker. Infrastructure as Code is the architect's blueprint — a precise, version-controlled document that anyone can use to reproduce the exact same building. If a fire destroys the house, you don't rely on memory; you hand the blueprint to a new crew and rebuild identically.

## 1. IaC Overview

### 1.1 What is Infrastructure as Code?

IaC is the practice of defining and managing infrastructure through code.

**Benefits:**
- Version control (Git)
- Reproducibility
- Automation
- Documentation
- Collaboration

### 1.2 IaC Tool Comparison

| Tool | Type | Language | Multi-Cloud |
|------|------|------|-------------|
| **Terraform** | Declarative | HCL | ✅ |
| CloudFormation | Declarative | JSON/YAML | AWS only |
| Deployment Manager | Declarative | YAML/Jinja | GCP only |
| Pulumi | Declarative | Python/TS etc | ✅ |
| Ansible | Procedural | YAML | ✅ |

---

## 2. Terraform Basics

### 2.1 Installation

```bash
# macOS
brew install terraform

# Linux
wget https://releases.hashicorp.com/terraform/1.6.0/terraform_1.6.0_linux_amd64.zip
unzip terraform_1.6.0_linux_amd64.zip
sudo mv terraform /usr/local/bin/

# Check version
terraform version
```

### 2.2 Core Concepts

```
┌─────────────────────────────────────────────────────────────┐
│  Terraform Workflow                                          │
│                                                             │
│  1. Write    → Write .tf files                              │
│  2. Init     → terraform init (download providers)          │
│  3. Plan     → terraform plan (preview changes)             │
│  4. Apply    → terraform apply (apply infrastructure)       │
│  5. Destroy  → terraform destroy (destroy infrastructure)   │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 HCL Syntax

```hcl
# Provider configuration
provider "aws" {
  region = "ap-northeast-2"
}

# Resource definition
resource "aws_instance" "web" {
  ami           = "ami-12345678"
  instance_type = "t3.micro"

  tags = {
    Name = "WebServer"
  }
}

# Variables
variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.micro"
}

# Outputs
output "public_ip" {
  value = aws_instance.web.public_ip
}

# Local values
locals {
  environment = "production"
  common_tags = {
    Environment = local.environment
    ManagedBy   = "Terraform"
  }
}

# Data source (reference existing resources)
data "aws_ami" "amazon_linux" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }
}
```

---

## 3. AWS Infrastructure Setup

### 3.1 VPC + EC2 Example

```hcl
# main.tf

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.region
}

# VPC
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true

  tags = {
    Name = "${var.project}-vpc"
  }
}

# Public subnet
resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.${count.index + 1}.0/24"
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = "${var.project}-public-${count.index + 1}"
  }
}

# Internet gateway
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "${var.project}-igw"
  }
}

# Route table
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name = "${var.project}-public-rt"
  }
}

resource "aws_route_table_association" "public" {
  count          = 2
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

# Security group
resource "aws_security_group" "web" {
  name        = "${var.project}-web-sg"
  description = "Web server security group"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.ssh_allowed_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# EC2 instance
resource "aws_instance" "web" {
  ami                    = data.aws_ami.amazon_linux.id
  instance_type          = var.instance_type
  subnet_id              = aws_subnet.public[0].id
  vpc_security_group_ids = [aws_security_group.web.id]
  key_name               = var.key_name

  user_data = <<-EOF
    #!/bin/bash
    dnf update -y
    dnf install -y nginx
    systemctl start nginx
    systemctl enable nginx
    echo "<h1>Hello from Terraform!</h1>" > /usr/share/nginx/html/index.html
  EOF

  tags = {
    Name = "${var.project}-web"
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_ami" "amazon_linux" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }
}
```

```hcl
# variables.tf

variable "region" {
  description = "AWS region"
  type        = string
  default     = "ap-northeast-2"
}

variable "project" {
  description = "Project name"
  type        = string
  default     = "myapp"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.micro"
}

variable "key_name" {
  description = "SSH key pair name"
  type        = string
}

variable "ssh_allowed_cidr" {
  description = "CIDR block allowed for SSH"
  type        = string
  default     = "0.0.0.0/0"
}
```

```hcl
# outputs.tf

output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.main.id
}

output "public_ip" {
  description = "Web server public IP"
  value       = aws_instance.web.public_ip
}

output "website_url" {
  description = "Website URL"
  value       = "http://${aws_instance.web.public_ip}"
}
```

---

## 4. GCP Infrastructure Setup

### 4.1 VPC + Compute Engine Example

```hcl
# main.tf

terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# VPC
resource "google_compute_network" "main" {
  name                    = "${var.name_prefix}-vpc"
  auto_create_subnetworks = false
}

# Subnet
resource "google_compute_subnetwork" "public" {
  name          = "${var.name_prefix}-subnet"
  ip_cidr_range = "10.0.1.0/24"
  region        = var.region
  network       = google_compute_network.main.id
}

# Firewall rule - HTTP
resource "google_compute_firewall" "http" {
  name    = "${var.name_prefix}-allow-http"
  network = google_compute_network.main.name

  allow {
    protocol = "tcp"
    ports    = ["80", "443"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["http-server"]
}

# Firewall rule - SSH
resource "google_compute_firewall" "ssh" {
  name    = "${var.name_prefix}-allow-ssh"
  network = google_compute_network.main.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = [var.ssh_allowed_cidr]
  target_tags   = ["ssh-server"]
}

# Compute Engine instance
resource "google_compute_instance" "web" {
  name         = "${var.name_prefix}-web"
  machine_type = var.machine_type
  zone         = "${var.region}-a"

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = 20
    }
  }

  network_interface {
    network    = google_compute_network.main.name
    subnetwork = google_compute_subnetwork.public.name

    access_config {
      // Assign external IP
    }
  }

  metadata_startup_script = <<-EOF
    #!/bin/bash
    apt-get update
    apt-get install -y nginx
    echo "<h1>Hello from Terraform on GCP!</h1>" > /var/www/html/index.html
  EOF

  tags = ["http-server", "ssh-server"]

  labels = {
    environment = var.environment
  }
}
```

```hcl
# variables.tf

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "asia-northeast3"
}

variable "name_prefix" {
  description = "Resource name prefix"
  type        = string
  default     = "myapp"
}

variable "machine_type" {
  description = "Compute Engine machine type"
  type        = string
  default     = "e2-micro"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

variable "ssh_allowed_cidr" {
  description = "CIDR allowed for SSH"
  type        = string
  default     = "0.0.0.0/0"
}
```

```hcl
# outputs.tf

output "instance_ip" {
  description = "Instance external IP"
  value       = google_compute_instance.web.network_interface[0].access_config[0].nat_ip
}

output "website_url" {
  description = "Website URL"
  value       = "http://${google_compute_instance.web.network_interface[0].access_config[0].nat_ip}"
}
```

---

## 5. State Management

### 5.1 Remote State Backend

**AWS S3 Backend:**
```hcl
terraform {
  backend "s3" {
    bucket         = "my-terraform-state"
    key            = "prod/terraform.tfstate"
    region         = "ap-northeast-2"
    encrypt        = true
    dynamodb_table = "terraform-locks"  # State locking
  }
}
```

**GCP Cloud Storage Backend:**
```hcl
terraform {
  backend "gcs" {
    bucket = "my-terraform-state"
    prefix = "prod/terraform.tfstate"
  }
}
```

### 5.2 State Commands

```bash
# List state
terraform state list

# Show state
terraform state show aws_instance.web

# Remove resource from state (keep actual resource)
terraform state rm aws_instance.web

# Move state (refactoring)
terraform state mv aws_instance.old aws_instance.new

# Import state (existing resource)
terraform import aws_instance.web i-1234567890abcdef0
```

---

## 6. Modules

### 6.1 Module Structure

```
modules/
├── vpc/
│   ├── main.tf
│   ├── variables.tf
│   └── outputs.tf
└── ec2/
    ├── main.tf
    ├── variables.tf
    └── outputs.tf
```

### 6.2 Module Definition

```hcl
# modules/vpc/main.tf

resource "aws_vpc" "this" {
  cidr_block           = var.cidr_block
  enable_dns_hostnames = true

  tags = merge(var.tags, {
    Name = var.name
  })
}

resource "aws_subnet" "public" {
  count                   = length(var.public_subnets)
  vpc_id                  = aws_vpc.this.id
  cidr_block              = var.public_subnets[count.index]
  availability_zone       = var.availability_zones[count.index]
  map_public_ip_on_launch = true

  tags = merge(var.tags, {
    Name = "${var.name}-public-${count.index + 1}"
  })
}
```

```hcl
# modules/vpc/variables.tf

variable "name" {
  type = string
}

variable "cidr_block" {
  type    = string
  default = "10.0.0.0/16"
}

variable "public_subnets" {
  type = list(string)
}

variable "availability_zones" {
  type = list(string)
}

variable "tags" {
  type    = map(string)
  default = {}
}
```

```hcl
# modules/vpc/outputs.tf

output "vpc_id" {
  value = aws_vpc.this.id
}

output "public_subnet_ids" {
  value = aws_subnet.public[*].id
}
```

### 6.3 Module Usage

```hcl
# main.tf

module "vpc" {
  source = "./modules/vpc"

  name               = "myapp"
  cidr_block         = "10.0.0.0/16"
  public_subnets     = ["10.0.1.0/24", "10.0.2.0/24"]
  availability_zones = ["ap-northeast-2a", "ap-northeast-2c"]

  tags = {
    Environment = "production"
  }
}

module "ec2" {
  source = "./modules/ec2"

  name          = "myapp-web"
  subnet_id     = module.vpc.public_subnet_ids[0]
  instance_type = "t3.micro"
}
```

---

## 7. Workspaces

```bash
# List workspaces
terraform workspace list

# Create new workspace
terraform workspace new dev
terraform workspace new prod

# Switch workspace
terraform workspace select prod

# Show current workspace
terraform workspace show
```

```hcl
# Workspace-specific configuration
locals {
  environment = terraform.workspace

  instance_type = {
    dev  = "t3.micro"
    prod = "t3.large"
  }
}

resource "aws_instance" "web" {
  instance_type = local.instance_type[local.environment]
  # ...
}
```

---

## 8. Best Practices

### 8.1 Directory Structure

```
terraform/
├── environments/
│   ├── dev/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── terraform.tfvars
│   └── prod/
│       ├── main.tf
│       ├── variables.tf
│       └── terraform.tfvars
├── modules/
│   ├── vpc/
│   ├── ec2/
│   └── rds/
└── global/
    └── iam/
```

### 8.2 Code Style

```hcl
# Resource naming convention
resource "aws_instance" "web" { }  # Singular
resource "aws_subnet" "public" { } # Use count/for_each for plural

# Variable defaults
variable "instance_type" {
  description = "EC2 instance type"  # Always include description
  type        = string
  default     = "t3.micro"
}

# Tag consistency
locals {
  common_tags = {
    Project     = var.project
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}
```

### 8.3 Security

```hcl
# Sensitive variables
variable "db_password" {
  type      = string
  sensitive = true
}

# Sensitive outputs
output "db_password" {
  value     = var.db_password
  sensitive = true
}
```

---

## 9. CI/CD Integration

### 9.1 GitHub Actions

```yaml
# .github/workflows/terraform.yml
name: Terraform

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  terraform:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: 1.6.0

      - name: Terraform Init
        run: terraform init

      - name: Terraform Plan
        run: terraform plan -no-color
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}

      - name: Terraform Apply
        if: github.ref == 'refs/heads/main'
        run: terraform apply -auto-approve
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
```

---

## 10. Next Steps

- [17_Monitoring_Logging_Cost.md](./17_Monitoring_Logging_Cost.md) - Monitoring
- [Docker/](../Docker/) - Kubernetes IaC

---

## Exercises

### Exercise 1: IaC Tool Selection

Your team is choosing an IaC tool. Evaluate which tool is most appropriate for each scenario.

| Scenario | Best Tool | Reason |
|---|---|---|
| Manage resources on both AWS and GCP from one codebase | ? | ? |
| Your team only uses AWS and wants native JSON/YAML support | ? | ? |
| You need to configure OS packages and software on existing VMs | ? | ? |
| You want to write infrastructure in TypeScript instead of HCL | ? | ? |

<details>
<summary>Show Answer</summary>

| Scenario | Best Tool | Reason |
|---|---|---|
| Manage resources on both AWS and GCP from one codebase | **Terraform** | Multi-cloud support; single HCL codebase with provider plugins for each cloud |
| Your team only uses AWS and wants native JSON/YAML support | **CloudFormation** | Native AWS service; no extra tooling needed; deep integration with AWS services |
| You need to configure OS packages and software on existing VMs | **Ansible** | Procedural tool designed for configuration management (SSH-based, agentless); Terraform handles provisioning, Ansible handles configuration |
| You want to write infrastructure in TypeScript instead of HCL | **Pulumi** | Supports general-purpose languages (Python, TypeScript, Go, C#); full IDE support, type safety |

Note: Terraform and CloudFormation are both declarative — you define the desired state, and the tool figures out how to achieve it. Ansible is procedural — you define the steps to execute in order. In practice, Terraform (provisioning) + Ansible (configuration) is a common combination.

</details>

---

### Exercise 2: Terraform Workflow

You have written a Terraform configuration to create an S3 bucket and an EC2 instance. List the exact commands to run and explain what each does, from starting a new project to applying the configuration for the first time.

<details>
<summary>Show Answer</summary>

```bash
# Step 1: Initialize the working directory
# Downloads provider plugins (hashicorp/aws) and sets up the backend
terraform init

# Step 2: Validate the configuration syntax
# Catches HCL syntax errors and type mismatches before touching real infrastructure
terraform validate

# Step 3: Preview the changes (dry run)
# Shows what resources will be created/modified/destroyed; does NOT change anything
terraform plan

# Step 4: Apply the configuration
# Creates the actual resources; prompts for confirmation unless -auto-approve is used
terraform apply

# Optional: Destroy all resources managed by this configuration
terraform destroy
```

**What happens internally during `terraform apply`:**
1. Reads the current state (`.tfstate` file or remote backend)
2. Calls the AWS API to check the current real-world state
3. Calculates the diff between desired state (`.tf` files) and actual state
4. Executes API calls to create/update/delete resources in the correct dependency order
5. Writes the new state to the backend

Important: Never manually edit `.tfstate` files — use `terraform state` commands instead. Corrupting the state file is the most common Terraform disaster.

</details>

---

### Exercise 3: Remote State and Locking

A team of 3 engineers all run `terraform apply` on the same production infrastructure. What problems can arise, and how do you configure a remote backend with state locking to prevent them?

<details>
<summary>Show Answer</summary>

**Problems with local state in a team environment:**
- **State conflicts**: Two engineers apply simultaneously → both read the same old state → both think they're making the same change → one overwrites the other's state, causing drift
- **State loss**: State file exists only on one laptop → engineer leaves or laptop breaks → state is lost → Terraform no longer knows what it manages
- **No history**: No audit trail of who changed what and when

**Solution: Remote backend with state locking**

```hcl
# backend.tf
terraform {
  backend "s3" {
    bucket         = "mycompany-terraform-state"
    key            = "production/terraform.tfstate"
    region         = "ap-northeast-2"
    encrypt        = true                        # Encrypt state at rest
    dynamodb_table = "terraform-state-locks"     # Lock table
  }
}
```

**Create the DynamoDB lock table (one-time setup):**
```bash
aws dynamodb create-table \
    --table-name terraform-state-locks \
    --attribute-definitions AttributeName=LockID,AttributeType=S \
    --key-schema AttributeName=LockID,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST
```

**How locking works:**
1. Engineer A runs `terraform apply` → acquires a lock in DynamoDB (writes a LockID entry)
2. Engineer B runs `terraform apply` simultaneously → tries to acquire lock → lock is held → Terraform exits with an error: "Error acquiring the state lock"
3. Engineer A finishes → lock is released
4. Engineer B can now proceed safely

**For GCP, use Cloud Storage backend (locking is built-in via object versioning):**
```hcl
terraform {
  backend "gcs" {
    bucket = "mycompany-terraform-state"
    prefix = "production/terraform.tfstate"
  }
}
```

</details>

---

### Exercise 4: Variables and Outputs

You have this Terraform resource for an RDS instance. Refactor it to use variables with descriptions and add outputs for the database endpoint and port.

```hcl
resource "aws_db_instance" "main" {
  identifier        = "myapp-db"
  engine            = "mysql"
  engine_version    = "8.0"
  instance_class    = "db.t3.micro"
  allocated_storage = 20
  username          = "admin"
  password          = "SuperSecret123!"
  skip_final_snapshot = true
}
```

<details>
<summary>Show Answer</summary>

```hcl
# variables.tf
variable "db_identifier" {
  description = "The name of the RDS instance"
  type        = string
  default     = "myapp-db"
}

variable "db_instance_class" {
  description = "The instance type of the RDS instance"
  type        = string
  default     = "db.t3.micro"
}

variable "db_allocated_storage" {
  description = "The allocated storage size in gigabytes"
  type        = number
  default     = 20
}

variable "db_username" {
  description = "The master username for the database"
  type        = string
  default     = "admin"
}

variable "db_password" {
  description = "The master password for the database"
  type        = string
  sensitive   = true  # Never shown in plan/apply output or state list
}
```

```hcl
# main.tf
resource "aws_db_instance" "main" {
  identifier        = var.db_identifier
  engine            = "mysql"
  engine_version    = "8.0"
  instance_class    = var.db_instance_class
  allocated_storage = var.db_allocated_storage
  username          = var.db_username
  password          = var.db_password
  skip_final_snapshot = true
}
```

```hcl
# outputs.tf
output "db_endpoint" {
  description = "The connection endpoint for the RDS instance"
  value       = aws_db_instance.main.endpoint
}

output "db_port" {
  description = "The port the database is listening on"
  value       = aws_db_instance.main.port
}
```

```bash
# Set the sensitive password via environment variable (never in terraform.tfvars)
export TF_VAR_db_password="SuperSecret123!"
terraform apply
```

Key points:
- Mark passwords as `sensitive = true` so they are redacted in logs and CLI output
- Never hardcode credentials in `.tf` files that get committed to version control
- Pass sensitive values via `TF_VAR_` environment variables or a secrets manager integration (e.g., Vault provider)

</details>

---

### Exercise 5: Terraform Module Design

You need to deploy the same three-tier architecture (VPC + web tier + database tier) to both `dev` and `prod` environments. The `dev` environment uses smaller, cheaper instances. Describe the module and directory structure, and show how `prod/main.tf` would call the modules differently from `dev/main.tf`.

<details>
<summary>Show Answer</summary>

**Directory structure:**
```
terraform/
├── modules/
│   ├── vpc/
│   │   ├── main.tf       # VPC, subnets, IGW, route tables
│   │   ├── variables.tf  # cidr_block, name, etc.
│   │   └── outputs.tf    # vpc_id, public_subnet_ids, private_subnet_ids
│   ├── web/
│   │   ├── main.tf       # EC2 + Auto Scaling Group + ALB
│   │   ├── variables.tf  # instance_type, subnet_ids, etc.
│   │   └── outputs.tf    # alb_dns_name
│   └── database/
│       ├── main.tf       # RDS instance + subnet group + security group
│       ├── variables.tf  # db_instance_class, db_name, subnet_ids
│       └── outputs.tf    # db_endpoint, db_port
├── environments/
│   ├── dev/
│   │   ├── main.tf
│   │   └── terraform.tfvars
│   └── prod/
│       ├── main.tf
│       └── terraform.tfvars
```

**environments/dev/main.tf:**
```hcl
module "vpc" {
  source     = "../../modules/vpc"
  name       = "myapp-dev"
  cidr_block = "10.0.0.0/16"
}

module "web" {
  source        = "../../modules/web"
  instance_type = "t3.micro"        # Cheap dev instance
  subnet_ids    = module.vpc.public_subnet_ids
}

module "database" {
  source            = "../../modules/database"
  db_instance_class = "db.t3.micro" # Smallest RDS
  subnet_ids        = module.vpc.private_subnet_ids
}
```

**environments/prod/main.tf:**
```hcl
module "vpc" {
  source     = "../../modules/vpc"
  name       = "myapp-prod"
  cidr_block = "10.1.0.0/16"
}

module "web" {
  source        = "../../modules/web"
  instance_type = "t3.large"        # Production-grade instances
  subnet_ids    = module.vpc.public_subnet_ids
}

module "database" {
  source            = "../../modules/database"
  db_instance_class = "db.r6g.large"  # Production RDS
  multi_az          = true             # High availability
  subnet_ids        = module.vpc.private_subnet_ids
}
```

Benefits of this structure:
- Module code is written once and reused across environments — DRY principle
- Environment-specific parameters are isolated in each environment's `main.tf` and `terraform.tfvars`
- Different Terraform state files per environment prevent a dev `destroy` from affecting prod
- Adding a new environment (e.g., `staging`) is just creating a new directory and calling the existing modules

</details>

---

## References

- [Terraform Documentation](https://www.terraform.io/docs)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [Terraform Google Provider](https://registry.terraform.io/providers/hashicorp/google/latest/docs)
- [Terraform Best Practices](https://www.terraform-best-practices.com/)
