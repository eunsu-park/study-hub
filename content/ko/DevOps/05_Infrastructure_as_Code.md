# 레슨 5: Infrastructure as Code

**이전**: [GitHub Actions 심화](./04_GitHub_Actions_Deep_Dive.md) | **다음**: [Terraform 고급](./06_Terraform_Advanced.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Infrastructure as Code(IaC)의 원칙과 수동 인프라 프로비저닝을 대체하는 이유를 설명할 수 있다
2. 프로바이더, 리소스, 변수, 출력을 포함한 HCL 문법으로 Terraform 구성을 작성할 수 있다
3. Terraform 워크플로우(init, plan, apply, destroy)를 실행할 수 있다
4. Terraform 상태를 관리하고, 팀 협업을 위해 원격 상태 백엔드가 필수적인 이유를 이해할 수 있다
5. 유지보수성을 위해 입력 변수, 출력 값, 로컬 값을 사용하여 Terraform 코드를 구조화할 수 있다

---

Infrastructure as Code(IaC)는 수동 프로세스나 대화형 도구가 아닌 기계가 읽을 수 있는 구성 파일을 통해 인프라를 관리하고 프로비저닝하는 실천 방법입니다. IaC 이전에는 서버를 프로비저닝한다는 것은 클라우드 콘솔을 클릭하거나, 임의의 스크립트를 실행하거나, 운영 팀에 티켓을 제출하는 것을 의미했습니다. 이러한 접근 방식은 느리고, 오류가 발생하기 쉬우며, 안정적으로 재현하기 불가능합니다. IaC는 버전 관리가 애플리케이션 코드에 가져온 것과 동일한 엄격함을 인프라에 적용합니다. 모든 변경이 추적, 검토, 테스트되며 재현 가능합니다.

> **비유 -- 건축 설계도:** 수동 인프라 관리는 건설 팀에게 구두 지시를 내려 집을 짓는 것과 같습니다. 모든 집이 조금씩 다르게 나오고, 기억에 의존하여 집을 복제하는 것은 신뢰할 수 없습니다. IaC는 건축 설계도입니다. 정확하고, 버전 관리되며, 어떤 건설 팀이든 사용하여 동일한 결과를 만들 수 있습니다.

## 1. Infrastructure as Code가 필요한 이유

### 수동 인프라의 문제점

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

### IaC의 이점

| 이점 | 설명 |
|------|------|
| **재현성** | 매번 동일한 환경 생성 |
| **버전 관리** | Git에서 모든 인프라 변경 추적 |
| **코드 리뷰** | Pull Request를 통한 인프라 변경 검토 |
| **셀프서비스** | 개발자가 자체 환경을 프로비저닝 |
| **속도** | 복잡한 환경을 수 분 내에 생성 |
| **문서화** | 코드 자체가 문서 |
| **재해 복구** | 코드에서 전체 인프라 재생성 |
| **비용 관리** | 미사용 환경 삭제, 리소스 감사 |

---

## 2. IaC 접근 방식

### 선언적 vs 명령적

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

### 가변(Mutable) vs 불변(Immutable) 인프라

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

## 3. IaC 도구 현황

| 도구 | 유형 | 언어 | 프로바이더 | 상태 |
|------|------|------|-----------|------|
| **Terraform** | 선언적 | HCL | 멀티 클라우드 | 명시적 상태 파일 |
| **OpenTofu** | 선언적 | HCL | 멀티 클라우드 | 명시적 상태 파일 |
| **AWS CloudFormation** | 선언적 | JSON/YAML | AWS 전용 | AWS 관리 |
| **Pulumi** | 선언적 | Python/TS/Go | 멀티 클라우드 | 관리형 또는 셀프 호스팅 |
| **Ansible** | 명령적/선언적 | YAML | 멀티 클라우드 | 상태 없음 (에이전트리스) |
| **AWS CDK** | 명령적 | Python/TS/Java | AWS 전용 | CloudFormation 스택 |
| **Crossplane** | 선언적 | YAML (K8s CRD) | 멀티 클라우드 | Kubernetes 상태 |

이 레슨에서는 가장 널리 채택된 멀티 클라우드 IaC 도구인 **Terraform**에 집중합니다.

---

## 4. Terraform 기초

### 설치

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

## 5. Terraform 구성 구조

### 프로젝트 레이아웃

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

### 프로바이더 구성

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

### 리소스

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

### 변수

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

### 출력

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

## 6. Terraform 워크플로우

### 핵심 명령어

```
terraform init ──▶ terraform plan ──▶ terraform apply ──▶ terraform destroy
     │                    │                   │                    │
  Download            Show what           Execute the         Tear down all
  providers           will change         planned changes     resources
  Initialize          (dry run)           (asks for           (asks for
  backend                                 confirmation)       confirmation)
```

### 단계별 실행

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

### Plan 출력 기호

```
+ create           A new resource will be created
- destroy          An existing resource will be destroyed
~ update in-place  An existing resource will be modified
-/+ replace        A resource will be destroyed and recreated
<= read            A data source will be read
```

---

## 7. 상태 관리

Terraform 상태는 구성을 실제 리소스에 매핑하는 JSON 파일입니다. Terraform 프로젝트에서 가장 중요한 아티팩트입니다.

### 상태가 추적하는 내용

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

### 로컬 vs 원격 상태

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

### 원격 백엔드 구성 (AWS S3)

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

### 상태 명령어

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

## 8. 로컬 값과 데이터 소스

### 로컬 값

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

### 데이터 소스

데이터 소스를 사용하면 기존 리소스나 외부 데이터를 조회할 수 있습니다.

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

## 9. 리소스 의존성과 라이프사이클

### 암시적 의존성

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

### 명시적 의존성

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

### 라이프사이클 규칙

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

## 연습 문제

### 연습 문제 1: 첫 번째 Terraform 프로젝트

간단한 웹 서버 설정을 프로비저닝하는 Terraform 구성을 만드십시오:
1. AWS 프로바이더 구성으로 `providers.tf` 생성
2. 리전, 인스턴스 유형, 프로젝트 이름 변수로 `variables.tf` 생성
3. VPC, 퍼블릭 서브넷, 보안 그룹(포트 80, 443, 22), nginx user_data가 있는 EC2 인스턴스로 `main.tf` 생성
4. 인스턴스의 퍼블릭 IP와 URL을 보여주는 `outputs.tf` 생성
5. `terraform init`, `terraform validate`, `terraform plan` 실행
6. plan 출력을 검토하고 각 리소스가 무엇을 하는지 설명

### 연습 문제 2: 상태 관리

연습 문제 1에서 시작하여:
1. 원격 상태를 위한 S3 백엔드 구성 생성
2. 상태 잠금 없이 두 명의 개발자가 동시에 `terraform apply`를 실행하면 어떻게 되는지 설명
3. `terraform state list`와 `terraform state show`를 하나의 리소스에 대해 실행
4. `terraform state mv`와 `terraform state rm`을 사용할 시나리오 설명

### 연습 문제 3: 변수와 환경

연습 문제 1의 구성을 여러 환경을 지원하도록 확장하십시오:
1. 변수를 사용하여 인스턴스 유형 제어 (dev에는 micro, production에는 large)
2. dev와 production을 위한 별도의 `.tfvars` 파일 생성
3. 인스턴스 유형을 허용 목록으로 제한하는 검증 규칙 추가
4. 데이터베이스 비밀번호를 위한 민감한 변수 추가
5. 환경별 변수로 적용하는 방법 표시: `terraform apply -var-file=dev.tfvars`

### 연습 문제 4: 데이터 소스

구성에 데이터 소스를 추가하십시오:
1. 하드코딩 대신 `aws_ami` 데이터 소스를 사용하여 최신 Amazon Linux 2023 AMI 찾기
2. `aws_availability_zones`를 사용하여 AZ에 걸쳐 리소스 분산
3. `aws_caller_identity`를 사용하여 계정 ID로 리소스에 태그 지정
4. `resource` 블록과 `data` 블록의 차이점 설명

---

**이전**: [GitHub Actions 심화](./04_GitHub_Actions_Deep_Dive.md) | [개요](00_Overview.md) | **다음**: [Terraform 고급](./06_Terraform_Advanced.md)

**License**: CC BY-NC 4.0
