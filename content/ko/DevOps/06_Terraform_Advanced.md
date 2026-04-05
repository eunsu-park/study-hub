# 레슨 6: Terraform 고급

**이전**: [Infrastructure as Code](./05_Infrastructure_as_Code.md) | **다음**: [구성 관리](./07_Configuration_Management.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 적절한 입력/출력 인터페이스를 갖춘 재사용 가능한 Terraform 모듈을 설계하고 게시할 수 있다
2. 워크스페이스를 사용하여 단일 구성에서 여러 환경을 관리할 수 있다
3. 복잡한 인프라 시나리오를 위해 데이터 소스와 프로비저너를 활용할 수 있다
4. Terraform 상태와 실제 인프라 간의 구성 드리프트를 감지하고 수정할 수 있다
5. import 명령을 사용하여 기존 인프라를 Terraform 관리 하에 가져올 수 있다
6. DRY 구성을 위한 Terragrunt와 인프라 테스팅을 위한 Terratest를 포함한 고급 패턴을 적용할 수 있다

---

Terraform의 기초인 프로바이더, 리소스, 변수, 상태만으로는 시작할 수 있지만, 프로덕션 수준의 인프라에는 고급 패턴이 필요합니다. 모듈은 프로젝트 간 코드 중복을 방지합니다. 워크스페이스는 별도의 코드베이스 없이 여러 환경을 관리합니다. 드리프트 감지는 승인되지 않은 수동 변경을 포착합니다. 테스팅 프레임워크는 인프라 코드가 실제 클라우드 리소스를 건드리기 전에 검증합니다. 이 레슨에서는 취미 수준의 Terraform과 프로덕션 Terraform을 구분하는 패턴과 도구를 다룹니다.

> **비유 -- 소프트웨어 라이브러리:** Terraform 모듈은 소프트웨어 라이브러리와 같습니다. 동일한 네트워킹 코드를 모든 프로젝트에 복사하는 대신 모듈을 한 번 게시하고 어디서든 임포트합니다. 잘 설계된 라이브러리가 깔끔한 API(함수 시그니처, 반환 타입)를 갖는 것처럼, 잘 설계된 모듈은 명확한 입력(변수), 출력, 문서를 갖습니다.

## 1. 모듈

Terraform 모듈은 재사용 가능하고 자체 완결적인 Terraform 구성 패키지입니다.

### 모듈 구조

```
modules/
└── vpc/
    ├── main.tf          # Resources
    ├── variables.tf     # Input interface
    ├── outputs.tf       # Output interface
    ├── versions.tf      # Provider requirements
    └── README.md        # Usage documentation
```

### 모듈 작성

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

### 모듈 사용

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

### 모듈 소스

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

## 2. 워크스페이스

워크스페이스를 사용하면 단일 Terraform 구성으로 여러 환경(dev, staging, production)을 관리할 수 있습니다.

### 워크스페이스 명령어

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

### 구성에서 워크스페이스 사용

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

### 원격 백엔드에서의 워크스페이스

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

### 워크스페이스 vs 별도 디렉토리

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

## 3. for_each와 Dynamic 블록

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

### Dynamic 블록

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

## 4. 드리프트 감지

구성 드리프트는 실제 인프라가 Terraform 상태와 달라질 때 발생하며, 보통 수동 변경에 의해 발생합니다.

### 드리프트 감지

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

### CI/CD에서의 드리프트 감지

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

Import는 기존 인프라를 Terraform 관리 하에 가져옵니다.

### Import 워크플로우

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

### Import 블록 (Terraform 1.5+)

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

### 스크립트를 사용한 대량 Import

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

Terragrunt는 Terraform을 감싸는 얇은 래퍼로, 구성을 DRY하게 유지하고 여러 환경을 관리하기 위한 추가 도구를 제공합니다.

### Terragrunt 프로젝트 구조

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

### 루트 terragrunt.hcl

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

### 컴포넌트 terragrunt.hcl

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

### Terragrunt 명령어

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

## 7. Terratest를 사용한 테스팅

Terratest는 실제 인프라를 배포하고 검증한 후 삭제하여 Terraform 코드를 테스트하는 Go 라이브러리입니다.

### 기본 테스트 구조

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

### 테스트 실행

```bash
# Run all tests
cd test
go test -v -timeout 30m

# Run a specific test
go test -v -timeout 30m -run TestVpcModule

# Run with shorter timeout for faster feedback
go test -v -timeout 10m -run TestVpcModule
```

### Terraform 검증 및 Plan 테스트 (배포 없음)

```bash
# Fast validation without deploying (no cloud costs)
# Use in CI for every PR

terraform init -backend=false        # Skip backend initialization
terraform validate                   # Syntax and consistency check
terraform plan -input=false          # Plan without apply
```

---

## 8. 모범 사례 요약

### 모듈 설계

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

### 상태 관리

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

## 연습 문제

### 연습 문제 1: 재사용 가능한 모듈 만들기

Application Load Balancer(ALB)를 위한 Terraform 모듈을 만드십시오:
1. 입력 받기: 이름, VPC ID, 서브넷 ID, 상태 검사 경로, 대상 포트
2. ALB, 대상 그룹, 리스너(HTTP 및 HTTPS 리다이렉트), 보안 그룹 생성
3. ALB DNS 이름, ARN, 대상 그룹 ARN 출력
4. 대상 그룹에 등록된 EC2 인스턴스도 생성하는 루트 구성에서 모듈 사용
5. 상태 검사 경로에 대한 입력 검증 추가 ("/"로 시작해야 함)

### 연습 문제 2: 워크스페이스를 사용한 다중 환경

워크스페이스를 사용하는 Terraform 구성을 설정하십시오:
1. dev, staging, production 워크스페이스 생성
2. 워크스페이스별로 다른 인스턴스 유형과 수 구성
3. 충돌을 피하기 위해 워크스페이스별로 다른 CIDR 블록 사용
4. 워크스페이스 간 전환 및 적용 방법 표시
5. 실수로 프로덕션을 변경하는 것을 방지하기 위한 안전 장치 논의

### 연습 문제 3: 레거시 인프라 Import

콘솔을 통해 생성된 기존 AWS 설정이 있습니다:
- VPC 1개, 서브넷 2개, 보안 그룹 1개, EC2 인스턴스 2개
1. 이 리소스들에 대한 Terraform 구성 작성
2. 각 리소스에 대한 import 명령어 작성
3. import가 성공적으로 완료되었는지 검증하는 방법 설명
4. import 후 `terraform plan`이 변경 사항을 표시하면 어떻게 할 것인지

### 연습 문제 4: 드리프트 감지 파이프라인

다음을 포함하는 Terraform CI/CD 파이프라인을 설계하십시오:
1. PR 시: `terraform fmt`, `terraform validate`, `terraform plan` 실행
2. main 머지 시: dev에 대해 `terraform apply` 자동 실행
3. 매일 예약: Slack 알림과 함께 드리프트 감지 실행
4. 완전한 GitHub Actions 워크플로우 YAML 작성
5. 프로덕션 환경에 대한 적절한 안전 장치 포함

---

**이전**: [Infrastructure as Code](./05_Infrastructure_as_Code.md) | [개요](00_Overview.md) | **다음**: [구성 관리](./07_Configuration_Management.md)

**License**: CC BY-NC 4.0
